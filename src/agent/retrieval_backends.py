"""Pluggable retrieval backends for imported knowledge (v1.4).

v1.3 retrieved knowledge chunks with the same content-addressed firing math the
memory bank uses: a chunk vector (from the deterministic offline encoder) dotted
with the radius-scaled query. That is perfect for CI — it is fast, needs no
download, and reproduces exactly — but it is **not semantic**: the deterministic
encoder hashes the whole string, so a paraphrase of a chunk is essentially an
orthogonal random vector. Real coding/project docs need retrieval *by meaning*.

This module introduces a small backend seam so the workbench can keep the
deterministic backend as the default (tests, offline) while optionally using a
semantic backend (MiniLM via the existing ``TextEncoder`` path) when a real
embedding model is available. The two backends are interchangeable behind one
interface; nothing about the concept-cell geometry, the memory bank, or the
knowledge library changes.

Design rules honoured here:

* The deterministic backend reproduces v1.3 retrieval **exactly** (same
  activations, same ordering, same skip/empty behaviour).
* The semantic backend imports its model **lazily**; importing this module never
  triggers a download, so tests stay offline.
* If the semantic model is unavailable, construction raises
  :class:`BackendUnavailableError` so callers can fall back gracefully.
* Every returned candidate preserves its source/chunk provenance and records
  which backend produced it (``backend_name`` / ``is_semantic``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .knowledge_sources import KnowledgeChunk
from .orchestrator import EncoderProtocol

_RADIUS = 0.9
_DETERMINISTIC = "deterministic"
_SEMANTIC = "semantic"


class BackendUnavailableError(RuntimeError):
    """Raised when a requested retrieval backend cannot be constructed.

    The common case is asking for the semantic backend when the embedding model
    (sentence-transformers / torch) is not installed or cannot load offline.
    Callers should catch this and fall back to the deterministic backend.
    """


@dataclass
class KnowledgeCandidate:
    """One retrieved knowledge chunk with its provenance and score.

    ``activation`` is the backend's score for the chunk against the query (a
    radius-scaled dot product for the deterministic backend, a cosine similarity
    for the semantic backend). ``backend_name`` / ``is_semantic`` record which
    backend produced the result so a query audit can show it. ``version`` is the
    owning source's version, enriched by the retrieval wrapper.
    """

    source_id: str
    chunk_id: str
    activation: float
    rank: int
    source_name: str
    domain: str
    authority: str
    chunk_text: str
    source_section: Optional[str] = None
    version: Optional[str] = None
    backend_name: str = _DETERMINISTIC
    is_semantic: bool = False


@runtime_checkable
class RetrievalBackend(Protocol):
    """A backend that indexes knowledge chunks and ranks them for a query."""

    backend_name: str
    is_semantic: bool

    def index_chunks(self, chunks: Sequence[KnowledgeChunk]) -> None:
        ...

    def retrieve(self, query_text: str, top_k: int = 5) -> List[KnowledgeCandidate]:
        ...


def _candidate_from(chunk: KnowledgeChunk, score: float, rank: int,
                    backend_name: str, is_semantic: bool) -> KnowledgeCandidate:
    return KnowledgeCandidate(
        source_id=chunk.source_id,
        chunk_id=chunk.chunk_id,
        activation=round(float(score), 6),
        rank=rank,
        source_name=chunk.source_name,
        domain=chunk.domain.value,
        authority=chunk.authority.value,
        chunk_text=chunk.chunk_text,
        source_section=chunk.source_section,
        backend_name=backend_name,
        is_semantic=is_semantic,
    )


class DeterministicRetrievalBackend:
    """Offline, content-addressed backend — the default for tests and CI.

    Reproduces v1.3 knowledge retrieval exactly: each chunk vector is
    unit-normalised to a cell weight, the query is scaled to ``radius``, and
    ``activation = w . q``. The same deterministic encoder used by the memory
    bank is used here, so writing then querying the *same* text reproduces the
    same vector and ranking. It is not semantic — a paraphrase will not rank a
    chunk highly — which is exactly why it is reproducible.
    """

    backend_name = _DETERMINISTIC
    is_semantic = False

    def __init__(self, encoder: EncoderProtocol, radius: float = _RADIUS):
        self._encoder = encoder
        self._radius = float(radius)
        self._chunks: List[KnowledgeChunk] = []

    def index_chunks(self, chunks: Sequence[KnowledgeChunk]) -> None:
        self._chunks = list(chunks)

    def retrieve(self, query_text: str,
                 top_k: int = 5) -> List[KnowledgeCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not self._chunks:
            return []

        q_raw = np.asarray(self._encoder.encode_one(query_text), dtype=np.float64)
        qn = np.linalg.norm(q_raw)
        if qn < 1e-12:
            return []
        q = q_raw / qn * self._radius

        scored: List[tuple[float, KnowledgeChunk]] = []
        for chunk in self._chunks:
            x = np.asarray(self._encoder.encode_one(chunk.chunk_text),
                           dtype=np.float64)
            xn = np.linalg.norm(x)
            if xn < 1e-12:
                continue
            w = x / xn
            scored.append((float(w @ q), chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            _candidate_from(chunk, score, rank, self.backend_name,
                            self.is_semantic)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


class SemanticRetrievalBackend:
    """Meaning-based backend backed by a real sentence embedder (MiniLM).

    Chunks are embedded once at index time into a unit-normalised matrix;
    retrieval scores each chunk by cosine similarity to the embedded query, so a
    paraphrase of a chunk still ranks it highly. The embedder is loaded lazily
    through :func:`_load_minilm` (the existing ``TextEncoder`` path); if it
    cannot load, construction raises :class:`BackendUnavailableError`.

    An ``embedder`` exposing ``encode_one(text) -> vector`` may be injected
    (used by tests to mock the model without any download).
    """

    backend_name = _SEMANTIC
    is_semantic = True

    def __init__(self, embedder: Optional[EncoderProtocol] = None,
                 model_name: str = "all-MiniLM-L6-v2"):
        self._embedder = embedder if embedder is not None else _load_minilm(
            model_name)
        self._chunks: List[KnowledgeChunk] = []
        self._matrix: Optional[np.ndarray] = None  # (N, D) unit-normalised

    def index_chunks(self, chunks: Sequence[KnowledgeChunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._matrix = None
            return
        rows: List[np.ndarray] = []
        for chunk in self._chunks:
            rows.append(self._unit(self._embedder.encode_one(chunk.chunk_text)))
        self._matrix = np.vstack(rows)

    def retrieve(self, query_text: str,
                 top_k: int = 5) -> List[KnowledgeCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not self._chunks or self._matrix is None or not query_text.strip():
            return []

        q = self._unit(self._embedder.encode_one(query_text))
        if not np.any(q):
            return []
        scores = self._matrix @ q  # cosine similarity, both unit-normalised

        order = np.argsort(-scores)[:top_k]
        return [
            _candidate_from(self._chunks[idx], float(scores[idx]), rank,
                            self.backend_name, self.is_semantic)
            for rank, idx in enumerate(order, start=1)
        ]

    @staticmethod
    def _unit(vec) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.zeros_like(v)
        return v / n


class _MiniLMEmbedder:
    """Adapter exposing ``encode_one`` over the existing ``TextEncoder``."""

    def __init__(self, model_name: str):
        from concept_cells.encoders import TextEncoder
        self._encoder = TextEncoder(model_name)

    def encode_one(self, text: str) -> np.ndarray:
        return self._encoder.encode([text]).embeddings[0]


def _load_minilm(model_name: str = "all-MiniLM-L6-v2") -> EncoderProtocol:
    """Build the MiniLM embedder lazily; never imported at module load.

    Raises :class:`BackendUnavailableError` if sentence-transformers / torch are
    missing or the model cannot be loaded (e.g. offline with no cache). This is
    what lets the workbench fall back to the deterministic backend gracefully.
    """
    try:
        return _MiniLMEmbedder(model_name)
    except Exception as exc:  # noqa: BLE001 - any load failure is "unavailable"
        raise BackendUnavailableError(
            f"semantic backend unavailable: {exc}") from exc


def resolve_backend_name(explicit: Optional[str] = None) -> str:
    """Resolve the requested backend name from an explicit value or the env.

    Precedence: an explicit argument wins; otherwise the
    ``KNOWLEDGE_RETRIEVAL_BACKEND`` environment variable; otherwise
    ``"deterministic"``. The value is lower-cased and trimmed.
    """
    name = explicit if explicit is not None else os.environ.get(
        "KNOWLEDGE_RETRIEVAL_BACKEND", "")
    name = (name or "").strip().lower()
    return name or _DETERMINISTIC


def make_backend(name: Optional[str], *,
                 deterministic_encoder: EncoderProtocol,
                 semantic_embedder: Optional[EncoderProtocol] = None
                 ) -> RetrievalBackend:
    """Construct a backend by name.

    ``deterministic`` (default) wraps ``deterministic_encoder``. ``semantic``
    builds :class:`SemanticRetrievalBackend`, which raises
    :class:`BackendUnavailableError` if the model cannot load. An unknown name
    falls back to deterministic.
    """
    resolved = resolve_backend_name(name)
    if resolved == _SEMANTIC:
        return SemanticRetrievalBackend(embedder=semantic_embedder)
    return DeterministicRetrievalBackend(deterministic_encoder)
