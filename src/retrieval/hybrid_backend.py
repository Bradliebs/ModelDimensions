"""Hybrid knowledge-retrieval backend (v2.2) — opt-in, composes v1.x backends.

The hybrid backend blends three signals for each candidate chunk:

* a **keyword** overlap score (lexical, paraphrase-robust at the token level);
* an optional **semantic** cosine score, only when a real embedding component
  is supplied — it is never loaded implicitly, so the default stays offline;
* a gentle **authority** multiplier from the source metadata.

It composes the frozen :class:`DeterministicRetrievalBackend` (whose geometric
activation is reused purely as a deterministic tie-breaker / fallback ordering)
and an optional :class:`SemanticRetrievalBackend`. It does not reimplement the
concept-cell geometry or the embedding model.

Safety is inherited, not re-decided here: deleted/inactive chunks are excluded
upstream (and defensively filtered again below), stale chunks are *labelled*
upstream and pass through untouched, and the memory verifier / AnswerGuard run
on a separate path. The hybrid backend only reorders the chunks it is handed
and records an auditable report of why.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from agent.retrieval_backends import (
    BackendUnavailableError,
    DeterministicRetrievalBackend,
    EncoderProtocol,
    KnowledgeCandidate,
    SemanticRetrievalBackend,
)
from agent.knowledge_sources import KnowledgeChunk

from .scoring import (
    HybridRetrievalReport,
    HybridWeights,
    ScoredChunk,
    authority_weight,
    combine_scores,
    exact_phrase_match,
    keyword_overlap,
)

__all__ = ["HybridRetrievalBackend"]

# How many rejected candidates to record in the audit report. Bounded so the
# audit stays readable on large corpora ("where practical").
_MAX_REJECTED_RECORDED = 10


class HybridRetrievalBackend:
    """Blended keyword + (optional) semantic + authority retrieval backend.

    ``semantic_embedder`` is an object exposing ``encode_one(text) -> vector``.
    When supplied, the semantic component is activated; if constructing it
    fails (:class:`BackendUnavailableError`), the backend degrades silently to
    keyword + geometry and reports ``semantic_available = False``.
    """

    backend_name = "hybrid"

    def __init__(
        self,
        deterministic_encoder: EncoderProtocol,
        *,
        semantic_embedder: Optional[EncoderProtocol] = None,
        weights: Optional[HybridWeights] = None,
    ) -> None:
        self._geometry = DeterministicRetrievalBackend(deterministic_encoder)
        self._weights = weights or HybridWeights()
        self._semantic: Optional[SemanticRetrievalBackend] = None
        if semantic_embedder is not None:
            try:
                self._semantic = SemanticRetrievalBackend(embedder=semantic_embedder)
            except BackendUnavailableError:
                self._semantic = None
        # ``is_semantic`` reflects whether a semantic component is actually live,
        # so the audit and ``KnowledgeCandidate.is_semantic`` never overclaim.
        self.is_semantic = self._semantic is not None
        self._chunks: List[KnowledgeChunk] = []
        self.last_report: Optional[HybridRetrievalReport] = None

    def index_chunks(self, chunks: Sequence[KnowledgeChunk]) -> None:
        self._chunks = list(chunks)
        active = [c for c in self._chunks if c.active]
        self._geometry.index_chunks(active)
        if self._semantic is not None:
            try:
                self._semantic.index_chunks(active)
            except Exception:
                # An injected embedder that fails at index time is treated as
                # unavailable: degrade to keyword + geometry rather than crash a
                # query. Safety is preserved; only the optional signal is lost.
                self._semantic = None
                self.is_semantic = False

    def retrieve(self, query_text: str, top_k: int = 5) -> List[KnowledgeCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        active = [c for c in self._chunks if c.active]
        inactive = [c for c in self._chunks if not c.active]

        if not active or not query_text.strip():
            self.last_report = HybridRetrievalReport(
                backend_name=self.backend_name,
                semantic_available=self.is_semantic,
                weights=self._weights_dict(),
                selected=[],
                rejected=[
                    self._reject(c, "inactive_source", 0.0, None, 0.0, False)
                    for c in inactive[:_MAX_REJECTED_RECORDED]
                ],
            )
            return []

        # Deterministic geometry activation per chunk (reused as a stable
        # tie-breaker and as the fallback ordering signal).
        geo_by_id = {
            cand.chunk_id: cand.activation
            for cand in self._geometry.retrieve(query_text, top_k=len(active))
        }
        # Optional semantic cosine per chunk.
        sem_by_id: dict = {}
        if self._semantic is not None:
            try:
                sem_by_id = {
                    cand.chunk_id: cand.activation
                    for cand in self._semantic.retrieve(query_text, top_k=len(active))
                }
            except Exception:
                # Degrade to keyword + geometry on a runtime embedder failure.
                self._semantic = None
                self.is_semantic = False
                sem_by_id = {}

        scored: List[ScoredChunk] = []
        for chunk in active:
            kw = keyword_overlap(query_text, chunk.chunk_text)
            sem = sem_by_id.get(chunk.chunk_id) if self.is_semantic else None
            auth_w = authority_weight(chunk.authority.value)
            exact = exact_phrase_match(query_text, chunk.chunk_text)
            combined = combine_scores(
                kw,
                sem,
                auth_w,
                weights=self._weights,
                semantic_available=self.is_semantic,
                exact_match=exact,
            )
            scored.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    source_name=chunk.source_name,
                    authority=chunk.authority.value,
                    keyword_score=round(kw, 6),
                    semantic_score=(round(float(sem), 6) if sem is not None else None),
                    authority_weight=round(auth_w, 6),
                    exact_match=exact,
                    combined_score=combined,
                    selected=False,
                )
            )

        # Fully deterministic ordering: combined score, then geometry activation
        # as a reproducible tie-breaker, then chunk_id for total stability.
        scored.sort(
            key=lambda s: (
                -s.combined_score,
                -geo_by_id.get(s.chunk_id, 0.0),
                s.chunk_id,
            )
        )

        selected = scored[:top_k]
        rejected = scored[top_k:]

        candidates: List[KnowledgeCandidate] = []
        chunk_by_id = {c.chunk_id: c for c in active}
        for rank, s in enumerate(selected, start=1):
            s.selected = True
            s.rank = rank
            chunk = chunk_by_id[s.chunk_id]
            candidates.append(
                KnowledgeCandidate(
                    source_id=chunk.source_id,
                    chunk_id=chunk.chunk_id,
                    activation=s.combined_score,
                    rank=rank,
                    source_name=chunk.source_name,
                    domain=chunk.domain.value,
                    authority=chunk.authority.value,
                    chunk_text=chunk.chunk_text,
                    source_section=chunk.source_section,
                    backend_name=self.backend_name,
                    is_semantic=self.is_semantic,
                )
            )

        rejected_records = []
        for s in rejected[:_MAX_REJECTED_RECORDED]:
            s.reason = "below_top_k"
            rejected_records.append(s.to_dict())
        for c in inactive[:_MAX_REJECTED_RECORDED]:
            rejected_records.append(
                self._reject(c, "inactive_source", 0.0, None, 0.0, False)
            )

        self.last_report = HybridRetrievalReport(
            backend_name=self.backend_name,
            semantic_available=self.is_semantic,
            weights=self._weights_dict(),
            selected=[s.to_dict() for s in selected],
            rejected=rejected_records,
        )
        return candidates

    def _weights_dict(self) -> dict:
        return {
            "keyword": self._weights.keyword,
            "semantic": self._weights.semantic,
            "authority_influence": self._weights.authority_influence,
            "exact_bonus": self._weights.exact_bonus,
        }

    @staticmethod
    def _reject(chunk: KnowledgeChunk, reason: str, kw: float,
                sem: Optional[float], auth_w: float, exact: bool) -> dict:
        return ScoredChunk(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            source_name=chunk.source_name,
            authority=chunk.authority.value,
            keyword_score=round(kw, 6),
            semantic_score=sem,
            authority_weight=round(auth_w, 6),
            exact_match=exact,
            combined_score=0.0,
            selected=False,
            reason=reason,
        ).to_dict()
