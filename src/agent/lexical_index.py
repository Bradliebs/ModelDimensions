"""BM25 lexical index over the concept-cell bank's source texts.

The V1 retrieval path is bi-encoder cosine over a whitened MiniLM
embedding (see :class:`src.agent.streaming_bank.StreamingBank`). exp27's
diagnostic on the 8-question multi-hop set showed two distinct failure
modes:

  - **Reranker-fixable** (e.g. Q4 Ottawa/Canada): keyword cell is inside
    the cosine top-50 but ranked too low to make the gate; cross-encoder
    rerank lifts it. Already supported in :mod:`src.agent.reranker`.
  - **Reranker-can't-help** (e.g. Q2 French Connection / Don Ellis): the
    keyword cell is **outside the cosine top-50 entirely**, so no amount
    of re-scoring the top-50 will surface it. This is what lexical /
    BM25 retrieval is for: a sparse lexical signal that catches
    rare-named entities (proper nouns, technical terms) the dense
    encoder folds onto a generic neighbourhood.

This module provides a thin, deterministic wrapper around
``rank_bm25.BM25Okapi`` plus a persisted on-disk layout that fails
closed if the corpus changes underneath it. It is the *retrieval* layer
only — composition with the dense bank lives in
:mod:`src.agent.hybrid_retriever`.

Persistence layout (under ``results/v1_bank/bm25_index/`` by default)::

    manifest.json   { corpus_hash, tokenizer_version, n_docs, k1, b,
                      created_at, source }
    index.pkl       pickled (BM25Okapi, cell_ids: np.ndarray[int64])

Stale-index detection: ``manifest.corpus_hash`` is the SHA-256 of the
concatenated cell ids and source-text lengths produced at build time. On
load, if the caller supplies a bank whose hash differs, ``load()``
raises ``LexicalIndexStale`` rather than silently returning a stale
ranking — silent BM25 stale-ness would be a wrong-answer source. The
hash is intentionally *not* over the full text (would be slow on 5.7M
cells) but ids + lengths is enough to catch the failure modes that
matter: cells added, removed, or rewritten.

Tokenizer: aligned with :mod:`src.agent.v1_answer_verifier`'s lexical
contract (``[A-Za-z0-9']+`` runs, lowercased) so that what BM25 ranks
on matches what the verifier checks. Stopwords are dropped at index
time *and* query time for consistency. The tokenizer version string is
embedded in the manifest so a future tokenizer change forces a rebuild.

Out of scope (Phase 1):
  - Live overlay updates. The index is a snapshot. Mutating the overlay
    after build requires a rebuild; the manifest hash will detect it.
  - Tantivy / Lucene backends. ``rank_bm25`` is pure Python; it's slow
    to build on the full 5.7M-cell bank (Phase 1.5 will measure that),
    but its query latency is fine for the cascade.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agent.streaming_bank import StreamingBank


# Token pattern intentionally matches v1_answer_verifier._TOKEN_RE so the
# verifier and the lexical index agree on what counts as a token. Bumping
# this requires bumping TOKENIZER_VERSION below to force index rebuilds.
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# Cheap stopword set. Mirrors v1_answer_verifier._STOPWORDS to keep
# behaviour consistent across retrieval and verification; do NOT import
# from the verifier (the verifier is the frozen V1 contract and should
# not export internals).
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "having",
    "of", "for", "to", "in", "on", "at", "by", "with", "from", "as",
    "and", "or", "but", "if", "then", "else", "than", "that", "this",
    "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "him", "we", "our", "us", "you", "your",
    "i", "me", "my",
    "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "not", "no", "yes",
    "can", "could", "would", "should", "may", "might", "must", "shall",
    "will", "won", "wont",
    "there", "here", "also", "such", "some", "any", "all", "each",
    "more", "most", "less", "least", "very", "only", "just", "even",
    "into", "onto", "about", "between", "during", "after", "before",
    "above", "below", "over", "under", "again", "still",
    "yet", "so", "because", "while", "until", "though", "although",
})

# Bump when _TOKEN_RE or _STOPWORDS changes. ``load`` rejects indices
# whose manifest version doesn't match — silent tokenizer drift is a
# wrong-answer source.
TOKENIZER_VERSION = "v1.lex.1"

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class LexicalIndexStale(RuntimeError):
    """Raised when the loaded index's corpus hash or tokenizer version
    does not match the bank/tokenizer the caller currently has."""


def tokenize(text: str) -> List[str]:
    """Lowercase, regex-tokenize, drop stopwords and bare-digit length-1
    tokens. Returns an empty list for empty/None input."""
    if not text:
        return []
    out: List[str] = []
    for tok in _TOKEN_RE.findall(text):
        t = tok.lower()
        if t in _STOPWORDS:
            continue
        if len(t) < 2:
            continue
        out.append(t)
    return out


def _corpus_hash(cell_ids: Sequence[int],
                 texts: Sequence[Optional[str]]) -> str:
    """SHA-256 over (id, text-length) pairs. Cheap to compute on 5.7M
    cells (~seconds), strong enough to catch any add/remove/rewrite."""
    h = hashlib.sha256()
    for cid, text in zip(cell_ids, texts):
        h.update(str(int(cid)).encode("ascii"))
        h.update(b"\x00")
        h.update(str(len(text) if text else 0).encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass(frozen=True)
class LexicalIndexManifest:
    """On-disk metadata; serialized as JSON next to ``index.pkl``."""
    corpus_hash: str
    tokenizer_version: str
    n_docs: int
    k1: float
    b: float
    created_at: float
    source: str  # human-readable provenance hint (e.g. bank path + limit)

    def as_dict(self) -> dict:
        return {
            "corpus_hash": self.corpus_hash,
            "tokenizer_version": self.tokenizer_version,
            "n_docs": int(self.n_docs),
            "k1": float(self.k1),
            "b": float(self.b),
            "created_at": float(self.created_at),
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LexicalIndexManifest":
        return cls(
            corpus_hash=str(d["corpus_hash"]),
            tokenizer_version=str(d["tokenizer_version"]),
            n_docs=int(d["n_docs"]),
            k1=float(d["k1"]),
            b=float(d["b"]),
            created_at=float(d["created_at"]),
            source=str(d.get("source", "")),
        )


class LexicalIndex:
    """In-memory BM25 index over a fixed snapshot of bank source texts.

    The index is built once via :meth:`build_from_texts` (or the
    higher-level :meth:`build_from_bank`) and persisted to disk via
    :meth:`save`. Production callers load via :meth:`load`, passing the
    same corpus snapshot, and the load fails closed on stale corpus.
    """

    MANIFEST_NAME = "manifest.json"
    INDEX_NAME = "index.pkl"

    def __init__(self) -> None:
        self._bm25 = None  # rank_bm25.BM25Okapi, set by build/load
        self._cell_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._manifest: Optional[LexicalIndexManifest] = None

    # ---- introspection ----

    @property
    def n_docs(self) -> int:
        return int(self._cell_ids.shape[0])

    @property
    def manifest(self) -> Optional[LexicalIndexManifest]:
        return self._manifest

    @property
    def cell_ids(self) -> np.ndarray:
        """Read-only view of the cell id array (parallel to BM25 doc index)."""
        return self._cell_ids

    # ---- build ----

    def build_from_texts(
        self,
        cell_ids: Sequence[int],
        texts: Sequence[Optional[str]],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        source: str = "",
    ) -> None:
        """Build the BM25 index over the given (cell_id, text) pairs.

        Empty / None texts are indexed as empty token lists so the cell
        id is still present; BM25 will rank them at the bottom for any
        non-empty query, which is the correct behaviour (no text = no
        lexical signal). The cell id mapping is preserved so a caller
        passing in tombstones can mask them out at query time.
        """
        from rank_bm25 import BM25Okapi  # lazy: kept out of import path

        if len(cell_ids) != len(texts):
            raise ValueError(
                f"cell_ids ({len(cell_ids)}) and texts ({len(texts)}) "
                f"must be parallel"
            )
        ids = np.asarray([int(c) for c in cell_ids], dtype=np.int64)
        tokenized: List[List[str]] = [tokenize(t) for t in texts]
        # BM25Okapi crashes on a fully empty corpus (avgdl divides by 0).
        # Two defenses: (a) require at least one doc; (b) ensure at least
        # one token globally — inject a sentinel if not. The sentinel can
        # never be retrieved because it isn't a real query term.
        if not tokenized:
            raise ValueError("cannot build lexical index over empty corpus")
        if not any(tokenized):
            tokenized[0] = ["__lex_empty_sentinel__"]
        bm25 = BM25Okapi(tokenized, k1=float(k1), b=float(b))
        self._bm25 = bm25
        self._cell_ids = ids
        self._manifest = LexicalIndexManifest(
            corpus_hash=_corpus_hash(cell_ids, texts),
            tokenizer_version=TOKENIZER_VERSION,
            n_docs=int(len(cell_ids)),
            k1=float(k1),
            b=float(b),
            created_at=time.time(),
            source=str(source),
        )

    def build_from_bank(
        self,
        bank: "StreamingBank",
        *,
        limit: Optional[int] = None,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        progress_callback=None,
    ) -> None:
        """Build over the bank's cells in id order.

        ``limit`` caps the corpus to the first N cells by id (for Phase
        1 prove-out runs); production builds pass ``limit=None``.
        Tombstoned ids are excluded — there is no point indexing a cell
        the bank will never return. Source texts are bulk-fetched in
        chunks so peak RAM stays bounded.
        """
        ids_all = [int(c) for c in bank.cell_ids
                   if int(c) not in bank._tombstoned_ids]
        if limit is not None:
            ids_all = ids_all[: int(limit)]

        # SQLite's default SQLITE_LIMIT_VARIABLE_NUMBER is 999 on older
        # builds and 32766 on 3.32+. fetch_source_texts uses an IN-list
        # with one placeholder per id, so the chunk size must stay below
        # the limit of whichever SQLite the runtime is linked against.
        # 900 is a portable safe value.
        chunk = 900
        texts: List[Optional[str]] = []
        for i in range(0, len(ids_all), chunk):
            slab = ids_all[i: i + chunk]
            texts.extend(bank.fetch_source_texts(slab))
            if progress_callback is not None:
                try:
                    progress_callback({
                        "loaded": min(i + chunk, len(ids_all)),
                        "total": len(ids_all),
                    })
                except Exception:
                    pass
        self.build_from_texts(
            ids_all, texts,
            k1=k1, b=b,
            source=f"streaming_bank:{bank.db_path}"
                   f"{f' limit={limit}' if limit is not None else ''}",
        )

    # ---- query ----

    def topk(
        self,
        query: str,
        k: int = 200,
        *,
        excluded_ids: Optional[Iterable[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(cell_ids, scores)`` for the top ``k`` BM25 docs.

        Both arrays are length ``min(k, n_docs)`` ordered by descending
        score. An empty / all-stopword query yields zero hits (returns
        empty arrays); the cascade caller will then fall back to the
        dense bank's ``topk``. ``excluded_ids`` (e.g. tombstones added
        after build) are masked out before ranking.
        """
        if self._bm25 is None:
            raise RuntimeError("lexical index not built/loaded")
        tokens = tokenize(query)
        if not tokens:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )
        scores = self._bm25.get_scores(tokens)
        if excluded_ids:
            ex = {int(c) for c in excluded_ids}
            if ex:
                mask = np.fromiter(
                    (int(c) not in ex for c in self._cell_ids),
                    dtype=bool, count=self.n_docs,
                )
                scores = np.where(mask, scores, -np.inf)
        k_eff = min(int(k), self.n_docs)
        if k_eff <= 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )
        # Partial sort: argpartition + final sort over the top-k slice.
        # BM25 score 0 is meaningful (no matching term) and we want to
        # drop it; -inf survived the mask above, so filter both here.
        if k_eff < scores.shape[0]:
            idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
        else:
            idx = np.arange(scores.shape[0])
        idx = idx[np.argsort(-scores[idx])]
        sel_scores = scores[idx]
        # Drop zero / -inf scores. A doc with score 0 shares no terms with
        # the query and should not be ranked.
        keep = sel_scores > 0.0
        return (
            self._cell_ids[idx][keep].astype(np.int64, copy=False),
            sel_scores[keep].astype(np.float32, copy=False),
        )

    # ---- persistence ----

    def save(self, directory: str | Path) -> None:
        """Persist manifest + pickled BM25 to ``directory``."""
        if self._bm25 is None or self._manifest is None:
            raise RuntimeError("nothing to save: index not built")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / self.MANIFEST_NAME).write_text(
            json.dumps(self._manifest.as_dict(), indent=2),
            encoding="utf-8",
        )
        with (d / self.INDEX_NAME).open("wb") as fh:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "cell_ids": self._cell_ids,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_corpus_hash: Optional[str] = None,
    ) -> "LexicalIndex":
        """Load a persisted index. If ``expected_corpus_hash`` is set
        and does not match the manifest, raises :class:`LexicalIndexStale`."""
        d = Path(directory)
        manifest = LexicalIndexManifest.from_dict(
            json.loads((d / cls.MANIFEST_NAME).read_text(encoding="utf-8"))
        )
        if manifest.tokenizer_version != TOKENIZER_VERSION:
            raise LexicalIndexStale(
                f"tokenizer version mismatch: index built with "
                f"{manifest.tokenizer_version!r}, current is "
                f"{TOKENIZER_VERSION!r}"
            )
        if (expected_corpus_hash is not None
                and manifest.corpus_hash != expected_corpus_hash):
            raise LexicalIndexStale(
                f"corpus hash mismatch: index built over a different "
                f"snapshot (manifest {manifest.corpus_hash[:12]}…, "
                f"expected {expected_corpus_hash[:12]}…)"
            )
        with (d / cls.INDEX_NAME).open("rb") as fh:
            payload = pickle.load(fh)
        obj = cls()
        obj._bm25 = payload["bm25"]
        obj._cell_ids = np.asarray(payload["cell_ids"], dtype=np.int64)
        obj._manifest = manifest
        return obj


__all__ = [
    "LexicalIndex",
    "LexicalIndexManifest",
    "LexicalIndexStale",
    "TOKENIZER_VERSION",
    "DEFAULT_K1",
    "DEFAULT_B",
    "tokenize",
]
