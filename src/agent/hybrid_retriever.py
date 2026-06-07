"""Hybrid retriever: BM25 lexical pre-filter → dense cosine scoring.

The V1 retrieval path is single-stage cosine over the full 5.7M-cell
bank. exp27 showed this has a hard failure mode for proper-noun queries
whose target cell lies outside the cosine top-50 entirely (e.g. Q2
French Connection / Don Ellis from
:mod:`experiments.exp20_multihop_probe`). A cross-encoder rerank cannot
fix it because the right cell never enters the candidate pool.

This module composes :class:`src.agent.lexical_index.LexicalIndex` with
:class:`src.agent.streaming_bank.StreamingBank`. The algorithm is
deliberately the simplest two-stage cascade that lets BM25 widen the
candidate pool without touching the gate or the verifier:

  1. BM25 top-``k_lexical`` by sparse score (default 200).
  2. The dense bank's whitened weights for those ``k_lexical`` cell ids
     are dotted with the encoded query (via ``StreamingBank.weights_for``)
     to produce per-candidate cosine activations on the same scale as
     :meth:`StreamingBank.topk`.
  3. Return top-``k_final`` by cosine, with the BM25 score and lexical
     rank carried as diagnostic side channels.

Cosine activations are the silence-gate signal of record. Lexical
scores are surfaced for diagnostics only — BM25 is too noisy a
confidence signal to fire silence on, and using it would mean two
fundamentally different distributions feeding the same threshold. The
gate stays cosine-only.

Fallback behaviour: when ``LexicalIndex.topk`` returns no hits (e.g.
all-stopword query or no lexical overlap with the corpus), the
retriever falls back to the bank's dense ``topk`` over the full
corpus. This preserves V1 behaviour on encoder-friendly queries where
lexical retrieval would only hurt.

Return shape mirrors :meth:`StreamingBank.topk` so existing pipeline
code consumes it unchanged. The two new keys (``lexical_scores``,
``lexical_ranks``) are additive — callers that don't care can ignore
them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agent.lexical_index import LexicalIndex
    from src.agent.streaming_bank import StreamingBank


DEFAULT_K_LEXICAL = 200
DEFAULT_K_FINAL = 10


class HybridRetriever:
    """Compose a sparse lexical index with the dense bank.

    The retriever is stateless other than holding references to the
    bank and the index. One instance per pipeline; ``topk`` is thread-
    compatible with the bank (single-threaded in the HTTPServer used
    by the workspace, and the bank itself is read-only after load).
    """

    def __init__(
        self,
        lexical_index: "LexicalIndex",
        bank: "StreamingBank",
    ) -> None:
        self.lexical_index = lexical_index
        self.bank = bank

    def topk(
        self,
        query: str,
        encoded_query: np.ndarray,
        *,
        k_lexical: int = DEFAULT_K_LEXICAL,
        k_final: int = DEFAULT_K_FINAL,
    ) -> dict:
        """Run the BM25 → cosine cascade and return the top ``k_final``.

        ``query`` is the raw question (for BM25 scoring).
        ``encoded_query`` is the already-whitened dense vector (so the
        caller controls the encoder identity, mirroring the pattern in
        :meth:`AnswerPipeline.ask`).

        Tombstones are honoured at both stages: the BM25 mask excludes
        tombstoned ids before ranking, and the dense fallback path uses
        :meth:`StreamingBank.topk` which already masks tombstones.

        Return dict mirrors :meth:`StreamingBank.topk` and adds:
            ``lexical_scores``: BM25 scores for the returned cells (or
                                zeros if the dense fallback fired)
            ``lexical_ranks``:  1-based BM25 ranks (or ``None`` per cell
                                in the dense fallback case)
            ``stage``:          ``"hybrid"`` or ``"dense_fallback"``
        """
        if encoded_query.shape != (self.bank.dim,):
            raise ValueError(
                f"encoded_query shape {encoded_query.shape} != "
                f"({self.bank.dim},)"
            )

        excluded = self.bank._tombstoned_ids or None
        lex_ids, lex_scores = self.lexical_index.topk(
            query, k=int(k_lexical), excluded_ids=excluded,
        )

        if lex_ids.size == 0:
            # Dense fallback. Preserves V1 behaviour for queries with no
            # lexical signal (rare on real queries, but the test path
            # exercises this branch via short stopword-only inputs).
            dense = self.bank.topk(encoded_query, k=int(k_final))
            k_eff = int(dense["cell_ids"].shape[0])
            return {
                "cell_ids": dense["cell_ids"],
                "activations": dense["activations"],
                "thetas": dense["thetas"],
                "lexical_scores": np.zeros(k_eff, dtype=np.float32),
                "lexical_ranks": [None] * k_eff,
                "stage": "dense_fallback",
            }

        # Some lexical hits may be overlay or base cells whose weights
        # are still in the bank's in-RAM array — weights_for handles
        # both. If any id is unknown to the bank the index is stale and
        # we want to fail loudly rather than silently drop it.
        weights = self.bank.weights_for(lex_ids)
        activations = (weights @ encoded_query).astype(np.float32, copy=False)
        thetas = np.fromiter(
            (float(self.bank.thetas[self.bank._id_to_row[int(c)]])
             for c in lex_ids),
            dtype=np.float32,
            count=lex_ids.shape[0],
        )

        k_eff = min(int(k_final), activations.shape[0])
        if k_eff <= 0:
            return {
                "cell_ids": np.empty(0, dtype=np.int64),
                "activations": np.empty(0, dtype=np.float32),
                "thetas": np.empty(0, dtype=np.float32),
                "lexical_scores": np.empty(0, dtype=np.float32),
                "lexical_ranks": [],
                "stage": "hybrid",
            }
        # Re-sort the lexical pool by cosine activation.
        if k_eff < activations.shape[0]:
            order = np.argpartition(-activations, k_eff - 1)[:k_eff]
        else:
            order = np.arange(activations.shape[0])
        order = order[np.argsort(-activations[order])]
        # 1-based lexical rank within the BM25 result, preserved for
        # diagnostics (exp28 wants to know whether the chosen cell was
        # the BM25 winner or a cosine-promoted runner-up).
        ranks_1based = [int(o) + 1 for o in order]
        return {
            "cell_ids": lex_ids[order].astype(np.int64, copy=False),
            "activations": activations[order],
            "thetas": thetas[order],
            "lexical_scores": lex_scores[order].astype(np.float32, copy=False),
            "lexical_ranks": ranks_1based,
            "stage": "hybrid",
        }


__all__ = [
    "HybridRetriever",
    "DEFAULT_K_LEXICAL",
    "DEFAULT_K_FINAL",
]
