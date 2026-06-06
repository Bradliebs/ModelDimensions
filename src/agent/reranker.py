"""Cross-encoder reranker — second-stage scoring of (query, cell-text) pairs.

The bi-encoder retrieval in :mod:`src.agent.streaming_bank` is fast but
operates on independent embeddings: query and cell are encoded
separately and compared by cosine similarity. When several cells embed
to nearly the same point in the encoder's space the bi-encoder cannot
tell them apart (the ``top-1 - top-2`` margin collapses, which is
exactly what trips the V1 silence gate on the encoder-ambiguous knowns
identified in `experiments/exp18_v1_pipeline_questions_eval.py`).

A cross-encoder reads ``(query, passage)`` jointly through one
transformer pass and emits a single scalar relevance score. It is
slower per-pair than a bi-encoder cosine (per-query cost is O(k * fwd)
vs. one matmul) but more discriminating because self-attention spans
both texts. We use it as a re-ranker over the top-k bi-encoder
candidates only — never as a primary retrieval mechanism.

Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — 80 MB, trained
on MS MARCO passage ranking, the canonical light cross-encoder. Outputs
unbounded logits; on this task they typically sit in roughly
``[-10, +10]`` so margins are on a *different scale* from the cosine
activations the silence gate was tuned for (``DEFAULT_MARGIN_THRESHOLD
= 0.03``). Callers that gate on rerank scores must pick their own
threshold; see `experiments/exp19_reranker_sweep.py` for calibration.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """One :class:`sentence_transformers.CrossEncoder` per process."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL,
                 device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy-loaded on first :meth:`score`

    def _ensure_loaded(self) -> None:
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CrossEncoder(self.model_name, device=dev)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Score each ``(query, passage)`` pair; returns ``(N,)`` float32.

        Order of the returned array matches ``passages``. Empty
        ``passages`` returns an empty array without loading the model.
        """
        if not passages:
            return np.zeros((0,), dtype=np.float32)
        self._ensure_loaded()
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)


__all__ = ["CrossEncoderReranker", "DEFAULT_RERANKER_MODEL"]
