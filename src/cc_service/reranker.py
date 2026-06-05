"""Cross-encoder reranker for the memory service.

Stage 2 of two-stage retrieval. The bank (bi-encoder) provides a recall pool;
this cross-encoder rescores each (query, candidate_text) pair jointly, which is
far more precise than the bi-encoder dot product but too expensive to run over
the whole bank — hence it only sees the top candidates from stage 1.

Lazy-loaded so import stays cheap and the model is only pulled when a query
actually asks for reranking.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class RerankerSingleton:
    """One cross-encoder instance per process."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base",
                 device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy-loaded on first rerank

    def _ensure_loaded(self) -> None:
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CrossEncoder(self.model_name, device=dev)

    def score(self, query: str, candidates: List[str],
              batch_size: int = 32) -> np.ndarray:
        """Return a relevance score per candidate (higher = more relevant).

        Scores are model-raw logits; only their relative order is meaningful.
        """
        if not candidates:
            return np.zeros((0,), dtype=np.float32)
        self._ensure_loaded()
        pairs = [(query, c) for c in candidates]
        scores = self._model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)
