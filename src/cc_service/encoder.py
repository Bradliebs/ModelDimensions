"""Encoder wrapper for the memory service.

Loads the sentence encoder once at process start and serves embeddings from
memory. Lazy-loaded so that import-time is fast (sentence-transformers loads in
~1-3s).

Some retrieval-tuned encoders (BGE, E5) require asymmetric query/passage
instruction prefixes to perform — the query side gets a short instruction and
the passage side gets none (BGE) or a different tag (E5). Encoding the query
without the prefix silently degrades recall, so the prefixes are wired here per
model and the caller flags whether a text is a query or a passage.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


# Known embedding dims so we don't force a model load just to report the dim.
_KNOWN_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
}

# Per-model (query_prefix, passage_prefix). Empty strings = no prefix.
_MODEL_PREFIXES = {
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
}


class EncoderSingleton:
    """One sentence-encoder instance per process."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy-loaded on first encode
        self.query_prefix, self.passage_prefix = _MODEL_PREFIXES.get(
            model_name, ("", "")
        )

    @property
    def dim(self) -> int:
        if self.model_name in _KNOWN_DIMS:
            return _KNOWN_DIMS[self.model_name]
        # Lazy fallback: load and query
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.model_name, device=dev)

    def encode(self, texts: List[str], batch_size: int = 32,
               is_query: bool = False) -> np.ndarray:
        """Encode a batch of texts to (N, D) float32.

        Pass is_query=True for search queries so the query-side instruction
        prefix is applied (a no-op for models without prefixes, e.g. MiniLM).
        """
        self._ensure_loaded()
        prefix = self.query_prefix if is_query else self.passage_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we whiten ourselves; raw output preferred
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        """Convenience: encode a single text, return (D,) float32."""
        return self.encode([text], is_query=is_query)[0]
