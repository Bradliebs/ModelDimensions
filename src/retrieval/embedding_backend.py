"""Offline, deterministic token-hashing embedder for hybrid retrieval tests.

This embedder lets the hybrid backend exercise its *semantic* code path with
zero downloads and fully reproducible output. It is **not** a neural model: it
hashes word tokens into a fixed-width vector (a signed bag-of-words / hashing
trick). Two texts that share tokens land close together under cosine
similarity, so paraphrases that reuse vocabulary score highly — but genuine
synonyms with no shared tokens do not. Treat it as a deterministic stand-in for
a real embedding model, useful for offline tests and as a lightweight default;
reach for a true model (e.g. MiniLM) when real semantics matter.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .scoring import tokenize

__all__ = ["OfflineHashingEmbedder"]


class OfflineHashingEmbedder:
    """Deterministic ``encode_one`` implementing the EncoderProtocol.

    Each token is hashed to a bucket index and a sign; occurrences accumulate
    into a dense vector. The vector is returned un-normalised — the semantic
    backend L2-normalises before computing cosine similarity, so the only thing
    that matters here is direction, which is a deterministic function of the
    token multiset.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim

    def encode_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            v[bucket] += sign
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            # Empty / all-stopword text: return a tiny non-zero constant vector
            # so downstream normalisation does not divide by zero.
            v = np.full(self.dim, 1e-6, dtype=np.float32)
        return v
