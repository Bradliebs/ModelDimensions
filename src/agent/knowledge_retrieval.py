"""Retrieval over imported knowledge chunks (v1.3).

This is the knowledge-side parallel to ``candidate_retrieval``: it ranks chunks
by the same activation the memory bank would compute (unit chunk vector dotted
with the radius-scaled query), but it runs over the **knowledge library**, never
the memory bank. Keeping a separate code path guarantees imported knowledge and
project memory cannot cross-contaminate: a knowledge query can only ever return
knowledge chunks, and a memory query can only ever return memories.

It reads the chunk vectors through the shared encoder so ranking is consistent
with the rest of the system; it does not mutate any bank or change any geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .knowledge_sources import KnowledgeChunk
from .orchestrator import EncoderProtocol

_RADIUS = 0.9


@dataclass
class KnowledgeCandidate:
    """One retrieved knowledge chunk with its provenance and activation."""

    source_id: str
    chunk_id: str
    activation: float
    rank: int
    source_name: str
    domain: str
    authority: str
    chunk_text: str
    source_section: str | None = None


def retrieve_knowledge(encoder: EncoderProtocol,
                       chunks: Sequence[KnowledgeChunk],
                       query_text: str,
                       k: int = 5,
                       radius: float = _RADIUS) -> List[KnowledgeCandidate]:
    """Return the top-``k`` knowledge chunks for ``query_text`` by activation.

    The activation reproduces the memory bank's firing math exactly: each chunk
    vector is unit-normalised to a cell weight and the query is scaled to
    ``radius``; ``activation = w . q``. Only the supplied chunks are considered,
    so passing active chunks only is how a deleted source is excluded.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not chunks:
        return []

    q_raw = np.asarray(encoder.encode_one(query_text), dtype=np.float64)
    qn = np.linalg.norm(q_raw)
    if qn < 1e-12:
        return []
    q = q_raw / qn * radius

    scored: List[tuple[float, KnowledgeChunk]] = []
    for chunk in chunks:
        x = np.asarray(encoder.encode_one(chunk.chunk_text), dtype=np.float64)
        xn = np.linalg.norm(x)
        if xn < 1e-12:
            continue
        w = x / xn
        scored.append((float(w @ q), chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    candidates: List[KnowledgeCandidate] = []
    for rank, (act, chunk) in enumerate(scored[:k], start=1):
        candidates.append(KnowledgeCandidate(
            source_id=chunk.source_id,
            chunk_id=chunk.chunk_id,
            activation=round(act, 6),
            rank=rank,
            source_name=chunk.source_name,
            domain=chunk.domain.value,
            authority=chunk.authority.value,
            chunk_text=chunk.chunk_text,
            source_section=chunk.source_section,
        ))
    return candidates
