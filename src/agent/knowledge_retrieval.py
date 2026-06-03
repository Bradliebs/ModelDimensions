"""Retrieval over imported knowledge chunks (v1.3, backend-pluggable in v1.4).

This is the knowledge-side parallel to ``candidate_retrieval``: it ranks chunks
over the **knowledge library**, never the memory bank. Keeping a separate code
path guarantees imported knowledge and project memory cannot cross-contaminate:
a knowledge query can only ever return knowledge chunks, and a memory query can
only ever return memories.

v1.4 moves the actual ranking behind a :class:`RetrievalBackend` (see
``retrieval_backends``). The default backend is the deterministic, offline one
that reproduces v1.3 behaviour exactly; an optional semantic backend can be
supplied for meaning-based retrieval. This module keeps the small wrapper
``retrieve_knowledge`` so existing callers are unchanged, and enriches each
returned candidate with its source ``version`` when a lookup is provided.

``KnowledgeCandidate`` is defined in ``retrieval_backends`` and re-exported here
so existing imports (``from agent.knowledge_retrieval import KnowledgeCandidate``)
keep working.
"""
from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

from .knowledge_sources import KnowledgeChunk
from .orchestrator import EncoderProtocol
from .retrieval_backends import (
    DeterministicRetrievalBackend,
    KnowledgeCandidate,
    RetrievalBackend,
)

__all__ = ["KnowledgeCandidate", "retrieve_knowledge"]

_RADIUS = 0.9


def retrieve_knowledge(encoder: EncoderProtocol,
                       chunks: Sequence[KnowledgeChunk],
                       query_text: str,
                       k: int = 5,
                       radius: float = _RADIUS,
                       *,
                       backend: Optional[RetrievalBackend] = None,
                       source_versions: Optional[Mapping[str, str]] = None
                       ) -> List[KnowledgeCandidate]:
    """Return the top-``k`` knowledge chunks for ``query_text``.

    When no ``backend`` is given, a :class:`DeterministicRetrievalBackend` is
    used, reproducing v1.3 ranking exactly. Passing only the active chunks is
    how a deleted source is excluded. ``source_versions`` (source_id -> version)
    enriches each candidate's ``version`` field for the query audit.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not chunks:
        return []

    if backend is None:
        backend = DeterministicRetrievalBackend(encoder, radius=radius)

    backend.index_chunks(chunks)
    candidates = backend.retrieve(query_text, top_k=k)

    if source_versions:
        for cand in candidates:
            cand.version = source_versions.get(cand.source_id)
    return candidates
