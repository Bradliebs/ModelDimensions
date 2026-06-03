"""Opt-in hybrid knowledge-retrieval backend (v2.2).

This package layers a *hybrid* retrieval backend on top of the frozen v1.x
retrieval core in :mod:`agent.retrieval_backends`. It never reimplements the
concept-cell geometry or the embedding model; it *composes* the existing
deterministic and (optional) semantic backends and blends their scores with a
keyword-overlap signal.

Nothing here is mandatory. The default backend stays deterministic and offline;
the hybrid backend is enabled only by explicit flag/config and degrades safely
to the deterministic path when no semantic component is available.
"""
