"""SLM orchestration layer for Concept Cells (v0.8).

This package adds an *auditable* small-language-model controller around the
frozen concept-cell memory. The SLM never stores facts itself: it only routes
intent, compresses text for memory writes, and proposes bindings. All factual
answers are produced from fired memory cells by the response policy, never by
the SLM directly.

Nothing in this package modifies the core concept-cell geometry, whitening,
scaling, write, query, or Oja-binding behaviour.
"""
