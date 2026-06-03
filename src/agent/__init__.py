"""Agent orchestration layer for the concept-cell memory (v0.8).

This package wires together three things that are deliberately kept separate:

  1. the (untrusted) SLM controller, which classifies intent and proposes text;
  2. the (trusted) concept-cell memory bank, which is the only source of facts;
  3. the response policy, which guarantees every answer is grounded in fired
     cells or is an explicit refusal.

The SLM layer is optional and injectable. Pass ``controller=None`` to run the
memory bank with no language model at all.
"""
