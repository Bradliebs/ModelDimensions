"""Query routing between project memory and imported knowledge (v1.3).

A query can want different things: a record of a past decision ("what did we
agree"), an external reference ("what do the docs say"), or both. The planner is
a small, deterministic, rule-based router — it inspects the query text for
intent cues and returns a route plus, for knowledge queries, the likely domain
and whether a domain-specific caution applies (medical/legal answers are always
informational only).

It makes no network calls and runs no model; it only decides *where to look*.
The service is free to override the route when a side turns out to be silent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .knowledge_sources import KnowledgeDomain


class QueryRoute(str, Enum):
    MEMORY_ONLY = "memory_only"
    KNOWLEDGE_ONLY = "knowledge_only"
    BOTH = "both"
    GENERAL_MODEL_NOT_GROUNDED = "general_model_not_grounded"


@dataclass
class QueryPlan:
    route: QueryRoute
    domain: Optional[KnowledgeDomain]
    medical_caution: bool
    reason: str


# Phrases that signal the user is asking about a recorded decision/state.
_MEMORY_CUES = (
    "did we decide", "what did we agree", "what changed", "did we agree",
    "what did we decide", "what was decided", "our decision",
)

# Phrases that signal the user wants an external reference answer.
_KNOWLEDGE_CUES = (
    "according to docs", "according to the docs", "how do i code",
    "how do i write", "what does python", "what does pydantic",
    "what does the doc", "in the documentation", "per the docs",
)

# Medical-intent terms (route to knowledge, but with a caution).
_MEDICAL_TERMS = (
    "medical", "symptom", "symptoms", "dosage", "dose", "diagnosis",
    "diagnose", "prescription", "prescribe", "treatment", "disease",
    "patient", "mg ", "milligram",
)


def _contains_any(text: str, needles) -> bool:
    return any(n in text for n in needles)


def plan_query(query_text: str) -> QueryPlan:
    """Route ``query_text`` to memory, knowledge, both, or ungrounded model."""
    text = query_text.lower().strip()
    if not text:
        return QueryPlan(QueryRoute.GENERAL_MODEL_NOT_GROUNDED, None, False,
                         "empty query")

    has_memory = _contains_any(text, _MEMORY_CUES)
    has_medical = _contains_any(re.sub(r"\s+", " ", text + " "), _MEDICAL_TERMS)
    has_knowledge = _contains_any(text, _KNOWLEDGE_CUES)

    if has_medical:
        return QueryPlan(QueryRoute.KNOWLEDGE_ONLY, KnowledgeDomain.MEDICAL,
                         True, "medical-intent terms detected; "
                         "answer must be informational only")

    if has_memory and not has_knowledge:
        return QueryPlan(QueryRoute.MEMORY_ONLY, None, False,
                         "decision/agreement cue detected")

    if has_knowledge and not has_memory:
        return QueryPlan(QueryRoute.KNOWLEDGE_ONLY, None, False,
                         "documentation/reference cue detected")

    # Ambiguous (both cues, or neither): look in both and label sources apart.
    return QueryPlan(QueryRoute.BOTH, None, False,
                     "ambiguous intent; checking memory and knowledge "
                     "separately")
