"""Deterministic, non-LLM rules controller.

This is the simplest possible controller: pure keyword and pattern matching,
no model, no randomness. It exists as a baseline in the v0.9 stress tests so
that LLM-based controllers can be measured against a floor that is, by
construction, 100% JSON-valid and never malformed.

It implements the same ``SLMController`` interface as every other controller
and never touches the memory bank.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from .controller import SLMController
from .schemas import (
    IntentType,
    MemoryBindCandidate,
    MemoryIntent,
    MemoryWriteCandidate,
    SLMControllerDecision,
)

WRITE_KEYWORDS = {
    "remember", "note", "store", "save", "memorize", "memorise",
    "record", "log",
}
DELETE_KEYWORDS = {
    "delete", "forget", "remove", "erase", "unlearn", "drop",
}
BIND_KEYWORDS = {
    "bind", "group", "link", "associate", "combine", "connect", "merge",
}
QUESTION_WORDS = {
    "what", "who", "when", "where", "why", "how", "which", "whose", "whom",
}

_MEMID_RE = re.compile(r"\b(mem[-_][0-9a-zA-Z]+)\b")
_WORD_RE = re.compile(r"[a-z0-9']+")
_LEADING_WRITE_RE = re.compile(
    r"^\s*(please\s+)?(" + "|".join(sorted(WRITE_KEYWORDS)) + r")(\s+that)?[:,]?\s*",
    re.I,
)


class RulesController(SLMController):
    """Keyword/pattern controller with no language model in the loop."""

    def classify_intent(self, user_text: str) -> SLMControllerDecision:
        text = user_text.strip()
        tokens = set(_WORD_RE.findall(text.lower()))

        if tokens & DELETE_KEYWORDS:
            mem_id = self._extract_memory_id(text)
            intent = MemoryIntent(
                intent=IntentType.DELETE,
                confidence=0.9 if mem_id else 0.5,
                rationale="matched delete keyword",
            )
            return SLMControllerDecision(intent=intent, delete_memory_id=mem_id)

        if tokens & BIND_KEYWORDS:
            intent = MemoryIntent(
                intent=IntentType.BIND,
                confidence=0.7,
                rationale="matched bind keyword",
            )
            return SLMControllerDecision(intent=intent)

        if tokens & WRITE_KEYWORDS:
            intent = MemoryIntent(
                intent=IntentType.WRITE,
                confidence=0.85,
                rationale="matched write keyword",
            )
            return SLMControllerDecision(
                intent=intent,
                write_candidate=self.compress_for_memory(text),
            )

        first = ""
        m = _WORD_RE.search(text.lower())
        if m:
            first = m.group(0)
        if first in QUESTION_WORDS or text.endswith("?"):
            intent = MemoryIntent(
                intent=IntentType.QUERY,
                confidence=0.8,
                rationale="matched interrogative form",
            )
            return SLMControllerDecision(intent=intent, query_text=text)

        if text:
            # Default to recall: safe, because grounding refuses on no fire.
            intent = MemoryIntent(
                intent=IntentType.QUERY,
                confidence=0.4,
                rationale="defaulted to recall (no action keyword)",
            )
            return SLMControllerDecision(intent=intent, query_text=text)

        intent = MemoryIntent(
            intent=IntentType.UNKNOWN,
            confidence=0.0,
            rationale="empty input",
        )
        return SLMControllerDecision(intent=intent)

    def compress_for_memory(self, user_text: str) -> MemoryWriteCandidate:
        canonical = _LEADING_WRITE_RE.sub("", user_text.strip()).rstrip(" .")
        canonical = canonical.strip() or user_text.strip()
        return MemoryWriteCandidate(
            canonical_text=canonical, source="user", tags=[]
        )

    def suggest_bindings(
        self, memory_items: Sequence[dict]
    ) -> Optional[MemoryBindCandidate]:
        ids = [m["memory_id"] for m in memory_items if m.get("memory_id")]
        if len(ids) < 2:
            return None
        return MemoryBindCandidate(
            memory_ids=ids,
            bound_group_id="grp-rules-" + "-".join(ids[:4]),
            rationale="rules controller grouped all provided memories",
        )

    @staticmethod
    def _extract_memory_id(text: str) -> Optional[str]:
        m = _MEMID_RE.search(text)
        return m.group(1) if m else None
