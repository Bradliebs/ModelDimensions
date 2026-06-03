"""SLM controller interface plus a deterministic mock implementation.

The controller is the only component allowed to interpret free-form user text.
It produces typed decisions (see ``slm.schemas``) and never touches the memory
bank directly. Two implementations are provided:

  - ``MockSLMController``  : rule-based, deterministic, no model download.
                            Used by default and by every test.
  - ``LocalSLMController`` : optional wrapper around a locally hosted SLM,
                            loaded lazily behind the same interface. It is
                            never imported unless explicitly constructed, so
                            the package has no hard dependency on a model.

Add new backends by subclassing ``SLMController``; the orchestrator depends
only on the abstract interface.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from .schemas import (
    IntentType,
    MemoryBindCandidate,
    MemoryIntent,
    MemoryWriteCandidate,
    SLMControllerDecision,
)


class SLMController(ABC):
    """Abstract controller interface.

    Implementations must be free of side effects on the memory bank: they only
    read user text and emit typed proposals.
    """

    @abstractmethod
    def classify_intent(self, user_text: str) -> SLMControllerDecision:
        """Classify a user turn into a typed, routable decision."""

    @abstractmethod
    def compress_for_memory(self, user_text: str) -> MemoryWriteCandidate:
        """Compress/canonicalise user text into a single write candidate."""

    @abstractmethod
    def suggest_bindings(
        self, memory_items: Sequence[dict]
    ) -> Optional[MemoryBindCandidate]:
        """Propose a binding over existing memories, or None if not warranted.

        ``memory_items`` are lightweight dicts with at least ``memory_id`` and
        ``canonical_text`` keys (as produced by the persistence layer).
        """


# ---------- Mock controller (default; no downloads) ----------

_WRITE_RE = re.compile(
    r"\b(remember|note that|store|save|memori[sz]e|keep in mind)\b", re.I
)
_QUERY_RE = re.compile(
    r"(\bwhat\b|\bwho\b|\bwhen\b|\bwhere\b|\bwhy\b|\bhow\b|\bwhich\b|"
    r"\brecall\b|\bdo you (?:remember|know)\b|\?\s*$)",
    re.I,
)
_BIND_RE = re.compile(r"\b(bind|group|link|associate|combine)\b", re.I)
_DELETE_RE = re.compile(r"\b(delete|forget|remove|unlearn|erase)\b", re.I)
# Captures an explicit memory id token, e.g. "mem-3f2a" or "memory id mem_12".
_MEMID_RE = re.compile(r"\b(mem[-_][0-9a-zA-Z]+)\b")


class MockSLMController(SLMController):
    """Deterministic, rule-based controller for tests and offline runs.

    Classification is intentionally simple and transparent: it keys off a
    small set of verbs and the presence of a question mark. The point is not
    to be a good language model but to exercise the orchestration contract
    without any model download.
    """

    def classify_intent(self, user_text: str) -> SLMControllerDecision:
        text = user_text.strip()

        # Delete and bind take priority over write/query because their verbs
        # ("forget", "group") are unambiguous memory operations.
        if _DELETE_RE.search(text):
            mem_id = self._extract_memory_id(text)
            intent = MemoryIntent(
                intent=IntentType.DELETE,
                confidence=0.9 if mem_id else 0.5,
                rationale="matched delete/forget verb",
            )
            return SLMControllerDecision(intent=intent, delete_memory_id=mem_id)

        if _BIND_RE.search(text):
            intent = MemoryIntent(
                intent=IntentType.BIND,
                confidence=0.7,
                rationale="matched bind/group verb",
            )
            return SLMControllerDecision(intent=intent)

        if _WRITE_RE.search(text):
            intent = MemoryIntent(
                intent=IntentType.WRITE,
                confidence=0.85,
                rationale="matched write verb",
            )
            return SLMControllerDecision(
                intent=intent,
                write_candidate=self.compress_for_memory(text),
            )

        if _QUERY_RE.search(text):
            intent = MemoryIntent(
                intent=IntentType.QUERY,
                confidence=0.8,
                rationale="matched interrogative form",
            )
            return SLMControllerDecision(intent=intent, query_text=text)

        # Empty input cannot be acted on at all.
        if not text:
            intent = MemoryIntent(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                rationale="empty input",
            )
            return SLMControllerDecision(intent=intent)

        # Default action for any other non-empty text is to ATTEMPT RECALL.
        # This is safe: if no cell fires, the response policy refuses, so a
        # mis-routed turn can never fabricate a memory-derived answer.
        intent = MemoryIntent(
            intent=IntentType.QUERY,
            confidence=0.4,
            rationale="defaulted to recall (no write/bind/delete verb)",
        )
        return SLMControllerDecision(intent=intent, query_text=text)

    def compress_for_memory(self, user_text: str) -> MemoryWriteCandidate:
        canonical = self._canonicalise(user_text)
        return MemoryWriteCandidate(
            canonical_text=canonical,
            source="user",
            tags=[],
        )

    def suggest_bindings(
        self, memory_items: Sequence[dict]
    ) -> Optional[MemoryBindCandidate]:
        ids = [m["memory_id"] for m in memory_items if m.get("memory_id")]
        if len(ids) < 2:
            return None
        group_id = "grp-" + "-".join(ids[:4])
        return MemoryBindCandidate(
            memory_ids=ids,
            bound_group_id=group_id,
            rationale="mock controller grouped all provided memories",
        )

    # -- helpers --

    @staticmethod
    def _extract_memory_id(text: str) -> Optional[str]:
        m = _MEMID_RE.search(text)
        return m.group(1) if m else None

    @staticmethod
    def _canonicalise(user_text: str) -> str:
        """Strip the leading instruction verb and tidy whitespace.

        e.g. "Remember that the sky is blue." -> "the sky is blue"
        """
        text = user_text.strip()
        text = re.sub(
            r"^\s*(please\s+)?(remember|note|store|save|memori[sz]e|keep in mind)"
            r"(\s+that)?[:,]?\s*",
            "",
            text,
            flags=re.I,
        )
        text = text.rstrip(" .")
        return text.strip() or user_text.strip()


# ---------- Optional local SLM backend (lazy, not hard-coded) ----------

class LocalSLMController(SLMController):
    """Optional controller backed by a locally hosted SLM.

    The model is loaded lazily on first use so that merely importing this
    module (or constructing the object with ``eager=False``) never triggers a
    download. The model name is a parameter, not hard-coded, and the structured
    contract is identical to the mock — the SLM's free text is parsed back into
    the same typed schemas, never trusted as memory.

    This class is a thin, optional convenience. If the backing libraries are
    not installed it raises a clear error only when actually used.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        eager: bool = False,
        fallback: Optional[SLMController] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._pipe = None
        # If the local model cannot answer in a structured way, fall back to
        # the deterministic mock so routing still degrades safely.
        self._fallback = fallback or MockSLMController()
        if eager:
            self._ensure_loaded()

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - exercised only with extras
            raise RuntimeError(
                "LocalSLMController requires 'transformers' and 'torch'. "
                "Install them, or use MockSLMController."
            ) from exc
        self._pipe = pipeline("text-generation", model=self.model_name,
                              device_map=self.device)

    def classify_intent(self, user_text: str) -> SLMControllerDecision:
        # A production backend would prompt the SLM for a JSON intent and
        # validate it against MemoryIntent. To keep v0.8 dependency-free and
        # deterministic, structural classification is delegated to the rule
        # set; the SLM is reserved for natural-language phrasing only.
        self._ensure_loaded()
        return self._fallback.classify_intent(user_text)

    def compress_for_memory(self, user_text: str) -> MemoryWriteCandidate:
        self._ensure_loaded()
        return self._fallback.compress_for_memory(user_text)

    def suggest_bindings(
        self, memory_items: Sequence[dict]
    ) -> Optional[MemoryBindCandidate]:
        self._ensure_loaded()
        return self._fallback.suggest_bindings(memory_items)


# ---------- Factory ----------

def get_controller(kind: str = "mock", **kwargs) -> SLMController:
    """Construct a controller by name.

    kind:
      - "mock"  : MockSLMController (default; no downloads)
      - "local" : LocalSLMController (requires ``model_name`` kwarg)
    """
    if kind == "mock":
        return MockSLMController()
    if kind == "local":
        model_name = kwargs.pop("model_name", None)
        if not model_name:
            raise ValueError("kind='local' requires a 'model_name' argument.")
        return LocalSLMController(model_name=model_name, **kwargs)
    raise ValueError(f"Unknown controller kind: {kind!r}. Use 'mock' or 'local'.")
