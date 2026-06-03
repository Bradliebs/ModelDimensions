"""Strict local-SLM controller that only ever emits validated decisions.

This controller wraps a *text-generating* local SLM and forces its output
through a Pydantic gate. The SLM is asked to return JSON describing a memory
decision; the controller then:

  * extracts and parses the JSON,
  * validates it against the schema,
  * retries exactly once on malformed/invalid output,
  * falls back to a safe intent (QUERY or UNKNOWN) if both attempts fail,
  * never touches the memory bank.

The actual model is reached through an injectable ``GenerationBackend`` so the
whole thing is testable with a fake backend and no model download. A lazy
``TransformersBackend`` is provided for real local models.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ValidationError

from .controller import SLMController
from .schemas import (
    IntentType,
    MemoryBindCandidate,
    MemoryIntent,
    MemoryWriteCandidate,
    SLMControllerDecision,
)


# ---------- generation backend ----------

@runtime_checkable
class GenerationBackend(Protocol):
    """Anything that turns a prompt string into raw model text."""

    def generate(self, prompt: str) -> str:
        ...


class TransformersBackend:
    """Lazy local text-generation backend.

    Loaded only on first ``generate`` so importing this module never triggers
    a download. Model name is a parameter, not hard-coded.
    """

    def __init__(self, model_name: str, device: Optional[str] = None,
                 max_new_tokens: int = 256):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - only with extras
            raise RuntimeError(
                "TransformersBackend requires 'transformers' and 'torch'."
            ) from exc
        self._pipe = pipeline(
            "text-generation", model=self.model_name, device_map=self.device
        )

    def generate(self, prompt: str) -> str:  # pragma: no cover - needs a model
        self._ensure_loaded()
        out = self._pipe(prompt, max_new_tokens=self.max_new_tokens,
                        do_sample=False, return_full_text=False)
        return out[0]["generated_text"]


# ---------- diagnostics ----------

@dataclass
class ControllerStats:
    """Aggregate counters for auditing controller behaviour."""

    calls: int = 0
    valid_first_attempt: int = 0
    retried: int = 0
    recovered_on_retry: int = 0
    fallback_used: int = 0
    malformed: int = 0  # unusable after retry -> fallback


@dataclass
class LastCall:
    """Per-call diagnostic snapshot for tests and experiments."""

    raw_attempts: List[str] = field(default_factory=list)
    used_fallback: bool = False
    retried: bool = False
    valid_first_attempt: bool = False


# ---------- raw decision schema (the SLM's claimed output) ----------

class _RawDecision(BaseModel):
    """Loose schema for parsing the SLM's JSON before mapping to the strict one."""

    intent: str
    confidence: float = 0.5
    rationale: str = ""
    query_text: Optional[str] = None
    canonical_text: Optional[str] = None
    tags: List[str] = []
    suggested_epsilon: Optional[float] = None
    delete_memory_id: Optional[str] = None
    bind_memory_ids: Optional[List[str]] = None
    bound_group_id: Optional[str] = None


def _extract_json(text: str) -> Optional[str]:
    """Pull the first balanced-looking JSON object out of model text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


# ---------- the controller ----------

_VALID_INTENTS = {i.value for i in IntentType}


class LocalJSONController(SLMController):
    """SLM controller that yields only schema-validated decisions."""

    def __init__(self, model_name: str,
                 backend: Optional[GenerationBackend] = None,
                 fallback_intent: IntentType = IntentType.QUERY):
        if fallback_intent not in (IntentType.QUERY, IntentType.UNKNOWN):
            raise ValueError("fallback_intent must be QUERY or UNKNOWN")
        self.model_name = model_name
        self.backend = backend or TransformersBackend(model_name)
        self.fallback_intent = fallback_intent
        self.stats = ControllerStats()
        self.last_call = LastCall()

    # -- public interface --

    def classify_intent(self, user_text: str) -> SLMControllerDecision:
        self.stats.calls += 1
        self.last_call = LastCall()

        raw1 = self._generate(self._intent_prompt(user_text))
        self.last_call.raw_attempts.append(raw1)
        decision = self._parse_decision(raw1, user_text)
        if decision is not None:
            self.stats.valid_first_attempt += 1
            self.last_call.valid_first_attempt = True
            return decision

        # Retry exactly once with a stricter reminder.
        self.stats.retried += 1
        self.last_call.retried = True
        raw2 = self._generate(self._intent_prompt(user_text, retry=True))
        self.last_call.raw_attempts.append(raw2)
        decision = self._parse_decision(raw2, user_text)
        if decision is not None:
            self.stats.recovered_on_retry += 1
            return decision

        # Both attempts failed: fall back safely.
        self.stats.malformed += 1
        self.stats.fallback_used += 1
        self.last_call.used_fallback = True
        return self._fallback_decision(user_text)

    def compress_for_memory(self, user_text: str) -> MemoryWriteCandidate:
        raw = self._generate(self._compress_prompt(user_text))
        parsed = self._parse_raw(raw)
        if parsed is not None and parsed.canonical_text:
            try:
                return MemoryWriteCandidate(
                    canonical_text=parsed.canonical_text,
                    source="user",
                    tags=list(parsed.tags),
                    suggested_epsilon=parsed.suggested_epsilon,
                )
            except ValidationError:
                pass
        # Safe fallback: trivial canonicalisation, never empty.
        canonical = user_text.strip() or "(empty)"
        return MemoryWriteCandidate(canonical_text=canonical, source="user")

    def suggest_bindings(
        self, memory_items: Sequence[dict]
    ) -> Optional[MemoryBindCandidate]:
        valid_ids = [m["memory_id"] for m in memory_items if m.get("memory_id")]
        if len(valid_ids) < 2:
            return None
        raw = self._generate(self._bind_prompt(memory_items))
        parsed = self._parse_raw(raw)
        if parsed is None or not parsed.bind_memory_ids or not parsed.bound_group_id:
            return None
        # Never fabricate: keep only ids that actually exist in the bank.
        valid_set = set(valid_ids)
        kept = [m for m in parsed.bind_memory_ids if m in valid_set]
        if len(kept) < 2:
            return None
        try:
            return MemoryBindCandidate(
                memory_ids=kept,
                bound_group_id=parsed.bound_group_id,
                rationale=parsed.rationale or "local SLM binding proposal",
            )
        except ValidationError:
            return None

    # -- internals --

    def _generate(self, prompt: str) -> str:
        return self.backend.generate(prompt)

    def _parse_raw(self, raw: str) -> Optional[_RawDecision]:
        blob = _extract_json(raw)
        if blob is None:
            return None
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        try:
            return _RawDecision(**data)
        except ValidationError:
            return None

    def _parse_decision(self, raw: str,
                        user_text: str) -> Optional[SLMControllerDecision]:
        parsed = self._parse_raw(raw)
        if parsed is None:
            return None
        intent_str = (parsed.intent or "").strip().lower()
        if intent_str not in _VALID_INTENTS:
            return None
        intent_type = IntentType(intent_str)
        try:
            intent = MemoryIntent(
                intent=intent_type,
                confidence=max(0.0, min(1.0, float(parsed.confidence))),
                rationale=parsed.rationale or "",
            )
        except (ValidationError, ValueError, TypeError):
            return None

        decision = SLMControllerDecision(intent=intent)

        if intent_type == IntentType.QUERY:
            decision.query_text = parsed.query_text or user_text
        elif intent_type == IntentType.WRITE:
            if not parsed.canonical_text:
                return None  # write with nothing to store is invalid
            try:
                decision.write_candidate = MemoryWriteCandidate(
                    canonical_text=parsed.canonical_text,
                    source="user",
                    tags=list(parsed.tags),
                    suggested_epsilon=parsed.suggested_epsilon,
                )
            except ValidationError:
                return None
        elif intent_type == IntentType.DELETE:
            decision.delete_memory_id = parsed.delete_memory_id
        elif intent_type == IntentType.BIND:
            if (parsed.bind_memory_ids and len(parsed.bind_memory_ids) >= 2
                    and parsed.bound_group_id):
                try:
                    decision.bind_candidate = MemoryBindCandidate(
                        memory_ids=list(parsed.bind_memory_ids),
                        bound_group_id=parsed.bound_group_id,
                        rationale=parsed.rationale or "",
                    )
                except ValidationError:
                    decision.bind_candidate = None
        return decision

    def _fallback_decision(self, user_text: str) -> SLMControllerDecision:
        if self.fallback_intent == IntentType.UNKNOWN or not user_text.strip():
            intent = MemoryIntent(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                rationale="model output unusable after retry; fell back",
            )
            return SLMControllerDecision(intent=intent)
        intent = MemoryIntent(
            intent=IntentType.QUERY,
            confidence=0.2,
            rationale="model output unusable after retry; fell back to recall",
        )
        return SLMControllerDecision(intent=intent, query_text=user_text)

    # -- prompts (kept simple; the contract is the JSON, not the wording) --

    @staticmethod
    def _intent_prompt(user_text: str, retry: bool = False) -> str:
        reminder = (
            "Your previous answer was not valid JSON. Reply with ONLY a JSON "
            "object and nothing else.\n" if retry else ""
        )
        return (
            f"{reminder}You are a memory controller. Classify the user message "
            "into one of: query, write, bind, delete, unknown. Respond with a "
            'JSON object: {"intent": ..., "confidence": 0-1, "rationale": ..., '
            '"query_text": ..., "canonical_text": ..., "delete_memory_id": ..., '
            '"bind_memory_ids": ..., "bound_group_id": ...}.\n'
            f"User message: {user_text}\nJSON:"
        )

    @staticmethod
    def _compress_prompt(user_text: str) -> str:
        return (
            "Compress the following into a single canonical fact. Respond with "
            'JSON {"intent": "write", "canonical_text": ...}.\n'
            f"Text: {user_text}\nJSON:"
        )

    @staticmethod
    def _bind_prompt(memory_items: Sequence[dict]) -> str:
        listing = "; ".join(
            f"{m.get('memory_id')}={m.get('canonical_text', '')}"
            for m in memory_items
        )
        return (
            "Propose a binding over related memories. Respond with JSON "
            '{"intent": "bind", "bind_memory_ids": [...], "bound_group_id": ...}.'
            f"\nMemories: {listing}\nJSON:"
        )
