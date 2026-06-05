"""Optional injectable SLM equivalence judge (v1.0-rc2).

The deterministic verifier (``agent.verifier``) is the trust anchor. This module
adds an *optional* second opinion from a generative model, behind a strict
Pydantic gate and one hard rule:

    the SLM can never override a deterministic REJECT.

A model that says "these mean the same thing" cannot resurrect a candidate the
deterministic rules already rejected for a number/date/entity/negation/antonym
flip. The SLM may only act on cases the deterministic layer left as ``ACCEPT``
or ``AMBIGUOUS``, and even then any malformed or invalid output falls back to
``AMBIGUOUS`` (refuse, do not ground).

The model is reached through an injectable ``GenerationBackend`` so the whole
thing is testable with a fake backend and no model download.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from slm.schemas import VerificationVerdict


@runtime_checkable
class GenerationBackend(Protocol):
    """Anything that turns a prompt string into raw model text."""

    def generate(self, prompt: str) -> str:
        ...


class _RawJudgment(BaseModel):
    """Loose schema the model output is validated against before mapping."""

    verdict: str
    reason: str = ""


class EquivalenceJudgment(BaseModel):
    """The validated, typed judgment surfaced to callers."""

    verdict: VerificationVerdict
    reason: str = ""


_PROMPT = (
    "You are a strict fact-equivalence checker. Decide whether the CANDIDATE "
    "states the same fact as the QUERY. Reply with JSON only: "
    '{{"verdict": "accept" | "reject" | "ambiguous", "reason": "<short>"}}. '
    "Use 'accept' only if the facts match, 'reject' if any entity, number, "
    "date, or polarity differs, otherwise 'ambiguous'.\n"
    "QUERY: {query}\nCANDIDATE: {candidate}\nJSON:"
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeStats:
    """Aggregate counters for auditing judge behaviour."""

    calls: int = 0
    deterministic_reject_shortcircuit: int = 0
    valid: int = 0
    malformed_fallback: int = 0


class EquivalenceJudge:
    """SLM-backed equivalence judge that cannot loosen the deterministic call.

    ``judge`` takes the deterministic verdict and the two texts. If the
    deterministic verdict is ``REJECT`` it returns ``REJECT`` immediately
    without consulting the model. Otherwise it queries the backend, validates
    the JSON, and returns the model's verdict -- except that a model ``REJECT``
    is always honoured (more conservative is always allowed) and any malformed
    output becomes ``AMBIGUOUS``.
    """

    def __init__(self, backend: GenerationBackend):
        self.backend = backend
        self.stats = JudgeStats()

    def judge(self, query_text: str, candidate_text: str,
              deterministic_verdict: VerificationVerdict) -> EquivalenceJudgment:
        self.stats.calls += 1

        # Hard rule: deterministic REJECT is final. The model is not consulted.
        if deterministic_verdict is VerificationVerdict.REJECT:
            self.stats.deterministic_reject_shortcircuit += 1
            return EquivalenceJudgment(
                verdict=VerificationVerdict.REJECT,
                reason="deterministic reject is final; SLM not consulted",
            )

        prompt = _PROMPT.format(query=query_text, candidate=candidate_text)
        try:
            raw = self.backend.generate(prompt)
        except Exception:
            self.stats.malformed_fallback += 1
            return EquivalenceJudgment(
                verdict=VerificationVerdict.AMBIGUOUS,
                reason="backend error; falling back to ambiguous",
            )

        parsed = self._parse(raw)
        if parsed is None:
            self.stats.malformed_fallback += 1
            return EquivalenceJudgment(
                verdict=VerificationVerdict.AMBIGUOUS,
                reason="malformed judge output; falling back to ambiguous",
            )

        self.stats.valid += 1
        return parsed

    @staticmethod
    def _parse(raw: str) -> Optional[EquivalenceJudgment]:
        match = _JSON_RE.search(raw or "")
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        try:
            loose = _RawJudgment.model_validate(data)
        except ValidationError:
            return None
        verdict_token = loose.verdict.strip().lower()
        try:
            verdict = VerificationVerdict(verdict_token)
        except ValueError:
            return None
        return EquivalenceJudgment(verdict=verdict, reason=loose.reason)
