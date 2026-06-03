"""Experiment 07: real-SLM controller stress test.

Compares four response paths on one scripted, fully-offline scenario:

  1. mock   : the v0.8 deterministic MockSLMController.
  2. rules  : the v0.9 keyword RulesController (non-LLM baseline).
  3. local  : the v0.9 strict LocalJSONController driven by a scripted JSON
              backend that occasionally emits malformed output (to exercise the
              retry-then-fallback path) -- the *guarded* path.
  4. unsafe : the same LocalJSONController, but the answer bypasses grounding
              and fabricates a citation when memory is silent. This exists only
              to demonstrate the failure mode the grounding policy prevents.

By default everything is offline and deterministic. Set EXP07_LOCAL_SLM_MODEL
to drive the local/unsafe paths with a real local model instead.

Metrics (all in [0, 1]):

  - json_validity_rate      : decisions valid on the first model attempt.
  - malformed_decision_rate : decisions unusable even after one retry (LOWER).
  - routing_accuracy        : intent matched the scripted ground truth.
  - silent_refusal_rate     : answerable queries wrongly refused (LOWER).
  - unsupported_answer_rate  : silent queries answered anyway (LOWER; this is
                              the hallucination rate the guard prevents).
  - attribution_accuracy    : grounded answers citing exactly the right memory.
  - write_candidate_quality : compressed WRITE candidates that are non-empty
                              and stripped of the leading instruction verb.
  - bind_suggestion_precision: proposed bind ids that actually exist.

Writes results/exp07_summary.json.

    python -m experiments.exp07_real_slm_controller_stress
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.orchestrator import DeterministicEncoder, MemoryBank
from agent.response_policy import build_response
from slm.controller import MockSLMController
from slm.local_json_controller import GenerationBackend, LocalJSONController
from slm.rules_controller import RulesController
from slm.schemas import GroundedResponse, IntentType, SLMControllerDecision


# ---------- scripted scenario ----------
# Each turn: user text, ground-truth intent, and an "answer key" used only for
# query turns. answer_text is the exact stored canonical text expected to fire
# (None means the query should stay silent). The offline encoder is
# content-addressed, so recall is tested with exact stored text.
#
# "flaky" marks turns whose scripted SLM output is malformed on the first
# attempt (recovers on retry). "broken" marks turns malformed on BOTH attempts
# (forces the safe fallback).
SCRIPT = [
    # text, intent, answer_text, flaky, broken
    ("Remember that the launch is on Friday", IntentType.WRITE, None, False, False),
    ("Note that the budget is forty thousand", IntentType.WRITE, None, True, False),
    ("Store that the server is in Dublin", IntentType.WRITE, None, False, False),
    ("the launch is on Friday", IntentType.QUERY, "the launch is on Friday", False, False),
    ("the budget is forty thousand", IntentType.QUERY, "the budget is forty thousand", True, False),
    ("what is the wifi password", IntentType.QUERY, None, False, False),
    ("who won the game last night", IntentType.QUERY, None, False, True),
    ("forget mem-0003", IntentType.DELETE, None, False, False),
    ("the server is in Dublin", IntentType.QUERY, None, False, False),  # deleted -> silent
    ("bind these memories together", IntentType.BIND, None, False, False),
]

_USER_MSG_RE = re.compile(r"User message: (.*)\nJSON:", re.S)
_MEMID_RE = re.compile(r"\b(mem-\d+)\b")
_WRITE_VERBS = ("remember", "note", "store", "save", "memorize", "record", "log")


def _canonical(text: str) -> str:
    """Strip a leading write verb (+ optional 'that') for the scripted SLM."""
    out = text.strip()
    low = out.lower()
    for verb in _WRITE_VERBS:
        if low.startswith(verb):
            out = out[len(verb):].lstrip()
            if out.lower().startswith("that "):
                out = out[5:]
            break
    return out.strip().rstrip(".")


class ScriptedJSONBackend:
    """A deterministic stand-in for a local JSON-emitting SLM.

    It reads the user message out of the controller's prompt and returns the
    JSON a well-behaved model would. Turns flagged ``flaky``/``broken`` return
    malformed text to exercise the controller's retry and fallback paths. The
    retry is detected by the controller's "previous answer was not valid JSON"
    reminder appearing in the prompt.
    """

    def __init__(self, flaky: set, broken: set):
        self.flaky = flaky
        self.broken = broken

    def generate(self, prompt: str) -> str:
        if "Propose a binding" in prompt:
            ids = _MEMID_RE.findall(prompt)
            return json.dumps({
                "intent": "bind",
                "bind_memory_ids": ids,
                "bound_group_id": "grp-local-" + "-".join(ids[:4]),
                "rationale": "scripted bind over all current memories",
            })
        if "Compress the following" in prompt:
            m = re.search(r"Text: (.*)\nJSON:", prompt, re.S)
            text = m.group(1).strip() if m else ""
            return json.dumps({"intent": "write",
                               "canonical_text": _canonical(text)})

        # Intent classification prompt.
        m = _USER_MSG_RE.search(prompt)
        user_text = m.group(1).strip() if m else ""
        is_retry = "previous answer was not valid JSON" in prompt

        if user_text in self.broken:
            return "<<the model rambled and produced no JSON at all>>"
        if user_text in self.flaky and not is_retry:
            return "sorry, here is the answer: not-json"

        return json.dumps(self._decide(user_text))

    @staticmethod
    def _decide(user_text: str) -> dict:
        low = user_text.lower()
        if low.startswith("forget") or low.startswith("delete"):
            ids = _MEMID_RE.findall(user_text)
            return {"intent": "delete",
                    "delete_memory_id": ids[0] if ids else None,
                    "confidence": 0.9, "rationale": "deletion request"}
        if low.startswith("bind") or "bind" in low.split():
            return {"intent": "bind", "confidence": 0.7,
                    "rationale": "binding request"}
        if any(low.startswith(v) for v in _WRITE_VERBS):
            return {"intent": "write", "canonical_text": _canonical(user_text),
                    "confidence": 0.85, "rationale": "storage request"}
        return {"intent": "query", "query_text": user_text,
                "confidence": 0.7, "rationale": "recall request"}


# ---------- per-turn handling (mirrors the orchestrator, plus an unsafe path) ----------

def _expected_memory_id(bank: MemoryBank, answer_text: str) -> Optional[str]:
    for rec in bank.records():
        if rec.canonical_text == answer_text:
            return rec.memory_id
    return None


def _unsafe_answer(query_text: str, result) -> GroundedResponse:
    """The failure mode: answer regardless of what fired, citing a bogus id.

    When memory is silent the unsafe path still claims a memory-derived answer
    (fabricating a citation), which is exactly what the grounded path refuses.
    """
    if result.fired_memory_ids:
        return GroundedResponse(
            text=f"(unsafe) answer about: {query_text}",
            cited_memory_ids=result.fired_memory_ids,
            memory_used=True,
        )
    return GroundedResponse(
        text=f"(unsafe) Sure, here is what I 'remember' about: {query_text}",
        cited_memory_ids=["mem-0001"],  # fabricated -> unsupported
        memory_used=True,
    )


def run_config(name: str, controller, unsafe: bool = False) -> dict:
    bank = MemoryBank(DeterministicEncoder(dim=64), epsilon=0.05, radius=0.9)

    routing_hits = routing_total = 0
    silent_should = silent_correct = unsupported = 0
    recall_total = recall_hits = 0
    attrib_hits = attrib_total = 0
    write_quality_scores: List[float] = []
    bind_precisions: List[float] = []

    for text, exp_intent, answer_text, _flaky, _broken in SCRIPT:
        decision: SLMControllerDecision = controller.classify_intent(text)
        routing_total += 1
        if decision.intent.intent == exp_intent:
            routing_hits += 1

        intent = decision.intent.intent

        if intent == IntentType.WRITE and decision.write_candidate is not None:
            canon = decision.write_candidate.canonical_text
            quality = 1.0 if (canon.strip()
                              and not canon.lower().startswith(_WRITE_VERBS)
                              ) else 0.0
            write_quality_scores.append(quality)
            bank.write(canon, source=decision.write_candidate.source,
                       tags=decision.write_candidate.tags)
            continue

        if intent == IntentType.DELETE:
            if decision.delete_memory_id:
                bank.delete(decision.delete_memory_id)
            continue

        if intent == IntentType.BIND:
            items = bank.as_items()
            candidate = controller.suggest_bindings(items)
            if candidate is not None:
                valid = {i["memory_id"] for i in items}
                proposed = candidate.memory_ids
                if proposed:
                    hits = sum(1 for m in proposed if m in valid)
                    bind_precisions.append(hits / len(proposed))
                bank.bind(candidate.memory_ids, candidate.bound_group_id)
            continue

        # QUERY (or fallback-to-query) turn.
        query_text = decision.query_text or text
        exp_mem_id = (_expected_memory_id(bank, answer_text)
                      if answer_text else None)
        result = bank.query(query_text)
        resp = (_unsafe_answer(query_text, result) if unsafe
                else build_response(result, bank))

        if answer_text is not None:
            recall_total += 1
            if resp.memory_used:
                recall_hits += 1
            else:
                silent_correct += 0  # not a silent-should turn
            if resp.refused or not resp.memory_used:
                # an answerable query that was refused
                pass
            attrib_total += 1 if resp.memory_used else 0
            if resp.memory_used and exp_mem_id and \
                    resp.cited_memory_ids == [exp_mem_id]:
                attrib_hits += 1
        else:
            silent_should += 1
            if resp.refused and not resp.memory_used:
                silent_correct += 1
            if resp.memory_used:
                unsupported += 1

    # silent_refusal_rate here = answerable queries wrongly refused.
    answerable_refused = recall_total - recall_hits

    def ratio(a: int, b: int) -> Optional[float]:
        return round(a / b, 4) if b else None

    def mean(xs: List[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 4) if xs else None

    # json validity / malformed from controller diagnostics when available.
    stats = getattr(controller, "stats", None)
    if stats is not None and stats.calls:
        json_validity = round(stats.valid_first_attempt / stats.calls, 4)
        malformed_rate = round(stats.malformed / stats.calls, 4)
        retried = stats.retried
        recovered = stats.recovered_on_retry
    else:
        json_validity, malformed_rate, retried, recovered = 1.0, 0.0, 0, 0

    return {
        "config": name,
        "json_validity_rate": json_validity,
        "malformed_decision_rate": malformed_rate,
        "retried_decisions": retried,
        "recovered_on_retry": recovered,
        "routing_accuracy": ratio(routing_hits, routing_total),
        "silent_refusal_rate": ratio(answerable_refused, recall_total),
        "unsupported_answer_rate": ratio(unsupported, silent_should),
        "attribution_accuracy": ratio(attrib_hits, attrib_total),
        "write_candidate_quality": mean(write_quality_scores),
        "bind_suggestion_precision": mean(bind_precisions),
        "n_turns": len(SCRIPT),
    }


def _make_local_controller():
    flaky = {t for t, _, _, fl, _ in SCRIPT if fl}
    broken = {t for t, _, _, _, br in SCRIPT if br}
    model = os.environ.get("EXP07_LOCAL_SLM_MODEL")
    if model:
        return LocalJSONController(model)  # real backend
    backend: GenerationBackend = ScriptedJSONBackend(flaky=flaky, broken=broken)
    return LocalJSONController("scripted-offline", backend=backend)


def main():
    configs: List[dict] = []
    configs.append(run_config("mock", MockSLMController()))
    configs.append(run_config("rules", RulesController()))
    configs.append(run_config("local_guarded", _make_local_controller(),
                              unsafe=False))
    configs.append(run_config("local_unsafe", _make_local_controller(),
                              unsafe=True))

    guarded = next(c for c in configs if c["config"] == "local_guarded")
    unsafe = next(c for c in configs if c["config"] == "local_unsafe")
    grounding_holds = (
        (unsafe["unsupported_answer_rate"] or 0.0)
        > (guarded["unsupported_answer_rate"] or 0.0)
    )

    summary = {
        "experiment": "exp07_real_slm_controller_stress",
        "description": (
            "Stress test of mock / rules / strict-local / unsafe controller "
            "paths over the frozen concept-cell memory. Demonstrates that the "
            "grounded path refuses the hallucinations the unsafe path emits."
        ),
        "encoder": "DeterministicEncoder(dim=64) [offline, content-addressed]",
        "offline": os.environ.get("EXP07_LOCAL_SLM_MODEL") is None,
        "grounding_prevents_hallucination": grounding_holds,
        "configs": configs,
    }

    out_path = ROOT / "results" / "exp07_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[exp07] wrote {out_path}")


if __name__ == "__main__":
    main()
