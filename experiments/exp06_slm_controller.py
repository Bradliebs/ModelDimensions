"""Experiment 06: auditable SLM controller over the concept-cell memory.

Compares three configurations on a small, fully offline scripted scenario:

  1. cells_only   : memory bank with no SLM (every input is a query).
  2. mock_slm     : memory bank + deterministic MockSLMController.
  3. local_slm    : same, but with a locally hosted SLM, IF one is configured
                    via the EXP06_LOCAL_SLM_MODEL environment variable.
                    Skipped otherwise so the experiment stays download-free.

Metrics (all in [0, 1], higher is better unless noted):

  - routing_accuracy        : intent classified as the scripted ground truth.
  - silent_refusal_rate     : of turns that SHOULD find nothing, how many were
                              correctly refused as "memory silent".
  - unsupported_answer_rate : of turns that should find nothing, how many
                              produced a memory-used answer anyway (LOWER is
                              better; this is the hallucination rate).
  - attribution_accuracy    : of grounded answers, how many cited exactly the
                              expected memory.
  - query_success_rate      : of turns that SHOULD recall, how many fired.

Writes results/exp06_summary.json.

    python -m experiments.exp06_slm_controller
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.orchestrator import DeterministicEncoder, MemoryBank, Orchestrator
from slm.controller import MockSLMController, get_controller
from slm.schemas import IntentType


# A scripted scenario. Each step has the user text, the ground-truth intent,
# and (for queries) whether recall is expected and which canonical text holds
# the answer. Query turns use the exact stored text because the offline
# deterministic encoder is content-addressed (an honest recall proxy).
SCRIPT = [
    # (user_text, expected_intent, expects_recall, answer_text_or_None)
    ("Remember that the launch is on Friday", IntentType.WRITE, False, None),
    ("Remember that the budget is forty thousand", IntentType.WRITE, False, None),
    ("the launch is on Friday", IntentType.QUERY, True, "the launch is on Friday"),
    ("What is the weather on Mars?", IntentType.QUERY, False, None),
    ("the budget is forty thousand", IntentType.QUERY, True, "the budget is forty thousand"),
    ("flibbertigibbet nonsense token", IntentType.QUERY, False, None),
]


def _expected_memory_id(bank: MemoryBank, answer_text: str) -> Optional[str]:
    for rec in bank.records():
        if rec.canonical_text == answer_text:
            return rec.memory_id
    return None


def run_config(name: str, use_controller: bool,
               controller_kind: str = "mock",
               **controller_kwargs) -> dict:
    enc = DeterministicEncoder(dim=64)
    bank = MemoryBank(enc, epsilon=0.05, radius=0.9)
    controller = None
    if use_controller:
        if controller_kind == "mock":
            controller = MockSLMController()
        else:
            controller = get_controller(controller_kind, **controller_kwargs)
    orch = Orchestrator(bank, controller)

    # Baseline fairness: with no controller there is no intent routing, so the
    # bank cannot tell a write from a query. We pre-seed it via the direct API
    # (simulating out-of-band writes) so recall is still measurable. With a
    # controller, writes flow through the scripted WRITE turns instead.
    if controller is None:
        canon = MockSLMController()
        for user_text, exp_intent, _, _ in SCRIPT:
            if exp_intent == IntentType.WRITE:
                bank.write(canon.compress_for_memory(user_text).canonical_text)

    routing_hits = routing_total = 0
    recall_hits = recall_total = 0
    silent_should = silent_correct = 0
    unsupported = 0
    attrib_hits = attrib_total = 0

    for user_text, exp_intent, expects_recall, answer_text in SCRIPT:
        # Routing accuracy is only meaningful when a controller is present.
        if controller is not None:
            decision = controller.classify_intent(user_text)
            routing_total += 1
            if decision.intent.intent == exp_intent:
                routing_hits += 1

        # Without a controller, writes were pre-seeded; skip those turns so we
        # do not query the bank with the raw "Remember that..." text.
        if controller is None and exp_intent == IntentType.WRITE:
            continue

        # Determine the expected memory id BEFORE handling (writes happen here).
        exp_mem_id = (
            _expected_memory_id(bank, answer_text)
            if (expects_recall and answer_text)
            else None
        )

        resp = orch.handle(user_text)

        # Only score recall/grounding on query-style turns.
        if exp_intent == IntentType.QUERY:
            if expects_recall:
                recall_total += 1
                if resp.memory_used:
                    recall_hits += 1
                    attrib_total += 1
                    if exp_mem_id and resp.cited_memory_ids == [exp_mem_id]:
                        attrib_hits += 1
            else:
                silent_should += 1
                if resp.refused and not resp.memory_used:
                    silent_correct += 1
                if resp.memory_used:
                    unsupported += 1

    def ratio(a: int, b: int) -> Optional[float]:
        return round(a / b, 4) if b else None

    return {
        "config": name,
        "routing_accuracy": ratio(routing_hits, routing_total),
        "query_success_rate": ratio(recall_hits, recall_total),
        "silent_refusal_rate": ratio(silent_correct, silent_should),
        "unsupported_answer_rate": ratio(unsupported, silent_should),
        "attribution_accuracy": ratio(attrib_hits, attrib_total),
        "n_turns": len(SCRIPT),
    }


def main():
    configs: List[dict] = []
    configs.append(run_config("cells_only", use_controller=False))
    configs.append(run_config("mock_slm", use_controller=True,
                              controller_kind="mock"))

    local_model = os.environ.get("EXP06_LOCAL_SLM_MODEL")
    if local_model:
        try:
            configs.append(run_config(
                "local_slm", use_controller=True,
                controller_kind="local", model_name=local_model,
            ))
        except Exception as exc:  # pragma: no cover - only with a real model
            configs.append({
                "config": "local_slm",
                "error": f"{type(exc).__name__}: {exc}",
            })
    else:
        configs.append({
            "config": "local_slm",
            "skipped": "set EXP06_LOCAL_SLM_MODEL to enable (kept offline by default)",
        })

    summary = {
        "experiment": "exp06_slm_controller",
        "description": (
            "Auditable SLM controller over the frozen concept-cell memory: "
            "routing, grounding, and hallucination-refusal metrics."
        ),
        "encoder": "DeterministicEncoder(dim=64) [offline, content-addressed]",
        "configs": configs,
    }

    out_path = ROOT / "results" / "exp06_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\n[exp06] wrote {out_path}")


if __name__ == "__main__":
    main()
