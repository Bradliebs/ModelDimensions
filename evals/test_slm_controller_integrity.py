"""Integrity tests for the SLM controller layer (v0.8).

These run fully offline: a ``DeterministicEncoder`` (no model download) plus the
rule-based ``MockSLMController``. They assert the auditable guarantees:

  * intents are classified into the right typed action;
  * binding proposals are well-formed;
  * a silent bank refuses instead of answering;
  * fired memories are cited in grounded answers;
  * the SLM cannot smuggle in an unsupported memory id;
  * exact-id delete actually removes the memory.

Run with:  python -m pytest evals/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.orchestrator import DeterministicEncoder, MemoryBank, Orchestrator
from agent.response_policy import enforce_grounding, ground_from_cells
from slm.controller import MockSLMController
from slm.schemas import IntentType


def _fresh():
    enc = DeterministicEncoder(dim=64)
    bank = MemoryBank(enc, epsilon=0.05, radius=0.9)
    ctrl = MockSLMController()
    orch = Orchestrator(bank, ctrl)
    return enc, bank, ctrl, orch


# ---------- intent classification ----------

def test_query_intent_classification():
    ctrl = MockSLMController()
    decision = ctrl.classify_intent("What is the capital of France?")
    assert decision.intent.intent == IntentType.QUERY
    assert decision.query_text is not None


def test_write_intent_classification():
    ctrl = MockSLMController()
    decision = ctrl.classify_intent("Remember that the sky is blue.")
    assert decision.intent.intent == IntentType.WRITE
    assert decision.write_candidate is not None
    # canonicalisation strips the leading verb
    assert "sky is blue" in decision.write_candidate.canonical_text.lower()
    assert "remember" not in decision.write_candidate.canonical_text.lower()


def test_delete_intent_classification_with_id():
    ctrl = MockSLMController()
    decision = ctrl.classify_intent("Please forget mem-0003 now.")
    assert decision.intent.intent == IntentType.DELETE
    assert decision.delete_memory_id == "mem-0003"


def test_bind_suggestion_formatting():
    ctrl = MockSLMController()
    items = [
        {"memory_id": "mem-0001", "canonical_text": "a"},
        {"memory_id": "mem-0002", "canonical_text": "b"},
    ]
    cand = ctrl.suggest_bindings(items)
    assert cand is not None
    assert cand.memory_ids == ["mem-0001", "mem-0002"]
    assert cand.bound_group_id  # non-empty


def test_bind_suggestion_requires_two():
    ctrl = MockSLMController()
    assert ctrl.suggest_bindings([{"memory_id": "mem-0001"}]) is None


# ---------- grounding guarantees ----------

def test_silent_memory_refusal():
    _, bank, _, orch = _fresh()
    # Nothing stored -> any query must be a silent refusal.
    resp = orch.handle("What did I tell you earlier?")
    assert resp.refused is True
    assert resp.memory_used is False
    assert resp.cited_memory_ids == []
    assert "silent" in resp.text.lower()


def test_fired_memory_grounded_answer():
    _, bank, _, _ = _fresh()
    # The deterministic encoder is content-addressed by exact string, so a
    # cell fires only for the exact stored canonical text. Recall therefore
    # uses that exact text (the offline proxy for semantic recall).
    rec = bank.write("the launch is on Friday")
    result = bank.query("the launch is on Friday")
    resp = ground_from_cells(result, bank)
    assert resp.memory_used is True
    assert resp.refused is False
    assert rec.memory_id in resp.cited_memory_ids
    assert "launch is on friday" in resp.text.lower()


def test_no_answer_when_no_cell_fires():
    _, bank, _, orch = _fresh()
    orch.handle("Remember that the launch is on Friday.")
    # Unrelated query -> no cell should fire -> refusal, no citation.
    resp = orch.handle("What is the weather on Mars?")
    assert resp.refused is True
    assert resp.cited_memory_ids == []


def test_exact_delete_removes_memory():
    _, bank, _, orch = _fresh()
    rec = bank.write("the launch is on Friday")
    assert bank.get(rec.memory_id) is not None
    resp = orch.handle(f"forget {rec.memory_id}")
    assert resp.refused is False
    assert bank.get(rec.memory_id) is None
    # And the now-deleted memory no longer answers queries.
    again = orch.handle("What about the launch is on Friday?")
    assert again.refused is True


def test_delete_unknown_id_refuses():
    _, bank, _, orch = _fresh()
    resp = orch.handle("forget mem-9999")
    assert resp.refused is True


def test_mock_slm_cannot_bypass_response_policy():
    _, bank, _, _ = _fresh()
    rec = bank.write("grounded fact A")
    result = bank.query("grounded fact A")
    assert not result.silent  # it fired
    fired = result.fired_memory_ids
    # SLM claims a memory that did not fire -> must be refused.
    forged = enforce_grounding(
        slm_text="Trust me, mem-7777 says otherwise.",
        fired_memory_ids=fired,
        claimed_memory_ids=fired + ["mem-7777"],
    )
    assert forged.refused is True
    assert forged.cited_memory_ids == []


def test_ground_from_cells_only_cites_real_records():
    _, bank, _, _ = _fresh()
    rec = bank.write("grounded fact B")
    result = bank.query("grounded fact B")
    grounded = ground_from_cells(result, bank)
    assert grounded.memory_used is True
    assert grounded.cited_memory_ids == [rec.memory_id]
    assert "grounded fact b" in grounded.text.lower()


# ---------- no-SLM path ----------

def test_orchestrator_runs_without_controller():
    enc = DeterministicEncoder(dim=64)
    bank = MemoryBank(enc)
    bank.write("the sky is blue")
    orch = Orchestrator(bank, controller=None)
    resp = orch.handle("the sky is blue")
    assert resp.memory_used is True
    assert len(resp.cited_memory_ids) == 1
