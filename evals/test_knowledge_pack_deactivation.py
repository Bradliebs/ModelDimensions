"""v7.0 deactivation: reversible, emergency path, audited.

Pins that deactivation moves an active pack to inactive (not deleted), that it is
reversible via a reactivation, that emergency deactivation requires an explicit
actor and reason, and that an unknown/inactive pack fails closed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import knowledge_pack_activation as kpa  # noqa: E402
from _kp_activation_helpers import (  # noqa: E402
    FIXED_NOW,
    approval_for,
    passing_evidence,
    write_hf_pack,
)


def _manager(tmp_path):
    return kpa.ActivationStateManager(
        state_path=tmp_path / "config" / "state.jsonl",
        audit_path=tmp_path / "reports" / "audit.jsonl")


def _activate(mgr, pack):
    identity = kpa.load_pack_identity(pack)
    evidence = passing_evidence(identity)
    request = kpa.KnowledgePackActivationRequest(
        identity=identity, approval=approval_for(identity, evidence=evidence),
        evidence=evidence, current_state=kpa.PackLifecycleState.EVALUATED)
    return mgr.activate(request, write=True, now=FIXED_NOW), identity


def test_emergency_deactivation_marks_inactive_not_deleted(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    mgr = _manager(tmp_path)
    _, identity = _activate(mgr, pack)
    result = mgr.deactivate("hf-demo", emergency=True, actor="oncall",
                            reason="incident-123", write=True, now=FIXED_NOW)
    assert result.success and result.emergency
    record = mgr.load_state().find("hf-demo")
    assert record.status == kpa.PackLifecycleState.INACTIVE
    # Pack still on disk.
    assert (pack / "manifest.json").exists()
    assert (pack / "knowledge.jsonl").exists()


def test_emergency_requires_actor_and_reason(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    mgr = _manager(tmp_path)
    _activate(mgr, pack)
    result = mgr.deactivate("hf-demo", emergency=True, actor=" ", reason="",
                            write=True, now=FIXED_NOW)
    assert not result.success
    assert kpa.ActivationFindingCode.APPROVAL_SCOPE_INVALID in {
        f.code for f in result.findings}


def test_normal_deactivation_requires_scoped_approval(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    mgr = _manager(tmp_path)
    _, identity = _activate(mgr, pack)
    # Approval scoped to activate cannot deactivate.
    wrong = approval_for(identity, scope=kpa.ActivationScope.ACTIVATE)
    result = mgr.deactivate("hf-demo", approval=wrong, write=True, now=FIXED_NOW)
    assert not result.success
    ok = approval_for(identity, scope=kpa.ActivationScope.EMERGENCY_DEACTIVATE)
    result2 = mgr.deactivate("hf-demo", approval=ok, write=True, now=FIXED_NOW)
    assert result2.success


def test_deactivation_is_reversible(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    mgr = _manager(tmp_path)
    _, identity = _activate(mgr, pack)
    mgr.deactivate("hf-demo", emergency=True, actor="op", reason="pause",
                   write=True, now=FIXED_NOW)
    assert mgr.load_state().find("hf-demo").status == kpa.PackLifecycleState.INACTIVE
    # Reactivate via a reactivation-scoped approval.
    evidence = passing_evidence(identity)
    reactivate = approval_for(identity, scope=kpa.ActivationScope.REACTIVATE,
                              evidence=evidence)
    request = kpa.KnowledgePackActivationRequest(
        identity=identity, approval=reactivate, evidence=evidence,
        current_state=kpa.PackLifecycleState.INACTIVE)
    result = mgr.activate(request, write=True, now=FIXED_NOW)
    assert result.success
    assert mgr.load_state().find("hf-demo").status == kpa.PackLifecycleState.ACTIVE


def test_deactivate_unknown_pack_fails(tmp_path):
    mgr = _manager(tmp_path)
    result = mgr.deactivate("nope", emergency=True, actor="op", reason="x",
                            write=True, now=FIXED_NOW)
    assert not result.success
    assert kpa.ActivationFindingCode.PACK_NOT_FOUND in {
        f.code for f in result.findings}


def test_deactivate_dry_run_writes_nothing(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    mgr = _manager(tmp_path)
    _activate(mgr, pack)
    state_before = mgr.state_path.read_bytes()
    result = mgr.deactivate("hf-demo", emergency=True, actor="op", reason="x",
                            write=False, now=FIXED_NOW)
    assert result.success and not result.written
    assert mgr.state_path.read_bytes() == state_before  # unchanged
