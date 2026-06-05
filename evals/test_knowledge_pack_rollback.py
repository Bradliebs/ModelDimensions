"""v7.0 rollback: restore a prior recorded active state, fail-closed.

Pins that rollback restores a previously recorded state by hash, that a stale
current-state hash invalidates the rollback approval, that a missing or mutated
target fails closed, that a new audit event is written, and that the newer pack's
files are preserved on disk (rollback is not deletion).
"""
from __future__ import annotations

import sys
from dataclasses import replace
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


def _request(identity, *, current=kpa.PackLifecycleState.EVALUATED):
    evidence = passing_evidence(identity)
    return kpa.KnowledgePackActivationRequest(
        identity=identity, approval=approval_for(identity, evidence=evidence),
        evidence=evidence, current_state=current)


def _setup_two_states(tmp_path):
    """Activate v1 (state H1), supersede with v2 (state H2). Return manager + hashes."""
    v1 = write_hf_pack(tmp_path / "packs" / "hf-v1",
                       pack_id="hf-v1", dataset_revision="rev-1")
    v2 = write_hf_pack(tmp_path / "packs" / "hf-v2",
                       pack_id="hf-v2", dataset_revision="rev-2")
    id1 = kpa.load_pack_identity(v1)
    id2 = kpa.load_pack_identity(v2)
    mgr = _manager(tmp_path)
    mgr.activate(_request(id1), write=True, now=FIXED_NOW)
    h1 = mgr.load_state().state_hash
    evidence2 = passing_evidence(id2)
    supersede_request = kpa.KnowledgePackActivationRequest(
        identity=id2,
        approval=approval_for(id2, scope=kpa.ActivationScope.SUPERSEDE,
                              evidence=evidence2),
        evidence=evidence2)
    mgr.supersede(old_pack_id="hf-v1", request=supersede_request,
                  write=True, now=FIXED_NOW)
    h2 = mgr.load_state().state_hash
    dirs = {"hf-v1": str(v1), "hf-v2": str(v2)}
    return mgr, id1, h1, h2, dirs


def test_rollback_restores_prior_state(tmp_path):
    mgr, id1, h1, h2, dirs = _setup_two_states(tmp_path)
    approval = approval_for(id1, scope=kpa.ActivationScope.ROLLBACK)
    request = kpa.KnowledgePackRollbackRequest(
        target_state_hash=h1, current_state_hash=h2, approval=approval,
        reason="revert")
    result = mgr.rollback(request, available_pack_dirs=dirs, write=True,
                          now=FIXED_NOW)
    assert result.success
    active = [r.pack_id for r in mgr.load_state().active()]
    assert active == ["hf-v1"]
    assert mgr.load_state().state_hash == h1


def test_rollback_preserves_newer_pack_files(tmp_path):
    mgr, id1, h1, h2, dirs = _setup_two_states(tmp_path)
    approval = approval_for(id1, scope=kpa.ActivationScope.ROLLBACK)
    request = kpa.KnowledgePackRollbackRequest(
        target_state_hash=h1, current_state_hash=h2, approval=approval)
    mgr.rollback(request, available_pack_dirs=dirs, write=True, now=FIXED_NOW)
    assert "hf-v2" not in [r.pack_id for r in mgr.load_state().active()]
    assert (Path(dirs["hf-v2"]) / "manifest.json").exists()  # not deleted


def test_rollback_writes_new_audit_event(tmp_path):
    mgr, id1, h1, h2, dirs = _setup_two_states(tmp_path)
    before = len(mgr.load_audit())
    approval = approval_for(id1, scope=kpa.ActivationScope.ROLLBACK)
    request = kpa.KnowledgePackRollbackRequest(
        target_state_hash=h1, current_state_hash=h2, approval=approval)
    mgr.rollback(request, available_pack_dirs=dirs, write=True, now=FIXED_NOW)
    audit = mgr.load_audit()
    assert len(audit) == before + 1
    assert audit[-1].action == kpa.AuditAction.ROLLBACK
    assert audit[-1].success


def test_stale_current_state_hash_blocks(tmp_path):
    mgr, id1, h1, h2, dirs = _setup_two_states(tmp_path)
    approval = approval_for(id1, scope=kpa.ActivationScope.ROLLBACK)
    request = kpa.KnowledgePackRollbackRequest(
        target_state_hash=h1, current_state_hash="pkgstate-stale",
        approval=approval)
    result = mgr.rollback(request, available_pack_dirs=dirs, write=True,
                          now=FIXED_NOW)
    assert not result.success
    assert kpa.ActivationFindingCode.STALE_ROLLBACK_APPROVAL in {
        f.code for f in result.findings}


def test_missing_target_state_blocks(tmp_path):
    mgr, id1, h1, h2, dirs = _setup_two_states(tmp_path)
    approval = approval_for(id1, scope=kpa.ActivationScope.ROLLBACK)
    request = kpa.KnowledgePackRollbackRequest(
        target_state_hash="pkgstate-nonexistent", current_state_hash=h2,
        approval=approval)
    result = mgr.rollback(request, available_pack_dirs=dirs, write=True,
                          now=FIXED_NOW)
    assert not result.success
    assert kpa.ActivationFindingCode.ROLLBACK_TARGET_MISSING in {
        f.code for f in result.findings}


def test_wrong_scope_blocks_rollback(tmp_path):
    mgr, id1, h1, h2, dirs = _setup_two_states(tmp_path)
    approval = approval_for(id1, scope=kpa.ActivationScope.ACTIVATE)
    request = kpa.KnowledgePackRollbackRequest(
        target_state_hash=h1, current_state_hash=h2, approval=approval)
    result = mgr.rollback(request, available_pack_dirs=dirs, write=True,
                          now=FIXED_NOW)
    assert not result.success
    assert kpa.ActivationFindingCode.APPROVAL_SCOPE_INVALID in {
        f.code for f in result.findings}


def test_mutated_target_pack_blocks_rollback(tmp_path):
    mgr, id1, h1, h2, dirs = _setup_two_states(tmp_path)
    # Mutate the v1 pack content after H1 was recorded.
    v1_knowledge = Path(dirs["hf-v1"]) / "knowledge.jsonl"
    v1_knowledge.write_text(
        v1_knowledge.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    write_hf_pack(Path(dirs["hf-v1"]), pack_id="hf-v1",
                  dataset_revision="rev-1",
                  chunks=[("hfchunk-0001", "MUTATED")])
    approval = approval_for(id1, scope=kpa.ActivationScope.ROLLBACK)
    request = kpa.KnowledgePackRollbackRequest(
        target_state_hash=h1, current_state_hash=h2, approval=approval)
    result = mgr.rollback(request, available_pack_dirs=dirs, write=True,
                          now=FIXED_NOW)
    assert not result.success
    assert kpa.ActivationFindingCode.ROLLBACK_TARGET_INVALID in {
        f.code for f in result.findings}


def test_rollback_dry_run_writes_nothing(tmp_path):
    mgr, id1, h1, h2, dirs = _setup_two_states(tmp_path)
    state_before = mgr.state_path.read_bytes()
    audit_before = mgr.audit_path.read_bytes()
    approval = approval_for(id1, scope=kpa.ActivationScope.ROLLBACK)
    request = kpa.KnowledgePackRollbackRequest(
        target_state_hash=h1, current_state_hash=h2, approval=approval)
    result = mgr.rollback(request, available_pack_dirs=dirs, write=False,
                          now=FIXED_NOW)
    assert result.success and not result.written
    assert mgr.state_path.read_bytes() == state_before
    assert mgr.audit_path.read_bytes() == audit_before
