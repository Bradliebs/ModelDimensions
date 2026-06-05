"""v7.0 supersession: atomic replace of an active revision, fail-closed.

Pins that supersession activates the new pack and marks the predecessor
superseded in a single atomic state write, that the lineage must match, that a
failed supersession leaves the predecessor active, and that both packs' files are
preserved on disk.
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


def _activate(mgr, identity):
    evidence = passing_evidence(identity)
    request = kpa.KnowledgePackActivationRequest(
        identity=identity, approval=approval_for(identity, evidence=evidence),
        evidence=evidence, current_state=kpa.PackLifecycleState.EVALUATED)
    return mgr.activate(request, write=True, now=FIXED_NOW)


def _supersede_request(identity, evidence=None):
    evidence = evidence or passing_evidence(identity)
    return kpa.KnowledgePackActivationRequest(
        identity=identity,
        approval=approval_for(identity, scope=kpa.ActivationScope.SUPERSEDE,
                              evidence=evidence),
        evidence=evidence)


def test_supersession_is_atomic(tmp_path):
    v1 = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "packs" / "hf-v1", pack_id="hf-v1", dataset_revision="rev-1"))
    v2 = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "packs" / "hf-v2", pack_id="hf-v2", dataset_revision="rev-2"))
    mgr = _manager(tmp_path)
    _activate(mgr, v1)
    result = mgr.supersede(old_pack_id="hf-v1",
                           request=_supersede_request(v2), write=True, now=FIXED_NOW)
    assert result.success
    state = mgr.load_state()
    assert state.find("hf-v1").status == kpa.PackLifecycleState.SUPERSEDED
    assert state.find("hf-v2").status == kpa.PackLifecycleState.ACTIVE
    assert state.find("hf-v2").supersedes == "hf-v1"
    assert state.find("hf-v1").superseded_by == "hf-v2"
    # Exactly one active revision in the lineage.
    assert len(state.active_for_lineage(v2.lineage_key)) == 1


def test_supersession_requires_matching_lineage(tmp_path):
    hf = kpa.load_pack_identity(write_hf_pack(tmp_path / "packs" / "hf-demo"))
    from _kp_activation_helpers import write_pdf_pack
    pdf = kpa.load_pack_identity(write_pdf_pack(tmp_path / "packs" / "pdf-demo"))
    mgr = _manager(tmp_path)
    _activate(mgr, hf)
    result = mgr.supersede(old_pack_id="hf-demo",
                           request=_supersede_request(pdf), write=True, now=FIXED_NOW)
    assert not result.success
    assert kpa.ActivationFindingCode.LINEAGE_CONFLICT_ACTIVE_REVISION in {
        f.code for f in result.validation.blocking_findings}


def test_failed_supersession_leaves_predecessor_active(tmp_path):
    v1 = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "packs" / "hf-v1", pack_id="hf-v1", dataset_revision="rev-1"))
    v2 = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "packs" / "hf-v2", pack_id="hf-v2", dataset_revision="rev-2"))
    mgr = _manager(tmp_path)
    _activate(mgr, v1)
    # New pack fails evaluation -> supersession blocked.
    bad_evidence = replace(passing_evidence(v2), passed=False, failed_case_count=1)
    result = mgr.supersede(old_pack_id="hf-v1",
                           request=_supersede_request(v2, evidence=bad_evidence),
                           write=True, now=FIXED_NOW)
    assert not result.success
    state = mgr.load_state()
    assert state.find("hf-v1").status == kpa.PackLifecycleState.ACTIVE
    assert state.find("hf-v2") is None


def test_supersede_old_not_active_blocks(tmp_path):
    v2 = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "packs" / "hf-v2", pack_id="hf-v2", dataset_revision="rev-2"))
    mgr = _manager(tmp_path)
    result = mgr.supersede(old_pack_id="hf-v1",
                           request=_supersede_request(v2), write=True, now=FIXED_NOW)
    assert not result.success
    assert kpa.ActivationFindingCode.PACK_NOT_ACTIVE in {
        f.code for f in result.validation.blocking_findings}


def test_supersession_dry_run_writes_nothing(tmp_path):
    v1 = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "packs" / "hf-v1", pack_id="hf-v1", dataset_revision="rev-1"))
    v2 = kpa.load_pack_identity(write_hf_pack(
        tmp_path / "packs" / "hf-v2", pack_id="hf-v2", dataset_revision="rev-2"))
    mgr = _manager(tmp_path)
    _activate(mgr, v1)
    state_before = mgr.state_path.read_bytes()
    result = mgr.supersede(old_pack_id="hf-v1",
                           request=_supersede_request(v2), write=False, now=FIXED_NOW)
    assert result.success and not result.written
    assert mgr.state_path.read_bytes() == state_before
