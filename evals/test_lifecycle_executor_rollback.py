"""End-to-end governed rollback via the real v7.0 primitive."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.lifecycle_action_executor as lae  # noqa: E402
import agent.regression_review_queue as rrq  # noqa: E402
from _lifecycle_executor_helpers import (  # noqa: E402
    FIXED_DT, activate_audit_record, approved_action_request, execution_approval,
    manager, manifest, two_pack_manifest,
)

_RECCODE = rrq.MonitoringRecommendationCode


def _rollback_setup(tmp_path):
    # prior good state = p1 only; regressed current = p1 + p2.
    prior = manifest(packs=(("p1", "1.0", "packfp-a", "src1", "r1"),))
    current = two_pack_manifest()
    ar, rec, _ = approved_action_request(
        recommendation=_RECCODE.ROLLBACK_RECOMMENDED, mf=current,
        candidate_pack_ids=("p2",), rollback_target=prior.state_hash)
    mgr = manager(tmp_path, current, audit_records=[activate_audit_record(prior)])
    approval = execution_approval(ar)
    return ar, rec, mgr, approval, prior, current


def test_rollback_dry_run_writes_nothing(tmp_path):
    ar, rec, mgr, approval, _, current = _rollback_setup(tmp_path)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=False, now=FIXED_DT)
    assert res.status == lae.ExecutionStatus.DRY_RUN_READY
    assert mgr.load_state().state_hash == current.state_hash


def test_rollback_restores_prior_state(tmp_path):
    ar, rec, mgr, approval, prior, _ = _rollback_setup(tmp_path)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    assert res.status == lae.ExecutionStatus.EXECUTED, res.message
    assert mgr.load_state().state_hash == prior.state_hash
    assert res.lifecycle_primitive_called == "knowledge_pack_activation.rollback"


def test_rollback_marks_action_executed(tmp_path):
    ar, rec, mgr, approval, _, _ = _rollback_setup(tmp_path)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    assert res.updated_action_request.status == rrq.ActionRequestStatus.EXECUTED


def test_rollback_to_current_state_is_refused(tmp_path):
    # A rollback target equal to the current state is "already resolved".
    current = two_pack_manifest()
    ar, rec, _ = approved_action_request(
        recommendation=_RECCODE.ROLLBACK_RECOMMENDED, mf=current,
        candidate_pack_ids=("p2",), rollback_target=current.state_hash)
    mgr = manager(tmp_path, current)
    approval = execution_approval(ar)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    assert res.status == lae.ExecutionStatus.VALIDATION_FAILED
    assert mgr.load_state().state_hash == current.state_hash
