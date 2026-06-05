"""Operational follow-up execution (bounded record; no active-state change)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.lifecycle_action_executor as lae  # noqa: E402
import agent.regression_review_queue as rrq  # noqa: E402
from _lifecycle_executor_helpers import (  # noqa: E402
    FIXED_DT, approved_action_request, execution_approval, manager,
    two_pack_manifest,
)

_RECCODE = rrq.MonitoringRecommendationCode


def _follow_up(tmp_path, recommendation):
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(
        recommendation=recommendation, mf=mf, candidate_pack_ids=("p1",))
    mgr = manager(tmp_path, mf)
    approval = execution_approval(ar)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    return res, mgr, mf


def test_investigate_creates_follow_up_without_state_change(tmp_path):
    res, mgr, mf = _follow_up(tmp_path, _RECCODE.INVESTIGATE)
    assert res.status == lae.ExecutionStatus.EXECUTED, res.message
    assert res.operational_follow_up is not None
    assert res.operational_follow_up.follow_up_type == lae.FollowUpType.INVESTIGATION
    assert mgr.load_state().state_hash == mf.state_hash


def test_watch_creates_follow_up(tmp_path):
    res, mgr, mf = _follow_up(tmp_path, _RECCODE.KEEP_ACTIVE_WITH_WATCH)
    assert res.status == lae.ExecutionStatus.EXECUTED, res.message
    assert res.operational_follow_up.follow_up_type == lae.FollowUpType.WATCH
    assert mgr.load_state().state_hash == mf.state_hash


def test_follow_up_marks_action_executed(tmp_path):
    res, _, _ = _follow_up(tmp_path, _RECCODE.INVESTIGATE)
    assert res.updated_action_request.status == rrq.ActionRequestStatus.EXECUTED
    assert res.lifecycle_audit_ref


def test_follow_up_dry_run_writes_nothing(tmp_path):
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(
        recommendation=_RECCODE.INVESTIGATE, mf=mf, candidate_pack_ids=("p1",))
    mgr = manager(tmp_path, mf)
    approval = execution_approval(ar)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=False, now=FIXED_DT)
    assert res.status == lae.ExecutionStatus.DRY_RUN_READY
    assert res.operational_follow_up is None
    assert mgr.load_state().state_hash == mf.state_hash
