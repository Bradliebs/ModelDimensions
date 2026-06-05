"""Atomicity, replay protection, persistence, and fail-closed guarantees."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.lifecycle_action_executor as lae  # noqa: E402
from _lifecycle_executor_helpers import (  # noqa: E402
    FIXED_DT, approved_action_request, deactivate_p2, execution_approval,
    manager, two_pack_manifest,
)


def _deactivate(tmp_path):
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf, candidate_pack_ids=("p1",))
    mgr = manager(tmp_path, mf)
    approval = execution_approval(ar)
    return ar, rec, mgr, approval, mf


def test_idempotent_replay_returns_already_executed(tmp_path):
    ar, rec, mgr, approval, _ = _deactivate(tmp_path)
    first = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    assert first.status == lae.ExecutionStatus.EXECUTED
    after_first = mgr.load_state().state_hash
    second = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], prior_results=[first],
        write=True, now=FIXED_DT)
    assert second.status == lae.ExecutionStatus.ALREADY_EXECUTED
    assert not second.written
    assert mgr.load_state().state_hash == after_first  # no second mutation


def test_stale_state_fails_closed(tmp_path):
    # Approval minted against `mf`, but the live state drifted afterwards.
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf, candidate_pack_ids=("p1",))
    drifted = deactivate_p2(mf)
    mgr = manager(tmp_path, drifted)
    approval = execution_approval(ar)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    assert res.status == lae.ExecutionStatus.VALIDATION_FAILED
    assert not res.written
    assert mgr.load_state().state_hash == drifted.state_hash  # untouched


def test_failed_validation_is_never_successful(tmp_path):
    ar, rec, mgr, _, mf = _deactivate(tmp_path)
    res = lae.execute_lifecycle_action(
        ar, None, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    assert not res.succeeded
    assert mgr.load_state().state_hash == mf.state_hash


def test_execution_audit_round_trip(tmp_path):
    ar, rec, mgr, approval, _ = _deactivate(tmp_path)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    audit_path = tmp_path / "reviews" / "lifecycle_execution_audit.jsonl"
    lae.append_execution_audit([res.audit], audit_path)
    loaded = lae.load_execution_audit(audit_path)
    assert len(loaded) == 1
    assert loaded[0].audit_record_hash == res.audit.audit_record_hash


def test_execution_audit_is_append_only(tmp_path):
    ar, rec, mgr, approval, _ = _deactivate(tmp_path)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    audit_path = tmp_path / "reviews" / "lifecycle_execution_audit.jsonl"
    lae.append_execution_audit([res.audit], audit_path)
    lae.append_execution_audit([res.audit], audit_path)
    assert len(lae.load_execution_audit(audit_path)) == 2  # history never rewritten


def test_follow_up_merge_is_idempotent(tmp_path):
    from _lifecycle_executor_helpers import approved_action_request as _aar
    import agent.regression_review_queue as rrq
    mf = two_pack_manifest()
    ar, _, _ = _aar(recommendation=rrq.MonitoringRecommendationCode.INVESTIGATE,
                    mf=mf, candidate_pack_ids=("p1",))
    fu = lae.make_operational_follow_up(ar, requested_by="op",
                                        requested_at="2024-01-01T00:00:00+00:00")
    merged = lae.add_follow_up([fu], fu)
    assert len(merged) == 1
