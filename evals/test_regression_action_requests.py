"""v7.2 regression action requests: a request is not an execution.

Pins that an action request is created only by an approval, that its id is
deterministic, that marking it executed requires an external lifecycle audit
reference, and that a stale action cannot be executed. This layer never performs
a lifecycle action — it only records one the separate layer reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.regression_review_queue as rrq  # noqa: E402
from _regression_review_helpers import FIXED_NOW, manifest, run_dict  # noqa: E402

_REC = rrq.MonitoringRecommendationCode
_D = rrq.ReviewDecision
_ROLE = rrq.ReviewerRole
_AS = rrq.ActionRequestStatus


def _approved(rec=_REC.ROLLBACK_RECOMMENDED, rollback_target="state-prev"):
    mf = manifest()
    rd = run_dict(manifest_obj=mf, recommendation=rec,
                  rollback_target=rollback_target)
    item = rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW)
    validation = rrq.validate_review_item(
        item, current_state=mf, rollback_target_exists=True, now=FIXED_NOW)
    out = rrq.review_item(item, _D.APPROVE, reviewer="gov",
                          role=_ROLE.GOVERNANCE_APPROVER, reason="confirmed",
                          validation=validation, reviewed_at=FIXED_NOW)
    return out


def test_action_request_id_is_deterministic_and_prefixed():
    a = _approved().action_request
    b = _approved().action_request
    assert a.action_request_id == b.action_request_id
    assert a.action_request_id.startswith(rrq.ACTION_REQUEST_PREFIX)


def test_rollback_request_carries_target_and_approval_type():
    ar = _approved().action_request
    assert ar.requested_action == rrq.ActionRequestType.REQUEST_ROLLBACK
    assert ar.rollback_target_state_hash == "state-prev"
    assert ar.required_execution_approval_type == "rollback_approval"
    assert ar.status == _AS.REQUESTED


def test_request_starts_unexecuted():
    ar = _approved().action_request
    assert not ar.executed
    assert ar.executed_at == ""
    assert ar.execution_record_id == ""


def test_mark_executed_requires_external_audit_reference():
    ar = _approved().action_request
    try:
        rrq.mark_action_result(ar, _AS.EXECUTED)
    except ValueError:
        pass
    else:
        raise AssertionError("executed must require an execution record + audit ref")
    done = rrq.mark_action_result(
        ar, _AS.EXECUTED, execution_record_id="lifecycle-rec-1",
        lifecycle_audit_ref="audit-1", executed_at=FIXED_NOW)
    assert done.executed
    assert done.execution_record_id == "lifecycle-rec-1"
    assert done.lifecycle_audit_ref == "audit-1"


def test_mark_executed_is_idempotent_evidence():
    ar = _approved().action_request
    done = rrq.mark_action_result(
        ar, _AS.EXECUTED, execution_record_id="rec-1",
        lifecycle_audit_ref="audit-1", executed_at=FIXED_NOW)
    # re-marking the already-executed action is rejected, not silently re-run
    try:
        rrq.mark_action_result(done, _AS.EXECUTED, execution_record_id="rec-1",
                               lifecycle_audit_ref="audit-1")
    except ValueError:
        return
    raise AssertionError("an already-executed action must not be re-executed")


def test_stale_action_cannot_be_executed():
    ar = _approved().action_request
    stale = rrq.mark_action_result(ar, _AS.STALE)
    try:
        rrq.mark_action_result(stale, _AS.EXECUTED, execution_record_id="r",
                               lifecycle_audit_ref="a")
    except ValueError:
        return
    raise AssertionError("a stale action must not be executable")


def test_failed_action_records_reason():
    ar = _approved().action_request
    failed = rrq.mark_action_result(ar, _AS.FAILED, failure_reason="precheck failed")
    assert failed.status == _AS.FAILED
    assert failed.failure_reason == "precheck failed"


def test_non_approval_decisions_create_no_action_request():
    mf = manifest()
    rd = run_dict(manifest_obj=mf, recommendation=_REC.DEACTIVATE_RECOMMENDED)
    item = rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW)
    out = rrq.review_item(item, _D.REJECT, reviewer="alice",
                          role=_ROLE.MONITORING_REVIEWER, reason="no",
                          reviewed_at=FIXED_NOW)
    assert out.action_request is None


def test_action_request_roundtrips():
    ar = _approved().action_request
    again = rrq.RegressionActionRequest.from_dict(ar.to_dict())
    assert again == ar
