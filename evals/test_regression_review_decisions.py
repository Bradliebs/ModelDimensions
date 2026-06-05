"""v7.2 regression review queue: governed decisions.

Pins reviewer-identity and reason requirements, role gating, valid decision
results, that a stale item cannot be approved, that an unresolved evidence
request blocks approval, and that approval emits an action request without ever
executing a lifecycle action.
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
_S = rrq.ReviewItemStatus


def _item(rec=_REC.DEACTIVATE_RECOMMENDED, mf=None):
    mf = mf if mf is not None else manifest()
    rd = run_dict(manifest_obj=mf, recommendation=rec)
    return rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW), mf


def _valid(item, mf):
    return rrq.validate_review_item(item, current_state=mf, now=FIXED_NOW)


def test_reviewer_identity_is_required():
    item, mf = _item()
    try:
        rrq.review_item(item, _D.REJECT, reviewer="  ", role=_ROLE.MONITORING_REVIEWER,
                        reason="bad")
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing reviewer")


def test_reason_is_required():
    item, mf = _item()
    try:
        rrq.review_item(item, _D.REJECT, reviewer="alice",
                        role=_ROLE.MONITORING_REVIEWER, reason="")
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing reason")


def test_role_gating_blocks_unauthorised_approval():
    item, mf = _item()
    try:
        rrq.review_item(item, _D.APPROVE, reviewer="alice",
                        role=_ROLE.MONITORING_REVIEWER, reason="looks right",
                        validation=_valid(item, mf))
    except ValueError:
        return
    raise AssertionError("monitoring_reviewer must not approve")


def test_reject_sets_status_and_emits_no_action():
    item, mf = _item()
    out = rrq.review_item(item, _D.REJECT, reviewer="alice",
                          role=_ROLE.MONITORING_REVIEWER, reason="false alarm",
                          reviewed_at=FIXED_NOW)
    assert out.item.status == _S.REJECTED
    assert out.action_request is None
    assert out.record.review_record_id.startswith(rrq.REVIEW_RECORD_PREFIX)
    assert out.record.decision == _D.REJECT


def test_defer_records_defer_until():
    item, mf = _item()
    out = rrq.review_item(item, _D.DEFER, reviewer="alice",
                          role=_ROLE.MONITORING_REVIEWER, reason="need owner",
                          defer_until="2024-02-01", reviewed_at=FIXED_NOW)
    assert out.item.status == _S.DEFERRED
    assert out.record.defer_until == "2024-02-01"
    assert out.action_request is None


def test_request_more_evidence_sets_status():
    item, mf = _item()
    out = rrq.review_item(item, _D.REQUEST_MORE_EVIDENCE, reviewer="alice",
                          role=_ROLE.MONITORING_REVIEWER, reason="ambiguous",
                          requested_evidence="more cases", reviewed_at=FIXED_NOW)
    assert out.item.status == _S.MORE_EVIDENCE_REQUESTED
    assert out.record.requested_evidence == "more cases"


def test_mark_duplicate_links_canonical():
    item, mf = _item()
    out = rrq.review_item(item, _D.MARK_DUPLICATE, reviewer="alice",
                          role=_ROLE.MONITORING_REVIEWER, reason="dup",
                          duplicate_of_review_item_id="regrev-canon",
                          reviewed_at=FIXED_NOW)
    assert out.item.status == _S.DUPLICATE
    assert out.record.duplicate_of_review_item_id == "regrev-canon"
    assert out.action_request is None


def test_close_without_action():
    item, mf = _item()
    out = rrq.review_item(item, _D.CLOSE_WITHOUT_ACTION, reviewer="alice",
                          role=_ROLE.MONITORING_REVIEWER, reason="no longer relevant",
                          reviewed_at=FIXED_NOW)
    assert out.item.status == _S.CLOSED_NO_ACTION
    assert out.action_request is None


def test_approval_emits_action_request_only():
    item, mf = _item()
    out = rrq.review_item(item, _D.APPROVE, reviewer="gov",
                          role=_ROLE.GOVERNANCE_APPROVER, reason="confirmed regression",
                          validation=_valid(item, mf), reviewed_at=FIXED_NOW)
    assert out.item.status == _S.ACTION_REQUESTED
    assert out.action_request is not None
    ar = out.action_request
    assert ar.requested_action == rrq.ActionRequestType.REQUEST_DEACTIVATION
    assert ar.status == rrq.ActionRequestStatus.REQUESTED
    assert ar.required_execution_approval_type == "deactivation_approval"
    assert out.record.action_request_id == ar.action_request_id


def test_stale_item_cannot_be_approved():
    item, mf = _item()
    drift = manifest(packs=(("p1", "1.0", "packfp-CHANGED", "src1", "r1"),))
    stale_validation = rrq.validate_review_item(item, current_state=drift, now=FIXED_NOW)
    try:
        rrq.review_item(item, _D.APPROVE, reviewer="gov",
                        role=_ROLE.GOVERNANCE_APPROVER, reason="x",
                        validation=stale_validation)
    except ValueError:
        return
    raise AssertionError("a stale item must not be approvable")


def test_approval_without_validation_is_rejected():
    item, mf = _item()
    try:
        rrq.review_item(item, _D.APPROVE, reviewer="gov",
                        role=_ROLE.GOVERNANCE_APPROVER, reason="x")
    except ValueError:
        return
    raise AssertionError("approval requires a validation")


def test_unresolved_evidence_request_blocks_approval():
    item, mf = _item()
    out = rrq.review_item(item, _D.REQUEST_MORE_EVIDENCE, reviewer="alice",
                          role=_ROLE.MONITORING_REVIEWER, reason="ambiguous",
                          reviewed_at=FIXED_NOW)
    try:
        rrq.review_item(out.item, _D.APPROVE, reviewer="gov",
                        role=_ROLE.GOVERNANCE_APPROVER, reason="now sure",
                        validation=_valid(out.item, mf))
    except ValueError:
        return
    raise AssertionError("an unresolved evidence request must block approval")


def test_terminal_item_cannot_be_re_decided():
    item, mf = _item()
    out = rrq.review_item(item, _D.REJECT, reviewer="alice",
                          role=_ROLE.MONITORING_REVIEWER, reason="no",
                          reviewed_at=FIXED_NOW)
    try:
        rrq.review_item(out.item, _D.APPROVE, reviewer="gov",
                        role=_ROLE.GOVERNANCE_APPROVER, reason="x",
                        validation=_valid(out.item, mf))
    except ValueError:
        return
    raise AssertionError("a rejected item must not be re-decided")


def test_record_binds_evidence_fingerprints():
    item, mf = _item()
    out = rrq.review_item(item, _D.REJECT, reviewer="alice",
                          role=_ROLE.MONITORING_REVIEWER, reason="no",
                          reviewed_at=FIXED_NOW)
    rec = out.record
    assert rec.monitoring_run_fingerprint == item.monitoring_run_fingerprint
    assert rec.recommendation_fingerprint == item.recommendation_fingerprint
    assert rec.active_state_hash_at_review == item.active_state_hash
    assert rec.reviewer == "alice"
