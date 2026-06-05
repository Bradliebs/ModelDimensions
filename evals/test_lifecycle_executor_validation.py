"""Independent revalidation (fail-closed) for the v7.3 executor."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.lifecycle_action_executor as lae  # noqa: E402
from _lifecycle_executor_helpers import (  # noqa: E402
    FIXED_DT, approved_action_request, deactivate_p2, execution_approval,
    two_pack_manifest,
)

_CODE = lae.ExecutionFindingCode


def _codes(validation):
    return {f.code for f in validation.findings}


def test_valid_deactivation_passes():
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf, candidate_pack_ids=("p1",))
    approval = execution_approval(ar)
    v = lae.validate_lifecycle_execution(
        ar, approval, mf, review_history=[rec], now=FIXED_DT)
    assert v.ok, _codes(v)


def test_missing_action_request_fails_closed():
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf)
    approval = execution_approval(ar)
    v = lae.validate_lifecycle_execution(
        None, approval, mf, review_history=[rec], now=FIXED_DT)
    assert not v.ok
    assert _CODE.ACTION_REQUEST_MISSING in _codes(v)


def test_missing_execution_approval_fails_closed():
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf)
    v = lae.validate_lifecycle_execution(
        ar, None, mf, review_history=[rec], now=FIXED_DT)
    assert not v.ok
    assert _CODE.EXECUTION_APPROVAL_MISSING in _codes(v)


def test_missing_review_record_fails_closed():
    mf = two_pack_manifest()
    ar, _, _ = approved_action_request(mf=mf)
    approval = execution_approval(ar)
    v = lae.validate_lifecycle_execution(
        ar, approval, mf, review_history=[], now=FIXED_DT)
    assert not v.ok
    assert _CODE.REVIEW_RECORD_MISSING in _codes(v)


def test_expired_approval_fails_closed():
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf)
    approval = execution_approval(ar, expires_at="2023-01-01T00:00:00+00:00")
    v = lae.validate_lifecycle_execution(
        ar, approval, mf, review_history=[rec], now=FIXED_DT)
    assert not v.ok
    assert _CODE.EXECUTION_APPROVAL_EXPIRED in _codes(v)


def test_consumed_single_use_approval_fails_closed():
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf)
    approval = execution_approval(ar)
    v = lae.validate_lifecycle_execution(
        ar, approval, mf, review_history=[rec], approval_consumed=True,
        now=FIXED_DT)
    assert not v.ok
    assert _CODE.EXECUTION_APPROVAL_CONSUMED in _codes(v)


def test_changed_active_state_fails_closed():
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf, candidate_pack_ids=("p1",))
    approval = execution_approval(ar)
    # Live state drifted after the approval was minted.
    drifted = deactivate_p2(mf)
    v = lae.validate_lifecycle_execution(
        ar, approval, drifted, review_history=[rec], now=FIXED_DT)
    assert not v.ok
    assert _CODE.ACTIVE_STATE_CHANGED in _codes(v)


def test_action_mismatch_fails_closed():
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf)
    approval = execution_approval(ar)
    tampered = replace(approval, approved_action="request_rollback")
    v = lae.validate_lifecycle_execution(
        ar, tampered, mf, review_history=[rec], now=FIXED_DT)
    assert not v.ok
    assert _CODE.APPROVAL_ACTION_MISMATCH in _codes(v)


def test_already_executed_action_request_fails_closed():
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf)
    approval = execution_approval(ar)
    executed = replace(ar, status=lae.ActionRequestStatus.EXECUTED)
    v = lae.validate_lifecycle_execution(
        executed, approval, mf, review_history=[rec], now=FIXED_DT)
    assert not v.ok
    assert _CODE.ACTION_REQUEST_NOT_ACTIONABLE in _codes(v) \
        or _CODE.ACTION_REQUEST_ALREADY_EXECUTED in _codes(v)
