"""End-to-end governed deactivation via the real v7.0 primitive."""
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


def _execute(tmp_path, *, write, prior_results=()):
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf, candidate_pack_ids=("p1",))
    mgr = manager(tmp_path, mf)
    approval = execution_approval(ar)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], prior_results=prior_results,
        write=write, now=FIXED_DT)
    return res, mgr, mf


def test_dry_run_writes_nothing(tmp_path):
    res, mgr, mf = _execute(tmp_path, write=False)
    assert res.status == lae.ExecutionStatus.DRY_RUN_READY
    assert not res.written
    assert mgr.load_state().state_hash == mf.state_hash


def test_write_deactivates_only_the_target(tmp_path):
    res, mgr, _ = _execute(tmp_path, write=True)
    assert res.status == lae.ExecutionStatus.EXECUTED, res.message
    assert res.verification.passed, res.verification.findings
    after = mgr.load_state()
    assert not after.find("p1").is_active
    assert after.find("p2").is_active  # unrelated pack untouched


def test_executed_action_request_is_marked(tmp_path):
    res, _, _ = _execute(tmp_path, write=True)
    assert res.updated_action_request.status == rrq.ActionRequestStatus.EXECUTED
    assert res.updated_action_request.lifecycle_audit_ref
    assert res.lifecycle_audit_ref  # the v7.0 primitive's audit hash


def test_audit_record_is_produced(tmp_path):
    res, _, _ = _execute(tmp_path, write=True)
    assert res.audit is not None
    assert res.audit.result_status == lae.ExecutionStatus.EXECUTED
    assert res.audit.pre_state_hash != res.audit.post_state_hash
    assert res.audit.lifecycle_primitive_called == \
        "knowledge_pack_activation.deactivate"


def test_review_approval_alone_cannot_execute(tmp_path):
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(mf=mf, candidate_pack_ids=("p1",))
    mgr = manager(tmp_path, mf)
    # An approved review record exists, but NO execution approval is supplied.
    res = lae.execute_lifecycle_action(
        ar, None, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    assert res.status == lae.ExecutionStatus.VALIDATION_FAILED
    assert not res.written
    assert mgr.load_state().state_hash == mf.state_hash
