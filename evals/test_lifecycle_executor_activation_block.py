"""Activation-block execution (no active-state mutation; exact fingerprint)."""
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


def _block_setup(tmp_path):
    mf = two_pack_manifest()
    ar, rec, _ = approved_action_request(
        recommendation=_RECCODE.BLOCK_FUTURE_ACTIVATION, mf=mf,
        candidate_pack_ids=("p1",))
    mgr = manager(tmp_path, mf)
    approval = execution_approval(ar)
    return ar, rec, mgr, approval, mf


def test_block_does_not_change_active_state(tmp_path):
    ar, rec, mgr, approval, mf = _block_setup(tmp_path)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    assert res.status == lae.ExecutionStatus.EXECUTED, res.message
    assert res.activation_block is not None
    # A block records intent; it does NOT deactivate the pack.
    after = mgr.load_state()
    assert after.state_hash == mf.state_hash
    assert after.find("p1").is_active


def test_block_targets_exact_fingerprint(tmp_path):
    ar, rec, mgr, approval, mf = _block_setup(tmp_path)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    block = res.activation_block
    assert block.pack_fingerprint == "packfp-a"
    assert lae.is_fingerprint_blocked([block], "packfp-a")
    assert not lae.is_fingerprint_blocked([block], "packfp-b")


def test_block_is_idempotent(tmp_path):
    ar, rec, mgr, approval, _ = _block_setup(tmp_path)
    first = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=True, now=FIXED_DT)
    second = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], prior_blocks=[first.activation_block],
        write=True, now=FIXED_DT)
    assert second.status == lae.ExecutionStatus.ALREADY_EXECUTED


def test_block_dry_run_writes_nothing(tmp_path):
    ar, rec, mgr, approval, mf = _block_setup(tmp_path)
    res = lae.execute_lifecycle_action(
        ar, approval, executor_identity="op", executor_role="lifecycle_operator",
        state_manager=mgr, review_history=[rec], write=False, now=FIXED_DT)
    assert res.status == lae.ExecutionStatus.DRY_RUN_READY
    assert mgr.load_state().state_hash == mf.state_hash
