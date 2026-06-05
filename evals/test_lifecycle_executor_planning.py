"""Deterministic execution plan tests for the v7.3 executor."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.lifecycle_action_executor as lae  # noqa: E402
import agent.regression_review_queue as rrq  # noqa: E402
from _lifecycle_executor_helpers import (  # noqa: E402
    activate_audit_record, approved_action_request, execution_approval, manifest,
    two_pack_manifest,
)

_RECCODE = rrq.MonitoringRecommendationCode


def test_deactivation_plan_expects_state_change():
    mf = two_pack_manifest()
    ar, _, _ = approved_action_request(mf=mf, candidate_pack_ids=("p1",))
    approval = execution_approval(ar)
    plan = lae.build_execution_plan(ar, approval, mf)
    assert plan.lifecycle_primitive == "knowledge_pack_activation.deactivate"
    assert plan.expected_state_hash != plan.current_state_hash
    assert "config/active_knowledge_packs.jsonl" in plan.files_expected_to_change
    assert any("pack" in f for f in plan.files_expected_unchanged)


def test_plan_fingerprint_is_deterministic():
    mf = two_pack_manifest()
    ar, _, _ = approved_action_request(mf=mf, candidate_pack_ids=("p1",))
    approval = execution_approval(ar)
    p1 = lae.build_execution_plan(ar, approval, mf)
    p2 = lae.build_execution_plan(ar, approval, mf)
    assert p1.plan_fingerprint == p2.plan_fingerprint
    assert p1.plan_fingerprint.startswith("lifeplan-")


def test_rollback_plan_targets_prior_state():
    prior = manifest(packs=(("p1", "1.0", "packfp-a", "src1", "r1"),))
    current = two_pack_manifest()
    ar, _, _ = approved_action_request(
        recommendation=_RECCODE.ROLLBACK_RECOMMENDED, mf=current,
        candidate_pack_ids=("p2",), rollback_target=prior.state_hash)
    approval = execution_approval(ar)
    plan = lae.build_execution_plan(
        ar, approval, current, lifecycle_history=[activate_audit_record(prior)])
    assert plan.lifecycle_primitive == "knowledge_pack_activation.rollback"
    assert plan.expected_state_hash == prior.state_hash
    assert not plan.risk_findings


def test_activation_block_plan_does_not_change_active_state():
    mf = two_pack_manifest()
    ar, _, _ = approved_action_request(
        recommendation=_RECCODE.BLOCK_FUTURE_ACTIVATION, mf=mf,
        candidate_pack_ids=("p1",))
    approval = execution_approval(ar)
    plan = lae.build_execution_plan(ar, approval, mf)
    assert plan.lifecycle_primitive == "activation_block_record"
    assert plan.expected_state_hash == plan.current_state_hash


def test_follow_up_plan_does_not_change_active_state():
    mf = two_pack_manifest()
    ar, _, _ = approved_action_request(
        recommendation=_RECCODE.INVESTIGATE, mf=mf, candidate_pack_ids=("p1",))
    approval = execution_approval(ar)
    plan = lae.build_execution_plan(ar, approval, mf)
    assert plan.lifecycle_primitive == "operational_follow_up"
    assert plan.expected_state_hash == plan.current_state_hash
