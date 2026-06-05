"""v7.0 activation validation: approval + evaluation binding, fail-closed.

Pins that activation requires an explicit approval bound to the exact pack and
evaluation, that a passing evaluation alone never activates, and that drift in
fingerprint, manifest, licence, environment, scope, or expiry fails closed.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from datetime import timedelta
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


def _request(identity, *, evidence=None, approval=None, environment="default",
             current=kpa.PackLifecycleState.EVALUATED):
    return kpa.KnowledgePackActivationRequest(
        identity=identity,
        approval=approval or approval_for(identity, evidence=evidence),
        evidence=evidence,
        environment=environment,
        current_state=current,
    )


def _validate(identity, **kw):
    return kpa.validate_activation(
        _request(identity, **kw), state=kpa.ActivationStateManifest(), now=FIXED_NOW)


def test_happy_path_validates(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = passing_evidence(identity)
    result = _validate(identity, evidence=evidence,
                       approval=approval_for(identity, evidence=evidence))
    assert result.ok, [f.code.value for f in result.blocking_findings]
    assert result.to_state == kpa.PackLifecycleState.ACTIVE


def test_missing_evidence_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    result = _validate(identity, evidence=None,
                       approval=approval_for(identity))
    assert not result.ok
    codes = {f.code for f in result.blocking_findings}
    assert kpa.ActivationFindingCode.EVALUATION_NOT_PASSED in codes


def test_failing_evaluation_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = replace(passing_evidence(identity), passed=False, failed_case_count=2)
    result = _validate(identity, evidence=evidence,
                       approval=approval_for(identity, evidence=evidence))
    assert not result.ok
    codes = {f.code for f in result.blocking_findings}
    assert kpa.ActivationFindingCode.EVALUATION_NOT_PASSED in codes
    assert kpa.ActivationFindingCode.EVALUATION_BLOCKING_FAILURE in codes


def test_forbidden_source_bleed_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = replace(passing_evidence(identity), forbidden_source_hit_count=1)
    result = _validate(identity, evidence=evidence,
                       approval=approval_for(identity, evidence=evidence))
    assert kpa.ActivationFindingCode.EVALUATION_FORBIDDEN_BLEED in {
        f.code for f in result.blocking_findings}


def test_evidence_bound_to_other_pack_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = replace(passing_evidence(identity), pack_fingerprint="packfp-other")
    result = _validate(identity, evidence=evidence,
                       approval=approval_for(identity, evidence=evidence))
    assert kpa.ActivationFindingCode.EVALUATION_PACK_MISMATCH in {
        f.code for f in result.blocking_findings}


def test_approval_fingerprint_mismatch_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = passing_evidence(identity)
    approval = replace(approval_for(identity, evidence=evidence),
                       pack_fingerprint="packfp-stale")
    result = _validate(identity, evidence=evidence, approval=approval)
    assert kpa.ActivationFindingCode.PACK_FINGERPRINT_MISMATCH in {
        f.code for f in result.blocking_findings}


def test_approval_manifest_hash_mismatch_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = passing_evidence(identity)
    approval = replace(approval_for(identity, evidence=evidence),
                       manifest_hash="packmanifest-wrong")
    result = _validate(identity, evidence=evidence, approval=approval)
    assert kpa.ActivationFindingCode.MANIFEST_HASH_MISMATCH in {
        f.code for f in result.blocking_findings}


def test_expired_approval_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = passing_evidence(identity)
    expired = (FIXED_NOW - timedelta(days=1)).isoformat()
    approval = replace(approval_for(identity, evidence=evidence),
                       expires_at=expired)
    result = _validate(identity, evidence=evidence, approval=approval)
    assert kpa.ActivationFindingCode.APPROVAL_EXPIRED in {
        f.code for f in result.blocking_findings}


def test_wrong_environment_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = passing_evidence(identity)
    approval = approval_for(identity, evidence=evidence, environment="staging")
    result = _validate(identity, evidence=evidence, approval=approval,
                       environment="production")
    assert kpa.ActivationFindingCode.APPROVAL_ENVIRONMENT_MISMATCH in {
        f.code for f in result.blocking_findings}


def test_wrong_scope_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = passing_evidence(identity)
    approval = approval_for(identity, evidence=evidence,
                            scope=kpa.ActivationScope.EMERGENCY_DEACTIVATE)
    result = _validate(identity, evidence=evidence, approval=approval)
    assert kpa.ActivationFindingCode.APPROVAL_SCOPE_INVALID in {
        f.code for f in result.blocking_findings}


def test_eval_pack_cannot_activate_as_knowledge(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p", pack_kind="eval"))
    evidence = passing_evidence(identity)
    result = _validate(identity, evidence=evidence,
                       approval=approval_for(identity, evidence=evidence))
    assert kpa.ActivationFindingCode.PACK_NOT_KNOWLEDGE in {
        f.code for f in result.blocking_findings}


def test_licence_drift_after_approval_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = passing_evidence(identity)
    approval = approval_for(identity, evidence=evidence, bind_licence=True)
    drifted = replace(identity, licence_snapshot="changed-licence")
    result = kpa.validate_activation(
        kpa.KnowledgePackActivationRequest(
            identity=drifted, approval=approval, evidence=evidence,
            current_state=kpa.PackLifecycleState.EVALUATED),
        state=kpa.ActivationStateManifest(), now=FIXED_NOW)
    assert kpa.ActivationFindingCode.LICENCE_CHANGED in {
        f.code for f in result.blocking_findings}


def test_invalid_transition_from_imported_blocks(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = passing_evidence(identity)
    result = _validate(identity, evidence=evidence,
                       approval=approval_for(identity, evidence=evidence),
                       current=kpa.PackLifecycleState.IMPORTED)
    assert kpa.ActivationFindingCode.INVALID_STATE_TRANSITION in {
        f.code for f in result.blocking_findings}


def test_stale_evaluation_blocks_under_age_policy(tmp_path):
    identity = kpa.load_pack_identity(write_hf_pack(tmp_path / "p"))
    evidence = passing_evidence(identity)
    approval = approval_for(identity, evidence=evidence)
    policy = replace(kpa.DEFAULT_THRESHOLD_POLICY, max_evaluation_age_days=1)
    result = kpa.validate_activation(
        kpa.KnowledgePackActivationRequest(
            identity=identity, approval=approval, evidence=evidence,
            current_state=kpa.PackLifecycleState.EVALUATED),
        state=kpa.ActivationStateManifest(), threshold_policy=policy, now=FIXED_NOW)
    assert kpa.ActivationFindingCode.EVALUATION_STALE in {
        f.code for f in result.blocking_findings}


def test_transition_table():
    S = kpa.PackLifecycleState
    assert kpa.transition_allowed(S.EVALUATED, S.ACTIVATION_PENDING)
    assert kpa.transition_allowed(S.ACTIVATION_PENDING, S.ACTIVE)
    assert kpa.transition_allowed(S.ACTIVE, S.INACTIVE)
    assert kpa.transition_allowed(S.INACTIVE, S.ACTIVE)
    assert kpa.transition_allowed(S.ACTIVE, S.SUPERSEDED)
    assert not kpa.transition_allowed(S.IMPORTED, S.ACTIVE)
    assert not kpa.transition_allowed(S.EVALUATION_FAILED, S.ACTIVE)
    assert not kpa.transition_allowed(S.BLOCKED, S.ACTIVE)
    assert not kpa.transition_allowed(S.RETIRED, S.ACTIVE)
    assert not kpa.transition_allowed(S.SUPERSEDED, S.ACTIVE)
