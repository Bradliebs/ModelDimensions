"""Shared fixtures for the v7.2 regression review queue tests.

Builds deterministic monitoring-run dicts (as ``MonitoringRun.to_dict()`` would
produce) and the matching activation manifest, so review-item identities,
fingerprints and staleness checks are reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.active_pack_monitor as apm  # noqa: E402
from agent.knowledge_pack_activation import (  # noqa: E402
    ActivationStateManifest,
    ActivePackState,
    PackLifecycleState,
)

FIXED_NOW = "2024-01-01T00:00:00+00:00"

_REC = apm.MonitoringRecommendationCode
_SEV = apm.MonitoringSeverity
_CONF = apm.ConfidenceBand
_FCODE = apm.RegressionFindingCode


def manifest(packs=(("p1", "1.0", "packfp-a", "src1", "r1"),)):
    """Build an activation manifest of ACTIVE packs from compact tuples."""
    records = tuple(
        ActivePackState(
            pack_id=pid, pack_version=ver, pack_fingerprint=fp,
            source_type="hf", source_id=sid, source_revision=rev,
            status=PackLifecycleState.ACTIVE, activated_at=FIXED_NOW,
            activation_approval_id="ap1", evaluation_report_id="ev1")
        for pid, ver, fp, sid, rev in packs)
    return ActivationStateManifest(records=records)


def run_dict(*, recommendation=_REC.DEACTIVATE_RECOMMENDED,
             confidence=_CONF.HIGH, severity=_SEV.CRITICAL,
             manifest_obj=None, candidate_pack_ids=("p1",),
             affected_cases=("c1",), rationale=("expected_source_dropped",),
             rollback_target="", finding_code=_FCODE.EXPECTED_SOURCE_DROPPED,
             baseline_hash="monbase-1", corpus_fingerprint="moncorpus-1",
             retrieval_config_fingerprint="retrcfg-1",
             policy_fingerprint="polfp-1", created_at=FIXED_NOW):
    """Produce a monitoring-run dict for a chosen recommendation (deterministic)."""
    mf = manifest_obj if manifest_obj is not None else manifest()
    snapshot = apm.snapshot_from_state(mf, captured_at=created_at)
    rec = apm.MonitoringRecommendation(
        recommendation=recommendation, rationale_codes=tuple(rationale),
        confidence=confidence, affected_cases=tuple(affected_cases),
        candidate_pack_ids=tuple(candidate_pack_ids),
        rollback_target=rollback_target, generated_at=created_at)
    findings = (apm.RegressionFinding(
        finding_code=finding_code, severity=severity,
        case_ids=tuple(affected_cases), candidate_pack_ids=tuple(candidate_pack_ids),
        confidence=confidence),) if severity is not None else ()
    run = apm.MonitoringRun(
        monitoring_run_id="monrun-fixed1", snapshot=snapshot,
        baseline_hash=baseline_hash, corpus_fingerprint=corpus_fingerprint,
        retrieval_config_fingerprint=retrieval_config_fingerprint,
        policy_id="monitor-strict-v1", policy_fingerprint=policy_fingerprint,
        metrics=apm.MonitoringMetrics(), deltas=(), findings=findings,
        recommendation=rec, created_at=created_at)
    return run.to_dict()
