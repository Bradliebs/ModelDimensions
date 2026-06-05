"""v7.1 active-pack monitoring: deterministic, advisory-only recommendations.

The recommender maps findings + attribution to exactly one code from a closed
set. It never performs a lifecycle action: a rollback recommendation is advice
for a human to act on through the separately governed rollback path, not a
rollback. Insufficient evidence blocks any strong keep/rollback recommendation.
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


def _snapshot():
    rec = ActivePackState(
        pack_id="p1", pack_version="1.0", pack_fingerprint="packfp-a",
        source_type="hf", source_id="src1", source_revision="r1",
        status=PackLifecycleState.ACTIVE, activated_at=FIXED_NOW,
        activation_approval_id="ap1", evaluation_report_id="ev1")
    return apm.snapshot_from_state(
        ActivationStateManifest(records=(rec,)), captured_at=FIXED_NOW)


def _finding(code, severity, case_id="c0"):
    return apm.RegressionFinding(finding_code=code, severity=severity,
                                case_ids=(case_id,))


def _recommend(findings, **kw):
    base = dict(findings=findings, metrics=apm.MonitoringMetrics(),
                snapshot=_snapshot(), generated_at=FIXED_NOW)
    base.update(kw)
    return apm.recommend(**base)


def _pack_specific():
    return apm.AttributionResult(apm.AttributionClass.PACK_SPECIFIC, ("p1",),
                                apm.ConfidenceBand.HIGH)


def test_clean_run_recommends_keep_active():
    rec = _recommend([])
    assert rec.recommendation == apm.MonitoringRecommendationCode.KEEP_ACTIVE


def test_minor_degradation_recommends_keep_active_with_watch():
    findings = [_finding(apm.RegressionFindingCode.DUPLICATE_INTERFERENCE_INCREASED,
                         apm.MonitoringSeverity.LOW)]
    rec = _recommend(findings)
    assert rec.recommendation == apm.MonitoringRecommendationCode.KEEP_ACTIVE_WITH_WATCH


def test_instability_recommends_investigate():
    findings = [_finding(apm.RegressionFindingCode.INSTABILITY_DETECTED,
                         apm.MonitoringSeverity.MEDIUM)]
    rec = _recommend(findings)
    assert rec.recommendation == apm.MonitoringRecommendationCode.INVESTIGATE


def test_isolated_material_high_regression_recommends_deactivate():
    findings = [_finding(apm.RegressionFindingCode.EXPECTED_SOURCE_DROPPED,
                         apm.MonitoringSeverity.HIGH)]
    rec = _recommend(findings, pack_attribution=_pack_specific())
    assert rec.recommendation == apm.MonitoringRecommendationCode.DEACTIVATE_RECOMMENDED


def test_critical_with_valid_rollback_and_pack_attribution_recommends_rollback():
    findings = [_finding(apm.RegressionFindingCode.FORBIDDEN_SOURCE_INTRODUCED,
                         apm.MonitoringSeverity.CRITICAL)]
    rec = _recommend(findings, pack_attribution=_pack_specific(),
                     rollback_target="pkgstate-known-good")
    assert rec.recommendation == apm.MonitoringRecommendationCode.ROLLBACK_RECOMMENDED
    assert rec.rollback_target == "pkgstate-known-good"
    assert rec.required_human_approval == "rollback_approval"


def test_critical_without_attribution_recommends_investigate():
    findings = [_finding(apm.RegressionFindingCode.CRITICAL_CASE_FAILED,
                         apm.MonitoringSeverity.CRITICAL)]
    rec = _recommend(findings)
    assert rec.recommendation == apm.MonitoringRecommendationCode.INVESTIGATE


def test_insufficient_sample_blocks_strong_recommendation():
    findings = [_finding(apm.RegressionFindingCode.INSUFFICIENT_SAMPLE,
                         apm.MonitoringSeverity.MEDIUM)]
    rec = _recommend(findings)
    assert rec.recommendation == (
        apm.MonitoringRecommendationCode.INSUFFICIENT_EVIDENCE_TO_RECOMMEND)


def test_incompatible_baseline_blocks_recommendation():
    findings = [_finding(apm.RegressionFindingCode.BASELINE_INCOMPATIBLE,
                         apm.MonitoringSeverity.HIGH)]
    rec = _recommend(findings)
    assert rec.recommendation == (
        apm.MonitoringRecommendationCode.INSUFFICIENT_EVIDENCE_TO_RECOMMEND)


def test_recommendation_is_advisory_only_and_performs_no_action():
    rec = _recommend([])
    assert rec.advisory_only is True
    assert rec.to_dict()["advisory_only"] is True
    # The recommendation object exposes no lifecycle action.
    for action in ("activate", "deactivate", "rollback", "supersede", "apply",
                   "execute", "write"):
        assert not hasattr(rec, action)


def test_recommendation_is_deterministic():
    findings = [_finding(apm.RegressionFindingCode.FORBIDDEN_SOURCE_INTRODUCED,
                         apm.MonitoringSeverity.CRITICAL)]
    a = _recommend(findings, pack_attribution=_pack_specific(),
                   rollback_target="t").to_dict()
    b = _recommend(findings, pack_attribution=_pack_specific(),
                   rollback_target="t").to_dict()
    assert a == b
