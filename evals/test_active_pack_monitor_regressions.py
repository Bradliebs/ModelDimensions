"""v7.1 active-pack monitoring: regression finding detection.

These pin which deterministic finding codes fire for which observed conditions,
that an incompatible baseline short-circuits comparison, that too small a sample
is flagged, and that critical findings sort ahead of everything else.
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
CORPUS_FP = "moncorpus-fixed"
RCFG = "retrcfg-fixed"


def _snapshot():
    rec = ActivePackState(
        pack_id="p1", pack_version="1.0", pack_fingerprint="packfp-a",
        source_type="hf", source_id="src1", source_revision="r1",
        status=PackLifecycleState.ACTIVE, activated_at=FIXED_NOW,
        activation_approval_id="ap1", evaluation_report_id="ev1")
    return apm.snapshot_from_state(
        ActivationStateManifest(records=(rec,)), captured_at=FIXED_NOW)


def _clean_result(case_id, **kw):
    base = dict(
        case_id=case_id, case_class=apm.MonitoringCaseClass.PACK_RELEVANT,
        severity=apm.MonitoringSeverity.MEDIUM, hit=True,
        expected_source_recall=1.0, wrong_source_rate=0.0, passed=True)
    base.update(kw)
    return apm.MonitoringCaseResult(**base)


def _baseline(results=None):
    results = results or [_clean_result(f"c{i}") for i in range(4)]
    return apm.create_baseline(
        baseline_id="b1", baseline_type=apm.BaselineType.PRE_ACTIVATION,
        snapshot=_snapshot(), corpus_fingerprint=CORPUS_FP,
        retrieval_config_fingerprint=RCFG, results=results, created_at=FIXED_NOW)


def _detect(results, baseline=None, corpus_fp=CORPUS_FP, rcfg=RCFG):
    return apm.detect_regressions(
        baseline=baseline or _baseline(), results=results, snapshot=_snapshot(),
        corpus_fingerprint=corpus_fp, retrieval_config_fingerprint=rcfg)


def _codes(findings):
    return {f.finding_code for f in findings}


def test_incompatible_baseline_short_circuits():
    findings = _detect([_clean_result(f"c{i}") for i in range(4)],
                       corpus_fp="moncorpus-different")
    codes = _codes(findings)
    assert codes == {apm.RegressionFindingCode.BASELINE_INCOMPATIBLE}


def test_forbidden_source_introduction_is_critical():
    results = [_clean_result("c0", selected_forbidden_source_count=1)] + [
        _clean_result(f"c{i}") for i in range(1, 4)]
    findings = _detect(results)
    forb = [f for f in findings
            if f.finding_code == apm.RegressionFindingCode.FORBIDDEN_SOURCE_INTRODUCED]
    assert forb and forb[0].is_critical
    assert forb[0].severity == apm.MonitoringSeverity.CRITICAL


def test_inactive_pack_retrieval_is_critical():
    results = [_clean_result("c0", inactive_pack_hit_count=1)] + [
        _clean_result(f"c{i}") for i in range(1, 4)]
    assert apm.RegressionFindingCode.INACTIVE_PACK_RETRIEVED in _codes(_detect(results))


def test_superseded_pack_retrieval_is_detected():
    results = [_clean_result("c0", superseded_pack_hit_count=1)] + [
        _clean_result(f"c{i}") for i in range(1, 4)]
    assert apm.RegressionFindingCode.SUPERSEDED_PACK_RETRIEVED in _codes(_detect(results))


def test_expected_source_drop_below_baseline_is_detected():
    results = [_clean_result("c0", expected_source_recall=0.0)] + [
        _clean_result(f"c{i}") for i in range(1, 4)]
    assert apm.RegressionFindingCode.EXPECTED_SOURCE_DROPPED in _codes(_detect(results))


def test_unsupported_citation_is_detected():
    results = [_clean_result("c0", unsupported_citation_count=1)] + [
        _clean_result(f"c{i}") for i in range(1, 4)]
    assert apm.RegressionFindingCode.UNSUPPORTED_CITATION_INTRODUCED in _codes(
        _detect(results))


def test_unrelated_control_regression_is_detected():
    base = _baseline([
        _clean_result("ctrl", case_class=apm.MonitoringCaseClass.UNRELATED_CONTROL,
                      control_case=True),
        _clean_result("c1"), _clean_result("c2"), _clean_result("c3")])
    results = [
        _clean_result("ctrl", case_class=apm.MonitoringCaseClass.UNRELATED_CONTROL,
                      control_case=True, passed=False),
        _clean_result("c1"), _clean_result("c2"), _clean_result("c3")]
    assert apm.RegressionFindingCode.UNRELATED_CASE_REGRESSED in _codes(
        _detect(results, baseline=base))


def test_insufficient_sample_is_flagged():
    findings = _detect([_clean_result("c0")])
    assert apm.RegressionFindingCode.INSUFFICIENT_SAMPLE in _codes(findings)


def test_critical_case_failure_is_flagged_and_sorts_first():
    results = [
        _clean_result("crit", severity=apm.MonitoringSeverity.CRITICAL,
                      critical_case=True, passed=False),
        _clean_result("c1", expected_source_recall=0.0),
        _clean_result("c2"), _clean_result("c3")]
    findings = _detect(results)
    codes = _codes(findings)
    assert apm.RegressionFindingCode.CRITICAL_CASE_FAILED in codes
    # Critical findings sort ahead of medium ones.
    assert findings[0].severity == apm.MonitoringSeverity.CRITICAL


def test_aggregate_improved_but_critical_failed_override():
    base = _baseline([
        _clean_result("crit", severity=apm.MonitoringSeverity.CRITICAL,
                      critical_case=True, expected_source_recall=1.0),
        _clean_result("c1", expected_source_recall=0.5),
        _clean_result("c2"), _clean_result("c3")])
    # Current: aggregate recall improves, but the critical case fails.
    results = [
        _clean_result("crit", severity=apm.MonitoringSeverity.CRITICAL,
                      critical_case=True, passed=False, expected_source_recall=1.0),
        _clean_result("c1", expected_source_recall=1.0),
        _clean_result("c2"), _clean_result("c3")]
    codes = _codes(_detect(results, baseline=base))
    assert apm.RegressionFindingCode.AGGREGATE_IMPROVED_BUT_CRITICAL_FAILED in codes


def test_clean_run_against_self_has_no_safety_findings():
    results = [_clean_result(f"c{i}") for i in range(4)]
    findings = _detect(results, baseline=_baseline(results))
    assert not [f for f in findings if f.is_critical]


def test_detection_is_deterministic():
    results = [_clean_result("c0", selected_forbidden_source_count=1)] + [
        _clean_result(f"c{i}") for i in range(1, 4)]
    a = [f.to_dict() for f in _detect(results)]
    b = [f.to_dict() for f in _detect(results)]
    assert a == b
