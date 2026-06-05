"""v7.1 active-pack monitoring: metrics and source-status classification.

These pin the safety arithmetic: expected-source drops, wrong-source increases,
forbidden-source introductions, unsupported citations, unrelated-control
regressions and inactive/superseded pack retrievals are each measured; critical
cases are counted separately from the aggregate; and an aggregate that holds or
improves never erases the separate critical-failure count.
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


def _result(case_id, **kw):
    base = dict(
        case_id=case_id, case_class=apm.MonitoringCaseClass.PACK_RELEVANT,
        severity=apm.MonitoringSeverity.MEDIUM, hit=True,
        expected_source_recall=1.0, wrong_source_rate=0.0, passed=True)
    base.update(kw)
    return apm.MonitoringCaseResult(**base)


def test_critical_cases_are_counted_separately_from_aggregate():
    results = [
        _result("crit", severity=apm.MonitoringSeverity.CRITICAL,
                critical_case=True, passed=False),
        _result("ok1"), _result("ok2"), _result("ok3"),
    ]
    metrics = apm.compute_metrics(results)
    assert metrics.case_count == 4
    assert metrics.critical_case_count == 1
    assert metrics.critical_failed_count == 1
    assert metrics.passed_count == 3


def test_expected_source_recall_drop_is_measured():
    results = [_result("c1", expected_source_recall=0.5),
               _result("c2", expected_source_recall=1.0)]
    metrics = apm.compute_metrics(results)
    assert metrics.expected_source_recall == 0.75


def test_wrong_source_rate_increase_is_measured():
    results = [_result("c1", wrong_source_rate=0.4),
               _result("c2", wrong_source_rate=0.0)]
    metrics = apm.compute_metrics(results)
    assert metrics.wrong_source_rate == 0.2


def test_forbidden_introduction_is_gated_at_selected_and_cited():
    results = [_result("c1", selected_forbidden_source_count=1,
                       cited_forbidden_source_count=1,
                       raw_forbidden_source_hit_count=3)]
    metrics = apm.compute_metrics(results)
    # Gated count = selected + cited (raw is recorded separately, not gated).
    assert metrics.forbidden_source_hit_count == 2
    assert metrics.raw_forbidden_source_hit_count == 3


def test_unsupported_citation_is_measured():
    results = [_result("c1", unsupported_citation_count=2,
                       uncited_factual_claim_count=1)]
    metrics = apm.compute_metrics(results)
    assert metrics.unsupported_citation_count == 2
    assert metrics.uncited_factual_claim_count == 1


def test_unrelated_control_regression_is_counted():
    results = [
        _result("ctrl", case_class=apm.MonitoringCaseClass.UNRELATED_CONTROL,
                control_case=True, selected_forbidden_source_count=1,
                passed=False),
        _result("ok"),
    ]
    metrics = apm.compute_metrics(results)
    assert metrics.unrelated_case_regression_count == 1


def test_inactive_and_superseded_pack_hits_are_measured():
    results = [_result("c1", inactive_pack_hit_count=1, superseded_pack_hit_count=2,
                       retired_pack_hit_count=1)]
    metrics = apm.compute_metrics(results)
    assert metrics.inactive_pack_hit_count == 1
    assert metrics.superseded_pack_hit_count == 2
    assert metrics.retired_pack_hit_count == 1


def _state():
    records = (
        ActivePackState(pack_id="active", pack_version="2", pack_fingerprint="f2",
                        source_type="hf", source_id="s2", source_revision="r2",
                        status=PackLifecycleState.ACTIVE, activated_at=FIXED_NOW,
                        activation_approval_id="ap", evaluation_report_id="ev"),
        ActivePackState(pack_id="old", pack_version="1", pack_fingerprint="f1",
                        source_type="hf", source_id="s1", source_revision="r1",
                        status=PackLifecycleState.SUPERSEDED, activated_at=FIXED_NOW,
                        activation_approval_id="ap", evaluation_report_id="ev"),
        ActivePackState(pack_id="gone", pack_version="0", pack_fingerprint="f0",
                        source_type="hf", source_id="s0", source_revision="r0",
                        status=PackLifecycleState.RETIRED, activated_at=FIXED_NOW,
                        activation_approval_id="ap", evaluation_report_id="ev"),
    )
    return ActivationStateManifest(records=records)


def test_classify_pack_hits_distinguishes_status():
    inactive, superseded, retired, env = apm.classify_pack_hits(
        _state(), ["active", "old", "gone", "unknown"])
    assert superseded == 1   # 'old'
    assert retired == 1      # 'gone'
    assert inactive == 1     # 'unknown' (not governed-active)
    assert env == 0


def test_aggregate_improvement_does_not_mask_critical_failure():
    # Aggregate recall is perfect, but a critical case fails.
    results = [
        _result("crit", severity=apm.MonitoringSeverity.CRITICAL,
                critical_case=True, passed=False, expected_source_recall=1.0),
        _result("c1"), _result("c2"), _result("c3"),
    ]
    metrics = apm.compute_metrics(results)
    assert metrics.expected_source_recall == 1.0
    # The critical failure is still visible as a separate count.
    assert metrics.critical_failed_count == 1
