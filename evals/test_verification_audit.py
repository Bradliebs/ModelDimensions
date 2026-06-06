"""Tests for the V1.1 verification audit layer."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.agent.verification_audit import (
    load_candidate_audit_cases,
    render_verification_audit_markdown,
    rule_of_three_upper_bound,
    run_verification_audit,
    wilson_interval,
)
from scripts.run_verification_audit import main as audit_main


FIXTURE = ROOT / "evals" / "fixtures" / "v1_1_independent_verification_cases.jsonl"


def test_given_zero_events_when_rule_of_three_then_reports_upper_bound():
    # Act
    upper = rule_of_three_upper_bound(0, 30)
    interval = wilson_interval(0, 30)

    # Assert
    assert upper == 0.1
    assert 0.0 <= interval.lower <= interval.upper < 0.12


def test_given_independent_fixture_when_loaded_then_contains_required_categories():
    # Act
    cases = load_candidate_audit_cases(FIXTURE)
    categories = {case.category for case in cases}

    # Assert
    assert len(cases) >= 20
    assert "preserved_paraphrase" in categories
    assert "negation_flip" in categories
    assert "number_flip" in categories
    assert "entity_swap" in categories
    assert "natural_near_miss" in categories


def test_given_fixture_when_audit_runs_then_metrics_include_counts_and_intervals():
    # Arrange
    cases = load_candidate_audit_cases(FIXTURE)

    # Act
    report = run_verification_audit(cases, thresholds=(0.35, 0.50, 0.80))
    metrics = {item.name: item for item in report.metrics}

    # Assert
    assert metrics["false_accept_rate"].total > 0
    assert metrics["false_accept_rate"].wilson_95.upper >= 0.0
    assert metrics["coverage"].positive < metrics["coverage"].total
    assert len(report.risk_coverage_curve) == 3


def test_given_fixture_when_rendered_then_report_names_scientific_bounds():
    # Arrange
    cases = load_candidate_audit_cases(FIXTURE)
    report = run_verification_audit(cases, thresholds=(0.50,))

    # Act
    markdown = render_verification_audit_markdown(report)

    # Assert
    assert "Wilson 95%" in markdown
    assert "Rule-of-three upper" in markdown
    assert "Risk / Coverage Curve" in markdown
    assert "Detector Ablations" in markdown


def test_given_negation_case_when_detector_ablated_then_false_accept_risk_is_visible():
    # Arrange
    cases = load_candidate_audit_cases(FIXTURE)

    # Act
    report = run_verification_audit(cases)
    by_detector = {item.detector_name: item for item in report.detector_ablations}

    # Assert
    negation = by_detector["detect_negation_flip"]
    assert "flip_rotation_negation" in negation.newly_accepted_case_ids
    assert negation.ablated_false_accepts > negation.baseline_false_accepts


def test_given_cli_when_run_then_writes_json_and_markdown(tmp_path: Path):
    # Arrange
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"

    # Act
    exit_code = audit_main([
        "--cases", str(FIXTURE),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
        "--thresholds", "0.50",
    ])

    # Assert
    assert exit_code == 0
    assert out_json.exists()
    assert "V1.1 verification audit" in out_md.read_text(encoding="utf-8")