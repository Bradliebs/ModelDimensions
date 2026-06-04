"""Tests for the v6.2A Clean Data Intake Lane (assessment-only, read-only).

The intake lane assesses an external dataset's *declared metadata* before it may
become eval samples or trusted knowledge. It downloads nothing, writes no memory
ledger, writes no source registry, creates no source/memory proposal, and writes
to disk only when an explicit ``--out`` path is given. These tests pin that
contract and the required decision invariants:

* a missing licence, an unknown provenance, possible/declared PII, and a
  non-commercial or research-only licence can never auto-classify as
  ``approved_for_knowledge``;
* ``approved_for_eval`` is a distinct, weaker tier than
  ``approved_for_knowledge``;
* assessment is deterministic; stdout mode writes nothing; ``--out`` writes only
  the assessment report;
* the module imports no durable-state writer.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import data_intake as di  # noqa: E402
from agent.data_intake import (  # noqa: E402
    DataIntakeAssessment,
    DataIntakeDecision,
    DataIntakeFinding,
    ExternalDatasetCandidate,
    FindingCode,
    IntakeLane,
    assess_candidate,
    assess_candidates,
    load_candidates,
    write_assessment_report,
)

_DEMO_CANDIDATES = ROOT / "demos" / "data_intake_candidates.jsonl"
_README = ROOT / "README.md"

_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)


def _open_knowledge_candidate(**overrides) -> ExternalDatasetCandidate:
    """An open-licensed candidate that is affirmatively clean for knowledge."""
    base = dict(
        dataset_id="demo/open-qa",
        source_url="https://huggingface.co/datasets/demo/open-qa",
        source_type="qa_dataset",
        title="Open QA",
        description="Open-licensed question/answer pairs from public-domain text.",
        licence="cc-by-4.0",
        provenance="huggingface:demo/open-qa (curated)",
        publisher="Demo Collective",
        language="en",
        task_categories=("question-answering",),
        size_hint="42MB",
        last_updated="2025-11-01",
        intended_use="knowledge_base",
        contains_personal_data=False,
        is_synthetic=False,
        sample_available=True,
        notes="clean public corpus",
    )
    base.update(overrides)
    return ExternalDatasetCandidate(**base)


# 1. open licensed dataset can be approved_for_eval. -------------------------

def test_open_licensed_dataset_can_be_approved_for_eval():
    # A synthetic, open-licensed eval set lands in the distinct eval tier.
    candidate = _open_knowledge_candidate(
        dataset_id="demo/open-synth", licence="apache-2.0", is_synthetic=True,
        intended_use="evaluation")
    assessment = assess_candidate(candidate, now=_NOW)
    assert assessment.decision == DataIntakeDecision.APPROVED_FOR_EVAL
    assert assessment.permits_eval_use is True


# 2. open licensed dataset is knowledge only with good provenance + use. -----

def test_open_licensed_dataset_knowledge_only_when_provenance_and_use_ok():
    ok = assess_candidate(_open_knowledge_candidate(), now=_NOW)
    assert ok.decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE
    assert ok.lane == IntakeLane.KNOWLEDGE_CANDIDATE

    bad_provenance = assess_candidate(
        _open_knowledge_candidate(provenance="unknown"), now=_NOW)
    assert bad_provenance.approved_for_knowledge is False

    bad_use = assess_candidate(
        _open_knowledge_candidate(intended_use="benchmark"), now=_NOW)
    assert bad_use.approved_for_knowledge is False


# 3. missing licence is not approved_for_knowledge. --------------------------

def test_missing_licence_not_approved_for_knowledge():
    assessment = assess_candidate(
        _open_knowledge_candidate(licence=None), now=_NOW)
    assert assessment.approved_for_knowledge is False
    assert assessment.decision == DataIntakeDecision.NEEDS_REVIEW
    assert any(f.code == FindingCode.MISSING_LICENSE for f in assessment.findings)


# 4. unknown provenance is not approved_for_knowledge. -----------------------

def test_unknown_provenance_not_approved_for_knowledge():
    for provenance in ("", "unknown"):
        assessment = assess_candidate(
            _open_knowledge_candidate(provenance=provenance), now=_NOW)
        assert assessment.approved_for_knowledge is False
        assert any(f.code == FindingCode.UNKNOWN_PROVENANCE
                   for f in assessment.findings)


# 5. non-commercial licence is not approved_for_knowledge by default. --------

def test_non_commercial_licence_not_approved_for_knowledge():
    assessment = assess_candidate(
        _open_knowledge_candidate(licence="cc-by-nc-4.0"), now=_NOW)
    assert assessment.approved_for_knowledge is False
    assert assessment.decision == DataIntakeDecision.APPROVED_FOR_EVAL
    assert any(f.code == FindingCode.NON_COMMERCIAL_LICENSE
               for f in assessment.findings)


# 6. research-only licence is not approved_for_knowledge by default. ---------

def test_research_only_licence_not_approved_for_knowledge():
    assessment = assess_candidate(
        _open_knowledge_candidate(licence="research-only"), now=_NOW)
    assert assessment.approved_for_knowledge is False
    assert assessment.decision == DataIntakeDecision.APPROVED_FOR_EVAL
    assert any(f.code == FindingCode.RESEARCH_ONLY_LICENSE
               for f in assessment.findings)


# 7. possible / declared PII is blocked or needs_review. ---------------------

def test_possible_pii_is_blocked_or_needs_review():
    possible = assess_candidate(
        _open_knowledge_candidate(contains_personal_data=None), now=_NOW)
    assert possible.decision in (
        DataIntakeDecision.BLOCKED, DataIntakeDecision.NEEDS_REVIEW)
    assert possible.approved_for_knowledge is False


def test_declared_pii_is_blocked():
    declared = assess_candidate(
        _open_knowledge_candidate(contains_personal_data=True), now=_NOW)
    assert declared.decision == DataIntakeDecision.BLOCKED
    assert declared.blocked is True


# 8. approved_for_eval does not imply approved_for_knowledge. ----------------

def test_approved_for_eval_does_not_imply_knowledge():
    eval_only = assess_candidate(
        _open_knowledge_candidate(licence="cc-by-nc-4.0"), now=_NOW)
    assert eval_only.permits_eval_use is True
    assert eval_only.approved_for_knowledge is False
    assert (DataIntakeDecision.APPROVED_FOR_EVAL
            != DataIntakeDecision.APPROVED_FOR_KNOWLEDGE)


# 9. synthetic dataset can be eval_only. -------------------------------------

def test_synthetic_dataset_can_be_eval_only():
    assessment = assess_candidate(
        _open_knowledge_candidate(is_synthetic=True, intended_use="general"),
        now=_NOW)
    assert assessment.permits_eval_use is True
    assert assessment.approved_for_knowledge is False
    assert assessment.lane == IntakeLane.SYNTHETIC_EXAMPLES


# 10. blocked candidate cannot be a knowledge candidate. ---------------------

def test_blocked_candidate_cannot_be_knowledge_candidate():
    assessment = assess_candidate(
        _open_knowledge_candidate(contains_personal_data=True), now=_NOW)
    assert assessment.blocked is True
    assert assessment.lane != IntakeLane.KNOWLEDGE_CANDIDATE
    assert assessment.lane == IntakeLane.BLOCKED
    assert assessment.approved_for_knowledge is False


# 11. stale dataset is eval-only, not knowledge. -----------------------------

def test_stale_dataset_is_eval_only():
    assessment = assess_candidate(
        _open_knowledge_candidate(last_updated="2018-02-01"), now=_NOW)
    assert assessment.decision == DataIntakeDecision.APPROVED_FOR_EVAL
    assert any(f.code == FindingCode.STALE_DATASET for f in assessment.findings)


# 12. assessment is deterministic. -------------------------------------------

def test_assessment_is_deterministic():
    a = assess_candidate(_open_knowledge_candidate(), now=_NOW)
    b = assess_candidate(_open_knowledge_candidate(), now=_NOW)
    assert a.to_dict() == b.to_dict()


# 13. the demo candidate file assesses to the expected spread. ---------------

def test_demo_candidates_assess_as_expected():
    candidates = load_candidates(_DEMO_CANDIDATES)
    assert len(candidates) == 7
    by_id = {c.dataset_id: a for c, a in
             zip(candidates, assess_candidates(candidates, now=_NOW))}
    assert by_id["openqa/public-qa"].decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE
    assert by_id["synth/eval-instructions"].decision == DataIntakeDecision.APPROVED_FOR_EVAL
    assert by_id["scrape/no-licence"].approved_for_knowledge is False
    assert by_id["research/nc-corpus"].approved_for_knowledge is False
    assert by_id["forum/threads"].decision in (
        DataIntakeDecision.BLOCKED, DataIntakeDecision.NEEDS_REVIEW)
    assert by_id["misc/unknown-origin"].approved_for_knowledge is False
    assert by_id["legacy/old-news"].approved_for_knowledge is False


# 14. the module imports no durable-state writer. ----------------------------

def test_data_intake_imports_no_writer():
    import ast

    source = Path(di.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    forbidden = {
        "MemoryLedger", "save_registry", "load_registry",
        "save_memory_review_queue", "save_source_review_queue",
        "propose_source_updates", "build_memory_proposals",
        "agent.memory_ledger", "agent.source_registry",
        "agent.source_proposals", "agent.memory_proposals",
    }
    leaked = imported & forbidden
    assert not leaked, f"data intake must not import writers/proposers: {leaked}"
    code = source.replace(ast.get_docstring(tree, clean=False) or "", "", 1)
    for token in ("MemoryLedger", "save_registry", "propose_source",
                  "memory_review_queue", "source_review_queue"):
        assert token not in code, f"data intake must not reference {token!r}"


# 15. assessment writes nothing by default (stdout mode). --------------------

def test_assessment_writes_nothing_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    assess_candidate(_open_knowledge_candidate(), now=_NOW)
    assess_candidates(load_candidates(_DEMO_CANDIDATES), now=_NOW)
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after  # nothing was created


# 16. the report writer writes only the assessment report. -------------------

def test_write_assessment_report_only_on_explicit_path(tmp_path):
    import json

    assessment = assess_candidate(_open_knowledge_candidate(), now=_NOW)
    out = tmp_path / "nested" / "report.json"
    path = write_assessment_report(assessment, out)
    assert path == out
    assert out.exists()
    # The report is the only file created in the tree.
    created = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert created == [out]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["_record"] == "data_intake_assessment"
    assert payload["decision"] == "approved_for_knowledge"
    assert "findings" in payload


# 17. to_dict exposes the decision, lane, tiers, and findings. ---------------

def test_to_dict_exposes_decision_lane_and_findings():
    payload = assess_candidate(_open_knowledge_candidate(), now=_NOW).to_dict()
    for key in ("candidate", "decision", "lane", "approved_for_eval",
                "approved_for_knowledge", "permits_eval_use",
                "permits_knowledge_use", "blocked", "needs_review",
                "quarantined", "rationale", "findings"):
        assert key in payload
    assert isinstance(payload["findings"], list)


# 18. CLI: stdout mode writes nothing; --out writes only the report. ---------

def test_cli_stdout_mode_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    before = sorted(p.name for p in tmp_path.iterdir())
    capsys.readouterr()
    argv = ["data-intake", "assess", "--candidate", str(_DEMO_CANDIDATES),
            "--now", "2026-06-04T00:00:00+00:00"]
    assert workbench.main(argv) == 0
    out = capsys.readouterr().out
    assert "# Data intake assessment (v6.2A; assessment-only)" in out
    assert "Decision:" in out
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after  # stdout mode created no files


def test_cli_out_writes_only_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    out = tmp_path / "report.json"
    capsys.readouterr()
    argv = ["data-intake", "assess", "--candidate", str(_DEMO_CANDIDATES),
            "--now", "2026-06-04T00:00:00+00:00", "--out", str(out)]
    assert workbench.main(argv) == 0
    created = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert created == [out]  # the report is the only durable write


def test_cli_deterministic_output(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    argv = ["data-intake", "assess", "--candidate", str(_DEMO_CANDIDATES),
            "--now", "2026-06-04T00:00:00+00:00"]
    capsys.readouterr()
    assert workbench.main(argv) == 0
    first = capsys.readouterr().out
    assert workbench.main(argv) == 0
    second = capsys.readouterr().out
    assert first == second


# 19. README documents the v6.2A data intake lane. ---------------------------

def test_readme_documents_data_intake():
    text = _README.read_text(encoding="utf-8")
    assert "## v6.2A" in text
    assert "data intake" in text.lower()
    assert "assessment-only" in text.lower()
