"""Tests for the v2.3 Value Sprint harness (``src/agent/value_sprint_harness.py``).

The harness is read-only measurement built on top of the **frozen** assistant
path. It changes no semantics: it runs queries, records what the upstream
composer / grounding / AnswerGuard already decided, and emits proposals and
reports. These tests assert the *plumbing* — that each trust signal is captured
faithfully — rather than re-asserting per-row routing (which depends on the
over-permissive backend and is documented as an honest finding).

Two service shapes are used:

* a bare memory-only ``WorkbenchService`` (no knowledge pack) to exercise the
  memory-routed behaviours — conflict surfacing, model-prior fallback, honest
  refusal — without the over-permissive knowledge backend masking them;
* the real M365 / Coding pack built into ``tmp_path`` to exercise stale-source
  flagging and the end-to-end report against the canonical query file.

Nothing here touches geometry, verifier, grounding policy, lifecycle, citation
semantics, MemoryLedger, or KnowledgeLibrary writes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import pack_builder  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from agent.value_sprint_harness import (  # noqa: E402
    DEMO_SEED_MEMORIES,
    EXPECTED_CONFLICT,
    EXPECTED_HONEST_REFUSAL,
    EXPECTED_MODEL_PRIOR,
    EXPECTED_PACK_GAP,
    EXPECTED_STALE_FLAGGED,
    OUTCOME_HONEST_REFUSAL,
    SprintQuery,
    extract_pack_gap,
    load_queries,
    load_rows,
    run_query,
    run_sprint,
    summarize,
    write_reports,
)

_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_QUERIES = ROOT / "demos" / "value_sprint_queries.jsonl"

# The verbatim PowerApps section (a STALE knowledge source) — used to confirm
# stale-source flagging is carried through the harness row.
_POWERAPPS_QUERY = (
    "Power Fx is the low-code formula language used in Power Apps canvas apps; "
    "it is declarative and spreadsheet-like, recalculating values when their "
    "inputs change."
)


def _pack_service(tmp_path, monkeypatch):
    """Build the real M365 pack into an isolated registry (deterministic)."""
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    return pack_builder.open_pack_service(registry, report.pack_id)


def _memory_service():
    """A bare memory-only service seeded with the demo project memories."""
    service = WorkbenchService()
    for text in DEMO_SEED_MEMORIES:
        service.add_memory(text, source="value-sprint-test")
    return service


# 1. the runner works fully offline and writes both reports. ------------------

def test_runner_offline_and_reports_generated(tmp_path, monkeypatch):
    service = _pack_service(tmp_path, monkeypatch)
    for text in DEMO_SEED_MEMORIES:
        service.add_memory(text, source="value-sprint-test")
    queries = load_queries(_QUERIES)
    assert len(queries) >= 25  # the value query set is realistic and broad

    rows = run_sprint(service, queries)
    summary = summarize(rows)

    md = tmp_path / "value_sprint_latest.md"
    jsonl = tmp_path / "value_sprint_latest.jsonl"
    write_reports(rows, summary, md_path=md, jsonl_path=jsonl,
                  pack_label="m365_coding_assistant", backend_label="deterministic")

    assert md.exists() and jsonl.exists()
    assert "v2.3 Value Sprint report" in md.read_text(encoding="utf-8")
    # The JSONL round-trips back into rows.
    reloaded = load_rows(jsonl)
    assert len(reloaded) == len(rows)
    # Summary tallies are self-consistent with the rows.
    assert summary.query_count == len(rows)
    assert summary.model_prior_count >= 1
    assert summary.pack_gap_count >= 1
    assert summary.refused_count >= 1
    assert summary.stale_flagged_count >= 1


# 2. the deterministic backend is the default and is recorded per row. --------

def test_deterministic_backend_is_default(tmp_path, monkeypatch):
    service = _pack_service(tmp_path, monkeypatch)
    rows = run_sprint(service, load_queries(_QUERIES))
    assert rows
    assert all(r.retrieval_backend == "deterministic" for r in rows)


# 3. an unseeded decision-recall query is refused and labelled honestly. ------

def test_refused_query_labelled(tmp_path, monkeypatch):
    service = _memory_service()
    spec = SprintQuery(
        query="What did we decide about the production database rollout plan?",
        note="unseeded decision recall",
        category="no_evidence",
        expected_outcome=EXPECTED_HONEST_REFUSAL)
    row = run_query(service, spec, retrieval_backend="deterministic")
    assert row.refused is True
    assert row.outcome_label == OUTCOME_HONEST_REFUSAL
    assert row.pack_gap_detected is False  # an honest refusal is not a gap


# 4. a near-miss of a seeded memory surfaces a conflict warning. --------------

def test_conflict_warning_carried(tmp_path, monkeypatch):
    service = _memory_service()  # seeds "...Friday afternoon"
    spec = SprintQuery(
        query="the supplier delivery is on Monday afternoon",
        note="near-miss of seeded Friday memory",
        category="conflict",
        expected_outcome=EXPECTED_CONFLICT)
    row = run_query(service, spec, retrieval_backend="deterministic")
    assert row.conflict_warning is True


# 5. a labelled model-prior fallback is captured (and not mistaken for refusal).

def test_model_prior_labelled(tmp_path, monkeypatch):
    service = _memory_service()
    spec = SprintQuery(
        query="What did we decide about the mobile app public launch date?",
        note="unseeded decision; model prior allowed",
        category="model_prior",
        expected_outcome=EXPECTED_MODEL_PRIOR,
        allow_model_prior=True)
    row = run_query(service, spec, retrieval_backend="deterministic")
    assert row.model_prior_used is True
    assert row.refused is False


# 6. the verbatim PowerApps query flags its stale source. ---------------------

def test_stale_warning_carried(tmp_path, monkeypatch):
    service = _pack_service(tmp_path, monkeypatch)
    spec = SprintQuery(
        query=_POWERAPPS_QUERY,
        note="PowerApps stale source",
        category="powerapps_formula",
        expected_outcome=EXPECTED_STALE_FLAGGED)
    row = run_query(service, spec, retrieval_backend="deterministic")
    assert row.stale_warning is True


# 7. the AnswerGuard verdict is captured on every row. ------------------------

def test_guard_verdict_captured(tmp_path, monkeypatch):
    service = _pack_service(tmp_path, monkeypatch)
    rows = run_sprint(service, load_queries(_QUERIES))
    assert all(r.guard_verdict in {"ACCEPT", "REJECT"} for r in rows)


# 8. pack gaps are proposed but never written to memory or knowledge. ---------

def test_pack_gaps_suggested_but_not_written(tmp_path, monkeypatch):
    service = _memory_service()
    before = len(service.ledger.entries())

    spec = SprintQuery(
        query="What did we decide about the data residency review vendor?",
        note="missing decision",
        category="project_decision",
        expected_outcome=EXPECTED_PACK_GAP)
    row = run_query(service, spec, retrieval_backend="deterministic")

    # A proposal is produced ...
    assert row.pack_gap_detected is True
    assert row.suggested_pack_update is not None
    assert row.suggested_pack_update["suggested_source_type"] == "memory_proposal"
    # ... but nothing was written to the ledger.
    assert len(service.ledger.entries()) == before


def test_extract_pack_gap_is_advisory_only():
    spec = SprintQuery(
        query="What did we decide about the data residency review vendor?",
        category="project_decision",
        expected_outcome=EXPECTED_PACK_GAP)
    gap = extract_pack_gap(spec, grounded=False)
    assert gap is not None
    assert gap.affected_query == spec.query
    # A grounded value query produces no gap.
    grounded_spec = SprintQuery(query="anything", expected_outcome="grounded_useful")
    assert extract_pack_gap(grounded_spec, grounded=True) is None


def test_report_metrics_extended_with_fluency_and_inert_by_default():
    """v2.8: the report-metrics tuple grows to eight (v2.7 partition + v2.8
    fluency) and every report field defaults to zero on a non-report row."""
    from agent.value_sprint_harness import SprintRow, _report_metrics
    from slm.assistant_composer import (
        SECTION_FACTUAL,
        SECTION_JUDGEMENT,
        AnswerSpan,
        ReportSection,
        ReportStructure,
    )

    # No report -> eight zeros (inert outside report mode).
    assert _report_metrics(None) == (0, 0, 0, 0, 0, 0, 0, 0)

    # The new report-fluency fields exist on SprintRow and default to zero
    # (additive and backward-compatible; older rows deserialise unchanged).
    from dataclasses import fields

    defaults = {f.name: f.default for f in fields(SprintRow)}
    for name in ("report_duplicate_span_count", "report_truncated_span_count",
                 "report_section_overlap_count",
                 "report_judgement_placeholder_count"):
        assert name in defaults, name
        assert defaults[name] == 0

    # A small report produces the eight-tuple in (partition..., fluency...) order.
    report = ReportStructure(sections=[
        ReportSection(title="Executive summary", kind=SECTION_FACTUAL,
                      spans=[AnswerSpan(text="Pydantic validates input.",
                                        citation_id="src:a", source_name="a")]),
        ReportSection(title="Recommendation", kind=SECTION_JUDGEMENT,
                      judgement_text="[JUDGEMENT — not grounded in evidence]\nx",
                      is_placeholder=True),
    ])
    metrics = _report_metrics(report)
    assert len(metrics) == 8
    assert metrics[0] == 1  # one factual section
    assert metrics[1] == 1  # one cited claim
    assert metrics[2] == 1  # one judgement block
    assert metrics[7] == 1  # one judgement placeholder

