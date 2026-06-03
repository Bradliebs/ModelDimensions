"""Tests for the v2.1.1 Value Sprint instrumentation.

The Value Sprint (``src/agent/value_sprint.py``) is read-only measurement: it
runs a fixed query set through the **frozen** assistant path and records an
audit row per query. These tests build the real M365 / Coding pack into an
isolated ``tmp_path`` registry and assert the sprint's honest behaviour:

* it runs fully offline (template composer, no SLM);
* every row carries an audit trail (a route);
* verbatim section queries ground and cite;
* the verbatim PowerApps query flags its stale source;
* an empty-memory decision-recall query is refused (no evidence to ground);
* the over-permissive backend can ground an irrelevant chunk for an
  out-of-domain query (recorded honestly as a false grounding) yet the
  grounding guard still keeps forbidden terms out of the answer;
* the summary tallies match the rows.

Nothing here touches geometry, grounding policy, verifier, lifecycle, or
citation semantics. The pack is built into ``tmp_path`` so tracked sources stay
pristine.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import pack_builder  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent.value_sprint import (  # noqa: E402
    load_queries,
    render_markdown,
    run_value_sprint,
    summarize,
)

_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_QUERIES = ROOT / "demos" / "value_sprint_queries.jsonl"

_POWERAPPS_NOTE = "PowerApps Power Fx"
_AWS_FORBIDDEN = ("AWS", "S3 bucket", "Lambda")


def _service(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    return pack_builder.open_pack_service(registry, report.pack_id)


def _run(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    queries = load_queries(_QUERIES)
    rows = run_value_sprint(service, queries)
    return rows


# 1. the sprint runs offline and every row has an audit trail. --------------

def test_sprint_runs_with_audit_trail(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch)
    assert len(rows) == 10
    for row in rows:
        assert row.route, f"missing route for: {row.note}"
        assert row.composer_backend  # template composer recorded


# 2. verbatim section queries ground and cite. ------------------------------

def test_verbatim_queries_ground_and_cite(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch)
    grounded = [r for r in rows if r.grounded]
    # The five verbatim section queries should all ground with citations.
    assert len(grounded) >= 5
    for row in grounded:
        assert not row.refused
        assert row.citations_count > 0


# 3. the verbatim PowerApps query flags its stale source. -------------------

def test_powerapps_query_flags_stale(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch)
    powerapps = [r for r in rows if _POWERAPPS_NOTE in r.note and r.grounded]
    assert powerapps, "expected a grounded PowerApps row"
    assert any(r.stale_warning for r in powerapps), \
        "PowerApps source is stale; a stale caution was expected"


# 4. the empty-ledger decision-recall query refuses (no evidence). ----------

def test_decision_recall_refuses(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch)
    recall = [r for r in rows if "Decision recall" in r.note]
    assert recall, "expected the decision-recall row"
    for row in recall:
        assert row.refused, "empty-ledger decision recall should refuse"
        assert row.citations_count == 0


# 5. out-of-domain query: honest over-permissive grounding, no term leak. ----
# The deterministic backend grounds an irrelevant chunk (a false grounding)
# rather than refusing, but the grounding guard keeps forbidden AWS terms out.

def test_out_of_domain_false_grounding_no_leak(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch)
    aws = [r for r in rows if "Out-of-domain" in r.note]
    assert aws, "expected the out-of-domain AWS row"
    for row in aws:
        # Records the over-permissive retrieval honestly.
        assert row.false_grounding == (row.grounded
                                       and row.missing_evidence_expected)
        # Even when grounding an irrelevant source, forbidden terms never leak.
        for term in _AWS_FORBIDDEN:
            assert term.lower() not in row.answer_snippet.lower()


# 6. the summary tallies match the rows. ------------------------------------

def test_summary_matches_rows(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch)
    summary = summarize(rows)
    assert summary.query_count == len(rows)
    assert summary.grounded_count == sum(1 for r in rows if r.grounded)
    assert summary.refused_count == sum(1 for r in rows if r.refused)
    assert summary.stale_flagged_count == sum(
        1 for r in rows if r.stale_warning)
    assert summary.pack_gap_count == sum(1 for r in rows if r.pack_gap)
    assert summary.false_grounding_count == sum(
        1 for r in rows if r.false_grounding)
    # The report renders without error and surfaces the honest finding.
    md = render_markdown(rows, summary, pack_label="test-pack")
    assert "Value Sprint report" in md
    assert "Per-query audit" in md
    assert "over-permissive" in md.lower()
