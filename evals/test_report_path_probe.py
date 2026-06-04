"""Tests for the v3.0.1 report-path probe (measurement only).

These tests prove the probe is a faithful, **read-only** localisation layer:

* it traces each case through the full report path — raw candidates -> selected
  evidence -> final report citations — and reports the documented stage shape;
* its per-stage forbidden source/term detection works (proven on synthetic
  stage items so the assertion does not depend on retrieval contents);
* it is **deterministic** — the same pack and cases yield byte-identical results
  across two runs;
* it **mutates no state** — the MemoryLedger, the proposal queue, and the
  knowledge library are unchanged after a full probe;
* the existing v3.0 retrieval eval still behaves identically alongside it.

The pack is always built into an isolated ``tmp_path`` registry so the tracked
demo packs stay pristine. Nothing here changes retrieval, ranking, source
selection, grounding, composer, or memory behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from agent import pack_builder  # noqa: E402
from agent import retrieval_eval_harness as reh  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402

_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_CASES = ROOT / "demos" / "retrieval_report_path_cases.jsonl"
_V30_CASES = ROOT / "demos" / "retrieval_eval_cases.jsonl"


def _build_service(tmp_path, monkeypatch) -> WorkbenchService:
    """Build the assistant pack into an isolated registry and bind a service.

    Mirrors the v3.0 harness test: ``chdir`` to the repo root so the manifest's
    relative source paths resolve, write the pack under ``tmp_path``, and bind
    the hybrid (lexical) retrieval backend — the path the probe traces.
    """
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    pack = registry.get_pack(report.pack_id)
    return WorkbenchService.from_pack(
        pack, registry=registry, knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder())


# 1. the report-path case dataset loads with the documented shape. -----------

def test_probe_cases_load():
    cases = reh.load_cases(_CASES)
    assert len(cases) == 4
    by_id = {c.case_id: c for c in cases}
    assert "least-priv-not-copilot" in by_id
    near = by_id["least-priv-not-copilot"]
    assert near.expected_sources == ["Microsoft 365 Admin Patterns"]
    assert "SharePoint Teams Copilot Studio Notes" in near.forbidden_sources
    assert "Copilot Studio" in near.forbidden_topic_terms
    # the gap probe declares nothing to find.
    assert by_id["data-residency-gap"].has_expected is False


# 2. the probe reports the documented per-stage + summary metric shape. ------

def test_probe_metric_shape(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)
    results = reh.run_probe(service, cases)
    summary = reh.summarize_probe(results)

    assert summary.case_count == len(cases)
    for rate in (summary.raw_off_topic_rate,
                 summary.selected_evidence_off_topic_rate,
                 summary.final_citation_off_topic_rate):
        assert 0.0 <= rate <= 1.0
    assert set(summary.bleed_stage_counts) == {"raw", "selected", "final"}
    assert summary.bleed_reproduced == (summary.cases_with_bleed > 0)

    for r in results:
        data = r.to_dict()
        for key in ("raw", "selected", "final", "final_citations",
                    "bleed_introduced_stage"):
            assert key in data
        for stage_name in ("raw", "selected", "final"):
            obs = r.stage(stage_name)
            assert obs.stage == stage_name
            # chunk ids are de-duplicated, so never exceed the raw item count.
            assert obs.item_count >= len(obs.chunk_ids)
            assert obs.off_topic_count == len(obs.off_topic_hits)
        # the first-bleed stage is one of the three stages, or None.
        assert r.bleed_introduced_stage in (None, "raw", "selected", "final")


# 3. per-stage forbidden source/term detection actually fires. ---------------

def test_stage_detection_positive():
    """On synthetic stage items the bleed rule flags a forbidden source/term."""
    case = reh.RetrievalEvalCase(
        query="least privilege",
        expected_sources=["Microsoft 365 Admin Patterns"],
        forbidden_sources=["SharePoint Teams Copilot Studio Notes"],
        forbidden_topic_terms=["connector"],
    )
    items = [
        {"source_name": "Microsoft 365 Admin Patterns", "source_id": "",
         "chunk_id": "chk-1", "text": "use PIM for just-in-time activation"},
        {"source_name": "SharePoint Teams Copilot Studio Notes",
         "source_id": "", "chunk_id": "chk-2",
         "text": "connector availability varies for custom agents"},
    ]
    hits = reh._stage_off_topic_hits(case, items)
    # the forbidden source is flagged; the expected source is never counted.
    assert any("forbidden source" in h for h in hits)
    assert all("Admin Patterns" not in h for h in hits)
    # wrong-source rate: one of two items is not the expected source.
    assert reh._stage_wrong_source_rate(case, items) == 0.5


def test_stage_detection_negative():
    """A clean stage (only the expected source) yields no bleed."""
    case = reh.RetrievalEvalCase(
        query="least privilege",
        expected_sources=["Microsoft 365 Admin Patterns"],
        forbidden_topic_terms=["connector"],
    )
    items = [
        {"source_name": "Microsoft 365 Admin Patterns", "source_id": "",
         "chunk_id": "chk-1", "text": "least-privilege roles and PIM"},
    ]
    assert reh._stage_off_topic_hits(case, items) == []
    assert reh._stage_wrong_source_rate(case, items) == 0.0


# 4. the probe is deterministic for a fixed pack + cases. --------------------

def test_probe_deterministic(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)

    first = [r.to_dict() for r in reh.run_probe(service, cases)]
    second = [r.to_dict() for r in reh.run_probe(service, cases)]
    assert first == second

    s1 = reh.summarize_probe(reh.run_probe(service, cases)).to_dict()
    s2 = reh.summarize_probe(reh.run_probe(service, cases)).to_dict()
    assert s1 == s2


# 5. probing mutates no state (no memory / proposal / knowledge writes). ------

def test_probe_no_mutation(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)

    ledger_before = service.export_ledger()
    proposals_before = service.list_proposals()
    audit_before = service.query_knowledge(cases[0].query)
    sources_before = sorted(c["source_name"] for c in audit_before.candidates)

    reh.run_probe(service, cases)

    assert service.export_ledger() == ledger_before
    assert service.list_proposals() == proposals_before
    audit_after = service.query_knowledge(cases[0].query)
    sources_after = sorted(c["source_name"] for c in audit_after.candidates)
    assert sources_after == sources_before


# 6. the existing v3.0 retrieval eval still behaves identically alongside. ----

def test_v30_eval_unchanged_alongside_probe(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    v30_cases = reh.load_cases(_V30_CASES)

    baseline = {r.case_id: r.to_dict() for r in reh.run_eval(service, v30_cases)}
    # run the probe in between — it must leave the v3.0 eval result identical.
    reh.run_probe(service, reh.load_cases(_CASES))
    after = {r.case_id: r.to_dict() for r in reh.run_eval(service, v30_cases)}
    assert after == baseline
