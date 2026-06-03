"""Tests for the v2.0 M365 + coding assistant pack.

This is the first real *operational* assistant pack: it combines Microsoft 365
governance notes (domain ``microsoft``) with this project's own coding stack
notes (domain ``coding``) under ``demos/m365_coding_sources/``. The tests
exercise it end to end — the spec loads, builds from local Markdown with full
provenance, its evaluation questions run through the frozen knowledge query
path, both domains survive into the knowledge library, the combined
memory/knowledge/model-prior routing reports the right flags offline, and no
imported knowledge leaks into the MemoryLedger. Nothing here touches concept-cell
geometry, grounding policy, or lifecycle semantics; the pack is always built into
an isolated ``tmp_path`` registry so the tracked demo packs stay pristine.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.project_packs import PackRegistry  # noqa: E402
from agent.review_service import ReviewService  # noqa: E402
from agent import pack_builder, pack_evaluator  # noqa: E402

_SPEC = ROOT / "packs" / "starter" / "m365_coding_assistant.yaml"
_EVAL = ROOT / "demos" / "pack_eval_questions" / "m365_coding_assistant_eval.jsonl"

_MICROSOFT_SOURCES = {
    "Purview Sensitivity Labels",
    "Endpoint DLP",
    "Copilot Studio Agent Patterns",
    "Power Platform DLP",
    "SharePoint Governance",
}
_CODING_SOURCES = {
    "Pydantic v2 Notes",
    "Streamlit Workbench Notes",
    "PowerShell Launcher Notes",
}


def _build(tmp_path, monkeypatch) -> tuple[PackRegistry, str]:
    """Build the assistant pack from local docs into an isolated registry.

    ``chdir`` to the repo root so the spec's relative source paths resolve, but
    write the built pack under ``tmp_path`` so the test leaves no trace.
    """
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_SPEC)
    report = pack_builder.build_pack(plan, registry)
    return registry, report.pack_id


# 1. the pack spec loads with the expected provenance defaults. -------------

def test_pack_spec_loads():
    plan = pack_builder.PackBuildPlan.from_file(_SPEC)
    assert plan.pack_name == "m365-coding-assistant"
    assert plan.enabled is True
    assert plan.allow_url_ingestion is False
    assert len(plan.sources) == 8
    domains = {s.domain for s in plan.sources}
    assert domains == {"microsoft", "coding"}
    for source in plan.sources:
        assert source.authority == "reputable"
        assert source.version == "local-notes-v1"
        assert source.staleness_policy == "review_required"


# 2. the pack builds from the local docs (no skips). ------------------------

def test_pack_builds_from_local_docs(tmp_path, monkeypatch):
    registry, pack_id = _build(tmp_path, monkeypatch)
    service = pack_builder.open_pack_service(registry, pack_id)
    names = {s["source_name"] for s in service.list_knowledge_sources()}
    assert _MICROSOFT_SOURCES <= names
    assert _CODING_SOURCES <= names
    assert len(names) == 8


# 3. the build preserves domain / authority / version metadata. -------------

def test_pack_preserves_source_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_SPEC)
    report = pack_builder.build_pack(plan, registry)

    assert report.source_count == 8
    assert not report.skipped

    purview = next(s for s in report.sources
                   if s["source_name"] == "Purview Sensitivity Labels")
    assert purview["domain"] == "microsoft"
    assert purview["authority"] == "reputable"
    assert purview["version"] == "local-notes-v1"
    assert purview["staleness_policy"] == "review_required"
    assert purview["chunks"] >= 1

    pydantic = next(s for s in report.sources
                    if s["source_name"] == "Pydantic v2 Notes")
    assert pydantic["domain"] == "coding"
    assert pydantic["authority"] == "reputable"
    assert pydantic["version"] == "local-notes-v1"
    assert pydantic["chunks"] >= 1


# 4. the evaluation questions run and pass against the built pack. ----------

def test_eval_questions_run(tmp_path, monkeypatch):
    registry, pack_id = _build(tmp_path, monkeypatch)
    service = pack_builder.open_pack_service(registry, pack_id)

    results = pack_evaluator.evaluate_pack_file(service, _EVAL)
    assert results, "expected at least one eval question"
    failed = [r for r in results if not r.passed]
    assert not failed, "; ".join(f"{r.question_id}: {r.reason}" for r in failed)


# 5. a wrong-topic question does not pass against an unrelated source. ------

def test_wrong_topic_does_not_pass_against_unrelated_source(tmp_path,
                                                            monkeypatch):
    registry, pack_id = _build(tmp_path, monkeypatch)
    service = pack_builder.open_pack_service(registry, pack_id)

    # The pack has no AWS content, so asserting it as the expected source for a
    # cross-cloud query must fail rather than fabricate a match.
    question = pack_builder.PackEvalQuestion(
        query="How do I configure AWS IAM roles and S3 bucket policies?",
        expected_source="AWS IAM Reference",
        expected_chunk_contains="S3 bucket policy",
    )
    result = pack_evaluator.evaluate_question(service, question)
    assert not result.passed
    assert "expected" in result.reason.lower()


# 6. the built pack can be opened and inspected by the ReviewService. -------

def test_pack_opens_in_review_service(tmp_path, monkeypatch):
    registry, pack_id = _build(tmp_path, monkeypatch)
    service = pack_builder.open_pack_service(registry, pack_id)
    review = ReviewService(service, registry)

    state = review.get_dashboard_state()
    assert state["knowledge_sources"] == 8
    table = review.get_knowledge_table()
    assert {row["domain"] for row in table} == {"microsoft", "coding"}
    assert {row["authority"] for row in table} == {"reputable"}


# 7. both M365 and coding domains survive into the knowledge library. -------

def test_both_domains_present_in_knowledge(tmp_path, monkeypatch):
    registry, pack_id = _build(tmp_path, monkeypatch)
    service = pack_builder.open_pack_service(registry, pack_id)

    by_domain: dict[str, set[str]] = {}
    for src in service.list_knowledge_sources():
        by_domain.setdefault(src["domain"], set()).add(src["source_name"])
    assert by_domain.get("microsoft") == _MICROSOFT_SOURCES
    assert by_domain.get("coding") == _CODING_SOURCES


# 8. the combined query path reports memory / knowledge / model-prior flags. -

def test_combined_query_reports_routing_flags(tmp_path, monkeypatch):
    registry, pack_id = _build(tmp_path, monkeypatch)
    service = pack_builder.open_pack_service(registry, pack_id)

    # A query drawn from a source chunk routes to knowledge, not memory, and is
    # not a model-prior fallback.
    chunk = next(c for c in service.knowledge.list_chunks(active_only=True)
                 if c.source_name == "Pydantic v2 Notes")
    hit = service.query_all(chunk.chunk_text)
    assert hit.knowledge_used is True
    assert hit.memory_used is False
    assert hit.model_prior_used is False

    # A decision-recall query routes to project memory only; with an empty
    # ledger nothing grounds it, so the answer falls back to the model prior
    # and neither memory nor knowledge is used.
    miss = service.query_all("What did we decide about the production database?")
    assert miss.knowledge_used is False
    assert miss.memory_used is False
    assert miss.model_prior_used is True


# 9. imported knowledge is not written as project memory. -------------------

def test_imported_knowledge_is_not_project_memory(tmp_path, monkeypatch):
    registry, pack_id = _build(tmp_path, monkeypatch)
    service = pack_builder.open_pack_service(registry, pack_id)
    review = ReviewService(service, registry)

    # Knowledge is populated...
    assert len(review.get_knowledge_table()) == 8
    # ...but importing knowledge creates no memory ledger entries.
    memory_table = review.get_memory_table()
    assert all(len(rows) == 0 for rows in memory_table.values()), memory_table


# 10. an HF coding fixture may be used as eval samples only, never knowledge. -

def test_hf_coding_fixture_is_eval_only(tmp_path, monkeypatch):
    from agent.hf_dataset_importer import (
        HFDatasetMode,
        HFDatasetSpec,
        import_hf_dataset_to_pack,
    )

    registry, pack_id = _build(tmp_path, monkeypatch)
    pack = registry.get_pack(pack_id)
    assert pack is not None

    service = pack_builder.open_pack_service(registry, pack_id)
    before = len(service.list_knowledge_sources())
    assert before == 8

    # Import the offline coding fixture strictly in eval mode. The unknown
    # licence is allowed for eval (with a warning) but would be rejected for
    # knowledge mode, so this can never become trusted knowledge.
    spec = HFDatasetSpec(
        dataset_id="demo/small-coding",
        mode=HFDatasetMode.EVAL,
        domain="coding",
        authority="reputable",
        local_fixture=str(ROOT / "demos" / "hf_fixtures"
                          / "small_coding_dataset.jsonl"),
    )
    report = import_hf_dataset_to_pack(spec, pack)

    # The samples land in an eval file, not the knowledge library.
    assert report.accepted is True
    assert report.mode == HFDatasetMode.EVAL.value
    assert report.imported_count > 0
    assert report.eval_output_path is not None
    assert report.knowledge_source_id is None

    # The knowledge library is untouched: still the 8 curated sources, and no
    # memory ledger pollution.
    after = pack_builder.open_pack_service(registry, pack_id)
    assert len(after.list_knowledge_sources()) == 8
    review = ReviewService(after, registry)
    memory_table = review.get_memory_table()
    assert all(len(rows) == 0 for rows in memory_table.values()), memory_table
