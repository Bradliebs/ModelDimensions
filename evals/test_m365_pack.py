"""Tests for the v1.9 Microsoft 365 consultant pack.

These exercise the first *real* curated knowledge pack template end to end: the
spec loads, builds from the local field-note Markdown under
``demos/m365_sources/`` with full provenance, and its evaluation questions run
through the frozen knowledge query path. Nothing here touches the concept-cell
geometry, grounding policy, or lifecycle semantics — building reuses
``KnowledgeLibrary.import_text_file`` and evaluating reads through
``WorkbenchService.query_knowledge``. The pack is always built into an isolated
``tmp_path`` registry so the tracked demo packs stay pristine.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.project_packs import PackRegistry  # noqa: E402
from agent.review_service import ReviewService  # noqa: E402
from agent import pack_builder, pack_evaluator  # noqa: E402

_SPEC = ROOT / "packs" / "starter" / "m365_consultant_pack.yaml"
_EVAL = ROOT / "demos" / "pack_eval_questions" / "m365_consultant_eval.jsonl"


def _build(tmp_path, monkeypatch) -> tuple[PackRegistry, str]:
    """Build the real M365 pack from local docs into an isolated registry.

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
    assert plan.pack_name == "m365-consultant"
    assert plan.enabled is True
    assert plan.allow_url_ingestion is False
    assert len(plan.sources) == 6
    for source in plan.sources:
        assert source.domain == "microsoft"
        assert source.authority == "reputable"
        assert source.version == "local-notes-v1"
        assert source.staleness_policy == "review_required"


# 2. the pack builds from the local docs (no skips). ------------------------

def test_pack_builds_from_local_docs(tmp_path, monkeypatch):
    registry, pack_id = _build(tmp_path, monkeypatch)
    service = pack_builder.open_pack_service(registry, pack_id)
    names = {s["source_name"] for s in service.list_knowledge_sources()}
    assert "Purview Sensitivity Labels" in names
    assert "Defender for Cloud Apps" in names
    assert "SharePoint Governance" in names
    assert len(names) == 6


# 3. the build preserves domain / authority / version metadata. -------------

def test_pack_preserves_source_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_SPEC)
    report = pack_builder.build_pack(plan, registry)

    assert report.source_count == 6
    assert not report.skipped
    src = next(s for s in report.sources
               if s["source_name"] == "Purview Sensitivity Labels")
    assert src["domain"] == "microsoft"
    assert src["authority"] == "reputable"
    assert src["version"] == "local-notes-v1"
    assert src["staleness_policy"] == "review_required"
    assert src["chunks"] >= 1


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
    assert state["knowledge_sources"] == 6
    table = review.get_knowledge_table()
    assert {row["domain"] for row in table} == {"microsoft"}
    assert {row["authority"] for row in table} == {"reputable"}


# 7. imported M365 knowledge is not written as project memory. --------------

def test_imported_knowledge_is_not_project_memory(tmp_path, monkeypatch):
    registry, pack_id = _build(tmp_path, monkeypatch)
    service = pack_builder.open_pack_service(registry, pack_id)
    review = ReviewService(service, registry)

    # Knowledge is populated...
    assert len(review.get_knowledge_table()) == 6
    # ...but importing knowledge creates no memory ledger entries.
    memory_table = review.get_memory_table()
    assert all(len(rows) == 0 for rows in memory_table.values()), memory_table
