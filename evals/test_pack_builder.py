"""Tests for the v1.8 curated knowledge pack builder and evaluator.

These tests exercise the additive pack-building layer end to end: importing
local sources into an isolated pack with full provenance, then validating
retrieval/source behaviour with pack-specific evaluation questions. Nothing here
touches the frozen geometry, grounding policy, or lifecycle semantics — the
builder reuses ``KnowledgeLibrary.import_text_file`` and the evaluator reads
through ``WorkbenchService.query_knowledge``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.project_packs import PackRegistry  # noqa: E402
from agent import pack_builder, pack_evaluator  # noqa: E402

_CODING_DOC = (
    "# Coding reference\n\n"
    "## Pydantic model validation\n\n"
    "Pydantic validates data against a typed model and raises a "
    "ValidationError when the input does not satisfy the declared type.\n\n"
    "## JSONL persistence\n\n"
    "JSONL stores one JSON object per line, which makes append-only logs "
    "easy to stream line by line.\n"
)
_MEDICAL_DOC = (
    "# Hydration reference\n\n"
    "Staying hydrated supports normal bodily function. This text is "
    "informational only and is not a substitute for professional care.\n"
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _coding_plan(doc: Path) -> pack_builder.PackBuildPlan:
    return pack_builder.PackBuildPlan(
        pack_name="coding-reference",
        description="local coding reference",
        default_domain="coding",
        sources=[pack_builder.PackSourceSpec(
            source_name="Python Coding Reference",
            path_or_url=str(doc),
            domain="coding",
            authority="official",
            version="3.12",
            licence="CC-BY-4.0",
            staleness_policy="review-quarterly",
        )],
    )


# 1. a local source spec imports into the correct pack. ---------------------

def test_local_source_imports_into_correct_pack(tmp_path):
    doc = _write(tmp_path, "coding.md", _CODING_DOC)
    registry = PackRegistry(tmp_path / "packs")
    report = pack_builder.build_pack(_coding_plan(doc), registry)

    pack = registry.get_pack("coding-reference")
    assert pack is not None
    assert report.pack_id == pack.pack_id
    service = pack_builder.open_pack_service(registry, pack.pack_id)
    names = {s["source_name"] for s in service.list_knowledge_sources()}
    assert "Python Coding Reference" in names


# 2. pack build preserves domain / authority / version metadata. ------------

def test_pack_build_preserves_source_metadata(tmp_path):
    doc = _write(tmp_path, "coding.md", _CODING_DOC)
    registry = PackRegistry(tmp_path / "packs")
    report = pack_builder.build_pack(_coding_plan(doc), registry)

    assert report.source_count == 1
    src = report.sources[0]
    assert src["domain"] == "coding"
    assert src["authority"] == "official"
    assert src["version"] == "3.12"
    assert src["licence"] == "CC-BY-4.0"
    assert src["staleness_policy"] == "review-quarterly"
    assert src["chunks"] >= 1


# 3. pack eval passes when the expected source / chunk is retrieved. ---------

def test_pack_eval_passes_for_expected_source(tmp_path):
    doc = _write(tmp_path, "coding.md", _CODING_DOC)
    registry = PackRegistry(tmp_path / "packs")
    pack_builder.build_pack(_coding_plan(doc), registry)
    service = pack_builder.open_pack_service(registry, "coding-reference")

    question = pack_builder.PackEvalQuestion(
        query="How does Pydantic validate data?",
        expected_domain="coding",
        expected_source="Python Coding Reference",
        expected_authority="official",
        expected_chunk_contains="ValidationError",
        required_flags=["knowledge_used"],
    )
    result = pack_evaluator.evaluate_question(service, question)
    assert result.passed, result.reason


# 4. pack eval fails when the wrong source is retrieved. --------------------

def test_pack_eval_fails_for_wrong_source(tmp_path):
    doc = _write(tmp_path, "coding.md", _CODING_DOC)
    registry = PackRegistry(tmp_path / "packs")
    pack_builder.build_pack(_coding_plan(doc), registry)
    service = pack_builder.open_pack_service(registry, "coding-reference")

    question = pack_builder.PackEvalQuestion(
        query="How does Pydantic validate data?",
        expected_source="Some Other Source",
    )
    result = pack_evaluator.evaluate_question(service, question)
    assert not result.passed
    assert "expected" in result.reason.lower()


# 5. medical eval requires the informational_only flag. ---------------------

def test_medical_eval_requires_informational_only_flag(tmp_path):
    doc = _write(tmp_path, "medical.md", _MEDICAL_DOC)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan(
        pack_name="medical-reference",
        default_domain="medical",
        sources=[pack_builder.PackSourceSpec(
            source_name="Medical Reference (example)",
            path_or_url=str(doc),
            domain="medical",
            authority="reputable",
        )],
    )
    pack_builder.build_pack(plan, registry)
    service = pack_builder.open_pack_service(registry, "medical-reference")

    question = pack_builder.PackEvalQuestion(
        query="What guidance is given about hydration?",
        expected_domain="medical",
        required_flags=["informational_only"],
    )
    result = pack_evaluator.evaluate_question(service, question)
    assert result.passed, result.reason
    flag_check = next(c for c in result.checks
                      if c["name"] == "flag:informational_only")
    assert flag_check["ok"]


# 6. disabled URL ingestion does not run silently. --------------------------

def test_disabled_url_ingestion_is_recorded_not_run(tmp_path):
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan(
        pack_name="url-pack",
        allow_url_ingestion=False,
        sources=[pack_builder.PackSourceSpec(
            source_name="Remote Doc",
            path_or_url="https://example.com/reference.md",
            domain="general",
            authority="unknown",
        )],
    )
    report = pack_builder.build_pack(plan, registry)

    assert report.source_count == 0
    assert len(report.skipped) == 1
    assert report.skipped[0]["source_name"] == "Remote Doc"
    assert "url ingestion is disabled" in report.skipped[0]["reason"].lower()


# 7. pack report includes source counts and eval results. -------------------

def test_pack_report_includes_source_counts_and_eval(tmp_path):
    doc = _write(tmp_path, "coding.md", _CODING_DOC)
    registry = PackRegistry(tmp_path / "packs")
    pack_builder.build_pack(_coding_plan(doc), registry)
    built = registry.get_pack("coding-reference")
    service = pack_builder.open_pack_service(registry, built.pack_id)

    question = pack_builder.PackEvalQuestion(
        query="How does JSONL persistence store records?",
        expected_source="Python Coding Reference",
        expected_chunk_contains="one JSON object per line",
        required_flags=["knowledge_used"],
    )
    results = pack_evaluator.evaluate_pack(service, [question])
    report = pack_builder.report_for_pack(
        service, pack_id=built.pack_id, pack_name=built.name,
        eval_results=results)

    payload = report.to_dict()
    assert payload["source_count"] == 1
    assert payload["total_chunks"] >= 1
    assert payload["eval"]["total"] == 1
    assert payload["eval"]["passed"] == 1
    assert payload["eval"]["results"][0]["passed"] is True
