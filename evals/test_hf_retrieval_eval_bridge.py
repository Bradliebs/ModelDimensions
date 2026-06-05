"""Tests for the v6.9 HF eval-pack -> retrieval-eval bridge (Phase F).

These verify that an imported eval pack (Phase D) becomes inert retrieval-eval
cases that the *unchanged* v3.0 retrieval-eval harness loads and scores, that
lineage travels with each case, that writing the cases file activates nothing,
and that the bridge imports no service / retrieval activator / writer.
"""
from __future__ import annotations

import ast
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import hf_retrieval_eval_bridge as bridge  # noqa: E402
from agent.hf_content_inspector import inspect_normalized_rows  # noqa: E402
from agent.hf_eval_pack_importer import (  # noqa: E402
    HFEvalPackImportRequest,
    import_eval_pack,
)
from agent.hf_import_lifecycle import (  # noqa: E402
    HFApprovalScope,
    HFDatasetApproval,
    HFImportIntent,
    HFImportRequest,
    sample_rows,
)
from agent.hf_retrieval_eval_bridge import (  # noqa: E402
    eval_pack_to_retrieval_cases,
    load_eval_pack,
    questions_to_retrieval_cases,
    result_to_retrieval_cases,
    write_retrieval_cases,
)
from agent.hf_row_normalizer import HFNormalizationProfile, normalize_sample  # noqa: E402
from agent.retrieval_eval_harness import RetrievalEvalCase, load_cases  # noqa: E402

_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
_DEMO_CASES = ROOT / "demos" / "retrieval_hf_import_cases.jsonl"


def _approval(**overrides):
    base = dict(
        approval_id="hfappr-bridge",
        dataset_id="demo/governed-qa",
        dataset_revision="main",
        metadata_fingerprint="hfmeta-x",
        intake_assessment_fingerprint="",
        approved_by="reviewer@local",
        approved_at="2026-05-02T09:00:00+00:00",
        approval_scope=HFApprovalScope.EVAL_AND_KNOWLEDGE,
        approved_split_names=("train",),
        approved_columns=("id", "question", "answer", "context"),
        row_limit=100,
        approved_intended_use="eval and knowledge",
        licence_snapshot="apache-2.0",
        provenance_snapshot="hf:demo",
    )
    base.update(overrides)
    return HFDatasetApproval(**base)


def _request(**overrides):
    base = dict(
        dataset_id="demo/governed-qa",
        dataset_revision="main",
        split="train",
        intent=HFImportIntent.EVAL,
        requested_columns=("id", "question", "answer", "context"),
        requested_row_limit=100,
        streaming=True,
        licence_snapshot="apache-2.0",
        provenance_snapshot="hf:demo",
    )
    base.update(overrides)
    return HFImportRequest(**base)


def _import_result(tmp_path: Path, *, write=False, pack_dir=None):
    fixture = tmp_path / "qa.jsonl"
    fixture.write_text(
        '{"id": "q1", "question": "What is X?", "answer": "It is Y.", "context": "bg"}\n'
        '{"id": "q2", "question": "What is Z?", "answer": "It is W."}\n',
        encoding="utf-8")
    sample = sample_rows(_request(), fixture_path=fixture, approval=_approval(),
                        key_field="id", now=_NOW)
    norm = normalize_sample(
        sample, profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    inspect = inspect_normalized_rows(norm.normalized_rows, now=_NOW)
    req = HFEvalPackImportRequest(
        pack_id="demo-eval", approval=_approval(), request=_request(),
        normalized_rows=norm.normalized_rows, inspections=inspect.inspections,
        pack_dir=pack_dir, write=write)
    return import_eval_pack(req, now=_NOW)


# 1. In-memory result -> retrieval cases carry lineage. ---------------------


def test_result_to_retrieval_cases_carries_lineage(tmp_path):
    result = _import_result(tmp_path)
    cases = result_to_retrieval_cases(result)
    assert len(cases) == 2
    for case, question in zip(cases, result.questions):
        assert case["case_id"] == question.question_id
        assert case["expected_sources"] == ["demo/governed-qa"]
        assert case["minimum_hit_k"] == 0  # gap probe; no forced hit
        assert "hf_import" in case["tags"]


# 2. The emitted cases load into the unchanged retrieval harness. -----------


def test_cases_load_into_unchanged_harness(tmp_path):
    result = _import_result(tmp_path)
    cases = result_to_retrieval_cases(result)
    loaded = [RetrievalEvalCase.from_dict(c) for c in cases]
    assert all(isinstance(c, RetrievalEvalCase) for c in loaded)
    assert loaded[0].query == "What is X?"
    # Gap probe: no expected target asserted -> hit constraint is vacuous.
    assert loaded[0].has_expected is True  # dataset id is an expected source
    assert loaded[0].minimum_hit_k == 0


# 3. A written pack round-trips through the bridge reader. ------------------


def test_written_pack_round_trips(tmp_path):
    pack_dir = tmp_path / "pack"
    result = _import_result(tmp_path, write=True, pack_dir=str(pack_dir))
    assert result.written is True
    manifest, questions = load_eval_pack(pack_dir)
    assert manifest["pack_kind"] == "eval"
    assert len(questions) == 2
    cases = eval_pack_to_retrieval_cases(pack_dir)
    assert [c["case_id"] for c in cases] == \
        [q.question_id for q in result.questions]


# 4. Reading a non-eval directory fails closed. -----------------------------


def test_non_eval_pack_is_rejected(tmp_path):
    pack_dir = tmp_path / "knowledge-pack"
    pack_dir.mkdir()
    (pack_dir / "manifest.json").write_text(
        '{"pack_kind": "knowledge"}', encoding="utf-8")
    (pack_dir / "eval_questions.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_pack(pack_dir)


# 5. A missing pack file fails closed. --------------------------------------


def test_missing_pack_files_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_pack(tmp_path / "does-not-exist")


# 6. Writing the cases file is atomic and self-documenting. -----------------


def test_write_retrieval_cases_is_inert_and_atomic(tmp_path):
    result = _import_result(tmp_path)
    cases = result_to_retrieval_cases(result)
    out = tmp_path / "cases.jsonl"
    written = write_retrieval_cases(cases, out, now=_NOW)
    assert Path(written) == out
    text = out.read_text(encoding="utf-8")
    assert "activates nothing" in text  # inert-by-construction banner
    # The harness can load exactly the cases we wrote (comments skipped).
    loaded = load_cases(out)
    assert len(loaded) == len(cases)


# 7. The cases file never leaks expected answers. ---------------------------


def test_written_cases_never_leak_answers(tmp_path):
    result = _import_result(tmp_path)
    cases = result_to_retrieval_cases(result)
    out = tmp_path / "cases.jsonl"
    write_retrieval_cases(cases, out, now=_NOW)
    text = out.read_text(encoding="utf-8")
    assert "It is Y." not in text
    assert "It is W." not in text


# 8. The shipped demo cases file loads through the harness. -----------------


def test_demo_cases_file_loads(tmp_path):
    loaded = load_cases(_DEMO_CASES)
    assert loaded, "demo retrieval HF-import cases should be non-empty"
    assert all("hf_import" in c.tags for c in loaded)
    assert all(c.minimum_hit_k == 0 for c in loaded)


# 9. The transform is deterministic. ----------------------------------------


def test_transform_is_deterministic(tmp_path):
    result = _import_result(tmp_path)
    a = questions_to_retrieval_cases(result.questions)
    b = questions_to_retrieval_cases(result.questions)
    assert a == b


# 10. The bridge imports no service / activator / writer. -------------------


def test_module_imports_no_forbidden_dependencies():
    source = Path(bridge.__file__).read_text(encoding="utf-8")
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
        "save_registry", "MemoryLedger", "KnowledgeLibrary",
        "WorkbenchService", "agent.workbench_service",
        "agent.memory_ledger", "agent.source_registry",
        "agent.knowledge_library", "agent.hf_dataset_importer",
        "agent.hf_knowledge_pack_importer", "agent.project_packs",
        "datasets", "huggingface_hub", "requests", "urllib", "httpx",
        "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"retrieval-eval bridge must not import: {leaked}"
