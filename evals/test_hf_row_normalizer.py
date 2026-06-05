"""Tests for v6.9 governed Hugging Face row normalization (Phase C).

These cover profile-driven canonical mapping, fail-closed rejection of rows that
lack a required role, explicit column-map override of the profile defaults,
deterministic row ids that change with content/revision, and the import-purity
guarantee that the normalizer pulls in no writer.
"""
from __future__ import annotations

import ast
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import hf_row_normalizer as norm  # noqa: E402
from agent.hf_import_lifecycle import (  # noqa: E402
    ADAPTER_VERSION,
    HFRowAccessProvenance,
    HFSampledRow,
)
from agent.hf_row_normalizer import (  # noqa: E402
    HFNormalizationProfile,
    HFRowValidationCode,
    compute_row_id,
    get_schema_profile,
    normalize_row,
)

_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)


def _sampled(fields, *, index=0, key="", revision="main", split="train",
             dataset_id="demo/governed-qa") -> HFSampledRow:
    prov = HFRowAccessProvenance(
        dataset_id=dataset_id,
        dataset_revision=revision,
        split=split,
        source_row_index=index,
        source_row_key=key,
        retrieved_at=_NOW.isoformat(),
        adapter_version=ADAPTER_VERSION,
        streaming=True,
        row_content_hash="sha256:deadbeef",
        schema_fingerprint="hfschema-aaaaaaaaaaaaaaaa",
    )
    return HFSampledRow(fields=dict(fields), provenance=prov)


# 1. A well-formed QA row normalizes to canonical roles. --------------------


def test_question_answer_row_normalizes():
    row = _sampled({"question": "What is X?", "answer": "It is Y.",
                    "context": "Background."}, key="q1")
    normalized, result = normalize_row(
        row, profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    assert result.valid is True
    assert normalized is not None
    assert normalized.normalized_fields == {
        "question": "What is X?", "answer": "It is Y.", "context": "Background."}
    assert normalized.profile is HFNormalizationProfile.QUESTION_ANSWER
    assert normalized.row_id.startswith("hfrow-")


# 2. A missing required role fails closed (no normalized row). ---------------


def test_missing_required_role_fails_closed():
    row = _sampled({"question": "What is X?"}, key="q1")  # no answer
    normalized, result = normalize_row(
        row, profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    assert normalized is None
    assert result.valid is False
    codes = {f.code for f in result.findings}
    assert HFRowValidationCode.MISSING_REQUIRED_FIELD in codes


# 3. An empty required value fails closed. ----------------------------------


def test_empty_required_value_fails_closed():
    row = _sampled({"question": "What is X?", "answer": "   "}, key="q1")
    normalized, result = normalize_row(
        row, profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    assert normalized is None
    assert result.valid is False
    codes = {f.code for f in result.findings}
    assert HFRowValidationCode.EMPTY_REQUIRED_FIELD in codes


# 4. Default column aliases resolve canonical roles. ------------------------


def test_default_aliases_resolve_roles():
    row = _sampled({"query": "What is X?", "response": "It is Y."}, key="q1")
    normalized, result = normalize_row(
        row, profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    assert result.valid is True
    assert normalized.normalized_fields == {
        "question": "What is X?", "answer": "It is Y."}


# 5. An explicit column map overrides the profile defaults. -----------------


def test_explicit_column_map_overrides_defaults():
    row = _sampled({"prompt_text": "What is X?", "gold": "It is Y.",
                    "question": "decoy"}, key="q1")
    normalized, result = normalize_row(
        row, profile=HFNormalizationProfile.QUESTION_ANSWER,
        column_map={"question": "prompt_text", "answer": "gold"}, now=_NOW)
    assert result.valid is True
    assert normalized.normalized_fields["question"] == "What is X?"
    assert normalized.normalized_fields["answer"] == "It is Y."


# 6. The generic profile keeps all non-empty fields. ------------------------


def test_generic_record_keeps_all_fields():
    row = _sampled({"a": "one", "b": "", "c": 3}, key="g1")
    normalized, result = normalize_row(
        row, profile=HFNormalizationProfile.GENERIC_RECORD, now=_NOW)
    assert result.valid is True
    assert normalized.normalized_fields == {"a": "one", "c": "3"}


# 7. An empty generic record is rejected. -----------------------------------


def test_empty_generic_record_rejected():
    row = _sampled({"a": "", "b": None}, key="g1")
    normalized, result = normalize_row(
        row, profile=HFNormalizationProfile.GENERIC_RECORD, now=_NOW)
    assert normalized is None
    assert result.valid is False
    assert any(f.code is HFRowValidationCode.EMPTY_ROW for f in result.findings)


# 8. The row id is deterministic for identical content. ---------------------


def test_row_id_is_deterministic():
    row = _sampled({"question": "Q?", "answer": "A."}, key="q1")
    a, _ = normalize_row(row, profile=HFNormalizationProfile.QUESTION_ANSWER,
                         now=_NOW)
    b, _ = normalize_row(row, profile=HFNormalizationProfile.QUESTION_ANSWER,
                         now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert a.row_id == b.row_id  # timestamp does not affect the id


# 9. The row id changes with content. ---------------------------------------


def test_row_id_changes_with_content():
    a, _ = normalize_row(
        _sampled({"question": "Q?", "answer": "A."}, key="q1"),
        profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    b, _ = normalize_row(
        _sampled({"question": "Q?", "answer": "DIFFERENT."}, key="q1"),
        profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    assert a.row_id != b.row_id


# 10. The row id changes with revision. -------------------------------------


def test_row_id_changes_with_revision():
    a, _ = normalize_row(
        _sampled({"question": "Q?", "answer": "A."}, key="q1", revision="main"),
        profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    b, _ = normalize_row(
        _sampled({"question": "Q?", "answer": "A."}, key="q1", revision="v2"),
        profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    assert a.row_id != b.row_id


# 11. compute_row_id matches the seed contract. -----------------------------


def test_compute_row_id_contract():
    rid = compute_row_id("demo/ds", "main", "train", "k1",
                         {"answer": "A.", "question": "Q?"})
    # Independent of dict insertion order (canonicalised internally).
    rid2 = compute_row_id("demo/ds", "main", "train", "k1",
                          {"question": "Q?", "answer": "A."})
    assert rid == rid2
    assert rid.startswith("hfrow-")
    assert len(rid) == len("hfrow-") + 16


# 12. Every profile has a coherent schema. ----------------------------------


def test_all_profiles_have_schema():
    for profile in HFNormalizationProfile:
        schema = get_schema_profile(profile)
        assert schema.profile is profile
        assert set(schema.required_roles).issubset(set(schema.roles))


# 13. normalize_sample aggregates accepted and rejected rows. ---------------


def test_normalize_sample_aggregates(tmp_path):
    from agent.hf_import_lifecycle import (
        HFApprovalScope, HFDatasetApproval, HFImportIntent, HFImportRequest,
        sample_rows)

    fixture = tmp_path / "qa.jsonl"
    fixture.write_text(
        '{"id": "q1", "question": "Q1?", "answer": "A1."}\n'
        '{"id": "q2", "question": "Q2 only?"}\n'
        '{"id": "q3", "question": "Q3?", "answer": "A3."}\n',
        encoding="utf-8")
    approval = HFDatasetApproval(
        approval_id="a", dataset_id="demo/ds", dataset_revision="main",
        metadata_fingerprint="hfmeta-x", approved_by="r", approved_at="t",
        approval_scope=HFApprovalScope.EVAL_AND_KNOWLEDGE,
        approved_split_names=("train",),
        approved_columns=("id", "question", "answer"),
        row_limit=10, approved_intended_use="eval and knowledge")
    request = HFImportRequest(
        dataset_id="demo/ds", dataset_revision="main", split="train",
        intent=HFImportIntent.KNOWLEDGE,
        requested_columns=("id", "question", "answer"), requested_row_limit=10)
    sample = sample_rows(request, fixture_path=fixture, approval=approval,
                         key_field="id", now=_NOW)
    report = norm.normalize_sample(
        sample, profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    assert report.normalized_count == 2
    assert report.rejected_count == 1
    assert report.all_normalized is False


# 14. The normalization report round-trips and never leaks via markdown. ----


def test_report_round_trips_and_markdown_is_safe(tmp_path):
    row = _sampled({"question": "secret-question?", "answer": "secret-answer."},
                   key="q1")
    normalized, result = normalize_row(
        row, profile=HFNormalizationProfile.QUESTION_ANSWER, now=_NOW)
    report = norm.HFNormalizationReport(
        profile=HFNormalizationProfile.QUESTION_ANSWER,
        dataset_id="demo/ds", dataset_revision="main", split="train",
        normalized_rows=(normalized,), validation_results=(result,),
        normalized_at=_NOW.isoformat())
    out = norm.write_normalization_report(report, tmp_path / "norm.json")
    assert out.exists()
    md = norm.render_normalization_markdown(report)
    assert "secret-answer" not in md
    assert report.to_dict()["_record"] == "hf_normalization_report"


# 15. The module imports no forbidden writers/clients. ----------------------


def test_module_imports_no_forbidden_dependencies():
    source = Path(norm.__file__).read_text(encoding="utf-8")
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
        "agent.memory_ledger", "agent.source_registry",
        "agent.knowledge_library", "agent.hf_dataset_importer",
        "agent.project_packs", "datasets", "huggingface_hub", "requests",
        "urllib", "httpx", "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"normalizer must not import: {leaked}"
