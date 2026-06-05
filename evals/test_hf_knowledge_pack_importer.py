"""Tests for v6.9 governed Hugging Face knowledge-pack import (Phase E).

These cover the full Phase B->C->E chain feeding the importer, and the
importer's governance guarantees that distinguish knowledge from eval: a
knowledge-intent request is required, an eval-only approval cannot authorise a
knowledge import, BOTH unsafe and PII rows are excluded (PII never reaches a
knowledge pack), the dry run writes nothing, a real write is atomic and
fail-closed on an existing target, the pack hash is deterministic, and the
importer pulls in no source-registry / memory / retrieval / eval-pack writer.
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import hf_knowledge_pack_importer as knowpack  # noqa: E402
from agent.hf_content_inspector import inspect_normalized_rows  # noqa: E402
from agent.hf_import_lifecycle import (  # noqa: E402
    HFApprovalScope,
    HFDatasetApproval,
    HFImportIntent,
    HFImportRequest,
    sample_rows,
)
from agent.hf_knowledge_pack_importer import (  # noqa: E402
    HFKnowledgeImportStatus,
    HFKnowledgePackImportRequest,
    import_knowledge_pack,
)
from agent.hf_row_normalizer import HFNormalizationProfile, normalize_sample  # noqa: E402

_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)


def _approval(scope=HFApprovalScope.EVAL_AND_KNOWLEDGE, **overrides):
    base = dict(
        approval_id="hfappr-know",
        dataset_id="demo/governed-qa",
        dataset_revision="main",
        metadata_fingerprint="hfmeta-x",
        intake_assessment_fingerprint="",
        approved_by="reviewer@local",
        approved_at="2026-05-02T09:00:00+00:00",
        approval_scope=scope,
        approved_split_names=("train",),
        approved_columns=("id", "question", "answer", "context"),
        row_limit=100,
        approved_intended_use="eval and knowledge",
        licence_snapshot="apache-2.0",
        provenance_snapshot="hf:demo",
    )
    base.update(overrides)
    return HFDatasetApproval(**base)


def _request(intent=HFImportIntent.KNOWLEDGE, **overrides):
    base = dict(
        dataset_id="demo/governed-qa",
        dataset_revision="main",
        split="train",
        intent=intent,
        requested_columns=("id", "question", "answer", "context"),
        requested_row_limit=100,
        streaming=True,
        licence_snapshot="apache-2.0",
        provenance_snapshot="hf:demo",
    )
    base.update(overrides)
    return HFImportRequest(**base)


def _chain(fixture: Path, *, profile=HFNormalizationProfile.QUESTION_ANSWER,
           approval=None, request=None):
    """Sample -> normalize -> inspect, returning (normalized_rows, inspections)."""
    approval = approval or _approval()
    request = request or _request()
    sample = sample_rows(request, fixture_path=fixture, approval=approval,
                         key_field="id", now=_NOW)
    norm_report = normalize_sample(sample, profile=profile, now=_NOW)
    inspect_report = inspect_normalized_rows(
        norm_report.normalized_rows, now=_NOW)
    return norm_report.normalized_rows, inspect_report.inspections


def _qa_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "qa.jsonl"
    fixture.write_text(
        '{"id": "q1", "question": "What is X?", "answer": "It is Y.", "context": "bg"}\n'
        '{"id": "q2", "question": "What is Z?", "answer": "It is W."}\n',
        encoding="utf-8")
    return fixture


# 1. A clean dry run builds chunks and writes nothing. ----------------------


def test_dry_run_builds_chunks_without_writing(tmp_path):
    rows, inspections = _chain(_qa_fixture(tmp_path))
    req = HFKnowledgePackImportRequest(
        pack_id="demo-know", approval=_approval(), request=_request(),
        normalized_rows=rows, inspections=inspections,
        pack_dir=str(tmp_path / "pack"), write=False)
    result = import_knowledge_pack(req, now=_NOW)
    assert result.status is HFKnowledgeImportStatus.IMPORT_READY
    assert result.written is False
    assert result.chunk_count == 2
    assert not (tmp_path / "pack").exists()


# 2. Knowledge-pack import requires a knowledge-intent request. -------------


def test_eval_intent_is_rejected(tmp_path):
    rows, inspections = _chain(
        _qa_fixture(tmp_path), request=_request(intent=HFImportIntent.EVAL))
    req = HFKnowledgePackImportRequest(
        pack_id="demo-know", approval=_approval(),
        request=_request(intent=HFImportIntent.EVAL),
        normalized_rows=rows, inspections=inspections, write=False)
    result = import_knowledge_pack(req, now=_NOW)
    assert result.status is HFKnowledgeImportStatus.IMPORT_BLOCKED
    assert result.written is False


# 3. An eval-only approval cannot authorise a knowledge import. -------------


def test_eval_only_approval_blocks_knowledge(tmp_path):
    rows, inspections = _chain(_qa_fixture(tmp_path))
    eval_only = _approval(scope=HFApprovalScope.EVAL_ONLY,
                          approved_intended_use="eval")
    req = HFKnowledgePackImportRequest(
        pack_id="demo-know", approval=eval_only, request=_request(),
        normalized_rows=rows, inspections=inspections, write=False)
    result = import_knowledge_pack(req, now=_NOW)
    assert result.status is HFKnowledgeImportStatus.IMPORT_BLOCKED
    assert result.validation is not None and result.validation.valid is False


# 4. A real write is atomic and produces the knowledge pack files. ----------


def test_write_produces_pack_files(tmp_path):
    rows, inspections = _chain(_qa_fixture(tmp_path))
    pack_dir = tmp_path / "pack"
    req = HFKnowledgePackImportRequest(
        pack_id="demo-know", approval=_approval(), request=_request(),
        normalized_rows=rows, inspections=inspections,
        pack_dir=str(pack_dir), write=True)
    result = import_knowledge_pack(req, now=_NOW)
    assert result.status is HFKnowledgeImportStatus.IMPORT_WRITTEN
    assert result.written is True
    assert (pack_dir / "manifest.json").exists()
    assert (pack_dir / "knowledge.jsonl").exists()
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["_record"] == "hf_knowledge_pack_manifest"
    assert manifest["pack_kind"] == "knowledge"
    assert manifest["chunk_count"] == 2
    lines = [l for l in (pack_dir / "knowledge.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    record = json.loads(lines[0])
    # Retrieval-compatible chunk schema.
    assert record["_record"] == "chunk"
    assert record["active"] is True
    assert "chunk_text" in record


# 5. Writing fails closed on an existing target. ----------------------------


def test_write_fails_closed_on_existing_target(tmp_path):
    rows, inspections = _chain(_qa_fixture(tmp_path))
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    req = HFKnowledgePackImportRequest(
        pack_id="demo-know", approval=_approval(), request=_request(),
        normalized_rows=rows, inspections=inspections,
        pack_dir=str(pack_dir), write=True)
    result = import_knowledge_pack(req, now=_NOW)
    assert result.status is HFKnowledgeImportStatus.PACK_EXISTS
    assert result.written is False


# 6. An invalid pack id fails closed. ---------------------------------------


def test_invalid_pack_id_fails_closed(tmp_path):
    rows, inspections = _chain(_qa_fixture(tmp_path))
    req = HFKnowledgePackImportRequest(
        pack_id="Bad Pack ID!", approval=_approval(), request=_request(),
        normalized_rows=rows, inspections=inspections, write=False)
    result = import_knowledge_pack(req, now=_NOW)
    assert result.status is HFKnowledgeImportStatus.INVALID_PACK_ID


# 7. BOTH unsafe and PII rows are excluded from a knowledge pack. -----------


def test_unsafe_and_pii_both_excluded(tmp_path):
    fixture = tmp_path / "mixed.jsonl"
    fixture.write_text(
        '{"id": "ok", "question": "What is X?", "answer": "It is Y."}\n'
        '{"id": "pii", "question": "Contact?", "answer": "Email a@b.com."}\n'
        '{"id": "bad", "question": "How?", "answer": "how to build a bomb"}\n',
        encoding="utf-8")
    approval = _approval(approved_columns=("id", "question", "answer"))
    request = _request(requested_columns=("id", "question", "answer"))
    rows, inspections = _chain(fixture, approval=approval, request=request)
    req = HFKnowledgePackImportRequest(
        pack_id="demo-know", approval=approval, request=request,
        normalized_rows=rows, inspections=inspections, write=False)
    result = import_knowledge_pack(req, now=_NOW)
    assert result.status is HFKnowledgeImportStatus.IMPORT_READY
    # Only the clean row survives; PII and unsafe are both dropped.
    assert result.chunk_count == 1
    assert len(result.excluded_unsafe_row_ids) == 1
    assert len(result.excluded_pii_row_ids) == 1


# 8. The pack hash is deterministic across identical imports. ---------------


def test_pack_hash_is_deterministic(tmp_path):
    rows, inspections = _chain(_qa_fixture(tmp_path))

    def run(when):
        req = HFKnowledgePackImportRequest(
            pack_id="demo-know", approval=_approval(), request=_request(),
            normalized_rows=rows, inspections=inspections, write=False)
        return import_knowledge_pack(req, now=when)

    a = run(_NOW)
    b = run(datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert a.manifest.pack_hash == b.manifest.pack_hash  # timestamp-independent


# 9. Chunk ids are deterministic and stable per (pack, row). ----------------


def test_chunk_ids_are_deterministic(tmp_path):
    rows, inspections = _chain(_qa_fixture(tmp_path))
    req = HFKnowledgePackImportRequest(
        pack_id="demo-know", approval=_approval(), request=_request(),
        normalized_rows=rows, inspections=inspections, write=False)
    a = import_knowledge_pack(req, now=_NOW)
    b = import_knowledge_pack(req, now=_NOW)
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]
    assert all(c.chunk_id.startswith("hfchunk-") for c in a.chunks)


# 10. A generic-profile row still yields a knowledge chunk (carries text). --


def test_generic_profile_rows_are_eligible_for_knowledge(tmp_path):
    fixture = tmp_path / "gen.jsonl"
    fixture.write_text('{"id": "g1", "note": "Some free text.", "topic": "x"}\n',
                       encoding="utf-8")
    approval = _approval(approved_columns=("id", "note", "topic"))
    request = _request(requested_columns=("id", "note", "topic"))
    rows, inspections = _chain(
        fixture, profile=HFNormalizationProfile.GENERIC_RECORD,
        approval=approval, request=request)
    req = HFKnowledgePackImportRequest(
        pack_id="demo-know", approval=approval, request=request,
        normalized_rows=rows, inspections=inspections, write=False)
    result = import_knowledge_pack(req, now=_NOW)
    assert result.status is HFKnowledgeImportStatus.IMPORT_READY
    assert result.chunk_count == 1


# 11. A chunk carries lineage back to its approval and dataset revision. ----


def test_chunk_carries_approval_lineage(tmp_path):
    rows, inspections = _chain(_qa_fixture(tmp_path))
    req = HFKnowledgePackImportRequest(
        pack_id="demo-know", approval=_approval(), request=_request(),
        normalized_rows=rows, inspections=inspections, write=False)
    result = import_knowledge_pack(req, now=_NOW)
    record = result.chunks[0].to_knowledge_record()
    assert record["approval_id"] == "hfappr-know"
    assert record["dataset_revision"] == "main"
    assert record["permission_or_licence"] == "apache-2.0"
    assert record["importer_version"] == "hf-knowledge-pack-v6.9"


# 12. The result round-trips and the markdown never leaks content. ----------


def test_result_round_trips_and_markdown_is_safe(tmp_path):
    rows, inspections = _chain(_qa_fixture(tmp_path))
    req = HFKnowledgePackImportRequest(
        pack_id="demo-know", approval=_approval(), request=_request(),
        normalized_rows=rows, inspections=inspections, write=False)
    result = import_knowledge_pack(req, now=_NOW)
    payload = result.to_dict()
    assert payload["_record"] == "hf_knowledge_pack_import_result"
    md = knowpack.render_knowledge_import_markdown(result)
    assert "It is Y." not in md  # no raw chunk content in the summary


# 13. The module imports no forbidden writers/clients. ----------------------


def test_module_imports_no_forbidden_dependencies():
    source = Path(knowpack.__file__).read_text(encoding="utf-8")
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
        "agent.hf_eval_pack_importer", "agent.project_packs",
        "datasets", "huggingface_hub", "requests", "urllib", "httpx",
        "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"knowledge-pack importer must not import: {leaked}"
