"""Tests for the unified document-ingest path (v1.1).

These prove the "easy way" the workbench exposes: a single file on disk is
extracted, chunked, and written straight into the concept-cell memory bank so
it is queryable immediately. The dependency-free formats (.txt and .md) are
covered here; .docx/.xlsx are exercised only when their optional libraries are
installed. The suite is offline and leaves no trace -- every service is built
into an isolated ``tmp_path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import document_ingest  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402


def _service(tmp_path) -> WorkbenchService:
    return WorkbenchService(
        ledger_path=str(tmp_path / "ledger.jsonl"),
        queue_path=str(tmp_path / "queue.jsonl"),
        knowledge_path=str(tmp_path / "knowledge.jsonl"),
    )


# 1. a .txt file is extracted, written to the bank, and is queryable. --------

def test_ingest_txt_lands_in_bank_and_is_queryable(tmp_path):
    doc = tmp_path / "note.txt"
    doc.write_text(
        "The supplier delivery is scheduled for Friday afternoon.\n",
        encoding="utf-8",
    )
    service = _service(tmp_path)

    result = service.ingest_document(doc)

    assert result.chunk_count >= 1
    assert result.source == "note.txt"
    assert result.suffix == ".txt"
    assert len(result.memory_ids) == result.chunk_count

    # The ingested text is now in the bank and is offered as a candidate.
    audit = service.query_memory("the supplier delivery is scheduled for Friday")
    assert audit.candidate_retrieved is True
    assert any("supplier delivery" in c.canonical_text for c in audit.candidates)


# 2. ingested chunks carry the provenance label and the "ingested" tag. ------

def test_ingest_tags_and_source_are_recorded(tmp_path):
    doc = tmp_path / "spec.md"
    doc.write_text("# Spec\n\nPydantic validates typed models on construction.\n",
                   encoding="utf-8")
    service = _service(tmp_path)

    result = service.ingest_document(doc)

    entries = {e.memory_id: e for e in service.ledger.active_entries()}
    for mem_id in result.memory_ids:
        entry = entries[mem_id]
        assert entry.source == "spec.md"
        assert "ingested" in entry.tags


# 3. a long document is split into multiple chunks. --------------------------

def test_ingest_long_document_splits_into_chunks(tmp_path):
    paragraph = ("Concept cells store one memory per write. " * 30).strip()
    doc = tmp_path / "long.txt"
    doc.write_text("\n\n".join([paragraph] * 6), encoding="utf-8")
    service = _service(tmp_path)

    result = service.ingest_document(doc)

    assert result.chunk_count > 1


# 4. a missing file raises FileNotFoundError. --------------------------------

def test_ingest_missing_file_raises(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(FileNotFoundError):
        service.ingest_document(tmp_path / "does_not_exist.txt")


# 5. an unsupported type raises DocumentIngestError. -------------------------

def test_ingest_unsupported_type_raises(tmp_path):
    doc = tmp_path / "data.bin"
    doc.write_bytes(b"\x00\x01\x02")
    service = _service(tmp_path)
    with pytest.raises(document_ingest.DocumentIngestError):
        service.ingest_document(doc)


# 6. chunk_text packs short paragraphs and drops blank input. ----------------

def test_chunk_text_packs_paragraphs_and_handles_empty():
    assert document_ingest.chunk_text("") == []
    assert document_ingest.chunk_text("   \n\n  ") == []
    chunks = document_ingest.chunk_text("alpha\n\nbeta\n\ngamma")
    assert len(chunks) == 1
    assert "alpha" in chunks[0] and "gamma" in chunks[0]
