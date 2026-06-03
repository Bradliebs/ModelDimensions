"""Tests for v1.3 source-governed knowledge import.

These cover the import contract: importing a document creates a KnowledgeSource
and chunks; imported chunks are never written to the memory ledger; an imported
chunk can be retrieved by a knowledge query; ``query_all`` distinguishes
``memory_used`` from ``knowledge_used``; deleting a source prevents its chunks
from being cited; a coding source includes its version (or warns when unknown);
a medical source returns an informational-only flag; a memory query never cites
knowledge as a project decision; and a knowledge query never marks
``memory_used`` true.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.workbench_service import WorkbenchService  # noqa: E402

_DELIVERY = "the supplier delivery is on Friday afternoon"


def _service(tmp_path) -> WorkbenchService:
    return WorkbenchService(
        ledger_path=str(tmp_path / "ledger.jsonl"),
        queue_path=str(tmp_path / "queue.jsonl"),
        knowledge_path=str(tmp_path / "knowledge.jsonl"),
    )


def _write_doc(tmp_path, text: str, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


_CODING_DOC = (
    "# Pydantic\n\n"
    "Pydantic validates data against a typed model when the model is "
    "constructed and raises a ValidationError on bad input.\n\n"
    "# JSONL\n\n"
    "JSONL stores one JSON object per line for append-only ledgers.\n"
)

_MEDICAL_DOC = (
    "# Dosage\n\n"
    "Paracetamol is commonly taken for pain relief; typical adult dosage "
    "guidance is described in the leaflet.\n"
)


# 1. importing knowledge creates a KnowledgeSource and chunks. --------------

def test_import_creates_source_and_chunks(tmp_path):
    service = _service(tmp_path)
    doc = _write_doc(tmp_path, _CODING_DOC)

    source = service.import_knowledge(
        doc, domain="coding", authority="official",
        source_name="Pydantic Notes", version="2.x")

    sources = service.list_knowledge_sources()
    assert len(sources) == 1
    assert sources[0]["source_id"] == source.source_id
    chunks = service.knowledge.list_chunks(source_id=source.source_id)
    assert len(chunks) >= 2


# 2. knowledge chunks are not written to the MemoryLedger. ------------------

def test_import_does_not_touch_memory_ledger(tmp_path):
    service = _service(tmp_path)
    doc = _write_doc(tmp_path, _CODING_DOC)

    service.import_knowledge(
        doc, domain="coding", authority="official",
        source_name="Pydantic Notes", version="2.x")

    assert service.export_ledger() == []
    assert service.bank.records() == []


# 3. query_knowledge can retrieve an imported chunk. ------------------------

def test_query_knowledge_retrieves_chunk(tmp_path):
    service = _service(tmp_path)
    doc = _write_doc(tmp_path, _CODING_DOC)
    service.import_knowledge(
        doc, domain="coding", authority="official",
        source_name="Pydantic Notes", version="2.x")

    audit = service.query_knowledge("how do I code Pydantic validation")

    assert audit.knowledge_used is True
    assert audit.candidates
    assert any("Pydantic" in c["text"] for c in audit.candidates)


# 4. query_all distinguishes memory_used and knowledge_used. ----------------

def test_query_all_distinguishes_memory_and_knowledge(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_DELIVERY, source="seed")
    doc = _write_doc(tmp_path, _CODING_DOC)
    service.import_knowledge(
        doc, domain="coding", authority="official",
        source_name="Pydantic Notes", version="2.x")

    mem = service.query_all(
        "what did we decide about " + _DELIVERY)
    assert mem.memory_used is True
    assert mem.knowledge_used is False

    know = service.query_all("how do I code Pydantic validation")
    assert know.knowledge_used is True
    assert know.memory_used is False


# 5. deleting a knowledge source prevents citation. -------------------------

def test_delete_source_prevents_citation(tmp_path):
    service = _service(tmp_path)
    doc = _write_doc(tmp_path, _CODING_DOC)
    source = service.import_knowledge(
        doc, domain="coding", authority="official",
        source_name="Pydantic Notes", version="2.x")

    before = service.query_knowledge("how do I code Pydantic validation")
    assert before.knowledge_used is True

    assert service.delete_knowledge_source(source.source_id) is True

    after = service.query_knowledge("how do I code Pydantic validation")
    assert after.knowledge_used is False
    assert after.candidates == []
    assert service.list_knowledge_sources() == []


# 6. coding domain includes version / unknown-version warning. --------------

def test_coding_version_and_unknown_warning(tmp_path):
    service = _service(tmp_path)
    versioned = _write_doc(tmp_path, _CODING_DOC, name="versioned.md")
    service.import_knowledge(
        versioned, domain="coding", authority="official",
        source_name="Pydantic Notes", version="2.x")

    audit = service.query_knowledge("how do I code Pydantic validation")
    assert any("2.x" in c for c in audit.cautions)

    service2 = WorkbenchService(
        ledger_path=str(tmp_path / "l2.jsonl"),
        queue_path=str(tmp_path / "q2.jsonl"),
        knowledge_path=str(tmp_path / "k2.jsonl"),
    )
    unversioned = _write_doc(tmp_path, _CODING_DOC, name="unversioned.md")
    service2.import_knowledge(
        unversioned, domain="coding", authority="community",
        source_name="Forum Post")

    audit2 = service2.query_knowledge("how do I code Pydantic validation")
    assert any("no recorded" in c for c in audit2.cautions)


# 7. medical domain returns an informational-only flag. ---------------------

def test_medical_domain_is_informational_only(tmp_path):
    service = _service(tmp_path)
    doc = _write_doc(tmp_path, _MEDICAL_DOC, name="med.md")
    service.import_knowledge(
        doc, domain="medical", authority="reputable",
        source_name="Health Leaflet")

    audit = service.query_knowledge("what is the dosage for pain relief")

    assert audit.domain == "medical"
    assert audit.informational_only is True
    assert any("not medical advice" in c.lower()
               or "informational only" in c.lower() for c in audit.cautions)


# 8. a memory query does not cite knowledge as a project decision. ----------

def test_memory_query_does_not_cite_knowledge(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_DELIVERY, source="seed")
    doc = _write_doc(tmp_path, _CODING_DOC)
    service.import_knowledge(
        doc, domain="coding", authority="official",
        source_name="Pydantic Notes", version="2.x")

    audit = service.query_memory("what did we decide about " + _DELIVERY)

    # Memory grounded, and only memory ids are ever cited.
    assert audit.memory_used is True
    assert audit.cited_memory_ids
    for cid in audit.cited_memory_ids:
        assert cid.startswith("mem-")


# 9. a knowledge query does not mark memory_used true. ----------------------

def test_knowledge_query_does_not_set_memory_used(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_DELIVERY, source="seed")
    doc = _write_doc(tmp_path, _CODING_DOC)
    service.import_knowledge(
        doc, domain="coding", authority="official",
        source_name="Pydantic Notes", version="2.x")

    audit = service.query_knowledge("how do I code Pydantic validation")

    assert not hasattr(audit, "memory_used") or getattr(audit, "memory_used", False) is False
    assert audit.knowledge_used is True
