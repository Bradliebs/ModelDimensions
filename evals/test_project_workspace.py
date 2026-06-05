"""Tests for the usable project workspace (v7.0 data-to-answer layer).

These pin the user-facing journey and the governance guarantees of the thin
project layer:

* create a project, add a clean PDF, and it becomes *Available for answers*;
* processing delegates to the governed importer and writes **only** the pack's
  own files — no memory, no activation, no source-registry mutation;
* documents the governance pipeline does not approve for knowledge (missing
  provenance/permission, confidential markers, non-PDF uploads) are never
  auto-imported and never become answerable;
* Ask is scoped to a single project's ready documents and writes nothing;
* a cited Markdown report can be rendered and exported to a file.

The deterministic PDF fixture writer is reused from the importer workflow tests.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import project_workspace as pw  # noqa: E402

# Reuse the deterministic PDF fixture writer from the importer workflow tests.
from test_pdf_import_workflow import (  # noqa: E402
    _PARAGRAPHS, _clean_pdf_bytes, _make_pdf, _page,
)

_NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)

# Metadata that yields an "approved for knowledge" intake decision (verified
# against the governed pipeline).
_META = dict(
    source_url="https://example.org/report.pdf",
    owner="Platform Team",
    permission="cc-by-4.0",
    authority_level="official",
)

# A query with strong lexical overlap with one of the fixture paragraphs so the
# offline retrieval backend matches it.
_MATCHING_QUERY = (
    "reliability budgets reviewed every month regression threshold investigation"
)


def _confidential_pdf_bytes() -> bytes:
    pages = [
        _page("CONFIDENTIAL - DO NOT DISTRIBUTE", _PARAGRAPHS[0], _PARAGRAPHS[1]),
        _page(_PARAGRAPHS[2], _PARAGRAPHS[3], _PARAGRAPHS[4]),
    ]
    return _make_pdf(pages)


def _workspace(tmp_path: Path) -> pw.ProjectWorkspace:
    return pw.ProjectWorkspace(root=tmp_path / "projects")


def _add_clean(ws: pw.ProjectWorkspace, project_id: str, *, filename="clean.pdf"):
    return ws.add_document(project_id, _clean_pdf_bytes(), filename=filename,
                           now=_NOW, **_META)


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #

def test_create_and_list_projects(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("Acme Diligence", description="Q3 review")
    assert project.name == "Acme Diligence"
    assert project.document_count == 0
    assert project.ready_count == 0

    listed = ws.list_projects()
    assert [p.project_id for p in listed] == [project.project_id]
    assert ws.get_project(project.project_id).name == "Acme Diligence"


def test_get_unknown_project_is_none_and_add_raises(tmp_path):
    ws = _workspace(tmp_path)
    assert ws.get_project("nope") is None
    with pytest.raises(KeyError):
        ws.add_document("nope", _clean_pdf_bytes(), filename="x.pdf")


# --------------------------------------------------------------------------- #
# Adding a clean document — the routine path
# --------------------------------------------------------------------------- #

def test_clean_document_becomes_available_for_answers(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    doc = _add_clean(ws, project.project_id)

    assert doc.status == pw.STATUS_READY
    assert doc.available_for_answers is True
    assert doc.status_label == "Available for answers"
    assert doc.chunk_count > 0
    assert doc.pack_id

    docs = ws.list_documents(project.project_id)
    assert len(docs) == 1 and docs[0].available_for_answers
    assert ws.get_project(project.project_id).ready_count == 1


def test_processing_writes_only_governed_pack_files(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    doc = _add_clean(ws, project.project_id)

    pack = ws._registry.get_pack(project.project_id)
    doc_pack_dir = pack.root_path / "documents" / doc.pack_id
    written = {p.name for p in doc_pack_dir.iterdir()}
    assert written == {"manifest.json", "knowledge.jsonl"}
    # The governed importer really imported chunk records.
    body = (doc_pack_dir / "knowledge.jsonl").read_text(encoding="utf-8")
    assert '"_record": "chunk"' in body


def test_processing_does_not_write_memory_or_activate(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    _add_clean(ws, project.project_id)

    pack = ws._registry.get_pack(project.project_id)
    # The project's own memory/proposal/bank stores stay empty: no memory write.
    assert pack.memory_ledger_path.read_text(encoding="utf-8").strip() == ""
    assert pack.proposal_queue_path.read_text(encoding="utf-8").strip() == ""
    assert pack.memory_bank_path.read_text(encoding="utf-8").strip() == ""
    # No pack was activated.
    assert ws._registry.get_active_pack() is None


# --------------------------------------------------------------------------- #
# Exception paths — never auto-imported, never answerable
# --------------------------------------------------------------------------- #

def test_document_without_provenance_needs_review(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    doc = ws.add_document(project.project_id, _clean_pdf_bytes(),
                          filename="bare.pdf", now=_NOW,
                          intake_mode="external_trusted_source")

    assert doc.status == pw.STATUS_NEEDS_REVIEW
    assert doc.available_for_answers is False
    assert doc.pack_id == ""
    pack = ws._registry.get_pack(project.project_id)
    assert not (pack.root_path / "documents" / "bare").exists()


def test_non_pdf_upload_is_blocked(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    doc = ws.add_document(project.project_id, b"this is not a pdf",
                          filename="note.pdf", now=_NOW, **_META)

    assert doc.status == pw.STATUS_BLOCKED
    assert doc.available_for_answers is False
    assert doc.pack_id == ""


def test_confidential_document_is_not_auto_processed(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    doc = ws.add_document(project.project_id, _confidential_pdf_bytes(),
                          filename="secret.pdf", now=_NOW, **_META)

    assert doc.available_for_answers is False
    assert doc.status in (pw.STATUS_NEEDS_REVIEW, pw.STATUS_BLOCKED)
    assert doc.pack_id == ""


# --------------------------------------------------------------------------- #
# Ask — scoped to one project's ready documents
# --------------------------------------------------------------------------- #

def test_ask_uses_only_ready_documents(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    _add_clean(ws, project.project_id)

    answer = ws.ask(project.project_id, _MATCHING_QUERY)
    assert answer.inspector.available is True
    assert answer.inspector.retrieved_count >= 1
    assert answer.view.status_label  # renders a status


def test_ask_ignores_documents_that_are_not_ready(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    # Only a needs-review document exists -> nothing is answerable.
    ws.add_document(project.project_id, _clean_pdf_bytes(), filename="bare.pdf",
                    now=_NOW, intake_mode="external_trusted_source")

    answer = ws.ask(project.project_id, _MATCHING_QUERY)
    assert answer.inspector.retrieved_count == 0
    assert answer.citations == ()


def test_ask_is_scoped_to_a_single_project(tmp_path):
    ws = _workspace(tmp_path)
    a = ws.create_project("A")
    b = ws.create_project("B")
    _add_clean(ws, a.project_id)

    answer_a = ws.ask(a.project_id, _MATCHING_QUERY)
    answer_b = ws.ask(b.project_id, _MATCHING_QUERY)
    assert answer_a.inspector.retrieved_count >= 1
    assert answer_b.inspector.retrieved_count == 0


def test_ask_with_no_evidence_renders(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("Empty")
    answer = ws.ask(project.project_id, _MATCHING_QUERY)
    assert answer.view.status_label
    assert answer.citations == ()


def test_ask_writes_nothing_to_the_project(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    _add_clean(ws, project.project_id)
    pack = ws._registry.get_pack(project.project_id)
    docs_before = pack.root_path.joinpath("documents.jsonl").read_text(
        encoding="utf-8")

    ws.ask(project.project_id, _MATCHING_QUERY)

    assert pack.memory_ledger_path.read_text(encoding="utf-8").strip() == ""
    assert pack.proposal_queue_path.read_text(encoding="utf-8").strip() == ""
    assert ws._registry.get_active_pack() is None
    # The document index is unchanged: Ask records nothing.
    assert pack.root_path.joinpath("documents.jsonl").read_text(
        encoding="utf-8") == docs_before


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

def test_report_includes_question_and_renders(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("Acme")
    _add_clean(ws, project.project_id)
    answer = ws.ask(project.project_id, _MATCHING_QUERY)

    report = ws.render_report(project.project_id, answer,
                              generated_at=_NOW.isoformat())
    assert "Acme" in report
    assert _MATCHING_QUERY in report
    assert len(report) > 0


def test_export_report_writes_a_file(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("Acme")
    _add_clean(ws, project.project_id)
    answer = ws.ask(project.project_id, _MATCHING_QUERY)

    out = ws.export_report(project.project_id, answer,
                           out_dir=tmp_path / "exports",
                           generated_at=_NOW.isoformat())
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ws.render_report(
        project.project_id, answer, generated_at=_NOW.isoformat())
