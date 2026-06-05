"""Tests for the PDF intake usability fix (local vs external intake modes).

These pin the intake-policy correction: ordinary user-owned project PDFs are
usable by default, while stricter governance is preserved for external trusted
knowledge. The correction adds no new review queue, lifecycle state, or
governance framework — only an intake *mode* and richer extraction reporting.

Twelve scenarios cover the contract:

1.  a clean text PDF becomes *Available for answers* (Ready) by default;
2.  old document metadata does not block local project use;
3.  a missing source URL does not block local project use;
4.  an explicit user permission declaration permits basic project use;
5.  an external trusted source still requires provenance and a licence;
6.  a second (fallback) parser is attempted after a zero-text first pass;
7.  a scanned PDF is reported as ``requires_ocr`` (never silently usable);
8.  poor but non-zero extraction stays previewable (``poor_but_previewable``);
9.  an encrypted PDF remains blocked regardless of mode;
10. raw finding codes are hidden from the normal (plain-language) UI;
11. processing a local document writes no memory;
12. processing a local document activates nothing globally.

PDF fixtures are generated programmatically (no large binaries committed) and
reuse the deterministic writers from the adapter and importer test modules.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import project_workspace as pw  # noqa: E402
from agent.data_intake import DataIntakeDecision  # noqa: E402
from agent.pdf_intake_adapter import (  # noqa: E402
    PdfFindingCode,
    assess_pdf_intake,
    plain_language_findings,
    technical_finding_lines,
)

# Reuse the deterministic adapter fixture writer (supports metadata + flags).
from test_pdf_intake_adapter import (  # noqa: E402
    _PARAGRAPH, _clean_pdf, _make_pdf, _write_pdf,
)
# Reuse the importer fixture writer (proven to import all the way to Ready).
from test_pdf_import_workflow import _clean_pdf_bytes  # noqa: E402

_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)
_WS_NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _workspace(tmp_path: Path) -> pw.ProjectWorkspace:
    return pw.ProjectWorkspace(root=tmp_path / "projects")


# --------------------------------------------------------------------------- #
# 1. Clean text PDF becomes Ready by default (local mode).
# --------------------------------------------------------------------------- #

def test_clean_local_pdf_becomes_ready_by_default(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    doc = ws.add_document(project.project_id, _clean_pdf_bytes(),
                          filename="clean.pdf", now=_WS_NOW)

    assert doc.status == pw.STATUS_READY
    assert doc.available_for_answers is True
    assert doc.pack_id


# --------------------------------------------------------------------------- #
# 2. Old metadata does not block local use.
# --------------------------------------------------------------------------- #

def test_old_metadata_does_not_block_local_use(tmp_path):
    # Same proven structure, but with a creation date well past the staleness
    # window (>3 years before _WS_NOW). The replacement keeps byte length, so no
    # stream offsets shift.
    stale = _clean_pdf_bytes().replace(
        b"D:20250101000000Z", b"D:20180101000000Z")
    assert b"D:20180101000000Z" in stale

    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    doc = ws.add_document(project.project_id, stale,
                          filename="old.pdf", now=_WS_NOW)

    assert doc.status == pw.STATUS_READY
    assert doc.available_for_answers is True


# --------------------------------------------------------------------------- #
# 3. Missing source URL does not block local use.
# --------------------------------------------------------------------------- #

def test_missing_source_url_does_not_block_local_use(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    # No source_url, owner, permission, or authority_level supplied.
    doc = ws.add_document(project.project_id, _clean_pdf_bytes(),
                          filename="bare.pdf", now=_WS_NOW)

    assert doc.status == pw.STATUS_READY
    assert doc.available_for_answers is True


# --------------------------------------------------------------------------- #
# 4. User permission declaration permits basic project use.
# --------------------------------------------------------------------------- #

def test_permission_declaration_permits_project_use(tmp_path):
    path = _write_pdf(tmp_path, "declared.pdf", _clean_pdf())
    result = assess_pdf_intake(
        path, intake_mode="local_project_document",
        owner_permission_declared=True, now=_NOW)

    # A declared permission satisfies the basic project-use permission check.
    assert result.candidate.has_permission is True
    assert not any(f.code == PdfFindingCode.PDF_MISSING_PERMISSION
                   for f in result.findings)


# --------------------------------------------------------------------------- #
# 5. External trusted source still requires provenance and licence.
# --------------------------------------------------------------------------- #

def test_external_source_still_requires_provenance_and_licence(tmp_path):
    path = _write_pdf(tmp_path, "external.pdf", _clean_pdf())
    result = assess_pdf_intake(
        path, intake_mode="external_trusted_source",
        intended_use="knowledge_candidate", now=_NOW)

    assert result.approved_for_knowledge is False
    codes = {f.code for f in result.findings}
    assert PdfFindingCode.PDF_MISSING_PROVENANCE in codes
    assert PdfFindingCode.PDF_MISSING_PERMISSION in codes


# --------------------------------------------------------------------------- #
# 6. A second parser is attempted after a zero-text first pass.
# --------------------------------------------------------------------------- #

def test_fallback_parser_attempted_after_zero_extraction(tmp_path):
    # Break the page tree so the primary (page-walk) parser recovers no text;
    # the flat fallback scan must then recover the content-stream text.
    broken = _clean_pdf().replace(b"/Type /Page /Parent", b"/Type /Pg /Parent")
    path = _write_pdf(tmp_path, "broken_tree.pdf", broken)
    result = assess_pdf_intake(
        path, intake_mode="local_project_document", now=_NOW)

    quality = result.candidate.metadata.parse_quality
    assert quality.fallback_attempted is True
    # The fallback recovered usable text, so this is not unusable/OCR.
    assert quality.extracted_char_count > 0
    assert quality.extraction_outcome in ("good", "acceptable")
    assert result.decision != DataIntakeDecision.BLOCKED


# --------------------------------------------------------------------------- #
# 7. Scanned PDF is reported as requires_ocr (never silently usable).
# --------------------------------------------------------------------------- #

def test_scanned_pdf_reports_requires_ocr(tmp_path):
    data = _make_pdf([""], image_only=True, title="Scan",
                     creation_date="D:20250101000000Z")
    path = _write_pdf(tmp_path, "scan.pdf", data)
    result = assess_pdf_intake(
        path, intake_mode="local_project_document", now=_NOW)

    quality = result.candidate.metadata.parse_quality
    assert result.candidate.metadata.requires_ocr is True
    assert quality.extraction_outcome == "requires_ocr"
    # A 0.0 score is never assigned without recording why.
    assert quality.extraction_quality_score == 0.0
    assert quality.zero_text_reason
    assert result.approved_for_knowledge is False


def test_multi_page_scanned_pdf_is_not_poor_but_previewable(tmp_path):
    # Regression: a scanned PDF with several empty image pages must not be
    # mislabelled "poor_but_previewable" by the newline separators that join the
    # empty pages. The honest character count is zero, so the outcome is OCR.
    data = _make_pdf(["", "", "", ""], image_only=True, title="Big Scan",
                     creation_date="D:20250101000000Z")
    path = _write_pdf(tmp_path, "multiscan.pdf", data)
    result = assess_pdf_intake(
        path, intake_mode="local_project_document", now=_NOW)

    quality = result.candidate.metadata.parse_quality
    assert quality.extracted_char_count == 0
    assert quality.pages_with_text == 0
    assert quality.extraction_outcome == "requires_ocr"
    assert quality.extraction_quality_band == "unusable"
    assert result.candidate.metadata.has_text_layer is False
    assert result.candidate.metadata.requires_ocr is True



# --------------------------------------------------------------------------- #
# 8. Poor but non-zero extraction stays previewable.
# --------------------------------------------------------------------------- #

def test_poor_non_zero_extraction_stays_previewable(tmp_path):
    data = _make_pdf([_PARAGRAPH, "", ""], title="Sparse",
                     creation_date="D:20250101000000Z")
    path = _write_pdf(tmp_path, "sparse.pdf", data)
    result = assess_pdf_intake(
        path, intake_mode="local_project_document", now=_NOW)

    quality = result.candidate.metadata.parse_quality
    assert quality.extracted_char_count > 0
    assert quality.extraction_outcome == "poor_but_previewable"


# --------------------------------------------------------------------------- #
# 9. Encrypted PDF remains blocked regardless of mode.
# --------------------------------------------------------------------------- #

def test_encrypted_pdf_remains_blocked_in_local_mode(tmp_path):
    data = _make_pdf([_PARAGRAPH], encrypted=True, title="Secret",
                     creation_date="D:20250101000000Z")
    path = _write_pdf(tmp_path, "enc.pdf", data)
    result = assess_pdf_intake(
        path, intake_mode="local_project_document", now=_NOW)

    assert result.decision in (
        DataIntakeDecision.BLOCKED, DataIntakeDecision.NEEDS_REVIEW)
    assert result.approved_for_knowledge is False
    assert any(f.code == PdfFindingCode.PDF_ENCRYPTED for f in result.findings)


# --------------------------------------------------------------------------- #
# 10. Raw finding codes are hidden from the normal (plain-language) UI.
# --------------------------------------------------------------------------- #

def test_raw_codes_hidden_from_plain_language_ui(tmp_path):
    path = _write_pdf(tmp_path, "external.pdf", _clean_pdf())
    result = assess_pdf_intake(
        path, intake_mode="external_trusted_source", now=_NOW)

    plain = plain_language_findings(result)
    assert plain  # there are governance findings to show
    # No raw code token (e.g. "pdf_missing_permission") leaks into the plain UI.
    code_tokens = {f.code.value for f in result.findings}
    for line in plain:
        for token in code_tokens:
            assert token not in line

    # The technical projection *does* expose the raw codes for Advanced details.
    technical = technical_finding_lines(result)
    joined = "\n".join(technical)
    assert any(token in joined for token in code_tokens)


# --------------------------------------------------------------------------- #
# 11. Processing a local document writes no memory.
# --------------------------------------------------------------------------- #

def test_local_processing_writes_no_memory(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    ws.add_document(project.project_id, _clean_pdf_bytes(),
                    filename="clean.pdf", now=_WS_NOW)

    pack = ws._registry.get_pack(project.project_id)
    assert pack.memory_ledger_path.read_text(encoding="utf-8").strip() == ""
    assert pack.proposal_queue_path.read_text(encoding="utf-8").strip() == ""
    assert pack.memory_bank_path.read_text(encoding="utf-8").strip() == ""


# --------------------------------------------------------------------------- #
# 12. Processing a local document activates nothing globally.
# --------------------------------------------------------------------------- #

def test_local_processing_activates_nothing(tmp_path):
    ws = _workspace(tmp_path)
    project = ws.create_project("P")
    ws.add_document(project.project_id, _clean_pdf_bytes(),
                    filename="clean.pdf", now=_WS_NOW)

    assert ws._registry.get_active_pack() is None
