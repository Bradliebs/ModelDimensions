"""Tests for the v6.4 Governed PDF Intake Adapter (assessment-only).

The adapter inspects a *local* PDF file deterministically and classifies it for
governed intake. These tests pin its contract:

* deterministic file hashing and metadata inspection (no third-party PDF lib);
* a clean text PDF can reach the eval tier; a clean, provenanced, permitted PDF
  can become a knowledge candidate;
* missing provenance, missing permission, possible PII, confidential markers,
  encryption, scanned/image-only content, and poor extraction quality can never
  auto-classify as ``approved_for_knowledge``;
* ``approved_for_eval`` stays distinct from ``approved_for_knowledge``;
* embedded PDF metadata is always flagged unverified;
* the adapter performs no OCR, builds no document fragments, imports no
  durable-state or retrieval-index writer, writes nothing in stdout mode, and
  writes only the assessment report under ``--out``.

PDF fixtures are generated programmatically (no large binaries committed).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import pdf_intake_adapter as pia  # noqa: E402
from agent.data_intake import DataIntakeDecision, IntakeLane  # noqa: E402
from agent.pdf_intake_adapter import (  # noqa: E402
    PdfFindingCode,
    PdfIntakeResult,
    assess_pdf_intake,
    calculate_pdf_hash,
    inspect_pdf_metadata,
)

_README = ROOT / "README.md"
_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)

# A paragraph long enough (~300 chars) to score a "good" extraction band.
_PARAGRAPH = (
    "This document describes the quarterly operating review for the platform "
    "team. It summarises throughput, latency, and reliability across the core "
    "services and lists the agreed follow up actions for the next planning "
    "cycle along with the owners responsible for each workstream this quarter."
)


# --------------------------------------------------------------------------- #
# Minimal, deterministic PDF fixture writer (matches the adapter's reader).
# --------------------------------------------------------------------------- #


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(text: str) -> str:
    ops = ["BT", "/F1 12 Tf", "72 720 Td"]
    for i, line in enumerate(text.split("\n")):
        if i:
            ops.append("0 -14 Td")
        ops.append(f"({_escape(line)}) Tj")
    ops.append("ET")
    return "\n".join(ops)


def _make_pdf(
    pages,
    *,
    title="",
    author="",
    creator="",
    producer="",
    creation_date="",
    mod_date="",
    encrypted=False,
    image_only=False,
    acroform=False,
    embedded_file=False,
    header=b"%PDF-1.4\n",
) -> bytes:
    n_pages = len(pages)
    content_start = 3
    page_start = content_start + n_pages
    info_num = page_start + n_pages
    image_num = info_num + 1
    encrypt_num = image_num + 1
    embedded_num = encrypt_num + 1
    page_nums = [page_start + i for i in range(n_pages)]

    chunks = [header]

    def obj(num: int, body: str) -> None:
        chunks.append(f"{num} 0 obj\n{body}\nendobj\n".encode("latin-1"))

    acro = " /AcroForm << /Fields [] >>" if acroform else ""
    obj(1, f"<< /Type /Catalog /Pages 2 0 R{acro} >>")
    kids = " ".join(f"{p} 0 R" for p in page_nums)
    obj(2, f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>")

    for i, page_text in enumerate(pages):
        cnum = content_start + i
        pnum = page_start + i
        if image_only:
            content = "q 100 0 0 100 72 600 cm /Im0 Do Q"
            res = f" /Resources << /XObject << /Im0 {image_num} 0 R >> >>"
        else:
            content = _content_stream(page_text)
            res = ""
        obj(cnum, f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
        obj(pnum, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                  f"/Contents {cnum} 0 R{res} >>")

    info_parts = []
    if title:
        info_parts.append(f"/Title ({_escape(title)})")
    if author:
        info_parts.append(f"/Author ({_escape(author)})")
    if creator:
        info_parts.append(f"/Creator ({_escape(creator)})")
    if producer:
        info_parts.append(f"/Producer ({_escape(producer)})")
    if creation_date:
        info_parts.append(f"/CreationDate ({creation_date})")
    if mod_date:
        info_parts.append(f"/ModDate ({mod_date})")
    obj(info_num, "<< " + " ".join(info_parts) + " >>")

    if image_only:
        obj(image_num, "<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
                       "/BitsPerComponent 8 /ColorSpace /DeviceGray /Length 1 >>\n"
                       "stream\n\x00\nendstream")
    if embedded_file:
        obj(embedded_num, "<< /Type /EmbeddedFile /Length 3 >>\nstream\nabc\nendstream")

    trailer = f"trailer\n<< /Root 1 0 R /Info {info_num} 0 R"
    if encrypted:
        trailer += f" /Encrypt {encrypt_num} 0 R"
    trailer += " >>\n"
    chunks.append(trailer.encode("latin-1"))
    if encrypted:
        chunks.append(
            f"{encrypt_num} 0 obj\n<< /Filter /Standard /V 2 /R 3 >>\nendobj\n".encode("latin-1"))
    chunks.append(b"%%EOF\n")
    return b"".join(chunks)


def _write_pdf(tmp_path, name, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _clean_pdf(**overrides) -> bytes:
    base = dict(
        pages=[_PARAGRAPH],
        title="Quarterly Operating Review",
        author="Platform Team",
        creator="ModelDimensions",
        producer="ModelDimensions",
        creation_date="D:20250101000000Z",
    )
    base.update(overrides)
    return _make_pdf(**base)


# --------------------------------------------------------------------------- #
# 1. Clean text PDF can reach the eval tier. ---------------------------------
# --------------------------------------------------------------------------- #


def test_clean_text_pdf_can_be_approved_for_eval(tmp_path):
    path = _write_pdf(tmp_path, "clean_eval.pdf", _clean_pdf())
    result = assess_pdf_intake(
        path, source_url="https://example.org/report.pdf",
        permission="cc-by-4.0", intended_use="eval", now=_NOW)
    assert isinstance(result, PdfIntakeResult)
    assert result.decision == DataIntakeDecision.APPROVED_FOR_EVAL
    assert result.lane == IntakeLane.EVAL_ONLY
    assert result.permits_eval_use is True
    assert result.approved_for_knowledge is False
    assert any(f.code == PdfFindingCode.PDF_EVAL_ONLY_INTENDED_USE
               for f in result.findings)


# 2. Clean, provenanced, permitted PDF can become a knowledge candidate. ------

def test_clean_pdf_with_provenance_and_permission_becomes_knowledge(tmp_path):
    path = _write_pdf(tmp_path, "clean_knowledge.pdf", _clean_pdf())
    result = assess_pdf_intake(
        path, source_url="https://example.org/report.pdf",
        owner="Platform Team", permission="cc-by-4.0",
        intended_use="knowledge_candidate", authority_level="official", now=_NOW)
    assert result.decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE
    assert result.lane == IntakeLane.KNOWLEDGE_CANDIDATE
    assert result.approved_for_knowledge is True
    assert result.candidate.metadata.parse_quality.extraction_quality_band == "good"


def test_internal_owned_pdf_can_become_knowledge(tmp_path):
    path = _write_pdf(tmp_path, "internal.pdf", _clean_pdf())
    result = assess_pdf_intake(
        path, owner="Acme Internal", authority_level="internal",
        intended_use="knowledge_candidate", now=_NOW)
    assert result.decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE
    assert any(f.code == PdfFindingCode.PDF_APPROVED_INTERNAL_SOURCE
               for f in result.findings)


# 3. Missing provenance prevents approved_for_knowledge. ---------------------

def test_missing_provenance_prevents_knowledge(tmp_path):
    path = _write_pdf(tmp_path, "no_prov.pdf", _clean_pdf())
    result = assess_pdf_intake(path, permission="cc-by-4.0", now=_NOW)
    assert result.approved_for_knowledge is False
    assert result.decision == DataIntakeDecision.NEEDS_REVIEW
    assert any(f.code == PdfFindingCode.PDF_MISSING_PROVENANCE
               for f in result.findings)


# 4. Missing permission prevents knowledge unless internal-owned. ------------

def test_missing_permission_prevents_knowledge_unless_internal_owned(tmp_path):
    path = _write_pdf(tmp_path, "no_perm.pdf", _clean_pdf())
    external = assess_pdf_intake(
        path, source_url="https://example.org/report.pdf",
        authority_level="external", now=_NOW)
    assert external.approved_for_knowledge is False
    assert any(f.code == PdfFindingCode.PDF_MISSING_PERMISSION
               for f in external.findings)

    internal = assess_pdf_intake(
        path, owner="Acme Internal", authority_level="internal", now=_NOW)
    assert internal.approved_for_knowledge is True
    assert not any(f.code == PdfFindingCode.PDF_MISSING_PERMISSION
                   for f in internal.findings)


# 5. Embedded metadata is marked unverified. ---------------------------------

def test_embedded_metadata_is_marked_unverified(tmp_path):
    path = _write_pdf(tmp_path, "meta.pdf", _clean_pdf())
    result = assess_pdf_intake(
        path, source_url="https://example.org/report.pdf",
        permission="cc-by-4.0", now=_NOW)
    assert result.candidate.metadata.title == "Quarterly Operating Review"
    assert any(f.code == PdfFindingCode.PDF_METADATA_UNVERIFIED
               for f in result.findings)


# 6. Encrypted PDF is blocked or needs_review. -------------------------------

def test_encrypted_pdf_is_blocked_or_review(tmp_path):
    path = _write_pdf(tmp_path, "enc.pdf", _clean_pdf(encrypted=True))
    result = assess_pdf_intake(
        path, source_url="https://example.org/report.pdf",
        permission="cc-by-4.0", now=_NOW)
    assert result.decision in (
        DataIntakeDecision.BLOCKED, DataIntakeDecision.NEEDS_REVIEW)
    assert result.approved_for_knowledge is False
    assert result.candidate.metadata.is_encrypted is True
    assert any(f.code == PdfFindingCode.PDF_ENCRYPTED for f in result.findings)


# 7. No-text-layer (scanned) PDF requires OCR and is not knowledge. ----------

def test_scanned_pdf_requires_ocr_and_is_not_knowledge(tmp_path):
    data = _make_pdf([""], image_only=True, title="Scan",
                     creation_date="D:20250101000000Z")
    path = _write_pdf(tmp_path, "scan.pdf", data)
    result = assess_pdf_intake(
        path, source_url="https://example.org/scan.pdf",
        permission="cc-by-4.0", now=_NOW)
    meta = result.candidate.metadata
    assert meta.has_text_layer is False
    assert meta.requires_ocr is True
    assert result.approved_for_knowledge is False
    codes = {f.code for f in result.findings}
    assert PdfFindingCode.PDF_NO_TEXT_LAYER in codes
    assert PdfFindingCode.PDF_REQUIRES_OCR in codes


# 8. Low extraction quality prevents knowledge. ------------------------------

def test_low_extraction_quality_prevents_knowledge(tmp_path):
    data = _make_pdf([_PARAGRAPH, "", ""], title="Sparse coverage",
                     creation_date="D:20250101000000Z")
    path = _write_pdf(tmp_path, "lowq.pdf", data)
    result = assess_pdf_intake(
        path, source_url="https://example.org/lowq.pdf",
        permission="cc-by-4.0", now=_NOW)
    assert result.candidate.metadata.parse_quality.extraction_quality_band in (
        "poor", "unusable")
    assert result.approved_for_knowledge is False
    assert any(f.code == PdfFindingCode.PDF_LOW_TEXT_EXTRACTION_QUALITY
               for f in result.findings)


# 9. Possible PII causes blocked or needs_review. ----------------------------

def test_possible_pii_causes_review(tmp_path):
    text = _PARAGRAPH + " Contact john.doe@example.com for the SSN 123-45-6789."
    data = _make_pdf([text], title="Contact list",
                     creation_date="D:20250101000000Z")
    path = _write_pdf(tmp_path, "pii.pdf", data)
    result = assess_pdf_intake(
        path, source_url="https://example.org/pii.pdf",
        permission="cc-by-4.0", now=_NOW)
    assert result.decision in (
        DataIntakeDecision.NEEDS_REVIEW, DataIntakeDecision.BLOCKED)
    assert result.approved_for_knowledge is False
    assert any(f.code == PdfFindingCode.PDF_POSSIBLE_PII for f in result.findings)


# 10. Confidential marker causes needs_review. -------------------------------

def test_confidential_marker_causes_review(tmp_path):
    text = _PARAGRAPH + " This document is CONFIDENTIAL and proprietary."
    data = _make_pdf([text], title="Strategy",
                     creation_date="D:20250101000000Z")
    path = _write_pdf(tmp_path, "conf.pdf", data)
    result = assess_pdf_intake(
        path, source_url="https://example.org/conf.pdf",
        permission="cc-by-4.0", now=_NOW)
    assert result.decision == DataIntakeDecision.NEEDS_REVIEW
    assert result.approved_for_knowledge is False
    assert any(f.code == PdfFindingCode.PDF_CONFIDENTIAL_MARKER
               for f in result.findings)


# 11. Stale metadata creates a finding but does not declare content false. ----

def test_stale_metadata_creates_finding_not_false_claim(tmp_path):
    data = _clean_pdf(creation_date="D:20000101000000Z",
                      mod_date="D:20000101000000Z")
    path = _write_pdf(tmp_path, "stale.pdf", data)
    result = assess_pdf_intake(
        path, source_url="https://example.org/old.pdf",
        permission="cc-by-4.0", now=_NOW)
    stale = [f for f in result.findings
             if f.code == PdfFindingCode.PDF_STALE_BY_METADATA]
    assert stale, "expected a staleness finding"
    assert result.approved_for_knowledge is False  # eval tier, not knowledge
    assert result.decision == DataIntakeDecision.APPROVED_FOR_EVAL
    # The message must frame the document as possibly outdated, never as false.
    assert "outdated" in stale[0].message.lower()
    assert "not as false" in stale[0].message.lower()


# 12. Malformed PDF fails closed with a structured assessment. ---------------

def test_malformed_pdf_fails_closed(tmp_path):
    path = _write_pdf(tmp_path, "broken.pdf", b"this is not a pdf at all")
    result = assess_pdf_intake(path, now=_NOW)
    assert result.decision == DataIntakeDecision.BLOCKED
    assert result.candidate.metadata.read_error != ""
    assert any(f.code == PdfFindingCode.PDF_UNREADABLE for f in result.findings)


def test_missing_file_fails_closed(tmp_path):
    result = assess_pdf_intake(tmp_path / "nope.pdf", now=_NOW)
    assert result.decision == DataIntakeDecision.BLOCKED
    assert "not found" in result.candidate.metadata.read_error


def test_unsupported_extension_fails_closed(tmp_path):
    path = _write_pdf(tmp_path, "notes.txt", b"hello")
    result = assess_pdf_intake(path, now=_NOW)
    assert result.decision == DataIntakeDecision.BLOCKED
    assert "unsupported extension" in result.candidate.metadata.read_error


def test_embedded_files_cause_review(tmp_path):
    data = _clean_pdf(embedded_file=True)
    path = _write_pdf(tmp_path, "embed.pdf", data)
    result = assess_pdf_intake(
        path, source_url="https://example.org/embed.pdf",
        permission="cc-by-4.0", now=_NOW)
    assert result.candidate.metadata.embedded_file_count >= 1
    assert result.decision == DataIntakeDecision.NEEDS_REVIEW
    assert any(f.code == PdfFindingCode.PDF_EMBEDDED_FILES_PRESENT
               for f in result.findings)


# 13. approved_for_eval does not imply approved_for_knowledge. ---------------

def test_approved_for_eval_does_not_imply_knowledge(tmp_path):
    path = _write_pdf(tmp_path, "eval.pdf", _clean_pdf())
    result = assess_pdf_intake(
        path, source_url="https://example.org/report.pdf",
        permission="cc-by-4.0", intended_use="eval", now=_NOW)
    assert result.permits_eval_use is True
    assert result.approved_for_eval is True
    assert result.approved_for_knowledge is False


# 14. Assessment is deterministic across repeated runs. ----------------------

def test_assessment_is_deterministic(tmp_path):
    path = _write_pdf(tmp_path, "det.pdf", _clean_pdf())
    a = assess_pdf_intake(path, source_url="https://example.org/report.pdf",
                          permission="cc-by-4.0", now=_NOW)
    b = assess_pdf_intake(path, source_url="https://example.org/report.pdf",
                          permission="cc-by-4.0", now=_NOW)
    assert a.to_dict() == b.to_dict()


# 15. File hash is deterministic. --------------------------------------------

def test_file_hash_is_deterministic(tmp_path):
    import hashlib

    data = _clean_pdf()
    path = _write_pdf(tmp_path, "hash.pdf", data)
    first = calculate_pdf_hash(path)
    second = calculate_pdf_hash(path)
    assert first == second
    assert first == hashlib.sha256(data).hexdigest()
    assert inspect_pdf_metadata(path).file_hash == first


# 16. The adapter imports no writer, downloader, OCR, or retrieval index. -----

def test_adapter_imports_no_writer_or_retrieval():
    import ast

    source = Path(pia.__file__).read_text(encoding="utf-8")
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
        "MemoryLedger", "save_registry", "load_registry",
        "save_memory_review_queue", "save_source_review_queue",
        "propose_source_updates", "build_memory_proposals",
        "agent.memory_ledger", "agent.source_registry",
        # retrieval / chunk-pack writers must never be imported here
        "agent.retrieval", "agent.hybrid_retrieval", "agent.pack_builder",
        "agent.knowledge_packs",
        # OCR / dataset-download clients must never be imported here
        "pytesseract", "pdf2image", "datasets", "huggingface_hub",
        "requests", "urllib", "urllib.request", "httpx",
    }
    leaked = imported & forbidden
    assert not leaked, f"pdf adapter must not import forbidden modules: {leaked}"

    body = source.replace(ast.get_docstring(tree, clean=False) or "", "", 1)
    for token in ("load_dataset", "hf_hub_download", "snapshot_download",
                  "requests.get", "urlopen", "MemoryLedger", "save_registry",
                  "propose_source", "memory_review_queue", "source_review_queue",
                  "pytesseract", "pdf2image", "add_to_index"):
        assert token not in body, f"pdf adapter must not reference {token!r}"


# 17. No network is touched during assessment. -------------------------------

def test_assessment_does_no_network(tmp_path, monkeypatch):
    import socket

    def _boom(*args, **kwargs):  # pragma: no cover - only fires on a violation
        raise AssertionError("pdf adapter attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    path = _write_pdf(tmp_path, "net.pdf", _clean_pdf())
    result = assess_pdf_intake(path, source_url="https://example.org/report.pdf",
                               permission="cc-by-4.0", now=_NOW)
    assert isinstance(result, PdfIntakeResult)


# 18. stdout mode writes nothing; --out writes only the assessment report. ----

def test_cli_stdout_mode_writes_nothing(tmp_path, monkeypatch, capsys):
    pdf = _write_pdf(tmp_path, "cli.pdf", _clean_pdf())
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    before = sorted(p.name for p in work.iterdir())
    capsys.readouterr()
    argv = ["pdf-intake", "assess", "--pdf", str(pdf),
            "--now", "2026-06-04T00:00:00+00:00"]
    assert workbench.main(argv) == 0
    out = capsys.readouterr().out
    assert "# PDF intake assessment (v6.4; assessment-only)" in out
    assert "Decision:" in out
    after = sorted(p.name for p in work.iterdir())
    assert before == after  # stdout mode created no files


def test_cli_out_writes_only_the_report(tmp_path, monkeypatch, capsys):
    pdf = _write_pdf(tmp_path, "cli.pdf", _clean_pdf())
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    out = work / "pdf_report.json"
    capsys.readouterr()
    argv = ["pdf-intake", "assess", "--pdf", str(pdf),
            "--source-url", "https://example.org/report.pdf",
            "--permission", "cc-by-4.0",
            "--now", "2026-06-04T00:00:00+00:00", "--out", str(out)]
    assert workbench.main(argv) == 0
    created = [p for p in work.rglob("*") if p.is_file()]
    assert created == [out]  # the report is the only durable write


def test_cli_report_contents_are_pdf_assessment_record(tmp_path, monkeypatch, capsys):
    import json

    pdf = _write_pdf(tmp_path, "cli.pdf", _clean_pdf())
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    out = work / "pdf_report.json"
    capsys.readouterr()
    argv = ["pdf-intake", "assess", "--pdf", str(pdf),
            "--source-url", "https://example.org/report.pdf",
            "--permission", "cc-by-4.0",
            "--now", "2026-06-04T00:00:00+00:00", "--out", str(out)]
    assert workbench.main(argv) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["_record"] == "pdf_intake_assessment"
    assert payload["decision"] in {d.value for d in DataIntakeDecision}


def test_cli_deterministic_output(tmp_path, monkeypatch, capsys):
    pdf = _write_pdf(tmp_path, "cli.pdf", _clean_pdf())
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    argv = ["pdf-intake", "assess", "--pdf", str(pdf),
            "--source-url", "https://example.org/report.pdf",
            "--permission", "cc-by-4.0",
            "--now", "2026-06-04T00:00:00+00:00"]
    capsys.readouterr()
    assert workbench.main(argv) == 0
    first = capsys.readouterr().out
    assert workbench.main(argv) == 0
    second = capsys.readouterr().out
    assert first == second


# 19. README documents the v6.4 PDF intake adapter. --------------------------

def test_readme_documents_pdf_intake():
    text = _README.read_text(encoding="utf-8").lower()
    assert "## v6.4" in text
    assert "pdf" in text
    assert "intake" in text


# 20. result.to_dict exposes the candidate, external candidate, and findings. -

def test_result_to_dict_shape(tmp_path):
    path = _write_pdf(tmp_path, "shape.pdf", _clean_pdf())
    payload = assess_pdf_intake(
        path, source_url="https://example.org/report.pdf",
        permission="cc-by-4.0", now=_NOW).to_dict()
    assert payload["_record"] == "pdf_intake_assessment"
    assert payload["external_candidate"]["source_type"] == "pdf_document"
    assert payload["candidate"]["metadata"]["file_name"] == "shape.pdf"
    assert isinstance(payload["findings"], list)
    assert "parse_quality" in payload["candidate"]["metadata"]
