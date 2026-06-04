"""Tests for the v6.5 PDF Parse Quality + Chunk Preview layer (preview-only).

This layer shows exactly what extracted PDF text and proposed chunks would look
like BEFORE any knowledge-pack creation or retrieval indexing. These tests pin
its contract:

* per-page extraction preserves page order and page numbers, and never silently
  drops empty pages;
* chunk identifiers are deterministic across repeated previews;
* repeated headers/footers, duplicate and conservative near-duplicate chunks,
  sparse/poor pages, fragmented and table-risk chunks, possible PII, and
  confidentiality markers are all flagged;
* page-crossing chunks are labelled only when page merging is requested;
* ``preview_ready_for_import`` is advisory only and stays False for no-text-layer,
  OCR-required, blocked-intake, sensitive, or poor-quality documents;
* the module imports no OCR, LLM, retrieval, registry, memory, pack, or proposal
  writer, touches no network, writes nothing in stdout mode, and writes only the
  preview report under ``--out``.

PDF fixtures are generated programmatically (no large binaries committed). Each
page's text is emitted as a single literal string so that paragraph and line
structure survives the deterministic, stdlib-only reader.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import pdf_chunk_preview as pcp  # noqa: E402
from agent.data_intake import DataIntakeDecision  # noqa: E402
from agent.pdf_chunk_preview import (  # noqa: E402
    PdfChunkPreview,
    PdfChunkWarningCode,
    PdfPageQualityBand,
    PdfPreviewStatus,
    analyse_page_quality,
    extract_pdf_pages,
    preview_pdf_chunks,
)

_README = ROOT / "README.md"
_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)

# Sentence-terminated paragraphs long enough to score a "good" extraction band.
_P1 = (
    "The platform reliability program tracks throughput and latency across every "
    "core service so the team can plan capacity for the upcoming quarter with care."
)
_P2 = (
    "Each workstream lists a named owner who is accountable for the agreed follow "
    "up actions and for reporting progress at the weekly operating review on time."
)
_P3 = (
    "Reliability budgets are reviewed every month and any regression beyond the "
    "agreed threshold triggers a documented investigation and a remediation plan."
)


def _page(*paragraphs: str) -> str:
    return "\n\n".join(paragraphs)


# --------------------------------------------------------------------------- #
# Deterministic single-literal PDF fixture writer (matches the v6.4 reader).
# --------------------------------------------------------------------------- #


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", "\\n")
    )


def _text_stream(text: str) -> str:
    return "\n".join(["BT", "/F1 12 Tf", "72 720 Td", f"({_escape(text)}) Tj", "ET"])


def _make_pdf(
    pages,
    *,
    title="Quarterly Operating Review",
    author="Platform Team",
    creator="ModelDimensions",
    producer="ModelDimensions",
    creation_date="D:20250101000000Z",
    image_only=False,
    encrypted=False,
    header=b"%PDF-1.4\n",
) -> bytes:
    n_pages = len(pages)
    content_start = 3
    page_start = content_start + n_pages
    info_num = page_start + n_pages
    image_num = info_num + 1
    encrypt_num = image_num + 1
    page_nums = [page_start + i for i in range(n_pages)]

    chunks = [header]

    def obj(num: int, body: str) -> None:
        chunks.append(f"{num} 0 obj\n{body}\nendobj\n".encode("latin-1"))

    obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{p} 0 R" for p in page_nums)
    obj(2, f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>")

    for i, page_text in enumerate(pages):
        cnum = content_start + i
        pnum = page_start + i
        if image_only:
            content = "q 100 0 0 100 72 600 cm /Im0 Do Q"
            res = f" /Resources << /XObject << /Im0 {image_num} 0 R >> >>"
        else:
            content = _text_stream(page_text)
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
    obj(info_num, "<< " + " ".join(info_parts) + " >>")

    if image_only:
        obj(image_num, "<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
                       "/BitsPerComponent 8 /ColorSpace /DeviceGray /Length 1 >>\n"
                       "stream\n\x00\nendstream")

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


def _write(tmp_path, name, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _clean_three_page(tmp_path, name="clean.pdf") -> Path:
    pages = [
        _page("PAGEONEMARKER " + _P1, _P2, _P3),
        _page("PAGETWOMARKER " + _P3, _P1, _P2),
        _page("PAGETHREEMARKER " + _P2, _P3, _P1),
    ]
    return _write(tmp_path, name, _make_pdf(pages))


def _preview_clean(tmp_path, **kwargs):
    path = _clean_three_page(tmp_path)
    return preview_pdf_chunks(
        path,
        source_url="https://example.org/report.pdf",
        owner="Platform Team",
        permission="cc-by-4.0",
        intended_use="knowledge_candidate",
        authority_level="official",
        now=_NOW,
        **kwargs,
    )


def _doc_text(preview: PdfChunkPreview) -> str:
    return "\n\n".join(p.text for p in preview.pages)


# --------------------------------------------------------------------------- #
# 1. Page order preserved and page numbers traceable. ------------------------
# --------------------------------------------------------------------------- #


def test_page_order_preserved(tmp_path):
    preview = _preview_clean(tmp_path)
    assert preview.page_count == 3
    assert [p.page_number for p in preview.pages] == [1, 2, 3]
    assert "PAGEONEMARKER" in preview.pages[0].text
    assert "PAGETWOMARKER" in preview.pages[1].text
    assert "PAGETHREEMARKER" in preview.pages[2].text


def test_page_numbers_traceable_on_chunks(tmp_path):
    preview = _preview_clean(tmp_path)
    assert preview.chunks
    for chunk in preview.chunks:
        assert 1 <= chunk.page_start <= chunk.page_end <= preview.page_count
        assert chunk.file_hash == preview.file_hash


def test_empty_pages_not_dropped(tmp_path):
    pages = [_page(_P1, _P2), "", _page(_P3, _P1)]
    path = _write(tmp_path, "withblank.pdf", _make_pdf(pages))
    extracted = extract_pdf_pages(path, file_hash="h", file_name="withblank.pdf")
    assert [p.page_number for p in extracted] == [1, 2, 3]
    assert extracted[1].is_empty is True


# 2. Chunk identifiers are deterministic. ------------------------------------

def test_chunk_ids_are_deterministic(tmp_path):
    first = _preview_clean(tmp_path)
    second = _preview_clean(tmp_path)
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]
    assert all(c.chunk_id.startswith("pdfchk-") for c in first.chunks)


def test_repeated_runs_produce_identical_output(tmp_path):
    first = _preview_clean(tmp_path)
    second = _preview_clean(tmp_path)
    assert first.to_dict() == second.to_dict()


# 3. Repeated headers and footers are detected. ------------------------------

def test_repeated_headers_and_footers_detected(tmp_path):
    header = "Acme Platform Operations Handbook"
    footer = "Distribution list update pending"
    pages = [
        _page(header, _P1, _P2, footer),
        _page(header, _P3, _P1, footer),
        _page(header, _P2, _P3, footer),
    ]
    path = _write(tmp_path, "hdr.pdf", _make_pdf(pages))
    preview = preview_pdf_chunks(path, now=_NOW)
    assert header in preview.repeated_headers
    assert footer in preview.repeated_footers
    codes = {code for c in preview.chunks for code in c.warning_codes}
    assert PdfChunkWarningCode.CHUNK_REPEATED_HEADER.value in codes
    assert PdfChunkWarningCode.CHUNK_REPEATED_FOOTER.value in codes


# 4. Duplicate and near-duplicate chunks. ------------------------------------

def test_duplicate_chunks_detected(tmp_path):
    same = _page(_P1, _P2, _P3)
    path = _write(tmp_path, "dup.pdf", _make_pdf([same, same]))
    preview = preview_pdf_chunks(path, now=_NOW)
    assert preview.summary.duplicate_chunk_count >= 1


def test_near_duplicate_chunks_classified_conservatively(tmp_path):
    page_a = _page(_P1, _P2, _P3)
    page_b = _page(_P1, _P2, _P3 + " A tiny extra clause was appended here lately.")
    path = _write(tmp_path, "near.pdf", _make_pdf([page_a, page_b]))
    preview = preview_pdf_chunks(path, now=_NOW)
    assert preview.summary.near_duplicate_chunk_count >= 1
    assert preview.summary.duplicate_chunk_count == 0


# 5. Sparse and poor-quality pages. ------------------------------------------

def test_sparse_pages_flagged(tmp_path):
    pages = [_page(_P1, _P2, _P3), "Note."]
    path = _write(tmp_path, "sparse.pdf", _make_pdf(pages))
    preview = preview_pdf_chunks(path, now=_NOW)
    sparse = [q for q in preview.page_quality if q.is_sparse]
    assert sparse
    assert sparse[0].quality_band in (PdfPageQualityBand.POOR, PdfPageQualityBand.UNUSABLE)


def test_poor_quality_pages_lower_preview_readiness(tmp_path):
    pages = ["Short.", "Tiny."]
    path = _write(tmp_path, "poor.pdf", _make_pdf(pages))
    preview = preview_pdf_chunks(
        path, source_url="https://example.org/x.pdf", permission="cc-by-4.0",
        intended_use="knowledge_candidate", authority_level="official", now=_NOW)
    assert preview.summary.preview_ready_for_import is False


# 6. Chunk boundaries respect words and prefer paragraphs. -------------------

def test_chunk_boundaries_do_not_split_words(tmp_path):
    preview = _preview_clean(tmp_path, chunk_size=200, overlap=0, max_chunk_size=320)
    doc = _doc_text(preview)
    for chunk in preview.chunks:
        cs, ce = chunk.char_offset_start, chunk.char_offset_end
        if cs > 0:
            assert doc[cs - 1].isspace(), f"chunk {chunk.chunk_id} starts mid-word"
        if ce < len(doc):
            assert doc[ce].isspace(), f"chunk {chunk.chunk_id} ends mid-word"


def test_paragraph_boundaries_preferred(tmp_path):
    page = _page(_P1, _P2, _P3)
    path = _write(tmp_path, "para.pdf", _make_pdf([page]))
    preview = preview_pdf_chunks(path, chunk_size=320, overlap=0, max_chunk_size=420, now=_NOW)
    doc = _doc_text(preview)
    # At least one chunk must end exactly at a paragraph break (or document end).
    assert any(
        c.char_offset_end >= len(doc) or doc[c.char_offset_end:c.char_offset_end + 2] == "\n\n"
        for c in preview.chunks
    )


# 7. Page-crossing chunks are labelled only when merging is requested. -------

def test_page_crossing_chunks_labelled_when_merging(tmp_path):
    pages = [_P1, _P2]
    path = _write(tmp_path, "merge.pdf", _make_pdf(pages))
    preview = preview_pdf_chunks(
        path, chunk_size=2000, overlap=0, respect_page_boundaries=False, now=_NOW)
    crossing = [c for c in preview.chunks if c.crosses_page_boundary]
    assert crossing
    assert PdfChunkWarningCode.CHUNK_CROSSES_PAGE_BOUNDARY.value in crossing[0].warning_codes


def test_page_bounded_chunks_do_not_cross_by_default(tmp_path):
    preview = _preview_clean(tmp_path)
    assert all(c.crosses_page_boundary is False for c in preview.chunks)


# 8. Fragmented and table-risk chunks. ---------------------------------------

def test_fragmented_chunks_warned(tmp_path):
    fragments = "\n".join(f"item field token {i:02d}" for i in range(20))
    path = _write(tmp_path, "frag.pdf", _make_pdf([fragments]))
    preview = preview_pdf_chunks(path, chunk_size=2000, overlap=0, now=_NOW)
    codes = {code for c in preview.chunks for code in c.warning_codes}
    assert PdfChunkWarningCode.CHUNK_FRAGMENTED.value in codes


def test_table_risk_chunks_warned(tmp_path):
    rows = "\n".join(
        ["Name      Region      Value"]
        + [f"item{i:02d}      west      {i*3}" for i in range(8)]
    )
    path = _write(tmp_path, "table.pdf", _make_pdf([rows]))
    preview = preview_pdf_chunks(path, chunk_size=2000, overlap=0, now=_NOW)
    assert preview.summary.table_risk_chunk_count >= 1


# 9. Sensitivity triggers human review. --------------------------------------

def test_possible_pii_triggers_review(tmp_path):
    page = _page(_P1, "Please contact john.doe@example.com for the onboarding pack.", _P2)
    path = _write(tmp_path, "pii.pdf", _make_pdf([page]))
    preview = preview_pdf_chunks(path, now=_NOW)
    assert preview.summary.possible_pii_chunk_count >= 1
    assert preview.summary.review_required_count >= 1
    assert preview.summary.preview_ready_for_import is False


def test_confidential_markers_trigger_review(tmp_path):
    page = _page("CONFIDENTIAL - do not distribute outside the company.", _P1, _P2)
    path = _write(tmp_path, "conf.pdf", _make_pdf([page]))
    preview = preview_pdf_chunks(path, now=_NOW)
    assert preview.summary.confidential_marker_chunk_count >= 1
    assert preview.summary.review_required_count >= 1
    assert preview.summary.preview_ready_for_import is False


# 10. No-text-layer / OCR-required / blocked intake are never preview-ready. -

def test_no_text_layer_pdf_not_preview_ready(tmp_path):
    path = _write(tmp_path, "scan.pdf", _make_pdf([""], image_only=True))
    preview = preview_pdf_chunks(path, now=_NOW)
    assert preview.has_text_layer is False
    assert preview.requires_ocr is True
    assert preview.summary.preview_ready_for_import is False


def test_blocked_intake_cannot_become_preview_ready(tmp_path):
    path = _clean_three_page(tmp_path)
    from agent.pdf_intake_adapter import assess_pdf_intake

    base = assess_pdf_intake(
        path, source_url="https://example.org/report.pdf",
        permission="cc-by-4.0", intended_use="knowledge_candidate",
        authority_level="official", now=_NOW)
    blocked = dataclasses.replace(base, decision=DataIntakeDecision.BLOCKED)
    preview = preview_pdf_chunks(path, intake_result=blocked, now=_NOW)
    assert preview.intake_blocked is True
    assert preview.chunks == ()
    assert preview.summary.preview_ready_for_import is False


# 11. A clean, governed document is advisory-ready but approves nothing. -----

def test_clean_document_is_preview_ready_but_advisory_only(tmp_path):
    preview = _preview_clean(tmp_path)
    assert preview.summary.preview_ready_for_import is True
    # Advisory only: the preview never escalates to an approval or import action.
    assert preview.preview_status in (PdfPreviewStatus.PREVIEW_ONLY, PdfPreviewStatus.NEEDS_REVIEW)
    record = preview.to_dict()
    assert record["_record"] == "pdf_chunk_preview"
    # Advisory only: no chunk is ever marked approved; status stays preview/needs-review.
    assert all(
        c["preview_status"] in ("preview_only", "needs_review") for c in record["chunks"]
    )
    assert "preview_ready_for_import" in record["summary"]


def test_preview_ready_requires_text_layer_and_quality(tmp_path):
    preview = _preview_clean(tmp_path)
    assert preview.has_text_layer is True
    assert preview.requires_ocr is False
    assert preview.extraction_quality_band in ("acceptable", "good")


# 12. Page-quality metrics are deterministic and well-formed. ----------------

def test_page_quality_metrics_present(tmp_path):
    preview = _preview_clean(tmp_path)
    q = preview.page_quality[0]
    assert q.char_count > 0
    assert q.word_count > 0
    assert q.line_count > 0
    assert 0.0 <= q.text_density <= 1.0
    assert 0.0 <= q.short_line_ratio <= 1.0
    assert q.quality_band in set(PdfPageQualityBand)


def test_analyse_page_quality_empty_page_is_unusable():
    page = pcp.PdfPageText(
        file_hash="h", file_name="f.pdf", page_number=1, text="",
        char_count=0, char_offset_start=0, char_offset_end=0)
    q = analyse_page_quality(page)
    assert q.quality_band == PdfPageQualityBand.UNUSABLE
    assert q.is_empty is True


# 13. Import purity: no writer, retrieval, OCR, LLM, or network dependency. --

def test_module_imports_no_forbidden_dependencies():
    import ast

    source = Path(pcp.__file__).read_text(encoding="utf-8")
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
        "agent.retrieval", "agent.hybrid_retrieval", "agent.pack_builder",
        "agent.knowledge_packs",
        "pytesseract", "pdf2image", "datasets", "huggingface_hub",
        "requests", "urllib", "urllib.request", "httpx", "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"preview must not import forbidden modules: {leaked}"

    body = source.replace(ast.get_docstring(tree, clean=False) or "", "", 1)
    for token in ("load_dataset", "hf_hub_download", "snapshot_download",
                  "requests.get", "urlopen", "MemoryLedger", "save_registry",
                  "propose_source", "memory_review_queue", "source_review_queue",
                  "pytesseract", "pdf2image", "add_to_index", "build_pack",
                  "chat.completions"):
        assert token not in body, f"preview must not reference {token!r}"


def test_preview_does_no_network(tmp_path, monkeypatch):
    import socket

    def _boom(*args, **kwargs):  # pragma: no cover - only fires on a violation
        raise AssertionError("preview attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    preview = _preview_clean(tmp_path)
    assert isinstance(preview, PdfChunkPreview)


# 14. The preview itself writes nothing; only --out persists a report. -------

def test_preview_creates_no_files(tmp_path):
    before = sorted(p.name for p in tmp_path.iterdir())
    _preview_clean(tmp_path)  # creates the input pdf, nothing else
    after = sorted(p.name for p in tmp_path.iterdir())
    assert after == sorted(before + ["clean.pdf"])


def test_cli_stdout_mode_writes_nothing(tmp_path, monkeypatch, capsys):
    pdf = _clean_three_page(tmp_path, "cli.pdf")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    before = sorted(p.name for p in work.iterdir())
    capsys.readouterr()
    argv = ["pdf-preview", "inspect", "--pdf", str(pdf),
            "--now", "2026-06-04T00:00:00+00:00"]
    assert workbench.main(argv) == 0
    out = capsys.readouterr().out
    assert "PREVIEW ONLY" in out
    assert "NOT APPROVED FOR IMPORT" in out
    after = sorted(p.name for p in work.iterdir())
    assert before == after


def test_cli_out_writes_only_the_report(tmp_path, monkeypatch, capsys):
    pdf = _clean_three_page(tmp_path, "cli.pdf")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    out = work / "preview.md"
    capsys.readouterr()
    argv = ["pdf-preview", "inspect", "--pdf", str(pdf),
            "--now", "2026-06-04T00:00:00+00:00", "--out", str(out)]
    assert workbench.main(argv) == 0
    created = [p for p in work.rglob("*") if p.is_file()]
    assert created == [out]


def test_cli_json_out_persists_full_text(tmp_path, monkeypatch, capsys):
    pdf = _clean_three_page(tmp_path, "cli.pdf")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    out = work / "preview.json"
    capsys.readouterr()
    argv = ["pdf-preview", "inspect", "--pdf", str(pdf), "--format", "json",
            "--now", "2026-06-04T00:00:00+00:00", "--out", str(out)]
    assert workbench.main(argv) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["_record"] == "pdf_chunk_preview"
    assert any("text" in chunk for chunk in payload["chunks"])


def test_cli_bad_now_returns_two(tmp_path, monkeypatch):
    pdf = _clean_three_page(tmp_path, "cli.pdf")
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    argv = ["pdf-preview", "inspect", "--pdf", str(pdf), "--now", "not-a-date"]
    assert workbench.main(argv) == 2


# 15. Markdown surfaces sensitivity warnings before text previews. -----------

def test_markdown_lists_sensitivity_before_text(tmp_path):
    page = _page(_P1, "Reach us at jane.roe@example.com any weekday.", _P2)
    path = _write(tmp_path, "pii2.pdf", _make_pdf([page]))
    preview = preview_pdf_chunks(path, now=_NOW)
    markdown = pcp.render_chunk_preview_markdown(preview)
    assert "Sensitivity warnings" in markdown
    assert markdown.index("Sensitivity warnings") < markdown.index("Proposed chunks")


# 16. README documents the v6.5 slice. ---------------------------------------

def test_readme_documents_chunk_preview():
    text = _README.read_text(encoding="utf-8").lower()
    assert "## v6.5" in text
    assert "chunk preview" in text
