"""Tests for the v6.6 approved PDF-to-knowledge-pack importer (write path).

These tests pin the contract that turns an already-reviewed v6.5 chunk preview
into a deterministic knowledge pack:

* advisory preview readiness alone never imports; explicit structured approval
  is required and is bound to both the source file hash and a deterministic
  preview fingerprint;
* only the exact approved chunk ids are imported, rejected/unknown chunks fail
  closed, and a changed preview or changed PDF hash invalidates the approval;
* blocking PDF conditions and blocking chunk warnings (possible PII, confidential
  markers, malformed unicode, low source quality, human-review-required) prevent
  import;
* imported chunk text is byte-identical to the preview, full source/page/approval
  lineage is preserved, and content/manifest hashes are deterministic;
* validation and dry-run write nothing, ``--write`` creates only the intended
  pack files, an existing target fails closed, and a write failure leaves no
  partial pack;
* the importer imports no retrieval/memory/registry/proposal writer and no OCR
  or LLM client, and touches no network.

PDF fixtures are generated programmatically; synthetic preview dicts are used to
exercise the warning paths deterministically.
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import pdf_pack_importer as ppi  # noqa: E402
from agent.pdf_chunk_preview import PdfChunkPreview, preview_pdf_chunks  # noqa: E402
from agent.pdf_pack_importer import (  # noqa: E402
    ImportFindingCode,
    ImportStatus,
    PdfImportApproval,
    PdfImportRequest,
    compute_preview_fingerprint,
    content_hash,
    import_pdf_pack,
    validate_import,
)

_README = ROOT / "README.md"
_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)
_IMPORT_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
_PACK_ID = "pdf_demo_pack"

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


def _make_pdf(pages, *, header=b"%PDF-1.4\n") -> bytes:
    n_pages = len(pages)
    content_start = 3
    page_start = content_start + n_pages
    info_num = page_start + n_pages
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
        content = _text_stream(page_text)
        obj(cnum, f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
        obj(pnum, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                  f"/Contents {cnum} 0 R >>")

    obj(info_num, "<< /Title (Quarterly Operating Review) /Author (Platform Team) "
                  "/Creator (ModelDimensions) /Producer (ModelDimensions) "
                  "/CreationDate (D:20250101000000Z) >>")
    chunks.append(f"trailer\n<< /Root 1 0 R /Info {info_num} 0 R >>\n".encode("latin-1"))
    chunks.append(b"%%EOF\n")
    return b"".join(chunks)


def _clean_three_page(tmp_path, name="clean.pdf") -> Path:
    pages = [
        _page("PAGEONEMARKER " + _P1, _P2, _P3),
        _page("PAGETWOMARKER " + _P3, _P1, _P2),
        _page("PAGETHREEMARKER " + _P2, _P3, _P1),
    ]
    path = tmp_path / name
    path.write_bytes(_make_pdf(pages))
    return path


def _clean_preview(tmp_path, name="clean.pdf", **kwargs) -> PdfChunkPreview:
    path = _clean_three_page(tmp_path, name)
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


def _preview_dict(tmp_path, **kwargs) -> dict:
    return _clean_preview(tmp_path, **kwargs).to_dict(include_full_text=True)


def _approval_for(preview: dict, *, chunk_ids=None, **overrides) -> PdfImportApproval:
    ids = chunk_ids if chunk_ids is not None else [c["chunk_id"] for c in preview["chunks"]]
    defaults = dict(
        approval_id="appr-001",
        approved_by="reviewer@local",
        approved_at="2026-06-04T00:00:00+00:00",
        source_file_hash=preview["file_hash"],
        preview_fingerprint=compute_preview_fingerprint(preview),
        approved_chunk_ids=tuple(ids),
        intended_pack_id=_PACK_ID,
        intended_use="knowledge_candidate",
        source_title="Quarterly Operating Review",
        authority_level="official",
        provenance="declared by document owner",
        permission_or_licence="cc-by-4.0",
        domain="general",
    )
    defaults.update(overrides)
    return PdfImportApproval(**defaults)


def _synthetic_preview(*, warning_codes=(), ready=True, **top) -> dict:
    """A minimal valid preview dict with one chunk carrying the given warnings."""
    text = _P1 + "\n\n" + _P2
    chunk = {
        "chunk_id": "pdfchk-synthetic1",
        "file_hash": "synhash",
        "file_name": "synthetic.pdf",
        "page_start": 1,
        "page_end": 1,
        "char_offset_start": 0,
        "char_offset_end": len(text),
        "char_count": len(text),
        "extraction_method": "text_layer",
        "crosses_page_boundary": False,
        "preview_status": "preview_only",
        "requires_human_review": bool(warning_codes),
        "warning_codes": list(warning_codes),
        "warnings": [],
        "text_preview": text[:200],
        "text": text,
    }
    preview = {
        "_record": "pdf_chunk_preview",
        "file_hash": "synhash",
        "file_name": "synthetic.pdf",
        "file_path": "synthetic.pdf",
        "page_count": 1,
        "chunk_size": 1000,
        "overlap": 150,
        "max_chunk_size": 1500,
        "respect_page_boundaries": True,
        "intake_decision": "approved_for_knowledge",
        "intake_blocked": False,
        "has_text_layer": True,
        "requires_ocr": False,
        "extraction_quality_band": "good",
        "preview_status": "preview_only",
        "intake_findings": [],
        "repeated_headers": [],
        "repeated_footers": [],
        "duplicate_page_pairs": [],
        "near_duplicate_page_pairs": [],
        "pages": [],
        "page_quality": [],
        "chunks": [chunk],
        "summary": {"preview_ready_for_import": ready},
    }
    preview.update(top)
    return preview


def _request(preview, approval, *, pack_dir=None, write=False) -> PdfImportRequest:
    return PdfImportRequest(
        preview=preview,
        approval=approval,
        pack_id=_PACK_ID,
        pack_dir=str(pack_dir) if pack_dir is not None else None,
        write=write,
    )


# --------------------------------------------------------------------------- #
# 1-2. Readiness alone never imports; explicit approval is required.
# --------------------------------------------------------------------------- #


def test_preview_readiness_alone_does_not_import(tmp_path):
    preview = _preview_dict(tmp_path)
    assert preview["summary"]["preview_ready_for_import"] is True
    empty = _approval_for(preview, chunk_ids=[])
    validation = validate_import(preview, empty, pack_id=_PACK_ID)
    assert validation.valid is False
    assert ImportFindingCode.APPROVAL_MISSING in {f.code for f in validation.findings}


def test_explicit_approval_required(tmp_path):
    preview = _preview_dict(tmp_path)
    result = import_pdf_pack(_request(preview, _approval_for(preview, chunk_ids=[])))
    assert result.written is False
    assert result.status is ImportStatus.APPROVAL_MISMATCH


# 3-4. Approval is bound to source hash and preview fingerprint. --------------


def test_approval_must_match_source_hash(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview, source_file_hash="0" * 64)
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert validation.status is ImportStatus.SOURCE_CHANGED
    assert ImportFindingCode.SOURCE_HASH_MISMATCH in {f.code for f in validation.findings}


def test_approval_must_match_preview_fingerprint(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview, preview_fingerprint="pdfprev-0000000000000000")
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert validation.status is ImportStatus.PREVIEW_CHANGED
    assert ImportFindingCode.PREVIEW_FINGERPRINT_MISMATCH in {f.code for f in validation.findings}


# 5-7. Only approved chunks imported; rejected/unknown fail closed. -----------


def test_only_approved_chunk_ids_imported(tmp_path):
    preview = _preview_dict(tmp_path)
    first = preview["chunks"][0]["chunk_id"]
    approval = _approval_for(preview, chunk_ids=[first])
    result = import_pdf_pack(_request(preview, approval))
    assert result.status is ImportStatus.DRY_RUN_READY
    assert [c.original_preview_chunk_id for c in result.imported_chunks] == [first]
    assert result.chunk_count == 1
    assert len(result.validation.excluded_chunk_ids) == len(preview["chunks"]) - 1


def test_rejected_chunks_not_imported(tmp_path):
    preview = _preview_dict(tmp_path)
    ids = [c["chunk_id"] for c in preview["chunks"]]
    approval = _approval_for(preview, chunk_ids=ids[:1], rejected_chunk_ids=tuple(ids[1:]))
    result = import_pdf_pack(_request(preview, approval))
    assert result.status is ImportStatus.DRY_RUN_READY
    imported = {c.original_preview_chunk_id for c in result.imported_chunks}
    assert imported == {ids[0]}
    for rejected in ids[1:]:
        assert rejected not in imported


def test_approved_and_rejected_same_chunk_fails_closed(tmp_path):
    preview = _preview_dict(tmp_path)
    ids = [c["chunk_id"] for c in preview["chunks"]]
    approval = _approval_for(preview, chunk_ids=ids, rejected_chunk_ids=(ids[0],))
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert ImportFindingCode.APPROVAL_CHUNK_REJECTED in {f.code for f in validation.findings}


def test_unknown_chunk_ids_fail_closed(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview, chunk_ids=["pdfchk-doesnotexist"])
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert validation.status is ImportStatus.APPROVAL_MISMATCH
    assert ImportFindingCode.APPROVAL_CHUNK_UNKNOWN in {f.code for f in validation.findings}


# 8-9. Changed preview / changed PDF invalidates approval. -------------------


def test_changed_preview_invalidates_approval(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)  # bound to the default chunking
    # Re-preview with different chunk settings -> different fingerprint.
    changed = _clean_preview(tmp_path, chunk_size=300, overlap=50).to_dict(include_full_text=True)
    validation = validate_import(changed, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert ImportFindingCode.PREVIEW_FINGERPRINT_MISMATCH in {f.code for f in validation.findings}


def test_changed_pdf_hash_invalidates_approval(tmp_path):
    preview = _preview_dict(tmp_path, name="a.pdf")
    approval = _approval_for(preview)
    other = _clean_preview(tmp_path, name="b.pdf").to_dict(include_full_text=True)
    # Different file name still hashes the same bytes; force genuinely different bytes.
    pages = [_page("DIFFERENT " + _P2, _P3), _page("OTHER " + _P1, _P3)]
    p = tmp_path / "c.pdf"
    p.write_bytes(_make_pdf(pages))
    other = preview_pdf_chunks(
        p, source_url="https://example.org/report.pdf", owner="Platform Team",
        permission="cc-by-4.0", intended_use="knowledge_candidate",
        authority_level="official", now=_NOW,
    ).to_dict(include_full_text=True)
    assert other["file_hash"] != preview["file_hash"]
    validation = validate_import(other, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert validation.status is ImportStatus.SOURCE_CHANGED


# 10-12. Blocking warnings / PII / confidential prevent import. --------------


def test_blocking_warnings_prevent_import():
    preview = _synthetic_preview(warning_codes=["chunk_low_source_quality"])
    approval = _approval_for(preview)
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert validation.status is ImportStatus.BLOCKED
    assert ImportFindingCode.BLOCKING_CHUNK_WARNING in {f.code for f in validation.findings}


def test_unresolved_pii_prevents_import():
    preview = _synthetic_preview(warning_codes=["chunk_possible_pii"])
    approval = _approval_for(preview)
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert ImportFindingCode.UNRESOLVED_PII in {f.code for f in validation.findings}


def test_pii_chunk_may_be_excluded_then_import_proceeds():
    # A clean chunk plus a PII chunk; approving only the clean one is allowed.
    preview = _synthetic_preview()
    pii_chunk = dict(preview["chunks"][0])
    pii_chunk["chunk_id"] = "pdfchk-pii0001"
    pii_chunk["warning_codes"] = ["chunk_possible_pii"]
    preview["chunks"] = [preview["chunks"][0], pii_chunk]
    approval = _approval_for(preview, chunk_ids=["pdfchk-synthetic1"])
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is True
    assert "pdfchk-pii0001" in validation.excluded_chunk_ids


def test_unresolved_confidential_prevents_import():
    preview = _synthetic_preview(warning_codes=["chunk_confidential_marker"])
    approval = _approval_for(preview)
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert ImportFindingCode.UNRESOLVED_CONFIDENTIAL_CONTENT in {f.code for f in validation.findings}


def test_confidential_allowed_only_by_explicit_policy():
    preview = _synthetic_preview(warning_codes=["chunk_confidential_marker"])
    approval = _approval_for(preview, allow_confidential=True)
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is True


# 13-15. Text immutability, lineage, deterministic hashes. -------------------


def test_chunk_text_remains_identical_to_preview(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    result = import_pdf_pack(_request(preview, approval), now=_IMPORT_NOW)
    by_id = {c.original_preview_chunk_id: c for c in result.imported_chunks}
    for chunk in preview["chunks"]:
        assert by_id[chunk["chunk_id"]].text == chunk["text"]


def test_page_lineage_preserved(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    result = import_pdf_pack(_request(preview, approval), now=_IMPORT_NOW)
    by_id = {c.original_preview_chunk_id: c for c in result.imported_chunks}
    for chunk in preview["chunks"]:
        imported = by_id[chunk["chunk_id"]]
        assert imported.page_start == chunk["page_start"]
        assert imported.page_end == chunk["page_end"]
        assert imported.char_offset_start == chunk["char_offset_start"]
        assert imported.char_offset_end == chunk["char_offset_end"]
        assert imported.source_file_hash == preview["file_hash"]
        assert imported.approval_id == approval.approval_id
        assert imported.approved_by == approval.approved_by


def test_content_hashes_are_deterministic(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    first = import_pdf_pack(_request(preview, approval), now=_IMPORT_NOW)
    second = import_pdf_pack(_request(preview, approval), now=_IMPORT_NOW)
    assert [c.content_hash for c in first.imported_chunks] == [
        c.content_hash for c in second.imported_chunks
    ]
    for c in first.imported_chunks:
        assert c.content_hash == content_hash(c.text)


def test_wrong_approved_content_hash_is_rejected(tmp_path):
    preview = _preview_dict(tmp_path)
    first = preview["chunks"][0]["chunk_id"]
    approval = _approval_for(
        preview,
        chunk_ids=[first],
        approved_chunk_content_hashes={first: "sha256:deadbeef"},
    )
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert ImportFindingCode.CONTENT_HASH_MISMATCH in {f.code for f in validation.findings}


def test_imported_chunk_count_equals_approved_count(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    result = import_pdf_pack(_request(preview, approval), now=_IMPORT_NOW)
    assert result.chunk_count == len(approval.approved_chunk_ids)


# 16. Manifest is deterministic (content identity ignores timestamps). -------


def test_pack_manifest_is_deterministic(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    first = import_pdf_pack(_request(preview, approval), now=_IMPORT_NOW)
    later = datetime(2027, 1, 1, tzinfo=timezone.utc)
    second = import_pdf_pack(_request(preview, approval), now=later)
    # Stable content identity is timestamp-independent.
    assert first.manifest.manifest_hash == second.manifest.manifest_hash
    assert first.manifest.preview_fingerprint == second.manifest.preview_fingerprint
    # Operational timestamp differs.
    assert first.manifest.import_timestamp != second.manifest.import_timestamp


def test_manifest_carries_full_lineage(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    manifest = import_pdf_pack(_request(preview, approval), now=_IMPORT_NOW).manifest.to_dict()
    for key in (
        "pack_id", "pack_version", "source_count", "chunk_count", "source_file_hash",
        "preview_fingerprint", "approval_id", "approval_actor", "approval_timestamp",
        "import_timestamp", "intended_use", "authority_level", "provenance",
        "permission_or_licence", "excluded_chunk_ids", "unresolved_nonblocking_findings",
        "importer_version", "manifest_hash",
    ):
        assert key in manifest


# 17-18. Validate and dry-run write nothing. ---------------------------------


def test_dry_run_writes_nothing(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    pack_dir = tmp_path / "out" / _PACK_ID
    result = import_pdf_pack(_request(preview, approval, pack_dir=pack_dir, write=False),
                             now=_IMPORT_NOW)
    assert result.status is ImportStatus.DRY_RUN_READY
    assert result.written is False
    assert not pack_dir.exists()
    assert not (tmp_path / "out").exists()


def test_validate_writes_nothing(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    pack_dir = tmp_path / "out" / _PACK_ID
    before = sorted(p.name for p in tmp_path.iterdir())
    validation = validate_import(preview, approval, pack_id=_PACK_ID, pack_dir=str(pack_dir))
    assert validation.valid is True
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after
    assert not pack_dir.exists()


# 19-21. Safe write semantics. -----------------------------------------------


def test_write_creates_only_intended_files(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    pack_dir = tmp_path / "out" / _PACK_ID
    result = import_pdf_pack(_request(preview, approval, pack_dir=pack_dir, write=True),
                             now=_IMPORT_NOW)
    assert result.status is ImportStatus.IMPORTED
    assert result.written is True
    assert pack_dir.is_dir()
    names = sorted(p.name for p in pack_dir.iterdir())
    assert names == ["knowledge.jsonl", "manifest.json"]
    # knowledge.jsonl carries one source, one document, and N chunk records.
    records = [json.loads(line) for line in
               (pack_dir / "knowledge.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [r["_record"] for r in records]
    assert kinds.count("source") == 1
    assert kinds.count("document") == 1
    assert kinds.count("chunk") == len(preview["chunks"])


def test_existing_target_fails_closed(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    pack_dir = tmp_path / "out" / _PACK_ID
    pack_dir.mkdir(parents=True)
    sentinel = pack_dir / "keep.txt"
    sentinel.write_text("do not touch", encoding="utf-8")
    result = import_pdf_pack(_request(preview, approval, pack_dir=pack_dir, write=True),
                             now=_IMPORT_NOW)
    assert result.status is ImportStatus.TARGET_EXISTS
    assert result.written is False
    assert sentinel.read_text(encoding="utf-8") == "do not touch"
    assert sorted(p.name for p in pack_dir.iterdir()) == ["keep.txt"]


def test_partial_output_cleaned_on_write_failure(tmp_path, monkeypatch):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    pack_dir = tmp_path / "out" / _PACK_ID

    def _boom(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(ppi.os, "rename", _boom)
    result = import_pdf_pack(_request(preview, approval, pack_dir=pack_dir, write=True),
                             now=_IMPORT_NOW)
    assert result.status is ImportStatus.WRITE_FAILED
    assert result.written is False
    assert not pack_dir.exists()
    # No leftover temporary import directory beside the target.
    leftovers = [p for p in (tmp_path / "out").iterdir()
                 if p.name.startswith(f".{_PACK_ID}.import-")]
    assert leftovers == []


def test_write_requires_explicit_out(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    result = import_pdf_pack(_request(preview, approval, pack_dir=None, write=True),
                             now=_IMPORT_NOW)
    assert result.status is ImportStatus.WRITE_FAILED
    assert result.written is False


# 22-26. No retrieval / memory / registry / proposal / OCR / LLM / network. --


def test_no_retrieval_or_pack_system_files_written(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    pack_dir = tmp_path / "out" / _PACK_ID
    import_pdf_pack(_request(preview, approval, pack_dir=pack_dir, write=True), now=_IMPORT_NOW)
    names = {p.name for p in pack_dir.iterdir()}
    # No bank/ledger/proposals/registry geometry is produced by the importer.
    assert names == {"manifest.json", "knowledge.jsonl"}


def test_no_memory_ledger_write(tmp_path, monkeypatch):
    import importlib

    ledger_mod = importlib.import_module("agent.memory_ledger")
    calls = []
    original = ledger_mod.MemoryLedger.add

    def _spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ledger_mod.MemoryLedger, "add", _spy)
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    pack_dir = tmp_path / "out" / _PACK_ID
    import_pdf_pack(_request(preview, approval, pack_dir=pack_dir, write=True), now=_IMPORT_NOW)
    assert calls == []


def test_no_source_registry_or_demo_state_mutated(tmp_path):
    registry = ROOT / "demos" / "source_registry.jsonl"
    before = registry.read_bytes() if registry.exists() else None
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    pack_dir = tmp_path / "out" / _PACK_ID
    import_pdf_pack(_request(preview, approval, pack_dir=pack_dir, write=True), now=_IMPORT_NOW)
    after = registry.read_bytes() if registry.exists() else None
    assert before == after


def test_module_imports_no_forbidden_dependencies():
    source = Path(ppi.__file__).read_text(encoding="utf-8")
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
        "MemoryLedger", "save_registry", "save_memory_review_queue",
        "save_source_review_queue", "propose_source_updates", "build_memory_proposals",
        "agent.memory_ledger", "agent.source_registry", "agent.workbench_service",
        "agent.retrieval", "agent.hybrid_retrieval", "agent.pack_builder",
        "agent.project_packs", "agent.knowledge_packs", "WorkbenchService", "PackRegistry",
        "pytesseract", "pdf2image", "datasets", "huggingface_hub",
        "requests", "urllib", "urllib.request", "httpx", "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"importer must not import forbidden modules: {leaked}"

    body = source.replace(ast.get_docstring(tree, clean=False) or "", "", 1)
    for token in (
        "load_dataset", "hf_hub_download", "snapshot_download", "requests.get", "urlopen",
        "MemoryLedger", "save_registry", "propose_source", "memory_review_queue",
        "source_review_queue", "pytesseract", "pdf2image", "add_to_index", "build_pack",
        "WorkbenchService", "PackRegistry", "chat.completions",
    ):
        assert token not in body, f"importer must not reference {token!r}"


def test_importer_does_no_network(tmp_path, monkeypatch):
    import socket

    def _boom(*args, **kwargs):  # pragma: no cover - only fires on a violation
        raise AssertionError("importer attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    pack_dir = tmp_path / "out" / _PACK_ID
    result = import_pdf_pack(_request(preview, approval, pack_dir=pack_dir, write=True),
                             now=_IMPORT_NOW)
    assert result.written is True


# 27. Determinism of validation. ---------------------------------------------


def test_validation_is_deterministic(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    first = validate_import(preview, approval, pack_id=_PACK_ID)
    second = validate_import(preview, approval, pack_id=_PACK_ID)
    assert first.to_dict() == second.to_dict()


def test_blocked_intake_prevents_import():
    preview = _synthetic_preview(intake_blocked=True)
    approval = _approval_for(preview)
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert ImportFindingCode.BLOCKING_PDF_FINDING in {f.code for f in validation.findings}


def test_no_text_layer_prevents_import():
    preview = _synthetic_preview(has_text_layer=False, requires_ocr=True)
    approval = _approval_for(preview)
    validation = validate_import(preview, approval, pack_id=_PACK_ID)
    assert validation.valid is False
    assert ImportFindingCode.BLOCKING_PDF_FINDING in {f.code for f in validation.findings}


# 28-30. CLI surface. --------------------------------------------------------


def _write_inputs(tmp_path, preview, approval):
    preview_path = tmp_path / "preview.json"
    approval_path = tmp_path / "approval.json"
    preview_path.write_text(json.dumps(preview), encoding="utf-8")
    approval_path.write_text(json.dumps(approval.to_dict()), encoding="utf-8")
    return preview_path, approval_path


def _workbench():
    sys.path.insert(0, str(ROOT / "app"))
    import workbench  # noqa: E402
    return workbench


def test_cli_validate_writes_nothing(tmp_path, monkeypatch, capsys):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    preview_path, approval_path = _write_inputs(tmp_path, preview, approval)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    workbench = _workbench()
    capsys.readouterr()
    argv = ["pdf-pack", "validate", "--preview", str(preview_path),
            "--approval", str(approval_path), "--pack-id", _PACK_ID]
    assert workbench.main(argv) == 0
    out = capsys.readouterr().out
    assert "validation" in out.lower()
    assert sorted(p.name for p in work.iterdir()) == []


def test_cli_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    preview_path, approval_path = _write_inputs(tmp_path, preview, approval)
    pack_dir = tmp_path / "packs" / _PACK_ID
    workbench = _workbench()
    capsys.readouterr()
    argv = ["pdf-pack", "import", "--preview", str(preview_path),
            "--approval", str(approval_path), "--pack-id", _PACK_ID,
            "--out", str(pack_dir), "--dry-run", "--now", "2026-06-05T12:00:00+00:00"]
    assert workbench.main(argv) == 0
    assert not pack_dir.exists()


def test_cli_write_creates_pack(tmp_path, capsys):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    preview_path, approval_path = _write_inputs(tmp_path, preview, approval)
    pack_dir = tmp_path / "packs" / _PACK_ID
    workbench = _workbench()
    capsys.readouterr()
    argv = ["pdf-pack", "import", "--preview", str(preview_path),
            "--approval", str(approval_path), "--pack-id", _PACK_ID,
            "--out", str(pack_dir), "--write", "--now", "2026-06-05T12:00:00+00:00"]
    assert workbench.main(argv) == 0
    assert sorted(p.name for p in pack_dir.iterdir()) == ["knowledge.jsonl", "manifest.json"]


def test_cli_bad_now_returns_2(tmp_path):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview)
    preview_path, approval_path = _write_inputs(tmp_path, preview, approval)
    workbench = _workbench()
    argv = ["pdf-pack", "validate", "--preview", str(preview_path),
            "--approval", str(approval_path), "--pack-id", _PACK_ID, "--now", "not-a-date"]
    assert workbench.main(argv) == 2


def test_cli_invalid_approval_validate_returns_1(tmp_path, capsys):
    preview = _preview_dict(tmp_path)
    approval = _approval_for(preview, source_file_hash="0" * 64)
    preview_path, approval_path = _write_inputs(tmp_path, preview, approval)
    workbench = _workbench()
    capsys.readouterr()
    argv = ["pdf-pack", "validate", "--preview", str(preview_path),
            "--approval", str(approval_path), "--pack-id", _PACK_ID]
    assert workbench.main(argv) == 1


# 31. README documents v6.6. -------------------------------------------------


def test_readme_documents_v66():
    text = _README.read_text(encoding="utf-8")
    assert "## v6.6" in text
    assert "preview fingerprint" in text.lower()
    assert "dry-run" in text.lower() or "dry run" in text.lower()
