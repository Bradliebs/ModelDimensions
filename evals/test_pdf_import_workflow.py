"""Governance tests for the Consultant Workbench PDF import workflow.

These tests pin the four prohibitions the Imports page must honour: uploading,
assessing, previewing, validating, dry-running, evaluating, proposing, and
requesting activation must never create a pack, never activate a pack, never
create memory, never change retrieval, and never bypass provenance/permission
checks. The only write is the explicit, confirmed pack creation, and even then
only the pack's own files are written.

PDF fixtures are generated programmatically (same deterministic writer used by
the importer tests).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import pdf_import_workflow as wf  # noqa: E402
from agent.pdf_pack_importer import ImportStatus  # noqa: E402

_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)
_IMPORT_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)

# Nine distinct, sentence-terminated paragraphs (one document, no repeated
# content) so the preview yields clean, non-duplicate chunks for the tests.
_PARAGRAPHS = (
    "The platform reliability program tracks throughput and latency across every "
    "core service so the team can plan capacity for the upcoming quarter with care.",
    "Capacity headroom for the payments tier is reviewed against forecast demand so "
    "that seasonal traffic peaks never exhaust the provisioned compute reservations.",
    "The incident commander rotation publishes a weekly schedule and confirms that "
    "every on-call engineer has acknowledged their shift before the handover meeting.",
    "Each workstream lists a named owner who is accountable for the agreed follow "
    "up actions and for reporting progress at the weekly operating review on time.",
    "Deployment freezes are announced two business days ahead of major launches so "
    "that downstream teams can reschedule risky migrations away from the freeze window.",
    "Customer escalations above the agreed severity are mirrored into the reliability "
    "channel where a triage lead assigns an owner and a target resolution timestamp.",
    "Reliability budgets are reviewed every month and any regression beyond the "
    "agreed threshold triggers a documented investigation and a remediation plan.",
    "Latency objectives for the search path are expressed as percentile targets that "
    "the dashboard evaluates against a rolling thirty day window of production traffic.",
    "Postmortems are circulated within five business days and capture the contributing "
    "factors, the corrective actions, and the owners responsible for each follow up.",
)


# --------------------------------------------------------------------------- #
# Deterministic PDF fixture writer (matches the v6.4 reader).
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


def _page(*paragraphs: str) -> str:
    return "\n\n".join(paragraphs)


def _clean_pdf_bytes() -> bytes:
    pages = [
        _page("PAGEONEMARKER " + _PARAGRAPHS[0], _PARAGRAPHS[1], _PARAGRAPHS[2]),
        _page("PAGETWOMARKER " + _PARAGRAPHS[3], _PARAGRAPHS[4], _PARAGRAPHS[5]),
        _page("PAGETHREEMARKER " + _PARAGRAPHS[6], _PARAGRAPHS[7], _PARAGRAPHS[8]),
    ]
    return _make_pdf(pages)


def _snapshot(root: Path) -> dict:
    """Map every file under ``root`` to its bytes (for write-detection)."""
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _stage(tmp_path: Path) -> Path:
    return wf.stage_upload(
        _clean_pdf_bytes(), filename="quarterly review.pdf",
        workspace=tmp_path / "staging",
    )


def _preview_dict(path: Path) -> dict:
    cp = wf.preview(
        path, source_url="https://example.org/report.pdf", owner="Platform Team",
        permission="cc-by-4.0", intended_use="knowledge_candidate",
        authority_level="official", now=_NOW,
    )
    return wf.preview_to_dict(cp)


def _approval(preview: dict, *, pack_id="quarterly-review", approved_ids=None,
              excluded_ids=()):
    ids = (approved_ids if approved_ids is not None
           else [c["chunk_id"] for c in preview["chunks"]])
    return wf.build_approval(
        preview, approval_id="appr-001", approved_by="reviewer@local",
        approved_at="2026-06-04T00:00:00+00:00", approved_chunk_ids=ids,
        excluded_chunk_ids=excluded_ids, intended_pack_id=pack_id,
        intended_use="knowledge_candidate", source_title="Quarterly Operating Review",
        authority_level="official", provenance="https://example.org/report.pdf",
        permission_or_licence="cc-by-4.0", domain="general",
    )


# --------------------------------------------------------------------------- #
# Shape / availability
# --------------------------------------------------------------------------- #

def test_backend_is_available_in_this_build():
    status = wf.backend_availability()
    assert status.available is True
    assert status.missing == ()


def test_workflow_covers_all_required_capabilities():
    keys = {s.key for s in wf.PDF_WORKFLOW_STEPS}
    required = {
        "upload", "metadata", "assess", "quality", "findings", "preview",
        "exclude", "approval", "dry_run", "create", "evaluate", "activate",
    }
    assert required <= keys


def test_normalize_pack_id_produces_valid_ids():
    from agent.pdf_pack_importer import is_valid_pack_id

    assert is_valid_pack_id(wf.normalize_pack_id("Quarterly Operating Review!"))
    assert is_valid_pack_id(wf.normalize_pack_id("   "))


# --------------------------------------------------------------------------- #
# Read-only steps write nothing
# --------------------------------------------------------------------------- #

def test_stage_assess_preview_validate_dry_run_write_no_pack(tmp_path):
    path = _stage(tmp_path)
    before = _snapshot(tmp_path)

    result = wf.assess(
        path, source_url="https://example.org/report.pdf", owner="Platform Team",
        permission="cc-by-4.0", intended_use="knowledge_candidate",
        authority_level="official", now=_NOW,
    )
    assert result is not None
    # extraction quality is available (step 4) and findings (step 5)
    assert result.candidate.metadata.parse_quality.extraction_quality_band

    preview = _preview_dict(path)
    assert preview["chunks"]

    approval = _approval(preview)
    validation = wf.validate(preview, approval, pack_id="quarterly-review")
    assert validation is not None

    dry = wf.dry_run(preview, approval, pack_id="quarterly-review", now=_IMPORT_NOW)
    assert dry.written is False

    after = _snapshot(tmp_path)
    assert after == before  # no pack, no state, nothing written


# --------------------------------------------------------------------------- #
# Prohibition: uploading must not create a pack / activate / update retrieval
# --------------------------------------------------------------------------- #

def test_create_pack_without_confirm_writes_nothing(tmp_path):
    path = _stage(tmp_path)
    preview = _preview_dict(path)
    approval = _approval(preview)
    import_root = tmp_path / "packs"
    before = _snapshot(tmp_path)

    result = wf.create_pack(
        preview, approval, pack_id="quarterly-review", import_root=import_root,
        confirm=False, now=_IMPORT_NOW,
    )
    assert result.written is False
    assert not wf.pack_dir_for("quarterly-review", import_root=import_root).exists()
    assert _snapshot(tmp_path) == before


def test_create_pack_with_confirm_writes_only_pack_files(tmp_path):
    path = _stage(tmp_path)
    preview = _preview_dict(path)
    approval = _approval(preview)
    import_root = tmp_path / "packs"

    result = wf.create_pack(
        preview, approval, pack_id="quarterly-review", import_root=import_root,
        confirm=True, now=_IMPORT_NOW,
    )
    assert result.written is True
    assert result.status == ImportStatus.IMPORTED

    pack_dir = wf.pack_dir_for("quarterly-review", import_root=import_root)
    written = {p.name for p in pack_dir.iterdir()}
    assert written == {"manifest.json", "knowledge.jsonl"}

    # Nothing was written outside the pack directory itself.
    outside = [
        str(p) for p in tmp_path.rglob("*")
        if p.is_file() and pack_dir not in p.parents and p != path
    ]
    assert outside == []


def test_chunk_exclusion_is_honoured(tmp_path):
    path = _stage(tmp_path)
    preview = _preview_dict(path)
    all_ids = [c["chunk_id"] for c in preview["chunks"]]
    assert len(all_ids) >= 2
    excluded = all_ids[-1]
    approved = all_ids[:-1]

    approval = _approval(preview, pack_id="subset-pack", approved_ids=approved,
                         excluded_ids=(excluded,))
    result = wf.create_pack(
        preview, approval, pack_id="subset-pack",
        import_root=tmp_path / "packs", confirm=True, now=_IMPORT_NOW,
    )
    assert result.written is True
    # Imported chunks carry their source preview chunk id as lineage; the
    # excluded chunk must not appear and only the approved subset is imported.
    imported_sources = {c.original_preview_chunk_id for c in result.imported_chunks}
    assert excluded not in imported_sources
    assert imported_sources == set(approved)


# --------------------------------------------------------------------------- #
# Retrieval evaluation is read-only
# --------------------------------------------------------------------------- #

def test_evaluate_is_read_only_and_returns_summary(tmp_path):
    path = _stage(tmp_path)
    preview = _preview_dict(path)
    approval = _approval(preview)
    import_root = tmp_path / "packs"
    wf.create_pack(
        preview, approval, pack_id="quarterly-review", import_root=import_root,
        confirm=True, now=_IMPORT_NOW,
    )
    pack_dir = wf.pack_dir_for("quarterly-review", import_root=import_root)
    before = _snapshot(pack_dir)

    outcome = wf.evaluate(pack_dir)
    assert outcome.summary.case_count > 0
    assert outcome.summary.case_count == len(outcome.results)

    assert _snapshot(pack_dir) == before  # evaluation never mutates the pack


# --------------------------------------------------------------------------- #
# Registry proposal is a proposal only
# --------------------------------------------------------------------------- #

def test_propose_registry_is_proposal_only(tmp_path):
    path = _stage(tmp_path)
    preview = _preview_dict(path)
    approval = _approval(preview)
    import_root = tmp_path / "packs"
    wf.create_pack(
        preview, approval, pack_id="quarterly-review", import_root=import_root,
        confirm=True, now=_IMPORT_NOW,
    )
    pack_dir = wf.pack_dir_for("quarterly-review", import_root=import_root)
    registry = tmp_path / "source_registry.jsonl"
    before = _snapshot(tmp_path)

    proposal = wf.propose_registry(
        pack_dir, registry_path=registry, now=_IMPORT_NOW)
    assert proposal.requires_human_approval is True
    assert proposal.status == "proposed"
    assert not registry.exists()  # proposing never writes the registry
    assert _snapshot(tmp_path) == before


# --------------------------------------------------------------------------- #
# Prohibition: activation request must not activate
# --------------------------------------------------------------------------- #

def test_request_activation_never_activates(tmp_path):
    path = _stage(tmp_path)
    preview = _preview_dict(path)
    approval = _approval(preview)
    import_root = tmp_path / "packs"
    wf.create_pack(
        preview, approval, pack_id="quarterly-review", import_root=import_root,
        confirm=True, now=_IMPORT_NOW,
    )
    pack_dir = wf.pack_dir_for("quarterly-review", import_root=import_root)
    state_path = tmp_path / "active_knowledge_packs.jsonl"
    audit_path = tmp_path / "activation_audit.jsonl"
    before = _snapshot(tmp_path)

    validation = wf.request_activation(
        pack_dir, approval_id="act-001", approved_by="reviewer@local",
        approved_at="2026-06-05T12:00:00+00:00", state_path=state_path,
        audit_path=audit_path, now=_IMPORT_NOW,
    )
    # Fail-closed: without bound evaluation evidence, activation is not granted.
    assert validation.ok is False
    assert not state_path.exists()
    assert not audit_path.exists()
    assert _snapshot(tmp_path) == before  # nothing activated, nothing written


# --------------------------------------------------------------------------- #
# End-to-end happy path
# --------------------------------------------------------------------------- #

def test_full_workflow_end_to_end(tmp_path):
    path = _stage(tmp_path)
    assessment = wf.assess(
        path, source_url="https://example.org/report.pdf", owner="Platform Team",
        permission="cc-by-4.0", intended_use="knowledge_candidate",
        authority_level="official", now=_NOW,
    )
    assert not assessment.blocked

    preview = _preview_dict(path)
    approval = _approval(preview)
    assert wf.validate(preview, approval, pack_id="quarterly-review") is not None
    assert wf.dry_run(preview, approval, pack_id="quarterly-review",
                      now=_IMPORT_NOW).written is False

    result = wf.create_pack(
        preview, approval, pack_id="quarterly-review",
        import_root=tmp_path / "packs", confirm=True, now=_IMPORT_NOW,
    )
    assert result.written is True
    pack_dir = wf.pack_dir_for("quarterly-review", import_root=tmp_path / "packs")

    assert wf.evaluate(pack_dir).summary.case_count > 0
    assert wf.propose_registry(pack_dir, registry_path=tmp_path / "reg.jsonl",
                               now=_IMPORT_NOW).requires_human_approval is True
    assert wf.request_activation(
        pack_dir, approval_id="act-001", approved_by="reviewer@local",
        approved_at="2026-06-05T12:00:00+00:00",
        state_path=tmp_path / "state.jsonl", audit_path=tmp_path / "audit.jsonl",
        now=_IMPORT_NOW,
    ).ok is False
