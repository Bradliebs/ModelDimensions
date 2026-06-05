"""Tests for the v6.8 imported-PDF-pack -> source registry proposal (proposal-only).

This layer reads an imported pack's ``manifest.json`` (written by the v6.6
importer) and proposes a :class:`SourceRegistryEntry` for human review. It is
strictly a proposal: it requires human approval, applies nothing, writes no
registry/memory/pack/queue, and changes no retrieval/ranking/grounding behaviour.

The tests cover crafted manifests (clean/active, draft-forcing gaps, duplicate
conflict, unknown authority, excluded chunks), a real end-to-end manifest written
by the v6.6 importer (schema-drift guard), determinism, the ready-to-paste
registry record contract, the only writer, CLI behaviour, and import-purity +
no-network + no-registry-write safety.
"""
from __future__ import annotations

import ast
import json
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import pdf_pack_registry_proposal as ppr  # noqa: E402
from agent import source_registry as sr  # noqa: E402
from agent.pdf_chunk_preview import preview_pdf_chunks  # noqa: E402
from agent.pdf_pack_importer import (  # noqa: E402
    PdfImportApproval,
    PdfImportRequest,
    compute_preview_fingerprint,
    import_pdf_pack,
)
from agent.source_registry import (  # noqa: E402
    AuthorityLevel,
    SourceRegistryEntry,
    SourceStatus,
)

_README = ROOT / "README.md"
_DEMO_MANIFEST = ROOT / "demos" / "pdf_pack_manifest_example.json"
_DEMO_REGISTRY = ROOT / "demos" / "source_registry.jsonl"
_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
_IMPORT_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
_PREVIEW_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Crafted-manifest helpers
# --------------------------------------------------------------------------- #


def _clean_manifest(**overrides) -> dict:
    """A complete, clean v6.6 manifest (proposes an active source)."""
    manifest = {
        "_record": "pdf_knowledge_pack_manifest",
        "pack_id": "policy-handbook",
        "name": "Policy Handbook",
        "description": "PDF knowledge pack imported from the policy handbook",
        "created_at": "2026-05-30T09:15:00+00:00",
        "default_knowledge_backend": "deterministic",
        "default_domain": "general",
        "settings": {},
        "pack_version": "1.0",
        "source_count": 1,
        "chunk_count": 8,
        "source_file_hash": "sha256:" + "a" * 64,
        "preview_fingerprint": "pdfprev-0123456789abcdef",
        "approval_id": "pdfappr-policy-handbook",
        "approval_actor": "reviewer@local",
        "approval_timestamp": "2026-05-30T09:14:00+00:00",
        "import_timestamp": "2026-05-30T09:15:00+00:00",
        "intended_use": "knowledge",
        "authority_level": "reputable",
        "provenance": "internal governance library",
        "permission_or_licence": "internal use approved",
        "excluded_chunk_ids": [],
        "unresolved_nonblocking_findings": [],
        "importer_version": "v6.6",
        "manifest_hash": "pdfpack-0123456789abcdef",
    }
    manifest.update(overrides)
    return manifest


def _codes(proposal) -> set[str]:
    return {f.code for f in proposal.findings}


# --------------------------------------------------------------------------- #
# Real v6.6 import fixture (schema-drift guard)
# --------------------------------------------------------------------------- #


def _escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace("(", "\\(")
            .replace(")", "\\)").replace("\n", "\\n"))


def _text_stream(text: str) -> str:
    return "\n".join(["BT", "/F1 12 Tf", "72 720 Td",
                      f"({_escape(text)}) Tj", "ET"])


def _make_pdf(pages, *, title) -> bytes:
    n_pages = len(pages)
    content_start = 3
    page_start = content_start + n_pages
    info_num = page_start + n_pages
    page_nums = [page_start + i for i in range(n_pages)]
    chunks = [b"%PDF-1.4\n"]

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
    obj(info_num, f"<< /Title ({title}) /Author (Team) "
                  "/Creator (ModelDimensions) /Producer (ModelDimensions) "
                  "/CreationDate (D:20250101000000Z) >>")
    chunks.append(
        f"trailer\n<< /Root 1 0 R /Info {info_num} 0 R >>\n".encode("latin-1"))
    chunks.append(b"%%EOF\n")
    return b"".join(chunks)


_PAGES = [
    ("The platform reliability budget ceiling is fixed at 4200 error-minutes "
     "for every quarter and the capacity planning team watches throughput and "
     "latency against that ceiling. Each service owner confirms the 4200 "
     "error-minute reliability budget before approving any production rollout."),
    ("The named accountable owner for capacity planning is Dana Okafor, who "
     "chairs the weekly operating review and signs off the agreed follow up "
     "actions. Dana Okafor is recorded as the single accountable owner for the "
     "platform reliability programme for the current quarter."),
]


def _import_real_pack(tmp_path: Path, *, pack_id: str = "quarterly_review",
                      source_title: str = "Quarterly Operating Review") -> Path:
    pdf_path = tmp_path / f"{pack_id}.pdf"
    pdf_path.write_bytes(_make_pdf(_PAGES, title=source_title))
    preview = preview_pdf_chunks(
        pdf_path, source_url="https://example.org/report.pdf",
        owner="Platform Team", permission="cc-by-4.0",
        intended_use="knowledge_candidate", authority_level="official",
        now=_PREVIEW_NOW).to_dict(include_full_text=True)
    chunk_ids = [c["chunk_id"] for c in preview["chunks"]]
    approval = PdfImportApproval(
        approval_id=f"appr-{pack_id}", approved_by="reviewer@local",
        approved_at="2026-06-04T00:00:00+00:00",
        source_file_hash=preview["file_hash"],
        preview_fingerprint=compute_preview_fingerprint(preview),
        approved_chunk_ids=tuple(chunk_ids),
        intended_pack_id=pack_id, intended_use="knowledge_candidate",
        source_title=source_title, authority_level="official",
        provenance="declared by document owner",
        permission_or_licence="cc-by-4.0", domain="general")
    pack_dir = tmp_path / "packs" / pack_id
    result = import_pdf_pack(
        PdfImportRequest(preview=preview, approval=approval, pack_id=pack_id,
                         pack_dir=str(pack_dir), write=True),
        now=_IMPORT_NOW)
    assert result.written, result.status
    return pack_dir


# --------------------------------------------------------------------------- #
# 1. Clean manifest -> active proposal, approval-gated.
# --------------------------------------------------------------------------- #


def test_clean_manifest_proposes_active_source():
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(), now=_NOW)
    assert proposal.decision == ppr.ProposalDecision.PROPOSE_NEW_ACTIVE
    assert proposal.proposed_entry.status is SourceStatus.ACTIVE
    assert proposal.requires_human_approval is True
    assert proposal.status == "proposed"
    assert _codes(proposal) == {ppr.ProposalFindingCode.COMPLETE_LINEAGE}


# 2. Missing provenance forces a draft proposal. -----------------------------


def test_missing_provenance_forces_draft():
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(provenance=""), now=_NOW)
    assert proposal.decision == ppr.ProposalDecision.PROPOSE_NEW_DRAFT
    assert proposal.proposed_entry.status is SourceStatus.DRAFT
    assert ppr.ProposalFindingCode.MISSING_PROVENANCE in _codes(proposal)


# 3. Missing permission/licence forces a draft proposal. ---------------------


def test_missing_permission_forces_draft():
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(permission_or_licence=""), now=_NOW)
    assert proposal.decision == ppr.ProposalDecision.PROPOSE_NEW_DRAFT
    assert proposal.proposed_entry.status is SourceStatus.DRAFT
    assert ppr.ProposalFindingCode.MISSING_PERMISSION in _codes(proposal)


# 4. Eval-only intended use forces a draft proposal. -------------------------


def test_eval_only_intent_forces_draft():
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(intended_use="eval_only"), now=_NOW)
    assert proposal.decision == ppr.ProposalDecision.PROPOSE_NEW_DRAFT
    assert proposal.proposed_entry.status is SourceStatus.DRAFT
    assert ppr.ProposalFindingCode.EVAL_ONLY_INTENT in _codes(proposal)


# 5. Unknown authority is mapped to UNKNOWN and flagged (info, not draft). ----


def test_unknown_authority_mapped_and_flagged():
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(authority_level="vendor-blog"), now=_NOW)
    assert proposal.proposed_entry.authority_level is AuthorityLevel.UNKNOWN
    assert ppr.ProposalFindingCode.UNKNOWN_PACK_AUTHORITY in _codes(proposal)
    # Unknown authority alone is informational; the pack stays proposable active.
    assert proposal.decision == ppr.ProposalDecision.PROPOSE_NEW_ACTIVE


# 6. Excluded chunks and unresolved findings are surfaced. -------------------


def test_excluded_chunks_and_unresolved_findings_surfaced():
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(excluded_chunk_ids=["chk-3"],
                        unresolved_nonblocking_findings=["sparse chunk"]),
        now=_NOW)
    codes = _codes(proposal)
    assert ppr.ProposalFindingCode.EXCLUDED_CHUNKS_PRESENT in codes
    assert ppr.ProposalFindingCode.UNRESOLVED_IMPORT_FINDINGS in codes
    # The unresolved (warning) finding forces a draft.
    assert proposal.proposed_entry.status is SourceStatus.DRAFT


# 7. Duplicate source_id against an existing registry -> conflict, not add. ---


def test_duplicate_source_id_flags_conflict():
    existing = [SourceRegistryEntry(source_id="policy-handbook")]
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(), existing_entries=existing, now=_NOW)
    assert proposal.decision == ppr.ProposalDecision.CONFLICT_EXISTING_SOURCE
    assert proposal.has_conflict is True
    assert ppr.ProposalFindingCode.DUPLICATE_SOURCE_ID in _codes(proposal)
    # A conflicting pack is never proposed active.
    assert proposal.proposed_entry.status is SourceStatus.DRAFT


# 8. No duplicate when the id is absent from the registry. -------------------


def test_no_conflict_when_source_id_absent():
    existing = [SourceRegistryEntry(source_id="something-else")]
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(), existing_entries=existing, now=_NOW)
    assert proposal.has_conflict is False
    assert ppr.ProposalFindingCode.DUPLICATE_SOURCE_ID not in _codes(proposal)


# 9. Proposed entry is a valid, ready-to-paste registry record. --------------


def test_proposed_entry_round_trips_through_registry_model():
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(), now=_NOW)
    record = proposal.proposed_entry.to_dict()
    assert record["_record"] == "source_registry_entry"
    # The proposed record loads cleanly back into the registry model.
    reloaded = SourceRegistryEntry.from_dict(record)
    assert reloaded == proposal.proposed_entry


# 10. Lineage fields are carried verbatim from the manifest. -----------------


def test_lineage_fields_carried_from_manifest():
    manifest = _clean_manifest()
    proposal = ppr.propose_registry_entry_from_manifest(manifest, now=_NOW)
    assert proposal.manifest_hash == manifest["manifest_hash"]
    assert proposal.source_file_hash == manifest["source_file_hash"]
    assert proposal.preview_fingerprint == manifest["preview_fingerprint"]
    assert proposal.approval_id == manifest["approval_id"]
    assert proposal.proposed_entry.owner == manifest["approval_actor"]
    # The lineage note records the audit trail in the entry's notes.
    assert manifest["approval_id"] in proposal.proposed_entry.notes


# 11. proposal_id is deterministic and content-derived. ----------------------


def test_proposal_id_is_deterministic_and_content_derived():
    first = ppr.propose_registry_entry_from_manifest(_clean_manifest(), now=_NOW)
    second = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(), now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    # Same content identity -> same id regardless of when proposed.
    assert first.proposal_id == second.proposal_id
    assert first.proposal_id.startswith("pdfregprop-")
    other = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(manifest_hash="pdfpack-ffffffffffffffff"), now=_NOW)
    assert other.proposal_id != first.proposal_id


# 12. Approval gating holds across every decision path. ----------------------


@pytest.mark.parametrize("overrides", [
    {},
    {"provenance": ""},
    {"intended_use": "eval_only"},
    {"authority_level": "blog"},
])
def test_always_requires_approval_and_proposed_status(overrides):
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(**overrides), now=_NOW)
    assert proposal.requires_human_approval is True
    assert proposal.status == "proposed"


# 13. Findings are sorted error-first then by code. --------------------------


def test_findings_sorted_error_first():
    existing = [SourceRegistryEntry(source_id="policy-handbook")]
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(provenance="", permission_or_licence=""),
        existing_entries=existing, now=_NOW)
    ranks = [ppr._SEVERITY_RANK[f.severity] for f in proposal.findings]
    assert ranks == sorted(ranks)
    assert proposal.findings[0].code == ppr.ProposalFindingCode.DUPLICATE_SOURCE_ID


# 14. load_pack_manifest reads a file and a directory, and validates record. -


def test_load_pack_manifest_file_dir_and_validation(tmp_path):
    manifest = _clean_manifest()
    # From a file.
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(manifest), encoding="utf-8")
    assert ppr.load_pack_manifest(mf)["pack_id"] == "policy-handbook"
    # From a directory (reads manifest.json inside).
    assert ppr.load_pack_manifest(tmp_path)["pack_id"] == "policy-handbook"
    # Wrong record type fails cleanly.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"_record": "something_else"}), encoding="utf-8")
    with pytest.raises(ValueError):
        ppr.load_pack_manifest(bad)


# 15. A manifest with no pack_id fails cleanly. ------------------------------


def test_missing_pack_id_raises():
    manifest = _clean_manifest()
    manifest.pop("pack_id")
    with pytest.raises(ValueError):
        ppr.propose_registry_entry_from_manifest(manifest, now=_NOW)


# 16. The proposal serialises deterministically. -----------------------------


def test_proposal_to_dict_and_json_deterministic():
    a = ppr.propose_registry_entry_from_manifest(_clean_manifest(), now=_NOW)
    b = ppr.propose_registry_entry_from_manifest(_clean_manifest(), now=_NOW)
    assert a.to_dict() == b.to_dict()
    assert ppr.registry_proposal_to_json(a) == ppr.registry_proposal_to_json(b)
    assert a.to_dict()["_record"] == "pdf_pack_registry_proposal"


# 17. The markdown render is deterministic and proposal-framed. --------------


def test_render_markdown_is_deterministic_and_framed():
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(), now=_NOW)
    first = ppr.render_registry_proposal_markdown(proposal)
    second = ppr.render_registry_proposal_markdown(proposal)
    assert first == second
    assert "proposal-only" in first
    assert "Ready-to-paste registry record" in first
    assert proposal.decision in first


# 18. write_registry_proposal writes ONLY the given file. --------------------


def test_write_registry_proposal_writes_only_target(tmp_path):
    proposal = ppr.propose_registry_entry_from_manifest(
        _clean_manifest(), now=_NOW)
    out = tmp_path / "nested" / "proposal.json"
    ppr.write_registry_proposal(proposal, out)
    # Only the requested file exists under the tree.
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written == [out]
    # And it round-trips to the same dict.
    assert json.loads(out.read_text(encoding="utf-8")) == proposal.to_dict()


# 19. Real v6.6 import -> proposal reads the real manifest (schema guard). ----


def test_real_imported_pack_manifest_is_consumed(tmp_path):
    pack_dir = _import_real_pack(tmp_path)
    on_disk = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    proposal = ppr.propose_registry_entry_from_pack(pack_dir, now=_NOW)
    # The proposal's lineage matches the manifest the importer actually wrote.
    assert proposal.pack_id == on_disk["pack_id"]
    assert proposal.manifest_hash == on_disk["manifest_hash"]
    assert proposal.source_file_hash == on_disk["source_file_hash"]
    assert proposal.preview_fingerprint == on_disk["preview_fingerprint"]
    assert proposal.approval_id == on_disk["approval_id"]
    # A clean import (declared authority/provenance/permission) proposes active.
    assert proposal.decision == ppr.ProposalDecision.PROPOSE_NEW_ACTIVE
    assert proposal.proposed_entry.authority_level is AuthorityLevel.OFFICIAL


# 20. Real pack against a registry already holding the id -> conflict. -------


def test_real_pack_conflict_against_registry(tmp_path):
    pack_dir = _import_real_pack(tmp_path)
    registry = tmp_path / "registry.jsonl"
    sr.save_registry([SourceRegistryEntry(source_id="quarterly_review")],
                     registry)
    proposal = ppr.propose_registry_entry_from_pack(
        pack_dir, registry_path=registry, now=_NOW)
    assert proposal.has_conflict is True


# 21. Proposing never writes the source registry. ----------------------------


def test_proposing_does_not_write_registry(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):  # pragma: no cover - only fires on a violation
        raise AssertionError("proposal layer must never call save_registry")

    monkeypatch.setattr(sr, "save_registry", _boom)
    pack_dir = _import_real_pack(tmp_path)
    registry = tmp_path / "registry.jsonl"
    # Build a registry without the spy interfering, then re-arm the spy.
    monkeypatch.undo()
    sr.save_registry([SourceRegistryEntry(source_id="other")], registry)
    before = registry.read_bytes()
    monkeypatch.setattr(sr, "save_registry", _boom)
    ppr.propose_registry_entry_from_pack(pack_dir, registry_path=registry,
                                         now=_NOW)
    assert registry.read_bytes() == before


# 22. The demo registry is never modified by a conflict-checked proposal. ----


def test_demo_registry_untouched_by_proposal():
    before = _DEMO_REGISTRY.read_bytes()
    ppr.propose_registry_entry_from_pack(
        _DEMO_MANIFEST, registry_path=_DEMO_REGISTRY, now=_NOW)
    assert _DEMO_REGISTRY.read_bytes() == before


# 23. The demo manifest is well-formed and proposes an active source. --------


def test_demo_manifest_proposes_active():
    proposal = ppr.propose_registry_entry_from_pack(_DEMO_MANIFEST, now=_NOW)
    assert proposal.decision == ppr.ProposalDecision.PROPOSE_NEW_ACTIVE
    assert proposal.proposed_entry.authority_level is AuthorityLevel.REPUTABLE


# 24. Module imports no forbidden dependencies (read-only by construction). ---


def test_module_imports_no_forbidden_dependencies():
    source = Path(ppr.__file__).read_text(encoding="utf-8")
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
        "save_registry", "MemoryLedger", "import_pdf_pack", "WorkbenchService",
        "PackRegistry", "agent.memory_ledger", "agent.workbench_service",
        "agent.pdf_pack_importer", "agent.project_packs", "agent.pack_builder",
        "agent.retrieval", "agent.hybrid_retrieval",
        "requests", "urllib", "urllib.request", "httpx", "openai", "anthropic",
        "pytesseract", "pdf2image", "datasets", "huggingface_hub",
    }
    leaked = imported & forbidden
    assert not leaked, f"proposal layer must not import: {leaked}"

    # Strip every docstring (module/class/function) before the token check so a
    # docstring that *documents* the non-call of a forbidden writer does not
    # count as a reference to it.
    body = source
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                body = body.replace(doc, "", 1)
    # Word-boundary matching so legitimate type names (e.g. PdfPackRegistry-
    # Proposal) are not flagged as references to the forbidden PackRegistry.
    for token in ("save_registry", "MemoryLedger", "import_pdf_pack",
                  "WorkbenchService", "PackRegistry", "requests.get", "urlopen",
                  "add_to_index", "load_dataset"):
        assert not re.search(rf"\b{re.escape(token)}\b", body), \
            f"proposal layer must not reference {token!r}"


# 25. The layer opens no socket. ---------------------------------------------


def test_proposal_does_no_network(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):  # pragma: no cover - only fires on a violation
        raise AssertionError("proposal layer attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    proposal = ppr.propose_registry_entry_from_pack(_DEMO_MANIFEST, now=_NOW)
    assert proposal.pack_id == "m365-conditional-access-guide"


# 26. CLI prints markdown and writes nothing without --out. ------------------


def test_cli_stdout_writes_nothing(tmp_path, capsys, monkeypatch):
    import app.workbench as wb

    monkeypatch.chdir(tmp_path)
    rc = wb._source_registry_cli(
        ["propose-from-pack", "--pack", str(_DEMO_MANIFEST)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "proposal-only" in out
    assert not list(tmp_path.rglob("*.json"))


# 27. CLI --out writes only the proposal file; registry untouched. -----------


def test_cli_out_writes_only_proposal(tmp_path, capsys):
    import app.workbench as wb

    before = _DEMO_REGISTRY.read_bytes()
    out = tmp_path / "proposal.json"
    rc = wb._source_registry_cli(
        ["propose-from-pack", "--pack", str(_DEMO_MANIFEST),
         "--registry", str(_DEMO_REGISTRY), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["_record"] == "pdf_pack_registry_proposal"
    assert _DEMO_REGISTRY.read_bytes() == before


# 28. CLI fails cleanly on a missing pack. -----------------------------------


def test_cli_missing_pack_exits_one(tmp_path, capsys):
    import app.workbench as wb

    rc = wb._source_registry_cli(
        ["propose-from-pack", "--pack", str(tmp_path / "nope.json")])
    assert rc == 1
    assert "failed" in capsys.readouterr().err


# 29. README documents the v6.8 slice. ---------------------------------------


def test_readme_documents_v68():
    text = _README.read_text(encoding="utf-8")
    assert "## v6.8" in text
