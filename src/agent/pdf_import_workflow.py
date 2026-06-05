"""Governed PDF import workflow — the Consultant Workbench Imports controller.

This module is the single mutation-capable layer behind the Imports page. It
orchestrates the existing, already-governed PDF pipeline (intake assessment ->
chunk preview -> approval -> validation -> dry-run -> explicit pack creation ->
retrieval evaluation -> registry proposal -> activation request) and shapes the
results for display.

Governance guarantees preserved here:

* Staging an uploaded PDF and assessing, previewing, validating, dry-running,
  evaluating, proposing, and requesting activation **never** write a pack, never
  create memory, never activate a pack, never change retrieval state, and never
  bypass the provenance/permission checks the backend enforces.
* The only call that writes anything is :func:`create_pack` with
  ``confirm=True``; even then it writes **only** pack files (manifest + chunks)
  under the import workspace. It does not activate the pack, touch memory, the
  source registry, or retrieval.

Heavy imports are deferred so that importing this module (and detecting backend
availability) stays cheap and side-effect free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]

PDF_IMPORT_WORKFLOW_VERSION = "consultant-workbench-pdf-workflow-v1.0"

# Where the live UI writes user-created packs. Kept out of the curated demo
# packs; created packs are runtime artefacts and are not committed.
DEFAULT_IMPORT_ROOT = ROOT / "packs" / "pdf_imports"
DEFAULT_EVAL_CASES = ROOT / "demos" / "retrieval_pdf_import_cases.jsonl"
DEFAULT_REGISTRY = ROOT / "demos" / "source_registry.jsonl"

_PACK_ID_RE = re.compile(r"[^a-z0-9_-]+")


# --------------------------------------------------------------------------- #
# Workflow shape (for page navigation / progress display)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WorkflowStep:
    key: str
    label: str
    blurb: str


PDF_WORKFLOW_STEPS: Tuple[WorkflowStep, ...] = (
    WorkflowStep("upload", "1. Upload",
                 "Drag and drop or select a PDF. The file is staged locally; nothing is imported."),
    WorkflowStep("metadata", "2. Source metadata",
                 "Declare provenance, owner, permission/licence, authority, and intended use."),
    WorkflowStep("assess", "3. Intake assessment",
                 "Deterministic, read-only governance assessment of the document."),
    WorkflowStep("quality", "4. Extraction quality",
                 "Text-layer, OCR need, and extraction-quality band for the document."),
    WorkflowStep("findings", "5. Findings & risk",
                 "Provenance, permission, PII, and confidentiality findings to review."),
    WorkflowStep("preview", "6. Chunk preview",
                 "Proposed, preview-only chunks with per-chunk warnings. No import yet."),
    WorkflowStep("exclude", "7. Chunk selection",
                 "Approve a subset of chunks; explicitly exclude the rest."),
    WorkflowStep("approval", "8. Approval scope",
                 "Bind a structured approval to this exact source and preview."),
    WorkflowStep("dry_run", "9. Dry-run import",
                 "Validate and simulate the import. Writes nothing."),
    WorkflowStep("create", "10. Create knowledge pack",
                 "Explicitly write the pack files. Does not activate or change retrieval."),
    WorkflowStep("evaluate", "11. Retrieval evaluation",
                 "Measure retrieval over the created pack, read-only."),
    WorkflowStep("activate", "12. Activation request",
                 "Request activation through governance. Never activates here."),
)


# --------------------------------------------------------------------------- #
# Backend availability
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BackendAvailability:
    available: bool
    reason: str
    missing: Tuple[str, ...] = ()


_REQUIRED_MODULES = (
    "agent.pdf_intake_adapter",
    "agent.pdf_chunk_preview",
    "agent.pdf_pack_importer",
    "agent.pdf_pack_registry_proposal",
    "agent.knowledge_pack_activation",
)


def backend_availability() -> BackendAvailability:
    """Detect whether the governed PDF backend is importable. Pure, no writes."""
    import importlib

    missing = []
    for name in _REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception:  # pragma: no cover - defensive; missing optional deps
            missing.append(name)
    if missing:
        return BackendAvailability(
            available=False,
            reason="The governed PDF backend is not available in this build.",
            missing=tuple(missing),
        )
    return BackendAvailability(
        available=True,
        reason="The governed PDF import backend is available.",
    )


def _require_backend() -> None:
    status = backend_availability()
    if not status.available:
        raise RuntimeError(status.reason)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def normalize_pack_id(text: str) -> str:
    """Coerce free text into a valid pack id (``^[a-z0-9][a-z0-9_-]{1,63}$``)."""
    slug = _PACK_ID_RE.sub("-", (text or "").strip().lower()).strip("-_")
    slug = slug or "pdf-pack"
    if not slug[0].isalnum():
        slug = "p" + slug
    return slug[:64]


def stage_upload(data: bytes, *, filename: str, workspace) -> Path:
    """Write uploaded bytes into a staging workspace and return the path.

    This is local staging of the uploaded file only — it is not governed state,
    creates no pack, and triggers no import.
    """
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    safe = _PACK_ID_RE.sub("_", (filename or "upload.pdf").strip().lower())
    if not safe.endswith(".pdf"):
        safe = safe + ".pdf"
    target = ws / safe
    target.write_bytes(data)
    return target


# --------------------------------------------------------------------------- #
# Step 3-5: assessment (read-only)
# --------------------------------------------------------------------------- #

def assess(path, *, source_url: str = "", owner: str = "", permission: str = "",
           intended_use: str = "", authority_level: str = "", now=None):
    """Assess one staged PDF. Reads the file only; writes nothing."""
    _require_backend()
    from agent.pdf_intake_adapter import assess_pdf_intake

    return assess_pdf_intake(
        path, source_url=source_url, owner=owner, permission=permission,
        intended_use=intended_use, authority_level=authority_level, now=now,
    )


# --------------------------------------------------------------------------- #
# Step 6: chunk preview (read-only)
# --------------------------------------------------------------------------- #

def preview(path, *, intake_result=None, chunk_size: int = 1000,
            overlap: int = 150, max_chunk_size: int = 1500,
            source_url: str = "", owner: str = "", permission: str = "",
            intended_use: str = "", authority_level: str = "", now=None):
    """Build a preview-only chunk preview. Writes nothing."""
    _require_backend()
    from agent.pdf_chunk_preview import preview_pdf_chunks

    return preview_pdf_chunks(
        path, intake_result=intake_result, chunk_size=chunk_size,
        overlap=overlap, max_chunk_size=max_chunk_size,
        source_url=source_url, owner=owner, permission=permission,
        intended_use=intended_use, authority_level=authority_level, now=now,
    )


def preview_to_dict(chunk_preview) -> dict:
    """Serialise a preview with full chunk text (needed for import)."""
    return chunk_preview.to_dict(include_full_text=True)


# --------------------------------------------------------------------------- #
# Step 7-8: approval (pure data; the only authorisation gate)
# --------------------------------------------------------------------------- #

def build_approval(preview_dict: dict, *, approval_id: str, approved_by: str,
                   approved_at: str, approved_chunk_ids: Sequence[str],
                   excluded_chunk_ids: Sequence[str] = (),
                   intended_pack_id: str = "", intended_use: str = "",
                   source_title: str = "", authority_level: str = "",
                   provenance: str = "", permission_or_licence: str = "",
                   domain: str = "general", allow_confidential: bool = False):
    """Construct a structured import approval bound to this exact preview.

    Approval is pure data. It does not import anything; it authorises a later,
    explicit import of exactly the named chunks.
    """
    _require_backend()
    from agent.pdf_pack_importer import (
        DEFAULT_AUTHORITY, PdfImportApproval, compute_preview_fingerprint,
    )

    return PdfImportApproval(
        approval_id=approval_id,
        approved_by=approved_by,
        approved_at=approved_at,
        source_file_hash=str(preview_dict.get("file_hash", "")),
        preview_fingerprint=compute_preview_fingerprint(preview_dict),
        approved_chunk_ids=tuple(approved_chunk_ids),
        rejected_chunk_ids=tuple(excluded_chunk_ids),
        intended_pack_id=intended_pack_id,
        intended_use=intended_use,
        source_title=source_title,
        authority_level=authority_level or DEFAULT_AUTHORITY,
        provenance=provenance,
        permission_or_licence=permission_or_licence,
        domain=domain or "general",
        allow_confidential=allow_confidential,
    )


def validate(preview_dict: dict, approval, *, pack_id: str):
    """Validate an approval against a preview. Performs no writes."""
    _require_backend()
    from agent.pdf_pack_importer import validate_import

    return validate_import(preview_dict, approval, pack_id=pack_id)


# --------------------------------------------------------------------------- #
# Step 9-10: dry-run and explicit pack creation
# --------------------------------------------------------------------------- #

def dry_run(preview_dict: dict, approval, *, pack_id: str, now=None):
    """Simulate the import without writing anything (``write=False``)."""
    _require_backend()
    from agent.pdf_pack_importer import PdfImportRequest, import_pdf_pack

    request = PdfImportRequest(
        preview=preview_dict, approval=approval, pack_id=pack_id,
        pack_dir=None, write=False,
    )
    return import_pdf_pack(request, now=now)


def pack_dir_for(pack_id: str, *, import_root=None) -> Path:
    """The directory a created pack would live in. Does not create it."""
    root = Path(import_root) if import_root is not None else DEFAULT_IMPORT_ROOT
    return root / pack_id


def create_pack(preview_dict: dict, approval, *, pack_id: str,
                import_root=None, confirm: bool = False,
                pack_version: str = "1.0", description: str = "", now=None):
    """Create the knowledge pack on disk — only on explicit ``confirm=True``.

    Without ``confirm`` this is a dry-run that writes nothing. With ``confirm``
    it writes only the pack's ``manifest.json`` and ``knowledge.jsonl`` under the
    import workspace. It never activates the pack, never creates memory, and
    never changes the source registry or retrieval state.
    """
    _require_backend()
    from agent.pdf_pack_importer import PdfImportRequest, import_pdf_pack

    target = pack_dir_for(pack_id, import_root=import_root)
    request = PdfImportRequest(
        preview=preview_dict, approval=approval, pack_id=pack_id,
        pack_dir=str(target) if confirm else None,
        write=bool(confirm), pack_version=pack_version, description=description,
    )
    return import_pdf_pack(request, now=now)


# --------------------------------------------------------------------------- #
# Step 11: retrieval evaluation (read-only)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvaluationOutcome:
    summary: object
    results: Tuple[object, ...]


def evaluate(pack_dir, *, cases_path=None):
    """Evaluate retrieval over a created pack. Read-only measurement.

    Builds a throwaway read-only service over the pack with the offline hashing
    embedder (deterministic, no network, no model download) and runs the
    imported-PDF retrieval cases. Mutates nothing.
    """
    _require_backend()
    from agent.project_packs import PackRegistry
    from agent.retrieval_eval_harness import (
        load_pdf_import_cases, run_pdf_import_eval, summarize_pdf_import_eval,
    )
    from agent.workbench_service import WorkbenchService
    from retrieval.embedding_backend import OfflineHashingEmbedder

    pack_dir = Path(pack_dir)
    registry = PackRegistry(pack_dir.parent)
    pack = registry.get_pack(pack_dir.name)
    if pack is None:
        raise FileNotFoundError(f"no pack found at {pack_dir}")
    service = WorkbenchService.from_pack(
        pack, registry=registry, knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder(),
    )

    cases = load_pdf_import_cases(
        cases_path if cases_path is not None else DEFAULT_EVAL_CASES)
    results = run_pdf_import_eval(service, cases, pack_label=pack_dir.name)
    summary = summarize_pdf_import_eval(results)
    return EvaluationOutcome(summary=summary, results=tuple(results))


# --------------------------------------------------------------------------- #
# Step 12a: source-registry proposal (read-only proposal)
# --------------------------------------------------------------------------- #

def propose_registry(pack_dir, *, registry_path=None, now=None):
    """Propose a source-registry entry for a created pack. Writes nothing."""
    _require_backend()
    from agent.pdf_pack_registry_proposal import propose_registry_entry_from_pack

    # The proposal reads the registry read-only for a duplicate-source check; a
    # missing registry simply means there are no existing entries to compare.
    candidate = registry_path if registry_path is not None else DEFAULT_REGISTRY
    existing = candidate if candidate is not None and Path(candidate).exists() else None
    return propose_registry_entry_from_pack(
        pack_dir, registry_path=existing, now=now)


# --------------------------------------------------------------------------- #
# Step 12b: activation request (governed; never activates)
# --------------------------------------------------------------------------- #

def request_activation(pack_dir, *, approval_id: str, approved_by: str,
                       approved_at: str, environment: str = "default",
                       state_path=None, audit_path=None, now=None):
    """Build and validate an activation request. Never activates.

    Returns the fail-closed validation findings so a human can see exactly what
    governance requires before activation. This call reads the active-state
    manifest read-only and writes nothing.
    """
    _require_backend()
    from agent.knowledge_pack_activation import (
        DEFAULT_AUDIT_PATH, DEFAULT_STATE_PATH, ActivationScope,
        ActivationStateManager, KnowledgePackActivationApproval,
        KnowledgePackActivationRequest, load_pack_identity, validate_activation,
    )

    identity = load_pack_identity(pack_dir)
    approval = KnowledgePackActivationApproval(
        approval_id=approval_id,
        pack_id=identity.pack_id,
        pack_version=identity.pack_version,
        pack_fingerprint=identity.content_fingerprint,
        manifest_hash=identity.manifest_hash,
        approved_by=approved_by,
        approved_at=approved_at,
        approval_scope=ActivationScope.ACTIVATE,
        approved_environment=environment,
    )
    request = KnowledgePackActivationRequest(
        identity=identity, approval=approval, evidence=None,
        environment=environment,
    )
    manager = ActivationStateManager(
        state_path=state_path if state_path is not None else DEFAULT_STATE_PATH,
        audit_path=audit_path if audit_path is not None else DEFAULT_AUDIT_PATH,
    )
    state = manager.load_state()
    return validate_activation(request, state=state, now=now)
