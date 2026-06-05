"""Usable project workspace — the data-to-answer layer (v7.0).

This module is the thin, user-facing layer that lets a non-technical user run
the whole journey without a CLI, JSON editing, or governance vocabulary:

    create a project -> add a PDF -> process it -> ask a question ->
    get a cited answer -> inspect evidence -> export a report.

It is a *composition* layer only. It invents no new governance, queue,
lifecycle, monitoring, source type, retrieval, ranking, or grounding. Every
governed action is delegated to an existing, already-tested component:

* A **project** is one :class:`agent.project_packs.ProjectPack` (the existing
  pack primitive), created through :class:`agent.project_packs.PackRegistry`.
* **Adding/processing a document** delegates entirely to
  :mod:`agent.pdf_import_workflow` — the single mutation-capable controller.
  The only write is the governed, confirmed pack creation; it never activates a
  pack, never writes memory, and never touches the source registry.
* **Asking** builds a read-only :class:`agent.workbench_service.WorkbenchService`
  over the project's ready documents and runs the governed orchestrator through
  :func:`agent.console_model.answer_ask`. Memory is *not* in scope for Ask, so
  only imported knowledge is citable — exactly as the rest of the product.

Friendly statuses (Ready / Needs review / Blocked / Failed) are a pure
projection of the existing intake *decision* and import result. They add no new
decision logic: a document the governance pipeline does not approve for
knowledge is never auto-imported and never becomes answerable.

Heavy imports are deferred so importing this module stays cheap and side-effect
free, matching :mod:`agent.pdf_import_workflow`.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]

PROJECT_WORKSPACE_VERSION = "project-workspace-v7.0"

# Where the live UI stores user projects. Runtime artefacts, not committed.
DEFAULT_PROJECTS_ROOT = ROOT / "packs" / "projects"

# Per-project index of uploaded documents and their friendly status. This is a
# plain projection of governed results, not a new state store.
_DOCUMENTS_FILE = "documents.jsonl"
# Sub-directory under a project holding the governed per-document packs.
_DOCUMENTS_DIR = "documents"


# --------------------------------------------------------------------------- #
# Friendly status vocabulary (plain language; no governance jargon)
# --------------------------------------------------------------------------- #

STATUS_UPLOADING = "uploading"
STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_BLOCKED = "blocked"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class DocumentStatus:
    key: str
    label: str
    available_for_answers: bool
    blurb: str


_DOCUMENT_STATUSES: Tuple[DocumentStatus, ...] = (
    DocumentStatus(STATUS_UPLOADING, "Uploading", False,
                   "The file is being received."),
    DocumentStatus(STATUS_PROCESSING, "Processing", False,
                   "The document is being checked and prepared."),
    DocumentStatus(STATUS_READY, "Available for answers", True,
                   "The document is ready and its content can be cited in answers."),
    DocumentStatus(STATUS_NEEDS_REVIEW, "Needs review", False,
                   "Something needs a person to confirm before this document can "
                   "be used — usually a missing source or permission."),
    DocumentStatus(STATUS_BLOCKED, "Blocked", False,
                   "This document cannot be used as it is (for example it is "
                   "encrypted, scanned without text, or restricted)."),
    DocumentStatus(STATUS_FAILED, "Could not process", False,
                   "Processing did not complete. The document was not added."),
)

DOCUMENT_STATUS_BY_KEY = {s.key: s for s in _DOCUMENT_STATUSES}


def status_label(key: str) -> str:
    status = DOCUMENT_STATUS_BY_KEY.get(key)
    return status.label if status else key


def status_is_available(key: str) -> bool:
    status = DOCUMENT_STATUS_BY_KEY.get(key)
    return bool(status and status.available_for_answers)


# --------------------------------------------------------------------------- #
# Views (plain data the UI renders directly)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ProjectView:
    project_id: str
    name: str
    description: str
    created_at: str
    document_count: int
    ready_count: int

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "document_count": self.document_count,
            "ready_count": self.ready_count,
        }


@dataclass(frozen=True)
class DocumentView:
    doc_id: str
    filename: str
    pack_id: str
    status: str
    status_label: str
    available_for_answers: bool
    decision: str
    reason: str
    chunk_count: int
    created_at: str
    findings: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "pack_id": self.pack_id,
            "status": self.status,
            "status_label": self.status_label,
            "available_for_answers": self.available_for_answers,
            "decision": self.decision,
            "reason": self.reason,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at,
            "findings": list(self.findings),
        }


@dataclass(frozen=True)
class ProjectAnswer:
    """A project-scoped answer plus its read-only evidence projections."""

    query: str
    result: Any          # raw orchestrator result (read-only)
    view: Any            # console_model.AskView
    inspector: Any       # console_model.EvidenceInspectorView
    citations: Tuple[Any, ...]


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #

class ProjectWorkspace:
    """A thin, governed facade over projects and their documents."""

    def __init__(self, root: Optional[str | Path] = None):
        from agent.project_packs import PackRegistry

        self.root = Path(root) if root is not None else DEFAULT_PROJECTS_ROOT
        self._registry = PackRegistry(self.root)

    # -- projects ---------------------------------------------------------

    def create_project(self, name: str, *, description: str = "") -> ProjectView:
        """Create a new project. Reuses the governed pack primitive."""
        pack = self._registry.create_pack(name, description=description)
        (pack.root_path / _DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)
        return self._project_view(pack)

    def list_projects(self) -> List[ProjectView]:
        return [self._project_view(p) for p in self._registry.list_packs()]

    def get_project(self, project_id: str) -> Optional[ProjectView]:
        pack = self._registry.get_pack(project_id)
        return self._project_view(pack) if pack is not None else None

    def _project_view(self, pack) -> ProjectView:
        docs = self._read_documents(pack)
        ready = sum(1 for d in docs if d.available_for_answers)
        return ProjectView(
            project_id=pack.pack_id,
            name=pack.name,
            description=pack.description,
            created_at=pack.created_at,
            document_count=len(docs),
            ready_count=ready,
        )

    def _require_pack(self, project_id: str):
        pack = self._registry.get_pack(project_id)
        if pack is None:
            raise KeyError(f"no such project: {project_id!r}")
        return pack

    # -- documents --------------------------------------------------------

    def add_document(self, project_id: str, data: bytes, *, filename: str,
                     source_url: str = "", owner: str = "", permission: str = "",
                     intended_use: str = "knowledge_candidate",
                     authority_level: str = "", domain: str = "general",
                     intake_mode: str = "local_project_document",
                     owner_permission_declared: bool = False,
                     now=None) -> DocumentView:
        """Add and process one PDF through the governed pipeline.

        Routine path (the document is approved for knowledge): the document is
        previewed, every previewed chunk is approved, and the governed pack is
        created. The document becomes *Available for answers*.

        Exception path (not approved for knowledge): nothing is imported. The
        document is recorded as *Needs review* or *Blocked* with a plain reason.
        Genuinely risky documents are never processed automatically.

        Uploads default to ``intake_mode="local_project_document"``: a user's own
        project files are usable by default (missing provenance, missing source
        URL, or stale metadata are warnings, not blockers). Content/safety
        blockers — encryption, no text layer, scanned-only, poor extraction, PII,
        confidential/restricted markers — are unchanged.
        """
        from agent import pdf_import_workflow as wf

        pack = self._require_pack(project_id)
        doc_id = self._unique_doc_id(pack, filename)

        check = wf.validate_upload(data, filename=filename)
        if not check.ok:
            record = self._make_record(
                doc_id=doc_id, filename=filename, pack_id="",
                status=STATUS_BLOCKED, decision="rejected", reason=check.reason,
                chunk_count=0, findings=(), now=now)
            self._append_document(pack, record)
            return self._to_view(record)

        with tempfile.TemporaryDirectory() as staging:
            try:
                path = wf.stage_upload(data, filename=filename, workspace=staging)
                assessment = wf.assess(
                    path, source_url=source_url, owner=owner,
                    permission=permission, intended_use=intended_use,
                    authority_level=authority_level, intake_mode=intake_mode,
                    owner_permission_declared=owner_permission_declared, now=now)
            except Exception as exc:  # defensive: malformed/unreadable PDF
                record = self._make_record(
                    doc_id=doc_id, filename=filename, pack_id="",
                    status=STATUS_FAILED, decision="failed",
                    reason=f"The document could not be read ({exc}).",
                    chunk_count=0, findings=(), now=now)
                self._append_document(pack, record)
                return self._to_view(record)

            findings = tuple(f.message for f in assessment.findings)

            if not assessment.approved_for_knowledge:
                status, decision = self._exception_status(assessment)
                record = self._make_record(
                    doc_id=doc_id, filename=filename, pack_id="",
                    status=status, decision=decision,
                    reason=assessment.rationale or status_label(status),
                    chunk_count=0, findings=findings, now=now)
                self._append_document(pack, record)
                return self._to_view(record)

            # Routine path: approved for knowledge -> create the governed pack.
            record_now = _now_iso(now)
            try:
                preview = wf.preview(
                    path, intake_result=assessment, source_url=source_url,
                    owner=owner, permission=permission,
                    intended_use=intended_use, authority_level=authority_level,
                    now=now)
                preview_dict = wf.preview_to_dict(preview)
                approved_ids = [c["chunk_id"] for c in preview_dict["chunks"]]
                pack_id = doc_id
                approval = wf.build_approval(
                    preview_dict, approval_id=f"appr-{doc_id}",
                    approved_by="project_owner",
                    approved_at=record_now,
                    approved_chunk_ids=approved_ids,
                    intended_pack_id=pack_id, intended_use=intended_use,
                    source_title=filename, authority_level=authority_level,
                    provenance=source_url, permission_or_licence=permission,
                    domain=domain)
                import_root = pack.root_path / _DOCUMENTS_DIR
                result = wf.create_pack(
                    preview_dict, approval, pack_id=pack_id,
                    import_root=import_root, confirm=True,
                    description=f"Imported from {filename}", now=now)
            except Exception as exc:  # defensive
                record = self._make_record(
                    doc_id=doc_id, filename=filename, pack_id="",
                    status=STATUS_FAILED, decision="failed",
                    reason=f"The document could not be processed ({exc}).",
                    chunk_count=0, findings=findings, now=now)
                self._append_document(pack, record)
                return self._to_view(record)

            chunk_count = len(result.imported_chunks)
            ready = bool(result.written and chunk_count > 0)
            record = self._make_record(
                doc_id=doc_id, filename=filename,
                pack_id=pack_id if ready else "",
                status=STATUS_READY if ready else STATUS_NEEDS_REVIEW,
                decision=assessment.decision.value,
                reason=("Ready." if ready else
                        "No usable passages were extracted; a person should check "
                        "the document."),
                chunk_count=chunk_count, findings=findings,
                created_at=record_now, now=now)
            self._append_document(pack, record)
            return self._to_view(record)

    def list_documents(self, project_id: str) -> List[DocumentView]:
        pack = self._require_pack(project_id)
        return [self._to_view(r) for r in self._read_documents_raw(pack)]

    # -- ask --------------------------------------------------------------

    def ask(self, project_id: str, query: str, *,
            registry_path: Optional[Path] = None) -> ProjectAnswer:
        """Answer a question using only this project's ready documents.

        Read-only: builds a throwaway service over the merged ready-document
        knowledge and runs the governed orchestrator. Writes nothing, creates no
        memory, activates nothing.
        """
        from agent import console_model as cm

        pack = self._require_pack(project_id)
        with tempfile.TemporaryDirectory() as workdir:
            service = self._project_view_service(pack, Path(workdir))
            result = cm.answer_ask(service, query, registry_path=registry_path)
            view = cm.build_ask(result)
            inspector = cm.build_evidence_inspector(result)
            citations = tuple(cm.build_citation_details(result))
        return ProjectAnswer(query=query, result=result, view=view,
                             inspector=inspector, citations=citations)

    def render_report(self, project_id: str, answer: ProjectAnswer, *,
                      generated_at: str = "") -> str:
        """Render a cited, governed Markdown report for an answer."""
        from agent import console_model as cm

        project = self.get_project(project_id)
        title = project.name if project else project_id
        header = (f"# {title} — answer report\n\n"
                  f"**Question:** {answer.query}\n\n")
        body = cm.render_ask_markdown(
            answer.view, citations=list(answer.citations),
            inspector=answer.inspector, generated_at=generated_at)
        return header + body

    def export_report(self, project_id: str, answer: ProjectAnswer, *,
                      out_dir: str | Path, filename: str = "answer_report.md",
                      generated_at: str = "") -> Path:
        """Write a report to a file the user chose. User-initiated write only."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        target = out / filename
        target.write_text(
            self.render_report(project_id, answer, generated_at=generated_at),
            encoding="utf-8")
        return target

    # -- internals --------------------------------------------------------

    def _project_view_service(self, pack, workdir: Path):
        """Build a read-only service over the project's ready documents."""
        from agent.workbench_service import WorkbenchService
        from retrieval.embedding_backend import OfflineHashingEmbedder

        merged = workdir / "project_knowledge.jsonl"
        docs_dir = pack.root_path / _DOCUMENTS_DIR
        lines: List[str] = []
        for record in self._read_documents_raw(pack):
            if not record.get("available_for_answers"):
                continue
            pack_id = record.get("pack_id") or ""
            knowledge = docs_dir / pack_id / "knowledge.jsonl"
            if not knowledge.exists():
                continue
            for line in knowledge.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    lines.append(line)
        merged.write_text("\n".join(lines) + ("\n" if lines else ""),
                          encoding="utf-8")
        return WorkbenchService(
            knowledge_path=str(merged), knowledge_backend="hybrid",
            semantic_embedder=OfflineHashingEmbedder(), fresh=False)

    @staticmethod
    def _exception_status(assessment) -> Tuple[str, str]:
        if assessment.blocked or assessment.quarantined:
            return STATUS_BLOCKED, assessment.decision.value
        return STATUS_NEEDS_REVIEW, assessment.decision.value

    def _unique_doc_id(self, pack, filename: str) -> str:
        from agent.pdf_import_workflow import normalize_pack_id

        stem = Path(filename or "document").stem or "document"
        base = normalize_pack_id(stem)
        existing = {r.get("doc_id") for r in self._read_documents_raw(pack)}
        if base not in existing:
            return base
        suffix = 2
        while f"{base}-{suffix}" in existing:
            suffix += 1
        return f"{base}-{suffix}"

    @staticmethod
    def _make_record(*, doc_id: str, filename: str, pack_id: str, status: str,
                     decision: str, reason: str, chunk_count: int,
                     findings: Tuple[str, ...], created_at: str = "",
                     now=None) -> dict:
        return {
            "_record": "project_document",
            "doc_id": doc_id,
            "filename": filename,
            "pack_id": pack_id,
            "status": status,
            "decision": decision,
            "reason": reason,
            "chunk_count": int(chunk_count),
            "available_for_answers": status_is_available(status),
            "findings": list(findings),
            "created_at": created_at or _now_iso(now),
        }

    @staticmethod
    def _to_view(record: dict) -> DocumentView:
        status = str(record.get("status", STATUS_FAILED))
        return DocumentView(
            doc_id=str(record.get("doc_id", "")),
            filename=str(record.get("filename", "")),
            pack_id=str(record.get("pack_id", "")),
            status=status,
            status_label=status_label(status),
            available_for_answers=bool(record.get("available_for_answers")),
            decision=str(record.get("decision", "")),
            reason=str(record.get("reason", "")),
            chunk_count=int(record.get("chunk_count", 0)),
            created_at=str(record.get("created_at", "")),
            findings=tuple(record.get("findings", []) or ()),
        )

    def _documents_path(self, pack) -> Path:
        return pack.root_path / _DOCUMENTS_FILE

    def _read_documents_raw(self, pack) -> List[dict]:
        path = self._documents_path(pack)
        if not path.exists():
            return []
        out: List[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # skip a corrupt line, keep the rest
                continue
        return out

    def _read_documents(self, pack) -> List[DocumentView]:
        return [self._to_view(r) for r in self._read_documents_raw(pack)]

    def _append_document(self, pack, record: dict) -> None:
        path = self._documents_path(pack)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def _now_iso(now=None) -> str:
    from datetime import datetime, timezone

    if now is None:
        now = datetime.now(timezone.utc)
    return now.isoformat()
