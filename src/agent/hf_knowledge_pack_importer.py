"""Governed Hugging Face knowledge-pack importer — Phase E (v6.9).

This is the *second* write phase of the governed lifecycle, and it writes
**knowledge material only** — an on-disk knowledge pack of retrievable chunks.
It is deliberately separate from the eval-pack importer (Phase D) because
**approval for eval use is not approval for knowledge use**.

Governance honoured here, by construction:

* **Knowledge needs a knowledge approval.** This importer requires a
  *knowledge-intent* request validated against an approval whose scope permits
  knowledge. A knowledge intent under an eval-only approval fails closed — the
  validation layer raises ``KNOWLEDGE_REQUIRES_KNOWLEDGE_APPROVAL``.
* **Approved is not imported.** A valid approval alone writes nothing. The
  caller must pass an explicit ``pack_dir`` and ``write=True``; the default is a
  dry run that builds the pack in memory and writes nothing.
* **Imported is not activated.** Writing a knowledge pack does not register a
  source, mutate the source registry, touch the ``MemoryLedger``, or activate
  anything in retrieval. The pack is an inert artefact on disk; turning it on is
  a separate, separately-governed step that this module cannot perform.
* **PII *and* unsafe content are excluded.** Unlike the eval path, a row whose
  Phase C inspection blocks *either* eval (unsafe) *or* knowledge (PII) is
  dropped. PII caps a row at eval; it never reaches a knowledge pack. This is
  the eval/knowledge asymmetry enforced from the knowledge side.
* **Fail closed.** Invalid approval, wrong intent, an existing target directory,
  or no eligible rows all produce a no-write result.

This module imports no writer beyond its own atomic pack write: it cannot touch
the ``MemoryLedger``, the source registry, a proposal, a retrieval index, the
eval-pack importer, or an LLM. The docstring is explicit so the import-purity
test can strip it before scanning the body.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from agent.hf_content_inspector import HFContentInspection
from agent.hf_import_lifecycle import (
    HFDatasetApproval,
    HFImportIntent,
    HFImportRequest,
    HFImportValidation,
    validate_import_request,
)
from agent.hf_row_normalizer import HFNormalizationProfile, HFNormalizedRow

KNOWLEDGE_IMPORTER_VERSION = "hf-knowledge-pack-v6.9"
DEFAULT_PACK_VERSION = "1.0"
CHUNK_ID_PREFIX = "hfchunk-"
DEFAULT_DOMAIN = "external"
DEFAULT_AUTHORITY = "reference"
DEFAULT_KNOWLEDGE_BACKEND = "deterministic"

_PACK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")

# Human-friendly labels for composing a chunk's text from its normalized roles.
_ROLE_LABELS: Dict[str, str] = {
    "question": "Question",
    "answer": "Answer",
    "context": "Context",
    "instruction": "Instruction",
    "response": "Response",
    "input": "Input",
    "title": "Title",
    "text": "Text",
    "label": "Label",
    "query": "Query",
    "passage": "Passage",
    "relevance": "Relevance",
    "expected": "Expected",
    "choices": "Choices",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(now: Optional[datetime]) -> str:
    return (now or _utc_now()).isoformat()


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _label(role: str) -> str:
    return _ROLE_LABELS.get(role, role.replace("_", " ").title())


def _compose_chunk_text(row: HFNormalizedRow) -> str:
    """Compose a deterministic labelled chunk body from the row's roles.

    Roles are emitted in the row's own sorted order (``text_items``) so the same
    row always yields the same text. Empty roles are skipped.
    """
    parts = [f"{_label(role)}: {text}" for role, text in row.text_items() if text]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HFKnowledgeChunk:
    """One retrievable knowledge chunk derived from a normalized row."""

    chunk_id: str
    source_id: str
    document_id: str
    chunk_text: str
    domain: str
    authority: str
    source_name: str
    source_section: str
    profile: HFNormalizationProfile
    dataset_id: str
    dataset_revision: str
    split: str
    source_row_id: str
    source_row_key: str
    content_hash: str
    approval_id: str
    approved_by: str
    approved_at: str
    intended_use: str
    provenance: str
    permission_or_licence: str
    import_timestamp: str

    def to_knowledge_record(self) -> dict:
        """Pack-compatible ``chunk`` record plus additive v6.9 lineage fields."""
        return {
            # Existing knowledge-pack chunk schema (retrieval-compatible).
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_text": self.chunk_text,
            "domain": self.domain,
            "authority": self.authority,
            "source_name": self.source_name,
            "source_section": self.source_section,
            "active": True,
            "_record": "chunk",
            # Additive HF import lineage (ignored by existing readers).
            "pack_id": self.source_id,
            "profile": self.profile.value,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "source_row_id": self.source_row_id,
            "source_row_key": self.source_row_key,
            "content_hash": self.content_hash,
            "approval_id": self.approval_id,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "intended_use": self.intended_use,
            "provenance": self.provenance,
            "permission_or_licence": self.permission_or_licence,
            "import_timestamp": self.import_timestamp,
            "importer_version": KNOWLEDGE_IMPORTER_VERSION,
        }

    def to_dict(self) -> dict:
        record = self.to_knowledge_record()
        record["_record"] = "hf_knowledge_chunk"
        return record


def compute_chunk_id(pack_id: str, source_row_id: str) -> str:
    """``hfchunk-<sha256(pack_id|row_id)[:16]>`` — stable per (pack, row)."""
    return CHUNK_ID_PREFIX + _sha256_hex(f"{pack_id}|{source_row_id}")[:16]


@dataclass(frozen=True)
class HFKnowledgePackManifest:
    """The manifest for an imported HF knowledge pack (pack-format compatible)."""

    pack_id: str
    pack_version: str
    name: str
    description: str
    dataset_id: str
    dataset_revision: str
    split: str
    profile: HFNormalizationProfile
    domain: str
    authority: str
    approval_id: str
    approved_by: str
    approved_at: str
    metadata_fingerprint: str
    intake_assessment_fingerprint: str
    provenance: str
    permission_or_licence: str
    chunk_count: int
    excluded_unsafe_count: int
    excluded_pii_count: int
    excluded_empty_count: int
    importer_version: str
    created_at: str
    pack_hash: str

    def to_dict(self) -> dict:
        return {
            "_record": "hf_knowledge_pack_manifest",
            # Pack-compatible keys first so existing tooling can read the pack.
            "pack_id": self.pack_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "default_knowledge_backend": DEFAULT_KNOWLEDGE_BACKEND,
            "default_domain": self.domain,
            "settings": {},
            "pack_kind": "knowledge",
            # Lineage / provenance.
            "pack_version": self.pack_version,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "profile": self.profile.value,
            "authority": self.authority,
            "approval_id": self.approval_id,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "metadata_fingerprint": self.metadata_fingerprint,
            "intake_assessment_fingerprint": self.intake_assessment_fingerprint,
            "provenance": self.provenance,
            "permission_or_licence": self.permission_or_licence,
            "chunk_count": self.chunk_count,
            "excluded_unsafe_count": self.excluded_unsafe_count,
            "excluded_pii_count": self.excluded_pii_count,
            "excluded_empty_count": self.excluded_empty_count,
            "importer_version": self.importer_version,
            "pack_hash": self.pack_hash,
        }


class HFKnowledgeImportStatus(str, Enum):
    """The outcome of a knowledge-pack import (dry-run or write)."""

    IMPORT_READY = "import_ready"  # valid dry run; nothing written
    IMPORT_WRITTEN = "import_written"  # pack written to disk
    IMPORT_BLOCKED = "import_blocked"  # approval/intent invalid
    NO_ELIGIBLE_ROWS = "no_eligible_rows"  # nothing left after filtering
    PACK_EXISTS = "pack_exists"  # target dir already exists
    INVALID_PACK_ID = "invalid_pack_id"


@dataclass(frozen=True)
class HFKnowledgePackImportResult:
    """The full outcome of a knowledge-pack import."""

    status: HFKnowledgeImportStatus
    written: bool
    pack_dir: Optional[str]
    written_paths: Tuple[str, ...]
    chunks: Tuple[HFKnowledgeChunk, ...]
    manifest: Optional[HFKnowledgePackManifest]
    validation: Optional[HFImportValidation]
    message: str
    excluded_unsafe_row_ids: Tuple[str, ...] = ()
    excluded_pii_row_ids: Tuple[str, ...] = ()
    excluded_empty_row_ids: Tuple[str, ...] = ()

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def ok(self) -> bool:
        return self.status in (HFKnowledgeImportStatus.IMPORT_READY,
                               HFKnowledgeImportStatus.IMPORT_WRITTEN)

    def to_dict(self) -> dict:
        return {
            "_record": "hf_knowledge_pack_import_result",
            "status": self.status.value,
            "written": self.written,
            "pack_dir": self.pack_dir,
            "written_paths": list(self.written_paths),
            "chunk_count": self.chunk_count,
            "message": self.message,
            "excluded_unsafe_row_ids": list(self.excluded_unsafe_row_ids),
            "excluded_pii_row_ids": list(self.excluded_pii_row_ids),
            "excluded_empty_row_ids": list(self.excluded_empty_row_ids),
            "validation": self.validation.to_dict() if self.validation else None,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class HFKnowledgePackImportRequest:
    """An explicit request to build (and optionally write) an HF knowledge pack."""

    pack_id: str
    approval: HFDatasetApproval
    request: HFImportRequest
    normalized_rows: Tuple[HFNormalizedRow, ...]
    inspections: Tuple[HFContentInspection, ...] = ()
    pack_dir: Optional[str] = None
    write: bool = False
    pack_name: str = ""
    pack_description: str = ""
    domain: str = DEFAULT_DOMAIN
    authority: str = DEFAULT_AUTHORITY
    metadata: object = None
    assessment: object = None


# --------------------------------------------------------------------------- #
# Building chunks (pure; no writes)
# --------------------------------------------------------------------------- #


def build_chunk(
    row: HFNormalizedRow, *, pack_id: str, domain: str, authority: str,
    approval: HFDatasetApproval,
) -> Optional[HFKnowledgeChunk]:
    """Derive one knowledge chunk from a normalized row, or ``None`` if empty.

    Every profile can yield a knowledge chunk as long as it carries some text;
    the chunk body is the composed, labelled set of normalized roles.
    """
    chunk_text = _compose_chunk_text(row)
    if not chunk_text:
        return None
    return HFKnowledgeChunk(
        chunk_id=compute_chunk_id(pack_id, row.row_id),
        source_id=pack_id,
        document_id=f"{row.dataset_id}@{row.dataset_revision}:{row.split}",
        chunk_text=chunk_text,
        domain=domain,
        authority=authority,
        source_name=row.dataset_id,
        source_section=f"{row.split} #{row.source_row_index}",
        profile=row.profile,
        dataset_id=row.dataset_id,
        dataset_revision=row.dataset_revision,
        split=row.split,
        source_row_id=row.row_id,
        source_row_key=row.source_row_key,
        content_hash=row.content_hash,
        approval_id=approval.approval_id,
        approved_by=approval.approved_by,
        approved_at=approval.approved_at,
        intended_use=approval.approved_intended_use,
        provenance=approval.provenance_snapshot,
        permission_or_licence=approval.licence_snapshot,
        import_timestamp="",  # set by the importer at pack-build time
    )


def _pack_hash(
    pack_id: str,
    approval: HFDatasetApproval,
    request: HFImportRequest,
    chunks: Sequence[HFKnowledgeChunk],
) -> str:
    payload = {
        "pack_id": pack_id,
        "dataset_id": approval.dataset_id,
        "dataset_revision": approval.dataset_revision,
        "split": request.split,
        "approval_id": approval.approval_id,
        "metadata_fingerprint": approval.metadata_fingerprint,
        "intake_assessment_fingerprint": approval.intake_assessment_fingerprint,
        "importer_version": KNOWLEDGE_IMPORTER_VERSION,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "chunk_text": c.chunk_text,
                "content_hash": c.content_hash,
                "source_row_id": c.source_row_id,
            }
            for c in chunks
        ],
    }
    return "hfknowpack-" + _sha256_hex(_canonical_json(payload))[:16]


# --------------------------------------------------------------------------- #
# Atomic pack write
# --------------------------------------------------------------------------- #


def _atomic_write_pack(pack_dir: Path, files: Mapping[str, str]) -> List[str]:
    """Write ``files`` into ``pack_dir`` atomically; never leave a partial pack."""
    parent = pack_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(
        prefix=f".{pack_dir.name}.hfknow-", dir=str(parent)))
    try:
        for name, content in files.items():
            (tmp_dir / name).write_text(content, encoding="utf-8")
        os.rename(tmp_dir, pack_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return [str(pack_dir / name) for name in files]


def _render_pack_files(
    manifest: HFKnowledgePackManifest,
    chunks: Sequence[HFKnowledgeChunk],
) -> Dict[str, str]:
    manifest_json = json.dumps(
        manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
    knowledge_jsonl = "\n".join(
        json.dumps(c.to_knowledge_record(), ensure_ascii=False) for c in chunks)
    if knowledge_jsonl:
        knowledge_jsonl += "\n"
    return {
        "manifest.json": manifest_json + "\n",
        "knowledge.jsonl": knowledge_jsonl,
    }


# --------------------------------------------------------------------------- #
# Top-level import entry point
# --------------------------------------------------------------------------- #


def import_knowledge_pack(
    request: HFKnowledgePackImportRequest, *, now: Optional[datetime] = None,
) -> HFKnowledgePackImportResult:
    """Validate a knowledge-intent request and, only when asked, write the pack.

    Fail-closed order: pack-id shape, then approval/intent validation (a
    knowledge intent under an eval-only approval is rejected), then eligible-row
    filtering (rows that block knowledge — unsafe *or* PII — are excluded, and
    empty rows are dropped). Dry run (``write=False``) builds everything and
    writes nothing; ``write=True`` with a non-existent ``pack_dir`` writes
    atomically.
    """
    created_at = _iso(now)

    if not _PACK_ID_RE.match(request.pack_id or ""):
        return HFKnowledgePackImportResult(
            status=HFKnowledgeImportStatus.INVALID_PACK_ID, written=False,
            pack_dir=None, written_paths=(), chunks=(), manifest=None,
            validation=None,
            message=f"invalid pack id {request.pack_id!r}")

    # Knowledge pack requires a knowledge-intent request.
    if request.request.intent is not HFImportIntent.KNOWLEDGE:
        return HFKnowledgePackImportResult(
            status=HFKnowledgeImportStatus.IMPORT_BLOCKED, written=False,
            pack_dir=None, written_paths=(), chunks=(), manifest=None,
            validation=None,
            message="knowledge-pack import requires a knowledge-intent request")

    validation = validate_import_request(
        request.approval, request.request,
        metadata=request.metadata, assessment=request.assessment, now=now)
    if not validation.valid:
        return HFKnowledgePackImportResult(
            status=HFKnowledgeImportStatus.IMPORT_BLOCKED, written=False,
            pack_dir=None, written_paths=(), chunks=(), manifest=None,
            validation=validation,
            message="; ".join(validation.messages))

    # Filter rows: drop those whose content inspection blocks knowledge.
    # A row that blocks eval (unsafe) is counted as unsafe; a row that blocks
    # knowledge only (PII) is counted as PII. Both are excluded.
    inspections = {i.row_id: i for i in request.inspections}
    excluded_unsafe: List[str] = []
    excluded_pii: List[str] = []
    excluded_empty: List[str] = []
    chunks: List[HFKnowledgeChunk] = []
    for row in request.normalized_rows:
        inspection = inspections.get(row.row_id)
        if inspection is not None:
            if inspection.blocks_eval:
                excluded_unsafe.append(row.row_id)
                continue
            if inspection.blocks_knowledge:
                excluded_pii.append(row.row_id)
                continue
        chunk = build_chunk(
            row, pack_id=request.pack_id, domain=request.domain,
            authority=request.authority, approval=request.approval)
        if chunk is None:
            excluded_empty.append(row.row_id)
            continue
        chunks.append(chunk)

    if not chunks:
        return HFKnowledgePackImportResult(
            status=HFKnowledgeImportStatus.NO_ELIGIBLE_ROWS, written=False,
            pack_dir=None, written_paths=(), chunks=(), manifest=None,
            validation=validation,
            message="no eligible rows remained after filtering",
            excluded_unsafe_row_ids=tuple(excluded_unsafe),
            excluded_pii_row_ids=tuple(excluded_pii),
            excluded_empty_row_ids=tuple(excluded_empty))

    # Stamp the import timestamp onto every chunk now that it is known.
    chunks = [
        HFKnowledgeChunk(**{**c.__dict__, "import_timestamp": created_at})
        for c in chunks
    ]

    profile = chunks[0].profile
    manifest = HFKnowledgePackManifest(
        pack_id=request.pack_id,
        pack_version=DEFAULT_PACK_VERSION,
        name=request.pack_name
        or f"HF knowledge pack: {request.approval.dataset_id}",
        description=request.pack_description
        or f"Governed knowledge pack imported from {request.approval.dataset_id}@"
           f"{request.approval.dataset_revision}.",
        dataset_id=request.approval.dataset_id,
        dataset_revision=request.approval.dataset_revision,
        split=request.request.split,
        profile=profile,
        domain=request.domain,
        authority=request.authority,
        approval_id=request.approval.approval_id,
        approved_by=request.approval.approved_by,
        approved_at=request.approval.approved_at,
        metadata_fingerprint=request.approval.metadata_fingerprint,
        intake_assessment_fingerprint=request.approval.intake_assessment_fingerprint,
        provenance=request.approval.provenance_snapshot,
        permission_or_licence=request.approval.licence_snapshot,
        chunk_count=len(chunks),
        excluded_unsafe_count=len(excluded_unsafe),
        excluded_pii_count=len(excluded_pii),
        excluded_empty_count=len(excluded_empty),
        importer_version=KNOWLEDGE_IMPORTER_VERSION,
        created_at=created_at,
        pack_hash=_pack_hash(request.pack_id, request.approval, request.request,
                             chunks),
    )

    if not request.write:
        return HFKnowledgePackImportResult(
            status=HFKnowledgeImportStatus.IMPORT_READY, written=False,
            pack_dir=request.pack_dir, written_paths=(),
            chunks=tuple(chunks), manifest=manifest, validation=validation,
            message=f"dry run: {len(chunks)} knowledge chunk(s) ready; "
                    "nothing written",
            excluded_unsafe_row_ids=tuple(excluded_unsafe),
            excluded_pii_row_ids=tuple(excluded_pii),
            excluded_empty_row_ids=tuple(excluded_empty))

    if not request.pack_dir:
        return HFKnowledgePackImportResult(
            status=HFKnowledgeImportStatus.IMPORT_BLOCKED, written=False,
            pack_dir=None, written_paths=(), chunks=tuple(chunks),
            manifest=manifest, validation=validation,
            message="write requested but no pack_dir given",
            excluded_unsafe_row_ids=tuple(excluded_unsafe),
            excluded_pii_row_ids=tuple(excluded_pii),
            excluded_empty_row_ids=tuple(excluded_empty))

    pack_dir = Path(request.pack_dir)
    if pack_dir.exists():
        return HFKnowledgePackImportResult(
            status=HFKnowledgeImportStatus.PACK_EXISTS, written=False,
            pack_dir=str(pack_dir), written_paths=(),
            chunks=tuple(chunks), manifest=manifest, validation=validation,
            message=f"target pack dir already exists: {pack_dir}",
            excluded_unsafe_row_ids=tuple(excluded_unsafe),
            excluded_pii_row_ids=tuple(excluded_pii),
            excluded_empty_row_ids=tuple(excluded_empty))

    files = _render_pack_files(manifest, chunks)
    written_paths = _atomic_write_pack(pack_dir, files)
    return HFKnowledgePackImportResult(
        status=HFKnowledgeImportStatus.IMPORT_WRITTEN, written=True,
        pack_dir=str(pack_dir), written_paths=tuple(written_paths),
        chunks=tuple(chunks), manifest=manifest, validation=validation,
        message=f"wrote knowledge pack with {len(chunks)} chunk(s)",
        excluded_unsafe_row_ids=tuple(excluded_unsafe),
        excluded_pii_row_ids=tuple(excluded_pii),
        excluded_empty_row_ids=tuple(excluded_empty))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_knowledge_import_markdown(
    result: HFKnowledgePackImportResult,
) -> str:
    """Deterministic summary. Never prints raw chunk content."""
    manifest = result.manifest
    lines = [
        "# Governed Hugging Face knowledge-pack import (v6.9)",
        "",
        f"- status: **{result.status.value}**",
        f"- written: {result.written}",
        f"- chunks: {result.chunk_count}",
        f"- excluded (unsafe): {len(result.excluded_unsafe_row_ids)}",
        f"- excluded (pii): {len(result.excluded_pii_row_ids)}",
        f"- excluded (empty): {len(result.excluded_empty_row_ids)}",
    ]
    if manifest is not None:
        lines += [
            f"- pack id: `{manifest.pack_id}`",
            f"- dataset: `{manifest.dataset_id}@{manifest.dataset_revision}`",
            f"- split: `{manifest.split}`",
            f"- profile: `{manifest.profile.value}`",
            f"- domain: `{manifest.domain}`",
            f"- authority: `{manifest.authority}`",
            f"- pack hash: `{manifest.pack_hash}`",
        ]
    if result.message:
        lines += ["", f"> {result.message}"]
    return "\n".join(lines) + "\n"
