"""Approved PDF-to-knowledge-pack importer (v6.6).

This module adds a *narrow, explicit, human-approved* path that converts an
already-reviewed v6.5 PDF chunk preview into a deterministic knowledge pack on
disk. It is the controlled pack *write* path and nothing more.

It consumes the structured output of v6.5 (a ``pdf_chunk_preview`` record). It
never re-parses the PDF, never performs OCR, and never calls an LLM. Import is
gated on a structured human approval that is bound to both the source file hash
and a deterministic preview fingerprint; advisory ``preview_ready_for_import``
is *not* approval, and an accepted PDF intake is *not* knowledge approval.

This module deliberately does NOT:

* write to the MemoryLedger;
* create or approve memory proposals;
* create or apply source proposals;
* mutate the source registry;
* alter retrieval algorithms, ranking, grounding, or chat routing;
* update or rebuild a live retrieval index;
* perform OCR or call an LLM;
* summarise, paraphrase, rewrite, normalise, merge, or split chunk text;
* import directly from a raw PDF or from a PDF intake result;
* treat ``preview_ready_for_import`` as approval;
* import unresolved possible-PII or confidential content;
* overwrite an existing pack silently;
* leave a partial pack behind on failure.

Determinism: given an identical PDF hash, preview, approval, pack id, and
importer version, the generated chunk records and the stable manifest content
are identical. Only the explicitly documented operational timestamp fields
(``import_timestamp`` / ``created_at``) carry the wall clock, and they are
excluded from the content hashes and the manifest hash so the *content
identity* of a pack is timestamp-independent.

The docstring above is intentionally explicit about the forbidden operations so
the import-purity test can strip it before scanning the module body.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from agent.pdf_chunk_preview import PdfChunkPreview, PdfChunkWarningCode

# --------------------------------------------------------------------------- #
# Versions and constants.
# --------------------------------------------------------------------------- #

IMPORTER_VERSION = "v6.6"
DEFAULT_PACK_VERSION = "1.0"
PREVIEW_FINGERPRINT_VERSION = "v6.5"
SUPPORTED_PREVIEW_RECORD = "pdf_chunk_preview"
SUPPORTED_PREVIEW_RECORDS = (SUPPORTED_PREVIEW_RECORD,)

DEFAULT_DOMAIN = "general"
DEFAULT_KNOWLEDGE_BACKEND = "deterministic"
DEFAULT_AUTHORITY = "unknown"
PDF_SOURCE_TYPE = "pdf_document"

# Pack ids are filesystem-safe, lowercase identifiers.
import re as _re  # local alias keeps the public import list explicit

_PACK_ID_RE = _re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")

# Warning codes that block import when present on an approved chunk. Possible
# PII and confidentiality are handled separately because they have explicit
# resolution/exclusion paths; the rest fail closed by mere presence.
_BLOCKING_CHUNK_WARNINGS = frozenset(
    {
        PdfChunkWarningCode.CHUNK_MALFORMED_UNICODE.value,
        PdfChunkWarningCode.CHUNK_LOW_SOURCE_QUALITY.value,
        PdfChunkWarningCode.CHUNK_REQUIRES_HUMAN_REVIEW.value,
    }
)
_PII_WARNING = PdfChunkWarningCode.CHUNK_POSSIBLE_PII.value
_CONFIDENTIAL_WARNING = PdfChunkWarningCode.CHUNK_CONFIDENTIAL_MARKER.value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> str:
    return (dt or _utc_now()).isoformat()


def _sha1_id(prefix: str, payload: str, *, width: int = 8) -> str:
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:width]
    return f"{prefix}-{digest}"


def content_hash(text: str) -> str:
    """Deterministic content hash for an imported chunk's exact text."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_valid_pack_id(pack_id: str) -> bool:
    return bool(pack_id) and bool(_PACK_ID_RE.match(pack_id))


# --------------------------------------------------------------------------- #
# Enumerations.
# --------------------------------------------------------------------------- #


class ImportStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    DRY_RUN_READY = "dry_run_ready"
    IMPORTED = "imported"
    BLOCKED = "blocked"
    APPROVAL_MISMATCH = "approval_mismatch"
    PREVIEW_CHANGED = "preview_changed"
    SOURCE_CHANGED = "source_changed"
    TARGET_EXISTS = "target_exists"
    WRITE_FAILED = "write_failed"


class ImportFindingSeverity(str, Enum):
    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"


class ImportFindingCode(str, Enum):
    APPROVAL_MISSING = "approval_missing"
    APPROVAL_IDENTITY_MISSING = "approval_identity_missing"
    APPROVAL_CHUNK_UNKNOWN = "approval_chunk_unknown"
    APPROVAL_CHUNK_REJECTED = "approval_chunk_rejected"
    SOURCE_HASH_MISMATCH = "source_hash_mismatch"
    PREVIEW_FINGERPRINT_MISMATCH = "preview_fingerprint_mismatch"
    PREVIEW_NOT_READY = "preview_not_ready"
    BLOCKING_PDF_FINDING = "blocking_pdf_finding"
    BLOCKING_CHUNK_WARNING = "blocking_chunk_warning"
    UNRESOLVED_PII = "unresolved_pii"
    UNRESOLVED_CONFIDENTIAL_CONTENT = "unresolved_confidential_content"
    TARGET_PACK_EXISTS = "target_pack_exists"
    PACK_ID_INVALID = "pack_id_invalid"
    CHUNK_COUNT_MISMATCH = "chunk_count_mismatch"
    CHUNK_TEXT_CHANGED = "chunk_text_changed"
    CONTENT_HASH_MISMATCH = "content_hash_mismatch"
    UNSUPPORTED_PACK_VERSION = "unsupported_pack_version"
    WRITE_NOT_REQUESTED = "write_not_requested"
    IMPORT_VALIDATED = "import_validated"
    IMPORT_COMPLETED = "import_completed"


# --------------------------------------------------------------------------- #
# Frozen models.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PdfImportFinding:
    """One validation or import observation. Blocking findings stop the write."""

    code: ImportFindingCode
    severity: ImportFindingSeverity
    message: str
    chunk_id: str = ""

    @property
    def is_blocking(self) -> bool:
        return self.severity is ImportFindingSeverity.BLOCKING

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class PdfImportApproval:
    """A structured, human-supplied approval bound to one preview and source.

    Approval is the *only* thing that authorises an import. It must name the
    exact chunk ids approved, must match the current source file hash and the
    current deterministic preview fingerprint, and may not approve unknown
    chunk ids. Advisory preview readiness is never a substitute for approval.
    """

    approval_id: str
    approved_by: str
    approved_at: str
    source_file_hash: str
    preview_fingerprint: str
    approved_chunk_ids: Tuple[str, ...]
    rejected_chunk_ids: Tuple[str, ...] = ()
    approval_notes: str = ""
    unresolved_findings_acknowledged: Tuple[str, ...] = ()
    intended_pack_id: str = ""
    intended_use: str = ""
    # Lineage/provenance the human declares for the resulting pack.
    source_title: str = ""
    authority_level: str = DEFAULT_AUTHORITY
    provenance: str = ""
    permission_or_licence: str = ""
    domain: str = DEFAULT_DOMAIN
    # Explicit policy switch: only when True may confidential/restricted chunks
    # be imported. Possible PII has no such switch and must be excluded.
    allow_confidential: bool = False
    # Optional tamper check: chunk_id -> expected content hash at approval time.
    approved_chunk_content_hashes: Mapping[str, str] = field(default_factory=dict)

    @property
    def has_identity(self) -> bool:
        return bool(self.approval_id and self.approved_by and self.approved_at)

    def to_dict(self) -> dict:
        return {
            "_record": "pdf_import_approval",
            "approval_id": self.approval_id,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "source_file_hash": self.source_file_hash,
            "preview_fingerprint": self.preview_fingerprint,
            "approved_chunk_ids": list(self.approved_chunk_ids),
            "rejected_chunk_ids": list(self.rejected_chunk_ids),
            "approval_notes": self.approval_notes,
            "unresolved_findings_acknowledged": list(self.unresolved_findings_acknowledged),
            "intended_pack_id": self.intended_pack_id,
            "intended_use": self.intended_use,
            "source_title": self.source_title,
            "authority_level": self.authority_level,
            "provenance": self.provenance,
            "permission_or_licence": self.permission_or_licence,
            "domain": self.domain,
            "allow_confidential": self.allow_confidential,
            "approved_chunk_content_hashes": dict(self.approved_chunk_content_hashes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PdfImportApproval":
        def _str(key: str, default: str = "") -> str:
            value = data.get(key, default)
            return str(value) if value is not None else default

        def _tuple(key: str) -> Tuple[str, ...]:
            value = data.get(key) or []
            if isinstance(value, (str, bytes)):
                return (str(value),)
            return tuple(str(v) for v in value)

        hashes_raw = data.get("approved_chunk_content_hashes") or {}
        hashes = {str(k): str(v) for k, v in dict(hashes_raw).items()}
        return cls(
            approval_id=_str("approval_id"),
            approved_by=_str("approved_by"),
            approved_at=_str("approved_at"),
            source_file_hash=_str("source_file_hash"),
            preview_fingerprint=_str("preview_fingerprint"),
            approved_chunk_ids=_tuple("approved_chunk_ids"),
            rejected_chunk_ids=_tuple("rejected_chunk_ids"),
            approval_notes=_str("approval_notes"),
            unresolved_findings_acknowledged=_tuple("unresolved_findings_acknowledged"),
            intended_pack_id=_str("intended_pack_id"),
            intended_use=_str("intended_use"),
            source_title=_str("source_title"),
            authority_level=_str("authority_level", DEFAULT_AUTHORITY) or DEFAULT_AUTHORITY,
            provenance=_str("provenance"),
            permission_or_licence=_str("permission_or_licence"),
            domain=_str("domain", DEFAULT_DOMAIN) or DEFAULT_DOMAIN,
            allow_confidential=bool(data.get("allow_confidential", False)),
            approved_chunk_content_hashes=hashes,
        )


@dataclass(frozen=True)
class PdfImportRequest:
    """A fully-resolved request to validate or import an approved preview."""

    preview: dict
    approval: PdfImportApproval
    pack_id: str
    pack_dir: Optional[str] = None
    write: bool = False
    pack_version: str = DEFAULT_PACK_VERSION
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "_record": "pdf_import_request",
            "pack_id": self.pack_id,
            "pack_dir": self.pack_dir,
            "write": self.write,
            "pack_version": self.pack_version,
            "description": self.description,
            "approval": self.approval.to_dict(),
            "preview_fingerprint": compute_preview_fingerprint(self.preview),
        }


@dataclass(frozen=True)
class PdfImportValidation:
    """The outcome of validating an approval against a preview before any write."""

    status: ImportStatus
    valid: bool
    pack_id: str
    source_file_hash: str
    preview_fingerprint: str
    approved_chunk_count: int
    importable_chunk_ids: Tuple[str, ...]
    excluded_chunk_ids: Tuple[str, ...]
    findings: Tuple[PdfImportFinding, ...]

    @property
    def blocking_findings(self) -> Tuple[PdfImportFinding, ...]:
        return tuple(f for f in self.findings if f.is_blocking)

    def to_dict(self) -> dict:
        return {
            "_record": "pdf_import_validation",
            "status": self.status.value,
            "valid": self.valid,
            "pack_id": self.pack_id,
            "source_file_hash": self.source_file_hash,
            "preview_fingerprint": self.preview_fingerprint,
            "approved_chunk_count": self.approved_chunk_count,
            "importable_chunk_ids": list(self.importable_chunk_ids),
            "excluded_chunk_ids": list(self.excluded_chunk_ids),
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass(frozen=True)
class PdfImportedChunk:
    """One imported chunk with complete, immutable source/page/approval lineage."""

    pack_id: str
    source_id: str
    document_id: str
    chunk_id: str
    original_preview_chunk_id: str
    source_name: str
    source_title: str
    source_file_name: str
    source_file_hash: str
    preview_fingerprint: str
    approval_id: str
    approved_by: str
    approved_at: str
    page_start: int
    page_end: int
    char_offset_start: int
    char_offset_end: int
    extraction_method: str
    warning_codes: Tuple[str, ...]
    domain: str
    authority_level: str
    intended_use: str
    provenance: str
    permission_or_licence: str
    import_timestamp: str
    content_hash: str
    text: str

    @property
    def source_section(self) -> str:
        if self.page_start == self.page_end:
            return f"page {self.page_start}"
        return f"pages {self.page_start}-{self.page_end}"

    def to_knowledge_record(self) -> dict:
        """Pack-compatible ``chunk`` record plus additive v6.6 lineage fields."""
        return {
            # Existing knowledge-pack chunk schema (retrieval-compatible).
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_text": self.text,
            "domain": self.domain,
            "authority": self.authority_level,
            "source_name": self.source_name,
            "source_section": self.source_section,
            "active": True,
            "_record": "chunk",
            # Additive PDF import lineage (ignored by existing readers).
            "pack_id": self.pack_id,
            "original_preview_chunk_id": self.original_preview_chunk_id,
            "source_title": self.source_title,
            "source_file_name": self.source_file_name,
            "source_file_hash": self.source_file_hash,
            "preview_fingerprint": self.preview_fingerprint,
            "approval_id": self.approval_id,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_offset_start": self.char_offset_start,
            "char_offset_end": self.char_offset_end,
            "extraction_method": self.extraction_method,
            "warning_codes": list(self.warning_codes),
            "intended_use": self.intended_use,
            "provenance": self.provenance,
            "permission_or_licence": self.permission_or_licence,
            "import_timestamp": self.import_timestamp,
            "content_hash": self.content_hash,
            "importer_version": IMPORTER_VERSION,
        }

    def to_dict(self) -> dict:
        record = self.to_knowledge_record()
        record["_record"] = "pdf_imported_chunk"
        return record


@dataclass(frozen=True)
class PdfKnowledgePackManifest:
    """The manifest for an imported PDF knowledge pack (pack-format compatible)."""

    pack_id: str
    pack_version: str
    name: str
    description: str
    source_count: int
    chunk_count: int
    source_file_hash: str
    preview_fingerprint: str
    approval_id: str
    approval_actor: str
    approval_timestamp: str
    import_timestamp: str
    intended_use: str
    authority_level: str
    provenance: str
    permission_or_licence: str
    domain: str
    excluded_chunk_ids: Tuple[str, ...]
    unresolved_nonblocking_findings: Tuple[str, ...]
    importer_version: str
    manifest_hash: str

    def to_dict(self) -> dict:
        return {
            "_record": "pdf_knowledge_pack_manifest",
            # Pack-compatible keys first so existing tooling can read the pack.
            "pack_id": self.pack_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.import_timestamp,
            "default_knowledge_backend": DEFAULT_KNOWLEDGE_BACKEND,
            "default_domain": self.domain,
            "settings": {},
            # v6.6 lineage / provenance.
            "pack_version": self.pack_version,
            "source_count": self.source_count,
            "chunk_count": self.chunk_count,
            "source_file_hash": self.source_file_hash,
            "preview_fingerprint": self.preview_fingerprint,
            "approval_id": self.approval_id,
            "approval_actor": self.approval_actor,
            "approval_timestamp": self.approval_timestamp,
            "import_timestamp": self.import_timestamp,
            "intended_use": self.intended_use,
            "authority_level": self.authority_level,
            "provenance": self.provenance,
            "permission_or_licence": self.permission_or_licence,
            "excluded_chunk_ids": list(self.excluded_chunk_ids),
            "unresolved_nonblocking_findings": list(self.unresolved_nonblocking_findings),
            "importer_version": self.importer_version,
            "manifest_hash": self.manifest_hash,
        }


@dataclass(frozen=True)
class PdfImportResult:
    """The full outcome of an import (dry-run or write)."""

    status: ImportStatus
    validation: PdfImportValidation
    written: bool
    pack_dir: Optional[str]
    written_paths: Tuple[str, ...]
    imported_chunks: Tuple[PdfImportedChunk, ...]
    manifest: Optional[PdfKnowledgePackManifest]
    message: str
    findings: Tuple[PdfImportFinding, ...] = ()

    @property
    def chunk_count(self) -> int:
        return len(self.imported_chunks)

    def to_dict(self) -> dict:
        return {
            "_record": "pdf_import_result",
            "status": self.status.value,
            "written": self.written,
            "pack_dir": self.pack_dir,
            "written_paths": list(self.written_paths),
            "chunk_count": self.chunk_count,
            "message": self.message,
            "validation": self.validation.to_dict(),
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "imported_chunks": [c.to_dict() for c in self.imported_chunks],
            "findings": [f.to_dict() for f in self.findings],
        }


# --------------------------------------------------------------------------- #
# Preview normalisation and fingerprinting.
# --------------------------------------------------------------------------- #


def _as_preview_dict(preview) -> dict:
    """Accept an in-memory :class:`PdfChunkPreview` or a loaded preview dict."""
    if isinstance(preview, PdfChunkPreview):
        return preview.to_dict(include_full_text=True)
    if isinstance(preview, Mapping):
        return dict(preview)
    raise TypeError(
        "preview must be a PdfChunkPreview or a loaded preview dict, "
        f"not {type(preview).__name__}"
    )


def compute_preview_fingerprint(preview) -> str:
    """Deterministic fingerprint of a chunk preview.

    Inputs: fingerprint version, source file hash, chunk settings, and the
    ordered list of (chunk id, page mapping, character offsets, warning codes).
    Chunk text is not included because it is fully determined by the source
    file hash and the character offsets under deterministic extraction, so any
    change to the text necessarily changes the file hash or the offsets.
    """
    data = _as_preview_dict(preview)
    chunks = data.get("chunks") or []
    payload = {
        "version": PREVIEW_FINGERPRINT_VERSION,
        "file_hash": data.get("file_hash", ""),
        "chunk_size": data.get("chunk_size"),
        "overlap": data.get("overlap"),
        "max_chunk_size": data.get("max_chunk_size"),
        "respect_page_boundaries": data.get("respect_page_boundaries"),
        "chunks": [
            {
                "chunk_id": c.get("chunk_id", ""),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "char_offset_start": c.get("char_offset_start"),
                "char_offset_end": c.get("char_offset_end"),
                "warning_codes": sorted(c.get("warning_codes") or []),
            }
            for c in chunks
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "pdfprev-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _chunk_index(preview: dict) -> Dict[str, dict]:
    return {c.get("chunk_id", ""): c for c in (preview.get("chunks") or [])}


def _ordered_chunk_ids(preview: dict) -> List[str]:
    return [c.get("chunk_id", "") for c in (preview.get("chunks") or [])]


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #


def _blocking_status(findings: Sequence[PdfImportFinding]) -> ImportStatus:
    """Pick the most specific blocking status from the findings."""
    codes = {f.code for f in findings if f.is_blocking}
    if ImportFindingCode.SOURCE_HASH_MISMATCH in codes:
        return ImportStatus.SOURCE_CHANGED
    if ImportFindingCode.PREVIEW_FINGERPRINT_MISMATCH in codes:
        return ImportStatus.PREVIEW_CHANGED
    if ImportFindingCode.TARGET_PACK_EXISTS in codes:
        return ImportStatus.TARGET_EXISTS
    approval_codes = {
        ImportFindingCode.APPROVAL_MISSING,
        ImportFindingCode.APPROVAL_IDENTITY_MISSING,
        ImportFindingCode.APPROVAL_CHUNK_UNKNOWN,
        ImportFindingCode.APPROVAL_CHUNK_REJECTED,
    }
    if codes & approval_codes:
        return ImportStatus.APPROVAL_MISMATCH
    blocked_codes = {
        ImportFindingCode.BLOCKING_PDF_FINDING,
        ImportFindingCode.BLOCKING_CHUNK_WARNING,
        ImportFindingCode.UNRESOLVED_PII,
        ImportFindingCode.UNRESOLVED_CONFIDENTIAL_CONTENT,
    }
    if codes & blocked_codes:
        return ImportStatus.BLOCKED
    return ImportStatus.INVALID


def validate_import(
    preview,
    approval: PdfImportApproval,
    *,
    pack_id: str,
    pack_dir: Optional[str] = None,
    allow_replace: bool = False,
) -> PdfImportValidation:
    """Validate an approval against a preview. Performs no writes.

    Fails closed: every problem is recorded as a finding, and any blocking
    finding makes the validation invalid. A valid result means the named,
    approved chunk ids exist, the approval is bound to the current source hash
    and preview fingerprint, no blocking PDF or chunk condition remains, and
    the target path would not be silently overwritten.
    """
    preview = _as_preview_dict(preview)
    findings: List[PdfImportFinding] = []

    def block(code: ImportFindingCode, message: str, chunk_id: str = "") -> None:
        findings.append(PdfImportFinding(code, ImportFindingSeverity.BLOCKING, message, chunk_id))

    def warn(code: ImportFindingCode, message: str, chunk_id: str = "") -> None:
        findings.append(PdfImportFinding(code, ImportFindingSeverity.WARNING, message, chunk_id))

    def info(code: ImportFindingCode, message: str, chunk_id: str = "") -> None:
        findings.append(PdfImportFinding(code, ImportFindingSeverity.INFO, message, chunk_id))

    source_hash = str(preview.get("file_hash", ""))
    fingerprint = compute_preview_fingerprint(preview)

    # --- Approval presence and identity. ---
    if approval is None:
        block(ImportFindingCode.APPROVAL_MISSING, "no approval record supplied")
        return PdfImportValidation(
            status=ImportStatus.APPROVAL_MISMATCH,
            valid=False,
            pack_id=pack_id,
            source_file_hash=source_hash,
            preview_fingerprint=fingerprint,
            approved_chunk_count=0,
            importable_chunk_ids=(),
            excluded_chunk_ids=(),
            findings=tuple(findings),
        )
    if not approval.has_identity:
        block(
            ImportFindingCode.APPROVAL_IDENTITY_MISSING,
            "approval is missing approval_id, approved_by, or approved_at",
        )
    if not approval.approved_chunk_ids:
        block(ImportFindingCode.APPROVAL_MISSING, "approval names no approved chunk ids")

    # --- Pack id validity and agreement with the approval. ---
    if not is_valid_pack_id(pack_id):
        block(ImportFindingCode.PACK_ID_INVALID, f"pack id {pack_id!r} is not a valid identifier")
    if approval.intended_pack_id and approval.intended_pack_id != pack_id:
        block(
            ImportFindingCode.PACK_ID_INVALID,
            f"approval intends pack {approval.intended_pack_id!r}, not {pack_id!r}",
        )

    # --- Preview record support. ---
    record = str(preview.get("_record", ""))
    if record not in SUPPORTED_PREVIEW_RECORDS:
        block(
            ImportFindingCode.UNSUPPORTED_PACK_VERSION,
            f"unsupported preview record {record!r}; expected {SUPPORTED_PREVIEW_RECORD!r}",
        )

    # --- Binding: source hash and preview fingerprint. ---
    if source_hash != approval.source_file_hash:
        block(
            ImportFindingCode.SOURCE_HASH_MISMATCH,
            "approval source_file_hash does not match the preview file hash",
        )
    if fingerprint != approval.preview_fingerprint:
        block(
            ImportFindingCode.PREVIEW_FINGERPRINT_MISMATCH,
            "approval preview_fingerprint does not match the current preview",
        )

    # --- Source-level PDF gating (intake is not knowledge approval). ---
    if bool(preview.get("intake_blocked", False)):
        block(ImportFindingCode.BLOCKING_PDF_FINDING, "source PDF intake is blocked")
    if not bool(preview.get("has_text_layer", False)):
        block(ImportFindingCode.BLOCKING_PDF_FINDING, "source PDF has no extractable text layer")
    if bool(preview.get("requires_ocr", False)):
        block(ImportFindingCode.BLOCKING_PDF_FINDING, "source PDF requires OCR (out of scope)")

    summary = preview.get("summary") or {}
    if not bool(summary.get("preview_ready_for_import", False)):
        info(
            ImportFindingCode.PREVIEW_NOT_READY,
            "advisory preview readiness is False; explicit approval still governs import",
        )

    # --- Chunk-level checks over the approved set. ---
    index = _chunk_index(preview)
    rejected = set(approval.rejected_chunk_ids)
    importable: List[str] = []
    seen_approved = set()

    for chunk_id in approval.approved_chunk_ids:
        if chunk_id in seen_approved:
            continue
        seen_approved.add(chunk_id)
        if chunk_id in rejected:
            block(
                ImportFindingCode.APPROVAL_CHUNK_REJECTED,
                "chunk is both approved and rejected",
                chunk_id,
            )
            continue
        chunk = index.get(chunk_id)
        if chunk is None:
            block(
                ImportFindingCode.APPROVAL_CHUNK_UNKNOWN,
                "approved chunk id is not present in the preview",
                chunk_id,
            )
            continue
        if "text" not in chunk or chunk.get("text") is None:
            block(
                ImportFindingCode.CHUNK_TEXT_CHANGED,
                "preview chunk has no full text to import (re-export preview as JSON)",
                chunk_id,
            )
            continue

        expected_hash = approval.approved_chunk_content_hashes.get(chunk_id)
        if expected_hash and expected_hash != content_hash(str(chunk.get("text", ""))):
            block(
                ImportFindingCode.CONTENT_HASH_MISMATCH,
                "chunk text differs from the approved content hash",
                chunk_id,
            )
            continue

        codes = set(chunk.get("warning_codes") or [])
        if _PII_WARNING in codes:
            block(
                ImportFindingCode.UNRESOLVED_PII,
                "approved chunk has unresolved possible PII; exclude it instead",
                chunk_id,
            )
            continue
        if _CONFIDENTIAL_WARNING in codes and not approval.allow_confidential:
            block(
                ImportFindingCode.UNRESOLVED_CONFIDENTIAL_CONTENT,
                "approved chunk is marked confidential; excluded unless allow_confidential is set",
                chunk_id,
            )
            continue
        blocking_here = codes & _BLOCKING_CHUNK_WARNINGS
        if blocking_here:
            block(
                ImportFindingCode.BLOCKING_CHUNK_WARNING,
                f"approved chunk has blocking warning(s): {', '.join(sorted(blocking_here))}",
                chunk_id,
            )
            continue

        nonblocking = codes - {_PII_WARNING, _CONFIDENTIAL_WARNING} - _BLOCKING_CHUNK_WARNINGS
        if nonblocking:
            warn(
                ImportFindingCode.BLOCKING_CHUNK_WARNING,
                f"approved chunk carries non-blocking warning(s): {', '.join(sorted(nonblocking))}",
                chunk_id,
            )
        importable.append(chunk_id)

    # --- Target path safety. ---
    if pack_dir is not None:
        target = Path(pack_dir)
        if target.exists() and not allow_replace:
            block(
                ImportFindingCode.TARGET_PACK_EXISTS,
                f"target pack path already exists: {target}",
            )

    blocking = [f for f in findings if f.is_blocking]
    valid = not blocking
    excluded = sorted(set(_ordered_chunk_ids(preview)) - set(importable))

    if valid:
        status = ImportStatus.VALID
        info(ImportFindingCode.IMPORT_VALIDATED, "approval validated; ready for dry-run or write")
    else:
        status = _blocking_status(findings)

    return PdfImportValidation(
        status=status,
        valid=valid,
        pack_id=pack_id,
        source_file_hash=source_hash,
        preview_fingerprint=fingerprint,
        approved_chunk_count=len(seen_approved),
        importable_chunk_ids=tuple(importable),
        excluded_chunk_ids=tuple(excluded),
        findings=tuple(findings),
    )


# --------------------------------------------------------------------------- #
# Building imported chunks and the manifest (pure; no writes).
# --------------------------------------------------------------------------- #


def _build_imported_chunks(
    preview: dict,
    approval: PdfImportApproval,
    *,
    pack_id: str,
    importable_ids: Sequence[str],
    import_timestamp: str,
) -> List[PdfImportedChunk]:
    file_name = str(preview.get("file_name", ""))
    file_hash = str(preview.get("file_hash", ""))
    fingerprint = compute_preview_fingerprint(preview)
    source_title = approval.source_title or file_name or pack_id
    source_id = _sha1_id("src", f"{pack_id}|{file_hash}")
    document_id = _sha1_id("doc", f"{pack_id}|{file_hash}|{file_name}")
    index = _chunk_index(preview)

    # Preserve the deterministic preview order for the approved subset.
    importable = set(importable_ids)
    ordered = [cid for cid in _ordered_chunk_ids(preview) if cid in importable]

    chunks: List[PdfImportedChunk] = []
    for chunk_id in ordered:
        chunk = index[chunk_id]
        text = str(chunk.get("text", ""))
        chash = content_hash(text)
        chunks.append(
            PdfImportedChunk(
                pack_id=pack_id,
                source_id=source_id,
                document_id=document_id,
                chunk_id=_sha1_id("chk", f"{pack_id}|{chunk_id}|{chash}"),
                original_preview_chunk_id=chunk_id,
                source_name=source_title,
                source_title=source_title,
                source_file_name=file_name,
                source_file_hash=file_hash,
                preview_fingerprint=fingerprint,
                approval_id=approval.approval_id,
                approved_by=approval.approved_by,
                approved_at=approval.approved_at,
                page_start=int(chunk.get("page_start", 0)),
                page_end=int(chunk.get("page_end", 0)),
                char_offset_start=int(chunk.get("char_offset_start", 0)),
                char_offset_end=int(chunk.get("char_offset_end", 0)),
                extraction_method=str(chunk.get("extraction_method", "")),
                warning_codes=tuple(chunk.get("warning_codes") or []),
                domain=approval.domain or DEFAULT_DOMAIN,
                authority_level=approval.authority_level or DEFAULT_AUTHORITY,
                intended_use=approval.intended_use,
                provenance=approval.provenance,
                permission_or_licence=approval.permission_or_licence,
                import_timestamp=import_timestamp,
                content_hash=chash,
                text=text,
            )
        )
    return chunks


def _manifest_hash(
    *,
    pack_id: str,
    pack_version: str,
    source_file_hash: str,
    preview_fingerprint: str,
    approval_id: str,
    intended_use: str,
    authority_level: str,
    provenance: str,
    permission_or_licence: str,
    domain: str,
    excluded_chunk_ids: Sequence[str],
    unresolved_nonblocking_findings: Sequence[str],
    chunk_signature: Sequence[Tuple[str, str]],
) -> str:
    """Hash the *stable content identity* of a pack (no operational timestamps)."""
    payload = {
        "importer_version": IMPORTER_VERSION,
        "pack_id": pack_id,
        "pack_version": pack_version,
        "source_file_hash": source_file_hash,
        "preview_fingerprint": preview_fingerprint,
        "approval_id": approval_id,
        "intended_use": intended_use,
        "authority_level": authority_level,
        "provenance": provenance,
        "permission_or_licence": permission_or_licence,
        "domain": domain,
        "excluded_chunk_ids": sorted(excluded_chunk_ids),
        "unresolved_nonblocking_findings": sorted(unresolved_nonblocking_findings),
        "chunks": [list(pair) for pair in chunk_signature],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "pdfpack-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _build_manifest(
    preview: dict,
    approval: PdfImportApproval,
    validation: PdfImportValidation,
    chunks: Sequence[PdfImportedChunk],
    *,
    pack_id: str,
    pack_version: str,
    description: str,
    import_timestamp: str,
) -> PdfKnowledgePackManifest:
    source_title = approval.source_title or str(preview.get("file_name", "")) or pack_id
    nonblocking = sorted(
        {
            f.message
            for f in validation.findings
            if f.severity is ImportFindingSeverity.WARNING
        }
        | set(approval.unresolved_findings_acknowledged)
    )
    chunk_signature = [(c.chunk_id, c.content_hash) for c in chunks]
    mhash = _manifest_hash(
        pack_id=pack_id,
        pack_version=pack_version,
        source_file_hash=str(preview.get("file_hash", "")),
        preview_fingerprint=validation.preview_fingerprint,
        approval_id=approval.approval_id,
        intended_use=approval.intended_use,
        authority_level=approval.authority_level or DEFAULT_AUTHORITY,
        provenance=approval.provenance,
        permission_or_licence=approval.permission_or_licence,
        domain=approval.domain or DEFAULT_DOMAIN,
        excluded_chunk_ids=validation.excluded_chunk_ids,
        unresolved_nonblocking_findings=nonblocking,
        chunk_signature=chunk_signature,
    )
    return PdfKnowledgePackManifest(
        pack_id=pack_id,
        pack_version=pack_version,
        name=source_title,
        description=description or f"PDF knowledge pack imported from {source_title}",
        source_count=1 if chunks else 0,
        chunk_count=len(chunks),
        source_file_hash=str(preview.get("file_hash", "")),
        preview_fingerprint=validation.preview_fingerprint,
        approval_id=approval.approval_id,
        approval_actor=approval.approved_by,
        approval_timestamp=approval.approved_at,
        import_timestamp=import_timestamp,
        intended_use=approval.intended_use,
        authority_level=approval.authority_level or DEFAULT_AUTHORITY,
        provenance=approval.provenance,
        permission_or_licence=approval.permission_or_licence,
        domain=approval.domain or DEFAULT_DOMAIN,
        excluded_chunk_ids=validation.excluded_chunk_ids,
        unresolved_nonblocking_findings=tuple(nonblocking),
        importer_version=IMPORTER_VERSION,
        manifest_hash=mhash,
    )


def _knowledge_records(
    manifest: PdfKnowledgePackManifest,
    chunks: Sequence[PdfImportedChunk],
) -> List[dict]:
    """Build the deterministic ``knowledge.jsonl`` record list (source/doc/chunks)."""
    if not chunks:
        return []
    first = chunks[0]
    source_record = {
        "source_id": first.source_id,
        "source_name": first.source_title,
        "source_type": PDF_SOURCE_TYPE,
        "domain": manifest.domain,
        "authority": manifest.authority_level,
        "path_or_url": first.source_file_name,
        "staleness_policy": "static",
        "licence": manifest.permission_or_licence,
        "retrieved_at": manifest.import_timestamp,
        "published_at": None,
        "version": manifest.pack_version,
        "active": True,
        "_record": "source",
        "source_file_hash": manifest.source_file_hash,
        "preview_fingerprint": manifest.preview_fingerprint,
        "approval_id": manifest.approval_id,
        "provenance": manifest.provenance,
        "intended_use": manifest.intended_use,
    }
    document_record = {
        "document_id": first.document_id,
        "source_id": first.source_id,
        "title": first.source_title,
        "path_or_url": first.source_file_name,
        "_record": "document",
        "source_file_hash": manifest.source_file_hash,
        "page_count": max((c.page_end for c in chunks), default=0),
    }
    records: List[dict] = [source_record, document_record]
    records.extend(c.to_knowledge_record() for c in chunks)
    return records


# --------------------------------------------------------------------------- #
# Safe, atomic pack write.
# --------------------------------------------------------------------------- #


def _atomic_write_pack(pack_dir: Path, files: Mapping[str, str]) -> List[str]:
    """Write ``files`` into ``pack_dir`` atomically via a temp dir then rename.

    The parent directory is created (bounded to the requested path). Files are
    written into a sibling temporary directory and only renamed into place once
    all writes succeed; on any failure the temporary directory is removed so no
    partial pack is ever left behind. The target must not already exist.
    """
    parent = pack_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{pack_dir.name}.import-", dir=str(parent)))
    try:
        for name, content in files.items():
            (tmp_dir / name).write_text(content, encoding="utf-8")
        os.rename(tmp_dir, pack_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return [str(pack_dir / name) for name in files]


def _render_pack_files(
    manifest: PdfKnowledgePackManifest,
    chunks: Sequence[PdfImportedChunk],
) -> Dict[str, str]:
    manifest_json = json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
    records = _knowledge_records(manifest, chunks)
    knowledge_jsonl = "\n".join(
        json.dumps(r, ensure_ascii=False) for r in records
    )
    if knowledge_jsonl:
        knowledge_jsonl += "\n"
    return {"manifest.json": manifest_json + "\n", "knowledge.jsonl": knowledge_jsonl}


# --------------------------------------------------------------------------- #
# Top-level import entry point.
# --------------------------------------------------------------------------- #


def import_pdf_pack(request: PdfImportRequest, *, now: Optional[datetime] = None) -> PdfImportResult:
    """Validate an approved preview and, only when explicitly requested, write it.

    Dry-run is the default: with ``request.write`` False the importer validates,
    builds the manifest and chunk records in memory, and writes nothing. Only
    with ``request.write`` True (and an explicit ``pack_dir``) does it create the
    pack files, fail-closed on an existing target, and clean up on any error.
    """
    preview = _as_preview_dict(request.preview)
    approval = request.approval
    pack_id = request.pack_id
    import_timestamp = _iso(now)

    validation = validate_import(
        preview,
        approval,
        pack_id=pack_id,
        pack_dir=request.pack_dir if request.write else None,
    )

    if not validation.valid:
        return PdfImportResult(
            status=validation.status,
            validation=validation,
            written=False,
            pack_dir=request.pack_dir,
            written_paths=(),
            imported_chunks=(),
            manifest=None,
            message="validation failed; nothing written",
            findings=validation.findings,
        )

    chunks = _build_imported_chunks(
        preview,
        approval,
        pack_id=pack_id,
        importable_ids=validation.importable_chunk_ids,
        import_timestamp=import_timestamp,
    )
    manifest = _build_manifest(
        preview,
        approval,
        validation,
        chunks,
        pack_id=pack_id,
        pack_version=request.pack_version,
        description=request.description,
        import_timestamp=import_timestamp,
    )

    if len(chunks) != len(validation.importable_chunk_ids):  # defensive invariant
        finding = PdfImportFinding(
            ImportFindingCode.CHUNK_COUNT_MISMATCH,
            ImportFindingSeverity.BLOCKING,
            "imported chunk count does not equal the approved chunk count",
        )
        return PdfImportResult(
            status=ImportStatus.INVALID,
            validation=validation,
            written=False,
            pack_dir=request.pack_dir,
            written_paths=(),
            imported_chunks=(),
            manifest=None,
            message="chunk count invariant violated; nothing written",
            findings=validation.findings + (finding,),
        )

    if not request.write:
        finding = PdfImportFinding(
            ImportFindingCode.WRITE_NOT_REQUESTED,
            ImportFindingSeverity.INFO,
            "dry-run only; pass write=True with an explicit pack_dir to create the pack",
        )
        return PdfImportResult(
            status=ImportStatus.DRY_RUN_READY,
            validation=validation,
            written=False,
            pack_dir=request.pack_dir,
            written_paths=(),
            imported_chunks=tuple(chunks),
            manifest=manifest,
            message=f"dry-run ready: {len(chunks)} chunk(s) would be imported",
            findings=validation.findings + (finding,),
        )

    if not request.pack_dir:
        finding = PdfImportFinding(
            ImportFindingCode.WRITE_NOT_REQUESTED,
            ImportFindingSeverity.BLOCKING,
            "write requested without an explicit pack_dir",
        )
        return PdfImportResult(
            status=ImportStatus.WRITE_FAILED,
            validation=validation,
            written=False,
            pack_dir=None,
            written_paths=(),
            imported_chunks=tuple(chunks),
            manifest=manifest,
            message="no pack_dir supplied; nothing written",
            findings=validation.findings + (finding,),
        )

    pack_dir = Path(request.pack_dir)
    if pack_dir.exists():
        finding = PdfImportFinding(
            ImportFindingCode.TARGET_PACK_EXISTS,
            ImportFindingSeverity.BLOCKING,
            f"target pack path already exists: {pack_dir}",
        )
        return PdfImportResult(
            status=ImportStatus.TARGET_EXISTS,
            validation=validation,
            written=False,
            pack_dir=str(pack_dir),
            written_paths=(),
            imported_chunks=tuple(chunks),
            manifest=manifest,
            message="target pack already exists; refusing to overwrite",
            findings=validation.findings + (finding,),
        )

    files = _render_pack_files(manifest, chunks)
    try:
        written_paths = _atomic_write_pack(pack_dir, files)
    except OSError as exc:
        finding = PdfImportFinding(
            ImportFindingCode.WRITE_NOT_REQUESTED,
            ImportFindingSeverity.BLOCKING,
            f"write failed: {exc.__class__.__name__}: {exc}",
        )
        return PdfImportResult(
            status=ImportStatus.WRITE_FAILED,
            validation=validation,
            written=False,
            pack_dir=str(pack_dir),
            written_paths=(),
            imported_chunks=tuple(chunks),
            manifest=manifest,
            message="write failed; partial output cleaned up",
            findings=validation.findings + (finding,),
        )

    completed = PdfImportFinding(
        ImportFindingCode.IMPORT_COMPLETED,
        ImportFindingSeverity.INFO,
        f"imported {len(chunks)} chunk(s) into {pack_dir}",
    )
    return PdfImportResult(
        status=ImportStatus.IMPORTED,
        validation=validation,
        written=True,
        pack_dir=str(pack_dir),
        written_paths=tuple(written_paths),
        imported_chunks=tuple(chunks),
        manifest=manifest,
        message=f"imported {len(chunks)} chunk(s) into {pack_dir}",
        findings=validation.findings + (completed,),
    )


# --------------------------------------------------------------------------- #
# Loaders (read-only).
# --------------------------------------------------------------------------- #


def load_preview(path) -> dict:
    """Load a v6.5 preview JSON document (must include full chunk text)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("preview file does not contain a JSON object")
    return data


def load_approval(path) -> PdfImportApproval:
    """Load a structured approval JSON document."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("approval file does not contain a JSON object")
    return PdfImportApproval.from_dict(data)


# --------------------------------------------------------------------------- #
# Deterministic renderers (read-only).
# --------------------------------------------------------------------------- #


def render_validation_markdown(validation: PdfImportValidation) -> str:
    lines: List[str] = []
    lines.append("# PDF pack import validation (v6.6)")
    lines.append("")
    lines.append(
        "> Validation is read-only. No knowledge pack, retrieval index, source "
        "registry, or memory state is created or changed by validating."
    )
    lines.append("")
    lines.append(f"- **Pack id**: {validation.pack_id}")
    lines.append(f"- **Status**: {validation.status.value}")
    lines.append(f"- **Valid**: {validation.valid}")
    lines.append(f"- **Source file hash**: {validation.source_file_hash}")
    lines.append(f"- **Preview fingerprint**: {validation.preview_fingerprint}")
    lines.append(f"- **Approved chunks**: {validation.approved_chunk_count}")
    lines.append(f"- **Importable chunks**: {len(validation.importable_chunk_ids)}")
    lines.append(f"- **Excluded chunks**: {len(validation.excluded_chunk_ids)}")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if not validation.findings:
        lines.append("- (none)")
    else:
        lines.append("| Severity | Code | Chunk | Message |")
        lines.append("| --- | --- | --- | --- |")
        for f in validation.findings:
            lines.append(
                f"| {f.severity.value} | {f.code.value} | {f.chunk_id or '-'} | {f.message} |"
            )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_import_result_markdown(result: PdfImportResult) -> str:
    lines: List[str] = []
    title = "imported" if result.written else "dry-run / not written"
    lines.append(f"# PDF pack import result (v6.6 — {title})")
    lines.append("")
    if not result.written:
        lines.append(
            "> Nothing was written. Validation and dry-run never create pack files; "
            "an explicit write with an explicit pack directory is required."
        )
        lines.append("")
    lines.append(f"- **Status**: {result.status.value}")
    lines.append(f"- **Written**: {result.written}")
    lines.append(f"- **Pack dir**: {result.pack_dir or '-'}")
    lines.append(f"- **Imported chunks**: {result.chunk_count}")
    if result.manifest is not None:
        lines.append(f"- **Manifest hash**: {result.manifest.manifest_hash}")
        lines.append(f"- **Preview fingerprint**: {result.manifest.preview_fingerprint}")
        lines.append(f"- **Approval**: {result.manifest.approval_id} by {result.manifest.approval_actor}")
    lines.append(f"- **Message**: {result.message}")
    lines.append("")
    if result.written_paths:
        lines.append("## Written files")
        lines.append("")
        for p in result.written_paths:
            lines.append(f"- {p}")
        lines.append("")
    lines.append("## Findings")
    lines.append("")
    if not result.findings:
        lines.append("- (none)")
    else:
        lines.append("| Severity | Code | Chunk | Message |")
        lines.append("| --- | --- | --- | --- |")
        for f in result.findings:
            lines.append(
                f"| {f.severity.value} | {f.code.value} | {f.chunk_id or '-'} | {f.message} |"
            )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
