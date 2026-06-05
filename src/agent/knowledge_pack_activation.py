"""Governed knowledge-pack activation, deactivation, rollback and supersession (v7.0).

A knowledge pack is *imported* (its ``manifest.json`` + ``knowledge.jsonl`` exist on
disk) and may be *evaluated* (a retrieval-evaluation report exists). Neither makes
the pack **active** in retrieval. This module adds the missing governed lifecycle
between import/evaluation and retrieval eligibility.

Cardinal rules (enforced + tested):

* Imported does not mean active.
* Evaluated does not mean active.
* A passing evaluation is *evidence*, never *approval*.
* Activation requires an explicit, structured :class:`KnowledgePackActivationApproval`.
* Approval binds to an exact pack fingerprint **and** an exact evaluation fingerprint.
* Deactivation is reversible; rollback preserves history; supersession is atomic.
* Only one active revision per source lineage unless an explicit coexistence policy
  permits otherwise (default ``exclusive_latest``).

The module is **generic**: it reads any pack directory that carries a knowledge
manifest (Hugging Face, PDF, or a future DOCX/Markdown/TXT/CSV importer) plus a
``knowledge.jsonl`` of ``_record="chunk"`` rows. It is source-agnostic by design —
there is no Hugging-Face-only or PDF-only activation path.

Boundaries (verified by import-purity tests): this module imports **only the
standard library**. It never writes a MemoryLedger, never applies a memory
proposal, never mutates a source registry, never rewrites pack content or chunks,
never re-runs chunking, and never calls an LLM. The only things it writes are the
governed activation-state manifest and the append-only activation audit log, and
then only when a caller passes an explicit write flag.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ACTIVATION_LAYER_VERSION = "knowledge-pack-activation-v7.0"

CONTENT_FP_PREFIX = "packfp-"
MANIFEST_HASH_PREFIX = "packmanifest-"
LIFECYCLE_HASH_PREFIX = "pkglc-"
AUDIT_HASH_PREFIX = "pkgaudit-"
STATE_HASH_PREFIX = "pkgstate-"

# Manifest ``_record`` discriminators this layer recognises as knowledge packs.
KNOWLEDGE_MANIFEST_RECORDS = frozenset({
    "hf_knowledge_pack_manifest",
    "pdf_knowledge_pack_manifest",
})
# An eval pack manifest — recognised so it can be *rejected* for knowledge use.
EVAL_MANIFEST_RECORDS = frozenset({
    "hf_eval_pack_manifest",
})

MANIFEST_FILE = "manifest.json"
KNOWLEDGE_FILE = "knowledge.jsonl"

# Default governed state locations (relative to the project root). Callers may
# override both; tests always point them at a temp directory.
DEFAULT_STATE_PATH = "config/active_knowledge_packs.jsonl"
DEFAULT_AUDIT_PATH = "reports/knowledge_pack_activation_audit.jsonl"


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PackLifecycleState(str, Enum):
    """The lifecycle state of a knowledge pack."""

    IMPORTED = "imported"
    EVALUATION_FAILED = "evaluation_failed"
    EVALUATED = "evaluated"
    ACTIVATION_PENDING = "activation_pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUPERSEDED = "superseded"
    RETIRED = "retired"
    BLOCKED = "blocked"


class ActivationScope(str, Enum):
    """What transition an activation approval authorises."""

    ACTIVATE = "activate"
    REACTIVATE = "reactivate"
    SUPERSEDE = "supersede"
    ROLLBACK = "rollback"
    EMERGENCY_DEACTIVATE = "emergency_deactivate"


class CoexistencePolicy(str, Enum):
    """How many revisions of one source lineage may be active together."""

    EXCLUSIVE_LATEST = "exclusive_latest"
    EXPLICIT_MULTI_VERSION = "explicit_multi_version"
    ENVIRONMENT_SCOPED = "environment_scoped"
    TOPIC_PARTITIONED = "topic_partitioned"


class AuditAction(str, Enum):
    """The lifecycle action recorded in the audit log."""

    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"
    EMERGENCY_DEACTIVATE = "emergency_deactivate"
    ROLLBACK = "rollback"
    SUPERSEDE = "supersede"


class ActivationFindingCode(str, Enum):
    """Why a requested transition was rejected (or flagged)."""

    PACK_ID_MISMATCH = "pack_id_mismatch"
    PACK_VERSION_MISMATCH = "pack_version_mismatch"
    PACK_FINGERPRINT_MISMATCH = "pack_fingerprint_mismatch"
    MANIFEST_HASH_MISMATCH = "manifest_hash_mismatch"
    EVALUATION_FINGERPRINT_MISMATCH = "evaluation_fingerprint_mismatch"
    EVALUATION_PACK_MISMATCH = "evaluation_pack_mismatch"
    EVALUATION_NOT_PASSED = "evaluation_not_passed"
    EVALUATION_BLOCKING_FAILURE = "evaluation_blocking_failure"
    EVALUATION_FORBIDDEN_BLEED = "evaluation_forbidden_bleed"
    EVALUATION_UNSUPPORTED_CITATION = "evaluation_unsupported_citation"
    EVALUATION_UNCITED_CLAIM = "evaluation_uncited_claim"
    EVALUATION_REGRESSION = "evaluation_regression"
    EVALUATION_RECALL_BELOW_POLICY = "evaluation_recall_below_policy"
    EVALUATION_STALE = "evaluation_stale"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_SCOPE_INVALID = "approval_scope_invalid"
    APPROVAL_ENVIRONMENT_MISMATCH = "approval_environment_mismatch"
    LICENCE_CHANGED = "licence_changed"
    PROVENANCE_CHANGED = "provenance_changed"
    AUTHORITY_BELOW_POLICY = "authority_below_policy"
    PACK_NOT_KNOWLEDGE = "pack_not_knowledge"
    PACK_BLOCKED_OR_RETIRED = "pack_blocked_or_retired"
    INVALID_STATE_TRANSITION = "invalid_state_transition"
    LINEAGE_CONFLICT_ACTIVE_REVISION = "lineage_conflict_active_revision"
    DUPLICATE_SOURCE_FINGERPRINT = "duplicate_source_fingerprint"
    PREDECESSOR_AND_SUCCESSOR_ACTIVE = "predecessor_and_successor_active"
    MUTUALLY_EXCLUSIVE_CONFLICT = "mutually_exclusive_conflict"
    COEXISTENCE_NOT_PERMITTED = "coexistence_not_permitted"
    STALE_ROLLBACK_APPROVAL = "stale_rollback_approval"
    ROLLBACK_TARGET_MISSING = "rollback_target_missing"
    ROLLBACK_TARGET_INVALID = "rollback_target_invalid"
    PACK_NOT_FOUND = "pack_not_found"
    PACK_NOT_ACTIVE = "pack_not_active"
    OK = "ok"


class FindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# Findings that block a transition (everything else is advisory).
_BLOCKING_CODES = frozenset({
    ActivationFindingCode.PACK_ID_MISMATCH,
    ActivationFindingCode.PACK_VERSION_MISMATCH,
    ActivationFindingCode.PACK_FINGERPRINT_MISMATCH,
    ActivationFindingCode.MANIFEST_HASH_MISMATCH,
    ActivationFindingCode.EVALUATION_FINGERPRINT_MISMATCH,
    ActivationFindingCode.EVALUATION_PACK_MISMATCH,
    ActivationFindingCode.EVALUATION_NOT_PASSED,
    ActivationFindingCode.EVALUATION_BLOCKING_FAILURE,
    ActivationFindingCode.EVALUATION_FORBIDDEN_BLEED,
    ActivationFindingCode.EVALUATION_UNSUPPORTED_CITATION,
    ActivationFindingCode.EVALUATION_UNCITED_CLAIM,
    ActivationFindingCode.EVALUATION_REGRESSION,
    ActivationFindingCode.EVALUATION_RECALL_BELOW_POLICY,
    ActivationFindingCode.EVALUATION_STALE,
    ActivationFindingCode.APPROVAL_EXPIRED,
    ActivationFindingCode.APPROVAL_SCOPE_INVALID,
    ActivationFindingCode.APPROVAL_ENVIRONMENT_MISMATCH,
    ActivationFindingCode.LICENCE_CHANGED,
    ActivationFindingCode.PROVENANCE_CHANGED,
    ActivationFindingCode.AUTHORITY_BELOW_POLICY,
    ActivationFindingCode.PACK_NOT_KNOWLEDGE,
    ActivationFindingCode.PACK_BLOCKED_OR_RETIRED,
    ActivationFindingCode.INVALID_STATE_TRANSITION,
    ActivationFindingCode.LINEAGE_CONFLICT_ACTIVE_REVISION,
    ActivationFindingCode.DUPLICATE_SOURCE_FINGERPRINT,
    ActivationFindingCode.PREDECESSOR_AND_SUCCESSOR_ACTIVE,
    ActivationFindingCode.MUTUALLY_EXCLUSIVE_CONFLICT,
    ActivationFindingCode.COEXISTENCE_NOT_PERMITTED,
    ActivationFindingCode.STALE_ROLLBACK_APPROVAL,
    ActivationFindingCode.ROLLBACK_TARGET_MISSING,
    ActivationFindingCode.ROLLBACK_TARGET_INVALID,
    ActivationFindingCode.PACK_NOT_FOUND,
    ActivationFindingCode.PACK_NOT_ACTIVE,
})


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationFinding:
    """One reason a transition was rejected or flagged."""

    code: ActivationFindingCode
    message: str

    @property
    def severity(self) -> FindingSeverity:
        return (FindingSeverity.ERROR if self.code in _BLOCKING_CODES
                else FindingSeverity.INFO)

    @property
    def blocking(self) -> bool:
        return self.code in _BLOCKING_CODES

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "blocking": self.blocking,
        }


# ---------------------------------------------------------------------------
# Threshold policy + evaluation evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationThresholdPolicy:
    """The bar an evaluation must clear before a pack may be activated."""

    policy_id: str = "default-strict-v1"
    max_failed_cases: int = 0
    max_forbidden_source_hits: int = 0
    max_unsupported_citations: int = 0
    max_uncited_factual_claims: int = 0
    max_unrelated_regressions: int = 0
    min_expected_source_recall: float = 1.0
    min_cited_expected_source_recall: float = 1.0
    max_evaluation_age_days: Optional[int] = None
    require_pass_flag: bool = True

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "max_failed_cases": self.max_failed_cases,
            "max_forbidden_source_hits": self.max_forbidden_source_hits,
            "max_unsupported_citations": self.max_unsupported_citations,
            "max_uncited_factual_claims": self.max_uncited_factual_claims,
            "max_unrelated_regressions": self.max_unrelated_regressions,
            "min_expected_source_recall": self.min_expected_source_recall,
            "min_cited_expected_source_recall": self.min_cited_expected_source_recall,
            "max_evaluation_age_days": self.max_evaluation_age_days,
            "require_pass_flag": self.require_pass_flag,
        }


DEFAULT_THRESHOLD_POLICY = EvaluationThresholdPolicy()


@dataclass(frozen=True)
class KnowledgePackEvaluationEvidence:
    """Deterministic, fingerprint-bound evidence that a pack was evaluated."""

    evaluation_report_id: str
    pack_fingerprint: str
    evaluated_at: str = ""
    evaluator: str = ""
    case_count: int = 0
    passed_case_count: int = 0
    failed_case_count: int = 0
    forbidden_source_hit_count: int = 0
    unrelated_case_regression_count: int = 0
    unsupported_citation_count: int = 0
    uncited_factual_claim_count: int = 0
    expected_source_recall: float = 0.0
    cited_expected_source_recall: float = 0.0
    passed: bool = False
    threshold_policy_id: str = ""
    failure_classifications: Tuple[str, ...] = ()
    evaluation_report_fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        payload = {
            "v": ACTIVATION_LAYER_VERSION,
            "evaluation_report_id": self.evaluation_report_id,
            "pack_fingerprint": self.pack_fingerprint,
            "case_count": self.case_count,
            "passed_case_count": self.passed_case_count,
            "failed_case_count": self.failed_case_count,
            "forbidden_source_hit_count": self.forbidden_source_hit_count,
            "unrelated_case_regression_count": self.unrelated_case_regression_count,
            "unsupported_citation_count": self.unsupported_citation_count,
            "uncited_factual_claim_count": self.uncited_factual_claim_count,
            "expected_source_recall": round(self.expected_source_recall, 6),
            "cited_expected_source_recall": round(self.cited_expected_source_recall, 6),
            "passed": self.passed,
            "failure_classifications": sorted(self.failure_classifications),
        }
        return "evalfp-" + _sha256_hex(_canonical(payload))[:20]

    @property
    def report_fingerprint(self) -> str:
        return self.evaluation_report_fingerprint or self.compute_fingerprint()

    def to_dict(self) -> dict:
        return {
            "_record": "knowledge_pack_evaluation_evidence",
            "evaluation_report_id": self.evaluation_report_id,
            "evaluation_report_fingerprint": self.report_fingerprint,
            "pack_fingerprint": self.pack_fingerprint,
            "evaluated_at": self.evaluated_at,
            "evaluator": self.evaluator,
            "case_count": self.case_count,
            "passed_case_count": self.passed_case_count,
            "failed_case_count": self.failed_case_count,
            "forbidden_source_hit_count": self.forbidden_source_hit_count,
            "unrelated_case_regression_count": self.unrelated_case_regression_count,
            "unsupported_citation_count": self.unsupported_citation_count,
            "uncited_factual_claim_count": self.uncited_factual_claim_count,
            "expected_source_recall": self.expected_source_recall,
            "cited_expected_source_recall": self.cited_expected_source_recall,
            "passed": self.passed,
            "threshold_policy_id": self.threshold_policy_id,
            "failure_classifications": list(self.failure_classifications),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgePackEvaluationEvidence":
        return cls(
            evaluation_report_id=str(data.get("evaluation_report_id", "")),
            pack_fingerprint=str(data.get("pack_fingerprint", "")),
            evaluated_at=str(data.get("evaluated_at", "")),
            evaluator=str(data.get("evaluator", "")),
            case_count=int(data.get("case_count", 0)),
            passed_case_count=int(data.get("passed_case_count", 0)),
            failed_case_count=int(data.get("failed_case_count", 0)),
            forbidden_source_hit_count=int(data.get("forbidden_source_hit_count", 0)),
            unrelated_case_regression_count=int(
                data.get("unrelated_case_regression_count", 0)),
            unsupported_citation_count=int(data.get("unsupported_citation_count", 0)),
            uncited_factual_claim_count=int(data.get("uncited_factual_claim_count", 0)),
            expected_source_recall=float(data.get("expected_source_recall", 0.0)),
            cited_expected_source_recall=float(
                data.get("cited_expected_source_recall", 0.0)),
            passed=bool(data.get("passed", False)),
            threshold_policy_id=str(data.get("threshold_policy_id", "")),
            failure_classifications=tuple(data.get("failure_classifications", ()) or ()),
            evaluation_report_fingerprint=str(
                data.get("evaluation_report_fingerprint", "")),
        )


# ---------------------------------------------------------------------------
# Pack identity + fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgePackIdentity:
    """Deterministic identity of an imported knowledge pack on disk."""

    pack_id: str
    pack_version: str
    source_type: str
    source_id: str
    source_revision: str
    manifest_hash: str
    content_fingerprint: str
    chunk_count: int
    authority_level: str
    intended_use: str
    licence_snapshot: str
    provenance_snapshot: str
    created_at: str
    importer_version: str
    pack_kind: str = "knowledge"

    @property
    def lineage_key(self) -> Tuple[str, str]:
        """The source lineage a pack belongs to (one active revision by default)."""
        return (self.source_type, self.source_id)

    @property
    def is_knowledge_pack(self) -> bool:
        return self.pack_kind == "knowledge"

    def to_dict(self) -> dict:
        return {
            "_record": "knowledge_pack_identity",
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "manifest_hash": self.manifest_hash,
            "content_fingerprint": self.content_fingerprint,
            "chunk_count": self.chunk_count,
            "authority_level": self.authority_level,
            "intended_use": self.intended_use,
            "licence_snapshot": self.licence_snapshot,
            "provenance_snapshot": self.provenance_snapshot,
            "created_at": self.created_at,
            "importer_version": self.importer_version,
            "pack_kind": self.pack_kind,
        }


def compute_manifest_hash(manifest: dict) -> str:
    """Hash a pack manifest deterministically over its parsed content."""
    return MANIFEST_HASH_PREFIX + _sha256_hex(_canonical(manifest))[:20]


def compute_content_fingerprint(*, pack_id: str, pack_version: str,
                                source_id: str, source_revision: str,
                                manifest_hash: str,
                                chunk_ids: Sequence[str],
                                chunk_hashes: Sequence[str]) -> str:
    """``packfp-<sha256(canonical_payload)[:20]>``.

    The fingerprint changes when chunk content, chunk IDs, chunk order, the
    source revision, the manifest identity, or the pack version change. Chunk
    order is preserved (it is meaningful for retrieval lineage).
    """
    payload = {
        "v": ACTIVATION_LAYER_VERSION,
        "pack_id": pack_id,
        "pack_version": pack_version,
        "source_id": source_id,
        "source_revision": source_revision,
        "manifest_hash": manifest_hash,
        "chunk_ids": list(chunk_ids),
        "chunk_hashes": list(chunk_hashes),
    }
    return CONTENT_FP_PREFIX + _sha256_hex(_canonical(payload))[:20]


def _infer_source_type(manifest: dict, record: str) -> str:
    explicit = manifest.get("source_type")
    if explicit:
        return str(explicit)
    if record == "hf_knowledge_pack_manifest" or manifest.get("dataset_id"):
        return "huggingface"
    if record == "pdf_knowledge_pack_manifest" or manifest.get("source_file_hash") \
            or manifest.get("preview_fingerprint"):
        return "pdf_document"
    if record == "manifest" or "default_knowledge_backend" in manifest:
        return "native"
    return "unknown"


def _infer_source_id(manifest: dict) -> str:
    for key in ("source_id", "dataset_id", "source_file_hash"):
        value = manifest.get(key)
        if value:
            return str(value)
    return str(manifest.get("pack_id", ""))


def _infer_source_revision(manifest: dict) -> str:
    for key in ("source_revision", "dataset_revision", "preview_fingerprint"):
        value = manifest.get(key)
        if value:
            return str(value)
    return ""


def _read_chunk_hashes(knowledge_path: Path) -> Tuple[List[str], List[str]]:
    """Return ordered (chunk_ids, content_hashes) from a ``knowledge.jsonl``."""
    chunk_ids: List[str] = []
    chunk_hashes: List[str] = []
    if not knowledge_path.exists():
        return chunk_ids, chunk_hashes
    with knowledge_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed knowledge.jsonl line in {knowledge_path}") from exc
            if record.get("_record") != "chunk":
                continue
            chunk_id = str(record.get("chunk_id", ""))
            content_hash = record.get("content_hash")
            if not content_hash:
                content_hash = "sha256:" + _sha256_hex(
                    str(record.get("chunk_text", "")))
            chunk_ids.append(chunk_id)
            chunk_hashes.append(str(content_hash))
    return chunk_ids, chunk_hashes


def load_pack_manifest(pack_path: Path) -> dict:
    """Read a pack's ``manifest.json`` (accepts a directory or a manifest file)."""
    path = Path(pack_path)
    if path.is_dir():
        path = path / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed manifest at {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return data


def load_pack_identity(pack_dir: str | Path) -> KnowledgePackIdentity:
    """Build a :class:`KnowledgePackIdentity` from a pack directory on disk.

    Reads ``manifest.json`` and ``knowledge.jsonl`` only; never writes, never
    parses the original source, never re-chunks.
    """
    root = Path(pack_dir)
    manifest = load_pack_manifest(root)
    record = str(manifest.get("_record", "manifest"))

    pack_kind = manifest.get("pack_kind")
    if not pack_kind:
        pack_kind = ("eval" if record in EVAL_MANIFEST_RECORDS
                     else "knowledge" if record in KNOWLEDGE_MANIFEST_RECORDS
                     else "native")

    manifest_hash = (str(manifest.get("manifest_hash"))
                     if manifest.get("manifest_hash")
                     else str(manifest.get("pack_hash"))
                     if manifest.get("pack_hash")
                     else compute_manifest_hash(manifest))

    knowledge_path = root / KNOWLEDGE_FILE if root.is_dir() else root.parent / KNOWLEDGE_FILE
    chunk_ids, chunk_hashes = _read_chunk_hashes(knowledge_path)

    pack_id = str(manifest.get("pack_id", root.name if root.is_dir() else ""))
    pack_version = str(manifest.get("pack_version", ""))
    source_id = _infer_source_id(manifest)
    source_revision = _infer_source_revision(manifest)

    content_fp = compute_content_fingerprint(
        pack_id=pack_id, pack_version=pack_version,
        source_id=source_id, source_revision=source_revision,
        manifest_hash=manifest_hash,
        chunk_ids=chunk_ids, chunk_hashes=chunk_hashes,
    )

    chunk_count = manifest.get("chunk_count")
    if chunk_count is None:
        chunk_count = len(chunk_ids)

    authority = str(manifest.get("authority_level")
                    or manifest.get("authority") or "unknown")
    intended_use = str(manifest.get("intended_use")
                       or ("knowledge" if pack_kind == "knowledge" else pack_kind))
    licence = str(manifest.get("permission_or_licence")
                  or manifest.get("licence") or "")
    provenance = str(manifest.get("provenance") or "")

    return KnowledgePackIdentity(
        pack_id=pack_id,
        pack_version=pack_version,
        source_type=_infer_source_type(manifest, record),
        source_id=source_id,
        source_revision=source_revision,
        manifest_hash=manifest_hash,
        content_fingerprint=content_fp,
        chunk_count=int(chunk_count),
        authority_level=authority,
        intended_use=intended_use,
        licence_snapshot=licence,
        provenance_snapshot=provenance,
        created_at=str(manifest.get("created_at", "")),
        importer_version=str(manifest.get("importer_version", "")),
        pack_kind=str(pack_kind),
    )


# ---------------------------------------------------------------------------
# Activation approval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgePackActivationApproval:
    """The structured, fingerprint-bound authorisation to change pack state.

    Approval is never inferred from import, passing tests, a passing evaluation,
    preview readiness, existing pack presence, or source-registry status. It must
    be supplied explicitly and it binds to an exact pack and evaluation.
    """

    approval_id: str
    pack_id: str
    pack_version: str
    pack_fingerprint: str
    manifest_hash: str
    approved_by: str
    approved_at: str
    approval_scope: ActivationScope
    approved_environment: str = "default"
    evaluation_report_id: str = ""
    evaluation_report_fingerprint: str = ""
    evaluation_threshold_policy: str = ""
    acknowledged_nonblocking_findings: Tuple[str, ...] = ()
    activation_notes: str = ""
    expires_at: str = ""
    licence_snapshot: str = ""
    provenance_snapshot: str = ""
    minimum_authority_level: str = ""

    def is_expired(self, now: datetime) -> bool:
        if not self.expires_at:
            return False
        parsed = _parse_dt(self.expires_at)
        if parsed is None:
            return False
        return now > parsed

    def to_dict(self) -> dict:
        return {
            "_record": "knowledge_pack_activation_approval",
            "approval_id": self.approval_id,
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "pack_fingerprint": self.pack_fingerprint,
            "manifest_hash": self.manifest_hash,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "approval_scope": self.approval_scope.value,
            "approved_environment": self.approved_environment,
            "evaluation_report_id": self.evaluation_report_id,
            "evaluation_report_fingerprint": self.evaluation_report_fingerprint,
            "evaluation_threshold_policy": self.evaluation_threshold_policy,
            "acknowledged_nonblocking_findings":
                list(self.acknowledged_nonblocking_findings),
            "activation_notes": self.activation_notes,
            "expires_at": self.expires_at,
            "licence_snapshot": self.licence_snapshot,
            "provenance_snapshot": self.provenance_snapshot,
            "minimum_authority_level": self.minimum_authority_level,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgePackActivationApproval":
        scope_raw = str(data.get("approval_scope", "activate"))
        try:
            scope = ActivationScope(scope_raw)
        except ValueError as exc:
            raise ValueError(f"unknown approval_scope: {scope_raw!r}") from exc
        return cls(
            approval_id=str(data.get("approval_id", "")),
            pack_id=str(data.get("pack_id", "")),
            pack_version=str(data.get("pack_version", "")),
            pack_fingerprint=str(data.get("pack_fingerprint", "")),
            manifest_hash=str(data.get("manifest_hash", "")),
            approved_by=str(data.get("approved_by", "")),
            approved_at=str(data.get("approved_at", "")),
            approval_scope=scope,
            approved_environment=str(data.get("approved_environment", "default")),
            evaluation_report_id=str(data.get("evaluation_report_id", "")),
            evaluation_report_fingerprint=str(
                data.get("evaluation_report_fingerprint", "")),
            evaluation_threshold_policy=str(
                data.get("evaluation_threshold_policy", "")),
            acknowledged_nonblocking_findings=tuple(
                data.get("acknowledged_nonblocking_findings", ()) or ()),
            activation_notes=str(data.get("activation_notes", "")),
            expires_at=str(data.get("expires_at", "")),
            licence_snapshot=str(data.get("licence_snapshot", "")),
            provenance_snapshot=str(data.get("provenance_snapshot", "")),
            minimum_authority_level=str(data.get("minimum_authority_level", "")),
        )


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_activation_approval(path: str | Path) -> KnowledgePackActivationApproval:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return KnowledgePackActivationApproval.from_dict(data)


def load_evaluation_evidence(path: str | Path) -> KnowledgePackEvaluationEvidence:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return KnowledgePackEvaluationEvidence.from_dict(data)


# ---------------------------------------------------------------------------
# Active-pack state + state manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivePackState:
    """One lifecycle/configuration record in the active-state manifest.

    Stores only lifecycle metadata — never pack contents or chunk text.
    """

    pack_id: str
    pack_version: str
    pack_fingerprint: str
    source_type: str
    source_id: str
    source_revision: str
    status: PackLifecycleState
    activated_at: str = ""
    activation_approval_id: str = ""
    evaluation_report_id: str = ""
    previous_active_pack: str = ""
    supersedes: str = ""
    superseded_by: str = ""
    environment: str = "default"
    authority_level: str = ""
    intended_use: str = "knowledge"
    coexistence_policy: CoexistencePolicy = CoexistencePolicy.EXCLUSIVE_LATEST

    @property
    def lineage_key(self) -> Tuple[str, str]:
        return (self.source_type, self.source_id)

    @property
    def is_active(self) -> bool:
        return self.status == PackLifecycleState.ACTIVE

    def _hash_payload(self) -> dict:
        return {
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "pack_fingerprint": self.pack_fingerprint,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "status": self.status.value,
            "activation_approval_id": self.activation_approval_id,
            "evaluation_report_id": self.evaluation_report_id,
            "previous_active_pack": self.previous_active_pack,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "environment": self.environment,
            "coexistence_policy": self.coexistence_policy.value,
        }

    @property
    def lifecycle_record_hash(self) -> str:
        return LIFECYCLE_HASH_PREFIX + _sha256_hex(_canonical(self._hash_payload()))[:20]

    def with_status(self, status: PackLifecycleState, **changes) -> "ActivePackState":
        from dataclasses import replace
        return replace(self, status=status, **changes)

    def to_dict(self) -> dict:
        return {
            "_record": "active_knowledge_pack",
            **self._hash_payload(),
            "activated_at": self.activated_at,
            "authority_level": self.authority_level,
            "intended_use": self.intended_use,
            "lifecycle_record_hash": self.lifecycle_record_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActivePackState":
        return cls(
            pack_id=str(data.get("pack_id", "")),
            pack_version=str(data.get("pack_version", "")),
            pack_fingerprint=str(data.get("pack_fingerprint", "")),
            source_type=str(data.get("source_type", "")),
            source_id=str(data.get("source_id", "")),
            source_revision=str(data.get("source_revision", "")),
            status=PackLifecycleState(str(data.get("status", "inactive"))),
            activated_at=str(data.get("activated_at", "")),
            activation_approval_id=str(data.get("activation_approval_id", "")),
            evaluation_report_id=str(data.get("evaluation_report_id", "")),
            previous_active_pack=str(data.get("previous_active_pack", "")),
            supersedes=str(data.get("supersedes", "")),
            superseded_by=str(data.get("superseded_by", "")),
            environment=str(data.get("environment", "default")),
            authority_level=str(data.get("authority_level", "")),
            intended_use=str(data.get("intended_use", "knowledge")),
            coexistence_policy=CoexistencePolicy(
                str(data.get("coexistence_policy", "exclusive_latest"))),
        )


@dataclass(frozen=True)
class ActivationStateManifest:
    """The current active-state of every governed knowledge pack."""

    records: Tuple[ActivePackState, ...] = ()

    def active(self, *, environment: Optional[str] = None) -> List[ActivePackState]:
        out = [r for r in self.records if r.is_active]
        if environment is not None:
            out = [r for r in out if r.environment == environment]
        return out

    def find(self, pack_id: str) -> Optional[ActivePackState]:
        for record in self.records:
            if record.pack_id == pack_id:
                return record
        return None

    def active_for_lineage(self, lineage_key: Tuple[str, str], *,
                           environment: Optional[str] = None
                           ) -> List[ActivePackState]:
        return [r for r in self.active(environment=environment)
                if r.lineage_key == lineage_key]

    @property
    def state_hash(self) -> str:
        payload = sorted(r.lifecycle_record_hash for r in self.records)
        return STATE_HASH_PREFIX + _sha256_hex(_canonical(payload))[:20]

    def upsert(self, record: ActivePackState) -> "ActivationStateManifest":
        kept = tuple(r for r in self.records if r.pack_id != record.pack_id)
        return ActivationStateManifest(records=kept + (record,))

    def to_records(self) -> List[dict]:
        return [r.to_dict() for r in self.records]

    @classmethod
    def from_records(cls, rows: Sequence[dict]) -> "ActivationStateManifest":
        return cls(records=tuple(
            ActivePackState.from_dict(r) for r in rows
            if r.get("_record") == "active_knowledge_pack"))


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgePackActivationValidation:
    """The outcome of validating a requested activation transition."""

    pack_id: str
    pack_fingerprint: str
    evaluation_fingerprint: str
    requested_scope: ActivationScope
    from_state: PackLifecycleState
    to_state: PackLifecycleState
    findings: Tuple[ActivationFinding, ...]

    @property
    def blocking_findings(self) -> Tuple[ActivationFinding, ...]:
        return tuple(f for f in self.findings if f.blocking)

    @property
    def ok(self) -> bool:
        return not self.blocking_findings

    def to_dict(self) -> dict:
        return {
            "_record": "knowledge_pack_activation_validation",
            "pack_id": self.pack_id,
            "pack_fingerprint": self.pack_fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "requested_scope": self.requested_scope.value,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "ok": self.ok,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


_ALLOWED_TRANSITIONS: Dict[PackLifecycleState, frozenset] = {
    PackLifecycleState.IMPORTED: frozenset({
        PackLifecycleState.EVALUATED,
        PackLifecycleState.EVALUATION_FAILED,
        PackLifecycleState.BLOCKED,
    }),
    PackLifecycleState.EVALUATION_FAILED: frozenset({
        PackLifecycleState.EVALUATED,
        PackLifecycleState.BLOCKED,
    }),
    PackLifecycleState.EVALUATED: frozenset({
        PackLifecycleState.ACTIVATION_PENDING,
        # An explicit activation approval collapses the transient
        # ``activation_pending`` step, so a direct evaluated -> active edge is
        # permitted when an approval is present.
        PackLifecycleState.ACTIVE,
        PackLifecycleState.BLOCKED,
    }),
    PackLifecycleState.ACTIVATION_PENDING: frozenset({
        PackLifecycleState.ACTIVE,
        PackLifecycleState.EVALUATED,
        PackLifecycleState.BLOCKED,
    }),
    PackLifecycleState.ACTIVE: frozenset({
        PackLifecycleState.INACTIVE,
        PackLifecycleState.SUPERSEDED,
        PackLifecycleState.RETIRED,
    }),
    PackLifecycleState.INACTIVE: frozenset({
        PackLifecycleState.ACTIVE,
        PackLifecycleState.RETIRED,
    }),
    PackLifecycleState.SUPERSEDED: frozenset({
        PackLifecycleState.INACTIVE,
        PackLifecycleState.RETIRED,
    }),
    PackLifecycleState.RETIRED: frozenset(),
    PackLifecycleState.BLOCKED: frozenset(),
}


def transition_allowed(from_state: PackLifecycleState,
                       to_state: PackLifecycleState) -> bool:
    """True iff ``from_state -> to_state`` is a permitted lifecycle edge."""
    if from_state == to_state:
        return False
    return to_state in _ALLOWED_TRANSITIONS.get(from_state, frozenset())


# ---------------------------------------------------------------------------
# Conflict / coexistence checks
# ---------------------------------------------------------------------------


def detect_conflicts(identity: KnowledgePackIdentity,
                     state: ActivationStateManifest, *,
                     policy: CoexistencePolicy = CoexistencePolicy.EXCLUSIVE_LATEST,
                     environment: str = "default",
                     ignore_pack_ids: Sequence[str] = ()) -> List[ActivationFinding]:
    """Detect activation conflicts against the current active set."""
    findings: List[ActivationFinding] = []
    ignore = set(ignore_pack_ids)

    if identity.pack_kind != "knowledge":
        findings.append(ActivationFinding(
            ActivationFindingCode.PACK_NOT_KNOWLEDGE,
            f"pack {identity.pack_id!r} is intended for {identity.pack_kind!r}, "
            "not knowledge retrieval"))

    active = [r for r in state.active(environment=environment)
              if r.pack_id not in ignore]

    for record in active:
        if record.pack_fingerprint == identity.content_fingerprint \
                and record.pack_id != identity.pack_id:
            findings.append(ActivationFinding(
                ActivationFindingCode.DUPLICATE_SOURCE_FINGERPRINT,
                f"pack {record.pack_id!r} is already active with the same "
                f"content fingerprint {identity.content_fingerprint}"))

    same_lineage = [r for r in active if r.lineage_key == identity.lineage_key
                    and r.pack_id != identity.pack_id]
    if same_lineage and policy == CoexistencePolicy.EXCLUSIVE_LATEST:
        names = ", ".join(sorted(r.pack_id for r in same_lineage))
        findings.append(ActivationFinding(
            ActivationFindingCode.LINEAGE_CONFLICT_ACTIVE_REVISION,
            f"source lineage {identity.lineage_key} already has an active "
            f"revision ({names}); exclusive_latest permits only one — supersede "
            "the existing revision instead of activating alongside it"))

    if policy == CoexistencePolicy.ENVIRONMENT_SCOPED:
        clash = [r for r in same_lineage if r.environment == environment]
        if clash:
            findings.append(ActivationFinding(
                ActivationFindingCode.COEXISTENCE_NOT_PERMITTED,
                f"environment_scoped policy already has an active revision for "
                f"{identity.lineage_key} in environment {environment!r}"))

    return findings


# ---------------------------------------------------------------------------
# Activation request + validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgePackActivationRequest:
    """Everything needed to validate and (optionally) perform an activation."""

    identity: KnowledgePackIdentity
    approval: KnowledgePackActivationApproval
    evidence: Optional[KnowledgePackEvaluationEvidence] = None
    environment: str = "default"
    coexistence_policy: CoexistencePolicy = CoexistencePolicy.EXCLUSIVE_LATEST
    current_state: PackLifecycleState = PackLifecycleState.EVALUATED


_SCOPE_TARGET_STATE = {
    ActivationScope.ACTIVATE: PackLifecycleState.ACTIVE,
    ActivationScope.REACTIVATE: PackLifecycleState.ACTIVE,
    ActivationScope.SUPERSEDE: PackLifecycleState.ACTIVE,
    ActivationScope.ROLLBACK: PackLifecycleState.ACTIVE,
    ActivationScope.EMERGENCY_DEACTIVATE: PackLifecycleState.INACTIVE,
}


def _authority_rank(level: str) -> int:
    order = {"official": 4, "reputable": 3, "community": 2, "reference": 2,
             "external": 1, "unknown": 0, "": 0}
    return order.get((level or "").lower(), 1)


def _validate_evidence(evidence: Optional[KnowledgePackEvaluationEvidence],
                       identity: KnowledgePackIdentity,
                       approval: KnowledgePackActivationApproval,
                       policy: EvaluationThresholdPolicy,
                       now: datetime) -> List[ActivationFinding]:
    findings: List[ActivationFinding] = []
    if evidence is None:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_NOT_PASSED,
            "no evaluation evidence supplied; a passing evaluation report bound "
            "to this exact pack fingerprint is required to activate"))
        return findings

    if evidence.pack_fingerprint != identity.content_fingerprint:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_PACK_MISMATCH,
            f"evaluation evidence is bound to {evidence.pack_fingerprint}, not "
            f"the pack being activated ({identity.content_fingerprint})"))

    if approval.evaluation_report_fingerprint \
            and approval.evaluation_report_fingerprint != evidence.report_fingerprint:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_FINGERPRINT_MISMATCH,
            "approval is bound to a different evaluation report fingerprint than "
            "the supplied evidence"))

    if approval.evaluation_report_id \
            and approval.evaluation_report_id != evidence.evaluation_report_id:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_FINGERPRINT_MISMATCH,
            "approval names a different evaluation_report_id than the evidence"))

    if policy.require_pass_flag and not evidence.passed:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_NOT_PASSED,
            "evaluation evidence does not record an overall pass"))
    if evidence.failed_case_count > policy.max_failed_cases:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_BLOCKING_FAILURE,
            f"{evidence.failed_case_count} failed case(s) exceed policy maximum "
            f"{policy.max_failed_cases}"))
    if evidence.forbidden_source_hit_count > policy.max_forbidden_source_hits:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_FORBIDDEN_BLEED,
            f"{evidence.forbidden_source_hit_count} forbidden-source hit(s) exceed "
            f"policy maximum {policy.max_forbidden_source_hits}"))
    if evidence.unsupported_citation_count > policy.max_unsupported_citations:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_UNSUPPORTED_CITATION,
            f"{evidence.unsupported_citation_count} unsupported citation(s) exceed "
            f"policy maximum {policy.max_unsupported_citations}"))
    if evidence.uncited_factual_claim_count > policy.max_uncited_factual_claims:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_UNCITED_CLAIM,
            f"{evidence.uncited_factual_claim_count} uncited factual claim(s) exceed "
            f"policy maximum {policy.max_uncited_factual_claims}"))
    if evidence.unrelated_case_regression_count > policy.max_unrelated_regressions:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_REGRESSION,
            f"{evidence.unrelated_case_regression_count} unrelated-case "
            f"regression(s) exceed policy maximum {policy.max_unrelated_regressions}"))
    if evidence.expected_source_recall < policy.min_expected_source_recall:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_RECALL_BELOW_POLICY,
            f"expected source recall {evidence.expected_source_recall} is below "
            f"policy minimum {policy.min_expected_source_recall}"))
    if evidence.cited_expected_source_recall < policy.min_cited_expected_source_recall:
        findings.append(ActivationFinding(
            ActivationFindingCode.EVALUATION_RECALL_BELOW_POLICY,
            f"cited expected source recall {evidence.cited_expected_source_recall} "
            f"is below policy minimum {policy.min_cited_expected_source_recall}"))

    if policy.max_evaluation_age_days is not None and evidence.evaluated_at:
        evaluated = _parse_dt(evidence.evaluated_at)
        if evaluated is not None:
            age_days = (now - evaluated).days
            if age_days > policy.max_evaluation_age_days:
                findings.append(ActivationFinding(
                    ActivationFindingCode.EVALUATION_STALE,
                    f"evaluation is {age_days} day(s) old, beyond the policy "
                    f"maximum of {policy.max_evaluation_age_days}"))
    return findings


def validate_activation(request: KnowledgePackActivationRequest, *,
                        state: ActivationStateManifest,
                        threshold_policy: EvaluationThresholdPolicy =
                        DEFAULT_THRESHOLD_POLICY,
                        now: Optional[datetime] = None
                        ) -> KnowledgePackActivationValidation:
    """Fail-closed validation of an activation/reactivation/supersession request."""
    now = now or datetime.now(timezone.utc)
    identity = request.identity
    approval = request.approval
    findings: List[ActivationFinding] = []

    to_state = _SCOPE_TARGET_STATE.get(
        approval.approval_scope, PackLifecycleState.ACTIVE)

    # --- approval binds to this exact pack ---
    if approval.pack_id != identity.pack_id:
        findings.append(ActivationFinding(
            ActivationFindingCode.PACK_ID_MISMATCH,
            f"approval pack_id {approval.pack_id!r} != pack {identity.pack_id!r}"))
    if approval.pack_version != identity.pack_version:
        findings.append(ActivationFinding(
            ActivationFindingCode.PACK_VERSION_MISMATCH,
            f"approval pack_version {approval.pack_version!r} != "
            f"{identity.pack_version!r}"))
    if approval.pack_fingerprint != identity.content_fingerprint:
        findings.append(ActivationFinding(
            ActivationFindingCode.PACK_FINGERPRINT_MISMATCH,
            "approval pack_fingerprint does not match the pack on disk "
            "(content changed after approval)"))
    if approval.manifest_hash and approval.manifest_hash != identity.manifest_hash:
        findings.append(ActivationFinding(
            ActivationFindingCode.MANIFEST_HASH_MISMATCH,
            "approval manifest_hash does not match the pack manifest on disk"))

    # --- approval scope / environment / expiry ---
    if approval.approval_scope not in (
            ActivationScope.ACTIVATE, ActivationScope.REACTIVATE,
            ActivationScope.SUPERSEDE, ActivationScope.ROLLBACK):
        findings.append(ActivationFinding(
            ActivationFindingCode.APPROVAL_SCOPE_INVALID,
            f"approval scope {approval.approval_scope.value!r} does not permit "
            "an activation"))
    if approval.is_expired(now):
        findings.append(ActivationFinding(
            ActivationFindingCode.APPROVAL_EXPIRED,
            f"approval expired at {approval.expires_at}"))
    if approval.approved_environment != request.environment:
        findings.append(ActivationFinding(
            ActivationFindingCode.APPROVAL_ENVIRONMENT_MISMATCH,
            f"approval environment {approval.approved_environment!r} != requested "
            f"{request.environment!r}"))

    # --- licence / provenance drift after approval ---
    if approval.licence_snapshot \
            and approval.licence_snapshot != identity.licence_snapshot:
        findings.append(ActivationFinding(
            ActivationFindingCode.LICENCE_CHANGED,
            "licence changed since approval"))
    if approval.provenance_snapshot \
            and approval.provenance_snapshot != identity.provenance_snapshot:
        findings.append(ActivationFinding(
            ActivationFindingCode.PROVENANCE_CHANGED,
            "provenance changed since approval"))
    if approval.minimum_authority_level \
            and _authority_rank(identity.authority_level) \
            < _authority_rank(approval.minimum_authority_level):
        findings.append(ActivationFinding(
            ActivationFindingCode.AUTHORITY_BELOW_POLICY,
            f"authority {identity.authority_level!r} is below required "
            f"{approval.minimum_authority_level!r}"))

    # --- pack must be a knowledge pack, not blocked/retired ---
    if identity.pack_kind != "knowledge":
        findings.append(ActivationFinding(
            ActivationFindingCode.PACK_NOT_KNOWLEDGE,
            f"pack is intended for {identity.pack_kind!r}, not knowledge use"))
    if request.current_state in (PackLifecycleState.BLOCKED,
                                 PackLifecycleState.RETIRED):
        findings.append(ActivationFinding(
            ActivationFindingCode.PACK_BLOCKED_OR_RETIRED,
            f"pack is {request.current_state.value}; cannot be activated without "
            "an explicit separate policy"))

    # --- state transition must be legal ---
    if not transition_allowed(request.current_state, to_state):
        findings.append(ActivationFinding(
            ActivationFindingCode.INVALID_STATE_TRANSITION,
            f"{request.current_state.value} -> {to_state.value} is not a permitted "
            "lifecycle transition"))

    # --- evaluation evidence (only for forward activations) ---
    if to_state == PackLifecycleState.ACTIVE \
            and approval.approval_scope != ActivationScope.ROLLBACK:
        findings.extend(_validate_evidence(
            request.evidence, identity, approval, threshold_policy, now))

    # --- conflict / coexistence ---
    if to_state == PackLifecycleState.ACTIVE:
        findings.extend(detect_conflicts(
            identity, state, policy=request.coexistence_policy,
            environment=request.environment,
            ignore_pack_ids=(identity.pack_id,)))

    evidence_fp = (request.evidence.report_fingerprint
                   if request.evidence is not None else "")
    return KnowledgePackActivationValidation(
        pack_id=identity.pack_id,
        pack_fingerprint=identity.content_fingerprint,
        evaluation_fingerprint=evidence_fp,
        requested_scope=approval.approval_scope,
        from_state=request.current_state,
        to_state=to_state,
        findings=tuple(findings),
    )


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationAuditRecord:
    """One immutable entry in the append-only activation audit log."""

    action: AuditAction
    actor: str
    timestamp: str
    previous_state_hash: str
    requested_state: str
    resulting_state_hash: str
    approval_id: str
    pack_fingerprint: str
    evaluation_fingerprint: str
    reason: str
    success: bool
    failure_findings: Tuple[str, ...]
    resulting_state_records: Tuple[dict, ...]

    def _hash_payload(self) -> dict:
        return {
            "action": self.action.value,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "previous_state_hash": self.previous_state_hash,
            "requested_state": self.requested_state,
            "resulting_state_hash": self.resulting_state_hash,
            "approval_id": self.approval_id,
            "pack_fingerprint": self.pack_fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "reason": self.reason,
            "success": self.success,
            "failure_findings": list(self.failure_findings),
            "resulting_state_records": list(self.resulting_state_records),
        }

    @property
    def audit_record_hash(self) -> str:
        return AUDIT_HASH_PREFIX + _sha256_hex(_canonical(self._hash_payload()))[:20]

    def to_dict(self) -> dict:
        return {
            "_record": "knowledge_pack_activation_audit",
            **self._hash_payload(),
            "audit_record_hash": self.audit_record_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActivationAuditRecord":
        return cls(
            action=AuditAction(str(data.get("action", "activate"))),
            actor=str(data.get("actor", "")),
            timestamp=str(data.get("timestamp", "")),
            previous_state_hash=str(data.get("previous_state_hash", "")),
            requested_state=str(data.get("requested_state", "")),
            resulting_state_hash=str(data.get("resulting_state_hash", "")),
            approval_id=str(data.get("approval_id", "")),
            pack_fingerprint=str(data.get("pack_fingerprint", "")),
            evaluation_fingerprint=str(data.get("evaluation_fingerprint", "")),
            reason=str(data.get("reason", "")),
            success=bool(data.get("success", False)),
            failure_findings=tuple(data.get("failure_findings", ()) or ()),
            resulting_state_records=tuple(data.get("resulting_state_records", ()) or ()),
        )


# ---------------------------------------------------------------------------
# Operation result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgePackActivationRecord:
    """The result of an activate/reactivate/supersede operation."""

    success: bool
    written: bool
    action: AuditAction
    validation: KnowledgePackActivationValidation
    state: ActivationStateManifest
    audit: Optional[ActivationAuditRecord]
    message: str

    def to_dict(self) -> dict:
        return {
            "_record": "knowledge_pack_activation_record",
            "success": self.success,
            "written": self.written,
            "action": self.action.value,
            "message": self.message,
            "validation": self.validation.to_dict(),
            "active_packs": [r.pack_id for r in self.state.active()],
            "state_hash": self.state.state_hash,
            "audit_record_hash": self.audit.audit_record_hash if self.audit else "",
        }


@dataclass(frozen=True)
class KnowledgePackDeactivationRecord:
    """The result of a (normal or emergency) deactivation."""

    success: bool
    written: bool
    emergency: bool
    pack_id: str
    state: ActivationStateManifest
    audit: Optional[ActivationAuditRecord]
    findings: Tuple[ActivationFinding, ...]
    message: str

    def to_dict(self) -> dict:
        return {
            "_record": "knowledge_pack_deactivation_record",
            "success": self.success,
            "written": self.written,
            "emergency": self.emergency,
            "pack_id": self.pack_id,
            "message": self.message,
            "findings": [f.to_dict() for f in self.findings],
            "state_hash": self.state.state_hash,
            "audit_record_hash": self.audit.audit_record_hash if self.audit else "",
        }


@dataclass(frozen=True)
class KnowledgePackRollbackRequest:
    """A request to restore a previously recorded active state."""

    target_state_hash: str
    current_state_hash: str
    approval: KnowledgePackActivationApproval
    reason: str = ""


@dataclass(frozen=True)
class KnowledgePackRollbackResult:
    """The result of a rollback operation."""

    success: bool
    written: bool
    target_state_hash: str
    state: ActivationStateManifest
    audit: Optional[ActivationAuditRecord]
    findings: Tuple[ActivationFinding, ...]
    message: str

    def to_dict(self) -> dict:
        return {
            "_record": "knowledge_pack_rollback_result",
            "success": self.success,
            "written": self.written,
            "target_state_hash": self.target_state_hash,
            "message": self.message,
            "findings": [f.to_dict() for f in self.findings],
            "state_hash": self.state.state_hash,
            "active_packs": [r.pack_id for r in self.state.active()],
            "audit_record_hash": self.audit.audit_record_hash if self.audit else "",
        }


@dataclass(frozen=True)
class KnowledgePackSupersessionRecord:
    """The result of an atomic supersession (new active, old superseded)."""

    success: bool
    written: bool
    old_pack_id: str
    new_pack_id: str
    validation: KnowledgePackActivationValidation
    state: ActivationStateManifest
    audit: Optional[ActivationAuditRecord]
    message: str

    def to_dict(self) -> dict:
        return {
            "_record": "knowledge_pack_supersession_record",
            "success": self.success,
            "written": self.written,
            "old_pack_id": self.old_pack_id,
            "new_pack_id": self.new_pack_id,
            "message": self.message,
            "validation": self.validation.to_dict(),
            "active_packs": [r.pack_id for r in self.state.active()],
            "state_hash": self.state.state_hash,
            "audit_record_hash": self.audit.audit_record_hash if self.audit else "",
        }


# ---------------------------------------------------------------------------
# State manager (the only component that writes governed state)
# ---------------------------------------------------------------------------


def _atomic_write_lines(path: Path, lines: Sequence[str]) -> None:
    """Write ``lines`` to ``path`` atomically (temp file in same dir + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=path.name + ".", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


class ActivationStateManager:
    """Reads/writes the governed active-state manifest and audit log.

    Every durable change is dry-run by default and only persisted when the
    caller passes ``write=True``. The current active-state manifest is a small
    overwrite-on-change file; the audit log is strictly append-only.
    """

    def __init__(self, *, state_path: str | Path = DEFAULT_STATE_PATH,
                 audit_path: str | Path = DEFAULT_AUDIT_PATH):
        self.state_path = Path(state_path)
        self.audit_path = Path(audit_path)

    # -- load --

    def load_state(self) -> ActivationStateManifest:
        if not self.state_path.exists():
            return ActivationStateManifest()
        rows: List[dict] = []
        with self.state_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append(json.loads(line))
        return ActivationStateManifest.from_records(rows)

    def load_audit(self) -> List[ActivationAuditRecord]:
        if not self.audit_path.exists():
            return []
        out: List[ActivationAuditRecord] = []
        with self.audit_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out.append(ActivationAuditRecord.from_dict(json.loads(line)))
        return out

    # -- persist --

    def _write_state(self, state: ActivationStateManifest) -> None:
        lines = [_canonical(r.to_dict()) for r in
                 sorted(state.records, key=lambda r: r.pack_id)]
        _atomic_write_lines(self.state_path, lines)

    def _append_audit(self, record: ActivationAuditRecord) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical(record.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _commit(self, new_state: ActivationStateManifest,
                audit: ActivationAuditRecord, *, write: bool) -> None:
        if not write:
            return
        self._write_state(new_state)
        self._append_audit(audit)

    # -- operations --

    def activate(self, request: KnowledgePackActivationRequest, *,
                 actor: str = "operator", write: bool = False,
                 threshold_policy: EvaluationThresholdPolicy =
                 DEFAULT_THRESHOLD_POLICY,
                 reason: str = "",
                 now: Optional[datetime] = None) -> KnowledgePackActivationRecord:
        now = now or datetime.now(timezone.utc)
        state = self.load_state()
        existing = state.find(request.identity.pack_id)
        current = existing.status if existing is not None else request.current_state
        request = _with_current_state(request, current)
        validation = validate_activation(
            request, state=state, threshold_policy=threshold_policy, now=now)

        action = (AuditAction.ACTIVATE
                  if request.approval.approval_scope != ActivationScope.REACTIVATE
                  else AuditAction.ACTIVATE)

        if not validation.ok:
            audit = self._build_audit(
                action, actor, now, state, state, request.approval,
                request.identity.content_fingerprint, validation.evaluation_fingerprint,
                reason or "activation rejected", success=False, findings=validation)
            self._commit(state, audit, write=False)  # never persist a failure's state
            if write:
                self._append_audit(audit)
            return KnowledgePackActivationRecord(
                success=False, written=False, action=action, validation=validation,
                state=state, audit=audit,
                message="activation blocked: " + _summarise(validation))

        record = ActivePackState(
            pack_id=request.identity.pack_id,
            pack_version=request.identity.pack_version,
            pack_fingerprint=request.identity.content_fingerprint,
            source_type=request.identity.source_type,
            source_id=request.identity.source_id,
            source_revision=request.identity.source_revision,
            status=PackLifecycleState.ACTIVE,
            activated_at=now.isoformat(),
            activation_approval_id=request.approval.approval_id,
            evaluation_report_id=(request.evidence.evaluation_report_id
                                  if request.evidence else ""),
            previous_active_pack=existing.pack_id if existing else "",
            environment=request.environment,
            authority_level=request.identity.authority_level,
            intended_use=request.identity.intended_use,
            coexistence_policy=request.coexistence_policy,
        )
        new_state = state.upsert(record)
        audit = self._build_audit(
            action, actor, now, state, new_state, request.approval,
            request.identity.content_fingerprint, validation.evaluation_fingerprint,
            reason or "pack activated", success=True, findings=validation)
        self._commit(new_state, audit, write=write)
        return KnowledgePackActivationRecord(
            success=True, written=write, action=action, validation=validation,
            state=new_state, audit=audit,
            message=("activated" if write else "dry-run: validation passed, "
                     "nothing written"))

    def deactivate(self, pack_id: str, *,
                   approval: Optional[KnowledgePackActivationApproval] = None,
                   actor: str = "operator", emergency: bool = False,
                   reason: str = "", write: bool = False,
                   now: Optional[datetime] = None
                   ) -> KnowledgePackDeactivationRecord:
        now = now or datetime.now(timezone.utc)
        state = self.load_state()
        existing = state.find(pack_id)
        findings: List[ActivationFinding] = []

        if existing is None:
            findings.append(ActivationFinding(
                ActivationFindingCode.PACK_NOT_FOUND,
                f"no governed state for pack {pack_id!r}"))
        elif not existing.is_active:
            findings.append(ActivationFinding(
                ActivationFindingCode.PACK_NOT_ACTIVE,
                f"pack {pack_id!r} is {existing.status.value}, not active"))

        # Emergency deactivation may bypass evaluation prerequisites but still
        # requires an explicit actor + reason. Normal deactivation requires an
        # approval scoped to emergency_deactivate or reactivate-family removal.
        if emergency:
            if not actor.strip() or not reason.strip():
                findings.append(ActivationFinding(
                    ActivationFindingCode.APPROVAL_SCOPE_INVALID,
                    "emergency deactivation requires an explicit actor and reason"))
        else:
            if approval is None:
                findings.append(ActivationFinding(
                    ActivationFindingCode.APPROVAL_SCOPE_INVALID,
                    "deactivation requires an approval (or use emergency mode "
                    "with an explicit actor and reason)"))
            elif approval.approval_scope not in (
                    ActivationScope.EMERGENCY_DEACTIVATE,):
                findings.append(ActivationFinding(
                    ActivationFindingCode.APPROVAL_SCOPE_INVALID,
                    f"approval scope {approval.approval_scope.value!r} does not "
                    "permit deactivation"))
            elif approval.pack_id != pack_id:
                findings.append(ActivationFinding(
                    ActivationFindingCode.PACK_ID_MISMATCH,
                    "approval is for a different pack"))

        action = (AuditAction.EMERGENCY_DEACTIVATE if emergency
                  else AuditAction.DEACTIVATE)
        blocking = [f for f in findings if f.blocking]
        if blocking:
            audit = self._build_audit(
                action, actor, now, state, state, approval,
                existing.pack_fingerprint if existing else "", "",
                reason or "deactivation rejected", success=False,
                findings_list=findings)
            if write:
                self._append_audit(audit)
            return KnowledgePackDeactivationRecord(
                success=False, written=False, emergency=emergency, pack_id=pack_id,
                state=state, audit=audit, findings=tuple(findings),
                message="deactivation blocked: " + "; ".join(
                    f.message for f in blocking))

        assert existing is not None
        new_record = existing.with_status(
            PackLifecycleState.INACTIVE, activated_at=existing.activated_at)
        new_state = state.upsert(new_record)
        audit = self._build_audit(
            action, actor, now, state, new_state, approval,
            existing.pack_fingerprint, "",
            reason or ("emergency deactivation" if emergency else "deactivated"),
            success=True, findings_list=findings)
        self._commit(new_state, audit, write=write)
        return KnowledgePackDeactivationRecord(
            success=True, written=write, emergency=emergency, pack_id=pack_id,
            state=new_state, audit=audit, findings=tuple(findings),
            message=("deactivated" if write else "dry-run: would deactivate"))

    def supersede(self, *, old_pack_id: str,
                  request: KnowledgePackActivationRequest,
                  actor: str = "operator", write: bool = False,
                  threshold_policy: EvaluationThresholdPolicy =
                  DEFAULT_THRESHOLD_POLICY, reason: str = "",
                  now: Optional[datetime] = None
                  ) -> KnowledgePackSupersessionRecord:
        now = now or datetime.now(timezone.utc)
        state = self.load_state()
        old = state.find(old_pack_id)
        new_identity = request.identity

        # Validate the new pack ignoring the old pack as a lineage conflict — it
        # is being replaced, not coexisted with.
        current = PackLifecycleState.EVALUATED
        request = _with_current_state(request, current)
        validation = validate_activation(
            request, state=_without(state, old_pack_id),
            threshold_policy=threshold_policy, now=now)

        extra: List[ActivationFinding] = []
        if old is None or not old.is_active:
            extra.append(ActivationFinding(
                ActivationFindingCode.PACK_NOT_ACTIVE,
                f"old pack {old_pack_id!r} is not currently active"))
        elif old.lineage_key != new_identity.lineage_key:
            extra.append(ActivationFinding(
                ActivationFindingCode.LINEAGE_CONFLICT_ACTIVE_REVISION,
                "supersession requires matching source lineage between the old "
                "and new pack"))

        all_findings = tuple(validation.findings) + tuple(extra)
        merged = KnowledgePackActivationValidation(
            pack_id=validation.pack_id, pack_fingerprint=validation.pack_fingerprint,
            evaluation_fingerprint=validation.evaluation_fingerprint,
            requested_scope=validation.requested_scope,
            from_state=validation.from_state, to_state=validation.to_state,
            findings=all_findings)

        if not merged.ok:
            audit = self._build_audit(
                AuditAction.SUPERSEDE, actor, now, state, state, request.approval,
                new_identity.content_fingerprint, validation.evaluation_fingerprint,
                reason or "supersession rejected", success=False, findings=merged)
            if write:
                self._append_audit(audit)
            return KnowledgePackSupersessionRecord(
                success=False, written=False, old_pack_id=old_pack_id,
                new_pack_id=new_identity.pack_id, validation=merged, state=state,
                audit=audit, message="supersession blocked: " + _summarise(merged))

        # Atomic single transition: new -> active, old -> superseded.
        assert old is not None
        new_record = ActivePackState(
            pack_id=new_identity.pack_id, pack_version=new_identity.pack_version,
            pack_fingerprint=new_identity.content_fingerprint,
            source_type=new_identity.source_type, source_id=new_identity.source_id,
            source_revision=new_identity.source_revision,
            status=PackLifecycleState.ACTIVE, activated_at=now.isoformat(),
            activation_approval_id=request.approval.approval_id,
            evaluation_report_id=(request.evidence.evaluation_report_id
                                  if request.evidence else ""),
            previous_active_pack=old_pack_id, supersedes=old_pack_id,
            environment=request.environment,
            authority_level=new_identity.authority_level,
            intended_use=new_identity.intended_use,
            coexistence_policy=request.coexistence_policy)
        old_record = old.with_status(
            PackLifecycleState.SUPERSEDED, superseded_by=new_identity.pack_id)
        new_state = state.upsert(old_record).upsert(new_record)
        audit = self._build_audit(
            AuditAction.SUPERSEDE, actor, now, state, new_state, request.approval,
            new_identity.content_fingerprint, validation.evaluation_fingerprint,
            reason or f"{new_identity.pack_id} supersedes {old_pack_id}",
            success=True, findings=merged)
        self._commit(new_state, audit, write=write)
        return KnowledgePackSupersessionRecord(
            success=True, written=write, old_pack_id=old_pack_id,
            new_pack_id=new_identity.pack_id, validation=merged, state=new_state,
            audit=audit,
            message=("superseded" if write else "dry-run: would supersede"))

    def rollback(self, request: KnowledgePackRollbackRequest, *,
                 available_pack_dirs: Optional[Dict[str, str]] = None,
                 actor: str = "operator", write: bool = False,
                 now: Optional[datetime] = None) -> KnowledgePackRollbackResult:
        now = now or datetime.now(timezone.utc)
        state = self.load_state()
        findings: List[ActivationFinding] = []

        if request.approval.approval_scope != ActivationScope.ROLLBACK:
            findings.append(ActivationFinding(
                ActivationFindingCode.APPROVAL_SCOPE_INVALID,
                "rollback requires an approval scoped to rollback"))
        if request.current_state_hash != state.state_hash:
            findings.append(ActivationFinding(
                ActivationFindingCode.STALE_ROLLBACK_APPROVAL,
                "current active state changed after the rollback approval was "
                "issued; re-approve against the new state"))

        target = find_state_by_hash(self.load_audit(), request.target_state_hash)
        if target is None:
            findings.append(ActivationFinding(
                ActivationFindingCode.ROLLBACK_TARGET_MISSING,
                f"no recorded state matches target hash "
                f"{request.target_state_hash}"))
        else:
            for rec in target.active():
                if available_pack_dirs is not None \
                        and rec.pack_id not in available_pack_dirs:
                    findings.append(ActivationFinding(
                        ActivationFindingCode.ROLLBACK_TARGET_MISSING,
                        f"target pack {rec.pack_id!r} no longer exists on disk"))
                    continue
                if available_pack_dirs is not None:
                    identity = load_pack_identity(available_pack_dirs[rec.pack_id])
                    if identity.content_fingerprint != rec.pack_fingerprint:
                        findings.append(ActivationFinding(
                            ActivationFindingCode.ROLLBACK_TARGET_INVALID,
                            f"target pack {rec.pack_id!r} fingerprint changed; "
                            "cannot restore a different pack"))

        blocking = [f for f in findings if f.blocking]
        if blocking or target is None:
            audit = self._build_audit(
                AuditAction.ROLLBACK, actor, now, state, state, request.approval,
                "", "", request.reason or "rollback rejected", success=False,
                findings_list=findings)
            if write:
                self._append_audit(audit)
            return KnowledgePackRollbackResult(
                success=False, written=False,
                target_state_hash=request.target_state_hash, state=state,
                audit=audit, findings=tuple(findings),
                message="rollback blocked: " + "; ".join(
                    f.message for f in blocking) if blocking
                else "rollback blocked: target state not found")

        new_state = target
        audit = self._build_audit(
            AuditAction.ROLLBACK, actor, now, state, new_state, request.approval,
            "", "", request.reason or f"rollback to {request.target_state_hash}",
            success=True, findings_list=findings)
        self._commit(new_state, audit, write=write)
        return KnowledgePackRollbackResult(
            success=True, written=write,
            target_state_hash=request.target_state_hash, state=new_state,
            audit=audit, findings=tuple(findings),
            message=("rolled back" if write else "dry-run: would roll back"))

    # -- audit construction --

    def _build_audit(self, action: AuditAction, actor: str, now: datetime,
                     prev: ActivationStateManifest, result: ActivationStateManifest,
                     approval: Optional[KnowledgePackActivationApproval],
                     pack_fp: str, eval_fp: str, reason: str, *, success: bool,
                     findings: Optional[KnowledgePackActivationValidation] = None,
                     findings_list: Optional[Sequence[ActivationFinding]] = None
                     ) -> ActivationAuditRecord:
        if findings is not None:
            fail = [f.code.value for f in findings.blocking_findings]
        elif findings_list is not None:
            fail = [f.code.value for f in findings_list if f.blocking]
        else:
            fail = []
        return ActivationAuditRecord(
            action=action, actor=actor, timestamp=now.isoformat(),
            previous_state_hash=prev.state_hash,
            requested_state=action.value,
            resulting_state_hash=result.state_hash,
            approval_id=approval.approval_id if approval else "",
            pack_fingerprint=pack_fp, evaluation_fingerprint=eval_fp,
            reason=reason, success=success, failure_findings=tuple(fail),
            resulting_state_records=tuple(result.to_records()))


def _with_current_state(request: KnowledgePackActivationRequest,
                        current: PackLifecycleState
                        ) -> KnowledgePackActivationRequest:
    from dataclasses import replace
    return replace(request, current_state=current)


def _without(state: ActivationStateManifest, pack_id: str
             ) -> ActivationStateManifest:
    return ActivationStateManifest(
        records=tuple(r for r in state.records if r.pack_id != pack_id))


def _summarise(validation: KnowledgePackActivationValidation) -> str:
    return "; ".join(f.message for f in validation.blocking_findings) \
        or "no blocking findings"


def find_state_by_hash(audit: Sequence[ActivationAuditRecord],
                       target_hash: str) -> Optional[ActivationStateManifest]:
    """Reconstruct a prior active-state manifest from the audit log by hash."""
    for record in reversed(list(audit)):
        if record.success and record.resulting_state_hash == target_hash:
            return ActivationStateManifest.from_records(
                list(record.resulting_state_records))
    return None


# ---------------------------------------------------------------------------
# Retrieval integration — pack-selection boundary only
# ---------------------------------------------------------------------------


def select_active_pack_ids(state: ActivationStateManifest, *,
                           environment: Optional[str] = None) -> List[str]:
    """The pack IDs retrieval is permitted to load (active + env-matched only).

    Ignores imported-but-inactive, superseded, retired and blocked packs. This
    is the *only* adapter retrieval consults; retrieval algorithms are unchanged.
    """
    return sorted(r.pack_id for r in state.active(environment=environment))


def select_active_pack_dirs(state: ActivationStateManifest,
                            packs_root: str | Path, *,
                            environment: Optional[str] = None) -> List[Path]:
    """Resolve active pack IDs to on-disk directories under ``packs_root``."""
    root = Path(packs_root)
    out: List[Path] = []
    for pack_id in select_active_pack_ids(state, environment=environment):
        candidate = root / pack_id
        if (candidate / MANIFEST_FILE).exists():
            out.append(candidate)
    return out


# ---------------------------------------------------------------------------
# Markdown renderers (deterministic; no sensitive content)
# ---------------------------------------------------------------------------


def render_state_markdown(state: ActivationStateManifest) -> str:
    lines = ["# Active knowledge-pack state", "",
             f"State hash: `{state.state_hash}`", ""]
    if not state.records:
        lines.append("_No governed knowledge packs._")
        return "\n".join(lines)
    lines.append("| Pack | Version | Status | Source | Revision | Fingerprint |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in sorted(state.records, key=lambda x: x.pack_id):
        lines.append(
            f"| {r.pack_id} | {r.pack_version} | {r.status.value} | "
            f"{r.source_type}:{r.source_id} | {r.source_revision} | "
            f"`{r.pack_fingerprint}` |")
    return "\n".join(lines)


def render_history_markdown(audit: Sequence[ActivationAuditRecord]) -> str:
    lines = ["# Knowledge-pack activation history", ""]
    if not audit:
        lines.append("_No activation history._")
        return "\n".join(lines)
    lines.append("| Timestamp | Action | Actor | Success | Result state | Reason |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in audit:
        lines.append(
            f"| {r.timestamp} | {r.action.value} | {r.actor} | {r.success} | "
            f"`{r.resulting_state_hash}` | {r.reason} |")
    return "\n".join(lines)


def render_validation_markdown(validation: KnowledgePackActivationValidation) -> str:
    lines = [
        "# Activation validation",
        "",
        f"- Pack: `{validation.pack_id}`",
        f"- Pack fingerprint: `{validation.pack_fingerprint}`",
        f"- Evaluation fingerprint: `{validation.evaluation_fingerprint}`",
        f"- Requested scope: {validation.requested_scope.value}",
        f"- Transition: {validation.from_state.value} -> {validation.to_state.value}",
        f"- Result: {'OK' if validation.ok else 'BLOCKED'}",
        "",
    ]
    if validation.findings:
        lines.append("## Findings")
        for f in validation.findings:
            mark = "x" if f.blocking else " "
            lines.append(f"- [{mark}] **{f.code.value}** ({f.severity.value}): {f.message}")
    else:
        lines.append("_No findings._")
    return "\n".join(lines)
