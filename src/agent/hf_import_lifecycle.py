"""Governed Hugging Face import lifecycle — Phase A/B (v6.9).

This module is the *front of* the governed Hugging Face ingestion path. It does
two things and nothing more:

* **Phase A — explicit, scoped, revision-bound approval contracts.** A dataset
  may only be imported under a :class:`HFDatasetApproval` that names an exact
  dataset id, an exact revision, and the deterministic *metadata fingerprint*
  (:func:`compute_metadata_fingerprint`) of the card it was approved against. An
  approval declares its scope (``eval_only`` / ``knowledge_only`` /
  ``eval_and_knowledge``), the approved splits and columns, a row limit, and a
  snapshot of the licence and provenance it was granted against.
  :func:`validate_import_request` is **fail-closed**: it returns invalid the
  moment the dataset id, revision, fingerprint, licence, provenance, columns,
  split, row limit, or intended use drifts from what was approved, the approval
  has expired, the underlying v6.2A intake decision is not an approval, or a
  knowledge import is attempted under an eval-only approval.

* **Phase B — bounded, offline-first row sampling.** :func:`sample_rows` reads at
  most the approved number of rows from a local JSONL fixture, *streaming line by
  line* so the full file is never materialised, stopping the instant the cap is
  reached. It records per-row access provenance (dataset id, revision, split,
  source index, retrieval timestamp, adapter version, streaming flag, a row
  content hash, and a schema fingerprint). It never reaches the network.

What this module deliberately does **not** do (and cannot, by construction — it
imports none of the relevant writers):

* approve a dataset (approvals are human-authored records this module only reads);
* treat a v6.2A assessment as an import approval;
* treat ``approved_for_eval`` as ``approved_for_knowledge``;
* download an unbounded dataset or materialise a large dataset in memory;
* normalise, inspect, or import rows (those are later phases);
* write the ``MemoryLedger``, the source registry, a proposal, a retrieval index,
  or a knowledge/eval pack;
* call any LLM.

The only durable write here is :func:`write_sample_result`, which writes a single
sample-result JSON to a caller-supplied path.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from agent.data_intake import DataIntakeAssessment, DataIntakeDecision
from agent.hf_data_adapter import HuggingFaceDatasetMetadata

LIFECYCLE_VERSION = "v6.9"
ADAPTER_VERSION = "hf-lifecycle-v6.9"

# Bounded sampling defaults. The point of this path is small governed samples,
# not bulk ingestion: a request may never exceed HARD_MAX_ROWS, and the default
# sample is DEFAULT_SAMPLE_ROWS.
DEFAULT_SAMPLE_ROWS = 100
HARD_MAX_ROWS = 1000

FINGERPRINT_PREFIX = "hfmeta-"
ASSESSMENT_FINGERPRINT_PREFIX = "hfintake-"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _as_tuple(value) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    return tuple(str(v).strip() for v in value if str(v).strip())


def _opt_str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class HFApprovalScope(str, Enum):
    """What an approval authorises. Eval and knowledge are deliberately split."""

    EVAL_ONLY = "eval_only"
    KNOWLEDGE_ONLY = "knowledge_only"
    EVAL_AND_KNOWLEDGE = "eval_and_knowledge"


class HFImportIntent(str, Enum):
    """The intended landing tier of a single import request."""

    EVAL = "eval"
    KNOWLEDGE = "knowledge"


class HFValidationCode(str, Enum):
    """Fail-closed reasons an import request does not match its approval."""

    APPROVAL_VALID = "approval_valid"
    DATASET_ID_MISMATCH = "dataset_id_mismatch"
    REVISION_MISMATCH = "revision_mismatch"
    METADATA_FINGERPRINT_MISMATCH = "metadata_fingerprint_mismatch"
    INTAKE_FINGERPRINT_MISMATCH = "intake_fingerprint_mismatch"
    LICENCE_CHANGED = "licence_changed"
    PROVENANCE_CHANGED = "provenance_changed"
    COLUMNS_NOT_APPROVED = "columns_not_approved"
    SPLIT_NOT_APPROVED = "split_not_approved"
    ROW_LIMIT_EXCEEDED = "row_limit_exceeded"
    INTENDED_USE_MISMATCH = "intended_use_mismatch"
    STREAMING_NOT_APPROVED = "streaming_not_approved"
    APPROVAL_EXPIRED = "approval_expired"
    INTAKE_NOT_APPROVED = "intake_not_approved"
    KNOWLEDGE_REQUIRES_KNOWLEDGE_APPROVAL = "knowledge_requires_knowledge_approval"
    EVAL_REQUIRES_EVAL_APPROVAL = "eval_requires_eval_approval"


class HFSampleStatus(str, Enum):
    """The outcome of a bounded sampling attempt."""

    SAMPLE_READY = "sample_ready"
    ROW_LIMIT_REACHED = "row_limit_reached"
    SAMPLE_BLOCKED = "sample_blocked"
    SCHEMA_MISMATCH = "schema_mismatch"
    SOURCE_UNAVAILABLE = "source_unavailable"
    REVISION_UNAVAILABLE = "revision_unavailable"


# --------------------------------------------------------------------------- #
# Metadata fingerprinting (Phase A)
# --------------------------------------------------------------------------- #


def canonical_metadata_payload(
    metadata: HuggingFaceDatasetMetadata,
    *,
    assessment: Optional[DataIntakeAssessment] = None,
) -> Dict[str, object]:
    """The deterministic inputs a metadata fingerprint is computed over.

    Binds the approval to the *declared identity and governance posture* of the
    card: id, revision-relevant fields, licence, provenance markers (gated /
    private), publisher, task categories, language, schema/feature declaration,
    intended use, and — when supplied — the v6.2A intake finding codes. Volatile
    popularity counters (downloads, likes) are deliberately excluded so a
    fingerprint tracks governance, not fashion.
    """
    schema_keys: Tuple[str, ...] = ()
    if isinstance(metadata.dataset_info, dict):
        features = metadata.dataset_info.get("features")
        if isinstance(features, dict):
            schema_keys = tuple(sorted(str(k) for k in features))
        elif isinstance(features, list):
            schema_keys = tuple(sorted(
                str(f.get("name")) for f in features
                if isinstance(f, dict) and f.get("name")))

    payload: Dict[str, object] = {
        "dataset_id": metadata.dataset_id,
        "licence": (metadata.license or "").strip().lower(),
        "task_categories": sorted(metadata.task_categories),
        "language": sorted(metadata.language),
        "size_categories": sorted(metadata.size_categories),
        "gated": bool(metadata.gated),
        "private": bool(metadata.private),
        "contains_personal_data": metadata.contains_personal_data,
        "schema_keys": list(schema_keys),
    }
    if assessment is not None:
        payload["intake_decision"] = assessment.decision.value
        payload["intake_findings"] = sorted(
            f.code.value for f in assessment.findings)
    return payload


def compute_metadata_fingerprint(
    metadata: HuggingFaceDatasetMetadata,
    *,
    assessment: Optional[DataIntakeAssessment] = None,
) -> str:
    """``hfmeta-<sha256(canonical_payload)[:16]>`` — deterministic, offline."""
    payload = canonical_metadata_payload(metadata, assessment=assessment)
    return FINGERPRINT_PREFIX + _sha256_hex(_canonical_json(payload))[:16]


def compute_assessment_fingerprint(assessment: DataIntakeAssessment) -> str:
    """A deterministic fingerprint of a v6.2A intake assessment.

    Binds only the *stable governance posture* — candidate identity, decision,
    lane, and the sorted finding codes — and deliberately excludes volatile
    fields (the ``assessed_at`` timestamp, free-text rationale) so the same card
    and ruleset always produce the same fingerprint regardless of when it ran.
    """
    payload = {
        "dataset_id": assessment.candidate.dataset_id,
        "decision": assessment.decision.value,
        "lane": assessment.lane.value,
        "findings": sorted(f.code.value for f in assessment.findings),
    }
    return ASSESSMENT_FINGERPRINT_PREFIX + _sha256_hex(
        _canonical_json(payload))[:16]


# --------------------------------------------------------------------------- #
# Approval / request / policy records (Phase A)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HFImportPolicy:
    """Operator-set bounds applied on top of an approval (fail-closed)."""

    row_limit: int = DEFAULT_SAMPLE_ROWS
    hard_max_rows: int = HARD_MAX_ROWS
    streaming_preferred: bool = True
    allow_unknown_licence_for_eval: bool = False
    pii_policy: str = "block_knowledge"
    content_policy: str = "block_on_unsafe"

    @property
    def effective_row_limit(self) -> int:
        return max(0, min(int(self.row_limit), int(self.hard_max_rows)))

    def to_dict(self) -> dict:
        return {
            "row_limit": self.row_limit,
            "hard_max_rows": self.hard_max_rows,
            "effective_row_limit": self.effective_row_limit,
            "streaming_preferred": self.streaming_preferred,
            "allow_unknown_licence_for_eval": self.allow_unknown_licence_for_eval,
            "pii_policy": self.pii_policy,
            "content_policy": self.content_policy,
        }


@dataclass(frozen=True)
class HFDatasetApproval:
    """An explicit, scoped, revision- and fingerprint-bound import approval.

    This is a *human-authored* governance record. The lifecycle reads it; it is
    never produced automatically from an assessment.
    """

    approval_id: str
    dataset_id: str
    dataset_revision: str
    metadata_fingerprint: str
    approved_by: str
    approved_at: str
    approval_scope: HFApprovalScope
    approved_split_names: Tuple[str, ...] = field(default_factory=tuple)
    approved_columns: Tuple[str, ...] = field(default_factory=tuple)
    row_limit: int = DEFAULT_SAMPLE_ROWS
    streaming_allowed: bool = True
    approved_intended_use: str = ""
    licence_snapshot: str = ""
    provenance_snapshot: str = ""
    pii_policy: str = "block_knowledge"
    content_policy: str = "block_on_unsafe"
    intake_assessment_fingerprint: str = ""
    approval_notes: str = ""
    expires_at: Optional[str] = None

    @property
    def permits_eval(self) -> bool:
        return self.approval_scope in (
            HFApprovalScope.EVAL_ONLY, HFApprovalScope.EVAL_AND_KNOWLEDGE)

    @property
    def permits_knowledge(self) -> bool:
        return self.approval_scope in (
            HFApprovalScope.KNOWLEDGE_ONLY, HFApprovalScope.EVAL_AND_KNOWLEDGE)

    @classmethod
    def from_dict(cls, payload: dict) -> "HFDatasetApproval":
        if not isinstance(payload, dict):
            raise ValueError("HF approval record must be a JSON object")
        for required in ("approval_id", "dataset_id", "dataset_revision",
                         "metadata_fingerprint", "approved_by", "approved_at",
                         "approval_scope"):
            if not str(payload.get(required, "")).strip():
                raise ValueError(f"HF approval is missing '{required}'")
        return cls(
            approval_id=str(payload["approval_id"]),
            dataset_id=str(payload["dataset_id"]),
            dataset_revision=str(payload["dataset_revision"]),
            metadata_fingerprint=str(payload["metadata_fingerprint"]),
            approved_by=str(payload["approved_by"]),
            approved_at=str(payload["approved_at"]),
            approval_scope=HFApprovalScope(str(payload["approval_scope"])),
            approved_split_names=_as_tuple(payload.get("approved_split_names")),
            approved_columns=_as_tuple(payload.get("approved_columns")),
            row_limit=int(payload.get("row_limit", DEFAULT_SAMPLE_ROWS)),
            streaming_allowed=bool(payload.get("streaming_allowed", True)),
            approved_intended_use=str(payload.get("approved_intended_use", "")),
            licence_snapshot=str(payload.get("licence_snapshot", "")),
            provenance_snapshot=str(payload.get("provenance_snapshot", "")),
            pii_policy=str(payload.get("pii_policy", "block_knowledge")),
            content_policy=str(payload.get("content_policy", "block_on_unsafe")),
            intake_assessment_fingerprint=str(
                payload.get("intake_assessment_fingerprint", "")),
            approval_notes=str(payload.get("approval_notes", "")),
            expires_at=_opt_str(payload.get("expires_at")),
        )

    def to_dict(self) -> dict:
        return {
            "_record": "hf_dataset_approval",
            "approval_id": self.approval_id,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "metadata_fingerprint": self.metadata_fingerprint,
            "intake_assessment_fingerprint": self.intake_assessment_fingerprint,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "approval_scope": self.approval_scope.value,
            "approved_split_names": list(self.approved_split_names),
            "approved_columns": list(self.approved_columns),
            "row_limit": self.row_limit,
            "streaming_allowed": self.streaming_allowed,
            "approved_intended_use": self.approved_intended_use,
            "licence_snapshot": self.licence_snapshot,
            "provenance_snapshot": self.provenance_snapshot,
            "pii_policy": self.pii_policy,
            "content_policy": self.content_policy,
            "approval_notes": self.approval_notes,
            "expires_at": self.expires_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class HFImportRequest:
    """A concrete request to sample/import from an approved dataset."""

    dataset_id: str
    dataset_revision: str
    split: str
    intent: HFImportIntent
    requested_columns: Tuple[str, ...] = field(default_factory=tuple)
    requested_row_limit: int = DEFAULT_SAMPLE_ROWS
    streaming: bool = True
    provenance_snapshot: str = ""
    licence_snapshot: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "HFImportRequest":
        if not isinstance(payload, dict):
            raise ValueError("HF import request must be a JSON object")
        for required in ("dataset_id", "dataset_revision", "split", "intent"):
            if not str(payload.get(required, "")).strip():
                raise ValueError(f"HF import request is missing '{required}'")
        return cls(
            dataset_id=str(payload["dataset_id"]),
            dataset_revision=str(payload["dataset_revision"]),
            split=str(payload["split"]),
            intent=HFImportIntent(str(payload["intent"])),
            requested_columns=_as_tuple(payload.get("requested_columns")),
            requested_row_limit=int(
                payload.get("requested_row_limit", DEFAULT_SAMPLE_ROWS)),
            streaming=bool(payload.get("streaming", True)),
            provenance_snapshot=str(payload.get("provenance_snapshot", "")),
            licence_snapshot=str(payload.get("licence_snapshot", "")),
        )

    def to_dict(self) -> dict:
        return {
            "_record": "hf_import_request",
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "intent": self.intent.value,
            "requested_columns": list(self.requested_columns),
            "requested_row_limit": self.requested_row_limit,
            "streaming": self.streaming,
            "provenance_snapshot": self.provenance_snapshot,
            "licence_snapshot": self.licence_snapshot,
        }


@dataclass(frozen=True)
class HFImportValidation:
    """The fail-closed verdict of checking a request against an approval."""

    valid: bool
    codes: Tuple[HFValidationCode, ...]
    messages: Tuple[str, ...]
    approval_id: str
    dataset_id: str
    dataset_revision: str
    scope: HFApprovalScope
    intent: HFImportIntent
    effective_row_limit: int
    checked_at: str

    @property
    def ok(self) -> bool:
        return self.valid

    def to_dict(self) -> dict:
        return {
            "_record": "hf_import_validation",
            "valid": self.valid,
            "codes": [c.value for c in self.codes],
            "messages": list(self.messages),
            "approval_id": self.approval_id,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "scope": self.scope.value,
            "intent": self.intent.value,
            "effective_row_limit": self.effective_row_limit,
            "checked_at": self.checked_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# Validation (Phase A) — fail-closed
# --------------------------------------------------------------------------- #


def _expired(expires_at: Optional[str], now: datetime) -> bool:
    if not expires_at:
        return False
    try:
        deadline = datetime.fromisoformat(expires_at)
    except ValueError:
        # An unparseable expiry is treated as expired: fail closed.
        return True
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    return now > deadline


def validate_import_request(
    approval: HFDatasetApproval,
    request: HFImportRequest,
    *,
    metadata: Optional[HuggingFaceDatasetMetadata] = None,
    assessment: Optional[DataIntakeAssessment] = None,
    policy: Optional[HFImportPolicy] = None,
    now: Optional[datetime] = None,
) -> HFImportValidation:
    """Decide, fail-closed, whether ``request`` is covered by ``approval``.

    When ``metadata`` is supplied the current card is re-fingerprinted and
    compared to the approval's bound fingerprint, and the live licence/provenance
    are compared to the approval's snapshots. When ``assessment`` is supplied the
    underlying v6.2A decision must be an approval (not blocked/quarantine/review)
    and its fingerprint must match. Any drift returns ``valid=False``.
    """
    now = now or _utc_now()
    policy = policy or HFImportPolicy(row_limit=approval.row_limit)
    codes: List[HFValidationCode] = []
    messages: List[str] = []

    def fail(code: HFValidationCode, message: str) -> None:
        codes.append(code)
        messages.append(message)

    if request.dataset_id != approval.dataset_id:
        fail(HFValidationCode.DATASET_ID_MISMATCH,
             f"request dataset_id {request.dataset_id!r} != approved "
             f"{approval.dataset_id!r}")
    if request.dataset_revision != approval.dataset_revision:
        fail(HFValidationCode.REVISION_MISMATCH,
             f"request revision {request.dataset_revision!r} != approved "
             f"{approval.dataset_revision!r}")

    # Scope vs intent: eval/knowledge are separately governed.
    if request.intent is HFImportIntent.KNOWLEDGE and not approval.permits_knowledge:
        fail(HFValidationCode.KNOWLEDGE_REQUIRES_KNOWLEDGE_APPROVAL,
             "knowledge import requires a knowledge-scoped approval; this "
             f"approval is {approval.approval_scope.value}")
    if request.intent is HFImportIntent.EVAL and not approval.permits_eval:
        fail(HFValidationCode.EVAL_REQUIRES_EVAL_APPROVAL,
             "eval import requires an eval-scoped approval; this approval is "
             f"{approval.approval_scope.value}")

    # Split / columns must be inside what was approved.
    if approval.approved_split_names and request.split not in approval.approved_split_names:
        fail(HFValidationCode.SPLIT_NOT_APPROVED,
             f"split {request.split!r} is not in approved splits "
             f"{list(approval.approved_split_names)}")
    if approval.approved_columns and request.requested_columns:
        extra = [c for c in request.requested_columns
                 if c not in approval.approved_columns]
        if extra:
            fail(HFValidationCode.COLUMNS_NOT_APPROVED,
                 f"requested columns {extra} are outside approved columns "
                 f"{list(approval.approved_columns)}")

    # Row limit: request must not exceed approval, and approval is clamped to
    # the policy hard cap.
    ceiling = min(approval.row_limit, policy.effective_row_limit)
    if request.requested_row_limit > approval.row_limit:
        fail(HFValidationCode.ROW_LIMIT_EXCEEDED,
             f"requested row limit {request.requested_row_limit} exceeds "
             f"approved {approval.row_limit}")

    if request.streaming and not approval.streaming_allowed:
        fail(HFValidationCode.STREAMING_NOT_APPROVED,
             "streaming requested but the approval does not allow streaming")

    if approval.approved_intended_use and request.intent.value not in \
            approval.approved_intended_use:
        # approved_intended_use is a free-text scope label; require the intent
        # token to appear in it. Mismatch fails closed.
        fail(HFValidationCode.INTENDED_USE_MISMATCH,
             f"intent {request.intent.value!r} not covered by approved "
             f"intended use {approval.approved_intended_use!r}")

    if _expired(approval.expires_at, now):
        fail(HFValidationCode.APPROVAL_EXPIRED,
             f"approval expired at {approval.expires_at}")

    # Licence / provenance drift against the snapshots the approval was granted
    # against (compared from request or live metadata).
    live_licence = None
    live_provenance = None
    if metadata is not None:
        live_licence = (metadata.license or "").strip().lower()
        fingerprint = compute_metadata_fingerprint(metadata, assessment=assessment)
        if fingerprint != approval.metadata_fingerprint:
            fail(HFValidationCode.METADATA_FINGERPRINT_MISMATCH,
                 f"current metadata fingerprint {fingerprint} != approved "
                 f"{approval.metadata_fingerprint}")
    if request.licence_snapshot:
        live_licence = request.licence_snapshot.strip().lower()
    if request.provenance_snapshot:
        live_provenance = request.provenance_snapshot.strip()

    if approval.licence_snapshot and live_licence is not None and \
            live_licence != approval.licence_snapshot.strip().lower():
        fail(HFValidationCode.LICENCE_CHANGED,
             f"licence {live_licence!r} differs from approved snapshot "
             f"{approval.licence_snapshot!r}")
    if approval.provenance_snapshot and live_provenance is not None and \
            live_provenance != approval.provenance_snapshot.strip():
        fail(HFValidationCode.PROVENANCE_CHANGED,
             f"provenance {live_provenance!r} differs from approved snapshot "
             f"{approval.provenance_snapshot!r}")

    # Underlying intake decision must itself be an approval.
    if assessment is not None:
        if assessment.decision not in (DataIntakeDecision.APPROVED_FOR_EVAL,
                                       DataIntakeDecision.APPROVED_FOR_KNOWLEDGE):
            fail(HFValidationCode.INTAKE_NOT_APPROVED,
                 f"intake decision {assessment.decision.value!r} is not an "
                 "import approval")
        elif request.intent is HFImportIntent.KNOWLEDGE and \
                assessment.decision is not DataIntakeDecision.APPROVED_FOR_KNOWLEDGE:
            fail(HFValidationCode.INTAKE_NOT_APPROVED,
                 "knowledge import requires an intake decision of "
                 "approved_for_knowledge")
        if approval.intake_assessment_fingerprint:
            fingerprint = compute_assessment_fingerprint(assessment)
            if fingerprint != approval.intake_assessment_fingerprint:
                fail(HFValidationCode.INTAKE_FINGERPRINT_MISMATCH,
                     f"intake fingerprint {fingerprint} != approved "
                     f"{approval.intake_assessment_fingerprint}")

    valid = not codes
    if valid:
        codes.append(HFValidationCode.APPROVAL_VALID)
        messages.append("request is covered by the approval")
    return HFImportValidation(
        valid=valid,
        codes=tuple(codes),
        messages=tuple(messages),
        approval_id=approval.approval_id,
        dataset_id=approval.dataset_id,
        dataset_revision=approval.dataset_revision,
        scope=approval.approval_scope,
        intent=request.intent,
        effective_row_limit=max(0, ceiling),
        checked_at=now.isoformat(),
    )


def load_approval(path) -> HFDatasetApproval:
    """Read a single approval record from a JSON file (read-only)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HFDatasetApproval.from_dict(payload)


# --------------------------------------------------------------------------- #
# Bounded sampling (Phase B)
# --------------------------------------------------------------------------- #


def _row_content_hash(row: dict) -> str:
    return "sha256:" + _sha256_hex(_canonical_json(row))


def _schema_fingerprint(keys: Iterable[str]) -> str:
    return "hfschema-" + _sha256_hex(
        _canonical_json(sorted(str(k) for k in keys)))[:16]


@dataclass(frozen=True)
class HFRowAccessProvenance:
    """Where one sampled row came from (audit lineage)."""

    dataset_id: str
    dataset_revision: str
    split: str
    source_row_index: int
    source_row_key: str
    retrieved_at: str
    adapter_version: str
    streaming: bool
    row_content_hash: str
    schema_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "source_row_index": self.source_row_index,
            "source_row_key": self.source_row_key,
            "retrieved_at": self.retrieved_at,
            "adapter_version": self.adapter_version,
            "streaming": self.streaming,
            "row_content_hash": self.row_content_hash,
            "schema_fingerprint": self.schema_fingerprint,
        }


@dataclass(frozen=True)
class HFSampledRow:
    """One bounded sampled row: its raw fields plus access provenance."""

    fields: Dict[str, object]
    provenance: HFRowAccessProvenance

    def to_dict(self) -> dict:
        return {
            "fields": dict(self.fields),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class HFSampleResult:
    """The auditable outcome of a bounded sampling attempt."""

    status: HFSampleStatus
    dataset_id: str
    dataset_revision: str
    split: str
    requested_row_limit: int
    effective_row_limit: int
    rows: Tuple[HFSampledRow, ...]
    schema_fingerprint: str
    truncated: bool
    blocked_reason: Optional[str]
    validation: Optional[HFImportValidation]
    sampled_at: str
    adapter_version: str = ADAPTER_VERSION

    @property
    def sampled_row_count(self) -> int:
        return len(self.rows)

    @property
    def ready(self) -> bool:
        return self.status in (HFSampleStatus.SAMPLE_READY,
                               HFSampleStatus.ROW_LIMIT_REACHED)

    def to_dict(self) -> dict:
        return {
            "_record": "hf_sample_result",
            "status": self.status.value,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "requested_row_limit": self.requested_row_limit,
            "effective_row_limit": self.effective_row_limit,
            "sampled_row_count": self.sampled_row_count,
            "truncated": self.truncated,
            "blocked_reason": self.blocked_reason,
            "schema_fingerprint": self.schema_fingerprint,
            "adapter_version": self.adapter_version,
            "sampled_at": self.sampled_at,
            "validation": self.validation.to_dict() if self.validation else None,
            "rows": [r.to_dict() for r in self.rows],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, dict]]:
    """Yield ``(line_index, row)`` from a JSONL fixture, streaming line by line.

    Comment lines (``#``) and blank lines are skipped. The whole file is never
    held in memory — rows are produced one at a time so the caller can stop early.
    """
    index = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError("each fixture row must be a JSON object")
            yield index, row
            index += 1


def _blocked_result(request: HFImportRequest, *, status: HFSampleStatus,
                    reason: str, validation: Optional[HFImportValidation],
                    effective_limit: int, now: datetime) -> HFSampleResult:
    return HFSampleResult(
        status=status,
        dataset_id=request.dataset_id,
        dataset_revision=request.dataset_revision,
        split=request.split,
        requested_row_limit=request.requested_row_limit,
        effective_row_limit=effective_limit,
        rows=(),
        schema_fingerprint="",
        truncated=False,
        blocked_reason=reason,
        validation=validation,
        sampled_at=now.isoformat(),
    )


def sample_rows(
    request: HFImportRequest,
    *,
    fixture_path,
    approval: HFDatasetApproval,
    metadata: Optional[HuggingFaceDatasetMetadata] = None,
    assessment: Optional[DataIntakeAssessment] = None,
    policy: Optional[HFImportPolicy] = None,
    key_field: Optional[str] = None,
    now: Optional[datetime] = None,
) -> HFSampleResult:
    """Read at most the approved number of rows from a local JSONL fixture.

    Fail-closed: the request is validated against the approval first. A failed
    validation returns a ``sample_blocked`` result with no rows. A missing
    fixture returns ``source_unavailable``. Reading streams line by line and
    stops the instant the cap is reached; the full file is never materialised.
    """
    now = now or _utc_now()
    policy = policy or HFImportPolicy(row_limit=approval.row_limit)

    validation = validate_import_request(
        approval, request, metadata=metadata, assessment=assessment,
        policy=policy, now=now)
    cap = validation.effective_row_limit
    if not validation.valid:
        return _blocked_result(
            request, status=HFSampleStatus.SAMPLE_BLOCKED,
            reason="; ".join(validation.messages), validation=validation,
            effective_limit=cap, now=now)

    path = Path(fixture_path)
    if not path.exists():
        return _blocked_result(
            request, status=HFSampleStatus.SOURCE_UNAVAILABLE,
            reason=f"fixture not found: {path}", validation=validation,
            effective_limit=cap, now=now)

    rows: List[HFSampledRow] = []
    truncated = False
    first_schema: Optional[str] = None
    schema_mismatch = False
    retrieved_at = now.isoformat()

    for index, row in _iter_jsonl(path):
        if len(rows) >= cap:
            # There is at least one more row than the cap allows: stop now.
            truncated = True
            break
        schema = _schema_fingerprint(row.keys())
        if first_schema is None:
            first_schema = schema
        elif schema != first_schema:
            schema_mismatch = True
        source_key = ""
        if key_field and key_field in row:
            source_key = str(row[key_field])
        provenance = HFRowAccessProvenance(
            dataset_id=request.dataset_id,
            dataset_revision=request.dataset_revision,
            split=request.split,
            source_row_index=index,
            source_row_key=source_key,
            retrieved_at=retrieved_at,
            adapter_version=ADAPTER_VERSION,
            streaming=request.streaming,
            row_content_hash=_row_content_hash(row),
            schema_fingerprint=schema,
        )
        rows.append(HFSampledRow(fields=dict(row), provenance=provenance))

    status = HFSampleStatus.ROW_LIMIT_REACHED if truncated else HFSampleStatus.SAMPLE_READY
    if schema_mismatch:
        status = HFSampleStatus.SCHEMA_MISMATCH
    return HFSampleResult(
        status=status,
        dataset_id=request.dataset_id,
        dataset_revision=request.dataset_revision,
        split=request.split,
        requested_row_limit=request.requested_row_limit,
        effective_row_limit=cap,
        rows=tuple(rows),
        schema_fingerprint=first_schema or "",
        truncated=truncated,
        blocked_reason=None,
        validation=validation,
        sampled_at=retrieved_at,
    )


# --------------------------------------------------------------------------- #
# Rendering / writing
# --------------------------------------------------------------------------- #


def render_sample_result_markdown(result: HFSampleResult) -> str:
    """Deterministic, human-readable summary. Never prints raw row content."""
    lines = [
        "# Governed Hugging Face sample (v6.9; bounded, read-only)",
        "",
        f"- dataset: `{result.dataset_id}`",
        f"- revision: `{result.dataset_revision}`",
        f"- split: `{result.split}`",
        f"- status: **{result.status.value}**",
        f"- requested row limit: {result.requested_row_limit}",
        f"- effective row limit: {result.effective_row_limit}",
        f"- sampled rows: {result.sampled_row_count}",
        f"- truncated at cap: {result.truncated}",
        f"- schema fingerprint: `{result.schema_fingerprint or 'n/a'}`",
        f"- adapter: `{result.adapter_version}`",
    ]
    if result.blocked_reason:
        lines += ["", f"> blocked: {result.blocked_reason}"]
    if result.validation is not None:
        lines += ["", "## Approval validation",
                  f"- valid: {result.validation.valid}",
                  f"- codes: {', '.join(c.value for c in result.validation.codes)}"]
    return "\n".join(lines) + "\n"


def write_sample_result(result: HFSampleResult, path) -> Path:
    """The only durable write: a single sample-result JSON to ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_json() + "\n", encoding="utf-8")
    return out
