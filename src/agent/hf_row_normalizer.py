"""Governed Hugging Face row normalization — Phase C (v6.9).

Phase B produces *bounded, raw* sampled rows with access provenance. This module
turns one such row into a **canonical, validated, deterministically-identified**
record under an explicit *normalization profile*, and nothing more.

It is deliberately narrow:

* A :class:`HFNormalizationProfile` names a known record shape
  (question/answer, instruction/response, document text, classification,
  retrieval pair, benchmark case, or a passthrough generic record). Each profile
  has a :class:`HFSchemaProfile` declaring its canonical roles, which roles are
  required, and the default source columns that map onto each role.
* :func:`normalize_row` resolves each canonical role from the raw row (an
  explicit ``column_map`` wins over the profile defaults), coerces values to
  trimmed strings, and **fails closed** with a :class:`HFRowValidationResult`
  when a required role is missing or empty.
* Every accepted row gets a deterministic id
  ``hfrow-<sha256(dataset_id|revision|split|source_key|normalized_payload)[:16]>``
  so the *same row content under the same approval* always yields the *same id*,
  and a different revision or different content yields a different id.

What this module does **not** do (and cannot — it imports no writer):

* sample or download anything (that is Phase B);
* inspect content for PII or safety (that is :mod:`agent.hf_content_inspector`);
* import a row into an eval pack or knowledge pack (later phases);
* write the ``MemoryLedger``, source registry, a proposal, or a retrieval index;
* call any LLM.

The only durable write is :func:`write_normalization_report`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from agent.hf_import_lifecycle import HFSampledRow, HFSampleResult

NORMALIZER_VERSION = "hf-normalizer-v6.9"
ROW_ID_PREFIX = "hfrow-"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _coerce_text(value) -> str:
    """Coerce a raw field value to a trimmed string (lists join with newlines)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "\n".join(_coerce_text(v) for v in value if _coerce_text(v))
    return _canonical_json(value)


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


class HFNormalizationProfile(str, Enum):
    """A known target record shape for a normalized row."""

    QUESTION_ANSWER = "question_answer"
    INSTRUCTION_RESPONSE = "instruction_response"
    DOCUMENT_TEXT = "document_text"
    CLASSIFICATION = "classification"
    RETRIEVAL_PAIR = "retrieval_pair"
    BENCHMARK_CASE = "benchmark_case"
    GENERIC_RECORD = "generic_record"


@dataclass(frozen=True)
class HFSchemaProfile:
    """Declares a profile's canonical roles and how they map from raw columns."""

    profile: HFNormalizationProfile
    roles: Tuple[str, ...]
    required_roles: Tuple[str, ...]
    default_columns: Mapping[str, Tuple[str, ...]]
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.value,
            "roles": list(self.roles),
            "required_roles": list(self.required_roles),
            "default_columns": {k: list(v) for k, v in self.default_columns.items()},
            "description": self.description,
        }


PROFILE_SCHEMAS: Dict[HFNormalizationProfile, HFSchemaProfile] = {
    HFNormalizationProfile.QUESTION_ANSWER: HFSchemaProfile(
        profile=HFNormalizationProfile.QUESTION_ANSWER,
        roles=("question", "answer", "context"),
        required_roles=("question", "answer"),
        default_columns={
            "question": ("question", "query", "prompt", "q"),
            "answer": ("answer", "answers", "response", "a"),
            "context": ("context", "passage", "evidence"),
        },
        description="A question with its answer and optional grounding context.",
    ),
    HFNormalizationProfile.INSTRUCTION_RESPONSE: HFSchemaProfile(
        profile=HFNormalizationProfile.INSTRUCTION_RESPONSE,
        roles=("instruction", "response", "input"),
        required_roles=("instruction", "response"),
        default_columns={
            "instruction": ("instruction", "prompt", "task"),
            "response": ("response", "output", "completion", "answer"),
            "input": ("input", "context"),
        },
        description="An instruction with its response and optional input.",
    ),
    HFNormalizationProfile.DOCUMENT_TEXT: HFSchemaProfile(
        profile=HFNormalizationProfile.DOCUMENT_TEXT,
        roles=("text", "title"),
        required_roles=("text",),
        default_columns={
            "text": ("text", "content", "body", "document"),
            "title": ("title", "heading", "name"),
        },
        description="A free-text document with an optional title.",
    ),
    HFNormalizationProfile.CLASSIFICATION: HFSchemaProfile(
        profile=HFNormalizationProfile.CLASSIFICATION,
        roles=("text", "label"),
        required_roles=("text", "label"),
        default_columns={
            "text": ("text", "sentence", "content", "input"),
            "label": ("label", "class", "category", "target"),
        },
        description="A text with its classification label.",
    ),
    HFNormalizationProfile.RETRIEVAL_PAIR: HFSchemaProfile(
        profile=HFNormalizationProfile.RETRIEVAL_PAIR,
        roles=("query", "passage", "relevance"),
        required_roles=("query", "passage"),
        default_columns={
            "query": ("query", "question", "anchor"),
            "passage": ("passage", "document", "positive", "context"),
            "relevance": ("relevance", "label", "score"),
        },
        description="A query paired with a candidate passage and optional label.",
    ),
    HFNormalizationProfile.BENCHMARK_CASE: HFSchemaProfile(
        profile=HFNormalizationProfile.BENCHMARK_CASE,
        roles=("question", "expected", "choices"),
        required_roles=("question", "expected"),
        default_columns={
            "question": ("question", "prompt", "problem"),
            "expected": ("expected", "answer", "target", "label", "solution"),
            "choices": ("choices", "options", "candidates"),
        },
        description="A benchmark question with its expected answer and choices.",
    ),
    HFNormalizationProfile.GENERIC_RECORD: HFSchemaProfile(
        profile=HFNormalizationProfile.GENERIC_RECORD,
        roles=(),
        required_roles=(),
        default_columns={},
        description="A passthrough record; all non-empty string fields are kept.",
    ),
}


def get_schema_profile(profile: HFNormalizationProfile) -> HFSchemaProfile:
    return PROFILE_SCHEMAS[profile]


# --------------------------------------------------------------------------- #
# Validation records
# --------------------------------------------------------------------------- #


class HFRowValidationSeverity(str, Enum):
    """How strongly a row-validation finding constrains the outcome."""

    REJECT = "reject"  # row is not normalized
    REVIEW = "review"  # normalized but flagged
    INFO = "info"  # advisory only


class HFRowValidationCode(str, Enum):
    """Stable reasons a raw row does or does not normalize under a profile."""

    ROW_VALID = "row_valid"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    EMPTY_REQUIRED_FIELD = "empty_required_field"
    EMPTY_ROW = "empty_row"
    UNMAPPED_COLUMNS = "unmapped_columns"
    NOT_A_RECORD = "not_a_record"


@dataclass(frozen=True)
class HFRowValidationFinding:
    """A single deterministic observation about a row's normalization."""

    code: HFRowValidationCode
    severity: HFRowValidationSeverity
    message: str
    field_name: str = ""

    @property
    def rejects(self) -> bool:
        return self.severity is HFRowValidationSeverity.REJECT

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field_name,
        }


@dataclass(frozen=True)
class HFRowValidationResult:
    """The fail-closed verdict of normalizing one row."""

    valid: bool
    findings: Tuple[HFRowValidationFinding, ...]
    row_id: str
    source_row_index: int

    @property
    def ok(self) -> bool:
        return self.valid

    def to_dict(self) -> dict:
        return {
            "_record": "hf_row_validation_result",
            "valid": self.valid,
            "row_id": self.row_id,
            "source_row_index": self.source_row_index,
            "findings": [f.to_dict() for f in self.findings],
        }


# --------------------------------------------------------------------------- #
# Normalized row
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HFNormalizedRow:
    """A canonical, validated, deterministically-identified row."""

    row_id: str
    profile: HFNormalizationProfile
    dataset_id: str
    dataset_revision: str
    split: str
    source_row_index: int
    source_row_key: str
    normalized_fields: Dict[str, str]
    unmapped_columns: Tuple[str, ...]
    content_hash: str
    schema_fingerprint: str
    normalized_at: str
    adapter_version: str = NORMALIZER_VERSION

    def text_items(self) -> Tuple[Tuple[str, str], ...]:
        """(role, text) pairs for downstream content inspection; sorted, stable."""
        return tuple(sorted(self.normalized_fields.items()))

    def to_dict(self) -> dict:
        return {
            "_record": "hf_normalized_row",
            "row_id": self.row_id,
            "profile": self.profile.value,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "source_row_index": self.source_row_index,
            "source_row_key": self.source_row_key,
            "normalized_fields": dict(self.normalized_fields),
            "unmapped_columns": list(self.unmapped_columns),
            "content_hash": self.content_hash,
            "schema_fingerprint": self.schema_fingerprint,
            "normalized_at": self.normalized_at,
            "adapter_version": self.adapter_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def compute_row_id(
    dataset_id: str,
    dataset_revision: str,
    split: str,
    source_row_key: str,
    normalized_payload: Mapping[str, str],
) -> str:
    """``hfrow-<sha256(id|revision|split|source_key|canonical_payload)[:16]>``."""
    seed = "|".join((
        dataset_id,
        dataset_revision,
        split,
        source_row_key,
        _canonical_json(dict(normalized_payload)),
    ))
    return ROW_ID_PREFIX + _sha256_hex(seed)[:16]


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def _resolve_role(
    role: str,
    row: Mapping[str, object],
    column_map: Optional[Mapping[str, str]],
    defaults: Mapping[str, Tuple[str, ...]],
) -> Tuple[Optional[str], str]:
    """Return ``(source_column, text)`` for a canonical role, or ``(None, "")``.

    An explicit ``column_map`` entry wins over the profile defaults; default
    candidates are tried in declared order. The first present column wins,
    deterministically.
    """
    if column_map and role in column_map:
        col = column_map[role]
        if col in row:
            return col, _coerce_text(row[col])
        return col, ""
    for candidate in defaults.get(role, ()):
        if candidate in row:
            return candidate, _coerce_text(row[candidate])
    return None, ""


def normalize_row(
    sampled_row: HFSampledRow,
    *,
    profile: HFNormalizationProfile,
    column_map: Optional[Mapping[str, str]] = None,
    now: Optional[datetime] = None,
) -> Tuple[Optional[HFNormalizedRow], HFRowValidationResult]:
    """Normalize one sampled row under ``profile``; fail closed on missing roles."""
    now = now or _utc_now()
    schema = get_schema_profile(profile)
    raw = sampled_row.fields
    prov = sampled_row.provenance
    findings: List[HFRowValidationFinding] = []

    normalized: Dict[str, str] = {}
    consumed: List[str] = []

    if profile is HFNormalizationProfile.GENERIC_RECORD:
        for key in sorted(raw):
            text = _coerce_text(raw[key])
            if text:
                normalized[str(key)] = text
                consumed.append(str(key))
        if not normalized:
            findings.append(HFRowValidationFinding(
                code=HFRowValidationCode.EMPTY_ROW,
                severity=HFRowValidationSeverity.REJECT,
                message="generic record has no non-empty fields"))
    else:
        for role in schema.roles:
            source_col, text = _resolve_role(role, raw, column_map,
                                             schema.default_columns)
            required = role in schema.required_roles
            if source_col is None:
                if required:
                    findings.append(HFRowValidationFinding(
                        code=HFRowValidationCode.MISSING_REQUIRED_FIELD,
                        severity=HFRowValidationSeverity.REJECT,
                        message=f"required role {role!r} has no source column",
                        field_name=role))
                continue
            if not text:
                if required:
                    findings.append(HFRowValidationFinding(
                        code=HFRowValidationCode.EMPTY_REQUIRED_FIELD,
                        severity=HFRowValidationSeverity.REJECT,
                        message=f"required role {role!r} resolved to an empty value",
                        field_name=role))
                continue
            normalized[role] = text
            consumed.append(source_col)

    unmapped = tuple(sorted(str(k) for k in raw if str(k) not in set(consumed)))
    if unmapped and profile is not HFNormalizationProfile.GENERIC_RECORD:
        findings.append(HFRowValidationFinding(
            code=HFRowValidationCode.UNMAPPED_COLUMNS,
            severity=HFRowValidationSeverity.INFO,
            message=f"{len(unmapped)} source column(s) not used by this profile",
            field_name=",".join(unmapped)))

    valid = not any(f.rejects for f in findings)
    if not valid:
        result = HFRowValidationResult(
            valid=False, findings=tuple(findings), row_id="",
            source_row_index=prov.source_row_index)
        return None, result

    row_id = compute_row_id(
        prov.dataset_id, prov.dataset_revision, prov.split,
        prov.source_row_key, normalized)
    findings.insert(0, HFRowValidationFinding(
        code=HFRowValidationCode.ROW_VALID,
        severity=HFRowValidationSeverity.INFO,
        message="row normalized under profile"))
    result = HFRowValidationResult(
        valid=True, findings=tuple(findings), row_id=row_id,
        source_row_index=prov.source_row_index)
    normalized_row = HFNormalizedRow(
        row_id=row_id,
        profile=profile,
        dataset_id=prov.dataset_id,
        dataset_revision=prov.dataset_revision,
        split=prov.split,
        source_row_index=prov.source_row_index,
        source_row_key=prov.source_row_key,
        normalized_fields=dict(normalized),
        unmapped_columns=unmapped,
        content_hash=prov.row_content_hash,
        schema_fingerprint=prov.schema_fingerprint,
        normalized_at=now.isoformat(),
    )
    return normalized_row, result


# --------------------------------------------------------------------------- #
# Sample-level report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HFNormalizationReport:
    """The auditable outcome of normalizing a whole bounded sample."""

    profile: HFNormalizationProfile
    dataset_id: str
    dataset_revision: str
    split: str
    normalized_rows: Tuple[HFNormalizedRow, ...]
    validation_results: Tuple[HFRowValidationResult, ...]
    normalized_at: str
    adapter_version: str = NORMALIZER_VERSION

    @property
    def normalized_count(self) -> int:
        return len(self.normalized_rows)

    @property
    def rejected_count(self) -> int:
        return sum(1 for r in self.validation_results if not r.valid)

    @property
    def all_normalized(self) -> bool:
        return self.rejected_count == 0 and self.normalized_count > 0

    def to_dict(self) -> dict:
        return {
            "_record": "hf_normalization_report",
            "profile": self.profile.value,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "normalized_count": self.normalized_count,
            "rejected_count": self.rejected_count,
            "adapter_version": self.adapter_version,
            "normalized_at": self.normalized_at,
            "normalized_rows": [r.to_dict() for r in self.normalized_rows],
            "validation_results": [r.to_dict() for r in self.validation_results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def normalize_sample(
    sample_result: HFSampleResult,
    *,
    profile: HFNormalizationProfile,
    column_map: Optional[Mapping[str, str]] = None,
    now: Optional[datetime] = None,
) -> HFNormalizationReport:
    """Normalize every row of a bounded sample under one profile (read-only)."""
    now = now or _utc_now()
    normalized_rows: List[HFNormalizedRow] = []
    results: List[HFRowValidationResult] = []
    for sampled_row in sample_result.rows:
        normalized, result = normalize_row(
            sampled_row, profile=profile, column_map=column_map, now=now)
        results.append(result)
        if normalized is not None:
            normalized_rows.append(normalized)
    return HFNormalizationReport(
        profile=profile,
        dataset_id=sample_result.dataset_id,
        dataset_revision=sample_result.dataset_revision,
        split=sample_result.split,
        normalized_rows=tuple(normalized_rows),
        validation_results=tuple(results),
        normalized_at=now.isoformat(),
    )


# --------------------------------------------------------------------------- #
# Rendering / writing
# --------------------------------------------------------------------------- #


def render_normalization_markdown(report: HFNormalizationReport) -> str:
    """Deterministic summary. Never prints raw normalized field content."""
    lines = [
        "# Governed Hugging Face normalization (v6.9; read-only)",
        "",
        f"- dataset: `{report.dataset_id}`",
        f"- revision: `{report.dataset_revision}`",
        f"- split: `{report.split}`",
        f"- profile: **{report.profile.value}**",
        f"- normalized rows: {report.normalized_count}",
        f"- rejected rows: {report.rejected_count}",
        f"- adapter: `{report.adapter_version}`",
    ]
    if report.rejected_count:
        lines += ["", "## Rejected rows"]
        for result in report.validation_results:
            if result.valid:
                continue
            codes = ", ".join(f.code.value for f in result.findings if f.rejects)
            lines.append(f"- row {result.source_row_index}: {codes}")
    return "\n".join(lines) + "\n"


def write_normalization_report(report: HFNormalizationReport, path) -> Path:
    """The only durable write: a single normalization-report JSON to ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_json() + "\n", encoding="utf-8")
    return out
