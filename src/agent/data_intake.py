"""v6.2A Clean Data Intake Lane — governed, read-only dataset assessment.

This module is an **assessment layer only**. It looks at the *declared metadata*
of an external dataset or external content and returns a deterministic
classification. It never:

* downloads full external datasets,
* writes to the MemoryLedger,
* writes to the source registry,
* creates or applies source proposals,
* creates or applies memory proposals,
* changes retrieval, ranking, grounding, or chat routing.

It mutates durable project state only when the caller passes an explicit output
path to :func:`write_assessment_report`, and even then it writes *only* an
assessment report — nothing else. The module imports no memory/source/proposal
writer, so it cannot mutate that state even by accident.

Core invariants enforced here:

* External data must not become trusted knowledge automatically.
* ``approved_for_eval`` is a distinct, weaker tier than
  ``approved_for_knowledge``.
* A missing licence, an unknown provenance, possible/declared PII, and
  non-commercial or research-only licences can never auto-classify as
  ``approved_for_knowledge``.
* ``blocked`` means no import should occur; ``needs_review`` means a human must
  decide before any use.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Thresholds and vocabularies (all deterministic; no network, no download).
# --------------------------------------------------------------------------- #

STALE_AFTER_DAYS = 1095  # ~3 years: older than this is not "fresh" knowledge.
TOO_LARGE_BYTES = 2_000_000_000  # 2 GB hint cap for an initial import.
TOO_LARGE_EXAMPLES = 5_000_000  # row-count hint cap for an initial import.

_OPEN_COMMERCIAL_LICENCES = frozenset({
    "apache-2.0", "apache2.0", "apache", "mit", "bsd", "bsd-2-clause",
    "bsd-3-clause", "isc", "cc0-1.0", "cc0", "cc-by-4.0", "cc-by-3.0",
    "cc-by-sa-4.0", "cc-by-sa-3.0", "odc-by-1.0", "odbl-1.0", "mpl-2.0",
    "unlicense", "public-domain",
})
_UNKNOWN_LICENCE_VALUES = frozenset({
    "", "unknown", "unspecified", "other", "none", "null", "tbd", "n/a", "na",
})
_NON_COMMERCIAL_MARKERS: Tuple[str, ...] = ("-nc", "noncommercial", "non-commercial")
_RESEARCH_ONLY_MARKERS: Tuple[str, ...] = (
    "research-only", "research only", "research_only", "research-use",
    "research use", "academic-only", "academic only", "non-commercial-research",
)
_RESEARCH_ONLY_VALUES = frozenset({"research", "academic"})

_UNKNOWN_PROVENANCE_VALUES = frozenset({
    "", "unknown", "unspecified", "none", "null", "n/a", "na", "unverified",
})

_BENCHMARK_USE_MARKERS: Tuple[str, ...] = (
    "benchmark", "eval", "evaluation", "test set", "testing", "validation",
)
_BENCHMARK_SOURCE_TYPES = frozenset({"benchmark", "eval", "evaluation"})

_PII_DECLARED_VALUES = frozenset({"true", "yes", "y", "present", "declared", "1"})
_PII_ABSENT_VALUES = frozenset({"false", "no", "n", "absent", "none", "clean", "0"})

_SUPPORTED_EXTENSIONS = frozenset({
    "jsonl", "json", "csv", "tsv", "parquet", "txt", "text", "md", "markdown",
})
_UNSUPPORTED_EXTENSIONS = frozenset({
    "pdf", "doc", "docx", "html", "htm", "xml", "zip", "gz", "tgz", "tar",
    "bin", "db", "sqlite", "xls", "xlsx", "exe",
})

_LOW_QUALITY_MARKERS: Tuple[str, ...] = ("low_quality", "low quality", "noisy", "dirty")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


# --------------------------------------------------------------------------- #
# Enums.
# --------------------------------------------------------------------------- #


class DataIntakeDecision(str, Enum):
    """The five terminal classifications for an external dataset candidate."""

    APPROVED_FOR_EVAL = "approved_for_eval"
    APPROVED_FOR_KNOWLEDGE = "approved_for_knowledge"
    NEEDS_REVIEW = "needs_review"
    QUARANTINE = "quarantine"
    BLOCKED = "blocked"


class IntakeLane(str, Enum):
    """The routing bucket a candidate is placed into after assessment."""

    EVAL_ONLY = "eval_only"
    KNOWLEDGE_CANDIDATE = "knowledge_candidate"
    SYNTHETIC_EXAMPLES = "synthetic_examples"
    BENCHMARK = "benchmark"
    QUARANTINE = "quarantine"
    BLOCKED = "blocked"


class IntakeSeverity(str, Enum):
    """How strongly a finding constrains the decision (worst finding wins)."""

    BLOCK = "block"  # -> blocked
    QUARANTINE = "quarantine"  # -> quarantine
    REVIEW = "review"  # -> needs_review
    KNOWLEDGE_BLOCK = "knowledge_block"  # -> caps at approved_for_eval
    INFO = "info"  # -> no constraint


_SEVERITY_RANK = {
    IntakeSeverity.BLOCK: 5,
    IntakeSeverity.QUARANTINE: 4,
    IntakeSeverity.REVIEW: 3,
    IntakeSeverity.KNOWLEDGE_BLOCK: 2,
    IntakeSeverity.INFO: 1,
}


class FindingCode(str, Enum):
    """Stable, auditable reason codes for intake findings."""

    MISSING_LICENSE = "missing_license"
    UNKNOWN_LICENSE = "unknown_license"
    NON_COMMERCIAL_LICENSE = "non_commercial_license"
    RESEARCH_ONLY_LICENSE = "research_only_license"
    UNKNOWN_PROVENANCE = "unknown_provenance"
    MISSING_DATASET_CARD = "missing_dataset_card"
    POSSIBLE_PII = "possible_pii"
    DECLARED_PERSONAL_DATA = "declared_personal_data"
    UNSUPPORTED_FORMAT = "unsupported_format"
    TOO_LARGE_FOR_INITIAL_IMPORT = "too_large_for_initial_import"
    STALE_DATASET = "stale_dataset"
    LOW_QUALITY_SAMPLE = "low_quality_sample"
    SYNTHETIC_BUT_UNVERIFIED = "synthetic_but_unverified"
    APPROVED_PUBLIC_BENCHMARK = "approved_public_benchmark"
    APPROVED_SYNTHETIC_EVAL = "approved_synthetic_eval"
    APPROVED_OPEN_LICENSE = "approved_open_license"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


# --------------------------------------------------------------------------- #
# Data structures.
# --------------------------------------------------------------------------- #


def _coerce_bool(value, *, default: Optional[bool]) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _PII_DECLARED_VALUES:
        return True
    if text in _PII_ABSENT_VALUES:
        return False
    return default


def _coerce_pii(value) -> Optional[bool]:
    """True = declared PII, False = declared no PII, None = unknown/possible."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _PII_DECLARED_VALUES:
        return True
    if text in _PII_ABSENT_VALUES:
        return False
    return None


@dataclass(frozen=True)
class ExternalDatasetCandidate:
    """Declared metadata about an external dataset awaiting assessment.

    This is *metadata only*. No dataset content is downloaded or stored here.
    """

    dataset_id: str
    source_url: str = ""
    source_type: str = ""
    title: str = ""
    description: str = ""
    licence: Optional[str] = None
    provenance: str = ""
    publisher: str = ""
    language: str = ""
    task_categories: Tuple[str, ...] = field(default_factory=tuple)
    size_hint: Optional[str] = None
    last_updated: Optional[str] = None
    intended_use: str = ""
    contains_personal_data: Optional[bool] = None
    is_synthetic: bool = False
    sample_available: Optional[bool] = None
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "ExternalDatasetCandidate":
        if "dataset_id" not in payload or not str(payload["dataset_id"]).strip():
            raise ValueError("candidate is missing a non-empty 'dataset_id'")
        categories = payload.get("task_categories") or ()
        if isinstance(categories, str):
            categories = (categories,)
        return cls(
            dataset_id=str(payload["dataset_id"]),
            source_url=str(payload.get("source_url", "") or ""),
            source_type=str(payload.get("source_type", "") or ""),
            title=str(payload.get("title", "") or ""),
            description=str(payload.get("description", "") or ""),
            licence=payload.get("licence", payload.get("license")),
            provenance=str(payload.get("provenance", "") or ""),
            publisher=str(payload.get("publisher", "") or ""),
            language=str(payload.get("language", "") or ""),
            task_categories=tuple(str(c) for c in categories),
            size_hint=payload.get("size_hint"),
            last_updated=payload.get("last_updated"),
            intended_use=str(payload.get("intended_use", "") or ""),
            contains_personal_data=_coerce_pii(payload.get("contains_personal_data")),
            is_synthetic=bool(_coerce_bool(payload.get("is_synthetic"), default=False)),
            sample_available=_coerce_bool(payload.get("sample_available"), default=None),
            notes=str(payload.get("notes", "") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "title": self.title,
            "description": self.description,
            "licence": self.licence,
            "provenance": self.provenance,
            "publisher": self.publisher,
            "language": self.language,
            "task_categories": list(self.task_categories),
            "size_hint": self.size_hint,
            "last_updated": self.last_updated,
            "intended_use": self.intended_use,
            "contains_personal_data": self.contains_personal_data,
            "is_synthetic": self.is_synthetic,
            "sample_available": self.sample_available,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DataIntakeFinding:
    """A single deterministic observation about a candidate."""

    code: FindingCode
    severity: IntakeSeverity
    message: str

    @property
    def blocks_knowledge(self) -> bool:
        return self.severity in (
            IntakeSeverity.BLOCK,
            IntakeSeverity.QUARANTINE,
            IntakeSeverity.REVIEW,
            IntakeSeverity.KNOWLEDGE_BLOCK,
        )

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "blocks_knowledge": self.blocks_knowledge,
        }


@dataclass(frozen=True)
class DataIntakeAssessment:
    """The deterministic result of assessing one candidate."""

    candidate: ExternalDatasetCandidate
    decision: DataIntakeDecision
    lane: IntakeLane
    findings: Tuple[DataIntakeFinding, ...] = ()
    rationale: str = ""
    assessed_at: str = field(default_factory=lambda: _utc_now().isoformat())

    @property
    def approved_for_eval(self) -> bool:
        """True only for the distinct eval-only tier (not the knowledge tier)."""
        return self.decision == DataIntakeDecision.APPROVED_FOR_EVAL

    @property
    def approved_for_knowledge(self) -> bool:
        return self.decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE

    @property
    def permits_eval_use(self) -> bool:
        """Knowledge approval implies eval is permitted; eval approval does not
        imply knowledge."""
        return self.decision in (
            DataIntakeDecision.APPROVED_FOR_EVAL,
            DataIntakeDecision.APPROVED_FOR_KNOWLEDGE,
        )

    @property
    def permits_knowledge_use(self) -> bool:
        return self.decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE

    @property
    def blocked(self) -> bool:
        return self.decision == DataIntakeDecision.BLOCKED

    @property
    def needs_review(self) -> bool:
        return self.decision == DataIntakeDecision.NEEDS_REVIEW

    @property
    def quarantined(self) -> bool:
        return self.decision == DataIntakeDecision.QUARANTINE

    def to_dict(self) -> dict:
        return {
            "_record": "data_intake_assessment",
            "candidate": self.candidate.to_dict(),
            "decision": self.decision.value,
            "lane": self.lane.value,
            "approved_for_eval": self.approved_for_eval,
            "approved_for_knowledge": self.approved_for_knowledge,
            "permits_eval_use": self.permits_eval_use,
            "permits_knowledge_use": self.permits_knowledge_use,
            "blocked": self.blocked,
            "needs_review": self.needs_review,
            "quarantined": self.quarantined,
            "rationale": self.rationale,
            "findings": [f.to_dict() for f in self.findings],
            "assessed_at": self.assessed_at,
        }


# --------------------------------------------------------------------------- #
# Per-dimension assessors (pure functions).
# --------------------------------------------------------------------------- #


def _finding(code: FindingCode, severity: IntakeSeverity, message: str) -> DataIntakeFinding:
    return DataIntakeFinding(code=code, severity=severity, message=message)


def _has_marker(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def _assess_licence(candidate: ExternalDatasetCandidate) -> List[DataIntakeFinding]:
    if candidate.licence is None or _norm(candidate.licence) == "":
        return [_finding(FindingCode.MISSING_LICENSE, IntakeSeverity.REVIEW,
                         "licence is missing; a human must classify it before any use")]
    lic = _norm(candidate.licence)
    if lic in _UNKNOWN_LICENCE_VALUES:
        return [_finding(FindingCode.UNKNOWN_LICENSE, IntakeSeverity.REVIEW,
                         f"licence '{candidate.licence}' is unknown; a human must classify it")]
    if _has_marker(lic, _NON_COMMERCIAL_MARKERS):
        return [_finding(FindingCode.NON_COMMERCIAL_LICENSE, IntakeSeverity.KNOWLEDGE_BLOCK,
                         f"licence '{candidate.licence}' is non-commercial; eval-only, not knowledge")]
    if _has_marker(lic, _RESEARCH_ONLY_MARKERS) or lic in _RESEARCH_ONLY_VALUES:
        return [_finding(FindingCode.RESEARCH_ONLY_LICENSE, IntakeSeverity.KNOWLEDGE_BLOCK,
                         f"licence '{candidate.licence}' is research-only; eval-only, not knowledge")]
    if lic in _OPEN_COMMERCIAL_LICENCES:
        return [_finding(FindingCode.APPROVED_OPEN_LICENSE, IntakeSeverity.INFO,
                         f"licence '{candidate.licence}' permits commercial reuse")]
    return [_finding(FindingCode.UNKNOWN_LICENSE, IntakeSeverity.REVIEW,
                     f"licence '{candidate.licence}' is not recognised; a human must classify it")]


def _assess_provenance(candidate: ExternalDatasetCandidate) -> List[DataIntakeFinding]:
    prov = _norm(candidate.provenance)
    if prov in _UNKNOWN_PROVENANCE_VALUES:
        return [_finding(FindingCode.UNKNOWN_PROVENANCE, IntakeSeverity.REVIEW,
                         "provenance is unknown; the dataset's origin must be established")]
    return []


def _assess_dataset_card(candidate: ExternalDatasetCandidate) -> List[DataIntakeFinding]:
    if _norm(candidate.title) == "" and _norm(candidate.description) == "":
        return [_finding(FindingCode.MISSING_DATASET_CARD, IntakeSeverity.REVIEW,
                         "no dataset card (title/description) to verify the dataset")]
    return []


def _assess_pii(candidate: ExternalDatasetCandidate) -> List[DataIntakeFinding]:
    state = candidate.contains_personal_data
    if state is True:
        return [_finding(FindingCode.DECLARED_PERSONAL_DATA, IntakeSeverity.BLOCK,
                         "dataset declares personal data; blocked from import")]
    if state is None:
        return [_finding(FindingCode.POSSIBLE_PII, IntakeSeverity.REVIEW,
                         "PII status is undeclared; a human must review before any use")]
    return []


def _is_benchmark_use(candidate: ExternalDatasetCandidate) -> bool:
    if _norm(candidate.source_type) in _BENCHMARK_SOURCE_TYPES:
        return True
    return _has_marker(_norm(candidate.intended_use), _BENCHMARK_USE_MARKERS)


def _assess_intended_use(candidate: ExternalDatasetCandidate) -> List[DataIntakeFinding]:
    if _is_benchmark_use(candidate):
        return [_finding(FindingCode.APPROVED_PUBLIC_BENCHMARK, IntakeSeverity.KNOWLEDGE_BLOCK,
                         "intended use is a benchmark/eval set; eval-only, not training knowledge")]
    return []


def _assess_synthetic(candidate: ExternalDatasetCandidate) -> List[DataIntakeFinding]:
    if not candidate.is_synthetic:
        return []
    if candidate.sample_available is True:
        return [_finding(FindingCode.APPROVED_SYNTHETIC_EVAL, IntakeSeverity.KNOWLEDGE_BLOCK,
                         "synthetic data with a sample; eval-only, not trusted knowledge")]
    return [_finding(FindingCode.SYNTHETIC_BUT_UNVERIFIED, IntakeSeverity.KNOWLEDGE_BLOCK,
                     "synthetic data with no sample to verify; eval-only, not knowledge")]


def _url_extension(source_url: str) -> str:
    path = re.split(r"[?#]", source_url, maxsplit=1)[0]
    match = re.search(r"\.([A-Za-z0-9]+)$", path.strip())
    return match.group(1).lower() if match else ""


def _assess_format(candidate: ExternalDatasetCandidate) -> List[DataIntakeFinding]:
    ext = _url_extension(candidate.source_url)
    if ext and ext in _UNSUPPORTED_EXTENSIONS:
        return [_finding(FindingCode.UNSUPPORTED_FORMAT, IntakeSeverity.QUARANTINE,
                         f"source format '.{ext}' is not a supported import format")]
    return []


def _is_too_large(size_hint: Optional[str]) -> bool:
    if not size_hint:
        return False
    text = str(size_hint).strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([a-z]+)?", text)
    if not match:
        return False
    value = float(match.group(1))
    unit = (match.group(2) or "").strip()
    if unit in ("tb", "tib"):
        return True
    if unit in ("gb", "gib"):
        return value * 1_000_000_000 > TOO_LARGE_BYTES
    if unit in ("examples", "rows", "samples", "records"):
        return value > TOO_LARGE_EXAMPLES
    return False


def _assess_size(candidate: ExternalDatasetCandidate) -> List[DataIntakeFinding]:
    if _is_too_large(candidate.size_hint):
        return [_finding(FindingCode.TOO_LARGE_FOR_INITIAL_IMPORT, IntakeSeverity.QUARANTINE,
                         f"declared size '{candidate.size_hint}' is too large for an initial import")]
    return []


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _assess_freshness(candidate: ExternalDatasetCandidate, *, now: datetime) -> List[DataIntakeFinding]:
    parsed = _parse_date(candidate.last_updated)
    if parsed is None:
        return []
    if (now - parsed).days > STALE_AFTER_DAYS:
        return [_finding(FindingCode.STALE_DATASET, IntakeSeverity.KNOWLEDGE_BLOCK,
                         f"last updated {candidate.last_updated}; too stale for trusted knowledge")]
    return []


def _assess_quality(candidate: ExternalDatasetCandidate) -> List[DataIntakeFinding]:
    if _has_marker(_norm(candidate.notes), _LOW_QUALITY_MARKERS):
        return [_finding(FindingCode.LOW_QUALITY_SAMPLE, IntakeSeverity.QUARANTINE,
                         "notes flag a low-quality sample; held until cleaned")]
    return []


# --------------------------------------------------------------------------- #
# Decision and lane derivation.
# --------------------------------------------------------------------------- #


def _worst_severity(findings: Sequence[DataIntakeFinding]) -> IntakeSeverity:
    worst = IntakeSeverity.INFO
    for finding in findings:
        if _SEVERITY_RANK[finding.severity] > _SEVERITY_RANK[worst]:
            worst = finding.severity
    return worst


def _decide(findings: Sequence[DataIntakeFinding]) -> DataIntakeDecision:
    worst = _worst_severity(findings)
    if worst == IntakeSeverity.BLOCK:
        return DataIntakeDecision.BLOCKED
    if worst == IntakeSeverity.QUARANTINE:
        return DataIntakeDecision.QUARANTINE
    if worst == IntakeSeverity.REVIEW:
        return DataIntakeDecision.NEEDS_REVIEW
    if worst == IntakeSeverity.KNOWLEDGE_BLOCK:
        return DataIntakeDecision.APPROVED_FOR_EVAL
    return DataIntakeDecision.APPROVED_FOR_KNOWLEDGE


def _lane(decision: DataIntakeDecision, candidate: ExternalDatasetCandidate) -> IntakeLane:
    if decision == DataIntakeDecision.BLOCKED:
        return IntakeLane.BLOCKED
    if decision in (DataIntakeDecision.QUARANTINE, DataIntakeDecision.NEEDS_REVIEW):
        return IntakeLane.QUARANTINE
    if decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE:
        return IntakeLane.KNOWLEDGE_CANDIDATE
    # approved_for_eval: route by nature of the data.
    if candidate.is_synthetic:
        return IntakeLane.SYNTHETIC_EXAMPLES
    if _is_benchmark_use(candidate):
        return IntakeLane.BENCHMARK
    return IntakeLane.EVAL_ONLY


_RATIONALE_HEAD = {
    DataIntakeDecision.APPROVED_FOR_KNOWLEDGE: "Approved for knowledge (and eval).",
    DataIntakeDecision.APPROVED_FOR_EVAL: "Approved for eval only; not trusted knowledge.",
    DataIntakeDecision.NEEDS_REVIEW: "Needs human review before any use.",
    DataIntakeDecision.QUARANTINE: "Quarantined; held until cleaned or cleared.",
    DataIntakeDecision.BLOCKED: "Blocked; no import should occur.",
}


def _rationale(decision: DataIntakeDecision, findings: Sequence[DataIntakeFinding]) -> str:
    head = _RATIONALE_HEAD[decision]
    drivers = [f.message for f in findings if f.severity != IntakeSeverity.INFO]
    if not drivers:
        drivers = [f.message for f in findings]
    return head + (" " + "; ".join(drivers) if drivers else "")


# --------------------------------------------------------------------------- #
# Public assessment entry points.
# --------------------------------------------------------------------------- #


def assess_candidate(
    candidate: ExternalDatasetCandidate,
    *,
    now: Optional[datetime] = None,
) -> DataIntakeAssessment:
    """Assess one candidate deterministically. Reads metadata only; writes nothing."""
    moment = now or _utc_now()
    findings: List[DataIntakeFinding] = []
    findings.extend(_assess_licence(candidate))
    findings.extend(_assess_provenance(candidate))
    findings.extend(_assess_dataset_card(candidate))
    findings.extend(_assess_pii(candidate))
    findings.extend(_assess_intended_use(candidate))
    findings.extend(_assess_synthetic(candidate))
    findings.extend(_assess_format(candidate))
    findings.extend(_assess_size(candidate))
    findings.extend(_assess_freshness(candidate, now=moment))
    findings.extend(_assess_quality(candidate))

    decision = _decide(findings)
    if decision == DataIntakeDecision.NEEDS_REVIEW:
        findings.append(_finding(FindingCode.NEEDS_HUMAN_REVIEW, IntakeSeverity.REVIEW,
                                 "a human must review and decide before any use"))
    lane = _lane(decision, candidate)
    rationale = _rationale(decision, findings)
    return DataIntakeAssessment(
        candidate=candidate,
        decision=decision,
        lane=lane,
        findings=tuple(findings),
        rationale=rationale,
        assessed_at=moment.isoformat(),
    )


def assess_candidates(
    candidates: Sequence[ExternalDatasetCandidate],
    *,
    now: Optional[datetime] = None,
) -> List[DataIntakeAssessment]:
    return [assess_candidate(candidate, now=now) for candidate in candidates]


def load_candidates(path) -> List[ExternalDatasetCandidate]:
    """Read candidate metadata from a JSON or JSONL file. Reads only; no download."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    payloads: List[dict] = []
    if source.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            payloads.append(json.loads(stripped))
    else:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            payloads.extend(parsed)
        else:
            payloads.append(parsed)
    return [ExternalDatasetCandidate.from_dict(p) for p in payloads]


# --------------------------------------------------------------------------- #
# Reporting (the ONLY durable write, and only to an explicit path).
# --------------------------------------------------------------------------- #


def write_assessment_report(
    assessments,
    out_path,
) -> Path:
    """Write an assessment report to ``out_path`` (the only durable write).

    Accepts a single assessment or a sequence. Writes *only* the report; it never
    touches memory, the source registry, or any proposal queue.
    """
    if isinstance(assessments, DataIntakeAssessment):
        payload = assessments.to_dict()
    else:
        payload = [a.to_dict() for a in assessments]
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_intake_assessment_markdown(assessment: DataIntakeAssessment) -> str:
    """Render a deterministic, human-readable report for one assessment."""
    lines = [
        "# Data intake assessment (v6.2A; assessment-only)",
        "",
        f"- Dataset: {assessment.candidate.dataset_id}",
        f"- Decision: {assessment.decision.value}",
        f"- Lane: {assessment.lane.value}",
        f"- Approved for knowledge: {str(assessment.approved_for_knowledge).lower()}",
        f"- Approved for eval: {str(assessment.approved_for_eval).lower()}",
        f"- Permits eval use: {str(assessment.permits_eval_use).lower()}",
        "",
        "## Rationale",
        assessment.rationale,
        "",
        "## Findings",
    ]
    if assessment.findings:
        for finding in assessment.findings:
            lines.append(f"- [{finding.severity.value}] {finding.code.value}: {finding.message}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def render_intake_summary_markdown(assessments: Sequence[DataIntakeAssessment]) -> str:
    """Render a one-line-per-candidate summary across many assessments."""
    lines = ["# Data intake summary (v6.2A; assessment-only)", ""]
    for assessment in assessments:
        lines.append(
            f"- {assessment.candidate.dataset_id}: {assessment.decision.value} "
            f"(lane: {assessment.lane.value})"
        )
    return "\n".join(lines)
