"""v5.2 Memory Conflict + Staleness Detector — read-only risk detection.

Before any future memory-write path exists, this module checks v5.0/v5.1 memory
proposals against existing memory / project-state records and reports *risk*:
duplicates, contradictions, supersession, staleness, missing evidence, low
confidence, and not-writeable categories. It is **detection only**:

* nothing here writes to the ``MemoryLedger`` or the concept-cell bank — there is
  no import of either, by construction;
* it never mutates a proposal queue, an existing-memory file, or any source;
* it never applies, approves, or writes a proposal;
* it changes no retrieval, ranking, source-selection, grounding, or composer
  behaviour, and introduces no autonomous action.

The core rule: **conflict detection reports risk; it does not resolve it or
write memory.** Every finding is a signal for a human, never an action.

Detection is deterministic and *conservative* (lexical token overlap, polarity,
and date comparison — not semantic). When a relationship is ambiguous it is
reported at a lower severity (``possible_duplicate``) rather than asserted, and
no risk is ever silently dropped.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.memory_proposal_quality import MemoryProposalQuality, MemoryProposalType


# --------------------------------------------------------------------------- #
# Severity + risk codes
# --------------------------------------------------------------------------- #
class MemoryConflictSeverity:
    """Severity of a conflict/staleness finding (advisory only)."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class MemoryConflictCode:
    """Stable string codes for the kind of risk a finding reports."""

    DUPLICATE_CLAIM = "duplicate_claim"
    POSSIBLE_DUPLICATE = "possible_duplicate"
    CONTRADICTS_EXISTING_MEMORY = "contradicts_existing_memory"
    SUPERSEDES_EXISTING_MEMORY = "supersedes_existing_memory"
    SUPERSEDED_BY_EXISTING_MEMORY = "superseded_by_existing_memory"
    STALE_PROJECT_STATE = "stale_project_state"
    STALE_USER_PREFERENCE = "stale_user_preference"
    MISSING_EVIDENCE = "missing_evidence"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_CANDIDATE = "invalid_candidate"
    NON_FACTUAL_NOT_WRITEABLE = "non_factual_not_writeable"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


_SEVERITY_BY_CODE: Dict[str, str] = {
    MemoryConflictCode.DUPLICATE_CLAIM: MemoryConflictSeverity.WARNING,
    MemoryConflictCode.POSSIBLE_DUPLICATE: MemoryConflictSeverity.INFO,
    MemoryConflictCode.CONTRADICTS_EXISTING_MEMORY: MemoryConflictSeverity.ERROR,
    MemoryConflictCode.SUPERSEDES_EXISTING_MEMORY: MemoryConflictSeverity.WARNING,
    MemoryConflictCode.SUPERSEDED_BY_EXISTING_MEMORY:
        MemoryConflictSeverity.WARNING,
    MemoryConflictCode.STALE_PROJECT_STATE: MemoryConflictSeverity.WARNING,
    MemoryConflictCode.STALE_USER_PREFERENCE: MemoryConflictSeverity.INFO,
    MemoryConflictCode.MISSING_EVIDENCE: MemoryConflictSeverity.WARNING,
    MemoryConflictCode.LOW_CONFIDENCE: MemoryConflictSeverity.WARNING,
    MemoryConflictCode.INVALID_CANDIDATE: MemoryConflictSeverity.ERROR,
    MemoryConflictCode.NON_FACTUAL_NOT_WRITEABLE: MemoryConflictSeverity.INFO,
    MemoryConflictCode.NEEDS_HUMAN_REVIEW: MemoryConflictSeverity.INFO,
}

# Deterministic sort order for severities (most severe first).
_SEVERITY_RANK = {
    MemoryConflictSeverity.ERROR: 0,
    MemoryConflictSeverity.WARNING: 1,
    MemoryConflictSeverity.INFO: 2,
}


# --------------------------------------------------------------------------- #
# Proposal-type groupings (kept local; mirror v5.0 public constants)
# --------------------------------------------------------------------------- #
_FACTUAL_TYPES = frozenset({
    MemoryProposalType.FACT,
    MemoryProposalType.DECISION,
    MemoryProposalType.PROJECT_STATE,
    MemoryProposalType.USER_PREFERENCE,
})
_TRACKED_NON_FACTUAL_TYPES = frozenset({
    MemoryProposalType.TODO,
    MemoryProposalType.OPEN_QUESTION,
    MemoryProposalType.SOURCE_GAP,
})
_EVIDENCE_REQUIRED_TYPES = frozenset({
    MemoryProposalType.FACT,
    MemoryProposalType.DECISION,
    MemoryProposalType.PROJECT_STATE,
})

# --------------------------------------------------------------------------- #
# Tunable, deterministic thresholds
# --------------------------------------------------------------------------- #
# A factual proposal below this confidence is flagged low_confidence.
_LOW_CONFIDENCE_THRESHOLD = 0.6
# Content-token Jaccard at/above which two claims are a "possible duplicate".
_POSSIBLE_DUPLICATE_JACCARD = 0.7
# Overlap at/above which a polarity flip is treated as a contradiction.
_CONTRADICTION_OVERLAP = 0.5
# Overlap at/above which two project-state claims are "the same topic" (so a
# non-identical newer claim supersedes the older one).
_SUPERSEDE_JACCARD = 0.6
# Staleness windows (conservative: a preference ages far slower than state).
_PROJECT_STATE_STALE_DAYS = 90
_USER_PREFERENCE_STALE_DAYS = 365

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "to", "of", "and", "for", "in", "on", "at",
    "by", "it", "that", "this", "as", "with", "be", "was", "were", "will",
    "we", "i", "you", "they", "or", "but", "if", "then", "so", "from",
})
_NEGATION_MARKERS = (
    " not ", " no ", " never ", " cannot ", "n't", " without ", " none ",
    " neither ", " nor ",
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _normalized_key(claim: str) -> Tuple[str, ...]:
    """Token tuple used for exact-duplicate comparison (case/space-insensitive)."""
    return tuple(_tokens(claim))


def _content_tokens(claim: str) -> frozenset:
    """Significant tokens (minus stopwords and negation words) for overlap."""
    out = set()
    for token in _tokens(claim):
        if token in _STOPWORDS:
            continue
        out.add(token)
    return frozenset(out)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _has_negation(claim: str) -> bool:
    padded = f" {claim.lower()} "
    return any(marker in padded for marker in _NEGATION_MARKERS)


def _parse_dt(value) -> Optional[datetime]:
    """Parse an ISO date/datetime into an aware UTC datetime (or None)."""
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    parsed: Optional[datetime] = None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text[:10])
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# --------------------------------------------------------------------------- #
# Existing-memory input model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExistingMemoryRecord:
    """An existing memory / project-state record a proposal is checked against.

    This is the *current* memory the detector compares proposals to. It is read
    only — the detector never writes or mutates it.
    """

    memory_id: str
    memory_type: str = ""
    claim: str = ""
    evidence_source_id: str = ""
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ExistingMemoryRecord":
        memory_id = data.get("memory_id", data.get("id", ""))
        memory_type = data.get(
            "memory_type", data.get("proposal_type", data.get("type", "")))
        claim = data.get("claim", data.get("text", data.get("canonical_text", "")))
        return cls(
            memory_id=str(memory_id),
            memory_type=str(memory_type),
            claim=str(claim),
            evidence_source_id=str(data.get("evidence_source_id", "")),
            created_at=data.get("created_at"),
        )


# --------------------------------------------------------------------------- #
# Finding + report output models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemoryConflictFinding:
    """One read-only risk finding for one proposal. Advisory; never an action."""

    proposal_id: str
    proposal_type: str
    claim: str
    code: str
    severity: str
    detail: str
    existing_id: str = ""
    existing_claim: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["_record"] = "memory_conflict_finding"
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryConflictFinding":
        return cls(
            proposal_id=str(data["proposal_id"]),
            proposal_type=str(data.get("proposal_type", "")),
            claim=str(data.get("claim", "")),
            code=str(data["code"]),
            severity=str(data.get("severity", "")),
            detail=str(data.get("detail", "")),
            existing_id=str(data.get("existing_id", "")),
            existing_claim=str(data.get("existing_claim", "")),
        )


def _finding_sort_key(f: MemoryConflictFinding) -> tuple:
    return (f.proposal_id, _SEVERITY_RANK.get(f.severity, 9), f.code,
            f.existing_id)


@dataclass(frozen=True)
class MemoryConflictReport:
    """A deterministic, read-only set of conflict/staleness findings."""

    findings: Tuple[MemoryConflictFinding, ...] = ()

    def counts_by_severity(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    def counts_by_code(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.findings:
            counts[f.code] = counts.get(f.code, 0) + 1
        return counts

    def to_jsonl(self) -> str:
        ordered = sorted(self.findings, key=_finding_sort_key)
        return "\n".join(f.to_json() for f in ordered)


# --------------------------------------------------------------------------- #
# Loading (pure reads; no writes, no ledger)
# --------------------------------------------------------------------------- #
def load_existing_memory(path) -> List[ExistingMemoryRecord]:
    """Load existing memory records from a JSONL file (``#`` comments allowed)."""
    out: List[ExistingMemoryRecord] = []
    file_path = Path(path)
    if not file_path.exists():
        return out
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(ExistingMemoryRecord.from_dict(json.loads(stripped)))
    return out


def load_proposals_for_check(path) -> List[MemoryProposalQuality]:
    """Load proposals from a v5.0 export *or* a v5.1 review-queue JSONL.

    A review-queue entry wraps the proposal in ``proposal_snapshot``; a v5.0
    export is the proposal dict itself. Either is accepted so an approved
    proposal can be checked — checking never writes it. ``#`` lines are comments.
    """
    out: List[MemoryProposalQuality] = []
    file_path = Path(path)
    if not file_path.exists():
        return out
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        data = json.loads(stripped)
        if isinstance(data.get("proposal_snapshot"), dict):
            data = data["proposal_snapshot"]
        out.append(MemoryProposalQuality.from_dict(data))
    return out


# --------------------------------------------------------------------------- #
# Detection (pure; deterministic; conservative)
# --------------------------------------------------------------------------- #
def _is_evidence_supported(proposal: MemoryProposalQuality) -> bool:
    return bool(proposal.evidence_source_id.strip()
                and proposal.evidence_text.strip())


def _pair_relation(proposal: MemoryProposalQuality,
                   existing: ExistingMemoryRecord, now: datetime
                   ) -> Optional[Tuple[str, str, str]]:
    """Return the single strongest (code, severity, detail) for one pair, or None.

    Priority: exact duplicate > contradiction (polarity flip) > supersession
    (same project-state topic) > possible duplicate. One relation per pair keeps
    the report from double-counting the same overlap.
    """
    if not existing.claim.strip():
        return None

    if _normalized_key(proposal.claim) == _normalized_key(existing.claim):
        return (MemoryConflictCode.DUPLICATE_CLAIM,
                _SEVERITY_BY_CODE[MemoryConflictCode.DUPLICATE_CLAIM],
                f"exact duplicate of existing memory {existing.memory_id}")

    p_content = _content_tokens(proposal.claim)
    e_content = _content_tokens(existing.claim)
    overlap = _jaccard(p_content, e_content)

    if overlap >= _CONTRADICTION_OVERLAP \
            and (_has_negation(proposal.claim) ^ _has_negation(existing.claim)):
        return (MemoryConflictCode.CONTRADICTS_EXISTING_MEMORY,
                _SEVERITY_BY_CODE[MemoryConflictCode.CONTRADICTS_EXISTING_MEMORY],
                f"polarity conflict with existing memory {existing.memory_id}")

    if proposal.proposal_type == MemoryProposalType.PROJECT_STATE \
            and existing.memory_type == MemoryProposalType.PROJECT_STATE \
            and overlap >= _SUPERSEDE_JACCARD:
        proposal_dt = _parse_dt(proposal.created_at) or now
        existing_dt = _parse_dt(existing.created_at)
        if existing_dt is not None and proposal_dt < existing_dt:
            return (MemoryConflictCode.SUPERSEDED_BY_EXISTING_MEMORY,
                    _SEVERITY_BY_CODE[
                        MemoryConflictCode.SUPERSEDED_BY_EXISTING_MEMORY],
                    f"newer existing memory {existing.memory_id} may supersede "
                    f"this project-state claim")
        return (MemoryConflictCode.SUPERSEDES_EXISTING_MEMORY,
                _SEVERITY_BY_CODE[MemoryConflictCode.SUPERSEDES_EXISTING_MEMORY],
                f"updates the project-state in existing memory "
                f"{existing.memory_id}")

    if overlap >= _POSSIBLE_DUPLICATE_JACCARD:
        return (MemoryConflictCode.POSSIBLE_DUPLICATE,
                _SEVERITY_BY_CODE[MemoryConflictCode.POSSIBLE_DUPLICATE],
                f"high overlap with existing memory {existing.memory_id} "
                f"(jaccard {overlap:.2f})")

    return None


def _staleness_finding(proposal: MemoryProposalQuality, now: datetime
                       ) -> Optional[Tuple[str, str, str]]:
    """Conservative staleness check on a proposal's own ``created_at`` vs now."""
    created = _parse_dt(proposal.created_at)
    if created is None:
        return None
    age = now - created
    if proposal.proposal_type == MemoryProposalType.PROJECT_STATE \
            and age > timedelta(days=_PROJECT_STATE_STALE_DAYS):
        return (MemoryConflictCode.STALE_PROJECT_STATE,
                _SEVERITY_BY_CODE[MemoryConflictCode.STALE_PROJECT_STATE],
                f"project-state claim is {age.days} days old "
                f"(> {_PROJECT_STATE_STALE_DAYS}d); may no longer hold")
    if proposal.proposal_type == MemoryProposalType.USER_PREFERENCE \
            and age > timedelta(days=_USER_PREFERENCE_STALE_DAYS):
        return (MemoryConflictCode.STALE_USER_PREFERENCE,
                _SEVERITY_BY_CODE[MemoryConflictCode.STALE_USER_PREFERENCE],
                f"user preference is {age.days} days old "
                f"(> {_USER_PREFERENCE_STALE_DAYS}d); confirm it still holds")
    return None


def detect_memory_conflicts(proposals: List[MemoryProposalQuality],
                            existing: List[ExistingMemoryRecord], *,
                            now: Optional[datetime] = None
                            ) -> MemoryConflictReport:
    """Detect conflict/staleness risk for proposals vs existing memory (read-only).

    Returns a deterministic :class:`MemoryConflictReport`. Pure: it reads the
    proposals and existing records and writes nothing — no ledger, no queue, no
    memory. Every proposal that is factual and otherwise clean still receives a
    ``needs_human_review`` finding, because nothing is ever written automatically.
    """
    now = now or datetime.now(timezone.utc)
    findings: List[MemoryConflictFinding] = []

    for proposal in proposals:
        produced: List[MemoryConflictFinding] = []

        def _add(rel: Optional[Tuple[str, str, str]],
                 existing_id: str = "", existing_claim: str = "") -> None:
            if rel is None:
                return
            code, severity, detail = rel
            produced.append(MemoryConflictFinding(
                proposal_id=proposal.proposal_id,
                proposal_type=proposal.proposal_type,
                claim=proposal.claim,
                code=code,
                severity=severity,
                detail=detail,
                existing_id=existing_id,
                existing_claim=existing_claim,
            ))

        # Not-writeable categories (surfaced explicitly, never approvable here).
        if proposal.proposal_type == MemoryProposalType.INVALID_CANDIDATE:
            _add((MemoryConflictCode.INVALID_CANDIDATE,
                  _SEVERITY_BY_CODE[MemoryConflictCode.INVALID_CANDIDATE],
                  proposal.invalid_reason
                  or "invalid candidate; not writeable as memory"))
        elif proposal.proposal_type in _TRACKED_NON_FACTUAL_TYPES:
            _add((MemoryConflictCode.NON_FACTUAL_NOT_WRITEABLE,
                  _SEVERITY_BY_CODE[MemoryConflictCode.NON_FACTUAL_NOT_WRITEABLE],
                  f"{proposal.proposal_type} is tracked but not directly "
                  f"writeable as memory"))

        if proposal.proposal_type in _FACTUAL_TYPES:
            if proposal.proposal_type in _EVIDENCE_REQUIRED_TYPES \
                    and not _is_evidence_supported(proposal):
                _add((MemoryConflictCode.MISSING_EVIDENCE,
                      _SEVERITY_BY_CODE[MemoryConflictCode.MISSING_EVIDENCE],
                      "evidence-required type has no bound evidence source/text"))
            if proposal.confidence < _LOW_CONFIDENCE_THRESHOLD:
                _add((MemoryConflictCode.LOW_CONFIDENCE,
                      _SEVERITY_BY_CODE[MemoryConflictCode.LOW_CONFIDENCE],
                      f"confidence {proposal.confidence} below "
                      f"{_LOW_CONFIDENCE_THRESHOLD}"))
            _add(_staleness_finding(proposal, now))
            for record in existing:
                _add(_pair_relation(proposal, record, now),
                     existing_id=record.memory_id, existing_claim=record.claim)

        # A factual proposal with no error/warning still needs a human: nothing
        # is ever written automatically.
        if proposal.proposal_type in _FACTUAL_TYPES and not any(
                f.severity in (MemoryConflictSeverity.ERROR,
                               MemoryConflictSeverity.WARNING)
                for f in produced):
            _add((MemoryConflictCode.NEEDS_HUMAN_REVIEW,
                  _SEVERITY_BY_CODE[MemoryConflictCode.NEEDS_HUMAN_REVIEW],
                  "no automatic conflict; still requires human approval before "
                  "any future write"))

        findings.extend(produced)

    findings.sort(key=_finding_sort_key)
    return MemoryConflictReport(findings=tuple(findings))


# --------------------------------------------------------------------------- #
# Deterministic rendering / export (the only writer is write_conflict_report)
# --------------------------------------------------------------------------- #
def render_conflict_report_markdown(report: MemoryConflictReport) -> str:
    """Deterministic Markdown view of a conflict report. No side effects."""
    ordered = sorted(report.findings, key=_finding_sort_key)
    lines: List[str] = ["# Memory conflict + staleness report (v5.2)", ""]
    lines.append("Read-only risk detection. Conflict detection reports risk; it "
                 "**does not resolve it or write memory** — nothing here writes "
                 "the memory ledger, mutates existing memory, applies a "
                 "proposal, or changes retrieval, ranking, source selection, "
                 "grounding, or composers.")
    lines.append("")
    lines.append(f"- proposals with findings: "
                 f"{len({f.proposal_id for f in ordered})}")
    lines.append(f"- findings: {len(ordered)}")
    lines.append("")

    sev_counts = report.counts_by_severity()
    lines.append("## Summary by severity")
    lines.append("")
    if sev_counts:
        for severity in sorted(sev_counts,
                               key=lambda s: _SEVERITY_RANK.get(s, 9)):
            lines.append(f"- {severity}: {sev_counts[severity]}")
    else:
        lines.append("- (no findings)")
    lines.append("")

    code_counts = report.counts_by_code()
    lines.append("## Summary by code")
    lines.append("")
    if code_counts:
        for code in sorted(code_counts):
            lines.append(f"- {code}: {code_counts[code]}")
    else:
        lines.append("- (no findings)")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    if ordered:
        lines.append("| severity | code | proposal_id | proposal_type | "
                     "existing_id | detail |")
        lines.append("|---|---|---|---|---|---|")
        for f in ordered:
            detail = f.detail.replace("|", "\\|")
            lines.append(
                f"| {f.severity} | {f.code} | {f.proposal_id} "
                f"| {f.proposal_type} | {f.existing_id} | {detail} |")
    else:
        lines.append("(no findings)")
    return "\n".join(lines)


def write_conflict_report(report: MemoryConflictReport, path) -> None:
    """Write the conflict report to a JSONL file (sorted; deterministic).

    This is the **only** writer in this module. It writes a read-only report and
    never touches the memory ledger, existing memory, the proposal queue, or any
    source file.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = report.to_jsonl()
    file_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
