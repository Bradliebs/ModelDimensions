"""v4.0 Source Registry Foundation — read-only source metadata (no retrieval change).

This module adds a place to record *metadata about sources* — their authority,
freshness, ownership, review status, supersession, and notes — separately from
the source text itself. It is deliberately a thin, inspectable layer:

* The registry is **metadata, not truth.** A registry entry describes a source;
  it never replaces the source text, and nothing here decides what a query is
  answered with. The evidence is still the imported knowledge chunk; the
  registry only annotates where that evidence came from and how fresh it is.
* It changes **no** retrieval, ranking, source-selection, grounding, composer,
  or memory behaviour. Loading, inspecting, and computing an effective freshness
  status are all pure reads. The only writer (``save_registry``) is an explicit
  API a human calls; it is never invoked automatically and the CLI never writes.
* A later slice *may* let the registry influence retrieval or ranking. In v4.0 it
  does not — that boundary is the whole point of this foundation.

The on-disk format mirrors the rest of the workbench: one self-describing JSON
object per line, with ``#`` comment lines allowed, readable by eye.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


# -- controlled vocabularies --------------------------------------------------

class AuthorityLevel(str, Enum):
    """How much weight a source's origin carries (mirrors the knowledge model).

    ``official`` (the vendor/standard itself) and ``reputable`` (a well-regarded
    secondary source) are trusted; ``community`` and ``unknown`` are weaker. The
    registry only *records* this; it does not act on it in v4.0.
    """

    OFFICIAL = "official"
    REPUTABLE = "reputable"
    COMMUNITY = "community"
    UNKNOWN = "unknown"


class SourceStatus(str, Enum):
    """The lifecycle state a source is in.

    * ``active`` — in use and considered current.
    * ``stale`` — past its review window; still readable but flagged.
    * ``deprecated`` — retired, usually because something supersedes it.
    * ``draft`` — not yet ratified; present for tracking, not for relying on.
    """

    ACTIVE = "active"
    STALE = "stale"
    DEPRECATED = "deprecated"
    DRAFT = "draft"


# -- registry entry -----------------------------------------------------------

@dataclass(frozen=True)
class SourceRegistryEntry:
    """One source's metadata. ``source_id`` is the only required field.

    Every other field defaults safely so a sparse, hand-written entry loads
    without error. Lists default to empty; optional dates and ``stale_after_days``
    default to ``None`` (which makes freshness computation a no-op for that
    entry). The entry is frozen: computing an effective status never mutates it.
    """

    source_id: str
    title: str = ""
    source_type: str = ""
    authority_level: AuthorityLevel = AuthorityLevel.UNKNOWN
    owner: str = ""
    created_at: Optional[str] = None
    last_reviewed_at: Optional[str] = None
    freshness_policy: str = "static"
    stale_after_days: Optional[int] = None
    topics: List[str] = field(default_factory=list)
    linked_decisions: List[str] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)
    superseded_by: List[str] = field(default_factory=list)
    status: SourceStatus = SourceStatus.ACTIVE
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "SourceRegistryEntry":
        """Build an entry from a plain dict, validating the controlled fields.

        Missing optional fields default safely. An invalid ``status`` or
        ``authority_level`` fails cleanly with a ``ValueError`` naming the field;
        a missing/empty ``source_id`` also fails cleanly.
        """
        source_id = str(data.get("source_id") or "").strip()
        if not source_id:
            raise ValueError("source registry entry requires a non-empty source_id")

        try:
            authority = AuthorityLevel(data.get("authority_level", "unknown"))
        except ValueError:
            raise ValueError(
                f"invalid authority_level {data.get('authority_level')!r} "
                f"for source {source_id!r}; "
                f"expected one of {[a.value for a in AuthorityLevel]}")

        try:
            status = SourceStatus(data.get("status", "active"))
        except ValueError:
            raise ValueError(
                f"invalid status {data.get('status')!r} for source {source_id!r}; "
                f"expected one of {[s.value for s in SourceStatus]}")

        stale_after = data.get("stale_after_days")
        stale_after_days = None if stale_after is None else int(stale_after)

        return cls(
            source_id=source_id,
            title=str(data.get("title", "")),
            source_type=str(data.get("source_type", "")),
            authority_level=authority,
            owner=str(data.get("owner", "")),
            created_at=data.get("created_at"),
            last_reviewed_at=data.get("last_reviewed_at"),
            freshness_policy=str(data.get("freshness_policy", "static")),
            stale_after_days=stale_after_days,
            topics=list(data.get("topics") or []),
            linked_decisions=list(data.get("linked_decisions") or []),
            supersedes=list(data.get("supersedes") or []),
            superseded_by=list(data.get("superseded_by") or []),
            status=status,
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict (enums rendered as their string values)."""
        data = asdict(self)
        data["authority_level"] = self.authority_level.value
        data["status"] = self.status.value
        data["_record"] = "source_registry_entry"
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# -- load / save --------------------------------------------------------------

def load_registry(path: str | Path) -> List[SourceRegistryEntry]:
    """Load registry entries from a JSONL file (``#`` lines are comments).

    A pure read: it parses the file and returns entries, touching nothing else.
    """
    entries: List[SourceRegistryEntry] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(SourceRegistryEntry.from_dict(json.loads(line)))
    return entries


def save_registry(entries: List[SourceRegistryEntry], path: str | Path) -> None:
    """Write registry entries to a JSONL file.

    This is the **only** writer in the module and it is never called
    automatically — neither the CLI nor any retrieval/compose path invokes it.
    It exists so a human (or an explicit future tool) can persist a registry on
    purpose. It writes only the registry file; no source text, ledger, queue, or
    knowledge library is touched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.to_json() + "\n")


# -- inspection helpers (all pure) -------------------------------------------

def index_by_id(entries: List[SourceRegistryEntry]
                ) -> Dict[str, SourceRegistryEntry]:
    """Index entries by ``source_id`` (last wins on duplicate ids)."""
    return {e.source_id: e for e in entries}


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO date or datetime into an aware UTC datetime, or ``None``."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def compute_effective_status(entry: SourceRegistryEntry, *,
                             now: Optional[datetime] = None) -> SourceStatus:
    """Compute a source's *effective* freshness status without mutating anything.

    This is a pure read: it returns a :class:`SourceStatus` derived from the
    entry's stored status, review date, and staleness window. It never changes
    the entry (which is frozen) and never writes to disk — staleness is a
    *computed view*, not a stored fact, in v4.0.

    Rules:

    * ``deprecated`` and ``draft`` are terminal lifecycle states — returned as-is.
    * If the entry sets no ``stale_after_days`` or no ``last_reviewed_at``, there
      is nothing to age against, so the stored status is returned unchanged.
    * Otherwise, if the review is older than ``stale_after_days``, the effective
      status is ``stale``; if it is within the window, it is ``active``.
    """
    if entry.status in (SourceStatus.DEPRECATED, SourceStatus.DRAFT):
        return entry.status
    if entry.stale_after_days is None or not entry.last_reviewed_at:
        return entry.status
    reviewed = _parse_iso(entry.last_reviewed_at)
    if reviewed is None:
        return entry.status
    now = now or datetime.now(timezone.utc)
    age_days = (now - reviewed).days
    if age_days > entry.stale_after_days:
        return SourceStatus.STALE
    return SourceStatus.ACTIVE


def supersession_warnings(entries: List[SourceRegistryEntry]) -> List[str]:
    """Return read-only integrity notes about supersession links.

    Flags dangling references (a ``supersedes``/``superseded_by`` id with no
    matching entry) and asymmetric links (A supersedes B but B does not record
    ``superseded_by`` A). This reports; it never edits the registry.
    """
    ids = index_by_id(entries)
    warnings: List[str] = []
    for entry in entries:
        for target in entry.supersedes:
            if target not in ids:
                warnings.append(
                    f"{entry.source_id} supersedes unknown source {target}")
            elif entry.source_id not in ids[target].superseded_by:
                warnings.append(
                    f"{entry.source_id} supersedes {target}, but {target} "
                    f"does not record superseded_by {entry.source_id}")
        for target in entry.superseded_by:
            if target not in ids:
                warnings.append(
                    f"{entry.source_id} is superseded by unknown source {target}")
    return warnings


# -- rendering (deterministic, no side effects) ------------------------------

def render_registry_markdown(entries: List[SourceRegistryEntry], *,
                             now: Optional[datetime] = None) -> str:
    """Render a deterministic Markdown table of all entries (sorted by id).

    Output depends only on the entries and the supplied ``now`` (used for the
    computed effective-status column), so the same registry renders identically
    every time. No side effects.
    """
    lines: List[str] = []
    lines.append("# Source registry")
    lines.append("")
    lines.append("Read-only source metadata (v4.0). The registry annotates "
                 "sources; it never replaces source text or changes retrieval.")
    lines.append("")
    lines.append(f"- entries: {len(entries)}")
    warnings = supersession_warnings(entries)
    lines.append(f"- supersession warnings: {len(warnings)}")
    lines.append("")
    lines.append("| source_id | title | authority | status | effective | "
                 "owner | topics |")
    lines.append("|---|---|---|---|---|---|---|")
    for entry in sorted(entries, key=lambda e: e.source_id):
        effective = compute_effective_status(entry, now=now).value
        topics = ", ".join(entry.topics) if entry.topics else "—"
        lines.append(
            f"| {entry.source_id} | {entry.title or '—'} | "
            f"{entry.authority_level.value} | {entry.status.value} | "
            f"{effective} | {entry.owner or '—'} | {topics} |")
    lines.append("")
    return "\n".join(lines)


def render_entry_markdown(entry: SourceRegistryEntry,
                          entries: Optional[List[SourceRegistryEntry]] = None, *,
                          now: Optional[datetime] = None) -> str:
    """Render a deterministic detailed view of a single entry. No side effects."""
    effective = compute_effective_status(entry, now=now).value
    lines: List[str] = []
    lines.append(f"# Source: {entry.source_id}")
    lines.append("")
    lines.append(f"- title: {entry.title or '—'}")
    lines.append(f"- source_type: {entry.source_type or '—'}")
    lines.append(f"- authority_level: {entry.authority_level.value}")
    lines.append(f"- owner: {entry.owner or '—'}")
    lines.append(f"- created_at: {entry.created_at or '—'}")
    lines.append(f"- last_reviewed_at: {entry.last_reviewed_at or '—'}")
    lines.append(f"- freshness_policy: {entry.freshness_policy}")
    lines.append(f"- stale_after_days: "
                 f"{'—' if entry.stale_after_days is None else entry.stale_after_days}")
    lines.append(f"- status (stored): {entry.status.value}")
    lines.append(f"- status (effective): {effective}")
    lines.append(f"- topics: {', '.join(entry.topics) if entry.topics else '—'}")
    lines.append(f"- linked_decisions: "
                 f"{', '.join(entry.linked_decisions) if entry.linked_decisions else '—'}")
    lines.append(f"- supersedes: "
                 f"{', '.join(entry.supersedes) if entry.supersedes else '—'}")
    lines.append(f"- superseded_by: "
                 f"{', '.join(entry.superseded_by) if entry.superseded_by else '—'}")
    lines.append(f"- notes: {entry.notes or '—'}")
    if entries is not None:
        warnings = [w for w in supersession_warnings(entries)
                    if entry.source_id in w]
        if warnings:
            lines.append("")
            lines.append("## Supersession warnings")
            lines.append("")
            for w in warnings:
                lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)


# -- v4.1 audit layer (read-only; reports lifecycle/metadata risk) -----------
#
# The audit walks the registry and *reports* findings about each source's
# lifecycle (stale/deprecated/draft), its metadata completeness (owner, review
# date, topics, authority), and its supersession integrity (dangling/asymmetric
# links). It is the same kind of pure read as ``compute_effective_status`` and
# ``supersession_warnings``: it never mutates an entry (frozen), never writes a
# file (``save_registry`` stays the only writer and is never called here), and
# never touches retrieval, ranking, source-selection, grounding, composer, or
# memory. Crucially, the audit **does not decide a source is false** — source
# text remains the evidence; this only flags where the *metadata* carries risk.


class FindingSeverity(str, Enum):
    """How much attention a finding warrants.

    * ``error`` — a registry integrity problem (broken supersession lineage).
    * ``warning`` — a lifecycle or metadata gap that needs human review.
    * ``info`` — a normal, intentional state worth surfacing (deprecated/draft,
      minor metadata completeness notes).
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FindingCode:
    """Stable string codes for audit findings (grouped by what they describe).

    Codes are plain strings so reports serialise readably and tests pin exact
    values. :data:`FINDING_SEVERITY` maps each code to its severity.
    """

    # lifecycle / freshness
    STALE_BY_POLICY = "stale_by_policy"
    STALE_BY_STATUS = "stale_by_status"
    DEPRECATED_SOURCE = "deprecated_source"
    DRAFT_SOURCE = "draft_source"
    # metadata completeness
    MISSING_LAST_REVIEWED_AT = "missing_last_reviewed_at"
    MISSING_OWNER = "missing_owner"
    MISSING_TOPICS = "missing_topics"
    UNKNOWN_AUTHORITY_LEVEL = "unknown_authority_level"
    # supersession integrity
    DANGLING_SUPERSEDES = "dangling_supersedes"
    DANGLING_SUPERSEDED_BY = "dangling_superseded_by"
    ASYMMETRIC_SUPERSESSION = "asymmetric_supersession"
    ACTIVE_SOURCE_SUPERSEDED = "active_source_superseded"
    DEPRECATED_SOURCE_WITHOUT_SUCCESSOR = "deprecated_source_without_successor"


# Single source of truth for each code's severity. (The model collapses a
# missing ``authority_level`` to ``unknown``, so a missing/explicit-unknown
# authority both surface as ``unknown_authority_level`` — see README.)
FINDING_SEVERITY: Dict[str, FindingSeverity] = {
    FindingCode.STALE_BY_POLICY: FindingSeverity.WARNING,
    FindingCode.STALE_BY_STATUS: FindingSeverity.WARNING,
    FindingCode.DEPRECATED_SOURCE: FindingSeverity.INFO,
    FindingCode.DRAFT_SOURCE: FindingSeverity.INFO,
    FindingCode.MISSING_LAST_REVIEWED_AT: FindingSeverity.WARNING,
    FindingCode.MISSING_OWNER: FindingSeverity.WARNING,
    FindingCode.MISSING_TOPICS: FindingSeverity.INFO,
    FindingCode.UNKNOWN_AUTHORITY_LEVEL: FindingSeverity.INFO,
    FindingCode.DANGLING_SUPERSEDES: FindingSeverity.ERROR,
    FindingCode.DANGLING_SUPERSEDED_BY: FindingSeverity.ERROR,
    FindingCode.ASYMMETRIC_SUPERSESSION: FindingSeverity.ERROR,
    FindingCode.ACTIVE_SOURCE_SUPERSEDED: FindingSeverity.WARNING,
    FindingCode.DEPRECATED_SOURCE_WITHOUT_SUCCESSOR: FindingSeverity.WARNING,
}

# Deterministic ordering: errors first, then warnings, then info.
_SEVERITY_RANK = {
    FindingSeverity.ERROR: 0,
    FindingSeverity.WARNING: 1,
    FindingSeverity.INFO: 2,
}


@dataclass(frozen=True)
class SourceRegistryFinding:
    """One audit observation about one source. Pure data; no behaviour.

    ``source_id`` is the source the finding is about, ``code`` is a stable
    :class:`FindingCode` value, ``severity`` its mapped :class:`FindingSeverity`,
    and ``message`` a human-readable explanation. A finding reports *metadata or
    lifecycle risk*; it never asserts the source's content is wrong.
    """

    source_id: str
    code: str
    severity: FindingSeverity
    message: str

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "_record": "source_registry_finding",
        }


@dataclass(frozen=True)
class SourceRegistryAuditReport:
    """The full result of auditing a registry. Deterministic and inert.

    ``findings`` are pre-sorted (severity, code, source_id). ``counts_by_severity``
    and ``counts_by_code`` are summary tallies. ``entry_count`` is how many
    entries were audited. Building this report writes nothing and mutates nothing.
    """

    findings: List[SourceRegistryFinding]
    counts_by_severity: Dict[str, int]
    counts_by_code: Dict[str, int]
    entry_count: int

    @property
    def error_count(self) -> int:
        return self.counts_by_severity.get(FindingSeverity.ERROR.value, 0)

    @property
    def warning_count(self) -> int:
        return self.counts_by_severity.get(FindingSeverity.WARNING.value, 0)

    @property
    def info_count(self) -> int:
        return self.counts_by_severity.get(FindingSeverity.INFO.value, 0)

    def findings_for(self, source_id: str) -> List[SourceRegistryFinding]:
        return [f for f in self.findings if f.source_id == source_id]

    def to_dict(self) -> dict:
        return {
            "entry_count": self.entry_count,
            "counts_by_severity": dict(self.counts_by_severity),
            "counts_by_code": dict(self.counts_by_code),
            "findings": [f.to_dict() for f in self.findings],
            "_record": "source_registry_audit_report",
        }


def _finding(source_id: str, code: str, message: str) -> SourceRegistryFinding:
    """Build a finding, resolving severity from the code's mapping."""
    return SourceRegistryFinding(
        source_id=source_id,
        code=code,
        severity=FINDING_SEVERITY[code],
        message=message,
    )


def audit_registry(entries: List[SourceRegistryEntry], *,
                   now: Optional[datetime] = None) -> SourceRegistryAuditReport:
    """Audit a registry and return a deterministic, read-only findings report.

    Pure: it reads the entries (and the supplied ``now`` for freshness) and
    returns a report. It never mutates an entry, never writes a file, and never
    affects retrieval/ranking/grounding/composer/memory. Findings describe
    *metadata and lifecycle risk only* — they never claim a source's text is
    false.

    Checks, per entry:

    * **lifecycle / freshness** — ``stale_by_status`` (stored status is
      ``stale``), ``stale_by_policy`` (review window expired so the *computed*
      effective status is stale while stored status is active), ``deprecated_source``
      and ``draft_source`` (surfacing those intentional states);
    * **metadata completeness** — ``missing_last_reviewed_at``, ``missing_owner``,
      ``missing_topics``, ``unknown_authority_level``;
    * **supersession integrity** — ``dangling_supersedes`` /
      ``dangling_superseded_by`` (link to a non-existent id),
      ``asymmetric_supersession`` (a one-sided link), ``active_source_superseded``
      (an ``active`` source something supersedes — likely should be deprecated),
      and ``deprecated_source_without_successor`` (retired with no successor).
    """
    ids = index_by_id(entries)
    findings: List[SourceRegistryFinding] = []

    for entry in entries:
        sid = entry.source_id

        # -- lifecycle / freshness --
        if entry.status is SourceStatus.STALE:
            findings.append(_finding(
                sid, FindingCode.STALE_BY_STATUS,
                "stored status is 'stale'; source is flagged for review"))
        elif compute_effective_status(entry, now=now) is SourceStatus.STALE:
            findings.append(_finding(
                sid, FindingCode.STALE_BY_POLICY,
                f"last_reviewed_at is older than stale_after_days "
                f"({entry.stale_after_days}); effective status computes stale"))

        if entry.status is SourceStatus.DEPRECATED:
            findings.append(_finding(
                sid, FindingCode.DEPRECATED_SOURCE,
                "source is deprecated (retired from active use)"))
        if entry.status is SourceStatus.DRAFT:
            findings.append(_finding(
                sid, FindingCode.DRAFT_SOURCE,
                "source is a draft (not yet ratified)"))

        # -- metadata completeness --
        if not entry.last_reviewed_at:
            findings.append(_finding(
                sid, FindingCode.MISSING_LAST_REVIEWED_AT,
                "no last_reviewed_at recorded; freshness cannot be assessed"))
        if not entry.owner:
            findings.append(_finding(
                sid, FindingCode.MISSING_OWNER,
                "no owner recorded; source has no accountable owner"))
        if not entry.topics:
            findings.append(_finding(
                sid, FindingCode.MISSING_TOPICS,
                "no topics recorded; source is harder to classify/discover"))
        if entry.authority_level is AuthorityLevel.UNKNOWN:
            findings.append(_finding(
                sid, FindingCode.UNKNOWN_AUTHORITY_LEVEL,
                "authority_level is unknown (missing or unestablished)"))

        # -- supersession integrity --
        if entry.status is SourceStatus.ACTIVE and entry.superseded_by:
            findings.append(_finding(
                sid, FindingCode.ACTIVE_SOURCE_SUPERSEDED,
                f"status is active but superseded_by "
                f"{', '.join(entry.superseded_by)}; consider deprecating"))
        if entry.status is SourceStatus.DEPRECATED and not entry.superseded_by:
            findings.append(_finding(
                sid, FindingCode.DEPRECATED_SOURCE_WITHOUT_SUCCESSOR,
                "source is deprecated but records no superseded_by successor"))

        for target in entry.supersedes:
            if target not in ids:
                findings.append(_finding(
                    sid, FindingCode.DANGLING_SUPERSEDES,
                    f"supersedes unknown source {target}"))
            elif sid not in ids[target].superseded_by:
                findings.append(_finding(
                    sid, FindingCode.ASYMMETRIC_SUPERSESSION,
                    f"supersedes {target}, but {target} does not record "
                    f"superseded_by {sid}"))
        for target in entry.superseded_by:
            if target not in ids:
                findings.append(_finding(
                    sid, FindingCode.DANGLING_SUPERSEDED_BY,
                    f"superseded_by unknown source {target}"))
            elif sid not in ids[target].supersedes:
                findings.append(_finding(
                    sid, FindingCode.ASYMMETRIC_SUPERSESSION,
                    f"superseded_by {target}, but {target} does not record "
                    f"supersedes {sid}"))

    findings.sort(key=lambda f: (_SEVERITY_RANK[f.severity], f.code, f.source_id))

    counts_by_severity: Dict[str, int] = {}
    counts_by_code: Dict[str, int] = {}
    for finding in findings:
        counts_by_severity[finding.severity.value] = (
            counts_by_severity.get(finding.severity.value, 0) + 1)
        counts_by_code[finding.code] = counts_by_code.get(finding.code, 0) + 1

    return SourceRegistryAuditReport(
        findings=findings,
        counts_by_severity=counts_by_severity,
        counts_by_code=counts_by_code,
        entry_count=len(entries),
    )


def render_audit_markdown(report: SourceRegistryAuditReport) -> str:
    """Render a deterministic Markdown view of an audit report. No side effects.

    Output depends only on the report (itself already deterministic), so the same
    registry audits to identical text every time. Nothing is written.
    """
    lines: List[str] = []
    lines.append("# Source registry audit")
    lines.append("")
    lines.append("Read-only lifecycle/metadata risk report (v4.1). The audit "
                 "flags metadata risk; it never decides a source is false, and "
                 "it changes no retrieval/ranking/grounding behaviour.")
    lines.append("")
    lines.append(f"- entries audited: {report.entry_count}")
    lines.append(f"- findings: {len(report.findings)} "
                 f"(error {report.error_count}, warning {report.warning_count}, "
                 f"info {report.info_count})")
    lines.append("")

    lines.append("## Summary by severity")
    lines.append("")
    if report.counts_by_severity:
        for severity in (FindingSeverity.ERROR, FindingSeverity.WARNING,
                         FindingSeverity.INFO):
            count = report.counts_by_severity.get(severity.value, 0)
            if count:
                lines.append(f"- {severity.value}: {count}")
    else:
        lines.append("- (no findings)")
    lines.append("")

    lines.append("## Summary by code")
    lines.append("")
    if report.counts_by_code:
        for code in sorted(report.counts_by_code):
            lines.append(f"- {code}: {report.counts_by_code[code]}")
    else:
        lines.append("- (no findings)")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    if report.findings:
        lines.append("| severity | code | source_id | message |")
        lines.append("|---|---|---|---|")
        for f in report.findings:
            lines.append(
                f"| {f.severity.value} | {f.code} | {f.source_id} | {f.message} |")
    else:
        lines.append("No findings — every source's lifecycle and metadata are "
                     "complete and its supersession links are consistent.")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# v4.2 — Source update proposal generator (read-only; tasks, not truth claims)
# -----------------------------------------------------------------------------
# This layer turns audit findings (v4.1) into structured *maintenance proposals*
# a human can review and apply by hand. It is strictly proposal-generation:
#
#   * it mutates NO registry file (``save_registry`` is still the only registry
#     writer, and is never called from here),
#   * it mutates NO source/knowledge file,
#   * it changes NO retrieval/ranking/source-selection/grounding/composer/memory
#     behaviour, and writes NO memory ledger,
#   * every proposal carries ``requires_human_approval=True`` and
#     ``status="proposed"`` — nothing is ever applied automatically.
#
# A proposal is a TASK ("a human should look at this metadata"), never a truth
# claim about a source's content. The only file this layer may write is the
# explicit proposal export requested via :func:`write_proposals`; it never
# touches the registry or source truth.


class ProposalType:
    """Stable string codes for the kinds of maintenance task a proposal asks for."""

    REVIEW_STALE_SOURCE = "review_stale_source"
    ADD_MISSING_OWNER = "add_missing_owner"
    ADD_LAST_REVIEWED_AT = "add_last_reviewed_at"
    ADD_TOPICS = "add_topics"
    RESOLVE_DANGLING_SUPERSESSION = "resolve_dangling_supersession"
    RESOLVE_ASYMMETRIC_SUPERSESSION = "resolve_asymmetric_supersession"
    SET_SUCCESSOR_FOR_DEPRECATED_SOURCE = "set_successor_for_deprecated_source"
    CLARIFY_DRAFT_SOURCE = "clarify_draft_source"
    REVIEW_UNKNOWN_AUTHORITY = "review_unknown_authority"


# Which finding code becomes which proposal type. Findings absent from this map
# produce no proposal on purpose:
#   * ``deprecated_source`` is a settled, intentional state (no task needed);
#   * ``active_source_superseded`` is advisory ("consider deprecating") — the
#     deprecation itself, once a human makes it, is what triggers a
#     ``set_successor_for_deprecated_source`` proposal.
PROPOSAL_TYPE_BY_FINDING: Dict[str, str] = {
    FindingCode.STALE_BY_POLICY: ProposalType.REVIEW_STALE_SOURCE,
    FindingCode.STALE_BY_STATUS: ProposalType.REVIEW_STALE_SOURCE,
    FindingCode.MISSING_OWNER: ProposalType.ADD_MISSING_OWNER,
    FindingCode.MISSING_LAST_REVIEWED_AT: ProposalType.ADD_LAST_REVIEWED_AT,
    FindingCode.MISSING_TOPICS: ProposalType.ADD_TOPICS,
    FindingCode.DANGLING_SUPERSEDES: ProposalType.RESOLVE_DANGLING_SUPERSESSION,
    FindingCode.DANGLING_SUPERSEDED_BY: ProposalType.RESOLVE_DANGLING_SUPERSESSION,
    FindingCode.ASYMMETRIC_SUPERSESSION: ProposalType.RESOLVE_ASYMMETRIC_SUPERSESSION,
    FindingCode.DEPRECATED_SOURCE_WITHOUT_SUCCESSOR:
        ProposalType.SET_SUCCESSOR_FOR_DEPRECATED_SOURCE,
    FindingCode.DRAFT_SOURCE: ProposalType.CLARIFY_DRAFT_SOURCE,
    FindingCode.UNKNOWN_AUTHORITY_LEVEL: ProposalType.REVIEW_UNKNOWN_AUTHORITY,
}

# Static, human-readable task text per proposal type. Each is an instruction to
# a person, never an automated action.
_PROPOSED_ACTION: Dict[str, str] = {
    ProposalType.REVIEW_STALE_SOURCE:
        "Review the source and refresh last_reviewed_at, or confirm its status.",
    ProposalType.ADD_MISSING_OWNER:
        "Assign an accountable owner for this source.",
    ProposalType.ADD_LAST_REVIEWED_AT:
        "Record a last_reviewed_at date so freshness can be assessed.",
    ProposalType.ADD_TOPICS:
        "Add subject topics so the source is easier to classify and discover.",
    ProposalType.RESOLVE_DANGLING_SUPERSESSION:
        "Fix or remove the supersession link that points to a non-existent source.",
    ProposalType.RESOLVE_ASYMMETRIC_SUPERSESSION:
        "Add the missing reciprocal supersession back-link.",
    ProposalType.SET_SUCCESSOR_FOR_DEPRECATED_SOURCE:
        "Record a superseded_by successor for this deprecated source.",
    ProposalType.CLARIFY_DRAFT_SOURCE:
        "Ratify the draft or confirm it should remain a draft.",
    ProposalType.REVIEW_UNKNOWN_AUTHORITY:
        "Establish and record the source's authority_level.",
}


@dataclass(frozen=True)
class SourceUpdateProposal:
    """One maintenance task derived from one audit finding. Pure data; inert.

    A proposal records *what a human might change* and *why*, never an applied
    change. ``requires_human_approval`` is always ``True`` and ``status`` is
    always ``"proposed"``; ``current_value`` describes the present metadata so a
    reviewer has context. The proposal never asserts the source's content is
    wrong — it only flags metadata worth a human's attention.
    """

    proposal_id: str
    source_id: str
    proposal_type: str
    severity: FindingSeverity
    finding_code: str
    current_value: str
    proposed_action: str
    rationale: str
    requires_human_approval: bool = True
    status: str = "proposed"
    created_at: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "source_id": self.source_id,
            "proposal_type": self.proposal_type,
            "severity": self.severity.value,
            "finding_code": self.finding_code,
            "current_value": self.current_value,
            "proposed_action": self.proposed_action,
            "rationale": self.rationale,
            "requires_human_approval": self.requires_human_approval,
            "status": self.status,
            "created_at": self.created_at,
            "notes": self.notes,
            "_record": "source_update_proposal",
        }


def _proposal_id(source_id: str, finding_code: str, rationale: str) -> str:
    """Deterministic, stable id for a proposal.

    Derived only from the source, the finding code, and the finding's message,
    so the same registry always yields the same id and distinct findings (even
    of the same code, e.g. two dangling links) get distinct ids.
    """
    digest = hashlib.sha1(
        f"{source_id}|{finding_code}|{rationale}".encode("utf-8")).hexdigest()
    return f"srcprop-{digest[:10]}"


def _current_value(entry: SourceRegistryEntry, proposal_type: str) -> str:
    """A short, read-only description of the entry's present metadata."""
    if proposal_type == ProposalType.REVIEW_STALE_SOURCE:
        return (f"status={entry.status.value}, "
                f"last_reviewed_at={entry.last_reviewed_at!r}, "
                f"stale_after_days={entry.stale_after_days}")
    if proposal_type == ProposalType.ADD_MISSING_OWNER:
        return f"owner={entry.owner!r}"
    if proposal_type == ProposalType.ADD_LAST_REVIEWED_AT:
        return f"last_reviewed_at={entry.last_reviewed_at!r}"
    if proposal_type == ProposalType.ADD_TOPICS:
        return f"topics={entry.topics}"
    if proposal_type in (ProposalType.RESOLVE_DANGLING_SUPERSESSION,
                         ProposalType.RESOLVE_ASYMMETRIC_SUPERSESSION):
        return (f"supersedes={entry.supersedes}, "
                f"superseded_by={entry.superseded_by}")
    if proposal_type == ProposalType.SET_SUCCESSOR_FOR_DEPRECATED_SOURCE:
        return (f"status={entry.status.value}, "
                f"superseded_by={entry.superseded_by}")
    if proposal_type == ProposalType.CLARIFY_DRAFT_SOURCE:
        return f"status={entry.status.value}"
    if proposal_type == ProposalType.REVIEW_UNKNOWN_AUTHORITY:
        return f"authority_level={entry.authority_level.value}"
    return ""


def propose_source_updates(entries: List[SourceRegistryEntry], *,
                           now: Optional[datetime] = None,
                           ) -> List[SourceUpdateProposal]:
    """Generate deterministic, read-only maintenance proposals from an audit.

    Runs :func:`audit_registry` internally and converts each mapped finding into
    a :class:`SourceUpdateProposal`. Pure: it reads the entries (and ``now`` for
    freshness) and returns a list. It calls neither :func:`save_registry` nor any
    source/memory writer, and changes no retrieval/ranking/grounding behaviour. A
    clean registry yields an empty list. Output is sorted deterministically by
    (severity, proposal_type, source_id, proposal_id).
    """
    report = audit_registry(entries, now=now)
    ids = index_by_id(entries)
    proposals: List[SourceUpdateProposal] = []

    for finding in report.findings:
        proposal_type = PROPOSAL_TYPE_BY_FINDING.get(finding.code)
        if proposal_type is None:
            continue
        entry = ids[finding.source_id]
        proposals.append(SourceUpdateProposal(
            proposal_id=_proposal_id(finding.source_id, finding.code,
                                     finding.message),
            source_id=finding.source_id,
            proposal_type=proposal_type,
            severity=finding.severity,
            finding_code=finding.code,
            current_value=_current_value(entry, proposal_type),
            proposed_action=_PROPOSED_ACTION[proposal_type],
            rationale=finding.message,
        ))

    proposals.sort(key=lambda p: (_SEVERITY_RANK[p.severity], p.proposal_type,
                                  p.source_id, p.proposal_id))
    return proposals


def proposals_to_jsonl(proposals: List[SourceUpdateProposal]) -> str:
    """Serialise proposals to deterministic JSONL (one proposal per line)."""
    return "\n".join(
        json.dumps(p.to_dict(), ensure_ascii=False) for p in proposals)


def render_proposals_markdown(proposals: List[SourceUpdateProposal]) -> str:
    """Render a deterministic Markdown view of proposals. No side effects.

    Output depends only on the (already deterministic) proposals, so the same
    registry renders identical text every time. Nothing is written.
    """
    counts_by_type: Dict[str, int] = {}
    for p in proposals:
        counts_by_type[p.proposal_type] = counts_by_type.get(p.proposal_type, 0) + 1

    lines: List[str] = []
    lines.append("# Source update proposals")
    lines.append("")
    lines.append("Read-only maintenance proposals generated from the source "
                 "registry audit (v4.2). Proposals are **tasks, not truth "
                 "claims**: each requires human approval and nothing is applied "
                 "automatically. No registry/source file is modified, and no "
                 "retrieval/ranking/grounding/memory behaviour changes.")
    lines.append("")
    lines.append(f"- proposals: {len(proposals)}")
    lines.append("- every proposal requires human approval (status: proposed)")
    lines.append("")

    lines.append("## Summary by type")
    lines.append("")
    if counts_by_type:
        for proposal_type in sorted(counts_by_type):
            lines.append(f"- {proposal_type}: {counts_by_type[proposal_type]}")
    else:
        lines.append("- (no proposals)")
    lines.append("")

    lines.append("## Proposals")
    lines.append("")
    if proposals:
        lines.append("| severity | proposal_type | source_id | finding_code "
                     "| proposed_action | approval |")
        lines.append("|---|---|---|---|---|---|")
        for p in proposals:
            approval = "required" if p.requires_human_approval else "not required"
            lines.append(
                f"| {p.severity.value} | {p.proposal_type} | {p.source_id} "
                f"| {p.finding_code} | {p.proposed_action} | {approval} |")
    else:
        lines.append("No proposals — the registry audit found nothing to "
                     "maintain.")
    lines.append("")
    return "\n".join(lines)


def write_proposals(proposals: List[SourceUpdateProposal],
                    path: str | Path) -> None:
    """Write proposals to a JSONL file. The ONLY file this layer may write.

    This export writer touches *only* the given path; it never writes the
    registry (``save_registry`` remains the sole registry writer and is not
    called here) and never writes any source or memory file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(proposals_to_jsonl(proposals), encoding="utf-8")
