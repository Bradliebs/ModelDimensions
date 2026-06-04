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
