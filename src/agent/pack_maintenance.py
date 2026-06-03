"""Pack source maintenance and freshness analysis (v2.2).

This module is **read-only**. It computes a freshness *status* for each source
in a pack's knowledge library from that source's existing provenance fields
(``staleness_policy``, ``retrieved_at``, ``published_at``, ``version``) plus a
per-pack :class:`FreshnessPolicy`. It adds **no** fields to the frozen
``KnowledgeSource`` model, writes nothing, and changes no grounding, citation,
verifier, lifecycle or refusal semantics.

Its job is to surface which sources have gone — or are going — stale, so an
operator can act before a stale-but-cited answer misleads someone. The system
already *labels* stale evidence at query time (``_staleness_cautions`` in the
workbench service); this layer turns that single label into a maintainable
inventory: which sources need review, why, and at what risk.

Status meaning:

* ``current``     — trusted as up to date (declared static, or young enough).
* ``review_due``  — flagged for human review (declared ``review_required``, or
  past the review threshold but not yet past the stale window).
* ``stale``       — declared stale, or older than its stale window.
* ``unknown``     — no date is recorded, so age cannot be assessed.

The lifecycle *write* actions (mark-reviewed / mark-historical / retire) and the
query-assist refusal behaviour they enable are intentionally **not** in this
module: they change citation eligibility and are a separate, ratified step.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .knowledge_library import KnowledgeLibrary
from .knowledge_sources import KnowledgeSource


class FreshnessStatus(str, Enum):
    """How much an operator should trust a source's currency."""

    CURRENT = "current"
    REVIEW_DUE = "review_due"
    STALE = "stale"
    UNKNOWN = "unknown"


# Operator-declared ``staleness_policy`` strings that short-circuit the
# time-based age check. ``STALE_POLICIES`` mirrors the workbench service's
# query-time stale label so the two views never disagree about "stale".
STALE_POLICIES = frozenset({"stale", "outdated", "deprecated"})
STATIC_POLICIES = frozenset({"static"})
REVIEW_POLICIES = frozenset({"review_required", "review-required"})

# Named review cadences -> review window in days (a per-source override that
# lets, e.g., a fast-moving Copilot note carry a shorter window than a stable
# architecture note even though both share the same domain).
CADENCE_DAYS: Dict[str, int] = {
    "volatile": 30,
    "review-monthly": 30,
    "review-quarterly": 90,
    "review-annually": 365,
}

# Most-urgent first, for display ordering.
_STATUS_ORDER = {
    FreshnessStatus.STALE: 0,
    FreshnessStatus.REVIEW_DUE: 1,
    FreshnessStatus.UNKNOWN: 2,
    FreshnessStatus.CURRENT: 3,
}

_RISK_BY_STATUS = {
    FreshnessStatus.STALE: "high",
    FreshnessStatus.REVIEW_DUE: "medium",
    FreshnessStatus.UNKNOWN: "medium",
}

_ACTION_BY_STATUS = {
    FreshnessStatus.STALE:
        "replace with a current source, or mark historical / retire it",
    FreshnessStatus.REVIEW_DUE:
        "review the source, confirm it is still current, then mark it reviewed",
    FreshnessStatus.UNKNOWN:
        "record a retrieved_at date / version so freshness can be assessed",
}

_RISK_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class FreshnessPolicy:
    """Per-pack rules for when a source becomes review-due or stale.

    Time-based windows apply only to sources whose ``staleness_policy`` is not
    one of the short-circuit declarations (static / stale / review_required) and
    not a named cadence. ``review_due_fraction`` is the fraction of the window
    at which a source flips to ``review_due`` (0.8 means "warn at 80% of the
    window"). It is clamped to ``(0, 1]``.
    """

    default_stale_after_days: int = 365
    domain_stale_after_days: Dict[str, int] = field(default_factory=dict)
    review_due_fraction: float = 0.8

    def window_days(self, domain: str, policy: str) -> int:
        policy = (policy or "").strip().lower()
        if policy in CADENCE_DAYS:
            return CADENCE_DAYS[policy]
        return self.domain_stale_after_days.get(
            domain, self.default_stale_after_days)

    @classmethod
    def from_manifest(cls, manifest: Optional[dict]) -> "FreshnessPolicy":
        """Read a ``policy.freshness`` block from a pack manifest dict.

        Missing keys fall back to the dataclass defaults, so a manifest without
        a freshness block yields the default policy.
        """
        policy = (manifest or {}).get("policy", {}) or {}
        freshness = policy.get("freshness", {}) or {}
        fraction = float(freshness.get("review_due_fraction", 0.8))
        fraction = min(1.0, max(0.01, fraction))
        return cls(
            default_stale_after_days=int(
                freshness.get("default_stale_after_days", 365)),
            domain_stale_after_days={
                str(k): int(v) for k, v in
                (freshness.get("domain_stale_after_days", {}) or {}).items()
            },
            review_due_fraction=fraction,
        )


@dataclass(frozen=True)
class SourceFreshness:
    """One source's freshness assessment (a row of the inventory)."""

    source_id: str
    source_name: str
    domain: str
    authority: str
    source_type: str
    version: Optional[str]
    staleness_policy: str
    retrieved_at: Optional[str]
    age_days: Optional[int]
    status: FreshnessStatus
    reason: str
    entries: int

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class RefreshItem:
    """One source that needs operator attention, with a suggested action."""

    source_id: str
    source_name: str
    status: FreshnessStatus
    risk: str
    reason: str
    suggested_action: str
    entries: int

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class MaintenanceReport:
    """A pack's source-health summary, with optional eval pass numbers."""

    source_count: int
    current_count: int
    review_due_count: int
    stale_count: int
    unknown_count: int
    total_entries: int
    eval_total: Optional[int] = None
    eval_passed: Optional[int] = None

    @property
    def eval_pass_rate(self) -> Optional[float]:
        if not self.eval_total:
            return None
        return self.eval_passed / self.eval_total if self.eval_passed else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["eval_pass_rate"] = self.eval_pass_rate
        return data


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _assess(source: KnowledgeSource, policy: FreshnessPolicy,
            now: datetime) -> Tuple[FreshnessStatus, str, Optional[int]]:
    """Return ``(status, reason, age_days)`` for one source."""
    declared = (source.staleness_policy or "").strip().lower()
    if declared in STALE_POLICIES:
        return (FreshnessStatus.STALE,
                f"declared '{source.staleness_policy}'", None)
    if declared in STATIC_POLICIES:
        return (FreshnessStatus.CURRENT,
                "declared static (does not decay)", None)
    if declared in REVIEW_POLICIES:
        return (FreshnessStatus.REVIEW_DUE,
                "declared review_required (not yet confirmed current)", None)

    stamp = _parse_dt(source.retrieved_at) or _parse_dt(source.published_at)
    if stamp is None:
        return (FreshnessStatus.UNKNOWN,
                "no retrieved_at / published_at date to assess age", None)

    age_days = max(0, (now - stamp).days)
    window = policy.window_days(source.domain.value, declared)
    review_at = int(window * policy.review_due_fraction)
    if age_days >= window:
        return (FreshnessStatus.STALE,
                f"age {age_days}d \u2265 stale window {window}d", age_days)
    if age_days >= review_at:
        return (FreshnessStatus.REVIEW_DUE,
                f"age {age_days}d \u2265 review threshold {review_at}d", age_days)
    return (FreshnessStatus.CURRENT,
            f"age {age_days}d < review threshold {review_at}d", age_days)


def build_inventory(library: KnowledgeLibrary,
                    policy: Optional[FreshnessPolicy] = None,
                    *, now: Optional[datetime] = None) -> List[SourceFreshness]:
    """Assess every active source in ``library``, most-urgent first."""
    policy = policy or FreshnessPolicy()
    now = now or datetime.now(timezone.utc)
    rows: List[SourceFreshness] = []
    for src in library.list_sources(active_only=True):
        status, reason, age = _assess(src, policy, now)
        entries = len(library.list_chunks(
            source_id=src.source_id, active_only=True))
        rows.append(SourceFreshness(
            source_id=src.source_id,
            source_name=src.source_name,
            domain=src.domain.value,
            authority=src.authority.value,
            source_type=src.source_type,
            version=src.version,
            staleness_policy=src.staleness_policy,
            retrieved_at=src.retrieved_at,
            age_days=age,
            status=status,
            reason=reason,
            entries=entries,
        ))
    rows.sort(key=lambda r: (_STATUS_ORDER[r.status], r.source_name.lower()))
    return rows


def build_refresh_plan(
        inventory: List[SourceFreshness]) -> List[RefreshItem]:
    """Turn an inventory into an action list for the non-current sources."""
    items: List[RefreshItem] = []
    for row in inventory:
        if row.status is FreshnessStatus.CURRENT:
            continue
        items.append(RefreshItem(
            source_id=row.source_id,
            source_name=row.source_name,
            status=row.status,
            risk=_RISK_BY_STATUS.get(row.status, "low"),
            reason=row.reason,
            suggested_action=_ACTION_BY_STATUS.get(row.status, "review"),
            entries=row.entries,
        ))
    items.sort(key=lambda i: (_RISK_ORDER.get(i.risk, 3),
                              i.source_name.lower()))
    return items


def build_maintenance_report(
        inventory: List[SourceFreshness], *,
        eval_total: Optional[int] = None,
        eval_passed: Optional[int] = None) -> MaintenanceReport:
    """Summarise source health, optionally folding in eval pass numbers."""
    counts = Counter(r.status for r in inventory)
    return MaintenanceReport(
        source_count=len(inventory),
        current_count=counts.get(FreshnessStatus.CURRENT, 0),
        review_due_count=counts.get(FreshnessStatus.REVIEW_DUE, 0),
        stale_count=counts.get(FreshnessStatus.STALE, 0),
        unknown_count=counts.get(FreshnessStatus.UNKNOWN, 0),
        total_entries=sum(r.entries for r in inventory),
        eval_total=eval_total,
        eval_passed=eval_passed,
    )
