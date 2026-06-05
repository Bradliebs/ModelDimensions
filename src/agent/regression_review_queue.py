"""Governed regression review queue (v7.2) — human review between monitoring and action.

Active-pack monitoring (v7.1) is read-only and advisory: it measures an active
pack set against an immutable baseline and emits a closed-set *recommendation*
(``keep_active`` … ``rollback_recommended``). A recommendation is **not**
approval, and approval is **not** execution. This module adds the missing,
explicitly governed human step between the two:

    active pack -> monitoring -> advisory recommendation -> REGRESSION REVIEW
    -> approve / reject / defer / request evidence / mark duplicate / close
    -> (approve only) action request -> separate lifecycle validation
    -> explicit lifecycle execution -> audit result

Cardinal rules enforced and tested here:

* A monitoring recommendation is not approval.
* A review approval is not lifecycle execution — approval emits a
  :class:`RegressionActionRequest` and nothing else.
* An approved review never deactivates, rolls back, supersedes, blocks or
  activates a pack. Lifecycle execution stays in the separate activation layer,
  which independently re-validates state, approval, fingerprints and policy.
* Every review binds to the *exact* monitoring evidence reviewed (run
  fingerprint, recommendation fingerprint, active-state hash, baseline hash,
  affected pack fingerprints, corpus/policy/retrieval-config identity).
* Stale monitoring evidence fails closed: a stale item can never be approved.
* Reviewer identity is never inferred; a decision without an identity or a
  reason is rejected.
* Review history is append-only; rejected and deferred decisions stay auditable.

This module is **pure and inert by construction**. It imports only the standard
library plus read-only data contracts from the monitoring and activation layers.
It imports no activation/deactivation/rollback/supersession executor, no
``ActivationStateManager``, no ``MemoryLedger`` writer, no source-registry
writer, no proposal applier, no retrieval-index or pack-content writer, and no
LLM client. Live state is loaded by the CLI and passed in read-only; the only
files this layer ever writes are its own queue, audit and action-request files.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Read-only contracts only. None of these names is a writer or an executor.
from agent.active_pack_monitor import (
    ConfidenceBand,
    MONITOR_LAYER_VERSION,
    MonitoringRecommendationCode,
)
from agent.knowledge_pack_activation import (
    ActivationStateManifest,
    ActivePackState,
    PackLifecycleState,
)

REVIEW_LAYER_VERSION = "regression-review-queue-v7.2"

REVIEW_ITEM_PREFIX = "regrev-"
REVIEW_RECORD_PREFIX = "regrev-rec-"
RECOMMENDATION_FP_PREFIX = "regrecfp-"
ACTION_REQUEST_PREFIX = "regact-"
AUDIT_PREFIX = "regaud-"

DEFAULT_QUEUE_PATH = "reviews/regression_review_queue.jsonl"
DEFAULT_AUDIT_PATH = "reviews/regression_review_audit.jsonl"
DEFAULT_ACTION_PATH = "reviews/regression_action_requests.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _age_days(created_at: str, now: Optional[str]) -> Optional[float]:
    if now is None:
        return None
    start = _parse_dt(created_at)
    end = _parse_dt(now)
    if start is None or end is None:
        return None
    return round((end - start).total_seconds() / 86400.0, 4)


# ---------------------------------------------------------------------------
# Status / decision / role / action vocabularies
# ---------------------------------------------------------------------------


class ReviewItemStatus:
    """Where a review item sits. Lifecycle-action status is kept separate."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    MORE_EVIDENCE_REQUESTED = "more_evidence_requested"
    DUPLICATE = "duplicate"
    CLOSED_NO_ACTION = "closed_no_action"
    STALE = "stale"
    SUPERSEDED = "superseded"
    ACTION_REQUESTED = "action_requested"
    ACTION_COMPLETED = "action_completed"
    ACTION_FAILED = "action_failed"


_REVIEWABLE_STATUSES = frozenset({
    ReviewItemStatus.PENDING,
    ReviewItemStatus.DEFERRED,
    ReviewItemStatus.MORE_EVIDENCE_REQUESTED,
})

_APPROVED_FAMILY = frozenset({
    ReviewItemStatus.APPROVED,
    ReviewItemStatus.ACTION_REQUESTED,
    ReviewItemStatus.ACTION_COMPLETED,
    ReviewItemStatus.ACTION_FAILED,
})


class ReviewDecision:
    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"
    REQUEST_MORE_EVIDENCE = "request_more_evidence"
    MARK_DUPLICATE = "mark_duplicate"
    CLOSE_WITHOUT_ACTION = "close_without_action"


_DECISIONS = frozenset({
    ReviewDecision.APPROVE, ReviewDecision.REJECT, ReviewDecision.DEFER,
    ReviewDecision.REQUEST_MORE_EVIDENCE, ReviewDecision.MARK_DUPLICATE,
    ReviewDecision.CLOSE_WITHOUT_ACTION,
})

# Decision -> resulting review-item status (approve is handled specially because
# it always emits an action request, so the item moves to action_requested).
_DECISION_RESULT_STATUS: Dict[str, str] = {
    ReviewDecision.APPROVE: ReviewItemStatus.ACTION_REQUESTED,
    ReviewDecision.REJECT: ReviewItemStatus.REJECTED,
    ReviewDecision.DEFER: ReviewItemStatus.DEFERRED,
    ReviewDecision.REQUEST_MORE_EVIDENCE: ReviewItemStatus.MORE_EVIDENCE_REQUESTED,
    ReviewDecision.MARK_DUPLICATE: ReviewItemStatus.DUPLICATE,
    ReviewDecision.CLOSE_WITHOUT_ACTION: ReviewItemStatus.CLOSED_NO_ACTION,
}


class ReviewerRole:
    MONITORING_REVIEWER = "monitoring_reviewer"
    PACK_OWNER = "pack_owner"
    GOVERNANCE_APPROVER = "governance_approver"
    LIFECYCLE_OPERATOR = "lifecycle_operator"


class ActionRequestType:
    REQUEST_INVESTIGATION = "request_investigation"
    REQUEST_WATCH = "request_watch"
    REQUEST_DEACTIVATION = "request_deactivation"
    REQUEST_ROLLBACK = "request_rollback"
    REQUEST_ACTIVATION_BLOCK = "request_activation_block"
    REQUEST_NEW_MONITORING_RUN = "request_new_monitoring_run"
    REQUEST_ADDITIONAL_EVIDENCE = "request_additional_evidence"


class ActionRequestStatus:
    REQUESTED = "requested"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"


_ACTION_RESULT_STATUSES = frozenset({
    ActionRequestStatus.VALIDATED, ActionRequestStatus.REJECTED,
    ActionRequestStatus.EXECUTED, ActionRequestStatus.FAILED,
    ActionRequestStatus.CANCELLED, ActionRequestStatus.STALE,
})


class StalenessCode:
    ACTIVE_STATE_CHANGED = "active_state_changed"
    PACK_FINGERPRINT_CHANGED = "pack_fingerprint_changed"
    MONITORING_RUN_INVALIDATED = "monitoring_run_invalidated"
    RECOMMENDATION_FINGERPRINT_CHANGED = "recommendation_fingerprint_changed"
    BASELINE_REPLACED = "baseline_replaced"
    POLICY_CHANGED = "policy_changed"
    ROLLBACK_TARGET_INVALID = "rollback_target_invalid"
    RECOMMENDATION_SUPERSEDED = "recommendation_superseded"
    PACK_NOT_ACTIVE = "pack_not_active"
    REGRESSION_NOT_PRESENT = "regression_not_present"


# Recommendation -> the single action type an approval may request.
_RECOMMENDATION_TO_ACTION: Dict[str, str] = {
    MonitoringRecommendationCode.DEACTIVATE_RECOMMENDED.value:
        ActionRequestType.REQUEST_DEACTIVATION,
    MonitoringRecommendationCode.ROLLBACK_RECOMMENDED.value:
        ActionRequestType.REQUEST_ROLLBACK,
    MonitoringRecommendationCode.BLOCK_FUTURE_ACTIVATION.value:
        ActionRequestType.REQUEST_ACTIVATION_BLOCK,
    MonitoringRecommendationCode.INVESTIGATE.value:
        ActionRequestType.REQUEST_INVESTIGATION,
    MonitoringRecommendationCode.KEEP_ACTIVE_WITH_WATCH.value:
        ActionRequestType.REQUEST_WATCH,
    MonitoringRecommendationCode.INSUFFICIENT_EVIDENCE_TO_RECOMMEND.value:
        ActionRequestType.REQUEST_ADDITIONAL_EVIDENCE,
    MonitoringRecommendationCode.KEEP_ACTIVE.value:
        ActionRequestType.REQUEST_WATCH,
}

# The execution-approval the *separate* lifecycle layer would require. ``none``
# means the requested action carries no lifecycle execution (investigation,
# watch, evidence) and is satisfied without touching pack state.
_ACTION_EXECUTION_APPROVAL: Dict[str, str] = {
    ActionRequestType.REQUEST_DEACTIVATION: "deactivation_approval",
    ActionRequestType.REQUEST_ROLLBACK: "rollback_approval",
    ActionRequestType.REQUEST_ACTIVATION_BLOCK: "activation_block_approval",
    ActionRequestType.REQUEST_INVESTIGATION: "none",
    ActionRequestType.REQUEST_WATCH: "none",
    ActionRequestType.REQUEST_ADDITIONAL_EVIDENCE: "none",
    ActionRequestType.REQUEST_NEW_MONITORING_RUN: "none",
}

# Recommendations imported into the review queue by default. ``keep_active`` is
# intentionally excluded — it stays in monitoring history and only enters the
# queue when explicitly requested.
_DEFAULT_IMPORTABLE = frozenset({
    MonitoringRecommendationCode.INVESTIGATE.value,
    MonitoringRecommendationCode.DEACTIVATE_RECOMMENDED.value,
    MonitoringRecommendationCode.ROLLBACK_RECOMMENDED.value,
    MonitoringRecommendationCode.BLOCK_FUTURE_ACTIVATION.value,
    MonitoringRecommendationCode.INSUFFICIENT_EVIDENCE_TO_RECOMMEND.value,
})

_SEVERITY_ORDER = ("info", "low", "medium", "high", "critical")


def _max_severity(severities: Sequence[str]) -> str:
    rank = -1
    out = "info"
    for sev in severities:
        try:
            r = _SEVERITY_ORDER.index(sev)
        except ValueError:
            r = 0
        if r > rank:
            rank = r
            out = sev
    return out


# ---------------------------------------------------------------------------
# Role policy (deterministic, declared-role only; no identity infrastructure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewRolePolicy:
    """Which declared roles may take which decisions. Pure, deterministic."""

    allowed: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def roles_for(self, decision: str) -> Tuple[str, ...]:
        return self.allowed.get(decision, ())

    def permits(self, decision: str, role: str) -> bool:
        return role in self.roles_for(decision)


_NON_APPROVE_ROLES = (
    ReviewerRole.MONITORING_REVIEWER,
    ReviewerRole.PACK_OWNER,
    ReviewerRole.GOVERNANCE_APPROVER,
)

DEFAULT_ROLE_POLICY = ReviewRolePolicy(allowed={
    ReviewDecision.APPROVE: (ReviewerRole.GOVERNANCE_APPROVER,),
    ReviewDecision.REJECT: _NON_APPROVE_ROLES,
    ReviewDecision.DEFER: _NON_APPROVE_ROLES,
    ReviewDecision.REQUEST_MORE_EVIDENCE: _NON_APPROVE_ROLES,
    ReviewDecision.MARK_DUPLICATE: _NON_APPROVE_ROLES,
    ReviewDecision.CLOSE_WITHOUT_ACTION: _NON_APPROVE_ROLES,
})


# ---------------------------------------------------------------------------
# Review item
# ---------------------------------------------------------------------------


def compute_recommendation_fingerprint(*, recommendation: str,
                                       rationale_codes: Sequence[str],
                                       affected_case_ids: Sequence[str],
                                       affected_pack_ids: Sequence[str],
                                       rollback_target: str,
                                       confidence: str,
                                       monitoring_run_fingerprint: str) -> str:
    """Deterministic fingerprint binding a recommendation to its exact evidence."""
    payload = {
        "recommendation": recommendation,
        "rationale_codes": sorted(rationale_codes),
        "affected_case_ids": sorted(affected_case_ids),
        "affected_pack_ids": sorted(affected_pack_ids),
        "rollback_target": rollback_target,
        "confidence": confidence,
        "monitoring_run_fingerprint": monitoring_run_fingerprint,
    }
    return RECOMMENDATION_FP_PREFIX + _sha256_hex(_canonical(payload))[:16]


@dataclass(frozen=True)
class RegressionReviewItem:
    """One review-worthy monitoring recommendation, bound to its exact evidence."""

    review_item_id: str
    monitoring_run_id: str
    monitoring_run_fingerprint: str
    recommendation: str
    recommendation_fingerprint: str
    recommendation_confidence: str
    active_state_hash: str
    baseline_hash: str
    corpus_fingerprints: Tuple[str, ...]
    policy_id: str
    policy_fingerprint: str
    retrieval_config_fingerprint: str
    affected_pack_ids: Tuple[str, ...]
    affected_pack_fingerprints: Tuple[str, ...]
    affected_case_ids: Tuple[str, ...]
    regression_findings: Tuple[dict, ...]
    severity: str
    proposed_action_type: str
    rollback_target_state_hash: str = ""
    created_at: str = ""
    status: str = ReviewItemStatus.PENDING

    @property
    def is_critical(self) -> bool:
        return self.severity == "critical"

    @property
    def is_pending(self) -> bool:
        return self.status == ReviewItemStatus.PENDING

    def to_dict(self) -> dict:
        return {
            "_record": "regression_review_item",
            "review_item_id": self.review_item_id,
            "monitoring_run_id": self.monitoring_run_id,
            "monitoring_run_fingerprint": self.monitoring_run_fingerprint,
            "recommendation": self.recommendation,
            "recommendation_fingerprint": self.recommendation_fingerprint,
            "recommendation_confidence": self.recommendation_confidence,
            "active_state_hash": self.active_state_hash,
            "baseline_hash": self.baseline_hash,
            "corpus_fingerprints": list(self.corpus_fingerprints),
            "policy_id": self.policy_id,
            "policy_fingerprint": self.policy_fingerprint,
            "retrieval_config_fingerprint": self.retrieval_config_fingerprint,
            "affected_pack_ids": list(self.affected_pack_ids),
            "affected_pack_fingerprints": list(self.affected_pack_fingerprints),
            "affected_case_ids": list(self.affected_case_ids),
            "regression_findings": [dict(f) for f in self.regression_findings],
            "severity": self.severity,
            "proposed_action_type": self.proposed_action_type,
            "rollback_target_state_hash": self.rollback_target_state_hash,
            "created_at": self.created_at,
            "status": self.status,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RegressionReviewItem":
        return cls(
            review_item_id=str(data["review_item_id"]),
            monitoring_run_id=str(data.get("monitoring_run_id", "")),
            monitoring_run_fingerprint=str(data.get("monitoring_run_fingerprint", "")),
            recommendation=str(data.get("recommendation", "")),
            recommendation_fingerprint=str(data.get("recommendation_fingerprint", "")),
            recommendation_confidence=str(data.get("recommendation_confidence", "")),
            active_state_hash=str(data.get("active_state_hash", "")),
            baseline_hash=str(data.get("baseline_hash", "")),
            corpus_fingerprints=tuple(str(v) for v in
                                      (data.get("corpus_fingerprints") or [])),
            policy_id=str(data.get("policy_id", "")),
            policy_fingerprint=str(data.get("policy_fingerprint", "")),
            retrieval_config_fingerprint=str(
                data.get("retrieval_config_fingerprint", "")),
            affected_pack_ids=tuple(str(v) for v in
                                    (data.get("affected_pack_ids") or [])),
            affected_pack_fingerprints=tuple(str(v) for v in
                                             (data.get("affected_pack_fingerprints") or [])),
            affected_case_ids=tuple(str(v) for v in
                                    (data.get("affected_case_ids") or [])),
            regression_findings=tuple(dict(f) for f in
                                      (data.get("regression_findings") or [])),
            severity=str(data.get("severity", "info")),
            proposed_action_type=str(data.get("proposed_action_type", "")),
            rollback_target_state_hash=str(data.get("rollback_target_state_hash", "")),
            created_at=str(data.get("created_at", "")),
            status=str(data.get("status", ReviewItemStatus.PENDING)),
        )


def _compute_review_item_id(*, monitoring_run_fingerprint: str,
                            recommendation_fingerprint: str,
                            affected_pack_ids: Sequence[str]) -> str:
    payload = {
        "monitoring_run_fingerprint": monitoring_run_fingerprint,
        "recommendation_fingerprint": recommendation_fingerprint,
        "affected_pack_ids": sorted(affected_pack_ids),
    }
    return REVIEW_ITEM_PREFIX + _sha256_hex(_canonical(payload))[:16]


# ---------------------------------------------------------------------------
# Phase C — import monitoring recommendations
# ---------------------------------------------------------------------------


def is_importable_recommendation(recommendation: str, *,
                                 include_keep_active: bool = False,
                                 include_watch: bool = True) -> bool:
    """Whether a recommendation creates a review item under default policy."""
    if recommendation == MonitoringRecommendationCode.KEEP_ACTIVE.value:
        return include_keep_active
    if recommendation == MonitoringRecommendationCode.KEEP_ACTIVE_WITH_WATCH.value:
        return include_watch
    return recommendation in _DEFAULT_IMPORTABLE


def _bounded_findings(run_dict: dict) -> Tuple[dict, ...]:
    """Copy only bounded finding metadata — never retrieved text."""
    out: List[dict] = []
    for f in run_dict.get("findings") or []:
        out.append({
            "finding_code": str(f.get("finding_code", "")),
            "severity": str(f.get("severity", "info")),
            "case_ids": [str(c) for c in (f.get("case_ids") or [])],
            "confidence": str(f.get("confidence", ConfidenceBand.MEDIUM.value)),
            "candidate_pack_ids": [str(p) for p in
                                   (f.get("candidate_pack_ids") or [])],
        })
    return tuple(out)


def import_monitoring_recommendation(monitoring_report: dict,
                                     recommendation: Optional[str] = None, *,
                                     created_at: Optional[str] = None,
                                     include_keep_active: bool = False,
                                     include_watch: bool = True,
                                     ) -> Optional[RegressionReviewItem]:
    """Build a :class:`RegressionReviewItem` from a monitoring run dict (pure).

    Returns ``None`` when the recommendation is not review-worthy under default
    policy (e.g. ``keep_active``). Passing ``recommendation`` explicitly forces
    import of that recommendation, even ``keep_active``. The item preserves all
    monitoring evidence references and copies only bounded finding metadata — no
    retrieved text is ever persisted.
    """
    if str(monitoring_report.get("_record", "")) != "active_pack_monitoring_run":
        raise ValueError(
            "monitoring_report is not an active_pack_monitoring_run record")

    rec_block = dict(monitoring_report.get("recommendation") or {})
    explicit = recommendation is not None
    rec = recommendation if explicit else str(rec_block.get("recommendation", ""))
    if not rec:
        raise ValueError("monitoring report has no recommendation")

    if not explicit and not is_importable_recommendation(
            rec, include_keep_active=include_keep_active,
            include_watch=include_watch):
        return None

    snapshot = dict(monitoring_report.get("snapshot") or {})
    snap_pack_ids = [str(p) for p in (snapshot.get("active_pack_ids") or [])]
    snap_pack_fps = [str(p) for p in (snapshot.get("active_pack_fingerprints") or [])]
    fp_by_id = dict(zip(snap_pack_ids, snap_pack_fps))

    candidate_ids = [str(p) for p in (rec_block.get("candidate_pack_ids") or [])]
    affected_pack_ids = tuple(candidate_ids or snap_pack_ids)
    affected_pack_fps = tuple(fp_by_id.get(pid, "") for pid in affected_pack_ids)

    affected_cases = tuple(str(c) for c in (rec_block.get("affected_cases") or []))
    rationale = [str(c) for c in (rec_block.get("rationale_codes") or [])]
    rollback_target = str(rec_block.get("rollback_target", ""))
    confidence = str(rec_block.get("confidence", ConfidenceBand.MEDIUM.value))
    run_fp = str(monitoring_report.get("record_hash", ""))

    rec_fp = compute_recommendation_fingerprint(
        recommendation=rec, rationale_codes=rationale,
        affected_case_ids=affected_cases, affected_pack_ids=affected_pack_ids,
        rollback_target=rollback_target, confidence=confidence,
        monitoring_run_fingerprint=run_fp)

    item_id = _compute_review_item_id(
        monitoring_run_fingerprint=run_fp, recommendation_fingerprint=rec_fp,
        affected_pack_ids=affected_pack_ids)

    findings = _bounded_findings(monitoring_report)
    severity = _max_severity([str(f["severity"]) for f in findings] or ["info"])

    corpus_fp = str(monitoring_report.get("corpus_fingerprint", ""))

    return RegressionReviewItem(
        review_item_id=item_id,
        monitoring_run_id=str(monitoring_report.get("monitoring_run_id", "")),
        monitoring_run_fingerprint=run_fp,
        recommendation=rec,
        recommendation_fingerprint=rec_fp,
        recommendation_confidence=confidence,
        active_state_hash=str(snapshot.get("active_state_hash", "")),
        baseline_hash=str(monitoring_report.get("baseline_hash", "")),
        corpus_fingerprints=(corpus_fp,) if corpus_fp else (),
        policy_id=str(monitoring_report.get("policy_id", "")),
        policy_fingerprint=str(monitoring_report.get("policy_fingerprint", "")),
        retrieval_config_fingerprint=str(
            monitoring_report.get("retrieval_config_fingerprint", "")),
        affected_pack_ids=affected_pack_ids,
        affected_pack_fingerprints=affected_pack_fps,
        affected_case_ids=affected_cases,
        regression_findings=findings,
        severity=severity,
        proposed_action_type=_RECOMMENDATION_TO_ACTION.get(
            rec, ActionRequestType.REQUEST_INVESTIGATION),
        rollback_target_state_hash=rollback_target,
        created_at=created_at if created_at is not None else _utc_now_iso(),
        status=ReviewItemStatus.PENDING,
    )


def _sorted_items(items) -> List[RegressionReviewItem]:
    return sorted(items, key=lambda r: r.review_item_id)


def add_review_item(items: Sequence[RegressionReviewItem],
                    new_item: RegressionReviewItem,
                    ) -> List[RegressionReviewItem]:
    """Merge a new item into a queue, idempotent by ``review_item_id`` (pure).

    An identical re-import is a no-op: the existing item (and its review state)
    is preserved. Deterministic order by id, so the queue saves byte-identically.
    """
    by_id = {r.review_item_id: r for r in items}
    if new_item.review_item_id not in by_id:
        by_id[new_item.review_item_id] = new_item
    return _sorted_items(by_id.values())


# ---------------------------------------------------------------------------
# Phase D — staleness validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionReviewValidation:
    review_item_id: str
    ok: bool
    stale: bool
    staleness_codes: Tuple[str, ...]
    reasons: Tuple[str, ...]
    checked_at: str = ""

    def to_dict(self) -> dict:
        return {
            "_record": "regression_review_validation",
            "review_item_id": self.review_item_id,
            "ok": self.ok,
            "stale": self.stale,
            "staleness_codes": list(self.staleness_codes),
            "reasons": list(self.reasons),
            "checked_at": self.checked_at,
        }


def validate_review_item(item: RegressionReviewItem, *,
                         current_state: Optional[ActivationStateManifest] = None,
                         current_run: Optional[dict] = None,
                         baseline_exists: bool = True,
                         rollback_target_exists: Optional[bool] = None,
                         superseded: bool = False,
                         now: Optional[str] = None) -> RegressionReviewValidation:
    """Fail-closed staleness check binding a review item to current evidence.

    A review item is **stale** (and therefore not approvable) when the active
    state, affected pack fingerprints, monitoring run, baseline, policy or
    rollback target it was created against no longer hold, or when the relevant
    pack is no longer active, or when a newer run has superseded it. Pure: reads
    the passed-in read-only state and never mutates anything.
    """
    codes: List[str] = []
    reasons: List[str] = []

    def _flag(code: str, reason: str) -> None:
        if code not in codes:
            codes.append(code)
            reasons.append(reason)

    if item.status in (ReviewItemStatus.STALE, ReviewItemStatus.SUPERSEDED):
        _flag(StalenessCode.RECOMMENDATION_SUPERSEDED,
              f"item is already {item.status}")

    if superseded:
        _flag(StalenessCode.RECOMMENDATION_SUPERSEDED,
              "a newer monitoring run supersedes this recommendation")

    if not baseline_exists:
        _flag(StalenessCode.BASELINE_REPLACED,
              "the baseline this run compared against no longer exists")

    if (item.rollback_target_state_hash and rollback_target_exists is False):
        _flag(StalenessCode.ROLLBACK_TARGET_INVALID,
              "the rollback target state no longer exists")

    if current_state is not None:
        if current_state.state_hash != item.active_state_hash:
            _flag(StalenessCode.ACTIVE_STATE_CHANGED,
                  "current active-state hash differs from the reviewed state")
        by_id = {r.pack_id: r for r in current_state.records}
        for pid, fp in zip(item.affected_pack_ids, item.affected_pack_fingerprints):
            rec = by_id.get(pid)
            if rec is None:
                _flag(StalenessCode.PACK_NOT_ACTIVE,
                      f"affected pack {pid!r} is no longer in the active state")
                continue
            if fp and rec.pack_fingerprint != fp:
                _flag(StalenessCode.PACK_FINGERPRINT_CHANGED,
                      f"affected pack {pid!r} fingerprint changed since review")
            if rec.status != PackLifecycleState.ACTIVE:
                _flag(StalenessCode.PACK_NOT_ACTIVE,
                      f"affected pack {pid!r} is {rec.status.value}, not active")

    if current_run is not None:
        run_fp = str(current_run.get("record_hash", ""))
        if run_fp and run_fp != item.monitoring_run_fingerprint:
            _flag(StalenessCode.MONITORING_RUN_INVALIDATED,
                  "the current monitoring run fingerprint differs")
        rec_block = dict(current_run.get("recommendation") or {})
        cur_rec = str(rec_block.get("recommendation", ""))
        if cur_rec and cur_rec != item.recommendation:
            _flag(StalenessCode.RECOMMENDATION_SUPERSEDED,
                  "the current run recommends a different action")

    stale = bool(codes)
    return RegressionReviewValidation(
        review_item_id=item.review_item_id, ok=not stale, stale=stale,
        staleness_codes=tuple(codes), reasons=tuple(reasons),
        checked_at=now if now is not None else _utc_now_iso())


# ---------------------------------------------------------------------------
# Phase E — review decision records + Phase G — action requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionActionRequest:
    """A request for a *separate* lifecycle action. Never an execution itself."""

    action_request_id: str
    review_item_id: str
    review_record_id: str
    requested_action: str
    requested_by: str
    requested_at: str
    active_state_hash: str
    affected_pack_ids: Tuple[str, ...]
    affected_pack_fingerprints: Tuple[str, ...]
    monitoring_run_fingerprint: str
    recommendation_fingerprint: str
    approval_reason: str
    required_execution_approval_type: str
    rollback_target_state_hash: str = ""
    status: str = ActionRequestStatus.REQUESTED
    executed_at: str = ""
    execution_record_id: str = ""
    lifecycle_audit_ref: str = ""
    failure_reason: str = ""

    @property
    def executed(self) -> bool:
        return self.status == ActionRequestStatus.EXECUTED

    def to_dict(self) -> dict:
        return {
            "_record": "regression_action_request",
            "action_request_id": self.action_request_id,
            "review_item_id": self.review_item_id,
            "review_record_id": self.review_record_id,
            "requested_action": self.requested_action,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
            "active_state_hash": self.active_state_hash,
            "affected_pack_ids": list(self.affected_pack_ids),
            "affected_pack_fingerprints": list(self.affected_pack_fingerprints),
            "monitoring_run_fingerprint": self.monitoring_run_fingerprint,
            "recommendation_fingerprint": self.recommendation_fingerprint,
            "approval_reason": self.approval_reason,
            "required_execution_approval_type": self.required_execution_approval_type,
            "rollback_target_state_hash": self.rollback_target_state_hash,
            "status": self.status,
            "executed_at": self.executed_at,
            "execution_record_id": self.execution_record_id,
            "lifecycle_audit_ref": self.lifecycle_audit_ref,
            "failure_reason": self.failure_reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RegressionActionRequest":
        return cls(
            action_request_id=str(data["action_request_id"]),
            review_item_id=str(data.get("review_item_id", "")),
            review_record_id=str(data.get("review_record_id", "")),
            requested_action=str(data.get("requested_action", "")),
            requested_by=str(data.get("requested_by", "")),
            requested_at=str(data.get("requested_at", "")),
            active_state_hash=str(data.get("active_state_hash", "")),
            affected_pack_ids=tuple(str(v) for v in
                                    (data.get("affected_pack_ids") or [])),
            affected_pack_fingerprints=tuple(str(v) for v in
                                             (data.get("affected_pack_fingerprints") or [])),
            monitoring_run_fingerprint=str(data.get("monitoring_run_fingerprint", "")),
            recommendation_fingerprint=str(data.get("recommendation_fingerprint", "")),
            approval_reason=str(data.get("approval_reason", "")),
            required_execution_approval_type=str(
                data.get("required_execution_approval_type", "none")),
            rollback_target_state_hash=str(data.get("rollback_target_state_hash", "")),
            status=str(data.get("status", ActionRequestStatus.REQUESTED)),
            executed_at=str(data.get("executed_at", "")),
            execution_record_id=str(data.get("execution_record_id", "")),
            lifecycle_audit_ref=str(data.get("lifecycle_audit_ref", "")),
            failure_reason=str(data.get("failure_reason", "")),
        )


@dataclass(frozen=True)
class RegressionReviewRecord:
    """An immutable record of one review decision. Append-only by construction."""

    review_record_id: str
    review_item_id: str
    decision: str
    reviewer: str
    role: str
    reason: str
    evidence_acknowledged: bool
    reviewed_at: str
    monitoring_run_fingerprint: str
    recommendation_fingerprint: str
    active_state_hash_at_review: str
    affected_pack_fingerprints_at_review: Tuple[str, ...]
    resulting_status: str
    defer_until: str = ""
    requested_evidence: str = ""
    duplicate_of_review_item_id: str = ""
    action_request_id: str = ""

    def to_dict(self) -> dict:
        return {
            "_record": "regression_review_record",
            "review_record_id": self.review_record_id,
            "review_item_id": self.review_item_id,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "role": self.role,
            "reason": self.reason,
            "evidence_acknowledged": self.evidence_acknowledged,
            "reviewed_at": self.reviewed_at,
            "monitoring_run_fingerprint": self.monitoring_run_fingerprint,
            "recommendation_fingerprint": self.recommendation_fingerprint,
            "active_state_hash_at_review": self.active_state_hash_at_review,
            "affected_pack_fingerprints_at_review":
                list(self.affected_pack_fingerprints_at_review),
            "resulting_status": self.resulting_status,
            "defer_until": self.defer_until,
            "requested_evidence": self.requested_evidence,
            "duplicate_of_review_item_id": self.duplicate_of_review_item_id,
            "action_request_id": self.action_request_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RegressionReviewRecord":
        return cls(
            review_record_id=str(data["review_record_id"]),
            review_item_id=str(data.get("review_item_id", "")),
            decision=str(data.get("decision", "")),
            reviewer=str(data.get("reviewer", "")),
            role=str(data.get("role", "")),
            reason=str(data.get("reason", "")),
            evidence_acknowledged=bool(data.get("evidence_acknowledged", False)),
            reviewed_at=str(data.get("reviewed_at", "")),
            monitoring_run_fingerprint=str(data.get("monitoring_run_fingerprint", "")),
            recommendation_fingerprint=str(data.get("recommendation_fingerprint", "")),
            active_state_hash_at_review=str(data.get("active_state_hash_at_review", "")),
            affected_pack_fingerprints_at_review=tuple(
                str(v) for v in (data.get("affected_pack_fingerprints_at_review") or [])),
            resulting_status=str(data.get("resulting_status", "")),
            defer_until=str(data.get("defer_until", "")),
            requested_evidence=str(data.get("requested_evidence", "")),
            duplicate_of_review_item_id=str(data.get("duplicate_of_review_item_id", "")),
            action_request_id=str(data.get("action_request_id", "")),
        )


def _compute_review_record_id(*, review_item_id: str, decision: str,
                              reviewed_at: str, reviewer: str) -> str:
    payload = {"review_item_id": review_item_id, "decision": decision,
               "reviewed_at": reviewed_at, "reviewer": reviewer}
    return REVIEW_RECORD_PREFIX + _sha256_hex(_canonical(payload))[:16]


def create_action_request(item: RegressionReviewItem,
                          record: RegressionReviewRecord, *,
                          requested_at: Optional[str] = None,
                          ) -> RegressionActionRequest:
    """Build the action request that an approval authorises (pure; inert).

    The action request is a *request*, not an execution. Its id is deterministic,
    so re-creating it from the same review record is idempotent.
    """
    action_type = item.proposed_action_type
    now = requested_at if requested_at is not None else _utc_now_iso()
    payload = {
        "review_record_id": record.review_record_id,
        "requested_action": action_type,
        "active_state_hash": item.active_state_hash,
    }
    action_id = ACTION_REQUEST_PREFIX + _sha256_hex(_canonical(payload))[:16]
    return RegressionActionRequest(
        action_request_id=action_id,
        review_item_id=item.review_item_id,
        review_record_id=record.review_record_id,
        requested_action=action_type,
        requested_by=record.reviewer,
        requested_at=now,
        active_state_hash=item.active_state_hash,
        affected_pack_ids=item.affected_pack_ids,
        affected_pack_fingerprints=item.affected_pack_fingerprints,
        monitoring_run_fingerprint=item.monitoring_run_fingerprint,
        recommendation_fingerprint=item.recommendation_fingerprint,
        approval_reason=record.reason,
        required_execution_approval_type=_ACTION_EXECUTION_APPROVAL.get(
            action_type, "none"),
        rollback_target_state_hash=item.rollback_target_state_hash,
        status=ActionRequestStatus.REQUESTED,
    )


@dataclass(frozen=True)
class ReviewOutcome:
    """The result of one review decision: updated item, record, optional request."""

    item: RegressionReviewItem
    record: RegressionReviewRecord
    action_request: Optional[RegressionActionRequest] = None


def review_item(item: RegressionReviewItem, decision: str, *,
                reviewer: str, role: str, reason: str,
                evidence_acknowledged: bool = False,
                validation: Optional[RegressionReviewValidation] = None,
                active_state_hash_at_review: Optional[str] = None,
                affected_pack_fingerprints_at_review: Optional[Sequence[str]] = None,
                defer_until: str = "", requested_evidence: str = "",
                duplicate_of_review_item_id: str = "",
                role_policy: ReviewRolePolicy = DEFAULT_ROLE_POLICY,
                reviewed_at: Optional[str] = None) -> ReviewOutcome:
    """Apply one review decision (pure). Returns a new item, record and request.

    Fail-closed: a missing reviewer identity or reason is rejected; a stale item
    cannot be approved; approval is only permitted when the reviewed active state
    and pack fingerprints still match; an approval emits a
    :class:`RegressionActionRequest` and never executes a lifecycle action.
    """
    if decision not in _DECISIONS:
        raise ValueError(f"unknown review decision {decision!r}")
    if not reviewer.strip():
        raise ValueError("reviewer identity is required (it is never inferred)")
    if not reason.strip():
        raise ValueError("a review reason is required")
    if not role_policy.permits(decision, role):
        raise ValueError(
            f"role {role!r} may not take decision {decision!r}")
    if item.status not in _REVIEWABLE_STATUSES:
        raise ValueError(
            f"review item {item.review_item_id} is {item.status!r}, "
            "not in a reviewable state")

    now = reviewed_at if reviewed_at is not None else _utc_now_iso()
    state_at = (active_state_hash_at_review if active_state_hash_at_review is not None
                else item.active_state_hash)
    fps_at = (tuple(affected_pack_fingerprints_at_review)
              if affected_pack_fingerprints_at_review is not None
              else item.affected_pack_fingerprints)

    action_request: Optional[RegressionActionRequest] = None

    if decision == ReviewDecision.APPROVE:
        if item.status == ReviewItemStatus.MORE_EVIDENCE_REQUESTED:
            raise ValueError(
                "an unresolved evidence request blocks approval; "
                "run a fresh monitoring run and re-import")
        if validation is None or not validation.ok:
            raise ValueError(
                "approval requires a non-stale validation of the review item")
        if state_at != item.active_state_hash:
            raise ValueError(
                "active state changed since review; re-validate before approval")
        if tuple(fps_at) != item.affected_pack_fingerprints:
            raise ValueError(
                "affected pack fingerprints changed since review; re-validate")
        if item.recommendation == MonitoringRecommendationCode.KEEP_ACTIVE.value:
            raise ValueError(
                "keep_active is not an actionable recommendation to approve")

    resulting_status = _DECISION_RESULT_STATUS[decision]
    record_id = _compute_review_record_id(
        review_item_id=item.review_item_id, decision=decision,
        reviewed_at=now, reviewer=reviewer)

    record = RegressionReviewRecord(
        review_record_id=record_id, review_item_id=item.review_item_id,
        decision=decision, reviewer=reviewer, role=role, reason=reason,
        evidence_acknowledged=evidence_acknowledged, reviewed_at=now,
        monitoring_run_fingerprint=item.monitoring_run_fingerprint,
        recommendation_fingerprint=item.recommendation_fingerprint,
        active_state_hash_at_review=state_at,
        affected_pack_fingerprints_at_review=tuple(fps_at),
        resulting_status=resulting_status, defer_until=defer_until,
        requested_evidence=requested_evidence,
        duplicate_of_review_item_id=duplicate_of_review_item_id)

    if decision == ReviewDecision.APPROVE:
        action_request = create_action_request(item, record, requested_at=now)
        record = replace(record, action_request_id=action_request.action_request_id)

    new_item = replace(item, status=resulting_status)
    return ReviewOutcome(item=new_item, record=record, action_request=action_request)


def supersede_review_item(item: RegressionReviewItem, *,
                          superseding_run_fingerprint: str,
                          reason: str = "", at: Optional[str] = None,
                          ) -> RegressionReviewItem:
    """Mark a pending item superseded by a newer monitoring run (pure; traceable)."""
    if item.status not in _REVIEWABLE_STATUSES:
        raise ValueError(
            f"only a reviewable item can be superseded; {item.review_item_id} "
            f"is {item.status!r}")
    return replace(item, status=ReviewItemStatus.SUPERSEDED)


def mark_action_result(action: RegressionActionRequest, status: str, *,
                       executed_at: Optional[str] = None,
                       execution_record_id: str = "",
                       lifecycle_audit_ref: str = "",
                       failure_reason: str = "") -> RegressionActionRequest:
    """Record an *externally* completed lifecycle action (pure; never executes).

    This only annotates the action request with an outcome that the separate
    lifecycle layer produced. Marking ``executed`` requires a real lifecycle
    execution record and audit reference, so an action can never be marked done
    without evidence. A stale action cannot be executed.
    """
    if status not in _ACTION_RESULT_STATUSES:
        raise ValueError(f"unknown action result status {status!r}")
    if action.status in (ActionRequestStatus.EXECUTED, ActionRequestStatus.CANCELLED):
        raise ValueError(
            f"action {action.action_request_id} is already {action.status!r}")
    if status == ActionRequestStatus.EXECUTED:
        if action.status == ActionRequestStatus.STALE:
            raise ValueError("a stale action request cannot be executed")
        if not execution_record_id.strip() or not lifecycle_audit_ref.strip():
            raise ValueError(
                "marking an action executed requires a lifecycle execution "
                "record id and an external lifecycle audit reference")
    return replace(
        action, status=status,
        executed_at=(executed_at if executed_at is not None else
                     (_utc_now_iso() if status == ActionRequestStatus.EXECUTED
                      else action.executed_at)),
        execution_record_id=execution_record_id or action.execution_record_id,
        lifecycle_audit_ref=lifecycle_audit_ref or action.lifecycle_audit_ref,
        failure_reason=failure_reason or action.failure_reason)


# ---------------------------------------------------------------------------
# Audit records (append-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionReviewAuditRecord:
    audit_id: str
    event: str
    review_item_id: str
    actor: str
    at: str
    review_record_id: str = ""
    action_request_id: str = ""
    status: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "_record": "regression_review_audit",
            "audit_id": self.audit_id,
            "event": self.event,
            "review_item_id": self.review_item_id,
            "review_record_id": self.review_record_id,
            "action_request_id": self.action_request_id,
            "status": self.status,
            "actor": self.actor,
            "at": self.at,
            "detail": self.detail,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RegressionReviewAuditRecord":
        return cls(
            audit_id=str(data["audit_id"]),
            event=str(data.get("event", "")),
            review_item_id=str(data.get("review_item_id", "")),
            actor=str(data.get("actor", "")),
            at=str(data.get("at", "")),
            review_record_id=str(data.get("review_record_id", "")),
            action_request_id=str(data.get("action_request_id", "")),
            status=str(data.get("status", "")),
            detail=str(data.get("detail", "")),
        )


def make_audit_record(*, event: str, review_item_id: str, actor: str, at: str,
                      review_record_id: str = "", action_request_id: str = "",
                      status: str = "", detail: str = "",
                      ) -> RegressionReviewAuditRecord:
    payload = {"event": event, "review_item_id": review_item_id,
               "review_record_id": review_record_id,
               "action_request_id": action_request_id, "at": at}
    audit_id = AUDIT_PREFIX + _sha256_hex(_canonical(payload))[:16]
    return RegressionReviewAuditRecord(
        audit_id=audit_id, event=event, review_item_id=review_item_id,
        actor=actor, at=at, review_record_id=review_record_id,
        action_request_id=action_request_id, status=status, detail=detail)


# ---------------------------------------------------------------------------
# Phase I — persistence (atomic; queue / audit / actions kept distinct)
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_jsonl(path: str | Path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(json.loads(line))
    return out


def load_review_queue(path: str | Path = DEFAULT_QUEUE_PATH,
                      ) -> List[RegressionReviewItem]:
    return [RegressionReviewItem.from_dict(d) for d in _read_jsonl(path)]


def save_review_queue(items: Sequence[RegressionReviewItem],
                      path: str | Path = DEFAULT_QUEUE_PATH) -> None:
    """Write the review queue (sorted by id; atomic). The only queue writer."""
    text = "".join(item.to_json() + "\n" for item in _sorted_items(items))
    _atomic_write_text(Path(path), text)


def load_action_requests(path: str | Path = DEFAULT_ACTION_PATH,
                         ) -> List[RegressionActionRequest]:
    return [RegressionActionRequest.from_dict(d) for d in _read_jsonl(path)]


def save_action_requests(actions: Sequence[RegressionActionRequest],
                         path: str | Path = DEFAULT_ACTION_PATH) -> None:
    """Write the action-request file (sorted by id; atomic). Distinct from queue."""
    ordered = sorted(actions, key=lambda a: a.action_request_id)
    text = "".join(a.to_json() + "\n" for a in ordered)
    _atomic_write_text(Path(path), text)


def add_action_request(actions: Sequence[RegressionActionRequest],
                       new_action: RegressionActionRequest,
                       ) -> List[RegressionActionRequest]:
    """Merge an action request, idempotent by id (a duplicate is a no-op)."""
    by_id = {a.action_request_id: a for a in actions}
    by_id.setdefault(new_action.action_request_id, new_action)
    return sorted(by_id.values(), key=lambda a: a.action_request_id)


def load_review_audit(path: str | Path = DEFAULT_AUDIT_PATH,
                      ) -> List[RegressionReviewAuditRecord]:
    return [RegressionReviewAuditRecord.from_dict(d) for d in _read_jsonl(path)]


def append_review_audit(records: Sequence[RegressionReviewAuditRecord],
                        path: str | Path = DEFAULT_AUDIT_PATH) -> None:
    """Append audit records (append-only; atomic). Never rewrites prior history.

    Existing records are read and re-emitted unchanged, then the new records are
    appended, then the whole file is replaced atomically — so a failed write
    leaves the previous file byte-identical and no prior decision is lost.
    """
    if not records:
        return
    existing = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    addition = "".join(r.to_json() + "\n" for r in records)
    _atomic_write_text(Path(path), existing + addition)


# ---------------------------------------------------------------------------
# Phase J — queue queries
# ---------------------------------------------------------------------------


def list_review_items(items: Sequence[RegressionReviewItem], *,
                      status: Optional[str] = None,
                      ) -> List[RegressionReviewItem]:
    out = [r for r in items if status is None or r.status == status]
    return _sorted_items(out)


def get_review_item(items: Sequence[RegressionReviewItem], review_item_id: str,
                    ) -> Optional[RegressionReviewItem]:
    for r in items:
        if r.review_item_id == review_item_id:
            return r
    return None


def list_action_requests(actions: Sequence[RegressionActionRequest], *,
                         status: Optional[str] = None,
                         ) -> List[RegressionActionRequest]:
    out = [a for a in actions if status is None or a.status == status]
    return sorted(out, key=lambda a: a.action_request_id)


def get_action_request(actions: Sequence[RegressionActionRequest],
                       action_request_id: str,
                       ) -> Optional[RegressionActionRequest]:
    for a in actions:
        if a.action_request_id == action_request_id:
            return a
    return None


# ---------------------------------------------------------------------------
# Phase L — summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionReviewSummary:
    pending_count: int
    approved_count: int
    rejected_count: int
    deferred_count: int
    evidence_requested_count: int
    duplicate_count: int
    closed_count: int
    stale_count: int
    superseded_count: int
    action_requested_count: int
    action_completed_count: int
    action_failed_count: int
    critical_pending_count: int
    oldest_pending_age: Optional[float] = None
    stale_pending_age: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "_record": "regression_review_summary",
            "pending_count": self.pending_count,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "deferred_count": self.deferred_count,
            "evidence_requested_count": self.evidence_requested_count,
            "duplicate_count": self.duplicate_count,
            "closed_count": self.closed_count,
            "stale_count": self.stale_count,
            "superseded_count": self.superseded_count,
            "action_requested_count": self.action_requested_count,
            "action_completed_count": self.action_completed_count,
            "action_failed_count": self.action_failed_count,
            "critical_pending_count": self.critical_pending_count,
            "oldest_pending_age": self.oldest_pending_age,
            "stale_pending_age": self.stale_pending_age,
        }


def summarize_review_queue(items: Sequence[RegressionReviewItem], *,
                           now: Optional[str] = None) -> RegressionReviewSummary:
    """Deterministic queue summary. Critical pending items are surfaced, not hidden."""
    def _count(status: str) -> int:
        return sum(1 for r in items if r.status == status)

    pending = [r for r in items if r.status == ReviewItemStatus.PENDING]
    stale = [r for r in items if r.status == ReviewItemStatus.STALE]
    pending_ages = [a for a in (_age_days(r.created_at, now) for r in pending)
                    if a is not None]
    stale_ages = [a for a in (_age_days(r.created_at, now) for r in stale)
                  if a is not None]

    approved = sum(1 for r in items if r.status in _APPROVED_FAMILY)

    return RegressionReviewSummary(
        pending_count=len(pending),
        approved_count=approved,
        rejected_count=_count(ReviewItemStatus.REJECTED),
        deferred_count=_count(ReviewItemStatus.DEFERRED),
        evidence_requested_count=_count(ReviewItemStatus.MORE_EVIDENCE_REQUESTED),
        duplicate_count=_count(ReviewItemStatus.DUPLICATE),
        closed_count=_count(ReviewItemStatus.CLOSED_NO_ACTION),
        stale_count=len(stale),
        superseded_count=_count(ReviewItemStatus.SUPERSEDED),
        action_requested_count=_count(ReviewItemStatus.ACTION_REQUESTED),
        action_completed_count=_count(ReviewItemStatus.ACTION_COMPLETED),
        action_failed_count=_count(ReviewItemStatus.ACTION_FAILED),
        critical_pending_count=sum(1 for r in pending if r.is_critical),
        oldest_pending_age=max(pending_ages) if pending_ages else None,
        stale_pending_age=max(stale_ages) if stale_ages else None,
    )


# ---------------------------------------------------------------------------
# Renderers (deterministic; advisory labelling)
# ---------------------------------------------------------------------------


def _critical_first(items: Sequence[RegressionReviewItem],
                    ) -> List[RegressionReviewItem]:
    return sorted(items, key=lambda r: (0 if r.is_critical else 1,
                                        r.status, r.review_item_id))


def render_review_queue_markdown(items: Sequence[RegressionReviewItem], *,
                                 now: Optional[str] = None) -> str:
    """Render the queue. Critical pending items first; action state never implied."""
    summary = summarize_review_queue(items, now=now)
    lines: List[str] = []
    lines.append("# Regression review queue")
    lines.append("")
    lines.append(f"_{REVIEW_LAYER_VERSION}; monitoring layer "
                 f"{MONITOR_LAYER_VERSION}. Review is governance, not execution: "
                 "an approval creates an action request only — it never "
                 "deactivates, rolls back, supersedes, blocks or activates a "
                 "pack._")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in summary.to_dict().items():
        if key == "_record":
            continue
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Items (critical first)")
    lines.append("")
    ordered = _critical_first(items)
    if not ordered:
        lines.append("_(queue empty)_")
        return "\n".join(lines) + "\n"
    lines.append("| review_item_id | status | severity | recommendation | "
                 "proposed_action | packs |")
    lines.append("|---|---|---|---|---|---|")
    for r in ordered:
        packs = ",".join(r.affected_pack_ids)
        crit = " **(CRITICAL)**" if r.is_critical else ""
        lines.append(
            f"| {r.review_item_id} | {r.status}{crit} | {r.severity} | "
            f"{r.recommendation} | {r.proposed_action_type} | {packs} |")
    return "\n".join(lines) + "\n"


def render_review_item_markdown(item: RegressionReviewItem, *,
                                validation: Optional[RegressionReviewValidation] = None,
                                action_request: Optional[RegressionActionRequest] = None,
                                ) -> str:
    """Render a single review item with explicit governance labelling."""
    lines: List[str] = []
    lines.append(f"# Review item {item.review_item_id}")
    lines.append("")
    if item.is_critical:
        lines.append("> **CRITICAL severity**")
        lines.append("")
    lines.append(f"- status: {item.status}")
    lines.append(f"- recommendation: {item.recommendation} "
                 f"(confidence {item.recommendation_confidence})")
    lines.append(f"- proposed action (NOT executed): {item.proposed_action_type}")
    lines.append(f"- monitoring run: {item.monitoring_run_id} "
                 f"({item.monitoring_run_fingerprint})")
    lines.append(f"- recommendation fingerprint: {item.recommendation_fingerprint}")
    lines.append(f"- active state hash: {item.active_state_hash}")
    lines.append(f"- baseline: {item.baseline_hash}")
    lines.append(f"- affected packs: {', '.join(item.affected_pack_ids) or '(none)'}")
    lines.append(f"- affected cases: {', '.join(item.affected_case_ids) or '(none)'}")
    if item.rollback_target_state_hash:
        lines.append(f"- rollback target state: {item.rollback_target_state_hash}")
    lines.append("")
    if validation is not None:
        label = "VALID" if validation.ok else "STALE — cannot be approved"
        lines.append(f"## Validation: {label}")
        lines.append("")
        if validation.staleness_codes:
            for code, reason in zip(validation.staleness_codes, validation.reasons):
                lines.append(f"- {code}: {reason}")
        else:
            lines.append("- monitoring evidence still matches current state")
        lines.append("")
    if item.status in _APPROVED_FAMILY:
        lines.append("## APPROVED FOR REQUEST ONLY")
        lines.append("")
        lines.append("This approval authorised an action request. Lifecycle "
                     "execution is a separate, independently validated step.")
        lines.append("")
    if action_request is not None:
        lines.append("## Action request (NOT EXECUTED)")
        lines.append("")
        lines.append(f"- action_request_id: {action_request.action_request_id}")
        lines.append(f"- requested_action: {action_request.requested_action}")
        lines.append(f"- status: {action_request.status}")
        lines.append(f"- required execution approval: "
                     f"{action_request.required_execution_approval_type}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_action_request_markdown(action: RegressionActionRequest) -> str:
    lines: List[str] = []
    lines.append(f"# Action request {action.action_request_id}")
    lines.append("")
    lines.append("> **NOT EXECUTED by this layer.** This is a request for the "
                 "separate lifecycle layer, which independently re-validates "
                 "state, approval, fingerprints, rollback target and policy "
                 "before any execution.")
    lines.append("")
    for key, value in action.to_dict().items():
        if key == "_record":
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value) or "(none)"
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"
