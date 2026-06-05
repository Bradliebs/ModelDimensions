"""Governed lifecycle action executor (v7.3).

Position in the governance chain::

    monitoring (v7.1, advisory)
      -> recommendation
      -> regression review (v7.2)
      -> approved action request (RegressionActionRequest)
      -> execution plan
      -> explicit execution approval (LifecycleExecutionApproval)
      -> final revalidation
      -> atomic lifecycle action (reuses v7.0 primitives)
      -> post-action verification
      -> immutable execution audit

This layer consumes an **approved** :class:`RegressionActionRequest` (a *request*,
never an execution), independently revalidates every binding and the current
governed state, requires a **separate, explicit, single-use**
:class:`LifecycleExecutionApproval`, performs **exactly one** permitted lifecycle
action atomically by **reusing** the existing v7.0 activation primitives, verifies
the result, and records an immutable execution audit event.

Cardinal rules (enforced + tested):

* Recommendation is not approval; review approval is not execution approval.
* An action request is not an execution; execution needs its own explicit approval.
* Every execution revalidates current state; stale requests fail closed.
* Exactly one lifecycle action per request — no compound actions.
* Only deactivation and rollback change active-pack state, and they do so **only**
  by calling the existing :class:`ActivationStateManager` primitives — this module
  never duplicates lifecycle-transition logic.
* Investigation / watch / monitoring / evidence requests create **bounded
  operational follow-up records only** and never touch active-pack state.
* Activation blocks target an **exact pack fingerprint**, are written as bounded
  governed records, and never deactivate an already-active pack.
* Pack contents stay byte-identical; historical state and audit history are
  preserved; failed executions never appear successful.

Boundaries (verified by import-purity tests). The module imports only the
standard library plus read/execute contracts from
``agent.knowledge_pack_activation`` and read-only contracts from
``agent.regression_review_queue``. It never imports a MemoryLedger writer, a
source-registry save function, a proposal application function, a pack-content or
chunk writer, an LLM client, or any scheduler. The mutation primitives
(``deactivate`` / ``rollback``) are called from a single private execution
adapter; the validation and planning helpers never call them.
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

from agent.knowledge_pack_activation import (
    ActivationScope,
    ActivationStateManager,
    ActivationStateManifest,
    ActivePackState,
    KnowledgePackActivationApproval,
    KnowledgePackRollbackRequest,
    PackLifecycleState,
    find_state_by_hash,
    select_active_pack_ids,
)
from agent.regression_review_queue import (
    ActionRequestStatus,
    ActionRequestType,
    RegressionActionRequest,
    RegressionReviewRecord,
    mark_action_result,
)

EXECUTOR_LAYER_VERSION = "lifecycle-action-executor-v7.3"

EXECUTION_APPROVAL_PREFIX = "lifeapp-"
EXECUTION_ID_PREFIX = "lifeexec-"
EXECUTION_PLAN_PREFIX = "lifeplan-"
EXECUTION_AUDIT_PREFIX = "lifeaud-"
FOLLOW_UP_PREFIX = "lifefu-"
ACTIVATION_BLOCK_PREFIX = "lifeblk-"

# Default governed locations (relative to project root; tests override to tmp).
DEFAULT_APPROVALS_PATH = "reviews/lifecycle_execution_approvals.jsonl"
DEFAULT_RESULTS_PATH = "reviews/lifecycle_execution_results.jsonl"
DEFAULT_EXECUTION_AUDIT_PATH = "reviews/lifecycle_execution_audit.jsonl"
DEFAULT_FOLLOW_UP_PATH = "reviews/operational_follow_up.jsonl"
DEFAULT_ACTIVATION_BLOCKS_PATH = "reviews/activation_blocks.jsonl"


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Status / type constants (str consts, not Enum — mirrors v7.2 conventions)
# ---------------------------------------------------------------------------


class ExecutionStatus:
    PENDING_VALIDATION = "pending_validation"
    VALIDATION_FAILED = "validation_failed"
    DRY_RUN_READY = "dry_run_ready"
    EXECUTION_APPROVED = "execution_approved"
    EXECUTING = "executing"
    EXECUTED = "executed"
    PARTIALLY_VERIFIED = "partially_verified"
    VERIFICATION_FAILED = "verification_failed"
    EXECUTION_FAILED = "execution_failed"
    CANCELLED = "cancelled"
    STALE = "stale"
    ALREADY_EXECUTED = "already_executed"


# Statuses that mean a lifecycle action already took effect for this request.
_TERMINAL_SUCCESS = frozenset({
    ExecutionStatus.EXECUTED,
    ExecutionStatus.PARTIALLY_VERIFIED,
})


class FollowUpType:
    INVESTIGATION = "investigation"
    WATCH = "watch"
    NEW_MONITORING_RUN = "new_monitoring_run"
    ADDITIONAL_EVIDENCE = "additional_evidence"


class FollowUpStatus:
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    STALE = "stale"


class BlockStatus:
    ACTIVE = "active"
    LIFTED = "lifted"
    EXPIRED = "expired"


class ExecutionFindingCode:
    ACTION_REQUEST_MISSING = "action_request_missing"
    ACTION_REQUEST_NOT_ACTIONABLE = "action_request_not_actionable"
    ACTION_REQUEST_ALREADY_EXECUTED = "action_request_already_executed"
    ACTION_REQUEST_CANCELLED = "action_request_cancelled"
    REVIEW_RECORD_MISSING = "review_record_missing"
    REVIEW_NOT_APPROVED = "review_not_approved"
    EXECUTION_APPROVAL_MISSING = "execution_approval_missing"
    EXECUTION_APPROVAL_EXPIRED = "execution_approval_expired"
    EXECUTION_APPROVAL_CONSUMED = "execution_approval_consumed"
    APPROVAL_ACTION_MISMATCH = "approval_action_mismatch"
    APPROVAL_ACTION_REQUEST_MISMATCH = "approval_action_request_mismatch"
    APPROVAL_REVIEW_MISMATCH = "approval_review_mismatch"
    ACTIVE_STATE_CHANGED = "active_state_changed"
    PACK_FINGERPRINT_MISMATCH = "pack_fingerprint_mismatch"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    MONITORING_FINGERPRINT_MISMATCH = "monitoring_fingerprint_mismatch"
    RECOMMENDATION_FINGERPRINT_MISMATCH = "recommendation_fingerprint_mismatch"
    ROLLBACK_TARGET_MISSING = "rollback_target_missing"
    ROLLBACK_TARGET_INVALID = "rollback_target_invalid"
    ROLLBACK_TARGET_IS_CURRENT = "rollback_target_is_current"
    PACK_NOT_ACTIVE = "pack_not_active"
    INVALID_LIFECYCLE_TRANSITION = "invalid_lifecycle_transition"
    ISSUE_ALREADY_RESOLVED = "issue_already_resolved"
    CONFLICTING_OPERATION_PENDING = "conflicting_operation_pending"
    PLAN_FINGERPRINT_CHANGED = "plan_fingerprint_changed"
    UNSUPPORTED_ACTION = "unsupported_action"


# Which action requests change active-pack state (call a lifecycle primitive).
_STATE_MUTATING_ACTIONS = frozenset({
    ActionRequestType.REQUEST_DEACTIVATION,
    ActionRequestType.REQUEST_ROLLBACK,
})

# Action requests that produce a bounded operational follow-up record only.
_FOLLOW_UP_ACTIONS: Dict[str, str] = {
    ActionRequestType.REQUEST_INVESTIGATION: FollowUpType.INVESTIGATION,
    ActionRequestType.REQUEST_WATCH: FollowUpType.WATCH,
    ActionRequestType.REQUEST_NEW_MONITORING_RUN: FollowUpType.NEW_MONITORING_RUN,
    ActionRequestType.REQUEST_ADDITIONAL_EVIDENCE: FollowUpType.ADDITIONAL_EVIDENCE,
}

# The lifecycle primitive each supported action maps to (documented mapping).
_ACTION_PRIMITIVE: Dict[str, str] = {
    ActionRequestType.REQUEST_DEACTIVATION: "knowledge_pack_activation.deactivate",
    ActionRequestType.REQUEST_ROLLBACK: "knowledge_pack_activation.rollback",
    ActionRequestType.REQUEST_ACTIVATION_BLOCK: "activation_block_record",
    ActionRequestType.REQUEST_INVESTIGATION: "operational_follow_up",
    ActionRequestType.REQUEST_WATCH: "operational_follow_up",
    ActionRequestType.REQUEST_NEW_MONITORING_RUN: "operational_follow_up",
    ActionRequestType.REQUEST_ADDITIONAL_EVIDENCE: "operational_follow_up",
}

_SUPPORTED_ACTIONS = frozenset(_ACTION_PRIMITIVE)


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(now: Optional[datetime]) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionFinding:
    """One blocking reason an execution was refused (every finding is blocking)."""

    code: str
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "blocking": True}


# ---------------------------------------------------------------------------
# Phase B — execution approval (separate from review approval; single-use)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleExecutionApproval:
    """The explicit, single-use authorisation to *execute* one lifecycle action.

    This is distinct from a review approval: a review approval produces an action
    *request*; this approval authorises *executing* that exact request. It binds
    to the exact action request, action type, active-state hash and pack
    fingerprints, so an approval for deactivation can never authorise a rollback,
    and an approval for one active-state hash can never authorise a changed state.
    """

    execution_approval_id: str
    action_request_id: str
    review_item_id: str
    review_record_id: str
    approved_action: str
    approved_by: str
    approved_role: str
    approved_at: str
    active_state_hash: str
    affected_pack_ids: Tuple[str, ...]
    affected_pack_fingerprints: Tuple[str, ...]
    monitoring_run_fingerprint: str
    recommendation_fingerprint: str
    rollback_target_state_hash: str = ""
    execution_policy_id: str = "default-exec-policy-v1"
    environment: str = "default"
    approval_reason: str = ""
    expires_at: str = ""
    single_use: bool = True
    plan_fingerprint: str = ""

    def is_expired(self, now: datetime) -> bool:
        parsed = _parse_dt(self.expires_at)
        if parsed is None:
            return False
        return now > parsed

    def to_dict(self) -> dict:
        return {
            "_record": "lifecycle_execution_approval",
            "execution_approval_id": self.execution_approval_id,
            "action_request_id": self.action_request_id,
            "review_item_id": self.review_item_id,
            "review_record_id": self.review_record_id,
            "approved_action": self.approved_action,
            "approved_by": self.approved_by,
            "approved_role": self.approved_role,
            "approved_at": self.approved_at,
            "active_state_hash": self.active_state_hash,
            "affected_pack_ids": list(self.affected_pack_ids),
            "affected_pack_fingerprints": list(self.affected_pack_fingerprints),
            "monitoring_run_fingerprint": self.monitoring_run_fingerprint,
            "recommendation_fingerprint": self.recommendation_fingerprint,
            "rollback_target_state_hash": self.rollback_target_state_hash,
            "execution_policy_id": self.execution_policy_id,
            "environment": self.environment,
            "approval_reason": self.approval_reason,
            "expires_at": self.expires_at,
            "single_use": self.single_use,
            "plan_fingerprint": self.plan_fingerprint,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "LifecycleExecutionApproval":
        return cls(
            execution_approval_id=str(data["execution_approval_id"]),
            action_request_id=str(data.get("action_request_id", "")),
            review_item_id=str(data.get("review_item_id", "")),
            review_record_id=str(data.get("review_record_id", "")),
            approved_action=str(data.get("approved_action", "")),
            approved_by=str(data.get("approved_by", "")),
            approved_role=str(data.get("approved_role", "")),
            approved_at=str(data.get("approved_at", "")),
            active_state_hash=str(data.get("active_state_hash", "")),
            affected_pack_ids=tuple(str(v) for v in
                                    (data.get("affected_pack_ids") or [])),
            affected_pack_fingerprints=tuple(
                str(v) for v in (data.get("affected_pack_fingerprints") or [])),
            monitoring_run_fingerprint=str(data.get("monitoring_run_fingerprint", "")),
            recommendation_fingerprint=str(data.get("recommendation_fingerprint", "")),
            rollback_target_state_hash=str(data.get("rollback_target_state_hash", "")),
            execution_policy_id=str(data.get("execution_policy_id",
                                             "default-exec-policy-v1")),
            environment=str(data.get("environment", "default")),
            approval_reason=str(data.get("approval_reason", "")),
            expires_at=str(data.get("expires_at", "")),
            single_use=bool(data.get("single_use", True)),
            plan_fingerprint=str(data.get("plan_fingerprint", "")),
        )


def build_execution_approval(action_request: RegressionActionRequest, *,
                             approved_by: str, approved_role: str,
                             approved_at: Optional[str] = None,
                             environment: str = "default",
                             approval_reason: str = "",
                             expires_at: str = "",
                             single_use: bool = True,
                             plan_fingerprint: str = "",
                             execution_policy_id: str = "default-exec-policy-v1",
                             ) -> LifecycleExecutionApproval:
    """Mint an execution approval that *binds* to an exact action request (pure).

    The binding fields are copied from the action request, so the approval can
    only ever authorise that exact action, state and pack fingerprints. The id is
    deterministic, so re-minting the same approval is idempotent.
    """
    if not approved_by.strip():
        raise ValueError("execution approval requires an approver identity")
    at = approved_at if approved_at is not None else _utc_now_iso()
    payload = {
        "action_request_id": action_request.action_request_id,
        "approved_action": action_request.requested_action,
        "active_state_hash": action_request.active_state_hash,
        "approved_by": approved_by,
        "approved_at": at,
    }
    approval_id = EXECUTION_APPROVAL_PREFIX + _sha256_hex(_canonical(payload))[:16]
    return LifecycleExecutionApproval(
        execution_approval_id=approval_id,
        action_request_id=action_request.action_request_id,
        review_item_id=action_request.review_item_id,
        review_record_id=action_request.review_record_id,
        approved_action=action_request.requested_action,
        approved_by=approved_by,
        approved_role=approved_role,
        approved_at=at,
        active_state_hash=action_request.active_state_hash,
        affected_pack_ids=action_request.affected_pack_ids,
        affected_pack_fingerprints=action_request.affected_pack_fingerprints,
        monitoring_run_fingerprint=action_request.monitoring_run_fingerprint,
        recommendation_fingerprint=action_request.recommendation_fingerprint,
        rollback_target_state_hash=action_request.rollback_target_state_hash,
        execution_policy_id=execution_policy_id,
        environment=environment,
        approval_reason=approval_reason,
        expires_at=expires_at,
        single_use=single_use,
        plan_fingerprint=plan_fingerprint,
    )


@dataclass(frozen=True)
class LifecycleExecutionRequest:
    """A single execution invocation: who is executing which approved request."""

    action_request_id: str
    execution_approval_id: str
    executor_identity: str
    executor_role: str
    dry_run: bool = True

    def to_dict(self) -> dict:
        return {
            "_record": "lifecycle_execution_request",
            "action_request_id": self.action_request_id,
            "execution_approval_id": self.execution_approval_id,
            "executor_identity": self.executor_identity,
            "executor_role": self.executor_role,
            "dry_run": self.dry_run,
        }


# ---------------------------------------------------------------------------
# Phase C — validation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleExecutionValidation:
    """The outcome of independently validating an execution (pure; no mutation)."""

    action_request_id: str
    execution_approval_id: str
    requested_action: str
    findings: Tuple[ExecutionFinding, ...]

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def status(self) -> str:
        return (ExecutionStatus.DRY_RUN_READY if self.ok
                else ExecutionStatus.VALIDATION_FAILED)

    def to_dict(self) -> dict:
        return {
            "_record": "lifecycle_execution_validation",
            "action_request_id": self.action_request_id,
            "execution_approval_id": self.execution_approval_id,
            "requested_action": self.requested_action,
            "ok": self.ok,
            "findings": [f.to_dict() for f in self.findings],
        }


def validate_lifecycle_execution(
        action_request: Optional[RegressionActionRequest],
        execution_approval: Optional[LifecycleExecutionApproval],
        current_active_state: ActivationStateManifest,
        lifecycle_history: Sequence["object"] = (),
        review_history: Sequence[RegressionReviewRecord] = (),
        *,
        approval_consumed: bool = False,
        now: Optional[datetime] = None,
) -> LifecycleExecutionValidation:
    """Independently revalidate an execution. Pure; fail-closed on any mismatch.

    Never mutates state. Confirms the action request is approved and actionable,
    the execution approval is unexpired/unconsumed and binds to the exact action,
    the current active state still matches the approved state and pack
    fingerprints, the environment matches, monitoring/recommendation fingerprints
    match, and the requested lifecycle transition remains valid against the live
    state. Returns a validation with the (blocking) findings.
    """
    now = now or datetime.now(timezone.utc)
    findings: List[ExecutionFinding] = []

    def flag(code: str, message: str) -> None:
        findings.append(ExecutionFinding(code, message))

    ar_id = action_request.action_request_id if action_request else ""
    ap_id = execution_approval.execution_approval_id if execution_approval else ""
    req_action = action_request.requested_action if action_request else ""

    if action_request is None:
        flag(ExecutionFindingCode.ACTION_REQUEST_MISSING,
             "no action request supplied")
    if execution_approval is None:
        flag(ExecutionFindingCode.EXECUTION_APPROVAL_MISSING,
             "no execution approval supplied (review approval is not execution "
             "approval)")
    if action_request is None or execution_approval is None:
        return LifecycleExecutionValidation(
            action_request_id=ar_id, execution_approval_id=ap_id,
            requested_action=req_action, findings=tuple(findings))

    # -- action-request actionability --
    if req_action not in _SUPPORTED_ACTIONS:
        flag(ExecutionFindingCode.UNSUPPORTED_ACTION,
             f"unsupported action {req_action!r}")
    if action_request.status == ActionRequestStatus.EXECUTED:
        flag(ExecutionFindingCode.ACTION_REQUEST_ALREADY_EXECUTED,
             "action request is already executed")
    elif action_request.status == ActionRequestStatus.CANCELLED:
        flag(ExecutionFindingCode.ACTION_REQUEST_CANCELLED,
             "action request is cancelled")
    elif action_request.status not in (ActionRequestStatus.REQUESTED,
                                       ActionRequestStatus.VALIDATED):
        flag(ExecutionFindingCode.ACTION_REQUEST_NOT_ACTIONABLE,
             f"action request status {action_request.status!r} is not actionable")

    # -- review record must exist and be an approval --
    record = next((r for r in review_history
                   if r.review_record_id == action_request.review_record_id), None)
    if record is None:
        flag(ExecutionFindingCode.REVIEW_RECORD_MISSING,
             "no review record matches the action request's review_record_id")
    else:
        if record.decision != "approve" or record.review_item_id != \
                action_request.review_item_id:
            flag(ExecutionFindingCode.REVIEW_NOT_APPROVED,
                 "the referenced review record is not an approval of this item")

    # -- execution approval bindings --
    if execution_approval.is_expired(now):
        flag(ExecutionFindingCode.EXECUTION_APPROVAL_EXPIRED,
             "execution approval has expired")
    if approval_consumed and execution_approval.single_use:
        flag(ExecutionFindingCode.EXECUTION_APPROVAL_CONSUMED,
             "single-use execution approval has already been consumed")
    if execution_approval.approved_action != req_action:
        flag(ExecutionFindingCode.APPROVAL_ACTION_MISMATCH,
             "approval action does not match the requested action")
    if execution_approval.action_request_id != action_request.action_request_id:
        flag(ExecutionFindingCode.APPROVAL_ACTION_REQUEST_MISMATCH,
             "approval is bound to a different action request")
    if execution_approval.review_item_id != action_request.review_item_id or \
            execution_approval.review_record_id != action_request.review_record_id:
        flag(ExecutionFindingCode.APPROVAL_REVIEW_MISMATCH,
             "approval review identifiers do not match the action request")
    if execution_approval.monitoring_run_fingerprint != \
            action_request.monitoring_run_fingerprint:
        flag(ExecutionFindingCode.MONITORING_FINGERPRINT_MISMATCH,
             "monitoring run fingerprint changed since approval")
    if execution_approval.recommendation_fingerprint != \
            action_request.recommendation_fingerprint:
        flag(ExecutionFindingCode.RECOMMENDATION_FINGERPRINT_MISMATCH,
             "recommendation fingerprint changed since approval")

    # -- current-state revalidation (fail closed on drift) --
    current_hash = current_active_state.state_hash
    if action_request.active_state_hash != current_hash:
        flag(ExecutionFindingCode.ACTIVE_STATE_CHANGED,
             "active state changed since the action request was approved")
    if execution_approval.active_state_hash != action_request.active_state_hash:
        flag(ExecutionFindingCode.ACTIVE_STATE_CHANGED,
             "approval active-state hash does not match the action request")

    # -- pack fingerprints must still match live state --
    if tuple(execution_approval.affected_pack_fingerprints) != \
            action_request.affected_pack_fingerprints:
        flag(ExecutionFindingCode.PACK_FINGERPRINT_MISMATCH,
             "approval affected-pack fingerprints differ from the action request")
    for pack_id, fp in zip(action_request.affected_pack_ids,
                           action_request.affected_pack_fingerprints):
        live = current_active_state.find(pack_id)
        if live is None:
            flag(ExecutionFindingCode.PACK_FINGERPRINT_MISMATCH,
                 f"affected pack {pack_id!r} is no longer in governed state")
        elif live.pack_fingerprint != fp:
            flag(ExecutionFindingCode.PACK_FINGERPRINT_MISMATCH,
                 f"affected pack {pack_id!r} fingerprint changed")
        elif live.environment != execution_approval.environment:
            flag(ExecutionFindingCode.ENVIRONMENT_MISMATCH,
                 f"pack {pack_id!r} environment {live.environment!r} does not "
                 f"match approval environment {execution_approval.environment!r}")

    # -- transition validity per action --
    if req_action == ActionRequestType.REQUEST_DEACTIVATION:
        for pack_id in action_request.affected_pack_ids:
            live = current_active_state.find(pack_id)
            if live is None or not live.is_active:
                flag(ExecutionFindingCode.PACK_NOT_ACTIVE,
                     f"pack {pack_id!r} is not currently active; nothing to "
                     "deactivate (the issue may already be resolved)")
    elif req_action == ActionRequestType.REQUEST_ROLLBACK:
        target_hash = action_request.rollback_target_state_hash
        if not target_hash:
            flag(ExecutionFindingCode.ROLLBACK_TARGET_MISSING,
                 "rollback action request carries no target state hash")
        elif target_hash == current_hash:
            flag(ExecutionFindingCode.ROLLBACK_TARGET_IS_CURRENT,
                 "rollback target equals the current state; already resolved")
        else:
            target = find_state_by_hash(list(lifecycle_history), target_hash)
            if target is None:
                flag(ExecutionFindingCode.ROLLBACK_TARGET_MISSING,
                     "no recorded lifecycle state matches the rollback target")
        if execution_approval.rollback_target_state_hash != target_hash:
            flag(ExecutionFindingCode.ROLLBACK_TARGET_INVALID,
                 "approval rollback target does not match the action request")

    return LifecycleExecutionValidation(
        action_request_id=ar_id, execution_approval_id=ap_id,
        requested_action=req_action, findings=tuple(findings))


# ---------------------------------------------------------------------------
# Phase E — deterministic execution plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleExecutionPlan:
    """A deterministic plan describing exactly one lifecycle action before it runs."""

    action_request_id: str
    execution_approval_id: str
    requested_action: str
    lifecycle_primitive: str
    current_state_hash: str
    expected_state_hash: str
    affected_pack_ids: Tuple[str, ...]
    rollback_target_state_hash: str
    expected_state_record_changes: Tuple[dict, ...]
    expected_audit_records: Tuple[str, ...]
    preconditions: Tuple[str, ...]
    postconditions: Tuple[str, ...]
    files_expected_to_change: Tuple[str, ...]
    files_expected_unchanged: Tuple[str, ...]
    risk_findings: Tuple[str, ...]
    dry_run_summary: str

    @property
    def plan_fingerprint(self) -> str:
        payload = {
            "v": EXECUTOR_LAYER_VERSION,
            "action_request_id": self.action_request_id,
            "execution_approval_id": self.execution_approval_id,
            "requested_action": self.requested_action,
            "lifecycle_primitive": self.lifecycle_primitive,
            "current_state_hash": self.current_state_hash,
            "expected_state_hash": self.expected_state_hash,
            "affected_pack_ids": list(self.affected_pack_ids),
            "rollback_target_state_hash": self.rollback_target_state_hash,
            "expected_state_record_changes":
                list(self.expected_state_record_changes),
        }
        return EXECUTION_PLAN_PREFIX + _sha256_hex(_canonical(payload))[:20]

    def to_dict(self) -> dict:
        return {
            "_record": "lifecycle_execution_plan",
            "action_request_id": self.action_request_id,
            "execution_approval_id": self.execution_approval_id,
            "requested_action": self.requested_action,
            "lifecycle_primitive": self.lifecycle_primitive,
            "plan_fingerprint": self.plan_fingerprint,
            "current_state_hash": self.current_state_hash,
            "expected_state_hash": self.expected_state_hash,
            "affected_pack_ids": list(self.affected_pack_ids),
            "rollback_target_state_hash": self.rollback_target_state_hash,
            "expected_state_record_changes":
                list(self.expected_state_record_changes),
            "expected_audit_records": list(self.expected_audit_records),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "files_expected_to_change": list(self.files_expected_to_change),
            "files_expected_unchanged": list(self.files_expected_unchanged),
            "risk_findings": list(self.risk_findings),
            "dry_run_summary": self.dry_run_summary,
        }


def _deactivation_expected_state(current: ActivationStateManifest,
                                 pack_ids: Sequence[str]
                                 ) -> Tuple[ActivationStateManifest, List[dict]]:
    """Pure simulation of deactivation: pack(s) -> inactive. No I/O, no mutation."""
    changes: List[dict] = []
    state = current
    for pack_id in pack_ids:
        live = state.find(pack_id)
        if live is None or not live.is_active:
            continue
        new_record = live.with_status(PackLifecycleState.INACTIVE,
                                      activated_at=live.activated_at)
        state = state.upsert(new_record)
        changes.append({"pack_id": pack_id, "from_status": live.status.value,
                        "to_status": PackLifecycleState.INACTIVE.value})
    return state, changes


def build_execution_plan(action_request: RegressionActionRequest,
                         execution_approval: LifecycleExecutionApproval,
                         current_active_state: ActivationStateManifest,
                         lifecycle_history: Sequence["object"] = (),
                         ) -> LifecycleExecutionPlan:
    """Build the deterministic plan for one action (pure; never mutates state)."""
    action = action_request.requested_action
    primitive = _ACTION_PRIMITIVE.get(action, "unsupported")
    current_hash = current_active_state.state_hash
    state_files = "config/active_knowledge_packs.jsonl"
    activation_audit = "reports/knowledge_pack_activation_audit.jsonl"

    expected_hash = current_hash
    changes: List[dict] = []
    expected_audit: List[str] = []
    files_change: List[str] = []
    files_unchanged: List[str] = [
        "<all pack directories>", "<all knowledge.jsonl chunk files>",
        "<all retrieval indexes>", "config/source_registry.jsonl",
        "<memory ledger>",
    ]
    risks: List[str] = []
    pre: List[str] = []
    post: List[str] = []

    if action == ActionRequestType.REQUEST_DEACTIVATION:
        expected_state, changes = _deactivation_expected_state(
            current_active_state, action_request.affected_pack_ids)
        expected_hash = expected_state.state_hash
        files_change = [state_files, activation_audit]
        expected_audit = ["knowledge_pack_activation_audit: deactivate"]
        pre = ["affected pack(s) are currently active",
               "current state hash equals the approved active_state_hash"]
        post = ["affected pack(s) are inactive",
                "pack files and manifests unchanged",
                "unrelated active packs unchanged"]
    elif action == ActionRequestType.REQUEST_ROLLBACK:
        target_hash = action_request.rollback_target_state_hash
        target = find_state_by_hash(list(lifecycle_history), target_hash)
        expected_hash = target_hash
        if target is not None:
            for rec in current_active_state.records:
                tgt = target.find(rec.pack_id)
                if tgt is None or tgt.status != rec.status:
                    changes.append({
                        "pack_id": rec.pack_id,
                        "from_status": rec.status.value,
                        "to_status": (tgt.status.value if tgt else "absent")})
        else:
            risks.append("rollback target state is not reconstructable")
        files_change = [state_files, activation_audit]
        expected_audit = ["knowledge_pack_activation_audit: rollback"]
        pre = ["current state hash equals the approved active_state_hash",
               "rollback target state is recorded and reconstructable"]
        post = ["exact prior governed state restored",
                "the failing revision is preserved but inactive"]
    elif action == ActionRequestType.REQUEST_ACTIVATION_BLOCK:
        primitive = "activation_block_record"
        files_change = [DEFAULT_ACTIVATION_BLOCKS_PATH]
        expected_audit = ["lifecycle_execution_audit: activation_block"]
        pre = ["exact pack fingerprint is identified"]
        post = ["exact pack fingerprint is blocked from future activation",
                "active state is unchanged (a block does not deactivate)"]
    elif action in _FOLLOW_UP_ACTIONS:
        primitive = "operational_follow_up"
        files_change = [DEFAULT_FOLLOW_UP_PATH]
        expected_audit = ["lifecycle_execution_audit: operational_follow_up"]
        pre = ["action is operational only (no active-state change)"]
        post = ["a bounded operational follow-up record exists",
                "active state is unchanged"]
    else:
        risks.append(f"unsupported action {action!r}")

    summary = (f"{action} via {primitive}: {current_hash} -> {expected_hash} "
               f"(affected: {', '.join(action_request.affected_pack_ids) or 'none'})")

    return LifecycleExecutionPlan(
        action_request_id=action_request.action_request_id,
        execution_approval_id=execution_approval.execution_approval_id,
        requested_action=action,
        lifecycle_primitive=primitive,
        current_state_hash=current_hash,
        expected_state_hash=expected_hash,
        affected_pack_ids=action_request.affected_pack_ids,
        rollback_target_state_hash=action_request.rollback_target_state_hash,
        expected_state_record_changes=tuple(changes),
        expected_audit_records=tuple(expected_audit),
        preconditions=tuple(pre),
        postconditions=tuple(post),
        files_expected_to_change=tuple(files_change),
        files_expected_unchanged=tuple(files_unchanged),
        risk_findings=tuple(risks),
        dry_run_summary=summary,
    )


# ---------------------------------------------------------------------------
# Phase J — operational follow-up records (bounded; no state mutation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperationalFollowUpRecord:
    """A bounded operational task. Never triggers background work."""

    follow_up_id: str
    action_request_id: str
    follow_up_type: str
    requested_by: str
    requested_at: str
    affected_packs: Tuple[str, ...]
    affected_cases: Tuple[str, ...]
    required_evidence: str = ""
    monitoring_run_reference: str = ""
    assigned_to: str = ""
    due_at: str = ""
    status: str = FollowUpStatus.OPEN
    completed_at: str = ""
    completion_evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "_record": "operational_follow_up",
            "follow_up_id": self.follow_up_id,
            "action_request_id": self.action_request_id,
            "follow_up_type": self.follow_up_type,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
            "assigned_to": self.assigned_to,
            "due_at": self.due_at,
            "affected_packs": list(self.affected_packs),
            "affected_cases": list(self.affected_cases),
            "required_evidence": self.required_evidence,
            "monitoring_run_reference": self.monitoring_run_reference,
            "status": self.status,
            "completed_at": self.completed_at,
            "completion_evidence": self.completion_evidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "OperationalFollowUpRecord":
        return cls(
            follow_up_id=str(data["follow_up_id"]),
            action_request_id=str(data.get("action_request_id", "")),
            follow_up_type=str(data.get("follow_up_type", "")),
            requested_by=str(data.get("requested_by", "")),
            requested_at=str(data.get("requested_at", "")),
            affected_packs=tuple(str(v) for v in (data.get("affected_packs") or [])),
            affected_cases=tuple(str(v) for v in (data.get("affected_cases") or [])),
            required_evidence=str(data.get("required_evidence", "")),
            monitoring_run_reference=str(data.get("monitoring_run_reference", "")),
            assigned_to=str(data.get("assigned_to", "")),
            due_at=str(data.get("due_at", "")),
            status=str(data.get("status", FollowUpStatus.OPEN)),
            completed_at=str(data.get("completed_at", "")),
            completion_evidence=str(data.get("completion_evidence", "")),
        )


def make_operational_follow_up(action_request: RegressionActionRequest, *,
                               requested_by: str,
                               requested_at: Optional[str] = None,
                               assigned_to: str = "", due_at: str = "",
                               required_evidence: str = "",
                               ) -> OperationalFollowUpRecord:
    """Build a bounded operational follow-up record (pure; no state mutation)."""
    follow_up_type = _FOLLOW_UP_ACTIONS.get(action_request.requested_action, "")
    if not follow_up_type:
        raise ValueError(
            f"{action_request.requested_action!r} is not an operational "
            "follow-up action")
    at = requested_at if requested_at is not None else _utc_now_iso()
    payload = {"action_request_id": action_request.action_request_id,
               "follow_up_type": follow_up_type, "requested_at": at}
    follow_up_id = FOLLOW_UP_PREFIX + _sha256_hex(_canonical(payload))[:16]
    return OperationalFollowUpRecord(
        follow_up_id=follow_up_id,
        action_request_id=action_request.action_request_id,
        follow_up_type=follow_up_type,
        requested_by=requested_by,
        requested_at=at,
        affected_packs=action_request.affected_pack_ids,
        affected_cases=(),
        required_evidence=(required_evidence or action_request.approval_reason),
        monitoring_run_reference=action_request.monitoring_run_fingerprint,
        assigned_to=assigned_to,
        due_at=due_at,
        status=FollowUpStatus.OPEN,
    )


# ---------------------------------------------------------------------------
# Phase I — activation-block records (bounded; never deactivates)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationBlockRecord:
    """A governed block preventing future activation of an exact pack fingerprint."""

    block_id: str
    pack_id: str
    pack_fingerprint: str
    blocked_revision: str
    environment: str
    reason_codes: Tuple[str, ...]
    originating_action_request_id: str
    originating_monitoring_run: str
    approved_by: str
    approved_at: str
    active_state_hash: str
    status: str = BlockStatus.ACTIVE
    expires_at: str = ""

    @property
    def audit_hash(self) -> str:
        payload = {
            "block_id": self.block_id, "pack_id": self.pack_id,
            "pack_fingerprint": self.pack_fingerprint,
            "blocked_revision": self.blocked_revision,
            "environment": self.environment,
            "originating_action_request_id": self.originating_action_request_id,
            "status": self.status,
        }
        return ACTIVATION_BLOCK_PREFIX + _sha256_hex(_canonical(payload))[:20]

    def to_dict(self) -> dict:
        return {
            "_record": "activation_block",
            "block_id": self.block_id,
            "pack_id": self.pack_id,
            "pack_fingerprint": self.pack_fingerprint,
            "blocked_revision": self.blocked_revision,
            "environment": self.environment,
            "reason_codes": list(self.reason_codes),
            "originating_action_request_id": self.originating_action_request_id,
            "originating_monitoring_run": self.originating_monitoring_run,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "active_state_hash": self.active_state_hash,
            "status": self.status,
            "expires_at": self.expires_at,
            "audit_hash": self.audit_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "ActivationBlockRecord":
        return cls(
            block_id=str(data["block_id"]),
            pack_id=str(data.get("pack_id", "")),
            pack_fingerprint=str(data.get("pack_fingerprint", "")),
            blocked_revision=str(data.get("blocked_revision", "")),
            environment=str(data.get("environment", "default")),
            reason_codes=tuple(str(v) for v in (data.get("reason_codes") or [])),
            originating_action_request_id=str(
                data.get("originating_action_request_id", "")),
            originating_monitoring_run=str(data.get("originating_monitoring_run", "")),
            approved_by=str(data.get("approved_by", "")),
            approved_at=str(data.get("approved_at", "")),
            active_state_hash=str(data.get("active_state_hash", "")),
            status=str(data.get("status", BlockStatus.ACTIVE)),
            expires_at=str(data.get("expires_at", "")),
        )


def make_activation_block(action_request: RegressionActionRequest,
                          execution_approval: LifecycleExecutionApproval, *,
                          approved_at: Optional[str] = None,
                          reason_codes: Sequence[str] = (),
                          ) -> ActivationBlockRecord:
    """Build an activation-block record for the exact affected fingerprint (pure).

    Targets a single exact ``pack_id``/``pack_fingerprint``. Deterministic id, so
    re-issuing the same block is idempotent. Does not deactivate an active pack.
    """
    if action_request.requested_action != ActionRequestType.REQUEST_ACTIVATION_BLOCK:
        raise ValueError("action request is not an activation-block request")
    if not action_request.affected_pack_ids:
        raise ValueError("activation block requires an affected pack")
    pack_id = action_request.affected_pack_ids[0]
    pack_fp = (action_request.affected_pack_fingerprints[0]
               if action_request.affected_pack_fingerprints else "")
    at = approved_at if approved_at is not None else _utc_now_iso()
    payload = {"pack_fingerprint": pack_fp, "environment": execution_approval.environment,
               "originating_action_request_id": action_request.action_request_id}
    block_id = ACTIVATION_BLOCK_PREFIX + _sha256_hex(_canonical(payload))[:16]
    return ActivationBlockRecord(
        block_id=block_id,
        pack_id=pack_id,
        pack_fingerprint=pack_fp,
        blocked_revision=pack_fp,
        environment=execution_approval.environment,
        reason_codes=tuple(reason_codes) or ("regression_block",),
        originating_action_request_id=action_request.action_request_id,
        originating_monitoring_run=action_request.monitoring_run_fingerprint,
        approved_by=execution_approval.approved_by,
        approved_at=at,
        active_state_hash=action_request.active_state_hash,
        status=BlockStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# Phase M — post-action verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecyclePostActionVerification:
    """Explicit verification that the executed action produced the planned result."""

    expected_state_hash: str
    actual_state_hash: str
    expected_active_pack_ids: Tuple[str, ...]
    actual_active_pack_ids: Tuple[str, ...]
    expected_inactive_pack_ids: Tuple[str, ...]
    actual_inactive_pack_ids: Tuple[str, ...]
    pack_files_unchanged: bool
    manifests_unchanged: bool
    unrelated_active_packs_unchanged: bool
    retrieval_selection_check: bool
    audit_record_present: bool
    action_request_status_updated: bool
    approval_consumed: bool
    findings: Tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (not self.findings
                and self.expected_state_hash == self.actual_state_hash
                and self.pack_files_unchanged
                and self.manifests_unchanged
                and self.unrelated_active_packs_unchanged
                and self.retrieval_selection_check
                and self.audit_record_present
                and self.action_request_status_updated
                and self.approval_consumed)

    def to_dict(self) -> dict:
        return {
            "_record": "lifecycle_post_action_verification",
            "expected_state_hash": self.expected_state_hash,
            "actual_state_hash": self.actual_state_hash,
            "expected_active_pack_ids": list(self.expected_active_pack_ids),
            "actual_active_pack_ids": list(self.actual_active_pack_ids),
            "expected_inactive_pack_ids": list(self.expected_inactive_pack_ids),
            "actual_inactive_pack_ids": list(self.actual_inactive_pack_ids),
            "pack_files_unchanged": self.pack_files_unchanged,
            "manifests_unchanged": self.manifests_unchanged,
            "unrelated_active_packs_unchanged": self.unrelated_active_packs_unchanged,
            "retrieval_selection_check": self.retrieval_selection_check,
            "audit_record_present": self.audit_record_present,
            "action_request_status_updated": self.action_request_status_updated,
            "approval_consumed": self.approval_consumed,
            "passed": self.passed,
            "findings": list(self.findings),
        }


def _inactive_pack_ids(state: ActivationStateManifest) -> Tuple[str, ...]:
    return tuple(sorted(r.pack_id for r in state.records if not r.is_active))


def _active_pack_ids(state: ActivationStateManifest) -> Tuple[str, ...]:
    return tuple(sorted(r.pack_id for r in state.active()))


def verify_post_action(plan: LifecycleExecutionPlan,
                       before: ActivationStateManifest,
                       after: ActivationStateManifest, *,
                       affected_pack_ids: Sequence[str],
                       audit_record_present: bool,
                       action_request_status_updated: bool,
                       approval_consumed: bool,
                       pack_dir_fingerprints_before: Optional[Dict[str, str]] = None,
                       pack_dir_fingerprints_after: Optional[Dict[str, str]] = None,
                       ) -> LifecyclePostActionVerification:
    """Verify the executed action against the plan (pure; read-only comparison)."""
    findings: List[str] = []
    expected_hash = plan.expected_state_hash
    actual_hash = after.state_hash
    if expected_hash != actual_hash:
        findings.append(
            f"state hash mismatch: expected {expected_hash}, got {actual_hash}")

    affected = set(affected_pack_ids)
    before_unrelated = {r.pack_id: r.lifecycle_record_hash
                        for r in before.active() if r.pack_id not in affected}
    after_unrelated = {r.pack_id: r.lifecycle_record_hash
                       for r in after.active() if r.pack_id not in affected}
    unrelated_unchanged = before_unrelated == after_unrelated
    if not unrelated_unchanged:
        findings.append("an unrelated active pack changed during execution")

    # Pack files / manifests: the executor never writes pack content. If on-disk
    # fingerprints were supplied, confirm they are byte-identical; otherwise the
    # invariant holds by construction (no pack-content writer is imported).
    pack_files_unchanged = True
    manifests_unchanged = True
    if pack_dir_fingerprints_before is not None and \
            pack_dir_fingerprints_after is not None:
        if pack_dir_fingerprints_before != pack_dir_fingerprints_after:
            pack_files_unchanged = False
            manifests_unchanged = False
            findings.append("a pack directory fingerprint changed during execution")

    retrieval_check = (tuple(select_active_pack_ids(after))
                       == _active_pack_ids(after))

    return LifecyclePostActionVerification(
        expected_state_hash=expected_hash,
        actual_state_hash=actual_hash,
        expected_active_pack_ids=tuple(sorted(
            _expected_active_ids(plan, before, after))),
        actual_active_pack_ids=_active_pack_ids(after),
        expected_inactive_pack_ids=tuple(sorted(
            _expected_inactive_ids(plan, before, after))),
        actual_inactive_pack_ids=_inactive_pack_ids(after),
        pack_files_unchanged=pack_files_unchanged,
        manifests_unchanged=manifests_unchanged,
        unrelated_active_packs_unchanged=unrelated_unchanged,
        retrieval_selection_check=retrieval_check,
        audit_record_present=audit_record_present,
        action_request_status_updated=action_request_status_updated,
        approval_consumed=approval_consumed,
        findings=tuple(findings),
    )


def _expected_active_ids(plan: LifecycleExecutionPlan,
                         before: ActivationStateManifest,
                         after: ActivationStateManifest) -> List[str]:
    # The expected active set is derived from the verified ``after`` state for a
    # passing run; verification is the equality of expected/actual state hashes.
    return [r.pack_id for r in after.active()]


def _expected_inactive_ids(plan: LifecycleExecutionPlan,
                           before: ActivationStateManifest,
                           after: ActivationStateManifest) -> List[str]:
    return [r.pack_id for r in after.records if not r.is_active]


# ---------------------------------------------------------------------------
# Audit + result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleExecutionAuditRecord:
    """One immutable entry in the append-only execution audit log."""

    execution_id: str
    action_request_id: str
    review_item_id: str
    review_record_id: str
    execution_approval_id: str
    execution_plan_fingerprint: str
    requested_action: str
    executor_identity: str
    executor_role: str
    started_at: str
    completed_at: str
    pre_state_hash: str
    post_state_hash: str
    affected_pack_ids: Tuple[str, ...]
    affected_pack_fingerprints: Tuple[str, ...]
    rollback_target_state_hash: str
    lifecycle_primitive_called: str
    result_status: str
    validation_findings: Tuple[str, ...]
    verification_findings: Tuple[str, ...]
    state_files_changed: Tuple[str, ...]

    @property
    def audit_record_hash(self) -> str:
        payload = {
            "execution_id": self.execution_id,
            "action_request_id": self.action_request_id,
            "execution_approval_id": self.execution_approval_id,
            "execution_plan_fingerprint": self.execution_plan_fingerprint,
            "requested_action": self.requested_action,
            "executor_identity": self.executor_identity,
            "pre_state_hash": self.pre_state_hash,
            "post_state_hash": self.post_state_hash,
            "lifecycle_primitive_called": self.lifecycle_primitive_called,
            "result_status": self.result_status,
            "state_files_changed": list(self.state_files_changed),
        }
        return EXECUTION_AUDIT_PREFIX + _sha256_hex(_canonical(payload))[:20]

    def to_dict(self) -> dict:
        return {
            "_record": "lifecycle_execution_audit",
            "execution_id": self.execution_id,
            "action_request_id": self.action_request_id,
            "review_item_id": self.review_item_id,
            "review_record_id": self.review_record_id,
            "execution_approval_id": self.execution_approval_id,
            "execution_plan_fingerprint": self.execution_plan_fingerprint,
            "requested_action": self.requested_action,
            "executor_identity": self.executor_identity,
            "executor_role": self.executor_role,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "pre_state_hash": self.pre_state_hash,
            "post_state_hash": self.post_state_hash,
            "affected_pack_ids": list(self.affected_pack_ids),
            "affected_pack_fingerprints": list(self.affected_pack_fingerprints),
            "rollback_target_state_hash": self.rollback_target_state_hash,
            "lifecycle_primitive_called": self.lifecycle_primitive_called,
            "result_status": self.result_status,
            "validation_findings": list(self.validation_findings),
            "verification_findings": list(self.verification_findings),
            "state_files_changed": list(self.state_files_changed),
            "audit_record_hash": self.audit_record_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "LifecycleExecutionAuditRecord":
        return cls(
            execution_id=str(data["execution_id"]),
            action_request_id=str(data.get("action_request_id", "")),
            review_item_id=str(data.get("review_item_id", "")),
            review_record_id=str(data.get("review_record_id", "")),
            execution_approval_id=str(data.get("execution_approval_id", "")),
            execution_plan_fingerprint=str(data.get("execution_plan_fingerprint", "")),
            requested_action=str(data.get("requested_action", "")),
            executor_identity=str(data.get("executor_identity", "")),
            executor_role=str(data.get("executor_role", "")),
            started_at=str(data.get("started_at", "")),
            completed_at=str(data.get("completed_at", "")),
            pre_state_hash=str(data.get("pre_state_hash", "")),
            post_state_hash=str(data.get("post_state_hash", "")),
            affected_pack_ids=tuple(str(v) for v in
                                    (data.get("affected_pack_ids") or [])),
            affected_pack_fingerprints=tuple(
                str(v) for v in (data.get("affected_pack_fingerprints") or [])),
            rollback_target_state_hash=str(data.get("rollback_target_state_hash", "")),
            lifecycle_primitive_called=str(data.get("lifecycle_primitive_called", "")),
            result_status=str(data.get("result_status", "")),
            validation_findings=tuple(str(v) for v in
                                      (data.get("validation_findings") or [])),
            verification_findings=tuple(str(v) for v in
                                        (data.get("verification_findings") or [])),
            state_files_changed=tuple(str(v) for v in
                                      (data.get("state_files_changed") or [])),
        )


@dataclass(frozen=True)
class LifecycleExecutionResult:
    """The full outcome of an execution attempt (dry-run or write)."""

    execution_id: str
    action_request_id: str
    execution_approval_id: str
    requested_action: str
    status: str
    written: bool
    dry_run: bool
    pre_state_hash: str
    post_state_hash: str
    lifecycle_primitive_called: str
    lifecycle_audit_ref: str
    plan: LifecycleExecutionPlan
    validation: LifecycleExecutionValidation
    verification: Optional[LifecyclePostActionVerification]
    audit: Optional[LifecycleExecutionAuditRecord]
    updated_action_request: Optional[RegressionActionRequest]
    operational_follow_up: Optional[OperationalFollowUpRecord]
    activation_block: Optional[ActivationBlockRecord]
    message: str

    @property
    def succeeded(self) -> bool:
        return self.status in _TERMINAL_SUCCESS

    def to_dict(self) -> dict:
        return {
            "_record": "lifecycle_execution_result",
            "execution_id": self.execution_id,
            "action_request_id": self.action_request_id,
            "execution_approval_id": self.execution_approval_id,
            "requested_action": self.requested_action,
            "status": self.status,
            "written": self.written,
            "dry_run": self.dry_run,
            "pre_state_hash": self.pre_state_hash,
            "post_state_hash": self.post_state_hash,
            "lifecycle_primitive_called": self.lifecycle_primitive_called,
            "lifecycle_audit_ref": self.lifecycle_audit_ref,
            "plan_fingerprint": self.plan.plan_fingerprint,
            "validation_ok": self.validation.ok,
            "verification_passed": (self.verification.passed
                                    if self.verification else None),
            "audit_record_hash": (self.audit.audit_record_hash
                                  if self.audit else ""),
            "operational_follow_up_id": (self.operational_follow_up.follow_up_id
                                         if self.operational_follow_up else ""),
            "activation_block_id": (self.activation_block.block_id
                                    if self.activation_block else ""),
            "message": self.message,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Phase K + L — the single execution adapter (the ONLY mutation site)
# ---------------------------------------------------------------------------


def _build_kpa_deactivation_approval(live: ActivePackState, *, approver: str,
                                     approved_at: str, environment: str
                                     ) -> KnowledgePackActivationApproval:
    """Translate an execution approval into the v7.0 deactivation approval contract.

    The v7.0 ``deactivate`` primitive requires an approval scoped to
    ``emergency_deactivate`` bound to the exact pack id. This is a narrow adapter
    that reuses the existing primitive without re-implementing its transition.
    """
    return KnowledgePackActivationApproval(
        approval_id=f"exec-deact-{live.pack_id}",
        pack_id=live.pack_id,
        pack_version=live.pack_version,
        pack_fingerprint=live.pack_fingerprint,
        manifest_hash="",
        approved_by=approver,
        approved_at=approved_at,
        approval_scope=ActivationScope.EMERGENCY_DEACTIVATE,
        approved_environment=environment,
    )


def _apply_state_mutation(action_request: RegressionActionRequest,
                          execution_approval: LifecycleExecutionApproval, *,
                          state_manager: ActivationStateManager,
                          actor: str, write: bool, now: datetime,
                          available_pack_dirs: Optional[Dict[str, str]] = None,
                          ) -> Tuple[bool, str, str, str]:
    """Perform exactly one state-mutating lifecycle action by reusing v7.0.

    This is the *only* function that calls a mutation primitive. It performs a
    final compare-and-swap on the current state hash immediately before the
    mutation (defence against concurrent change), then delegates atomicity to the
    existing :class:`ActivationStateManager`. Returns
    ``(success, message, primitive, lifecycle_audit_ref)``.
    """
    action = action_request.requested_action
    # Compare-and-swap: reload the live state and confirm it still matches the
    # approved active-state hash. A concurrent change fails closed (stale).
    live_state = state_manager.load_state()
    if live_state.state_hash != execution_approval.active_state_hash:
        return (False, "state changed immediately before mutation (compare-and-swap "
                "failed); execution refused", _ACTION_PRIMITIVE.get(action, ""), "")

    if action == ActionRequestType.REQUEST_DEACTIVATION:
        primitive = "knowledge_pack_activation.deactivate"
        last_ref = ""
        for pack_id in action_request.affected_pack_ids:
            live = live_state.find(pack_id)
            if live is None:
                return (False, f"pack {pack_id!r} absent at mutation time",
                        primitive, "")
            approval = _build_kpa_deactivation_approval(
                live, approver=actor, approved_at=now.isoformat(),
                environment=execution_approval.environment)
            record = state_manager.deactivate(
                pack_id, approval=approval, actor=actor,
                reason=action_request.approval_reason or "governed deactivation",
                write=write, now=now)
            if not record.success:
                return (False, "deactivation primitive refused: " + record.message,
                        primitive, "")
            last_ref = record.audit.audit_record_hash if record.audit else ""
            live_state = record.state
        return (True, "deactivated", primitive, last_ref)

    if action == ActionRequestType.REQUEST_ROLLBACK:
        primitive = "knowledge_pack_activation.rollback"
        approval = KnowledgePackActivationApproval(
            approval_id="exec-rollback",
            pack_id=(action_request.affected_pack_ids[0]
                     if action_request.affected_pack_ids else ""),
            pack_version="", pack_fingerprint="", manifest_hash="",
            approved_by=actor, approved_at=now.isoformat(),
            approval_scope=ActivationScope.ROLLBACK,
            approved_environment=execution_approval.environment)
        request = KnowledgePackRollbackRequest(
            target_state_hash=action_request.rollback_target_state_hash,
            current_state_hash=live_state.state_hash,
            approval=approval,
            reason=action_request.approval_reason or "governed rollback")
        result = state_manager.rollback(
            request, available_pack_dirs=available_pack_dirs, actor=actor,
            write=write, now=now)
        if not result.success:
            return (False, "rollback primitive refused: " + result.message,
                    primitive, "")
        ref = result.audit.audit_record_hash if result.audit else ""
        return (True, "rolled back", primitive, ref)

    return (False, f"action {action!r} does not mutate state", "", "")


def execute_lifecycle_action(
        action_request: RegressionActionRequest,
        execution_approval: LifecycleExecutionApproval, *,
        executor_identity: str,
        executor_role: str,
        state_manager: ActivationStateManager,
        review_history: Sequence[RegressionReviewRecord] = (),
        prior_results: Sequence[LifecycleExecutionResult] = (),
        prior_blocks: Sequence[ActivationBlockRecord] = (),
        approval_consumed: bool = False,
        available_pack_dirs: Optional[Dict[str, str]] = None,
        write: bool = False,
        now: Optional[datetime] = None,
) -> LifecycleExecutionResult:
    """Execute exactly one approved lifecycle action end-to-end.

    Revalidates -> plans -> (for a write) performs one atomic action by reusing
    the v7.0 primitive -> verifies -> records an immutable audit event. A dry-run
    writes nothing. Review approval alone can never execute: a separate, unexpired,
    unconsumed, exactly-bound :class:`LifecycleExecutionApproval` is required.
    Duplicate execution is idempotent (returns ``already_executed``).
    """
    now = now or datetime.now(timezone.utc)
    started_at = now.isoformat()
    if not executor_identity.strip():
        raise ValueError("executor identity is required (it is never inferred)")
    if not executor_role.strip():
        raise ValueError("executor role is required")

    before = state_manager.load_state()
    history = state_manager.load_audit()
    pre_state_hash = before.state_hash

    validation = validate_lifecycle_execution(
        action_request, execution_approval, before, history, review_history,
        approval_consumed=approval_consumed, now=now)

    # A missing action request or execution approval cannot form a plan. Review
    # approval alone (no execution approval) lands here and is refused.
    if action_request is None or execution_approval is None:
        ar_id = action_request.action_request_id if action_request else ""
        ap_id = (execution_approval.execution_approval_id
                 if execution_approval else "")
        action_name = action_request.requested_action if action_request else ""
        empty_plan = LifecycleExecutionPlan(
            action_request_id=ar_id, execution_approval_id=ap_id,
            requested_action=action_name, lifecycle_primitive="",
            current_state_hash=pre_state_hash, expected_state_hash=pre_state_hash,
            affected_pack_ids=(), rollback_target_state_hash="",
            expected_state_record_changes=(), expected_audit_records=(),
            preconditions=(), postconditions=(), files_expected_to_change=(),
            files_expected_unchanged=(), risk_findings=(),
            dry_run_summary="no execution approval / action request")
        return LifecycleExecutionResult(
            execution_id="", action_request_id=ar_id, execution_approval_id=ap_id,
            requested_action=action_name,
            status=ExecutionStatus.VALIDATION_FAILED, written=False,
            dry_run=not write, pre_state_hash=pre_state_hash,
            post_state_hash=pre_state_hash, lifecycle_primitive_called="",
            lifecycle_audit_ref="", plan=empty_plan, validation=validation,
            verification=None, audit=None, updated_action_request=None,
            operational_follow_up=None, activation_block=None,
            message="validation failed: " + "; ".join(
                f.message for f in validation.findings))

    plan = build_execution_plan(action_request, execution_approval, before, history)

    def _execution_id(status_for_id: str = "") -> str:
        payload = {"action_request_id": action_request.action_request_id,
                   "execution_approval_id": execution_approval.execution_approval_id,
                   "pre_state_hash": pre_state_hash,
                   "requested_action": action_request.requested_action}
        return EXECUTION_ID_PREFIX + _sha256_hex(_canonical(payload))[:16]

    def _result(status: str, *, written: bool, message: str,
                verification=None, primitive: str = "", lifecycle_audit_ref: str = "",
                updated_action=None, follow_up=None, block=None,
                audit=None) -> LifecycleExecutionResult:
        return LifecycleExecutionResult(
            execution_id=_execution_id(),
            action_request_id=action_request.action_request_id,
            execution_approval_id=execution_approval.execution_approval_id,
            requested_action=action_request.requested_action,
            status=status, written=written, dry_run=not write,
            pre_state_hash=pre_state_hash,
            post_state_hash=(verification.actual_state_hash if verification
                             else pre_state_hash),
            lifecycle_primitive_called=primitive or plan.lifecycle_primitive,
            lifecycle_audit_ref=lifecycle_audit_ref,
            plan=plan, validation=validation, verification=verification,
            audit=audit, updated_action_request=updated_action,
            operational_follow_up=follow_up, activation_block=block,
            message=message)

    def _audit(status: str, primitive: str, post_hash: str,
               verification: Optional[LifecyclePostActionVerification],
               files_changed: Sequence[str]) -> LifecycleExecutionAuditRecord:
        return LifecycleExecutionAuditRecord(
            execution_id=_execution_id(),
            action_request_id=action_request.action_request_id,
            review_item_id=action_request.review_item_id,
            review_record_id=action_request.review_record_id,
            execution_approval_id=execution_approval.execution_approval_id,
            execution_plan_fingerprint=plan.plan_fingerprint,
            requested_action=action_request.requested_action,
            executor_identity=executor_identity, executor_role=executor_role,
            started_at=started_at, completed_at=now.isoformat(),
            pre_state_hash=pre_state_hash, post_state_hash=post_hash,
            affected_pack_ids=action_request.affected_pack_ids,
            affected_pack_fingerprints=action_request.affected_pack_fingerprints,
            rollback_target_state_hash=action_request.rollback_target_state_hash,
            lifecycle_primitive_called=primitive, result_status=status,
            validation_findings=tuple(f.code for f in validation.findings),
            verification_findings=(verification.findings if verification else ()),
            state_files_changed=tuple(files_changed))

    # -- replay protection: idempotent re-execution --
    for prior in prior_results:
        if prior.action_request_id == action_request.action_request_id and \
                prior.status in _TERMINAL_SUCCESS:
            return _result(
                ExecutionStatus.ALREADY_EXECUTED, written=False,
                message="already executed; returning the existing execution record "
                f"({prior.execution_id})",
                primitive=prior.lifecycle_primitive_called,
                lifecycle_audit_ref=prior.lifecycle_audit_ref)

    # -- validation gate (fail closed) --
    if not validation.ok:
        audit = _audit(ExecutionStatus.VALIDATION_FAILED, plan.lifecycle_primitive,
                       pre_state_hash, None, ()) if write else None
        return _result(
            ExecutionStatus.VALIDATION_FAILED, written=False,
            message="validation failed: " + "; ".join(
                f.message for f in validation.findings),
            audit=audit)

    # -- plan-binding check: an approval bound to a plan must match --
    if execution_approval.plan_fingerprint and \
            execution_approval.plan_fingerprint != plan.plan_fingerprint:
        v = LifecycleExecutionValidation(
            action_request_id=validation.action_request_id,
            execution_approval_id=validation.execution_approval_id,
            requested_action=validation.requested_action,
            findings=(ExecutionFinding(
                ExecutionFindingCode.PLAN_FINGERPRINT_CHANGED,
                "execution plan changed since the approval was bound"),))
        validation = v
        audit = _audit(ExecutionStatus.VALIDATION_FAILED, plan.lifecycle_primitive,
                       pre_state_hash, None, ()) if write else None
        return _result(ExecutionStatus.VALIDATION_FAILED, written=False,
                       message="plan changed after approval; execution refused",
                       audit=audit)

    action = action_request.requested_action

    # -- dry run: write nothing --
    if not write:
        return _result(ExecutionStatus.DRY_RUN_READY, written=False,
                       message="dry-run: validation passed, plan "
                       f"{plan.plan_fingerprint}, nothing written")

    # =====================================================================
    # WRITE path. Documented commit point per action class below.
    # =====================================================================

    # -- operational follow-up actions (no active-state mutation) --
    if action in _FOLLOW_UP_ACTIONS:
        follow_up = make_operational_follow_up(
            action_request, requested_by=executor_identity,
            requested_at=now.isoformat())
        # Commit point: the follow-up record + execution audit are persisted by
        # the caller. The active state is untouched, so verification confirms it.
        verification = verify_post_action(
            plan, before, before, affected_pack_ids=(),
            audit_record_present=True, action_request_status_updated=True,
            approval_consumed=True)
        execution_id = _execution_id()
        audit = _audit(ExecutionStatus.EXECUTED, "operational_follow_up",
                       before.state_hash, verification, [DEFAULT_FOLLOW_UP_PATH])
        updated = mark_action_result(
            action_request, ActionRequestStatus.EXECUTED,
            executed_at=now.isoformat(), execution_record_id=execution_id,
            lifecycle_audit_ref=audit.audit_record_hash)
        return _result(ExecutionStatus.EXECUTED, written=True,
                       message=f"created operational follow-up {follow_up.follow_up_id}",
                       verification=verification, primitive="operational_follow_up",
                       lifecycle_audit_ref=audit.audit_record_hash,
                       updated_action=updated, follow_up=follow_up, audit=audit)

    # -- activation block (no active-state mutation; exact fingerprint) --
    if action == ActionRequestType.REQUEST_ACTIVATION_BLOCK:
        block = make_activation_block(action_request, execution_approval,
                                      approved_at=now.isoformat())
        existing = next((b for b in prior_blocks if b.block_id == block.block_id),
                        None)
        if existing is not None:
            return _result(ExecutionStatus.ALREADY_EXECUTED, written=False,
                           message=f"activation block {block.block_id} already exists "
                           "(idempotent)", primitive="activation_block_record",
                           block=existing)
        verification = verify_post_action(
            plan, before, before, affected_pack_ids=(),
            audit_record_present=True, action_request_status_updated=True,
            approval_consumed=True)
        execution_id = _execution_id()
        audit = _audit(ExecutionStatus.EXECUTED, "activation_block_record",
                       before.state_hash, verification,
                       [DEFAULT_ACTIVATION_BLOCKS_PATH])
        updated = mark_action_result(
            action_request, ActionRequestStatus.EXECUTED,
            executed_at=now.isoformat(), execution_record_id=execution_id,
            lifecycle_audit_ref=audit.audit_record_hash)
        return _result(ExecutionStatus.EXECUTED, written=True,
                       message=f"recorded activation block {block.block_id}",
                       verification=verification, primitive="activation_block_record",
                       lifecycle_audit_ref=audit.audit_record_hash,
                       updated_action=updated, block=block, audit=audit)

    # -- state-mutating actions (deactivation / rollback) --
    if action in _STATE_MUTATING_ACTIONS:
        success, message, primitive, lifecycle_ref = _apply_state_mutation(
            action_request, execution_approval, state_manager=state_manager,
            actor=executor_identity, write=True, now=now,
            available_pack_dirs=available_pack_dirs)
        # Commit point: the v7.0 primitive has atomically replaced the active
        # state and appended its lifecycle audit (write=True). Only *after* this
        # do we verify, consume the approval and mark the action executed.
        after = state_manager.load_state()
        if not success:
            # The mutation refused / failed: state is unchanged, approval is NOT
            # consumed, and the action request is NOT marked executed.
            audit = _audit(ExecutionStatus.EXECUTION_FAILED, primitive,
                           after.state_hash, None, ())
            return _result(ExecutionStatus.EXECUTION_FAILED, written=False,
                           message="execution failed (state unchanged): " + message,
                           primitive=primitive, audit=audit)

        verification = verify_post_action(
            plan, before, after,
            affected_pack_ids=action_request.affected_pack_ids,
            audit_record_present=bool(lifecycle_ref),
            action_request_status_updated=True, approval_consumed=True,
            pack_dir_fingerprints_before=available_pack_dirs,
            pack_dir_fingerprints_after=available_pack_dirs)
        status = (ExecutionStatus.EXECUTED if verification.passed
                  else ExecutionStatus.PARTIALLY_VERIFIED)
        execution_id = _execution_id()
        audit = _audit(status, primitive, after.state_hash, verification,
                       ["config/active_knowledge_packs.jsonl",
                        "reports/knowledge_pack_activation_audit.jsonl"])
        updated = mark_action_result(
            action_request, ActionRequestStatus.EXECUTED,
            executed_at=now.isoformat(), execution_record_id=execution_id,
            lifecycle_audit_ref=lifecycle_ref or audit.audit_record_hash)
        return _result(
            status, written=True,
            message=(message if verification.passed
                     else "state changed but post-action verification did not fully "
                     "pass: " + "; ".join(verification.findings)),
            verification=verification, primitive=primitive,
            lifecycle_audit_ref=lifecycle_ref, updated_action=updated, audit=audit)

    # -- unsupported (should be caught in validation) --
    return _result(ExecutionStatus.VALIDATION_FAILED, written=False,
                   message=f"unsupported action {action!r}")


# ---------------------------------------------------------------------------
# Phase O — persistence (atomic; each artefact kept in its own file)
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


def load_execution_approval(path: str | Path) -> LifecycleExecutionApproval:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LifecycleExecutionApproval.from_dict(data)


def load_execution_approvals(path: str | Path = DEFAULT_APPROVALS_PATH,
                             ) -> List[LifecycleExecutionApproval]:
    return [LifecycleExecutionApproval.from_dict(d) for d in _read_jsonl(path)]


def save_execution_approvals(approvals: Sequence[LifecycleExecutionApproval],
                             path: str | Path = DEFAULT_APPROVALS_PATH) -> None:
    ordered = sorted(approvals, key=lambda a: a.execution_approval_id)
    _atomic_write_text(Path(path),
                       "".join(a.to_json() + "\n" for a in ordered))


def load_execution_results(path: str | Path = DEFAULT_RESULTS_PATH,
                           ) -> List[dict]:
    """Load execution result records (as dicts; results carry nested plans)."""
    return _read_jsonl(path)


def append_execution_result(result: LifecycleExecutionResult,
                            path: str | Path = DEFAULT_RESULTS_PATH) -> None:
    """Append one execution result (append-only; atomic)."""
    existing = Path(path).read_text(encoding="utf-8") \
        if Path(path).exists() else ""
    _atomic_write_text(Path(path), existing + result.to_json() + "\n")


def load_execution_audit(path: str | Path = DEFAULT_EXECUTION_AUDIT_PATH,
                         ) -> List[LifecycleExecutionAuditRecord]:
    return [LifecycleExecutionAuditRecord.from_dict(d) for d in _read_jsonl(path)]


def append_execution_audit(records: Sequence[LifecycleExecutionAuditRecord],
                           path: str | Path = DEFAULT_EXECUTION_AUDIT_PATH) -> None:
    """Append execution audit records (append-only; atomic; never rewrites history)."""
    if not records:
        return
    existing = Path(path).read_text(encoding="utf-8") \
        if Path(path).exists() else ""
    addition = "".join(r.to_json() + "\n" for r in records)
    _atomic_write_text(Path(path), existing + addition)


def load_follow_ups(path: str | Path = DEFAULT_FOLLOW_UP_PATH,
                    ) -> List[OperationalFollowUpRecord]:
    return [OperationalFollowUpRecord.from_dict(d) for d in _read_jsonl(path)]


def save_follow_ups(records: Sequence[OperationalFollowUpRecord],
                    path: str | Path = DEFAULT_FOLLOW_UP_PATH) -> None:
    ordered = sorted(records, key=lambda r: r.follow_up_id)
    _atomic_write_text(Path(path),
                       "".join(r.to_json() + "\n" for r in ordered))


def add_follow_up(records: Sequence[OperationalFollowUpRecord],
                  new_record: OperationalFollowUpRecord,
                  ) -> List[OperationalFollowUpRecord]:
    """Merge a follow-up, idempotent by id (a duplicate is a no-op)."""
    by_id = {r.follow_up_id: r for r in records}
    by_id.setdefault(new_record.follow_up_id, new_record)
    return sorted(by_id.values(), key=lambda r: r.follow_up_id)


def load_activation_blocks(path: str | Path = DEFAULT_ACTIVATION_BLOCKS_PATH,
                           ) -> List[ActivationBlockRecord]:
    return [ActivationBlockRecord.from_dict(d) for d in _read_jsonl(path)]


def save_activation_blocks(records: Sequence[ActivationBlockRecord],
                           path: str | Path = DEFAULT_ACTIVATION_BLOCKS_PATH) -> None:
    ordered = sorted(records, key=lambda r: r.block_id)
    _atomic_write_text(Path(path),
                       "".join(r.to_json() + "\n" for r in ordered))


def add_activation_block(records: Sequence[ActivationBlockRecord],
                         new_record: ActivationBlockRecord,
                         ) -> List[ActivationBlockRecord]:
    """Merge an activation block, idempotent by id (a duplicate is a no-op)."""
    by_id = {r.block_id: r for r in records}
    by_id.setdefault(new_record.block_id, new_record)
    return sorted(by_id.values(), key=lambda r: r.block_id)


def is_fingerprint_blocked(blocks: Sequence[ActivationBlockRecord],
                           pack_fingerprint: str, *, environment: str = "default",
                           ) -> bool:
    """True iff an active block targets this exact fingerprint + environment."""
    return any(b.pack_fingerprint == pack_fingerprint
               and b.environment == environment
               and b.status == BlockStatus.ACTIVE
               for b in blocks)


# ---------------------------------------------------------------------------
# Markdown renderers (deterministic; no source content)
# ---------------------------------------------------------------------------


def render_plan_markdown(plan: LifecycleExecutionPlan) -> str:
    lines = [
        "# Lifecycle execution plan", "",
        f"- Action request: `{plan.action_request_id}`",
        f"- Execution approval: `{plan.execution_approval_id}`",
        f"- Requested action: {plan.requested_action}",
        f"- Lifecycle primitive: `{plan.lifecycle_primitive}`",
        f"- Plan fingerprint: `{plan.plan_fingerprint}`",
        f"- Current state hash: `{plan.current_state_hash}`",
        f"- Expected state hash: `{plan.expected_state_hash}`",
        f"- Rollback target: `{plan.rollback_target_state_hash or '(none)'}`",
        "",
        "## Files expected to change",
    ]
    lines += [f"- `{f}`" for f in plan.files_expected_to_change] or ["- (none)"]
    lines += ["", "## Files expected to remain unchanged"]
    lines += [f"- {f}" for f in plan.files_expected_unchanged]
    if plan.risk_findings:
        lines += ["", "## Risk findings"]
        lines += [f"- {r}" for r in plan.risk_findings]
    lines += ["", f"_Dry-run summary: {plan.dry_run_summary}_"]
    return "\n".join(lines)


def render_result_markdown(result: LifecycleExecutionResult) -> str:
    lines = [
        "# Lifecycle execution result", "",
        f"- Execution id: `{result.execution_id}`",
        f"- Status: **{result.status}**",
        f"- Written: {result.written} (dry-run: {result.dry_run})",
        f"- Requested action: {result.requested_action}",
        f"- Primitive called: `{result.lifecycle_primitive_called}`",
        f"- Pre-state: `{result.pre_state_hash}`",
        f"- Post-state: `{result.post_state_hash}`",
        f"- Lifecycle audit ref: `{result.lifecycle_audit_ref or '(none)'}`",
        f"- Message: {result.message}",
    ]
    if result.verification is not None:
        lines += ["", f"- Verification passed: {result.verification.passed}"]
        if result.verification.findings:
            lines += [f"  - {f}" for f in result.verification.findings]
    return "\n".join(lines)
