"""Shared fixtures for the v7.3 governed lifecycle action executor tests.

Reuses the real v7.2 regression-review pipeline to mint genuinely approved
:class:`RegressionActionRequest` objects, and the real v7.0
:class:`ActivationStateManager` so executions exercise the actual lifecycle
primitives (no duplicated transition logic).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.knowledge_pack_activation as kpa  # noqa: E402
import agent.lifecycle_action_executor as lae  # noqa: E402
import agent.regression_review_queue as rrq  # noqa: E402
from _regression_review_helpers import FIXED_NOW, manifest, run_dict  # noqa: E402

FIXED_DT = datetime(2024, 1, 1, tzinfo=timezone.utc)

_REC = rrq.MonitoringRecommendationCode
_D = rrq.ReviewDecision
_ROLE = rrq.ReviewerRole


def two_pack_manifest():
    """Two active packs p1 + p2 (so we can deactivate one and keep one)."""
    return manifest(packs=(("p1", "1.0", "packfp-a", "src1", "r1"),
                           ("p2", "2.0", "packfp-b", "src2", "r2")))


def deactivate_p2(mf):
    """A copy of ``mf`` with p2 inactive (a later governed state)."""
    p2 = mf.find("p2")
    return mf.upsert(p2.with_status(kpa.PackLifecycleState.INACTIVE,
                                    activated_at=p2.activated_at))


def approved_action_request(*, recommendation=_REC.DEACTIVATE_RECOMMENDED,
                            mf=None, candidate_pack_ids=("p1",),
                            rollback_target=""):
    """Mint a genuinely approved action request via the v7.2 pipeline."""
    mf = mf if mf is not None else manifest()
    rd = run_dict(manifest_obj=mf, recommendation=recommendation,
                  candidate_pack_ids=candidate_pack_ids,
                  rollback_target=rollback_target)
    item = rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW)
    validation = rrq.validate_review_item(item, current_state=mf, now=FIXED_NOW)
    out = rrq.review_item(item, _D.APPROVE, reviewer="gov",
                          role=_ROLE.GOVERNANCE_APPROVER,
                          reason="confirmed regression",
                          validation=validation, reviewed_at=FIXED_NOW)
    return out.action_request, out.record, mf


def write_state(path: Path, mf) -> None:
    """Seed an activation state file the manager can load."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), sort_keys=True, separators=(",", ":"))
             for r in sorted(mf.records, key=lambda r: r.pack_id)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_audit(path: Path, records) -> None:
    """Seed an append-only activation audit file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), sort_keys=True, separators=(",", ":"))
             for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def activate_audit_record(mf):
    """A synthetic successful ACTIVATE audit record that reconstructs ``mf``."""
    return kpa.ActivationAuditRecord(
        action=kpa.AuditAction.ACTIVATE, actor="seed", timestamp=FIXED_NOW,
        previous_state_hash="pkgstate-seed", requested_state="activate",
        resulting_state_hash=mf.state_hash, approval_id="seed-ap",
        pack_fingerprint="", evaluation_fingerprint="", reason="seed",
        success=True, failure_findings=(), resulting_state_records=tuple(
            mf.to_records()))


def manager(tmp_path: Path, mf, audit_records=()):
    """A real ActivationStateManager seeded with ``mf`` and optional audit."""
    state_path = tmp_path / "config" / "active_knowledge_packs.jsonl"
    audit_path = tmp_path / "reports" / "knowledge_pack_activation_audit.jsonl"
    write_state(state_path, mf)
    if audit_records:
        write_audit(audit_path, audit_records)
    return kpa.ActivationStateManager(state_path=state_path, audit_path=audit_path)


def execution_approval(action_request, *, approved_by="lifecycle-op",
                       approved_role=_ROLE.LIFECYCLE_OPERATOR,
                       environment="default", expires_at="",
                       single_use=True, plan_fingerprint=""):
    """Mint an execution approval bound to ``action_request``."""
    return lae.build_execution_approval(
        action_request, approved_by=approved_by, approved_role=approved_role,
        approved_at=FIXED_NOW, environment=environment, expires_at=expires_at,
        single_use=single_use, plan_fingerprint=plan_fingerprint,
        approval_reason="execution authorised")
