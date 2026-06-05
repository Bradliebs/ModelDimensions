"""Governed HF revision refresh & retirement/supersession — Phases G/H/I (v6.9).

Once a dataset revision has been approved and imported, two things change over
time: the *upstream* dataset moves on (a new revision, drifted metadata, a
changed licence), and the *local* pack eventually needs retiring or replacing.
This module governs both — as **decisions and plans**, never as actions.

Governance honoured here, by construction:

* **A revision approval covers exactly one revision.** Approval for revision R1
  is not approval for R2. ``assess_revision`` compares an existing approval
  against freshly observed metadata and, on any revision change, returns
  ``REQUIRE_REAPPROVAL`` — it can never declare a new revision "covered". A
  refresh therefore re-triggers approval; it never silently re-imports.
* **Metadata drift on the same revision is suspicious, not waved through.** If
  the dataset id and revision match but the recomputed metadata fingerprint
  differs (licence, schema, gating, or personal-data flags moved), the existing
  approval no longer matches what it approved, so this too requires re-approval.
* **Retirement and supersession are plans, not executions.** ``plan_retirement``
  and ``plan_supersession`` describe what *should* happen — which pack to
  deactivate, which revision supersedes it, which fresh approval is required —
  and produce an inert plan record. They deactivate nothing, mutate no source
  registry, touch no retrieval, and import no writer. Carrying out a plan is a
  separate, explicitly governed step outside this module.
* **Read-only and offline.** Every function here is a pure comparison over data
  the caller already holds. No network, no service, no ledger, no importer. The
  docstring is explicit so the import-purity test can strip it before scanning.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from agent.data_intake import DataIntakeAssessment
from agent.hf_data_adapter import HuggingFaceDatasetMetadata
from agent.hf_import_lifecycle import (
    HFDatasetApproval,
    compute_metadata_fingerprint,
)

REVISION_MANAGER_VERSION = "hf-revision-v6.9"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(now: Optional[datetime]) -> str:
    return (now or _utc_now()).isoformat()


# --------------------------------------------------------------------------- #
# Revision refresh (Phase G)
# --------------------------------------------------------------------------- #


class HFRevisionStatus(str, Enum):
    """The relationship between an approval and freshly observed metadata."""

    UP_TO_DATE = "up_to_date"            # same revision and fingerprint
    REVISION_CHANGED = "revision_changed"  # upstream moved to a new revision
    METADATA_DRIFT = "metadata_drift"    # same revision, fingerprint differs
    DATASET_MISMATCH = "dataset_mismatch"  # observed a different dataset entirely


class HFRefreshAction(str, Enum):
    """What the operator must do in response to a revision assessment."""

    NO_ACTION = "no_action"                # approval still matches; nothing to do
    REQUIRE_REAPPROVAL = "require_reapproval"  # a fresh approval is needed
    BLOCK = "block"                        # do not import under any approval


class HFRevisionFindingCode(str, Enum):
    """Specific reasons a refresh needs attention."""

    REVISION_ADVANCED = "revision_advanced"
    METADATA_FINGERPRINT_CHANGED = "metadata_fingerprint_changed"
    LICENCE_CHANGED = "licence_changed"
    LICENCE_NOW_UNKNOWN = "licence_now_unknown"
    PROVENANCE_CHANGED = "provenance_changed"
    NOW_GATED = "now_gated"
    NOW_PRIVATE = "now_private"
    NOW_DECLARES_PERSONAL_DATA = "now_declares_personal_data"
    DATASET_ID_MISMATCH = "dataset_id_mismatch"


@dataclass(frozen=True)
class HFRevisionFinding:
    """One reason the observed dataset differs from what was approved."""

    code: HFRevisionFindingCode
    message: str
    requires_reapproval: bool
    blocking: bool = False

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "requires_reapproval": self.requires_reapproval,
            "blocking": self.blocking,
        }


@dataclass(frozen=True)
class HFRevisionAssessment:
    """A decision about whether an existing approval still covers a dataset."""

    dataset_id: str
    approved_revision: str
    observed_revision: str
    approved_metadata_fingerprint: str
    observed_metadata_fingerprint: str
    status: HFRevisionStatus
    action: HFRefreshAction
    covered_by_existing_approval: bool
    findings: Tuple[HFRevisionFinding, ...]
    assessed_at: str
    manager_version: str = REVISION_MANAGER_VERSION

    @property
    def requires_reapproval(self) -> bool:
        return self.action is HFRefreshAction.REQUIRE_REAPPROVAL

    @property
    def blocked(self) -> bool:
        return self.action is HFRefreshAction.BLOCK

    def to_dict(self) -> dict:
        return {
            "_record": "hf_revision_assessment",
            "dataset_id": self.dataset_id,
            "approved_revision": self.approved_revision,
            "observed_revision": self.observed_revision,
            "approved_metadata_fingerprint": self.approved_metadata_fingerprint,
            "observed_metadata_fingerprint": self.observed_metadata_fingerprint,
            "status": self.status.value,
            "action": self.action.value,
            "covered_by_existing_approval": self.covered_by_existing_approval,
            "requires_reapproval": self.requires_reapproval,
            "blocked": self.blocked,
            "findings": [f.to_dict() for f in self.findings],
            "assessed_at": self.assessed_at,
            "manager_version": self.manager_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def assess_revision(
    approval: HFDatasetApproval, *,
    observed_metadata: HuggingFaceDatasetMetadata,
    observed_revision: Optional[str] = None,
    observed_assessment: Optional[DataIntakeAssessment] = None,
    now: Optional[datetime] = None,
) -> HFRevisionAssessment:
    """Decide whether ``approval`` still covers the freshly observed metadata.

    ``observed_revision`` is the dataset revision (git ref or commit) the caller
    actually fetched the metadata at; it defaults to the approval's own revision,
    meaning "same revision, re-checking for metadata drift". Fail-closed: any
    revision change, fingerprint drift, dataset mismatch, licence loss, or new
    gating/personal-data flag means the existing approval no longer applies and
    a fresh approval is required (or the import is blocked). Only an exact
    dataset + revision + fingerprint match is covered.
    """
    assessed_at = _iso(now)
    observed_fp = compute_metadata_fingerprint(
        observed_metadata, assessment=observed_assessment)
    findings: List[HFRevisionFinding] = []

    # Dataset identity must match before anything else is meaningful.
    if observed_metadata.dataset_id != approval.dataset_id:
        findings.append(HFRevisionFinding(
            code=HFRevisionFindingCode.DATASET_ID_MISMATCH,
            message=(f"observed dataset {observed_metadata.dataset_id!r} is not "
                     f"the approved dataset {approval.dataset_id!r}"),
            requires_reapproval=True, blocking=True))
        return HFRevisionAssessment(
            dataset_id=approval.dataset_id,
            approved_revision=approval.dataset_revision,
            observed_revision=observed_revision or approval.dataset_revision,
            approved_metadata_fingerprint=approval.metadata_fingerprint,
            observed_metadata_fingerprint=observed_fp,
            status=HFRevisionStatus.DATASET_MISMATCH,
            action=HFRefreshAction.BLOCK,
            covered_by_existing_approval=False,
            findings=tuple(findings), assessed_at=assessed_at)

    resolved_revision = observed_revision or approval.dataset_revision

    # Licence / provenance / risk-flag drift (reported regardless of revision).
    approved_licence = (approval.licence_snapshot or "").strip().lower()
    observed_licence = (observed_metadata.license or "").strip().lower()
    if approved_licence and observed_licence and observed_licence != approved_licence:
        findings.append(HFRevisionFinding(
            code=HFRevisionFindingCode.LICENCE_CHANGED,
            message=(f"licence changed from {approved_licence!r} to "
                     f"{observed_licence!r}"),
            requires_reapproval=True))
    if approved_licence and not observed_licence:
        findings.append(HFRevisionFinding(
            code=HFRevisionFindingCode.LICENCE_NOW_UNKNOWN,
            message="licence is no longer declared upstream",
            requires_reapproval=True, blocking=True))
    if observed_metadata.gated:
        findings.append(HFRevisionFinding(
            code=HFRevisionFindingCode.NOW_GATED,
            message="dataset is now gated upstream",
            requires_reapproval=True))
    if observed_metadata.private:
        findings.append(HFRevisionFinding(
            code=HFRevisionFindingCode.NOW_PRIVATE,
            message="dataset is now private upstream",
            requires_reapproval=True, blocking=True))
    if observed_metadata.contains_personal_data:
        findings.append(HFRevisionFinding(
            code=HFRevisionFindingCode.NOW_DECLARES_PERSONAL_DATA,
            message="dataset now declares personal data",
            requires_reapproval=True))

    # Revision / fingerprint comparison drives the headline status.
    revision_changed = resolved_revision != approval.dataset_revision
    fingerprint_changed = observed_fp != approval.metadata_fingerprint

    if revision_changed:
        findings.append(HFRevisionFinding(
            code=HFRevisionFindingCode.REVISION_ADVANCED,
            message=(f"revision advanced from {approval.dataset_revision!r} to "
                     f"{resolved_revision!r}; the existing approval does not "
                     "cover it"),
            requires_reapproval=True))
        status = HFRevisionStatus.REVISION_CHANGED
    elif fingerprint_changed:
        findings.append(HFRevisionFinding(
            code=HFRevisionFindingCode.METADATA_FINGERPRINT_CHANGED,
            message=("metadata fingerprint changed on the same revision; the "
                     "approval no longer matches the dataset"),
            requires_reapproval=True))
        status = HFRevisionStatus.METADATA_DRIFT
    else:
        status = HFRevisionStatus.UP_TO_DATE

    blocking = any(f.blocking for f in findings)
    needs_reapproval = any(f.requires_reapproval for f in findings)
    if blocking:
        action = HFRefreshAction.BLOCK
    elif needs_reapproval:
        action = HFRefreshAction.REQUIRE_REAPPROVAL
    else:
        action = HFRefreshAction.NO_ACTION
    covered = (status is HFRevisionStatus.UP_TO_DATE
               and action is HFRefreshAction.NO_ACTION)

    return HFRevisionAssessment(
        dataset_id=approval.dataset_id,
        approved_revision=approval.dataset_revision,
        observed_revision=resolved_revision,
        approved_metadata_fingerprint=approval.metadata_fingerprint,
        observed_metadata_fingerprint=observed_fp,
        status=status, action=action,
        covered_by_existing_approval=covered,
        findings=tuple(findings), assessed_at=assessed_at)


# --------------------------------------------------------------------------- #
# Retirement / supersession (Phases H / I) — plans only, never executed
# --------------------------------------------------------------------------- #


class HFRetirementReason(str, Enum):
    """Why an imported pack is being retired or replaced."""

    SUPERSEDED_BY_REVISION = "superseded_by_revision"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_REVOKED = "approval_revoked"
    SOURCE_REMOVED_UPSTREAM = "source_removed_upstream"
    POLICY_CHANGE = "policy_change"


@dataclass(frozen=True)
class HFRetirementPlan:
    """An inert plan to retire (or supersede) an imported pack.

    A plan records *what should happen* and is never self-executing: ``executed``
    is always ``False``. Carrying it out — deactivating a pack, registering a
    superseding one — is a separate, explicitly governed step this module never
    performs.
    """

    pack_id: str
    dataset_id: str
    dataset_revision: str
    reason: HFRetirementReason
    superseded_by_pack_id: str
    superseded_by_revision: str
    requires_approval_id: str
    planned_actions: Tuple[str, ...]
    rationale: str
    executed: bool
    planned_at: str
    manager_version: str = REVISION_MANAGER_VERSION

    @property
    def is_supersession(self) -> bool:
        return self.reason is HFRetirementReason.SUPERSEDED_BY_REVISION

    def to_dict(self) -> dict:
        return {
            "_record": "hf_retirement_plan",
            "pack_id": self.pack_id,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "reason": self.reason.value,
            "superseded_by_pack_id": self.superseded_by_pack_id,
            "superseded_by_revision": self.superseded_by_revision,
            "requires_approval_id": self.requires_approval_id,
            "planned_actions": list(self.planned_actions),
            "rationale": self.rationale,
            "is_supersession": self.is_supersession,
            "executed": self.executed,
            "planned_at": self.planned_at,
            "manager_version": self.manager_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def plan_retirement(
    pack_id: str, *,
    dataset_id: str,
    dataset_revision: str,
    reason: HFRetirementReason,
    rationale: str = "",
    requires_approval_id: str = "",
    now: Optional[datetime] = None,
) -> HFRetirementPlan:
    """Build an inert retirement plan for an imported pack (nothing executed)."""
    actions = [
        f"Mark pack {pack_id!r} inactive in its on-disk manifest.",
        "Confirm the pack is not referenced by an active retrieval source.",
        "Record the retirement decision in the human governance log.",
    ]
    if requires_approval_id:
        actions.append(
            f"Verify the governing approval {requires_approval_id!r} authorises "
            "this retirement.")
    return HFRetirementPlan(
        pack_id=pack_id,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        reason=reason,
        superseded_by_pack_id="",
        superseded_by_revision="",
        requires_approval_id=requires_approval_id,
        planned_actions=tuple(actions),
        rationale=rationale or f"Retiring {pack_id!r} ({reason.value}).",
        executed=False,
        planned_at=_iso(now))


def plan_supersession(
    *,
    old_pack_id: str,
    old_approval: HFDatasetApproval,
    revision_assessment: HFRevisionAssessment,
    new_pack_id: str = "",
    new_approval_id: str = "",
    rationale: str = "",
    now: Optional[datetime] = None,
) -> HFRetirementPlan:
    """Plan replacing a pack because the upstream revision advanced.

    Requires a ``revision_assessment`` whose status indicates a change; the plan
    explicitly demands a *fresh approval* for the new revision before any
    superseding pack may be imported — supersession never reuses the old
    approval.
    """
    new_revision = revision_assessment.observed_revision
    actions = [
        f"Obtain a fresh approval for {old_approval.dataset_id}@{new_revision} "
        "(the old approval does not cover the new revision).",
        f"Import the new revision under that approval as pack "
        f"{new_pack_id or '<new-pack-id>'!r} (dry-run, then explicit write).",
        f"Run the retrieval-eval bridge against the new pack before activation.",
        f"Mark the old pack {old_pack_id!r} inactive only after the new pack is "
        "imported and reviewed.",
        "Record the supersession decision in the human governance log.",
    ]
    return HFRetirementPlan(
        pack_id=old_pack_id,
        dataset_id=old_approval.dataset_id,
        dataset_revision=old_approval.dataset_revision,
        reason=HFRetirementReason.SUPERSEDED_BY_REVISION,
        superseded_by_pack_id=new_pack_id,
        superseded_by_revision=new_revision,
        requires_approval_id=new_approval_id,
        planned_actions=tuple(actions),
        rationale=rationale or (
            f"Revision advanced from {old_approval.dataset_revision!r} to "
            f"{new_revision!r}; superseding {old_pack_id!r} after re-approval."),
        executed=False,
        planned_at=_iso(now))


# --------------------------------------------------------------------------- #
# Durable writes (the only side effects in this module)
# --------------------------------------------------------------------------- #


def _atomic_write_json(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return str(path)


def write_revision_assessment(
    assessment: HFRevisionAssessment, path: str | Path,
) -> str:
    """Atomically write a revision assessment as JSON; returns the path."""
    return _atomic_write_json(Path(path), assessment.to_json() + "\n")


def write_retirement_plan(plan: HFRetirementPlan, path: str | Path) -> str:
    """Atomically write a retirement plan as JSON; returns the path."""
    return _atomic_write_json(Path(path), plan.to_json() + "\n")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_revision_markdown(assessment: HFRevisionAssessment) -> str:
    """Deterministic summary of a revision assessment (no raw row content)."""
    lines = [
        "# Governed HF revision refresh assessment (v6.9)",
        "",
        f"- dataset: `{assessment.dataset_id}`",
        f"- approved revision: `{assessment.approved_revision}`",
        f"- observed revision: `{assessment.observed_revision}`",
        f"- status: **{assessment.status.value}**",
        f"- action: **{assessment.action.value}**",
        f"- covered by existing approval: {assessment.covered_by_existing_approval}",
        "",
        "## Findings",
    ]
    if assessment.findings:
        for finding in assessment.findings:
            flags = []
            if finding.requires_reapproval:
                flags.append("re-approval")
            if finding.blocking:
                flags.append("blocking")
            suffix = f" ({', '.join(flags)})" if flags else ""
            lines.append(f"- `{finding.code.value}`: {finding.message}{suffix}")
    else:
        lines.append("- none — approval still matches the dataset")
    return "\n".join(lines) + "\n"


def render_retirement_markdown(plan: HFRetirementPlan) -> str:
    """Deterministic summary of a retirement / supersession plan."""
    lines = [
        "# Governed HF retirement / supersession plan (v6.9)",
        "",
        f"- pack: `{plan.pack_id}`",
        f"- dataset: `{plan.dataset_id}@{plan.dataset_revision}`",
        f"- reason: **{plan.reason.value}**",
        f"- executed: {plan.executed} (plans are never self-executing)",
    ]
    if plan.is_supersession:
        lines += [
            f"- superseded by: `{plan.superseded_by_pack_id or '<new-pack-id>'}`"
            f" @ `{plan.superseded_by_revision}`",
            f"- requires approval: `{plan.requires_approval_id or '<new-approval>'}`",
        ]
    lines += ["", "## Planned actions"]
    for idx, action in enumerate(plan.planned_actions, start=1):
        lines.append(f"{idx}. {action}")
    if plan.rationale:
        lines += ["", f"> {plan.rationale}"]
    return "\n".join(lines) + "\n"
