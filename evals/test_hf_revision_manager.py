"""Tests for v6.9 governed HF revision refresh & retirement — Phases G/H/I.

These verify the cardinal refresh rule — an approval for one revision never
covers another, and metadata drift on the same revision still forces
re-approval — plus the planning-only nature of retirement and supersession: the
plans describe what should happen, execute nothing, and the module imports no
writer / service / importer.
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import hf_revision_manager as revmgr  # noqa: E402
from agent.hf_data_adapter import HuggingFaceDatasetMetadata  # noqa: E402
from agent.hf_import_lifecycle import (  # noqa: E402
    HFApprovalScope,
    HFDatasetApproval,
    compute_metadata_fingerprint,
)
from agent.hf_revision_manager import (  # noqa: E402
    HFRefreshAction,
    HFRetirementReason,
    HFRevisionStatus,
    assess_revision,
    plan_retirement,
    plan_supersession,
    write_retirement_plan,
    write_revision_assessment,
)

_NOW = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
_BASE_META = ROOT / "demos" / "hf_dataset_metadata_example.json"
_NEXT_META = ROOT / "demos" / "hf_dataset_metadata_next_revision.json"


def _load_meta(path: Path) -> HuggingFaceDatasetMetadata:
    return HuggingFaceDatasetMetadata.from_dict(
        json.loads(path.read_text(encoding="utf-8")))


def _approval_for(metadata: HuggingFaceDatasetMetadata, *,
                  revision="main", **overrides) -> HFDatasetApproval:
    base = dict(
        approval_id="hfappr-rev",
        dataset_id=metadata.dataset_id,
        dataset_revision=revision,
        metadata_fingerprint=compute_metadata_fingerprint(metadata),
        intake_assessment_fingerprint="",
        approved_by="reviewer@local",
        approved_at="2026-05-02T09:00:00+00:00",
        approval_scope=HFApprovalScope.EVAL_AND_KNOWLEDGE,
        approved_split_names=("train",),
        approved_columns=("id", "question", "answer"),
        row_limit=100,
        approved_intended_use="eval and knowledge",
        licence_snapshot=metadata.license or "",
        provenance_snapshot="hf:demo",
    )
    base.update(overrides)
    return HFDatasetApproval(**base)


# 1. Same revision + identical metadata is covered, no action. --------------


def test_identical_metadata_is_up_to_date():
    meta = _load_meta(_BASE_META)
    approval = _approval_for(meta)
    result = assess_revision(approval, observed_metadata=meta, now=_NOW)
    assert result.status is HFRevisionStatus.UP_TO_DATE
    assert result.action is HFRefreshAction.NO_ACTION
    assert result.covered_by_existing_approval is True
    assert result.requires_reapproval is False


# 2. A new revision is never covered by the old approval. -------------------


def test_new_revision_requires_reapproval():
    meta = _load_meta(_BASE_META)
    approval = _approval_for(meta, revision="main")
    result = assess_revision(
        approval, observed_metadata=meta, observed_revision="v2", now=_NOW)
    assert result.status is HFRevisionStatus.REVISION_CHANGED
    assert result.action is HFRefreshAction.REQUIRE_REAPPROVAL
    assert result.covered_by_existing_approval is False
    codes = {f.code.value for f in result.findings}
    assert "revision_advanced" in codes


# 3. Metadata drift on the same revision still forces re-approval. ----------


def test_metadata_drift_same_revision_requires_reapproval():
    base = _load_meta(_BASE_META)
    nxt = _load_meta(_NEXT_META)  # same id+revision, licence changed
    approval = _approval_for(base, revision="main")
    result = assess_revision(
        approval, observed_metadata=nxt, observed_revision="main", now=_NOW)
    # Licence drift changes the fingerprint -> METADATA_DRIFT, re-approval.
    assert result.status is HFRevisionStatus.METADATA_DRIFT
    assert result.action is HFRefreshAction.REQUIRE_REAPPROVAL
    assert result.covered_by_existing_approval is False
    codes = {f.code.value for f in result.findings}
    assert "licence_changed" in codes
    assert "metadata_fingerprint_changed" in codes


# 4. A different dataset entirely is blocked. -------------------------------


def test_different_dataset_is_blocked():
    meta = _load_meta(_BASE_META)
    approval = _approval_for(meta)
    other = HuggingFaceDatasetMetadata.from_dict(
        {"id": "someone/other-dataset", "card_data": {"license": "mit"}})
    result = assess_revision(approval, observed_metadata=other, now=_NOW)
    assert result.status is HFRevisionStatus.DATASET_MISMATCH
    assert result.action is HFRefreshAction.BLOCK
    assert result.covered_by_existing_approval is False


# 5. Losing the licence upstream blocks the refresh. ------------------------


def test_licence_now_unknown_blocks():
    meta = _load_meta(_BASE_META)
    approval = _approval_for(meta)
    unlicensed = HuggingFaceDatasetMetadata.from_dict(
        {"id": "demo/governed-qa", "card_data": {
            "language": ["en"], "task_categories": ["question-answering"],
            "size_categories": ["n<1K"]}})
    result = assess_revision(approval, observed_metadata=unlicensed, now=_NOW)
    assert result.action is HFRefreshAction.BLOCK
    assert result.blocked is True
    codes = {f.code.value for f in result.findings}
    assert "licence_now_unknown" in codes


# 6. Becoming private upstream blocks the refresh. --------------------------


def test_now_private_blocks():
    meta = _load_meta(_BASE_META)
    approval = _approval_for(meta)
    private = HuggingFaceDatasetMetadata.from_dict(
        {"id": "demo/governed-qa", "card_data": {"license": "apache-2.0"},
         "private": True})
    result = assess_revision(approval, observed_metadata=private, now=_NOW)
    assert result.action is HFRefreshAction.BLOCK
    codes = {f.code.value for f in result.findings}
    assert "now_private" in codes


# 7. The assessment is deterministic (timestamp-independent fingerprint). ---


def test_assessment_is_deterministic():
    meta = _load_meta(_BASE_META)
    approval = _approval_for(meta)
    a = assess_revision(approval, observed_metadata=meta, now=_NOW)
    b = assess_revision(
        approval, observed_metadata=meta,
        now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert a.observed_metadata_fingerprint == b.observed_metadata_fingerprint
    assert a.status is b.status


# 8. A retirement plan executes nothing. ------------------------------------


def test_retirement_plan_is_inert():
    plan = plan_retirement(
        "demo-eval", dataset_id="demo/governed-qa", dataset_revision="main",
        reason=HFRetirementReason.APPROVAL_EXPIRED, now=_NOW)
    assert plan.executed is False
    assert plan.reason is HFRetirementReason.APPROVAL_EXPIRED
    assert plan.is_supersession is False
    assert plan.planned_actions  # describes steps, performs none
    payload = plan.to_dict()
    assert payload["_record"] == "hf_retirement_plan"
    assert payload["executed"] is False


# 9. A supersession plan demands a fresh approval for the new revision. -----


def test_supersession_requires_fresh_approval():
    meta = _load_meta(_BASE_META)
    approval = _approval_for(meta, revision="main")
    assessment = assess_revision(
        approval, observed_metadata=meta, observed_revision="v2", now=_NOW)
    plan = plan_supersession(
        old_pack_id="demo-eval", old_approval=approval,
        revision_assessment=assessment, new_pack_id="demo-eval-v2", now=_NOW)
    assert plan.is_supersession is True
    assert plan.executed is False
    assert plan.superseded_by_revision == "v2"
    assert plan.superseded_by_pack_id == "demo-eval-v2"
    # The plan text must call for a fresh approval, never reuse the old one.
    joined = " ".join(plan.planned_actions).lower()
    assert "fresh approval" in joined
    assert "does not cover the new revision" in joined


# 10. Writers are atomic and the markdown renders without raw content. ------


def test_writers_and_markdown(tmp_path):
    meta = _load_meta(_BASE_META)
    approval = _approval_for(meta)
    assessment = assess_revision(
        approval, observed_metadata=meta, observed_revision="v2", now=_NOW)
    rev_path = write_revision_assessment(assessment, tmp_path / "rev.json")
    assert Path(rev_path).exists()
    reloaded = json.loads(Path(rev_path).read_text(encoding="utf-8"))
    assert reloaded["_record"] == "hf_revision_assessment"

    plan = plan_supersession(
        old_pack_id="demo-eval", old_approval=approval,
        revision_assessment=assessment, new_pack_id="demo-eval-v2", now=_NOW)
    plan_path = write_retirement_plan(plan, tmp_path / "plan.json")
    assert Path(plan_path).exists()

    md = revmgr.render_revision_markdown(assessment)
    assert "require_reapproval" in md
    md2 = revmgr.render_retirement_markdown(plan)
    assert "never self-executing" in md2


# 11. The demo next-revision metadata illustrates licence drift. ------------


def test_demo_next_revision_shows_drift():
    base = _load_meta(_BASE_META)
    nxt = _load_meta(_NEXT_META)
    assert base.dataset_id == nxt.dataset_id
    assert base.license != nxt.license
    approval = _approval_for(base)
    result = assess_revision(
        approval, observed_metadata=nxt, observed_revision="main", now=_NOW)
    assert result.requires_reapproval is True


# 12. The module imports no forbidden writers/clients/services. -------------


def test_module_imports_no_forbidden_dependencies():
    source = Path(revmgr.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    forbidden = {
        "save_registry", "MemoryLedger", "KnowledgeLibrary",
        "WorkbenchService", "agent.workbench_service",
        "agent.memory_ledger", "agent.source_registry",
        "agent.knowledge_library", "agent.hf_dataset_importer",
        "agent.hf_eval_pack_importer", "agent.hf_knowledge_pack_importer",
        "agent.project_packs", "datasets", "huggingface_hub", "requests",
        "urllib", "httpx", "openai", "anthropic",
    }
    leaked = imported & forbidden
    assert not leaked, f"revision manager must not import: {leaked}"
