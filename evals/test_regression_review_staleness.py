"""v7.2 regression review queue: fail-closed staleness validation.

Pins that a review item is valid only while its bound evidence still holds, and
that any drift — active-state hash change, pack fingerprint change, pack no
longer active, baseline replaced, rollback target gone, a superseding run —
marks the item stale so it can never be approved.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.regression_review_queue as rrq  # noqa: E402
from agent.knowledge_pack_activation import (  # noqa: E402
    ActivationStateManifest, ActivePackState, PackLifecycleState)
from _regression_review_helpers import FIXED_NOW, manifest, run_dict  # noqa: E402

_REC = rrq.MonitoringRecommendationCode
_SC = rrq.StalenessCode


def _item(**kw):
    mf = kw.pop("manifest_obj", None)
    rd = run_dict(manifest_obj=mf, **kw) if mf is not None else run_dict(**kw)
    return rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW), rd


def test_matching_state_is_valid():
    mf = manifest()
    item, _ = _item(manifest_obj=mf)
    v = rrq.validate_review_item(item, current_state=mf, now=FIXED_NOW)
    assert v.ok and not v.stale
    assert v.staleness_codes == ()


def test_changed_active_state_hash_is_stale():
    mf = manifest()
    item, _ = _item(manifest_obj=mf)
    drifted = manifest(packs=(("p1", "1.0", "packfp-a", "src1", "r1"),
                              ("p2", "1.0", "packfp-b", "src2", "r2")))
    v = rrq.validate_review_item(item, current_state=drifted, now=FIXED_NOW)
    assert v.stale and not v.ok
    assert _SC.ACTIVE_STATE_CHANGED in v.staleness_codes


def test_changed_pack_fingerprint_is_stale():
    mf = manifest()
    item, _ = _item(manifest_obj=mf)
    changed = manifest(packs=(("p1", "1.0", "packfp-CHANGED", "src1", "r1"),))
    v = rrq.validate_review_item(item, current_state=changed, now=FIXED_NOW)
    assert v.stale
    assert _SC.PACK_FINGERPRINT_CHANGED in v.staleness_codes


def test_pack_no_longer_active_is_stale():
    mf = manifest()
    item, _ = _item(manifest_obj=mf)
    inactive = ActivationStateManifest(records=(ActivePackState(
        pack_id="p1", pack_version="1.0", pack_fingerprint="packfp-a",
        source_type="hf", source_id="src1", source_revision="r1",
        status=PackLifecycleState.INACTIVE),))
    v = rrq.validate_review_item(item, current_state=inactive, now=FIXED_NOW)
    assert v.stale
    assert _SC.PACK_NOT_ACTIVE in v.staleness_codes


def test_missing_pack_is_stale():
    mf = manifest()
    item, _ = _item(manifest_obj=mf)
    empty = ActivationStateManifest(records=())
    v = rrq.validate_review_item(item, current_state=empty, now=FIXED_NOW)
    assert v.stale
    assert _SC.PACK_NOT_ACTIVE in v.staleness_codes


def test_replaced_baseline_is_stale():
    mf = manifest()
    item, _ = _item(manifest_obj=mf)
    v = rrq.validate_review_item(item, current_state=mf, baseline_exists=False,
                                now=FIXED_NOW)
    assert v.stale
    assert _SC.BASELINE_REPLACED in v.staleness_codes


def test_invalid_rollback_target_is_stale():
    mf = manifest()
    rd = run_dict(manifest_obj=mf, recommendation=_REC.ROLLBACK_RECOMMENDED,
                  rollback_target="state-OLD")
    item = rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW)
    v = rrq.validate_review_item(item, current_state=mf,
                                rollback_target_exists=False, now=FIXED_NOW)
    assert v.stale
    assert _SC.ROLLBACK_TARGET_INVALID in v.staleness_codes


def test_superseding_run_marks_stale():
    mf = manifest()
    item, _ = _item(manifest_obj=mf)
    v = rrq.validate_review_item(item, current_state=mf, superseded=True,
                                now=FIXED_NOW)
    assert v.stale
    assert _SC.RECOMMENDATION_SUPERSEDED in v.staleness_codes


def test_different_current_run_invalidates():
    mf = manifest()
    item, _ = _item(manifest_obj=mf)
    newer = run_dict(manifest_obj=mf, corpus_fingerprint="moncorpus-NEW")
    v = rrq.validate_review_item(item, current_state=mf, current_run=newer,
                                now=FIXED_NOW)
    assert v.stale
    assert _SC.MONITORING_RUN_INVALIDATED in v.staleness_codes


def test_validation_roundtrips_to_dict():
    mf = manifest()
    item, _ = _item(manifest_obj=mf)
    v = rrq.validate_review_item(item, current_state=mf, now=FIXED_NOW)
    d = v.to_dict()
    assert d["ok"] is True
    assert d["review_item_id"] == item.review_item_id
