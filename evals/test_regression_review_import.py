"""v7.2 regression review queue: which recommendations enter review, and merge.

Pins the default import policy — actionable recommendations create review items,
``keep_active`` does not unless explicitly requested, and re-importing the same
recommendation is an idempotent no-op that preserves any existing review state.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.regression_review_queue as rrq  # noqa: E402
from dataclasses import replace  # noqa: E402
from _regression_review_helpers import FIXED_NOW, run_dict  # noqa: E402

_REC = rrq.MonitoringRecommendationCode


def _import(rec, **kw):
    return rrq.import_monitoring_recommendation(
        run_dict(recommendation=rec), created_at=FIXED_NOW, **kw)


def test_actionable_recommendations_are_imported():
    for rec in (_REC.INVESTIGATE, _REC.DEACTIVATE_RECOMMENDED,
                _REC.ROLLBACK_RECOMMENDED, _REC.BLOCK_FUTURE_ACTIVATION,
                _REC.INSUFFICIENT_EVIDENCE_TO_RECOMMEND):
        assert _import(rec) is not None, rec


def test_keep_active_is_not_imported_by_default():
    assert _import(_REC.KEEP_ACTIVE) is None


def test_keep_active_imports_only_when_explicitly_requested():
    rd = run_dict(recommendation=_REC.KEEP_ACTIVE)
    forced = rrq.import_monitoring_recommendation(
        rd, recommendation=_REC.KEEP_ACTIVE.value, created_at=FIXED_NOW)
    assert forced is not None
    assert forced.recommendation == _REC.KEEP_ACTIVE.value


def test_keep_active_with_watch_respects_include_watch_flag():
    assert _import(_REC.KEEP_ACTIVE_WITH_WATCH) is not None
    assert _import(_REC.KEEP_ACTIVE_WITH_WATCH, include_watch=False) is None


def test_insufficient_evidence_maps_to_additional_evidence_action():
    item = _import(_REC.INSUFFICIENT_EVIDENCE_TO_RECOMMEND)
    assert item.proposed_action_type == rrq.ActionRequestType.REQUEST_ADDITIONAL_EVIDENCE


def test_merge_into_queue_is_idempotent():
    item = _import(_REC.DEACTIVATE_RECOMMENDED)
    queue = rrq.add_review_item([], item)
    assert len(queue) == 1
    # a duplicate import is the same id -> no growth, existing item preserved
    reviewed = replace(item, status=rrq.ReviewItemStatus.REJECTED)
    queue2 = rrq.add_review_item([reviewed], _import(_REC.DEACTIVATE_RECOMMENDED))
    assert len(queue2) == 1
    assert queue2[0].review_item_id == item.review_item_id
    assert queue2[0].status == rrq.ReviewItemStatus.REJECTED


def test_queue_is_sorted_deterministically():
    a = _import(_REC.DEACTIVATE_RECOMMENDED)
    b = rrq.import_monitoring_recommendation(
        run_dict(recommendation=_REC.INVESTIGATE,
                 corpus_fingerprint="moncorpus-z"), created_at=FIXED_NOW)
    q1 = rrq.add_review_item(rrq.add_review_item([], a), b)
    q2 = rrq.add_review_item(rrq.add_review_item([], b), a)
    assert [r.review_item_id for r in q1] == [r.review_item_id for r in q2]
    assert [r.review_item_id for r in q1] == sorted(r.review_item_id for r in q1)
