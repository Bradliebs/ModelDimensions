"""v7.2 regression review queue: atomic, append-only persistence.

Pins that the queue/audit/action files are distinct, that writes are atomic and
deterministic (re-saving the same state is byte-identical), that the audit log is
append-only, and that no retrieved text is ever persisted.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.regression_review_queue as rrq  # noqa: E402
from _regression_review_helpers import FIXED_NOW, manifest, run_dict  # noqa: E402

_REC = rrq.MonitoringRecommendationCode
_D = rrq.ReviewDecision
_ROLE = rrq.ReviewerRole


def _item(rec=_REC.DEACTIVATE_RECOMMENDED, corpus="moncorpus-1"):
    rd = run_dict(recommendation=rec, corpus_fingerprint=corpus)
    return rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW)


def test_queue_save_load_roundtrip(tmp_path):
    path = tmp_path / "reviews" / "queue.jsonl"
    items = [_item(), _item(_REC.INVESTIGATE, corpus="moncorpus-2")]
    rrq.save_review_queue(items, path)
    loaded = rrq.load_review_queue(path)
    assert {i.review_item_id for i in loaded} == {i.review_item_id for i in items}


def test_queue_save_is_byte_identical_on_resave(tmp_path):
    path = tmp_path / "reviews" / "queue.jsonl"
    items = [_item(_REC.INVESTIGATE, corpus="moncorpus-2"), _item()]
    rrq.save_review_queue(items, path)
    first = path.read_bytes()
    rrq.save_review_queue(rrq.load_review_queue(path), path)
    assert path.read_bytes() == first


def test_missing_queue_loads_empty(tmp_path):
    assert rrq.load_review_queue(tmp_path / "nope.jsonl") == []


def test_audit_is_append_only(tmp_path):
    path = tmp_path / "reviews" / "audit.jsonl"
    r1 = rrq.make_audit_record(event="imported", review_item_id="regrev-1",
                               actor="alice", at=FIXED_NOW)
    rrq.append_review_audit([r1], path)
    first = path.read_text(encoding="utf-8")
    r2 = rrq.make_audit_record(event="decided", review_item_id="regrev-1",
                               actor="gov", at="2024-02-01T00:00:00+00:00",
                               status="rejected")
    rrq.append_review_audit([r2], path)
    after = path.read_text(encoding="utf-8")
    # prior history preserved verbatim, new record appended
    assert after.startswith(first)
    records = rrq.load_review_audit(path)
    assert [r.event for r in records] == ["imported", "decided"]


def test_action_requests_are_a_distinct_file(tmp_path):
    queue_path = tmp_path / "reviews" / "queue.jsonl"
    action_path = tmp_path / "reviews" / "actions.jsonl"
    mf = manifest()
    rd = run_dict(manifest_obj=mf, recommendation=_REC.DEACTIVATE_RECOMMENDED)
    item = rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW)
    validation = rrq.validate_review_item(item, current_state=mf, now=FIXED_NOW)
    out = rrq.review_item(item, _D.APPROVE, reviewer="gov",
                          role=_ROLE.GOVERNANCE_APPROVER, reason="ok",
                          validation=validation, reviewed_at=FIXED_NOW)
    rrq.save_review_queue([out.item], queue_path)
    rrq.save_action_requests([out.action_request], action_path)
    assert queue_path.exists() and action_path.exists()
    assert rrq.load_action_requests(action_path)[0].action_request_id \
        == out.action_request.action_request_id
    # queue file holds review items, not action requests
    assert "regression_review_item" in queue_path.read_text(encoding="utf-8")
    assert "regression_action_request" in action_path.read_text(encoding="utf-8")


def test_persisted_records_contain_no_retrieved_text(tmp_path):
    path = tmp_path / "reviews" / "queue.jsonl"
    rrq.save_review_queue([_item()], path)
    text = path.read_text(encoding="utf-8").lower()
    for banned in ("chunk_text", "retrieved_text", "passage", "snippet_text"):
        assert banned not in text


def test_add_action_request_is_idempotent():
    mf = manifest()
    rd = run_dict(manifest_obj=mf)
    item = rrq.import_monitoring_recommendation(rd, created_at=FIXED_NOW)
    v = rrq.validate_review_item(item, current_state=mf, now=FIXED_NOW)
    out = rrq.review_item(item, _D.APPROVE, reviewer="gov",
                          role=_ROLE.GOVERNANCE_APPROVER, reason="ok",
                          validation=v, reviewed_at=FIXED_NOW)
    actions = rrq.add_action_request([], out.action_request)
    actions = rrq.add_action_request(actions, out.action_request)
    assert len(actions) == 1
