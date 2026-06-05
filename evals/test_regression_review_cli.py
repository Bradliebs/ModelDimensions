"""v7.2 governed regression review CLI: governance-only guarantees.

The review CLI never executes a lifecycle action. ``import``/``decide`` write only
the review files and only with ``--write``; ``list``/``inspect``/``validate``/
``actions`` write nothing. An approval emits an action *request* (NOT EXECUTED)
and never creates the governed activation state or audit, never deactivates,
rolls back, supersedes, blocks or activates a pack. Output is deterministic and
labels review as governance, not execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "evals"))

import workbench  # noqa: E402
import agent.regression_review_queue as rrq  # noqa: E402
from _regression_review_helpers import FIXED_NOW, manifest, run_dict  # noqa: E402

_REC = rrq.MonitoringRecommendationCode
_D = rrq.ReviewDecision
_ROLE = rrq.ReviewerRole


def _write_report(tmp_path, *, recommendation=_REC.DEACTIVATE_RECOMMENDED,
                  manifest_obj=None):
    rd = run_dict(recommendation=recommendation, manifest_obj=manifest_obj)
    path = tmp_path / "mon.jsonl"
    path.write_text(json.dumps(rd) + "\n", encoding="utf-8")
    return path, rd


def _base(tmp_path):
    return [
        "regression-review", "--deterministic",
        "--queue", str(tmp_path / "queue.jsonl"),
        "--audit", str(tmp_path / "audit.jsonl"),
        "--actions", str(tmp_path / "actions.jsonl"),
        "--state", str(tmp_path / "missing_state.jsonl"),
    ]


def _imported_id(tmp_path, capsys, **kw):
    report, _ = _write_report(tmp_path, **kw)
    capsys.readouterr()
    rc = workbench.main(_base(tmp_path) + [
        "import", "--monitoring-report", str(report), "--write"])
    capsys.readouterr()
    assert rc == 0
    items = rrq.load_review_queue(tmp_path / "queue.jsonl")
    assert len(items) == 1
    return items[0].review_item_id


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


def test_import_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    report, _ = _write_report(tmp_path)
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    capsys.readouterr()
    rc = workbench.main(_base(tmp_path) + [
        "import", "--monitoring-report", str(report)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "dry-run" in text
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before


def test_import_write_creates_only_queue_and_audit(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    report, _ = _write_report(tmp_path)
    capsys.readouterr()
    workbench.main(_base(tmp_path) + [
        "import", "--monitoring-report", str(report), "--write"])
    capsys.readouterr()
    assert (tmp_path / "queue.jsonl").exists()
    assert (tmp_path / "audit.jsonl").exists()
    assert not (tmp_path / "actions.jsonl").exists()  # import emits no request
    # never creates the governed activation state or audit
    assert not (tmp_path / "config" / "active_knowledge_packs.jsonl").exists()


def test_import_keep_active_imports_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    report, _ = _write_report(tmp_path, recommendation=_REC.KEEP_ACTIVE)
    capsys.readouterr()
    rc = workbench.main(_base(tmp_path) + [
        "import", "--monitoring-report", str(report), "--write"])
    text = capsys.readouterr().out
    assert rc == 0
    assert "nothing review-worthy" in text
    assert not (tmp_path / "queue.jsonl").exists()


def test_import_is_idempotent(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    report, _ = _write_report(tmp_path)
    for _ in range(2):
        capsys.readouterr()
        workbench.main(_base(tmp_path) + [
            "import", "--monitoring-report", str(report), "--write"])
        capsys.readouterr()
    assert len(rrq.load_review_queue(tmp_path / "queue.jsonl")) == 1


# ---------------------------------------------------------------------------
# list / inspect / validate are read-only
# ---------------------------------------------------------------------------


def test_list_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _imported_id(tmp_path, capsys)
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    capsys.readouterr()
    rc = workbench.main(_base(tmp_path) + ["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Regression review queue" in out
    assert "governance, not execution" in out
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before


def test_inspect_writes_nothing_and_labels_request_only(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rid = _imported_id(tmp_path, capsys)
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    capsys.readouterr()
    rc = workbench.main(_base(tmp_path) + ["inspect", "--review-id", rid])
    out = capsys.readouterr().out
    assert rc == 0
    assert "proposed action (NOT executed)" in out
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before


def test_validate_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rid = _imported_id(tmp_path, capsys)
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    capsys.readouterr()
    rc = workbench.main(_base(tmp_path) + ["validate", "--review-id", rid])
    capsys.readouterr()
    assert rc in (0, 1)
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------


def test_decide_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rid = _imported_id(tmp_path, capsys)
    before = (tmp_path / "queue.jsonl").read_bytes()
    audit_before = (tmp_path / "audit.jsonl").read_bytes()
    capsys.readouterr()
    rc = workbench.main(_base(tmp_path) + [
        "decide", "--review-id", rid, "--decision", _D.REJECT,
        "--reviewer", "alice", "--role", _ROLE.MONITORING_REVIEWER,
        "--reason", "not a real regression"])
    text = capsys.readouterr().out
    assert rc == 0
    assert "dry-run" in text
    # queue + audit unchanged, no action file
    assert (tmp_path / "queue.jsonl").read_bytes() == before
    assert (tmp_path / "audit.jsonl").read_bytes() == audit_before
    assert not (tmp_path / "actions.jsonl").exists()


def test_decide_approve_emits_action_request_not_execution(tmp_path, monkeypatch,
                                                           capsys):
    monkeypatch.chdir(tmp_path)
    rid = _imported_id(tmp_path, capsys)
    capsys.readouterr()
    rc = workbench.main(_base(tmp_path) + [
        "decide", "--review-id", rid, "--decision", _D.APPROVE,
        "--reviewer", "gov", "--role", _ROLE.GOVERNANCE_APPROVER,
        "--reason", "confirmed", "--evidence-ack", "--write"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "APPROVED FOR REQUEST ONLY" in out
    assert "NOT EXECUTED" in out
    # an action request was emitted into the distinct actions file
    actions = rrq.load_action_requests(tmp_path / "actions.jsonl")
    assert len(actions) == 1
    assert actions[0].status == rrq.ActionRequestStatus.REQUESTED
    assert actions[0].executed_at == ""
    # review item moved to action_requested, NOT to any executed state
    item = rrq.get_review_item(
        rrq.load_review_queue(tmp_path / "queue.jsonl"), rid)
    assert item.status == rrq.ReviewItemStatus.ACTION_REQUESTED
    # never created the governed activation state/audit
    assert not (tmp_path / "config" / "active_knowledge_packs.jsonl").exists()
    assert not (tmp_path / "reports"
                / "knowledge_pack_activation_audit.jsonl").exists()


def test_approve_by_non_approver_role_is_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rid = _imported_id(tmp_path, capsys)
    capsys.readouterr()
    try:
        workbench.main(_base(tmp_path) + [
            "decide", "--review-id", rid, "--decision", _D.APPROVE,
            "--reviewer", "alice", "--role", _ROLE.MONITORING_REVIEWER,
            "--reason", "confirmed", "--evidence-ack", "--write"])
    except SystemExit as exc:
        assert "regression-review" in str(exc.code)
    else:  # pragma: no cover - approval by wrong role must fail
        raise AssertionError("expected approval by monitoring reviewer to fail")
    # nothing was persisted as approved / requested
    assert not (tmp_path / "actions.jsonl").exists()


def test_reject_keeps_monitoring_evidence_and_emits_no_action(tmp_path, monkeypatch,
                                                             capsys):
    monkeypatch.chdir(tmp_path)
    rid = _imported_id(tmp_path, capsys)
    capsys.readouterr()
    workbench.main(_base(tmp_path) + [
        "decide", "--review-id", rid, "--decision", _D.REJECT,
        "--reviewer", "alice", "--role", _ROLE.MONITORING_REVIEWER,
        "--reason", "false positive", "--write"])
    capsys.readouterr()
    item = rrq.get_review_item(
        rrq.load_review_queue(tmp_path / "queue.jsonl"), rid)
    assert item.status == rrq.ReviewItemStatus.REJECTED
    # rejection does not erase the underlying monitoring evidence
    assert item.recommendation_fingerprint
    assert item.regression_findings
    assert not (tmp_path / "actions.jsonl").exists()


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------


def test_actions_list_labels_not_executed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rid = _imported_id(tmp_path, capsys)
    capsys.readouterr()
    workbench.main(_base(tmp_path) + [
        "decide", "--review-id", rid, "--decision", _D.APPROVE,
        "--reviewer", "gov", "--role", _ROLE.GOVERNANCE_APPROVER,
        "--reason", "confirmed", "--evidence-ack", "--write"])
    capsys.readouterr()
    rc = workbench.main(_base(tmp_path) + ["actions"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NOT EXECUTED by this layer" in out


def test_output_is_deterministic(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _imported_id(tmp_path, capsys)
    capsys.readouterr()
    workbench.main(_base(tmp_path) + ["list"])
    out_a = capsys.readouterr().out
    workbench.main(_base(tmp_path) + ["list"])
    out_b = capsys.readouterr().out
    assert out_a == out_b
