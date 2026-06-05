"""CLI surface for the v7.3 governed lifecycle action executor."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import agent.regression_review_queue as rrq  # noqa: E402
import workbench  # noqa: E402
from _lifecycle_executor_helpers import (  # noqa: E402
    approved_action_request, execution_approval, two_pack_manifest, write_audit,
    write_state,
)

_ROLE = rrq.ReviewerRole.LIFECYCLE_OPERATOR


def _setup(tmp_path, *, recommendation=None, candidate_pack_ids=("p1",),
           rollback_target="", mf=None, audit_records=()):
    mf = mf if mf is not None else two_pack_manifest()
    kwargs = {"mf": mf, "candidate_pack_ids": candidate_pack_ids,
              "rollback_target": rollback_target}
    if recommendation is not None:
        kwargs["recommendation"] = recommendation
    ar, rec, _ = approved_action_request(**kwargs)
    approval = execution_approval(ar)

    state = tmp_path / "config" / "active_knowledge_packs.jsonl"
    audit = tmp_path / "reports" / "knowledge_pack_activation_audit.jsonl"
    actions = tmp_path / "reviews" / "action_requests.jsonl"
    records = tmp_path / "reviews" / "review_records.jsonl"
    approval_path = tmp_path / "reviews" / "exec_approval.json"

    write_state(state, mf)
    write_audit(audit, audit_records)
    rrq.save_action_requests([ar], actions)
    records.parent.mkdir(parents=True, exist_ok=True)
    records.write_text(rec.to_json() + "\n", encoding="utf-8")
    approval_path.write_text(approval.to_json(), encoding="utf-8")

    paths = {
        "state": str(state), "audit": str(audit), "actions": str(actions),
        "records": str(records), "approval": str(approval_path),
        "results": str(tmp_path / "reviews" / "results.jsonl"),
        "audit_out": str(tmp_path / "reviews" / "exec_audit.jsonl"),
        "follow_ups": str(tmp_path / "reviews" / "follow_ups.jsonl"),
        "blocks": str(tmp_path / "reviews" / "blocks.jsonl"),
    }
    return ar, approval, mf, paths


def _globals(paths):
    return [
        "lifecycle-executor", "--deterministic",
        "--state", paths["state"], "--audit", paths["audit"],
        "--actions", paths["actions"], "--review-records", paths["records"],
        "--results", paths["results"], "--audit-out", paths["audit_out"],
        "--follow-ups", paths["follow_ups"], "--blocks", paths["blocks"],
    ]


def test_inspect_writes_nothing(tmp_path, capsys):
    ar, _, _, paths = _setup(tmp_path)
    rc = workbench.main(_globals(paths) + [
        "inspect", "--action-request-id", ar.action_request_id])
    assert rc == 0
    assert ar.action_request_id in capsys.readouterr().out
    assert not Path(paths["results"]).exists()


def test_validate_passes_and_writes_nothing(tmp_path, capsys):
    ar, _, _, paths = _setup(tmp_path)
    rc = workbench.main(_globals(paths) + [
        "validate", "--action-request-id", ar.action_request_id,
        "--execution-approval", paths["approval"]])
    assert rc == 0
    assert "VALID" in capsys.readouterr().out
    assert not Path(paths["results"]).exists()


def test_plan_writes_nothing(tmp_path, capsys):
    ar, _, _, paths = _setup(tmp_path)
    rc = workbench.main(_globals(paths) + [
        "plan", "--action-request-id", ar.action_request_id,
        "--execution-approval", paths["approval"]])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lifeplan-" in out
    assert not Path(paths["results"]).exists()


def test_execute_requires_write_or_dry_run(tmp_path):
    ar, _, _, paths = _setup(tmp_path)
    # Neither --write nor --dry-run is supplied: argparse must reject it.
    try:
        workbench.main(_globals(paths) + [
            "execute", "--action-request-id", ar.action_request_id,
            "--execution-approval", paths["approval"], "--executor", "op",
            "--role", _ROLE])
    except SystemExit as exc:
        assert exc.code != 0
        return
    raise AssertionError("expected argparse to require --write or --dry-run")


def test_execute_dry_run_writes_nothing(tmp_path, capsys):
    ar, _, mf, paths = _setup(tmp_path)
    rc = workbench.main(_globals(paths) + [
        "execute", "--action-request-id", ar.action_request_id,
        "--execution-approval", paths["approval"], "--executor", "op",
        "--role", _ROLE, "--dry-run"])
    assert rc == 0
    assert not Path(paths["results"]).exists()
    # Active state file is unchanged.
    from agent import knowledge_pack_activation as kpa
    live = kpa.ActivationStateManager(
        state_path=paths["state"], audit_path=paths["audit"]).load_state()
    assert live.state_hash == mf.state_hash


def test_execute_write_deactivates_and_persists(tmp_path, capsys):
    ar, _, mf, paths = _setup(tmp_path)
    rc = workbench.main(_globals(paths) + [
        "execute", "--action-request-id", ar.action_request_id,
        "--execution-approval", paths["approval"], "--executor", "op",
        "--role", _ROLE, "--write"])
    assert rc == 0
    from agent import knowledge_pack_activation as kpa
    live = kpa.ActivationStateManager(
        state_path=paths["state"], audit_path=paths["audit"]).load_state()
    assert not live.find("p1").is_active
    assert live.find("p2").is_active
    assert Path(paths["results"]).exists()
    assert Path(paths["audit_out"]).exists()
    # Action request is now marked executed on disk.
    actions = rrq.load_action_requests(paths["actions"])
    updated = rrq.get_action_request(actions, ar.action_request_id)
    assert updated.status == rrq.ActionRequestStatus.EXECUTED


def test_execute_is_idempotent_via_action_status(tmp_path, capsys):
    ar, _, mf, paths = _setup(tmp_path)
    argv = _globals(paths) + [
        "execute", "--action-request-id", ar.action_request_id,
        "--execution-approval", paths["approval"], "--executor", "op",
        "--role", _ROLE, "--write"]
    assert workbench.main(argv) == 0
    from agent import knowledge_pack_activation as kpa
    first_hash = kpa.ActivationStateManager(
        state_path=paths["state"], audit_path=paths["audit"]).load_state().state_hash
    # Second run: the persisted action request is EXECUTED -> validation refuses.
    rc = workbench.main(argv)
    assert rc == 1  # validation fails closed
    second_hash = kpa.ActivationStateManager(
        state_path=paths["state"], audit_path=paths["audit"]).load_state().state_hash
    assert second_hash == first_hash  # no second mutation


def test_follow_ups_and_blocks_listing(tmp_path, capsys):
    ar, _, _, paths = _setup(tmp_path)
    rc = workbench.main(_globals(paths) + ["follow-ups"])
    assert rc == 0
    assert "no operational follow-up" in capsys.readouterr().out
    rc = workbench.main(_globals(paths) + ["activation-blocks"])
    assert rc == 0
    assert "no activation-block" in capsys.readouterr().out
