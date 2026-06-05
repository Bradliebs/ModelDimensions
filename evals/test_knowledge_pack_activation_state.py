"""v7.0 activation state manager: dry-run vs write, atomicity, conflicts.

Pins that validation/dry-run write nothing, ``--write`` activates the exact pack,
the state file is atomic, the audit log is append-only, a second active revision
of the same lineage is rejected, and pack files stay byte-identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import knowledge_pack_activation as kpa  # noqa: E402
from _kp_activation_helpers import (  # noqa: E402
    FIXED_NOW,
    approval_for,
    passing_evidence,
    write_hf_pack,
)


def _manager(tmp_path):
    return kpa.ActivationStateManager(
        state_path=tmp_path / "config" / "active_knowledge_packs.jsonl",
        audit_path=tmp_path / "reports" / "activation_audit.jsonl")


def _request(identity):
    evidence = passing_evidence(identity)
    return kpa.KnowledgePackActivationRequest(
        identity=identity, approval=approval_for(identity, evidence=evidence),
        evidence=evidence, current_state=kpa.PackLifecycleState.EVALUATED)


def _pack_bytes(pack_dir: Path):
    return {p.name: p.read_bytes() for p in sorted(pack_dir.iterdir())}


def test_dry_run_writes_nothing(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    identity = kpa.load_pack_identity(pack)
    mgr = _manager(tmp_path)
    result = mgr.activate(_request(identity), write=False, now=FIXED_NOW)
    assert result.success and not result.written
    assert not mgr.state_path.exists()
    assert not mgr.audit_path.exists()


def test_write_activates_exact_pack(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    identity = kpa.load_pack_identity(pack)
    before = _pack_bytes(pack)
    mgr = _manager(tmp_path)
    result = mgr.activate(_request(identity), write=True, now=FIXED_NOW)
    assert result.success and result.written
    state = mgr.load_state()
    active = state.active()
    assert [r.pack_id for r in active] == ["hf-demo"]
    assert active[0].pack_fingerprint == identity.content_fingerprint
    assert _pack_bytes(pack) == before  # pack untouched


def test_audit_is_append_only(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    identity = kpa.load_pack_identity(pack)
    mgr = _manager(tmp_path)
    mgr.activate(_request(identity), write=True, now=FIXED_NOW)
    mgr.deactivate("hf-demo",
                   approval=approval_for(
                       identity, scope=kpa.ActivationScope.EMERGENCY_DEACTIVATE),
                   write=True, now=FIXED_NOW)
    audit = mgr.load_audit()
    assert len(audit) == 2
    assert audit[0].action == kpa.AuditAction.ACTIVATE
    assert audit[1].action == kpa.AuditAction.DEACTIVATE


def test_blocked_activation_does_not_mutate_state(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    identity = kpa.load_pack_identity(pack)
    mgr = _manager(tmp_path)
    # No evidence => blocked.
    bad = kpa.KnowledgePackActivationRequest(
        identity=identity, approval=approval_for(identity), evidence=None,
        current_state=kpa.PackLifecycleState.EVALUATED)
    result = mgr.activate(bad, write=True, now=FIXED_NOW)
    assert not result.success and not result.written
    assert not mgr.state_path.exists()  # no state persisted for a failure
    assert mgr.audit_path.exists()  # failure is still audited
    assert mgr.load_audit()[0].success is False


def test_second_lineage_revision_rejected(tmp_path):
    v1 = write_hf_pack(tmp_path / "packs" / "hf-v1",
                       pack_id="hf-v1", dataset_revision="rev-1")
    v2 = write_hf_pack(tmp_path / "packs" / "hf-v2",
                       pack_id="hf-v2", dataset_revision="rev-2")
    id1 = kpa.load_pack_identity(v1)
    id2 = kpa.load_pack_identity(v2)
    mgr = _manager(tmp_path)
    mgr.activate(_request(id1), write=True, now=FIXED_NOW)
    result = mgr.activate(_request(id2), write=True, now=FIXED_NOW)
    assert not result.success
    assert kpa.ActivationFindingCode.LINEAGE_CONFLICT_ACTIVE_REVISION in {
        f.code for f in result.validation.blocking_findings}
    # The first revision is still the only active pack.
    assert [r.pack_id for r in mgr.load_state().active()] == ["hf-v1"]


def test_distinct_lineages_coexist(tmp_path):
    hf = kpa.load_pack_identity(write_hf_pack(tmp_path / "packs" / "hf-demo"))
    from _kp_activation_helpers import write_pdf_pack
    pdf = kpa.load_pack_identity(write_pdf_pack(tmp_path / "packs" / "pdf-demo"))
    mgr = _manager(tmp_path)
    mgr.activate(_request(hf), write=True, now=FIXED_NOW)
    result = mgr.activate(_request(pdf), write=True, now=FIXED_NOW)
    assert result.success
    assert sorted(r.pack_id for r in mgr.load_state().active()) == [
        "hf-demo", "pdf-demo"]


def test_state_file_is_valid_jsonl(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    identity = kpa.load_pack_identity(pack)
    mgr = _manager(tmp_path)
    mgr.activate(_request(identity), write=True, now=FIXED_NOW)
    for line in mgr.state_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            assert record["_record"] == "active_knowledge_pack"


def test_no_tmp_files_left_behind(tmp_path):
    pack = write_hf_pack(tmp_path / "packs" / "hf-demo")
    identity = kpa.load_pack_identity(pack)
    mgr = _manager(tmp_path)
    mgr.activate(_request(identity), write=True, now=FIXED_NOW)
    leftovers = list((tmp_path / "config").glob("*.tmp"))
    assert leftovers == []
