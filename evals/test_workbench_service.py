"""Tests for the v1.1 workbench service (offline, no model download).

These cover the workbench's contract over the frozen v1.0 path: add creates an
active memory; an exact query grounds and cites the memory id; a one-word
near-miss is retrieved but rejected; an ambiguous candidate is not grounded; a
deleted memory cannot be cited; the export reflects active/deleted status; and
the audit trail carries the four load-bearing fields.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.workbench_service import QueryAudit, WorkbenchService  # noqa: E402

_FRIDAY = "the supplier delivery is on Friday afternoon"
_MONDAY = "the supplier delivery is on Monday afternoon"


def _service(tmp_path) -> WorkbenchService:
    return WorkbenchService(ledger_path=str(tmp_path / "ledger.jsonl"))


# 1. add memory creates an active memory. -----------------------------------

def test_add_memory_creates_active_entry(tmp_path):
    service = _service(tmp_path)
    entry = service.add_memory(_FRIDAY, source="test", tags=["a"])

    assert entry.status == "active"
    assert entry.deleted_at is None
    assert entry.memory_id.startswith("mem-")
    assert service.ledger.get(entry.memory_id) is entry


# 2. query exact memory grounds with its memory_id. -------------------------

def test_query_exact_memory_grounds_with_id(tmp_path):
    service = _service(tmp_path)
    entry = service.add_memory(_FRIDAY)

    audit = service.query_memory(_FRIDAY)

    assert audit.candidate_retrieved is True
    assert audit.verifier_verdict == "accept"
    assert audit.memory_used is True
    assert audit.refused is False
    assert entry.memory_id in audit.cited_memory_ids


# 3. a one-word near-miss is retrieved but rejected. ------------------------

def test_query_near_miss_retrieves_but_rejects(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY)

    audit = service.query_memory(_MONDAY)

    assert audit.candidate_retrieved is True
    assert audit.verifier_verdict == "reject"
    assert audit.memory_used is False
    assert audit.refused is True
    assert audit.cited_memory_ids == []


# 4. an ambiguous candidate is not grounded. --------------------------------

def test_ambiguous_candidate_not_grounded(tmp_path):
    service = _service(tmp_path)
    # A stored fact and an unrelated query with no material mismatch and no
    # exact/containment/fixture match yields AMBIGUOUS -> refuse.
    service.add_memory("the primary server is located in the Dublin data center")

    audit = service.query_memory("tell me something about the weather")

    assert audit.candidate_retrieved is True
    assert audit.verifier_verdict == "ambiguous"
    assert audit.memory_used is False
    assert audit.refused is True


# 5. a deleted memory cannot be cited. --------------------------------------

def test_deleted_memory_cannot_be_cited(tmp_path):
    service = _service(tmp_path)
    entry = service.add_memory(_FRIDAY)

    assert service.delete_memory(entry.memory_id) is True

    audit = service.query_memory(_FRIDAY)
    assert audit.candidate_retrieved is False
    assert audit.memory_used is False
    assert audit.refused is True
    assert entry.memory_id not in audit.cited_memory_ids


# 6. export ledger includes active and deleted status. ----------------------

def test_export_ledger_includes_status(tmp_path):
    service = _service(tmp_path)
    kept = service.add_memory("a memory that stays active")
    removed = service.add_memory("a memory that will be deleted")
    service.delete_memory(removed.memory_id)

    rows = {r["memory_id"]: r for r in service.export_ledger()}

    assert rows[kept.memory_id]["status"] == "active"
    assert rows[removed.memory_id]["status"] == "deleted"
    assert rows[removed.memory_id]["deleted_at"] is not None


# 7. the audit trail carries the four load-bearing fields. ------------------

def test_audit_trail_fields_present(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY)

    audit = service.query_memory(_FRIDAY)

    assert isinstance(audit, QueryAudit)
    payload = audit.to_dict()
    for key in ("candidate_retrieved", "verifier_verdict",
                "memory_used", "refused"):
        assert key in payload


# 8. the ledger persists to its JSONL path. ---------------------------------

def test_ledger_persists_to_disk(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    service = WorkbenchService(ledger_path=str(ledger_path))
    service.add_memory(_FRIDAY, tags=["persist"])

    assert ledger_path.exists()
    contents = ledger_path.read_text(encoding="utf-8").strip()
    assert "Friday" in contents
    assert '"status": "active"' in contents
