"""Tests for v1.2 note ingestion + the approval queue (offline, no download).

These cover the ingestion contract: importing a note creates *pending* candidate
memories; a rejected candidate is never written; an approved candidate is written
through the frozen ``add_memory`` path and becomes queryable; an edit preserves
the pre-edit text; ``write_approved`` writes only approved candidates; pending
candidates are not queryable until written; the source file is preserved; and a
duplicate-looking candidate is flagged rather than silently merged.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.workbench_service import WorkbenchService  # noqa: E402

_NOTE = """\
# Decisions

- Decision: keep the v1.0 geometry frozen
- The supplier delivery is on Friday afternoon

# Results

- Result: all offline tests passed
"""

_DUP_NOTE = """\
# Results

- Result: all offline tests passed
- Result: all offline tests passed
"""


def _service(tmp_path) -> WorkbenchService:
    return WorkbenchService(
        ledger_path=str(tmp_path / "ledger.jsonl"),
        queue_path=str(tmp_path / "queue.jsonl"),
    )


def _write_note(tmp_path, text: str, name: str = "note.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# 1. importing a note creates pending proposals. ----------------------------

def test_import_creates_pending_proposals(tmp_path):
    service = _service(tmp_path)
    note = _write_note(tmp_path, _NOTE)

    batch = service.import_notes(note)

    assert len(batch) > 0
    pending = service.list_proposals(status="pending")
    assert len(pending) == len(batch)
    assert all(p.status.value == "pending" for p in pending)


# 2. a rejected proposal is never written. ----------------------------------

def test_rejected_proposal_is_not_written(tmp_path):
    service = _service(tmp_path)
    note = _write_note(tmp_path, _NOTE)
    batch = service.import_notes(note)

    target = batch.proposals[0]
    assert service.reject_proposal(target.proposal_id) is True

    written = service.write_approved_proposals()
    assert written == []
    texts = [row["canonical_text"] for row in service.export_ledger()]
    assert target.canonical_text not in texts


# 3. an approved proposal is written to the ledger. -------------------------

def test_approved_proposal_is_written(tmp_path):
    service = _service(tmp_path)
    note = _write_note(tmp_path, _NOTE)
    batch = service.import_notes(note)

    target = batch.proposals[0]
    assert service.approve_proposal(target.proposal_id) is True

    written = service.write_approved_proposals()
    assert len(written) == 1
    texts = [row["canonical_text"] for row in service.export_ledger()]
    assert target.canonical_text in texts


# 4. an edit preserves the pre-edit text in edited_from. --------------------

def test_edit_preserves_edited_from(tmp_path):
    service = _service(tmp_path)
    note = _write_note(tmp_path, _NOTE)
    batch = service.import_notes(note)

    target = batch.proposals[0]
    original = target.canonical_text
    assert service.edit_proposal(target.proposal_id, "a cleaner restatement") is True

    edited = service.proposals.get(target.proposal_id)
    assert edited.status.value == "edited"
    assert edited.edited_from == original
    assert edited.canonical_text == "a cleaner restatement"


# 5. write_approved writes only approved proposals. -------------------------

def test_write_approved_writes_only_approved(tmp_path):
    service = _service(tmp_path)
    note = _write_note(tmp_path, _NOTE)
    batch = service.import_notes(note)
    assert len(batch.proposals) >= 3

    approved = batch.proposals[0]
    rejected = batch.proposals[1]
    # batch.proposals[2] is left pending.
    service.approve_proposal(approved.proposal_id)
    service.reject_proposal(rejected.proposal_id)

    written = service.write_approved_proposals()
    assert len(written) == 1
    texts = [row["canonical_text"] for row in service.export_ledger()]
    assert approved.canonical_text in texts
    assert rejected.canonical_text not in texts
    assert batch.proposals[2].canonical_text not in texts


# 6. pending proposals are not queryable until written. ---------------------

def test_pending_proposals_not_queryable_until_written(tmp_path):
    service = _service(tmp_path)
    note = _write_note(tmp_path, _NOTE)
    batch = service.import_notes(note)
    target = next(p for p in batch.proposals
                  if "Friday afternoon" in p.canonical_text)

    # Before approval/write nothing is in the bank.
    before = service.query_memory(target.canonical_text)
    assert before.memory_used is False
    assert before.candidate_retrieved is False

    service.approve_proposal(target.proposal_id)
    service.write_approved_proposals()

    after = service.query_memory(target.canonical_text)
    assert after.memory_used is True
    assert after.candidate_retrieved is True


# 7. the imported source file is preserved on each proposal. ----------------

def test_source_file_is_preserved(tmp_path):
    service = _service(tmp_path)
    note = _write_note(tmp_path, _NOTE)
    batch = service.import_notes(note)

    assert all(p.source_file == str(note) for p in batch.proposals)
    assert all(p.source_section for p in batch.proposals)


# 8. a duplicate-looking proposal is flagged, not silently merged. ----------

def test_duplicate_proposal_is_flagged_not_merged(tmp_path):
    service = _service(tmp_path)
    note = _write_note(tmp_path, _DUP_NOTE, name="dup.md")
    batch = service.import_notes(note)

    # Both identical lines survive as distinct proposals (distinct ids).
    assert len(batch.proposals) == 2
    ids = {p.proposal_id for p in batch.proposals}
    assert len(ids) == 2
    # The second is flagged as a possible duplicate rather than dropped.
    assert any("duplicate" in p.reason for p in batch.proposals)


# 9. write_approved is idempotent (no duplicate memories on a second call). --

def test_write_approved_is_idempotent(tmp_path):
    service = _service(tmp_path)
    note = _write_note(tmp_path, _NOTE)
    batch = service.import_notes(note)
    service.approve_proposal(batch.proposals[0].proposal_id)

    first = service.write_approved_proposals()
    second = service.write_approved_proposals()

    assert len(first) == 1
    assert second == []
    active = [r for r in service.export_ledger() if r["status"] == "active"]
    assert len(active) == 1
