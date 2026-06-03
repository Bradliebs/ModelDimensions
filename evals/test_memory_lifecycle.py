"""Tests for v1.3 memory lifecycle: duplicates, conflicts, and supersession.

These cover the lifecycle contract: an exact duplicate of an existing memory is
detected; a high-overlap rewrite is flagged as a possible duplicate; a
Friday -> Monday rewrite is flagged as a conflict (the verifier rejects the
weekday flip); a plain approve refuses a flagged duplicate/conflict; an explicit
approve-as-new writes a conflicting memory anyway; an approve-superseding marks
the old memory superseded and removes it from the bank so it can no longer be
grounded as the current answer; a deleted memory still cannot be cited; old
ledger lines without the new supersession fields still load; and
``list_conflicts`` surfaces pending conflict proposals.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.memory_ledger import MemoryLedger  # noqa: E402
from agent.memory_lifecycle import LifecycleVerdict  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from slm.schemas import VerificationVerdict  # noqa: E402

_FRIDAY = "the supplier delivery is on Friday afternoon"
_MONDAY = "the supplier delivery is on Monday afternoon"


def _service(tmp_path) -> WorkbenchService:
    return WorkbenchService(
        ledger_path=str(tmp_path / "ledger.jsonl"),
        queue_path=str(tmp_path / "queue.jsonl"),
    )


def _write_note(tmp_path, text: str, name: str = "note.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _proposal_for(service, text: str):
    """Return the queued proposal whose canonical text matches ``text``."""
    target = text.lower().rstrip(".")
    for p in service.list_proposals(status=None):
        if p.canonical_text.lower().rstrip(".") == target:
            return p
    raise AssertionError(f"no proposal matched: {text!r}")


# 1. an exact duplicate is detected. ----------------------------------------

def test_exact_duplicate_is_detected(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")
    note = _write_note(tmp_path, f"# Facts\n\n- {_FRIDAY}\n")

    service.import_notes(note)
    proposal = _proposal_for(service, _FRIDAY)

    assert proposal.lifecycle_verdict == LifecycleVerdict.DUPLICATE.value
    assert proposal.lifecycle_candidate_id is not None


# 2. a possible duplicate is detected. --------------------------------------

def test_possible_duplicate_is_detected(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")
    reworded = "the supplier delivery happens on Friday afternoon"
    note = _write_note(tmp_path, f"# Facts\n\n- {reworded}\n")

    service.import_notes(note)
    proposal = _proposal_for(service, reworded)

    assert proposal.lifecycle_verdict == (
        LifecycleVerdict.POSSIBLE_DUPLICATE.value
    )


# 3. a Friday -> Monday proposal flags a conflict. --------------------------

def test_friday_to_monday_flags_conflict(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")
    note = _write_note(tmp_path, f"# Facts\n\n- {_MONDAY}\n")

    service.import_notes(note)
    proposal = _proposal_for(service, _MONDAY)

    assert proposal.lifecycle_verdict == LifecycleVerdict.CONFLICT.value
    assert proposal.lifecycle_candidate_id is not None


# 4. an approved superseding memory marks the old memory superseded. --------

def test_superseding_approval_marks_old_superseded(tmp_path):
    service = _service(tmp_path)
    old = service.add_memory(_FRIDAY, source="seed")
    note = _write_note(tmp_path, f"# Facts\n\n- {_MONDAY}\n")
    service.import_notes(note)
    proposal = _proposal_for(service, _MONDAY)

    assert service.approve_proposal_superseding(
        proposal.proposal_id, old.memory_id)
    service.write_approved_proposals()

    old_entry = service.ledger.get(old.memory_id)
    assert old_entry.status == "superseded"
    assert old_entry.superseded_by is not None


# 5. a superseded memory is not grounded as the current answer. -------------

def test_superseded_memory_not_grounded_as_current(tmp_path):
    service = _service(tmp_path)
    old = service.add_memory(_FRIDAY, source="seed")
    note = _write_note(tmp_path, f"# Facts\n\n- {_MONDAY}\n")
    service.import_notes(note)
    proposal = _proposal_for(service, _MONDAY)
    service.approve_proposal_superseding(proposal.proposal_id, old.memory_id)
    service.write_approved_proposals()

    audit = service.query_memory(_FRIDAY)

    # The old Friday memory is gone from the bank, so a Friday query can no
    # longer be grounded on it; the surviving Monday memory is a weekday flip
    # and the verifier rejects it rather than grounding.
    assert audit.verifier_verdict != VerificationVerdict.ACCEPT.value
    assert old.memory_id not in audit.cited_memory_ids


# 6. a deleted memory still cannot be cited. --------------------------------

def test_deleted_memory_cannot_be_cited(tmp_path):
    service = _service(tmp_path)
    entry = service.add_memory(_FRIDAY, source="seed")

    assert service.delete_memory(entry.memory_id)
    audit = service.query_memory(_FRIDAY)

    assert entry.memory_id not in audit.cited_memory_ids
    assert audit.verifier_verdict != VerificationVerdict.ACCEPT.value


# 7. a conflict proposal is not silently written. ---------------------------

def test_conflict_proposal_not_silently_written(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")
    note = _write_note(tmp_path, f"# Facts\n\n- {_MONDAY}\n")
    service.import_notes(note)
    proposal = _proposal_for(service, _MONDAY)

    # A plain approve must refuse a flagged conflict.
    assert service.approve_proposal(proposal.proposal_id) is False
    written = service.write_approved_proposals()

    assert written == []
    refreshed = service.proposals.get(proposal.proposal_id)
    assert refreshed.written is False


# 8. approve-as-new allows a conflicting memory only with explicit action. --

def test_approve_as_new_allows_conflict_explicitly(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")
    note = _write_note(tmp_path, f"# Facts\n\n- {_MONDAY}\n")
    service.import_notes(note)
    proposal = _proposal_for(service, _MONDAY)

    assert service.approve_proposal_as_new(proposal.proposal_id) is True
    written = service.write_approved_proposals()

    assert len(written) == 1
    # Nothing was superseded — both memories now exist in the ledger.
    assert all(e["status"] == "active" for e in service.export_ledger())


# 9. old ledger entries without supersession fields still load. -------------

def test_old_ledger_entries_without_fields_load(tmp_path):
    path = tmp_path / "legacy_ledger.jsonl"
    # A v1.2-era line: no ``supersedes`` / ``superseded_by`` keys at all.
    path.write_text(
        '{"memory_id": "mem-0001", "canonical_text": "legacy fact", '
        '"source": "seed", "tags": [], "status": "active", '
        '"created_at": "2024-01-01T00:00:00Z"}\n',
        encoding="utf-8",
    )

    ledger = MemoryLedger(path=str(path))
    entry = ledger.get("mem-0001")

    assert entry is not None
    assert entry.supersedes == []
    assert entry.superseded_by is None


# 10. list_conflicts returns pending conflict proposals. --------------------

def test_list_conflicts_returns_pending_conflicts(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")
    note = _write_note(tmp_path, f"# Facts\n\n- {_MONDAY}\n")
    service.import_notes(note)

    conflicts = service.list_conflicts()

    assert len(conflicts) == 1
    assert "monday" in conflicts[0].canonical_text.lower()
