"""Tests for the v1.7 review service (offline, no downloads).

These cover the operator contract: the dashboard reports the active pack and the
review backlog; pending proposals and flagged conflicts are listed; the memory
table separates active / deleted / superseded / disputed; the knowledge table
carries source/domain/authority metadata; an audited query preserves
``memory_used`` / ``knowledge_used`` / ``backend_name``; exporting the active
pack produces a bundle; and the review service never bypasses the approval gates
or the grounding policy.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.project_packs import (  # noqa: E402
    MANIFEST_FILE,
    LEDGER_FILE,
    PackRegistry,
)
from agent.review_service import ReviewService  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402

_FRIDAY = "the supplier delivery is on Friday afternoon"
_MONDAY = "the supplier delivery is on Monday afternoon"
_KNOWLEDGE = (
    "# Coding reference\n\n"
    "List comprehensions build a list from an iterable in one expression.\n"
)


def _service(tmp_path) -> WorkbenchService:
    """A global (pack-less) workbench service rooted in ``tmp_path``."""
    return WorkbenchService(
        ledger_path=str(tmp_path / "ledger.jsonl"),
        queue_path=str(tmp_path / "queue.jsonl"),
        knowledge_path=str(tmp_path / "knowledge.jsonl"),
    )


def _pack_service(tmp_path):
    """A pack-bound service plus its registry, rooted in ``tmp_path``."""
    registry = PackRegistry(tmp_path / "packs")
    pack = registry.create_pack("review-demo")
    registry.set_active_pack(pack.pack_id)
    service = WorkbenchService.from_pack(pack, registry=registry)
    return service, registry, pack


def _write_note(tmp_path, text: str, name: str = "note.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _proposal_for(service, text: str):
    target = text.lower().rstrip(".")
    for p in service.list_proposals(status=None):
        if p.canonical_text.lower().rstrip(".") == target:
            return p
    raise AssertionError(f"no proposal matched: {text!r}")


# 1. dashboard state includes the active pack. ------------------------------

def test_dashboard_state_includes_active_pack(tmp_path):
    service, registry, pack = _pack_service(tmp_path)
    review = ReviewService(service, registry)

    state = review.get_dashboard_state()

    assert state["active_pack"] is not None
    assert state["active_pack"]["pack_id"] == pack.pack_id
    assert "memory_counts" in state
    assert "knowledge_backend" in state


# 2. pending proposals are listed. ------------------------------------------

def test_pending_proposals_are_listed(tmp_path):
    service = _service(tmp_path)
    review = ReviewService(service)
    note = _write_note(tmp_path, f"# Facts\n\n- {_FRIDAY}\n")
    service.import_notes(note)

    pending = review.list_pending_proposals()

    assert pending
    assert any(_FRIDAY.lower() in p["canonical_text"].lower() for p in pending)
    assert review.get_dashboard_state()["pending_proposals"] == len(pending)


# 3. conflicts are listed. --------------------------------------------------

def test_conflicts_are_listed(tmp_path):
    service = _service(tmp_path)
    review = ReviewService(service)
    service.add_memory(_FRIDAY, source="seed")
    note = _write_note(tmp_path, f"# Facts\n\n- {_MONDAY}\n")
    service.import_notes(note)

    conflicts = review.list_conflicts()

    assert len(conflicts) == 1
    assert "monday" in conflicts[0]["canonical_text"].lower()
    assert review.get_dashboard_state()["conflicts"] == 1


# 4. memory table separates active / deleted / superseded / disputed. -------

def test_memory_table_separates_statuses(tmp_path):
    service = _service(tmp_path)
    review = ReviewService(service)

    active = service.add_memory("the build runs nightly at midnight", source="seed")
    deleted = service.add_memory("the staging server is eu-west-1", source="seed")
    disputed = service.add_memory("the api owner is team blue", source="seed")
    old = service.add_memory(_FRIDAY, source="seed")

    service.delete_memory(deleted.memory_id)
    service.dispute_memory(disputed.memory_id)
    # Supersede the Friday memory via the approval path.
    note = _write_note(tmp_path, f"# Facts\n\n- {_MONDAY}\n")
    service.import_notes(note)
    proposal = _proposal_for(service, _MONDAY)
    service.approve_proposal_superseding(proposal.proposal_id, old.memory_id)
    service.write_approved_proposals()

    table = review.get_memory_table()

    assert set(table) == {"active", "deleted", "superseded", "disputed"}
    active_ids = {r["memory_id"] for r in table["active"]}
    assert active.memory_id in active_ids
    assert deleted.memory_id in {r["memory_id"] for r in table["deleted"]}
    assert disputed.memory_id in {r["memory_id"] for r in table["disputed"]}
    assert old.memory_id in {r["memory_id"] for r in table["superseded"]}


# 5. knowledge table includes source metadata. -----------------------------

def test_knowledge_table_includes_source_metadata(tmp_path):
    service = _service(tmp_path)
    review = ReviewService(service)
    doc = _write_note(tmp_path, _KNOWLEDGE, name="reference.md")
    service.import_knowledge(
        doc, domain="coding", authority="official",
        source_name="Python Reference", version="3.12")

    table = review.get_knowledge_table()

    assert len(table) == 1
    row = table[0]
    assert row["source_name"] == "Python Reference"
    assert row["domain"] == "coding"
    assert row["authority"] == "official"
    assert row["version"] == "3.12"


# 6. an audited query preserves memory/knowledge/backend audit fields. ------

def test_audited_query_preserves_audit_fields(tmp_path):
    service = _service(tmp_path)
    review = ReviewService(service)
    service.add_memory(_FRIDAY, source="seed")

    audit = review.run_audited_query(_FRIDAY)

    assert "memory_used" in audit
    assert "knowledge_used" in audit
    assert audit["backend_name"] == service.knowledge_backend_name()
    # The exact stored fact grounds on memory.
    assert audit["memory_used"] is True


# 7. exporting the active pack produces a bundle with the manifest. ---------

def test_export_active_pack_creates_bundle(tmp_path):
    service, registry, pack = _pack_service(tmp_path)
    review = ReviewService(service, registry)
    service.add_memory(_FRIDAY, source="seed")

    out = review.export_active_pack(tmp_path / "bundle.zip")

    assert out.exists()
    with zipfile.ZipFile(out, "r") as bundle:
        names = set(bundle.namelist())
    assert MANIFEST_FILE in names
    assert LEDGER_FILE in names


# 8. the review service does not bypass approval or grounding. --------------

def test_review_service_does_not_bypass_approval_or_grounding(tmp_path):
    service = _service(tmp_path)
    review = ReviewService(service)
    service.add_memory(_FRIDAY, source="seed")
    note = _write_note(tmp_path, f"# Facts\n\n- {_MONDAY}\n")
    service.import_notes(note)
    proposal = _proposal_for(service, _MONDAY)

    # A plain approve of a flagged conflict must be refused, exactly as the
    # workbench refuses it — the review service adds no override.
    result = review.review_proposal(proposal.proposal_id, "approve")
    assert result["ok"] is False
    assert review.write_approved() == []

    # Grounding policy is intact: a never-stored fact is not grounded.
    audit = review.run_audited_query("the warehouse forklift is battery powered")
    assert audit["memory_used"] is False
