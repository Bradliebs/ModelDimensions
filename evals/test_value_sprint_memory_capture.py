"""Tests for v2.5B memory capture from pack gaps (proposal-only, opt-in).

These tests assert the safety boundary of the v2.5B adapter: a missing-decision
pack gap can be turned into a *PENDING* :class:`MemoryProposal` and queued for
human review, but it is never written to the MemoryLedger until a human approves
it and the workbench writes it through the frozen ``add_memory`` path.

Nothing here changes verifier, grounding, lifecycle, refusal, pack-isolation,
SLM, AnswerGuard, EvidenceRanker, or SufficiencyGate behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.memory_proposals import ProposalKind, ProposalStatus  # noqa: E402
from agent.proposal_queue import ProposalQueue  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from agent.value_sprint_harness import (  # noqa: E402
    EXPECTED_GROUNDED_USEFUL,
    EXPECTED_PACK_GAP,
    SprintQuery,
    emit_memory_proposals,
    extract_pack_gap,
    pack_gap_to_memory_proposal,
    run_query,
    run_sprint,
)

_DECISION_QUERY = "What did we decide about the data residency review vendor?"


def _memory_service() -> WorkbenchService:
    """A bare, memory-only service (no knowledge pack)."""
    return WorkbenchService()


def _memory_gap() -> object:
    """A missing-decision (memory_proposal) pack gap."""
    spec = SprintQuery(
        query=_DECISION_QUERY,
        note="data residency vendor decision",
        category="project_decision",
        expected_outcome=EXPECTED_PACK_GAP)
    gap = extract_pack_gap(spec, grounded=False)
    assert gap is not None
    assert gap.suggested_source_type == "memory_proposal"
    return gap


def _knowledge_gap() -> object:
    """A missing-knowledge (knowledge_source) pack gap."""
    spec = SprintQuery(
        query="What is the syntax for a Power Fx filter expression?",
        note="power fx filter",
        category="coding_howto",
        expected_outcome=EXPECTED_GROUNDED_USEFUL)
    gap = extract_pack_gap(spec, grounded=False)
    assert gap is not None
    assert gap.suggested_source_type == "knowledge_source"
    return gap


# 1. a memory_proposal gap converts to a PENDING proposal. --------------------

def test_memory_proposal_gap_converts_to_pending():
    proposal = pack_gap_to_memory_proposal(_memory_gap())

    assert proposal is not None
    assert proposal.status == ProposalStatus.PENDING
    assert proposal.written is False
    assert proposal.kind == ProposalKind.DECISION
    assert abs(proposal.confidence - 0.3) < 1e-9
    assert proposal.source_file == "(value-sprint pack gap)"
    assert proposal.canonical_text  # the suggested memory text
    assert "value-sprint" in proposal.reason
    assert "missing-decision" in proposal.reason


# 2. a knowledge_source gap is not routed (stays report-only). ----------------

def test_knowledge_source_gap_returns_none():
    assert pack_gap_to_memory_proposal(_knowledge_gap()) is None


# 3. queuing a proposal does not write to the MemoryLedger. -------------------

def test_queue_add_does_not_write_ledger(tmp_path):
    service = _memory_service()
    before = len(service.ledger.entries())

    proposal = pack_gap_to_memory_proposal(_memory_gap())
    queue = ProposalQueue(tmp_path / "sprint_proposals.jsonl")
    assert queue.add(proposal) is True

    # The proposal is queued ...
    assert len(queue.list_pending()) == 1
    # ... but no memory was written.
    assert len(service.ledger.entries()) == before


# 4. the decision is still refused before any approval/write. -----------------

def test_query_refuses_before_approval(tmp_path):
    service = _memory_service()
    proposal = pack_gap_to_memory_proposal(_memory_gap())
    ProposalQueue(tmp_path / "sprint_proposals.jsonl").add(proposal)

    # Queuing a proposal does not make the decision answerable.
    result = service.answer_query(_DECISION_QUERY)
    assert result.refused is True


# 5. explicit approval + write is required before a ledger entry appears. -----

def test_approval_and_write_required_before_ledger_entry():
    service = _memory_service()
    before = len(service.ledger.entries())

    proposal = pack_gap_to_memory_proposal(_memory_gap())
    assert service.proposals.add(proposal) is True

    # A pending (un-approved) proposal writes nothing.
    assert service.write_approved_proposals() == []
    assert len(service.ledger.entries()) == before

    # Only after explicit approval does the write produce a ledger entry.
    assert service.approve_proposal(proposal.proposal_id) is True
    written = service.write_approved_proposals()
    assert len(written) == 1
    assert len(service.ledger.entries()) == before + 1


# 6. the adapter / emit path is idempotent across re-runs. --------------------

def test_emit_memory_proposals_is_idempotent(tmp_path):
    service = _memory_service()
    rows = [
        run_query(service, SprintQuery(
            query=_DECISION_QUERY, note="data residency vendor decision",
            category="project_decision", expected_outcome=EXPECTED_PACK_GAP),
            retrieval_backend="deterministic"),
    ]
    queue_path = tmp_path / "sprint_proposals.jsonl"

    first = emit_memory_proposals(rows, queue_path=queue_path)
    assert len(first) == 1
    assert queue_path.exists()

    # Re-running adds nothing (deterministic id + queue dedup).
    second = emit_memory_proposals(rows, queue_path=queue_path)
    assert second == []
    assert len(ProposalQueue(queue_path).list_pending()) == 1


# 6b. knowledge-source gaps are skipped by the emit path. ---------------------

def test_emit_skips_knowledge_source_gaps(tmp_path):
    service = _memory_service()
    rows = [
        run_query(service, SprintQuery(
            query="What is the syntax for a Power Fx filter expression?",
            note="power fx filter", category="coding_howto",
            expected_outcome=EXPECTED_GROUNDED_USEFUL),
            retrieval_backend="deterministic"),
    ]
    queue_path = tmp_path / "sprint_proposals.jsonl"

    added = emit_memory_proposals(rows, queue_path=queue_path)
    assert added == []


# 7. a normal sprint run (no opt-in) creates no proposal queue. ---------------

def test_default_sprint_creates_no_proposal_queue(tmp_path):
    service = _memory_service()
    queries = [
        SprintQuery(query=_DECISION_QUERY, note="data residency vendor decision",
                    category="project_decision",
                    expected_outcome=EXPECTED_PACK_GAP),
    ]
    queue_path = tmp_path / "sprint_proposals.jsonl"

    # The default sprint path is report-only: it never touches a proposal queue.
    rows = run_sprint(service, queries, retrieval_backend="deterministic")
    assert any(r.pack_gap_detected for r in rows)
    assert not queue_path.exists()
