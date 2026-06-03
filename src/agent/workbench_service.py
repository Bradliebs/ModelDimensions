"""Workbench service: a thin orchestration layer over the frozen v1.0 path.

This is the single seam the v1.1 workbench (CLI or Streamlit) talks to. It owns
one ``MemoryBank`` (the frozen concept-cell substrate) and one ``MemoryLedger``
(human-facing provenance), and it composes the existing, untouched v1.0 pieces:

    retrieve_candidates  ->  verify_candidate  ->  ground_accepted_candidates

It adds no geometry, no new firing rule, and no new verifier logic. Its only job
is to turn a single user action into a structured, auditable result and to keep
the bank and ledger consistent (same minted ``memory_id`` in both, deletion
removes the cell *and* marks the ledger entry).

Every query returns a :class:`QueryAudit` whose fields are exactly the audit
trail the workbench surfaces: ``candidate_retrieved``, ``verifier_verdict``,
``memory_used``, and ``refused``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from agent.candidate_retrieval import (
    ground_accepted_candidates,
    retrieve_candidates,
)
from agent.memory_ledger import MemoryLedger
from agent.memory_proposals import ProposalBatch
from agent.note_ingestion import extract_candidate_memories, load_text_file
from agent.orchestrator import DeterministicEncoder, EncoderProtocol, MemoryBank
from agent.proposal_queue import ProposalQueue
from agent.verifier import verify_candidate
from slm.schemas import VerificationVerdict, VerifiedMemoryCandidate

# Retrieval is intentionally permissive (epsilon large) so a stored memory is
# always offered as a candidate; safety is decided by the verifier, never by
# retrieval. These match scripts/demo_v1.py.
_EPSILON = 0.25
_RADIUS = 0.9
_K = 5


@dataclass
class CandidateView:
    """One retrieved candidate plus the verifier's verdict on it."""

    memory_id: str
    canonical_text: str
    activation: float
    rank: int
    verdict: str  # "accept" | "reject" | "ambiguous"


@dataclass
class QueryAudit:
    """The full, structured audit trail for a single query.

    The four load-bearing fields the workbench always shows are
    ``candidate_retrieved``, ``verifier_verdict``, ``memory_used`` and
    ``refused``. ``verifier_verdict`` is the *overall* verdict across candidates:
    ``accept`` if any candidate was accepted, else ``reject`` if any material
    mismatch was found, else ``ambiguous``, else ``none`` when nothing was
    retrieved.
    """

    query: str
    candidate_retrieved: bool
    verifier_verdict: str
    memory_used: bool
    refused: bool
    response_text: str
    cited_memory_ids: List[str] = field(default_factory=list)
    candidates: List[CandidateView] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _overall_verdict(views: List[CandidateView]) -> str:
    if not views:
        return "none"
    verdicts = {v.verdict for v in views}
    if "accept" in verdicts:
        return "accept"
    if "reject" in verdicts:
        return "reject"
    return "ambiguous"


class WorkbenchService:
    """Owns one memory bank + ledger and exposes the workbench operations."""

    def __init__(self, ledger_path: Optional[str | Path] = None,
                 encoder: Optional[EncoderProtocol] = None,
                 k: int = _K, fresh: bool = False,
                 queue_path: Optional[str | Path] = None):
        self.bank = MemoryBank(
            encoder or DeterministicEncoder(dim=64),
            epsilon=_EPSILON,
            radius=_RADIUS,
        )
        # The v1.0 bank is in-memory only and mints its own ids, so it cannot be
        # restored from a saved ledger. ``fresh`` starts the ledger empty (and
        # overwrites any prior file on first write) to keep bank and ledger ids
        # aligned; the workbench app uses it because each launch rebuilds the
        # bank from scratch.
        self.ledger = MemoryLedger(ledger_path, load_existing=not fresh)
        # The proposal queue is the v1.2 ingestion holding area: candidate
        # memories live here until a human approves them. It follows the same
        # ``fresh`` rule as the ledger so a relaunch does not double-apply a
        # queue whose approved memories were already written.
        self.proposals = ProposalQueue(queue_path, load_existing=not fresh)
        self.k = k

    # -- write --

    def add_memory(self, text: str, source: Optional[str] = None,
                   tags: Optional[List[str]] = None):
        """Write a memory into the bank and record it in the ledger.

        Returns the created :class:`~agent.memory_ledger.LedgerEntry`. The bank
        mints the ``memory_id``; the ledger records the same id so the two stay
        aligned.
        """
        rec = self.bank.write(text, source=source or "user", tags=tags or [])
        return self.ledger.add(rec.memory_id, rec.canonical_text,
                               source=source, tags=tags)

    # -- query --

    def query_memory(self, query_text: str) -> QueryAudit:
        """Run retrieve -> verify -> ground and return the audit trail."""
        candidates = retrieve_candidates(self.bank, query_text, self.k)

        verified: List[VerifiedMemoryCandidate] = []
        views: List[CandidateView] = []
        for cand in candidates:
            verdict = verify_candidate(query_text, cand.canonical_text)
            verified.append(
                VerifiedMemoryCandidate(candidate=cand, verdict=verdict)
            )
            views.append(CandidateView(
                memory_id=cand.memory_id,
                canonical_text=cand.canonical_text,
                activation=cand.activation,
                rank=cand.rank,
                verdict=verdict.value,
            ))

        response = ground_accepted_candidates(verified)

        return QueryAudit(
            query=query_text,
            candidate_retrieved=bool(candidates),
            verifier_verdict=_overall_verdict(views),
            memory_used=response.memory_used,
            refused=response.refused,
            response_text=response.text,
            cited_memory_ids=list(response.cited_memory_ids),
            candidates=views,
        )

    # -- delete --

    def delete_memory(self, memory_id: str) -> bool:
        """Remove a memory from the bank and mark the ledger entry deleted.

        Returns True if the memory existed and was removed. After deletion the
        cell is gone from the bank, so the memory can no longer be retrieved or
        cited; the ledger keeps a ``deleted`` record for audit.
        """
        removed = self.bank.delete(memory_id)
        marked = self.ledger.mark_deleted(memory_id)
        return removed or marked

    # -- ingestion / approval queue (v1.2) --

    def import_notes(self, path: str | Path) -> ProposalBatch:
        """Import a note, extract candidate memories, and queue them.

        Extraction is deterministic and offline (no LLM). The returned batch
        contains every candidate; the proposals are queued as ``pending`` and
        nothing is written to the bank until a human approves them.
        """
        source_file = str(path)
        text = load_text_file(path)
        batch = extract_candidate_memories(text, source_file)
        self.proposals.add_batch(batch)
        return batch

    def list_proposals(self, status: Optional[str] = "pending"):
        """List queued proposals, filtered by status (``None`` for all)."""
        return self.proposals.list(status)

    def approve_proposal(self, proposal_id: str) -> bool:
        """Approve a proposal so it will be written by ``write_approved``."""
        return self.proposals.approve(proposal_id)

    def reject_proposal(self, proposal_id: str) -> bool:
        """Reject a proposal so it is never written to the bank."""
        return self.proposals.reject(proposal_id)

    def edit_proposal(self, proposal_id: str, new_text: str) -> bool:
        """Edit a proposal's text (status becomes ``edited``, audit preserved)."""
        return self.proposals.edit(proposal_id, new_text)

    def write_approved_proposals(self) -> List:
        """Write every approved, not-yet-written proposal into the bank/ledger.

        Approved proposals reuse the frozen :meth:`add_memory` path, so the bank
        and ledger stay aligned and the verifier/grounding policy is unchanged.
        Rejected proposals are skipped entirely. Each written proposal is marked
        so a second call cannot create duplicate memories. Returns the created
        ledger entries.
        """
        written = []
        for proposal in self.proposals.approved_unwritten():
            tags = list(proposal.tags)
            if proposal.kind.value not in tags:
                tags.append(proposal.kind.value)
            entry = self.add_memory(
                proposal.canonical_text,
                source=proposal.source_file,
                tags=tags,
            )
            self.proposals.mark_written(proposal.proposal_id)
            written.append(entry)
        return written

    # -- inspect / export --

    def export_ledger(self) -> List[dict]:
        """Return the full ledger (active and deleted) as plain dicts."""
        return self.ledger.export()

    def export_proposals(self) -> List[dict]:
        """Return the whole proposal queue as plain dicts."""
        return self.proposals.export()

    def seed_from(self, seed_path: str | Path) -> List[str]:
        """Load seed memories from a JSONL file, writing each into the bank.

        Each line is a JSON object with at least ``canonical_text`` and optional
        ``source`` and ``tags`` (any ``memory_id`` in the file is ignored so the
        bank and ledger share freshly minted, aligned ids). Returns the list of
        minted memory ids.
        """
        path = Path(seed_path)
        minted: List[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data["canonical_text"]
                entry = self.add_memory(
                    text,
                    source=data.get("source"),
                    tags=list(data.get("tags", [])),
                )
                minted.append(entry.memory_id)
        return minted
