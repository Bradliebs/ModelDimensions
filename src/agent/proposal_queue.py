"""JSONL-backed approval queue for candidate memories.

The queue holds :class:`MemoryProposal` objects awaiting human review. It mirrors
``MemoryLedger``: an in-memory dict that writes through to a JSONL file on every
mutation, ``path`` may be ``None`` for a purely in-memory queue, and a
``load_existing`` flag controls whether a prior file is read on construction.

The queue never touches the concept-cell substrate. Approving a proposal only
changes its status here; writing the approved memory into the bank/ledger is the
workbench service's job. Rejected proposals are kept (status ``rejected``) so the
review itself stays auditable — they are simply never written.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from agent.memory_proposals import (
    MemoryProposal,
    ProposalBatch,
    ProposalStatus,
)


class ProposalQueue:
    """An in-memory proposal queue that writes through to a JSONL file."""

    def __init__(self, path: Optional[str | Path] = None, *,
                 load_existing: bool = True):
        self.path = Path(path) if path is not None else None
        self._proposals: Dict[str, MemoryProposal] = {}
        if load_existing and self.path is not None and self.path.exists():
            self.load()

    # -- mutating operations --

    def add(self, proposal: MemoryProposal) -> bool:
        """Add a proposal. Returns False if its id is already queued.

        A duplicate id means the same text at the same source location, so it is
        skipped (idempotent re-import) rather than raising or silently merging
        different content.
        """
        if proposal.proposal_id in self._proposals:
            return False
        self._proposals[proposal.proposal_id] = proposal
        self._flush()
        return True

    def add_batch(self, batch: ProposalBatch) -> List[MemoryProposal]:
        """Add every proposal in a batch, skipping ids already queued.

        Returns the proposals that were newly added.
        """
        added: List[MemoryProposal] = []
        for proposal in batch.proposals:
            if proposal.proposal_id not in self._proposals:
                self._proposals[proposal.proposal_id] = proposal
                added.append(proposal)
        self._flush()
        return added

    def approve(self, proposal_id: str) -> bool:
        """Mark a proposal approved. Returns False if unknown or rejected."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.status == ProposalStatus.REJECTED:
            return False
        proposal.status = ProposalStatus.APPROVED
        self._flush()
        return True

    def reject(self, proposal_id: str) -> bool:
        """Mark a proposal rejected. Returns False if unknown."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return False
        proposal.status = ProposalStatus.REJECTED
        self._flush()
        return True

    def edit(self, proposal_id: str, new_text: str) -> bool:
        """Edit a proposal's text. Returns False if unknown or rejected.

        The pre-edit text is preserved in ``edited_from`` and the status becomes
        ``edited`` (the proposal still needs a final approval before it is
        written).
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.status == ProposalStatus.REJECTED:
            return False
        proposal.edited_from = proposal.canonical_text
        proposal.canonical_text = new_text.strip()
        proposal.status = ProposalStatus.EDITED
        self._flush()
        return True

    def approve_all(self) -> int:
        """Approve every pending or edited proposal. Returns the count."""
        count = 0
        for proposal in self._proposals.values():
            if proposal.status in (ProposalStatus.PENDING,
                                   ProposalStatus.EDITED):
                proposal.status = ProposalStatus.APPROVED
                count += 1
        if count:
            self._flush()
        return count

    def reject_all(self) -> int:
        """Reject every pending or edited proposal. Returns the count."""
        count = 0
        for proposal in self._proposals.values():
            if proposal.status in (ProposalStatus.PENDING,
                                   ProposalStatus.EDITED):
                proposal.status = ProposalStatus.REJECTED
                count += 1
        if count:
            self._flush()
        return count

    def mark_written(self, proposal_id: str) -> bool:
        """Record that an approved proposal has been written to the bank."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return False
        proposal.written = True
        self._flush()
        return True

    def set_lifecycle(self, proposal_id: str, verdict: str,
                      candidate_id: Optional[str], reason: str) -> bool:
        """Attach lifecycle analysis (verdict / matched memory / reason)."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return False
        proposal.lifecycle_verdict = verdict
        proposal.lifecycle_candidate_id = candidate_id
        proposal.lifecycle_reason = reason
        self._flush()
        return True

    def set_supersedes(self, proposal_id: str,
                       old_memory_id: Optional[str]) -> bool:
        """Record the memory id this proposal will supersede when written."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return False
        proposal.supersedes_memory_id = old_memory_id
        self._flush()
        return True

    # -- read operations --

    def get(self, proposal_id: str) -> Optional[MemoryProposal]:
        return self._proposals.get(proposal_id)

    def all(self) -> List[MemoryProposal]:
        """Every proposal in insertion order."""
        return list(self._proposals.values())

    def list(self, status: Optional[str] = None) -> List[MemoryProposal]:
        """Proposals filtered by status string, or all when ``status`` is None."""
        if status is None:
            return self.all()
        return [p for p in self._proposals.values() if p.status.value == status]

    def list_pending(self) -> List[MemoryProposal]:
        return self.list(ProposalStatus.PENDING.value)

    def approved_unwritten(self) -> List[MemoryProposal]:
        """Approved proposals that have not yet been written to the bank."""
        return [p for p in self._proposals.values()
                if p.status == ProposalStatus.APPROVED and not p.written]

    def export(self) -> List[dict]:
        """Return the whole queue as plain dicts."""
        return [p.to_dict() for p in self._proposals.values()]

    # -- persistence --

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for proposal in self._proposals.values():
                fh.write(proposal.to_json() + "\n")

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        self._proposals.clear()
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                proposal = MemoryProposal.from_json(line)
                self._proposals[proposal.proposal_id] = proposal

    def export_to(self, target: str | Path) -> Path:
        """Write the current queue to ``target`` and return the path."""
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for proposal in self._proposals.values():
                fh.write(proposal.to_json() + "\n")
        return path
