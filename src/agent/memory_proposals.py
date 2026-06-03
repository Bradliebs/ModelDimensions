"""Memory proposals: the candidate-memory data model for v1.2 ingestion.

A *proposal* is a candidate memory extracted from an imported note. It is **not**
a memory: nothing in this module touches the concept-cell substrate, the bank,
or the ledger. A proposal only becomes a real memory when a human approves it and
the workbench writes it through the frozen v1.0 path (``add_memory``).

This mirrors the ``LedgerEntry`` style in ``agent.memory_ledger``: small
dataclasses with explicit ``to_json`` / ``from_json`` so the proposal queue can
persist one JSON object per line, fully readable by eye.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProposalStatus(str, Enum):
    """Lifecycle of a proposal in the approval queue.

    A proposal starts ``PENDING``. A human moves it to ``APPROVED`` (write it),
    ``REJECTED`` (never write it), or ``EDITED`` (text changed, still needs a
    final approval before it is written).
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


class ProposalKind(str, Enum):
    """A coarse, rule-assigned category for a candidate memory."""

    DECISION = "decision"
    ASSUMPTION = "assumption"
    RISK = "risk"
    REQUIREMENT = "requirement"
    RESULT = "result"
    LIMITATION = "limitation"
    NEXT_STEP = "next_step"
    FACT = "fact"


@dataclass
class MemoryProposal:
    """One candidate memory extracted from a note, awaiting human review.

    ``written`` tracks whether an approved proposal has already been written to
    the bank/ledger, so re-running ``write-approved`` cannot create duplicate
    memories. ``edited_from`` keeps the pre-edit text so an edit is auditable.
    """

    proposal_id: str
    canonical_text: str
    source_file: str
    source_section: Optional[str] = None
    source_line_start: Optional[int] = None
    source_line_end: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    kind: ProposalKind = ProposalKind.FACT
    confidence: float = 0.5
    reason: str = ""
    status: ProposalStatus = ProposalStatus.PENDING
    edited_from: Optional[str] = None
    written: bool = False
    created_at: str = field(default_factory=_utc_now_iso)

    def to_json(self) -> str:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "MemoryProposal":
        data = json.loads(line)
        return cls(
            proposal_id=data["proposal_id"],
            canonical_text=data["canonical_text"],
            source_file=data["source_file"],
            source_section=data.get("source_section"),
            source_line_start=data.get("source_line_start"),
            source_line_end=data.get("source_line_end"),
            tags=list(data.get("tags", [])),
            kind=ProposalKind(data.get("kind", ProposalKind.FACT.value)),
            confidence=float(data.get("confidence", 0.5)),
            reason=data.get("reason", ""),
            status=ProposalStatus(data.get("status",
                                           ProposalStatus.PENDING.value)),
            edited_from=data.get("edited_from"),
            written=bool(data.get("written", False)),
            created_at=data.get("created_at", _utc_now_iso()),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        return data


@dataclass
class ProposalBatch:
    """The set of proposals produced by a single import of one note."""

    source_file: str
    proposals: List[MemoryProposal] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)

    def __len__(self) -> int:
        return len(self.proposals)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "created_at": self.created_at,
            "count": len(self.proposals),
            "proposals": [p.to_dict() for p in self.proposals],
        }
