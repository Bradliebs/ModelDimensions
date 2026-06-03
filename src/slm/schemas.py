"""Pydantic schemas for the SLM controller layer.

These are the only data contracts that cross the boundary between the
(untrusted, generative) SLM and the (trusted, geometric) concept-cell memory.
Keeping them explicit and validated is what makes the layer auditable: every
decision the SLM makes is a typed object that can be logged and inspected.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class IntentType(str, Enum):
    """What the user is asking the memory system to do."""

    QUERY = "query"     # retrieve from memory
    WRITE = "write"     # store a new memory
    BIND = "bind"       # bind several existing memories into one group
    DELETE = "delete"   # remove a memory by id
    UNKNOWN = "unknown"  # cannot be classified — must be refused


class MemoryIntent(BaseModel):
    """Classified intent for one user turn."""

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class MemoryWriteCandidate(BaseModel):
    """A compressed, canonicalised piece of text proposed for storage.

    The SLM proposes this; the bank decides the final id and threshold. The
    SLM is not allowed to invent the embedding — only the canonical text.
    """

    canonical_text: str = Field(min_length=1)
    source: str = "user"
    tags: List[str] = Field(default_factory=list)
    # Optional hint for the cell threshold margin; the bank may ignore it.
    suggested_epsilon: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MemoryBindCandidate(BaseModel):
    """A proposal to bind several existing memories under one group id."""

    memory_ids: List[str] = Field(min_length=2)
    bound_group_id: str = Field(min_length=1)
    rationale: str = ""


class MemoryQueryResult(BaseModel):
    """Outcome of querying the concept-cell bank with one embedded query.

    `silent` is True exactly when no cell fired. This is the load-bearing
    signal the response policy uses to refuse hallucinated answers.
    """

    query_text: str
    fired_memory_ids: List[str] = Field(default_factory=list)
    margins: List[float] = Field(default_factory=list)
    silent: bool = True


class SLMControllerDecision(BaseModel):
    """The full, typed decision the controller hands to the orchestrator."""

    intent: MemoryIntent
    query_text: Optional[str] = None
    write_candidate: Optional[MemoryWriteCandidate] = None
    bind_candidate: Optional[MemoryBindCandidate] = None
    delete_memory_id: Optional[str] = None


class VerificationVerdict(str, Enum):
    """Whether a retrieved candidate actually preserves the queried fact.

    Used by the v1.0-rc2 candidate-recall + verifier layer. ``ACCEPT`` is the
    only verdict the grounding path is allowed to act on; ``REJECT`` and
    ``AMBIGUOUS`` both lead to refusal.
    """

    ACCEPT = "accept"        # candidate preserves the fact; safe to ground
    REJECT = "reject"        # material mismatch detected; must not ground
    AMBIGUOUS = "ambiguous"  # equivalence unclear; refuse, do not ground


class MemoryCandidate(BaseModel):
    """A retrieved candidate cell. Retrieval is NOT grounding.

    A candidate is a fast, explicit pointer into the concept-cell substrate. It
    is deliberately *not* marked as used memory: nothing here is grounded until
    a verifier accepts it. ``threshold_status`` records whether the activation
    cleared the cell's own firing threshold, for auditing only.
    """

    memory_id: str
    canonical_text: str
    activation: float
    rank: int = Field(ge=1)
    threshold_status: str  # "above_threshold" | "below_threshold"


class VerifiedMemoryCandidate(BaseModel):
    """A candidate plus the verifier's verdict on whether it preserves the fact.

    ``source`` records which verifier produced the verdict
    ("deterministic", "slm", or "combined"). Only ``verdict == ACCEPT`` may be
    grounded.
    """

    candidate: MemoryCandidate
    verdict: VerificationVerdict
    reason: str = ""
    source: str = "deterministic"


class GroundedResponse(BaseModel):
    """The final response returned to the user.

    Invariants enforced by the response policy:
      - if `memory_used` is True, `cited_memory_ids` is non-empty.
      - if `refused` is True, `memory_used` is False and `cited_memory_ids`
        is empty (no memory-derived facts may accompany a refusal).
    """

    text: str
    cited_memory_ids: List[str] = Field(default_factory=list)
    memory_used: bool = False
    refused: bool = False

    @model_validator(mode="after")
    def _check_invariants(self) -> "GroundedResponse":
        # Memory cannot be "used" without naming which memories were used.
        if self.memory_used and not self.cited_memory_ids:
            raise ValueError(
                "memory_used=True requires at least one cited memory id"
            )
        # A refusal must carry no memory-derived content.
        if self.refused and (self.memory_used or self.cited_memory_ids):
            raise ValueError(
                "a refused response must not use or cite any memory"
            )
        return self
