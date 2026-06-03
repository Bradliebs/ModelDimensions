"""Candidate retrieval over the concept-cell substrate (v1.0-rc2).

Retrieval returns the top-k cells by activation so a verifier can decide which,
if any, actually preserve the queried fact. This is deliberately separated from
grounding:

  * ``retrieve_candidates`` reads the bank through its public surface only
    (``records()``, ``encoder``, ``radius``) and reproduces the exact activation
    the bank itself would compute (unit cell weight dotted with the
    radius-scaled query). It does not mutate the bank and changes no geometry.
  * Candidates are returned as ``MemoryCandidate`` objects and are **never**
    marked as used memory. Retrieving a candidate is not grounding.
  * ``ground_accepted_candidates`` builds a grounded response from verified
    candidates, citing only those with an ``ACCEPT`` verdict. It composes the
    existing ``GroundedResponse`` invariants; it does not modify the frozen
    response policy.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

import numpy as np

from slm.schemas import (
    GroundedResponse,
    MemoryCandidate,
    VerificationVerdict,
    VerifiedMemoryCandidate,
)

if TYPE_CHECKING:  # avoid a runtime import cycle with orchestrator
    from .orchestrator import MemoryBank


def retrieve_candidates(bank: "MemoryBank", query_text: str,
                        k: int = 5) -> List[MemoryCandidate]:
    """Return the top-``k`` candidate cells for ``query_text`` by activation.

    The activation matches the bank's own firing computation exactly: each
    stored vector sits at radius ``r``, so the cell weight is ``x_i/||x_i||``
    (unit) and the query is scaled to radius ``r``; ``activation = w_i . q``.
    ``threshold_status`` reports whether that activation cleared the cell's
    stored threshold, for auditing only -- it does not gate retrieval.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    recs = bank.records()
    if not recs:
        return []

    q_raw = np.asarray(bank.encoder.encode_one(query_text), dtype=np.float64)
    qn = np.linalg.norm(q_raw)
    if qn < 1e-12:
        return []
    q = q_raw / qn * bank.radius

    scored = []
    for rec in recs:
        x = np.asarray(rec.vector, dtype=np.float64)
        xn = np.linalg.norm(x)
        if xn < 1e-12:
            continue
        w = x / xn
        scored.append((float(w @ q), rec))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    candidates: List[MemoryCandidate] = []
    for rank, (act, rec) in enumerate(scored[:k], start=1):
        status = ("above_threshold" if act > rec.threshold
                  else "below_threshold")
        candidates.append(MemoryCandidate(
            memory_id=rec.memory_id,
            canonical_text=rec.canonical_text,
            activation=round(act, 6),
            rank=rank,
            threshold_status=status,
        ))
    return candidates


def ground_accepted_candidates(
        verified: Sequence[VerifiedMemoryCandidate]) -> GroundedResponse:
    """Build a grounded response citing only ACCEPT-verdict candidates.

    Any ``REJECT`` or ``AMBIGUOUS`` candidate is excluded, so a near-miss that
    fired loudly but failed verification can never reach the user as memory. If
    nothing is accepted the response refuses (``refused=True``) and cites
    nothing, satisfying the existing ``GroundedResponse`` invariants.
    """
    accepted = [v for v in verified
                if v.verdict is VerificationVerdict.ACCEPT]
    if not accepted:
        return GroundedResponse(
            text="Memory is silent: no candidate was verified as a match.",
            cited_memory_ids=[],
            memory_used=False,
            refused=True,
        )

    lines: List[str] = []
    cited: List[str] = []
    for v in accepted:
        cid = v.candidate.memory_id
        lines.append(f"- ({cid}) {v.candidate.canonical_text}")
        cited.append(cid)

    return GroundedResponse(
        text="From verified memory:\n" + "\n".join(lines),
        cited_memory_ids=cited,
        memory_used=True,
        refused=False,
    )
