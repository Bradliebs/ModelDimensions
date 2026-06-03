"""Memory lifecycle analysis: duplicate and conflict detection (v1.3).

This module decides how a *candidate* memory (a proposal) relates to the
memories already in the ledger, so a reviewer can see — before approving — that a
proposal is a duplicate of, or a conflict with, something already stored.

It reuses the frozen v1.0 pieces and adds no geometry: duplicate detection is
lexical, and conflict detection delegates to the deterministic
``verify_candidate`` (the same REJECT-on-material-mismatch rule the grounding
path uses). A material flip — a different date, number, entity, modal strength,
negation, or known antonym — that the verifier already rejects between a proposal
and an active memory is surfaced here as a conflict.

Nothing in this module mutates the bank or the ledger; it only reports.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, List, Optional, Set, Tuple

from agent.verifier import verify_candidate
from slm.schemas import VerificationVerdict

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# A small, generic stopword set so lexical overlap reflects content words. Kept
# local (not imported from the verifier) so this module stays self-contained.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "for", "of", "on", "in", "to", "at", "and", "or", "our", "we", "this",
    "that", "by", "with", "as", "it", "its", "their", "there", "will",
    "have", "has", "had", "from",
})

# Lexical-overlap thresholds (Jaccard over content tokens).
_DUPLICATE_OVERLAP = 0.6   # near-identical wording, no factual mismatch
_CONFLICT_OVERLAP = 0.4    # same subject with a material flip -> hard conflict


class LifecycleVerdict(str, Enum):
    """How a proposal relates to the existing memories."""

    NEW = "new"
    DUPLICATE = "duplicate"
    POSSIBLE_DUPLICATE = "possible_duplicate"
    POSSIBLE_CONFLICT = "possible_conflict"
    CONFLICT = "conflict"


# Severity order, worst first, used when a proposal touches several memories.
_SEVERITY = {
    LifecycleVerdict.CONFLICT: 4,
    LifecycleVerdict.DUPLICATE: 3,
    LifecycleVerdict.POSSIBLE_CONFLICT: 2,
    LifecycleVerdict.POSSIBLE_DUPLICATE: 1,
    LifecycleVerdict.NEW: 0,
}


@dataclass
class LifecycleCheck:
    """The lifecycle finding for a proposal against one existing memory."""

    proposal_id: str
    candidate_memory_id: Optional[str]
    verdict: LifecycleVerdict
    reason: str
    verifier_verdict: Optional[str] = None
    activation: Optional[float] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data


# ---------- lexical helpers ----------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!?,;:")


def _content_tokens(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)
            if t.lower() not in _STOPWORDS}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def lexical_overlap(a: str, b: str) -> float:
    """Content-token Jaccard overlap between two texts (stopwords removed).

    The same lexical measure the duplicate/conflict checks use, exposed so other
    components (such as the historical-memory scan) can rank by overlap without
    reaching into private helpers.
    """
    return _jaccard(_content_tokens(a), _content_tokens(b))


# ---------- public checks ----------

def check_duplicate(proposal_text: str,
                    existing_memory_text: str) -> Tuple[LifecycleVerdict, str]:
    """Decide whether a proposal duplicates an existing memory.

    Exact normalised text is a ``DUPLICATE``. High lexical overlap with no
    factual mismatch (the verifier does not reject) is a ``POSSIBLE_DUPLICATE``.
    Anything else is ``NEW`` (with respect to this one memory).
    """
    if _norm(proposal_text) == _norm(existing_memory_text):
        return LifecycleVerdict.DUPLICATE, "exact text match"

    overlap = _jaccard(_content_tokens(proposal_text),
                       _content_tokens(existing_memory_text))
    if overlap >= _DUPLICATE_OVERLAP:
        # Only call it a duplicate if there is no material mismatch — a flip is
        # a conflict, not a duplicate.
        verdict = verify_candidate(proposal_text, existing_memory_text)
        if verdict != VerificationVerdict.REJECT:
            return (LifecycleVerdict.POSSIBLE_DUPLICATE,
                    f"high lexical overlap ({overlap:.2f}), no factual mismatch")
    return LifecycleVerdict.NEW, ""


def check_conflict(
    proposal_text: str,
    existing_memory_text: str,
    verifier: Callable[[str, str], VerificationVerdict] = verify_candidate,
) -> Tuple[LifecycleVerdict, str, Optional[str]]:
    """Decide whether a proposal conflicts with an existing memory.

    Delegates to the deterministic verifier: a ``REJECT`` between the proposal
    and an active memory means a material flip (date / number / entity / modal /
    negation / antonym). With high lexical overlap that is a hard ``CONFLICT``;
    with lower overlap it is a ``POSSIBLE_CONFLICT``. No rejection is ``NEW``.

    Returns ``(verdict, reason, verifier_verdict_value)``.
    """
    verdict = verifier(proposal_text, existing_memory_text)
    if verdict == VerificationVerdict.REJECT:
        overlap = _jaccard(_content_tokens(proposal_text),
                           _content_tokens(existing_memory_text))
        if overlap >= _CONFLICT_OVERLAP:
            return (LifecycleVerdict.CONFLICT,
                    f"verifier REJECT with high overlap ({overlap:.2f}): "
                    "material flip on the same subject",
                    verdict.value)
        return (LifecycleVerdict.POSSIBLE_CONFLICT,
                f"verifier REJECT with low overlap ({overlap:.2f}): "
                "possible flip, different subject",
                verdict.value)
    return LifecycleVerdict.NEW, "", verdict.value


def analyse_proposal_against_ledger(
    proposal,
    ledger,
    candidate_retriever: Optional[Callable[[str], list]] = None,
    verifier: Callable[[str, str], VerificationVerdict] = verify_candidate,
) -> LifecycleCheck:
    """Analyse one proposal against the active memories in ``ledger``.

    ``candidate_retriever`` (optional) is a callable mapping a query string to a
    list of retrieved candidates (objects with ``memory_id``, ``canonical_text``
    and ``activation``); when provided it supplies the activation of the matched
    memory and focuses the comparison on the geometrically nearest memories.
    Without it, the proposal is compared against every active ledger entry.

    Returns the single most severe :class:`LifecycleCheck`. A proposal that
    matches nothing is ``NEW`` with no candidate memory.
    """
    text = proposal.canonical_text
    active = {e.memory_id: e for e in ledger.get_active_memories()}

    scored: List[Tuple[str, str, Optional[float]]] = []
    if candidate_retriever is not None:
        for cand in candidate_retriever(text):
            if cand.memory_id in active:
                scored.append((cand.memory_id, cand.canonical_text,
                               cand.activation))
    if not scored:
        scored = [(e.memory_id, e.canonical_text, None)
                  for e in active.values()]

    best: Optional[LifecycleCheck] = None
    for memory_id, memory_text, activation in scored:
        dup_verdict, dup_reason = check_duplicate(text, memory_text)
        con_verdict, con_reason, vv = check_conflict(text, memory_text, verifier)

        # Take the more severe of the duplicate / conflict views for this memory.
        if _SEVERITY[con_verdict] >= _SEVERITY[dup_verdict]:
            verdict, reason = con_verdict, con_reason
        else:
            verdict, reason = dup_verdict, dup_reason

        if verdict == LifecycleVerdict.NEW:
            continue

        check = LifecycleCheck(
            proposal_id=proposal.proposal_id,
            candidate_memory_id=memory_id,
            verdict=verdict,
            reason=reason,
            verifier_verdict=vv,
            activation=activation,
        )
        if best is None or _SEVERITY[verdict] > _SEVERITY[best.verdict]:
            best = check

    if best is None:
        return LifecycleCheck(
            proposal_id=proposal.proposal_id,
            candidate_memory_id=None,
            verdict=LifecycleVerdict.NEW,
            reason="no duplicate or conflict against active memories",
        )
    return best
