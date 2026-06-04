"""v5.0 Memory Proposal Quality Upgrade — typed, evidence-bound memory proposals.

This module raises the *quality* of candidate memory proposals so a human
reviewer sees typed, evidence-bound, reviewable claims that are clearly
separated from non-approvable junk (vague placeholders, operator instructions,
unsupported assertions). It is **proposal-quality only**:

* nothing here writes to the ``MemoryLedger`` or the concept-cell bank — there is
  no import of either, by construction;
* every proposal carries ``requires_human_approval = True`` and
  ``status = "proposed"``; nothing is ever auto-approved;
* it changes no retrieval, ranking, source-selection, grounding, or composer
  behaviour, and introduces no autonomous action.

A proposal becomes a real memory only when a human approves it and the workbench
writes it through the frozen v1.0 ``add_memory`` path. This slice never performs
that step.

The classifier is deterministic and *signal-based* (lexical markers + an
evidence-binding check), not semantic. It respects an upstream ``suggested_type``
hint but never trusts it: a hint is only honoured when the text and evidence
actually satisfy that type's rule, otherwise the candidate is downgraded — and
an unusable candidate is surfaced as ``INVALID_CANDIDATE`` *with a reason*, never
silently dropped.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Proposal types
# --------------------------------------------------------------------------- #
class MemoryProposalType:
    """String constants for the kind of memory a proposal represents.

    The first four are *factual memory* types — claims that, once approved by a
    human, could be written as durable memory. ``TODO`` / ``OPEN_QUESTION`` /
    ``SOURCE_GAP`` are valid, reviewable categories that are deliberately **not**
    factual memory (an action item, an unresolved question, and a missing-source
    note are tracked, not stored as truth). ``INVALID_CANDIDATE`` is never
    approvable: it is how vague, unsupported, instruction-like, or non-claim text
    is surfaced with an explicit reason instead of being discarded.
    """

    FACT = "fact"
    DECISION = "decision"
    PROJECT_STATE = "project_state"
    USER_PREFERENCE = "user_preference"
    TODO = "todo"
    OPEN_QUESTION = "open_question"
    SOURCE_GAP = "source_gap"
    INVALID_CANDIDATE = "invalid_candidate"


_FACTUAL_MEMORY_TYPES = frozenset({
    MemoryProposalType.FACT,
    MemoryProposalType.DECISION,
    MemoryProposalType.PROJECT_STATE,
    MemoryProposalType.USER_PREFERENCE,
})

_TRACKED_NON_FACTUAL_TYPES = frozenset({
    MemoryProposalType.TODO,
    MemoryProposalType.OPEN_QUESTION,
    MemoryProposalType.SOURCE_GAP,
})

_ALL_TYPES = (
    _FACTUAL_MEMORY_TYPES
    | _TRACKED_NON_FACTUAL_TYPES
    | {MemoryProposalType.INVALID_CANDIDATE}
)

# Types that must be bound to an evidence source to be valid. A user preference is
# intentionally *not* in this set: a stable preference can be stated by the user
# without a document citation.
_EVIDENCE_REQUIRED_TYPES = frozenset({
    MemoryProposalType.FACT,
    MemoryProposalType.DECISION,
    MemoryProposalType.PROJECT_STATE,
})

# A factual proposal is only "approvable as memory" at or above this confidence.
_APPROVABLE_CONFIDENCE_FLOOR = 0.5

_TYPE_LABEL = {
    MemoryProposalType.FACT: "FACT",
    MemoryProposalType.DECISION: "DECISION",
    MemoryProposalType.PROJECT_STATE: "PROJECT_STATE",
    MemoryProposalType.USER_PREFERENCE: "USER_PREFERENCE",
    MemoryProposalType.TODO: "TODO",
    MemoryProposalType.OPEN_QUESTION: "OPEN_QUESTION",
    MemoryProposalType.SOURCE_GAP: "SOURCE_GAP",
    MemoryProposalType.INVALID_CANDIDATE: "INVALID_CANDIDATE",
}

_STALENESS_RISK_BY_TYPE = {
    MemoryProposalType.FACT: "low",
    MemoryProposalType.DECISION: "low",
    MemoryProposalType.PROJECT_STATE: "high",
    MemoryProposalType.USER_PREFERENCE: "medium",
    MemoryProposalType.TODO: "high",
    MemoryProposalType.OPEN_QUESTION: "high",
    MemoryProposalType.SOURCE_GAP: "medium",
    MemoryProposalType.INVALID_CANDIDATE: "n/a",
}

_USEFULNESS_REASON_BY_TYPE = {
    MemoryProposalType.FACT:
        "supported factual claim worth recalling later",
    MemoryProposalType.DECISION:
        "records a decision already taken so it is not re-litigated",
    MemoryProposalType.PROJECT_STATE:
        "captures current project/repo state for later orientation",
    MemoryProposalType.USER_PREFERENCE:
        "stable user preference that should shape future work",
    MemoryProposalType.TODO:
        "tracked action item; not a fact to store as truth",
    MemoryProposalType.OPEN_QUESTION:
        "unresolved question; tracked, not stored as an answer",
    MemoryProposalType.SOURCE_GAP:
        "flags missing evidence; points at work, not a stored fact",
    MemoryProposalType.INVALID_CANDIDATE:
        "not an approvable memory; surfaced with a reason, never written",
}


# --------------------------------------------------------------------------- #
# Classification signals (deterministic, lexical)
# --------------------------------------------------------------------------- #
# Placeholder / filler that signals an unfilled stub rather than a real claim.
_VAGUE_MARKERS = (
    "tbd", "tba", "fixme", "lorem ipsum", "placeholder", "fill in", "fill me",
    "to be filled", "insert here", "your text here", "<placeholder>", "xxxxx",
)
_FILLER_ONLY_TOKENS = frozenset({
    "stuff", "something", "things", "thing", "etc", "misc", "whatever",
    "various", "other", "na", "tbd",
})

# Operator/agent tooling instructions — directions to the coding agent or VCS,
# not durable memory. These are surfaced as INVALID_CANDIDATE.
_OPERATOR_PHRASES = (
    "git commit", "git push", "git add", "commit and push", "push to origin",
    "push origin", "run pytest", "run the test", "run the tests",
    "run the suite", "run the eval", "re-run the", "rerun the", "open a pr",
    "create a pr", "open the pr",
)
_OPERATOR_LEADING_VERBS = frozenset({
    "commit", "push", "stage", "rerun", "rebase", "merge", "checkout",
    "deploy", "redeploy", "git", "pytest",
})

_OPEN_QUESTION_MARKERS = (
    "open question", "unresolved", "still unclear", "unclear whether",
    "unknown whether", "tbd whether", "need to determine", "to be decided",
    "not yet decided", "should we", "do we ",
)
_SOURCE_GAP_MARKERS = (
    "no source", "missing evidence", "no evidence", "source gap",
    "undocumented", "not documented", "lacks a source", "without a source",
    "no document", "no citation",
)
_TODO_MARKERS = (
    "todo", "to-do", "action item", "action:", "need to ", "needs to ",
    "should ", "must ", "follow up", "follow-up", "next step", "next steps",
    "plan to ", "schedule ", "migrate ", "investigate ",
)
_USER_PREFERENCE_MARKERS = (
    "prefers", "prefer ", "preference:", "likes to", "wants to always",
    "always use", "by convention", "as a rule", "favours", "favors",
)
_DECISION_MARKERS = (
    "decided", "decision:", "we chose", "chose to", "chose ", "selected ",
    "agreed to", "agreed that", "will use", "we will", "going with",
    "adopt ", "adopted ", "settled on",
)
_PROJECT_STATE_MARKERS = (
    "currently", "is now", "status:", "as of ", "tests pass", "is pushed",
    "is implemented", "the repo ", "the project ", "the branch ",
    "now exposes", "now supports", "is green", "are green",
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _has_marker(low: str, markers: Tuple[str, ...]) -> bool:
    return any(m in low for m in markers)


# --------------------------------------------------------------------------- #
# Input model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RawMemoryCandidate:
    """A raw candidate handed to the quality classifier.

    ``text`` is the proposed memory claim. ``evidence_*`` bind it to supporting
    retrieval evidence (a source and, optionally, a chunk). ``suggested_type`` is
    an *untrusted* upstream hint — it is validated, never believed on its own.
    ``note`` is free context retained for the reviewer.
    """

    text: str
    evidence_text: str = ""
    evidence_source_id: str = ""
    evidence_chunk_id: str = ""
    suggested_type: str = ""
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "RawMemoryCandidate":
        return cls(
            text=str(data.get("text", "")),
            evidence_text=str(data.get("evidence_text", "")),
            evidence_source_id=str(data.get("evidence_source_id", "")),
            evidence_chunk_id=str(data.get("evidence_chunk_id", "")),
            suggested_type=str(data.get("suggested_type", "")),
            note=str(data.get("note", "")),
        )


# --------------------------------------------------------------------------- #
# Output model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemoryProposalQuality:
    """A typed, evidence-bound, reviewable memory proposal.

    ``requires_human_approval`` is always ``True`` and ``status`` is always
    ``"proposed"`` in this slice: nothing here approves or writes a memory.
    ``invalid_reason`` is empty unless ``proposal_type`` is ``INVALID_CANDIDATE``,
    in which case it explains why the candidate is not approvable.
    """

    proposal_id: str
    proposal_type: str
    claim: str
    evidence_text: str = ""
    evidence_source_id: str = ""
    evidence_chunk_id: str = ""
    confidence: float = 0.0
    rationale: str = ""
    usefulness_reason: str = ""
    staleness_risk: str = "n/a"
    requires_human_approval: bool = True
    status: str = "proposed"
    invalid_reason: str = ""
    created_at: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["_record"] = "memory_proposal_quality"
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryProposalQuality":
        return cls(
            proposal_id=data["proposal_id"],
            proposal_type=data["proposal_type"],
            claim=data["claim"],
            evidence_text=data.get("evidence_text", ""),
            evidence_source_id=data.get("evidence_source_id", ""),
            evidence_chunk_id=data.get("evidence_chunk_id", ""),
            confidence=float(data.get("confidence", 0.0)),
            rationale=data.get("rationale", ""),
            usefulness_reason=data.get("usefulness_reason", ""),
            staleness_risk=data.get("staleness_risk", "n/a"),
            requires_human_approval=bool(
                data.get("requires_human_approval", True)),
            status=data.get("status", "proposed"),
            invalid_reason=data.get("invalid_reason", ""),
            created_at=data.get("created_at"),
        )


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #
def is_factual_memory_type(proposal_type: str) -> bool:
    """True for the types that could become durable memory once approved."""
    return proposal_type in _FACTUAL_MEMORY_TYPES


def is_tracked_non_factual_type(proposal_type: str) -> bool:
    """True for TODO / OPEN_QUESTION / SOURCE_GAP (tracked, not truth)."""
    return proposal_type in _TRACKED_NON_FACTUAL_TYPES


def is_evidence_supported(candidate: RawMemoryCandidate) -> bool:
    """A candidate is evidence-supported when it cites a source *and* text."""
    return bool(candidate.evidence_source_id.strip()
                and candidate.evidence_text.strip())


def is_approvable_as_memory(proposal: MemoryProposalQuality) -> bool:
    """Whether a human *could* approve this proposal into memory.

    Approvable means: a factual-memory type, not flagged invalid, and at or
    above the confidence floor. TODO / OPEN_QUESTION / SOURCE_GAP and every
    ``INVALID_CANDIDATE`` are surfaced for review but are **not** approvable as
    memory. (Even then, approval remains a separate human step this slice never
    performs.)
    """
    return (is_factual_memory_type(proposal.proposal_type)
            and not proposal.invalid_reason
            and proposal.confidence >= _APPROVABLE_CONFIDENCE_FLOOR)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def _is_vague_or_placeholder(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    low = stripped.lower()
    if _has_marker(low, _VAGUE_MARKERS):
        return True
    words = _tokens(low)
    if not words:
        return True
    if len(words) < 2:
        return True
    if all(w in _FILLER_ONLY_TOKENS for w in words):
        return True
    return False


def _is_operator_instruction(low: str) -> bool:
    if _has_marker(low, _OPERATOR_PHRASES):
        return True
    words = _tokens(low)
    return bool(words) and words[0] in _OPERATOR_LEADING_VERBS


def _detect_type_by_signal(low: str) -> str:
    """Return a strong-signal type, or "" when no marker fires.

    Order matters: an unresolved question and a missing-source note are detected
    before action items and factual markers so they are never mis-stored as
    truth.
    """
    if low.rstrip().endswith("?") or _has_marker(low, _OPEN_QUESTION_MARKERS):
        return MemoryProposalType.OPEN_QUESTION
    if _has_marker(low, _SOURCE_GAP_MARKERS):
        return MemoryProposalType.SOURCE_GAP
    if _has_marker(low, _TODO_MARKERS):
        return MemoryProposalType.TODO
    if _has_marker(low, _USER_PREFERENCE_MARKERS):
        return MemoryProposalType.USER_PREFERENCE
    if _has_marker(low, _DECISION_MARKERS):
        return MemoryProposalType.DECISION
    if _has_marker(low, _PROJECT_STATE_MARKERS):
        return MemoryProposalType.PROJECT_STATE
    return ""


def _validate_type(proposal_type: str,
                   candidate: RawMemoryCandidate) -> Tuple[bool, str]:
    """Check the chosen type's rule. Returns (ok, invalid_reason)."""
    if proposal_type in _TRACKED_NON_FACTUAL_TYPES \
            or proposal_type == MemoryProposalType.USER_PREFERENCE:
        return True, ""
    if proposal_type in _EVIDENCE_REQUIRED_TYPES:
        if not is_evidence_supported(candidate):
            return False, (
                f"{_TYPE_LABEL[proposal_type]} requires an evidence source and "
                f"supporting text; none provided")
        return True, ""
    return False, "unrecognised proposal type"


def classify_candidate(candidate: RawMemoryCandidate) -> Tuple[str, str]:
    """Classify a raw candidate into a typed proposal.

    Returns ``(proposal_type, invalid_reason)``. An unusable candidate is
    classified as ``INVALID_CANDIDATE`` with a non-empty reason — never dropped.
    """
    text = candidate.text.strip()
    if _is_vague_or_placeholder(text):
        return (MemoryProposalType.INVALID_CANDIDATE,
                "vague or placeholder text; not an approvable memory claim")

    low = text.lower()
    if _is_operator_instruction(low):
        return (MemoryProposalType.INVALID_CANDIDATE,
                "operator/agent instruction, not a durable factual memory")

    detected = _detect_type_by_signal(low)
    if detected:
        proposal_type = detected
    else:
        hint = (candidate.suggested_type or "").strip().lower()
        proposal_type = hint if hint in _FACTUAL_MEMORY_TYPES \
            else MemoryProposalType.FACT

    ok, reason = _validate_type(proposal_type, candidate)
    if not ok:
        return MemoryProposalType.INVALID_CANDIDATE, reason
    return proposal_type, ""


# --------------------------------------------------------------------------- #
# Scoring + proposal construction
# --------------------------------------------------------------------------- #
def _overlap_bonus(claim: str, evidence_text: str) -> float:
    claim_tokens = set(_tokens(claim))
    if not claim_tokens:
        return 0.0
    shared = claim_tokens & set(_tokens(evidence_text))
    return round(0.15 * (len(shared) / len(claim_tokens)), 3)


def _confidence_for(proposal_type: str,
                    candidate: RawMemoryCandidate) -> float:
    if proposal_type == MemoryProposalType.INVALID_CANDIDATE:
        return 0.0
    if proposal_type in _TRACKED_NON_FACTUAL_TYPES:
        # Valid and tracked, but not a truth claim — kept below the approvable
        # floor so it is never treated as factual memory.
        return 0.4
    if proposal_type == MemoryProposalType.USER_PREFERENCE:
        return 0.7 if is_evidence_supported(candidate) else 0.55
    # FACT / DECISION / PROJECT_STATE are evidence-bound by validation.
    base = 0.6
    if candidate.evidence_chunk_id.strip():
        base += 0.2
    base += _overlap_bonus(candidate.text, candidate.evidence_text)
    return round(min(base, 0.95), 3)


def _proposal_id(claim: str, evidence_source_id: str,
                 evidence_chunk_id: str) -> str:
    key = f"{claim.strip()}|{evidence_source_id.strip()}|" \
          f"{evidence_chunk_id.strip()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"memprop-{digest}"


def _rationale_for(proposal_type: str, candidate: RawMemoryCandidate,
                   invalid_reason: str) -> str:
    if proposal_type == MemoryProposalType.INVALID_CANDIDATE:
        return invalid_reason
    if proposal_type in _EVIDENCE_REQUIRED_TYPES:
        return (f"{_TYPE_LABEL[proposal_type]} bound to evidence "
                f"{candidate.evidence_source_id or '(unknown source)'}; "
                f"requires human approval before any memory write")
    return (f"classified as {_TYPE_LABEL[proposal_type]} by signal; "
            f"requires human approval before any memory write")


def build_proposal(candidate: RawMemoryCandidate, *,
                   created_at: Optional[str] = None) -> MemoryProposalQuality:
    """Turn one raw candidate into a typed, evidence-bound proposal.

    The proposal always carries ``requires_human_approval=True`` and
    ``status="proposed"``. No memory is written.
    """
    proposal_type, invalid_reason = classify_candidate(candidate)
    confidence = _confidence_for(proposal_type, candidate)
    return MemoryProposalQuality(
        proposal_id=_proposal_id(candidate.text, candidate.evidence_source_id,
                                 candidate.evidence_chunk_id),
        proposal_type=proposal_type,
        claim=candidate.text.strip(),
        evidence_text=candidate.evidence_text.strip(),
        evidence_source_id=candidate.evidence_source_id.strip(),
        evidence_chunk_id=candidate.evidence_chunk_id.strip(),
        confidence=confidence,
        rationale=_rationale_for(proposal_type, candidate, invalid_reason),
        usefulness_reason=_USEFULNESS_REASON_BY_TYPE[proposal_type],
        staleness_risk=_STALENESS_RISK_BY_TYPE[proposal_type],
        requires_human_approval=True,
        status="proposed",
        invalid_reason=invalid_reason,
        created_at=created_at,
    )


def build_memory_proposals(candidates: List[RawMemoryCandidate], *,
                           created_at: Optional[str] = None
                           ) -> List[MemoryProposalQuality]:
    """Build proposals for many candidates, sorted deterministically by id."""
    proposals = [build_proposal(c, created_at=created_at) for c in candidates]
    proposals.sort(key=lambda p: (p.proposal_id, p.proposal_type))
    return proposals


# --------------------------------------------------------------------------- #
# Loading + deterministic rendering / export
# --------------------------------------------------------------------------- #
def load_memory_candidates(path) -> List[RawMemoryCandidate]:
    """Load raw candidates from a JSONL file (``#`` comment lines allowed)."""
    out: List[RawMemoryCandidate] = []
    file_path = Path(path)
    if not file_path.exists():
        return out
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(RawMemoryCandidate.from_dict(json.loads(stripped)))
    return out


def memory_proposals_to_jsonl(proposals: List[MemoryProposalQuality]) -> str:
    """Serialise proposals as deterministic JSONL (sorted by id)."""
    ordered = sorted(proposals, key=lambda p: (p.proposal_id, p.proposal_type))
    return "\n".join(p.to_json() for p in ordered)


def write_memory_proposals(proposals: List[MemoryProposalQuality], path) -> None:
    """Write proposals to a JSONL file.

    This is the **only** writer in this module. It writes a proposal-review
    export and never touches the memory ledger, the bank, or any source file.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = memory_proposals_to_jsonl(proposals)
    file_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def render_memory_proposals_markdown(
        proposals: List[MemoryProposalQuality]) -> str:
    """Deterministic Markdown view of a memory-proposal review set."""
    ordered = sorted(proposals, key=lambda p: (p.proposal_id, p.proposal_type))
    lines: List[str] = ["# Memory proposal review (v5.0)", ""]
    lines.append("All proposals require human approval; none are written. "
                 "Approvable = factual type, evidence-bound, above confidence "
                 "floor.")
    lines.append("")

    counts: dict = {}
    approvable = 0
    for proposal in ordered:
        counts[proposal.proposal_type] = counts.get(proposal.proposal_type, 0) + 1
        if is_approvable_as_memory(proposal):
            approvable += 1

    lines.append("## Summary by type")
    lines.append("")
    lines.append("| type | count |")
    lines.append("|---|---|")
    for proposal_type in sorted(counts):
        lines.append(f"| {_TYPE_LABEL[proposal_type]} | {counts[proposal_type]} |")
    lines.append(f"| **approvable as memory** | **{approvable}** |")
    lines.append("")

    lines.append("## Proposals")
    lines.append("")
    lines.append("| proposal_id | type | approvable | confidence | claim | "
                 "evidence_source_id | invalid_reason |")
    lines.append("|---|---|---|---|---|---|---|")
    for proposal in ordered:
        approv = "yes" if is_approvable_as_memory(proposal) else "no"
        claim = proposal.claim.replace("|", "\\|")
        reason = proposal.invalid_reason.replace("|", "\\|")
        lines.append(
            f"| {proposal.proposal_id} | {_TYPE_LABEL[proposal.proposal_type]} "
            f"| {approv} | {proposal.confidence} | {claim} "
            f"| {proposal.evidence_source_id} | {reason} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# v5.1 Memory proposal review queue (review-state only; approved != written)
# --------------------------------------------------------------------------- #
# This layer lets a human triage v5.0 memory proposals — approve, reject, or
# defer them — without ever writing memory. It is review-state only:
#
# * the only file it writes is the explicit review-queue JSONL passed to
#   :func:`save_memory_review_queue` (used by import and review);
# * it still imports no ``MemoryLedger``/``MemoryBank``/``add_memory`` path, so
#   nothing here can write durable memory by construction;
# * **approved does not mean written** — ``written`` stays ``False`` and
#   ``written_at`` stays ``None`` in v5.1. Approving a proposal authorises a
#   future write; it does not perform one.
#
# Every queue entry keeps the full originating ``proposal_snapshot`` so a review
# is auditable on its own.


class MemoryReviewStatus:
    """Stable string codes for where a memory proposal sits in human review."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


_MEMORY_REVIEW_STATUSES = frozenset({
    MemoryReviewStatus.PENDING, MemoryReviewStatus.APPROVED,
    MemoryReviewStatus.REJECTED, MemoryReviewStatus.DEFERRED,
})

# Allowed review-status transitions. ``approved`` and ``rejected`` are terminal
# in v5.1 — there is no reopen step, so a rejected proposal never silently
# becomes approved, and an approved proposal never means written.
_ALLOWED_MEMORY_REVIEW_TRANSITIONS: Dict[str, frozenset] = {
    MemoryReviewStatus.PENDING: frozenset({
        MemoryReviewStatus.APPROVED, MemoryReviewStatus.REJECTED,
        MemoryReviewStatus.DEFERRED}),
    MemoryReviewStatus.DEFERRED: frozenset({
        MemoryReviewStatus.APPROVED, MemoryReviewStatus.REJECTED}),
    MemoryReviewStatus.APPROVED: frozenset(),
    MemoryReviewStatus.REJECTED: frozenset(),
}


def memory_review_status_transition_allowed(current: str, new: str) -> bool:
    """Whether ``current -> new`` is a permitted review transition (pure)."""
    return new in _ALLOWED_MEMORY_REVIEW_TRANSITIONS.get(current, frozenset())


@dataclass(frozen=True)
class MemoryProposalReview:
    """One review record for one v5.0 memory proposal. Review-state only; inert.

    It records a human's *decision* about a memory proposal — approve, reject,
    defer — and never the writing of that memory. ``written`` is always
    ``False`` and ``written_at`` always ``None`` in v5.1: approving a proposal
    authorises a future memory write, it does not make one. ``proposal_snapshot``
    keeps the full originating proposal so the queue is auditable on its own.
    """

    proposal_id: str
    proposal_type: str
    claim: str
    review_status: str = MemoryReviewStatus.PENDING
    reviewer: str = ""
    reviewed_at: Optional[str] = None
    review_note: str = ""
    written: bool = False
    written_at: Optional[str] = None
    proposal_snapshot: dict = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "claim": self.claim,
            "review_status": self.review_status,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "review_note": self.review_note,
            "written": self.written,
            "written_at": self.written_at,
            "proposal_snapshot": self.proposal_snapshot,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "_record": "memory_proposal_review",
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryProposalReview":
        return cls(
            proposal_id=str(data["proposal_id"]),
            proposal_type=str(data.get("proposal_type", "")),
            claim=str(data.get("claim", "")),
            review_status=str(
                data.get("review_status", MemoryReviewStatus.PENDING)),
            reviewer=str(data.get("reviewer", "")),
            reviewed_at=data.get("reviewed_at"),
            review_note=str(data.get("review_note", "")),
            written=bool(data.get("written", False)),
            written_at=data.get("written_at"),
            proposal_snapshot=dict(data.get("proposal_snapshot") or {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    @classmethod
    def pending_from_proposal(cls, proposal: dict, *,
                              created_at: Optional[str] = None,
                              ) -> "MemoryProposalReview":
        """Build a fresh ``pending`` review from a proposal dict (snapshot kept)."""
        return cls(
            proposal_id=str(proposal["proposal_id"]),
            proposal_type=str(proposal.get("proposal_type", "")),
            claim=str(proposal.get("claim", "")),
            review_status=MemoryReviewStatus.PENDING,
            proposal_snapshot=dict(proposal),
            created_at=created_at,
            updated_at=created_at,
        )


# -- queue load / save --------------------------------------------------------

def load_memory_proposal_dicts(path) -> List[dict]:
    """Load v5.0 memory-proposal records (raw dicts) from a JSONL file (pure read).

    ``#`` lines are comments. This reads proposals so they can be queued for
    review; it touches nothing else.
    """
    dicts: List[dict] = []
    file_path = Path(path)
    if not file_path.exists():
        return dicts
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        dicts.append(json.loads(stripped))
    return dicts


def load_memory_review_queue(path) -> List[MemoryProposalReview]:
    """Load the memory review queue from JSONL (a missing file is an empty queue).

    A pure read: ``#`` lines are comments, and absence of the file simply means
    no proposals have been imported yet.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []
    reviews: List[MemoryProposalReview] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        reviews.append(MemoryProposalReview.from_dict(json.loads(stripped)))
    return reviews


def _sorted_memory_queue(reviews) -> List[MemoryProposalReview]:
    """Deterministic queue order: by ``proposal_id``."""
    return sorted(reviews, key=lambda r: r.proposal_id)


def save_memory_review_queue(reviews: List[MemoryProposalReview], path) -> None:
    """Write the memory review queue to JSONL (sorted by proposal_id; deterministic).

    This is the **only** writer in the review layer. It touches *only* the queue
    path — never the memory ledger, the bank, or any source file, and it never
    writes a memory.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for review in _sorted_memory_queue(reviews):
            handle.write(review.to_json() + "\n")


# -- queue operations (all pure; persistence is the caller's explicit save) ---

def import_memory_proposals_to_queue(proposals: List[dict],
                                     existing: List[MemoryProposalReview], *,
                                     created_at: Optional[str] = None,
                                     ) -> List[MemoryProposalReview]:
    """Merge memory-proposal dicts into a review queue as ``pending`` records.

    Idempotent: existing reviews are preserved unchanged (keyed by
    ``proposal_id``), and only proposals whose id is not already queued are
    added. Deterministic: returns the merged queue sorted by ``proposal_id``, so
    re-importing the same proposals is a no-op and saves byte-identically while
    preserving any existing approved/rejected/deferred state. Pure — it writes
    nothing.
    """
    by_id = {r.proposal_id: r for r in existing}
    for proposal in proposals:
        pid = str(proposal["proposal_id"])
        if pid in by_id:
            continue
        by_id[pid] = MemoryProposalReview.pending_from_proposal(
            proposal, created_at=created_at)
    return _sorted_memory_queue(by_id.values())


def review_memory_proposal(review: MemoryProposalReview, new_status: str, *,
                           reviewer: Optional[str] = None,
                           note: Optional[str] = None,
                           reviewed_at: Optional[str] = None,
                           updated_at: Optional[str] = None,
                           ) -> MemoryProposalReview:
    """Return a new review with ``new_status`` applied (review-state only).

    Validates the transition against
    :data:`_ALLOWED_MEMORY_REVIEW_TRANSITIONS` and raises ``ValueError`` cleanly
    on an unknown or disallowed status — the input review (frozen) is never
    mutated, so a failed transition leaves state intact. ``written``/
    ``written_at`` are held at ``False``/``None``: **approved does not mean
    written** in v5.1.
    """
    if new_status not in _MEMORY_REVIEW_STATUSES:
        raise ValueError(f"unknown review status {new_status!r}")
    if not memory_review_status_transition_allowed(
            review.review_status, new_status):
        raise ValueError(
            f"invalid review transition {review.review_status!r} -> "
            f"{new_status!r} for proposal {review.proposal_id}")
    return replace(
        review,
        review_status=new_status,
        reviewer=review.reviewer if reviewer is None else reviewer,
        reviewed_at=review.reviewed_at if reviewed_at is None else reviewed_at,
        review_note=review.review_note if note is None else note,
        updated_at=review.updated_at if updated_at is None else updated_at,
        written=False,
        written_at=None,
    )


def apply_memory_review_to_queue(reviews: List[MemoryProposalReview],
                                 proposal_id: str, new_status: str, *,
                                 reviewer: Optional[str] = None,
                                 note: Optional[str] = None,
                                 reviewed_at: Optional[str] = None,
                                 updated_at: Optional[str] = None,
                                 ) -> List[MemoryProposalReview]:
    """Apply a review transition to one queued proposal, returning a new queue.

    Pure: builds a new list (same membership), replacing only the targeted
    review and re-sorting deterministically. Raises ``ValueError`` if
    ``proposal_id`` is not queued. Writes nothing — persistence is the caller's
    explicit :func:`save_memory_review_queue`.
    """
    found = False
    updated: List[MemoryProposalReview] = []
    for review in reviews:
        if review.proposal_id == proposal_id:
            updated.append(review_memory_proposal(
                review, new_status, reviewer=reviewer, note=note,
                reviewed_at=reviewed_at, updated_at=updated_at))
            found = True
        else:
            updated.append(review)
    if not found:
        raise ValueError(f"no review for proposal_id {proposal_id!r}")
    return _sorted_memory_queue(updated)


def render_memory_review_queue_markdown(
        reviews: List[MemoryProposalReview]) -> str:
    """Render a deterministic Markdown view of the memory review queue.

    Output depends only on the (deterministically sorted) reviews, so the same
    queue renders identical text every time. Nothing is written.
    """
    counts: Dict[str, int] = {}
    for r in reviews:
        counts[r.review_status] = counts.get(r.review_status, 0) + 1
    ordered = _sorted_memory_queue(reviews)

    lines: List[str] = []
    lines.append("# Memory proposal review queue")
    lines.append("")
    lines.append("Human review state for v5.0 memory proposals (v5.1). "
                 "Reviewing records a decision only: **approved does not mean "
                 "written** — nothing here writes the memory ledger, applies a "
                 "memory, or changes retrieval, ranking, source selection, "
                 "grounding, or composers. `written` stays false in v5.1.")
    lines.append("")
    lines.append(f"- reviews: {len(ordered)}")
    written_n = sum(1 for r in ordered if r.written)
    lines.append(f"- written: {written_n} (always 0 in v5.1)")
    lines.append("")

    lines.append("## Summary by status")
    lines.append("")
    if counts:
        for status in sorted(counts):
            lines.append(f"- {status}: {counts[status]}")
    else:
        lines.append("- (no reviews)")
    lines.append("")

    lines.append("## Reviews")
    lines.append("")
    if ordered:
        lines.append("| review_status | proposal_id | proposal_type | "
                     "written | claim |")
        lines.append("|---|---|---|---|---|")
        for r in ordered:
            claim = r.claim.replace("|", "\\|")
            lines.append(
                f"| {r.review_status} | {r.proposal_id} | {r.proposal_type} "
                f"| {str(r.written).lower()} | {claim} |")
    else:
        lines.append("(queue is empty)")
    return "\n".join(lines)
