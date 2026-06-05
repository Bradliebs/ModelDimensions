"""v6.0 Governed chat orchestrator (read-only).

This module routes a user's chat question into one of a small set of *safe*
answer modes and renders a result. It is a **read-only orchestration skeleton**:
it may answer, cite, label judgement, refuse, or emit follow-up *proposals* for
human review — and it must never mutate durable state.

What "read-only" means here, enforced by construction:

* it never writes the memory ledger, the memory bank, or any review queue;
* it never writes or edits the source registry;
* it never applies a memory or source proposal;
* it never runs an autonomous action (deploy, commit, delete, ...);
* it changes no retrieval / ranking / grounding / composer behaviour — those are
  the frozen upstream paths, which this layer only *reads from*.

It wires only into existing read-only paths:

* :meth:`WorkbenchService.answer_query` for evidence-grounded answers (the
  frozen grounding + composer pipeline);
* :func:`agent.memory_proposal_quality.build_memory_proposals` to turn a
  memory-write instruction into a *proposal only*;
* :func:`agent.source_registry.propose_source_updates` to turn a
  source-maintenance request into *proposals only*.

The module imports none of the writers (no ``MemoryLedger``, no ``save_registry``,
no review-queue writer), so it cannot mutate durable state even by accident.
``ChatOrchestratorResult.state_mutation_attempted`` is therefore always ``False``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from agent.memory_proposal_quality import (
    RawMemoryCandidate,
    build_memory_proposals,
)
from agent.source_registry import load_registry, propose_source_updates
from slm.assistant_composer import JUDGEMENT_LABEL


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ChatMode:
    """The safe answer modes the orchestrator may produce.

    Every mode is non-mutating. ``PROPOSE_*`` modes emit proposals for human
    review; they never apply them. ``UNSUPPORTED_REQUEST`` is a polite refusal
    for anything that would require changing durable state or running an action.
    """

    EVIDENCE_ANSWER = "evidence_answer"
    MEMORY_CONTEXT = "memory_context"
    REPORT_STYLE_ANSWER = "report_style_answer"
    JUDGEMENT_ONLY = "judgement_only"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROPOSE_MEMORY = "propose_memory"
    PROPOSE_SOURCE_UPDATE = "propose_source_update"
    UNSUPPORTED_REQUEST = "unsupported_request"


@dataclass(frozen=True)
class ChatIntent:
    """The router's first-pass classification of a query. Pure data.

    ``mode`` is the *target* answer mode the router selected; the orchestrator
    may still downgrade an evidence-seeking mode to ``INSUFFICIENT_EVIDENCE``
    when retrieval grounds nothing.
    """

    query: str
    mode: str
    reason: str
    wants_report: bool = False
    judgement_requested: bool = False


@dataclass(frozen=True)
class ChatOrchestratorResult:
    """The outcome of answering one chat query. Inert, read-only.

    ``state_mutation_attempted`` is always ``False``: this layer has no write
    path. ``proposed_memory`` / ``proposed_source_updates`` carry the *proposal*
    dictionaries (each ``requires_human_approval`` and ``status="proposed"``),
    never an applied change.
    """

    query: str
    mode: str
    answer_text: str
    citations: List[str] = field(default_factory=list)
    judgement_labelled: bool = False
    proposed_memory_count: int = 0
    proposed_source_update_count: int = 0
    refusal_reason: str = ""
    state_mutation_attempted: bool = False
    intent_mode: str = ""
    route_reason: str = ""
    proposed_memory: List[dict] = field(default_factory=list)
    proposed_source_updates: List[dict] = field(default_factory=list)
    related_sources: List[str] = field(default_factory=list)
    # v6.2 evidence-bound answer UX metadata. Every field below is populated
    # deterministically from data the frozen pipeline already produced (composer
    # citations + the read-only audit); none of them trigger retrieval or a
    # write. ``unsupported_claim_count`` is 0 by construction in this governed
    # surface: grounded facts are always cited and judgement is always labelled,
    # so no uncited factual claim can be emitted.
    evidence_used_count: int = 0
    evidence_gap_count: int = 0
    evidence_summary: str = ""
    missing_evidence: List[str] = field(default_factory=list)
    answer_has_citations: bool = False
    judgement_present: bool = False
    unsupported_claim_count: int = 0
    # v7.1 evidence inspector projection: one read-only dict per retrieved
    # knowledge candidate (source/chunk provenance, the backend's relevance
    # score, and whether it was cited in the final answer). Built solely from
    # the audit the frozen pipeline already produced — it triggers no retrieval,
    # changes no ranking, and writes nothing. ``cited`` reflects whether the
    # candidate's citation id appears in the composer's own citation list; the
    # ``score`` is the backend's relevance activation and is never a confidence.
    evidence_detail: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "_record": "chat_orchestrator_result",
            "query": self.query,
            "mode": self.mode,
            "answer_text": self.answer_text,
            "citations": list(self.citations),
            "judgement_labelled": self.judgement_labelled,
            "proposed_memory_count": self.proposed_memory_count,
            "proposed_source_update_count": self.proposed_source_update_count,
            "refusal_reason": self.refusal_reason,
            "state_mutation_attempted": self.state_mutation_attempted,
            "intent_mode": self.intent_mode,
            "route_reason": self.route_reason,
            "related_sources": list(self.related_sources),
            "evidence_used_count": self.evidence_used_count,
            "evidence_gap_count": self.evidence_gap_count,
            "evidence_summary": self.evidence_summary,
            "missing_evidence": list(self.missing_evidence),
            "answer_has_citations": self.answer_has_citations,
            "judgement_present": self.judgement_present,
            "unsupported_claim_count": self.unsupported_claim_count,
            "evidence_detail": [dict(d) for d in self.evidence_detail],
        }


# --------------------------------------------------------------------------- #
# Deterministic first-pass router
# --------------------------------------------------------------------------- #
# Imperative system actions that are never a knowledge question in this tool.
# These are refused even when phrased as a question ("can you git push?") — chat
# can *propose*, never *apply*. Ordered first so safety wins ties.
_HARD_ACTION_MARKERS = (
    "apply the proposal", "apply this proposal", "apply the update",
    "write to the ledger", "write to memory now", "approve and write",
    "save the registry", "update the registry file", "edit the registry",
    "overwrite the registry", "rm -",
)
# Action words that ALSO appear in legitimate questions ("how do we remove the
# stale role?", "what's the best way to deploy least-privilege?"). These are
# refused only when the query is an imperative *command*, not when it asks how or
# whether to do the thing — answering is non-mutating, so it is always safe.
_SOFT_ACTION_MARKERS = (
    "delete ", "drop ", "rm -", "remove the ", "execute ", "run pytest",
    "deploy", "push to", "git commit", "git push", "commit and push",
    "overwrite",
)
# Openers that mark a query as a knowledge or advice *question* rather than an
# imperative command. A question is never refused as an action request; it routes
# onward to an evidence answer or a labelled judgement.
_QUESTION_OPENERS = (
    "how ", "how's", "how do", "how can", "how should", "how would",
    "what ", "what's", "what is", "what are", "why ", "why's",
    "when ", "where ", "which ", "who ", "whose ",
    "should i", "should we", "do we", "do you", "is it", "are there",
    "can we", "could we",
    "explain", "describe", "summarize", "summarise", "tell me", "give me",
    "show me", "list ", "walk me through", "compare ", "outline ",
)
# "Remember this" style instructions -> a memory proposal only.
_MEMORY_WRITE_MARKERS = (
    "remember that", "remember this", "store this", "save this to memory",
    "note that", "memorize", "add to memory", "keep in mind that",
    "record that", "make a note that",
)
# Source-maintenance requests -> a source-update proposal only.
_SOURCE_UPDATE_MARKERS = (
    "source registry", "stale source", "source update", "source maintenance",
    "missing owner", "review the sources", "registry audit", "propose source",
    "source metadata", "outdated source",
)
# Queries that ask about remembered context.
_MEMORY_CONTEXT_MARKERS = (
    "from memory", "what do we know about", "our notes", "we decided",
    "according to memory", "from our records", "what did we decide",
    "do we remember",
)
# Queries asking for a report-style answer.
_REPORT_MARKERS = (
    "report", "consultant", "executive summary", "write up", "write-up",
    "brief on", "summary report",
)
# Queries seeking an opinion / recommendation -> judgement, always labelled.
_JUDGEMENT_MARKERS = (
    "should i", "should we", "what do you think", "your opinion",
    "do you recommend", "would you recommend", "is it better",
    "what's your take", "which is best", "best approach", "do you think",
    "recommend",
)


def _is_question_like(low: str) -> bool:
    """True when the query asks something rather than commanding an action.

    A trailing ``?`` or a knowledge/advice opener (``how``/``what``/``should
    we`` ...) marks the query as a question. Questions are answered, not refused
    as action requests, even when they mention an action word.
    """
    if low.endswith("?"):
        return True
    return low.startswith(_QUESTION_OPENERS)


def route_query(query: str) -> ChatIntent:
    """Classify a query into a first-pass answer mode (deterministic, lexical).

    Precedence is safety-first: an action/mutation *command* is refused before
    any answer is attempted; then memory-write and source-maintenance
    instructions become proposals; then memory-context, report, and judgement
    framings; and anything else is treated as a factual question for an evidence
    answer. A real question that merely mentions an action word ("how do we
    remove the stale role?") is answered, not refused — only an imperative
    command ("delete the registry file") or a hard system directive is.
    """
    q = (query or "").strip()
    low = q.lower()

    def has(markers) -> bool:
        return any(m in low for m in markers)

    question = _is_question_like(low)
    if has(_HARD_ACTION_MARKERS) or (has(_SOFT_ACTION_MARKERS) and not question):
        return ChatIntent(
            query=q, mode=ChatMode.UNSUPPORTED_REQUEST,
            reason="request asks to change durable state or run an action")
    if has(_MEMORY_WRITE_MARKERS):
        return ChatIntent(
            query=q, mode=ChatMode.PROPOSE_MEMORY,
            reason="memory-write instruction; emitted as a proposal only")
    if has(_SOURCE_UPDATE_MARKERS):
        return ChatIntent(
            query=q, mode=ChatMode.PROPOSE_SOURCE_UPDATE,
            reason="source-maintenance request; emitted as a proposal only")

    wants_report = has(_REPORT_MARKERS)
    if has(_MEMORY_CONTEXT_MARKERS):
        return ChatIntent(
            query=q, mode=ChatMode.MEMORY_CONTEXT, wants_report=wants_report,
            reason="query asks about remembered context")
    if has(_JUDGEMENT_MARKERS):
        return ChatIntent(
            query=q, mode=ChatMode.JUDGEMENT_ONLY, judgement_requested=True,
            wants_report=wants_report,
            reason="query seeks an opinion or recommendation; judgement is "
                   "labelled")
    if wants_report:
        return ChatIntent(
            query=q, mode=ChatMode.REPORT_STYLE_ANSWER, wants_report=True,
            reason="query requests a report-style answer")
    return ChatIntent(
        query=q, mode=ChatMode.EVIDENCE_ANSWER,
        reason="factual question routed to an evidence-grounded answer")


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def _strip_memory_directive(query: str) -> str:
    """Drop a leading "remember that"/"note that" directive from a claim."""
    low = query.lower()
    best_idx = -1
    best_end = 0
    for marker in _MEMORY_WRITE_MARKERS:
        idx = low.find(marker)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx = idx
            best_end = idx + len(marker)
    if best_idx == -1:
        return query.strip()
    tail = query[best_end:].strip().lstrip(":").strip()
    return tail or query.strip()


def _judgement_prose(grounded: bool) -> str:
    if grounded:
        return ("This is an interpretation drawn from the cited evidence above, "
                "not an additional grounded fact.")
    return ("This is an interpretation, not a grounded fact; the active pack "
            "holds no citable evidence that settles the question.")


def _has_memory_citation(citations: List[str]) -> bool:
    return any(str(c).startswith("mem:") for c in citations)


_EVIDENCE_GAP = ("No citable evidence was retrieved for this query from the "
                 "active knowledge pack, so no grounded answer can be given.")


def _related_sources_from_audit(audit: dict, *, limit: int = 3) -> List[str]:
    """The closest knowledge sources retrieval considered (read-only).

    Reads only the audit the frozen pipeline already produced and returns the
    distinct source names of the retrieved-but-not-grounding candidates, so an
    unanswered question can point at the nearest covered topics. It never
    retrieves, ranks, or mutates anything — it only inspects the audit dict.
    """
    know = (audit or {}).get("knowledge") or {}
    names: List[str] = []
    for cand in know.get("candidates") or []:
        name = cand.get("source_name")
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _evidence_detail_from_audit(audit: dict,
                                citations: List[str]) -> List[dict]:
    """Project the retrieved knowledge candidates for the evidence inspector.

    Reads only ``audit["knowledge"]["candidates"]`` — the candidate dicts the
    frozen retrieval pipeline already produced — and returns one inert dict per
    candidate with its provenance (source, chunk, section, version), authority,
    domain, the backend's relevance ``score`` (its ``activation``; never a
    confidence), its ``rank``, and a ``cited`` flag set when the candidate's
    ``src:<chunk_id>`` citation appears in the composer's own citation list.

    It never retrieves, re-ranks, or writes anything; it only reshapes data the
    pipeline already computed so the UI can show retrieved-vs-cited stages.
    """
    cited_ids = set(citations or ())
    know = (audit or {}).get("knowledge") or {}
    detail: List[dict] = []
    for cand in know.get("candidates") or []:
        chunk_id = str(cand.get("chunk_id") or "")
        citation_id = f"src:{chunk_id}" if chunk_id else ""
        detail.append({
            "citation_id": citation_id,
            "source_id": str(cand.get("source_id") or ""),
            "chunk_id": chunk_id,
            "source_name": str(cand.get("source_name") or ""),
            "domain": str(cand.get("domain") or ""),
            "authority": str(cand.get("authority") or ""),
            "version": str(cand.get("version") or ""),
            "section": str(cand.get("section") or ""),
            "score": float(cand.get("activation") or 0.0),
            "rank": int(cand.get("rank") or 0),
            "backend_name": str(cand.get("backend_name") or ""),
            "text": str(cand.get("text") or ""),
            "cited": bool(citation_id) and citation_id in cited_ids,
        })
    return detail


def _insufficient_reason(audit: dict) -> str:
    """The evidence-gap message, enriched with the relevance gate's reason.

    The frozen relevance gate already recorded *why* the evidence was
    insufficient; surfacing its ``sufficiency_reason`` makes the refusal
    specific instead of generic. Read-only: it only reads the audit.
    """
    relevance = (audit or {}).get("relevance") or {}
    reason = str(relevance.get("sufficiency_reason") or "").strip()
    if reason:
        return _EVIDENCE_GAP + " " + reason
    return _EVIDENCE_GAP


# --------------------------------------------------------------------------- #
# v6.2 evidence-bound answer UX (deterministic formatters)
# --------------------------------------------------------------------------- #
# These formatters compose the *answer_text* surface from data the frozen
# pipeline already produced. They add no LLM call, perform no retrieval, and
# write nothing — they only reshape already-grounded composer text, the
# composer's own citations, and the read-only audit into a clearer, auditable
# structure (Answer / Evidence used / Evidence gaps / Next safe action; a
# labelled judgement that separates fact from opinion; and a governed refusal
# that names a safe alternative).
_NEXT_ACTION_EVIDENCE = (
    "Base any decision on the cited evidence above. To extend this answer, add a "
    "source to the active pack or ask a more specific question; this chat will "
    "not write memory, edit the registry, or apply changes.")
_NEXT_ACTION_INSUFFICIENT = (
    "Add a pack source that covers this topic, or narrow the question to a "
    "covered topic, then ask again. This chat will not invent an answer or write "
    "any state.")
_NEXT_ACTION_JUDGEMENT = (
    "Judgement is advisory; validate it against the cited evidence (or gather "
    "evidence where none exists) before acting. This chat will not act on the "
    "recommendation or write any state.")
_GOVERNANCE_BOUNDARY = (
    "This chat is read-only by construction: it has no write path to memory, the "
    "source registry, the review queue, or the shell, so it cannot apply a "
    "durable change even when asked.")


def _evidence_summary_line(used: int, gaps: int) -> str:
    """One deterministic audit line: counts plus the standing guarantees."""
    return (f"{used} evidence item(s) cited; {gaps} evidence gap(s) noted; every "
            "factual claim is citation-backed and any judgement is labelled.")


def summarize_evidence_gaps(audit: dict) -> List[str]:
    """Name the evidence gaps the frozen relevance gate already recorded.

    Reads only the audit dict the pipeline produced: the relevance gate's
    ``sufficiency_reason`` and, when present, the closest rejected candidate's
    reason. Returns an ordered, de-duplicated list of gap strings (empty when
    the gate recorded none). It retrieves, ranks, and writes nothing.
    """
    relevance = (audit or {}).get("relevance") or {}
    gaps: List[str] = []
    reason = str(relevance.get("sufficiency_reason") or "").strip()
    if reason:
        gaps.append(reason)
    rejected = relevance.get("top_rejected") or {}
    rej_reason = str(rejected.get("reason") or "").strip()
    if rej_reason:
        rej_id = str(rejected.get("citation_id") or "").strip()
        item = (f"closest rejected evidence {rej_id}: {rej_reason}"
                if rej_id else rej_reason)
        if item not in gaps:
            gaps.append(item)
    return gaps


def format_evidence_bound_answer(answer_text: str, citations: List[str],
                                 gaps: List[str]) -> str:
    """Structure a grounded answer: Answer / Evidence used / gaps / next action.

    Pure and deterministic. ``answer_text`` is the frozen composer's grounded
    text (passed through unchanged), ``citations`` are the composer's own
    citations, and ``gaps`` come from the audit. No judgement is added here —
    judgement is its own labelled mode.
    """
    parts = ["Answer:\n" + (answer_text.strip() or "(no answer)")]
    if citations:
        cited = "\n".join(f"- {c}" for c in citations)
        parts.append(f"Evidence used ({len(citations)}):\n{cited}")
    else:
        parts.append("Evidence used: none cited.")
    if gaps:
        parts.append("Evidence gaps:\n" + "\n".join(f"- {g}" for g in gaps))
    else:
        parts.append("Evidence gaps:\n- None identified within the cited scope; "
                     "this answer reflects only the active pack.")
    parts.append("Next safe action:\n" + _NEXT_ACTION_EVIDENCE)
    return "\n\n".join(parts)


def format_insufficient_evidence_answer(query: str, reason: str,
                                        missing: List[str],
                                        related: List[str]) -> str:
    """Explain a refusal constructively: asked / why / what resolves it / next.

    Names the specific missing evidence instead of a generic "I don't know",
    and points at the closest covered topics so the asker can narrow or
    rephrase. Pure and deterministic; reads no state.
    """
    parts = [f"Asked:\n- {query.strip()}"]
    parts.append("Why this cannot be answered from current evidence:\n- " + reason)
    if missing:
        parts.append("Evidence that would resolve the gap:\n"
                     + "\n".join(f"- {m}" for m in missing))
    else:
        parts.append("Evidence that would resolve the gap:\n- A pack source that "
                     "directly addresses this question.")
    if related:
        parts.append("Closest topics already in the pack:\n- "
                     + "; ".join(related)
                     + "\nTry narrowing the question to one of those, or "
                     "rephrasing it.")
    parts.append("Next safe action:\n" + _NEXT_ACTION_INSUFFICIENT)
    return "\n\n".join(parts)


def format_labelled_judgement(fact_text: str, citations: List[str], *,
                              grounded: bool) -> str:
    """Separate evidence-backed facts from labelled judgement and assumptions.

    Judgement is always prefixed with :data:`JUDGEMENT_LABEL`. When evidence
    exists the facts and their citations are shown first; when none exists the
    answer says so plainly. Pure and deterministic.
    """
    parts: List[str] = []
    if grounded:
        parts.append("Evidence-backed facts:\n" + (fact_text.strip() or "(none)"))
        if citations:
            cited = "\n".join(f"- {c}" for c in citations)
            parts.append(f"Evidence used ({len(citations)}):\n{cited}")
        parts.append("Judgement:\n" + JUDGEMENT_LABEL + " "
                     + _judgement_prose(True))
        parts.append("Assumptions:\n- The cited evidence is current and applies "
                     "to your context; judgement adds no new facts.")
    else:
        parts.append("Judgement:\n" + JUDGEMENT_LABEL + " "
                     + _judgement_prose(False))
        parts.append("Evidence-backed facts:\n- None. " + _EVIDENCE_GAP)
        parts.append("Assumptions:\n- This is reasoning without retrieved "
                     "evidence; treat it as opinion, not fact.")
    parts.append("Next safe action:\n" + _NEXT_ACTION_JUDGEMENT)
    return "\n\n".join(parts)


def format_unsupported_governed_action(query: str) -> str:
    """Refuse a mutation/action request and offer a safe, in-bounds alternative.

    Explains the governance boundary (this chat has no write path) and points to
    what it *can* do instead: produce a proposal, checklist, or command plan for
    a human to review and run. Pure and deterministic.
    """
    return "\n\n".join([
        "Refused:\n- This request asks to change durable state or run an action, "
        "which this read-only chat will not do.",
        "Governance boundary:\n- " + _GOVERNANCE_BOUNDARY,
        "Safe alternative:\n- Ask this chat to *propose* the change instead (a "
        "memory or source proposal you can review and approve), or ask for a "
        "checklist or command plan you can run yourself. Facts will be cited and "
        "any judgement will be labelled.",
    ])


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class ChatOrchestrator:
    """Route a query and render a safe, non-mutating answer.

    ``service`` is any object exposing the frozen read-only
    :meth:`answer_query` (a :class:`WorkbenchService`). ``registry_path`` is an
    optional source-registry JSONL used only to *generate* source-update
    proposals; it is read, never written.
    """

    def __init__(self, service, *, registry_path: Optional[str] = None) -> None:
        self._service = service
        self._registry_path = registry_path

    # -- public entry point --------------------------------------------------
    def answer(self, query: str, *, now=None) -> ChatOrchestratorResult:
        intent = route_query(query)
        if intent.mode == ChatMode.UNSUPPORTED_REQUEST:
            return self._unsupported(intent)
        if intent.mode == ChatMode.PROPOSE_MEMORY:
            return self._propose_memory(intent)
        if intent.mode == ChatMode.PROPOSE_SOURCE_UPDATE:
            return self._propose_source_update(intent, now=now)
        return self._answer_from_evidence(intent)

    # -- non-evidence branches ----------------------------------------------
    def _unsupported(self, intent: ChatIntent) -> ChatOrchestratorResult:
        answer_text = format_unsupported_governed_action(intent.query)
        reason = ("Read-only chat: durable mutations and actions are refused; it "
                  "can propose a change or produce a plan for human review.")
        return ChatOrchestratorResult(
            query=intent.query, mode=ChatMode.UNSUPPORTED_REQUEST,
            answer_text=answer_text, refusal_reason=reason,
            intent_mode=intent.mode, route_reason=intent.reason)

    def _propose_memory(self, intent: ChatIntent) -> ChatOrchestratorResult:
        claim = _strip_memory_directive(intent.query)
        proposals = build_memory_proposals([RawMemoryCandidate(text=claim)])
        payload = [p.to_dict() for p in proposals]
        answer_text = (
            f"Prepared {len(proposals)} memory proposal(s) for human review. "
            "Nothing was written to memory; each proposal requires human "
            "approval before it could become durable memory.")
        return ChatOrchestratorResult(
            query=intent.query, mode=ChatMode.PROPOSE_MEMORY,
            answer_text=answer_text,
            proposed_memory_count=len(proposals), proposed_memory=payload,
            intent_mode=intent.mode, route_reason=intent.reason)

    def _propose_source_update(self, intent: ChatIntent, *,
                               now=None) -> ChatOrchestratorResult:
        if not self._registry_path:
            return ChatOrchestratorResult(
                query=intent.query, mode=ChatMode.PROPOSE_SOURCE_UPDATE,
                answer_text="No source registry is configured for this chat "
                            "session, so no source-update proposals can be "
                            "generated.",
                intent_mode=intent.mode, route_reason=intent.reason)
        entries = load_registry(self._registry_path)
        proposals = propose_source_updates(entries, now=now)
        payload = [p.to_dict() for p in proposals]
        answer_text = (
            f"Generated {len(proposals)} source-maintenance proposal(s) for "
            "human review. The registry was not modified; each proposal "
            "requires human approval before any change.")
        return ChatOrchestratorResult(
            query=intent.query, mode=ChatMode.PROPOSE_SOURCE_UPDATE,
            answer_text=answer_text,
            proposed_source_update_count=len(proposals),
            proposed_source_updates=payload,
            intent_mode=intent.mode, route_reason=intent.reason)

    # -- evidence-grounded branches -----------------------------------------
    def _answer_from_evidence(self,
                              intent: ChatIntent) -> ChatOrchestratorResult:
        composer = None
        if intent.wants_report:
            from slm.assistant_composer import ConsultantReportComposer
            composer = ConsultantReportComposer()
        result = self._service.answer_query(intent.query, composer=composer)
        citations = list(result.answer.citations)
        grounded = (not result.refused) and bool(citations)

        if intent.mode == ChatMode.JUDGEMENT_ONLY:
            return self._judgement_result(intent, result, citations, grounded)

        if not grounded:
            return self._insufficient_result(intent, result)

        mode = intent.mode
        if mode == ChatMode.MEMORY_CONTEXT and not _has_memory_citation(citations):
            mode = ChatMode.EVIDENCE_ANSWER
        answer_text = format_evidence_bound_answer(
            result.answer.text, citations, [])
        judgement_present = JUDGEMENT_LABEL in answer_text
        evidence_detail = _evidence_detail_from_audit(
            getattr(result, "audit", None) or {}, citations)
        return ChatOrchestratorResult(
            query=intent.query, mode=mode, answer_text=answer_text,
            citations=citations,
            judgement_labelled=judgement_present,
            evidence_used_count=len(citations),
            evidence_gap_count=0,
            evidence_summary=_evidence_summary_line(len(citations), 0),
            answer_has_citations=bool(citations),
            judgement_present=judgement_present,
            unsupported_claim_count=0,
            evidence_detail=evidence_detail,
            intent_mode=intent.mode, route_reason=intent.reason)

    def _insufficient_result(self, intent: ChatIntent,
                             result) -> ChatOrchestratorResult:
        """Refuse with a constructive next step instead of a dead end.

        The relevance gate's reason makes the gap specific, and the closest
        retrieved (but not grounding) source names are surfaced so the asker can
        narrow or rephrase. All of this is read from the audit the frozen
        pipeline already produced — nothing is retrieved, ranked, or written.
        """
        audit = getattr(result, "audit", None) or {}
        related = _related_sources_from_audit(audit)
        reason = _insufficient_reason(audit)
        missing = summarize_evidence_gaps(audit)
        answer_text = format_insufficient_evidence_answer(
            intent.query, reason, missing, related)
        return ChatOrchestratorResult(
            query=intent.query, mode=ChatMode.INSUFFICIENT_EVIDENCE,
            answer_text=answer_text, refusal_reason=reason,
            related_sources=related,
            evidence_used_count=0,
            evidence_gap_count=len(missing),
            evidence_summary=_evidence_summary_line(0, len(missing)),
            missing_evidence=missing,
            answer_has_citations=False,
            judgement_present=False,
            unsupported_claim_count=0,
            intent_mode=intent.mode, route_reason=intent.reason)

    def _judgement_result(self, intent: ChatIntent, result,
                          citations: List[str],
                          grounded: bool) -> ChatOrchestratorResult:
        if grounded:
            answer_text = format_labelled_judgement(
                result.answer.text, citations, grounded=True)
            evidence_detail = _evidence_detail_from_audit(
                getattr(result, "audit", None) or {}, citations)
            return ChatOrchestratorResult(
                query=intent.query, mode=ChatMode.JUDGEMENT_ONLY,
                answer_text=answer_text, citations=citations,
                judgement_labelled=True,
                evidence_used_count=len(citations),
                evidence_gap_count=0,
                evidence_summary=_evidence_summary_line(len(citations), 0),
                answer_has_citations=bool(citations),
                judgement_present=True,
                unsupported_claim_count=0,
                evidence_detail=evidence_detail,
                intent_mode=intent.mode, route_reason=intent.reason)
        answer_text = format_labelled_judgement("", [], grounded=False)
        return ChatOrchestratorResult(
            query=intent.query, mode=ChatMode.JUDGEMENT_ONLY,
            answer_text=answer_text, judgement_labelled=True,
            refusal_reason=_EVIDENCE_GAP,
            evidence_used_count=0,
            evidence_gap_count=1,
            evidence_summary=_evidence_summary_line(0, 1),
            missing_evidence=[_EVIDENCE_GAP],
            answer_has_citations=False,
            judgement_present=True,
            unsupported_claim_count=0,
            intent_mode=intent.mode, route_reason=intent.reason)


# --------------------------------------------------------------------------- #
# Deterministic rendering
# --------------------------------------------------------------------------- #
def render_chat_result_markdown(result: ChatOrchestratorResult) -> str:
    """Render a chat result as deterministic Markdown for the CLI."""
    lines = [
        "# Chat answer (v6.0; read-only)",
        "",
        f"- Query: {result.query}",
        f"- Mode: {result.mode}",
        f"- Judgement labelled: {str(result.judgement_labelled).lower()}",
        "- State mutation attempted: "
        f"{str(result.state_mutation_attempted).lower()}",
        "",
        "## Answer",
        result.answer_text or "(no answer)",
    ]
    if result.citations:
        lines.append("")
        lines.append("## Citations")
        lines.extend(f"- {c}" for c in result.citations)
    if result.refusal_reason:
        lines.append("")
        lines.append("## Evidence gap")
        lines.append(result.refusal_reason)
    if result.related_sources:
        lines.append("")
        lines.append("## Closest topics in the pack")
        lines.extend(f"- {s}" for s in result.related_sources)
    if result.proposed_memory_count:
        lines.append("")
        lines.append(f"## Proposed memory records ({result.proposed_memory_count})"
                     " — require human approval")
        for p in result.proposed_memory:
            lines.append(f"- [{p.get('proposal_type')}] {p.get('claim')} "
                         f"(status={p.get('status')})")
    if result.proposed_source_update_count:
        lines.append("")
        lines.append("## Proposed source updates "
                     f"({result.proposed_source_update_count}) — require human "
                     "approval")
        for p in result.proposed_source_updates:
            lines.append(f"- [{p.get('proposal_type')}] {p.get('source_id')}: "
                         f"{p.get('proposed_action')} "
                         f"(status={p.get('status')})")
    return "\n".join(lines)
