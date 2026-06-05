"""Consultant Workbench UI view-model layer (UI v1.0) — pure and read-only.

This module turns the existing governed backend into plain-language, plain-data
*view models* for the Consultant Workbench product shell. It is deliberately
**pure and read-only**:

* it imports no writer and calls no mutation path;
* it never activates, deactivates, rolls back, supersedes, writes memory,
  saves a registry/queue, or executes a lifecycle action;
* every loader tolerates missing files and returns an honest empty state rather
  than fabricating metrics.

The Streamlit shell (``app/workbench.py``) renders these view models. All
retrieval, ranking, grounding, composition and governance stay in the backend
layers — this file only *reads* and *labels*.

The spec that requested this work called it "v7.0", but the backend already
ships milestones v7.0–v7.3. To avoid a version collision the product is named
"Consultant Workbench" and ships on its own UI track (``UI v1.0``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Pure, light-weight governance modules (no ML stack). Heavy modules
# (chat orchestrator, workbench service) are imported lazily where needed so a
# read-only page like Sources never pays for the retrieval stack.
from agent import knowledge_pack_activation as _kpa
from agent import lifecycle_action_executor as _lae
from agent import memory_proposal_quality as _mpq
from agent import regression_review_queue as _rrq
from agent import source_registry as _sr

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Product identity
# ---------------------------------------------------------------------------

PRODUCT_NAME = "Consultant Workbench"
CONSOLE_UI_VERSION = "consultant-workbench-ui-v1.0"
PRODUCT_TAGLINE = (
    "Governed answers, evidence, reports, sources and memory in one workspace."
)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NavSection:
    """One top-level workspace in the shell."""

    key: str
    label: str
    blurb: str

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "blurb": self.blurb}


NAV_SECTIONS: Tuple[NavSection, ...] = (
    NavSection("home", "Home", "What this system can do and what needs attention."),
    NavSection("projects", "Projects", "Create a project, add documents, ask questions and export reports."),
    NavSection("ask", "Ask", "Ask a question and inspect the governed answer and evidence."),
    NavSection("reports", "Reports", "Generate a consultant-style written report from evidence."),
    NavSection("sources", "Sources", "The source registry: authority, freshness and warnings."),
    NavSection("imports", "Imports", "Bring new knowledge in — what is supported today."),
    NavSection("packs", "Knowledge Packs", "Which knowledge packs are active right now."),
    NavSection("reviews", "Reviews", "Proposals and actions awaiting human approval."),
    NavSection("monitoring", "Monitoring", "Advisory regression signals on active packs."),
    NavSection("memory", "Memory", "The concept-memory ledger and proposed memories."),
    NavSection("settings", "Settings", "Environment, backend and data locations."),
)


def navigation() -> Tuple[NavSection, ...]:
    """Return the ordered navigation sections for the shell."""

    return NAV_SECTIONS


# ---------------------------------------------------------------------------
# Plain-language label maps
# ---------------------------------------------------------------------------

AUTHORITY_LABELS: Dict[str, str] = {
    "official": "Official",
    "reputable": "Reputable",
    "community": "Community",
    "unknown": "Unknown authority",
}

SOURCE_STATUS_LABELS: Dict[str, str] = {
    "active": "Active",
    "stale": "Needs review (stale)",
    "deprecated": "Deprecated",
    "draft": "Draft",
}

PACK_STATE_LABELS: Dict[str, str] = {
    "imported": "Imported",
    "evaluation_failed": "Evaluation failed",
    "evaluated": "Evaluated",
    "activation_pending": "Activation pending",
    "active": "Active",
    "inactive": "Inactive",
    "superseded": "Superseded",
    "retired": "Retired",
    "blocked": "Blocked",
}

REVIEW_STATUS_LABELS: Dict[str, str] = {
    "pending": "Awaiting review",
    "approved": "Approved",
    "rejected": "Rejected",
    "deferred": "Deferred",
    "more_evidence_requested": "More evidence requested",
    "duplicate": "Duplicate",
    "closed_no_action": "Closed (no action)",
    "stale": "Stale",
    "superseded": "Superseded",
    "action_requested": "Action requested",
    "action_completed": "Action completed",
    "action_failed": "Action failed",
    # action-request statuses
    "requested": "Requested",
    "validated": "Validated (approved for request only)",
    "executed": "Executed",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "written": "Written",
}

RECOMMENDATION_LABELS: Dict[str, str] = {
    "keep_active": "Keep active",
    "keep_active_with_watch": "Keep active — watch",
    "investigate": "Investigate",
    "deactivate_recommended": "Deactivation suggested (advisory)",
    "rollback_recommended": "Rollback suggested (advisory)",
    "block_future_activation": "Block future activation (advisory)",
    "insufficient_evidence_to_recommend": "Insufficient evidence to recommend",
}

# Answer grounding status by chat mode. ``(label, grounded)``.
_MODE_STATUS: Dict[str, Tuple[str, bool]] = {
    "evidence_answer": ("Grounded in evidence", True),
    "report_style_answer": ("Grounded report", True),
    "memory_context": ("Memory context", True),
    "judgement_only": ("Judgement only — not grounded", False),
    "insufficient_evidence": ("Insufficient evidence", False),
    "propose_memory": ("Proposal drafted — nothing written", False),
    "propose_source_update": ("Proposal drafted — nothing written", False),
    "unsupported_request": ("Unsupported request", False),
}

_MODE_NEXT_ACTION: Dict[str, str] = {
    "evidence_answer": "Review the cited evidence before acting on this answer.",
    "report_style_answer": "Review the report's citations before sharing it.",
    "memory_context": "This reflects remembered context; confirm against current sources.",
    "judgement_only": "Treat the judgement as advisory; confirm with evidence before acting.",
    "insufficient_evidence": "Add a source or refine the question — nothing was answered from weak evidence.",
    "propose_memory": "A memory was drafted for human review. Nothing was written.",
    "propose_source_update": "A source update was drafted for human review. Nothing was written.",
    "unsupported_request": "This request is governed; use the Reviews workspace to action it.",
}

# Answer-mode options offered on the Ask page. ``Auto`` uses the governed router.
ANSWER_MODES: Tuple[Tuple[str, str], ...] = (
    ("auto", "Auto (governed routing)"),
    ("evidence_answer", "Evidence answer"),
    ("report_style_answer", "Consultant report"),
    ("judgement_only", "Judgement"),
    ("memory_context", "Memory context"),
)


def _judgement_label() -> str:
    """Return the canonical judgement marker, degrading gracefully."""

    try:  # pragma: no cover - exercised indirectly; guarded for light imports
        from slm.assistant_composer import JUDGEMENT_LABEL

        return JUDGEMENT_LABEL
    except Exception:  # pragma: no cover - defensive only
        return "[JUDGEMENT — not grounded in evidence]"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsoleConfig:
    """Resolved data locations the read-only view models load from."""

    root: Path
    ledger_path: Path
    registry_path: Path
    pack_state_path: Path
    pack_audit_path: Path
    packs_root: Path
    memory_review_path: Path
    source_review_path: Path
    regression_queue_path: Path
    regression_actions_path: Path
    monitoring_history_path: Path
    follow_up_path: Path
    activation_blocks_path: Path
    execution_results_path: Path
    environment: str = "default"
    backend: str = "hybrid"

    @classmethod
    def default(cls, root: Optional[Path] = None) -> "ConsoleConfig":
        base = Path(root) if root is not None else ROOT
        return cls(
            root=base,
            ledger_path=base / "demos" / "workbench_ledger.jsonl",
            registry_path=base / "demos" / "source_registry.jsonl",
            pack_state_path=base / "config" / "active_knowledge_packs.jsonl",
            pack_audit_path=base / "reports" / "knowledge_pack_activation_audit.jsonl",
            packs_root=base / "demos" / "packs",
            memory_review_path=base / "reports" / "memory_proposal_review_queue.jsonl",
            source_review_path=base / "reports" / "source_proposal_review_queue.jsonl",
            regression_queue_path=base / "reviews" / "regression_review_queue.jsonl",
            regression_actions_path=base / "reviews" / "regression_action_requests.jsonl",
            monitoring_history_path=base / "reports" / "active_pack_monitoring.jsonl",
            follow_up_path=base / "reviews" / "operational_follow_up.jsonl",
            activation_blocks_path=base / "reviews" / "activation_blocks.jsonl",
            execution_results_path=base / "reviews" / "lifecycle_execution_results.jsonl",
        )


# ---------------------------------------------------------------------------
# Shared view-model primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionError:
    """A section that could not load — surfaced safely, never crashes the app."""

    section: str
    message: str

    def to_dict(self) -> dict:
        return {"section": self.section, "message": self.message}


@dataclass(frozen=True)
class MetricCard:
    """A headline number for the Home dashboard."""

    key: str
    label: str
    value: str
    detail: str = ""
    tone: str = "neutral"  # neutral | good | warn | critical | unavailable

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "detail": self.detail,
            "tone": self.tone,
        }


@dataclass(frozen=True)
class EmptyState:
    """A clear, honest empty state for a section that has no data."""

    title: str
    detail: str

    def to_dict(self) -> dict:
        return {"title": self.title, "detail": self.detail}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file into a list of dicts, tolerating a missing file."""

    if not path.exists():
        return []
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _enum_value(value: Any) -> str:
    """Normalise an enum / string status into its plain string value."""

    return getattr(value, "value", value) if value is not None else ""


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HomeSummary:
    product_name: str
    tagline: str
    ui_version: str
    environment: str
    backend: str
    capabilities: Tuple["Capability", ...]
    metrics: Tuple[MetricCard, ...]
    plain_status: Tuple[str, ...]
    quick_links: Tuple[Tuple[str, str], ...]
    errors: Tuple[SectionError, ...] = ()

    def to_dict(self) -> dict:
        return {
            "product_name": self.product_name,
            "tagline": self.tagline,
            "ui_version": self.ui_version,
            "environment": self.environment,
            "backend": self.backend,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "metrics": [m.to_dict() for m in self.metrics],
            "plain_status": list(self.plain_status),
            "quick_links": [list(q) for q in self.quick_links],
            "errors": [e.to_dict() for e in self.errors],
        }


@dataclass(frozen=True)
class Capability:
    title: str
    description: str
    target: str  # nav key

    def to_dict(self) -> dict:
        return {"title": self.title, "description": self.description, "target": self.target}


CAPABILITIES: Tuple[Capability, ...] = (
    Capability("Ask grounded questions", "Get answers tied to cited evidence, with gaps shown honestly.", "ask"),
    Capability("Write consultant reports", "Turn evidence into a structured written report.", "reports"),
    Capability("Govern your sources", "Track authority, freshness and supersession across sources.", "sources"),
    Capability("Run knowledge packs", "See exactly which knowledge packs are live.", "packs"),
    Capability("Review before change", "Approve proposals and actions — approval is separate from execution.", "reviews"),
    Capability("Watch for regressions", "Read advisory monitoring signals on active packs.", "monitoring"),
)


def build_home(config: ConsoleConfig, *, now: Optional[datetime] = None) -> HomeSummary:
    """Assemble the Home dashboard from read-only sources, fault-isolated."""

    now = now or _utc_now()
    metrics: List[MetricCard] = []
    plain: List[str] = []
    errors: List[SectionError] = []

    # Packs ---------------------------------------------------------------
    try:
        packs = build_packs(config)
        metrics.append(MetricCard(
            "active_packs", "Active knowledge packs", str(packs.active_count),
            "Live in this environment", "good" if packs.active_count else "neutral"))
        if packs.active_count:
            plain.append(f"{packs.active_count} knowledge pack(s) are active.")
        else:
            plain.append("No knowledge packs are active.")
    except Exception as exc:  # pragma: no cover - defensive aggregation guard
        errors.append(SectionError("packs", str(exc)))

    # Sources -------------------------------------------------------------
    try:
        sources = build_sources(config, now=now)
        stale = sum(1 for r in sources.rows if r.status == "stale")
        warn = sources.warning_count + sources.error_count
        metrics.append(MetricCard(
            "sources", "Sources", str(len(sources.rows)),
            f"{stale} need review" if stale else "All within freshness policy",
            "warn" if stale else "neutral"))
        if warn:
            plain.append(f"{warn} source registry warning(s) need attention.")
        if stale:
            plain.append(f"{stale} source(s) are stale and should be reviewed.")
    except Exception as exc:  # pragma: no cover
        errors.append(SectionError("sources", str(exc)))

    # Reviews -------------------------------------------------------------
    try:
        reviews = build_reviews(config)
        metrics.append(MetricCard(
            "pending_reviews", "Awaiting review", str(reviews.pending_count),
            "Across proposals and actions",
            "warn" if reviews.pending_count else "good"))
        if reviews.pending_count:
            plain.append(f"{reviews.pending_count} item(s) are waiting for human review.")
        else:
            plain.append("Nothing is waiting for review.")
    except Exception as exc:  # pragma: no cover
        errors.append(SectionError("reviews", str(exc)))

    # Monitoring ----------------------------------------------------------
    try:
        monitoring = build_monitoring(config)
        if not monitoring.has_history:
            metrics.append(MetricCard(
                "monitoring", "Monitoring", "—",
                "No monitoring runs recorded yet", "unavailable"))
            plain.append("No monitoring history yet.")
        else:
            crit = len(monitoring.critical_findings)
            metrics.append(MetricCard(
                "monitoring", "Critical monitoring signals", str(crit),
                "Advisory only — never auto-applied",
                "critical" if crit else "good"))
            if crit:
                plain.append(f"{crit} critical monitoring signal(s) — advisory only.")
            else:
                plain.append("No critical monitoring signals (advisory).")
    except Exception as exc:  # pragma: no cover
        errors.append(SectionError("monitoring", str(exc)))

    quick_links = (
        ("Ask a question", "ask"),
        ("Review pending items", "reviews"),
        ("Check sources", "sources"),
        ("See active packs", "packs"),
    )

    return HomeSummary(
        product_name=PRODUCT_NAME,
        tagline=PRODUCT_TAGLINE,
        ui_version=CONSOLE_UI_VERSION,
        environment=config.environment,
        backend=config.backend,
        capabilities=CAPABILITIES,
        metrics=tuple(metrics),
        plain_status=tuple(plain),
        quick_links=quick_links,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerSection:
    """One visually distinct block of an answer."""

    title: str
    kind: str  # factual | judgement | gap | meta
    body: str = ""
    items: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "kind": self.kind,
            "body": self.body,
            "items": list(self.items),
        }


@dataclass(frozen=True)
class AskView:
    query: str
    mode: str
    status_label: str
    grounded: bool
    answer_text: str
    sections: Tuple[AnswerSection, ...]
    citations: Tuple[str, ...]
    evidence_used_count: int
    evidence_gap_count: int
    evidence_summary: str
    missing_evidence: Tuple[str, ...]
    related_sources: Tuple[str, ...]
    judgement_present: bool
    refused: bool
    refusal_reason: str
    safe_next_action: str
    proposed_memory_count: int
    proposed_source_update_count: int
    state_mutation_attempted: bool

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "status_label": self.status_label,
            "grounded": self.grounded,
            "answer_text": self.answer_text,
            "sections": [s.to_dict() for s in self.sections],
            "citations": list(self.citations),
            "evidence_used_count": self.evidence_used_count,
            "evidence_gap_count": self.evidence_gap_count,
            "evidence_summary": self.evidence_summary,
            "missing_evidence": list(self.missing_evidence),
            "related_sources": list(self.related_sources),
            "judgement_present": self.judgement_present,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "safe_next_action": self.safe_next_action,
            "proposed_memory_count": self.proposed_memory_count,
            "proposed_source_update_count": self.proposed_source_update_count,
            "state_mutation_attempted": self.state_mutation_attempted,
        }


def _split_factual_judgement(answer_text: str) -> Tuple[str, str]:
    """Split a composed answer into (factual, judgement) on the judgement marker."""

    label = _judgement_label()
    if label and label in answer_text:
        head, _, tail = answer_text.partition(label)
        return head.strip(), (label + tail).strip()
    return answer_text.strip(), ""


def build_ask(result: Any) -> AskView:
    """Shape a ``ChatOrchestratorResult`` into a read-only Ask view model.

    Pure function: it reads attributes off the result and never mutates state.
    """

    mode = str(getattr(result, "mode", "") or "")
    answer_text = str(getattr(result, "answer_text", "") or "")
    citations = tuple(str(c) for c in (getattr(result, "citations", ()) or ()))
    missing = tuple(str(m) for m in (getattr(result, "missing_evidence", ()) or ()))
    related = tuple(str(s) for s in (getattr(result, "related_sources", ()) or ()))
    evidence_used = int(getattr(result, "evidence_used_count", 0) or 0)
    evidence_gaps = int(getattr(result, "evidence_gap_count", 0) or 0)
    evidence_summary = str(getattr(result, "evidence_summary", "") or "")
    judgement_present = bool(getattr(result, "judgement_present", False)
                             or getattr(result, "judgement_labelled", False))
    has_citations = bool(getattr(result, "answer_has_citations", bool(citations)))
    refusal_reason = str(getattr(result, "refusal_reason", "") or "")

    status_label, grounded = _MODE_STATUS.get(mode, ("Answer", bool(has_citations)))
    if mode in ("evidence_answer", "report_style_answer") and not has_citations:
        status_label, grounded = "Partially grounded", False

    factual, judgement = _split_factual_judgement(answer_text)

    sections: List[AnswerSection] = []
    if mode == "judgement_only":
        sections.append(AnswerSection("Judgement", "judgement", body=answer_text))
    else:
        if factual:
            sections.append(AnswerSection("Answer", "factual", body=factual))
        if judgement:
            sections.append(AnswerSection("Judgement", "judgement", body=judgement))

    if citations:
        sections.append(AnswerSection("Citations", "factual", items=citations))
    if evidence_summary:
        sections.append(AnswerSection("Evidence used", "factual", body=evidence_summary))
    if missing or evidence_gaps:
        gap_body = "" if missing else f"{evidence_gaps} evidence gap(s) identified."
        sections.append(AnswerSection("Evidence gaps", "gap", body=gap_body, items=missing))
    if related and mode == "insufficient_evidence":
        sections.append(AnswerSection("Related sources", "meta", items=related))

    refused = mode in ("insufficient_evidence", "unsupported_request") or bool(refusal_reason)
    next_action = _MODE_NEXT_ACTION.get(mode, "Review the answer and its evidence before acting.")

    return AskView(
        query=str(getattr(result, "query", "") or ""),
        mode=mode,
        status_label=status_label,
        grounded=grounded,
        answer_text=answer_text,
        sections=tuple(sections),
        citations=citations,
        evidence_used_count=evidence_used,
        evidence_gap_count=evidence_gaps,
        evidence_summary=evidence_summary,
        missing_evidence=missing,
        related_sources=related,
        judgement_present=judgement_present,
        refused=refused,
        refusal_reason=refusal_reason,
        safe_next_action=next_action,
        proposed_memory_count=int(getattr(result, "proposed_memory_count", 0) or 0),
        proposed_source_update_count=int(getattr(result, "proposed_source_update_count", 0) or 0),
        state_mutation_attempted=bool(getattr(result, "state_mutation_attempted", False)),
    )


def answer_ask(service: Any, query: str, *,
               registry_path: Optional[Path] = None) -> Any:
    """Run the governed orchestrator and return its raw read-only result.

    Centralises orchestrator construction so the Ask page can build several
    views (answer, evidence inspector, citations) from a single run without
    invoking retrieval more than once. The orchestrator is read-only by design;
    this never writes durable state.
    """

    from agent.chat_orchestrator import ChatOrchestrator

    orchestrator = ChatOrchestrator(
        service, registry_path=str(registry_path) if registry_path else None)
    return orchestrator.answer(query)


def run_ask(service: Any, query: str, *, registry_path: Optional[Path] = None) -> AskView:
    """Convenience: run the governed orchestrator and shape the result.

    The orchestrator is read-only by design; this never writes durable state.
    """

    return build_ask(answer_ask(service, query, registry_path=registry_path))


# ---------------------------------------------------------------------------
# Ask — evidence inspector and citations (read-only projections)
# ---------------------------------------------------------------------------

# Plain-language gloss for the three evidence stages, surfaced verbatim in the
# UI so a reader never mistakes a relevance score for proof.
EVIDENCE_STAGE_GLOSS: Tuple[str, ...] = (
    "Retrieved means the pipeline considered it as potentially relevant.",
    "Cited means it was referenced in the final factual answer.",
    "A relevance score measures match strength — it is not a confidence or a "
    "measure of truth.",
)

AUTHORITY_LABELS: Dict[str, str] = {
    "official": "Official",
    "reputable": "Reputable",
    "trusted": "Trusted",
    "community": "Community",
    "unknown": "Unknown authority",
    "": "Unknown authority",
}


@dataclass(frozen=True)
class EvidenceItemView:
    """One retrieved knowledge candidate, shaped for the evidence inspector.

    ``score`` is the backend's relevance activation, never a confidence. ``cited``
    marks whether this candidate was referenced in the final factual answer.
    """

    citation_id: str
    source_id: str
    chunk_id: str
    source_name: str
    domain: str
    authority: str
    authority_label: str
    version: str
    section: str
    score: float
    rank: int
    backend_name: str
    cited: bool
    supporting_text: str

    def to_dict(self) -> dict:
        return {
            "citation_id": self.citation_id,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "source_name": self.source_name,
            "domain": self.domain,
            "authority": self.authority,
            "authority_label": self.authority_label,
            "version": self.version,
            "section": self.section,
            "score": self.score,
            "rank": self.rank,
            "backend_name": self.backend_name,
            "cited": self.cited,
            "supporting_text": self.supporting_text,
        }


@dataclass(frozen=True)
class EvidenceInspectorView:
    """Retrieved-vs-cited evidence stages for one answer (read-only).

    The current pipeline surfaces two genuine, distinct stages — every retrieved
    candidate and the subset cited in the final answer. A separate post-gate
    "selected" stage is not exposed by the backend, so it is not simulated.
    """

    retrieved: Tuple[EvidenceItemView, ...]
    cited: Tuple[EvidenceItemView, ...]
    retrieved_count: int
    cited_count: int
    stage_gloss: Tuple[str, ...]
    available: bool

    def to_dict(self) -> dict:
        return {
            "retrieved": [e.to_dict() for e in self.retrieved],
            "cited": [e.to_dict() for e in self.cited],
            "retrieved_count": self.retrieved_count,
            "cited_count": self.cited_count,
            "stage_gloss": list(self.stage_gloss),
            "available": self.available,
        }


_MAX_SUPPORTING_CHARS = 320


def _bounded_extract(text: str, *, limit: int = _MAX_SUPPORTING_CHARS) -> str:
    """Return a bounded supporting extract; never the full document body."""

    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _evidence_item_from_detail(detail: dict) -> EvidenceItemView:
    authority = str(detail.get("authority") or "")
    return EvidenceItemView(
        citation_id=str(detail.get("citation_id") or ""),
        source_id=str(detail.get("source_id") or ""),
        chunk_id=str(detail.get("chunk_id") or ""),
        source_name=str(detail.get("source_name") or ""),
        domain=str(detail.get("domain") or ""),
        authority=authority,
        authority_label=AUTHORITY_LABELS.get(authority, authority or "Unknown authority"),
        version=str(detail.get("version") or ""),
        section=str(detail.get("section") or ""),
        score=float(detail.get("score") or 0.0),
        rank=int(detail.get("rank") or 0),
        backend_name=str(detail.get("backend_name") or ""),
        cited=bool(detail.get("cited")),
        supporting_text=_bounded_extract(str(detail.get("text") or "")),
    )


def build_evidence_inspector(result: Any) -> EvidenceInspectorView:
    """Shape an orchestrator result's ``evidence_detail`` into the inspector.

    Pure: it reads the read-only projection the orchestrator already produced
    (one dict per retrieved candidate) and never triggers retrieval or a write.
    """

    detail = list(getattr(result, "evidence_detail", ()) or ())
    items = [_evidence_item_from_detail(d) for d in detail]
    retrieved = tuple(items)
    cited = tuple(i for i in items if i.cited)
    return EvidenceInspectorView(
        retrieved=retrieved,
        cited=cited,
        retrieved_count=len(retrieved),
        cited_count=len(cited),
        stage_gloss=EVIDENCE_STAGE_GLOSS,
        available=bool(retrieved),
    )


@dataclass(frozen=True)
class CitationView:
    """A single citation with the provenance the backend actually exposes.

    Fields the governed pipeline does not surface (page/row, pack id/version)
    are intentionally absent rather than inferred — citations are never invented.
    """

    citation_id: str
    kind: str  # knowledge | memory
    source_name: str
    source_id: str
    chunk_id: str
    authority: str
    authority_label: str
    version: str
    section: str
    supporting_text: str

    def to_dict(self) -> dict:
        return {
            "citation_id": self.citation_id,
            "kind": self.kind,
            "source_name": self.source_name,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "authority": self.authority,
            "authority_label": self.authority_label,
            "version": self.version,
            "section": self.section,
            "supporting_text": self.supporting_text,
        }


def build_citation_details(result: Any) -> Tuple[CitationView, ...]:
    """Build per-citation cards from the orchestrator result (read-only).

    Knowledge citations (``src:<chunk_id>``) are enriched from the evidence
    detail projection; memory citations (``mem:<id>``) are preserved as-is with
    no fabricated provenance. Citation ids are kept exactly as the composer
    emitted them.
    """

    by_citation: Dict[str, dict] = {}
    for d in getattr(result, "evidence_detail", ()) or ():
        cid = str(d.get("citation_id") or "")
        if cid:
            by_citation.setdefault(cid, d)

    cards: List[CitationView] = []
    for cid in getattr(result, "citations", ()) or ():
        cid = str(cid)
        detail = by_citation.get(cid)
        if detail is not None:
            authority = str(detail.get("authority") or "")
            cards.append(CitationView(
                citation_id=cid,
                kind="knowledge",
                source_name=str(detail.get("source_name") or ""),
                source_id=str(detail.get("source_id") or ""),
                chunk_id=str(detail.get("chunk_id") or ""),
                authority=authority,
                authority_label=AUTHORITY_LABELS.get(
                    authority, authority or "Unknown authority"),
                version=str(detail.get("version") or ""),
                section=str(detail.get("section") or ""),
                supporting_text=_bounded_extract(str(detail.get("text") or "")),
            ))
        else:
            kind = "memory" if cid.startswith("mem:") else "knowledge"
            cards.append(CitationView(
                citation_id=cid, kind=kind, source_name="", source_id="",
                chunk_id=cid.split(":", 1)[1] if ":" in cid else cid,
                authority="", authority_label="Unknown authority",
                version="", section="", supporting_text=""))
    return tuple(cards)


# ---------------------------------------------------------------------------
# Ask — evidence scope (read-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AskScopeView:
    """The governed evidence scope a query will run against, shown pre-submit.

    Read-only: it reports how many active knowledge packs and approved sources
    are in scope so the asker can see what is (and is not) being searched.
    Memory context is off by default and never silently enabled.
    """

    active_pack_count: int
    approved_source_count: int
    memory_context_enabled: bool
    plain_summary: Tuple[str, ...]
    warnings: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "active_pack_count": self.active_pack_count,
            "approved_source_count": self.approved_source_count,
            "memory_context_enabled": self.memory_context_enabled,
            "plain_summary": list(self.plain_summary),
            "warnings": list(self.warnings),
        }


def build_ask_scope(config: ConsoleConfig, *,
                    memory_context_enabled: bool = False) -> AskScopeView:
    """Summarise the governed active-knowledge scope for the Ask page.

    Counts only governed *active* packs and *active* approved sources, reusing
    the existing read-only pack and source view models. Never searches inactive,
    blocked, retired, or superseded knowledge.
    """

    warnings: List[str] = []
    try:
        active_packs = build_packs(config).active_count
    except Exception:  # pragma: no cover - defensive aggregation guard
        active_packs = 0
    try:
        sources = build_sources(config)
        approved_sources = sum(1 for r in sources.rows if r.status == "active")
    except Exception:  # pragma: no cover - defensive aggregation guard
        approved_sources = 0

    if active_packs == 0:
        warnings.append(
            "No active knowledge packs — answers will rely on approved sources "
            "only, and may be insufficient.")
    if active_packs == 0 and approved_sources == 0:
        warnings.append(
            "Evidence scope is empty. Add data or activate a pack before asking.")

    plain = (
        f"{active_packs} active knowledge pack(s)",
        f"{approved_sources} approved source(s)",
        "Memory context enabled" if memory_context_enabled
        else "Memory context disabled",
    )
    return AskScopeView(
        active_pack_count=active_packs,
        approved_source_count=approved_sources,
        memory_context_enabled=memory_context_enabled,
        plain_summary=plain,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Ask — export (pure serialisers; the only writer is the caller's file write)
# ---------------------------------------------------------------------------


def render_ask_markdown(view: AskView, *,
                        citations: Tuple[CitationView, ...] = (),
                        inspector: Optional[EvidenceInspectorView] = None,
                        generated_at: str = "") -> str:
    """Serialise an answer to Markdown, preserving evidence and labels.

    Pure string builder — it writes nothing. Citation ids, judgement labels,
    evidence gaps, and the selected status are all preserved so an exported
    answer stays auditable. Full source bodies are never embedded; only the
    bounded supporting extracts already present in the views are included.
    """

    lines: List[str] = [f"# Ask answer — {view.status_label}", ""]
    if generated_at:
        lines.append(f"_Generated: {generated_at}_")
        lines.append("")
    lines.append(f"**Question:** {view.query}")
    lines.append(f"**Answer mode:** {view.mode}")
    lines.append(f"**Grounded:** {'yes' if view.grounded else 'no'}")
    lines.append("")
    for section in view.sections:
        lines.append(f"## {section.title}")
        if section.body:
            lines.append(section.body)
        for item in section.items:
            lines.append(f"- {item}")
        lines.append("")
    if citations:
        lines.append("## Citations")
        for c in citations:
            prov = " · ".join(p for p in (
                c.source_name, c.authority_label,
                f"v{c.version}" if c.version else "") if p)
            lines.append(f"- `{c.citation_id}`" + (f" — {prov}" if prov else ""))
        lines.append("")
    if inspector is not None and inspector.available:
        lines.append("## Evidence inspector")
        lines.append(
            f"Retrieved {inspector.retrieved_count}; cited {inspector.cited_count}. "
            "Relevance score is match strength, not confidence.")
        for e in inspector.retrieved:
            tag = "cited" if e.cited else "retrieved only"
            lines.append(
                f"- `{e.citation_id}` ({tag}) — {e.source_name} · "
                f"{e.authority_label} · score {e.score:.3f} · rank {e.rank}")
        lines.append("")
    lines.append(f"**Next safe action:** {view.safe_next_action}")
    lines.append("")
    lines.append(
        "_Retrieval is not truth. Citations show the evidence used; authority "
        "and relevance are different. Judgement is labelled separately. "
        "Insufficient evidence is a valid outcome. Ask never writes memory._")
    return "\n".join(lines).rstrip() + "\n"


def render_ask_json(view: AskView, *,
                    citations: Tuple[CitationView, ...] = (),
                    inspector: Optional[EvidenceInspectorView] = None,
                    generated_at: str = "") -> str:
    """Serialise an answer to a JSON evidence bundle (pure; writes nothing)."""

    bundle = {
        "_record": "ask_evidence_bundle",
        "generated_at": generated_at,
        "answer": view.to_dict(),
        "citations": [c.to_dict() for c in citations],
        "evidence_inspector": inspector.to_dict() if inspector is not None else None,
        "disclaimers": list(EVIDENCE_STAGE_GLOSS),
    }
    return json.dumps(bundle, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRow:
    source_id: str
    title: str
    source_type: str
    authority: str
    authority_label: str
    status: str
    status_label: str
    owner: str
    last_reviewed_at: str
    topics: Tuple[str, ...]
    superseded_by: str

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "source_type": self.source_type,
            "authority": self.authority,
            "authority_label": self.authority_label,
            "status": self.status,
            "status_label": self.status_label,
            "owner": self.owner,
            "last_reviewed_at": self.last_reviewed_at,
            "topics": list(self.topics),
            "superseded_by": self.superseded_by,
        }


@dataclass(frozen=True)
class SourcesView:
    rows: Tuple[SourceRow, ...]
    warnings: Tuple[str, ...]
    warning_count: int
    error_count: int
    info_count: int
    empty: Optional[EmptyState]
    errors: Tuple[SectionError, ...] = ()

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "warnings": list(self.warnings),
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "info_count": self.info_count,
            "empty": self.empty.to_dict() if self.empty else None,
            "errors": [e.to_dict() for e in self.errors],
        }


def build_sources(config: ConsoleConfig, *, now: Optional[datetime] = None) -> SourcesView:
    """Build the source-registry view, tolerating a missing registry file."""

    now = now or _utc_now()
    if not config.registry_path.exists():
        return SourcesView(
            rows=(), warnings=(), warning_count=0, error_count=0, info_count=0,
            empty=EmptyState("No source registry", "No sources are registered in this build yet."))

    entries = _sr.load_registry(config.registry_path)
    rows: List[SourceRow] = []
    for entry in entries:
        try:
            status = _enum_value(_sr.compute_effective_status(entry, now=now))
        except Exception:  # pragma: no cover - defensive per-row guard
            status = _enum_value(getattr(entry, "status", ""))
        authority = _enum_value(getattr(entry, "authority_level", ""))
        rows.append(SourceRow(
            source_id=str(getattr(entry, "source_id", "")),
            title=str(getattr(entry, "title", "")),
            source_type=str(getattr(entry, "source_type", "")),
            authority=authority,
            authority_label=AUTHORITY_LABELS.get(authority, authority or "Unknown"),
            status=status,
            status_label=SOURCE_STATUS_LABELS.get(status, status or "Unknown"),
            owner=str(getattr(entry, "owner", "")),
            last_reviewed_at=str(getattr(entry, "last_reviewed_at", "") or ""),
            topics=tuple(str(t) for t in (getattr(entry, "topics", ()) or ())),
            superseded_by=str(getattr(entry, "superseded_by", "") or ""),
        ))

    warnings: List[str] = list(_sr.supersession_warnings(entries))
    warn_count = err_count = info_count = 0
    try:
        report = _sr.audit_registry(entries, now=now)
        warn_count = int(getattr(report, "warning_count", 0))
        err_count = int(getattr(report, "error_count", 0))
        info_count = int(getattr(report, "info_count", 0))
    except Exception:  # pragma: no cover - defensive
        pass

    return SourcesView(
        rows=tuple(rows), warnings=tuple(warnings),
        warning_count=warn_count, error_count=err_count, info_count=info_count,
        empty=None if rows else EmptyState("No sources", "The registry is empty."))


# ---------------------------------------------------------------------------
# Knowledge packs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackRow:
    pack_id: str
    pack_version: str
    status: str
    status_label: str
    is_active: bool
    source_type: str
    source_id: str
    authority: str
    intended_use: str
    fingerprint: str
    environment: str

    def to_dict(self) -> dict:
        return {
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "status": self.status,
            "status_label": self.status_label,
            "is_active": self.is_active,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "authority": self.authority,
            "intended_use": self.intended_use,
            "fingerprint": self.fingerprint,
            "environment": self.environment,
        }


@dataclass(frozen=True)
class PacksView:
    rows: Tuple[PackRow, ...]
    active_count: int
    inactive_count: int
    empty: Optional[EmptyState]
    errors: Tuple[SectionError, ...] = ()

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "active_count": self.active_count,
            "inactive_count": self.inactive_count,
            "empty": self.empty.to_dict() if self.empty else None,
            "errors": [e.to_dict() for e in self.errors],
        }


def build_packs(config: ConsoleConfig) -> PacksView:
    """Build the active-pack view from the governed activation manifest."""

    manager = _kpa.ActivationStateManager(
        state_path=config.pack_state_path, audit_path=config.pack_audit_path)
    state = manager.load_state()
    records = list(getattr(state, "records", ()) or ())
    if not records:
        return PacksView(rows=(), active_count=0, inactive_count=0,
                         empty=EmptyState("No activated packs",
                                          "No knowledge packs have been activated in this build."))

    active_ids = set(_kpa.select_active_pack_ids(state, environment=config.environment))
    rows: List[PackRow] = []
    for rec in records:
        status = _enum_value(getattr(rec, "status", ""))
        pack_id = str(getattr(rec, "pack_id", ""))
        is_active = pack_id in active_ids or status == "active"
        rows.append(PackRow(
            pack_id=pack_id,
            pack_version=str(getattr(rec, "pack_version", "")),
            status=status,
            status_label=PACK_STATE_LABELS.get(status, status or "Unknown"),
            is_active=is_active,
            source_type=str(getattr(rec, "source_type", "")),
            source_id=str(getattr(rec, "source_id", "")),
            authority=_enum_value(getattr(rec, "authority_level", "")),
            intended_use=str(getattr(rec, "intended_use", "") or ""),
            fingerprint=str(getattr(rec, "pack_fingerprint", "")),
            environment=str(getattr(rec, "environment", "") or "default"),
        ))

    active_count = sum(1 for r in rows if r.is_active)
    return PacksView(rows=tuple(rows), active_count=active_count,
                     inactive_count=len(rows) - active_count, empty=None)


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------

_PENDING_STATUSES = {"pending", "requested", "more_evidence_requested"}
_STALE_STATUSES = {"stale", "superseded", "cancelled"}


@dataclass(frozen=True)
class ReviewRow:
    category: str
    item_id: str
    title: str
    detail: str
    status: str
    status_label: str
    is_pending: bool
    is_stale: bool
    can_approve: bool
    affected_packs: Tuple[str, ...]
    created_at: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "item_id": self.item_id,
            "title": self.title,
            "detail": self.detail,
            "status": self.status,
            "status_label": self.status_label,
            "is_pending": self.is_pending,
            "is_stale": self.is_stale,
            "can_approve": self.can_approve,
            "affected_packs": list(self.affected_packs),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ReviewCategory:
    key: str
    label: str
    rows: Tuple[ReviewRow, ...]
    pending: int
    empty: Optional[EmptyState]

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "rows": [r.to_dict() for r in self.rows],
            "pending": self.pending,
            "empty": self.empty.to_dict() if self.empty else None,
        }


@dataclass(frozen=True)
class ReviewsView:
    categories: Tuple[ReviewCategory, ...]
    pending_count: int
    errors: Tuple[SectionError, ...] = ()

    def to_dict(self) -> dict:
        return {
            "categories": [c.to_dict() for c in self.categories],
            "pending_count": self.pending_count,
            "errors": [e.to_dict() for e in self.errors],
        }


def _make_review_row(category: str, item_id: str, title: str, detail: str,
                     status: str, affected: Sequence[str], created_at: str) -> ReviewRow:
    status = (status or "").lower()
    is_pending = status in _PENDING_STATUSES
    is_stale = status in _STALE_STATUSES
    return ReviewRow(
        category=category, item_id=item_id, title=title, detail=detail,
        status=status, status_label=REVIEW_STATUS_LABELS.get(status, status or "Unknown"),
        is_pending=is_pending, is_stale=is_stale,
        # Approval is only ever offered for genuinely pending, non-stale items.
        can_approve=is_pending and not is_stale,
        affected_packs=tuple(str(p) for p in affected), created_at=str(created_at or ""))


def build_reviews(config: ConsoleConfig) -> ReviewsView:
    """Aggregate every human-review queue into one read-only view, fault-isolated."""

    categories: List[ReviewCategory] = []
    errors: List[SectionError] = []

    def _category(key: str, label: str, rows: List[ReviewRow], empty_detail: str) -> None:
        pending = sum(1 for r in rows if r.is_pending)
        categories.append(ReviewCategory(
            key=key, label=label, rows=tuple(rows), pending=pending,
            empty=None if rows else EmptyState(f"No {label.lower()}", empty_detail)))

    # Memory proposals ----------------------------------------------------
    try:
        rows: List[ReviewRow] = []
        for r in _mpq.load_memory_review_queue(config.memory_review_path):
            rows.append(_make_review_row(
                "memory", str(getattr(r, "proposal_id", "")),
                str(getattr(r, "claim", "")) or str(getattr(r, "proposal_type", "")),
                f"Proposed {getattr(r, 'proposal_type', 'memory')}",
                _enum_value(getattr(r, "review_status", "")), (),
                str(getattr(r, "created_at", ""))))
        _category("memory", "Memory proposals", rows,
                  "No memory proposals are awaiting review.")
    except Exception as exc:  # pragma: no cover
        errors.append(SectionError("memory_proposals", str(exc)))

    # Source proposals ----------------------------------------------------
    try:
        rows = []
        for r in _sr.load_review_queue(config.source_review_path):
            rows.append(_make_review_row(
                "source", str(getattr(r, "proposal_id", "")),
                str(getattr(r, "source_id", "")),
                f"{getattr(r, 'proposal_type', 'update')} ({getattr(r, 'finding_code', '')})",
                _enum_value(getattr(r, "review_status", "")), (),
                str(getattr(r, "created_at", ""))))
        _category("source", "Source proposals", rows,
                  "No source updates are awaiting review.")
    except Exception as exc:  # pragma: no cover
        errors.append(SectionError("source_proposals", str(exc)))

    # Regression review items --------------------------------------------
    try:
        rows = []
        for r in _rrq.load_review_queue(config.regression_queue_path):
            rows.append(_make_review_row(
                "regression", str(getattr(r, "review_item_id", "")),
                _enum_value(getattr(r, "recommendation", "")) or "Regression review",
                f"Severity: {_enum_value(getattr(r, 'severity', ''))} · "
                f"proposed: {getattr(r, 'proposed_action_type', '')}",
                _enum_value(getattr(r, "status", "")),
                getattr(r, "affected_pack_ids", ()) or (),
                str(getattr(r, "created_at", ""))))
        _category("regression", "Regression reviews", rows,
                  "No regression reviews are open.")
    except Exception as exc:  # pragma: no cover
        errors.append(SectionError("regression_reviews", str(exc)))

    # Lifecycle action requests (approval is NOT execution) ---------------
    try:
        rows = []
        for r in _rrq.load_action_requests(config.regression_actions_path):
            rows.append(_make_review_row(
                "action", str(getattr(r, "action_request_id", "")),
                _enum_value(getattr(r, "requested_action", "")) or "Lifecycle action",
                "Approval here authorises the request only — not execution.",
                _enum_value(getattr(r, "status", "")),
                getattr(r, "affected_pack_ids", ()) or (),
                str(getattr(r, "requested_at", "") or getattr(r, "created_at", ""))))
        _category("action", "Action requests", rows,
                  "No lifecycle action requests are open.")
    except Exception as exc:  # pragma: no cover
        errors.append(SectionError("action_requests", str(exc)))

    pending_count = sum(c.pending for c in categories)
    return ReviewsView(categories=tuple(categories), pending_count=pending_count,
                       errors=tuple(errors))


# ---------------------------------------------------------------------------
# Monitoring (advisory only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringFinding:
    finding_code: str
    severity: str
    is_critical: bool
    diagnostic_note: str
    recommended_human_action: str
    affected_packs: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "finding_code": self.finding_code,
            "severity": self.severity,
            "is_critical": self.is_critical,
            "diagnostic_note": self.diagnostic_note,
            "recommended_human_action": self.recommended_human_action,
            "affected_packs": list(self.affected_packs),
        }


@dataclass(frozen=True)
class MonitoringView:
    has_history: bool
    advisory_notice: str
    recommendation: str
    recommendation_label: str
    confidence: str
    run_id: str
    created_at: str
    critical_findings: Tuple[MonitoringFinding, ...]
    other_findings: Tuple[MonitoringFinding, ...]
    run_count: int
    empty: Optional[EmptyState]
    errors: Tuple[SectionError, ...] = ()

    def to_dict(self) -> dict:
        return {
            "has_history": self.has_history,
            "advisory_notice": self.advisory_notice,
            "recommendation": self.recommendation,
            "recommendation_label": self.recommendation_label,
            "confidence": self.confidence,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "critical_findings": [f.to_dict() for f in self.critical_findings],
            "other_findings": [f.to_dict() for f in self.other_findings],
            "run_count": self.run_count,
            "empty": self.empty.to_dict() if self.empty else None,
            "errors": [e.to_dict() for e in self.errors],
        }


_ADVISORY_NOTICE = (
    "ADVISORY ONLY — monitoring never changes active packs. Any rollback or "
    "deactivation must be reviewed and approved by a human in the Reviews workspace."
)


def _finding_from_dict(data: dict) -> MonitoringFinding:
    severity = str(data.get("severity", "") or "")
    return MonitoringFinding(
        finding_code=str(data.get("finding_code", "")),
        severity=severity,
        is_critical=severity == "critical",
        diagnostic_note=str(data.get("diagnostic_note", "") or ""),
        recommended_human_action=str(data.get("recommended_human_action", "") or ""),
        affected_packs=tuple(str(p) for p in (data.get("candidate_pack_ids") or [])),
    )


def build_monitoring(config: ConsoleConfig) -> MonitoringView:
    """Build the advisory monitoring view from recorded runs (read-only)."""

    runs = [r for r in _read_jsonl(config.monitoring_history_path)
            if r.get("_record") == "active_pack_monitoring_run"]
    if not runs:
        return MonitoringView(
            has_history=False, advisory_notice=_ADVISORY_NOTICE, recommendation="",
            recommendation_label="", confidence="", run_id="", created_at="",
            critical_findings=(), other_findings=(), run_count=0,
            empty=EmptyState("No monitoring history",
                             "No monitoring runs have been recorded for the active packs yet."))

    latest = runs[-1]
    rec = latest.get("recommendation") or {}
    rec_code = str(rec.get("recommendation", "") or "")
    findings = [_finding_from_dict(f) for f in (latest.get("findings") or [])]
    critical = tuple(f for f in findings if f.is_critical)
    other = tuple(f for f in findings if not f.is_critical)

    return MonitoringView(
        has_history=True,
        advisory_notice=_ADVISORY_NOTICE,
        recommendation=rec_code,
        recommendation_label=RECOMMENDATION_LABELS.get(rec_code, rec_code or "Unknown"),
        confidence=str(rec.get("confidence", "") or ""),
        run_id=str(latest.get("monitoring_run_id", "")),
        created_at=str(latest.get("created_at", "")),
        critical_findings=critical,
        other_findings=other,
        run_count=len(runs),
        empty=None,
    )


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryRow:
    memory_id: str
    text: str
    status: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "status": self.status,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class MemoryView:
    rows: Tuple[MemoryRow, ...]
    active_count: int
    total_count: int
    proposal_pending: int
    empty: Optional[EmptyState]
    errors: Tuple[SectionError, ...] = ()

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "active_count": self.active_count,
            "total_count": self.total_count,
            "proposal_pending": self.proposal_pending,
            "empty": self.empty.to_dict() if self.empty else None,
            "errors": [e.to_dict() for e in self.errors],
        }


def build_memory(config: ConsoleConfig, *, ledger_rows: Optional[Sequence[dict]] = None) -> MemoryView:
    """Build the memory-ledger view from exported ledger rows (read-only).

    ``ledger_rows`` is the output of ``WorkbenchService.export_ledger()``. When
    omitted the raw demo ledger file is read directly so the view never needs a
    live service just to display memory.
    """

    errors: List[SectionError] = []
    rows: List[MemoryRow] = []

    raw = list(ledger_rows) if ledger_rows is not None else _read_jsonl(config.ledger_path)
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get("memory_id") or entry.get("id") or "")
        text = str(entry.get("canonical_text") or entry.get("text") or entry.get("claim") or "")
        status = str(entry.get("status") or entry.get("lifecycle") or "")
        if not (mid or text):
            continue
        extra = {k: entry[k] for k in ("operation", "created_at", "updated_at")
                 if k in entry}
        rows.append(MemoryRow(memory_id=mid, text=text, status=status, extra=extra))

    active = sum(1 for r in rows if r.status in ("", "active"))

    proposal_pending = 0
    try:
        proposal_pending = sum(
            1 for r in _mpq.load_memory_review_queue(config.memory_review_path)
            if _enum_value(getattr(r, "review_status", "")) == "pending")
    except Exception as exc:  # pragma: no cover
        errors.append(SectionError("memory_proposals", str(exc)))

    return MemoryView(
        rows=tuple(rows), active_count=active, total_count=len(rows),
        proposal_pending=proposal_pending,
        empty=None if rows else EmptyState("No memories yet",
                                           "The concept-memory ledger is empty in this build."),
        errors=tuple(errors))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportAdapter:
    key: str
    label: str
    supported: bool
    status_label: str
    note: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "supported": self.supported,
            "status_label": self.status_label,
            "note": self.note,
        }


@dataclass(frozen=True)
class ImportsView:
    adapters: Tuple[ImportAdapter, ...]
    lifecycle_steps: Tuple[str, ...]

    @property
    def supported_count(self) -> int:
        return sum(1 for a in self.adapters if a.supported)

    def to_dict(self) -> dict:
        return {
            "adapters": [a.to_dict() for a in self.adapters],
            "lifecycle_steps": list(self.lifecycle_steps),
            "supported_count": self.supported_count,
        }


_IMPORT_ADAPTERS: Tuple[ImportAdapter, ...] = (
    ImportAdapter("pdf", "PDF document", True, "Available",
                  "Assess, preview chunks, then import for evaluation."),
    ImportAdapter("hf_dataset", "Hugging Face dataset", True, "Available",
                  "Assess metadata and licence before any import."),
    ImportAdapter("markdown", "Markdown / notes", False, "Coming soon",
                  "Not available in this build."),
    ImportAdapter("docx", "Word document", False, "Coming soon",
                  "Not available in this build."),
    ImportAdapter("csv", "CSV / tabular", False, "Coming soon",
                  "Not available in this build."),
    ImportAdapter("url", "Web page / URL", False, "Coming soon",
                  "Not available in this build."),
)

_IMPORT_LIFECYCLE: Tuple[str, ...] = (
    "Assess the candidate",
    "Review the findings",
    "Approve the intended use",
    "Sample / preview the content",
    "Import as a knowledge pack",
    "Evaluate the pack",
    "Activate (governed approval)",
)


def build_imports() -> ImportsView:
    """Static, honest capability catalogue for knowledge intake."""

    return ImportsView(adapters=_IMPORT_ADAPTERS, lifecycle_steps=_IMPORT_LIFECYCLE)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettingsView:
    items: Tuple[Tuple[str, str], ...]
    read_only_notice: str

    def to_dict(self) -> dict:
        return {
            "items": [list(i) for i in self.items],
            "read_only_notice": self.read_only_notice,
        }


_READ_ONLY_NOTICE = (
    "This workspace is read-only. It never writes memory, changes sources, "
    "activates or rolls back packs, or executes lifecycle actions. Governed "
    "changes happen only through the approved review and execution paths."
)


def build_settings(config: ConsoleConfig) -> SettingsView:
    """Surface environment, backend and data locations (read-only)."""

    def _present(path: Path) -> str:
        return "present" if path.exists() else "not present in this build"

    items = (
        ("Product", f"{PRODUCT_NAME} ({CONSOLE_UI_VERSION})"),
        ("Environment", config.environment),
        ("Retrieval backend", config.backend),
        ("Memory ledger", f"{config.ledger_path.name} ({_present(config.ledger_path)})"),
        ("Source registry", f"{config.registry_path.name} ({_present(config.registry_path)})"),
        ("Active packs state", f"{config.pack_state_path.name} ({_present(config.pack_state_path)})"),
        ("Monitoring history", f"{config.monitoring_history_path.name} "
                               f"({_present(config.monitoring_history_path)})"),
    )
    return SettingsView(items=items, read_only_notice=_READ_ONLY_NOTICE)
