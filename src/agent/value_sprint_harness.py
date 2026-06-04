"""v2.3 Value Sprint Harness — practical-value measurement (read-only).

This module is the v2.3 evolution of the v2.1.1 Value Sprint
(:mod:`agent.value_sprint`). It exists for one reason: to **measure** whether
the frozen assistant produces practical value on realistic Brad / M365 / coding
workflows, instead of adding more hidden architecture.

It runs a fixed query set through the **frozen** assistant path
(:meth:`WorkbenchService.build_grounding_package` for the upstream decision plus
:meth:`WorkbenchService.answer_query` for the composed answer and guard verdict)
and records one audit row per query: how it routed, the composer ``mode`` that
was decided upstream, whether it grounded and cited, whether it was honestly
refused, whether a stale source was flagged, whether a near-miss / superseded
conflict surfaced, whether it fell back to a labelled model prior, which
retrieval backend served it, and the independent AnswerGuard verdict.

It changes **no** grounding, citation, verifier, lifecycle, refusal,
pack-isolation, SLM, AnswerGuard, or retrieval-safety semantics. It only
observes the existing path and classifies the outcome against the operator's
prior expectation. Where the pack could not deliver expected value (a refusal or
weak answer where value was expected), it emits a **pack-gap proposal** — a
suggestion only. Nothing here writes to the MemoryLedger or KnowledgeLibrary;
the only outputs are reports and proposals.

Honest boundaries (see the README for the full version):

* The deterministic backend is over-permissive — citation *presence* is not a
  relevance guarantee. ``grounded_cited`` means evidence was retrieved, not that
  the answer is correct.
* Usefulness still requires human judgement: the operator labels are captured,
  never inferred.
* Memory-routed paths (decision recall, conflict, model prior) only fire when
  project memories exist. The offline demo pack ships knowledge only, so the
  caller seeds memories to exercise those paths (the CLI seeds a small demo set;
  tests seed their own scenario).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from slm.assistant_composer import (
    JUDGEMENT_LABEL,
    SECTION_FACTUAL,
    SECTION_JUDGEMENT,
    AssistantComposer,
    ComposerMode,
)

from .memory_proposals import MemoryProposal, ProposalKind, ProposalStatus
from .workbench_service import WorkbenchService

# -- expected-outcome taxonomy (operator's prior expectation per query) -------

EXPECTED_GROUNDED_USEFUL = "grounded_useful"
EXPECTED_STALE_FLAGGED = "stale_flagged"
EXPECTED_HONEST_REFUSAL = "honest_refusal"
EXPECTED_CONFLICT = "conflict"
EXPECTED_MODEL_PRIOR = "model_prior"
EXPECTED_PACK_GAP = "pack_gap"

# Expectations under which the operator wanted grounded value from the pack.
_VALUE_EXPECTED = frozenset({EXPECTED_GROUNDED_USEFUL, EXPECTED_STALE_FLAGGED})

# Demo project memories the CLI seeds before running the canonical query file, so
# the memory-routed paths (decision recall, repo-milestone recall, near-miss
# conflict) actually fire. These are seeded into the *temporary* sprint service
# only — never the tracked ledger. Tests seed the same set to stay consistent.
# They are not "writes" in the lifecycle sense: nothing is approved or persisted
# to a tracked store; they exist for the duration of one in-memory sprint.
DEMO_SEED_MEMORIES = (
    "the supplier delivery is on Friday afternoon",
    "the team chose Azure Container Apps for the staging deployment",
    "the v2.2 milestone shipped the hybrid retrieval backend",
)

# -- observed outcome labels (derived from the upstream composer mode) --------

OUTCOME_GROUNDED_CITED = "grounded_cited"
OUTCOME_GROUNDED_NO_CITES = "grounded_no_cites"
OUTCOME_HONEST_REFUSAL = "honest_refusal"
OUTCOME_CONFLICT_EXPLAINED = "conflict_explained"
OUTCOME_MODEL_PRIOR_LABELLED = "model_prior_labelled"
OUTCOME_PACK_SUMMARY = "pack_summary"


@dataclass(frozen=True)
class OperatorScore:
    """Manual operator labels for one answer — captured, never inferred.

    ``useful`` / ``saved_time`` / ``reusable_output`` are tri-state: ``None``
    means "not yet scored". They are loaded from the query spec (so an operator
    can pre-fill them, or hand-edit the emitted JSONL and re-report) and are
    never derived from the audit.
    """

    useful: Optional[bool] = None
    saved_time: Optional[bool] = None
    reusable_output: Optional[bool] = None
    trust_level: Optional[str] = None  # "high" | "medium" | "low"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SprintQuery:
    """One sprint query plus the operator's prior expectation for it."""

    query: str
    note: str = ""
    category: str = ""
    expected_outcome: str = EXPECTED_GROUNDED_USEFUL
    allow_model_prior: bool = False
    operator: OperatorScore = field(default_factory=OperatorScore)


@dataclass(frozen=True)
class PackGapProposal:
    """A proposed pack update for a query the pack could not usefully answer.

    This is a **proposal only**. It is never written to the MemoryLedger or the
    KnowledgeLibrary; it is surfaced in the report so an operator can decide
    whether to turn it into an approved knowledge import or memory proposal.
    """

    missing_topic: str
    suggested_source_type: str   # "knowledge_source" | "memory_proposal"
    suggested_memory: str
    affected_query: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SprintRow:
    """The recorded audit for one query: expectation vs observed behaviour."""

    query: str
    note: str
    category: str
    expected_outcome: str
    actual_mode: str
    refused: bool
    citations_count: int
    stale_warning: bool
    conflict_warning: bool
    model_prior_used: bool
    retrieval_backend: str
    guard_verdict: str
    outcome_label: str
    pack_gap_detected: bool
    suggested_pack_update: Optional[dict] = None
    operator: dict = field(default_factory=lambda: OperatorScore().to_dict())
    answer_snippet: str = ""
    # v2.4 relevance & sufficiency gate verdict for this query (the gate decides
    # answerability between retrieval and grounding). Empty when no evidence was
    # retrieved (the gate only runs on the would-be-grounded path).
    relevance_verdict: str = ""
    relevance_label: str = ""
    # The single most-relevant candidate the ranker did NOT let lead, and why —
    # the most useful audit line for *why* the gate did what it did. Empty when
    # only one (or no) candidate was retrieved.
    rejected_candidate: str = ""
    rejected_reason: str = ""
    # v2.6 extractive-composer measurement. ``synthesis_breadth`` is the number
    # of distinct sources a grounded answer draws on (composer-independent: a
    # property of the pack and query, 0 when not grounded). ``span_count`` is
    # the number of citation-bound spans the composer emitted (0 for the
    # whole-chunk template composer; >=1 per source for the extractive one).
    synthesis_breadth: int = 0
    span_count: int = 0
    # v2.7 consultant-report measurement. All 0 unless the report composer ran.
    # ``report_factual_section_count`` / ``report_judgement_block_count`` are
    # the section partition; ``report_cited_claim_count`` is the number of
    # citation-bound spans across factual sections; ``report_unlabelled_
    # judgement_count`` is an integrity counter that must stay 0.
    report_factual_section_count: int = 0
    report_cited_claim_count: int = 0
    report_judgement_block_count: int = 0
    report_unlabelled_judgement_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SprintSummary:
    """Sprint-level tallies highlighting where value did and did not appear."""

    query_count: int
    grounded_count: int
    refused_count: int
    honest_refusal_count: int
    conflict_count: int
    model_prior_count: int
    stale_flagged_count: int
    pack_gap_count: int
    guard_reject_count: int
    expectation_met_count: int
    # v2.6: grounded answers that draw on >=2 distinct sources — the queries an
    # extractive multi-chunk composer can synthesise rather than echo.
    multi_source_grounded_count: int = 0
    # v2.7 integrity counter: judgement blocks emitted without the mandatory
    # label, summed across rows. Must be 0 — the guard rejects any such answer.
    report_unlabelled_judgement_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# -- loading ------------------------------------------------------------------

def load_queries(path: str | Path) -> List[SprintQuery]:
    """Load sprint queries from a JSONL file (``#`` lines are comments).

    The schema is a superset of the v2.1.1 format: the v2.1.1 keys are ignored
    here, and ``expected_outcome`` / ``category`` / ``allow_model_prior`` /
    ``operator`` drive the v2.3 harness. Unknown keys are tolerated so a single
    canonical query file serves both harnesses.
    """
    queries: List[SprintQuery] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        data = json.loads(line)
        op = data.get("operator") or {}
        queries.append(SprintQuery(
            query=data["query"],
            note=data.get("note", ""),
            category=data.get("category", ""),
            expected_outcome=data.get("expected_outcome",
                                      EXPECTED_GROUNDED_USEFUL),
            allow_model_prior=bool(data.get("allow_model_prior", False)),
            operator=OperatorScore(
                useful=op.get("useful"),
                saved_time=op.get("saved_time"),
                reusable_output=op.get("reusable_output"),
                trust_level=op.get("trust_level"),
                notes=op.get("notes", ""),
            ),
        ))
    return queries


# -- per-query execution ------------------------------------------------------

def _has_stale_warning(cautions: List[str]) -> bool:
    return any("stale" in c.lower() for c in cautions)


def _outcome_label(mode: ComposerMode, citations: int) -> str:
    if mode == ComposerMode.GROUNDED:
        return OUTCOME_GROUNDED_CITED if citations > 0 else OUTCOME_GROUNDED_NO_CITES
    if mode == ComposerMode.CONFLICT_EXPLANATION:
        return OUTCOME_CONFLICT_EXPLAINED
    if mode == ComposerMode.MODEL_PRIOR_LABELLED:
        return OUTCOME_MODEL_PRIOR_LABELLED
    if mode == ComposerMode.PACK_SUMMARY:
        return OUTCOME_PACK_SUMMARY
    return OUTCOME_HONEST_REFUSAL


def _expectation_met(spec: SprintQuery, mode: ComposerMode,
                     stale: bool, conflict: bool, grounded: bool) -> bool:
    """Whether the observed outcome matched the operator's prior expectation."""
    exp = spec.expected_outcome
    if exp == EXPECTED_GROUNDED_USEFUL:
        return grounded
    if exp == EXPECTED_STALE_FLAGGED:
        return grounded and stale
    if exp == EXPECTED_HONEST_REFUSAL:
        return mode == ComposerMode.REFUSAL
    if exp == EXPECTED_CONFLICT:
        return conflict
    if exp == EXPECTED_MODEL_PRIOR:
        return mode == ComposerMode.MODEL_PRIOR_LABELLED
    if exp == EXPECTED_PACK_GAP:
        return not grounded
    return False


def extract_pack_gap(spec: SprintQuery, *, grounded: bool) -> Optional[PackGapProposal]:
    """Propose a pack update when the pack could not deliver expected value.

    A proposal is emitted when value was expected but not grounded, or when the
    query was an explicit pack-gap probe. The proposal is advisory: it suggests a
    source type and a candidate memory but never writes anything.
    """
    value_expected = spec.expected_outcome in _VALUE_EXPECTED
    is_gap_probe = spec.expected_outcome == EXPECTED_PACK_GAP
    if not ((value_expected and not grounded) or is_gap_probe):
        return None
    topic = spec.note or spec.category or spec.query[:80]
    # Knowledge-shaped categories want a source import; decision/recall-shaped
    # categories want a memory proposal. Default to a knowledge source.
    memory_categories = {"project_decision", "repo_milestone", "decision_recall"}
    if spec.category in memory_categories:
        source_type = "memory_proposal"
        suggested = (f"Capture the decision behind '{topic}' as an approved "
                     "project memory so future recall can ground it.")
    else:
        source_type = "knowledge_source"
        suggested = (f"Import an authoritative source covering '{topic}' so the "
                     "pack can ground this query instead of refusing it.")
    return PackGapProposal(
        missing_topic=topic,
        suggested_source_type=source_type,
        suggested_memory=suggested,
        affected_query=spec.query,
    )


# -- v2.5B: opt-in memory capture from missing-decision gaps ------------------

def pack_gap_to_memory_proposal(
        gap: PackGapProposal) -> Optional[MemoryProposal]:
    """Convert a missing-decision pack gap into a *PENDING* memory proposal.

    Proposal-only and opt-in. This mints a reviewable candidate memory from a
    value-sprint gap whose ``suggested_source_type`` is ``"memory_proposal"``
    (a missing project decision, not missing knowledge). Knowledge-source gaps
    return ``None`` and stay report-only.

    The returned proposal is ``PENDING`` and ``written=False``: it is **not** a
    fact. It becomes a memory only if a human approves it and the workbench
    writes it through the frozen ``add_memory`` path. This function never
    approves, never writes, and never touches the MemoryLedger or the bank.
    """
    if gap.suggested_source_type != "memory_proposal":
        return None
    digest = hashlib.sha1(
        f"packgap|{gap.affected_query}".encode("utf-8")).hexdigest()[:10]
    return MemoryProposal(
        proposal_id=f"packgap-{digest}",
        canonical_text=gap.suggested_memory,
        source_file="(value-sprint pack gap)",
        kind=ProposalKind.DECISION,
        confidence=0.3,
        reason=("value-sprint missing-decision gap; a candidate only — requires "
                "human approval before it can be written as a memory"),
        status=ProposalStatus.PENDING,
    )


def emit_memory_proposals(rows: List["SprintRow"], *,
                          queue_path: str | Path) -> List[MemoryProposal]:
    """Opt-in: route missing-decision pack gaps into a sprint-scoped queue.

    For every row carrying a ``memory_proposal``-type pack gap, mint a PENDING
    :class:`MemoryProposal` and add it to a :class:`ProposalQueue` at
    ``queue_path`` — a **sprint-scoped** file, deliberately separate from the
    pack's real proposal queue so auto-generated suggestions never mix with the
    human-authored queue until an operator promotes them.

    Knowledge-source gaps are skipped. Nothing is approved or written; the queue
    only accumulates PENDING candidates for human review. The queue's id-based
    deduplication makes re-running idempotent. Returns the proposals newly added.
    """
    from .proposal_queue import ProposalQueue

    queue = ProposalQueue(queue_path, load_existing=True)
    added: List[MemoryProposal] = []
    for row in rows:
        gap_dict = row.suggested_pack_update
        if not gap_dict:
            continue
        proposal = pack_gap_to_memory_proposal(PackGapProposal(**gap_dict))
        if proposal is None:
            continue
        if queue.add(proposal):
            added.append(proposal)
    return added


def _report_metrics(report) -> tuple[int, int, int, int]:
    """Derive consultant-report metrics from an answer's report carrier.

    Returns ``(factual_section_count, cited_claim_count, judgement_block_count,
    unlabelled_judgement_count)``. All zeros when ``report`` is None (every
    non-report answer), so the metrics are inert outside report mode.
    """
    if report is None:
        return (0, 0, 0, 0)
    factual = [s for s in report.sections if s.kind == SECTION_FACTUAL]
    judgement = [s for s in report.sections if s.kind == SECTION_JUDGEMENT]
    cited_claims = sum(len(s.spans) for s in factual)
    unlabelled = sum(
        1 for s in judgement
        if not (s.judgement_text or "").startswith(JUDGEMENT_LABEL))
    return (len(factual), cited_claims, len(judgement), unlabelled)


def run_query(service: WorkbenchService, spec: SprintQuery, *,
              retrieval_backend: str,
              composer: Optional[AssistantComposer] = None) -> SprintRow:
    """Run one query through the frozen assistant path and record the audit.

    Two frozen, deterministic calls are made: ``build_grounding_package`` to read
    the upstream composer ``mode`` (the authoritative signal for refusal /
    conflict / model-prior, which the audit flags alone cannot distinguish), and
    ``answer_query`` to obtain the composed answer, cited ids, and the
    independent AnswerGuard verdict. Both use identical arguments, so they route
    consistently. ``composer`` selects how the (already-decided) grounded
    evidence is rendered; it can never change the grounding decision.
    """
    package = service.build_grounding_package(
        spec.query, allow_model_prior=spec.allow_model_prior)
    result = service.answer_query(
        spec.query, allow_model_prior=spec.allow_model_prior,
        composer=composer)

    mode = package.mode
    audit = result.audit or {}
    cautions = list(audit.get("cautions") or [])
    citations = len(result.evidence_ids)
    refused = bool(result.refused)
    grounded = (mode == ComposerMode.GROUNDED) and citations > 0
    stale = _has_stale_warning(cautions)
    conflict = (mode == ComposerMode.CONFLICT_EXPLANATION
                or package.historical_note is not None)
    # Model-prior is keyed off the *mode*, never the audit flag (that flag is
    # also true on plain refusals — the AnswerGuard lesson from v2.1.1).
    model_prior = mode == ComposerMode.MODEL_PRIOR_LABELLED
    guard_verdict = str((audit.get("guard") or {}).get("verdict", ""))
    relevance = audit.get("relevance") or {}
    relevance_verdict = str(relevance.get("verdict", ""))
    relevance_label = str(relevance.get("label", ""))
    rejected = relevance.get("top_rejected") or {}
    rejected_candidate = str(rejected.get("citation_id", ""))
    rejected_reason = str(rejected.get("reason", ""))

    gap = extract_pack_gap(spec, grounded=grounded)
    snippet = (result.answer.text or "").strip().replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:157] + "..."
    synthesis_breadth = citations if grounded else 0
    span_count = len(getattr(result.answer, "spans", None) or [])
    (report_factual_section_count, report_cited_claim_count,
     report_judgement_block_count, report_unlabelled_judgement_count) = (
        _report_metrics(getattr(result.answer, "report", None)))

    return SprintRow(
        query=spec.query,
        note=spec.note,
        category=spec.category,
        expected_outcome=spec.expected_outcome,
        actual_mode=mode.value,
        refused=refused,
        citations_count=citations,
        stale_warning=stale,
        conflict_warning=conflict,
        model_prior_used=model_prior,
        retrieval_backend=retrieval_backend,
        guard_verdict=guard_verdict,
        outcome_label=_outcome_label(mode, citations),
        pack_gap_detected=gap is not None,
        suggested_pack_update=gap.to_dict() if gap else None,
        operator=spec.operator.to_dict(),
        answer_snippet=snippet,
        relevance_verdict=relevance_verdict,
        relevance_label=relevance_label,
        rejected_candidate=rejected_candidate,
        rejected_reason=rejected_reason,
        synthesis_breadth=synthesis_breadth,
        span_count=span_count,
        report_factual_section_count=report_factual_section_count,
        report_cited_claim_count=report_cited_claim_count,
        report_judgement_block_count=report_judgement_block_count,
        report_unlabelled_judgement_count=report_unlabelled_judgement_count,
    )


def run_sprint(service: WorkbenchService, queries: List[SprintQuery], *,
               retrieval_backend: str = "deterministic",
               composer: Optional[AssistantComposer] = None) -> List[SprintRow]:
    """Run every sprint query against the active pack's service."""
    return [run_query(service, q, retrieval_backend=retrieval_backend,
                      composer=composer)
            for q in queries]


def summarize(rows: List[SprintRow]) -> SprintSummary:
    """Tally outcomes across the sprint rows."""
    def grounded(r: SprintRow) -> bool:
        return r.outcome_label == OUTCOME_GROUNDED_CITED
    return SprintSummary(
        query_count=len(rows),
        grounded_count=sum(1 for r in rows if grounded(r)),
        refused_count=sum(1 for r in rows if r.refused),
        honest_refusal_count=sum(
            1 for r in rows if r.outcome_label == OUTCOME_HONEST_REFUSAL),
        conflict_count=sum(1 for r in rows if r.conflict_warning),
        model_prior_count=sum(1 for r in rows if r.model_prior_used),
        stale_flagged_count=sum(1 for r in rows if r.stale_warning),
        pack_gap_count=sum(1 for r in rows if r.pack_gap_detected),
        guard_reject_count=sum(1 for r in rows if r.guard_verdict == "REJECT"),
        expectation_met_count=sum(
            1 for r in rows
            if _row_expectation_met(r)),
        multi_source_grounded_count=sum(
            1 for r in rows if grounded(r) and r.synthesis_breadth >= 2),
        report_unlabelled_judgement_count=sum(
            r.report_unlabelled_judgement_count for r in rows),
    )


def _row_expectation_met(r: SprintRow) -> bool:
    """Re-derive expectation-met from a row (so re-rendered reports agree)."""
    exp = r.expected_outcome
    grounded = r.outcome_label == OUTCOME_GROUNDED_CITED
    if exp == EXPECTED_GROUNDED_USEFUL:
        return grounded
    if exp == EXPECTED_STALE_FLAGGED:
        return grounded and r.stale_warning
    if exp == EXPECTED_HONEST_REFUSAL:
        return r.outcome_label == OUTCOME_HONEST_REFUSAL
    if exp == EXPECTED_CONFLICT:
        return r.conflict_warning
    if exp == EXPECTED_MODEL_PRIOR:
        return r.model_prior_used
    if exp == EXPECTED_PACK_GAP:
        return not grounded
    return False


# -- reporting ----------------------------------------------------------------

def _check(flag: bool) -> str:
    return "yes" if flag else "no"


def _tri(value: Optional[bool]) -> str:
    """Render a tri-state operator label: yes / no / unscored."""
    if value is None:
        return "-"
    return "yes" if value else "no"


def render_markdown(rows: List[SprintRow], summary: SprintSummary, *,
                    pack_label: str = "(active pack)",
                    backend_label: str = "deterministic") -> str:
    """Render a human-readable Markdown report of the sprint."""
    lines: List[str] = []
    lines.append("# v2.3 Value Sprint report")
    lines.append("")
    lines.append(f"- pack            : {pack_label}")
    lines.append(f"- retrieval       : {backend_label}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- queries            : {summary.query_count}")
    lines.append(f"- expectation met    : {summary.expectation_met_count}"
                 f"/{summary.query_count}")
    lines.append(f"- grounded + cited   : {summary.grounded_count} "
                 "(evidence retrieved; NOT a relevance guarantee)")
    lines.append(f"- honest refusals    : {summary.honest_refusal_count}")
    lines.append(f"- conflicts surfaced : {summary.conflict_count}")
    lines.append(f"- model-prior used   : {summary.model_prior_count}")
    lines.append(f"- stale-flagged      : {summary.stale_flagged_count}")
    lines.append(f"- pack gaps          : {summary.pack_gap_count} "
                 "(proposals only — never written)")
    lines.append(f"- guard rejects      : {summary.guard_reject_count}")
    lines.append("")
    lines.append("## Honest finding")
    lines.append("")
    lines.append(
        "`grounded + cited` means evidence was *retrieved*, not that the answer "
        "is correct. The trust behaviours that genuinely work today are "
        "refusal-on-no-evidence, stale-source flagging, near-miss/superseded "
        "conflict surfacing, labelled model-prior fallback, and the independent "
        "AnswerGuard verdict. Usefulness is the operator's call — the operator "
        "columns below are captured, never inferred.")
    lines.append("")
    lines.append(
        "Caveat (resolved in v2.4): with the over-permissive *deterministic* "
        "knowledge backend, a declarative near-miss query routes to both memory "
        "and knowledge, and knowledge used to ground it — masking conflict "
        "surfacing in integration. The v2.4 relevance & sufficiency gate now "
        "surfaces a rejected near-miss as a conflict even when a knowledge "
        "chunk was retrieved on the same query, and downgrades weak or "
        "out-of-domain evidence to a refusal instead of a confident grounding. "
        "The `relevance` column below shows the gate verdict per query.")
    lines.append("")
    lines.append("## Per-query audit")
    lines.append("")
    header = ("| # | category | expected | mode | relevance | cites | stale | "
              "conflict | prior | guard | met | gap |")
    sep = ("|---|----------|----------|------|-----------|-------|-------|"
           "----------|-------|-------|-----|-----|")
    lines.append(header)
    lines.append(sep)
    for idx, r in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {r.category or '-'} | {r.expected_outcome} | "
            f"{r.actual_mode} | {r.relevance_verdict or '-'} | "
            f"{r.citations_count} | {_check(r.stale_warning)} | "
            f"{_check(r.conflict_warning)} | {_check(r.model_prior_used)} | "
            f"{r.guard_verdict or '-'} | {_check(_row_expectation_met(r))} | "
            f"{_check(r.pack_gap_detected)} |")
    lines.append("")
    lines.append("## Operator scoring")
    lines.append("")
    lines.append("Machine columns are captured automatically; fill these in by "
                 "hand (edit the JSONL, then `value-sprint report`).")
    lines.append("")
    lines.append("| # | category | useful | saved_time | reusable | trust | notes |")
    lines.append("|---|----------|--------|------------|----------|-------|-------|")
    for idx, r in enumerate(rows, start=1):
        op = r.operator or {}
        lines.append(
            f"| {idx} | {r.category or '-'} | {_tri(op.get('useful'))} | "
            f"{_tri(op.get('saved_time'))} | {_tri(op.get('reusable_output'))} | "
            f"{op.get('trust_level') or '-'} | {op.get('notes') or '-'} |")
    rejected = [r for r in rows if r.rejected_candidate]
    if rejected:
        lines.append("")
        lines.append("## Top rejected evidence (why the gate held it back)")
        lines.append("")
        lines.append("The strongest candidate the ranker did *not* let lead, per "
                     "query — the audit line for why a verdict was reached.")
        lines.append("")
        for idx, r in enumerate(rows, start=1):
            if not r.rejected_candidate:
                continue
            lines.append(f"- **#{idx}** `{r.rejected_candidate}` — "
                         f"{r.rejected_reason or 'outranked by lead evidence'}")
    gaps = [r for r in rows if r.suggested_pack_update]
    if gaps:
        lines.append("")
        lines.append("## Pack-gap proposals (advisory — not written)")
        lines.append("")
        for r in gaps:
            gap = r.suggested_pack_update or {}
            lines.append(f"- **{gap.get('missing_topic', '?')}** "
                         f"({gap.get('suggested_source_type', '?')}): "
                         f"{gap.get('suggested_memory', '')}")
    lines.append("")
    return "\n".join(lines)


def write_reports(rows: List[SprintRow], summary: SprintSummary, *,
                  md_path: str | Path, jsonl_path: str | Path,
                  pack_label: str = "(active pack)",
                  backend_label: str = "deterministic") -> None:
    """Write the Markdown and JSONL reports, creating parent dirs as needed."""
    md = Path(md_path)
    jsonl = Path(jsonl_path)
    md.parent.mkdir(parents=True, exist_ok=True)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        render_markdown(rows, summary, pack_label=pack_label,
                        backend_label=backend_label),
        encoding="utf-8")
    out_lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in rows]
    jsonl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def load_rows(path: str | Path) -> List[SprintRow]:
    """Reload sprint rows from a previously written JSONL report.

    Used by ``value-sprint report`` so an operator can hand-edit the operator
    labels in the JSONL and regenerate the Markdown without re-running queries.
    """
    rows: List[SprintRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        rows.append(SprintRow(
            query=data["query"],
            note=data.get("note", ""),
            category=data.get("category", ""),
            expected_outcome=data.get("expected_outcome", ""),
            actual_mode=data.get("actual_mode", ""),
            refused=bool(data.get("refused", False)),
            citations_count=int(data.get("citations_count", 0)),
            stale_warning=bool(data.get("stale_warning", False)),
            conflict_warning=bool(data.get("conflict_warning", False)),
            model_prior_used=bool(data.get("model_prior_used", False)),
            retrieval_backend=data.get("retrieval_backend", ""),
            guard_verdict=data.get("guard_verdict", ""),
            outcome_label=data.get("outcome_label", ""),
            pack_gap_detected=bool(data.get("pack_gap_detected", False)),
            suggested_pack_update=data.get("suggested_pack_update"),
            operator=data.get("operator") or OperatorScore().to_dict(),
            answer_snippet=data.get("answer_snippet", ""),
            relevance_verdict=data.get("relevance_verdict", ""),
            relevance_label=data.get("relevance_label", ""),
            rejected_candidate=data.get("rejected_candidate", ""),
            rejected_reason=data.get("rejected_reason", ""),
        ))
    return rows
