"""v2.1.1 Value Sprint — practical-value instrumentation (read-only).

This module measures whether the existing pack produces *practical value* on
real workflows. It runs a fixed set of realistic queries through the **frozen**
assistant path (``WorkbenchService.answer_query`` with the default template
composer) and records an audit row per query: how it routed, whether it
grounded and cited, whether it was honestly refused, whether a stale source was
flagged, and whether it fell back to a model prior.

It is pure instrumentation. It adds no query path, changes no grounding,
citation, verifier, lifecycle, refusal, or pack-isolation semantics, and needs
no SLM. It only *observes* the existing path and tallies the result against the
operator's prior expectation for each query (useful / reusable / missing
evidence), so a "pack gap" — a query the operator expected value from but the
pack could not answer — is surfaced explicitly rather than hidden.

The deterministic, offline retrieval backend is **over-permissive**: it returns
a broad fixed set of citations for almost any query — verbatim, paraphrased, or
entirely out-of-domain — so citation *presence* is not a relevance signal. The
sprint exposes this honestly rather than papering over it: a query that should
have no evidence (e.g. out-of-domain) can still ground irrelevant chunks (a
``false_grounding``), while the grounding guard still prevents fabricated or
forbidden content from leaking. The trust behaviours that genuinely work today
are refusal-on-no-evidence (empty memory) and stale-source flagging; relevance-
ranked retrieval — actually answering the question asked — would need a semantic
backend.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from .workbench_service import WorkbenchService


@dataclass(frozen=True)
class ValueSprintQuery:
    """One sprint query plus the operator's prior expectation for it."""

    query: str
    note: str = ""
    useful_expected: bool = False
    reusable_output_expected: bool = False
    missing_evidence_expected: bool = False


@dataclass(frozen=True)
class ValueSprintRow:
    """The recorded audit for one query: expectation vs observed behaviour."""

    query: str
    note: str
    route: str
    refused: bool
    citations_count: int
    stale_warning: bool
    model_prior_used: bool
    useful_expected: bool
    reusable_output_expected: bool
    missing_evidence_expected: bool
    # Observed outcome, for honest expected-vs-actual comparison.
    missing_evidence_observed: bool
    grounded: bool
    pack_gap: bool
    false_grounding: bool
    composer_backend: str
    cautions: List[str] = field(default_factory=list)
    answer_snippet: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ValueSprintSummary:
    """Sprint-level tallies highlighting where value did and did not appear."""

    query_count: int
    grounded_count: int
    refused_count: int
    honest_refusal_count: int
    pack_gap_count: int
    false_grounding_count: int
    stale_flagged_count: int
    model_prior_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_queries(path: str | Path) -> List[ValueSprintQuery]:
    """Load sprint queries from a JSONL file (``#`` lines are comments)."""
    queries: List[ValueSprintQuery] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        data = json.loads(line)
        queries.append(ValueSprintQuery(
            query=data["query"],
            note=data.get("note", ""),
            useful_expected=bool(data.get("useful_expected", False)),
            reusable_output_expected=bool(
                data.get("reusable_output_expected", False)),
            missing_evidence_expected=bool(
                data.get("missing_evidence_expected", False)),
        ))
    return queries


def _has_stale_warning(cautions: List[str]) -> bool:
    return any("stale" in c.lower() for c in cautions)


def run_query(service: WorkbenchService,
              spec: ValueSprintQuery) -> ValueSprintRow:
    """Run one query through the frozen assistant path and record the audit."""
    result = service.answer_query(spec.query)  # template composer, offline
    audit = result.audit or {}
    cautions = list(audit.get("cautions") or [])
    citations = len(result.evidence_ids)
    refused = bool(result.refused)
    grounded = (not refused) and citations > 0
    # A pack gap: the operator expected a useful, grounded answer, but the pack
    # could not deliver one (refused, or grounded nothing). Honest refusals —
    # where missing evidence was the expected, correct outcome — are not gaps.
    pack_gap = (
        spec.useful_expected
        and not spec.missing_evidence_expected
        and not grounded
    )
    # A false grounding: a refusal was the expected, correct outcome (no
    # relevant evidence should exist), yet the over-permissive backend still
    # grounded and cited irrelevant chunks.
    false_grounding = spec.missing_evidence_expected and grounded
    snippet = (result.answer.text or "").strip().replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:157] + "..."
    return ValueSprintRow(
        query=spec.query,
        note=spec.note,
        route=str(audit.get("route", "")),
        refused=refused,
        citations_count=citations,
        stale_warning=_has_stale_warning(cautions),
        model_prior_used=bool(audit.get("model_prior_used", False)),
        useful_expected=spec.useful_expected,
        reusable_output_expected=spec.reusable_output_expected,
        missing_evidence_expected=spec.missing_evidence_expected,
        missing_evidence_observed=refused,
        grounded=grounded,
        pack_gap=pack_gap,
        false_grounding=false_grounding,
        composer_backend=result.composer_backend,
        cautions=cautions,
        answer_snippet=snippet,
    )


def run_value_sprint(service: WorkbenchService,
                     queries: List[ValueSprintQuery]) -> List[ValueSprintRow]:
    """Run every sprint query against the active pack's service."""
    return [run_query(service, q) for q in queries]


def summarize(rows: List[ValueSprintRow]) -> ValueSprintSummary:
    """Tally grounded answers, refusals, honest refusals, gaps, and warnings."""
    return ValueSprintSummary(
        query_count=len(rows),
        grounded_count=sum(1 for r in rows if r.grounded),
        refused_count=sum(1 for r in rows if r.refused),
        honest_refusal_count=sum(
            1 for r in rows if r.refused and r.missing_evidence_expected),
        pack_gap_count=sum(1 for r in rows if r.pack_gap),
        false_grounding_count=sum(1 for r in rows if r.false_grounding),
        stale_flagged_count=sum(1 for r in rows if r.stale_warning),
        model_prior_count=sum(1 for r in rows if r.model_prior_used),
    )


def _check(flag: bool) -> str:
    return "yes" if flag else "no"


def render_markdown(rows: List[ValueSprintRow],
                    summary: ValueSprintSummary, *,
                    pack_label: str = "(active pack)") -> str:
    """Render a human-readable Markdown report of the sprint."""
    lines: List[str] = []
    lines.append("# v2.1.1 Value Sprint report")
    lines.append("")
    lines.append(f"Pack: {pack_label}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- queries           : {summary.query_count}")
    lines.append(f"- grounded + cited  : {summary.grounded_count} "
                 "(retrieved evidence; NOT a relevance guarantee — see finding)")
    lines.append(f"- refused           : {summary.refused_count}")
    lines.append(f"- honest refusals   : {summary.honest_refusal_count} "
                 "(refusal was the expected, correct outcome)")
    lines.append(f"- false groundings  : {summary.false_grounding_count} "
                 "(should have had no evidence, but grounded anyway)")
    lines.append(f"- pack gaps         : {summary.pack_gap_count} "
                 "(expected value, pack could not deliver)")
    lines.append(f"- stale-flagged     : {summary.stale_flagged_count}")
    lines.append(f"- model-prior used  : {summary.model_prior_count}")
    lines.append("")
    lines.append("## Honest finding")
    lines.append("")
    lines.append(
        "The deterministic retrieval backend is **over-permissive**: it returns "
        "a broad fixed set of citations for almost any query — verbatim, "
        "paraphrased, or out-of-domain. So `grounded + cited` here means "
        "*evidence was retrieved*, **not** *the answer is relevant*. The "
        "trust behaviours that genuinely work today are refusal-on-no-evidence "
        "(empty memory) and stale-source flagging, plus the grounding guard "
        "that stops forbidden/fabricated content leaking even when an "
        "irrelevant source is cited. Relevance-ranked retrieval — actually "
        "answering the question asked — would need a semantic backend.")
    lines.append("")
    lines.append("## Per-query audit")
    lines.append("")
    lines.append("| # | note | route | grounded | cites | refused | stale | "
                 "false-ground | gap |")
    lines.append("|---|------|-------|----------|-------|---------|-------|"
                 "--------------|-----|")
    for idx, r in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {r.note or '-'} | {r.route or '-'} | "
            f"{_check(r.grounded)} | {r.citations_count} | "
            f"{_check(r.refused)} | {_check(r.stale_warning)} | "
            f"{_check(r.false_grounding)} | {_check(r.pack_gap)} |")
    false_grounds = [r for r in rows if r.false_grounding]
    if false_grounds:
        lines.append("")
        lines.append("## False groundings — over-permissive retrieval")
        lines.append("")
        for r in false_grounds:
            lines.append(f"- {r.note or r.query}")
    gaps = [r for r in rows if r.pack_gap]
    if gaps:
        lines.append("")
        lines.append("## Pack gaps — expected value the pack could not deliver")
        lines.append("")
        for r in gaps:
            lines.append(f"- {r.note or r.query}")
    lines.append("")
    return "\n".join(lines)
