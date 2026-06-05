"""v3.0 Retrieval Evaluation Harness — retrieval-quality measurement (read-only).

This module measures how good the pack's **retrieval** is *before* any change to
retrieval, ranking, or source-selection logic. It exists to expose a baseline,
not to improve it: if retrieval performs badly on a case, that is a finding, not
a failure of the harness.

The motivating defect is *relevance bleed* — for example an off-topic Copilot
Studio connector caution surfacing in a least-privilege report. That is not a
rendering problem; it is a retrieval / source-selection quality problem, and the
right place to measure it is the raw retrieval signal.

What it measures (and what it does not):

* It reads the **frozen, deterministic** retrieval path
  :meth:`WorkbenchService.query_knowledge` — the same path the v1.8 pack
  evaluator uses — and scores the *ordered retrieved candidates* against each
  case's expected/forbidden sources. This is the upstream retrieval signal, not
  the post-gate grounded evidence, and not the composed report.
* It changes **no** retrieval, ranking, source-selection, sufficiency, grounding,
  composer, or memory semantics. Every call is a read. Nothing is written to the
  MemoryLedger, the proposal queue, or the knowledge library. The only outputs
  are an in-memory result set and (optionally) a report the caller writes.

Determinism: for a fixed pack and backend, ``query_knowledge`` is deterministic,
so running the same cases twice yields byte-identical results. The hybrid backend
is the meaningful lexical retrieval path; the default deterministic backend is a
whole-string encoder whose recall is near-chance for non-identical queries (a
documented baseline, not a bug to fix in this slice).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .workbench_service import WorkbenchService


# -- eval case format ---------------------------------------------------------

@dataclass(frozen=True)
class RetrievalEvalCase:
    """One retrieval probe: a query plus what should and must-not be retrieved.

    A case declares its expectations and the harness scores the retrieved
    candidates against them. All matching is case-insensitive.

    * ``expected_sources`` — source names (substring match) or exact source ids
      that *should* be retrieved for this query.
    * ``expected_chunk_ids`` — specific chunk ids that *should* be retrieved
      (the ``chk-...`` id, with or without the ``src:`` citation prefix).
    * ``forbidden_sources`` — source names whose presence in the results is
      relevance bleed.
    * ``forbidden_topic_terms`` — terms whose presence in a *non-expected*
      retrieved chunk is relevance bleed (e.g. ``"Copilot Studio"`` on a
      least-privilege query).
    * ``minimum_hit_k`` — the expected source/chunk must appear within the top
      ``minimum_hit_k`` retrieved candidates for the case to pass. ``0`` makes
      the hit constraint vacuous (used for known-gap probes).
    * ``tags`` — free-form labels (e.g. ``m365``, ``purview``, ``copilot``,
      ``memory``, ``decision``, ``source_librarian``).

    A case with no expected sources or chunk ids is a **gap probe**: hit/recall
    are undefined (reported as ``None``) and the case passes as long as no
    forbidden source/term bleeds in.
    """

    query: str
    expected_sources: List[str] = field(default_factory=list)
    expected_chunk_ids: List[str] = field(default_factory=list)
    forbidden_sources: List[str] = field(default_factory=list)
    forbidden_topic_terms: List[str] = field(default_factory=list)
    notes: str = ""
    minimum_hit_k: int = 1
    tags: List[str] = field(default_factory=list)
    case_id: str = ""
    # v3.0.2 optional classification/hygiene fields. All defaulted so every
    # v3.0 / v3.0.1 case loads unchanged. They drive the hygiene probe only —
    # the v3.0 scorer and the v3.0.1 report-path probe never read them.
    #   expected_topic_terms     terms that *legitimately* belong to this topic
    #   allowed_neighbour_sources sources that are on-topic neighbours, not bleed
    #   expected_gap             True when no authoritative source should exist
    #   query_shape              direct | value_sprint | report | decision_lookup
    #   classification_notes     human context for the wrong-source taxonomy
    expected_topic_terms: List[str] = field(default_factory=list)
    allowed_neighbour_sources: List[str] = field(default_factory=list)
    expected_gap: bool = False
    query_shape: str = "direct"
    classification_notes: str = ""

    @property
    def has_expected(self) -> bool:
        """Whether the case declares any expected source or chunk id."""
        return bool(self.expected_sources or self.expected_chunk_ids)

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalEvalCase":
        return cls(
            query=data["query"],
            expected_sources=list(data.get("expected_sources") or []),
            expected_chunk_ids=list(data.get("expected_chunk_ids") or []),
            forbidden_sources=list(data.get("forbidden_sources") or []),
            forbidden_topic_terms=list(data.get("forbidden_topic_terms") or []),
            notes=str(data.get("notes", "")),
            minimum_hit_k=int(data.get("minimum_hit_k", 1)),
            tags=list(data.get("tags") or []),
            case_id=str(data.get("case_id", "")),
            expected_topic_terms=list(data.get("expected_topic_terms") or []),
            allowed_neighbour_sources=list(
                data.get("allowed_neighbour_sources") or []),
            expected_gap=bool(data.get("expected_gap", False)),
            query_shape=str(data.get("query_shape", "direct")),
            classification_notes=str(data.get("classification_notes", "")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalCaseResult:
    """The scored outcome for one retrieval eval case.

    ``hit_at_*`` and ``expected_source_recall`` / ``wrong_source_rate`` are
    ``None`` when the case declares nothing to find (a gap probe), so the
    aggregate can average only over cases that actually assert retrieval.
    """

    case_id: str
    query: str
    passed: bool
    reason: str
    retrieved_chunk_count: int
    retrieved_sources: List[str]
    first_hit_rank: Optional[int]
    hit_at_1: Optional[bool]
    hit_at_3: Optional[bool]
    hit_at_5: Optional[bool]
    expected_source_recall: Optional[float]
    wrong_source_rate: Optional[float]
    missing_expected_sources: List[str]
    off_topic_hits: List[str]
    tags: List[str] = field(default_factory=list)

    @property
    def missing_expected_source_count(self) -> int:
        return len(self.missing_expected_sources)

    @property
    def off_topic_inclusion_count(self) -> int:
        return len(self.off_topic_hits)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["missing_expected_source_count"] = self.missing_expected_source_count
        data["off_topic_inclusion_count"] = self.off_topic_inclusion_count
        return data


@dataclass(frozen=True)
class RetrievalEvalSummary:
    """Aggregate retrieval-quality tallies across all cases."""

    query_count: int
    pass_count: int
    fail_count: int
    scored_case_count: int           # cases that declared expectations
    gap_probe_count: int             # cases with no expected target
    hit_at_1: Optional[float]
    hit_at_3: Optional[float]
    hit_at_5: Optional[float]
    expected_source_recall: Optional[float]
    wrong_source_rate: Optional[float]
    off_topic_inclusion_rate: float  # fraction of cases with any off-topic hit
    off_topic_hit_count: int
    missing_expected_source_count: int
    retrieved_chunk_count: int

    def to_dict(self) -> dict:
        return asdict(self)


# -- loading ------------------------------------------------------------------

def load_cases(path: str | Path) -> List[RetrievalEvalCase]:
    """Load retrieval eval cases from a JSONL file (``#`` lines are comments)."""
    cases: List[RetrievalEvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(RetrievalEvalCase.from_dict(json.loads(line)))
    return cases


# -- matching helpers ---------------------------------------------------------

def _norm_chunk_id(value: str) -> str:
    """Strip a ``src:`` citation prefix so chunk ids compare consistently."""
    return value[4:] if value.startswith("src:") else value


def _source_matches(expected: str, candidate: dict) -> bool:
    """Whether a retrieved candidate satisfies an expected-source token.

    Matches on exact source id, or case-insensitive substring of the source
    name (the same friendly convention the pack evaluator uses for names).
    """
    name = str(candidate.get("source_name") or "")
    source_id = str(candidate.get("source_id") or "")
    return expected == source_id or expected.lower() in name.lower()


def _candidate_matches_case(case: RetrievalEvalCase, candidate: dict) -> bool:
    """Whether a candidate satisfies *any* of the case's expected targets."""
    if any(_source_matches(src, candidate) for src in case.expected_sources):
        return True
    chunk_id = _norm_chunk_id(str(candidate.get("chunk_id") or ""))
    return any(_norm_chunk_id(cid) == chunk_id
               for cid in case.expected_chunk_ids)


# -- per-case scoring ---------------------------------------------------------

def evaluate_case(service: WorkbenchService,
                  case: RetrievalEvalCase) -> RetrievalCaseResult:
    """Score one case against the frozen ``query_knowledge`` retrieval path.

    Read-only: a single ``query_knowledge`` call, then pure measurement over the
    ordered candidates. No state is mutated.
    """
    audit = service.query_knowledge(case.query)
    candidates = list(audit.candidates)
    retrieved_count = len(candidates)
    retrieved_sources: List[str] = []
    for cand in candidates:
        name = str(cand.get("source_name") or "")
        if name and name not in retrieved_sources:
            retrieved_sources.append(name)

    # First rank (1-based) at which any expected target appears.
    first_hit_rank: Optional[int] = None
    for idx, cand in enumerate(candidates, start=1):
        if _candidate_matches_case(case, cand):
            first_hit_rank = idx
            break

    if case.has_expected:
        hit_at_1 = first_hit_rank is not None and first_hit_rank <= 1
        hit_at_3 = first_hit_rank is not None and first_hit_rank <= 3
        hit_at_5 = first_hit_rank is not None and first_hit_rank <= 5
    else:
        hit_at_1 = hit_at_3 = hit_at_5 = None

    # Expected-source recall and which expected tokens were never retrieved.
    expected_tokens = list(case.expected_sources) + [
        f"chunk:{cid}" for cid in case.expected_chunk_ids]
    missing: List[str] = []
    matched = 0
    for src in case.expected_sources:
        if any(_source_matches(src, cand) for cand in candidates):
            matched += 1
        else:
            missing.append(src)
    for cid in case.expected_chunk_ids:
        norm = _norm_chunk_id(cid)
        if any(_norm_chunk_id(str(cand.get("chunk_id") or "")) == norm
               for cand in candidates):
            matched += 1
        else:
            missing.append(f"chunk:{cid}")
    expected_source_recall: Optional[float] = (
        matched / len(expected_tokens) if expected_tokens else None)

    # Wrong-source rate: only meaningful when named source(s) are expected.
    if case.expected_sources and retrieved_count:
        wrong = sum(
            1 for cand in candidates
            if not any(_source_matches(src, cand)
                       for src in case.expected_sources))
        wrong_source_rate: Optional[float] = wrong / retrieved_count
    else:
        wrong_source_rate = None

    # Off-topic bleed: a *non-expected* candidate from a forbidden source, or
    # carrying a forbidden topic term. Expected candidates are excluded so a
    # term that legitimately lives in the expected source is never a false hit.
    off_topic_hits: List[str] = []
    for cand in candidates:
        if case.has_expected and _candidate_matches_case(case, cand):
            continue
        name = str(cand.get("source_name") or "")
        text = str(cand.get("text") or "").lower()
        for forbidden in case.forbidden_sources:
            if _source_matches(forbidden, cand):
                off_topic_hits.append(f"{name}: forbidden source")
                break
        else:
            for term in case.forbidden_topic_terms:
                if term.lower() in text:
                    off_topic_hits.append(f"{name}: term '{term}'")
                    break

    # Pass/fail. Hit constraint applies only when the case expects something and
    # minimum_hit_k > 0; the no-bleed constraint always applies.
    reasons: List[str] = []
    hit_ok = True
    if case.has_expected and case.minimum_hit_k > 0:
        hit_ok = (first_hit_rank is not None
                  and first_hit_rank <= case.minimum_hit_k)
        if not hit_ok:
            where = (f"rank {first_hit_rank}" if first_hit_rank is not None
                     else "not retrieved")
            reasons.append(
                f"expected source missing within top-{case.minimum_hit_k} "
                f"({where})")
    if off_topic_hits:
        reasons.append(f"off-topic bleed: {', '.join(off_topic_hits)}")
    passed = hit_ok and not off_topic_hits
    if passed:
        reason = ("gap probe — no off-topic bleed"
                  if not case.has_expected else "retrieval within tolerance")
    else:
        reason = "; ".join(reasons)

    return RetrievalCaseResult(
        case_id=case.case_id or case.query[:40],
        query=case.query,
        passed=passed,
        reason=reason,
        retrieved_chunk_count=retrieved_count,
        retrieved_sources=retrieved_sources,
        first_hit_rank=first_hit_rank,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        expected_source_recall=expected_source_recall,
        wrong_source_rate=wrong_source_rate,
        missing_expected_sources=missing,
        off_topic_hits=off_topic_hits,
        tags=list(case.tags),
    )


def run_eval(service: WorkbenchService,
             cases: List[RetrievalEvalCase]) -> List[RetrievalCaseResult]:
    """Score every case through the frozen retrieval path (read-only)."""
    return [evaluate_case(service, case) for case in cases]


# -- aggregation --------------------------------------------------------------

def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def summarize(results: List[RetrievalCaseResult]) -> RetrievalEvalSummary:
    """Aggregate per-case results into retrieval-quality tallies.

    Hit-rate and recall averages are taken only over cases that declared
    expectations; gap probes (no expected target) are excluded from those means
    but still counted in ``query_count`` and the off-topic-bleed rate.
    """
    scored = [r for r in results if r.hit_at_1 is not None]
    recall_vals = [r.expected_source_recall for r in results
                   if r.expected_source_recall is not None]
    wrong_vals = [r.wrong_source_rate for r in results
                  if r.wrong_source_rate is not None]
    off_topic_cases = sum(1 for r in results if r.off_topic_inclusion_count)
    return RetrievalEvalSummary(
        query_count=len(results),
        pass_count=sum(1 for r in results if r.passed),
        fail_count=sum(1 for r in results if not r.passed),
        scored_case_count=len(scored),
        gap_probe_count=len(results) - len(scored),
        hit_at_1=_mean([1.0 if r.hit_at_1 else 0.0 for r in scored]),
        hit_at_3=_mean([1.0 if r.hit_at_3 else 0.0 for r in scored]),
        hit_at_5=_mean([1.0 if r.hit_at_5 else 0.0 for r in scored]),
        expected_source_recall=_mean(recall_vals),
        wrong_source_rate=_mean(wrong_vals),
        off_topic_inclusion_rate=(off_topic_cases / len(results)
                                  if results else 0.0),
        off_topic_hit_count=sum(r.off_topic_inclusion_count for r in results),
        missing_expected_source_count=sum(
            r.missing_expected_source_count for r in results),
        retrieved_chunk_count=sum(r.retrieved_chunk_count for r in results),
    )


# -- rendering ----------------------------------------------------------------

def _fmt_opt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(results: List[RetrievalCaseResult],
                    summary: RetrievalEvalSummary, *,
                    pack_label: str, backend_label: str) -> str:
    """Render a readable Markdown retrieval-quality report (no side effects)."""
    lines: List[str] = []
    lines.append(f"# Retrieval evaluation — {pack_label} ({backend_label})")
    lines.append("")
    lines.append("Read-only baseline of the frozen `query_knowledge` retrieval "
                 "path. Measures retrieval quality; changes nothing.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- queries: {summary.query_count} "
                 f"(scored {summary.scored_case_count}, "
                 f"gap probes {summary.gap_probe_count})")
    lines.append(f"- pass / fail: {summary.pass_count} / {summary.fail_count}")
    lines.append(f"- hit@1 / hit@3 / hit@5: {_fmt_opt(summary.hit_at_1)} / "
                 f"{_fmt_opt(summary.hit_at_3)} / {_fmt_opt(summary.hit_at_5)}")
    lines.append(f"- expected_source_recall: "
                 f"{_fmt_opt(summary.expected_source_recall)}")
    lines.append(f"- wrong_source_rate: {_fmt_opt(summary.wrong_source_rate)}")
    lines.append(f"- off_topic_inclusion_rate: "
                 f"{summary.off_topic_inclusion_rate:.2f} "
                 f"({summary.off_topic_hit_count} hit(s))")
    lines.append(f"- missing_expected_source_count: "
                 f"{summary.missing_expected_source_count}")
    lines.append(f"- retrieved_chunk_count: {summary.retrieved_chunk_count}")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| case | result | first hit | hit@1/3/5 | recall | "
                 "wrong-src | off-topic | reason |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        rank = "—" if r.first_hit_rank is None else str(r.first_hit_rank)
        hits = "/".join(
            "—" if v is None else ("Y" if v else "n")
            for v in (r.hit_at_1, r.hit_at_3, r.hit_at_5))
        lines.append(
            f"| {r.case_id} | {verdict} | {rank} | {hits} | "
            f"{_fmt_opt(r.expected_source_recall)} | "
            f"{_fmt_opt(r.wrong_source_rate)} | "
            f"{r.off_topic_inclusion_count} | {r.reason} |")
    lines.append("")
    return "\n".join(lines)


def write_reports(results: List[RetrievalCaseResult],
                  summary: RetrievalEvalSummary, *,
                  md_path: str | Path, jsonl_path: str | Path,
                  pack_label: str, backend_label: str) -> None:
    """Write the Markdown and JSONL reports (only when a caller asks for them)."""
    md_path = Path(md_path)
    jsonl_path = Path(jsonl_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_markdown(results, summary, pack_label=pack_label,
                        backend_label=backend_label),
        encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"summary": summary.to_dict()}) + "\n")
        for r in results:
            handle.write(json.dumps(r.to_dict()) + "\n")


# =============================================================================
# v3.0.1 Report-path probe — localise *where* relevance bleed is introduced
# =============================================================================
#
# The v3.0 harness above measures only the *raw* retrieval signal
# (``query_knowledge``). A known historical defect — an off-topic Copilot Studio
# connector caution in a least-privilege report — did **not** reproduce at that
# raw layer. The bleed, if it exists, must therefore be introduced *later* in the
# report path. This probe instruments the full path read-only and reports the
# first stage at which a forbidden source/term appears:
#
#   STAGE 1  raw       — ``query_knowledge(q).candidates`` (ordered candidates)
#   STAGE 2  selected  — ``build_grounding_package(q).evidence`` (post-gate,
#                         re-ordered evidence the composer is allowed to cite)
#   STAGE 3  final     — ``answer_query(q, ConsultantReportComposer()).answer``
#                         (the final cited report spans / citations)
#
# Every call is a read. ``query_knowledge``, ``build_grounding_package`` and
# ``answer_query`` mutate no ledger, proposal queue, report, or pack — the same
# read-only path the value sprint already exercises. This probe changes **no**
# retrieval, ranking, source-selection, grounding, composer, or memory semantics;
# it only observes and tallies. If no bleed is found, that is the honest finding.

_STAGE_NAMES = ("raw", "selected", "final")


def _stage_off_topic_hits(case: RetrievalEvalCase,
                          items: List[dict]) -> List[str]:
    """Forbidden source/term hits among a stage's *non-expected* items.

    Identical bleed rule to :func:`evaluate_case`, applied to any normalised
    stage item (``source_name`` / ``source_id`` / ``chunk_id`` / ``text``):
    an expected item is never counted, then a forbidden source or a forbidden
    topic term in the item's text is recorded as bleed.
    """
    hits: List[str] = []
    for item in items:
        if case.has_expected and _candidate_matches_case(case, item):
            continue
        name = str(item.get("source_name") or "")
        text = str(item.get("text") or "").lower()
        for forbidden in case.forbidden_sources:
            if _source_matches(forbidden, item):
                hits.append(f"{name or '?'}: forbidden source")
                break
        else:
            for term in case.forbidden_topic_terms:
                if term.lower() in text:
                    hits.append(f"{name or '?'}: term '{term}'")
                    break
    return hits


def _stage_wrong_source_rate(case: RetrievalEvalCase,
                             items: List[dict]) -> Optional[float]:
    """Fraction of a stage's items that are not an expected source.

    ``None`` when the case names no expected source or the stage is empty, so an
    aggregate can average only over stages where the rate is defined.
    """
    if not case.expected_sources or not items:
        return None
    wrong = sum(
        1 for item in items
        if not any(_source_matches(src, item)
                   for src in case.expected_sources))
    return wrong / len(items)


@dataclass(frozen=True)
class StageObservation:
    """What one report-path stage retrieved/selected/cited for one case."""

    stage: str                       # "raw" | "selected" | "final"
    item_count: int
    source_names: List[str]
    chunk_ids: List[str]
    off_topic_hits: List[str]
    wrong_source_rate: Optional[float]

    @property
    def off_topic_count(self) -> int:
        return len(self.off_topic_hits)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["off_topic_count"] = self.off_topic_count
        return data


@dataclass(frozen=True)
class ReportPathCaseResult:
    """Per-case, per-stage bleed localisation for one query."""

    case_id: str
    query: str
    raw: StageObservation
    selected: StageObservation
    final: StageObservation
    final_citations: List[str]
    bleed_introduced_stage: Optional[str]  # first stage with bleed, else None
    tags: List[str] = field(default_factory=list)

    def stage(self, name: str) -> StageObservation:
        return {"raw": self.raw, "selected": self.selected,
                "final": self.final}[name]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "raw": self.raw.to_dict(),
            "selected": self.selected.to_dict(),
            "final": self.final.to_dict(),
            "final_citations": list(self.final_citations),
            "bleed_introduced_stage": self.bleed_introduced_stage,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ReportPathSummary:
    """Aggregate bleed-localisation tallies across all probed cases."""

    case_count: int
    raw_off_topic_rate: float
    selected_evidence_off_topic_rate: float
    final_citation_off_topic_rate: float
    raw_wrong_source_rate: Optional[float]
    selected_wrong_source_rate: Optional[float]
    final_wrong_source_rate: Optional[float]
    bleed_stage_counts: dict          # {"raw": n, "selected": n, "final": n}
    cases_with_bleed: int
    bleed_reproduced: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_evidence_items(evidence) -> List[dict]:
    """Normalise grounding-package evidence into stage-item dicts.

    Only knowledge evidence carries a source; memory evidence is kept (with a
    ``None`` source_name and its ``mem:`` id as chunk id) so a forbidden term
    bleeding through a memory would still be observed.
    """
    items: List[dict] = []
    for ev in evidence:
        citation = str(getattr(ev, "citation_id", "") or "")
        items.append({
            "source_name": getattr(ev, "source_name", None),
            "source_id": None,
            "chunk_id": _norm_chunk_id(citation),
            "text": getattr(ev, "text", "") or "",
        })
    return items


def _norm_span_items(spans) -> List[dict]:
    """Normalise composed report spans into stage-item dicts (final stage)."""
    items: List[dict] = []
    for span in spans:
        citation = str(getattr(span, "citation_id", "") or "")
        items.append({
            "source_name": getattr(span, "source_name", None),
            "source_id": None,
            "chunk_id": _norm_chunk_id(citation),
            "text": getattr(span, "text", "") or "",
        })
    return items


def _build_stage(stage: str, case: RetrievalEvalCase,
                 items: List[dict]) -> StageObservation:
    source_names: List[str] = []
    chunk_ids: List[str] = []
    for item in items:
        name = str(item.get("source_name") or "")
        if name and name not in source_names:
            source_names.append(name)
        chunk = str(item.get("chunk_id") or "")
        if chunk and chunk not in chunk_ids:
            chunk_ids.append(chunk)
    return StageObservation(
        stage=stage,
        item_count=len(items),
        source_names=source_names,
        chunk_ids=chunk_ids,
        off_topic_hits=_stage_off_topic_hits(case, items),
        wrong_source_rate=_stage_wrong_source_rate(case, items),
    )


def probe_case(service: WorkbenchService,
               case: RetrievalEvalCase) -> ReportPathCaseResult:
    """Trace one case through the full report path and localise any bleed.

    Read-only: three reads of the frozen path (``query_knowledge``,
    ``build_grounding_package``, ``answer_query`` with the consultant report
    composer), then pure measurement. Nothing is mutated.
    """
    # Lazy import: keeps the composer dependency out of module import (and mirrors
    # the value sprint, which imports the composer only when it composes).
    from slm.assistant_composer import ConsultantReportComposer

    # STAGE 1 — raw retrieved candidates.
    audit = service.query_knowledge(case.query)
    raw_items = list(audit.candidates)
    raw = _build_stage("raw", case, raw_items)

    # STAGE 2 — evidence selected/passed into the composer (post-gate).
    package = service.build_grounding_package(case.query)
    selected_items = _norm_evidence_items(package.evidence)
    selected = _build_stage("selected", case, selected_items)

    # STAGE 3 — final cited report spans / citations.
    result = service.answer_query(
        case.query, composer=ConsultantReportComposer())
    final_items = _norm_span_items(result.answer.spans)
    final = _build_stage("final", case, final_items)

    bleed_introduced_stage: Optional[str] = None
    for obs in (raw, selected, final):
        if obs.off_topic_count:
            bleed_introduced_stage = obs.stage
            break

    return ReportPathCaseResult(
        case_id=case.case_id or case.query[:40],
        query=case.query,
        raw=raw,
        selected=selected,
        final=final,
        final_citations=list(result.answer.citations),
        bleed_introduced_stage=bleed_introduced_stage,
        tags=list(case.tags),
    )


def run_probe(service: WorkbenchService,
              cases: List[RetrievalEvalCase]) -> List[ReportPathCaseResult]:
    """Probe every case through the full report path (read-only)."""
    return [probe_case(service, case) for case in cases]


def summarize_probe(
        results: List[ReportPathCaseResult]) -> ReportPathSummary:
    """Aggregate per-case stage observations into bleed-localisation tallies."""
    n = len(results)

    def _off_rate(stage: str) -> float:
        if not n:
            return 0.0
        return sum(1 for r in results if r.stage(stage).off_topic_count) / n

    def _wrong_mean(stage: str) -> Optional[float]:
        vals = [r.stage(stage).wrong_source_rate for r in results
                if r.stage(stage).wrong_source_rate is not None]
        return _mean(vals)

    stage_counts = {name: sum(1 for r in results
                              if r.bleed_introduced_stage == name)
                    for name in _STAGE_NAMES}
    cases_with_bleed = sum(1 for r in results
                           if r.bleed_introduced_stage is not None)
    return ReportPathSummary(
        case_count=n,
        raw_off_topic_rate=_off_rate("raw"),
        selected_evidence_off_topic_rate=_off_rate("selected"),
        final_citation_off_topic_rate=_off_rate("final"),
        raw_wrong_source_rate=_wrong_mean("raw"),
        selected_wrong_source_rate=_wrong_mean("selected"),
        final_wrong_source_rate=_wrong_mean("final"),
        bleed_stage_counts=stage_counts,
        cases_with_bleed=cases_with_bleed,
        bleed_reproduced=cases_with_bleed > 0,
    )


def render_probe_markdown(results: List[ReportPathCaseResult],
                          summary: ReportPathSummary, *,
                          pack_label: str, backend_label: str) -> str:
    """Render the report-path probe as Markdown (no side effects)."""
    lines: List[str] = []
    lines.append(f"# Report-path probe — {pack_label} ({backend_label})")
    lines.append("")
    lines.append("Read-only trace of the full report path. Localises the first "
                 "stage at which a forbidden source/term appears; changes "
                 "nothing.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- cases: {summary.case_count}")
    if summary.bleed_reproduced:
        lines.append(f"- **bleed reproduced**: yes "
                     f"({summary.cases_with_bleed} case(s))")
        order = ", ".join(f"{name}={summary.bleed_stage_counts[name]}"
                          for name in _STAGE_NAMES)
        lines.append(f"- first-bleed stage counts: {order}")
    else:
        lines.append("- **bleed reproduced**: no — not reproduced at any stage "
                     "(raw, selected, or final)")
    lines.append(f"- off-topic rate raw / selected / final: "
                 f"{summary.raw_off_topic_rate:.2f} / "
                 f"{summary.selected_evidence_off_topic_rate:.2f} / "
                 f"{summary.final_citation_off_topic_rate:.2f}")
    lines.append(f"- wrong-source rate raw / selected / final: "
                 f"{_fmt_opt(summary.raw_wrong_source_rate)} / "
                 f"{_fmt_opt(summary.selected_wrong_source_rate)} / "
                 f"{_fmt_opt(summary.final_wrong_source_rate)}")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| case | raw off / wrong | selected off / wrong | "
                 "final off / wrong | first bleed |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        def _cell(obs: StageObservation) -> str:
            return f"{obs.off_topic_count} / {_fmt_opt(obs.wrong_source_rate)}"
        bleed = r.bleed_introduced_stage or "not reproduced"
        lines.append(
            f"| {r.case_id} | {_cell(r.raw)} | {_cell(r.selected)} | "
            f"{_cell(r.final)} | {bleed} |")
    lines.append("")
    return "\n".join(lines)


def write_probe_reports(results: List[ReportPathCaseResult],
                        summary: ReportPathSummary, *,
                        md_path: str | Path, jsonl_path: str | Path,
                        pack_label: str, backend_label: str) -> None:
    """Write the probe Markdown and JSONL reports (only when asked)."""
    md_path = Path(md_path)
    jsonl_path = Path(jsonl_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_probe_markdown(results, summary, pack_label=pack_label,
                              backend_label=backend_label),
        encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"summary": summary.to_dict()}) + "\n")
        for r in results:
            handle.write(json.dumps(r.to_dict()) + "\n")


# =============================================================================
# v3.0.2 Harder corpus + source/chunk hygiene probe — classification only
# =============================================================================
#
# The v3.0 harness measures raw retrieval; the v3.0.1 probe localises bleed
# across the report path. Both found the historical off-topic Copilot Studio
# caution does *not* reproduce, and that the non-zero ``wrong_source_rate`` is
# made of *other retrieved sources* — but neither distinguishes a **forbidden
# bleed** from a **legitimate on-topic neighbour**. This layer adds that
# classification plus chunk/source hygiene diagnostics, so a non-expected
# candidate is sorted into one of four buckets:
#
#   forbidden_bleed    — matches a forbidden source or a forbidden topic term
#   on_topic_neighbour — not expected, but an allowed neighbour source or shares
#                        an expected topic term (legitimate adjacent evidence)
#   ambiguous          — neither clearly forbidden nor clearly on-topic
#   expected_gap       — a known no-answer case; retrieval surfaced something the
#                        system must not present as an authoritative source
#
# Every call is a read of the frozen ``query_knowledge`` path. This layer changes
# no retrieval, ranking, source-selection, grounding, composer, report-rendering,
# or memory behaviour; it only observes and classifies. A clean result (no
# forbidden bleed reproduced) is recorded honestly, never forced into a failure.

CLASS_EXPECTED = "expected"
CLASS_FORBIDDEN_BLEED = "forbidden_bleed"
CLASS_ON_TOPIC_NEIGHBOUR = "on_topic_neighbour"
CLASS_AMBIGUOUS = "ambiguous"
CLASS_EXPECTED_GAP = "expected_gap"

_WRONG_SOURCE_CLASSES = (
    CLASS_FORBIDDEN_BLEED, CLASS_ON_TOPIC_NEIGHBOUR,
    CLASS_AMBIGUOUS, CLASS_EXPECTED_GAP,
)


def _matched_topic_terms(case: RetrievalEvalCase, candidate: dict) -> List[str]:
    """Expected topic terms present in a candidate's text or source name."""
    text = str(candidate.get("text") or "").lower()
    name = str(candidate.get("source_name") or "").lower()
    return [t for t in case.expected_topic_terms
            if t.lower() in text or t.lower() in name]


def _forbidden_terms_found(case: RetrievalEvalCase,
                           candidate: dict) -> List[str]:
    """Forbidden topic terms present in a candidate's text."""
    text = str(candidate.get("text") or "").lower()
    return [t for t in case.forbidden_topic_terms if t.lower() in text]


def _candidate_is_forbidden(case: RetrievalEvalCase, candidate: dict) -> bool:
    """Whether a candidate trips a forbidden source or forbidden topic term."""
    if any(_source_matches(src, candidate) for src in case.forbidden_sources):
        return True
    return bool(_forbidden_terms_found(case, candidate))


def classify_candidate(case: RetrievalEvalCase, candidate: dict) -> str:
    """Sort one retrieved candidate into the wrong-source taxonomy.

    An expected candidate is ``expected`` (a hit, not a wrong source). Otherwise
    the order is: forbidden bleed first (it dominates), then a gap case marks
    every non-forbidden candidate ``expected_gap`` (the system must not pass it
    off as authoritative), then an allowed-neighbour source or a shared expected
    topic term makes it an ``on_topic_neighbour``, else ``ambiguous``.
    """
    if case.has_expected and _candidate_matches_case(case, candidate):
        return CLASS_EXPECTED
    if _candidate_is_forbidden(case, candidate):
        return CLASS_FORBIDDEN_BLEED
    if case.expected_gap:
        return CLASS_EXPECTED_GAP
    if any(_source_matches(src, candidate)
           for src in case.allowed_neighbour_sources):
        return CLASS_ON_TOPIC_NEIGHBOUR
    if _matched_topic_terms(case, candidate):
        return CLASS_ON_TOPIC_NEIGHBOUR
    return CLASS_AMBIGUOUS


@dataclass(frozen=True)
class CandidateDiagnostic:
    """Per-candidate hygiene + classification record (read-only observation)."""

    rank: int
    source_name: str
    chunk_id: str
    classification: str
    topic_density: Optional[float]          # matched / total expected terms
    matched_topic_terms: List[str]
    forbidden_terms_found: List[str]
    contains_forbidden_terms: bool
    is_mixed_topic: bool                     # expected AND forbidden term present

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HygieneCaseResult:
    """Per-case classification + chunk/source hygiene for one query."""

    case_id: str
    query: str
    query_shape: str
    expected_gap: bool
    passed: bool
    reason: str
    retrieved_chunk_count: int
    first_hit_rank: Optional[int]
    classification_counts: dict              # over non-expected candidates
    mixed_topic_chunk_count: int
    chunk_with_forbidden_terms_count: int
    chunk_topic_density: Optional[float]     # mean per-candidate density
    source_topic_overlap: Optional[float]    # expected-source chunks on-topic
    candidates: List[CandidateDiagnostic] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def forbidden_bleed_count(self) -> int:
        return self.classification_counts.get(CLASS_FORBIDDEN_BLEED, 0)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "query_shape": self.query_shape,
            "expected_gap": self.expected_gap,
            "passed": self.passed,
            "reason": self.reason,
            "retrieved_chunk_count": self.retrieved_chunk_count,
            "first_hit_rank": self.first_hit_rank,
            "classification_counts": dict(self.classification_counts),
            "mixed_topic_chunk_count": self.mixed_topic_chunk_count,
            "chunk_with_forbidden_terms_count":
                self.chunk_with_forbidden_terms_count,
            "chunk_topic_density": self.chunk_topic_density,
            "source_topic_overlap": self.source_topic_overlap,
            "candidates": [c.to_dict() for c in self.candidates],
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class HygieneSummary:
    """Aggregate wrong-source taxonomy + hygiene tallies across all cases."""

    case_count: int
    pass_count: int
    fail_count: int
    wrong_source_classification: dict        # bucket -> total candidate count
    mixed_topic_chunk_count: int
    chunk_with_forbidden_terms_count: int
    mean_chunk_topic_density: Optional[float]
    mean_source_topic_overlap: Optional[float]
    value_sprint_query_pass_rate: Optional[float]
    gap_case_pass_rate: Optional[float]
    forbidden_bleed_reproduced: bool

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose_case(service: WorkbenchService,
                  case: RetrievalEvalCase) -> HygieneCaseResult:
    """Classify every retrieved candidate and measure chunk/source hygiene.

    Read-only: a single ``query_knowledge`` call, then pure measurement over the
    ordered candidates. No state is mutated.
    """
    audit = service.query_knowledge(case.query)
    candidates = list(audit.candidates)

    diagnostics: List[CandidateDiagnostic] = []
    counts = {name: 0 for name in _WRONG_SOURCE_CLASSES}
    first_hit_rank: Optional[int] = None
    densities: List[float] = []
    mixed = 0
    forbidden_chunks = 0
    expected_source_chunks = 0
    expected_source_on_topic = 0

    for idx, cand in enumerate(candidates, start=1):
        is_expected = case.has_expected and _candidate_matches_case(case, cand)
        if is_expected and first_hit_rank is None:
            first_hit_rank = idx
        classification = classify_candidate(case, cand)
        if classification != CLASS_EXPECTED:
            counts[classification] += 1

        matched = _matched_topic_terms(case, cand)
        forbidden = _forbidden_terms_found(case, cand)
        density: Optional[float] = (
            len(matched) / len(case.expected_topic_terms)
            if case.expected_topic_terms else None)
        if density is not None:
            densities.append(density)
        is_mixed = bool(matched and forbidden)
        if is_mixed:
            mixed += 1
        if forbidden:
            forbidden_chunks += 1
        if case.expected_sources and any(
                _source_matches(src, cand) for src in case.expected_sources):
            expected_source_chunks += 1
            if matched:
                expected_source_on_topic += 1

        diagnostics.append(CandidateDiagnostic(
            rank=idx,
            source_name=str(cand.get("source_name") or ""),
            chunk_id=_norm_chunk_id(str(cand.get("chunk_id") or "")),
            classification=classification,
            topic_density=density,
            matched_topic_terms=matched,
            forbidden_terms_found=forbidden,
            contains_forbidden_terms=bool(forbidden),
            is_mixed_topic=is_mixed,
        ))

    chunk_topic_density = _mean(densities)
    source_topic_overlap: Optional[float] = (
        expected_source_on_topic / expected_source_chunks
        if (expected_source_chunks and case.expected_topic_terms) else None)

    # Pass logic: forbidden bleed always fails. A normal case also needs its
    # expected source within minimum_hit_k; a gap case has nothing to hit and
    # passes as long as nothing forbidden bled in.
    forbidden_count = counts[CLASS_FORBIDDEN_BLEED]
    reasons: List[str] = []
    hit_ok = True
    if not case.expected_gap and case.has_expected and case.minimum_hit_k > 0:
        hit_ok = (first_hit_rank is not None
                  and first_hit_rank <= case.minimum_hit_k)
        if not hit_ok:
            where = (f"rank {first_hit_rank}" if first_hit_rank is not None
                     else "not retrieved")
            reasons.append(
                f"expected source missing within top-{case.minimum_hit_k} "
                f"({where})")
    if forbidden_count:
        reasons.append(f"forbidden bleed x{forbidden_count}")
    passed = hit_ok and forbidden_count == 0
    if passed:
        reason = ("gap case — no forbidden bleed" if case.expected_gap
                  else "clean — expected hit, no forbidden bleed")
    else:
        reason = "; ".join(reasons)

    return HygieneCaseResult(
        case_id=case.case_id or case.query[:40],
        query=case.query,
        query_shape=case.query_shape,
        expected_gap=case.expected_gap,
        passed=passed,
        reason=reason,
        retrieved_chunk_count=len(candidates),
        first_hit_rank=first_hit_rank,
        classification_counts=counts,
        mixed_topic_chunk_count=mixed,
        chunk_with_forbidden_terms_count=forbidden_chunks,
        chunk_topic_density=chunk_topic_density,
        source_topic_overlap=source_topic_overlap,
        candidates=diagnostics,
        tags=list(case.tags),
    )


def run_hygiene(service: WorkbenchService,
                cases: List[RetrievalEvalCase]) -> List[HygieneCaseResult]:
    """Diagnose every case through the frozen retrieval path (read-only)."""
    return [diagnose_case(service, case) for case in cases]


def summarize_hygiene(results: List[HygieneCaseResult]) -> HygieneSummary:
    """Aggregate per-case classification + hygiene into corpus-level tallies."""
    counts = {name: 0 for name in _WRONG_SOURCE_CLASSES}
    for r in results:
        for name in _WRONG_SOURCE_CLASSES:
            counts[name] += r.classification_counts.get(name, 0)

    vs_cases = [r for r in results if r.query_shape == "value_sprint"]
    gap_cases = [r for r in results if r.expected_gap]
    densities = [r.chunk_topic_density for r in results
                 if r.chunk_topic_density is not None]
    overlaps = [r.source_topic_overlap for r in results
                if r.source_topic_overlap is not None]
    return HygieneSummary(
        case_count=len(results),
        pass_count=sum(1 for r in results if r.passed),
        fail_count=sum(1 for r in results if not r.passed),
        wrong_source_classification=counts,
        mixed_topic_chunk_count=sum(r.mixed_topic_chunk_count for r in results),
        chunk_with_forbidden_terms_count=sum(
            r.chunk_with_forbidden_terms_count for r in results),
        mean_chunk_topic_density=_mean(densities),
        mean_source_topic_overlap=_mean(overlaps),
        value_sprint_query_pass_rate=(
            _mean([1.0 if r.passed else 0.0 for r in vs_cases])
            if vs_cases else None),
        gap_case_pass_rate=(
            _mean([1.0 if r.passed else 0.0 for r in gap_cases])
            if gap_cases else None),
        forbidden_bleed_reproduced=counts[CLASS_FORBIDDEN_BLEED] > 0,
    )


def render_hygiene_markdown(results: List[HygieneCaseResult],
                            summary: HygieneSummary, *,
                            pack_label: str, backend_label: str) -> str:
    """Render the harder-corpus hygiene probe as Markdown (no side effects)."""
    lines: List[str] = []
    lines.append(f"# Harder-corpus hygiene probe — {pack_label} "
                 f"({backend_label})")
    lines.append("")
    lines.append("Read-only classification of every retrieved candidate into "
                 "the wrong-source taxonomy, plus chunk/source hygiene. Changes "
                 "no retrieval/ranking/source/composer/grounding/memory "
                 "behaviour.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- cases: {summary.case_count} "
                 f"(pass {summary.pass_count} / fail {summary.fail_count})")
    if summary.forbidden_bleed_reproduced:
        lines.append(
            f"- **forbidden bleed reproduced**: yes "
            f"({summary.wrong_source_classification[CLASS_FORBIDDEN_BLEED]} "
            f"candidate(s))")
    else:
        lines.append("- **forbidden bleed reproduced**: no — every non-expected "
                     "candidate is an on-topic neighbour, ambiguous, or an "
                     "expected-gap surface")
    wsc = summary.wrong_source_classification
    lines.append(f"- wrong-source classification: "
                 f"forbidden_bleed={wsc[CLASS_FORBIDDEN_BLEED]}, "
                 f"on_topic_neighbour={wsc[CLASS_ON_TOPIC_NEIGHBOUR]}, "
                 f"ambiguous={wsc[CLASS_AMBIGUOUS]}, "
                 f"expected_gap={wsc[CLASS_EXPECTED_GAP]}")
    lines.append(f"- mixed_topic_chunk_count: "
                 f"{summary.mixed_topic_chunk_count}")
    lines.append(f"- chunk_with_forbidden_terms_count: "
                 f"{summary.chunk_with_forbidden_terms_count}")
    lines.append(f"- mean_chunk_topic_density: "
                 f"{_fmt_opt(summary.mean_chunk_topic_density)}")
    lines.append(f"- mean_source_topic_overlap: "
                 f"{_fmt_opt(summary.mean_source_topic_overlap)}")
    lines.append(f"- value_sprint_query_pass_rate: "
                 f"{_fmt_opt(summary.value_sprint_query_pass_rate)}")
    lines.append(f"- gap_case_pass_rate: "
                 f"{_fmt_opt(summary.gap_case_pass_rate)}")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| case | shape | result | first hit | f_bleed | neighbour | "
                 "ambig | gap | mixed | density | src-overlap |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        rank = "—" if r.first_hit_rank is None else str(r.first_hit_rank)
        c = r.classification_counts
        lines.append(
            f"| {r.case_id} | {r.query_shape} | {verdict} | {rank} | "
            f"{c.get(CLASS_FORBIDDEN_BLEED, 0)} | "
            f"{c.get(CLASS_ON_TOPIC_NEIGHBOUR, 0)} | "
            f"{c.get(CLASS_AMBIGUOUS, 0)} | "
            f"{c.get(CLASS_EXPECTED_GAP, 0)} | "
            f"{r.mixed_topic_chunk_count} | "
            f"{_fmt_opt(r.chunk_topic_density)} | "
            f"{_fmt_opt(r.source_topic_overlap)} |")
    lines.append("")
    return "\n".join(lines)


def write_hygiene_reports(results: List[HygieneCaseResult],
                          summary: HygieneSummary, *,
                          md_path: str | Path, jsonl_path: str | Path,
                          pack_label: str, backend_label: str) -> None:
    """Write the hygiene Markdown and JSONL reports (only when asked)."""
    md_path = Path(md_path)
    jsonl_path = Path(jsonl_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_hygiene_markdown(results, summary, pack_label=pack_label,
                                backend_label=backend_label),
        encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"summary": summary.to_dict()}) + "\n")
        for r in results:
            handle.write(json.dumps(r.to_dict()) + "\n")


# =============================================================================
# v6.7 Imported PDF Retrieval Evaluation — read-only
# =============================================================================
#
# The v6.6 importer turns an *approved* PDF chunk preview into a knowledge pack
# (manifest.json + knowledge.jsonl). Importing a pack does not make it
# trustworthy: an imported PDF must *earn* its place by improving retrieval and
# citation without bleeding wrong sources, dragging in off-topic neighbours, or
# regressing the existing baseline. This layer measures exactly that, and only
# measures — it reuses the same frozen read path the v3.0 harness and the v3.0.1
# report-path probe use (``query_knowledge`` -> ``build_grounding_package`` ->
# ``answer_query``) and writes nothing to any pack, index, ledger, registry, or
# proposal queue.
#
# Three honest stages are traced per case:
#   * raw      — the ordered ``query_knowledge`` candidates (always top-k; the
#                backends apply no score gate, so an off-topic query still
#                returns the pack's only chunks here — that is documented, not a
#                defect).
#   * selected — the grounded evidence after the v2.4 relevance/sufficiency gate
#                (``build_grounding_package``). The gate is downgrade-only: a
#                weak/no-support query yields *empty* evidence. This is where
#                pack **isolation** is provable: an unrelated query selects no
#                PDF chunk even though raw retrieval surfaced one.
#   * final    — the cited report spans/citations from the consultant report
#                composer. Citations can only reference selected evidence.
#
# Page lineage: retrieval candidates expose page provenance only through their
# ``section`` string ("page N" / "pages N-M"), so this layer parses that and
# builds a chunk_id -> (source, pages) map from the raw stage to enrich the
# selected/final stages (selection is always a subset of retrieval).
#
# Comparison: the caller may supply a *baseline* result set (the same cases run
# against a service that does **not** have the imported pack). The comparison is
# pure: it diffs the two result sets to show which cases the pack improved, left
# unchanged, or regressed, and whether it introduced any new wrong-source or
# forbidden-source hits or affected any unrelated case. It never re-runs
# retrieval and never mutates anything.

# -- pass/fail thresholds (explicit, deterministic) ---------------------------

# An expected-content case must retrieve *all* its expected sources to pass; with
# a single expected source this is simply "the expected source was retrieved".
MIN_EXPECTED_SOURCE_RECALL = 1.0
# Token-overlap ratio at or above which two retrieved chunks are "near
# duplicates" (exact-text duplicates are counted separately).
_NEAR_DUP_JACCARD = 0.8

# -- failure classifications (one per case; diagnostic only) ------------------

PDF_NO_FAILURE = "no_failure"
PDF_RETRIEVAL_MISS = "retrieval_miss"
PDF_WRONG_SOURCE_RANKED = "wrong_source_ranked"
PDF_EXPECTED_SOURCE_LOW_RANK = "expected_source_low_rank"
PDF_EXPECTED_CHUNK_LOW_RANK = "expected_chunk_low_rank"
PDF_OFF_TOPIC_NEIGHBOUR = "off_topic_neighbour"
PDF_DUPLICATE_CHUNK_INTERFERENCE = "duplicate_chunk_interference"
PDF_CHUNK_BOUNDARY_FAILURE = "chunk_boundary_failure"
PDF_PAGE_LINEAGE_FAILURE = "page_lineage_failure"
PDF_SELECTION_DROP = "selection_drop"
PDF_CITATION_DROP = "citation_drop"
PDF_INSUFFICIENT_EVIDENCE_CORRECT = "insufficient_evidence_correct"
PDF_EXPECTED_GAP = "expected_gap"
PDF_AMBIGUOUS_CASE = "ambiguous_case"
PDF_PACK_REGRESSION = "pack_regression"

PDF_FAILURE_CLASSES = (
    PDF_NO_FAILURE, PDF_RETRIEVAL_MISS, PDF_WRONG_SOURCE_RANKED,
    PDF_EXPECTED_SOURCE_LOW_RANK, PDF_EXPECTED_CHUNK_LOW_RANK,
    PDF_OFF_TOPIC_NEIGHBOUR, PDF_DUPLICATE_CHUNK_INTERFERENCE,
    PDF_CHUNK_BOUNDARY_FAILURE, PDF_PAGE_LINEAGE_FAILURE, PDF_SELECTION_DROP,
    PDF_CITATION_DROP, PDF_INSUFFICIENT_EVIDENCE_CORRECT, PDF_EXPECTED_GAP,
    PDF_AMBIGUOUS_CASE, PDF_PACK_REGRESSION,
)


# -- eval case format ---------------------------------------------------------

@dataclass(frozen=True)
class PdfImportRetrievalCase:
    """One imported-PDF retrieval probe: a query plus its full expectations.

    A case declares what should be retrieved/selected/cited and what must never
    be, across all three stages. Source tokens match on exact ``source_id`` or
    case-insensitive substring of the source name (the same friendly convention
    the rest of the harness uses). Chunk ids match after stripping a ``src:``
    citation prefix.

    * ``pack_id`` — the imported pack under evaluation (label only).
    * ``expected_source_ids`` / ``expected_chunk_ids`` — what *should* surface.
    * ``expected_page_ranges`` — ``[[start, end], ...]`` page ranges that the
      expected content should carry through (page-lineage check).
    * ``forbidden_source_ids`` / ``forbidden_chunk_ids`` — presence is bleed.
    * ``expected_terms`` / ``forbidden_terms`` — on-topic / off-topic markers.
    * ``expected_answer_mode`` — ``"grounded"`` or ``"refusal"`` (``""`` = any).
    * ``expected_citation_source_ids`` / ``expected_citation_chunk_ids`` — what
      the final answer should cite.
    * ``pdf_only`` — the answer exists *only* in imported PDF content (drives the
      newly-answerable comparison signal).
    * ``unrelated`` — the answer must **not** use imported PDF content (isolation
      probe); list the pack's source in ``forbidden_source_ids``.
    * ``expected_gap`` — no source should support the query (insufficient but
      correct).
    * ``ambiguous`` — multiple sources legitimately compete (no single truth).
    * ``minimum_hit_k`` — expected source/chunk must appear within this rank.
    """

    case_id: str
    query: str
    pack_id: str = ""
    expected_source_ids: List[str] = field(default_factory=list)
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_page_ranges: List[List[int]] = field(default_factory=list)
    forbidden_source_ids: List[str] = field(default_factory=list)
    forbidden_chunk_ids: List[str] = field(default_factory=list)
    expected_terms: List[str] = field(default_factory=list)
    forbidden_terms: List[str] = field(default_factory=list)
    expected_answer_mode: str = ""
    expected_citation_source_ids: List[str] = field(default_factory=list)
    expected_citation_chunk_ids: List[str] = field(default_factory=list)
    pdf_only: bool = False
    unrelated: bool = False
    expected_gap: bool = False
    ambiguous: bool = False
    minimum_hit_k: int = 5
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def has_expected(self) -> bool:
        """Whether the case declares any expected source or chunk id."""
        return bool(self.expected_source_ids or self.expected_chunk_ids)

    @property
    def expects_pdf_content(self) -> bool:
        """Whether a correct answer should be grounded in imported PDF content."""
        return (self.has_expected and not self.unrelated
                and not self.expected_gap)

    @classmethod
    def from_dict(cls, data: dict) -> "PdfImportRetrievalCase":
        return cls(
            case_id=str(data["case_id"]),
            query=str(data["query"]),
            pack_id=str(data.get("pack_id", "")),
            expected_source_ids=list(data.get("expected_source_ids") or []),
            expected_chunk_ids=list(data.get("expected_chunk_ids") or []),
            expected_page_ranges=[list(r) for r in
                                  (data.get("expected_page_ranges") or [])],
            forbidden_source_ids=list(data.get("forbidden_source_ids") or []),
            forbidden_chunk_ids=list(data.get("forbidden_chunk_ids") or []),
            expected_terms=list(data.get("expected_terms") or []),
            forbidden_terms=list(data.get("forbidden_terms") or []),
            expected_answer_mode=str(data.get("expected_answer_mode", "")),
            expected_citation_source_ids=list(
                data.get("expected_citation_source_ids") or []),
            expected_citation_chunk_ids=list(
                data.get("expected_citation_chunk_ids") or []),
            pdf_only=bool(data.get("pdf_only", False)),
            unrelated=bool(data.get("unrelated", False)),
            expected_gap=bool(data.get("expected_gap", False)),
            ambiguous=bool(data.get("ambiguous", False)),
            minimum_hit_k=int(data.get("minimum_hit_k", 5)),
            notes=str(data.get("notes", "")),
            tags=list(data.get("tags") or []),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_pdf_import_cases(path: str | Path) -> List[PdfImportRetrievalCase]:
    """Load imported-PDF eval cases from JSONL (``#`` lines are comments)."""
    cases: List[PdfImportRetrievalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(PdfImportRetrievalCase.from_dict(json.loads(line)))
    return cases


# -- stage items / traces -----------------------------------------------------

@dataclass(frozen=True)
class PdfStageItem:
    """One observed item at a stage, carrying every lineage signal available."""

    stage: str
    chunk_id: str
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    pack_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    rank: Optional[int] = None
    score: Optional[float] = None
    selection_reason: str = ""
    text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PdfStageTrace:
    """The ordered items observed at one stage (raw / selected / final)."""

    stage: str
    items: List[PdfStageItem] = field(default_factory=list)

    @property
    def chunk_ids(self) -> List[str]:
        return [it.chunk_id for it in self.items if it.chunk_id]

    @property
    def source_names(self) -> List[str]:
        out: List[str] = []
        for it in self.items:
            name = it.source_name or ""
            if name and name not in out:
                out.append(name)
        return out

    def to_dict(self) -> dict:
        return {"stage": self.stage,
                "items": [it.to_dict() for it in self.items]}


# -- metric blocks ------------------------------------------------------------

@dataclass(frozen=True)
class PdfRetrievalMetrics:
    """Raw-retrieval quality metrics for one case (``None`` when undefined)."""

    retrieved_chunk_count: int
    hit_at_1: Optional[bool]
    hit_at_3: Optional[bool]
    hit_at_5: Optional[bool]
    expected_source_recall: Optional[float]
    expected_chunk_recall: Optional[float]
    expected_page_recall: Optional[float]
    wrong_source_rate: Optional[float]
    forbidden_source_hit_count: int
    forbidden_chunk_hit_count: int
    off_topic_inclusion_rate: float
    duplicate_chunk_rate: float
    near_duplicate_chunk_rate: float
    missing_expected_source_count: int
    missing_expected_chunk_count: int
    rank_of_first_expected_source: Optional[int]
    rank_of_first_expected_chunk: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PdfSelectedMetrics:
    """Selected-evidence (post-relevance-gate) metrics for one case."""

    selected_chunk_count: int
    selected_expected_source_recall: Optional[float]
    selected_expected_chunk_recall: Optional[float]
    selected_wrong_source_rate: Optional[float]
    selected_forbidden_source_count: int
    selected_off_topic_inclusion_rate: float
    selected_page_lineage_accuracy: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PdfCitationMetrics:
    """Final report/chat citation metrics for one case."""

    cited_chunk_count: int
    cited_expected_source_recall: Optional[float]
    cited_expected_chunk_recall: Optional[float]
    cited_wrong_source_rate: Optional[float]
    cited_forbidden_source_count: int
    citation_page_lineage_accuracy: Optional[float]
    uncited_factual_claim_count: int
    unsupported_citation_count: int
    citation_set_matches_selected_evidence: bool
    final_answer_grounded: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PdfImportCaseResult:
    """The full three-stage outcome for one imported-PDF eval case."""

    case_id: str
    query: str
    pack_id: str
    passed: bool
    reasons: List[str]
    classification: str
    answer_mode: str
    refused: bool
    raw: PdfStageTrace
    selected: PdfStageTrace
    final: PdfStageTrace
    retrieval: PdfRetrievalMetrics
    selected_metrics: PdfSelectedMetrics
    citation: PdfCitationMetrics
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "pack_id": self.pack_id,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "classification": self.classification,
            "answer_mode": self.answer_mode,
            "refused": self.refused,
            "raw": self.raw.to_dict(),
            "selected": self.selected.to_dict(),
            "final": self.final.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "selected_metrics": self.selected_metrics.to_dict(),
            "citation": self.citation.to_dict(),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class PdfImportEvalSummary:
    """Aggregate tallies across all imported-PDF eval cases."""

    case_count: int
    pass_count: int
    fail_count: int
    forbidden_source_hit_total: int
    forbidden_chunk_hit_total: int
    cited_forbidden_source_total: int
    uncited_factual_claim_total: int
    unsupported_citation_total: int
    expected_source_recall: Optional[float]
    expected_chunk_recall: Optional[float]
    expected_page_recall: Optional[float]
    cited_expected_source_recall: Optional[float]
    classification_counts: Dict[str, int]
    forbidden_bleed_reproduced: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PdfImportComparison:
    """Pure baseline-vs-pack diff across the same cases (no re-run)."""

    case_count: int
    cases_improved: int
    cases_unchanged: int
    cases_regressed: int
    new_wrong_source_hits: int
    new_forbidden_source_hits: int
    newly_answerable_cases: int
    unrelated_cases_affected: int
    improved_case_ids: List[str]
    regressed_case_ids: List[str]
    unrelated_affected_case_ids: List[str]

    @property
    def pack_helps(self) -> bool:
        """Whether the pack improved at least one case and regressed none."""
        return self.cases_improved > 0 and self.cases_regressed == 0

    def to_dict(self) -> dict:
        return asdict(self)


# -- helpers ------------------------------------------------------------------

def _parse_pages(section: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse a candidate ``section`` into a (page_start, page_end) pair.

    Understands the importer's two forms: ``"page N"`` -> ``(N, N)`` and
    ``"pages N-M"`` -> ``(N, M)``. Anything else yields ``(None, None)``.
    """
    if not section:
        return (None, None)
    nums = [int(n) for n in re.findall(r"\d+", section)]
    if not nums:
        return (None, None)
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (nums[0], nums[1])


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _pdf_source_token_matches(token: str, source_id: Optional[str],
                              source_name: Optional[str]) -> bool:
    """Whether a source token matches a (source_id, source_name) pair."""
    if not token:
        return False
    if source_id and token == source_id:
        return True
    name = (source_name or "").lower()
    return bool(name) and token.lower() in name


def _item_matches_any_source(item: PdfStageItem, tokens: List[str]) -> bool:
    return any(_pdf_source_token_matches(tok, item.source_id, item.source_name)
               for tok in tokens)


def _pages_overlap(a: Tuple[Optional[int], Optional[int]],
                   rng: List[int]) -> bool:
    """Whether item pages ``a`` overlap an expected ``[start, end]`` range."""
    ps, pe = a
    if ps is None or pe is None or len(rng) < 2:
        return False
    return ps <= rng[1] and rng[0] <= pe


def _recall(expected: List[str], hit_fn) -> Optional[float]:
    """Fraction of ``expected`` tokens satisfied by ``hit_fn(token)``."""
    if not expected:
        return None
    hits = sum(1 for tok in expected if hit_fn(tok))
    return hits / len(expected)


# -- stage construction (read-only) -------------------------------------------

def _pdf_raw_items(candidates: List[dict],
                   pack_label: str) -> List[PdfStageItem]:
    """Build raw-stage items from ``query_knowledge`` candidate dicts."""
    items: List[PdfStageItem] = []
    for cand in candidates:
        ps, pe = _parse_pages(cand.get("section"))
        items.append(PdfStageItem(
            stage="raw",
            chunk_id=_norm_chunk_id(str(cand.get("chunk_id") or "")),
            source_id=str(cand.get("source_id") or "") or None,
            source_name=cand.get("source_name"),
            pack_id=pack_label or None,
            page_start=ps,
            page_end=pe,
            rank=cand.get("rank"),
            score=cand.get("activation"),
            selection_reason="retrieved",
            text=cand.get("text", "") or "",
        ))
    return items


def _lineage_map(raw_items: List[PdfStageItem]) -> Dict[str, PdfStageItem]:
    """Map chunk_id -> raw item so later stages can recover source/page lineage."""
    out: Dict[str, PdfStageItem] = {}
    for it in raw_items:
        if it.chunk_id and it.chunk_id not in out:
            out[it.chunk_id] = it
    return out


def _pdf_selected_items(evidence, lineage: Dict[str, PdfStageItem],
                        pack_label: str) -> List[PdfStageItem]:
    """Build selected-stage items from grounding-package evidence."""
    items: List[PdfStageItem] = []
    for ev in evidence:
        if getattr(ev, "kind", "") != "knowledge":
            continue
        chunk_id = _norm_chunk_id(str(getattr(ev, "citation_id", "") or ""))
        src = lineage.get(chunk_id)
        items.append(PdfStageItem(
            stage="selected",
            chunk_id=chunk_id,
            source_id=(src.source_id if src else None),
            source_name=getattr(ev, "source_name", None)
            or (src.source_name if src else None),
            pack_id=pack_label or None,
            page_start=(src.page_start if src else None),
            page_end=(src.page_end if src else None),
            rank=(src.rank if src else None),
            score=(src.score if src else None),
            selection_reason="passed_relevance_gate",
            text=getattr(ev, "text", "") or "",
        ))
    return items


def _pdf_final_items(spans, lineage: Dict[str, PdfStageItem],
                     pack_label: str) -> List[PdfStageItem]:
    """Build final-stage items from composed report spans."""
    items: List[PdfStageItem] = []
    for span in spans:
        citation = str(getattr(span, "citation_id", "") or "")
        chunk_id = _norm_chunk_id(citation)
        src = lineage.get(chunk_id)
        items.append(PdfStageItem(
            stage="final",
            chunk_id=chunk_id,
            source_id=(src.source_id if src else None),
            source_name=getattr(span, "source_name", None)
            or (src.source_name if src else None),
            pack_id=pack_label or None,
            page_start=(src.page_start if src else None),
            page_end=(src.page_end if src else None),
            rank=(src.rank if src else None),
            score=(src.score if src else None),
            selection_reason="cited" if citation else "uncited_claim",
            text=getattr(span, "text", "") or "",
        ))
    return items


# -- metric computation -------------------------------------------------------

def _pdf_retrieval_metrics(case: PdfImportRetrievalCase,
                           items: List[PdfStageItem]) -> PdfRetrievalMetrics:
    norm_expected_chunks = {_norm_chunk_id(c) for c in case.expected_chunk_ids}
    norm_forbidden_chunks = {_norm_chunk_id(c) for c in case.forbidden_chunk_ids}

    # First-hit ranks.
    first_src_rank: Optional[int] = None
    first_chunk_rank: Optional[int] = None
    for it in items:
        if (first_src_rank is None
                and _item_matches_any_source(it, case.expected_source_ids)):
            first_src_rank = it.rank
        if (first_chunk_rank is None and it.chunk_id in norm_expected_chunks):
            first_chunk_rank = it.rank

    def _hit_at(k: int) -> Optional[bool]:
        if not case.expected_source_ids:
            return None
        return first_src_rank is not None and first_src_rank <= k

    src_recall = _recall(
        case.expected_source_ids,
        lambda tok: any(_pdf_source_token_matches(
            tok, it.source_id, it.source_name) for it in items))
    chunk_recall = _recall(
        case.expected_chunk_ids,
        lambda tok: _norm_chunk_id(tok) in {it.chunk_id for it in items})
    page_recall: Optional[float]
    if case.expected_page_ranges:
        covered = sum(1 for rng in case.expected_page_ranges
                      if any(_pages_overlap((it.page_start, it.page_end), rng)
                             for it in items))
        page_recall = covered / len(case.expected_page_ranges)
    else:
        page_recall = None

    forbidden_src = sum(
        1 for it in items
        if _item_matches_any_source(it, case.forbidden_source_ids))
    forbidden_chunk = sum(1 for it in items
                          if it.chunk_id in norm_forbidden_chunks)

    # Wrong-source rate: retrieved items whose source is neither expected nor an
    # allowed on-topic case (ambiguous cases tolerate competing sources).
    wrong_rate: Optional[float]
    if items and case.expected_source_ids and not case.ambiguous:
        wrong = sum(1 for it in items
                    if not _item_matches_any_source(
                        it, case.expected_source_ids))
        wrong_rate = wrong / len(items)
    else:
        wrong_rate = None

    if case.forbidden_terms and items:
        off = sum(1 for it in items
                  if any(term.lower() in (it.text or "").lower()
                         for term in case.forbidden_terms))
        off_rate = off / len(items)
    else:
        off_rate = 0.0

    # Duplicate / near-duplicate detection over retrieved chunk text.
    dup = 0
    near = 0
    seen_text: List[str] = []
    for it in items:
        is_dup = any(it.text == prev for prev in seen_text)
        if is_dup:
            dup += 1
        elif any(_jaccard(it.text, prev) >= _NEAR_DUP_JACCARD
                 for prev in seen_text):
            near += 1
        seen_text.append(it.text)
    n = len(items)

    missing_src = sum(
        1 for tok in case.expected_source_ids
        if not any(_pdf_source_token_matches(tok, it.source_id, it.source_name)
                   for it in items))
    missing_chunk = sum(
        1 for c in norm_expected_chunks
        if c not in {it.chunk_id for it in items})

    return PdfRetrievalMetrics(
        retrieved_chunk_count=n,
        hit_at_1=_hit_at(1),
        hit_at_3=_hit_at(3),
        hit_at_5=_hit_at(5),
        expected_source_recall=src_recall,
        expected_chunk_recall=chunk_recall,
        expected_page_recall=page_recall,
        wrong_source_rate=wrong_rate,
        forbidden_source_hit_count=forbidden_src,
        forbidden_chunk_hit_count=forbidden_chunk,
        off_topic_inclusion_rate=off_rate,
        duplicate_chunk_rate=(dup / n if n else 0.0),
        near_duplicate_chunk_rate=(near / n if n else 0.0),
        missing_expected_source_count=missing_src,
        missing_expected_chunk_count=missing_chunk,
        rank_of_first_expected_source=first_src_rank,
        rank_of_first_expected_chunk=first_chunk_rank,
    )


def _pdf_selected_metrics(case: PdfImportRetrievalCase,
                          items: List[PdfStageItem]) -> PdfSelectedMetrics:
    norm_expected_chunks = {_norm_chunk_id(c) for c in case.expected_chunk_ids}
    src_recall = _recall(
        case.expected_source_ids,
        lambda tok: any(_pdf_source_token_matches(
            tok, it.source_id, it.source_name) for it in items))
    chunk_recall = _recall(
        case.expected_chunk_ids,
        lambda tok: _norm_chunk_id(tok) in {it.chunk_id for it in items})
    forbidden_src = sum(
        1 for it in items
        if _item_matches_any_source(it, case.forbidden_source_ids))
    if items and case.expected_source_ids and not case.ambiguous:
        wrong = sum(1 for it in items
                    if not _item_matches_any_source(
                        it, case.expected_source_ids))
        wrong_rate: Optional[float] = wrong / len(items)
    else:
        wrong_rate = None
    if case.forbidden_terms and items:
        off = sum(1 for it in items
                  if any(term.lower() in (it.text or "").lower()
                         for term in case.forbidden_terms))
        off_rate = off / len(items)
    else:
        off_rate = 0.0

    # Page-lineage accuracy over selected items that match an expected chunk.
    page_acc: Optional[float]
    if case.expected_page_ranges and norm_expected_chunks:
        matched = [it for it in items if it.chunk_id in norm_expected_chunks]
        if matched:
            ok = sum(1 for it in matched
                     if any(_pages_overlap((it.page_start, it.page_end), rng)
                            for rng in case.expected_page_ranges))
            page_acc = ok / len(matched)
        else:
            page_acc = None
    else:
        page_acc = None

    return PdfSelectedMetrics(
        selected_chunk_count=len(items),
        selected_expected_source_recall=src_recall,
        selected_expected_chunk_recall=chunk_recall,
        selected_wrong_source_rate=wrong_rate,
        selected_forbidden_source_count=forbidden_src,
        selected_off_topic_inclusion_rate=off_rate,
        selected_page_lineage_accuracy=page_acc,
    )


def _pdf_citation_metrics(case: PdfImportRetrievalCase,
                          final_items: List[PdfStageItem],
                          selected_items: List[PdfStageItem],
                          answer) -> PdfCitationMetrics:
    cited = [it for it in final_items if it.selection_reason == "cited"]
    expected_cite_sources = (case.expected_citation_source_ids
                             or case.expected_source_ids)
    expected_cite_chunks = (case.expected_citation_chunk_ids
                            or case.expected_chunk_ids)
    src_recall = _recall(
        expected_cite_sources,
        lambda tok: any(_pdf_source_token_matches(
            tok, it.source_id, it.source_name) for it in cited))
    chunk_recall = _recall(
        expected_cite_chunks,
        lambda tok: _norm_chunk_id(tok) in {it.chunk_id for it in cited})
    forbidden_src = sum(
        1 for it in cited
        if _item_matches_any_source(it, case.forbidden_source_ids))
    if cited and case.expected_source_ids and not case.ambiguous:
        wrong = sum(1 for it in cited
                    if not _item_matches_any_source(
                        it, case.expected_source_ids))
        wrong_rate: Optional[float] = wrong / len(cited)
    else:
        wrong_rate = None

    norm_expected_chunks = {_norm_chunk_id(c) for c in expected_cite_chunks}
    page_acc: Optional[float]
    if case.expected_page_ranges and norm_expected_chunks:
        matched = [it for it in cited if it.chunk_id in norm_expected_chunks]
        if matched:
            ok = sum(1 for it in matched
                     if any(_pages_overlap((it.page_start, it.page_end), rng)
                            for rng in case.expected_page_ranges))
            page_acc = ok / len(matched)
        else:
            page_acc = None
    else:
        page_acc = None

    uncited_claims = sum(
        1 for it in final_items
        if it.selection_reason == "uncited_claim" and (it.text or "").strip())

    selected_chunk_set = {it.chunk_id for it in selected_items}
    citation_chunk_set = {it.chunk_id for it in cited if it.chunk_id}
    unsupported = sum(1 for cid in citation_chunk_set
                      if cid not in selected_chunk_set)
    citation_matches = citation_chunk_set.issubset(selected_chunk_set)

    refused = bool(getattr(answer, "refused", False))
    grounded = (not refused) and bool(getattr(answer, "citations", None))

    return PdfCitationMetrics(
        cited_chunk_count=len(cited),
        cited_expected_source_recall=src_recall,
        cited_expected_chunk_recall=chunk_recall,
        cited_wrong_source_rate=wrong_rate,
        cited_forbidden_source_count=forbidden_src,
        citation_page_lineage_accuracy=page_acc,
        uncited_factual_claim_count=uncited_claims,
        unsupported_citation_count=unsupported,
        citation_set_matches_selected_evidence=citation_matches,
        final_answer_grounded=grounded,
    )


# -- verdict + classification -------------------------------------------------

def pdf_case_verdict(case: PdfImportRetrievalCase,
                     retr: PdfRetrievalMetrics,
                     sel: PdfSelectedMetrics,
                     cit: PdfCitationMetrics) -> Tuple[bool, List[str]]:
    """Deterministic pass/fail with human-readable reasons.

    Hard gates (any failure fails the case): no forbidden source/chunk in the
    *selected evidence* or *final citations*, no uncited factual claims, no
    unsupported citations, citations are a subset of selected evidence.
    Expected-content cases must also retrieve their expected source within
    ``minimum_hit_k`` and recall it fully; gap/unrelated cases must stay
    ungrounded.

    Raw retrieval is deliberately *not* gated on forbidden presence: the
    retrieval backends apply no score threshold and always return the top-k
    candidates, so a forbidden/unrelated source in the same corpus is expected
    to appear at the raw stage. Isolation is therefore measured where the system
    can actually enforce it — the relevance-gated selected stage and the cited
    answer. Raw forbidden presence is still recorded (and drives the
    wrong-source classification of *failing* expected-content cases).
    """
    reasons: List[str] = []

    if sel.selected_forbidden_source_count:
        reasons.append("selected evidence included a forbidden source")
    if cit.cited_forbidden_source_count:
        reasons.append("final answer cited a forbidden source")
    if cit.uncited_factual_claim_count:
        reasons.append("final answer made an uncited factual claim")
    if cit.unsupported_citation_count:
        reasons.append("final answer cited evidence not in selected set")
    if not cit.citation_set_matches_selected_evidence:
        reasons.append("citation set is not a subset of selected evidence")

    if case.expects_pdf_content:
        if (retr.expected_source_recall is None
                or retr.expected_source_recall < MIN_EXPECTED_SOURCE_RECALL):
            reasons.append("expected source was not fully retrieved")
        if (retr.rank_of_first_expected_source is None
                or retr.rank_of_first_expected_source > case.minimum_hit_k):
            reasons.append(
                f"expected source not within top-{case.minimum_hit_k}")
        if case.expected_chunk_ids and (retr.expected_chunk_recall or 0.0) <= 0:
            reasons.append("expected chunk was not retrieved")
        if (case.expected_page_ranges
                and (retr.expected_page_recall or 0.0) < 1.0):
            reasons.append("expected page lineage not preserved in retrieval")
        if case.expected_citation_chunk_ids and (
                cit.cited_expected_chunk_recall or 0.0) <= 0:
            reasons.append("expected chunk was not cited")
    elif case.expected_gap:
        if cit.final_answer_grounded:
            reasons.append("gap case produced a grounded answer")
    elif case.unrelated:
        if sel.selected_chunk_count:
            reasons.append("unrelated query selected imported PDF evidence")

    return (not reasons, reasons)


def classify_pdf_case(case: PdfImportRetrievalCase,
                      retr: PdfRetrievalMetrics,
                      sel: PdfSelectedMetrics,
                      cit: PdfCitationMetrics,
                      passed: bool) -> str:
    """Assign one diagnostic classification (priority cascade)."""
    if case.expected_gap:
        return (PDF_INSUFFICIENT_EVIDENCE_CORRECT if passed
                else PDF_EXPECTED_GAP)
    if case.unrelated:
        if sel.selected_chunk_count or cit.cited_chunk_count:
            return PDF_OFF_TOPIC_NEIGHBOUR
        return PDF_NO_FAILURE
    if passed:
        return PDF_NO_FAILURE

    # Failing expected-content cases: localise the first broken stage.
    if (retr.forbidden_source_hit_count or sel.selected_forbidden_source_count
            or cit.cited_forbidden_source_count):
        return PDF_WRONG_SOURCE_RANKED
    if case.ambiguous:
        return PDF_AMBIGUOUS_CASE
    if case.expects_pdf_content:
        if retr.rank_of_first_expected_source is None:
            return PDF_RETRIEVAL_MISS
        if retr.rank_of_first_expected_source > case.minimum_hit_k:
            return PDF_EXPECTED_SOURCE_LOW_RANK
        if case.expected_chunk_ids and (retr.expected_chunk_recall or 0.0) <= 0:
            if retr.near_duplicate_chunk_rate or retr.duplicate_chunk_rate:
                return PDF_DUPLICATE_CHUNK_INTERFERENCE
            return PDF_EXPECTED_CHUNK_LOW_RANK
        if (case.expected_page_ranges
                and (retr.expected_page_recall or 0.0) < 1.0):
            return PDF_CHUNK_BOUNDARY_FAILURE
        if (sel.selected_expected_source_recall is not None
                and sel.selected_expected_source_recall <= 0):
            return PDF_SELECTION_DROP
        if (cit.cited_expected_source_recall is not None
                and cit.cited_expected_source_recall <= 0):
            return PDF_CITATION_DROP
        if (sel.selected_page_lineage_accuracy is not None
                and sel.selected_page_lineage_accuracy < 1.0) or (
                cit.citation_page_lineage_accuracy is not None
                and cit.citation_page_lineage_accuracy < 1.0):
            return PDF_PAGE_LINEAGE_FAILURE
    if retr.off_topic_inclusion_rate > 0:
        return PDF_OFF_TOPIC_NEIGHBOUR
    return PDF_WRONG_SOURCE_RANKED


# -- per-case evaluation (read-only) ------------------------------------------

def evaluate_pdf_import_case(service: WorkbenchService,
                             case: PdfImportRetrievalCase, *,
                             pack_label: str = "") -> PdfImportCaseResult:
    """Trace one case through raw -> selected -> final and score it.

    Read-only: three reads of the frozen path (``query_knowledge``,
    ``build_grounding_package``, ``answer_query`` with the consultant report
    composer), then pure measurement. Nothing is mutated.
    """
    # Lazy import mirrors :func:`probe_case`: the composer dependency stays out
    # of module import so the harness imports nothing that can compose or write.
    from slm.assistant_composer import ConsultantReportComposer

    label = pack_label or case.pack_id

    # STAGE 1 — raw retrieved candidates.
    audit = service.query_knowledge(case.query)
    raw_items = _pdf_raw_items(list(audit.candidates), label)
    lineage = _lineage_map(raw_items)

    # STAGE 2 — evidence selected past the relevance/sufficiency gate.
    package = service.build_grounding_package(case.query)
    selected_items = _pdf_selected_items(package.evidence, lineage, label)

    # STAGE 3 — final cited report spans.
    result = service.answer_query(
        case.query, composer=ConsultantReportComposer())
    final_items = _pdf_final_items(result.answer.spans, lineage, label)

    retr = _pdf_retrieval_metrics(case, raw_items)
    sel = _pdf_selected_metrics(case, selected_items)
    cit = _pdf_citation_metrics(case, final_items, selected_items, result.answer)

    passed, reasons = pdf_case_verdict(case, retr, sel, cit)
    classification = classify_pdf_case(case, retr, sel, cit, passed)

    return PdfImportCaseResult(
        case_id=case.case_id or case.query[:40],
        query=case.query,
        pack_id=label,
        passed=passed,
        reasons=reasons,
        classification=classification,
        answer_mode=result.answer.mode.value,
        refused=bool(result.refused),
        raw=PdfStageTrace("raw", raw_items),
        selected=PdfStageTrace("selected", selected_items),
        final=PdfStageTrace("final", final_items),
        retrieval=retr,
        selected_metrics=sel,
        citation=cit,
        tags=list(case.tags),
    )


def run_pdf_import_eval(service: WorkbenchService,
                        cases: List[PdfImportRetrievalCase], *,
                        pack_label: str = "") -> List[PdfImportCaseResult]:
    """Evaluate every case against ``service`` (read-only)."""
    return [evaluate_pdf_import_case(service, c, pack_label=pack_label)
            for c in cases]


def summarize_pdf_import_eval(
        results: List[PdfImportCaseResult]) -> PdfImportEvalSummary:
    """Aggregate per-case results into imported-PDF eval tallies."""
    counts = {name: 0 for name in PDF_FAILURE_CLASSES}
    for r in results:
        counts[r.classification] = counts.get(r.classification, 0) + 1

    src_recall = _mean([r.retrieval.expected_source_recall for r in results
                        if r.retrieval.expected_source_recall is not None])
    chunk_recall = _mean([r.retrieval.expected_chunk_recall for r in results
                          if r.retrieval.expected_chunk_recall is not None])
    page_recall = _mean([r.retrieval.expected_page_recall for r in results
                         if r.retrieval.expected_page_recall is not None])
    cite_recall = _mean(
        [r.citation.cited_expected_source_recall for r in results
         if r.citation.cited_expected_source_recall is not None])

    forbidden_total = sum(r.retrieval.forbidden_source_hit_count
                          for r in results)
    return PdfImportEvalSummary(
        case_count=len(results),
        pass_count=sum(1 for r in results if r.passed),
        fail_count=sum(1 for r in results if not r.passed),
        forbidden_source_hit_total=forbidden_total,
        forbidden_chunk_hit_total=sum(
            r.retrieval.forbidden_chunk_hit_count for r in results),
        cited_forbidden_source_total=sum(
            r.citation.cited_forbidden_source_count for r in results),
        uncited_factual_claim_total=sum(
            r.citation.uncited_factual_claim_count for r in results),
        unsupported_citation_total=sum(
            r.citation.unsupported_citation_count for r in results),
        expected_source_recall=src_recall,
        expected_chunk_recall=chunk_recall,
        expected_page_recall=page_recall,
        cited_expected_source_recall=cite_recall,
        classification_counts=counts,
        forbidden_bleed_reproduced=any(
            r.selected_metrics.selected_forbidden_source_count
            or r.citation.cited_forbidden_source_count
            for r in results),
    )


# -- baseline-vs-pack comparison (pure) ---------------------------------------

def _case_signature(r: PdfImportCaseResult) -> dict:
    """A small comparable signature of one case outcome."""
    return {
        "expected_source_hit": bool(r.retrieval.hit_at_5),
        "grounded": r.citation.final_answer_grounded,
        "forbidden": (r.retrieval.forbidden_source_hit_count
                      + r.citation.cited_forbidden_source_count),
        "wrong": (r.retrieval.wrong_source_rate or 0.0),
        "cited_expected": (r.citation.cited_expected_source_recall or 0.0),
    }


def compare_pdf_import_eval(
        baseline: List[PdfImportCaseResult],
        with_pack: List[PdfImportCaseResult]) -> PdfImportComparison:
    """Diff two result sets (same cases, with vs without the pack).

    Pure: it re-runs nothing. Cases are matched by ``case_id``; a case present
    in only one set is ignored (the caller is expected to run identical cases).
    """
    base_by_id = {r.case_id: r for r in baseline}
    improved: List[str] = []
    regressed: List[str] = []
    unrelated_affected: List[str] = []
    unchanged = 0
    new_wrong = 0
    new_forbidden = 0
    newly_answerable = 0

    for wp in with_pack:
        base = base_by_id.get(wp.case_id)
        if base is None:
            continue
        b = _case_signature(base)
        w = _case_signature(wp)

        if w["forbidden"] > b["forbidden"]:
            new_forbidden += 1
        if w["wrong"] > b["wrong"] + 1e-9:
            new_wrong += 1

        # Unrelated cases must be unaffected by enabling the pack.
        is_unrelated = "unrelated" in wp.tags
        if is_unrelated and (
                wp.selected.chunk_ids or wp.final.chunk_ids):
            unrelated_affected.append(wp.case_id)

        gained = (
            (not b["expected_source_hit"] and w["expected_source_hit"])
            or (not b["grounded"] and w["grounded"])
            or (w["cited_expected"] > b["cited_expected"] + 1e-9))
        lost = (
            (b["expected_source_hit"] and not w["expected_source_hit"])
            or (b["grounded"] and not w["grounded"])
            or (w["forbidden"] > b["forbidden"])
            or (w["wrong"] > b["wrong"] + 1e-9))

        if not b["grounded"] and w["grounded"]:
            newly_answerable += 1

        if lost:
            regressed.append(wp.case_id)
        elif gained:
            improved.append(wp.case_id)
        else:
            unchanged += 1

    return PdfImportComparison(
        case_count=len(with_pack),
        cases_improved=len(improved),
        cases_unchanged=unchanged,
        cases_regressed=len(regressed),
        new_wrong_source_hits=new_wrong,
        new_forbidden_source_hits=new_forbidden,
        newly_answerable_cases=newly_answerable,
        unrelated_cases_affected=len(unrelated_affected),
        improved_case_ids=improved,
        regressed_case_ids=regressed,
        unrelated_affected_case_ids=unrelated_affected,
    )


# -- rendering / writing ------------------------------------------------------

def render_pdf_import_markdown(
        results: List[PdfImportCaseResult],
        summary: PdfImportEvalSummary, *,
        pack_label: str, backend_label: str,
        comparison: Optional[PdfImportComparison] = None) -> str:
    """Render the imported-PDF retrieval evaluation as Markdown (no side effects)."""
    lines: List[str] = []
    lines.append(f"# Imported PDF retrieval evaluation — {pack_label} "
                 f"({backend_label})")
    lines.append("")
    lines.append("Read-only, three-stage trace (raw retrieval -> selected "
                 "evidence -> final citations) of an approved imported PDF "
                 "pack. Changes no retrieval/ranking/chunking/grounding/"
                 "composer/memory behaviour. Recommendations are diagnostic "
                 "only.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- cases: {summary.case_count} "
                 f"(pass {summary.pass_count} / fail {summary.fail_count})")
    if summary.forbidden_bleed_reproduced:
        lines.append("- **forbidden bleed reproduced**: yes")
    else:
        lines.append("- **forbidden bleed reproduced**: no — zero forbidden "
                     "sources/chunks at raw, selected, or cited stages")
    lines.append(f"- expected-source recall: "
                 f"{_fmt_opt(summary.expected_source_recall)} | "
                 f"expected-chunk recall: "
                 f"{_fmt_opt(summary.expected_chunk_recall)} | "
                 f"expected-page recall: "
                 f"{_fmt_opt(summary.expected_page_recall)}")
    lines.append(f"- cited expected-source recall: "
                 f"{_fmt_opt(summary.cited_expected_source_recall)}")
    lines.append(f"- uncited factual claims: "
                 f"{summary.uncited_factual_claim_total} | "
                 f"unsupported citations: "
                 f"{summary.unsupported_citation_total}")
    lines.append("")
    if comparison is not None:
        lines.append("## Baseline vs pack")
        lines.append("")
        lines.append(f"- improved: {comparison.cases_improved} | "
                     f"unchanged: {comparison.cases_unchanged} | "
                     f"regressed: {comparison.cases_regressed}")
        lines.append(f"- newly answerable: "
                     f"{comparison.newly_answerable_cases}")
        lines.append(f"- new wrong-source hits: "
                     f"{comparison.new_wrong_source_hits} | "
                     f"new forbidden-source hits: "
                     f"{comparison.new_forbidden_source_hits}")
        lines.append(f"- unrelated cases affected: "
                     f"{comparison.unrelated_cases_affected}")
        lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| case | result | classification | hit@5 | src/chunk/page "
                 "recall | sel | cited | fbd | mode |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        hit5 = "—" if r.retrieval.hit_at_5 is None else (
            "y" if r.retrieval.hit_at_5 else "n")
        recall = (f"{_fmt_opt(r.retrieval.expected_source_recall)}/"
                  f"{_fmt_opt(r.retrieval.expected_chunk_recall)}/"
                  f"{_fmt_opt(r.retrieval.expected_page_recall)}")
        fbd = (r.retrieval.forbidden_source_hit_count
               + r.citation.cited_forbidden_source_count)
        lines.append(
            f"| {r.case_id} | {verdict} | {r.classification} | {hit5} | "
            f"{recall} | {r.selected_metrics.selected_chunk_count} | "
            f"{r.citation.cited_chunk_count} | {fbd} | {r.answer_mode} |")
    lines.append("")
    return "\n".join(lines)


def write_pdf_import_reports(
        results: List[PdfImportCaseResult],
        summary: PdfImportEvalSummary, *,
        md_path: str | Path, jsonl_path: str | Path,
        pack_label: str, backend_label: str,
        comparison: Optional[PdfImportComparison] = None) -> None:
    """Write the imported-PDF eval Markdown and JSONL reports (only when asked)."""
    md_path = Path(md_path)
    jsonl_path = Path(jsonl_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_pdf_import_markdown(
            results, summary, pack_label=pack_label,
            backend_label=backend_label, comparison=comparison),
        encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        payload = {"summary": summary.to_dict()}
        if comparison is not None:
            payload["comparison"] = comparison.to_dict()
        handle.write(json.dumps(payload) + "\n")
        for r in results:
            handle.write(json.dumps(r.to_dict()) + "\n")
