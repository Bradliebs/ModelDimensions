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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

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
