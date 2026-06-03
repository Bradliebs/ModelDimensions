"""v2.4 Relevance & Sufficiency Gate — between retrieval and grounding.

.. note::
   This module is now a thin **backward-compatible adapter**. The canonical
   implementation lives in two focused modules:

   * :mod:`retrieval.evidence_ranker` — scores and orders retrieved candidates
     by auditable positive/negative relevance signals (the *ranker*).
   * :mod:`retrieval.sufficiency_gate` — turns the ranker's strongest
     substantive score into an answerability verdict (the *gate*).

   Existing callers and tests import :func:`assess_relevance`,
   :class:`RelevanceReport`, the ``SufficiencyVerdict`` enum, the ``LABEL_*``
   constants, and the lexical helpers from here; those names are re-exported
   unchanged so the frozen behaviour and the value-sprint contract are
   preserved. New code should prefer the two modules above directly.

The frozen retrieve -> verify -> ground path is *recall-biased*: the
deterministic backend returns its top-k candidates and
:meth:`WorkbenchService.build_grounding_package` previously grounded an answer
whenever *any* memory or knowledge chunk came back. Retrieval presence is not
relevance, so weakly-related or out-of-domain chunks could become a confident
grounded answer (the dominant failure mode the v2.3 value sprint measured).

The gate is **additive and downgrade-only**. It can move a would-be GROUNDED
decision to a partial answer, a refusal, or an explicit conflict; it can
re-order evidence so the strongest substantive item leads; it can never invent
evidence, upgrade a refusal into a grounded answer, or relax any frozen
guarantee (verifier, citations, lifecycle, pack isolation, AnswerGuard).

The relevance signal is lexical token overlap — deliberately simple and
explainable, not semantic understanding. Every decision carries a per-check
trace, a one-line sufficiency reason, and the top rejected candidate, so the
query-assist audit, the value-sprint report, and the AnswerGuard chain can all
see *why* a verdict was reached and *what* was held back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from slm.assistant_composer import EvidenceItem

# -- canonical implementation (re-exported for backward compatibility) --------
from retrieval.evidence_ranker import (  # noqa: F401
    EvidenceAssessment,
    EvidenceRanker,
    RankingReport,
    content_tokens,
    has_near_miss_conflict,
    infer_out_of_domain,
    is_metadata_header,
    is_milestone_query,
    overlap_coefficient,
)
from retrieval.sufficiency_gate import (  # noqa: F401
    LABEL_CONFLICT_MASKED,
    LABEL_GROUNDED_IRRELEVANT,
    LABEL_GROUNDED_PARTIAL,
    LABEL_GROUNDED_RELEVANT,
    LABEL_METADATA_HEADER_LEAD,
    LABEL_OUT_OF_DOMAIN_FALSE_GROUNDING,
    LABEL_WEAK_SUPPORT,
    MODERATE_OVERLAP,
    STRONG_OVERLAP,
    WEAK_FLOOR,
    SufficiencyGate,
    SufficiencyVerdict,
)

# Shared singletons — both are stateless and deterministic, so one instance is
# safe to reuse across every call.
_RANKER = EvidenceRanker()
_GATE = SufficiencyGate()


@dataclass(frozen=True)
class RelevanceReport:
    """The gate's decision for one query plus its full reasoning trace.

    This is the combined view the rest of the system consumes: the
    :class:`~retrieval.sufficiency_gate.SufficiencyGate` verdict together with
    the :class:`~retrieval.evidence_ranker.RankingReport` ordering and the top
    rejected candidate.
    """

    verdict: SufficiencyVerdict
    label: str
    sufficiency_reason: str
    lead_citation_id: Optional[str]
    ordered_evidence: List[EvidenceItem]
    assessments: List[EvidenceAssessment]
    checks: Dict[str, object]
    limitation: Optional[str] = None
    top_rejected: Optional[Dict[str, object]] = None

    @property
    def can_ground(self) -> bool:
        """Whether this verdict permits a (full or partial) grounded answer."""
        return self.verdict in (SufficiencyVerdict.RELEVANT,
                                SufficiencyVerdict.PARTIAL)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "label": self.label,
            "sufficiency_reason": self.sufficiency_reason,
            "lead_citation_id": self.lead_citation_id,
            "limitation": self.limitation,
            "checks": dict(self.checks),
            "assessments": [a.to_dict() for a in self.assessments],
            "top_rejected": dict(self.top_rejected) if self.top_rejected else None,
        }


def assess_relevance(query: str, evidence: List[EvidenceItem], *,
                     route: str,
                     conflict_signal: bool = False,
                     stale_ids: Optional[set] = None) -> RelevanceReport:
    """Decide whether retrieved evidence is sufficient to ground ``query``.

    Orchestrates the two canonical stages: the
    :class:`~retrieval.evidence_ranker.EvidenceRanker` scores and orders the
    candidates, then the :class:`~retrieval.sufficiency_gate.SufficiencyGate`
    maps the strongest substantive score onto an answerability verdict. The two
    are folded back into a single :class:`RelevanceReport` for callers.

    ``conflict_signal`` is set when the memory verifier rejected a near-miss
    candidate for this query: a contradiction must surface as CONFLICT before
    ordinary grounding, and the knowledge route must not mask it. ``stale_ids``
    is the set of citation ids backed by a stale source (a penalty, never a
    disqualifier — frozen staleness labelling is preserved).
    """
    ranking = _RANKER.rank(query, evidence, route=route,
                           conflict_signal=conflict_signal,
                           stale_ids=stale_ids)

    decision = _GATE.classify(
        best_score=ranking.best_score,
        best_domain_match=ranking.best_domain_match,
        has_best=ranking.best is not None,
        out_marker=ranking.out_marker,
        conflict_signal=conflict_signal,
        had_evidence=bool(evidence),
        had_header=ranking.had_header,
    )

    lead = (ranking.best.citation_id
            if (ranking.best is not None and decision.can_ground) else None)

    return RelevanceReport(
        verdict=decision.verdict,
        label=decision.label,
        sufficiency_reason=decision.reason,
        lead_citation_id=lead,
        ordered_evidence=ranking.ordered_evidence,
        assessments=ranking.assessments,
        checks=ranking.checks,
        limitation=decision.limitation,
        top_rejected=ranking.top_rejected,
    )
