"""v2.4 SufficiencyGate — turn a ranked-evidence score into an answerability verdict.

This is the second half of the v2.4 relevance & sufficiency layer (the first
half is :mod:`retrieval.evidence_ranker`, which *scores and orders* candidates).
The gate answers one narrow question: given the strongest substantive evidence
the ranker found, *is it good enough to ground this query, and how*?

Everything here is a **pure, deterministic, offline** function of the inputs:
no I/O, no model, no randomness. The gate is **downgrade-only** by construction
— it can return RELEVANT (full grounding), PARTIAL (limited answer), WEAK_MATCH
or NO_SUPPORT (refuse / propose a gap), or CONFLICT (a near-miss must surface
first). It can never invent evidence, upgrade a refusal into a grounded answer,
or relax any frozen guarantee (verifier, citations, lifecycle, pack isolation,
AnswerGuard). The thresholds operate on a length-robust overlap coefficient, so
the same query/evidence pair always yields the same verdict.

The verdict→effect contract (enforced upstream in
``WorkbenchService.build_grounding_package``):

* ``RELEVANT``   — strong substantive match → a fully grounded answer.
* ``PARTIAL``    — moderate match → a grounded answer with an explicit limit.
* ``WEAK_MATCH`` — too weak to lead → no confident answer (refuse / model prior).
* ``NO_SUPPORT`` — unrelated or out-of-domain → refuse or propose a pack gap.
* ``CONFLICT``   — a rejected near-miss contradicts the query → surface first.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SufficiencyVerdict(str, Enum):
    """How well the retrieved evidence supports answering the query."""

    RELEVANT = "relevant"        # strong substantive support -> full grounding
    PARTIAL = "partial"          # some support -> grounded with explicit limit
    WEAK_MATCH = "weak_match"    # too weak -> cannot ground a full answer
    CONFLICT = "conflict"        # a near-miss contradiction must surface first
    NO_SUPPORT = "no_support"    # nothing relevant -> refuse or propose a gap


# -- value-sprint diagnostic labels (what the gate prevented or allowed) ------

LABEL_GROUNDED_RELEVANT = "grounded_relevant"
LABEL_GROUNDED_PARTIAL = "grounded_partial"
LABEL_GROUNDED_IRRELEVANT = "grounded_irrelevant"
LABEL_WEAK_SUPPORT = "weak_support"
LABEL_CONFLICT_MASKED = "conflict_masked"
LABEL_METADATA_HEADER_LEAD = "metadata_header_lead"
LABEL_OUT_OF_DOMAIN_FALSE_GROUNDING = "out_of_domain_false_grounding"

# -- thresholds (overlap coefficient, length-robust) --------------------------

STRONG_OVERLAP = 0.5      # >= -> RELEVANT eligible
MODERATE_OVERLAP = 0.28   # >= -> PARTIAL eligible
WEAK_FLOOR = 0.12         # >= -> WEAK_MATCH; below -> NO_SUPPORT


@dataclass(frozen=True)
class SufficiencyDecision:
    """The gate's verdict for one query plus a one-line auditable reason."""

    verdict: SufficiencyVerdict
    label: str
    reason: str
    limitation: Optional[str] = None

    @property
    def can_ground(self) -> bool:
        """Whether this verdict permits a (full or partial) grounded answer."""
        return self.verdict in (SufficiencyVerdict.RELEVANT,
                                SufficiencyVerdict.PARTIAL)


class SufficiencyGate:
    """Map the ranker's best substantive score onto a sufficiency verdict.

    Thresholds are injectable so a caller could tune strictness, but the
    defaults reproduce the frozen v2.4 behaviour exactly. The gate holds no
    state and performs no I/O; ``classify`` is a pure function of its arguments.
    """

    def __init__(self, *, strong: float = STRONG_OVERLAP,
                 moderate: float = MODERATE_OVERLAP,
                 weak_floor: float = WEAK_FLOOR):
        self.strong = strong
        self.moderate = moderate
        self.weak_floor = weak_floor

    def classify(self, *, best_score: float, best_domain_match: bool,
                 has_best: bool, out_marker: Optional[str],
                 conflict_signal: bool, had_evidence: bool,
                 had_header: bool) -> SufficiencyDecision:
        """Decide answerability from the ranker's strongest substantive item.

        ``has_best`` is whether the ranker found any substantive lead candidate;
        ``best_score`` / ``best_domain_match`` describe it. ``out_marker`` is set
        when the query is clearly about an out-of-scope domain. ``conflict_signal``
        is set when the verifier rejected a high-overlap near-miss memory — that
        contradiction outranks ordinary grounding and is never masked by a
        knowledge chunk retrieved on the same query.
        """
        # 1. A near-miss contradiction outranks grounding — and is not masked by
        #    a knowledge chunk that happened to be retrieved on the same query.
        if conflict_signal:
            return SufficiencyDecision(
                SufficiencyVerdict.CONFLICT, LABEL_CONFLICT_MASKED,
                "a near-miss memory contradicts the query; conflict surfaced "
                "before grounding")

        # 2. The query is about an out-of-scope domain with no covering evidence.
        if out_marker is not None and (not has_best or not best_domain_match):
            return SufficiencyDecision(
                SufficiencyVerdict.NO_SUPPORT,
                LABEL_OUT_OF_DOMAIN_FALSE_GROUNDING,
                f"query is about out-of-scope topic {out_marker!r}; no pack "
                "evidence covers it")

        # 3. Nothing substantive to lead with (e.g. only metadata/source headers).
        if not has_best:
            label = LABEL_METADATA_HEADER_LEAD if had_header else LABEL_WEAK_SUPPORT
            return SufficiencyDecision(
                SufficiencyVerdict.WEAK_MATCH, label,
                "no substantive evidence available to lead a grounded answer")

        # 4. Strong substantive match -> full grounding.
        if best_score >= self.strong and best_domain_match:
            return SufficiencyDecision(
                SufficiencyVerdict.RELEVANT, LABEL_GROUNDED_RELEVANT,
                f"strong substantive match (overlap {best_score:.2f})")

        # 5. Moderate match -> grounded with an explicit limitation.
        if best_score >= self.moderate:
            return SufficiencyDecision(
                SufficiencyVerdict.PARTIAL, LABEL_GROUNDED_PARTIAL,
                f"partial match (overlap {best_score:.2f}); answer is limited "
                "to what the evidence directly supports",
                "Evidence only partially matches the question; treat the answer "
                "as partial and verify the rest.")

        # 6. Weak match -> cannot ground a full answer.
        if best_score >= self.weak_floor:
            return SufficiencyDecision(
                SufficiencyVerdict.WEAK_MATCH, LABEL_WEAK_SUPPORT,
                f"weak match (overlap {best_score:.2f}); insufficient to ground")

        # 7. Evidence was retrieved but is essentially unrelated.
        return SufficiencyDecision(
            SufficiencyVerdict.NO_SUPPORT, LABEL_GROUNDED_IRRELEVANT,
            f"retrieved evidence is unrelated (overlap {best_score:.2f})")
