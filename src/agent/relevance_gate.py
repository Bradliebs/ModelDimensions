"""v2.4 Relevance & Sufficiency Gate — between retrieval and grounding.

The frozen retrieve -> verify -> ground path is *recall-biased*: the
deterministic backend returns its top-k candidates and
:meth:`WorkbenchService.build_grounding_package` previously grounded an answer
whenever *any* memory or knowledge chunk came back. Retrieval presence is not
relevance, so weakly-related or out-of-domain chunks could become a confident
grounded answer (the dominant failure mode the v2.3 value sprint measured).

This module adds a thin, **deterministic, offline** sufficiency layer that sits
between retrieval and grounding and answers one question: *is the retrieved
evidence good enough to ground this query, and if so, which item should lead?*

It is **additive and downgrade-only**. It can move a would-be GROUNDED decision
to a partial answer, a refusal, or an explicit conflict; it can re-order
evidence so the strongest substantive item leads; it can never invent evidence,
upgrade a refusal into a grounded answer, or relax any frozen guarantee
(verifier, citations, lifecycle, pack isolation, AnswerGuard). The gate decides
*answerability*; hybrid retrieval may still improve candidate order upstream.

The relevance signal is lexical token overlap (the same family as
:func:`agent.memory_lifecycle.lexical_overlap`) — deliberately simple and
explainable, not semantic understanding. Every decision carries a per-check
trace and a one-line sufficiency reason so the query-assist audit, the
value-sprint report, and the AnswerGuard chain can all see *why* a verdict was
reached.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from slm.assistant_composer import EvidenceItem


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

STRONG_OVERLAP = 0.5   # >= -> RELEVANT eligible
MODERATE_OVERLAP = 0.28  # >= -> PARTIAL eligible
WEAK_FLOOR = 0.12      # >= -> WEAK_MATCH; below -> NO_SUPPORT

_MIN_CONTENT_TOKENS = 3         # an evidence chunk needs this many to be a lead
_DOMAIN_MISMATCH_FACTOR = 0.25  # multiplicative penalty when domains disagree
_STALE_PENALTY_FACTOR = 0.9     # mild penalty; stale never disqualifies on its own

# -- lexical helpers ----------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "with", "as", "that", "this", "these", "those", "it", "its", "from", "into",
    "what", "which", "who", "whom", "how", "when", "where", "why", "about",
    "i", "we", "you", "they", "he", "she", "my", "our", "your", "their",
    "me", "us", "can", "could", "should", "would", "will", "shall", "may",
    "have", "has", "had", "not", "no", "yes", "if", "then", "else", "so",
    "up", "out", "over", "under", "between", "any", "some", "all", "more",
    "say", "says", "said", "tell", "show", "give", "get", "use", "using",
    "please", "thanks", "thank", "there", "here", "than", "but", "also",
})

# Topic markers for domains the active packs do not cover. A query that is
# clearly *about* one of these, with no domain-matching substantive evidence,
# is out of scope and must refuse rather than ground on an incidental overlap.
_OUT_OF_DOMAIN_MARKERS = frozenset({
    "aws", "s3", "ec2", "iam", "lambda", "dynamodb", "cloudfront", "redshift",
    "gcp", "bigquery", "firestore", "pubsub",
})


def content_tokens(text: str) -> List[str]:
    """Lowercase content tokens with stopwords and punctuation removed."""
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if t not in _STOPWORDS and len(t) > 1]


def overlap_coefficient(query_tokens: set, evidence_tokens: set) -> float:
    """Szymkiewicz-Simpson overlap: |Q n E| / min(|Q|, |E|).

    Length-robust: a short query whose terms all appear in a long chunk scores
    high, and a long verbatim query that contains a short chunk also scores
    high. An irrelevant chunk that shares few distinctive terms scores low.
    """
    if not query_tokens or not evidence_tokens:
        return 0.0
    shared = len(query_tokens & evidence_tokens)
    return shared / min(len(query_tokens), len(evidence_tokens))


_METADATA_MARKERS = ("source:", "version:", "domain:", "authority:",
                     "staleness:", "section:", "retrieved:", "published:")


def is_metadata_header(text: str) -> bool:
    """True when a chunk is a source/metadata header rather than substance.

    These ``Source: ... Version: ... Authority: ...`` headers describe a source
    but assert nothing; they must never be used as substantive lead evidence.
    """
    low = (text or "").lower()
    marker_hits = sum(1 for m in _METADATA_MARKERS if m in low)
    if marker_hits >= 3:
        return True
    return False


def infer_out_of_domain(query_tokens: set) -> Optional[str]:
    """Return an out-of-scope domain marker present in the query, if any."""
    for marker in _OUT_OF_DOMAIN_MARKERS:
        if marker in query_tokens:
            return marker
    return None


def has_near_miss_conflict(query: str, candidates, *,
                           threshold: float = STRONG_OVERLAP) -> bool:
    """True when a *rejected* memory candidate is a genuine near-miss.

    The frozen verifier returns REJECT for any material mismatch on the nearest
    memory — which covers both a one-token contradiction (``Friday`` vs
    ``Monday``) and a merely-irrelevant nearest memory. Only the former should
    surface as a CONFLICT (and mask grounding), so we additionally require high
    lexical overlap between the query and the rejected candidate. This keeps the
    conflict-masking fix from firing on every query that happens to retrieve an
    unrelated memory the verifier correctly rejected.
    """
    q = set(content_tokens(query))
    if not q:
        return False
    for cand in candidates or []:
        if (cand or {}).get("verdict") != "reject":
            continue
        e = set(content_tokens((cand or {}).get("canonical_text", "")))
        if overlap_coefficient(q, e) >= threshold:
            return True
    return False


@dataclass(frozen=True)
class EvidenceAssessment:
    """The relevance assessment of a single piece of retrieved evidence."""

    citation_id: str
    kind: str
    topic_overlap: float
    domain_match: bool
    is_metadata_header: bool
    is_stale: bool
    substantive: bool
    score: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "citation_id": self.citation_id,
            "kind": self.kind,
            "topic_overlap": round(self.topic_overlap, 4),
            "domain_match": self.domain_match,
            "is_metadata_header": self.is_metadata_header,
            "is_stale": self.is_stale,
            "substantive": self.substantive,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RelevanceReport:
    """The gate's decision for one query plus its full reasoning trace."""

    verdict: SufficiencyVerdict
    label: str
    sufficiency_reason: str
    lead_citation_id: Optional[str]
    ordered_evidence: List[EvidenceItem]
    assessments: List[EvidenceAssessment]
    checks: Dict[str, object]
    limitation: Optional[str] = None

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
        }


def _assess_one(item: EvidenceItem, query_tokens: set, *,
                out_of_domain_marker: Optional[str],
                stale_ids: set) -> EvidenceAssessment:
    e_tokens = set(content_tokens(item.text))
    overlap = overlap_coefficient(query_tokens, e_tokens)
    header = is_metadata_header(item.text)
    is_stale = item.citation_id in stale_ids

    # Domain match: a memory item is always domain-fit for its own route. A
    # knowledge item with a declared domain is mismatched only when the query is
    # clearly about an out-of-scope domain and this item does not cover it.
    domain_match = True
    if out_of_domain_marker is not None and item.kind == "knowledge":
        domain_match = out_of_domain_marker in e_tokens

    substantive = (not header) and len(e_tokens) >= _MIN_CONTENT_TOKENS

    reasons: List[str] = []
    score = overlap
    if header:
        reasons.append("metadata/source header — cannot be substantive lead")
        score = 0.0
    if not domain_match:
        reasons.append(
            f"out-of-domain for query topic {out_of_domain_marker!r}")
        score *= _DOMAIN_MISMATCH_FACTOR
    if is_stale:
        reasons.append("stale source — penalised but still citable")
        score *= _STALE_PENALTY_FACTOR
    if not substantive and not header:
        reasons.append("too few content tokens to lead")

    return EvidenceAssessment(
        citation_id=item.citation_id,
        kind=item.kind,
        topic_overlap=overlap,
        domain_match=domain_match,
        is_metadata_header=header,
        is_stale=is_stale,
        substantive=substantive,
        score=score,
        reasons=reasons,
    )


def assess_relevance(query: str, evidence: List[EvidenceItem], *,
                     route: str,
                     conflict_signal: bool = False,
                     stale_ids: Optional[set] = None) -> RelevanceReport:
    """Decide whether retrieved evidence is sufficient to ground ``query``.

    ``conflict_signal`` is set when the memory verifier rejected a near-miss
    candidate for this query: a contradiction must surface as CONFLICT before
    ordinary grounding, and the knowledge route must not mask it. ``stale_ids``
    is the set of citation ids backed by a stale source (a penalty, never a
    disqualifier — frozen staleness labelling is preserved).
    """
    stale_ids = stale_ids or set()
    query_tokens = set(content_tokens(query))
    out_marker = infer_out_of_domain(query_tokens)

    assessments = [
        _assess_one(item, query_tokens,
                    out_of_domain_marker=out_marker, stale_ids=stale_ids)
        for item in evidence
    ]

    # Re-order: strongest substantive item first, headers/non-substantive last.
    order = sorted(
        range(len(evidence)),
        key=lambda i: (assessments[i].substantive, assessments[i].score),
        reverse=True,
    )
    ordered_evidence = [evidence[i] for i in order]
    ordered_assessments = [assessments[i] for i in order]

    substantive = [a for a in ordered_assessments if a.substantive]
    best = substantive[0] if substantive else None
    best_score = best.score if best else 0.0

    checks: Dict[str, object] = {
        "query_intent_match": _route_fit(route, best),
        "entity_topic_match": round(best_score, 4),
        "domain_match": best.domain_match if best else False,
        "claim_support": best_score >= MODERATE_OVERLAP,
        "contradiction_near_miss": bool(conflict_signal),
        "source_status_ok": True,
        "citation_substance": best is not None,
        "metadata_header_penalty": any(a.is_metadata_header
                                       for a in ordered_assessments),
        "stale_source_penalty": any(a.is_stale for a in ordered_assessments),
        "route_fit": _route_fit(route, best),
    }

    verdict, label, reason, limitation = _decide(
        best=best, best_score=best_score, out_marker=out_marker,
        conflict_signal=conflict_signal,
        had_evidence=bool(evidence),
        had_header=any(a.is_metadata_header for a in ordered_assessments),
    )

    lead = best.citation_id if (best and verdict in (
        SufficiencyVerdict.RELEVANT, SufficiencyVerdict.PARTIAL)) else None

    return RelevanceReport(
        verdict=verdict,
        label=label,
        sufficiency_reason=reason,
        lead_citation_id=lead,
        ordered_evidence=ordered_evidence,
        assessments=ordered_assessments,
        checks=checks,
        limitation=limitation,
    )


def _route_fit(route: str, best: Optional[EvidenceAssessment]) -> bool:
    """Whether the lead evidence kind fits the planned route intent."""
    if best is None:
        return False
    if route == "memory_only":
        return best.kind == "memory"
    if route == "knowledge_only":
        return best.kind == "knowledge"
    return True  # both / general routes accept either kind


def _decide(*, best: Optional[EvidenceAssessment], best_score: float,
            out_marker: Optional[str], conflict_signal: bool,
            had_evidence: bool, had_header: bool):
    """Map the best substantive score and signals onto a verdict and label."""
    # 1. A near-miss contradiction outranks grounding — and is not masked by a
    #    knowledge chunk that happened to be retrieved on the same query.
    if conflict_signal:
        return (SufficiencyVerdict.CONFLICT, LABEL_CONFLICT_MASKED,
                "a near-miss memory contradicts the query; conflict surfaced "
                "before grounding", None)

    # 2. The query is about an out-of-scope domain with no covering evidence.
    if out_marker is not None and (best is None
                                   or not best.domain_match):
        return (SufficiencyVerdict.NO_SUPPORT,
                LABEL_OUT_OF_DOMAIN_FALSE_GROUNDING,
                f"query is about out-of-scope topic {out_marker!r}; no pack "
                "evidence covers it", None)

    # 3. Nothing substantive to lead with (e.g. only metadata/source headers).
    if best is None:
        label = LABEL_METADATA_HEADER_LEAD if had_header else LABEL_WEAK_SUPPORT
        return (SufficiencyVerdict.WEAK_MATCH, label,
                "no substantive evidence available to lead a grounded answer",
                None)

    # 4. Strong substantive match -> full grounding.
    if best_score >= STRONG_OVERLAP and best.domain_match:
        return (SufficiencyVerdict.RELEVANT, LABEL_GROUNDED_RELEVANT,
                f"strong substantive match (overlap {best_score:.2f})", None)

    # 5. Moderate match -> grounded with an explicit limitation.
    if best_score >= MODERATE_OVERLAP:
        return (SufficiencyVerdict.PARTIAL, LABEL_GROUNDED_PARTIAL,
                f"partial match (overlap {best_score:.2f}); answer is limited "
                "to what the evidence directly supports",
                "Evidence only partially matches the question; treat the answer "
                "as partial and verify the rest.")

    # 6. Weak match -> cannot ground a full answer.
    if best_score >= WEAK_FLOOR:
        return (SufficiencyVerdict.WEAK_MATCH, LABEL_WEAK_SUPPORT,
                f"weak match (overlap {best_score:.2f}); insufficient to ground",
                None)

    # 7. Evidence was retrieved but is essentially unrelated.
    return (SufficiencyVerdict.NO_SUPPORT, LABEL_GROUNDED_IRRELEVANT,
            f"retrieved evidence is unrelated (overlap {best_score:.2f})", None)
