"""v2.4 EvidenceRanker — deterministic relevance scoring and ordering of evidence.

The frozen retrieve → verify → ground path is *recall-biased*: the deterministic
backend returns its top-k candidates and grounding used to fire whenever *any*
memory or knowledge chunk came back. Retrieval presence is not relevance, so a
weakly-related or out-of-domain chunk could become a confident grounded answer
(the dominant failure the v2.3 value sprint measured).

This module adds the *scoring and ordering* half of the v2.4 sufficiency layer.
It takes the query, the planned route intent, and the retrieved candidates, and
produces an auditable :class:`RankingReport`: each candidate is scored from a
small set of explainable **positive** and **negative** signals, the candidates
are sorted strongest-substantive-first, and the single most-relevant rejected
candidate is recorded with the reason it was not allowed to lead.

Signals (all auditable, all offline):

* **positive** — lexical topic overlap (Szymkiewicz-Simpson overlap coefficient
  between query and evidence content tokens); a project-milestone preference
  that favours seeded project memory when the query asks about a milestone /
  decision.
* **negative** — a metadata/source-header chunk cannot lead (score → 0); a
  topic/domain mismatch is heavily penalised; a stale source is mildly
  penalised (never disqualified — frozen staleness labelling is preserved); too
  few content tokens to be a substantive lead.

The relevance signal is lexical, not neural — deliberately simple and
explainable. The companion :class:`retrieval.sufficiency_gate.SufficiencyGate`
turns the ranker's strongest substantive score into the answerability verdict.
Nothing here invents evidence, upgrades a refusal, or relaxes a frozen guarantee.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from slm.assistant_composer import EvidenceItem

from .sufficiency_gate import STRONG_OVERLAP

# -- penalties / floors (multiplicative unless noted) -------------------------

_MIN_CONTENT_TOKENS = 3          # an evidence chunk needs this many to lead
_DOMAIN_MISMATCH_FACTOR = 0.25   # penalty when the query's domain disagrees
_STALE_PENALTY_FACTOR = 0.9      # mild; stale never disqualifies on its own
_MILESTONE_MEMORY_BONUS = 0.15   # additive boost for seeded memory on a
#                                  project-milestone/decision query (capped <=1)

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

# Markers that the query is asking about a project milestone / decision, where
# seeded project memory should be preferred over incidental knowledge chunks.
_MILESTONE_MARKERS = frozenset({
    "milestone", "milestones", "shipped", "ship", "ships", "release",
    "released", "sprint", "roadmap", "deliverable", "deliverables",
    "decision", "decided", "decide", "choose", "chose", "chosen",
})

_METADATA_MARKERS = ("source:", "version:", "domain:", "authority:",
                     "staleness:", "section:", "retrieved:", "published:")


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


def is_metadata_header(text: str) -> bool:
    """True when a chunk is a source/metadata header rather than substance.

    These ``Source: ... Version: ... Authority: ...`` headers describe a source
    but assert nothing; they must never be used as substantive lead evidence.
    """
    low = (text or "").lower()
    marker_hits = sum(1 for m in _METADATA_MARKERS if m in low)
    return marker_hits >= 3


def infer_out_of_domain(query_tokens: set) -> Optional[str]:
    """Return an out-of-scope domain marker present in the query, if any."""
    for marker in _OUT_OF_DOMAIN_MARKERS:
        if marker in query_tokens:
            return marker
    return None


def is_milestone_query(query_tokens: set) -> bool:
    """True when the query asks about a project milestone or decision."""
    return bool(query_tokens & _MILESTONE_MARKERS)


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
    rank: int = 0
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
            "rank": self.rank,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RankingReport:
    """The ranker's ordered output plus an auditable rejection trace.

    ``best`` is the strongest substantive assessment (the candidate that would
    lead a grounded answer) or ``None`` when nothing substantive survived.
    ``top_rejected`` records the most-relevant candidate that did *not* lead,
    with the reason it was held back — the single most useful line for an audit
    of *why* the gate did what it did.
    """

    ordered_evidence: List[EvidenceItem]
    assessments: List[EvidenceAssessment]
    best: Optional[EvidenceAssessment]
    out_marker: Optional[str]
    had_header: bool
    milestone_query: bool
    checks: Dict[str, object]
    top_rejected: Optional[Dict[str, object]] = None

    @property
    def best_score(self) -> float:
        return self.best.score if self.best else 0.0

    @property
    def best_domain_match(self) -> bool:
        return self.best.domain_match if self.best else False

    def to_dict(self) -> dict:
        return {
            "best_citation_id": self.best.citation_id if self.best else None,
            "best_score": round(self.best_score, 4),
            "out_marker": self.out_marker,
            "had_header": self.had_header,
            "milestone_query": self.milestone_query,
            "checks": dict(self.checks),
            "assessments": [a.to_dict() for a in self.assessments],
            "top_rejected": dict(self.top_rejected) if self.top_rejected else None,
        }


def _route_fit(route: str, best: Optional[EvidenceAssessment]) -> bool:
    """Whether the lead evidence kind fits the planned route intent."""
    if best is None:
        return False
    if route == "memory_only":
        return best.kind == "memory"
    if route == "knowledge_only":
        return best.kind == "knowledge"
    return True  # both / general routes accept either kind


class EvidenceRanker:
    """Score and order retrieved candidates by auditable relevance signals.

    The ranker is **pure and deterministic**: ``rank`` is a function of the
    query, the route intent, the candidates, and the supplied signals. It holds
    no state and performs no I/O. Penalty factors are injectable for tuning, but
    the defaults reproduce the frozen v2.4 behaviour exactly.
    """

    def __init__(self, *, domain_mismatch_factor: float = _DOMAIN_MISMATCH_FACTOR,
                 stale_penalty_factor: float = _STALE_PENALTY_FACTOR,
                 min_content_tokens: int = _MIN_CONTENT_TOKENS,
                 milestone_memory_bonus: float = _MILESTONE_MEMORY_BONUS):
        self.domain_mismatch_factor = domain_mismatch_factor
        self.stale_penalty_factor = stale_penalty_factor
        self.min_content_tokens = min_content_tokens
        self.milestone_memory_bonus = milestone_memory_bonus

    def rank(self, query: str, evidence: List[EvidenceItem], *,
             route: str, conflict_signal: bool = False,
             stale_ids: Optional[set] = None) -> RankingReport:
        """Rank ``evidence`` for ``query`` and return an auditable report.

        ``conflict_signal`` is forwarded into the ``checks`` trace (the gate
        consumes it to decide CONFLICT). ``stale_ids`` is the set of citation ids
        backed by a stale source — a penalty, never a disqualifier, so frozen
        staleness labelling is preserved.
        """
        stale_ids = stale_ids or set()
        query_tokens = set(content_tokens(query))
        out_marker = infer_out_of_domain(query_tokens)
        milestone = is_milestone_query(query_tokens)

        assessments = [
            self._assess_one(item, query_tokens, out_marker=out_marker,
                             stale_ids=stale_ids, milestone=milestone)
            for item in evidence
        ]

        # Re-order: strongest substantive item first, headers/non-substantive
        # last. Stable for ties so retrieval order is preserved among equals.
        order = sorted(
            range(len(evidence)),
            key=lambda i: (assessments[i].substantive, assessments[i].score),
            reverse=True,
        )
        ordered_evidence = [evidence[i] for i in order]
        ordered_assessments = [
            EvidenceAssessment(
                citation_id=assessments[i].citation_id,
                kind=assessments[i].kind,
                topic_overlap=assessments[i].topic_overlap,
                domain_match=assessments[i].domain_match,
                is_metadata_header=assessments[i].is_metadata_header,
                is_stale=assessments[i].is_stale,
                substantive=assessments[i].substantive,
                score=assessments[i].score,
                rank=rank,
                reasons=assessments[i].reasons,
            )
            for rank, i in enumerate(order, start=1)
        ]

        substantive = [a for a in ordered_assessments if a.substantive]
        best = substantive[0] if substantive else None

        top_rejected = self._top_rejected(ordered_assessments, best)
        had_header = any(a.is_metadata_header for a in ordered_assessments)

        checks: Dict[str, object] = {
            "query_intent_match": _route_fit(route, best),
            "entity_topic_match": round(best.score if best else 0.0, 4),
            "domain_match": best.domain_match if best else False,
            "claim_support": (best.score if best else 0.0) >= 0.28,
            "contradiction_near_miss": bool(conflict_signal),
            "source_status_ok": True,
            "citation_substance": best is not None,
            "metadata_header_penalty": had_header,
            "stale_source_penalty": any(a.is_stale for a in ordered_assessments),
            "milestone_preference": milestone,
            "route_fit": _route_fit(route, best),
        }

        return RankingReport(
            ordered_evidence=ordered_evidence,
            assessments=ordered_assessments,
            best=best,
            out_marker=out_marker,
            had_header=had_header,
            milestone_query=milestone,
            checks=checks,
            top_rejected=top_rejected,
        )

    def _assess_one(self, item: EvidenceItem, query_tokens: set, *,
                    out_marker: Optional[str], stale_ids: set,
                    milestone: bool) -> EvidenceAssessment:
        e_tokens = set(content_tokens(item.text))
        overlap = overlap_coefficient(query_tokens, e_tokens)
        header = is_metadata_header(item.text)
        is_stale = item.citation_id in stale_ids

        # Domain match: a memory item is always domain-fit for its own route. A
        # knowledge item with a declared domain is mismatched only when the
        # query is clearly about an out-of-scope domain and this item does not
        # cover it.
        domain_match = True
        if out_marker is not None and item.kind == "knowledge":
            domain_match = out_marker in e_tokens

        substantive = (not header) and len(e_tokens) >= self.min_content_tokens

        reasons: List[str] = []
        score = overlap
        if header:
            reasons.append("metadata/source header — cannot be substantive lead")
            score = 0.0
        if not domain_match:
            reasons.append(
                f"out-of-domain for query topic {out_marker!r}")
            score *= self.domain_mismatch_factor
        if is_stale:
            reasons.append("stale source — penalised but still citable")
            score *= self.stale_penalty_factor
        if not substantive and not header:
            reasons.append("too few content tokens to lead")
        if milestone and item.kind == "memory" and substantive:
            score = min(1.0, score + self.milestone_memory_bonus)
            reasons.append("project-milestone query — seeded project memory "
                           "preferred")

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

    @staticmethod
    def _top_rejected(ordered_assessments: List[EvidenceAssessment],
                      best: Optional[EvidenceAssessment]
                      ) -> Optional[Dict[str, object]]:
        """The most-relevant candidate that did not lead, with the reason.

        When a lead exists, the top rejected is the next strongest candidate by
        topic overlap. When nothing led (a refusal), it is the strongest
        candidate overall — i.e. the chunk that *almost* grounded the answer and
        the reason it was held back.
        """
        lead_id = best.citation_id if best else None
        pool = [a for a in ordered_assessments if a.citation_id != lead_id]
        if not pool:
            return None
        top = max(pool, key=lambda a: a.topic_overlap)
        reason = top.reasons[0] if top.reasons else "outranked by lead evidence"
        return {
            "citation_id": top.citation_id,
            "kind": top.kind,
            "topic_overlap": round(top.topic_overlap, 4),
            "reason": reason,
        }
