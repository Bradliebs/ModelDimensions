"""Pure, deterministic scoring helpers for the hybrid retrieval backend.

Everything in this module is a pure function or a plain dataclass: no I/O, no
model loading, no randomness. The hybrid backend uses these helpers to blend a
lexical keyword signal, an optional semantic-similarity signal, and a gentle
source-authority weight into a single combined score, and to record an
auditable report of why each candidate was selected or rejected.

The keyword signal here is honest about what it is: token-level lexical
overlap. It is paraphrase-robust in a way the whole-string-hash deterministic
encoder is not, but it is *not* neural semantics. True semantic similarity only
enters when a real embedding backend is supplied to the hybrid backend.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

__all__ = [
    "tokenize",
    "keyword_overlap",
    "exact_phrase_match",
    "authority_weight",
    "HybridWeights",
    "combine_scores",
    "ScoredChunk",
    "HybridRetrievalReport",
]

# A small, conservative stop-word set. Kept deliberately short so that domain
# terms are never discarded; the goal is only to stop trivial words from
# dominating the overlap coefficient.
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is",
        "are", "be", "with", "by", "as", "at", "it", "this", "that", "how",
        "do", "i", "can", "you", "my", "me", "we", "us", "from", "into",
        "what", "which", "use", "using", "used",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lower-case, split on non-alphanumeric runs, drop trivial stop words.

    Single-character tokens are dropped as well; they carry no useful overlap
    signal and only add noise.
    """

    if not text:
        return []
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in _STOP_WORDS
    ]


def keyword_overlap(query: str, text: str) -> float:
    """Overlap coefficient of query tokens found in ``text`` (range 0..1).

    Defined as ``|Q ∩ T| / |Q|`` so a chunk that contains *all* of the query's
    meaningful tokens scores ``1.0`` regardless of how much extra text it
    carries. This is recall-oriented on purpose: retrieval wants chunks that
    cover the query, not chunks that are lexically identical to it.
    """

    q = set(tokenize(query))
    if not q:
        return 0.0
    t = set(tokenize(text))
    return len(q & t) / len(q)


def exact_phrase_match(query: str, text: str) -> bool:
    """True when the normalised query appears verbatim inside ``text``.

    Normalisation collapses whitespace and lower-cases both sides so trivial
    formatting differences do not defeat the match. Used to give an exact
    keyword hit a decisive edge over a merely paraphrased near-match.
    """

    def _norm(s: str) -> str:
        return " ".join(s.lower().split())

    nq = _norm(query)
    if not nq:
        return False
    return nq in _norm(text)


# Authority is a *gentle* multiplier, never a gate. Source-status gating
# (deleted/stale) is handled upstream of the backend and must not be duplicated
# or weakened here.
_AUTHORITY_WEIGHTS = {
    "official": 1.0,
    "reputable": 0.85,
    "community": 0.6,
    "unknown": 0.4,
}


def authority_weight(authority: str) -> float:
    """Map a source-authority label to a gentle [0.4, 1.0] multiplier."""

    return _AUTHORITY_WEIGHTS.get((authority or "").strip().lower(), 0.4)


@dataclass(frozen=True)
class HybridWeights:
    """Blend weights for :func:`combine_scores`.

    ``keyword`` and ``semantic`` are the relative weights of the lexical and
    semantic signals when both are present. ``authority_influence`` is how much
    the source-authority multiplier is allowed to move the combined score (kept
    small so a high-authority irrelevant chunk never outranks a relevant one).
    ``exact_bonus`` is added when the query appears verbatim, so an exact
    keyword match outranks a paraphrase with a comparable blended score.
    """

    keyword: float = 0.5
    semantic: float = 0.5
    authority_influence: float = 0.1
    exact_bonus: float = 0.25


def combine_scores(
    keyword: float,
    semantic: Optional[float],
    authority_w: float,
    *,
    weights: HybridWeights,
    semantic_available: bool,
    exact_match: bool = False,
) -> float:
    """Blend the available signals into a single score.

    When no semantic signal is available the semantic weight is redistributed
    to the keyword term, so the deterministic/offline fallback is *not*
    penalised for the missing component. The authority multiplier nudges the
    base score within a narrow band, and an exact verbatim match adds a fixed
    bonus so it cannot be edged out by a paraphrase.
    """

    if semantic_available and semantic is not None:
        total = weights.keyword + weights.semantic
        base = (weights.keyword * keyword + weights.semantic * max(0.0, semantic)) / total
    else:
        base = keyword

    # Authority modulates within [1 - influence, 1.0]: a low-authority source
    # is gently discounted, never zeroed.
    modulated = base * ((1.0 - weights.authority_influence) + weights.authority_influence * authority_w)

    if exact_match:
        modulated += weights.exact_bonus

    return round(modulated, 6)


@dataclass
class ScoredChunk:
    """One chunk's scoring breakdown, for the retrieval audit."""

    chunk_id: str
    source_id: str
    source_name: str
    authority: str
    keyword_score: float
    semantic_score: Optional[float]
    authority_weight: float
    exact_match: bool
    combined_score: float
    selected: bool
    rank: Optional[int] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HybridRetrievalReport:
    """Auditable record of a single hybrid retrieval call.

    Captures which backend ran, whether a semantic component was active, the
    blend weights, and the per-candidate score breakdown for both selected and
    rejected chunks (rejected list is capped upstream where practical).
    """

    backend_name: str
    semantic_available: bool
    weights: dict
    selected: List[dict] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "backend_name": self.backend_name,
            "semantic_available": self.semantic_available,
            "weights": dict(self.weights),
            "selected": list(self.selected),
            "rejected": list(self.rejected),
        }
