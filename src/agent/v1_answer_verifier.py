"""V1 answer verifier: token-coverage check against the cited cells.

The existing `src/agent/verifier.py` is a deterministic *cell-vs-cell*
equivalence verifier (ACCEPT / AMBIGUOUS / REJECT). V1 needs a different
shape: given a generated free-text answer and the cells it was supposed
to draw from, decide whether the answer is grounded in those cells or
the model wandered off into unsupported claims.

This is a deliberately simple first version. The rule:

  1. Tokenize the answer into lower-cased word tokens.
  2. Drop stopwords, numbers under length 2, and tokens that appear in the
     question itself (the question's vocabulary doesn't prove grounding).
  3. The remaining tokens are the *content* tokens. Count how many appear
     in the concatenation of the cited cell texts (substring containment
     after the same tokenization).
  4. coverage = covered / total. Fire if coverage >= MIN_COVERAGE.

Edge cases:
  - If the answer has zero content tokens after filtering, coverage is
    treated as 1.0 (nothing to verify — the answer is either trivially
    grounded or trivially empty; the caller should still inspect length).
  - If the cited cells are empty, coverage is 0.0.

This is intentionally lexical, not semantic. Phi-3 paraphrasing of a
cited cell will still pass because the substantive nouns survive
paraphrasing. A confabulated proper noun or number will not appear in
the cells and will pull coverage down. That's the failure mode this
verifier exists to catch.

Future upgrades (out of scope for V1):
  - Per-sentence rather than per-answer coverage.
  - Entailment via a small NLI model.
  - Number/date strict-match checks (re-use rules from
    `src/agent/verifier.py`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence, Set

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "having",
    "of", "for", "to", "in", "on", "at", "by", "with", "from", "as",
    "and", "or", "but", "if", "then", "else", "than", "that", "this",
    "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "him", "we", "our", "us", "you", "your",
    "i", "me", "my",
    "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "not", "no", "yes",
    "can", "could", "would", "should", "may", "might", "must", "shall",
    "will", "won", "wont",
    "there", "here", "also", "such", "some", "any", "all", "each",
    "more", "most", "less", "least", "very", "only", "just", "even",
    "into", "onto", "about", "between", "during", "after", "before",
    "above", "below", "over", "under", "again", "still",
    "yet", "so", "because", "while", "until", "though", "although",
})

# Minimum fraction of content tokens that must appear in the cited cells.
# Tuned low for V1 because Phi-3 paraphrases freely; raising this would
# silence legitimate paraphrases. If the model confabulates, the offending
# tokens drag coverage well below 0.5 quickly.
MIN_COVERAGE: float = 0.50


@dataclass
class VerificationDecision:
    """Verdict from the V1 answer verifier."""

    grounded: bool
    coverage: float
    covered: int
    total: int
    threshold: float
    reason: str
    uncovered_tokens: list[str]

    def as_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "coverage": self.coverage,
            "covered": self.covered,
            "total": self.total,
            "threshold": self.threshold,
            "reason": self.reason,
            "uncovered_tokens": self.uncovered_tokens,
        }


def _content_tokens(text: str, exclude: Set[str] | None = None) -> list[str]:
    raw = [t.lower() for t in _TOKEN_RE.findall(text)]
    excluded = exclude or set()
    out: list[str] = []
    for t in raw:
        if t in _STOPWORDS:
            continue
        if t in excluded:
            continue
        if len(t) < 2:
            continue
        out.append(t)
    return out


def verify(answer: str, cell_texts: Sequence[str], question: str = "",
           min_coverage: float = MIN_COVERAGE) -> VerificationDecision:
    """Verify ``answer`` against ``cell_texts``; ``question`` is filtered out."""

    question_tokens = set(_content_tokens(question)) if question else set()

    answer_tokens = _content_tokens(answer, exclude=question_tokens)
    if not answer_tokens:
        return VerificationDecision(
            grounded=True,
            coverage=1.0,
            covered=0,
            total=0,
            threshold=min_coverage,
            reason="answer has no content tokens to verify",
            uncovered_tokens=[],
        )

    cell_blob = " ".join(cell_texts).lower()
    cell_token_set = set(_TOKEN_RE.findall(cell_blob))

    covered: list[str] = []
    uncovered: list[str] = []
    for tok in answer_tokens:
        if tok in cell_token_set:
            covered.append(tok)
        else:
            uncovered.append(tok)

    total = len(answer_tokens)
    coverage = len(covered) / total
    grounded = coverage >= min_coverage

    return VerificationDecision(
        grounded=grounded,
        coverage=coverage,
        covered=len(covered),
        total=total,
        threshold=min_coverage,
        reason=("coverage above threshold"
                if grounded
                else "coverage below threshold (answer drifted from cells)"),
        uncovered_tokens=uncovered[:25],
    )


__all__ = ["VerificationDecision", "verify", "MIN_COVERAGE"]
