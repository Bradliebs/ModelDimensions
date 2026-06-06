"""V1 answer verifier: token-coverage check against the cited cells.

The existing `src/agent/verifier.py` is a deterministic *cell-vs-cell*
equivalence verifier (ACCEPT / AMBIGUOUS / REJECT). V1 needs a different
shape: given a generated free-text answer and the cells it was supposed
to draw from, decide whether the answer is grounded in those cells or
the model wandered off into unsupported claims.

Two-stage check:

  STAGE A — token coverage (catches lexical drift):
    1. Tokenize the answer into lower-cased word tokens.
    2. Drop stopwords, numbers under length 2, and tokens that appear in the
       question itself (the question's vocabulary doesn't prove grounding).
    3. The remaining tokens are the *content* tokens. Count how many appear
       in the concatenation of the cited cell texts.
    4. coverage = covered / total. Pass iff coverage >= MIN_COVERAGE.

  STAGE B — strict numeric/date match (catches confabulated
  proper nouns and numbers; this is the harder failure mode):
    1. Pull every numeric run (regex ``\\d+(\\.\\d+)?``) and every
       month-name token from the answer.
    2. Drop any number/month that already appears in the question
       (the question's numbers/dates don't prove grounding).
    3. Every remaining number/month must appear in the cited cell texts.
    4. A single missing numeric/date token is a hard reject. This is the
       Friday-vs-Monday bet from the v1.0 research baseline applied at
       answer-vs-cells scope: a one-token factual flip is the failure
       mode this layer exists to catch.

Both stages must pass for ``grounded=True``. Stage B firing without
Stage A is rare because most numeric drift drags coverage down too, but
it does happen for short answers ("Yes, in 1812.") so we check both.

Edge cases:
  - If the answer has zero content tokens after filtering, coverage is
    treated as 1.0 (nothing to verify — trivially grounded or trivially
    empty; the caller should still inspect length).
  - If the cited cells are empty, coverage is 0.0.

This is intentionally lexical, not semantic. Phi-3 paraphrasing of a
cited cell will still pass because the substantive nouns survive
paraphrasing. A confabulated proper noun will still pull coverage down;
a confabulated date (e.g. "1969" not in cells) is rejected by Stage B.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence, Set

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
# Numeric runs: integers, decimals, dates with separators handled by the
# fact that we test substring-presence in the cell blob.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Citation markers we ASKED the generator to emit (e.g. "[1]", "[123]"):
# these are markup, not facts, so strip them before numeric extraction.
_CITATION_BRACKET_RE = re.compile(r"\[\d+\]")

# Word boundary used to strip bare cell-ID numerics that the generator
# was supposed to bracket but didn't (e.g. "fact 51098" instead of
# "fact [51098]"). Built per-call from the cell IDs we presented.
def _build_bare_cell_id_re(cell_ids: Sequence[int]) -> re.Pattern[str] | None:
    if not cell_ids:
        return None
    parts = sorted({str(int(c)) for c in cell_ids}, key=len, reverse=True)
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in parts) + r")\b")

_MONTHS: frozenset[str] = frozenset({
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
})

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
    # Stage B: numbers/dates in the answer that did not appear in the cells.
    # Empty list when stage B passed (or had nothing to check).
    uncited_numerics: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "coverage": self.coverage,
            "covered": self.covered,
            "total": self.total,
            "threshold": self.threshold,
            "reason": self.reason,
            "uncovered_tokens": self.uncovered_tokens,
            "uncited_numerics": self.uncited_numerics,
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


def _extract_numerics(text: str) -> list[str]:
    """Return numbers and month-name tokens found in ``text`` (deduplicated,
    insertion order preserved). Numbers keep their original form so a year
    "1969" matches "1969" but not "69"."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _NUMBER_RE.findall(text):
        if m not in seen:
            seen.add(m)
            out.append(m)
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in _MONTHS and raw not in seen:
            seen.add(raw)
            out.append(raw)
    return out


def verify(answer: str, cell_texts: Sequence[str], question: str = "",
           min_coverage: float = MIN_COVERAGE,
           cell_ids: Sequence[int] | None = None) -> VerificationDecision:
    """Verify ``answer`` against ``cell_texts``; ``question`` is filtered out.

    ``cell_ids`` is the set of bracketed IDs we presented in the prompt.
    When supplied, those numerics are treated as markup wherever they
    appear in the answer — bracketed or bare — so a generator that
    drops the brackets ("fact 51098" instead of "fact [51098]") is not
    penalised by Stage B as a confabulated numeric.
    """

    # Strip citation markers ("[1]", "[123]") before any extraction: they
    # are prompt-shaped markup, not facts, and would otherwise be classified
    # as uncited numerics.
    answer_clean = _CITATION_BRACKET_RE.sub(" ", answer)
    if cell_ids:
        bare_re = _build_bare_cell_id_re(cell_ids)
        if bare_re is not None:
            answer_clean = bare_re.sub(" ", answer_clean)

    question_tokens = set(_content_tokens(question)) if question else set()
    question_numerics = set(_extract_numerics(question)) if question else set()

    answer_tokens = _content_tokens(answer_clean, exclude=question_tokens)
    cell_blob_lower = " ".join(cell_texts).lower()

    if not answer_tokens:
        # Even an empty-content answer must not introduce uncited numerics.
        ans_nums = [n for n in _extract_numerics(answer_clean)
                    if n not in question_numerics]
        uncited = [n for n in ans_nums if n.lower() not in cell_blob_lower]
        if uncited:
            return VerificationDecision(
                grounded=False,
                coverage=1.0,
                covered=0,
                total=0,
                threshold=min_coverage,
                reason=f"uncited numeric/date in answer: {uncited[:3]}",
                uncovered_tokens=[],
                uncited_numerics=uncited[:10],
            )
        return VerificationDecision(
            grounded=True,
            coverage=1.0,
            covered=0,
            total=0,
            threshold=min_coverage,
            reason="answer has no content tokens to verify",
            uncovered_tokens=[],
            uncited_numerics=[],
        )

    cell_token_set = set(_TOKEN_RE.findall(cell_blob_lower))

    covered: list[str] = []
    uncovered: list[str] = []
    for tok in answer_tokens:
        if tok in cell_token_set:
            covered.append(tok)
        else:
            uncovered.append(tok)

    total = len(answer_tokens)
    coverage = len(covered) / total
    coverage_ok = coverage >= min_coverage

    # Stage B: every number/month in the answer (minus those already in the
    # question) must appear in the cell text blob.
    answer_numerics = _extract_numerics(answer_clean)
    novel_numerics = [n for n in answer_numerics if n not in question_numerics]
    uncited = [n for n in novel_numerics if n.lower() not in cell_blob_lower]
    numerics_ok = len(uncited) == 0

    grounded = coverage_ok and numerics_ok
    if grounded:
        reason = "coverage above threshold; all numerics cited"
    elif not coverage_ok and not numerics_ok:
        reason = (f"coverage below threshold AND uncited numerics "
                  f"({uncited[:3]})")
    elif not coverage_ok:
        reason = "coverage below threshold (answer drifted from cells)"
    else:
        reason = f"uncited numeric/date in answer: {uncited[:3]}"

    return VerificationDecision(
        grounded=grounded,
        coverage=coverage,
        covered=len(covered),
        total=total,
        threshold=min_coverage,
        reason=reason,
        uncovered_tokens=uncovered[:25],
        uncited_numerics=uncited[:10],
    )


__all__ = ["VerificationDecision", "verify", "MIN_COVERAGE"]
