"""V1 answer verifier: token-coverage check against the cited cells.

The existing `src/agent/verifier.py` is a deterministic *cell-vs-cell*
equivalence verifier (ACCEPT / AMBIGUOUS / REJECT). V1 needs a different
shape: given a generated free-text answer and the cells it was supposed
to draw from, decide whether the answer is grounded in those cells or
the model wandered off into unsupported claims.

Two-stage check (always on):

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

Two extra stages (opt-in via ``strict_nonsense=True``; the production
pipeline turns them on):

  STAGE C — query coherence (catches token-salad / repetitive queries
  before they consume a generation slot's worth of trust):
    Reject if the question has fewer than two wordlike tokens (alpha,
    length >= 3, contains a vowel), or if one token accounts for more
    than half the question's tokens. Catches "asdf qwerty zxcv hjkl",
    "blah blah blah", "1234567890 !@#$%^^*()", "test test test".

  STAGE D — query-evidence overlap (catches retrieval sets with low
  query-to-chunk relevance, the noise-margin failure mode):
    When the question has at least two content tokens, the cited cell
    blob must contain at least one of them. Catches "asdf qwerty zxcv
    hjkl" -> 12 cells about something unrelated, where Stage A passes
    because the answer paraphrases the random cells but the cells were
    never about the question.

Stages run A -> B -> C -> D; any failure short-circuits to grounded=False.
Stages C and D are off by default for backward compatibility with callers
that drive the verifier on hand-crafted test inputs.

Edge cases:
  - If the answer has zero content tokens after filtering, coverage is
    treated as 1.0 (nothing to verify - trivially grounded or trivially
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

# Stage C tunables: at least this many wordlike tokens must be present in
# the question, and no single token may account for more than this share of
# the question's tokens when the question has more than one token. Floor
# is 1 so single-word queries ("zebras?") still pass; the wordlike check
# rejects digit/symbol noise like "1234567890 !@#$%^&*()", and Stage D
# backstops multi-token salad like "asdf qwerty zxcv hjkl" via the
# query-evidence-overlap check.
MIN_QUERY_WORDLIKE: int = 1
MAX_DOMINANT_TOKEN_RATIO: float = 0.50

_VOWEL_RE = re.compile(r"[aeiouy]", re.IGNORECASE)


def _is_wordlike(token: str) -> bool:
    """A token that resembles a real word: at least three characters, all
    alphabetic, and contains a vowel. Catches alphabet salad like
    ``hjklhjklhjkl`` and digit/symbol noise."""
    if len(token) < 3:
        return False
    if not token.isalpha():
        return False
    return bool(_VOWEL_RE.search(token))


def _check_query_coherence(
    question: str,
    *,
    min_wordlike: int = MIN_QUERY_WORDLIKE,
    max_dominant_ratio: float = MAX_DOMINANT_TOKEN_RATIO,
) -> tuple[bool, str]:
    """Stage C — return ``(ok, reason)``. ``reason`` is empty when ``ok``."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(question)]
    if not tokens:
        return False, "query has no tokens"
    wordlike = [t for t in tokens if _is_wordlike(t)]
    if len(wordlike) < min_wordlike:
        return False, (
            f"query has only {len(wordlike)} wordlike token(s) "
            f"(need >= {min_wordlike})"
        )
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    most_common, most_count = max(counts.items(), key=lambda kv: kv[1])
    # Dominance is only meaningful at 3+ tokens. A 1- or 2-token query is
    # too short to call "repetitive" — "zebras?" has 1 token, "what zebras?"
    # has 2; both are fine. "test test test" (3 tokens, 100% dominance) is
    # not.
    if len(tokens) < 3:
        return True, ""
    ratio = most_count / len(tokens)
    if ratio > max_dominant_ratio:
        return False, (
            f"query is dominated by token {most_common!r} "
            f"({most_count}/{len(tokens)} = {ratio:.0%})"
        )
    return True, ""


def _check_query_evidence_overlap(
    question: str,
    cell_blob_lower: str,
) -> tuple[bool, str]:
    """Stage D — return ``(ok, reason)``. The cited cells must contain at
    least one content token from the question. Skipped (returns ok) when
    the question has fewer than two content tokens of its own — Stage C
    will have already screened those.

    Catches the noise pattern where the gate accepts a margin from random
    cells whose content has nothing to do with the query (\"asdf qwerty
    zxcv hjkl\" -> 12 cells about whatever happened to score). The answer
    will be lexically grounded in those cells (Stage A passes) but the
    cells themselves do not address the query."""
    q_content = set(_content_tokens(question))
    if len(q_content) < 2:
        return True, ""
    if not cell_blob_lower:
        return False, "no cells to verify against"
    overlap = {t for t in q_content if t in cell_blob_lower}
    if not overlap:
        return False, (
            f"cells share no content token with the query "
            f"(query content tokens: {sorted(q_content)[:5]})"
        )
    return True, ""


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
           cell_ids: Sequence[int] | None = None,
           strict_nonsense: bool = False) -> VerificationDecision:
    """Verify ``answer`` against ``cell_texts``; ``question`` is filtered out.

    ``cell_ids`` is the set of bracketed IDs we presented in the prompt.
    When supplied, those numerics are treated as markup wherever they
    appear in the answer — bracketed or bare — so a generator that
    drops the brackets ("fact 51098" instead of "fact [51098]") is not
    penalised by Stage B as a confabulated numeric.

    ``strict_nonsense`` enables Stage C (query coherence) and Stage D
    (query-evidence overlap). Off by default to keep callers that drive
    the verifier on hand-crafted inputs unchanged; the production
    pipeline turns it on.
    """

    # Stage C runs first when enabled: a token-salad query never reaches
    # Stage A/B because there is nothing to verify against.
    if strict_nonsense:
        c_ok, c_reason = _check_query_coherence(question)
        if not c_ok:
            return VerificationDecision(
                grounded=False,
                coverage=0.0,
                covered=0,
                total=0,
                threshold=min_coverage,
                reason=f"query incoherent: {c_reason}",
                uncovered_tokens=[],
                uncited_numerics=[],
            )

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

    # Stage D runs after we have the cell blob but before any pass/fail
    # decision: a query-evidence mismatch is fatal regardless of how
    # well the answer covers the (mismatched) cells.
    if strict_nonsense:
        d_ok, d_reason = _check_query_evidence_overlap(question, cell_blob_lower)
        if not d_ok:
            return VerificationDecision(
                grounded=False,
                coverage=0.0,
                covered=0,
                total=len(answer_tokens),
                threshold=min_coverage,
                reason=f"query-evidence mismatch: {d_reason}",
                uncovered_tokens=[],
                uncited_numerics=[],
            )

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
