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

Three extra stages (opt-in via ``strict_nonsense=True``; the production
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

  STAGE E — answer-entity / question-anchor co-occurrence (catches
  Stage-A false positives where the answer's substantive proper noun
  is confabulated but happens to appear *somewhere* in the cited
  cells, just not in connection with the question's subject):
    1. Extract the question's proper-noun runs (consecutive
       capitalised tokens, hyphenated names kept whole). Rank by
       specificity: multi-token runs (e.g. "French Connection",
       "Queen Elizabeth II") first; single-token runs (e.g. "Canada")
       second. Use the most-specific tier available — if any
       multi-token anchors exist, single-token anchors are not
       consulted. If the question has no proper-noun runs at all
       (e.g. "What gas is used in these balloons?"), fall back to the
       question's content tokens (stopwords removed) and tag the
       check as having used a weaker fallback.
    2. Extract the answer's proper-noun runs the same way and drop
       any that are subspans of any question run (same-name questions
       don't have to be re-grounded by the answer) or that
       lower-case-match the prompt-template whitelist (Fact, Answer,
       Question, ...). Each remaining run is a NOVEL ANSWER ENTITY.
    3. Normalise both sides: lowercase, strip honorifics from the
       front (King, Queen, Saint/St, Mount/Mt, Dr, Sir, ...), replace
       hyphens and apostrophes with spaces, collapse whitespace. So
       "King Charles III" matches "Charles III" in the cell, and
       "Jean-Philippe Rameau" matches "Jean-Philippe Rameau" or
       "Jean Philippe Rameau" but is preserved as one entity, not
       three independently-checkable tokens.
    4. For each novel answer entity, the requirement is: at least one
       cited cell whose normalised text contains BOTH the entity AND
       at least one normalised question anchor (substring match).
       Same-cell co-occurrence is the load-bearing constraint;
       any-cell anchoring lets through confabulations whose name
       happens to appear in an unrelated bank cell, which is the
       exp24-Q2 failure: "Jean-Philippe Rameau" exists in a generic
       France-music cell that never mentions "French Connection".
    5. A single novel answer entity that fails the co-occurrence
       check is a hard reject. The reject reason names the entity
       and the anchors it was checked against; the per-entity audit
       trail is exposed via ``stage_e_log`` on the decision.

Stages run A -> B -> C -> D -> E; any failure short-circuits to
grounded=False. Stages C, D, and E are off by default for backward
compatibility with callers that drive the verifier on hand-crafted
test inputs.

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


# Stage E whitelist: prompt-template artefacts and very common
# capitalized words that appear in answer text without carrying factual
# claim weight. Lowercased.
_STAGE_E_WHITELIST: frozenset[str] = frozenset({
    "fact", "facts", "answer", "question", "context",
    "yes", "no",
})

# Honorifics, titles, and geographic prefixes stripped from the START of
# an entity when normalising. The underlying entity is what we want to
# match against the cell, not the surface marker. Lowercased.
_STAGE_E_HONORIFICS: frozenset[str] = frozenset({
    "king", "queen", "prince", "princess", "lord", "lady",
    "sir", "dame", "dr", "mr", "mrs", "ms", "ms.",
    "saint", "st", "st.",
    "mount", "mt", "mt.", "lake", "river", "cape",
    "the", "a", "an",
})

# Maximal run of consecutive capitalised tokens. A capitalised token is
# a word starting with [A-Z]; internal hyphens and apostrophes keep the
# token together (so "Jean-Philippe" and "D'Antoni" are single tokens).
# Roman-numeral-like all-caps tokens (II, III, IV) match because [A-Z]
# followed by [A-Za-z]* permits zero or more letters of any case.
_STAGE_E_RUN_RE = re.compile(
    r"\b[A-Z][A-Za-z]*(?:[-'][A-Za-z]+)*"
    r"(?:\s+[A-Z][A-Za-z]*(?:[-'][A-Za-z]+)*)*"
    r"\b"
)

# Internal punctuation we replace with whitespace before substring match,
# so "Jean-Philippe Rameau" and "Jean Philippe Rameau" both normalise to
# "jean philippe rameau".
_STAGE_E_PUNCT_RE = re.compile(r"[^\w\s]")
_STAGE_E_WS_RE = re.compile(r"\s+")
# English possessive 's (straight or curly apostrophe) stripped before
# normalisation so "Helena's" -> "Helena" and matches "Helena" in cells.
_STAGE_E_POSSESSIVE_RE = re.compile(r"['\u2019]s\b", re.IGNORECASE)


def _normalise_for_anchor(s: str) -> str:
    """Lowercase ``s``, drop the English possessive ``'s``, replace
    internal punctuation with whitespace, and collapse runs of
    whitespace. Used both for cell text and for entity spans before
    substring matching."""
    s = _STAGE_E_POSSESSIVE_RE.sub("", s)
    return _STAGE_E_WS_RE.sub(" ", _STAGE_E_PUNCT_RE.sub(" ", s.lower())).strip()


def _normalise_entity(span: str) -> str:
    """Normalise an entity span and strip leading honorifics/titles. So
    "King Charles III" -> "charles iii" and "Mt. Logan" -> "logan"."""
    parts = _normalise_for_anchor(span).split()
    while parts and parts[0] in _STAGE_E_HONORIFICS:
        parts = parts[1:]
    return " ".join(parts)


def _extract_capitalised_runs(text: str) -> list[str]:
    """Return original-case capitalised runs found in ``text``,
    deduplicated by normalised form, insertion order preserved."""
    seen_norm: set[str] = set()
    out: list[str] = []
    for m in _STAGE_E_RUN_RE.finditer(text):
        span = m.group(0)
        norm = _normalise_entity(span)
        # Skip runs that normalise to something too short to be a useful
        # anchor (a single 1- or 2-character token; e.g. a stray "Mr"
        # or "St" with no following name). Stage E requires at least
        # length-3 to avoid noise; the normalised whole-run length
        # check is the cleanest place to enforce it.
        if len(norm) < 3:
            continue
        # Drop runs that are entirely template/whitelist words (e.g.
        # an answer that starts with the literal token "Answer:").
        if all(p in _STAGE_E_WHITELIST or p in _STOPWORDS or p in _MONTHS
               for p in norm.split()):
            continue
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        out.append(span)
    return out


def _classify_question_anchors(
    question: str,
) -> tuple[list[str], bool]:
    """Return ``(anchors, used_fallback)``. ``anchors`` is the list of
    NORMALISED question anchor strings to require co-occurrence against.
    Specificity tiering:

      tier 1: multi-token proper-noun runs (e.g. "Queen Elizabeth II",
              "French Connection") — most specific
      tier 2: single-token proper-noun runs (e.g. "Canada", "Helena")
      tier 3: question content tokens (stopwords removed) — fallback
              when the question has no proper-noun anchors at all

    The most-specific NON-EMPTY tier is used. ``used_fallback`` is True
    only when tier 3 is the source — that signals to callers that the
    Stage E check for this query is weaker than name-anchored matching."""
    runs = _extract_capitalised_runs(question)
    multi: list[str] = []
    single: list[str] = []
    for r in runs:
        norm = _normalise_entity(r)
        if not norm:
            continue
        if " " in norm:
            multi.append(norm)
        else:
            single.append(norm)
    if multi:
        # Deduplicate while preserving order.
        return list(dict.fromkeys(multi)), False
    if single:
        return list(dict.fromkeys(single)), False
    # Fallback: question has no proper-noun anchors. Use content tokens.
    content = [t for t in _content_tokens(question) if len(t) >= 3]
    return list(dict.fromkeys(content)), True


def _check_answer_anchor(
    answer_clean: str,
    question: str,
    cell_texts_normalised: list[str],
) -> tuple[bool, list[str], list[dict]]:
    """Stage E v2 — every novel answer entity must co-occur in some
    cited cell with at least one question anchor.

    Returns ``(ok, missing_entities, audit_log)`` where ``missing_entities``
    is the list of original-case answer entity spans that failed
    co-occurrence, and ``audit_log`` is a list of per-entity records
    suitable for serialisation alongside the verification decision."""
    answer_runs = _extract_capitalised_runs(answer_clean)
    if not answer_runs:
        return True, [], []

    # Question anchors — the runs we require co-occurrence against.
    q_anchors, used_fallback = _classify_question_anchors(question)
    # The question-as-text — used to drop answer entities that are
    # subspans of question runs (so "Napoleon" in the answer is not
    # treated as novel when the question already says "Napoleon").
    q_norm_full = _normalise_for_anchor(question)

    audit: list[dict] = []
    missing: list[str] = []

    for span in answer_runs:
        norm = _normalise_entity(span)
        if not norm:
            continue
        # Subspan check: an answer run that is wholly contained in the
        # question is not a novel claim.
        if norm in q_norm_full:
            continue

        if not q_anchors:
            # No anchors at all (question had no content tokens either).
            # Without anchors we cannot enforce co-occurrence; pass.
            audit.append({
                "stage": "E",
                "decision": "skip",
                "answer_entity": span,
                "question_anchors": [],
                "entity_found_in_some_cell": None,
                "anchor_found_in_same_cell_as_entity": None,
                "fallback_used": used_fallback,
                "reason": "no_question_anchors_available",
            })
            continue

        entity_found_anywhere = False
        co_located = False
        for cell_norm in cell_texts_normalised:
            if not cell_norm:
                continue
            if norm not in cell_norm:
                continue
            entity_found_anywhere = True
            if any(a in cell_norm for a in q_anchors):
                co_located = True
                break

        if co_located:
            audit.append({
                "stage": "E",
                "decision": "accept",
                "answer_entity": span,
                "question_anchors": list(q_anchors),
                "entity_found_in_some_cell": True,
                "anchor_found_in_same_cell_as_entity": True,
                "fallback_used": used_fallback,
                "reason": "answer_entity_colocated_with_question_anchor",
            })
        else:
            audit.append({
                "stage": "E",
                "decision": "reject",
                "answer_entity": span,
                "question_anchors": list(q_anchors),
                "entity_found_in_some_cell": entity_found_anywhere,
                "anchor_found_in_same_cell_as_entity": False,
                "fallback_used": used_fallback,
                "reason": (
                    "answer_entity_not_colocated_with_question_anchor"
                    if entity_found_anywhere
                    else "answer_entity_absent_from_cells"
                ),
            })
            missing.append(span)

    return (len(missing) == 0), missing, audit


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
    # Stage E: original-case answer entity spans (full proper-noun
    # runs, e.g. "Jean-Philippe Rameau") that failed the v2
    # co-occurrence check. Empty when Stage E passed, was not run, or
    # had nothing to check.
    unanchored_proper_nouns: list[str] = field(default_factory=list)
    # Stage E audit trail: one record per novel answer entity that was
    # evaluated, with decision, anchors checked, and whether the entity
    # was found anywhere in the cells and whether it co-located with a
    # question anchor. Empty when Stage E was not run.
    stage_e_log: list[dict] = field(default_factory=list)
    # Stage F: per-claim cross-cell-splice check report. Empty dict when
    # Stage F was not enabled.
    claim_verifier_report: dict = field(default_factory=dict)

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
            "unanchored_proper_nouns": self.unanchored_proper_nouns,
            "stage_e_log": self.stage_e_log,
            "claim_verifier_report": self.claim_verifier_report,
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
           strict_nonsense: bool = False,
           enable_claim_verification: bool = False) -> VerificationDecision:
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

    ``enable_claim_verification`` enables Stage F (Phase 2): every
    atomic claim's distinctive signals (numerics, proper-noun runs)
    must co-occur in a single cited cell. Off by default; opt-in via
    the answer pipeline. Requires ``cell_ids`` to be supplied so the
    failing claim's supporting cell can be reported. A Stage F
    rejection overrides Stages A/B/E to silence.
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

    # Stage E v2 (opt-in via strict_nonsense): every novel answer
    # entity (proper-noun run not in the question) must co-occur in
    # some cited cell with at least one question anchor. Same-cell
    # co-occurrence prevents the exp24-Q2 failure where a confabulated
    # name happens to appear in an unrelated cell.
    unanchored: list[str] = []
    stage_e_audit: list[dict] = []
    anchor_ok = True
    if strict_nonsense:
        cell_texts_normalised = [_normalise_for_anchor(t) for t in cell_texts]
        anchor_ok, unanchored, stage_e_audit = _check_answer_anchor(
            answer_clean=answer_clean,
            question=question,
            cell_texts_normalised=cell_texts_normalised,
        )

    grounded = coverage_ok and numerics_ok and anchor_ok
    if grounded:
        reason = "coverage above threshold; all numerics cited"
    elif not anchor_ok and coverage_ok and numerics_ok:
        reason = (
            f"answer entity not colocated with question anchor: "
            f"{unanchored[:3]}"
        )
    elif not coverage_ok and not numerics_ok:
        reason = (f"coverage below threshold AND uncited numerics "
                  f"({uncited[:3]})")
    elif not coverage_ok:
        reason = "coverage below threshold (answer drifted from cells)"
    elif not numerics_ok:
        reason = f"uncited numeric/date in answer: {uncited[:3]}"
    else:
        # coverage and numerics pass but anchor failed alongside one of them
        # (defensive — currently unreachable given the branches above).
        reason = (
            f"answer entity not colocated with question anchor: "
            f"{unanchored[:3]}"
        )

    # Stage F (opt-in): claim-level cross-cell-splice check. Runs only
    # when Stages A/B/E have all passed; an earlier-stage rejection has
    # higher diagnostic priority and there is nothing to splice if the
    # answer already failed coverage.
    claim_report: dict = {}
    if enable_claim_verification and grounded:
        # Imported lazily so the verifier module remains importable in
        # environments that don't ship the Stage F dependencies.
        from src.agent.claim_verifier import verify_claims as _verify_claims

        pairs: list[tuple[int, str]]
        if cell_ids is not None and len(cell_ids) == len(cell_texts):
            pairs = [(int(cid), txt) for cid, txt in zip(cell_ids, cell_texts)]
        else:
            pairs = [(idx, txt) for idx, txt in enumerate(cell_texts)]
        f_report = _verify_claims(answer_clean, pairs, question=question)
        claim_report = f_report.as_dict()
        if not f_report.grounded:
            grounded = False
            failed_preview = [
                {
                    "claim": fc.claim_text[:120],
                    "missing": fc.missing_in_best[:3],
                }
                for fc in f_report.failed[:3]
            ]
            reason = (
                f"claim verification failed: "
                f"{f_report.n_rejected}/{f_report.n_claims} claim(s) "
                f"unsupported; {failed_preview}"
            )

    return VerificationDecision(
        grounded=grounded,
        coverage=coverage,
        covered=len(covered),
        total=total,
        threshold=min_coverage,
        reason=reason,
        uncovered_tokens=uncovered[:25],
        uncited_numerics=uncited[:10],
        unanchored_proper_nouns=unanchored[:10],
        stage_e_log=stage_e_audit,
        claim_verifier_report=claim_report,
    )


__all__ = ["VerificationDecision", "verify", "MIN_COVERAGE"]
