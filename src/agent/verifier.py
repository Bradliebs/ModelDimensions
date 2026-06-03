"""Deterministic fact verifier (v1.0-rc2).

Exp 09 showed the limit of threshold-only semantic recall: MiniLM ranks
one-word near-miss flips (``Friday``->``Monday``, ``forty``->``ninety``,
``approved``->``rejected``) *above* genuine paraphrases in cosine space, so no
single threshold can admit paraphrases without also admitting dangerous
near-misses.

This module attacks that failure from the other side. Concept cells stay the
fast candidate substrate; this verifier decides, by deterministic lexical rules,
whether a retrieved candidate actually preserves the same fact as the query. It
never touches geometry, embeddings, or the grounding policy.

Design stance (deliberately conservative):

  * A *material* mismatch (different entity, number, date/weekday, a negation
    flip, or a known antonym flip) yields ``REJECT``.
  * No mismatch but no positive evidence of equivalence yields ``AMBIGUOUS``.
  * ``ACCEPT`` is only returned for an exact normalised match, strong lexical
    containment, or a fixture-defined safe paraphrase.

The asymmetry is intentional: a false ``ACCEPT`` grounds a wrong fact, which is
the failure we most want to avoid. A false ``AMBIGUOUS`` merely refuses a
correct paraphrase, which is safe.
"""
from __future__ import annotations

import re
from typing import FrozenSet, List, Set, Tuple

from slm.schemas import VerificationVerdict

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# Tokens dropped when comparing "content". Kept small and generic.
_STOPWORDS: FrozenSet[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "for", "of", "on", "in", "to", "at", "and", "or", "our", "we", "this",
    "that", "by", "with", "as", "it", "its", "their", "there", "will",
    "have", "has", "had", "from",
})

# Negation / cancellation markers. Asymmetric presence => fact flipped.
_NEGATIONS: FrozenSet[str] = frozenset({
    "not", "never", "no", "none", "cannot", "cant", "wont", "without",
    "failed", "fail", "cancelled", "canceled", "denied", "deny",
    "neither", "nor", "nothing",
})

# Weekdays and months, normalised by surface form.
_WEEKDAYS: FrozenSet[str] = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
})
_MONTHS: FrozenSet[str] = frozenset({
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
})

# Number words -> integer value. Used so "two" and "2" compare equal (avoids
# false rejects on harmless digit/word paraphrases) while real numeric
# substitutions still differ.
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
    # ordinals
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

# Symmetric antonym pairs. If one side has A and the other has B, the fact is
# flipped. Frozen, small, and explicit so behaviour is auditable.
_ANTONYM_PAIRS: FrozenSet[FrozenSet[str]] = frozenset(
    frozenset(p) for p in [
        ("increase", "decrease"), ("increased", "decreased"),
        ("rise", "fall"), ("rose", "fell"), ("up", "down"),
        ("approved", "rejected"), ("approve", "reject"),
        ("accept", "reject"), ("accepted", "rejected"),
        ("safe", "unsafe"), ("secure", "insecure"),
        ("allow", "deny"), ("allowed", "denied"),
        ("grant", "revoke"), ("granted", "revoked"),
        ("enable", "disable"), ("enabled", "disabled"),
        ("open", "closed"), ("start", "stop"),
        ("launch", "recall"), ("budget", "deficit"),
        ("surplus", "deficit"), ("profit", "loss"),
        ("visit", "leave"), ("arrive", "depart"),
        ("located", "relocated"), ("ships", "slips"),
        ("success", "failure"), ("pass", "fail"),
        ("before", "after"), ("buy", "sell"),
    ]
)

# Modal-strength flips: an *obligation* ("must"/"required"/"mandatory"/"shall")
# swapped for a *permission* or *option* ("may"/"might"/"optional") changes the
# fact even though no antonym pair is present. Kept separate from the antonym
# set so the obligation/permission semantics are auditable on their own. The
# "may" surface form also denotes a month; that collision can only cause a
# conservative (safe) false REJECT, never a false ACCEPT.
_MODAL_FLIP_PAIRS: FrozenSet[FrozenSet[str]] = frozenset(
    frozenset(p) for p in [
        ("must", "may"), ("must", "might"),
        ("shall", "may"), ("shall", "might"),
        ("required", "optional"), ("required", "may"),
        ("mandatory", "optional"), ("mandatory", "may"),
    ]
)


# ---------- tokenisation helpers ----------

def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS]


def _numbers(text: str) -> Set[int]:
    """Set of numeric values, from digits and number-words.

    Pure digit runs parse directly; tokens like ``v2`` contribute their digit
    run; number-words map through ``_NUMBER_WORDS``. Returning a set means
    repeated scale words ("thousand") collapse, which is fine for mismatch
    detection.
    """
    vals: Set[int] = set()
    for tok in _tokens(text):
        if tok.isdigit():
            vals.add(int(tok))
            continue
        digits = re.findall(r"\d+", tok)
        for d in digits:
            vals.add(int(d))
        if tok in _NUMBER_WORDS:
            vals.add(_NUMBER_WORDS[tok])
    return vals


def _proper_nouns(text: str) -> Set[str]:
    """Capitalised tokens that look like named entities.

    Excludes the first token of the string (sentence-initial capitalisation is
    not evidence of a proper noun) and excludes weekday/month names, which are
    handled by the date detector. Acronyms and place names survive.
    """
    raw = _TOKEN_RE.findall(text)
    ents: Set[str] = set()
    for i, tok in enumerate(raw):
        if i == 0:
            continue
        if not tok[0].isupper():
            continue
        low = tok.lower()
        if low in _WEEKDAYS or low in _MONTHS:
            continue
        ents.add(low)
    return ents


# ---------- individual deterministic detectors ----------

def detect_entity_mismatch(query_text: str, candidate_text: str) -> bool:
    """True iff query and candidate each name an entity the other does not.

    Conservative: only a genuine *substitution* (each side has a proper noun the
    other lacks) counts. One side merely adding an entity (e.g. "CEO") is not a
    mismatch, so paraphrases that introduce an acronym are not punished.
    """
    q = _proper_nouns(query_text)
    c = _proper_nouns(candidate_text)
    return bool((q - c) and (c - q))


def detect_number_mismatch(query_text: str, candidate_text: str) -> bool:
    """True iff each side carries a numeric value the other does not.

    "forty thousand" vs "ninety thousand" -> {40,1000} vs {90,1000} -> mismatch.
    "v2" vs "version two" -> {2} vs {2} -> no mismatch.
    """
    q = _numbers(query_text)
    c = _numbers(candidate_text)
    return bool((q - c) and (c - q))


def detect_date_or_weekday_mismatch(query_text: str,
                                    candidate_text: str) -> bool:
    """True iff weekdays differ or months differ between the two texts.

    Only fires when both sides mention a weekday (resp. month) and the sets
    differ. Same day or absence on one side does not trigger it.
    """
    q_days = {t for t in _tokens(query_text) if t in _WEEKDAYS}
    c_days = {t for t in _tokens(candidate_text) if t in _WEEKDAYS}
    if q_days and c_days and q_days != c_days:
        return True
    q_mon = {t for t in _tokens(query_text) if t in _MONTHS}
    c_mon = {t for t in _tokens(candidate_text) if t in _MONTHS}
    if q_mon and c_mon and q_mon != c_mon:
        return True
    return False


def detect_negation_flip(query_text: str, candidate_text: str) -> bool:
    """True iff one side carries a negation/cancellation marker the other lacks.

    "rotates every ninety days" vs "never rotates ..." -> flip.
    "backups are stored" vs "backups failed to reach" -> flip.
    """
    q_neg = any(t in _NEGATIONS for t in _tokens(query_text))
    c_neg = any(t in _NEGATIONS for t in _tokens(candidate_text))
    return q_neg != c_neg


def detect_antonym_flip_basic(query_text: str, candidate_text: str) -> bool:
    """True iff query and candidate sit on opposite sides of an antonym pair."""
    q = set(_tokens(query_text))
    c = set(_tokens(candidate_text))
    for pair in _ANTONYM_PAIRS:
        a, b = tuple(pair)
        if (a in q and b in c) or (b in q and a in c):
            return True
    return False


def detect_modal_flip(query_text: str, candidate_text: str) -> bool:
    """True iff query and candidate express different modal obligation strength.

    Catches an obligation/permission swap such as "the server must restart" vs
    "the server may restart", which carries no antonym or negation marker but
    still changes the fact. Symmetric and conservative, like the antonym
    detector. ("should not"/"must not" style flips are handled by the negation
    detector instead.)
    """
    q = set(_tokens(query_text))
    c = set(_tokens(candidate_text))
    for pair in _MODAL_FLIP_PAIRS:
        a, b = tuple(pair)
        if (a in q and b in c) or (b in q and a in c):
            return True
    return False


# Fixture of safe paraphrases that may be ACCEPTed even without exact match or
# containment. Stored as normalised content-token frozensets so trivial wording
# differences match. Intentionally tiny and explicit: paraphrase acceptance is
# opt-in, never inferred.
def _norm_key(text: str) -> FrozenSet[str]:
    return frozenset(_content_tokens(text))


_SAFE_PARAPHRASES: FrozenSet[Tuple[FrozenSet[str], FrozenSet[str]]] = frozenset({
    (
        _norm_key("the client meeting is on tuesday morning"),
        _norm_key("there is a morning client meeting on tuesday"),
    ),
    (
        _norm_key("nightly backups are stored in the frankfurt region"),
        _norm_key("we keep nightly backups in the frankfurt region"),
    ),
})


def _is_fixture_paraphrase(query_text: str, candidate_text: str) -> bool:
    qk, ck = _norm_key(query_text), _norm_key(candidate_text)
    return (qk, ck) in _SAFE_PARAPHRASES or (ck, qk) in _SAFE_PARAPHRASES


def _strong_containment(query_text: str, candidate_text: str) -> bool:
    """True iff one text's content tokens are a subset of the other's.

    Requires at least three shared-or-subset content tokens so short, generic
    overlaps do not trigger acceptance. Mismatch detectors run first, so this
    only sees pairs already free of entity/number/date/negation/antonym
    conflicts.
    """
    q = set(_content_tokens(query_text))
    c = set(_content_tokens(candidate_text))
    if not q or not c:
        return False
    smaller, larger = (q, c) if len(q) <= len(c) else (c, q)
    return len(smaller) >= 3 and smaller.issubset(larger)


_DETECTORS = (
    detect_entity_mismatch,
    detect_number_mismatch,
    detect_date_or_weekday_mismatch,
    detect_negation_flip,
    detect_antonym_flip_basic,
    detect_modal_flip,
)


def verify_candidate(query_text: str,
                     candidate_text: str) -> VerificationVerdict:
    """Decide whether ``candidate_text`` preserves the fact in ``query_text``.

    Order matters: any material mismatch short-circuits to ``REJECT`` before
    acceptance is even considered.
    """
    for detector in _DETECTORS:
        if detector(query_text, candidate_text):
            return VerificationVerdict.REJECT

    if _norm_key(query_text) == _norm_key(candidate_text):
        return VerificationVerdict.ACCEPT
    if _strong_containment(query_text, candidate_text):
        return VerificationVerdict.ACCEPT
    if _is_fixture_paraphrase(query_text, candidate_text):
        return VerificationVerdict.ACCEPT

    return VerificationVerdict.AMBIGUOUS
