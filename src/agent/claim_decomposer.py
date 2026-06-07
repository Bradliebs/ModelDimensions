"""Rule-based answer decomposer for Stage F claim verification.

Splits a generated answer into atomic claim spans for per-claim
grounding checks. The shape is intentionally lexical — sentence /
clause segmentation only, no semantic parse — because Stage F's job
is to catch cross-cell splices, not paraphrase fidelity.

A claim is a sentence (split on ``.``, ``!``, ``?``, ``;``), further
split on free-standing clausal boundaries (`` -- ``, `` — ``, and
comma-followed-by ``and|but|while|whereas|however|though|although``).
Each yielded span carries the original substring plus its
character offset in the input, so a failed Stage F decision can
point at the exact text.

Empty claims, citation-marker-only fragments (e.g. "[51098]."),
and the template self-silence string emitted by the pipeline when
the generator declines ("I drafted an answer but could not ground
it in memory.") are dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Sentence-terminator split. Preserves the rest of the text intact;
# offsets stay valid in the original string because we operate on
# slice positions, not on a re-joined output.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")
# Clausal boundary inside a sentence. ``-- `` and `` — `` are explicit
# dashes; the comma-conjunction pattern is anchored on whitespace and
# the word boundary so we don't split "and" inside a name.
_CLAUSE_SPLIT_RE = re.compile(
    r"\s+--\s+"
    r"|\s+\u2014\s+"
    r"|,\s+(?:and|but|while|whereas|however|though|although)\s+"
)
_CITATION_BRACKET_RE = re.compile(r"\[\s*\d+\s*\]")
# Strings the generator emits when refusing; never a real claim.
_SELF_SILENCE_PATTERNS = (
    "i drafted an answer but could not ground it",
    "no matching memory",
)
# A claim must have at least this many non-whitespace, non-punctuation
# characters to be eligible. Below this is fragment noise.
_MIN_CLAIM_CHARS: int = 8


@dataclass(frozen=True)
class ClaimSpan:
    """One atomic claim extracted from an answer."""

    text: str            # original substring (with original casing/punct)
    start: int           # inclusive char offset in the source answer
    end: int             # exclusive char offset in the source answer

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"bad span offsets: [{self.start}, {self.end})")


def _is_meaningful(fragment: str) -> bool:
    """Reject empty/citation-only/refusal fragments."""
    cleaned = _CITATION_BRACKET_RE.sub("", fragment).strip(" \t\r\n.,;:!?-—")
    if len(cleaned) < _MIN_CLAIM_CHARS:
        return False
    low = cleaned.lower()
    for pat in _SELF_SILENCE_PATTERNS:
        if pat in low:
            return False
    return True


def _iter_sentence_spans(text: str) -> Iterable[tuple[int, int]]:
    """Yield ``(start, end)`` for each sentence-level slice of ``text``."""
    if not text:
        return
    cursor = 0
    for m in _SENT_SPLIT_RE.finditer(text):
        end = m.start()
        if end > cursor:
            yield cursor, end
        cursor = m.end()
    if cursor < len(text):
        yield cursor, len(text)


def _iter_clause_spans(text: str, base: int) -> Iterable[tuple[int, int]]:
    """Yield clause-level ``(start, end)`` offsets *into the source* given
    ``text`` is ``source[base:base+len(text)]``."""
    cursor = 0
    for m in _CLAUSE_SPLIT_RE.finditer(text):
        end = m.start()
        if end > cursor:
            yield base + cursor, base + end
        cursor = m.end()
    if cursor < len(text):
        yield base + cursor, base + len(text)


def decompose(answer: str) -> list[ClaimSpan]:
    """Decompose ``answer`` into atomic claim spans.

    Returns an empty list when the answer is empty or contains nothing
    but refusal / citation fragments.
    """
    if not answer or not answer.strip():
        return []

    out: list[ClaimSpan] = []
    for s_start, s_end in _iter_sentence_spans(answer):
        sentence = answer[s_start:s_end]
        for c_start, c_end in _iter_clause_spans(sentence, base=s_start):
            fragment = answer[c_start:c_end].strip()
            if not _is_meaningful(fragment):
                continue
            # Re-locate the trimmed fragment inside the source so spans
            # are tight (don't include trailing punctuation/whitespace).
            inner_start = answer.find(fragment, c_start, c_end)
            if inner_start < 0:
                # Fallback: trimming changed exact bytes (rare). Use the
                # un-trimmed clause bounds.
                inner_start, inner_end = c_start, c_end
            else:
                inner_end = inner_start + len(fragment)
            out.append(ClaimSpan(text=fragment, start=inner_start, end=inner_end))
    return out


__all__ = ["ClaimSpan", "decompose"]
