"""Note ingestion: deterministic, rule-based extraction of candidate memories.

This module reads a Markdown / plain-text note and proposes candidate memories.
It is intentionally **not** an LLM: extraction is a small set of transparent
rules a reviewer can predict and audit.

Rules:

* Markdown headings define the current section; the heading becomes a context
  tag and is recorded as ``source_section``.
* Bullet points become candidate facts (one bullet -> one atomic proposal).
* Lines containing priority keywords (``decision``, ``result``, ``limitation``,
  ``next``, ``blocked``, ``deferred``, ``approved``, ``failed``, ``validated``)
  are prioritised: they become candidates even when not bulleted and carry a
  higher confidence and a more specific :class:`ProposalKind`.
* Proposals are kept atomic and short (one sentence, capped length).

Nothing here writes a memory. The output is a :class:`ProposalBatch` the
workbench queues for human review.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple

from agent.memory_proposals import (
    MemoryProposal,
    ProposalBatch,
    ProposalKind,
)

# Keywords that promote a line to a high-value candidate and map it to a kind.
# Order matters: the first matching keyword wins.
_KIND_KEYWORDS: List[Tuple[str, ProposalKind]] = [
    ("decision", ProposalKind.DECISION),
    ("decided", ProposalKind.DECISION),
    ("approved", ProposalKind.DECISION),
    ("validated", ProposalKind.RESULT),
    ("result", ProposalKind.RESULT),
    ("failed", ProposalKind.RESULT),
    ("limitation", ProposalKind.LIMITATION),
    ("blocked", ProposalKind.LIMITATION),
    ("deferred", ProposalKind.NEXT_STEP),
    ("next step", ProposalKind.NEXT_STEP),
    ("next", ProposalKind.NEXT_STEP),
    ("todo", ProposalKind.NEXT_STEP),
    ("risk", ProposalKind.RISK),
    ("assumption", ProposalKind.ASSUMPTION),
    ("assume", ProposalKind.ASSUMPTION),
    ("requirement", ProposalKind.REQUIREMENT),
    ("must", ProposalKind.REQUIREMENT),
    ("shall", ProposalKind.REQUIREMENT),
]

# The subset that marks a line as "prioritised" per the spec.
_PRIORITY_KEYWORDS = {
    "decision", "result", "limitation", "next", "blocked", "deferred",
    "approved", "failed", "validated",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^\s*```")

_MAX_LEN = 200


def load_text_file(path: str | Path) -> str:
    """Read a UTF-8 text/Markdown file and return its contents."""
    return Path(path).read_text(encoding="utf-8")


def split_markdown_sections(text: str) -> List[Tuple[Optional[str], int, int]]:
    """Split text into sections by Markdown heading.

    Returns a list of ``(heading, line_start, line_end)`` tuples (1-based,
    inclusive line numbers). ``heading`` is ``None`` for any preamble before the
    first heading. Headings themselves are not included in the body range.
    """
    lines = text.splitlines()
    sections: List[Tuple[Optional[str], int, int]] = []
    current_heading: Optional[str] = None
    body_start = 1  # 1-based line number where the current body begins

    for idx, raw in enumerate(lines, start=1):
        m = _HEADING_RE.match(raw)
        if m:
            # Close the previous section's body (idx-1 is the last body line).
            if idx - 1 >= body_start:
                sections.append((current_heading, body_start, idx - 1))
            current_heading = m.group(2).strip()
            body_start = idx + 1

    last_line = len(lines)
    if last_line >= body_start:
        sections.append((current_heading, body_start, last_line))
    return sections


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug


def _clean_inline(text: str) -> str:
    """Strip common inline Markdown so the proposal text reads plainly."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [t](url) -> t
    text = text.replace("`", "")
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?<!\w)[*_](?=\w)", "", text)  # leading emphasis marks
    return text.strip()


def _atomic(text: str) -> str:
    """Keep a proposal to one short sentence."""
    text = text.strip()
    # First sentence only when the line is long and clearly multi-sentence.
    if len(text) > _MAX_LEN:
        parts = re.split(r"(?<=[.!?])\s+", text)
        if parts:
            text = parts[0]
    if len(text) > _MAX_LEN:
        text = text[:_MAX_LEN].rstrip() + "…"
    return text


def _classify(line_lower: str) -> Tuple[ProposalKind, bool, Optional[str]]:
    """Return (kind, prioritised, matched_keyword) for a line."""
    for keyword, kind in _KIND_KEYWORDS:
        if keyword in line_lower:
            prioritised = any(p in line_lower for p in _PRIORITY_KEYWORDS)
            return kind, prioritised, keyword
    return ProposalKind.FACT, False, None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!?,;:")


def _proposal_id(source_file: str, line_start: int, text: str) -> str:
    key = f"{source_file}|{line_start}|{text}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"prop-{digest}"


def extract_candidate_memories(text: str,
                               source_file: str) -> ProposalBatch:
    """Extract candidate memories from note ``text``.

    Deterministic and offline. A line becomes a proposal if it is a bullet point
    or contains a priority keyword. Headings supply context tags and the
    ``source_section``. Duplicate-looking proposals (same normalised text) are
    flagged in ``reason`` rather than silently merged.
    """
    lines = text.splitlines()
    sections = split_markdown_sections(text)
    batch = ProposalBatch(source_file=source_file)

    seen: dict[str, str] = {}  # normalised text -> first proposal_id
    in_fence = False

    for heading, start, end in sections:
        section_tag = _slug(heading) if heading else None
        for line_no in range(start, end + 1):
            raw = lines[line_no - 1]
            if _FENCE_RE.match(raw):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            stripped = raw.strip()
            if not stripped:
                continue

            bullet = _BULLET_RE.match(raw)
            content = bullet.group(1) if bullet else stripped
            content = _clean_inline(content)
            if not content:
                continue

            kind, prioritised, keyword = _classify(content.lower())
            is_candidate = bool(bullet) or prioritised
            if not is_candidate:
                continue
            # Skip trivially short fragments.
            if len(content.split()) < 2:
                continue

            canonical = _atomic(content)
            tags: List[str] = []
            if section_tag:
                tags.append(section_tag)

            if prioritised:
                confidence = 0.8
            elif keyword:
                confidence = 0.65
            else:
                confidence = 0.55

            reason_bits = []
            if bullet:
                reason_bits.append("bullet point")
            if keyword:
                reason_bits.append(f"keyword '{keyword}'")
            if heading:
                reason_bits.append(f"section '{heading}'")
            reason = "; ".join(reason_bits) if reason_bits else "candidate line"

            norm = _normalize(canonical)
            if norm in seen:
                reason += f" (possible duplicate of {seen[norm]})"

            proposal = MemoryProposal(
                proposal_id=_proposal_id(source_file, line_no, canonical),
                canonical_text=canonical,
                source_file=source_file,
                source_section=heading,
                source_line_start=line_no,
                source_line_end=line_no,
                tags=tags,
                kind=kind,
                confidence=confidence,
                reason=reason,
            )
            if norm not in seen:
                seen[norm] = proposal.proposal_id
            batch.proposals.append(proposal)

    return batch
