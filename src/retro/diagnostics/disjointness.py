"""Corpus disjointness verification.

The "Reader, not memoriser" claim only holds if the training corpus and the
concept-cell bank do not share content. This module makes that property
*measurable*. Without it, a model that scores well on the Ignorance Test
could still be passing only because the random-neighbour condition is
hitting paragraphs the model already memorised from training.

Approach
========

We compute set-overlap at two granularities and report both, since each can
fail independently:

* **Paragraph fingerprint set** — SHA-1 of normalised paragraph text. Catches
  near-exact duplication that survives whitespace/case differences. Two
  Wikipedia paragraphs that match here are the same fact in the same words,
  regardless of which dump they were sourced from.

* **Article-title set** — only available when the corpus has titles. Catches
  the looser failure: corpus contains a *different* paragraph from the
  same Wikipedia article, which still leaks knowledge.

Both sets are bounded by a user-supplied policy threshold (default: any
overlap > 0.1% of the corpus is a hard fail). Operators can tighten or
relax via ``DisjointnessPolicy``.

The bank-side fingerprint set is built once from the SQLite bank's
``source_texts`` table; callers should cache it on disk for repeated runs
on different corpus shards.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Set


# Whitespace, internal punctuation that varies across dumps, and Unicode
# zero-width glue all get collapsed before hashing.
_WHITESPACE_RE = re.compile(r"\s+")
_ZWS_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")


def normalise_paragraph(text: str) -> str:
    """Canonical form for hashing. Stable across whitespace/case variants."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    cleaned = _ZWS_RE.sub("", text)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip().lower()
    return cleaned


def fingerprint(text: str) -> str:
    """SHA-1 hex of the normalised paragraph. Empty input is rejected."""
    n = normalise_paragraph(text)
    if not n:
        raise ValueError("cannot fingerprint empty paragraph")
    return hashlib.sha1(n.encode("utf-8")).hexdigest()


def normalise_title(title: str) -> str:
    """Looser canonical form for article titles."""
    if not isinstance(title, str):
        raise TypeError(f"expected str, got {type(title).__name__}")
    return _WHITESPACE_RE.sub(" ", title).strip().lower()


@dataclass(frozen=True)
class DisjointnessPolicy:
    """Configurable bounds for what counts as a disjointness failure."""

    max_paragraph_overlap_ratio: float = 0.001  # 0.1% of corpus paragraphs
    max_title_overlap_ratio: float = 0.0  # zero tolerance for article reuse
    min_paragraph_chars: int = 20  # ignore near-empty fragments


@dataclass(frozen=True)
class DisjointnessReport:
    """Output of one disjointness check."""

    bank_paragraph_count: int
    corpus_paragraph_count: int
    corpus_paragraph_overlap: int
    corpus_paragraph_overlap_ratio: float
    bank_title_count: int | None
    corpus_title_count: int | None
    corpus_title_overlap: int | None
    corpus_title_overlap_ratio: float | None
    policy: DisjointnessPolicy
    paragraph_pass: bool
    title_pass: bool
    sample_overlapping_fingerprints: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.paragraph_pass and self.title_pass

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["policy"] = asdict(self.policy)
        d["passed"] = self.passed
        return d


def iter_bank_source_texts(db_path: Path, *, batch_size: int = 10_000) -> Iterator[str]:
    """Stream all paragraph texts out of a cc_service-format SQLite bank.

    The bank schema (per ``src/cc_service/persistence.py``) carries the
    original source text in ``source_texts.text``, keyed by ``cell_id``.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"bank database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT text FROM source_texts")
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                return
            for (text,) in rows:
                if text:
                    yield text
    finally:
        conn.close()


def build_fingerprint_set(
    paragraphs: Iterable[str],
    *,
    min_chars: int = 20,
) -> Set[str]:
    """Build a SHA-1 fingerprint set from an iterable of paragraph strings.

    Paragraphs shorter than ``min_chars`` post-normalisation are skipped:
    they generate spurious collisions (e.g. one-word headers).
    """
    out: Set[str] = set()
    for p in paragraphs:
        n = normalise_paragraph(p)
        if len(n) < min_chars:
            continue
        out.add(hashlib.sha1(n.encode("utf-8")).hexdigest())
    return out


def build_title_set(titles: Iterable[str]) -> Set[str]:
    """Build a normalised title set; empty titles are dropped."""
    out: Set[str] = set()
    for t in titles:
        n = normalise_title(t)
        if n:
            out.add(n)
    return out


def check_disjointness(
    *,
    bank_fingerprints: Set[str],
    corpus_fingerprints: Set[str],
    policy: DisjointnessPolicy = DisjointnessPolicy(),
    bank_titles: Set[str] | None = None,
    corpus_titles: Set[str] | None = None,
    sample_overlap: int = 5,
) -> DisjointnessReport:
    """Pure set-arithmetic check; safe under unit tests with synthetic inputs.

    Numeric trace:
        bank_fingerprints = {a, b, c, d}, corpus_fingerprints = {c, d, e, f}
        overlap = {c, d}, |overlap| = 2, |corpus| = 4
        ratio = 2 / 4 = 0.5
        0.5 > policy.max_paragraph_overlap_ratio (0.001) -> paragraph_pass = False
    """
    if not isinstance(bank_fingerprints, set):
        raise TypeError("bank_fingerprints must be a set")
    if not isinstance(corpus_fingerprints, set):
        raise TypeError("corpus_fingerprints must be a set")
    corpus_n = len(corpus_fingerprints)
    if corpus_n == 0:
        raise ValueError("corpus_fingerprints is empty; nothing to verify")

    para_overlap = bank_fingerprints & corpus_fingerprints
    para_ratio = len(para_overlap) / corpus_n
    paragraph_pass = para_ratio <= policy.max_paragraph_overlap_ratio

    title_count_b: int | None = None
    title_count_c: int | None = None
    title_overlap_n: int | None = None
    title_ratio: float | None = None
    title_pass = True
    if corpus_titles is not None and bank_titles is not None:
        title_count_b = len(bank_titles)
        title_count_c = len(corpus_titles)
        if title_count_c == 0:
            title_pass = True
            title_overlap_n = 0
            title_ratio = 0.0
        else:
            title_overlap = bank_titles & corpus_titles
            title_overlap_n = len(title_overlap)
            title_ratio = title_overlap_n / title_count_c
            title_pass = title_ratio <= policy.max_title_overlap_ratio

    return DisjointnessReport(
        bank_paragraph_count=len(bank_fingerprints),
        corpus_paragraph_count=corpus_n,
        corpus_paragraph_overlap=len(para_overlap),
        corpus_paragraph_overlap_ratio=para_ratio,
        bank_title_count=title_count_b,
        corpus_title_count=title_count_c,
        corpus_title_overlap=title_overlap_n,
        corpus_title_overlap_ratio=title_ratio,
        policy=policy,
        paragraph_pass=paragraph_pass,
        title_pass=title_pass,
        sample_overlapping_fingerprints=sorted(para_overlap)[:sample_overlap],
    )


def render_report_markdown(report: DisjointnessReport) -> str:
    lines: List[str] = []
    lines.append("# Corpus disjointness report")
    lines.append("")
    lines.append("## Paragraph-level")
    lines.append("")
    lines.append(f"- Bank fingerprints: **{report.bank_paragraph_count:,}**")
    lines.append(f"- Corpus fingerprints: **{report.corpus_paragraph_count:,}**")
    lines.append(
        f"- Overlap: **{report.corpus_paragraph_overlap:,}** "
        f"({report.corpus_paragraph_overlap_ratio:.6f} of corpus)"
    )
    lines.append(
        f"- Policy max ratio: {report.policy.max_paragraph_overlap_ratio:.6f}"
    )
    lines.append(f"- Paragraph gate: **{'PASS' if report.paragraph_pass else 'FAIL'}**")
    lines.append("")
    if report.corpus_title_count is not None:
        lines.append("## Article-title level")
        lines.append("")
        lines.append(f"- Bank titles: **{report.bank_title_count:,}**")
        lines.append(f"- Corpus titles: **{report.corpus_title_count:,}**")
        lines.append(
            f"- Title overlap: **{report.corpus_title_overlap:,}** "
            f"({(report.corpus_title_overlap_ratio or 0):.6f} of corpus)"
        )
        lines.append(
            f"- Policy max ratio: {report.policy.max_title_overlap_ratio:.6f}"
        )
        lines.append(f"- Title gate: **{'PASS' if report.title_pass else 'FAIL'}**")
        lines.append("")
    lines.append(f"## Overall verdict: **{'PASS' if report.passed else 'FAIL'}**")
    if report.sample_overlapping_fingerprints:
        lines.append("")
        lines.append("Sample overlapping fingerprints (first few):")
        for fp in report.sample_overlapping_fingerprints:
            lines.append(f"- `{fp}`")
    return "\n".join(lines) + "\n"
