"""Deterministic, read-only PDF parse-quality and chunk-preview layer (v6.5).

This module shows exactly what extracted PDF text and proposed chunks would
look like BEFORE any knowledge-pack creation or retrieval indexing happens.
It is strictly a PREVIEW. It reuses the v6.4 governed PDF intake adapter
(``agent.pdf_intake_adapter``) for extraction and the intake decision, then
layers page-quality diagnostics and deterministic chunk proposals on top.

PREVIEW ONLY. This module deliberately does NOT:

* create or update a knowledge pack;
* create or update a retrieval index;
* write to the source registry;
* write to the MemoryLedger;
* create or apply source proposals;
* create or apply memory proposals;
* perform OCR;
* call an LLM;
* summarise or paraphrase extracted text;
* alter retrieval, ranking, grounding, or chat routing;
* persist full extracted text unless an explicit ``--out`` path is provided;
* mark previewed chunks as approved automatically.

Everything here is deterministic: the same file and the same settings always
produce byte-identical output, including chunk identifiers.

The docstring above is intentionally explicit about the forbidden operations
so the import-purity test can strip it before scanning the module body.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from agent.pdf_intake_adapter import (
    PdfFindingCode,
    PdfIntakeResult,
    assess_pdf_intake,
    sample_pdf_text,
)

# --------------------------------------------------------------------------- #
# Tunable, documented thresholds. All deterministic and easy to change.
# --------------------------------------------------------------------------- #

# Chunking defaults (characters).
DEFAULT_CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 1500
DEFAULT_OVERLAP = 150
MIN_CHUNK_CHARS = 200        # below this a chunk is flagged "too short"
CONSOLE_PREVIEW_CHARS = 240  # bounded text sample shown in console / markdown

# Page-quality thresholds.
SPARSE_PAGE_MAX_CHARS = 80      # at or below -> page is "sparse"
DENSE_PAGE_MIN_CHARS = 3000     # at or above -> page is "dense"
SHORT_LINE_CHARS = 35           # lines shorter than this count as "short"
HIGH_WHITESPACE_RATIO = 0.5     # suspicious whitespace share
HIGH_SHORT_LINE_RATIO = 0.6     # share of short lines that looks fragmented
HIGH_FRAGMENT_RATIO = 0.5       # share of non-sentence-terminated lines
HIGH_NONPRINTABLE_RATIO = 0.05
HIGH_REPLACEMENT_RATIO = 0.02

# Page-quality band cut-offs (deterministic score in [0, 1]).
PAGE_QUALITY_GOOD = 0.80
PAGE_QUALITY_ACCEPTABLE = 0.55
PAGE_QUALITY_POOR = 0.25

# Structural diagnostics.
NEAR_DUPLICATE_JACCARD = 0.85   # conservative near-duplicate threshold
HEADER_FOOTER_MIN_PAGES = 2     # a header/footer must repeat on >= this many pages
HEADER_FOOTER_MIN_RATIO = 0.6   # ...or on >= this share of pages
TABLE_RUN_SPACES = 2            # >= this many spaces signals a column gap
TABLE_MIN_LINES = 2             # number of table-like lines to flag a page
MULTICOLUMN_GAP_SPACES = 4      # a single wide gap suggests two columns
MULTICOLUMN_MIN_LINES = 2

# Readiness thresholds (advisory only).
MAX_UNUSABLE_PAGE_RATIO = 0.25
MAX_POOR_PAGE_RATIO = 0.5

REPLACEMENT_CHAR = "\ufffd"

# Sensitivity vocabularies (kept local so this module stays self-contained).
_CONFIDENTIAL_MARKERS = (
    "confidential",
    "do not distribute",
    "do-not-distribute",
    "proprietary and confidential",
    "company confidential",
)
_RESTRICTED_MARKERS = (
    "restricted",
    "internal use only",
    "internal-use-only",
    "not for distribution",
    "for internal use",
    "classified",
)
_PII_KEYWORDS = (
    "social security number",
    "passport number",
    "date of birth",
    "credit card number",
    "national insurance number",
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PAGE_NUMBER_ONLY_RE = re.compile(r"^(?:page\s+)?\d+$|^[-\u2013\u2014]\s*\d+\s*[-\u2013\u2014]$")
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?(?=\s|$)")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Enumerations.
# --------------------------------------------------------------------------- #


class PdfPageQualityBand(str, Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class PdfPreviewStatus(str, Enum):
    PREVIEW_ONLY = "preview_only"
    NEEDS_REVIEW = "needs_review"


class PdfChunkWarningCode(str, Enum):
    CHUNK_TOO_SHORT = "chunk_too_short"
    CHUNK_TOO_LONG = "chunk_too_long"
    CHUNK_FRAGMENTED = "chunk_fragmented"
    CHUNK_STARTS_MID_SENTENCE = "chunk_starts_mid_sentence"
    CHUNK_ENDS_MID_SENTENCE = "chunk_ends_mid_sentence"
    CHUNK_REPEATED_HEADER = "chunk_repeated_header"
    CHUNK_REPEATED_FOOTER = "chunk_repeated_footer"
    CHUNK_DUPLICATE = "chunk_duplicate"
    CHUNK_NEAR_DUPLICATE = "chunk_near_duplicate"
    CHUNK_TABLE_RISK = "chunk_table_risk"
    CHUNK_MULTICOLUMN_RISK = "chunk_multicolumn_risk"
    CHUNK_SPARSE = "chunk_sparse"
    CHUNK_BOILERPLATE_HEAVY = "chunk_boilerplate_heavy"
    CHUNK_MALFORMED_UNICODE = "chunk_malformed_unicode"
    CHUNK_POSSIBLE_PII = "chunk_possible_pii"
    CHUNK_CONFIDENTIAL_MARKER = "chunk_confidential_marker"
    CHUNK_CROSSES_PAGE_BOUNDARY = "chunk_crosses_page_boundary"
    CHUNK_LOW_SOURCE_QUALITY = "chunk_low_source_quality"
    CHUNK_REQUIRES_HUMAN_REVIEW = "chunk_requires_human_review"


# Warning codes that always force a human review before any future import.
_REVIEW_FORCING = frozenset(
    {
        PdfChunkWarningCode.CHUNK_POSSIBLE_PII,
        PdfChunkWarningCode.CHUNK_CONFIDENTIAL_MARKER,
        PdfChunkWarningCode.CHUNK_REQUIRES_HUMAN_REVIEW,
    }
)


# --------------------------------------------------------------------------- #
# Data models (frozen, deterministic, JSON-serialisable).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PdfPageText:
    """Extracted text for a single page, with provenance for traceability."""

    file_hash: str
    file_name: str
    page_number: int
    text: str
    char_count: int
    char_offset_start: int
    char_offset_end: int
    extraction_method: str = "text_layer"
    extraction_error: str = ""

    @property
    def is_empty(self) -> bool:
        return self.char_count == 0

    def to_dict(self) -> dict:
        return {
            "file_hash": self.file_hash,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "char_count": self.char_count,
            "char_offset_start": self.char_offset_start,
            "char_offset_end": self.char_offset_end,
            "extraction_method": self.extraction_method,
            "extraction_error": self.extraction_error,
            "is_empty": self.is_empty,
        }


@dataclass(frozen=True)
class PdfPageQuality:
    """Deterministic per-page quality metrics."""

    page_number: int
    char_count: int
    word_count: int
    line_count: int
    average_line_length: float
    empty_line_ratio: float
    suspicious_whitespace_ratio: float
    replacement_character_count: int
    non_printable_character_count: int
    repeated_line_count: int
    likely_header_footer_count: int
    sentence_fragment_count: int
    short_line_ratio: float
    text_density: float
    quality_band: PdfPageQualityBand

    @property
    def is_sparse(self) -> bool:
        return 0 < self.char_count <= SPARSE_PAGE_MAX_CHARS

    @property
    def is_empty(self) -> bool:
        return self.char_count == 0

    @property
    def is_dense(self) -> bool:
        return self.char_count >= DENSE_PAGE_MIN_CHARS

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "average_line_length": self.average_line_length,
            "empty_line_ratio": self.empty_line_ratio,
            "suspicious_whitespace_ratio": self.suspicious_whitespace_ratio,
            "replacement_character_count": self.replacement_character_count,
            "non_printable_character_count": self.non_printable_character_count,
            "repeated_line_count": self.repeated_line_count,
            "likely_header_footer_count": self.likely_header_footer_count,
            "sentence_fragment_count": self.sentence_fragment_count,
            "short_line_ratio": self.short_line_ratio,
            "text_density": self.text_density,
            "quality_band": self.quality_band.value,
            "is_empty": self.is_empty,
            "is_sparse": self.is_sparse,
            "is_dense": self.is_dense,
        }


@dataclass(frozen=True)
class PdfChunkWarning:
    """A single advisory warning attached to a proposed chunk."""

    chunk_id: str
    code: PdfChunkWarningCode
    message: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "code": self.code.value,
            "message": self.message,
        }


@dataclass(frozen=True)
class PdfChunkProposal:
    """A deterministic, preview-only proposed chunk. Never an approved chunk."""

    chunk_id: str
    file_hash: str
    file_name: str
    page_start: int
    page_end: int
    char_offset_start: int
    char_offset_end: int
    char_count: int
    text: str
    text_preview: str
    extraction_method: str
    crosses_page_boundary: bool
    warnings: Tuple[PdfChunkWarning, ...] = ()
    preview_status: PdfPreviewStatus = PdfPreviewStatus.PREVIEW_ONLY

    @property
    def warning_codes(self) -> Tuple[str, ...]:
        return tuple(w.code.value for w in self.warnings)

    @property
    def requires_human_review(self) -> bool:
        return any(w.code in _REVIEW_FORCING for w in self.warnings)

    def to_dict(self, include_full_text: bool = False) -> dict:
        payload = {
            "chunk_id": self.chunk_id,
            "file_hash": self.file_hash,
            "file_name": self.file_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_offset_start": self.char_offset_start,
            "char_offset_end": self.char_offset_end,
            "char_count": self.char_count,
            "extraction_method": self.extraction_method,
            "crosses_page_boundary": self.crosses_page_boundary,
            "preview_status": self.preview_status.value,
            "requires_human_review": self.requires_human_review,
            "warning_codes": list(self.warning_codes),
            "warnings": [w.to_dict() for w in self.warnings],
            "text_preview": self.text_preview,
        }
        if include_full_text:
            payload["text"] = self.text
        return payload


@dataclass(frozen=True)
class PdfChunkPreviewSummary:
    """Aggregate, advisory metrics for a single PDF preview."""

    page_count: int
    pages_with_text: int
    empty_page_count: int
    poor_quality_page_count: int
    unusable_page_count: int
    proposed_chunk_count: int
    warning_count: int
    duplicate_chunk_count: int
    near_duplicate_chunk_count: int
    fragmented_chunk_count: int
    table_risk_chunk_count: int
    possible_pii_chunk_count: int
    confidential_marker_chunk_count: int
    average_chunk_chars: float
    minimum_chunk_chars: int
    maximum_chunk_chars: int
    chunk_coverage_ratio: float
    review_required_count: int
    preview_ready_for_import: bool

    def to_dict(self) -> dict:
        return {
            "page_count": self.page_count,
            "pages_with_text": self.pages_with_text,
            "empty_page_count": self.empty_page_count,
            "poor_quality_page_count": self.poor_quality_page_count,
            "unusable_page_count": self.unusable_page_count,
            "proposed_chunk_count": self.proposed_chunk_count,
            "warning_count": self.warning_count,
            "duplicate_chunk_count": self.duplicate_chunk_count,
            "near_duplicate_chunk_count": self.near_duplicate_chunk_count,
            "fragmented_chunk_count": self.fragmented_chunk_count,
            "table_risk_chunk_count": self.table_risk_chunk_count,
            "possible_pii_chunk_count": self.possible_pii_chunk_count,
            "confidential_marker_chunk_count": self.confidential_marker_chunk_count,
            "average_chunk_chars": self.average_chunk_chars,
            "minimum_chunk_chars": self.minimum_chunk_chars,
            "maximum_chunk_chars": self.maximum_chunk_chars,
            "chunk_coverage_ratio": self.chunk_coverage_ratio,
            "review_required_count": self.review_required_count,
            "preview_ready_for_import": self.preview_ready_for_import,
        }


@dataclass(frozen=True)
class PdfChunkPreview:
    """The full preview for one PDF: pages, quality, diagnostics, and chunks."""

    file_hash: str
    file_name: str
    file_path: str
    page_count: int
    chunk_size: int
    overlap: int
    max_chunk_size: int
    respect_page_boundaries: bool
    intake_decision: str
    intake_blocked: bool
    has_text_layer: bool
    requires_ocr: bool
    extraction_quality_band: str
    pages: Tuple[PdfPageText, ...]
    page_quality: Tuple[PdfPageQuality, ...]
    chunks: Tuple[PdfChunkProposal, ...]
    repeated_headers: Tuple[str, ...]
    repeated_footers: Tuple[str, ...]
    duplicate_page_pairs: Tuple[Tuple[int, int], ...]
    near_duplicate_page_pairs: Tuple[Tuple[int, int], ...]
    summary: PdfChunkPreviewSummary
    preview_status: PdfPreviewStatus
    intake_findings: Tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: _utc_now().isoformat())

    def to_dict(self, include_full_text: bool = False) -> dict:
        return {
            "_record": "pdf_chunk_preview",
            "file_hash": self.file_hash,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "page_count": self.page_count,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "max_chunk_size": self.max_chunk_size,
            "respect_page_boundaries": self.respect_page_boundaries,
            "intake_decision": self.intake_decision,
            "intake_blocked": self.intake_blocked,
            "has_text_layer": self.has_text_layer,
            "requires_ocr": self.requires_ocr,
            "extraction_quality_band": self.extraction_quality_band,
            "preview_status": self.preview_status.value,
            "intake_findings": list(self.intake_findings),
            "repeated_headers": list(self.repeated_headers),
            "repeated_footers": list(self.repeated_footers),
            "duplicate_page_pairs": [list(p) for p in self.duplicate_page_pairs],
            "near_duplicate_page_pairs": [list(p) for p in self.near_duplicate_page_pairs],
            "pages": [p.to_dict() for p in self.pages],
            "page_quality": [q.to_dict() for q in self.page_quality],
            "chunks": [c.to_dict(include_full_text=include_full_text) for c in self.chunks],
            "summary": self.summary.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Page extraction (reuses the v6.4 reader; read-only, OCR-free).
# --------------------------------------------------------------------------- #


def _normalize_newlines(text: str) -> str:
    """Normalise line endings only. No other whitespace cleanup."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_pdf_pages(path, *, file_hash: str = "", file_name: str = "") -> List[PdfPageText]:
    """Extract per-page text in page order, preserving page numbers.

    Empty pages are never silently dropped: they appear with empty text. Only
    line endings are normalised; no destructive whitespace cleanup, no OCR, no
    text rewriting. Per-page extraction errors are recorded, not swallowed.
    """
    p = Path(path)
    if not file_name:
        file_name = p.name
    try:
        data = p.read_bytes()
    except (OSError, ValueError) as exc:
        return [
            PdfPageText(
                file_hash=file_hash,
                file_name=file_name,
                page_number=1,
                text="",
                char_count=0,
                char_offset_start=0,
                char_offset_end=0,
                extraction_method="unreadable",
                extraction_error=f"read_error: {exc.__class__.__name__}",
            )
        ]

    try:
        page_texts, _ = sample_pdf_text(data)
    except Exception as exc:  # defensive: never raise out of a preview
        return [
            PdfPageText(
                file_hash=file_hash,
                file_name=file_name,
                page_number=1,
                text="",
                char_count=0,
                char_offset_start=0,
                char_offset_end=0,
                extraction_method="parse_error",
                extraction_error=f"parse_error: {exc.__class__.__name__}",
            )
        ]

    pages: List[PdfPageText] = []
    offset = 0
    for index, raw in enumerate(page_texts, start=1):
        text = _normalize_newlines(raw or "")
        char_count = len(text)
        pages.append(
            PdfPageText(
                file_hash=file_hash,
                file_name=file_name,
                page_number=index,
                text=text,
                char_count=char_count,
                char_offset_start=offset,
                char_offset_end=offset + char_count,
                extraction_method="text_layer" if char_count else "empty",
            )
        )
        # +2 accounts for the "\n\n" separator inserted between pages in doc text.
        offset += char_count + 2
    return pages


# --------------------------------------------------------------------------- #
# Page-quality analysis.
# --------------------------------------------------------------------------- #


def _nonempty_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def analyse_page_quality(
    page: PdfPageText,
    *,
    repeated_lines: frozenset = frozenset(),
) -> PdfPageQuality:
    """Compute deterministic quality metrics for one page."""
    text = page.text
    char_count = len(text)
    lines = text.split("\n")
    line_count = len(lines) if text else 0
    words = re.findall(r"\S+", text)
    word_count = len(words)

    nonempty = [ln.strip() for ln in lines if ln.strip()]
    empty_line_count = line_count - len(nonempty)
    empty_line_ratio = round(empty_line_count / line_count, 4) if line_count else 0.0

    whitespace_chars = sum(1 for ch in text if ch.isspace())
    suspicious_whitespace_ratio = round(whitespace_chars / char_count, 4) if char_count else 0.0

    replacement_character_count = text.count(REPLACEMENT_CHAR)
    non_printable_character_count = sum(
        1 for ch in text if not ch.isprintable() and ch not in "\n\t"
    )

    seen: Dict[str, int] = {}
    for ln in nonempty:
        seen[ln] = seen.get(ln, 0) + 1
    repeated_line_count = sum(count for count in seen.values() if count > 1)

    likely_header_footer_count = sum(1 for ln in nonempty if ln in repeated_lines)

    sentence_fragment_count = sum(
        1
        for ln in nonempty
        if not _SENTENCE_END_RE.search(ln) and not _PAGE_NUMBER_ONLY_RE.match(ln)
    )

    short_lines = sum(1 for ln in nonempty if len(ln) < SHORT_LINE_CHARS)
    short_line_ratio = round(short_lines / len(nonempty), 4) if nonempty else 0.0

    non_whitespace = char_count - whitespace_chars
    text_density = round(non_whitespace / char_count, 4) if char_count else 0.0

    average_line_length = round(char_count / line_count, 2) if line_count else 0.0

    band = _page_quality_band(
        char_count=char_count,
        suspicious_whitespace_ratio=suspicious_whitespace_ratio,
        short_line_ratio=short_line_ratio,
        sentence_fragment_count=sentence_fragment_count,
        nonempty_count=len(nonempty),
        replacement_character_count=replacement_character_count,
        non_printable_character_count=non_printable_character_count,
    )

    return PdfPageQuality(
        page_number=page.page_number,
        char_count=char_count,
        word_count=word_count,
        line_count=line_count,
        average_line_length=average_line_length,
        empty_line_ratio=empty_line_ratio,
        suspicious_whitespace_ratio=suspicious_whitespace_ratio,
        replacement_character_count=replacement_character_count,
        non_printable_character_count=non_printable_character_count,
        repeated_line_count=repeated_line_count,
        likely_header_footer_count=likely_header_footer_count,
        sentence_fragment_count=sentence_fragment_count,
        short_line_ratio=short_line_ratio,
        text_density=text_density,
        quality_band=band,
    )


def _page_quality_band(
    *,
    char_count: int,
    suspicious_whitespace_ratio: float,
    short_line_ratio: float,
    sentence_fragment_count: int,
    nonempty_count: int,
    replacement_character_count: int,
    non_printable_character_count: int,
) -> PdfPageQualityBand:
    if char_count == 0:
        return PdfPageQualityBand.UNUSABLE

    score = 1.0
    if char_count and replacement_character_count / char_count > HIGH_REPLACEMENT_RATIO:
        score *= 0.3
    if char_count and non_printable_character_count / char_count > HIGH_NONPRINTABLE_RATIO:
        score *= 0.4
    if suspicious_whitespace_ratio > HIGH_WHITESPACE_RATIO:
        score *= 0.6
    if short_line_ratio > HIGH_SHORT_LINE_RATIO:
        score *= 0.7
    if nonempty_count and sentence_fragment_count / nonempty_count > HIGH_FRAGMENT_RATIO:
        score *= 0.8

    band = _band_from_score(score)
    # Sparse pages cannot be better than "poor", regardless of clean text.
    if char_count <= SPARSE_PAGE_MAX_CHARS and band in (
        PdfPageQualityBand.GOOD,
        PdfPageQualityBand.ACCEPTABLE,
    ):
        return PdfPageQualityBand.POOR
    return band


def _band_from_score(score: float) -> PdfPageQualityBand:
    if score >= PAGE_QUALITY_GOOD:
        return PdfPageQualityBand.GOOD
    if score >= PAGE_QUALITY_ACCEPTABLE:
        return PdfPageQualityBand.ACCEPTABLE
    if score >= PAGE_QUALITY_POOR:
        return PdfPageQualityBand.POOR
    return PdfPageQualityBand.UNUSABLE


# --------------------------------------------------------------------------- #
# Document-level structural diagnostics.
# --------------------------------------------------------------------------- #


def detect_repeated_headers_footers(
    pages: Sequence[PdfPageText],
) -> Tuple[List[str], List[str]]:
    """Return (repeated_headers, repeated_footers) across pages.

    A header is the first non-empty line of a page; a footer is the last. A
    line is "repeated" when it appears in that position on enough pages.
    """
    page_count = len(pages)
    if page_count < HEADER_FOOTER_MIN_PAGES:
        return [], []

    threshold = max(HEADER_FOOTER_MIN_PAGES, int(round(HEADER_FOOTER_MIN_RATIO * page_count)))
    header_counts: Dict[str, int] = {}
    footer_counts: Dict[str, int] = {}
    for page in pages:
        lines = _nonempty_lines(page.text)
        if not lines:
            continue
        first = lines[0]
        last = lines[-1]
        if not _PAGE_NUMBER_ONLY_RE.match(first):
            header_counts[first] = header_counts.get(first, 0) + 1
        if not _PAGE_NUMBER_ONLY_RE.match(last):
            footer_counts[last] = footer_counts.get(last, 0) + 1

    headers = sorted(line for line, count in header_counts.items() if count >= threshold)
    footers = sorted(
        line for line, count in footer_counts.items() if count >= threshold and line not in headers
    )
    return headers, footers


def _normalized_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _collapsed(text: str) -> str:
    return " ".join(text.split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_duplicate_pages(
    pages: Sequence[PdfPageText],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return (duplicate_pairs, near_duplicate_pairs) of page numbers.

    Near-duplicates use a conservative Jaccard threshold so that genuinely
    distinct pages are never mislabelled.
    """
    duplicates: List[Tuple[int, int]] = []
    near: List[Tuple[int, int]] = []
    collapsed = [_collapsed(p.text) for p in pages]
    word_sets = [frozenset(_normalized_words(p.text)) for p in pages]
    for i in range(len(pages)):
        if not collapsed[i]:
            continue
        for j in range(i + 1, len(pages)):
            if not collapsed[j]:
                continue
            if collapsed[i] == collapsed[j]:
                duplicates.append((pages[i].page_number, pages[j].page_number))
            elif _jaccard(word_sets[i], word_sets[j]) >= NEAR_DUPLICATE_JACCARD:
                near.append((pages[i].page_number, pages[j].page_number))
    return duplicates, near


def _looks_like_table(text: str) -> bool:
    table_lines = 0
    for ln in text.split("\n"):
        if len(re.findall(r" {%d,}" % TABLE_RUN_SPACES, ln)) >= 2 or "\t" in ln:
            table_lines += 1
    return table_lines >= TABLE_MIN_LINES


def _looks_multicolumn(text: str) -> bool:
    gap_lines = 0
    pattern = re.compile(r"\S {%d,}\S" % MULTICOLUMN_GAP_SPACES)
    for ln in text.split("\n"):
        if pattern.search(ln):
            gap_lines += 1
    return gap_lines >= MULTICOLUMN_MIN_LINES


# --------------------------------------------------------------------------- #
# Deterministic chunk proposals.
# --------------------------------------------------------------------------- #


def _chunk_id(
    file_hash: str,
    chunk_size: int,
    overlap: int,
    page_start: int,
    page_end: int,
    offset_start: int,
    offset_end: int,
) -> str:
    raw = "|".join(
        str(part)
        for part in (
            file_hash,
            chunk_size,
            overlap,
            page_start,
            page_end,
            offset_start,
            offset_end,
        )
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"pdfchk-{digest}"


def _slice_chunks(text: str, target: int, max_size: int, overlap: int) -> List[Tuple[int, int]]:
    """Return (start, end) char spans over ``text``.

    Boundaries prefer paragraph breaks, then sentence ends, then whitespace,
    and never split a word. Spans are trimmed of leading/trailing whitespace.
    Overlap re-includes trailing context but is snapped to a word boundary.
    """
    spans: List[Tuple[int, int]] = []
    n = len(text)
    if n == 0:
        return spans
    floor = max(1, min(target // 2, max_size // 2))
    start = 0
    while start < n:
        # Skip leading whitespace so spans begin on a word boundary.
        while start < n and text[start].isspace():
            start += 1
        if start >= n:
            break
        hard_end = min(start + max_size, n)
        ideal_end = min(start + target, n)
        if ideal_end >= n:
            end = n
        else:
            end = _boundary(text, start, ideal_end, hard_end, floor)
        # Trim trailing whitespace from the span.
        real_end = end
        while real_end > start and text[real_end - 1].isspace():
            real_end -= 1
        if real_end <= start:
            real_end = end
        spans.append((start, real_end))
        if real_end >= n:
            break
        next_start = real_end - overlap
        if next_start <= start:
            next_start = real_end
        else:
            next_start = _snap_forward(text, next_start)
            if next_start <= start:
                next_start = real_end
        start = next_start
    return spans


def _boundary(text: str, start: int, ideal_end: int, hard_end: int, floor: int) -> int:
    lower = start + floor
    # 1) Paragraph break (blank line) is the strongest preference.
    para = text.rfind("\n\n", lower, hard_end)
    if para != -1:
        return para
    # 2) Sentence end within the window (allow slight overshoot past ideal).
    best = -1
    for match in _SENTENCE_END_RE.finditer(text, start, hard_end):
        pos = match.end()
        if pos >= lower:
            best = pos
    if best != -1:
        return best
    # 3) Last whitespace at or before the ideal end.
    ws = _rfind_whitespace(text, lower, ideal_end)
    if ws != -1:
        return ws
    # 4) Whitespace anywhere up to the hard limit.
    ws = _rfind_whitespace(text, start + 1, hard_end)
    if ws != -1:
        return ws
    # 5) Single very long token: hard cut at the ideal end.
    return ideal_end


def _rfind_whitespace(text: str, lo: int, hi: int) -> int:
    for i in range(hi - 1, lo - 1, -1):
        if text[i].isspace():
            return i
    return -1


def _snap_forward(text: str, index: int) -> int:
    n = len(text)
    if index <= 0 or index >= n:
        return index
    # If we are inside a word, advance to the next whitespace boundary.
    if not text[index].isspace() and not text[index - 1].isspace():
        while index < n and not text[index].isspace():
            index += 1
    while index < n and text[index].isspace():
        index += 1
    return index


def _page_for_offset(spans: Sequence[Tuple[int, int, int]], offset: int) -> int:
    """Map a doc-text offset to a page number. ``spans`` are (page, start, end)."""
    chosen = spans[0][0] if spans else 1
    for page_number, g_start, g_end in spans:
        if g_start <= offset < g_end:
            return page_number
        if offset >= g_end:
            chosen = page_number
    return chosen


def propose_pdf_chunks(
    pages: Sequence[PdfPageText],
    *,
    file_hash: str,
    file_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    respect_page_boundaries: bool = True,
) -> List[Tuple[int, int, int, int, int, str, bool]]:
    """Produce deterministic raw chunk records before quality assessment.

    Each record is (page_start, page_end, offset_start, offset_end, char_count,
    text, crosses_page_boundary). When ``respect_page_boundaries`` is True
    (default) chunks never merge unrelated pages. When False, consecutive pages
    are concatenated and any chunk spanning a page break is flagged.
    """
    page_spans: List[Tuple[int, int, int]] = [
        (p.page_number, p.char_offset_start, p.char_offset_end) for p in pages
    ]
    doc_text = "\n\n".join(p.text for p in pages)
    records: List[Tuple[int, int, int, int, int, str, bool]] = []

    if respect_page_boundaries:
        for page in pages:
            if not page.text.strip():
                continue
            for local_start, local_end in _slice_chunks(
                page.text, chunk_size, max_chunk_size, overlap
            ):
                g_start = page.char_offset_start + local_start
                g_end = page.char_offset_start + local_end
                text = page.text[local_start:local_end]
                records.append(
                    (
                        page.page_number,
                        page.page_number,
                        g_start,
                        g_end,
                        len(text),
                        text,
                        False,
                    )
                )
    else:
        for g_start, g_end in _slice_chunks(doc_text, chunk_size, max_chunk_size, overlap):
            text = doc_text[g_start:g_end]
            page_start = _page_for_offset(page_spans, g_start)
            page_end = _page_for_offset(page_spans, max(g_start, g_end - 1))
            records.append(
                (
                    page_start,
                    page_end,
                    g_start,
                    g_end,
                    len(text),
                    text,
                    page_start != page_end,
                )
            )
    return records


# --------------------------------------------------------------------------- #
# Chunk quality assessment.
# --------------------------------------------------------------------------- #


def _marker_hits(lowered: str, markers: Sequence[str]) -> bool:
    return any(marker in lowered for marker in markers)


def assess_chunk_quality(
    chunk_id: str,
    text: str,
    *,
    crosses_page_boundary: bool,
    repeated_headers: frozenset,
    repeated_footers: frozenset,
    source_band: PdfPageQualityBand,
    duplicate: bool,
    near_duplicate: bool,
) -> List[PdfChunkWarning]:
    """Return the deterministic advisory warnings for a single chunk."""
    warnings: List[PdfChunkWarning] = []

    def add(code: PdfChunkWarningCode, message: str) -> None:
        warnings.append(PdfChunkWarning(chunk_id=chunk_id, code=code, message=message))

    char_count = len(text)
    lowered = text.lower()

    if char_count < MIN_CHUNK_CHARS:
        add(PdfChunkWarningCode.CHUNK_TOO_SHORT, f"chunk has {char_count} chars (< {MIN_CHUNK_CHARS})")
    if char_count > max(MAX_CHUNK_SIZE, MIN_CHUNK_CHARS):
        add(PdfChunkWarningCode.CHUNK_TOO_LONG, f"chunk has {char_count} chars (> {MAX_CHUNK_SIZE})")

    if crosses_page_boundary:
        add(PdfChunkWarningCode.CHUNK_CROSSES_PAGE_BOUNDARY, "chunk spans more than one page")

    stripped = text.strip()
    if stripped:
        first = stripped[0]
        if first.isalpha() and first.islower():
            add(PdfChunkWarningCode.CHUNK_STARTS_MID_SENTENCE, "chunk begins mid-sentence")
        last = stripped[-1]
        if last.isalnum():
            add(PdfChunkWarningCode.CHUNK_ENDS_MID_SENTENCE, "chunk ends mid-sentence")

    nonempty = _nonempty_lines(text)
    if nonempty:
        short = sum(1 for ln in nonempty if len(ln) < SHORT_LINE_CHARS)
        fragments = sum(
            1
            for ln in nonempty
            if not _SENTENCE_END_RE.search(ln) and not _PAGE_NUMBER_ONLY_RE.match(ln)
        )
        if short / len(nonempty) > HIGH_SHORT_LINE_RATIO and fragments / len(nonempty) > HIGH_FRAGMENT_RATIO:
            add(PdfChunkWarningCode.CHUNK_FRAGMENTED, "chunk text looks fragmented")
        boilerplate = sum(
            1 for ln in nonempty if ln in repeated_headers or ln in repeated_footers
        )
        if boilerplate and boilerplate / len(nonempty) >= 0.5:
            add(PdfChunkWarningCode.CHUNK_BOILERPLATE_HEAVY, "chunk is mostly boilerplate")
        if any(ln in repeated_headers for ln in nonempty):
            add(PdfChunkWarningCode.CHUNK_REPEATED_HEADER, "chunk contains a repeated header")
        if any(ln in repeated_footers for ln in nonempty):
            add(PdfChunkWarningCode.CHUNK_REPEATED_FOOTER, "chunk contains a repeated footer")

    if _looks_like_table(text):
        add(PdfChunkWarningCode.CHUNK_TABLE_RISK, "chunk contains table-like layout")
    if _looks_multicolumn(text):
        add(PdfChunkWarningCode.CHUNK_MULTICOLUMN_RISK, "chunk may have multi-column reading order")

    if char_count <= SPARSE_PAGE_MAX_CHARS:
        add(PdfChunkWarningCode.CHUNK_SPARSE, "chunk has very little text")

    if REPLACEMENT_CHAR in text:
        add(PdfChunkWarningCode.CHUNK_MALFORMED_UNICODE, "chunk contains replacement characters")

    if _EMAIL_RE.search(text) or _SSN_RE.search(text) or _marker_hits(lowered, _PII_KEYWORDS):
        add(PdfChunkWarningCode.CHUNK_POSSIBLE_PII, "chunk may contain personal data")

    if _marker_hits(lowered, _CONFIDENTIAL_MARKERS) or _marker_hits(lowered, _RESTRICTED_MARKERS):
        add(PdfChunkWarningCode.CHUNK_CONFIDENTIAL_MARKER, "chunk carries a confidentiality marker")

    if duplicate:
        add(PdfChunkWarningCode.CHUNK_DUPLICATE, "chunk duplicates an earlier chunk")
    elif near_duplicate:
        add(PdfChunkWarningCode.CHUNK_NEAR_DUPLICATE, "chunk closely resembles an earlier chunk")

    if source_band in (PdfPageQualityBand.POOR, PdfPageQualityBand.UNUSABLE):
        add(PdfChunkWarningCode.CHUNK_LOW_SOURCE_QUALITY, f"source page quality is {source_band.value}")

    if any(w.code in _REVIEW_FORCING for w in warnings):
        add(PdfChunkWarningCode.CHUNK_REQUIRES_HUMAN_REVIEW, "chunk requires human review before any import")

    return warnings


# --------------------------------------------------------------------------- #
# Preview assembly.
# --------------------------------------------------------------------------- #


def calculate_preview_summary(
    pages: Sequence[PdfPageText],
    page_quality: Sequence[PdfPageQuality],
    chunks: Sequence[PdfChunkProposal],
    *,
    doc_text_len: int,
    intake_blocked: bool,
    has_text_layer: bool,
    requires_ocr: bool,
    extraction_quality_band: str,
    intake_sensitive: bool,
) -> PdfChunkPreviewSummary:
    page_count = len(pages)
    pages_with_text = sum(1 for q in page_quality if q.char_count > 0)
    empty_page_count = sum(1 for q in page_quality if q.is_empty)
    poor_count = sum(1 for q in page_quality if q.quality_band == PdfPageQualityBand.POOR)
    unusable_count = sum(1 for q in page_quality if q.quality_band == PdfPageQualityBand.UNUSABLE)

    warning_count = sum(len(c.warnings) for c in chunks)
    duplicate_chunks = sum(1 for c in chunks if _has(c, PdfChunkWarningCode.CHUNK_DUPLICATE))
    near_dupe_chunks = sum(1 for c in chunks if _has(c, PdfChunkWarningCode.CHUNK_NEAR_DUPLICATE))
    fragmented_chunks = sum(1 for c in chunks if _has(c, PdfChunkWarningCode.CHUNK_FRAGMENTED))
    table_chunks = sum(1 for c in chunks if _has(c, PdfChunkWarningCode.CHUNK_TABLE_RISK))
    pii_chunks = sum(1 for c in chunks if _has(c, PdfChunkWarningCode.CHUNK_POSSIBLE_PII))
    confidential_chunks = sum(
        1 for c in chunks if _has(c, PdfChunkWarningCode.CHUNK_CONFIDENTIAL_MARKER)
    )
    review_required = sum(1 for c in chunks if c.requires_human_review)

    char_counts = [c.char_count for c in chunks]
    if char_counts:
        average_chunk_chars = round(sum(char_counts) / len(char_counts), 2)
        minimum_chunk_chars = min(char_counts)
        maximum_chunk_chars = max(char_counts)
    else:
        average_chunk_chars = 0.0
        minimum_chunk_chars = 0
        maximum_chunk_chars = 0

    coverage = _coverage_ratio(chunks, doc_text_len)

    ready = _is_preview_ready(
        intake_blocked=intake_blocked,
        has_text_layer=has_text_layer,
        requires_ocr=requires_ocr,
        extraction_quality_band=extraction_quality_band,
        intake_sensitive=intake_sensitive,
        page_count=page_count,
        unusable_count=unusable_count,
        poor_count=poor_count,
        proposed_chunk_count=len(chunks),
        review_required=review_required,
    )

    return PdfChunkPreviewSummary(
        page_count=page_count,
        pages_with_text=pages_with_text,
        empty_page_count=empty_page_count,
        poor_quality_page_count=poor_count,
        unusable_page_count=unusable_count,
        proposed_chunk_count=len(chunks),
        warning_count=warning_count,
        duplicate_chunk_count=duplicate_chunks,
        near_duplicate_chunk_count=near_dupe_chunks,
        fragmented_chunk_count=fragmented_chunks,
        table_risk_chunk_count=table_chunks,
        possible_pii_chunk_count=pii_chunks,
        confidential_marker_chunk_count=confidential_chunks,
        average_chunk_chars=average_chunk_chars,
        minimum_chunk_chars=minimum_chunk_chars,
        maximum_chunk_chars=maximum_chunk_chars,
        chunk_coverage_ratio=coverage,
        review_required_count=review_required,
        preview_ready_for_import=ready,
    )


def _has(chunk: PdfChunkProposal, code: PdfChunkWarningCode) -> bool:
    return any(w.code == code for w in chunk.warnings)


def _coverage_ratio(chunks: Sequence[PdfChunkProposal], doc_text_len: int) -> float:
    if doc_text_len <= 0:
        return 0.0
    intervals = sorted((c.char_offset_start, c.char_offset_end) for c in chunks)
    covered = 0
    cur_start = -1
    cur_end = -1
    for start, end in intervals:
        if start > cur_end:
            if cur_end > cur_start:
                covered += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end > cur_start:
        covered += cur_end - cur_start
    return round(min(1.0, covered / doc_text_len), 4)


def _is_preview_ready(
    *,
    intake_blocked: bool,
    has_text_layer: bool,
    requires_ocr: bool,
    extraction_quality_band: str,
    intake_sensitive: bool,
    page_count: int,
    unusable_count: int,
    poor_count: int,
    proposed_chunk_count: int,
    review_required: int,
) -> bool:
    """Conservative, advisory readiness signal. Never an approval."""
    if intake_blocked or not has_text_layer or requires_ocr or intake_sensitive:
        return False
    if extraction_quality_band not in ("acceptable", "good"):
        return False
    if proposed_chunk_count == 0 or review_required > 0:
        return False
    if page_count <= 0:
        return False
    if unusable_count / page_count > MAX_UNUSABLE_PAGE_RATIO:
        return False
    if (poor_count + unusable_count) / page_count > MAX_POOR_PAGE_RATIO:
        return False
    return True


def _intake_is_sensitive(intake: PdfIntakeResult) -> bool:
    sensitive_codes = {
        PdfFindingCode.PDF_POSSIBLE_PII,
        PdfFindingCode.PDF_CONFIDENTIAL_MARKER,
        PdfFindingCode.PDF_RESTRICTED_MARKER,
    }
    return any(finding.code in sensitive_codes for finding in intake.findings)


def preview_pdf_chunks(
    path,
    *,
    intake_result: Optional[PdfIntakeResult] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    respect_page_boundaries: bool = True,
    source_url: str = "",
    owner: str = "",
    permission: str = "",
    intended_use: str = "",
    authority_level: str = "",
    now: Optional[datetime] = None,
) -> PdfChunkPreview:
    """Build a deterministic, preview-only chunk preview for one PDF.

    Reuses the v6.4 intake assessment for the file hash, text-layer status,
    OCR requirement, extraction quality, and the governance decision. Nothing
    here mutates any registry, memory, pack, or retrieval state.
    """
    intake = intake_result or assess_pdf_intake(
        path,
        source_url=source_url,
        owner=owner,
        permission=permission,
        intended_use=intended_use,
        authority_level=authority_level,
        now=now,
    )
    metadata = intake.candidate.metadata
    file_hash = metadata.file_hash
    file_name = metadata.file_name

    if metadata.read_error or intake.blocked:
        pages: List[PdfPageText] = []
    else:
        pages = extract_pdf_pages(path, file_hash=file_hash, file_name=file_name)

    headers, footers = detect_repeated_headers_footers(pages)
    header_set = frozenset(headers)
    footer_set = frozenset(footers)
    repeated_lines = header_set | footer_set

    page_quality = tuple(
        analyse_page_quality(page, repeated_lines=repeated_lines) for page in pages
    )
    band_by_page = {q.page_number: q.quality_band for q in page_quality}

    duplicate_pairs, near_pairs = detect_duplicate_pages(pages)

    raw_records = propose_pdf_chunks(
        pages,
        file_hash=file_hash,
        file_name=file_name,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunk_size=max_chunk_size,
        respect_page_boundaries=respect_page_boundaries,
    )

    chunks = _finalize_chunks(
        raw_records,
        file_hash=file_hash,
        file_name=file_name,
        chunk_size=chunk_size,
        overlap=overlap,
        header_set=header_set,
        footer_set=footer_set,
        band_by_page=band_by_page,
    )

    doc_text_len = len("\n\n".join(p.text for p in pages))
    intake_sensitive = _intake_is_sensitive(intake)
    chunk_sensitive = any(
        _has(c, PdfChunkWarningCode.CHUNK_POSSIBLE_PII)
        or _has(c, PdfChunkWarningCode.CHUNK_CONFIDENTIAL_MARKER)
        for c in chunks
    )

    summary = calculate_preview_summary(
        pages,
        page_quality,
        chunks,
        doc_text_len=doc_text_len,
        intake_blocked=intake.blocked,
        has_text_layer=metadata.has_text_layer,
        requires_ocr=metadata.requires_ocr,
        extraction_quality_band=metadata.parse_quality.extraction_quality_band,
        intake_sensitive=intake_sensitive or chunk_sensitive,
    )

    preview_status = (
        PdfPreviewStatus.NEEDS_REVIEW
        if summary.review_required_count or not summary.preview_ready_for_import
        else PdfPreviewStatus.PREVIEW_ONLY
    )

    assessed_at = (now or _utc_now()).isoformat()

    return PdfChunkPreview(
        file_hash=file_hash,
        file_name=file_name,
        file_path=str(Path(path)),
        page_count=len(pages),
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunk_size=max_chunk_size,
        respect_page_boundaries=respect_page_boundaries,
        intake_decision=intake.decision.value,
        intake_blocked=intake.blocked,
        has_text_layer=metadata.has_text_layer,
        requires_ocr=metadata.requires_ocr,
        extraction_quality_band=metadata.parse_quality.extraction_quality_band,
        pages=tuple(pages),
        page_quality=page_quality,
        chunks=tuple(chunks),
        repeated_headers=tuple(headers),
        repeated_footers=tuple(footers),
        duplicate_page_pairs=tuple(duplicate_pairs),
        near_duplicate_page_pairs=tuple(near_pairs),
        summary=summary,
        preview_status=preview_status,
        intake_findings=tuple(f.code.value for f in intake.findings),
        generated_at=assessed_at,
    )


def _finalize_chunks(
    raw_records,
    *,
    file_hash: str,
    file_name: str,
    chunk_size: int,
    overlap: int,
    header_set: frozenset,
    footer_set: frozenset,
    band_by_page: Dict[int, PdfPageQualityBand],
) -> List[PdfChunkProposal]:
    chunks: List[PdfChunkProposal] = []
    seen_collapsed: Dict[str, str] = {}
    seen_word_sets: List[Tuple[str, frozenset]] = []

    for page_start, page_end, off_start, off_end, char_count, text, crosses in raw_records:
        chunk_id = _chunk_id(
            file_hash, chunk_size, overlap, page_start, page_end, off_start, off_end
        )
        collapsed = _collapsed(text)
        word_set = frozenset(_normalized_words(text))
        duplicate = collapsed in seen_collapsed
        near_duplicate = False
        if not duplicate:
            for _, prev_words in seen_word_sets:
                if _jaccard(word_set, prev_words) >= NEAR_DUPLICATE_JACCARD:
                    near_duplicate = True
                    break

        worst_band = PdfPageQualityBand.GOOD
        for page_number in range(page_start, page_end + 1):
            band = band_by_page.get(page_number, PdfPageQualityBand.GOOD)
            if _band_rank(band) > _band_rank(worst_band):
                worst_band = band

        warnings = assess_chunk_quality(
            chunk_id,
            text,
            crosses_page_boundary=crosses,
            repeated_headers=header_set,
            repeated_footers=footer_set,
            source_band=worst_band,
            duplicate=duplicate,
            near_duplicate=near_duplicate,
        )

        requires_review = any(w.code in _REVIEW_FORCING for w in warnings)
        status = PdfPreviewStatus.NEEDS_REVIEW if requires_review else PdfPreviewStatus.PREVIEW_ONLY

        chunks.append(
            PdfChunkProposal(
                chunk_id=chunk_id,
                file_hash=file_hash,
                file_name=file_name,
                page_start=page_start,
                page_end=page_end,
                char_offset_start=off_start,
                char_offset_end=off_end,
                char_count=char_count,
                text=text,
                text_preview=_bounded(text),
                extraction_method="text_layer",
                crosses_page_boundary=crosses,
                warnings=tuple(warnings),
                preview_status=status,
            )
        )
        if not duplicate:
            seen_collapsed[collapsed] = chunk_id
            seen_word_sets.append((chunk_id, word_set))

    return chunks


def _band_rank(band: PdfPageQualityBand) -> int:
    order = {
        PdfPageQualityBand.GOOD: 0,
        PdfPageQualityBand.ACCEPTABLE: 1,
        PdfPageQualityBand.POOR: 2,
        PdfPageQualityBand.UNUSABLE: 3,
    }
    return order[band]


def _bounded(text: str, limit: int = CONSOLE_PREVIEW_CHARS) -> str:
    collapsed = text.strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "\u2026"


# --------------------------------------------------------------------------- #
# Renderers.
# --------------------------------------------------------------------------- #


def chunk_preview_to_json(preview: PdfChunkPreview, *, include_full_text: bool = False) -> str:
    return json.dumps(
        preview.to_dict(include_full_text=include_full_text),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


def render_chunk_preview_markdown(preview: PdfChunkPreview) -> str:
    """Render a bounded, preview-only Markdown report.

    The header states clearly that this is PREVIEW ONLY and NOT APPROVED FOR
    IMPORT. Any possible-PII or confidentiality warnings are surfaced BEFORE
    the text previews so a reviewer sees them first.
    """
    s = preview.summary
    lines: List[str] = []
    lines.append("# PDF Chunk Preview (PREVIEW ONLY — NOT APPROVED FOR IMPORT)")
    lines.append("")
    lines.append(
        "> This is a read-only parse-quality and chunk preview. No knowledge pack, "
        "retrieval index, source registry, or memory state has been created or changed."
    )
    lines.append("")
    lines.append(f"- **File**: {preview.file_name}")
    lines.append(f"- **SHA-256**: {preview.file_hash}")
    lines.append(f"- **Pages**: {preview.page_count}")
    lines.append(f"- **Intake decision**: {preview.intake_decision}")
    lines.append(f"- **Has text layer**: {preview.has_text_layer}")
    lines.append(f"- **Requires OCR**: {preview.requires_ocr}")
    lines.append(f"- **Extraction quality**: {preview.extraction_quality_band}")
    lines.append(f"- **Chunk size / overlap**: {preview.chunk_size} / {preview.overlap}")
    lines.append(f"- **Respect page boundaries**: {preview.respect_page_boundaries}")
    lines.append(
        f"- **Preview ready for import (advisory only)**: {s.preview_ready_for_import}"
    )
    lines.append("")

    # Sensitivity warnings first.
    sensitive = [
        c
        for c in preview.chunks
        if _has(c, PdfChunkWarningCode.CHUNK_POSSIBLE_PII)
        or _has(c, PdfChunkWarningCode.CHUNK_CONFIDENTIAL_MARKER)
    ]
    if sensitive:
        lines.append("## ⚠ Sensitivity warnings (review before any import)")
        lines.append("")
        for chunk in sensitive:
            flagged = [
                w.code.value
                for w in chunk.warnings
                if w.code
                in (
                    PdfChunkWarningCode.CHUNK_POSSIBLE_PII,
                    PdfChunkWarningCode.CHUNK_CONFIDENTIAL_MARKER,
                )
            ]
            lines.append(
                f"- `{chunk.chunk_id}` (page {chunk.page_start}): {', '.join(flagged)}"
            )
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    for key, value in s.to_dict().items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    lines.append("## Page quality")
    lines.append("")
    lines.append("| Page | Chars | Words | Band | Short-line ratio | Whitespace ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for q in preview.page_quality:
        lines.append(
            f"| {q.page_number} | {q.char_count} | {q.word_count} | {q.quality_band.value} "
            f"| {q.short_line_ratio} | {q.suspicious_whitespace_ratio} |"
        )
    lines.append("")

    if preview.repeated_headers or preview.repeated_footers:
        lines.append("## Repeated headers / footers")
        lines.append("")
        for header in preview.repeated_headers:
            lines.append(f"- header: `{header}`")
        for footer in preview.repeated_footers:
            lines.append(f"- footer: `{footer}`")
        lines.append("")

    lines.append("## Proposed chunks")
    lines.append("")
    for chunk in preview.chunks:
        pages = (
            f"page {chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pages {chunk.page_start}-{chunk.page_end}"
        )
        lines.append(f"### `{chunk.chunk_id}` ({pages}, {chunk.char_count} chars)")
        lines.append("")
        if chunk.warning_codes:
            lines.append(f"- warnings: {', '.join(chunk.warning_codes)}")
        lines.append(f"- status: {chunk.preview_status.value}")
        lines.append("")
        lines.append("```text")
        lines.append(chunk.text_preview)
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_chunk_preview_report(preview: PdfChunkPreview, out_path, *, fmt: str = "markdown") -> Path:
    """Write the preview report to ``out_path``. The ONLY durable write.

    Full extracted text is persisted only when ``fmt == 'json'`` (explicit
    machine-readable export). The Markdown report contains bounded previews
    only. No temporary preview files are created.
    """
    path = Path(out_path)
    if fmt == "json":
        content = chunk_preview_to_json(preview, include_full_text=True)
    else:
        content = render_chunk_preview_markdown(preview)
    path.write_text(content, encoding="utf-8")
    return path
