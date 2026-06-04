"""v6.4 Governed PDF Intake Adapter — deterministic, read-only PDF assessment.

This module inspects a *local* PDF file and returns a deterministic governance
assessment: provenance, permission, parseability, sensitivity, freshness, and
intended use. It decides whether a PDF may even be *considered* for an eval pack,
a knowledge pack, the source registry, a memory proposal, or chat retrieval.

This slice is **assessment-only**. It never:

* creates document fragments, imports PDF text into retrieval, or updates any
  retrieval index,
* performs optical character recognition or any image-text recovery,
* writes to the MemoryLedger,
* writes to the source registry,
* creates or applies source/memory proposals,
* changes ranking, grounding, or chat routing,
* adds autonomous actions,
* trusts embedded PDF metadata as verified truth.

It reads only a short, bounded text sample for parse-quality scoring; it does not
summarise, embed, or fragment the document, and it never calls a model. It
mutates durable project state only when the caller passes an explicit ``--out``
path to :func:`write_pdf_assessment_report`, and even then it writes *only* an
assessment report. The module imports no memory/source/proposal writer and no
retrieval/index writer, so it cannot mutate that state even by accident.

It reuses the v6.2A Clean Data Intake contract: the shared
:class:`agent.data_intake.DataIntakeDecision`, :class:`agent.data_intake.IntakeLane`,
:class:`agent.data_intake.IntakeSeverity`, and
:class:`agent.data_intake.ExternalDatasetCandidate`, plus the same conservative,
worst-finding-wins severity to decision mapping. PDFs carry document-specific
risks (encryption, missing text layer, scanned images, extraction quality) that
have no dataset-metadata equivalent, so PDF-specific finding codes are defined
here while the decision tiers and routing remain the shared ones.

Core invariants carried forward:

* Uploaded PDFs must not become trusted knowledge automatically.
* ``approved_for_eval`` is a distinct, weaker tier than
  ``approved_for_knowledge``.
* Missing provenance, missing permission, possible PII, confidential/restricted
  markers, encryption, scanned/image-only content, and poor extraction quality
  can never auto-classify as ``approved_for_knowledge``.
* Embedded PDF metadata (``/Title``, ``/Author`` …) is informational only and is
  always flagged unverified.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from agent.data_intake import (
    DataIntakeDecision,
    ExternalDatasetCandidate,
    IntakeLane,
    IntakeSeverity,
)

# --------------------------------------------------------------------------- #
# Thresholds (all deterministic; documented and easy to change).
# --------------------------------------------------------------------------- #

SAMPLE_CHAR_LIMIT = 2000  # bounded deterministic text sample for parse scoring.
TOO_LARGE_BYTES = 50_000_000  # 50 MB cap for an initial import.
SPARSE_TEXT_MIN_CHARS = 200  # fewer extracted chars than this is "sparse".
LOW_DENSITY_CHARS_PER_PAGE = 100  # avg below this with images looks scanned.
TARGET_CHARS_PER_PAGE = 200  # density reference for the quality score.
STALE_AFTER_DAYS = 1095  # ~3 years (mirrors the v6.2A intake lane).
HIGH_WHITESPACE_RATIO = 0.5  # over this share of whitespace looks broken.
MAX_TEXT_CHARS = 200_000  # hard cap on retained extracted text (read-only).

# Extraction-quality score bands (0.0-1.0).
QUALITY_GOOD = 0.80
QUALITY_ACCEPTABLE = 0.55
QUALITY_POOR = 0.25  # below this is "unusable".

REPLACEMENT_CHAR = "\ufffd"

_INTERNAL_AUTHORITY = frozenset({"internal", "official", "owned"})
_EVAL_INTENDED_USE = frozenset({
    "eval", "evaluation", "benchmark", "test", "testing", "validation",
})

_CONFIDENTIAL_MARKERS: Tuple[str, ...] = (
    "confidential", "do not distribute", "do-not-distribute",
    "proprietary and confidential", "company confidential",
)
_RESTRICTED_MARKERS: Tuple[str, ...] = (
    "restricted", "internal use only", "internal-use-only",
    "not for distribution", "for internal use", "classified",
)
_PII_KEYWORDS: Tuple[str, ...] = (
    "social security number", "passport number", "date of birth",
    "credit card number", "national insurance number",
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Low-level PDF structure patterns (operate on raw bytes).
_OBJ_RE = re.compile(rb"(\d+)\s+0\s+obj(.*?)endobj", re.S)
_PAGE_RE = re.compile(rb"/Type\s*/Page(?![sA-Za-z])")
_PAGES_COUNT_RE = re.compile(rb"/Count\s+(\d+)")
_CONTENTS_SINGLE_RE = re.compile(rb"/Contents\s+(\d+)\s+0\s+R")
_CONTENTS_ARRAY_RE = re.compile(rb"/Contents\s*\[([^\]]*)\]")
_REF_RE = re.compile(rb"(\d+)\s+0\s+R")
_STREAM_RE = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)
_LITERAL_STR_RE = re.compile(rb"\(((?:[^()\\]|\\.)*)\)")
_INFO_REF_RE = re.compile(rb"/Info\s+(\d+)\s+0\s+R")
_IMAGE_RE = re.compile(rb"/Subtype\s*/Image\b")
_ACROFORM_RE = re.compile(rb"/AcroForm\b")
_EMBEDDED_FILE_RE = re.compile(rb"/EmbeddedFile\b")
_ENCRYPT_RE = re.compile(rb"/Encrypt\b")
_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


# --------------------------------------------------------------------------- #
# Enums (reuse the v6.2A decision tiers; PDF-specific finding codes).
# --------------------------------------------------------------------------- #


class PdfFindingCode(str, Enum):
    """Stable, auditable reason codes for PDF intake findings."""

    PDF_UNREADABLE = "pdf_unreadable"
    PDF_ENCRYPTED = "pdf_encrypted"
    PDF_PASSWORD_REQUIRED = "pdf_password_required"
    PDF_MISSING_PROVENANCE = "pdf_missing_provenance"
    PDF_MISSING_PERMISSION = "pdf_missing_permission"
    PDF_NO_TEXT_LAYER = "pdf_no_text_layer"
    PDF_REQUIRES_OCR = "pdf_requires_ocr"
    PDF_SCANNED_IMAGE_ONLY = "pdf_scanned_image_only"
    PDF_LOW_TEXT_EXTRACTION_QUALITY = "pdf_low_text_extraction_quality"
    PDF_SPARSE_TEXT = "pdf_sparse_text"
    PDF_TOO_LARGE_FOR_INITIAL_IMPORT = "pdf_too_large_for_initial_import"
    PDF_POSSIBLE_PII = "pdf_possible_pii"
    PDF_CONFIDENTIAL_MARKER = "pdf_confidential_marker"
    PDF_RESTRICTED_MARKER = "pdf_restricted_marker"
    PDF_STALE_BY_METADATA = "pdf_stale_by_metadata"
    PDF_NO_CREATION_DATE = "pdf_no_creation_date"
    PDF_EVAL_ONLY_INTENDED_USE = "pdf_eval_only_intended_use"
    PDF_TABLE_HEAVY = "pdf_table_heavy"
    PDF_IMAGE_HEAVY = "pdf_image_heavy"
    PDF_FORM_HEAVY = "pdf_form_heavy"
    PDF_EMBEDDED_FILES_PRESENT = "pdf_embedded_files_present"
    PDF_DUPLICATE_HEADERS_FOOTERS_RISK = "pdf_duplicate_headers_footers_risk"
    PDF_METADATA_UNVERIFIED = "pdf_metadata_unverified"
    PDF_APPROVED_OFFICIAL_SOURCE = "pdf_approved_official_source"
    PDF_APPROVED_INTERNAL_SOURCE = "pdf_approved_internal_source"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


_SEVERITY_RANK = {
    IntakeSeverity.BLOCK: 5,
    IntakeSeverity.QUARANTINE: 4,
    IntakeSeverity.REVIEW: 3,
    IntakeSeverity.KNOWLEDGE_BLOCK: 2,
    IntakeSeverity.INFO: 1,
}


# --------------------------------------------------------------------------- #
# Frozen data structures.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PdfParseQuality:
    """Deterministic parse-quality metrics for an inspected PDF."""

    extracted_char_count: int = 0
    pages_with_text: int = 0
    text_coverage_ratio: float = 0.0
    average_chars_per_page: float = 0.0
    suspicious_whitespace_ratio: float = 0.0
    replacement_character_count: int = 0
    extraction_quality_score: float = 0.0
    extraction_quality_band: str = "unusable"

    def to_dict(self) -> dict:
        return {
            "extracted_char_count": self.extracted_char_count,
            "pages_with_text": self.pages_with_text,
            "text_coverage_ratio": self.text_coverage_ratio,
            "average_chars_per_page": self.average_chars_per_page,
            "suspicious_whitespace_ratio": self.suspicious_whitespace_ratio,
            "replacement_character_count": self.replacement_character_count,
            "extraction_quality_score": self.extraction_quality_score,
            "extraction_quality_band": self.extraction_quality_band,
        }


@dataclass(frozen=True)
class PdfMetadataSummary:
    """Deterministic, *unverified* facts read from a local PDF file.

    Every field here is observed from the file bytes only. Embedded document
    metadata (title/author/…) is descriptive, not authoritative.
    """

    file_path: str
    file_name: str
    file_hash: str = ""
    file_size_bytes: int = 0
    page_count: int = 0
    title: str = ""
    author: str = ""
    creator: str = ""
    producer: str = ""
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    is_encrypted: bool = False
    password_required: Optional[bool] = None
    has_text_layer: bool = False
    extracted_text_char_count: int = 0
    extracted_text_sample: str = ""
    text_density: float = 0.0
    likely_scanned: bool = False
    requires_ocr: bool = False
    possible_tables: bool = False
    possible_images: bool = False
    possible_forms: bool = False
    embedded_file_count: int = 0
    parse_quality: PdfParseQuality = field(default_factory=PdfParseQuality)
    read_error: str = ""

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "title": self.title,
            "author": self.author,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date,
            "modification_date": self.modification_date,
            "is_encrypted": self.is_encrypted,
            "password_required": self.password_required,
            "has_text_layer": self.has_text_layer,
            "extracted_text_char_count": self.extracted_text_char_count,
            "extracted_text_sample": self.extracted_text_sample,
            "text_density": self.text_density,
            "likely_scanned": self.likely_scanned,
            "requires_ocr": self.requires_ocr,
            "possible_tables": self.possible_tables,
            "possible_images": self.possible_images,
            "possible_forms": self.possible_forms,
            "embedded_file_count": self.embedded_file_count,
            "parse_quality": self.parse_quality.to_dict(),
            "read_error": self.read_error,
        }


@dataclass(frozen=True)
class PdfIntakeCandidate:
    """A PDF plus the operator-declared governance inputs awaiting assessment.

    The declared fields are *operator-supplied* provenance and permission. They
    are deliberately separate from the embedded metadata, which is never trusted.
    """

    metadata: PdfMetadataSummary
    declared_source_url: str = ""
    declared_owner: str = ""
    declared_permission_or_licence: str = ""
    intended_use: str = ""
    authority_level: str = ""
    notes: str = ""

    @property
    def is_internal_owned(self) -> bool:
        return bool(self.declared_owner) and _norm(self.authority_level) in _INTERNAL_AUTHORITY

    @property
    def has_provenance(self) -> bool:
        return bool(self.declared_source_url.strip()) or bool(self.declared_owner.strip())

    @property
    def has_permission(self) -> bool:
        return bool(self.declared_permission_or_licence.strip()) or self.is_internal_owned

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "declared_source_url": self.declared_source_url,
            "declared_owner": self.declared_owner,
            "declared_permission_or_licence": self.declared_permission_or_licence,
            "intended_use": self.intended_use,
            "authority_level": self.authority_level,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PdfIntakeFinding:
    """A single deterministic observation about a PDF candidate."""

    code: PdfFindingCode
    severity: IntakeSeverity
    message: str

    @property
    def blocks_knowledge(self) -> bool:
        return self.severity in (
            IntakeSeverity.BLOCK,
            IntakeSeverity.QUARANTINE,
            IntakeSeverity.REVIEW,
            IntakeSeverity.KNOWLEDGE_BLOCK,
        )

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "blocks_knowledge": self.blocks_knowledge,
        }


@dataclass(frozen=True)
class PdfIntakeResult:
    """The deterministic result of assessing one local PDF file."""

    candidate: PdfIntakeCandidate
    external_candidate: ExternalDatasetCandidate
    decision: DataIntakeDecision
    lane: IntakeLane
    findings: Tuple[PdfIntakeFinding, ...] = ()
    rationale: str = ""
    assessed_at: str = field(default_factory=lambda: _utc_now().isoformat())

    @property
    def approved_for_eval(self) -> bool:
        return self.decision == DataIntakeDecision.APPROVED_FOR_EVAL

    @property
    def approved_for_knowledge(self) -> bool:
        return self.decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE

    @property
    def permits_eval_use(self) -> bool:
        return self.decision in (
            DataIntakeDecision.APPROVED_FOR_EVAL,
            DataIntakeDecision.APPROVED_FOR_KNOWLEDGE,
        )

    @property
    def permits_knowledge_use(self) -> bool:
        return self.decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE

    @property
    def blocked(self) -> bool:
        return self.decision == DataIntakeDecision.BLOCKED

    @property
    def needs_review(self) -> bool:
        return self.decision == DataIntakeDecision.NEEDS_REVIEW

    @property
    def quarantined(self) -> bool:
        return self.decision == DataIntakeDecision.QUARANTINE

    def to_dict(self) -> dict:
        return {
            "_record": "pdf_intake_assessment",
            "candidate": self.candidate.to_dict(),
            "external_candidate": self.external_candidate.to_dict(),
            "decision": self.decision.value,
            "lane": self.lane.value,
            "approved_for_eval": self.approved_for_eval,
            "approved_for_knowledge": self.approved_for_knowledge,
            "permits_eval_use": self.permits_eval_use,
            "permits_knowledge_use": self.permits_knowledge_use,
            "blocked": self.blocked,
            "needs_review": self.needs_review,
            "quarantined": self.quarantined,
            "rationale": self.rationale,
            "findings": [f.to_dict() for f in self.findings],
            "assessed_at": self.assessed_at,
        }


# --------------------------------------------------------------------------- #
# Minimal, deterministic, stdlib-only PDF reader (no third-party dependency).
# --------------------------------------------------------------------------- #


def calculate_pdf_hash(path) -> str:
    """Return a deterministic SHA-256 hash of the file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace")


def _unescape_literal(raw: bytes) -> str:
    text = raw.decode("latin-1", errors="replace")
    # PDF literal-string escapes that matter for plain text extraction.
    replacements = {
        "\\(": "(", "\\)": ")", "\\\\": "\\",
        "\\n": "\n", "\\r": "\r", "\\t": "\t",
    }
    for needle, value in replacements.items():
        text = text.replace(needle, value)
    return text


def _object_map(data: bytes) -> dict:
    return {int(num): body for num, body in _OBJ_RE.findall(data)}


def _stream_text(body: bytes) -> str:
    """Extract plain text operands from a single content-stream object body."""
    match = _STREAM_RE.search(body)
    if not match:
        return ""
    payload = match.group(1)
    if _EMBEDDED_FILE_RE.search(body):  # never inspect embedded-file payloads
        return ""
    if b"/FlateDecode" in body:
        try:
            payload = zlib.decompress(payload)
        except zlib.error:
            return ""
    parts = [_unescape_literal(m.group(1)) for m in _LITERAL_STR_RE.finditer(payload)]
    return " ".join(p for p in parts if p)


def _content_refs(page_body: bytes) -> List[int]:
    refs: List[int] = []
    single = _CONTENTS_SINGLE_RE.search(page_body)
    if single:
        refs.append(int(single.group(1)))
    array = _CONTENTS_ARRAY_RE.search(page_body)
    if array:
        refs.extend(int(n) for n in _REF_RE.findall(array.group(1)))
    return refs


def sample_pdf_text(data: bytes) -> Tuple[List[str], str]:
    """Return ``(per_page_texts, full_text)`` extracted from a PDF's bytes.

    Reads only declared text operands. Performs no optical recovery of image
    text and builds no document fragments; the result is a bounded plain string.
    """
    objects = _object_map(data)
    page_texts: List[str] = []
    for num in sorted(objects):
        body = objects[num]
        if not _PAGE_RE.search(body):
            continue
        page_text_parts: List[str] = []
        for ref in _content_refs(body):
            ref_body = objects.get(ref)
            if ref_body is not None:
                page_text_parts.append(_stream_text(ref_body))
        page_texts.append(" ".join(p for p in page_text_parts if p).strip())
    full_text = "\n".join(page_texts)[:MAX_TEXT_CHARS]
    return page_texts, full_text


def _info_dict_body(data: bytes, objects: dict) -> bytes:
    ref = _INFO_REF_RE.search(data)
    if ref:
        body = objects.get(int(ref.group(1)))
        if body is not None:
            return body
    # Fall back to any object that declares document-info keys.
    for body in objects.values():
        if b"/Producer" in body or b"/Title" in body or b"/Author" in body:
            return body
    return b""


def _info_value(info_body: bytes, key: str) -> str:
    match = re.search(re.escape(key.encode("latin-1")) + rb"\s*\(((?:[^()\\]|\\.)*)\)", info_body)
    if not match:
        return ""
    return _unescape_literal(match.group(1)).strip()


def _count_pages(data: bytes, objects: dict) -> int:
    direct = sum(1 for body in objects.values() if _PAGE_RE.search(body))
    if direct:
        return direct
    for body in objects.values():
        if b"/Type" in body and b"/Pages" in body:
            count = _PAGES_COUNT_RE.search(body)
            if count:
                return int(count.group(1))
    return 0


def calculate_parse_quality(
    page_texts: Sequence[str],
    page_count: int,
    full_text: str,
) -> PdfParseQuality:
    """Compute deterministic parse-quality metrics and a coarse band.

    The score multiplies page-coverage by a text-density factor, then penalises
    decode-replacement characters and pathological whitespace. Thresholds are
    module constants; the bands are intentionally coarse to avoid false
    precision.
    """
    char_count = len(full_text)
    pages_with_text = sum(1 for text in page_texts if text)
    effective_pages = page_count if page_count > 0 else len(page_texts)

    coverage = round(pages_with_text / effective_pages, 4) if effective_pages else 0.0
    avg_chars = round(char_count / effective_pages, 2) if effective_pages else 0.0

    whitespace = sum(1 for ch in full_text if ch.isspace())
    whitespace_ratio = round(whitespace / char_count, 4) if char_count else 0.0
    replacement_count = full_text.count(REPLACEMENT_CHAR)

    density_factor = min(1.0, avg_chars / TARGET_CHARS_PER_PAGE) if TARGET_CHARS_PER_PAGE else 0.0
    score = coverage * density_factor
    replacement_ratio = replacement_count / char_count if char_count else 0.0
    if replacement_ratio > 0.02:
        score *= max(0.0, 1.0 - min(0.5, replacement_ratio))
    if whitespace_ratio > HIGH_WHITESPACE_RATIO:
        score *= 0.5
    score = 0.0 if char_count == 0 else round(score, 2)

    if char_count == 0 or score < QUALITY_POOR:
        band = "unusable"
    elif score < QUALITY_ACCEPTABLE:
        band = "poor"
    elif score < QUALITY_GOOD:
        band = "acceptable"
    else:
        band = "good"

    return PdfParseQuality(
        extracted_char_count=char_count,
        pages_with_text=pages_with_text,
        text_coverage_ratio=coverage,
        average_chars_per_page=avg_chars,
        suspicious_whitespace_ratio=whitespace_ratio,
        replacement_character_count=replacement_count,
        extraction_quality_score=score,
        extraction_quality_band=band,
    )


def _possible_tables(page_texts: Sequence[str]) -> bool:
    # Heuristic: column-like alignment shows up as repeated multi-space runs or
    # tabs in the extracted operands.
    for text in page_texts:
        if "\t" in text or len(re.findall(r" {2,}", text)) >= 2:
            return True
    return False


def _duplicate_header_footer(page_texts: Sequence[str]) -> bool:
    if len(page_texts) < 2:
        return False
    heads = [t[:24].strip() for t in page_texts if t.strip()]
    return len(heads) != len(set(heads))


def inspect_pdf_metadata(path) -> PdfMetadataSummary:
    """Read deterministic facts from a local PDF. Fails closed on any error.

    Reads file bytes only; never downloads, never recovers image text, never
    writes. On a malformed/encrypted/unreadable file it returns a summary whose
    ``read_error`` is set so the caller can fail closed.
    """
    source = Path(path)
    file_path = str(source)
    file_name = source.name

    if not source.exists() or not source.is_file():
        return PdfMetadataSummary(file_path=file_path, file_name=file_name,
                                  read_error="file not found")
    if source.suffix.lower() != ".pdf":
        return PdfMetadataSummary(file_path=file_path, file_name=file_name,
                                  read_error=f"unsupported extension '{source.suffix}'")

    try:
        data = source.read_bytes()
    except OSError as exc:  # pragma: no cover - defensive
        return PdfMetadataSummary(file_path=file_path, file_name=file_name,
                                  read_error=f"read failure: {exc}")

    file_size = len(data)
    file_hash = hashlib.sha256(data).hexdigest()

    if not data[:1024].lstrip().startswith(b"%PDF-"):
        return PdfMetadataSummary(file_path=file_path, file_name=file_name,
                                  file_hash=file_hash, file_size_bytes=file_size,
                                  read_error="missing %PDF header (malformed)")

    try:
        objects = _object_map(data)
        page_count = _count_pages(data, objects)
        page_texts, full_text = sample_pdf_text(data)
        info_body = _info_dict_body(data, objects)
        is_encrypted = bool(_ENCRYPT_RE.search(data))
        possible_images = bool(_IMAGE_RE.search(data))
        possible_forms = bool(_ACROFORM_RE.search(data))
        embedded_file_count = len(_EMBEDDED_FILE_RE.findall(data))
    except Exception as exc:  # pragma: no cover - defensive; fail closed
        return PdfMetadataSummary(file_path=file_path, file_name=file_name,
                                  file_hash=file_hash, file_size_bytes=file_size,
                                  read_error=f"parse failure: {exc}")

    quality = calculate_parse_quality(page_texts, page_count, full_text)
    char_count = quality.extracted_char_count
    has_text_layer = char_count > 0
    avg_chars = quality.average_chars_per_page
    likely_scanned = possible_images and (
        not has_text_layer or avg_chars < LOW_DENSITY_CHARS_PER_PAGE)
    requires_ocr = not has_text_layer or likely_scanned
    text_density = round(char_count / file_size * 1000, 4) if file_size else 0.0

    return PdfMetadataSummary(
        file_path=file_path,
        file_name=file_name,
        file_hash=file_hash,
        file_size_bytes=file_size,
        page_count=page_count,
        title=_info_value(info_body, "/Title"),
        author=_info_value(info_body, "/Author"),
        creator=_info_value(info_body, "/Creator"),
        producer=_info_value(info_body, "/Producer"),
        creation_date=_info_value(info_body, "/CreationDate") or None,
        modification_date=_info_value(info_body, "/ModDate") or None,
        is_encrypted=is_encrypted,
        password_required=is_encrypted if is_encrypted else None,
        has_text_layer=has_text_layer,
        extracted_text_char_count=char_count,
        extracted_text_sample=full_text[:SAMPLE_CHAR_LIMIT],
        text_density=text_density,
        likely_scanned=likely_scanned,
        requires_ocr=requires_ocr,
        possible_tables=_possible_tables(page_texts),
        possible_images=possible_images,
        possible_forms=possible_forms,
        embedded_file_count=embedded_file_count,
        parse_quality=quality,
        read_error="",
    )


# --------------------------------------------------------------------------- #
# Date parsing and risk inference.
# --------------------------------------------------------------------------- #


def _parse_pdf_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    match = _PDF_DATE_RE.search(str(value))
    if not match:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    year = int(match.group(1))
    month = int(match.group(2) or 1)
    day = int(match.group(3) or 1)
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def _has_marker(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def _finding(code: PdfFindingCode, severity: IntakeSeverity, message: str) -> PdfIntakeFinding:
    return PdfIntakeFinding(code=code, severity=severity, message=message)


def infer_pdf_findings(
    candidate: PdfIntakeCandidate,
    *,
    now: Optional[datetime] = None,
) -> List[PdfIntakeFinding]:
    """Derive deterministic, conservative findings for a PDF candidate."""
    moment = now or _utc_now()
    meta = candidate.metadata
    findings: List[PdfIntakeFinding] = []

    # 1. Readability / encryption gate (fail closed first).
    if meta.read_error:
        findings.append(_finding(PdfFindingCode.PDF_UNREADABLE, IntakeSeverity.BLOCK,
                                 f"PDF could not be read ({meta.read_error}); blocked"))
        return findings
    if meta.is_encrypted:
        findings.append(_finding(PdfFindingCode.PDF_ENCRYPTED, IntakeSeverity.BLOCK,
                                 "PDF is encrypted; content cannot be verified, so it is blocked"))
        if meta.password_required:
            findings.append(_finding(PdfFindingCode.PDF_PASSWORD_REQUIRED, IntakeSeverity.REVIEW,
                                     "a password is required to open this PDF"))
        return findings
    if meta.page_count == 0:
        findings.append(_finding(PdfFindingCode.PDF_UNREADABLE, IntakeSeverity.BLOCK,
                                 "PDF reports zero pages; treated as unreadable and blocked"))
        return findings

    # 2. Provenance and permission (operator-declared; embedded metadata never counts).
    if not candidate.has_provenance:
        findings.append(_finding(PdfFindingCode.PDF_MISSING_PROVENANCE, IntakeSeverity.REVIEW,
                                 "no declared source URL or owner; provenance must be established"))
    if not candidate.has_permission:
        findings.append(_finding(PdfFindingCode.PDF_MISSING_PERMISSION, IntakeSeverity.REVIEW,
                                 "no declared permission/licence and not declared internal-owned"))

    # 3. Parseability: text layer, scanning, OCR, extraction quality.
    if not meta.has_text_layer:
        findings.append(_finding(PdfFindingCode.PDF_NO_TEXT_LAYER, IntakeSeverity.KNOWLEDGE_BLOCK,
                                 "no extractable text layer; not eligible for knowledge in this slice"))
    if meta.likely_scanned:
        findings.append(_finding(PdfFindingCode.PDF_SCANNED_IMAGE_ONLY, IntakeSeverity.KNOWLEDGE_BLOCK,
                                 "looks scanned/image-only; not eligible for knowledge in this slice"))
    if meta.requires_ocr:
        findings.append(_finding(PdfFindingCode.PDF_REQUIRES_OCR, IntakeSeverity.KNOWLEDGE_BLOCK,
                                 "would require OCR to read; OCR is deferred, so not knowledge here"))
    band = meta.parse_quality.extraction_quality_band
    if meta.has_text_layer and band in ("poor", "unusable"):
        findings.append(_finding(PdfFindingCode.PDF_LOW_TEXT_EXTRACTION_QUALITY,
                                 IntakeSeverity.KNOWLEDGE_BLOCK,
                                 f"text extraction quality is '{band}'; too low for trusted knowledge"))
    if 0 < meta.extracted_text_char_count < SPARSE_TEXT_MIN_CHARS:
        findings.append(_finding(PdfFindingCode.PDF_SPARSE_TEXT, IntakeSeverity.KNOWLEDGE_BLOCK,
                                 f"only {meta.extracted_text_char_count} chars extracted; too sparse for knowledge"))

    # 4. Size.
    if meta.file_size_bytes > TOO_LARGE_BYTES:
        findings.append(_finding(PdfFindingCode.PDF_TOO_LARGE_FOR_INITIAL_IMPORT,
                                 IntakeSeverity.QUARANTINE,
                                 f"file size {meta.file_size_bytes} bytes is too large for an initial import"))

    # 5. Sensitivity: PII and confidentiality markers.
    haystack = " ".join([
        meta.extracted_text_sample, meta.title, meta.author,
        meta.creator, meta.producer,
    ]).lower()
    if _EMAIL_RE.search(haystack) or _SSN_RE.search(haystack) or _has_marker(haystack, _PII_KEYWORDS):
        findings.append(_finding(PdfFindingCode.PDF_POSSIBLE_PII, IntakeSeverity.REVIEW,
                                 "text contains possible personal data; a human must review before any use"))
    if _has_marker(haystack, _CONFIDENTIAL_MARKERS):
        findings.append(_finding(PdfFindingCode.PDF_CONFIDENTIAL_MARKER, IntakeSeverity.REVIEW,
                                 "a confidentiality marker was found; a human must review before any use"))
    if _has_marker(haystack, _RESTRICTED_MARKERS):
        findings.append(_finding(PdfFindingCode.PDF_RESTRICTED_MARKER, IntakeSeverity.REVIEW,
                                 "a restricted-distribution marker was found; a human must review before any use"))

    # 6. Freshness (a finding only; never declares the content false).
    recent = _parse_pdf_date(meta.modification_date) or _parse_pdf_date(meta.creation_date)
    if recent is None:
        findings.append(_finding(PdfFindingCode.PDF_NO_CREATION_DATE, IntakeSeverity.INFO,
                                 "no creation/modification date in metadata to judge freshness"))
    elif (moment - recent).days > STALE_AFTER_DAYS:
        findings.append(_finding(PdfFindingCode.PDF_STALE_BY_METADATA, IntakeSeverity.KNOWLEDGE_BLOCK,
                                 "document metadata date is old; treat as possibly outdated, not as false"))

    # 7. Intended use (eval-only intent caps below the knowledge tier).
    if _norm(candidate.intended_use) in _EVAL_INTENDED_USE:
        findings.append(_finding(PdfFindingCode.PDF_EVAL_ONLY_INTENDED_USE, IntakeSeverity.KNOWLEDGE_BLOCK,
                                 f"declared intended use '{candidate.intended_use}' is eval-only, not knowledge"))

    # 8. Structure (informational).
    if meta.possible_tables:
        findings.append(_finding(PdfFindingCode.PDF_TABLE_HEAVY, IntakeSeverity.INFO,
                                 "table-like layout detected; structured extraction is deferred"))
    if meta.possible_images:
        findings.append(_finding(PdfFindingCode.PDF_IMAGE_HEAVY, IntakeSeverity.INFO,
                                 "images detected; image content is not interpreted in this slice"))
    if meta.possible_forms:
        findings.append(_finding(PdfFindingCode.PDF_FORM_HEAVY, IntakeSeverity.INFO,
                                 "interactive form fields detected; form data is not interpreted here"))
    if meta.embedded_file_count > 0:
        findings.append(_finding(PdfFindingCode.PDF_EMBEDDED_FILES_PRESENT, IntakeSeverity.REVIEW,
                                 f"{meta.embedded_file_count} embedded file(s) present; a human must review"))

    # 9. Metadata is informational only.
    if any([meta.title, meta.author, meta.creator, meta.producer]):
        findings.append(_finding(PdfFindingCode.PDF_METADATA_UNVERIFIED, IntakeSeverity.INFO,
                                 "embedded PDF metadata is descriptive and is treated as unverified"))

    # 10. Positive provenance notes (informational; never force approval).
    clean = not any(f.severity in (IntakeSeverity.BLOCK, IntakeSeverity.QUARANTINE,
                                   IntakeSeverity.REVIEW) for f in findings)
    if clean and candidate.has_provenance and candidate.has_permission:
        if _norm(candidate.authority_level) == "official":
            findings.append(_finding(PdfFindingCode.PDF_APPROVED_OFFICIAL_SOURCE, IntakeSeverity.INFO,
                                     "declared official source with provenance and permission"))
        elif candidate.is_internal_owned:
            findings.append(_finding(PdfFindingCode.PDF_APPROVED_INTERNAL_SOURCE, IntakeSeverity.INFO,
                                     "declared internal-owned source with provenance and permission"))

    return findings


# --------------------------------------------------------------------------- #
# Decision derivation (mirrors the v6.2A worst-finding-wins mapping).
# --------------------------------------------------------------------------- #


def _worst_severity(findings: Sequence[PdfIntakeFinding]) -> IntakeSeverity:
    worst = IntakeSeverity.INFO
    for finding in findings:
        if _SEVERITY_RANK[finding.severity] > _SEVERITY_RANK[worst]:
            worst = finding.severity
    return worst


def _decide(findings: Sequence[PdfIntakeFinding]) -> DataIntakeDecision:
    worst = _worst_severity(findings)
    if worst == IntakeSeverity.BLOCK:
        return DataIntakeDecision.BLOCKED
    if worst == IntakeSeverity.QUARANTINE:
        return DataIntakeDecision.QUARANTINE
    if worst == IntakeSeverity.REVIEW:
        return DataIntakeDecision.NEEDS_REVIEW
    if worst == IntakeSeverity.KNOWLEDGE_BLOCK:
        return DataIntakeDecision.APPROVED_FOR_EVAL
    return DataIntakeDecision.APPROVED_FOR_KNOWLEDGE


def _lane(decision: DataIntakeDecision) -> IntakeLane:
    if decision == DataIntakeDecision.BLOCKED:
        return IntakeLane.BLOCKED
    if decision in (DataIntakeDecision.QUARANTINE, DataIntakeDecision.NEEDS_REVIEW):
        return IntakeLane.QUARANTINE
    if decision == DataIntakeDecision.APPROVED_FOR_KNOWLEDGE:
        return IntakeLane.KNOWLEDGE_CANDIDATE
    return IntakeLane.EVAL_ONLY


_RATIONALE_HEAD = {
    DataIntakeDecision.APPROVED_FOR_KNOWLEDGE: "Approved as a knowledge candidate (and eval).",
    DataIntakeDecision.APPROVED_FOR_EVAL: "Approved for eval only; not trusted knowledge.",
    DataIntakeDecision.NEEDS_REVIEW: "Needs human review before any use.",
    DataIntakeDecision.QUARANTINE: "Quarantined; held until cleared.",
    DataIntakeDecision.BLOCKED: "Blocked; no import should occur.",
}


def _rationale(decision: DataIntakeDecision, findings: Sequence[PdfIntakeFinding]) -> str:
    head = _RATIONALE_HEAD[decision]
    drivers = [f.message for f in findings if f.severity != IntakeSeverity.INFO]
    if not drivers:
        drivers = [f.message for f in findings]
    return head + (" " + "; ".join(drivers) if drivers else "")


# --------------------------------------------------------------------------- #
# Mapping to the shared ExternalDatasetCandidate (for interop with v6.2A).
# --------------------------------------------------------------------------- #


def pdf_metadata_to_candidate(candidate: PdfIntakeCandidate) -> ExternalDatasetCandidate:
    """Adapt a PDF candidate into the shared :class:`ExternalDatasetCandidate`.

    Pure and deterministic; reads only already-inspected metadata. The embedded
    PDF metadata is *not* used as provenance — only the operator-declared source
    URL/owner are.
    """
    meta = candidate.metadata
    haystack = " ".join([meta.extracted_text_sample, meta.title]).lower()
    possible_pii = bool(
        _EMAIL_RE.search(haystack) or _SSN_RE.search(haystack)
        or _has_marker(haystack, _PII_KEYWORDS))
    source_url = candidate.declared_source_url or f"file://{meta.file_name}"
    provenance = ""
    if candidate.declared_source_url:
        provenance = f"pdf:{candidate.declared_source_url}"
    elif candidate.declared_owner:
        provenance = f"pdf-owner:{candidate.declared_owner}"
    else:
        provenance = "unknown"
    return ExternalDatasetCandidate(
        dataset_id=meta.file_name or meta.file_hash or meta.file_path,
        source_url=source_url,
        source_type="pdf_document",
        title=meta.title or meta.file_name,
        description=(meta.extracted_text_sample[:200] or "local PDF document"),
        licence=candidate.declared_permission_or_licence or None,
        provenance=provenance,
        publisher=candidate.declared_owner,
        language="",
        task_categories=(),
        size_hint=f"{meta.file_size_bytes} bytes",
        last_updated=meta.modification_date or meta.creation_date,
        intended_use=candidate.intended_use,
        contains_personal_data=None if possible_pii else False,
        is_synthetic=False,
        sample_available=meta.has_text_layer,
        notes="embedded PDF metadata is unverified",
    )


# --------------------------------------------------------------------------- #
# Public assessment entry points.
# --------------------------------------------------------------------------- #


def assess_pdf_intake(
    path,
    *,
    source_url: str = "",
    owner: str = "",
    permission: str = "",
    intended_use: str = "",
    authority_level: str = "",
    now: Optional[datetime] = None,
) -> PdfIntakeResult:
    """Assess one local PDF deterministically. Reads the file only; writes nothing."""
    moment = now or _utc_now()
    metadata = inspect_pdf_metadata(path)
    candidate = PdfIntakeCandidate(
        metadata=metadata,
        declared_source_url=source_url or "",
        declared_owner=owner or "",
        declared_permission_or_licence=permission or "",
        intended_use=intended_use or "",
        authority_level=authority_level or "",
        notes="embedded PDF metadata is treated as unverified",
    )
    findings = infer_pdf_findings(candidate, now=moment)
    decision = _decide(findings)
    if decision == DataIntakeDecision.NEEDS_REVIEW:
        findings.append(_finding(PdfFindingCode.NEEDS_HUMAN_REVIEW, IntakeSeverity.REVIEW,
                                 "a human must review and decide before any use"))
    lane = _lane(decision)
    rationale = _rationale(decision, findings)
    external = pdf_metadata_to_candidate(candidate)
    return PdfIntakeResult(
        candidate=candidate,
        external_candidate=external,
        decision=decision,
        lane=lane,
        findings=tuple(findings),
        rationale=rationale,
        assessed_at=moment.isoformat(),
    )


def assess_pdf_paths(
    paths: Sequence[str],
    *,
    source_url: str = "",
    owner: str = "",
    permission: str = "",
    intended_use: str = "",
    authority_level: str = "",
    now: Optional[datetime] = None,
) -> List[PdfIntakeResult]:
    return [
        assess_pdf_intake(path, source_url=source_url, owner=owner,
                          permission=permission, intended_use=intended_use,
                          authority_level=authority_level, now=now)
        for path in paths
    ]


# --------------------------------------------------------------------------- #
# Rendering and the single durable write (only to an explicit path).
# --------------------------------------------------------------------------- #


def render_pdf_assessment_markdown(result: PdfIntakeResult) -> str:
    """Render a deterministic, human-readable report for one PDF assessment."""
    meta = result.candidate.metadata
    quality = meta.parse_quality
    lines = [
        "# PDF intake assessment (v6.4; assessment-only)",
        "",
        f"- File: {meta.file_name}",
        f"- SHA-256: {meta.file_hash}",
        f"- Pages: {meta.page_count}",
        f"- Decision: {result.decision.value}",
        f"- Lane: {result.lane.value}",
        f"- Approved for knowledge: {str(result.approved_for_knowledge).lower()}",
        f"- Approved for eval: {str(result.approved_for_eval).lower()}",
        f"- Permits eval use: {str(result.permits_eval_use).lower()}",
        f"- Text layer: {str(meta.has_text_layer).lower()}",
        f"- Requires OCR: {str(meta.requires_ocr).lower()}",
        f"- Extraction quality: {quality.extraction_quality_band} "
        f"(score {quality.extraction_quality_score})",
        "",
        "## Declared governance inputs (operator-supplied; not from the file)",
        f"- Source URL: {result.candidate.declared_source_url or '(none)'}",
        f"- Owner: {result.candidate.declared_owner or '(none)'}",
        f"- Permission/licence: {result.candidate.declared_permission_or_licence or '(none)'}",
        f"- Authority level: {result.candidate.authority_level or '(none)'}",
        f"- Intended use: {result.candidate.intended_use or '(none)'}",
        "",
        "## Embedded metadata (unverified)",
        f"- Title: {meta.title or '(none)'}",
        f"- Author: {meta.author or '(none)'}",
        f"- Creator: {meta.creator or '(none)'}",
        f"- Producer: {meta.producer or '(none)'}",
        f"- Creation date: {meta.creation_date or '(none)'}",
        f"- Modification date: {meta.modification_date or '(none)'}",
        "",
        "## Rationale",
        result.rationale,
        "",
        "## Findings",
    ]
    if result.findings:
        for finding in result.findings:
            lines.append(f"- [{finding.severity.value}] {finding.code.value}: {finding.message}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def render_pdf_summary_markdown(results: Sequence[PdfIntakeResult]) -> str:
    """Render a one-line-per-file summary across many PDF assessments."""
    lines = ["# PDF intake summary (v6.4; assessment-only)", ""]
    for result in results:
        lines.append(
            f"- {result.candidate.metadata.file_name}: {result.decision.value} "
            f"(lane: {result.lane.value})"
        )
    return "\n".join(lines)


def assessment_to_json(result_or_results) -> str:
    """Serialise one result or a sequence of results to deterministic JSON text."""
    if isinstance(result_or_results, PdfIntakeResult):
        payload = result_or_results.to_dict()
    else:
        payload = [r.to_dict() for r in result_or_results]
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_pdf_assessment_report(result_or_results, out_path) -> Path:
    """Write a PDF assessment report to ``out_path`` (the only durable write).

    Accepts a single result or a sequence. Writes *only* the report; it never
    touches memory, the source registry, any proposal queue, or any retrieval
    index.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(assessment_to_json(result_or_results), encoding="utf-8")
    return path
