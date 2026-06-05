"""Format-agnostic document text extraction for the unified ``ingest`` path.

This module turns a single file on disk into plain text and then into bounded
chunks suitable for writing into the concept-cell memory bank. It is the one
place that knows how to read each supported document format:

* ``.txt`` / ``.md``  -- read directly as UTF-8 text;
* ``.pdf``            -- the stdlib-only governed reader
                         (:func:`agent.pdf_intake_adapter.sample_pdf_text`);
* ``.docx``           -- via ``python-docx`` (optional dependency);
* ``.xlsx`` / ``.xls`` -- via ``openpyxl`` (optional dependency).

The Word and Excel readers are imported lazily so the CLI stays light and the
core path has no hard third-party dependency. If a needed library is missing,
:class:`DocumentIngestError` is raised with the exact install hint.

This module only *extracts and chunks*. It does not embed, write to the bank,
or touch the ledger -- that is the caller's job
(:meth:`agent.workbench_service.WorkbenchService.ingest_document`).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from agent.pdf_intake_adapter import sample_pdf_text

# Upper bound on extracted text per document (characters). Mirrors the spirit
# of the PDF adapter's own cap so a single huge file cannot flood the bank.
MAX_TEXT_CHARS = 200_000

# Chunking defaults (characters). Kept local and simple so this module is
# self-contained and does not reach into the PDF preview internals.
CHUNK_SIZE = 1000
MIN_CHUNK_CHARS = 200

SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf", ".docx", ".xlsx", ".xls")


class DocumentIngestError(Exception):
    """Raised when a document cannot be read (unsupported type or missing lib)."""


def _read_plain_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    _, full_text = sample_pdf_text(path.read_bytes())
    return full_text


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - exercised only without the lib
        raise DocumentIngestError(
            "reading .docx files needs python-docx; install it with "
            "`pip install python-docx`"
        ) from exc
    document = docx.Document(str(path))
    parts: List[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _read_xlsx(path: Path) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - exercised only without the lib
        raise DocumentIngestError(
            "reading .xlsx files needs openpyxl; install it with "
            "`pip install openpyxl`"
        ) from exc
    workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
    parts: List[str] = []
    for sheet in workbook.worksheets:
        parts.append(f"# {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(value).strip() for value in row if value is not None]
            if cells:
                parts.append(" | ".join(cells))
    workbook.close()
    return "\n".join(parts)


_READERS = {
    ".txt": _read_plain_text,
    ".md": _read_plain_text,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".xlsx": _read_xlsx,
    ".xls": _read_xlsx,
}


def extract_text(path: Path) -> str:
    """Return the plain text of ``path``, dispatching on its file suffix.

    Raises :class:`FileNotFoundError` if the file is absent and
    :class:`DocumentIngestError` for an unsupported type or a missing optional
    library. The returned text is bounded to :data:`MAX_TEXT_CHARS`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        raise DocumentIngestError(
            f"unsupported document type '{suffix or path.name}'; "
            f"supported: {', '.join(SUPPORTED_SUFFIXES)}"
        )
    text = reader(path)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split ``text`` into paragraph-aware chunks of roughly ``chunk_size``.

    Paragraphs (separated by blank lines) are packed together until they reach
    ``chunk_size``; a paragraph longer than ``chunk_size`` is hard-split on
    whitespace without breaking a word. Returns a list of trimmed, non-empty
    chunks. An empty or whitespace-only document yields an empty list.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_hard_split(para, chunk_size))
            continue
        if not buffer:
            buffer = para
        elif len(buffer) + 2 + len(para) <= chunk_size:
            buffer = f"{buffer}\n\n{para}"
        else:
            chunks.append(buffer)
            buffer = para
    if buffer:
        chunks.append(buffer)
    return chunks


def _hard_split(text: str, chunk_size: int) -> List[str]:
    """Split an over-long paragraph on whitespace without breaking a word."""
    words = text.split()
    chunks: List[str] = []
    buffer = ""
    for word in words:
        if not buffer:
            buffer = word
        elif len(buffer) + 1 + len(word) <= chunk_size:
            buffer = f"{buffer} {word}"
        else:
            chunks.append(buffer)
            buffer = word
    if buffer:
        chunks.append(buffer)
    return chunks
