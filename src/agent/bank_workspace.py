"""Bank Management Workspace backend.

Small, local-first helpers for adding, previewing, searching, tombstoning, and
reloading V1 overlay knowledge. This module is intentionally built on the
existing OverlayStore and StreamingBank path instead of introducing a new
knowledge-management layer.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from src.agent.answer_pipeline import AnswerPipeline, PHI3_MODEL, _build_phi3_generator
from src.agent.bank_admin import OverlayStore, add_cell_from_text
from src.agent.streaming_bank import StreamingBank
from src.cc_service.encoder import EncoderSingleton


SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".pdf"})
DEFAULT_BANK = Path(r"H:\MiniLM\cc_service\bank.db")
DEFAULT_OVERLAY = Path("results/v1_bank/overlay.db")
DEFAULT_CHUNK_CHARS = 700
DEFAULT_CHUNK_OVERLAP = 80
MIN_CELL_CHARS = 40
MAX_PDF_STREAM_INFLATE_BYTES = 32 * 1024 * 1024

_PDF_STREAM_RE = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
_PDF_TEXT_RE = re.compile(rb"\((?:\\.|[^\\)])*\)")
_WS_RE = re.compile(r"\s+")


class WorkspaceError(RuntimeError):
    """Raised for user-facing workspace validation errors."""


class LazyPhi3Generator:
    """Load Phi-3 on the first generated answer, not during bank reload."""

    def __init__(self, *, model_name: str = PHI3_MODEL, use_4bit: bool = True) -> None:
        self.model_name = model_name
        self.use_4bit = use_4bit
        self._generate: Callable[[str], str] | None = None
        self._lock = threading.Lock()

    def __call__(self, prompt: str) -> str:
        if self._generate is None:
            with self._lock:
                if self._generate is None:
                    generate, _ = _build_phi3_generator(self.model_name, self.use_4bit)
                    self._generate = generate
        return self._generate(prompt)


def _http_post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class HostedHttpGenerator:
    """Generate answers via an OpenAI-compatible chat/completions endpoint.

    Backend-agnostic: point ``base_url`` at any OpenAI-compatible server
    (Azure OpenAI, a vLLM or Ollama GPU host, etc.). Deterministic
    (``temperature=0``) to match the local Phi-3 path so the grounded-answer
    contract sees comparable drafts. Raises ``WorkspaceError`` on any failure
    rather than returning an empty or fabricated string.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 512,
        transport: Callable[[str, dict, dict, float], dict] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._transport = transport if transport is not None else _http_post_json

    def __call__(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            body = self._transport(url, payload, headers, self.timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise WorkspaceError(f"Hosted LLM HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise WorkspaceError(f"Hosted LLM request failed: {exc}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise WorkspaceError(
                f"Hosted LLM returned an unexpected response shape: {body!r}"[:500]
            ) from exc
        if not isinstance(content, str):
            raise WorkspaceError("Hosted LLM returned non-text content.")
        return content.strip()


def build_generator_from_env(
    env: dict[str, str] | None = None,
) -> Callable[[str], str] | None:
    """Return a hosted generator if ``WORKSPACE_LLM_BASE_URL`` is set, else None.

    Returning ``None`` preserves the default local Phi-3 path, so the personal
    tier is unchanged unless an operator opts in to a hosted endpoint.
    """
    env = env if env is not None else os.environ
    base_url = (env.get("WORKSPACE_LLM_BASE_URL") or "").strip()
    if not base_url:
        return None
    model = (env.get("WORKSPACE_LLM_MODEL") or "gpt-4o-mini").strip()
    api_key = (env.get("WORKSPACE_LLM_API_KEY") or "").strip() or None
    try:
        timeout = float(env.get("WORKSPACE_LLM_TIMEOUT") or "60")
    except ValueError:
        timeout = 60.0
    try:
        max_tokens = int(env.get("WORKSPACE_LLM_MAX_TOKENS") or "512")
    except ValueError:
        max_tokens = 512
    return HostedHttpGenerator(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
    )


@dataclass(frozen=True)
class ExtractionResult:
    source_name: str
    suffix: str
    text: str
    pages_processed: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def char_count(self) -> int:
        return len(self.text)

    def as_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "suffix": self.suffix,
            "text": self.text,
            "char_count": self.char_count,
            "pages_processed": self.pages_processed,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CandidateCell:
    draft_id: str
    text: str
    label: str
    accepted: bool = True
    rejection_reason: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)

    def as_dict(self) -> dict:
        return {
            "draft_id": self.draft_id,
            "text": self.text,
            "label": self.label,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "char_count": self.char_count,
        }


@dataclass(frozen=True)
class IngestionDraft:
    source_name: str
    extracted_text: str
    candidates: tuple[CandidateCell, ...]
    warnings: tuple[str, ...] = ()
    pages_processed: int = 0

    def as_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "text_char_count": len(self.extracted_text),
            "pages_processed": self.pages_processed,
            "cells_proposed": len(self.candidates),
            "cells_accepted": sum(1 for c in self.candidates if c.accepted),
            "cells_rejected": sum(1 for c in self.candidates if not c.accepted),
            "warnings": list(self.warnings),
            "candidates": [c.as_dict() for c in self.candidates],
        }


@dataclass(frozen=True)
class IngestionReport:
    source_name: str
    pages_processed: int
    text_char_count: int
    cells_proposed: int
    cells_accepted: int
    cells_rejected: int
    duplicates_found: int
    source_identifier: str
    ingestion_timestamp: float
    warnings: tuple[str, ...]
    added_cell_ids: tuple[int, ...]
    retrieval_smoke_test: dict
    reload_required: bool

    def as_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "pages_processed": self.pages_processed,
            "text_char_count": self.text_char_count,
            "cells_proposed": self.cells_proposed,
            "cells_accepted": self.cells_accepted,
            "cells_rejected": self.cells_rejected,
            "duplicates_found": self.duplicates_found,
            "source_identifier": self.source_identifier,
            "ingestion_timestamp": self.ingestion_timestamp,
            "warnings": list(self.warnings),
            "added_cell_ids": list(self.added_cell_ids),
            "retrieval_smoke_test": dict(self.retrieval_smoke_test),
            "reload_required": self.reload_required,
        }


@dataclass(frozen=True)
class CellSearchResult:
    cell_id: int
    source: str
    label: str | None
    text: str
    tombstoned: bool

    def as_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "source": self.source,
            "label": self.label,
            "text": self.text,
            "tombstoned": self.tombstoned,
        }


@dataclass
class WorkspaceSession:
    bank_path: Path = DEFAULT_BANK
    overlay_path: Path = DEFAULT_OVERLAY
    top_k: int = 10
    use_4bit: bool = True
    generator: Callable[[str], str] | None = None
    pipeline: AnswerPipeline | None = None
    overlay_handle: OverlayStore | None = None
    overlay_loaded: bool = False
    reload_required: bool = False
    history: list[dict] = field(default_factory=list)
    qa_history_path: Path | None = None
    max_persisted_qa: int = 200
    reload_job: dict = field(default_factory=lambda: {
        "state": "idle",
        "stage": "not_started",
        "loaded": 0,
        "total": None,
        "percent": None,
        "rate_per_second": 0.0,
        "eta_seconds": None,
        "started_at": None,
        "finished_at": None,
        "error": "",
    })
    _reload_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _reload_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _qa_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.qa_history_path is None:
            self.qa_history_path = self.overlay_path.parent / "qa_history.jsonl"
        self._load_qa_history()

    def _load_qa_history(self) -> None:
        path = self.qa_history_path
        if path is None or not path.exists():
            return
        records: list[dict] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            return
        self.history = records[-int(self.max_persisted_qa):]
        if len(records) > int(self.max_persisted_qa):
            self._rewrite_qa_history(self.history)

    def _rewrite_qa_history(self, records: list[dict]) -> None:
        path = self.qa_history_path
        if path is None:
            return
        with self._qa_lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".tmp")
                with tmp.open("w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record) + "\n")
                tmp.replace(path)
            except OSError:
                pass

    def _persist_qa(self, record: dict) -> None:
        path = self.qa_history_path
        if path is None:
            return
        with self._qa_lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            except OSError:
                pass

    def _set_reload_job(self, **updates: object) -> None:
        with self._reload_lock:
            self.reload_job.update(updates)

    def reload_snapshot(self) -> dict:
        with self._reload_lock:
            return dict(self.reload_job)

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None
        if self.overlay_handle is not None:
            self.overlay_handle.close()
            self.overlay_handle = None

    def _progress_callback(self, progress: dict) -> None:
        loaded = int(progress.get("loaded") or 0)
        total = progress.get("total")
        total_int = int(total) if total is not None else None
        percent = None
        if total_int:
            percent = round((loaded / total_int) * 100.0, 1)
        self._set_reload_job(
            state="running",
            stage="loading_bank",
            loaded=loaded,
            total=total_int,
            percent=percent,
            rate_per_second=float(progress.get("rate_per_second") or 0.0),
            eta_seconds=progress.get("eta_seconds"),
        )

    def reload(self) -> None:
        self._set_reload_job(
            state="running",
            stage="closing_existing_pipeline",
            loaded=0,
            total=None,
            percent=None,
            rate_per_second=0.0,
            eta_seconds=None,
            started_at=time.time(),
            finished_at=None,
            error="",
        )
        self.close()
        overlay = OverlayStore(self.overlay_path) if self.overlay_path.exists() else None
        bank = None
        try:
            bank = StreamingBank(
                self.bank_path,
                report_every=100_000,
                overlay=overlay,
                progress_callback=self._progress_callback,
            )
            self._set_reload_job(stage="initializing_answer_pipeline")
            generator = self.generator
            if generator is None:
                generator = LazyPhi3Generator(use_4bit=self.use_4bit)
            self.pipeline = AnswerPipeline(
                bank_path=self.bank_path,
                bank=bank,
                top_k=self.top_k,
                generator=generator,
                use_4bit=self.use_4bit,
            )
            self.overlay_handle = overlay
            self.overlay_loaded = overlay is not None
            self.reload_required = False
            self._set_reload_job(
                state="succeeded",
                stage="ready",
                percent=100.0,
                eta_seconds=0.0,
                finished_at=time.time(),
                error="",
            )
        except Exception as exc:
            if bank is not None:
                bank.close()
            if overlay is not None:
                overlay.close()
            self.pipeline = None
            self.overlay_handle = None
            self.overlay_loaded = False
            self._set_reload_job(
                state="failed",
                stage="failed",
                finished_at=time.time(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def start_reload(self) -> dict:
        snapshot = self.reload_snapshot()
        if snapshot.get("state") == "running":
            return snapshot

        def _worker() -> None:
            try:
                self.reload()
            except Exception:
                pass

        self._reload_thread = threading.Thread(target=_worker, daemon=True)
        self._set_reload_job(
            state="running",
            stage="queued",
            loaded=0,
            total=None,
            percent=None,
            rate_per_second=0.0,
            eta_seconds=None,
            started_at=time.time(),
            finished_at=None,
            error="",
        )
        self._reload_thread.start()
        return self.reload_snapshot()

    def ask(self, question: str, on_phase=None) -> dict:
        if self.reload_snapshot().get("state") == "running":
            raise WorkspaceError("bank reload is still running; wait until the status says Bank loaded")
        if self.pipeline is None:
            raise WorkspaceError("bank is not loaded; click Reload bank before asking")
        result = self.pipeline.ask(question, on_phase=on_phase)  # type: ignore[union-attr]
        payload = result.as_dict()
        record = {"ts": time.time(), "question": question, "result": payload}
        self.history.append(record)
        self._persist_qa(record)
        return payload

    def question_history(self, limit: int = 20) -> list[dict]:
        return list(reversed(self.history[-int(limit):]))

    def mark_reload_required(self) -> None:
        self.reload_required = True

    def status(self) -> dict:
        overlay_stats = overlay_summary(self.overlay_path)
        pipeline_loaded = self.pipeline is not None
        n_cells = None
        encoder_model = ""
        if self.pipeline is not None:
            n_cells = int(self.pipeline.bank.n_cells)
            encoder_model = self.pipeline.bank.encoder_model
        return {
            "bank_path": str(self.bank_path),
            "bank_exists": self.bank_path.exists(),
            "overlay_path": str(self.overlay_path),
            "overlay_exists": self.overlay_path.exists(),
            "overlay_loaded": self.overlay_loaded,
            "reload_required": self.reload_required,
            "pipeline_loaded": pipeline_loaded,
            "n_cells_loaded": n_cells,
            "encoder_model": encoder_model,
            "overlay": overlay_stats,
            "reload": self.reload_snapshot(),
            "history_count": len(self.history),
        }


def _normalise_text(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _pdf_literal_to_text(raw: bytes) -> str:
    body = raw[1:-1]
    body = body.replace(rb"\(", b"(").replace(rb"\)", b")")
    body = body.replace(rb"\n", b"\n").replace(rb"\r", b"\n").replace(rb"\t", b" ")
    return body.decode("latin-1", errors="ignore")


def extract_pdf_text(data: bytes) -> tuple[str, int, tuple[str, ...]]:
    warnings: list[str] = []
    page_count = max(0, data.count(b"/Type /Page") - data.count(b"/Type /Pages"))
    pieces: list[str] = []
    streams = list(_PDF_STREAM_RE.findall(data))
    for stream in streams:
        payload = stream.strip(b"\r\n")
        for candidate in (payload,):
            try:
                decompressor = zlib.decompressobj()
                inflated = decompressor.decompress(candidate, MAX_PDF_STREAM_INFLATE_BYTES)
                if decompressor.unconsumed_tail:
                    warnings.append("PDF stream exceeded inflate limit; truncated.")
                payload = inflated
                break
            except zlib.error:
                pass
        literals = [_pdf_literal_to_text(m.group(0)) for m in _PDF_TEXT_RE.finditer(payload)]
        if literals:
            pieces.append(" ".join(literals))
    text = _normalise_text("\n\n".join(pieces))
    if not text:
        warnings.append("No extractable PDF text found; scanned or complex PDFs may need OCR.")
    if not streams:
        warnings.append("No PDF streams found.")
    return text, page_count, tuple(warnings)


def extract_source(source_name: str, data: bytes) -> ExtractionResult:
    suffix = Path(source_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise WorkspaceError(f"unsupported file type {suffix!r}; use .txt, .md, or .pdf")
    if suffix in {".txt", ".md"}:
        text = data.decode("utf-8", errors="replace")
        return ExtractionResult(source_name=source_name, suffix=suffix, text=text)
    text, pages, warnings = extract_pdf_text(data)
    return ExtractionResult(
        source_name=source_name,
        suffix=suffix,
        text=text,
        pages_processed=pages,
        warnings=warnings,
    )


def extract_source_from_base64(source_name: str, data_base64: str) -> ExtractionResult:
    try:
        data = base64.b64decode(data_base64, validate=True)
    except ValueError as exc:
        raise WorkspaceError("file payload is not valid base64") from exc
    return extract_source(source_name, data)


def split_candidate_cells(
    text: str,
    *,
    source_name: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chars: int = MIN_CELL_CHARS,
) -> tuple[CandidateCell, ...]:
    if chunk_chars <= min_chars:
        raise ValueError("chunk_chars must be greater than min_chars")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must be >= 0 and less than chunk_chars")
    paragraphs = [_normalise_text(p) for p in re.split(r"\n\s*\n", text) if _normalise_text(p)]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                chunk = para[start:start + chunk_chars].strip()
                if chunk:
                    chunks.append(chunk)
                if start + chunk_chars >= len(para):
                    break
                start += chunk_chars - overlap
            continue
        proposed = para if not current else f"{current}\n\n{para}"
        if len(proposed) <= chunk_chars:
            current = proposed
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)

    cells: list[CandidateCell] = []
    stem = Path(source_name).stem or "source"
    for idx, chunk in enumerate(chunks, 1):
        accepted = len(chunk) >= min_chars
        cells.append(
            CandidateCell(
                draft_id=f"draft-{idx:04d}",
                text=chunk,
                label=f"{stem}_cell_{idx:04d}",
                accepted=accepted,
                rejection_reason="too short" if not accepted else "",
            )
        )
    return tuple(cells)


def build_ingestion_draft(extraction: ExtractionResult) -> IngestionDraft:
    candidates = split_candidate_cells(extraction.text, source_name=extraction.source_name)
    warnings = list(extraction.warnings)
    if not candidates:
        warnings.append("No candidate cells produced from extracted text.")
    return IngestionDraft(
        source_name=extraction.source_name,
        extracted_text=extraction.text,
        candidates=candidates,
        warnings=tuple(warnings),
        pages_processed=extraction.pages_processed,
    )


def _overlay_duplicate_texts(overlay_path: Path) -> set[str]:
    if not overlay_path.exists():
        return set()
    with sqlite3.connect(str(overlay_path)) as conn:
        return {_normalise_text(row[0]).lower() for row in conn.execute("SELECT source_text FROM overlay_cells")}


def commit_candidate_cells(
    *,
    bank_path: Path,
    overlay_path: Path,
    candidates: Sequence[CandidateCell],
    source_name: str,
    encoder: EncoderSingleton | None = None,
    bank: StreamingBank | None = None,
    smoke_question: str = "",
    pages_processed: int = 0,
    extracted_char_count: int | None = None,
    warnings: Sequence[str] = (),
) -> IngestionReport:
    opened_bank = bank is None
    active_bank = bank if bank is not None else StreamingBank(bank_path, report_every=0)
    active_encoder = encoder if encoder is not None else EncoderSingleton(
        model_name=active_bank.encoder_model or "all-MiniLM-L6-v2"
    )
    added: list[int] = []
    report_warnings: list[str] = list(warnings)
    duplicates = 0
    existing_texts = _overlay_duplicate_texts(overlay_path)
    try:
        with OverlayStore(overlay_path) as overlay:
            for cell in candidates:
                if not cell.accepted:
                    continue
                key = _normalise_text(cell.text).lower()
                if key in existing_texts:
                    duplicates += 1
                    continue
                cell_id, _ = add_cell_from_text(
                    overlay,
                    bank_dim=active_bank.dim,
                    base_max_id=active_bank.base_max_cell_id(),
                    encoder=active_encoder,
                    whiten_fn=active_bank.whiten,
                    text=cell.text,
                    source=source_name,
                    label=cell.label,
                )
                added.append(cell_id)
                existing_texts.add(key)
    finally:
        if opened_bank:
            active_bank.close()
    smoke = {"ran": False, "query": smoke_question, "hit_added_cell": False, "top_cell_id": None}
    if smoke_question and added:
        try:
            with OverlayStore(overlay_path) as overlay:
                smoke_bank = StreamingBank(bank_path, report_every=0, overlay=overlay)
                try:
                    raw = active_encoder.encode_one(smoke_question, is_query=True)
                    top = smoke_bank.topk(smoke_bank.whiten(raw), k=5)
                    ids = [int(c) for c in top["cell_ids"]]
                    smoke = {
                        "ran": True,
                        "query": smoke_question,
                        "hit_added_cell": any(cid in set(added) for cid in ids),
                        "top_cell_id": ids[0] if ids else None,
                    }
                finally:
                    smoke_bank.close()
        except Exception as exc:
            report_warnings.append(f"retrieval smoke test failed: {type(exc).__name__}: {exc}")
    accepted_count = sum(1 for c in candidates if c.accepted)
    return IngestionReport(
        source_name=source_name,
        pages_processed=int(pages_processed),
        text_char_count=(
            int(extracted_char_count)
            if extracted_char_count is not None
            else sum(len(c.text) for c in candidates)
        ),
        cells_proposed=len(candidates),
        cells_accepted=len(added),
        cells_rejected=len(candidates) - accepted_count,
        duplicates_found=duplicates,
        source_identifier=source_name,
        ingestion_timestamp=time.time(),
        warnings=tuple(report_warnings),
        added_cell_ids=tuple(added),
        retrieval_smoke_test=smoke,
        reload_required=bool(added),
    )


def overlay_summary(overlay_path: Path) -> dict:
    if not overlay_path.exists():
        return {"cell_count": 0, "tombstone_count": 0, "history_count": 0}
    with sqlite3.connect(str(overlay_path)) as conn:
        cell_count = int(conn.execute("SELECT COUNT(*) FROM overlay_cells").fetchone()[0])
        tombstone_count = int(conn.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0])
        history_count = int(conn.execute("SELECT COUNT(*) FROM provenance_log").fetchone()[0])
    return {
        "cell_count": cell_count,
        "tombstone_count": tombstone_count,
        "history_count": history_count,
    }


def search_cells(bank_path: Path, overlay_path: Path, query: str, limit: int = 20) -> tuple[CellSearchResult, ...]:
    if not query.strip():
        return ()
    pattern = f"%{query.strip()}%"
    results: list[CellSearchResult] = []
    tombstoned: set[int] = set()
    if overlay_path.exists():
        with sqlite3.connect(str(overlay_path)) as conn:
            tombstoned = {int(row[0]) for row in conn.execute("SELECT base_cell_id FROM tombstones")}
            for row in conn.execute(
                """SELECT id, label, source_text FROM overlay_cells
                   WHERE source_text LIKE ? OR label LIKE ?
                   ORDER BY id DESC LIMIT ?""",
                (pattern, pattern, int(limit)),
            ):
                results.append(CellSearchResult(int(row[0]), "overlay", row[1], row[2], int(row[0]) in tombstoned))
    remaining = max(0, limit - len(results))
    if remaining and bank_path.exists():
        uri = f"file:{bank_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            for row in conn.execute(
                """SELECT c.id, c.label, s.text
                   FROM source_texts s
                   JOIN cells c ON c.id = s.cell_id
                   WHERE s.text LIKE ? OR c.label LIKE ?
                   ORDER BY c.id ASC LIMIT ?""",
                (pattern, pattern, int(remaining)),
            ):
                cell_id = int(row[0])
                results.append(CellSearchResult(cell_id, "base", row[1], row[2], cell_id in tombstoned))
    return tuple(results)


def tombstone_cell(overlay_path: Path, cell_id: int, reason: str) -> dict:
    with OverlayStore(overlay_path) as overlay:
        overlay.remove_cell(cell_id, reason=reason)
    return {"cell_id": int(cell_id), "reason": reason, "reload_required": True}


def history(overlay_path: Path) -> list[dict]:
    if not overlay_path.exists():
        return []
    with OverlayStore(overlay_path) as overlay:
        return overlay.provenance()


def candidates_from_payload(raw_candidates: Sequence[dict]) -> tuple[CandidateCell, ...]:
    out: list[CandidateCell] = []
    for idx, raw in enumerate(raw_candidates, 1):
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        out.append(
            CandidateCell(
                draft_id=str(raw.get("draft_id") or f"draft-{idx:04d}"),
                text=text,
                label=str(raw.get("label") or f"manual_cell_{idx:04d}"),
                accepted=bool(raw.get("accepted", True)),
                rejection_reason=str(raw.get("rejection_reason", "")),
            )
        )
    return tuple(out)