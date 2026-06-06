"""Tests for the local Bank Management Workspace backend."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.bank_admin import OverlayStore
from src.agent.bank_workspace import (
    CandidateCell,
    HostedHttpGenerator,
    WorkspaceError,
    WorkspaceSession,
    build_generator_from_env,
    build_ingestion_draft,
    commit_candidate_cells,
    extract_pdf_text,
    extract_source,
    history,
    search_cells,
    tombstone_cell,
)
from src.agent.streaming_bank import StreamingBank


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS whitening (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    mu          BLOB NOT NULL,
    w_matrix    BLOB NOT NULL,
    max_norm    REAL NOT NULL,
    fitted_at   REAL NOT NULL,
    reference_n INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS cells (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT,
    weight      BLOB NOT NULL,
    theta       REAL NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('single', 'bound')),
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS source_texts (
    cell_id INTEGER PRIMARY KEY,
    text    TEXT NOT NULL,
    FOREIGN KEY (cell_id) REFERENCES cells(id) ON DELETE CASCADE
);
"""


class _StubEncoder:
    def __init__(self, dim: int):
        self.dim = dim
        self.model_name = "stub-encoder"
        self.responses: dict[str, np.ndarray] = {}

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.responses.get(text, np.eye(self.dim, dtype=np.float32)[3])


def _unit(v: np.ndarray) -> np.ndarray:
    return v.astype(np.float32) / (np.linalg.norm(v) + 1e-8)


def _make_bank(path: Path, dim: int = 8) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_SQL)
    conn.execute("INSERT INTO meta VALUES (?, ?)", ("dim", json.dumps(dim)))
    conn.execute("INSERT INTO meta VALUES (?, ?)", ("encoder_model", json.dumps("stub-encoder")))
    mu = np.zeros(dim, dtype=np.float32)
    w_matrix = np.eye(dim, dtype=np.float32)
    conn.execute(
        """INSERT INTO whitening (id, mu, w_matrix, max_norm, fitted_at, reference_n)
           VALUES (1, ?, ?, ?, ?, ?)""",
        (mu.tobytes(), w_matrix.tobytes(), 1.0, time.time(), 100),
    )
    cells = [
        ("cell_zebra", _unit(np.eye(dim, dtype=np.float32)[0]), "Zebras have black and white stripes."),
        ("cell_paris", _unit(np.eye(dim, dtype=np.float32)[1]), "Paris is the capital of France."),
    ]
    for label, weight, source in cells:
        cur = conn.execute(
            """INSERT INTO cells (label, weight, theta, kind, created_at)
               VALUES (?, ?, ?, 'single', ?)""",
            (label, weight.astype(np.float32).tobytes(), 0.3, time.time()),
        )
        conn.execute("INSERT INTO source_texts (cell_id, text) VALUES (?, ?)", (cur.lastrowid, source))
    conn.commit()
    conn.close()


def _make_pdf(text: str) -> bytes:
    stream = f"BT ({text}) Tj ET".encode("latin-1")
    return b"\n".join([
        b"%PDF-1.4",
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj",
        b"4 0 obj << /Length " + str(len(stream)).encode("ascii") + b" >> stream",
        stream,
        b"endstream endobj",
        b"%%EOF",
    ])


def test_given_text_source_when_previewed_then_candidate_cells_are_proposed():
    # Arrange
    text = "Mars has two moons, Phobos and Deimos.\n\nThe moons are small and irregular."

    # Act
    extraction = extract_source("mars.md", text.encode("utf-8"))
    draft = build_ingestion_draft(extraction)

    # Assert
    assert draft.source_name == "mars.md"
    assert draft.candidates
    assert draft.candidates[0].accepted is True
    assert "Mars" in draft.extracted_text


def test_given_pdf_source_when_previewed_then_text_is_extracted():
    # Act
    extraction = extract_source("guide.pdf", _make_pdf("Alpha PDF fact lives here."))

    # Assert
    assert extraction.pages_processed == 1
    assert "Alpha PDF fact" in extraction.text


def test_given_zip_bomb_pdf_stream_when_extracted_then_inflation_is_bounded():
    # Arrange: a single zlib stream that inflates to far beyond the cap.
    import zlib
    from src.agent.bank_workspace import MAX_PDF_STREAM_INFLATE_BYTES

    bomb = zlib.compress(b"\x00" * (MAX_PDF_STREAM_INFLATE_BYTES + (4 * 1024 * 1024)))
    pdf = b"\n".join([
        b"%PDF-1.4",
        b"3 0 obj << /Type /Page >> endobj",
        b"4 0 obj << /Length " + str(len(bomb)).encode("ascii") + b" >> stream",
        bomb,
        b"endstream endobj",
        b"%%EOF",
    ])

    # Act
    text, _pages, warnings = extract_pdf_text(pdf)

    # Assert: it does not OOM, and it flags the truncation.
    assert any("inflate limit" in w for w in warnings)
    assert len(text) <= MAX_PDF_STREAM_INFLATE_BYTES


def test_given_approved_cells_when_committed_then_overlay_records_provenance(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    overlay_path = tmp_path / "overlay.db"
    _make_bank(bank_path)
    bank = StreamingBank(bank_path, report_every=0)
    encoder = _StubEncoder(bank.dim)
    encoder.responses["Mars has two moons, Phobos and Deimos."] = np.eye(bank.dim, dtype=np.float32)[3]

    # Act
    try:
        report = commit_candidate_cells(
            bank_path=bank_path,
            overlay_path=overlay_path,
            candidates=(CandidateCell("draft-1", "Mars has two moons, Phobos and Deimos.", "mars"),),
            source_name="manual:mars",
            encoder=encoder,
            bank=bank,
            pages_processed=0,
            extracted_char_count=37,
        )
    finally:
        bank.close()

    # Assert
    assert report.cells_accepted == 1
    assert report.reload_required is True
    assert report.text_char_count == 37
    log = history(overlay_path)
    assert log[0]["op"] == "add"
    assert log[0]["source"] == "manual:mars"


def test_given_duplicate_candidate_when_committed_then_duplicate_is_reported(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    overlay_path = tmp_path / "overlay.db"
    _make_bank(bank_path)
    bank = StreamingBank(bank_path, report_every=0)
    encoder = _StubEncoder(bank.dim)
    candidate = CandidateCell("draft-1", "Mars has two moons, Phobos and Deimos.", "mars")

    # Act
    try:
        first = commit_candidate_cells(bank_path=bank_path, overlay_path=overlay_path, candidates=(candidate,), source_name="manual", encoder=encoder, bank=bank)
        second = commit_candidate_cells(bank_path=bank_path, overlay_path=overlay_path, candidates=(candidate,), source_name="manual", encoder=encoder, bank=bank)
    finally:
        bank.close()

    # Assert
    assert first.cells_accepted == 1
    assert second.cells_accepted == 0
    assert second.duplicates_found == 1


def test_given_overlay_and_base_cells_when_searching_then_sources_are_marked(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    overlay_path = tmp_path / "overlay.db"
    _make_bank(bank_path)
    with OverlayStore(overlay_path) as overlay:
        overlay.add_cell(weight=np.eye(8, dtype=np.float32)[3], source_text="Mars has two moons.", source="manual", label="mars", base_max_id=2)

    # Act
    results = search_cells(bank_path, overlay_path, "Mars")

    # Assert
    assert results[0].source == "overlay"
    assert results[0].text == "Mars has two moons."


def test_given_tombstone_when_history_loaded_then_reload_is_required(tmp_path: Path):
    # Arrange
    overlay_path = tmp_path / "overlay.db"

    # Act
    result = tombstone_cell(overlay_path, 1, "incorrect source")
    log = history(overlay_path)

    # Assert
    assert result["reload_required"] is True
    assert log[0]["op"] == "remove"
    assert log[0]["reason"] == "incorrect source"


def test_given_session_when_reloaded_then_status_reports_overlay_loaded(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    overlay_path = tmp_path / "overlay.db"
    _make_bank(bank_path)
    with OverlayStore(overlay_path) as overlay:
        overlay.add_cell(weight=np.eye(8, dtype=np.float32)[3], source_text="Mars has two moons.", base_max_id=2)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=overlay_path, generator=lambda _: "unused", use_4bit=False)

    # Act
    try:
        session.reload()
        status = session.status()
    finally:
        session.close()

    # Assert
    assert status["overlay_loaded"] is True
    assert status["overlay"]["cell_count"] == 1
    assert status["reload"]["state"] == "succeeded"
    assert status["reload"]["percent"] == 100.0


def test_given_streaming_bank_progress_callback_when_reloaded_then_status_records_progress(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    progress: list[dict] = []

    # Act
    bank = StreamingBank(bank_path, report_every=1, progress_callback=progress.append)
    try:
        pass
    finally:
        bank.close()

    # Assert
    assert progress[0]["loaded"] == 0
    assert progress[-1]["loaded"] == progress[-1]["total"] == 2
    assert progress[-1]["eta_seconds"] == 0.0


def test_given_session_when_background_reload_started_then_status_eventually_ready(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)

    # Act
    try:
        initial = session.start_reload()
        deadline = time.time() + 5
        status = session.status()
        while status["reload"]["state"] == "running" and time.time() < deadline:
            time.sleep(0.01)
            status = session.status()
    finally:
        session.close()

    # Assert
    assert initial["state"] == "running"
    assert status["reload"]["state"] == "succeeded"
    assert status["pipeline_loaded"] is True


def test_given_unloaded_session_when_asked_then_user_gets_reload_error(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)

    # Act / Assert
    try:
        try:
            session.ask("What is Mars?")
        except WorkspaceError as exc:
            assert "Reload bank" in str(exc)
        else:
            raise AssertionError("Expected unloaded session to require explicit reload")
    finally:
        session.close()


def test_given_prompt_when_hosted_generator_then_posts_openai_payload_and_parses():
    # Arrange
    captured: dict = {}

    def fake_transport(url, payload, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": "  grounded draft  "}}]}

    gen = HostedHttpGenerator(
        base_url="https://endpoint.example/v1/",
        model="my-model",
        api_key="secret-token",
        timeout=12.0,
        max_tokens=256,
        transport=fake_transport,
    )

    # Act
    result = gen("Why is the sky blue?")

    # Assert
    assert result == "grounded draft"
    assert captured["url"] == "https://endpoint.example/v1/chat/completions"
    assert captured["payload"]["model"] == "my-model"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "Why is the sky blue?"}]
    assert captured["payload"]["max_tokens"] == 256
    assert captured["payload"]["temperature"] == 0
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["timeout"] == 12.0


def test_given_no_api_key_when_hosted_generator_then_no_auth_header():
    # Arrange
    captured: dict = {}

    def fake_transport(url, payload, headers, timeout):
        captured["headers"] = headers
        return {"choices": [{"message": {"content": "ok"}}]}

    gen = HostedHttpGenerator(base_url="http://h/v1", model="m", transport=fake_transport)

    # Act
    gen("hi")

    # Assert
    assert "Authorization" not in captured["headers"]


def test_given_bad_response_shape_when_hosted_generator_then_workspace_error():
    # Arrange
    gen = HostedHttpGenerator(
        base_url="http://h/v1",
        model="m",
        transport=lambda *a: {"unexpected": True},
    )

    # Act / Assert
    try:
        gen("hi")
    except WorkspaceError as exc:
        assert "unexpected response shape" in str(exc)
    else:
        raise AssertionError("Expected WorkspaceError on malformed response")


def test_given_transport_failure_when_hosted_generator_then_workspace_error():
    # Arrange
    def boom(*_args):
        raise ConnectionError("refused")

    gen = HostedHttpGenerator(base_url="http://h/v1", model="m", transport=boom)

    # Act / Assert
    try:
        gen("hi")
    except WorkspaceError as exc:
        assert "request failed" in str(exc)
    else:
        raise AssertionError("Expected WorkspaceError on transport failure")


def test_given_base_url_env_when_build_generator_then_hosted_instance():
    # Arrange
    env = {
        "WORKSPACE_LLM_BASE_URL": "http://gpu-host:8000/v1",
        "WORKSPACE_LLM_MODEL": "phi3-vllm",
        "WORKSPACE_LLM_API_KEY": "k",
        "WORKSPACE_LLM_TIMEOUT": "30",
        "WORKSPACE_LLM_MAX_TOKENS": "256",
    }

    # Act
    gen = build_generator_from_env(env)

    # Assert
    assert isinstance(gen, HostedHttpGenerator)
    assert gen.base_url == "http://gpu-host:8000/v1"
    assert gen.model == "phi3-vllm"
    assert gen.api_key == "k"
    assert gen.timeout == 30.0
    assert gen.max_tokens == 256


def test_given_no_base_url_env_when_build_generator_then_none():
    # Act / Assert
    assert build_generator_from_env({}) is None
    assert build_generator_from_env({"WORKSPACE_LLM_BASE_URL": "   "}) is None