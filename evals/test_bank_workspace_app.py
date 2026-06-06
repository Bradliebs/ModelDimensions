"""API tests for the local Bank Management Workspace."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bank_workspace import SecurityConfig, create_app
from src.agent.bank_workspace import WorkspaceSession


_SCHEMA_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE whitening (id INTEGER PRIMARY KEY CHECK (id = 1), mu BLOB NOT NULL, w_matrix BLOB NOT NULL, max_norm REAL NOT NULL, fitted_at REAL NOT NULL, reference_n INTEGER NOT NULL);
CREATE TABLE cells (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, weight BLOB NOT NULL, theta REAL NOT NULL, kind TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE source_texts (cell_id INTEGER PRIMARY KEY, text TEXT NOT NULL);
"""


def _make_bank(path: Path) -> None:
    dim = 4
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_SQL)
    conn.execute("INSERT INTO meta VALUES (?, ?)", ("dim", json.dumps(dim)))
    conn.execute("INSERT INTO meta VALUES (?, ?)", ("encoder_model", json.dumps("stub-encoder")))
    conn.execute(
        "INSERT INTO whitening VALUES (1, ?, ?, ?, ?, ?)",
        (np.zeros(dim, dtype=np.float32).tobytes(), np.eye(dim, dtype=np.float32).tobytes(), 1.0, time.time(), 1),
    )
    cur = conn.execute(
        "INSERT INTO cells (label, weight, theta, kind, created_at) VALUES (?, ?, ?, ?, ?)",
        ("cell_paris", np.eye(dim, dtype=np.float32)[0].tobytes(), 0.3, "single", time.time()),
    )
    conn.execute("INSERT INTO source_texts VALUES (?, ?)", (cur.lastrowid, "Paris is the capital of France."))
    conn.commit()
    conn.close()


def test_given_workspace_app_when_status_requested_then_returns_bank_state(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    client = TestClient(create_app(session))

    # Act
    response = client.get("/api/status")

    # Assert
    assert response.status_code == 200
    assert response.json()["bank_exists"] is True


def test_given_text_preview_request_when_posted_then_returns_candidate_cells(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    client = TestClient(create_app(session))

    # Act
    response = client.post(
        "/api/preview",
        json={"source_name": "notes.md", "text": "Saturn has rings and many moons."},
    )

    # Assert
    assert response.status_code == 200
    assert response.json()["cells_proposed"] == 1


def test_given_tombstone_request_when_posted_then_reload_status_is_set(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    client = TestClient(create_app(session))

    # Act
    response = client.post("/api/tombstone", json={"cell_id": 1, "reason": "bad source"})
    status = client.get("/api/status").json()

    # Assert
    assert response.status_code == 200
    assert status["reload_required"] is True


def test_given_unloaded_workspace_when_asked_then_conflict_explains_reload(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    client = TestClient(create_app(session))

    # Act
    response = client.post("/api/ask", json={"question": "What is Mars?"})

    # Assert
    assert response.status_code == 409
    assert "Reload bank" in response.json()["detail"]


def test_given_unloaded_workspace_when_ask_stream_then_failure_event_then_done(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    client = TestClient(create_app(session))

    # Act
    with client.stream("GET", "/api/ask/stream", params={"question": "What is Mars?"}) as response:
        body = "".join(response.iter_text())

    # Assert: the bank is not loaded, so the stream reports an honest failure
    # then terminates with a done event (never a fabricated answer).
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: failure" in body
    assert "not loaded" in body
    assert body.rstrip().endswith("event: done\ndata: {}")


def test_given_blank_question_when_ask_stream_then_422(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    client = TestClient(create_app(session))

    # Act
    response = client.get("/api/ask/stream", params={"question": "   "})

    # Assert
    assert response.status_code == 422


def test_given_reload_request_when_posted_then_background_status_is_reported(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    client = TestClient(create_app(session))

    # Act
    response = client.post("/api/reload", json={})
    deadline = time.time() + 5
    status = response.json()
    while status["reload"]["state"] == "running" and time.time() < deadline:
        time.sleep(0.01)
        status = client.get("/api/status").json()

    # Assert
    assert response.status_code == 200
    assert status["reload"]["state"] == "succeeded"
    assert status["pipeline_loaded"] is True


def test_given_question_history_when_requested_then_recent_questions_return(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    session.history.append({"ts": 1.0, "question": "What is Paris?", "result": {"answer": "Paris."}})
    client = TestClient(create_app(session))

    # Act
    response = client.get("/api/qa-history")

    # Assert
    assert response.status_code == 200
    assert response.json()["history"][0]["question"] == "What is Paris?"


def test_given_health_route_when_requested_then_reports_liveness(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    client = TestClient(create_app(session))

    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["pipeline_loaded"] is False


def test_given_qa_history_path_default_then_derived_from_overlay_dir(tmp_path: Path):
    # Arrange / Act
    overlay_path = tmp_path / "nested" / "overlay.db"
    session = WorkspaceSession(bank_path=tmp_path / "bank.db", overlay_path=overlay_path, generator=lambda _: "unused", use_4bit=False)

    # Assert
    assert session.qa_history_path == overlay_path.parent / "qa_history.jsonl"


def test_given_persisted_qa_history_when_new_session_starts_then_history_is_restored(tmp_path: Path):
    # Arrange
    qa_path = tmp_path / "qa_history.jsonl"
    first = WorkspaceSession(bank_path=tmp_path / "bank.db", overlay_path=tmp_path / "overlay.db", qa_history_path=qa_path, generator=lambda _: "unused", use_4bit=False)
    first._persist_qa({"ts": 1.0, "question": "What is Paris?", "result": {"answer": "Paris."}})
    first._persist_qa({"ts": 2.0, "question": "What is Mars?", "result": {"answer": "Mars."}})

    # Act
    second = WorkspaceSession(bank_path=tmp_path / "bank.db", overlay_path=tmp_path / "overlay.db", qa_history_path=qa_path, generator=lambda _: "unused", use_4bit=False)
    client = TestClient(create_app(second))
    response = client.get("/api/qa-history")

    # Assert
    assert response.status_code == 200
    history = response.json()["history"]
    assert history[0]["question"] == "What is Mars?"
    assert history[1]["question"] == "What is Paris?"


def test_given_persisted_qa_history_when_loaded_then_capped_at_max(tmp_path: Path):
    # Arrange
    qa_path = tmp_path / "qa_history.jsonl"
    writer = WorkspaceSession(bank_path=tmp_path / "bank.db", overlay_path=tmp_path / "overlay.db", qa_history_path=qa_path, generator=lambda _: "unused", use_4bit=False)
    for i in range(5):
        writer._persist_qa({"ts": float(i), "question": f"Q{i}", "result": {}})

    # Act
    reader = WorkspaceSession(bank_path=tmp_path / "bank.db", overlay_path=tmp_path / "overlay.db", qa_history_path=qa_path, max_persisted_qa=3, generator=lambda _: "unused", use_4bit=False)

    # Assert
    assert len(reader.history) == 3
    assert reader.history[0]["question"] == "Q2"
    assert reader.history[-1]["question"] == "Q4"


def test_given_loaded_pipeline_when_commit_then_reuses_loaded_bank(tmp_path: Path, monkeypatch):
    # Arrange
    from types import SimpleNamespace
    import app.bank_workspace as appmod

    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    sentinel_bank = object()
    sentinel_encoder = object()
    session.pipeline = SimpleNamespace(bank=sentinel_bank, encoder=sentinel_encoder)

    captured: dict = {}

    def fake_commit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(reload_required=False, as_dict=lambda: {"ok": True})

    monkeypatch.setattr(appmod, "commit_candidate_cells", fake_commit)
    client = TestClient(create_app(session))

    # Act
    response = client.post("/api/commit", json={"candidates": [], "source_name": "x.txt"})

    # Assert
    assert response.status_code == 200
    assert captured["bank"] is sentinel_bank
    assert captured["encoder"] is sentinel_encoder


def test_given_overflowing_qa_history_when_loaded_then_file_is_truncated(tmp_path: Path):
    # Arrange
    qa_path = tmp_path / "qa_history.jsonl"
    writer = WorkspaceSession(bank_path=tmp_path / "bank.db", overlay_path=tmp_path / "overlay.db", qa_history_path=qa_path, generator=lambda _: "unused", use_4bit=False)
    for i in range(6):
        writer._persist_qa({"ts": float(i), "question": f"Q{i}", "result": {}})

    # Act
    WorkspaceSession(bank_path=tmp_path / "bank.db", overlay_path=tmp_path / "overlay.db", qa_history_path=qa_path, max_persisted_qa=2, generator=lambda _: "unused", use_4bit=False)

    # Assert
    lines = [ln for ln in qa_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["question"] == "Q4"
    assert json.loads(lines[-1])["question"] == "Q5"


def _secured_client(tmp_path: Path, security: SecurityConfig) -> TestClient:
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    session = WorkspaceSession(bank_path=bank_path, overlay_path=tmp_path / "overlay.db", generator=lambda _: "unused", use_4bit=False)
    return TestClient(create_app(session, security=security))


def test_given_no_security_config_then_routes_stay_open(tmp_path: Path):
    # Arrange / Act
    client = _secured_client(tmp_path, SecurityConfig())
    response = client.post("/api/tombstone", json={"cell_id": 999, "reason": "x"})

    # Assert: reaches the handler (400 for missing cell), not blocked by auth (401).
    assert response.status_code != 401


def test_given_api_token_when_write_without_token_then_unauthorized(tmp_path: Path):
    # Arrange
    client = _secured_client(tmp_path, SecurityConfig(api_token="secret"))

    # Act
    response = client.post("/api/tombstone", json={"cell_id": 1, "reason": "x"})

    # Assert
    assert response.status_code == 401


def test_given_api_token_when_write_with_token_then_authorized(tmp_path: Path):
    # Arrange
    client = _secured_client(tmp_path, SecurityConfig(api_token="secret"))

    # Act
    response = client.post(
        "/api/tombstone",
        json={"cell_id": 999, "reason": "x"},
        headers={"Authorization": "Bearer secret"},
    )

    # Assert: passes auth; handler responds (not 401).
    assert response.status_code != 401


def test_given_api_token_when_reads_open_then_status_allowed_without_token(tmp_path: Path):
    # Arrange: token set, but reads not gated.
    client = _secured_client(tmp_path, SecurityConfig(api_token="secret"))

    # Act
    response = client.get("/api/status")

    # Assert
    assert response.status_code == 200


def test_given_require_auth_for_reads_when_status_without_token_then_unauthorized(tmp_path: Path):
    # Arrange
    client = _secured_client(tmp_path, SecurityConfig(api_token="secret", require_auth_for_reads=True))

    # Act
    response = client.get("/api/status")

    # Assert
    assert response.status_code == 401


def test_given_body_limit_when_request_exceeds_then_payload_too_large(tmp_path: Path):
    # Arrange
    client = _secured_client(tmp_path, SecurityConfig(max_body_bytes=64))

    # Act: oversized preview payload.
    response = client.post("/api/preview", json={"source_name": "x.txt", "text": "z" * 500})

    # Assert
    assert response.status_code == 413


def test_given_rate_limit_when_exceeded_then_too_many_requests(tmp_path: Path):
    # Arrange
    client = _secured_client(tmp_path, SecurityConfig(rate_limit_per_minute=2))

    # Act
    statuses = [client.post("/api/ask", json={"question": "hi"}).status_code for _ in range(3)]

    # Assert: first two pass the limiter, the third is throttled.
    assert statuses[0] != 429
    assert statuses[1] != 429
    assert statuses[2] == 429


def test_given_env_then_security_config_parsed(monkeypatch):
    # Arrange
    env = {
        "WORKSPACE_API_TOKEN": "tok",
        "WORKSPACE_REQUIRE_AUTH_READS": "true",
        "WORKSPACE_MAX_BODY_MB": "5",
        "WORKSPACE_RATE_LIMIT_PER_MIN": "30",
    }

    # Act
    cfg = SecurityConfig.from_env(env)

    # Assert
    assert cfg.api_token == "tok"
    assert cfg.require_auth_for_reads is True
    assert cfg.max_body_bytes == 5 * 1024 * 1024
    assert cfg.rate_limit_per_minute == 30

