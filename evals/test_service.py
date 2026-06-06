"""Tests for the stdlib HTTP service wrapper around :class:`AnswerPipeline`.

We spin up the server on an ephemeral port against a tiny in-memory bank
fixture and stub generator, then exercise /health and /ask via
``http.client``. No network listeners survive the test; the server is
shut down in teardown.
"""
from __future__ import annotations

import http.client
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.answer_pipeline import AnswerPipeline
from src.agent.service import make_server


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


def _make_bank(path: Path, dim: int = 4) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_SQL)
    conn.execute("INSERT INTO meta VALUES (?, ?)", ("dim", json.dumps(dim)))
    conn.execute("INSERT INTO meta VALUES (?, ?)",
                 ("encoder_model", json.dumps("stub-encoder")))
    mu = np.zeros(dim, dtype=np.float32)
    w_matrix = np.eye(dim, dtype=np.float32)
    conn.execute(
        """INSERT INTO whitening (id, mu, w_matrix, max_norm, fitted_at,
                                  reference_n) VALUES (1, ?, ?, ?, ?, ?)""",
        (mu.tobytes(), w_matrix.tobytes(), 1.0, time.time(), 100),
    )
    eye = np.eye(dim, dtype=np.float32)
    cells = [
        ("zebra", eye[0], 0.3, "Zebras have stripes and live in Africa."),
        ("paris", eye[1], 0.3, "Paris is the capital of France."),
        ("chess", eye[2], 0.3, "Chess is a board game with 64 squares."),
    ]
    for label, weight, theta, source in cells:
        cur = conn.execute(
            """INSERT INTO cells (label, weight, theta, kind, created_at)
               VALUES (?, ?, ?, 'single', ?)""",
            (label, weight.tobytes(), theta, time.time()),
        )
        conn.execute(
            "INSERT INTO source_texts (cell_id, text) VALUES (?, ?)",
            (cur.lastrowid, source),
        )
    conn.commit()
    conn.close()


class _StubEncoder:
    def __init__(self, dim: int = 4):
        self.dim = dim
        self.model_name = "stub-encoder"
        self.responses: dict[str, np.ndarray] = {}

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        if text in self.responses:
            return self.responses[text].astype(np.float32)
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(self.dim).astype(np.float32) * 0.01


@pytest.fixture
def running_service(tmp_path: Path):
    """Start the service in a background thread on an ephemeral port."""
    db = tmp_path / "bank.db"
    _make_bank(db, dim=4)
    encoder = _StubEncoder(dim=4)
    encoder.responses["What do zebras look like?"] = np.eye(4, dtype=np.float32)[0]

    pipeline = AnswerPipeline(
        bank_path=db,
        top_k=3,
        encoder=encoder,
        generator=lambda p: "Zebras have stripes. [1]",
    )

    # Port 0 -> kernel picks an ephemeral free port.
    server = make_server(pipeline, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        pipeline.close()


def _request(port: int, method: str, path: str,
             body: dict | None = None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        if body is None:
            conn.request(method, path)
        else:
            payload = json.dumps(body).encode("utf-8")
            conn.request(method, path, body=payload,
                         headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        status = resp.status
        raw = resp.read().decode("utf-8")
    finally:
        conn.close()
    return status, json.loads(raw) if raw else {}


def test_health_returns_pipeline_metadata(running_service: int):
    status, body = _request(running_service, "GET", "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["n_cells"] == 3
    assert body["encoder_model"] == "stub-encoder"


def test_ask_returns_grounded_answer(running_service: int):
    status, body = _request(
        running_service, "POST", "/ask",
        {"question": "What do zebras look like?"},
    )
    assert status == 200
    assert body["silence"] is False
    assert "stripes" in body["answer"].lower()
    # Top-k=3 means all 3 cells are sent and cited; we only need the
    # zebra cell (id=1) to be among them.
    assert 1 in body["citations"]


def test_ask_returns_silence_with_closest_topics(running_service: int):
    status, body = _request(
        running_service, "POST", "/ask",
        {"question": "What color is the king of Mars?"},
    )
    assert status == 200
    assert body["silence"] is True
    assert body["gate"]["fire"] is False
    assert len(body["closest_topics"]) >= 1


def test_ask_rejects_missing_question(running_service: int):
    status, body = _request(running_service, "POST", "/ask", {})
    assert status == 400
    assert "error" in body


def test_ask_rejects_empty_question(running_service: int):
    status, body = _request(
        running_service, "POST", "/ask", {"question": "   "}
    )
    assert status == 400


def test_unknown_path_returns_404(running_service: int):
    status, _ = _request(running_service, "GET", "/nope")
    assert status == 404
