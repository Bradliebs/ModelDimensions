"""Smoke test for the V1 batch eval harness (exp17).

Verifies the pure aggregation logic end-to-end against a tiny in-memory
fixture without booting Phi-3 or the real bank. We run three queries —
one known, one unknown, one drift — and assert the harness:

  - classifies each correctly (grounded / silence_gate / silence_drift)
  - reports per-class accuracy
  - records latency aggregates with non-NaN p50/p95/mean
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.answer_pipeline import AnswerPipeline
from experiments.exp17_v1_pipeline_eval import run_eval, _classify


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
        if text in self.responses:
            return self.responses[text].astype(np.float32)
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(self.dim).astype(np.float32) * 0.01


@pytest.fixture
def harness_pipeline(tmp_path: Path):
    db = tmp_path / "bank.db"
    dim = 8
    conn = sqlite3.connect(str(db))
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
        ("zebra", eye[0], "Zebras have stripes and live in Africa."),
        ("paris", eye[1], "Paris is the capital of France."),
    ]
    for label, w, src in cells:
        cur = conn.execute(
            """INSERT INTO cells (label, weight, theta, kind, created_at)
               VALUES (?, ?, ?, 'single', ?)""",
            (label, w.tobytes(), 0.3, time.time()),
        )
        conn.execute(
            "INSERT INTO source_texts (cell_id, text) VALUES (?, ?)",
            (cur.lastrowid, src),
        )
    conn.commit()
    conn.close()

    encoder = _StubEncoder(dim=dim)
    # known: aligned with cell_zebra
    encoder.responses["zebras?"] = eye[0]
    # unknown: orthogonal -> diffuse activations -> silence_gate
    encoder.responses["mars king color?"] = np.array(
        [0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32
    )
    # drift: aligned with cell_paris but generator confabulates
    encoder.responses["paris drift?"] = eye[1]

    def gen(prompt: str) -> str:
        # The harness invokes ask() per query; we differentiate by prompt
        # contents (the question is appended to the FACTS block).
        if "zebras?" in prompt:
            return "Zebras have stripes. [1]"
        if "paris drift?" in prompt:
            return "Quantum entanglement involves correlated photons."
        return ""

    pipeline = AnswerPipeline(
        bank_path=db,
        top_k=2,
        encoder=encoder,
        generator=gen,
    )
    yield pipeline
    pipeline.close()


def test_classify_maps_outcomes():
    assert _classify({"silence": False, "silence_reason": ""}) == "grounded"
    assert _classify(
        {"silence": True, "silence_reason": "gate: ..."}
    ) == "silence_gate"
    assert _classify(
        {"silence": True, "silence_reason": "verify: coverage=0.10 ..."}
    ) == "silence_drift"


def test_run_eval_aggregates_correctly(harness_pipeline):
    queries = {
        "known":   [{"query": "zebras?", "expected": "grounded"}],
        "unknown": [{"query": "mars king color?", "expected": "silence"}],
        "noise":   [{"query": "paris drift?", "expected": "silence"}],
    }
    summary = run_eval(harness_pipeline, queries)

    assert summary["n_queries"] == 3
    by_class = summary["by_class"]
    assert by_class["known"]["accuracy"] == 1.0
    assert by_class["known"]["outcome_counts"]["grounded"] == 1
    assert by_class["unknown"]["accuracy"] == 1.0
    assert by_class["unknown"]["outcome_counts"]["silence_gate"] == 1
    # Drift case: gate fires (vector aligns with paris), generator confabulates.
    assert by_class["noise"]["accuracy"] == 1.0
    assert by_class["noise"]["outcome_counts"]["silence_drift"] == 1

    lat = summary["latency_seconds"]
    assert lat["total_wall"]["n"] == 3
    assert lat["total_wall"]["p50"] is not None
    assert lat["total_wall"]["p50"] >= 0.0
    # Generate runs only when gate fires (known + drift = 2).
    assert lat["generate_when_run"]["n"] == 2
