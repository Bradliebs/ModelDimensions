"""Offline tests for the V1 answer pipeline.

Builds a tiny SQLite bank matching the production schema, wires in a stub
generator (so we don't need Phi-3 weights), and validates the four
behavioural contracts:

  1. Known question → grounded answer + correct citation.
  2. Unknown question → honest silence (gate did not fire).
  3. Generator drift → honest silence (verifier rejected).
  4. PipelineResult shape matches the documented contract.

We intentionally do NOT touch the 5.7M-cell production bank here; that's
the job of ``scripts/ask.py`` smoke runs.
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

from src.agent.answer_pipeline import (
    AnswerPipeline,
    PipelineResult,
    SILENCE_DRIFT,
    SILENCE_NO_MATCH,
)


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


def _make_bank(path: Path, dim: int,
               cells: list[tuple[str, np.ndarray, float, str]]) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_SQL)
    # Match cc_service: JSON-encoded meta values.
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
    for label, weight, theta, source in cells:
        cur = conn.execute(
            """INSERT INTO cells (label, weight, theta, kind, created_at)
               VALUES (?, ?, ?, 'single', ?)""",
            (label, weight.astype(np.float32).tobytes(),
             float(theta), time.time()),
        )
        conn.execute(
            "INSERT INTO source_texts (cell_id, text) VALUES (?, ?)",
            (cur.lastrowid, source),
        )
    conn.commit()
    conn.close()


class _StubEncoder:
    """Encoder that returns the vector keyed off the question string.

    Tests register ``encoder.responses[question] = vector`` and the
    pipeline gets that exact vector back.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.model_name = "stub-encoder"
        self.responses: dict[str, np.ndarray] = {}

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        if text in self.responses:
            return self.responses[text].astype(np.float32)
        # Default: tiny random-ish vector that won't fire on a known cell.
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32) * 0.01
        return v


def _unit(v: np.ndarray) -> np.ndarray:
    return v.astype(np.float32) / (np.linalg.norm(v) + 1e-8)


# ---------- fixtures ----------

@pytest.fixture
def tiny_bank(tmp_path: Path) -> Path:
    """3-cell bank with orthogonal-ish weights so retrieval is unambiguous."""
    db = tmp_path / "bank.db"
    dim = 8
    # Cells are unit vectors along distinct axes plus a bit of mass elsewhere.
    weights = []
    eye = np.eye(dim, dtype=np.float32)
    cells = [
        ("cell_zebra", eye[0], 0.3,
         "Zebras have black and white stripes and live in Africa."),
        ("cell_paris", eye[1], 0.3,
         "Paris is the capital of France and sits on the river Seine."),
        ("cell_chess", eye[2], 0.3,
         "Chess is a two-player strategy game played on a 64-square board."),
    ]
    _make_bank(db, dim, cells)
    return db


# ---------- tests ----------

def test_known_question_returns_grounded_answer(tiny_bank: Path):
    encoder = _StubEncoder(dim=8)
    # Question maps cleanly onto cell_zebra's axis.
    encoder.responses["What do zebras look like?"] = np.eye(8, dtype=np.float32)[0]

    def stub_gen(prompt: str) -> str:
        # Phi-3 stand-in: returns a paraphrase that uses the cell's vocabulary.
        return "Zebras have black and white stripes. [1]"

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=stub_gen,
    )
    try:
        result = pipeline.ask("What do zebras look like?")
    finally:
        pipeline.close()

    assert isinstance(result, PipelineResult)
    assert result.silence is False, result.silence_reason
    assert "stripes" in result.answer.lower()
    assert 1 in result.citations
    assert result.verification is not None
    assert result.verification["grounded"] is True
    assert result.gate["fire"] is True
    assert result.gate["margin"] > 0.05


def test_unknown_question_triggers_silence(tiny_bank: Path):
    encoder = _StubEncoder(dim=8)
    # Vector orthogonal to every cell axis → all activations ~0.
    encoder.responses["What color is the king of Mars?"] = (
        np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    )

    def stub_gen(prompt: str) -> str:  # pragma: no cover - shouldn't be called
        raise AssertionError("generator must not run when gate silences")

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=stub_gen,
    )
    try:
        result = pipeline.ask("What color is the king of Mars?")
    finally:
        pipeline.close()

    assert result.silence is True
    assert result.answer == SILENCE_NO_MATCH
    assert result.gate["fire"] is False
    assert result.citations == []
    assert result.verification is None


def test_drifting_generator_triggers_silence(tiny_bank: Path):
    encoder = _StubEncoder(dim=8)
    encoder.responses["Tell me about Paris."] = np.eye(8, dtype=np.float32)[1]

    def drift_gen(prompt: str) -> str:
        # Answer that has no overlap with the cited Paris cell — pure drift.
        return (
            "Quantum entanglement involves correlated photon polarizations "
            "across spacelike separated detectors."
        )

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=drift_gen,
    )
    try:
        result = pipeline.ask("Tell me about Paris.")
    finally:
        pipeline.close()

    assert result.silence is True
    assert result.answer == SILENCE_DRIFT
    assert result.gate["fire"] is True              # retrieval was fine
    assert result.verification is not None
    assert result.verification["grounded"] is False
    assert result.verification["coverage"] < 0.5
    # Citations are still surfaced so the caller can audit what was tried.
    assert 2 in result.citations


def test_pipeline_result_shape(tiny_bank: Path):
    encoder = _StubEncoder(dim=8)
    encoder.responses["chess?"] = np.eye(8, dtype=np.float32)[2]
    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=lambda p: "Chess is played on a 64-square board. [3]",
    )
    try:
        result = pipeline.ask("chess?")
    finally:
        pipeline.close()

    d = result.as_dict()
    for key in ("question", "answer", "silence", "silence_reason",
                "citations", "retrieval", "gate", "verification", "timings",
                "closest_topics"):
        assert key in d, f"missing key: {key}"
    assert set(d["retrieval"]) >= {"top_k_cell_ids", "activations", "thetas"}
    assert set(d["gate"]) >= {"fire", "top1_activation", "top2_activation",
                              "margin", "threshold", "reason"}
    assert all(t >= 0 for t in d["timings"].values())
    # closest_topics is informationally populated even on grounded answers.
    assert isinstance(d["closest_topics"], list)
    assert len(d["closest_topics"]) >= 1
    for topic in d["closest_topics"]:
        assert "topic" in topic and "activation" in topic


def test_silence_includes_closest_topics(tiny_bank: Path):
    """Progressive disclosure: silence response surfaces top-3 topic snippets."""
    encoder = _StubEncoder(dim=8)
    # Vector orthogonal to every cell axis -> diffuse activations -> silence.
    encoder.responses["What color is the king of Mars?"] = (
        np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    )

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=lambda p: "should not run",
    )
    try:
        result = pipeline.ask("What color is the king of Mars?")
    finally:
        pipeline.close()

    assert result.silence is True
    assert result.gate["fire"] is False
    # Top-3 topics surfaced; topic strings are the cell labels we stored.
    assert len(result.closest_topics) == 3
    topics = {t["topic"] for t in result.closest_topics}
    assert topics == {"cell_zebra", "cell_paris", "cell_chess"}
    # Activations sorted descending.
    acts = [t["activation"] for t in result.closest_topics]
    assert acts == sorted(acts, reverse=True)


def test_drift_with_uncited_year_rejected(tiny_bank: Path):
    """Stage B: an answer that introduces a year not in the cited cells fails."""
    encoder = _StubEncoder(dim=8)
    encoder.responses["When was Paris founded?"] = np.eye(8, dtype=np.float32)[1]

    def confab_year(prompt: str) -> str:
        # Token coverage with the Paris cell is fine, but "1872" is fabricated.
        return "Paris is the capital of France and was founded in 1872. [2]"

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=confab_year,
    )
    try:
        result = pipeline.ask("When was Paris founded?")
    finally:
        pipeline.close()

    assert result.silence is True
    assert result.answer == SILENCE_DRIFT
    assert result.verification is not None
    assert result.verification["grounded"] is False
    # "1872" should be reported as the uncited numeric.
    assert "1872" in result.verification["uncited_numerics"]


def test_encoder_mismatch_rejected(tiny_bank: Path):
    """Bank says encoder=stub-encoder; pipeline must reject a different one."""

    class WrongEncoder:
        dim = 8
        model_name = "different-encoder"

        def encode_one(self, text, is_query=False):  # pragma: no cover
            return np.zeros(8, dtype=np.float32)

    with pytest.raises(ValueError, match="encoder mismatch"):
        AnswerPipeline(
            bank_path=tiny_bank,
            encoder=WrongEncoder(),     # type: ignore[arg-type]
            generator=lambda p: "",
        )
