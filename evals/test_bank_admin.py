"""Tests for the V1 overlay editable bank (``src/agent/bank_admin.py``).

Contract from PLAN.md Step 4: ``add cell → answer changes → remove → silence``,
with provenance logging. The production bank is read-only; mutations land
in a separate overlay SQLite under ``results/v1_bank/overlay.db`` (here a
tmp_path overlay so tests don't touch the real path).
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
    SILENCE_NO_MATCH,
)
from src.agent.bank_admin import (
    OverlayStore,
    OverlayStoreError,
    add_cell_from_text,
)
from src.agent.streaming_bank import StreamingBank


# ---------- shared fixtures (mirror test_answer_pipeline.py) ----------

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
    def __init__(self, dim: int):
        self.dim = dim
        self.model_name = "stub-encoder"
        self.responses: dict[str, np.ndarray] = {}

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        if text in self.responses:
            return self.responses[text].astype(np.float32)
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32) * 0.01
        return v


@pytest.fixture
def tiny_bank(tmp_path: Path) -> Path:
    db = tmp_path / "bank.db"
    dim = 8
    eye = np.eye(dim, dtype=np.float32)
    cells = [
        ("cell_zebra", eye[0], 0.3,
         "Zebras have black and white stripes and live in Africa."),
        ("cell_paris", eye[1], 0.3,
         "Paris is the capital of France."),
        ("cell_chess", eye[2], 0.3,
         "Chess is a two-player strategy game on a 64-square board."),
    ]
    _make_bank(db, dim, cells)
    return db


# ---------- OverlayStore unit tests ----------

def test_overlay_creates_schema(tmp_path: Path):
    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        conn = sqlite3.connect(str(overlay.db_path))
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        conn.close()
    finally:
        overlay.close()
    for required in ("overlay_cells", "tombstones", "provenance_log", "meta"):
        assert required in tables, f"missing table {required}"


def test_overlay_add_allocates_above_base(tmp_path: Path):
    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        w = np.eye(8, dtype=np.float32)[3]
        new_id = overlay.add_cell(
            weight=w, source_text="Mars is a planet.",
            source="manual", label="cell_mars", base_max_id=3,
        )
        # Anchored at base_max_id = 3 → first overlay id = 4.
        assert new_id == 4
        new_id2 = overlay.add_cell(
            weight=w, source_text="Mars has two moons.",
            base_max_id=3,
        )
        assert new_id2 == 5
    finally:
        overlay.close()


def test_overlay_rejects_grown_base(tmp_path: Path):
    """If the user re-points the overlay at a larger base bank, refuse."""
    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        w = np.eye(8, dtype=np.float32)[3]
        overlay.add_cell(weight=w, source_text="hello", base_max_id=3)
        with pytest.raises(OverlayStoreError):
            overlay.add_cell(weight=w, source_text="bye", base_max_id=99)
    finally:
        overlay.close()


def test_overlay_provenance_log_records_add_and_remove(tmp_path: Path):
    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        w = np.eye(8, dtype=np.float32)[3]
        new_id = overlay.add_cell(
            weight=w, source_text="Mars is a planet.",
            source="seed", base_max_id=3,
        )
        overlay.remove_cell(2, reason="superseded by cell_mars")
        log = overlay.provenance()
    finally:
        overlay.close()

    assert len(log) == 2
    assert log[0]["op"] == "add"
    assert log[0]["cell_id"] == new_id
    assert log[0]["text"] == "Mars is a planet."
    assert log[0]["source"] == "seed"
    assert log[1]["op"] == "remove"
    assert log[1]["cell_id"] == 2
    assert log[1]["reason"] == "superseded by cell_mars"


def test_overlay_remove_requires_reason(tmp_path: Path):
    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        with pytest.raises(ValueError):
            overlay.remove_cell(2, reason="")
    finally:
        overlay.close()


def test_add_cell_from_text_uses_encoder_and_whiten(tmp_path: Path, tiny_bank: Path):
    bank = StreamingBank(str(tiny_bank))
    try:
        encoder = _StubEncoder(dim=bank.dim)
        encoder.responses["Mars is a red planet."] = (
            np.eye(bank.dim, dtype=np.float32)[3]
        )
        overlay = OverlayStore(tmp_path / "ov.db")
        try:
            new_id, weight = add_cell_from_text(
                overlay,
                bank_dim=bank.dim,
                base_max_id=int(bank.cell_ids.max()),
                encoder=encoder,
                whiten_fn=bank.whiten,
                text="Mars is a red planet.",
                source="test",
                label="cell_mars",
            )
        finally:
            overlay.close()
    finally:
        bank.close()

    assert new_id > int(bank.cell_ids.max())
    # Identity whitening + already-unit input → weight ≈ axis-3 unit vector.
    expected = np.eye(bank.dim, dtype=np.float32)[3]
    assert np.allclose(weight, expected, atol=1e-6)


# ---------- StreamingBank-with-overlay integration ----------

def test_streaming_bank_merges_overlay_arrays(tmp_path: Path, tiny_bank: Path):
    base = StreamingBank(str(tiny_bank))
    try:
        base_max = int(base.cell_ids.max())
    finally:
        base.close()

    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        w = np.eye(8, dtype=np.float32)[3]
        new_id = overlay.add_cell(
            weight=w, source_text="Mars has two moons.",
            label="cell_mars", base_max_id=base_max,
        )
    finally:
        overlay.close()

    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        bank = StreamingBank(str(tiny_bank), overlay=overlay)
        try:
            assert bank.n_cells == 4
            assert new_id in set(int(c) for c in bank.cell_ids)
            # Querying along axis-3 retrieves the overlay cell.
            q = np.eye(8, dtype=np.float32)[3]
            top = bank.topk(q, k=2)
            assert int(top["cell_ids"][0]) == new_id
            assert top["activations"][0] > 0.99
            # Source text reads from the overlay.
            txt = bank.fetch_source_text(new_id)
            assert txt == "Mars has two moons."
            labels = bank.fetch_labels([new_id])
            assert labels == ["cell_mars"]
        finally:
            bank.close()
    finally:
        overlay.close()


def test_streaming_bank_applies_tombstones(tmp_path: Path, tiny_bank: Path):
    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        # Tombstone the zebra cell (base id 1).
        overlay.remove_cell(1, reason="bad data")
    finally:
        overlay.close()

    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        bank = StreamingBank(str(tiny_bank), overlay=overlay)
        try:
            q = np.eye(8, dtype=np.float32)[0]  # zebra axis
            top = bank.topk(q, k=3)
            # Zebra (id=1) is suppressed; topk returns the next-best matches.
            assert 1 not in set(int(c) for c in top["cell_ids"])
            # Top activation against axis-0 should now be ~0 (other cells
            # are on different axes).
            assert top["activations"][0] < 0.01
        finally:
            bank.close()
    finally:
        overlay.close()


# ---------- end-to-end pipeline contract (PLAN.md Step 4) ----------

def test_add_cell_changes_pipeline_answer(tmp_path: Path, tiny_bank: Path):
    """Add overlay cell → previously-silent question now grounds."""
    encoder = _StubEncoder(dim=8)
    # axis-3 is initially empty → without overlay this question is silent.
    encoder.responses["What is Mars?"] = np.eye(8, dtype=np.float32)[3]
    encoder.responses["Mars is a red planet with two moons."] = (
        np.eye(8, dtype=np.float32)[3]
    )

    # Sanity: with no overlay, the question is silenced.
    pipeline_no_overlay = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=lambda p: "should not run",
    )
    try:
        result_silent = pipeline_no_overlay.ask("What is Mars?")
    finally:
        pipeline_no_overlay.close()
    assert result_silent.silence is True
    assert result_silent.answer == SILENCE_NO_MATCH

    # Now add an overlay cell on axis-3.
    base = StreamingBank(str(tiny_bank))
    try:
        base_max = int(base.cell_ids.max())
    finally:
        base.close()
    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        add_cell_from_text(
            overlay,
            bank_dim=8,
            base_max_id=base_max,
            encoder=encoder,
            whiten_fn=lambda v: v,  # identity whitening for test bank
            text="Mars is a red planet with two moons.",
            source="test",
            label="cell_mars",
        )
    finally:
        overlay.close()

    # Re-open with overlay → previously-silent question now grounds.
    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        bank = StreamingBank(str(tiny_bank), overlay=overlay)
        pipeline_with = AnswerPipeline(
            bank_path=tiny_bank,
            top_k=3,
            encoder=encoder,
            bank=bank,
            generator=lambda p: "Mars is a red planet with two moons. [4]",
        )
        try:
            result_grounded = pipeline_with.ask("What is Mars?")
        finally:
            pipeline_with.close()  # closes bank
    finally:
        overlay.close()

    assert result_grounded.silence is False, result_grounded.silence_reason
    assert result_grounded.gate["fire"] is True
    assert result_grounded.verification is not None
    assert result_grounded.verification["grounded"] is True
    assert "mars" in result_grounded.answer.lower()


def test_remove_cell_silences_pipeline(tmp_path: Path, tiny_bank: Path):
    """Remove a base cell → its question silences."""
    encoder = _StubEncoder(dim=8)
    encoder.responses["What do zebras look like?"] = (
        np.eye(8, dtype=np.float32)[0]
    )

    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        # Tombstone the zebra cell (base id = 1, first inserted).
        overlay.remove_cell(1, reason="test removal")
    finally:
        overlay.close()

    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        bank = StreamingBank(str(tiny_bank), overlay=overlay)
        pipeline = AnswerPipeline(
            bank_path=tiny_bank,
            top_k=3,
            encoder=encoder,
            bank=bank,
            generator=lambda p: "should not run if silenced",
        )
        try:
            result = pipeline.ask("What do zebras look like?")
        finally:
            pipeline.close()
    finally:
        overlay.close()

    assert result.silence is True
    assert result.answer == SILENCE_NO_MATCH


def test_add_then_remove_overlay_cell_silences(tmp_path: Path, tiny_bank: Path):
    """Add overlay cell, then tombstone it → question silences again."""
    encoder = _StubEncoder(dim=8)
    encoder.responses["What is Mars?"] = np.eye(8, dtype=np.float32)[3]
    encoder.responses["Mars is a red planet."] = np.eye(8, dtype=np.float32)[3]

    base = StreamingBank(str(tiny_bank))
    try:
        base_max = int(base.cell_ids.max())
    finally:
        base.close()

    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        new_id, _ = add_cell_from_text(
            overlay,
            bank_dim=8,
            base_max_id=base_max,
            encoder=encoder,
            whiten_fn=lambda v: v,
            text="Mars is a red planet.",
            source="test",
        )
        overlay.remove_cell(new_id, reason="test tombstone of overlay cell")
    finally:
        overlay.close()

    overlay = OverlayStore(tmp_path / "ov.db")
    try:
        bank = StreamingBank(str(tiny_bank), overlay=overlay)
        pipeline = AnswerPipeline(
            bank_path=tiny_bank,
            top_k=3,
            encoder=encoder,
            bank=bank,
            generator=lambda p: "should not run",
        )
        try:
            result = pipeline.ask("What is Mars?")
        finally:
            pipeline.close()
    finally:
        overlay.close()

    assert result.silence is True
    assert result.answer == SILENCE_NO_MATCH
