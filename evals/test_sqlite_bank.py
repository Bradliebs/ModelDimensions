"""Offline tests for the SqliteBank read-only adapter.

These build a tiny SQLite file matching the live MiniLM ``cc_service`` schema
(meta, cells, source_texts, whitening), insert a handful of synthetic cells,
and exercise the adapter end-to-end. No network, no encoder, no dependency on
the live 1.82M-cell production bank.

Run with:  python -m pytest evals/test_sqlite_bank.py -q
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
sys.path.insert(0, str(ROOT / "src"))

from agent.sqlite_bank import (
    QueryHit,
    ReadOnlyBankError,
    SqliteBank,
    WhiteningParams,
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


def _make_bank(
    path: Path,
    dim: int,
    cells: list[tuple[str, np.ndarray, float, str | None]],
    *,
    with_whitening: bool = True,
) -> None:
    """Build a SQLite file matching the cc_service schema and populate it."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("dim", json.dumps(dim)),
    )
    if with_whitening:
        mu = np.zeros(dim, dtype=np.float32)
        w_matrix = np.eye(dim, dtype=np.float32)
        conn.execute(
            """INSERT INTO whitening
               (id, mu, w_matrix, max_norm, fitted_at, reference_n)
               VALUES (1, ?, ?, ?, ?, ?)""",
            (mu.tobytes(), w_matrix.tobytes(), 2.0, time.time(), 1000),
        )
    for label, weight, theta, source_text in cells:
        cur = conn.execute(
            """INSERT INTO cells (label, weight, theta, kind, created_at)
               VALUES (?, ?, ?, 'single', ?)""",
            (label, weight.astype(np.float32).tobytes(), float(theta), time.time()),
        )
        cell_id = cur.lastrowid
        if source_text is not None:
            conn.execute(
                "INSERT INTO source_texts (cell_id, text) VALUES (?, ?)",
                (cell_id, source_text),
            )
    conn.commit()
    conn.close()


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---------- open / introspect ----------

def test_open_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SqliteBank(tmp_path / "does_not_exist.db")


def test_open_loads_dim_and_whitening(tmp_path: Path):
    db = tmp_path / "bank.db"
    _make_bank(db, dim=8, cells=[])
    bank = SqliteBank(db)
    try:
        assert bank.dim == 8
        assert len(bank) == 0
        assert isinstance(bank.whitening, WhiteningParams)
        assert bank.whitening.mu.shape == (8,)
        assert bank.whitening.w_matrix.shape == (8, 8)
        assert bank.whitening.max_norm == pytest.approx(2.0)
    finally:
        bank.close()


def test_open_without_whitening(tmp_path: Path):
    db = tmp_path / "bank.db"
    _make_bank(db, dim=4, cells=[], with_whitening=False)
    bank = SqliteBank(db)
    try:
        assert bank.whitening is None
        assert len(bank) == 0
    finally:
        bank.close()


def test_open_requires_dim_meta(tmp_path: Path):
    db = tmp_path / "bank.db"
    # Build the schema without inserting a dim row.
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="'dim'"):
        SqliteBank(db)


# ---------- query ----------

def test_query_returns_only_fired_cells_by_default(tmp_path: Path):
    db = tmp_path / "bank.db"
    e1, e2, e3 = (np.eye(4, dtype=np.float32)[i] for i in range(3))
    _make_bank(
        db,
        dim=4,
        cells=[
            ("cell-a", e1, 0.5, "first cell text"),
            ("cell-b", e2, 0.5, "second cell text"),
            ("cell-c", e3, 0.5, "third cell text"),
        ],
    )
    bank = SqliteBank(db)
    try:
        # Query aligned with e1 should fire only cell-a.
        hits = bank.query(e1, top_k=10)
        assert [h.label for h in hits] == ["cell-a"]
        assert hits[0].activation == pytest.approx(1.0)
        assert hits[0].theta == pytest.approx(0.5)
        assert hits[0].margin == pytest.approx(0.5)
        assert hits[0].source_text == "first cell text"
    finally:
        bank.close()


def test_query_empty_bank_returns_empty(tmp_path: Path):
    db = tmp_path / "bank.db"
    _make_bank(db, dim=4, cells=[])
    bank = SqliteBank(db)
    try:
        assert bank.query(np.zeros(4, dtype=np.float32)) == []
    finally:
        bank.close()


def test_query_silent_when_no_cell_fires(tmp_path: Path):
    db = tmp_path / "bank.db"
    e1 = np.eye(4, dtype=np.float32)[0]
    _make_bank(db, dim=4, cells=[("only", e1, 0.5, "text")])
    bank = SqliteBank(db)
    try:
        # Orthogonal query produces zero activation, below theta -> silent.
        orth = np.eye(4, dtype=np.float32)[1]
        assert bank.query(orth) == []
    finally:
        bank.close()


def test_query_include_silent_returns_below_threshold(tmp_path: Path):
    db = tmp_path / "bank.db"
    e1 = np.eye(4, dtype=np.float32)[0]
    _make_bank(db, dim=4, cells=[("only", e1, 0.5, "text")])
    bank = SqliteBank(db)
    try:
        orth = np.eye(4, dtype=np.float32)[1]
        hits = bank.query(orth, top_k=5, include_silent=True)
        assert len(hits) == 1
        assert hits[0].margin < 0.0
    finally:
        bank.close()


def test_query_top_k_orders_by_margin(tmp_path: Path):
    db = tmp_path / "bank.db"
    # Three cells whose weights all overlap with the query, with descending alignment.
    w_strong = _unit(np.array([1.0, 0.1, 0.0, 0.0]))
    w_medium = _unit(np.array([1.0, 0.5, 0.0, 0.0]))
    w_weak = _unit(np.array([1.0, 1.0, 0.0, 0.0]))
    _make_bank(
        db,
        dim=4,
        cells=[
            ("strong", w_strong, 0.1, "strong"),
            ("medium", w_medium, 0.1, "medium"),
            ("weak", w_weak, 0.1, "weak"),
        ],
    )
    bank = SqliteBank(db)
    try:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        hits = bank.query(q, top_k=2)
        assert [h.label for h in hits] == ["strong", "medium"]
        assert hits[0].margin > hits[1].margin
    finally:
        bank.close()


def test_query_rejects_wrong_shape(tmp_path: Path):
    db = tmp_path / "bank.db"
    _make_bank(db, dim=4, cells=[])
    bank = SqliteBank(db)
    try:
        with pytest.raises(ValueError, match=r"\(4,\)"):
            bank.query(np.zeros(5, dtype=np.float32))
        with pytest.raises(ValueError, match=r"\(4,\)"):
            bank.query(np.zeros((2, 4), dtype=np.float32))
    finally:
        bank.close()


def test_query_rejects_bad_top_k(tmp_path: Path):
    db = tmp_path / "bank.db"
    _make_bank(db, dim=4, cells=[])
    bank = SqliteBank(db)
    try:
        with pytest.raises(ValueError, match="top_k"):
            bank.query(np.zeros(4, dtype=np.float32), top_k=0)
    finally:
        bank.close()


# ---------- whitening application ----------

def test_whitening_round_trip(tmp_path: Path):
    db = tmp_path / "bank.db"
    # Use identity whitening with max_norm=2 so apply() just halves the input.
    _make_bank(db, dim=4, cells=[])
    bank = SqliteBank(db)
    try:
        raw = np.array([2.0, 4.0, 0.0, -2.0], dtype=np.float32)
        whitened = bank.whitening.apply(raw)
        assert whitened.shape == (4,)
        assert whitened == pytest.approx(np.array([1.0, 2.0, 0.0, -1.0], dtype=np.float32))
        # Batch form preserves shape too.
        batch = np.stack([raw, raw * 0])
        out = bank.whitening.apply(batch)
        assert out.shape == (2, 4)
    finally:
        bank.close()


# ---------- read-only enforcement ----------

@pytest.mark.parametrize("method", ["write", "delete", "bind"])
def test_mutating_methods_refuse(tmp_path: Path, method: str):
    db = tmp_path / "bank.db"
    _make_bank(db, dim=4, cells=[])
    bank = SqliteBank(db)
    try:
        with pytest.raises(ReadOnlyBankError):
            getattr(bank, method)("anything")
    finally:
        bank.close()


# ---------- end-to-end: bank with whitened cells answers a whitened query ----------

def test_end_to_end_whitened_query(tmp_path: Path):
    """A cell written from a known-direction vector fires for that direction
    after both have passed through the bank's stored whitening."""
    db = tmp_path / "bank.db"
    raw_direction = np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # We build the cell weight as if we had: whiten(raw_direction) -> unit.
    # Whitening here is identity + max_norm=2, so whitened = raw / 2.
    whitened_anchor = raw_direction / 2.0
    cell_weight = _unit(whitened_anchor)
    _make_bank(db, dim=4, cells=[("anchor", cell_weight, 0.5, "anchor text")])

    bank = SqliteBank(db)
    try:
        # Caller's job: take raw query, apply bank's whitening, hand to query.
        raw_query = raw_direction.copy()
        q = bank.whitening.apply(raw_query)
        hits = bank.query(q, top_k=1)
        assert len(hits) == 1
        assert hits[0].label == "anchor"
        assert hits[0].source_text == "anchor text"
        assert hits[0].margin > 0.0
    finally:
        bank.close()
