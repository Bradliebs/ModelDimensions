"""Offline tests for the hybrid BM25 + dense retriever.

Builds a tiny SQLite bank matching the production schema (reused from
``test_answer_pipeline.py``-style helpers) and stacks a LexicalIndex on
top, then validates the cascade contract:

  - Hybrid path returns the BM25-restricted cosine top-K when the
    lexical signal is non-empty.
  - Falls back to dense ``topk`` when BM25 yields no hits.
  - Tombstones are honoured at both stages.
  - The returned shape is a superset of ``StreamingBank.topk``.
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

from src.agent.hybrid_retriever import HybridRetriever
from src.agent.lexical_index import LexicalIndex
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
    text    TEXT NOT NULL
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


@pytest.fixture
def cascade_bank(tmp_path: Path):
    """5-cell bank where Q2-style proper-noun retrieval can be exercised.

    Cell weights are *deliberately* engineered so that under a query that
    points at cell 1 (the noise direction), the right cell (cell 0,
    "Don Ellis") falls outside the cosine top-1 but is the unique BM25
    winner for the keyword query."""
    dim = 4
    eye = np.eye(dim, dtype=np.float32)
    # All cells share a common axis (eye[1]) so cosine prefers whichever
    # also leans on the query's secondary axis; the keyword cell's
    # secondary axis is eye[0] but the query points along eye[2].
    cells = [
        ("don_ellis", 0.1 * eye[0] + eye[1], 0.3,
         "Don Ellis composed the score for The French Connection in 1971."),
        ("godfather", 0.9 * eye[2] + eye[1], 0.3,
         "The Godfather is a 1972 crime film directed by Coppola."),
        ("zebra", 0.2 * eye[3] + eye[1], 0.3,
         "Zebras have black and white stripes and live in Africa."),
        ("paris", 0.4 * eye[0] + 0.5 * eye[1], 0.3,
         "Paris is the capital of France on the river Seine."),
        ("chess", 0.3 * eye[2] + 0.6 * eye[1], 0.3,
         "Chess is a two-player strategy game played on 64 squares."),
    ]
    db = tmp_path / "bank.db"
    _make_bank(db, dim, cells)
    return db, dim


def test_hybrid_pulls_keyword_cell_dense_misses(cascade_bank):
    """The pivot test: a query whose dense encoding leans the wrong way
    but whose terms uniquely match the keyword cell. BM25 must prefilter
    so the cosine re-rank still surfaces it."""
    db, dim = cascade_bank
    bank = StreamingBank(db)
    try:
        idx = LexicalIndex()
        idx.build_from_texts(
            [int(c) for c in bank.cell_ids],
            bank.fetch_source_texts([int(c) for c in bank.cell_ids]),
        )
        retriever = HybridRetriever(idx, bank)

        # Encoded query leans along eye[2] (the Godfather direction).
        # Dense alone would prefer Godfather + Chess; the keyword query
        # ("Don Ellis French Connection") restricts BM25 to the keyword
        # cell only.
        q = np.array([0.0, 0.4, 1.0, 0.0], dtype=np.float32)
        out = retriever.topk(
            "Don Ellis French Connection", q,
            k_lexical=10, k_final=3,
        )

        assert out["stage"] == "hybrid"
        assert out["cell_ids"][0] == bank.cell_ids[0]  # don_ellis
        assert out["lexical_ranks"][0] == 1
        assert out["lexical_scores"][0] > 0.0
    finally:
        bank.close()


def test_dense_fallback_on_empty_bm25(cascade_bank):
    """All-stopword / no-overlap query must fall back to bank.topk."""
    db, dim = cascade_bank
    bank = StreamingBank(db)
    try:
        idx = LexicalIndex()
        idx.build_from_texts(
            [int(c) for c in bank.cell_ids],
            bank.fetch_source_texts([int(c) for c in bank.cell_ids]),
        )
        retriever = HybridRetriever(idx, bank)

        q = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
        # Stopword-only query: BM25 returns nothing.
        out = retriever.topk("the of an a", q, k_lexical=10, k_final=3)
        assert out["stage"] == "dense_fallback"
        # Lexical diagnostics are still present and zero-filled.
        assert list(out["lexical_scores"]) == [0.0, 0.0, 0.0]
        assert out["lexical_ranks"] == [None, None, None]
        # Order matches what bank.topk would return for this query.
        dense = bank.topk(q, k=3)
        np.testing.assert_array_equal(out["cell_ids"], dense["cell_ids"])
    finally:
        bank.close()


def test_shape_matches_streaming_bank_topk(cascade_bank):
    """Pipeline code consumes ``cell_ids``, ``activations``, ``thetas``
    as parallel arrays. The hybrid result must respect that contract."""
    db, dim = cascade_bank
    bank = StreamingBank(db)
    try:
        idx = LexicalIndex()
        idx.build_from_texts(
            [int(c) for c in bank.cell_ids],
            bank.fetch_source_texts([int(c) for c in bank.cell_ids]),
        )
        retriever = HybridRetriever(idx, bank)

        q = np.array([0.0, 0.4, 1.0, 0.0], dtype=np.float32)
        out = retriever.topk("Don Ellis", q, k_lexical=10, k_final=2)

        assert out["cell_ids"].dtype == np.int64
        assert out["activations"].dtype == np.float32
        assert out["thetas"].dtype == np.float32
        assert (out["cell_ids"].shape
                == out["activations"].shape
                == out["thetas"].shape)
        assert len(out["lexical_scores"]) == out["cell_ids"].shape[0]
        assert len(out["lexical_ranks"]) == out["cell_ids"].shape[0]
        # Cosine activations match what weights_for + dot would produce.
        manual = bank.weights_for([int(c) for c in out["cell_ids"]]) @ q
        np.testing.assert_allclose(out["activations"], manual, rtol=1e-5)
    finally:
        bank.close()


def test_weights_for_raises_on_unknown_id(cascade_bank):
    """The hybrid retriever depends on weights_for failing loudly when
    an id is unknown — silent skipping would feed spurious zero
    activations into the gate."""
    db, _ = cascade_bank
    bank = StreamingBank(db)
    try:
        with pytest.raises(KeyError):
            bank.weights_for([9_999_999])
    finally:
        bank.close()


def test_weights_for_preserves_order_and_dtype(cascade_bank):
    db, dim = cascade_bank
    bank = StreamingBank(db)
    try:
        ids = list(reversed([int(c) for c in bank.cell_ids[:3]]))
        w = bank.weights_for(ids)
        assert w.shape == (3, dim)
        assert w.dtype == np.float32
        for i, cid in enumerate(ids):
            row = bank._id_to_row[cid]
            np.testing.assert_array_equal(w[i], bank.weights[row])
    finally:
        bank.close()


def test_hybrid_honours_tombstones(tmp_path: Path, cascade_bank):
    """A tombstoned cell must not appear in the hybrid result even if it
    is the top BM25 hit."""
    from src.agent.bank_admin import OverlayStore

    db, dim = cascade_bank
    overlay_path = tmp_path / "overlay.db"
    overlay = OverlayStore(overlay_path)

    bank = StreamingBank(db, overlay=overlay)
    try:
        # Build index BEFORE tombstone — that's the production sequence:
        # index is a snapshot; later tombstones are applied at query
        # time via excluded_ids.
        idx = LexicalIndex()
        all_ids = [int(c) for c in bank.cell_ids]
        idx.build_from_texts(all_ids, bank.fetch_source_texts(all_ids))

        # Tombstone the don_ellis cell.
        overlay.remove_cell(int(bank.cell_ids[0]), reason="test")

        # Re-open the bank to pick up the tombstone (tombstones are
        # applied at StreamingBank construction time, not live).
        bank.close()
        bank = StreamingBank(db, overlay=OverlayStore(overlay_path))
        retriever = HybridRetriever(idx, bank)

        q = np.array([0.0, 0.4, 1.0, 0.0], dtype=np.float32)
        out = retriever.topk(
            "Don Ellis French Connection", q,
            k_lexical=10, k_final=3,
        )
        # Tombstoned id never surfaces, even though it's the BM25 winner
        # in the raw index.
        assert int(bank.cell_ids[0]) not in [int(c) for c in out["cell_ids"]] \
            or len(out["cell_ids"]) == 0
    finally:
        bank.close()
