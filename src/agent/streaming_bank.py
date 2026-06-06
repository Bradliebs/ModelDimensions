"""Streaming loader for the production concept-cell bank.

The existing `SqliteBank` in this module's sibling reads the whole `cells`
table with `cursor.fetchall()` and then `np.stack()`s the result. That
pattern peaks at ~25 GB of RAM on the 5.7M-cell production bank because
SQLite materializes every row tuple before NumPy gets a chance to compact
them. On a 48 GB machine that's just enough to trip OOM under any other
load.

`StreamingBank` pre-allocates the target arrays and fills them row-by-row
from a cursor, which holds peak RAM at ~8.75 GB (the size of the bank
arrays themselves). It is read-only and reuses the schema established by
`src/agent/sqlite_bank.py` and `src/cc_service/cc_service.py`:

    meta(key, value)
    whitening(id, mu BLOB, w_matrix BLOB, max_norm REAL, ...)
    cells(id INTEGER, label, weight BLOB, theta REAL, kind, created_at)
    source_texts(cell_id INTEGER, text TEXT)

It deliberately does NOT preload `source_texts` (would add several GB);
texts are fetched on demand for top-k hits via :meth:`fetch_source_text`.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np


def _decode_meta_value(raw: str) -> Any:
    """Meta values are JSON-encoded by ``cc_service.persistence``; decode
    if possible, otherwise fall back to the raw string."""
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


class StreamingBank:
    """Read-only loader that streams the production bank in ~constant memory.

    Use this for V1 workloads where the full ~5.7M-cell production bank
    has to live in RAM but the SqliteBank loader would OOM the machine.
    For test fixtures (a few thousand cells), `SqliteBank` is fine.
    """

    def __init__(self, db_path: str | Path,
                 report_every: int = 500_000) -> None:
        self.db_path = str(db_path)
        uri = f"file:{self.db_path}?mode=ro"
        # ``check_same_thread=False``: the pipeline may be constructed on one
        # thread and used on another (e.g. ``HTTPServer`` runs on its own
        # thread). The bank is read-only and callers serialize access (the
        # stdlib HTTPServer is single-threaded), so cross-thread reads are
        # safe.
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        cur = self._conn.cursor()

        meta = {
            k: _decode_meta_value(v)
            for k, v in cur.execute("SELECT key, value FROM meta")
        }
        self.dim = int(meta["dim"])
        self.encoder_model = str(meta.get("encoder_model", ""))
        self.theta_write_default = float(meta.get("theta_write_default", 0.3))

        wrow = cur.execute(
            "SELECT mu, w_matrix, max_norm FROM whitening WHERE id = 1"
        ).fetchone()
        if wrow is not None:
            mu_blob, w_blob, max_norm = wrow
            self.mu = np.frombuffer(mu_blob, dtype=np.float32).copy()
            self.w_matrix = np.frombuffer(w_blob, dtype=np.float32).reshape(
                self.dim, self.dim
            ).copy()
            self.max_norm = float(max_norm)
            self.has_whitening = True
        else:
            self.mu = None
            self.w_matrix = None
            self.max_norm = 1.0
            self.has_whitening = False

        self.n_cells = int(
            cur.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
        )

        self.weights = np.empty((self.n_cells, self.dim), dtype=np.float32)
        self.thetas = np.empty(self.n_cells, dtype=np.float32)
        self.cell_ids = np.empty(self.n_cells, dtype=np.int64)

        t0 = time.time()
        cur2 = self._conn.cursor()
        cur2.execute("SELECT id, weight, theta FROM cells ORDER BY id ASC")
        for i, (cell_id, weight_blob, theta) in enumerate(cur2):
            self.weights[i] = np.frombuffer(weight_blob, dtype=np.float32)
            self.thetas[i] = float(theta)
            self.cell_ids[i] = int(cell_id)
            if report_every and (i + 1) % report_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (self.n_cells - i - 1) / rate if rate > 0 else 0.0
                print(
                    f"  [streaming_bank] {i + 1:,}/{self.n_cells:,} "
                    f"({rate:,.0f}/s, eta {eta:.0f}s)",
                    flush=True,
                )
        self.load_seconds = time.time() - t0

    # ---- query path ----

    def whiten(self, raw_vector: np.ndarray) -> np.ndarray:
        """Apply the bank's stored whitening transform to an encoder output."""
        if not self.has_whitening:
            return raw_vector.astype(np.float32, copy=False)
        x = raw_vector.astype(np.float32, copy=False)
        centered = x - self.mu
        whitened = centered @ self.w_matrix
        scaled = whitened / (self.max_norm + 1e-8)
        return scaled.astype(np.float32)

    def topk(self, vector: np.ndarray, k: int = 10) -> dict:
        """Return the top-k cells by raw activation (vector . weight).

        Returned dict has ``cell_ids``, ``activations``, ``thetas`` as
        parallel arrays of length k, ordered by descending activation.
        Activations are the raw dot product against the whitened query;
        the silence gate works on these values directly.
        """
        if vector.shape != (self.dim,):
            raise ValueError(
                f"vector shape {vector.shape} != ({self.dim},)"
            )
        q = vector.astype(np.float32, copy=False)
        activations = self.weights @ q
        k = min(k, len(activations))
        idx = np.argpartition(-activations, k - 1)[:k]
        order = idx[np.argsort(-activations[idx])]
        return {
            "cell_ids": self.cell_ids[order].copy(),
            "activations": activations[order].copy(),
            "thetas": self.thetas[order].copy(),
        }

    # ---- source-text fetch (lazy) ----

    def fetch_source_text(self, cell_id: int) -> Optional[str]:
        row = self._conn.execute(
            "SELECT text FROM source_texts WHERE cell_id = ?", (int(cell_id),)
        ).fetchone()
        return row[0] if row else None

    def fetch_source_texts(self, cell_ids: Iterable[int]) -> List[Optional[str]]:
        """Bulk-fetch source texts for a small list of cell ids, preserving order."""
        ids = [int(c) for c in cell_ids]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = {
            cid: text
            for (cid, text) in self._conn.execute(
                f"SELECT cell_id, text FROM source_texts "
                f"WHERE cell_id IN ({placeholders})",
                ids,
            )
        }
        return [rows.get(cid) for cid in ids]

    def fetch_labels(self, cell_ids: Iterable[int]) -> List[Optional[str]]:
        """Bulk-fetch ``cells.label`` for a small list of ids, preserving order.

        Many cells in the production bank have NULL labels; callers should
        fall back to a truncated source-text snippet for display.
        """
        ids = [int(c) for c in cell_ids]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = {
            cid: label
            for (cid, label) in self._conn.execute(
                f"SELECT id, label FROM cells WHERE id IN ({placeholders})",
                ids,
            )
        }
        return [rows.get(cid) for cid in ids]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "StreamingBank":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
