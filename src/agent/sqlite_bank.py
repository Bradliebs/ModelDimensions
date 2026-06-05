"""Read-only adapter for a SQLite-backed concept-cell bank.

This module bridges the SLM controller layer in this repo to a concept-cell
bank that was written by a *separate* service (the MiniLM ``cc_service``
running at ``H:\\MiniLM\\cc_service\\bank.db``). That service vendored a
small copy of the geometry/binding code and persists cells as ``(weight BLOB,
theta REAL)`` rows in SQLite. Rather than couple the two repos, we re-read
the same schema here in a strictly read-only fashion.

Schema expected (subset of the MiniLM ``BankStore`` schema):

  meta(key TEXT, value TEXT JSON)             # 'dim' is required
  cells(id INT, label TEXT, weight BLOB,
        theta REAL, kind TEXT, created_at REAL)
  source_texts(cell_id INT, text TEXT)        # LEFT JOIN; may be absent
  whitening(id=1, mu BLOB, w_matrix BLOB,
            max_norm REAL, ...)               # optional row

The adapter loads ``(W, theta, cell_ids, source_texts)`` into memory on open
and answers ``query(vector)`` with a single matmul. It performs *no* writes,
no binding, and does not own an encoder; callers supply preprocessed query
vectors. The bank's stored whitening parameters are exposed so the caller can
apply the exact preprocessing the cells were written under.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WhiteningParams:
    """Frozen ZCA whitening + ball-scaling parameters read from the bank."""

    mu: np.ndarray          # (D,) float32
    w_matrix: np.ndarray    # (D, D) float32
    max_norm: float

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """Whiten and ball-scale a raw embedding (or batch thereof)."""
        single = raw.ndim == 1
        x = raw[None, :] if single else raw
        centered = x - self.mu[None, :]
        whitened = centered @ self.w_matrix
        scaled = whitened / (self.max_norm + 1e-8)
        out = scaled.astype(np.float32)
        return out[0] if single else out


@dataclass(frozen=True)
class QueryHit:
    """One result row from a bank query."""

    cell_id: int
    label: Optional[str]
    source_text: Optional[str]
    activation: float
    theta: float
    margin: float           # activation - theta; > 0 means the cell fired


class ReadOnlyBankError(RuntimeError):
    """Raised on any mutating call against a read-only bank."""


class SqliteBank:
    """Read-only view of a SQLite concept-cell bank.

    Opens the database via a ``file:...?mode=ro`` URI so the live writer (if
    any) is not affected; cell data is snapshotted into memory at construction
    time. Designed to be cheap for small banks (tests) and tolerable for the
    production 1.82M-cell bank (~2.8 GB for weights at D=384, plus source
    texts).
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"bank not found at {self.db_path}")
        # Read-only URI keeps us from accidentally mutating a live bank.
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.dim = self._load_dim()
        self.whitening = self._load_whitening()
        (
            self._weights,
            self._thetas,
            self._cell_ids,
            self._source_texts,
            self._labels,
        ) = self._load_cells()

    # -- introspection --

    def __len__(self) -> int:
        return len(self._cell_ids)

    @property
    def cell_ids(self) -> List[int]:
        return list(self._cell_ids)

    def close(self) -> None:
        self._conn.close()

    # -- read --

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        include_silent: bool = False,
    ) -> List[QueryHit]:
        """Score one query vector against every cell; return ranked hits.

        ``vector`` must already be preprocessed in the same space the cells
        were written in (typically: encode -> :py:meth:`WhiteningParams.apply`).
        With ``include_silent=False`` (default), only hits whose margin is
        strictly positive are returned, preserving the bank's rejection
        property.
        """
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(
                f"query vector must be shape ({self.dim},), got {vector.shape}"
            )
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if len(self._cell_ids) == 0:
            return []
        q = vector.astype(np.float32, copy=False)
        activations = self._weights @ q                      # (N,)
        margins = activations - self._thetas                 # (N,)
        if include_silent:
            order = np.argsort(-margins)[:top_k]
        else:
            fired_idx = np.flatnonzero(margins > 0.0)
            if fired_idx.size == 0:
                return []
            order = fired_idx[np.argsort(-margins[fired_idx])][:top_k]
        return [
            QueryHit(
                cell_id=int(self._cell_ids[i]),
                label=self._labels[i],
                source_text=self._source_texts[i],
                activation=float(activations[i]),
                theta=float(self._thetas[i]),
                margin=float(margins[i]),
            )
            for i in order
        ]

    # -- mutation (refused) --

    def write(self, *args, **kwargs):
        raise ReadOnlyBankError("SqliteBank is read-only; use the live cc_service to write")

    def delete(self, *args, **kwargs):
        raise ReadOnlyBankError("SqliteBank is read-only; use the live cc_service to delete")

    def bind(self, *args, **kwargs):
        raise ReadOnlyBankError("SqliteBank is read-only; use the live cc_service to bind")

    # -- private --

    def _load_dim(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", ("dim",)
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"bank at {self.db_path} has no 'dim' in meta table"
            )
        # meta.value is JSON-encoded per the cc_service convention.
        return int(json.loads(row[0]))

    def _load_whitening(self) -> Optional[WhiteningParams]:
        row = self._conn.execute(
            "SELECT mu, w_matrix, max_norm FROM whitening WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        mu_bytes, w_bytes, max_norm = row
        mu = np.frombuffer(mu_bytes, dtype=np.float32).reshape((self.dim,)).copy()
        w_matrix = np.frombuffer(w_bytes, dtype=np.float32).reshape((self.dim, self.dim)).copy()
        return WhiteningParams(
            mu=mu, w_matrix=w_matrix, max_norm=float(max_norm),
        )

    def _load_cells(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[Optional[str]], List[Optional[str]]]:
        rows = self._conn.execute(
            """SELECT c.id, c.label, c.weight, c.theta, s.text
                 FROM cells c
                 LEFT JOIN source_texts s ON s.cell_id = c.id
                 ORDER BY c.id ASC"""
        ).fetchall()
        if not rows:
            return (
                np.zeros((0, self.dim), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
                [],
                [],
            )
        weights = np.stack([
            np.frombuffer(r[2], dtype=np.float32).reshape((self.dim,)).copy()
            for r in rows
        ])
        thetas = np.array([r[3] for r in rows], dtype=np.float32)
        cell_ids = [int(r[0]) for r in rows]
        labels = [r[1] for r in rows]
        source_texts = [r[4] for r in rows]
        return weights, thetas, cell_ids, source_texts, labels
