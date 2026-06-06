"""Overlay editable bank — V1 admin API.

The production 5.7M-cell concept-cell bank at ``H:\\MiniLM\\cc_service\\bank.db``
is treated as immutable in V1. New cells and cell removals land in a
**separate** SQLite file, an *overlay store*, so a wrong ``add_cell`` cannot
corrupt the 13 GB production bank. The pipeline reads
``base ∪ overlay`` and applies any tombstones the overlay records.

Schema (created on first use under ``results/v1_bank/overlay.db`` by default):

  overlay_cells(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT,
    weight      BLOB NOT NULL,           -- whitened, ball-scaled, float32
    theta       REAL NOT NULL,
    source_text TEXT NOT NULL,
    source      TEXT,                    -- caller-supplied provenance hint
    created_at  REAL NOT NULL
  )

  tombstones(
    base_cell_id INTEGER PRIMARY KEY,    -- id of a cell to suppress
    reason       TEXT NOT NULL,
    removed_at   REAL NOT NULL
  )

  provenance_log(
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    op          TEXT NOT NULL,           -- 'add' | 'remove'
    cell_id     INTEGER NOT NULL,
    text        TEXT,
    source      TEXT,
    reason      TEXT,
    ts          REAL NOT NULL
  )

  meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)

Cell-id allocation: the overlay reserves an id range strictly above the base
bank's max id. ``allocate_id_base`` is set at first cell insert from the
caller-supplied ``base_max_id`` so overlay ids never collide with base ids.

This module is intentionally minimal. It does *not* implement encoder
swap-out, multi-user concurrency, or complex transaction recovery. The
overlay is single-writer: the admin CLI (or test harness) opens it, makes
one edit, closes it. Long-running services should be restarted to pick
up overlay changes.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS overlay_cells (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT,
    weight      BLOB NOT NULL,
    theta       REAL NOT NULL,
    source_text TEXT NOT NULL,
    source      TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tombstones (
    base_cell_id INTEGER PRIMARY KEY,
    reason       TEXT NOT NULL,
    removed_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS provenance_log (
    rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
    op      TEXT NOT NULL,
    cell_id INTEGER NOT NULL,
    text    TEXT,
    source  TEXT,
    reason  TEXT,
    ts      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class OverlayPayload:
    """Bulk-read snapshot of the overlay, suitable for merging into the
    in-RAM bank arrays at construction time.

    All arrays are parallel and ordered by overlay cell id ascending. Source
    texts and labels are returned as plain Python lists so a NULL label is
    represented by ``None``.
    """

    weights: np.ndarray            # (n_overlay, dim) float32
    cell_ids: np.ndarray           # (n_overlay,) int64
    thetas: np.ndarray             # (n_overlay,) float32
    source_texts: List[str]        # length n_overlay
    labels: List[Optional[str]]    # length n_overlay
    tombstoned_base_ids: set       # set[int]


class OverlayStoreError(RuntimeError):
    """Raised on schema / id-collision / tombstone-of-overlay-cell errors."""


class OverlayStore:
    """SQLite-backed overlay editable bank.

    Construct once per session. Every mutation is committed immediately;
    crash safety is left to SQLite. The store owns no encoder and no
    whitening parameters — callers supply already-whitened weight vectors.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "OverlayStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- meta ----

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    # ---- writes ----

    def add_cell(
        self,
        *,
        weight: np.ndarray,
        source_text: str,
        source: Optional[str] = None,
        label: Optional[str] = None,
        theta: float = 0.30,
        base_max_id: int,
    ) -> int:
        """Persist a new overlay cell. Returns its allocated ``cell_id``.

        Allocation: the first add fixes ``allocate_id_base`` to
        ``base_max_id + 1``. Subsequent overlay ids are assigned by SQLite
        AUTOINCREMENT against the same offset, so overlay ids never alias
        base ids.

        ``weight`` must already be in the bank's whitened, ball-scaled
        space (typically ``bank.whiten(encoder.encode_one(text)) /
        norm(...)`` — see ``add_cell_from_text`` for the high-level helper).
        """
        if weight.ndim != 1:
            raise ValueError(
                f"weight must be a 1-D vector, got shape {weight.shape}"
            )
        weight_f32 = weight.astype(np.float32, copy=False)

        # First insert: anchor the autoincrement above base_max_id.
        existing = self.get_meta("allocate_id_base")
        if existing is None:
            self.set_meta("allocate_id_base", str(int(base_max_id)))
            # Seed sqlite_sequence so AUTOINCREMENT starts at base_max_id+1.
            self._conn.execute(
                "INSERT OR REPLACE INTO sqlite_sequence(name, seq) "
                "VALUES ('overlay_cells', ?)",
                (int(base_max_id),),
            )
            self._conn.commit()
        else:
            anchored = int(existing)
            if int(base_max_id) > anchored:
                raise OverlayStoreError(
                    f"base_max_id={base_max_id} exceeds the overlay's "
                    f"anchored id base {anchored}; the overlay was created "
                    f"against a smaller base bank. Refusing to allocate."
                )

        ts = time.time()
        cur = self._conn.execute(
            """INSERT INTO overlay_cells
                 (label, weight, theta, source_text, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (label, weight_f32.tobytes(), float(theta),
             str(source_text), source, ts),
        )
        new_id = int(cur.lastrowid)
        self._conn.execute(
            """INSERT INTO provenance_log(op, cell_id, text, source, reason, ts)
               VALUES ('add', ?, ?, ?, NULL, ?)""",
            (new_id, str(source_text), source, ts),
        )
        self._conn.commit()
        return new_id

    def remove_cell(self, cell_id: int, reason: str) -> None:
        """Tombstone a cell so the merged bank suppresses it.

        Works against base cell ids and overlay cell ids alike. For
        overlay-owned cells the tombstone is the canonical removal record;
        we do not delete the row from ``overlay_cells`` so provenance is
        preserved. Idempotent on repeated calls (same reason wins).
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason is required and must be non-empty")
        ts = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO tombstones(base_cell_id, reason, removed_at)
               VALUES (?, ?, ?)""",
            (int(cell_id), reason, ts),
        )
        self._conn.execute(
            """INSERT INTO provenance_log(op, cell_id, text, source, reason, ts)
               VALUES ('remove', ?, NULL, NULL, ?, ?)""",
            (int(cell_id), reason, ts),
        )
        self._conn.commit()

    # ---- reads ----

    def load(self, dim: int) -> OverlayPayload:
        """Bulk-read the overlay into parallel arrays + tombstone set."""
        cur = self._conn.execute(
            "SELECT id, label, weight, theta, source_text "
            "FROM overlay_cells ORDER BY id ASC"
        )
        ids: list[int] = []
        labels: list[Optional[str]] = []
        thetas: list[float] = []
        source_texts: list[str] = []
        weight_blobs: list[bytes] = []
        for row in cur:
            cid, label, weight_blob, theta, source_text = row
            ids.append(int(cid))
            labels.append(label)
            thetas.append(float(theta))
            source_texts.append(str(source_text))
            weight_blobs.append(weight_blob)

        n = len(ids)
        if n == 0:
            weights = np.empty((0, dim), dtype=np.float32)
        else:
            weights = np.empty((n, dim), dtype=np.float32)
            for i, blob in enumerate(weight_blobs):
                w = np.frombuffer(blob, dtype=np.float32)
                if w.shape != (dim,):
                    raise OverlayStoreError(
                        f"overlay cell id={ids[i]} has weight shape "
                        f"{w.shape} but bank dim={dim}"
                    )
                weights[i] = w

        cur = self._conn.execute("SELECT base_cell_id FROM tombstones")
        tombstoned = {int(row[0]) for row in cur}

        return OverlayPayload(
            weights=weights,
            cell_ids=np.asarray(ids, dtype=np.int64),
            thetas=np.asarray(thetas, dtype=np.float32),
            source_texts=source_texts,
            labels=labels,
            tombstoned_base_ids=tombstoned,
        )

    def fetch_overlay_text(self, cell_id: int) -> Optional[str]:
        row = self._conn.execute(
            "SELECT source_text FROM overlay_cells WHERE id = ?",
            (int(cell_id),),
        ).fetchone()
        return row[0] if row else None

    def provenance(self) -> List[dict]:
        """Return the full provenance log, oldest first."""
        cur = self._conn.execute(
            "SELECT op, cell_id, text, source, reason, ts "
            "FROM provenance_log ORDER BY rowid ASC"
        )
        return [
            {"op": op, "cell_id": int(cid), "text": text,
             "source": source, "reason": reason, "ts": float(ts)}
            for (op, cid, text, source, reason, ts) in cur
        ]


def add_cell_from_text(
    overlay: OverlayStore,
    *,
    bank_dim: int,
    base_max_id: int,
    encoder,
    whiten_fn,
    text: str,
    source: Optional[str] = None,
    label: Optional[str] = None,
    theta: float = 0.30,
) -> Tuple[int, np.ndarray]:
    """Encode + whiten ``text`` and persist as an overlay cell.

    The cell weight is the L2-normalised whitened embedding, matching the
    write pattern in ``cc_service`` (each cell's ``w = x / ||x||``).
    Returns ``(new_cell_id, stored_weight)``; the weight is returned so
    callers can verify or smoke-test the write.

    ``encoder`` only needs an ``encode_one(text, is_query=False)`` method.
    ``whiten_fn`` is typically ``bank.whiten``.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    raw = encoder.encode_one(text, is_query=False)
    if raw.shape != (bank_dim,):
        raise OverlayStoreError(
            f"encoder produced shape {raw.shape}; expected ({bank_dim},)"
        )
    whitened = whiten_fn(raw).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(whitened))
    if norm < 1e-8:
        raise OverlayStoreError(
            "whitened embedding has near-zero norm; refusing to store"
        )
    weight = whitened / norm
    new_id = overlay.add_cell(
        weight=weight,
        source_text=text,
        source=source,
        label=label,
        theta=theta,
        base_max_id=int(base_max_id),
    )
    return new_id, weight


__all__ = [
    "OverlayStore",
    "OverlayPayload",
    "OverlayStoreError",
    "add_cell_from_text",
]
