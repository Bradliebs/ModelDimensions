"""scripts/make_subset_bank.py -- carve a small subset of cc_service bank.db.

The production bank at H:\\MiniLM\\cc_service\\bank.db is 13.7 GB (5.7M cells)
and the cc_service query path loads ALL weights into a single in-memory matrix
on first /query (peak ~30 GB during np.stack). That doesn't fit comfortably on
a 48 GB host while RetroGPT is also loaded on GPU and the OS pagecache is
warm. Result: the cold first query swap-thrashes.

This script copies the schema + meta + whitening verbatim, then copies the
first N single-kind cells (with their source_texts) into a fresh bank file.
Subset bank loads in <1 second and queries return in single-digit ms after
the cache fills.

The encoder identity (all-MiniLM-L6-v2, dim 384) and whitening transform are
preserved -- queries against the subset return the same answers they would
against the full bank IF the answer happens to be in the subset.

Usage:
    python scripts/make_subset_bank.py \\
        --source H:\\MiniLM\\cc_service\\bank.db \\
        --dest H:\\MiniLM\\cc_service\\bank_subset_50k.db \\
        --n 50000
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cc_service.persistence import SCHEMA_SQL  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50000,
                        help="number of single cells to copy (default 50000)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"source bank not found: {args.source}")
    if args.dest.exists():
        if not args.overwrite:
            parser.error(f"dest exists (use --overwrite): {args.dest}")
        args.dest.unlink()

    t0 = time.time()
    src = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    dst = sqlite3.connect(str(args.dest))
    dst.executescript(SCHEMA_SQL)

    # ---- meta (whole table) ----
    rows = src.execute("SELECT key, value FROM meta").fetchall()
    dst.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", rows)
    print(f"[subset] copied {len(rows)} meta rows")

    # ---- whitening (single row) ----
    rows = src.execute(
        "SELECT id, mu, w_matrix, max_norm, fitted_at, reference_n FROM whitening"
    ).fetchall()
    dst.executemany(
        "INSERT INTO whitening (id, mu, w_matrix, max_norm, fitted_at, reference_n) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    print(f"[subset] copied {len(rows)} whitening row")

    # ---- cells (first N single cells) ----
    print(f"[subset] copying first {args.n:,} single cells...")
    cur = src.execute(
        "SELECT id, label, weight, theta, kind, created_at FROM cells "
        "WHERE kind = 'single' ORDER BY id ASC LIMIT ?",
        (args.n,),
    )
    batch = []
    n_cells = 0
    cell_ids_set: set[int] = set()
    for row in cur:
        batch.append(row)
        cell_ids_set.add(row[0])
        if len(batch) >= 10000:
            dst.executemany(
                "INSERT INTO cells (id, label, weight, theta, kind, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )
            n_cells += len(batch)
            batch = []
    if batch:
        dst.executemany(
            "INSERT INTO cells (id, label, weight, theta, kind, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )
        n_cells += len(batch)
    print(f"[subset] copied {n_cells:,} cells")

    # ---- source_texts for those cells ----
    # Iterate in batches of cell ids since IN clause has a 999-param cap by default.
    cell_ids_list = sorted(cell_ids_set)
    chunk = 500
    n_texts = 0
    for i in range(0, len(cell_ids_list), chunk):
        ids = cell_ids_list[i:i + chunk]
        placeholders = ",".join("?" * len(ids))
        rows = src.execute(
            f"SELECT cell_id, text FROM source_texts WHERE cell_id IN ({placeholders})",
            ids,
        ).fetchall()
        if rows:
            dst.executemany(
                "INSERT INTO source_texts (cell_id, text) VALUES (?, ?)", rows
            )
            n_texts += len(rows)
    print(f"[subset] copied {n_texts:,} source_texts rows")

    dst.commit()
    dst.execute("ANALYZE")
    dst.commit()

    # Reset AUTOINCREMENT so the next inserted cell would go above max(id).
    max_id = dst.execute("SELECT MAX(id) FROM cells").fetchone()[0]
    dst.execute(
        "INSERT OR REPLACE INTO sqlite_sequence (name, seq) VALUES ('cells', ?)",
        (max_id,),
    )
    dst.commit()

    src.close()
    dst.close()

    size_mb = args.dest.stat().st_size / (1024 * 1024)
    print(f"[subset] wrote {args.dest} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
