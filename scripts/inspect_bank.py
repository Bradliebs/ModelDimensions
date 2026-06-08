"""One-shot bank inspector for cc_service. Read-only."""
from __future__ import annotations
import argparse
import json
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, type=Path)
    args = p.parse_args()
    if not args.db.exists():
        print(f"missing: {args.db}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(str(args.db))
    cur = conn.cursor()
    rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f"tables: {[r[0] for r in rows]}")
    for t in [r[0] for r in rows]:
        try:
            n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {n} rows")
        except Exception as e:
            print(f"  {t}: count failed ({e})")
    # Look for meta / config
    for t in ("meta", "config", "settings", "info"):
        if any(r[0] == t for r in rows):
            print(f"\n--- {t} ---")
            for row in cur.execute(f"SELECT * FROM {t} LIMIT 50").fetchall():
                print(row)
    # Try to find a cells table to get dim
    for cand in ("cells", "memory", "concept_cells"):
        if any(r[0] == cand for r in rows):
            schema = cur.execute(f"PRAGMA table_info({cand})").fetchall()
            print(f"\n{cand} schema:")
            for col in schema:
                print(f"  {col}")
            sample = cur.execute(f"SELECT * FROM {cand} LIMIT 1").fetchone()
            if sample:
                print(f"  sample row types: {[type(v).__name__ for v in sample]}")
                print(f"  sample row lengths: {[len(v) if isinstance(v,(bytes,str)) else '-' for v in sample]}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
