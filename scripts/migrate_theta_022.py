"""One-off migration: lower the per-cell firing gate from theta=0.30 to 0.22.

Rationale and evidence: calibrate_theta.py measured that on the BGE bank
foreign/no-answer queries peak at activation ~0.19 while true partial-mention
matches sit at p10~0.26. The inherited theta=0.30 gated out ~25% of genuine
hits (all code looked "empty"). 0.22 sits in the clean band: real hits fire by
default, unrelated queries still return empty.

Scope: only kind='single' cells (bound cells carry per-bind calibrated readout
thresholds and are left untouched). Reversible: re-run with --revert to restore
0.30 for cells currently at 0.22.

Usage (cc_service venv python):
    python migrate_theta_022.py --db h:\\ai_mem_bank\\bank_bge.db
    python migrate_theta_022.py --db h:\\ai_mem_bank\\bank_bge.db --revert
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

OLD = 0.30
NEW = 0.22


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--revert", action="store_true",
                    help="restore theta=0.30 for single cells at 0.22")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: db not found: {args.db}", file=sys.stderr)
        return 1

    src, dst = (NEW, OLD) if args.revert else (OLD, NEW)
    con = sqlite3.connect(args.db)
    try:
        pre = con.execute(
            "SELECT COUNT(*) FROM cells WHERE kind='single' AND theta=?",
            (src,),
        ).fetchone()[0]
        print(f"single cells at theta={src}: {pre}")
        con.execute(
            "UPDATE cells SET theta=? WHERE kind='single' AND theta=?",
            (dst, src),
        )
        con.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("theta_write_default", json.dumps(dst)),
        )
        con.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("theta_write_prev", json.dumps(src)),
        )
        con.commit()
        dist = con.execute(
            "SELECT theta, COUNT(*) FROM cells GROUP BY theta ORDER BY 2 DESC"
        ).fetchall()
        print(f"theta distribution now: {dist}")
        meta = con.execute(
            "SELECT value FROM meta WHERE key='theta_write_default'"
        ).fetchone()
        print(f"meta theta_write_default: {meta[0] if meta else None}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
