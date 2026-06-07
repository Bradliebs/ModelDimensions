"""Diagnostic: where in the bank (by cell_id) do the keyword cells for
the 8 multihop queries live? Tells us whether a 100k-cell BM25 index is
sufficient for the Phase 1 prove-out, or whether a larger build is
mandatory.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp20_multihop_probe import MULTIHOP_QUERIES  # noqa: E402


BANK = os.environ.get("MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db")


# Search seeds chosen to match the expected keyword cells per query.
# Use distinctive proper-noun strings that should appear in the *right*
# Wikipedia cell, not just any cell that happens to mention the keyword.
SEEDS = [
    # Q1 Napoleon -> Saint Helena -> English
    ("Q1 napoleon-saint-helena",
     ["Saint Helena", "Napoleon"], ["English"]),
    # Q2 Best Picture 1972 -> French Connection -> Don Ellis
    ("Q2 french-connection-don-ellis",
     ["French Connection"], ["Don Ellis"]),
    # Q3 dynamite -> Nobel -> Sweden
    ("Q3 alfred-nobel-sweden",
     ["Alfred Nobel"], ["Sweden", "Swedish"]),
    # Q4 2010 Olympics -> Vancouver/Canada -> Ottawa
    ("Q4 canada-ottawa",
     ["Canada"], ["Ottawa"]),
    # Q5 WWII monarch -> George VI -> Elizabeth II
    ("Q5 george-vi-succession",
     ["George VI"], ["Elizabeth II", "succeeded"]),
    # Q6 first man on moon (from MULTIHOP_QUERIES, ad hoc keywords)
    # Pull from the actual MULTIHOP_QUERIES below to stay aligned.
]


def search(conn, anchor_phrases: list[str], keyword_phrases: list[str]):
    """Find cells whose text contains any anchor AND any keyword phrase."""
    cur = conn.cursor()
    anchor_like = " OR ".join("text LIKE ?" for _ in anchor_phrases)
    keyword_like = " OR ".join("text LIKE ?" for _ in keyword_phrases)
    params = [f"%{a}%" for a in anchor_phrases] + \
             [f"%{k}%" for k in keyword_phrases]
    sql = f"""
        SELECT cell_id, text FROM source_texts
        WHERE ({anchor_like}) AND ({keyword_like})
        ORDER BY cell_id ASC
        LIMIT 5
    """
    return list(cur.execute(sql, params))


def main() -> int:
    uri = f"file:{BANK}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)

    print(f"\n{'='*70}\nKeyword cell discovery for the 8 multihop queries\n{'='*70}\n")
    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        q = item["query"]
        kws = item["expected_keywords"]
        print(f"[Q{i}] {q[:70]}")
        print(f"      expected keywords: {kws}")
        # Search for cells that contain any expected keyword. The first
        # bank id wins — that's the cell BM25 would surface.
        results = search(conn, kws, kws)  # anchor==keyword: any match
        if not results:
            print(f"      NO MATCH in bank by keyword text grep")
        else:
            ids = [r[0] for r in results]
            print(f"      first 5 matching cell ids: {ids}")
            in_100k = sum(1 for cid in ids if cid < 100_000)
            print(f"      within first 100k: {in_100k}/{len(ids)}")
            # Show the first hit's text snippet
            snip = (results[0][1] or "")[:200]
            print(f"      cell {results[0][0]}: {snip}...")
        print()

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
