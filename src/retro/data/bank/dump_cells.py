"""
dump_cells.py — regenerate cell_ids.npy + cell_tokens.npy from bank.db.

Same JOIN + ORDER as precompute_neighbors.load_bank, so positional indices
stay aligned with what /query and prepare_heldout produce. CPU only — no
GPU, no W_bank load. Use this after growing the bank when you do NOT want
to also rebuild train/val_neighbors (which precompute_neighbors.py would).

Usage:
    cd H:\\MiniLM\\nanogpt
    python data/bank/dump_cells.py
"""

import sqlite3
import time
from pathlib import Path

import numpy as np
import tiktoken

BANK_DB = r"H:\MiniLM\cc_service\bank.db"
OUT_DIR = Path(__file__).parent
NEIGHBOR_LEN = 64       # must match RetroConfig.neighbor_len


def main():
    t0 = time.time()
    tok_enc = tiktoken.get_encoding("gpt2")
    eot = tok_enc.eot_token

    conn = sqlite3.connect(f"file:{BANK_DB}?mode=ro", uri=True)
    n_total = conn.execute(
        "SELECT COUNT(*) FROM cells c JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single'"
    ).fetchone()[0]
    print(f"dumping {n_total:,} cells from bank...")

    cell_ids = np.zeros(n_total, dtype=np.uint32)
    cell_tokens = np.full((n_total, NEIGHBOR_LEN), eot, dtype=np.uint16)

    cur = conn.execute(
        "SELECT c.id, s.text FROM cells c "
        "JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single' ORDER BY c.id"
    )
    for i, (cid, text) in enumerate(cur):
        cell_ids[i] = cid
        toks = tok_enc.encode_ordinary(text)[:NEIGHBOR_LEN]
        if toks:
            cell_tokens[i, : len(toks)] = toks
        if (i + 1) % 200000 == 0:
            print(f"  {i+1:>9,}/{n_total:,}  ({time.time()-t0:.1f}s)")
    conn.close()

    np.save(OUT_DIR / "cell_ids.npy", cell_ids)
    np.save(OUT_DIR / "cell_tokens.npy", cell_tokens)
    print(f"\nwrote cell_ids.npy   ({cell_ids.nbytes/1e6:.1f} MB)")
    print(f"wrote cell_tokens.npy ({cell_tokens.nbytes/1e6:.1f} MB)")
    print(f"total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
