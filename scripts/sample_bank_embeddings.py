"""Sample cell embeddings from the cc_service bank and split into fit/eval.

Reads ``cells.weight`` BLOBs (float32, dim from meta) from the SQLite bank,
samples ``--n`` cells uniformly without replacement, splits them into fit
and eval halves with a fixed seed, and writes both as .npy files for
``scripts/verify_zca_isotropy.py``.

Usage:
    python scripts/sample_bank_embeddings.py \\
        --bank H:/MiniLM/cc_service/bank.db \\
        --n 40000 \\
        --out-fit results/bank_fit_emb.npy \\
        --out-eval results/bank_eval_emb.npy
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BANK = Path(r"H:\MiniLM\cc_service\bank.db")


def _embedding_dim(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
    if row is None:
        raise SystemExit("bank meta missing 'dim' row")
    return int(row[0].strip().strip('"'))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--n", type=int, default=40_000,
                        help="Total cells to sample (split 50/50 fit/eval)")
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-fit", type=Path, required=True)
    parser.add_argument("--out-eval", type=Path, required=True)
    parser.add_argument("--max-id-probe", type=int, default=None,
                        help="Override max cell id (skip MAX scan)")
    args = parser.parse_args(argv)

    if not args.bank.exists():
        parser.error(f"bank not found: {args.bank}")
    if not (0.0 < args.fit_fraction < 1.0):
        parser.error("--fit-fraction must be in (0, 1)")
    if args.n < 2:
        parser.error("--n must be >= 2")

    rng = np.random.default_rng(args.seed)
    conn = sqlite3.connect(str(args.bank))
    dim = _embedding_dim(conn)
    print(f"[sample] bank={args.bank} dim={dim}")

    if args.max_id_probe is not None:
        max_id = args.max_id_probe
    else:
        max_id = conn.execute("SELECT MAX(id) FROM cells").fetchone()[0]
    print(f"[sample] max cell id = {max_id:,}; sampling {args.n:,} ids")

    # Sample more than n because some ids may be missing (deleted cells).
    # Issue queries in small id-chunks so we stay well under SQLite's
    # default parameter limit (32766; older builds 999).
    target = args.n
    id_chunk = 900
    seen_ids: set[int] = set()
    embeddings: list[np.ndarray] = []
    t0 = time.time()
    while len(embeddings) < target:
        need = target - len(embeddings)
        # Oversample 1.2x to absorb missing ids.
        draw = rng.integers(low=1, high=max_id + 1, size=int(need * 1.2) + 16)
        draw = [int(x) for x in draw if int(x) not in seen_ids]
        if not draw:
            continue
        for i in range(0, len(draw), id_chunk):
            chunk = draw[i:i + id_chunk]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT id, weight FROM cells "
                f"WHERE id IN ({placeholders}) AND weight IS NOT NULL",
                chunk,
            ).fetchall()
            for cid, blob in rows:
                if cid in seen_ids:
                    continue
                arr = np.frombuffer(blob, dtype=np.float32)
                if arr.shape[0] != dim:
                    continue
                embeddings.append(arr.copy())
                seen_ids.add(cid)
                if len(embeddings) >= target:
                    break
            if len(embeddings) >= target:
                break
        print(f"  collected {len(embeddings):,}/{target:,} in {time.time()-t0:.1f}s")

    conn.close()
    arr = np.stack(embeddings[:target], axis=0)
    elapsed = time.time() - t0
    print(f"[sample] gathered {arr.shape} in {elapsed:.1f}s")

    # Shuffle then split — keeps fit/eval IID.
    perm = rng.permutation(arr.shape[0])
    arr = arr[perm]
    n_fit = int(round(arr.shape[0] * args.fit_fraction))
    fit = arr[:n_fit]
    ev = arr[n_fit:]
    print(f"[sample] split: fit={fit.shape}, eval={ev.shape}")

    for p in (args.out_fit, args.out_eval):
        p.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_fit, fit)
    np.save(args.out_eval, ev)
    print(f"[sample] wrote {args.out_fit} and {args.out_eval}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
