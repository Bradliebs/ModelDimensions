"""Rebuild the concept-cell bank with a different encoder, non-destructively.

Changing the encoder invalidates every stored weight vector (they live in the
old encoder's whitened space). This script re-encodes from the *stored source
text* — no dataset re-download — into a brand new database file, leaving the
original bank untouched and the swap fully reversible.

Usage (from h:\\ai_mem_bank, with the cc_service venv python):
    python ..\\MiniLM\\cc_service\\rebuild_bank.py \
        --src-db h:\\ai_mem_bank\\bank.db \
        --dst-db h:\\ai_mem_bank\\bank_bge.db \
        --encoder BAAI/bge-base-en-v1.5

The destination must not already exist (refuses to clobber). Bound cells (no
source text) are skipped — they can be re-bound after the rebuild.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# Import pyarrow before torch loads. On Windows this venv segfaults
# (access violation) if pyarrow's DLLs are loaded *after* torch's; the
# sentence-transformers -> sklearn -> pandas -> pyarrow chain otherwise
# triggers the crash at first encode. Loading it first is harmless.
try:
    import pyarrow  # noqa: F401
except Exception:
    pass


def _pin_hf_cache_to_h() -> None:
    """Force all model downloads/caches onto H: before importing ST/torch."""
    cache = Path(r"h:\ai_mem_bank\.cache")
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache / "sentence-transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))


def read_singles(src_db: str):
    """Return list of (label, theta, text) for every single-kind cell."""
    con = sqlite3.connect(src_db)
    try:
        rows = con.execute(
            "SELECT c.label, c.theta, s.text "
            "FROM cells c JOIN source_texts s ON s.cell_id = c.id "
            "WHERE c.kind = 'single' AND s.text IS NOT NULL "
            "ORDER BY c.id"
        ).fetchall()
    finally:
        con.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-db", required=True)
    ap.add_argument("--dst-db", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ref-sample", type=int, default=30000,
                    help="texts used to fit whitening (default 30000)")
    ap.add_argument("--batch-size", type=int, default=2000,
                    help="write_many batch size (default 2000)")
    args = ap.parse_args()

    src = Path(args.src_db)
    dst = Path(args.dst_db)
    if not src.exists():
        print(f"ERROR: source db not found: {src}", file=sys.stderr)
        return 1
    if dst.exists():
        print(f"ERROR: destination already exists, refusing to clobber: {dst}",
              file=sys.stderr)
        return 1

    _pin_hf_cache_to_h()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    # Import only after the cache env is set.
    from cc_service.memory import MemoryBank

    t0 = time.time()
    print(f"Reading single cells from {src} ...", flush=True)
    rows = read_singles(str(src))
    n = len(rows)
    print(f"  {n:,} single cells with source text", flush=True)
    if n < 200:
        print("ERROR: need >= 200 cells to fit whitening", file=sys.stderr)
        return 1

    labels = [r[0] for r in rows]
    thetas = [float(r[1]) for r in rows]
    texts = [r[2] for r in rows]

    bank = MemoryBank(db_path=str(dst), encoder_model=args.encoder,
                      device=args.device)
    print(f"Encoder: {args.encoder} (dim={bank.encoder.dim})", flush=True)

    # --- Fit whitening on a sample (evenly strided for label coverage) ---
    ref_n = min(args.ref_sample, n)
    stride = max(1, n // ref_n)
    ref_texts = texts[::stride][:ref_n]
    print(f"Fitting whitening on {len(ref_texts):,} reference texts ...",
          flush=True)
    bank.init_from_corpus(ref_texts)
    print(f"  whitening fitted at t+{time.time()-t0:.0f}s", flush=True)

    # --- Write all cells, grouped by theta so thresholds are preserved ---
    # (loaders used a single default theta, so this is typically one group.)
    from collections import defaultdict
    by_theta = defaultdict(lambda: ([], []))  # theta -> (texts, labels)
    for txt, lab, th in zip(texts, labels, thetas):
        bucket = by_theta[th]
        bucket[0].append(txt)
        bucket[1].append(lab)

    written = 0
    for theta, (t_texts, t_labels) in by_theta.items():
        print(f"Writing {len(t_texts):,} cells at theta={theta} ...", flush=True)
        for i in range(0, len(t_texts), args.batch_size):
            b_texts = t_texts[i:i + args.batch_size]
            b_labels = t_labels[i:i + args.batch_size]
            ids = bank.write_many(b_texts, labels=b_labels, theta=theta)
            written += len(ids)
            if (i // args.batch_size) % 10 == 0:
                rate = written / max(1e-6, time.time() - t0)
                print(f"  {written:,}/{n:,}  ({rate:.0f} cells/s, "
                      f"t+{time.time()-t0:.0f}s)", flush=True)

    bank.store.close()
    print(f"DONE: wrote {written:,} cells to {dst} in "
          f"{time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
