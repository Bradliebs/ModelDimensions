"""CLI: build a BM25 lexical index over the production bank.

    python scripts/build_lexical_index.py \
        --bank-path H:\\MiniLM\\cc_service\\bank.db \
        --out-dir results/v1_bank/bm25_index

For Phase 1 prove-out (8-question multihop eval), pass ``--limit N`` to
build over only the first N cells by id; subsequent runs can scale up
without re-indexing the smaller corpus.

Index format is a directory with three files:
    bm25.pkl       - pickled rank_bm25.BM25Okapi
    cell_ids.npy   - int64 array, parallel to BM25's internal corpus
    manifest.json  - corpus hash, tokenizer version, k1, b, n_docs

Exits 0 on success, 2 on error.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BANK = os.environ.get(
    "MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db"
)
DEFAULT_OVERLAY = os.environ.get("MD_OVERLAY_PATH", "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BM25 lexical index.")
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--overlay-path", default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory to write the index files into (will be created).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap corpus to the first N cells by id (Phase 1 prove-out).",
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_path: Path | None = None
    if args.overlay_path:
        overlay_path = Path(args.overlay_path)
        if not overlay_path.exists():
            print(
                f"ERROR: overlay not found: {overlay_path}", file=sys.stderr
            )
            return 2

    print(f"[build] loading bank: {bank_path}", flush=True)
    t0 = time.time()
    from src.agent.streaming_bank import StreamingBank
    if overlay_path is not None:
        from src.agent.bank_admin import OverlayStore
        bank = StreamingBank(str(bank_path), overlay=OverlayStore(overlay_path))
    else:
        bank = StreamingBank(str(bank_path))
    print(
        f"[build] bank ready in {time.time() - t0:.1f}s "
        f"(n_cells={bank.n_cells}, dim={bank.dim})",
        flush=True,
    )

    from src.agent.lexical_index import LexicalIndex
    idx = LexicalIndex()

    last_print = [time.time()]

    def progress(p: dict) -> None:
        # Throttle to once per ~2s to keep stdout readable.
        now = time.time()
        if now - last_print[0] < 2.0 and p["loaded"] < p["total"]:
            return
        last_print[0] = now
        loaded = p["loaded"]
        total = p["total"]
        pct = (100.0 * loaded / total) if total else 100.0
        print(
            f"[build]   fetched texts: {loaded}/{total} ({pct:.1f}%)",
            flush=True,
        )

    t0 = time.time()
    idx.build_from_bank(
        bank, limit=args.limit, k1=args.k1, b=args.b,
        progress_callback=progress,
    )
    print(
        f"[build] indexed {idx.n_docs} docs in {time.time() - t0:.1f}s",
        flush=True,
    )

    t0 = time.time()
    idx.save(out_dir)
    print(
        f"[build] wrote {out_dir} in {time.time() - t0:.1f}s",
        flush=True,
    )
    bank.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
