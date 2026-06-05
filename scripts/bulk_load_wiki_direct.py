"""bulk_load_wiki_direct.py — bulk-load Wikipedia EN directly into bank.db,
bypassing the HTTP service entirely.

The HTTP loader (load_wikipedia_en.py) is unreliable for long bulk runs because
the uvicorn-hosted service has been observed to die silently under sustained
write load (no traceback to stderr -- likely a torch/CUDA or sqlite3 native
crash). This script avoids the issue by calling MemoryBank.write_many() in
the same process, which means:
  - no HTTP layer / no uvicorn worker
  - one Python process holds the SQLite writer connection
  - direct error visibility
  - same write path as the service (same whitening, same insert logic)

The HTTP service MUST BE STOPPED before running this -- SQLite supports only
one writer at a time. After the bulk load finishes you can restart the service
normally.

Usage:
    cd H:\\MiniLM\\cc_service
    python bulk_load_wiki_direct.py --skip-articles 156700 --max-articles 300000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `cc_service` importable (lives in ../src/cc_service)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# datasets must import BEFORE numpy/torch on Windows to avoid pyarrow DLL crash
from datasets import load_dataset                              # noqa: E402

from cc_service.memory import MemoryBank                       # noqa: E402


def split_paragraphs(text: str, min_chars: int) -> list[str]:
    out: list[str] = []
    for para in text.split("\n\n"):
        clean = " ".join(para.strip().split())
        if len(clean) >= min_chars:
            out.append(clean)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default=r"H:\MiniLM\cc_service\bank.db")
    p.add_argument("--skip-articles", type=int, default=0)
    p.add_argument("--max-articles", type=int, default=100_000)
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=200,
                   help="Number of paragraphs per write_many call")
    p.add_argument("--progress-every", type=int, default=2000,
                   help="Print rate snapshot every N articles processed")
    p.add_argument("--encoder", default="all-MiniLM-L6-v2")
    args = p.parse_args()

    print(f"opening bank at {args.db_path} ...", flush=True)
    bank = MemoryBank(db_path=args.db_path, encoder_model=args.encoder)
    if not bank.is_initialized:
        sys.exit("Bank has no whitening fitted; cannot bulk-load.")
    n0 = bank.store.count()
    print(f"bank cells before: {n0:,}", flush=True)

    print(f"\nstreaming wikimedia/wikipedia 20231101.en "
          f"(skip {args.skip_articles:,}, take {args.max_articles:,})...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)

    n_seen = 0
    n_used = 0
    n_para = 0
    n_cells_written = 0
    pending: list[str] = []
    t0 = time.time()
    t_last = t0
    n_cells_last = 0

    try:
        for article in ds:
            n_seen += 1
            if n_seen <= args.skip_articles:
                if n_seen % 25_000 == 0:
                    print(f"  skipping {n_seen:,}/{args.skip_articles:,}...",
                          flush=True)
                continue
            if n_used >= args.max_articles:
                break

            n_used += 1
            for para in split_paragraphs(article.get("text", ""), args.min_chars):
                pending.append(para)
                n_para += 1
                if len(pending) >= args.batch_size:
                    ids = bank.write_many(pending)
                    n_cells_written += len(ids)
                    pending = []

            if n_used % args.progress_every == 0:
                now = time.time()
                dt = max(now - t_last, 1e-6)
                rate_window = (n_cells_written - n_cells_last) / dt
                rate_avg = n_cells_written / max(now - t0, 1e-6)
                print(f"  [art {n_used:>6,}/{args.max_articles:,}] "
                      f"para={n_para:>7,}  cells_added={n_cells_written:>7,}  "
                      f"win_rate={rate_window:.0f}/s  "
                      f"avg_rate={rate_avg:.0f}/s  "
                      f"elapsed={(now-t0)/60:.1f}m  "
                      f"last_article_idx={n_seen:,}",
                      flush=True)
                t_last = now
                n_cells_last = n_cells_written
    except KeyboardInterrupt:
        print(f"\n!! KeyboardInterrupt at article_idx={n_seen}, "
              f"flushing pending batch ({len(pending)} paragraphs)...",
              flush=True)

    # Flush final partial batch
    if pending:
        ids = bank.write_many(pending)
        n_cells_written += len(ids)

    n1 = bank.store.count()
    print(f"\n=== DONE ===", flush=True)
    print(f"articles processed: {n_used:,}", flush=True)
    print(f"paragraphs encoded: {n_para:,}", flush=True)
    print(f"cells written:      {n_cells_written:,}", flush=True)
    print(f"bank cells: {n0:,} -> {n1:,} (+{n1-n0:,})", flush=True)
    print(f"last article index seen: {n_seen:,}", flush=True)
    print(f"total time: {(time.time()-t0)/60:.2f}m", flush=True)


if __name__ == "__main__":
    main()
