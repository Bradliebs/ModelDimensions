"""In-process retrieval eval for comparing bank encoders / reranking.

Runs against a given database + encoder (+ optional reranker) without touching
the live service. Two automatic, label-free metrics that are fair to compare
across different encoders:

  1. Self-retrieval (partial query): for a sampled cell, query with the first
     ~half of its own source text and check the rank of that same cell in the
     results. Tests whether the geometry binds a partial mention back to its
     full passage. Reports recall@1, recall@5, and mean reciprocal rank (MRR).

  2. Domain purity@5: fraction of the top-5 neighbors that share the sampled
     cell's domain tag (the [tag] prefix on the label). Higher = cleaner
     semantic separation.

Probes are stratified across domains and seeded for reproducibility, so the
old (MiniLM) and new (BGE) banks see comparable difficulty.

Usage (from h:\\ai_mem_bank, cc_service venv python):
    python ..\\MiniLM\\cc_service\\eval_retrieval.py \
        --db h:\\ai_mem_bank\\bank.db --encoder all-MiniLM-L6-v2
    python ..\\MiniLM\\cc_service\\eval_retrieval.py \
        --db h:\\ai_mem_bank\\bank_bge.db --encoder BAAI/bge-base-en-v1.5 \
        --reranker BAAI/bge-reranker-base
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
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
    cache = Path(r"h:\ai_mem_bank\.cache")
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache / "sentence-transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))


_TAG_RE = re.compile(r"^\[([^\]]+)\]")


def domain_of(label: str | None) -> str:
    if label and label.startswith("["):
        m = _TAG_RE.match(label)
        if m:
            return m.group(1)
    return "NONE"


def half_query(text: str) -> str:
    """First ~half of the text (>= 8 tokens) as a partial-mention query."""
    toks = text.split()
    k = max(8, len(toks) // 2)
    return " ".join(toks[:k])


def sample_probes(db: str, per_domain: int, seed: int):
    """Return list of (cell_id, domain, source_text), stratified by domain."""
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT c.id, c.label, s.text "
            "FROM cells c JOIN source_texts s ON s.cell_id = c.id "
            "WHERE c.kind='single' AND s.text IS NOT NULL AND length(s.text) > 40"
        ).fetchall()
    finally:
        con.close()
    by_dom = defaultdict(list)
    for cid, label, text in rows:
        by_dom[domain_of(label)].append((cid, text))
    rng = random.Random(seed)
    probes = []
    for dom in sorted(by_dom):
        pool = by_dom[dom]
        rng.shuffle(pool)
        for cid, text in pool[:per_domain]:
            probes.append((cid, dom, text))
    return probes


def run_eval(db: str, encoder: str, reranker: str | None,
             per_domain: int, top_k: int, seed: int, device: str | None):
    _pin_hf_cache_to_h()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from cc_service.memory import MemoryBank

    probes = sample_probes(db, per_domain, seed)
    bank = MemoryBank(db_path=db, encoder_model=encoder, device=device,
                      reranker_model=reranker)

    modes = ["baseline"]
    if reranker:
        modes.append("reranked")

    results = {}
    for mode in modes:
        use_rerank = (mode == "reranked")
        hit1 = hit5 = 0
        rr_sum = 0.0
        purity_sum = 0.0
        n = 0
        t0 = time.time()
        for cid, dom, text in probes:
            q = half_query(text)
            hits = bank.query(q, top_k=top_k, include_silent=True,
                              rerank=use_rerank)
            ids = [h.cell_id for h in hits]
            doms = [domain_of(h.label) for h in hits]
            rank = ids.index(cid) + 1 if cid in ids else 0
            if rank == 1:
                hit1 += 1
            if rank and rank <= 5:
                hit5 += 1
            rr_sum += (1.0 / rank) if rank else 0.0
            top5 = doms[:5]
            if top5:
                purity_sum += sum(1 for d in top5 if d == dom) / len(top5)
            n += 1
        results[mode] = {
            "n_probes": n,
            "recall@1": round(hit1 / n, 4),
            "recall@5": round(hit5 / n, 4),
            "mrr": round(rr_sum / n, 4),
            "domain_purity@5": round(purity_sum / n, 4),
            "elapsed_s": round(time.time() - t0, 1),
        }
    bank.store.close()
    return {
        "db": db, "encoder": encoder, "reranker": reranker,
        "top_k": top_k, "per_domain": per_domain, "seed": seed,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--per-domain", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: db not found: {args.db}", file=sys.stderr)
        return 1

    report = run_eval(args.db, args.encoder, args.reranker,
                      args.per_domain, args.top_k, args.seed, args.device)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
