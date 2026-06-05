"""Probe: does the cross-encoder reranker rescue paraphrase queries?

The theta gate works on absolute bi-encoder activation. Paraphrase queries
(not literal substrings of stored text) activate near the foreign-noise floor
(~0.19), so the gate can't separate them from gibberish even though the correct
cell is often rank-1 by activation.

This probe runs realistic paraphrase queries plus a nonsense control through:
  * stage-1 only  : top candidates by bi-encoder activation (recall view)
  * stage-1+rerank: same pool rescored by BAAI/bge-reranker-base

If the reranker scores the correct domain highly for real queries and low for
the nonsense control, a top-k-by-activation + rerank pipeline is the right
default for a usable system (and rerank_score is the rejection signal, not theta).

Usage (cc_service venv python, from h:\\MiniLM\\cc_service):
    python probe_rerank.py --db h:\\ai_mem_bank\\bank_bge.db
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    import pyarrow  # noqa: F401
except Exception:
    pass


def _pin_hf_cache_to_h() -> None:
    cache = Path(r"h:\ai_mem_bank\.cache")
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache / "sentence-transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))


QUERIES = [
    "write a python function to reverse a string",
    "SQL query to select all users",
    "bash script to list files",
    "What is the capital of France?",
    "quantum velvet harbor sprocket lantern thistle cobalt zephyr trellis",  # control
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    _pin_hf_cache_to_h()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from cc_service.memory import MemoryBank

    bank = MemoryBank(db_path=args.db, encoder_model=args.encoder,
                      device=args.device, reranker_model=args.reranker)

    for q in QUERIES:
        print(f"\n=== {q} ===")
        print(" stage-1 (top by activation):")
        s1 = bank.query(q, top_k=args.top_k, include_silent=True, rerank=False)
        for h in s1:
            lab = (h.label or "")[:18]
            txt = (h.source_text or "")[:55].replace("\n", " ")
            print(f"   act={h.activation:6.3f}  {lab:<18} {txt}")
        print(" stage-2 (reranked):")
        s2 = bank.query(q, top_k=args.top_k, rerank=True,
                        candidate_pool=args.pool)
        for h in s2:
            lab = (h.label or "")[:18]
            txt = (h.source_text or "")[:55].replace("\n", " ")
            rs = h.rerank_score if h.rerank_score is not None else float("nan")
            print(f"   rr={rs:7.3f}  act={h.activation:6.3f}  {lab:<18} {txt}")

    bank.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
