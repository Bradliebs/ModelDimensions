"""Step 3 follow-on (recall via reranker) — V1 pipeline eval with an
opt-in cross-encoder reranker over the question-shaped probe set.

Why this exists:
    `experiments/exp18_v1_pipeline_questions_eval.py` showed that with
    question-shaped probes and the gate margin tuned to 0.03, recall on
    the 5.7M-cell bank tops out at 7/12 known. The remaining 5 misses
    all sit at encoder ``top1 - top2`` margins of ≤ 0.019 — the
    bi-encoder cannot separate the right cell from its distractors. No
    knob on the bi-encoder side can fix that; per
    `docs/PLAN.md`'s "encoder ceiling" note, the candidate remedies are
    a stronger encoder, a cross-encoder reranker, or query rewriting.

    This experiment validates the **reranker** option. A small
    `cross-encoder/ms-marco-MiniLM-L-6-v2` is dropped in as a
    second-stage scorer over the top-k bi-encoder candidates. The gate
    then operates on rerank-margin instead of cosine-margin. Because
    cross-encoder logits live on a different scale than cosine cosines
    (~[-10, +10] vs. [-1, +1]) the threshold is swept rather than
    inherited from the bi-encoder default.

Outputs:
    results/v1_pipeline_reranker_eval_m<threshold>.json (one per --rerank-margin)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the eval harness and the question set from exp18.
from experiments.exp17_v1_pipeline_eval import run_eval, DEFAULT_BANK
from experiments.exp18_v1_pipeline_questions_eval import load_query_set


def _output_path(threshold: float) -> Path:
    tag = f"{threshold:+.2f}".replace("+", "p").replace("-", "n").replace(".", "")
    return REPO_ROOT / "results" / f"v1_pipeline_reranker_eval_m{tag}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--rerank-margin", type=float, default=2.0,
        help="rerank top1-top2 logit margin to fire the gate (default 2.0)",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", default=None,
                        help="defaults to results/v1_pipeline_reranker_eval_m<tag>.json")
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    print(f"[exp19] loading reranker {args.reranker_model}...", flush=True)
    from src.agent.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker(model_name=args.reranker_model)
    # Force load now so timings below are clean.
    _ = reranker.score("warmup query", ["warmup passage"])

    print(f"[exp19] loading pipeline...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        use_4bit=not args.no_4bit,
        reranker=reranker,
        rerank_margin_threshold=args.rerank_margin,
    )
    print(f"[exp19] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    queries = load_query_set()
    summary = run_eval(pipeline, queries)
    pipeline.close()

    out_path = Path(args.output) if args.output else _output_path(args.rerank_margin)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary["bank_path"] = str(bank_path)
    summary["top_k"] = int(args.top_k)
    summary["rerank_margin_threshold"] = float(args.rerank_margin)
    summary["reranker_model"] = args.reranker_model
    summary["use_4bit"] = not args.no_4bit
    summary["produced_by"] = "experiments/exp19_reranker_sweep.py"
    summary["known_set"] = "12 question-shaped queries derived from paragraph probes"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print(f"V1 pipeline reranker-eval summary  (rerank_margin={args.rerank_margin})")
    print("=" * 72)
    for cls, info in summary["by_class"].items():
        acc = info["accuracy"]
        acc_str = f"{acc * 100:5.1f}%" if acc is not None else "  n/a"
        print(f"  {cls:<8} n={info['n']:<3} accuracy={acc_str}  "
              f"counts={info['outcome_counts']}")
    lat = summary["latency_seconds"]
    print()
    print(f"  latency (sec):")
    for k, v in lat.items():
        if v["n"] == 0:
            continue
        print(f"    {k:<22} n={v['n']:<3} p50={v['p50']:.3f}  "
              f"p95={v['p95']:.3f}  mean={v['mean']:.3f}")
    print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
