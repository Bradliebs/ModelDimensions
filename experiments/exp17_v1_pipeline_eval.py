"""Step 3 evaluation — V1 pipeline batch eval over the labelled probe set.

Reuses the 50 queries from `results/bank_selectivity_at_5p7m.json`
(20 known, 20 unknown, 10 noise). Runs each through the full V1 pipeline
(retrieve -> silence gate -> Phi-3 generate -> verify) and records:

    - per-query expected vs observed (grounded / silence_gate / silence_drift)
    - per-stage latencies (encode, retrieve, fetch, build_prompt, generate,
      verify, total)
    - per-query gate margin and verifier coverage when applicable

Aggregates into:

    accuracy:
      known   -> P(grounded)            (paraphrases not separately scored)
      unknown -> P(silence)             (must be high)
      noise   -> P(silence)             (must be very high)

    latency:
      total p50/p95/mean
      generate p50/p95/mean
      retrieve p50/p95/mean

Output: results/v1_pipeline_eval.json. No assertions here; this is
measurement, not regression. The companion test
`evals/test_v1_pipeline_eval_harness.py` exercises the harness against
the tiny test bank.

Runtime on a 3070 with Phi-3 4-bit: ~3-4 minutes (bank load ~30 s,
generation ~3 s × 50 queries on grounded paths, gated paths short-circuit).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SOURCE_QUERIES = REPO_ROOT / "results" / "bank_selectivity_at_5p7m.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_pipeline_eval.json"
DEFAULT_BANK = os.environ.get(
    "MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db"
)


def _classify(result_dict: dict) -> str:
    """Map a PipelineResult.as_dict() to one of:
       'grounded' | 'silence_gate' | 'silence_drift'."""
    if not result_dict["silence"]:
        return "grounded"
    reason = result_dict.get("silence_reason", "") or ""
    if reason.startswith("verify"):
        return "silence_drift"
    return "silence_gate"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _summarize_latencies(samples: Iterable[float]) -> dict:
    arr = [float(x) for x in samples]
    if not arr:
        return {"n": 0, "p50": None, "p95": None, "mean": None}
    return {
        "n": len(arr),
        "p50": _percentile(arr, 50),
        "p95": _percentile(arr, 95),
        "mean": float(mean(arr)),
    }


def load_query_set() -> dict[str, list[dict]]:
    data = json.loads(SOURCE_QUERIES.read_text(encoding="utf-8"))
    return {
        "known": [{"query": e["query"], "expected": "grounded"}
                  for e in data["known"]],
        "unknown": [{"query": e["query"], "expected": "silence"}
                    for e in data["unknown"]],
        "noise": [{"query": e["query"], "expected": "silence"}
                  for e in data["noise"]],
    }


def run_eval(pipeline, queries: dict[str, list[dict]]) -> dict:
    per_query: list[dict] = []
    by_class: dict[str, dict] = {}

    for cls, items in queries.items():
        records: list[dict] = []
        for item in items:
            t0 = time.time()
            result = pipeline.ask(item["query"])
            wall = time.time() - t0
            d = result.as_dict()
            outcome = _classify(d)
            rec = {
                "class": cls,
                "expected": item["expected"],
                "outcome": outcome,
                "wall_time_sec": wall,
                "gate_fire": d["gate"]["fire"],
                "gate_margin": d["gate"]["margin"],
                "verify_grounded": (d["verification"] or {}).get("grounded"),
                "verify_coverage": (d["verification"] or {}).get("coverage"),
                "n_citations": len(d["citations"]),
                "timings": d["timings"],
                "query_head": item["query"][:120],
            }
            records.append(rec)
            per_query.append(rec)

            tag = "OK" if (
                (item["expected"] == "grounded" and outcome == "grounded") or
                (item["expected"] == "silence" and outcome != "grounded")
            ) else "MISS"
            print(f"  [{cls:<7}] [{tag}] {outcome:<14} "
                  f"wall={wall:.2f}s margin={d['gate']['margin']:+.3f}  "
                  f"{item['query'][:80]!r}", flush=True)

        # Per-class accuracy.
        if cls == "known":
            score_n = sum(1 for r in records if r["outcome"] == "grounded")
        else:
            score_n = sum(1 for r in records if r["outcome"] != "grounded")
        by_class[cls] = {
            "n": len(records),
            "correct": score_n,
            "accuracy": score_n / len(records) if records else None,
            "outcome_counts": {
                "grounded": sum(1 for r in records if r["outcome"] == "grounded"),
                "silence_gate": sum(1 for r in records if r["outcome"] == "silence_gate"),
                "silence_drift": sum(1 for r in records if r["outcome"] == "silence_drift"),
            },
        }

    # Latency aggregates over the whole run.
    latency = {
        "total_wall": _summarize_latencies(r["wall_time_sec"] for r in per_query),
        "retrieve": _summarize_latencies(r["timings"].get("retrieve", 0.0)
                                         for r in per_query),
        "encode": _summarize_latencies(r["timings"].get("encode", 0.0)
                                       for r in per_query),
        # Generate is only measured on grounded-or-drift paths (gate fired).
        "generate_when_run": _summarize_latencies(
            r["timings"]["generate"] for r in per_query
            if "generate" in r["timings"]
        ),
        "verify_when_run": _summarize_latencies(
            r["timings"]["verify"] for r in per_query
            if "verify" in r["timings"]
        ),
    }

    return {
        "produced_by": "experiments/exp17_v1_pipeline_eval.py",
        "n_queries": len(per_query),
        "by_class": by_class,
        "latency_seconds": latency,
        "records": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2
    if not SOURCE_QUERIES.exists():
        print(f"ERROR: probe queries not found: {SOURCE_QUERIES}",
              file=sys.stderr)
        return 2

    print(f"[exp17] loading pipeline...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=args.margin,
        use_4bit=not args.no_4bit,
    )
    print(f"[exp17] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    queries = load_query_set()
    summary = run_eval(pipeline, queries)
    pipeline.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary["bank_path"] = str(bank_path)
    summary["top_k"] = int(args.top_k)
    summary["margin_threshold"] = float(args.margin)
    summary["use_4bit"] = not args.no_4bit
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("V1 pipeline eval summary")
    print("=" * 72)
    for cls, agg in summary["by_class"].items():
        acc = agg["accuracy"]
        acc_s = f"{acc * 100:.1f}%" if acc is not None else "n/a"
        print(f"  {cls:<8} n={agg['n']:<3} accuracy={acc_s:>6}  "
              f"counts={agg['outcome_counts']}")
    lat = summary["latency_seconds"]
    print()
    print("  latency (sec):")
    for k, v in lat.items():
        if v["n"] == 0:
            continue
        print(f"    {k:<22} n={v['n']:<3} p50={v['p50']:.3f}  "
              f"p95={v['p95']:.3f}  mean={v['mean']:.3f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
