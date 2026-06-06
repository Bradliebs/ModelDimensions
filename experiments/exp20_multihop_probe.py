"""Step 3 follow-on (multi-hop probe) — does the V1 pipeline already handle
questions that require combining facts from two distinct cells, or is
multi-cell synthesis the next real gap?

Why this exists:
    The probe sets in exp17/exp18 are single-fact lookups: every answer
    sits in one paragraph, the pipeline either retrieves that paragraph
    or it doesn't. A "pattern completion" / association-graph layer
    would help only if the failure mode is *right cells, bad synthesis*
    — i.e. the top-k contains the facts needed but Phi-3 cannot stitch
    them. If the failure mode is instead *wrong cells retrieved*, more
    associative machinery would not help; that's a retrieval-quality
    problem the reranker (exp19) addresses directly.

    This experiment runs eight questions whose answers require chaining
    two facts that almost certainly live in different Wikipedia
    paragraphs (e.g. "language spoken on the island where Napoleon was
    exiled in 1815" needs both the exile location and that location's
    language). It captures the top-k cell texts, the generated answer,
    the gate margin, and the verifier verdict so each result can be
    classified by hand:

        retrieval-miss  : right paragraphs not in top-k
        synthesis-miss  : right paragraphs in top-k, answer wrong/silent
        grounded        : end-to-end correct

Outputs:
    results/v1_pipeline_multihop_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BANK = os.environ.get("MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_pipeline_multihop_eval.json"


# Each entry: (question, expected_answer_keywords, hop_explanation).
# expected_answer_keywords is a small list of strings that, if any
# appear in the generated answer, indicate the system answered
# correctly. hop_explanation documents the two facts that have to be
# chained — it lives in the JSON so manual classification is fast.
MULTIHOP_QUERIES: list[dict] = [
    {
        "query": "What language is spoken on the island where Napoleon was exiled after Waterloo?",
        "expected_keywords": ["English"],
        "hops": (
            "(Napoleon, 1815 exile after Waterloo) -> Saint Helena;  "
            "(Saint Helena, official language) -> English."
        ),
    },
    {
        "query": "Who composed the music for the film that won the Academy Award for Best Picture in 1972?",
        "expected_keywords": ["Nino Rota"],
        "hops": (
            "(Best Picture, 1972) -> The Godfather;  "
            "(The Godfather, composer) -> Nino Rota."
        ),
    },
    {
        "query": "In which country was the inventor of dynamite born?",
        "expected_keywords": ["Sweden", "Swedish"],
        "hops": (
            "(dynamite, inventor) -> Alfred Nobel;  "
            "(Alfred Nobel, born) -> Stockholm, Sweden."
        ),
    },
    {
        "query": "What is the capital of the country where the 2010 Winter Olympics were held?",
        "expected_keywords": ["Ottawa"],
        "hops": (
            "(2010 Winter Olympics, host city) -> Vancouver, Canada;  "
            "(Canada, capital) -> Ottawa."
        ),
    },
    {
        "query": "Who succeeded the British monarch who reigned throughout the Second World War?",
        "expected_keywords": ["Elizabeth II", "Elizabeth"],
        "hops": (
            "(British monarch, WWII) -> George VI;  "
            "(George VI, succeeded by) -> Elizabeth II."
        ),
    },
    {
        "query": "What is the highest mountain in the country whose flag features a red maple leaf?",
        "expected_keywords": ["Logan"],
        "hops": (
            "(red maple leaf flag) -> Canada;  "
            "(Canada, highest mountain) -> Mount Logan."
        ),
    },
    {
        "query": "What religion was the founder of psychoanalysis raised in?",
        "expected_keywords": ["Jewish", "Judaism"],
        "hops": (
            "(psychoanalysis, founder) -> Sigmund Freud;  "
            "(Freud, religious background) -> Jewish."
        ),
    },
    {
        "query": "In what city was the author of 'The Old Man and the Sea' born?",
        "expected_keywords": ["Oak Park"],
        "hops": (
            "(The Old Man and the Sea, author) -> Ernest Hemingway;  "
            "(Hemingway, birthplace) -> Oak Park, Illinois."
        ),
    },
]


def _classify(d: dict) -> str:
    if not d["silence"]:
        return "grounded"
    reason = d.get("silence_reason", "") or ""
    return "silence_drift" if reason.startswith("verify") else "silence_gate"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _check_keywords(answer: str, keywords: list[str]) -> bool:
    a = (answer or "").lower()
    return any(kw.lower() in a for kw in keywords)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--top-cells-snippet-len", type=int, default=240,
                        help="character cap on cell-text snippets in the JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    print(f"[exp20] loading pipeline...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=args.margin,
        use_4bit=not args.no_4bit,
    )
    print(f"[exp20] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    records: list[dict] = []
    snip = int(args.top_cells_snippet_len)

    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        q = item["query"]
        keywords = item["expected_keywords"]
        t0 = time.time()
        result = pipeline.ask(q)
        wall = time.time() - t0
        d = result.as_dict()
        outcome = _classify(d)

        # Pull top-3 cell texts for hand classification.
        top_ids = d["retrieval"]["top_k_cell_ids"][:3]
        top_acts = d["retrieval"]["activations"][:3]
        try:
            top_texts = pipeline.bank.fetch_source_texts(top_ids)
        except Exception:
            top_texts = [None] * len(top_ids)
        top_cells = [
            {
                "cell_id": int(cid),
                "activation": float(act),
                "text_snippet": (t or "")[:snip] + ("..." if t and len(t) > snip else ""),
            }
            for cid, act, t in zip(top_ids, top_acts, top_texts)
        ]

        kw_hit = _check_keywords(d["answer"], keywords)

        rec = {
            "n": i,
            "query": q,
            "expected_keywords": keywords,
            "hops": item["hops"],
            "outcome": outcome,
            "answer": d["answer"],
            "kw_hit": kw_hit,
            "gate_margin": d["gate"]["margin"],
            "gate_threshold": d["gate"]["threshold"],
            "verifier_coverage": (d["verification"] or {}).get("coverage"),
            "verifier_uncovered": (d["verification"] or {}).get("uncovered_tokens", [])[:5],
            "verifier_uncited_numerics": (d["verification"] or {}).get("uncited_numerics", [])[:3],
            "citations": d["citations"][:5],
            "top_cells": top_cells,
            "wall_time_sec": wall,
            "silence_reason": d.get("silence_reason", ""),
        }
        records.append(rec)

        tag = "OK" if (outcome == "grounded" and kw_hit) else "MISS"
        print(
            f"  [{i}/{len(MULTIHOP_QUERIES)}] [{tag}] {outcome:<14} "
            f"kw={kw_hit}  margin={d['gate']['margin']:+.3f}  "
            f"wall={wall:.2f}s  {q[:70]}",
            flush=True,
        )
        if outcome == "grounded":
            print(f"            answer: {d['answer'][:160]}", flush=True)

    pipeline.close()

    # Summary
    n = len(records)
    grounded = sum(1 for r in records if r["outcome"] == "grounded")
    correct = sum(1 for r in records if r["outcome"] == "grounded" and r["kw_hit"])
    silence_gate = sum(1 for r in records if r["outcome"] == "silence_gate")
    silence_drift = sum(1 for r in records if r["outcome"] == "silence_drift")
    walls = [r["wall_time_sec"] for r in records]

    summary = {
        "produced_by": "experiments/exp20_multihop_probe.py",
        "bank_path": str(bank_path),
        "top_k": int(args.top_k),
        "margin_threshold": float(args.margin),
        "n_queries": n,
        "by_outcome": {
            "grounded": grounded,
            "grounded_with_kw_hit": correct,
            "silence_gate": silence_gate,
            "silence_drift": silence_drift,
        },
        "latency_seconds": {
            "n": n,
            "p50": _percentile(walls, 50),
            "p95": _percentile(walls, 95),
            "mean": float(mean(walls)) if walls else None,
        },
        "records": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("V1 multi-hop probe summary")
    print("=" * 72)
    print(f"  n={n}")
    print(f"  grounded            : {grounded}/{n}")
    print(f"  grounded + kw-hit   : {correct}/{n}    <- end-to-end correct")
    print(f"  silence_gate        : {silence_gate}/{n}")
    print(f"  silence_drift       : {silence_drift}/{n}")
    lat = summary["latency_seconds"]
    print(f"  latency             : p50={lat['p50']:.2f}s  p95={lat['p95']:.2f}s  mean={lat['mean']:.2f}s")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
