"""Step 3 follow-on (recall diagnostic) — V1 pipeline eval over a
question-shaped known set.

Why this exists:
    exp17 reused the 50 paragraph probes from
    `results/bank_selectivity_at_5p7m.json`. Diagnostic
    `experiments/exp17b_diagnose_misses.py` showed that 5 of 20 "known"
    queries are wikitext / category lists, 1 is BASIC code, and the
    remaining 14 are declarative paragraphs (not questions). The
    pipeline's prompt asks Phi-3 to "answer the question" — given a
    paragraph excerpt, Phi-3 either paraphrases unrelatedly (cov=0) or
    introduces years not in the cited cells (Stage B reject).

    This experiment runs the SAME pipeline against question-shaped
    queries derived from the same underlying paragraph probes — same
    facts, same bank content, just question-shaped — to measure recall
    fairly. Unknown + noise are reused unchanged.

Outputs:
    results/v1_pipeline_questions_eval.json
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

# Reuse the run_eval harness from exp17.
from experiments.exp17_v1_pipeline_eval import (
    run_eval,
    DEFAULT_BANK,
    SOURCE_QUERIES as PARAGRAPH_PROBES,
)


# Twelve question-shaped queries derived 1:1 from the well-formed prose
# paragraphs in results/bank_selectivity_at_5p7m.json. Each question
# targets a fact present in the source paragraph; the bank content is
# the same. Wikitext (#1, #3, #4, #14, #18), BASIC code (#8), and two
# context-free fragments (#6, #15) are dropped.
KNOWN_QUESTIONS: list[str] = [
    # from paragraph #2 (DNA / miswak)
    "What did a 2016 paper compare about DNA on miswak and toothbrushes?",
    # from paragraph #5 (Vancouver Film Critics Circle)
    "Which film won Best Canadian Film at the 2nd Vancouver Film Critics Circle Awards?",
    # from paragraph #7 (CBC / South Dublin)
    "Which book series referenced CBC in its satire of South Dublin culture?",
    # from paragraph #9 (Dover-Sherborn)
    "Which school suspended the book before other challenges were tracked by the ALA?",
    # from paragraph #10 (Serenity House)
    "What is the restored gazebo at the school now known as?",
    # from paragraph #11 (Messiah)
    "Whose descendant is the Messiah according to the Tanakh and Old Testament prophecies?",
    # from paragraph #12 (National Gallery)
    "When did he join the National Gallery staff after the war?",
    # from paragraph #13 (Gygax / TSR)
    "Who did Gygax persuade to remove Kevin Blume as president?",
    # from paragraph #16 (Achill Island)
    "What is the largest of the Irish isles?",
    # from paragraph #17 (Trump's confirmation)
    "Where was Trump confirmed in 1959?",
    # from paragraph #19 (Anakapalli)
    "In which Indian state is the Anakapalli Lok Sabha constituency?",
    # from paragraph #20 (Hong Kong equestrian)
    "Which two venues hosted the main equestrian events?",
]


def load_query_set() -> dict[str, list[dict]]:
    """Build the known-question set + reuse paragraph-probe unknown/noise."""
    paragraph_data = json.loads(PARAGRAPH_PROBES.read_text(encoding="utf-8"))
    return {
        "known": [{"query": q, "expected": "grounded"} for q in KNOWN_QUESTIONS],
        "unknown": [{"query": e["query"], "expected": "silence"}
                    for e in paragraph_data["unknown"]],
        "noise": [{"query": e["query"], "expected": "silence"}
                  for e in paragraph_data["noise"]],
    }


DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_pipeline_questions_eval.json"


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

    print(f"[exp18] loading pipeline...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=args.margin,
        use_4bit=not args.no_4bit,
    )
    print(f"[exp18] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    queries = load_query_set()
    summary = run_eval(pipeline, queries)
    pipeline.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary["bank_path"] = str(bank_path)
    summary["top_k"] = int(args.top_k)
    summary["margin_threshold"] = float(args.margin)
    summary["use_4bit"] = not args.no_4bit
    summary["produced_by"] = "experiments/exp18_v1_pipeline_questions_eval.py"
    summary["known_set"] = "12 question-shaped queries derived from paragraph probes"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("V1 pipeline questions-eval summary")
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
