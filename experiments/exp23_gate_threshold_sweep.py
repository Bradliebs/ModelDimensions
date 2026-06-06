"""Gate threshold sweep — recall vs precision across the full system.

Why this exists:
    exp22 v2 left the decomposer producing 6/8 clean sub-questions but
    only 3/8 ground end-to-end. The five silenced cases are at
    pass-2 margins +0.000 .. +0.026, all below the default gate of
    +0.030. The bottleneck has migrated from query formulation to
    retrieval recall. Before relaxing the gate we need to measure the
    precision cost on the validated probe set.

Design (single-pipeline-load, post-hoc sweep):
    Run the pipeline ONCE at margin=0.0 (gate disabled). For every
    query we capture (observed margin, verifier verdict, answer text).
    For each candidate threshold T we then derive what the outcome
    WOULD have been:
        observed_margin < T  ->  silence_gate
        observed_margin >= T ->  whatever the verifier said
    This is exact: lowering the threshold is monotonic in what the
    gate accepts; everything downstream (generation, verifier) is
    threshold-independent.

Surveys covered:
    1. exp18 known questions (12)            recall: want grounded
    2. paragraph-probe unknowns (10)         precision: want silence
    3. paragraph-probe noise (10)            precision: want silence
    4. exp20 multi-hop probe (8) via decomposer
                                              recall AND drift safety

Outputs:
    results/v1_gate_threshold_sweep.json
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

from experiments.exp17_v1_pipeline_eval import (  # noqa: E402
    SOURCE_QUERIES as PARAGRAPH_PROBES,
    DEFAULT_BANK,
    _classify,
    _percentile,
)
from experiments.exp18_v1_pipeline_questions_eval import (  # noqa: E402
    KNOWN_QUESTIONS,
)
from experiments.exp20_multihop_probe import (  # noqa: E402
    MULTIHOP_QUERIES,
    _check_keywords,
)
from experiments.exp22_decomposer_multihop import (  # noqa: E402
    _build_decompose_prompt,
    _clean_subquestion,
)


DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_gate_threshold_sweep.json"

THRESHOLDS = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]


def _capture_simple(pipeline, query: str) -> dict:
    """Run a single-pass query at margin=0 and capture what we need to
    post-hoc derive any-threshold outcome."""
    t0 = time.time()
    result = pipeline.ask(query)
    wall = time.time() - t0
    d = result.as_dict()
    outcome_at_zero = _classify(d)  # grounded or silence_drift only
    return {
        "query": query,
        "wall_seconds": wall,
        "observed_margin": float(d["gate"]["margin"]),
        "outcome_at_zero": outcome_at_zero,
        "verify_grounded": (d["verification"] or {}).get("grounded"),
        "verify_coverage": (d["verification"] or {}).get("coverage"),
        "answer": d["answer"],
        "n_citations": len(d["citations"]),
    }


def _capture_multihop(pipeline, decompose_tok, item: dict, n_pass: int = 3,
                      snip: int = 240) -> dict:
    """Run the exp22 decomposer flow at margin=0 for a multi-hop query."""
    q = item["query"]
    keywords = item["expected_keywords"]

    # Pass 1: encode + retrieve only.
    raw = pipeline.encoder.encode_one(q, is_query=True)
    whitened = pipeline.bank.whiten(raw)
    topk1 = pipeline.bank.topk(whitened, k=pipeline.top_k)
    feedback_ids = [int(c) for c in topk1["cell_ids"][:n_pass]]
    try:
        feedback_texts = pipeline.bank.fetch_source_texts(feedback_ids)
    except Exception:
        feedback_texts = [None] * len(feedback_ids)

    # Decompose.
    cells_for_prompt = [
        {"cell_id": cid, "text": txt or ""}
        for cid, txt in zip(feedback_ids, feedback_texts)
    ]
    decompose_prompt = _build_decompose_prompt(decompose_tok, q, cells_for_prompt)
    try:
        raw_subq = pipeline._generate(decompose_prompt)
    except Exception:
        raw_subq = ""
    sub_question = _clean_subquestion(raw_subq, q)

    # Pass 2.
    t0 = time.time()
    result = pipeline.ask(sub_question)
    wall = time.time() - t0
    d = result.as_dict()
    outcome_at_zero = _classify(d)
    kw_hit = _check_keywords(d["answer"], keywords)

    return {
        "query": q,
        "sub_question": sub_question,
        "expected_keywords": keywords,
        "wall_seconds": wall,
        "observed_margin": float(d["gate"]["margin"]),
        "outcome_at_zero": outcome_at_zero,
        "verify_grounded": (d["verification"] or {}).get("grounded"),
        "answer": d["answer"],
        "kw_hit": kw_hit,
    }


def _outcome_at_threshold(rec: dict, T: float) -> str:
    """Derive the outcome the gate would produce at threshold T from a
    margin=0 capture."""
    if rec["observed_margin"] < T:
        return "silence_gate"
    return rec["outcome_at_zero"]


def _sweep_simple(records: list[dict], expected: str, name: str) -> dict:
    """For a class with expected = 'grounded' or 'silence', return per-T
    counts and accuracy."""
    n = len(records)
    by_T: list[dict] = []
    for T in THRESHOLDS:
        outcomes = [_outcome_at_threshold(r, T) for r in records]
        grounded = sum(1 for o in outcomes if o == "grounded")
        silence_gate = sum(1 for o in outcomes if o == "silence_gate")
        silence_drift = sum(1 for o in outcomes if o == "silence_drift")
        if expected == "grounded":
            correct = grounded
        else:
            correct = silence_gate + silence_drift
        by_T.append({
            "T": T,
            "grounded": grounded,
            "silence_gate": silence_gate,
            "silence_drift": silence_drift,
            "correct": correct,
            "accuracy": correct / n if n else None,
        })
    return {"name": name, "expected": expected, "n": n, "by_threshold": by_T}


def _sweep_multihop(records: list[dict], name: str) -> dict:
    """Multi-hop: track grounded+kw_hit (correct) and grounded+!kw_hit
    (confidently wrong / drift past verifier)."""
    n = len(records)
    by_T: list[dict] = []
    for T in THRESHOLDS:
        outcomes = [_outcome_at_threshold(r, T) for r in records]
        kw_hits = [r["kw_hit"] for r in records]
        grounded = sum(1 for o in outcomes if o == "grounded")
        correct = sum(1 for o, k in zip(outcomes, kw_hits)
                      if o == "grounded" and k)
        confidently_wrong = sum(1 for o, k in zip(outcomes, kw_hits)
                                if o == "grounded" and not k)
        silence_gate = sum(1 for o in outcomes if o == "silence_gate")
        silence_drift = sum(1 for o in outcomes if o == "silence_drift")
        by_T.append({
            "T": T,
            "grounded": grounded,
            "grounded_kw_hit": correct,
            "grounded_kw_miss": confidently_wrong,
            "silence_gate": silence_gate,
            "silence_drift": silence_drift,
        })
    return {"name": name, "n": n, "by_threshold": by_T}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    print("[exp23] loading pipeline (margin=0)...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline, PHI3_MODEL
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=0.0,
        use_4bit=not args.no_4bit,
    )
    print(f"[exp23] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    from transformers import AutoTokenizer
    decompose_tok = AutoTokenizer.from_pretrained(PHI3_MODEL)

    paragraph_data = json.loads(PARAGRAPH_PROBES.read_text(encoding="utf-8"))
    unknowns = [e["query"] for e in paragraph_data["unknown"]]
    noise = [e["query"] for e in paragraph_data["noise"]]

    print("\n[exp23] survey 1/4: known questions ({} q)".format(len(KNOWN_QUESTIONS)), flush=True)
    known_recs = []
    for i, q in enumerate(KNOWN_QUESTIONS, 1):
        rec = _capture_simple(pipeline, q)
        known_recs.append(rec)
        print(f"  [{i:>2}/{len(KNOWN_QUESTIONS)}] m={rec['observed_margin']:+.3f}  "
              f"{rec['outcome_at_zero']:<14}  {q[:80]}", flush=True)

    print("\n[exp23] survey 2/4: unknown queries ({} q)".format(len(unknowns)), flush=True)
    unknown_recs = []
    for i, q in enumerate(unknowns, 1):
        rec = _capture_simple(pipeline, q)
        unknown_recs.append(rec)
        print(f"  [{i:>2}/{len(unknowns)}] m={rec['observed_margin']:+.3f}  "
              f"{rec['outcome_at_zero']:<14}  {q[:80]}", flush=True)

    print("\n[exp23] survey 3/4: noise queries ({} q)".format(len(noise)), flush=True)
    noise_recs = []
    for i, q in enumerate(noise, 1):
        rec = _capture_simple(pipeline, q)
        noise_recs.append(rec)
        print(f"  [{i:>2}/{len(noise)}] m={rec['observed_margin']:+.3f}  "
              f"{rec['outcome_at_zero']:<14}  {q[:80]}", flush=True)

    print("\n[exp23] survey 4/4: multi-hop via decomposer ({} q)".format(
        len(MULTIHOP_QUERIES)), flush=True)
    multihop_recs = []
    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        rec = _capture_multihop(pipeline, decompose_tok, item)
        multihop_recs.append(rec)
        tag = "OK" if (rec["outcome_at_zero"] == "grounded" and rec["kw_hit"]) else "MISS"
        print(f"  [{i}/{len(MULTIHOP_QUERIES)}] [{tag}] m={rec['observed_margin']:+.3f}  "
              f"{rec['outcome_at_zero']:<14}  kw={rec['kw_hit']}  "
              f"SQ={rec['sub_question'][:70]}", flush=True)

    pipeline.close()

    sweep = {
        "known": _sweep_simple(known_recs, "grounded", "exp18 known questions"),
        "unknown": _sweep_simple(unknown_recs, "silence", "paragraph-probe unknown"),
        "noise": _sweep_simple(noise_recs, "silence", "paragraph-probe noise"),
        "multihop": _sweep_multihop(multihop_recs, "exp20 multi-hop via decomposer"),
    }

    summary = {
        "produced_by": "experiments/exp23_gate_threshold_sweep.py",
        "bank_path": str(bank_path),
        "top_k": int(args.top_k),
        "thresholds": THRESHOLDS,
        "default_threshold": 0.030,
        "sweep": sweep,
        "captures": {
            "known": known_recs,
            "unknown": unknown_recs,
            "noise": noise_recs,
            "multihop": multihop_recs,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print(f"{'T':>6} | {'known/12':>8} | {'unk/10':>7} | {'noi/10':>7} | "
          f"{'mh-OK/8':>7} | {'mh-WRONG':>8}")
    print("-" * 78)
    n_known = sweep["known"]["n"]
    n_unk = sweep["unknown"]["n"]
    n_noi = sweep["noise"]["n"]
    n_mh = sweep["multihop"]["n"]
    for i, T in enumerate(THRESHOLDS):
        k = sweep["known"]["by_threshold"][i]["correct"]
        u = sweep["unknown"]["by_threshold"][i]["correct"]
        nz = sweep["noise"]["by_threshold"][i]["correct"]
        mh = sweep["multihop"]["by_threshold"][i]["grounded_kw_hit"]
        mh_w = sweep["multihop"]["by_threshold"][i]["grounded_kw_miss"]
        marker = "  <- DEFAULT" if abs(T - 0.030) < 1e-9 else ""
        print(f"{T:>+6.3f} | {k:>4}/{n_known:<3} | {u:>3}/{n_unk:<3} | "
              f"{nz:>3}/{n_noi:<3} | {mh:>3}/{n_mh:<3} | {mh_w:>3}/{n_mh:<3}{marker}")
    print()
    print("Reading the table:")
    print("  known/12  : higher is better (recall on legitimate questions)")
    print("  unk/10    : higher is better (correct silences on out-of-bank queries)")
    print("  noi/10    : higher is better (correct silences on noise)")
    print("  mh-OK/8   : higher is better (multi-hop end-to-end correct)")
    print("  mh-WRONG/8: lower is better  (multi-hop confidently-wrong / verifier miss)")
    print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
