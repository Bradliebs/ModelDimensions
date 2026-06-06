"""exp26: live verification of the direct-evidence rescue gate.

Runs the production pipeline (T = silence-gate default, rescue ON) over the
exp23 four-survey corpus and records each query's actual decision plus
its rescue audit. Produces a single JSON artefact and prints an at-a-glance
table comparing rescue ON vs the prior baseline.

This is the live counterpart to exp25 (offline floor sweep). exp23 is no
longer a valid post-hoc verifier with rescue wired in, because rescue
fires only when the gate fails -- and exp23 captured at margin=0 (gate
never fails). Hence a dedicated single-threshold harness here.

Surveys (identical to exp23):
    1. exp18 known questions             (12) -> expect grounded
    2. paragraph-probe unknown queries   (10) -> expect silence
    3. paragraph-probe noise queries     (10) -> expect silence
    4. exp20 multi-hop probe via decomposer (8) -> expect grounded + kw_hit

Output: results/v1_rescue_verification.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp17_v1_pipeline_eval import (  # noqa: E402
    SOURCE_QUERIES as PARAGRAPH_PROBES,
    DEFAULT_BANK,
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


DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_rescue_verification.json"


def _classify(d: dict) -> str:
    """Classify a PipelineResult.as_dict() into one of:

        grounded_normal    -- silence=False, no rescue audit
        grounded_via_rescue -- silence=False, rescue audit present
        silence_gate       -- silence=True, gate-only path
        silence_rescue_reject -- silence=True, rescue audit present (rescue evaluated and rejected)
        silence_drift      -- silence=True, verifier rejected after gate passed
    """
    if not d["silence"]:
        return "grounded_via_rescue" if d.get("rescue") else "grounded_normal"
    reason = d.get("silence_reason", "") or ""
    if reason.startswith("verify"):
        return "silence_drift"
    if d.get("rescue"):
        return "silence_rescue_reject"
    return "silence_gate"


def _capture_simple(pipeline, query: str) -> dict:
    t0 = time.time()
    result = pipeline.ask(query)
    wall = time.time() - t0
    d = result.as_dict()
    return {
        "query": query,
        "wall_seconds": wall,
        "observed_margin": float(d["gate"]["margin"]),
        "decision": _classify(d),
        "silence": bool(d["silence"]),
        "silence_reason": d.get("silence_reason"),
        "answer": d["answer"],
        "n_citations": len(d["citations"]),
        "rescue": d.get("rescue"),
        "verify_grounded": (d["verification"] or {}).get("grounded"),
        "verify_coverage": (d["verification"] or {}).get("coverage"),
    }


def _capture_multihop(pipeline, decompose_tok, item: dict, n_pass: int = 3) -> dict:
    """Replicates exp23's pass-1 retrieve + decompose + pass-2 ask flow."""
    q = item["query"]
    keywords = item["expected_keywords"]

    raw = pipeline.encoder.encode_one(q, is_query=True)
    whitened = pipeline.bank.whiten(raw)
    topk1 = pipeline.bank.topk(whitened, k=pipeline.top_k)
    feedback_ids = [int(c) for c in topk1["cell_ids"][:n_pass]]
    try:
        feedback_texts = pipeline.bank.fetch_source_texts(feedback_ids)
    except Exception:
        feedback_texts = [None] * len(feedback_ids)

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

    t0 = time.time()
    result = pipeline.ask(sub_question)
    wall = time.time() - t0
    d = result.as_dict()
    decision = _classify(d)
    grounded = decision in ("grounded_normal", "grounded_via_rescue")
    kw_hit = _check_keywords(d["answer"], keywords) if grounded else False

    return {
        "query": q,
        "sub_question": sub_question,
        "expected_keywords": keywords,
        "wall_seconds": wall,
        "observed_margin": float(d["gate"]["margin"]),
        "decision": decision,
        "silence": bool(d["silence"]),
        "silence_reason": d.get("silence_reason"),
        "answer": d["answer"],
        "rescue": d.get("rescue"),
        "kw_hit": bool(kw_hit),
    }


def _agg_simple(records: list[dict], expected: str) -> dict:
    """expected in {'grounded', 'silence'}."""
    n = len(records)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    grounded = (
        counts.get("grounded_normal", 0)
        + counts.get("grounded_via_rescue", 0)
    )
    silence = (
        counts.get("silence_gate", 0)
        + counts.get("silence_drift", 0)
        + counts.get("silence_rescue_reject", 0)
    )
    if expected == "grounded":
        correct = grounded
    else:
        correct = silence
    return {
        "n": n,
        "counts": counts,
        "correct": correct,
        "rescued": counts.get("grounded_via_rescue", 0),
        "rescue_rejected": counts.get("silence_rescue_reject", 0),
    }


def _agg_multihop(records: list[dict]) -> dict:
    n = len(records)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    grounded = sum(
        1 for r in records
        if r["decision"] in ("grounded_normal", "grounded_via_rescue")
    )
    grounded_kw_hit = sum(
        1 for r in records
        if r["decision"] in ("grounded_normal", "grounded_via_rescue")
        and r["kw_hit"]
    )
    grounded_kw_miss = grounded - grounded_kw_hit  # confidently wrong
    rescued_kw_hit = sum(
        1 for r in records
        if r["decision"] == "grounded_via_rescue" and r["kw_hit"]
    )
    rescued_kw_miss = sum(
        1 for r in records
        if r["decision"] == "grounded_via_rescue" and not r["kw_hit"]
    )
    return {
        "n": n,
        "counts": counts,
        "grounded": grounded,
        "grounded_kw_hit": grounded_kw_hit,
        "grounded_kw_miss": grounded_kw_miss,
        "rescued_kw_hit": rescued_kw_hit,
        "rescued_kw_miss": rescued_kw_miss,
    }


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

    print("[exp26] loading pipeline at production threshold (rescue ON)...",
          flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import (
        AnswerPipeline,
        PHI3_MODEL,
        RESCUE_RANK_WINDOW,
        RESCUE_ACTIVATION_FLOOR,
    )
    # margin_threshold=None -> uses v1_silence_gate.DEFAULT_MARGIN_THRESHOLD (0.015)
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        use_4bit=not args.no_4bit,
    )
    print(f"[exp26] pipeline ready in {time.time() - t0:.1f}s "
          f"(rw={RESCUE_RANK_WINDOW}, floor={RESCUE_ACTIVATION_FLOOR:.2f})",
          flush=True)

    from transformers import AutoTokenizer
    decompose_tok = AutoTokenizer.from_pretrained(PHI3_MODEL)

    paragraph_data = json.loads(PARAGRAPH_PROBES.read_text(encoding="utf-8"))
    unknowns = [e["query"] for e in paragraph_data["unknown"]]
    noise = [e["query"] for e in paragraph_data["noise"]]

    def _fmt_decision(rec: dict) -> str:
        d = rec["decision"]
        if d == "grounded_via_rescue":
            return "GROUNDED*"  # rescue fired
        if d == "grounded_normal":
            return "GROUNDED "
        if d == "silence_rescue_reject":
            reason = (rec.get("rescue") or {}).get("rescue_rejection_reason", "?")
            return f"SIL[rscX:{reason[:14]}]"
        if d == "silence_drift":
            return "SIL[drift]"
        return "SIL[gate]"

    print("\n[exp26] survey 1/4: known questions ({}q)".format(
        len(KNOWN_QUESTIONS)), flush=True)
    known_recs = []
    for i, q in enumerate(KNOWN_QUESTIONS, 1):
        rec = _capture_simple(pipeline, q)
        known_recs.append(rec)
        print(f"  [{i:>2}/{len(KNOWN_QUESTIONS)}] m={rec['observed_margin']:+.3f}  "
              f"{_fmt_decision(rec):<22}  {q[:70]}", flush=True)

    print("\n[exp26] survey 2/4: unknown queries ({}q)".format(len(unknowns)), flush=True)
    unknown_recs = []
    for i, q in enumerate(unknowns, 1):
        rec = _capture_simple(pipeline, q)
        unknown_recs.append(rec)
        print(f"  [{i:>2}/{len(unknowns)}] m={rec['observed_margin']:+.3f}  "
              f"{_fmt_decision(rec):<22}  {q[:70]}", flush=True)

    print("\n[exp26] survey 3/4: noise queries ({}q)".format(len(noise)), flush=True)
    noise_recs = []
    for i, q in enumerate(noise, 1):
        rec = _capture_simple(pipeline, q)
        noise_recs.append(rec)
        print(f"  [{i:>2}/{len(noise)}] m={rec['observed_margin']:+.3f}  "
              f"{_fmt_decision(rec):<22}  {q[:70]}", flush=True)

    print("\n[exp26] survey 4/4: multi-hop via decomposer ({}q)".format(
        len(MULTIHOP_QUERIES)), flush=True)
    multihop_recs = []
    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        rec = _capture_multihop(pipeline, decompose_tok, item)
        multihop_recs.append(rec)
        tag = "OK" if rec["decision"] in ("grounded_normal", "grounded_via_rescue") and rec["kw_hit"] else "MISS"
        print(f"  [{i}/{len(MULTIHOP_QUERIES)}] [{tag}] m={rec['observed_margin']:+.3f}  "
              f"{_fmt_decision(rec):<22}  kw={rec['kw_hit']}  "
              f"SQ={rec['sub_question'][:60]}", flush=True)

    pipeline.close()

    summary = {
        "produced_by": "experiments/exp26_rescue_verification.py",
        "bank_path": str(bank_path),
        "top_k": int(args.top_k),
        "rescue": {
            "enabled": True,
            "rank_window": RESCUE_RANK_WINDOW,
            "activation_floor": RESCUE_ACTIVATION_FLOOR,
        },
        "agg": {
            "known": _agg_simple(known_recs, "grounded"),
            "unknown": _agg_simple(unknown_recs, "silence"),
            "noise": _agg_simple(noise_recs, "silence"),
            "multihop": _agg_multihop(multihop_recs),
        },
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
    print("=" * 80)
    print(f"RESCUE rw={RESCUE_RANK_WINDOW}  floor={RESCUE_ACTIVATION_FLOOR:.2f}")
    print("-" * 80)

    a = summary["agg"]
    print(f"  known       : correct {a['known']['correct']}/{a['known']['n']}  "
          f"rescued={a['known']['rescued']}  "
          f"rescue_rejected={a['known']['rescue_rejected']}")
    print(f"  unknown     : correct {a['unknown']['correct']}/{a['unknown']['n']}  "
          f"rescued={a['unknown']['rescued']}  "
          f"rescue_rejected={a['unknown']['rescue_rejected']}")
    print(f"  noise       : correct {a['noise']['correct']}/{a['noise']['n']}  "
          f"rescued={a['noise']['rescued']}  "
          f"rescue_rejected={a['noise']['rescue_rejected']}")
    print(f"  multihop OK : {a['multihop']['grounded_kw_hit']}/{a['multihop']['n']}  "
          f"WRONG={a['multihop']['grounded_kw_miss']}/{a['multihop']['n']}  "
          f"rescued_kw_hit={a['multihop']['rescued_kw_hit']}  "
          f"rescued_kw_miss={a['multihop']['rescued_kw_miss']}")
    print()
    print("Counts (known):", a["known"]["counts"])
    print("Counts (unknown):", a["unknown"]["counts"])
    print("Counts (noise):", a["noise"]["counts"])
    print("Counts (multihop):", a["multihop"]["counts"])
    print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
