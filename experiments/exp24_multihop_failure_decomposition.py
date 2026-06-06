"""Step 3 diagnostic — where does the multi-hop chain actually break?

Context:
    exp23 at T=0.015 lands 4/8 multi-hop correct, 0/8 confidently-wrong.
    The four misses (Q2 French Connection, Q4 Canada capital, Q5 Charles III
    succession, Q6 Mt. Logan) all silenced with pass-2 gate margins in the
    0.000-0.014 band, i.e. below the current default. Before adopting any
    new mechanism (anchor-aware retrieval, typed handover, KB extraction),
    decompose each miss into the stage that actually failed so we know
    whether the next investment should sit at the gate, retrieval, the
    decomposer, the generator, or the verifier.

What this captures, per multi-hop query:
    - decomposer sub-question (the hop-1 -> hop-2 handover)
    - pass-2 top-K retrieval: cell ids, activations, keyword-presence
    - pass-2 stretched top-50 retrieval: rank of first keyword-bearing
      cell, so we can distinguish "right cell at rank 11" (raise K) from
      "right cell not in top-50" (retrieval fault)
    - gate margin and whether it would fire at T=0.010 / 0.015
    - the generated answer at margin=0 (gate disabled) and whether the
      expected keyword appears
    - verifier verdict and reason

The script then classifies each query into one fault category:

    DECOMPOSER     sub-question wrong (would prevent finding the answer
                   even with perfect retrieval)
    RETRIEVAL      right sub-question, but no keyword-bearing cell in
                   top-50
    GATE           right sub-question, keyword cell in top-K, margin
                   below the operating threshold -> silenced
    GENERATOR      keyword cell in top-K, gate would fire, but Phi-3
                   produced the wrong answer
    VERIFIER       Phi-3 produced the right answer but the verifier
                   silenced it (false-positive verify)
    OK             end-to-end correct at T=0.015

Outputs:
    results/v1_multihop_failure_decomposition.json
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

from experiments.exp20_multihop_probe import (  # noqa: E402
    MULTIHOP_QUERIES,
    _check_keywords,
)
from experiments.exp22_decomposer_multihop import (  # noqa: E402
    _build_decompose_prompt,
    _clean_subquestion,
)


DEFAULT_BANK = os.environ.get("MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_multihop_failure_decomposition.json"

# Operating thresholds we want to evaluate the gate against. The current
# default is 0.015 (post-exp23). 0.010 is the next candidate floor.
T_OPERATING = 0.015
T_AGGRESSIVE = 0.010

# How deep to look for keyword-bearing cells. K=10 is production; K=50
# tells us whether raising K alone would help.
STRETCH_K = 50


def _keyword_rank(texts: list[str | None], keywords: list[str]) -> int | None:
    """Return the 1-based rank of the first cell whose text contains any
    of the expected keywords (case-insensitive); None if no cell does."""
    lowered = [k.lower() for k in keywords]
    for i, t in enumerate(texts, 1):
        if not t:
            continue
        tl = t.lower()
        if any(k in tl for k in lowered):
            return i
    return None


def _classify_failure(rec: dict) -> str:
    """Apply the fault-category decision tree to a captured record.

    The order matters: each stage assumes the previous stages passed.
    """
    # If the run actually succeeded at the operating threshold, mark OK.
    if rec["final_outcome_at_T015"] == "OK":
        return "OK"

    # Decomposer: did the sub-question even contain the resolved entity
    # or relation we needed? Cheap heuristic: if the sub-question is
    # identical to the original multi-hop question (cleanup fallback
    # triggered), the decomposer effectively did nothing.
    if rec["sub_question"].strip().lower() == rec["query"].strip().lower():
        return "DECOMPOSER"

    # Retrieval: keyword nowhere in top-50.
    if rec["keyword_rank_in_top_50"] is None:
        return "RETRIEVAL"

    # Gate: keyword in top-K but margin below operating threshold.
    keyword_in_top_k = (
        rec["keyword_rank_in_top_k"] is not None
        and rec["keyword_rank_in_top_k"] <= rec["top_k_used"]
    )
    if keyword_in_top_k and not rec["gate_would_fire_at_T015"]:
        return "GATE"

    # If keyword only present beyond top-K, the gate was never the
    # blocker. Call this RETRIEVAL too (need higher K or better rerank).
    if not keyword_in_top_k:
        return "RETRIEVAL"

    # From here: keyword in top-K, gate would fire. Now look at the
    # answer at margin=0 (gate already disabled in capture).
    if not rec["kw_hit_in_answer"]:
        return "GENERATOR"

    # Answer contains the keyword, but verifier silenced.
    if rec["verify_grounded"] is False:
        return "VERIFIER"

    # Catch-all: shouldn't reach here if the tree is exhaustive.
    return "UNCLASSIFIED"


def _capture(pipeline, decompose_tok, item: dict, n_pass: int,
             snip: int) -> dict:
    q = item["query"]
    keywords = item["expected_keywords"]

    # --- Pass 1: encode + retrieve only ---
    raw = pipeline.encoder.encode_one(q, is_query=True)
    whitened = pipeline.bank.whiten(raw)
    topk1 = pipeline.bank.topk(whitened, k=pipeline.top_k)
    feedback_ids = [int(c) for c in topk1["cell_ids"][:n_pass]]
    try:
        feedback_texts = pipeline.bank.fetch_source_texts(feedback_ids)
    except Exception:  # noqa: BLE001
        feedback_texts = [None] * len(feedback_ids)

    # --- Decompose ---
    cells_for_prompt = [
        {"cell_id": cid, "text": txt or ""}
        for cid, txt in zip(feedback_ids, feedback_texts)
    ]
    decompose_prompt = _build_decompose_prompt(decompose_tok, q, cells_for_prompt)
    try:
        raw_subq = pipeline._generate(decompose_prompt)
    except Exception:  # noqa: BLE001
        raw_subq = ""
    sub_question = _clean_subquestion(raw_subq, q)

    # --- Pass 2 stretched retrieval (read-only, for diagnosis) ---
    raw2 = pipeline.encoder.encode_one(sub_question, is_query=True)
    whitened2 = pipeline.bank.whiten(raw2)
    topk_stretch = pipeline.bank.topk(whitened2, k=STRETCH_K)
    stretch_ids = [int(c) for c in topk_stretch["cell_ids"]]
    try:
        stretch_texts = pipeline.bank.fetch_source_texts(stretch_ids)
    except Exception:  # noqa: BLE001
        stretch_texts = [None] * len(stretch_ids)
    kw_rank_50 = _keyword_rank(stretch_texts, keywords)

    # --- Pass 2 actual ask (margin=0, so the gate cannot silence) ---
    result = pipeline.ask(sub_question)
    d = result.as_dict()

    top_k_ids = d["retrieval"]["top_k_cell_ids"]
    top_k_acts = d["retrieval"]["activations"]
    try:
        top_k_texts = pipeline.bank.fetch_source_texts(
            [int(c) for c in top_k_ids]
        )
    except Exception:  # noqa: BLE001
        top_k_texts = [None] * len(top_k_ids)
    kw_rank_k = _keyword_rank(top_k_texts, keywords)
    top_k_cells = [
        {
            "cell_id": int(cid),
            "activation": float(act),
            "has_keyword": bool(
                t and any(kw.lower() in t.lower() for kw in keywords)
            ),
            "text_snippet": (t or "")[:snip] + (
                "..." if t and len(t) > snip else ""
            ),
        }
        for cid, act, t in zip(top_k_ids, top_k_acts, top_k_texts)
    ]

    margin = float(d["gate"]["margin"])
    silence = bool(d["silence"])
    silence_reason = d.get("silence_reason", "") or ""
    answer = d["answer"] or ""
    verify = d.get("verification") or {}
    verify_grounded = verify.get("grounded")
    kw_hit = _check_keywords(answer, keywords)

    # Final outcome the user would see at the current operating gate.
    if margin < T_OPERATING:
        final_T015 = "silence_gate"
    elif silence:
        # Verifier silenced (Stage A/B/C/D).
        final_T015 = "silence_drift"
    else:
        final_T015 = "OK" if kw_hit else "wrong_grounded"

    rec = {
        "query": q,
        "sub_question": sub_question,
        "decompose_used_passages": len(cells_for_prompt),
        "expected_keywords": keywords,
        "hop_explanation": item["hops"],

        # Retrieval signals.
        "top_k_used": len(top_k_ids),
        "keyword_rank_in_top_k": kw_rank_k,
        "keyword_rank_in_top_50": kw_rank_50,

        # Gate signals.
        "observed_margin": margin,
        "gate_would_fire_at_T015": margin >= T_OPERATING,
        "gate_would_fire_at_T010": margin >= T_AGGRESSIVE,

        # Generator / verifier signals.
        "answer": answer,
        "kw_hit_in_answer": kw_hit,
        "verify_grounded": verify_grounded,
        "verify_reason": verify.get("reason"),
        "silence_at_margin_zero": silence,
        "silence_reason_at_margin_zero": silence_reason,

        # User-visible outcome at the production gate.
        "final_outcome_at_T015": final_T015,

        # Top-K cell table for inspection.
        "top_k_cells": top_k_cells,
    }
    rec["fault_category"] = _classify_failure(rec)
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--n-passages-for-decompose", type=int, default=3)
    parser.add_argument("--top-cells-snippet-len", type=int, default=240)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    print("[exp24] loading pipeline (margin=0 to capture every signal)...",
          flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline, PHI3_MODEL
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=0.0,
        use_4bit=not args.no_4bit,
    )
    print(f"[exp24] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    from transformers import AutoTokenizer
    decompose_tok = AutoTokenizer.from_pretrained(PHI3_MODEL)

    print(f"\n[exp24] decomposing {len(MULTIHOP_QUERIES)} multi-hop queries",
          flush=True)
    records: list[dict] = []
    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        t_q = time.time()
        rec = _capture(
            pipeline, decompose_tok, item,
            n_pass=int(args.n_passages_for_decompose),
            snip=int(args.top_cells_snippet_len),
        )
        rec["wall_seconds"] = time.time() - t_q
        records.append(rec)
        kw_k = rec["keyword_rank_in_top_k"]
        kw_50 = rec["keyword_rank_in_top_50"]
        print(
            f"  [{i}/{len(MULTIHOP_QUERIES)}] "
            f"m={rec['observed_margin']:+.3f}  "
            f"kw@K={kw_k!s:<4}  kw@50={kw_50!s:<4}  "
            f"out={rec['final_outcome_at_T015']:<14}  "
            f"fault={rec['fault_category']}",
            flush=True,
        )

    # --- Summary ---
    print("\n" + "=" * 78)
    print("Fault-category counts at T={:.3f}:".format(T_OPERATING))
    counts: dict[str, int] = {}
    for r in records:
        counts[r["fault_category"]] = counts.get(r["fault_category"], 0) + 1
    for cat in ("OK", "DECOMPOSER", "RETRIEVAL", "GATE",
                "GENERATOR", "VERIFIER", "UNCLASSIFIED"):
        n = counts.get(cat, 0)
        if n:
            print(f"  {cat:<14} {n}")
    print("=" * 78)
    print("Per-query table:")
    print("-" * 78)
    print(f"{'#':<3} {'fault':<13} {'margin':>7}  "
          f"{'kw@K':<5} {'kw@50':<5}  query")
    print("-" * 78)
    for i, r in enumerate(records, 1):
        print(
            f"{i:<3} {r['fault_category']:<13} "
            f"{r['observed_margin']:+.3f}  "
            f"{str(r['keyword_rank_in_top_k']):<5} "
            f"{str(r['keyword_rank_in_top_50']):<5}  "
            f"{r['query'][:60]}"
        )

    # --- Persist ---
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "produced_by": "experiments/exp24_multihop_failure_decomposition.py",
                "bank_path": str(bank_path),
                "top_k": args.top_k,
                "stretch_k": STRETCH_K,
                "operating_threshold": T_OPERATING,
                "aggressive_threshold": T_AGGRESSIVE,
                "fault_counts": counts,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
