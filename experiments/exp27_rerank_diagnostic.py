"""exp27 — measurement-only diagnostic: where does the keyword-bearing
cell land under cosine vs cross-encoder ranking on the 8-question
multi-hop set?

Why this exists:
    V1 baseline at commit a065ee1 lands 6/8 correct, 0/8 wrong on
    MULTIHOP_QUERIES (exp20). The two misses are silences: Q2 (French
    Connection -> Don Ellis) where the rescue floor 0.40 is not met by
    any of the top-3 cosine activations, and Q5 (Elizabeth II / WWII
    succession) where the keyword-bearing cell is at rank 8 by cosine
    so the gate margin top1-top2 is dominated by an off-target cell.

    Track B has three candidate paths -- cross-encoder rerank (surgical),
    hybrid BM25+MiniLM (biggest surface), or query expansion (variable).
    This experiment measures which one Q2 and Q5 actually need:

        - if cross-encoder rerank lifts the keyword cell into top-3 for
          a query, then a rerank-then-gate flow can fix it without
          touching the index.
        - if the keyword cell is not in the top-50 cosine pool at all,
          rerank cannot help; that query needs hybrid retrieval.
        - the other 6 queries (currently OK) are reported for context
          and to detect any regression risk from a future rerank wiring.

What this captures, per multi-hop query:
    - the decomposed sub-question (same flow as exp22/exp24)
    - top-50 cosine retrieval on that sub-question (cell ids, activations,
      keyword presence)
    - cross-encoder scores on the same 50 (query=sub_question, passages
      = source texts)
    - keyword rank under each ranking (1-based, None if no keyword cell
      in the 50)
    - top-3 cells under each ranking (id, score, has_keyword, snippet)
    - status flag: rerank_helps / rerank_indifferent / rerank_cant_help

Outputs:
    results/v1_rerank_diagnostic.json (+ stdout summary table)

Read-only. Does not modify the pipeline, the bank, or any production
code path. Run with the same .venv used for exp24/exp25.
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
)
from experiments.exp22_decomposer_multihop import (  # noqa: E402
    _build_decompose_prompt,
    _clean_subquestion,
)


DEFAULT_BANK = os.environ.get("MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_rerank_diagnostic.json"

STRETCH_K = 50          # candidate pool size for the rerank diagnostic
TOP_REPORT = 3          # how many top cells to surface per ranking


def _keyword_rank(texts: list[str | None], keywords: list[str]) -> int | None:
    """1-based rank of the first cell whose text contains any keyword."""
    lowered = [k.lower() for k in keywords]
    for i, t in enumerate(texts, 1):
        if not t:
            continue
        tl = t.lower()
        if any(k in tl for k in lowered):
            return i
    return None


def _has_keyword(text: str | None, keywords: list[str]) -> bool:
    if not text:
        return False
    tl = text.lower()
    return any(k.lower() in tl for k in keywords)


def _capture(pipeline, decompose_tok, reranker, item: dict,
             n_pass: int, snip: int) -> dict:
    q = item["query"]
    keywords = item["expected_keywords"]

    # --- Pass 1: encode + retrieve, no generation ---
    raw1 = pipeline.encoder.encode_one(q, is_query=True)
    whitened1 = pipeline.bank.whiten(raw1)
    topk1 = pipeline.bank.topk(whitened1, k=pipeline.top_k)
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

    # --- Pass 2 stretched cosine retrieval (top-50) ---
    raw2 = pipeline.encoder.encode_one(sub_question, is_query=True)
    whitened2 = pipeline.bank.whiten(raw2)
    topk_stretch = pipeline.bank.topk(whitened2, k=STRETCH_K)
    cosine_ids = [int(c) for c in topk_stretch["cell_ids"]]
    cosine_acts = [float(a) for a in topk_stretch["activations"]]
    try:
        cosine_texts = pipeline.bank.fetch_source_texts(cosine_ids)
    except Exception:  # noqa: BLE001
        cosine_texts = [None] * len(cosine_ids)

    # --- Cross-encoder rerank on the same 50 (query = sub_question) ---
    # Replace any None with empty string for the cross-encoder; record
    # which slots were empty so they cannot win on rerank ordering.
    rerank_passages = [t if t is not None else "" for t in cosine_texts]
    t_re = time.time()
    ce_scores = reranker.score(sub_question, rerank_passages)
    rerank_secs = time.time() - t_re

    # Sort indices by cross-encoder score, descending. Empty passages
    # would naturally score low; no mask needed.
    import numpy as np
    ce_order = np.argsort(-ce_scores).tolist()
    rerank_ids = [cosine_ids[i] for i in ce_order]
    rerank_texts = [cosine_texts[i] for i in ce_order]
    rerank_scores = [float(ce_scores[i]) for i in ce_order]

    # --- Keyword ranks under each ordering ---
    kw_rank_cosine = _keyword_rank(cosine_texts, keywords)
    kw_rank_rerank = _keyword_rank(rerank_texts, keywords)

    def _top_view(ids, scores, texts, n):
        out = []
        for i in range(min(n, len(ids))):
            t = texts[i]
            out.append({
                "cell_id": int(ids[i]),
                "score": float(scores[i]),
                "has_keyword": _has_keyword(t, keywords),
                "text_snippet": (
                    (t or "")[:snip] + ("..." if t and len(t) > snip else "")
                ),
            })
        return out

    # --- Status classification ---
    # rerank_helps:        cosine had keyword cell outside top-3 (or
    #                      missing) AND rerank places it in top-3.
    # rerank_indifferent:  both rankings put keyword cell within top-3.
    # rerank_cant_help:    keyword cell not in top-50 cosine pool at all.
    # rerank_neutral:      keyword cell in top-50 but rerank does not
    #                      lift it into top-3.
    if kw_rank_cosine is None:
        status = "rerank_cant_help"
    else:
        cos_in_top3 = kw_rank_cosine <= TOP_REPORT
        re_in_top3 = (kw_rank_rerank is not None
                      and kw_rank_rerank <= TOP_REPORT)
        if cos_in_top3 and re_in_top3:
            status = "rerank_indifferent"
        elif (not cos_in_top3) and re_in_top3:
            status = "rerank_helps"
        else:
            status = "rerank_neutral"

    return {
        "query": q,
        "sub_question": sub_question,
        "expected_keywords": keywords,
        "hop_explanation": item["hops"],
        "stretch_k": STRETCH_K,

        "kw_rank_cosine": kw_rank_cosine,
        "kw_rank_rerank": kw_rank_rerank,

        "cosine_top3": _top_view(cosine_ids, cosine_acts, cosine_texts,
                                 TOP_REPORT),
        "rerank_top3": _top_view(rerank_ids, rerank_scores, rerank_texts,
                                 TOP_REPORT),

        "rerank_seconds": rerank_secs,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--top-k", type=int, default=10,
                        help="pass-1 retrieval depth used to seed the decomposer")
    parser.add_argument("--n-passages-for-decompose", type=int, default=3)
    parser.add_argument("--reranker-model",
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--top-cells-snippet-len", type=int, default=240)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    print(f"[exp27] loading reranker {args.reranker_model}...", flush=True)
    from src.agent.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker(model_name=args.reranker_model)
    _ = reranker.score("warmup query", ["warmup passage"])

    print("[exp27] loading pipeline (margin=0; no generation needed for "
          "diagnostic, but decomposer uses Phi-3)...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline, PHI3_MODEL
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=0.0,
        use_4bit=not args.no_4bit,
    )
    print(f"[exp27] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    from transformers import AutoTokenizer
    decompose_tok = AutoTokenizer.from_pretrained(PHI3_MODEL)

    print(f"\n[exp27] running diagnostic on {len(MULTIHOP_QUERIES)} queries",
          flush=True)
    records: list[dict] = []
    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        t_q = time.time()
        rec = _capture(
            pipeline, decompose_tok, reranker, item,
            n_pass=int(args.n_passages_for_decompose),
            snip=int(args.top_cells_snippet_len),
        )
        rec["wall_seconds"] = time.time() - t_q
        records.append(rec)
        print(
            f"  [{i}/{len(MULTIHOP_QUERIES)}] "
            f"cos@kw={rec['kw_rank_cosine']!s:<5}  "
            f"re@kw={rec['kw_rank_rerank']!s:<5}  "
            f"status={rec['status']:<20}  "
            f"q={rec['query'][:60]}",
            flush=True,
        )

    # --- Summary ---
    counts: dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print("\n" + "=" * 78)
    print("Status counts:")
    for cat in ("rerank_helps", "rerank_indifferent",
                "rerank_neutral", "rerank_cant_help"):
        n = counts.get(cat, 0)
        if n:
            print(f"  {cat:<22} {n}")
    print("=" * 78)
    print("Per-query table:")
    print("-" * 78)
    print(f"{'#':<3} {'status':<22} {'cos@kw':<7} {'re@kw':<7}  query")
    print("-" * 78)
    for i, r in enumerate(records, 1):
        print(
            f"{i:<3} {r['status']:<22} "
            f"{str(r['kw_rank_cosine']):<7} "
            f"{str(r['kw_rank_rerank']):<7}  "
            f"{r['query'][:60]}"
        )

    # --- Decision banner ---
    helps = counts.get("rerank_helps", 0)
    cant = counts.get("rerank_cant_help", 0)
    print("\n" + "-" * 78)
    print(f"Decision signal: rerank lifts keyword cell into top-{TOP_REPORT} "
          f"for {helps} of {len(records)} queries; "
          f"{cant} require a fix the cross-encoder cannot provide "
          f"(keyword cell missing from top-{STRETCH_K} cosine pool).")
    if cant == 0 and helps > 0:
        print("  -> Path B (surgical cross-encoder rerank) is sufficient.")
    elif cant > 0:
        print("  -> Path C (hybrid BM25 + MiniLM) is required for "
              "rerank_cant_help queries.")
    else:
        print("  -> Neither path looks needed on this set; investigate further.")

    # --- Persist ---
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "produced_by": "experiments/exp27_rerank_diagnostic.py",
                "bank_path": str(bank_path),
                "stretch_k": STRETCH_K,
                "top_report": TOP_REPORT,
                "reranker_model": args.reranker_model,
                "status_counts": counts,
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
