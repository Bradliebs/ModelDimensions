"""exp29 - Stage F (claim verifier) regression + behaviour proof.

Run the production-recommended `hybrid_rerank_fallback` retrieval mode
on the exp28 multi-hop query set, once with Stage F off (current
production) and once with Stage F on (Phase 2 candidate). Compare
per-query verdicts to certify two things:

  1. CARDINAL RULE: Stage F never turns a wrong answer into a worse
     wrong answer, and never increases the wrong-answer count.
  2. NO BASELINE REGRESSION: every query that was `grounded` under
     Stage F off must still be `grounded` under Stage F on. A
     `grounded -> silence` flip is allowed iff the answer was a
     splice the verifier rightly distrusts; any such flip is logged
     for review but DOES count as a regression under the gating
     criterion (this experiment intentionally chooses safety over
     theoretical false-positive catching when the query set has no
     known false positives at Stage E v2's exit point).

Pass criterion:
  - 0 wrong under either configuration (V1 cardinal rule).
  - 0 `grounded -> silence` flips between off and on.
  - Bonus, not gating: any historical or constructed false positive
    that Stage F newly silences is logged.

Outputs:
  results/v1_claim_verifier.json
  results/v1_claim_verifier.md

This script depends on exp28's infrastructure (decomposer prompt,
mode toggling, keyword rank, verdict labelling) and the production
bank + lexical index built in Phase 1.5.
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

from experiments.exp20_multihop_probe import MULTIHOP_QUERIES  # noqa: E402
from experiments.exp22_decomposer_multihop import (  # noqa: E402
    _build_decompose_prompt,
    _clean_subquestion,
)
from experiments.exp28_hybrid_cascade import (  # noqa: E402
    _keyword_rank,
    _verdict,
)


DEFAULT_BANK = os.environ.get("MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db")
DEFAULT_LEXICAL = os.environ.get(
    "MD_LEXICAL_INDEX",
    str(REPO_ROOT / "results" / "v1_bank" / "bm25_tantivy"),
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_claim_verifier.json"
DEFAULT_MD = REPO_ROOT / "results" / "v1_claim_verifier.md"


def _run_call(
    pipeline, query: str, keywords: list[str], *,
    enable_claim_verification: bool,
) -> dict:
    pipeline.enable_claim_verification = bool(enable_claim_verification)
    t0 = time.time()
    result = pipeline.ask(query)
    wall = time.time() - t0

    cell_ids = list(result.retrieval.get("top_k_cell_ids", []))
    try:
        cell_texts = pipeline.bank.fetch_source_texts(cell_ids[:10])
    except Exception:  # noqa: BLE001
        cell_texts = [None] * len(cell_ids[:10])
    kw_rank = _keyword_rank(cell_texts, keywords)
    verdict = _verdict(result.answer, bool(result.silence), keywords)

    cv_report = {}
    if result.verification is not None:
        cv_report = result.verification.get("claim_verifier_report") or {}

    return {
        "enable_claim_verification": bool(enable_claim_verification),
        "verdict": verdict,
        "silence": bool(result.silence),
        "silence_reason": result.silence_reason,
        "answer": result.answer,
        "keyword_rank": kw_rank,
        "claim_verifier": {
            "grounded": cv_report.get("grounded"),
            "n_claims": cv_report.get("n_claims"),
            "n_skipped": cv_report.get("n_skipped"),
            "n_verified": cv_report.get("n_verified"),
            "n_rejected": cv_report.get("n_rejected"),
            "failed": cv_report.get("failed", []),
        },
        "wall_seconds": wall,
    }


def _write_markdown(out_path: Path, payload: dict) -> None:
    lines: list[str] = [
        "# exp29 - Stage F (claim verifier) regression + behaviour proof\n",
        f"- bank: `{payload['bank_path']}`",
        f"- lexical index: `{payload['lexical_path']}`",
        f"- retrieval mode: `hybrid_rerank_fallback` (production)",
        "",
        "## Verdict counts\n",
        "| Stage F | grounded | silence | wrong |",
        "|---------|---------:|--------:|------:|",
        f"| off     | {payload['counts']['off']['grounded']} | "
        f"{payload['counts']['off']['silence']} | "
        f"{payload['counts']['off']['wrong']} |",
        f"| on      | {payload['counts']['on']['grounded']} | "
        f"{payload['counts']['on']['silence']} | "
        f"{payload['counts']['on']['wrong']} |",
        "",
        "## Per-query verdicts (off -> on)\n",
        "| # | Query | off | on | delta |",
        "|---|-------|:---:|:---:|:---:|",
    ]
    for i, q in enumerate(payload["queries"], 1):
        v_off = q["off"]["verdict"]
        v_on = q["on"]["verdict"]
        if v_off == v_on:
            delta = "="
        elif v_off == "grounded" and v_on == "silence":
            delta = "**REGRESSION**"
        elif v_off == "wrong" and v_on == "silence":
            delta = "**caught**"
        elif v_off == "silence" and v_on == "grounded":
            delta = "rescue (impossible without retrieval change)"
        else:
            delta = f"{v_off} -> {v_on}"
        q_short = q["query"][:55].replace("|", "\\|")
        lines.append(f"| {i} | {q_short} | {v_off} | {v_on} | {delta} |")
    lines.append("")
    lines.append(f"**Regressions (grounded->silence)**: "
                 f"{payload['regressions']}")
    lines.append(f"**False-positive catches (wrong->silence)**: "
                 f"{payload['false_positive_catches']}")
    lines.append("")
    lines.append(f"**PASS**: {payload['pass']}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--lexical-index", default=DEFAULT_LEXICAL)
    parser.add_argument("--reranker-model",
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-margin", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lexical-k", type=int, default=200)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MD))
    parser.add_argument("--n-passages-for-decompose", type=int, default=3)
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    lex_path = Path(args.lexical_index)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2
    if not lex_path.exists():
        print(f"ERROR: lexical index not found: {lex_path}", file=sys.stderr)
        return 2

    print(f"[exp29] loading lexical index: {lex_path}", flush=True)
    from src.agent.lexical_index import load_lexical_index
    lex = load_lexical_index(lex_path)
    print(f"[exp29]   n_docs={lex.n_docs} backend={lex.manifest.backend}",
          flush=True)

    print(f"[exp29] loading reranker {args.reranker_model}...", flush=True)
    from src.agent.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker(model_name=args.reranker_model)
    _ = reranker.score("warmup query", ["warmup passage"])

    print(f"[exp29] loading pipeline (bank + Phi-3)...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline
    from src.agent.hybrid_retriever import HybridRetriever
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        use_4bit=not args.no_4bit,
        reranker=reranker,
        rerank_margin_threshold=args.rerank_margin,
        lexical_index=lex,
        lexical_k=args.lexical_k,
        hybrid_mode="fallback",
    )
    # Production cascade is "always wired in fallback mode".
    pipeline._hybrid = HybridRetriever(lex, pipeline.bank)
    print(f"[exp29] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    from transformers import AutoTokenizer
    from src.agent.answer_pipeline import PHI3_MODEL
    print(f"[exp29] loading decomposer tokenizer ({PHI3_MODEL})...",
          flush=True)
    decompose_tok = AutoTokenizer.from_pretrained(PHI3_MODEL)

    n = len(MULTIHOP_QUERIES)
    print(f"\n[exp29] running {n} queries x 2 configs = {n * 2} calls",
          flush=True)

    queries: list[dict] = []
    counts = {
        "off": {"grounded": 0, "silence": 0, "wrong": 0},
        "on":  {"grounded": 0, "silence": 0, "wrong": 0},
    }

    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        # Decompose once per query (mirrors exp28 / exp22 flow). The
        # hybrid retriever is on in both configs, so we run decompose
        # with hybrid temporarily off and reranker off, identical to
        # exp28's pass-1 seeding.
        saved_rerank = pipeline.reranker
        saved_hybrid = pipeline._hybrid
        pipeline.reranker = None
        pipeline._hybrid = None
        try:
            raw1 = pipeline.encoder.encode_one(item["query"], is_query=True)
            whitened1 = pipeline.bank.whiten(raw1)
            topk1 = pipeline.bank.topk(whitened1, k=pipeline.top_k)
            fb_ids = [int(c) for c in topk1["cell_ids"][:int(args.n_passages_for_decompose)]]
            try:
                fb_texts = pipeline.bank.fetch_source_texts(fb_ids)
            except Exception:  # noqa: BLE001
                fb_texts = [None] * len(fb_ids)
            cells_for_prompt = [
                {"cell_id": cid, "text": txt or ""}
                for cid, txt in zip(fb_ids, fb_texts)
            ]
            decompose_prompt = _build_decompose_prompt(
                decompose_tok, item["query"], cells_for_prompt,
            )
            try:
                raw_subq = pipeline._generate(decompose_prompt)
            except Exception:  # noqa: BLE001
                raw_subq = ""
            sub_question = _clean_subquestion(raw_subq, item["query"])
        finally:
            pipeline.reranker = saved_rerank
            pipeline._hybrid = saved_hybrid

        print(f"  [{i}/{n}] q={item['query'][:50]!r} -> sub={sub_question[:60]!r}",
              flush=True)

        off_rec = _run_call(
            pipeline, sub_question, item["expected_keywords"],
            enable_claim_verification=False,
        )
        counts["off"][off_rec["verdict"]] += 1
        print(f"    stage_F=off  verdict={off_rec['verdict']:<8} "
              f"kw_rank={off_rec['keyword_rank']!s:<4} "
              f"({off_rec['wall_seconds']:.1f}s)",
              flush=True)

        on_rec = _run_call(
            pipeline, sub_question, item["expected_keywords"],
            enable_claim_verification=True,
        )
        counts["on"][on_rec["verdict"]] += 1
        cv = on_rec["claim_verifier"]
        print(f"    stage_F=on   verdict={on_rec['verdict']:<8} "
              f"kw_rank={on_rec['keyword_rank']!s:<4} "
              f"({on_rec['wall_seconds']:.1f}s)  "
              f"claims={cv.get('n_claims')} verified={cv.get('n_verified')} "
              f"rejected={cv.get('n_rejected')}",
              flush=True)

        queries.append({
            "query": item["query"],
            "sub_question": sub_question,
            "expected_keywords": item["expected_keywords"],
            "off": off_rec,
            "on": on_rec,
        })

    regressions = 0
    fp_catches = 0
    for q in queries:
        v_off = q["off"]["verdict"]
        v_on = q["on"]["verdict"]
        if v_off == "grounded" and v_on == "silence":
            regressions += 1
        if v_off == "wrong" and v_on == "silence":
            fp_catches += 1

    passed = (
        counts["off"]["wrong"] == 0
        and counts["on"]["wrong"] == 0
        and regressions == 0
    )

    payload = {
        "produced_by": "experiments/exp29_claim_verifier.py",
        "bank_path": str(bank_path),
        "lexical_path": str(lex_path),
        "reranker_model": args.reranker_model,
        "rerank_margin": args.rerank_margin,
        "top_k": args.top_k,
        "lexical_k": args.lexical_k,
        "counts": counts,
        "regressions": regressions,
        "false_positive_catches": fp_catches,
        "pass": bool(passed),
        "queries": queries,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    md = Path(args.markdown)
    md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(md, payload)
    print(f"wrote {md}")

    print("\n" + "=" * 78)
    print("Verdict counts:")
    for cfg in ("off", "on"):
        c = counts[cfg]
        print(f"  stage_F={cfg:<3} grounded={c['grounded']}  "
              f"silence={c['silence']}  wrong={c['wrong']}")
    print(f"  grounded->silence regressions: {regressions}")
    print(f"  wrong->silence catches:        {fp_catches}")
    print("=" * 78)
    print(f"PASS criterion (0 wrong both configs, 0 regressions): "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  -> HALT: Stage F regressed the baseline or introduced wrong answers.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
