"""exp28 — gating proof for the Phase 1 hybrid-retrieval cascade.

Compares four retrieval configurations on the 8-question multi-hop set
(MULTIHOP_QUERIES from exp20), end-to-end through the V1 answer
pipeline:

    mode 1  cosine-only baseline          (V1 production behaviour)
    mode 2  cosine + cross-encoder rerank (Track B Path B)
    mode 3  BM25 -> cosine cascade        (Track B Path C, no rerank)
    mode 4  BM25 -> cosine -> rerank      (full cascade)

For each (query, mode) pair the script captures the top-3 cells, the
keyword-cell rank, gate fire/margin, generated answer, verifier verdict,
and a verdict label in {grounded, silence, wrong}:

    grounded    pipeline did not silence AND any expected keyword
                appears in the answer.
    silence     pipeline returned silence (gate rejected / verifier
                rejected / rescue failed).
    wrong       pipeline returned a non-silence answer that does NOT
                contain any expected keyword.

PASS CRITERION (Phase 1 plan): mode 4 must achieve >= 7/8 grounded and
exactly 0/8 wrong. A single wrong answer is a hard fail -- 'wrong is the
worst failure mode' (V1 cardinal constraint).

Outputs:
    results/v1_hybrid_cascade.json
    results/v1_hybrid_cascade.md
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


DEFAULT_BANK = os.environ.get("MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db")
DEFAULT_LEXICAL = os.environ.get(
    "MD_LEXICAL_INDEX",
    str(REPO_ROOT / "results" / "v1_bank" / "bm25_index"),
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_hybrid_cascade.json"
DEFAULT_MD = REPO_ROOT / "results" / "v1_hybrid_cascade.md"


MODES = (
    # (name, use_rerank, use_hybrid, hybrid_mode_when_hybrid_on)
    ("cosine_only",             False, False, "fallback"),
    ("cosine_rerank",           True,  False, "fallback"),
    ("hybrid_cosine_always",    False, True,  "always"),
    ("hybrid_rerank_always",    True,  True,  "always"),
    ("hybrid_rerank_fallback",  True,  True,  "fallback"),
)


def _keyword_rank(texts: list[str | None], keywords: list[str]) -> int | None:
    lowered = [k.lower() for k in keywords]
    for i, t in enumerate(texts, 1):
        if not t:
            continue
        tl = t.lower()
        if any(k in tl for k in lowered):
            return i
    return None


def _verdict(answer: str, silence: bool, keywords: list[str]) -> str:
    if silence:
        return "silence"
    ans_l = (answer or "").lower()
    if any(k.lower() in ans_l for k in keywords):
        return "grounded"
    return "wrong"


def _run_mode(pipeline, query: str, expected_keywords: list[str],
              *, mode: str, use_rerank: bool, use_hybrid: bool,
              hybrid_mode: str,
              reranker, hybrid, rerank_margin: float) -> dict:
    # Toggle pipeline attributes for this mode. All four pipeline
    # branches gate on these attributes and capture nothing closure-bound
    # at construction time (verified by reading answer_pipeline.ask).
    pipeline.reranker = reranker if use_rerank else None
    pipeline.rerank_margin_threshold = rerank_margin if use_rerank else None
    pipeline._hybrid = hybrid if use_hybrid else None
    pipeline.hybrid_mode = hybrid_mode

    t0 = time.time()
    result = pipeline.ask(query)
    wall = time.time() - t0

    cell_ids = list(result.retrieval.get("top_k_cell_ids", []))
    try:
        cell_texts = pipeline.bank.fetch_source_texts(cell_ids[:10])
    except Exception:  # noqa: BLE001
        cell_texts = [None] * len(cell_ids[:10])
    kw_rank = _keyword_rank(cell_texts, expected_keywords)

    verdict = _verdict(result.answer, bool(result.silence), expected_keywords)

    top3 = []
    for i, cid in enumerate(cell_ids[:3]):
        t = cell_texts[i] if i < len(cell_texts) else None
        top3.append({
            "cell_id": int(cid),
            "text_snippet": ((t or "")[:200]
                             + ("..." if t and len(t) > 200 else "")),
        })

    return {
        "mode": mode,
        "verdict": verdict,
        "silence": bool(result.silence),
        "silence_reason": result.silence_reason,
        "answer": result.answer,
        "keyword_rank": kw_rank,
        "top3": top3,
        "gate": {
            "fire": bool(result.gate.get("fire")),
            "margin": float(result.gate.get("margin", 0.0)),
            "threshold": float(result.gate.get("threshold", 0.0)),
            "top1_activation": float(result.gate.get("top1_activation", 0.0)),
            "top2_activation": float(result.gate.get("top2_activation", 0.0)),
        },
        "retrieval_stage": result.retrieval.get("retrieval_stage"),
        "rescue": result.rescue,
        "verification": (
            None if result.verification is None
            else {
                "grounded": bool(result.verification.get("grounded")),
                "coverage": float(result.verification.get("coverage", 0.0)),
                "uncited_numerics": list(
                    result.verification.get("uncited_numerics") or []
                ),
            }
        ),
        "wall_seconds": wall,
    }


def _write_markdown(out_path: Path, payload: dict) -> None:
    lines: list[str] = []
    lines.append("# exp28 \u2014 hybrid cascade gating proof\n")
    lines.append(f"- bank: `{payload['bank_path']}`")
    lines.append(f"- lexical index: `{payload['lexical_path']}`")
    lines.append(f"- reranker: `{payload['reranker_model']}`")
    lines.append(f"- rerank margin: {payload['rerank_margin']}")
    lines.append("")
    lines.append("## Verdict counts per mode\n")
    lines.append("| Mode | grounded | silence | wrong |")
    lines.append("|------|---------:|--------:|------:|")
    for mode in ("cosine_only", "cosine_rerank",
                 "hybrid_cosine_always", "hybrid_rerank_always",
                 "hybrid_rerank_fallback"):
        c = payload["counts_per_mode"][mode]
        lines.append(
            f"| {mode} | {c.get('grounded', 0)} | "
            f"{c.get('silence', 0)} | {c.get('wrong', 0)} |"
        )
    lines.append("")
    lines.append("## Per-query verdicts\n")
    lines.append(
        "| # | Query | cosine_only | cosine_rerank | "
        "hyb_cos_alw | hyb_rer_alw | hyb_rer_fb |"
    )
    lines.append("|---|-------|:---:|:---:|:---:|:---:|:---:|")
    for i, q in enumerate(payload["queries"], 1):
        verdicts = {m["mode"]: m["verdict"] for m in q["modes"]}
        q_short = q["query"][:60].replace("|", "\\|")
        lines.append(
            f"| {i} | {q_short} | "
            f"{verdicts.get('cosine_only', '?')} | "
            f"{verdicts.get('cosine_rerank', '?')} | "
            f"{verdicts.get('hybrid_cosine_always', '?')} | "
            f"{verdicts.get('hybrid_rerank_always', '?')} | "
            f"{verdicts.get('hybrid_rerank_fallback', '?')} |"
        )
    lines.append("")
    lines.append(f"**PASS**: {payload['pass']}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--lexical-index", default=DEFAULT_LEXICAL,
                        help="Directory containing a saved LexicalIndex.")
    parser.add_argument("--reranker-model",
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument(
        "--rerank-margin", type=float, default=0.5,
        help="Margin threshold when the reranker is active. The cross-"
             "encoder operates on a different score scale than cosine "
             "activation; this value is conservative for ms-marco-"
             "MiniLM-L-6-v2 (typical relevant-doc scores are ~6-12, "
             "irrelevant ~-5 to 0).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lexical-k", type=int, default=200)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MD))
    parser.add_argument(
        "--n-passages-for-decompose", type=int, default=3,
        help="Number of pass-1 cosine cells fed to the decomposer prompt. "
             "Mirrors exp22/exp27.",
    )
    parser.add_argument(
        "--no-decompose", action="store_true",
        help="Skip decomposition; ask each mode on the raw query. Use only "
             "to reproduce the original (defective) exp28 harness.",
    )
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    lex_path = Path(args.lexical_index)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2
    if not lex_path.exists():
        print(f"ERROR: lexical index not found: {lex_path}", file=sys.stderr)
        return 2

    print(f"[exp28] loading lexical index: {lex_path}", flush=True)
    from src.agent.lexical_index import load_lexical_index
    lex = load_lexical_index(lex_path)
    print(
        f"[exp28]   n_docs={lex.n_docs} backend={lex.manifest.backend}",
        flush=True,
    )

    print(f"[exp28] loading reranker {args.reranker_model}...", flush=True)
    from src.agent.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker(model_name=args.reranker_model)
    _ = reranker.score("warmup query", ["warmup passage"])

    print(f"[exp28] loading pipeline (bank + Phi-3)...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline
    from src.agent.hybrid_retriever import HybridRetriever
    # Build with reranker + lexical_index ON so all expensive resources
    # are loaded once. Mode toggling re-binds the attributes per call.
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        use_4bit=not args.no_4bit,
        reranker=reranker,
        rerank_margin_threshold=args.rerank_margin,
        lexical_index=lex,
        lexical_k=args.lexical_k,
    )
    hybrid = HybridRetriever(lex, pipeline.bank)
    print(f"[exp28] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    decompose_tok = None
    if not args.no_decompose:
        from transformers import AutoTokenizer
        from src.agent.answer_pipeline import PHI3_MODEL
        print(f"[exp28] loading decomposer tokenizer ({PHI3_MODEL})...",
              flush=True)
        decompose_tok = AutoTokenizer.from_pretrained(PHI3_MODEL)

    print(f"\n[exp28] running {len(MULTIHOP_QUERIES)} queries x "
          f"{len(MODES)} modes = "
          f"{len(MULTIHOP_QUERIES) * len(MODES)} pipeline calls"
          f"  (decompose={'off' if args.no_decompose else 'on'})",
          flush=True)

    queries: list[dict] = []
    counts_per_mode: dict[str, dict[str, int]] = {
        m[0]: {"grounded": 0, "silence": 0, "wrong": 0} for m in MODES
    }

    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        q_record = {
            "query": item["query"],
            "expected_keywords": item["expected_keywords"],
            "hops": item["hops"],
            "modes": [],
        }

        # Decomposition: one pass-1 cosine retrieval + one Phi-3 generation
        # per query (NOT per mode). All four modes ask the pipeline on the
        # SAME sub_question so the only variable being measured is the
        # retrieval mode of the answer-bearing pass-2 call. This mirrors
        # the exp22 / exp27 flow that V1's documented 6/8 baseline
        # depends on.
        ask_query = item["query"]
        sub_question_used: str | None = None
        if decompose_tok is not None:
            # Pass-1 cosine retrieval with no rerank/hybrid to seed the
            # decomposer. Temporarily clear both attributes so this call
            # is byte-identical to the baseline V1 retrieval used in
            # exp22 / exp27.
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
                sub_question_used = _clean_subquestion(raw_subq, item["query"])
                ask_query = sub_question_used
            finally:
                pipeline.reranker = saved_rerank
                pipeline._hybrid = saved_hybrid

        q_record["sub_question"] = sub_question_used
        print(
            f"  [{i}/{len(MULTIHOP_QUERIES)}] q={item['query'][:50]!r}"
            f"  ->  sub_q={(sub_question_used or '(no-decompose)')[:60]!r}",
            flush=True,
        )

        for mode_name, use_rerank, use_hybrid, hybrid_mode in MODES:
            rec = _run_mode(
                pipeline, ask_query, item["expected_keywords"],
                mode=mode_name,
                use_rerank=use_rerank, use_hybrid=use_hybrid,
                hybrid_mode=hybrid_mode,
                reranker=reranker, hybrid=hybrid,
                rerank_margin=args.rerank_margin,
            )
            rec["asked_query"] = ask_query
            q_record["modes"].append(rec)
            counts_per_mode[mode_name][rec["verdict"]] += 1
            print(
                f"    {mode_name:<26} "
                f"verdict={rec['verdict']:<8} "
                f"kw_rank={rec['keyword_rank']!s:<4} "
                f"gate_fire={int(rec['gate']['fire'])} "
                f"({rec['wall_seconds']:.1f}s)",
                flush=True,
            )
        queries.append(q_record)

    # Pass criterion: hybrid_rerank_fallback >= 7/8 grounded AND 0/8 wrong.
    # The fallback orchestration is the production-recommended mode -- it
    # preserves V1 cosine baseline on queries cosine grounds and only
    # attempts the BM25+rerank cascade as rescue on cosine silences.
    m_target = counts_per_mode["hybrid_rerank_fallback"]
    passed = (m_target["grounded"] >= 7) and (m_target["wrong"] == 0)

    payload = {
        "produced_by": "experiments/exp28_hybrid_cascade.py",
        "bank_path": str(bank_path),
        "lexical_path": str(lex_path),
        "reranker_model": args.reranker_model,
        "rerank_margin": args.rerank_margin,
        "top_k": args.top_k,
        "lexical_k": args.lexical_k,
        "counts_per_mode": counts_per_mode,
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
    print("Verdict counts per mode:")
    for mode_name in (m[0] for m in MODES):
        c = counts_per_mode[mode_name]
        print(f"  {mode_name:<26} grounded={c['grounded']}  "
              f"silence={c['silence']}  wrong={c['wrong']}")
    print("=" * 78)
    print(f"PASS criterion (mode hybrid_rerank_fallback: "
          f">=7 grounded, 0 wrong): "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  -> HALT: do not proceed to scale-up until Phase 1 passes.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
