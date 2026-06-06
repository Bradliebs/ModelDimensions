"""Phi-3 query decomposition for multi-hop retrieval (option B).

Hypothesis (from exp21 post-fix):
    Naive regex entity-append produced 2/8 grounded+correct on the
    multi-hop probe but at high cost (one confidently-wrong answer
    on Q2 because pass-1 retrieved a misleading cell, and 6/8 still
    silenced because the appended noun bag did not reformulate the
    question). The right move on a multi-hop query is to ask the
    SECOND question — not to add nouns to the FIRST one.

Design:
    Pass 1 (cheap): encode + retrieve top-k, no generation.
    Decompose:     Phi-3 reads (original question, top-3 cell texts)
                   and writes a single follow-up sub-question whose
                   answer is the user's answer. ("What language is
                   spoken on Saint Helena?" rather than "Where was
                   Napoleon exiled after Waterloo?")
    Pass 2:        pipeline.ask(sub_question) — gate + generate +
                   verify run normally on a focused query.

If the decomposer returns the original question unchanged (because
hop-1 already covered it) the system degrades to the single-pass
behaviour, no harm done. If the decomposer hallucinates an entity
not in the cells, Phi-3 typically still retrieves something close,
and the verifier catches confabulation.

Outputs:
    results/v1_pipeline_decomposer_multihop_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp20_multihop_probe import (  # noqa: E402
    MULTIHOP_QUERIES,
    _classify,
    _check_keywords,
    _percentile,
)


DEFAULT_BANK = os.environ.get("MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_pipeline_decomposer_multihop_eval.json"


DECOMPOSE_SYSTEM = (
    "You help a retrieval system answer multi-hop questions. "
    "You receive (1) a USER QUESTION and (2) PASSAGES the system retrieved on "
    "the first pass. The passages usually contain a FIRST FACT (e.g. who, "
    "where, what) but not the FINAL answer to the question. "
    "Your job: output exactly ONE follow-up sub-question whose ANSWER IS THE "
    "SAME as the answer to the USER QUESTION. The sub-question must keep the "
    "same WH-word and the same direction as the user question. "
    "Use the FIRST FACT from the passages to substitute a concrete entity for "
    "the descriptive phrase in the user question (e.g. replace 'the country "
    "whose flag has a red maple leaf' with 'Canada'). "
    "Match the GRAIN of the user question: if the user asks about a country, "
    "the sub-question must name a country (not a province, state, or city). "
    "Rules: output ONLY the sub-question, no preamble, no explanation, no "
    "citation, max 20 words. If the passages already directly answer the "
    "user question, output the original question unchanged."
)

DECOMPOSE_EXAMPLES = (
    "Example 1 (entity substitution)\n"
    "USER QUESTION: What language is spoken in the country where the Eiffel "
    "Tower is located?\n"
    "PASSAGES: [1] The Eiffel Tower is a wrought-iron tower in Paris, France.\n"
    "SUB-QUESTION: What language is spoken in France?\n"
    "\n"
    "Example 2 (entity substitution)\n"
    "USER QUESTION: Who is the spouse of the actor who played James Bond in "
    "Goldfinger?\n"
    "PASSAGES: [1] Sean Connery starred as James Bond in Goldfinger (1964).\n"
    "SUB-QUESTION: Who is the spouse of Sean Connery?\n"
    "\n"
    "Example 3 (succession direction: keep the user's direction; do NOT "
    "invert)\n"
    "USER QUESTION: Who succeeded the U.S. president who served during the "
    "Civil War?\n"
    "PASSAGES: [1] Abraham Lincoln was the 16th president and led the Union "
    "during the American Civil War.\n"
    "SUB-QUESTION: Who succeeded Abraham Lincoln as U.S. president?\n"
    "\n"
    "Example 4 (country grain: do NOT replace 'country' with a province)\n"
    "USER QUESTION: What is the capital of the country where the Sydney "
    "Opera House is located?\n"
    "PASSAGES: [1] The Sydney Opera House is in Sydney, New South Wales, "
    "Australia.\n"
    "SUB-QUESTION: What is the capital of Australia?\n"
)


def _build_decompose_prompt(tokenizer, question: str, cells: list[dict]) -> str:
    passages = "\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(cells) if c.get("text")
    )
    user = (
        f"{DECOMPOSE_EXAMPLES}\n"
        f"USER QUESTION: {question}\n"
        f"PASSAGES:\n{passages}\n"
        f"SUB-QUESTION:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": DECOMPOSE_SYSTEM},
         {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


_SUBQ_CLEANUP_RE = re.compile(r"^\s*(SUB-?QUESTION\s*:?\s*)+", re.IGNORECASE)


def _clean_subquestion(raw: str, original: str) -> str:
    """Phi-3 sometimes prefixes its answer with 'SUB-QUESTION:' or adds
    explanation after the first line. Take the first non-empty line and
    strip any leading SUB-QUESTION label."""
    if not raw:
        return original
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _SUBQ_CLEANUP_RE.sub("", line).strip()
        if line:
            return line
    return original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--n-passages-for-decompose", type=int, default=3)
    parser.add_argument("--top-cells-snippet-len", type=int, default=240)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    print("[exp22] loading pipeline...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline, PHI3_MODEL
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=args.margin,
        use_4bit=not args.no_4bit,
    )
    print(f"[exp22] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    # Load a fresh tokenizer for the decomposer prompt. Pipeline keeps
    # its tokenizer in a closure; reloading is ~1s and avoids changing
    # production code for an experiment.
    from transformers import AutoTokenizer
    decompose_tok = AutoTokenizer.from_pretrained(PHI3_MODEL)

    records: list[dict] = []
    snip = int(args.top_cells_snippet_len)
    n_pass = int(args.n_passages_for_decompose)

    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        q = item["query"]
        keywords = item["expected_keywords"]

        # --- Pass 1: encode + retrieve only ---
        t_p1 = time.time()
        raw = pipeline.encoder.encode_one(q, is_query=True)
        whitened = pipeline.bank.whiten(raw)
        topk1 = pipeline.bank.topk(whitened, k=pipeline.top_k)
        ids1 = [int(c) for c in topk1["cell_ids"]]
        acts1 = [float(a) for a in topk1["activations"]]
        feedback_ids = ids1[:n_pass]
        try:
            feedback_texts = pipeline.bank.fetch_source_texts(feedback_ids)
        except Exception:
            feedback_texts = [None] * len(feedback_ids)
        pass1_seconds = time.time() - t_p1

        # --- Decompose ---
        t_d = time.time()
        cells_for_prompt = [
            {"cell_id": cid, "text": txt or ""}
            for cid, txt in zip(feedback_ids, feedback_texts)
        ]
        decompose_prompt = _build_decompose_prompt(decompose_tok, q, cells_for_prompt)
        try:
            raw_subq = pipeline._generate(decompose_prompt)
        except Exception as exc:  # noqa: BLE001
            raw_subq = ""
            print(f"  [{i}] decomposer error: {exc}", flush=True)
        sub_question = _clean_subquestion(raw_subq, q)
        decompose_seconds = time.time() - t_d

        # --- Pass 2 ---
        t_p2 = time.time()
        result = pipeline.ask(sub_question)
        pass2_seconds = time.time() - t_p2
        d = result.as_dict()
        outcome = _classify(d)

        top_ids2 = d["retrieval"]["top_k_cell_ids"][:3]
        top_acts2 = d["retrieval"]["activations"][:3]
        try:
            top_texts2 = pipeline.bank.fetch_source_texts(top_ids2)
        except Exception:
            top_texts2 = [None] * len(top_ids2)
        top_cells_pass2 = [
            {
                "cell_id": int(cid),
                "activation": float(act),
                "text_snippet": (t or "")[:snip] + ("..." if t and len(t) > snip else ""),
            }
            for cid, act, t in zip(top_ids2, top_acts2, top_texts2)
        ]

        kw_hit = _check_keywords(d["answer"], keywords)
        same_as_original = sub_question.strip().lower() == q.strip().lower()

        rec = {
            "n": i,
            "query": q,
            "expected_keywords": keywords,
            "hops": item["hops"],
            "outcome": outcome,
            "answer": d["answer"],
            "kw_hit": kw_hit,
            "pass1": {
                "top_ids": ids1[:5],
                "top_activations": acts1[:5],
                "feedback_text_snippets": [
                    ((t or "")[:snip] + ("..." if t and len(t) > snip else ""))
                    for t in feedback_texts
                ],
                "wall_seconds": pass1_seconds,
            },
            "decompose": {
                "raw_output": raw_subq,
                "sub_question": sub_question,
                "same_as_original": same_as_original,
                "wall_seconds": decompose_seconds,
            },
            "pass2": {
                "gate_margin": d["gate"]["margin"],
                "gate_threshold": d["gate"]["threshold"],
                "verifier_coverage": (d["verification"] or {}).get("coverage"),
                "citations": d["citations"][:5],
                "top_cells": top_cells_pass2,
                "wall_seconds": pass2_seconds,
            },
            "silence_reason": d.get("silence_reason", ""),
            "total_wall_seconds": pass1_seconds + decompose_seconds + pass2_seconds,
        }
        records.append(rec)

        tag = "OK" if (outcome == "grounded" and kw_hit) else "MISS"
        print(
            f"  [{i}/{len(MULTIHOP_QUERIES)}] [{tag}] {outcome:<14} "
            f"kw={kw_hit}  m={d['gate']['margin']:+.3f}  "
            f"p1={pass1_seconds:.2f}s d={decompose_seconds:.2f}s p2={pass2_seconds:.2f}s",
            flush=True,
        )
        print(f"        Q : {q[:100]}", flush=True)
        print(f"        SQ: {sub_question[:100]}", flush=True)
        if outcome == "grounded":
            print(f"        A : {d['answer'][:160]}", flush=True)

    pipeline.close()

    n = len(records)
    grounded = sum(1 for r in records if r["outcome"] == "grounded")
    correct = sum(1 for r in records if r["outcome"] == "grounded" and r["kw_hit"])
    silence_gate = sum(1 for r in records if r["outcome"] == "silence_gate")
    silence_drift = sum(1 for r in records if r["outcome"] == "silence_drift")
    walls = [r["total_wall_seconds"] for r in records]
    same = sum(1 for r in records if r["decompose"]["same_as_original"])

    summary = {
        "produced_by": "experiments/exp22_decomposer_multihop.py",
        "bank_path": str(bank_path),
        "top_k": int(args.top_k),
        "margin_threshold": float(args.margin),
        "n_passages_for_decompose": n_pass,
        "n_queries": n,
        "by_outcome": {
            "grounded": grounded,
            "grounded_with_kw_hit": correct,
            "silence_gate": silence_gate,
            "silence_drift": silence_drift,
        },
        "decomposer": {
            "subq_same_as_original": same,
        },
        "latency_seconds_total": {
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
    print("V1 decomposer multi-hop summary")
    print("=" * 72)
    print(f"  n={n}")
    print(f"  grounded            : {grounded}/{n}   (exp20 baseline: 0/{n};  exp21: 3/{n})")
    print(f"  grounded + kw-hit   : {correct}/{n}   (exp20: 0/{n};  exp21: 2/{n})  <- end-to-end correct")
    print(f"  silence_gate        : {silence_gate}/{n}")
    print(f"  silence_drift       : {silence_drift}/{n}")
    print(f"  subq == original    : {same}/{n}")
    lat = summary["latency_seconds_total"]
    print(f"  total latency       : p50={lat['p50']:.2f}s  p95={lat['p95']:.2f}s  mean={lat['mean']:.2f}s")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
