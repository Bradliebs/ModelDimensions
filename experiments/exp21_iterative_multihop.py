"""Iterative two-pass retrieval prototype for multi-hop questions.

Hypothesis (from exp20):
    7/8 multi-hop failures were retrieval misses on the second hop:
    the encoder finds the hop-1 cell ("Hemingway", "Saint Helena",
    "Canada") but the system never queries the bank for hop-2
    ("Hemingway birthplace", "Saint Helena language", "Canada highest
    mountain"). The single pass cannot find paragraphs about an
    entity it has not yet identified.

Design:
    Pass 1 — encode original question, fetch top-3 cells.
    Entity extraction — regex proper-noun runs from those cell texts,
        filtered against the question's own tokens (so we surface
        *new* entities the system just discovered, not echoes of the
        query). No LLM call.
    Pass 2 — call pipeline.ask(question + " " + entities). Gate +
        generate + verify run as normal against pass-2 retrieval.

This is deliberately additive: it does not modify AnswerPipeline.
If the prototype works, the right next step is to fold it in behind
an opt-in flag.

Outputs:
    results/v1_pipeline_iterative_multihop_eval.json
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
DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_pipeline_iterative_multihop_eval.json"


# Words that look like proper nouns when sentence-initial but should not
# leak into expanded queries.
_STOPLIKE_CAPS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "They",
    "He", "She", "His", "Her", "Their", "Its", "There", "Here", "What",
    "When", "Where", "Who", "Why", "How", "Is", "Are", "Was", "Were",
    "Be", "Been", "Being", "Has", "Have", "Had", "Do", "Does", "Did",
    "Plot", "Awards", "Government", "Education", "Anything", "Released",
    "Although", "Because", "However", "Beginning",
}

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][A-Za-z]+(?:[\s.\-][A-Z][A-Za-z]+)*)\b")


def _extract_entities(
    cell_texts: list[str | None],
    question: str,
    max_entities: int = 6,
) -> list[str]:
    """Pull capitalised noun phrases from cell_texts, dedup, drop
    anything already present in the question (case-insensitive)."""
    qlow = question.lower()
    seen: set[str] = set()
    out: list[str] = []

    for text in cell_texts:
        if not text:
            continue
        for match in _PROPER_NOUN_RE.findall(text):
            phrase = match.strip(" .-")
            if not phrase:
                continue
            # Filter single-token sentence-initial caps that are not really entities.
            if " " not in phrase and "-" not in phrase and "." not in phrase:
                if phrase in _STOPLIKE_CAPS:
                    continue
            # Skip if the phrase (or any token in it) already appears
            # verbatim in the user's question.
            if phrase.lower() in qlow:
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(phrase)
            if len(out) >= max_entities:
                return out
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--top-cells-snippet-len", type=int, default=240)
    parser.add_argument("--n-feedback-cells", type=int, default=3)
    parser.add_argument("--max-entities", type=int, default=6)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    print("[exp21] loading pipeline...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=args.margin,
        use_4bit=not args.no_4bit,
    )
    print(f"[exp21] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    records: list[dict] = []
    snip = int(args.top_cells_snippet_len)

    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        q = item["query"]
        keywords = item["expected_keywords"]

        # --- Pass 1: cheap retrieval-only call to fetch hop-1 cells. ---
        # We bypass pipeline.ask to avoid running Phi-3 on a query that
        # we know is going to be silenced; instead we use the same
        # encode + topk path the pipeline does internally.
        t_p1 = time.time()
        raw = pipeline.encoder.encode_one(q, is_query=True)
        whitened = pipeline.bank.whiten(raw)
        topk1 = pipeline.bank.topk(whitened, k=pipeline.top_k)
        ids1 = [int(c) for c in topk1["cell_ids"]]
        acts1 = [float(a) for a in topk1["activations"]]
        feedback_ids = ids1[: args.n_feedback_cells]
        try:
            feedback_texts = pipeline.bank.fetch_source_texts(feedback_ids)
        except Exception:
            feedback_texts = [None] * len(feedback_ids)
        entities = _extract_entities(feedback_texts, q, max_entities=args.max_entities)
        pass1_seconds = time.time() - t_p1

        # --- Pass 2: full pipeline on expanded query. ---
        expanded_query = (q + " " + " ".join(entities)).strip()
        t_p2 = time.time()
        result = pipeline.ask(expanded_query)
        pass2_seconds = time.time() - t_p2
        d = result.as_dict()
        outcome = _classify(d)

        # Pull top-3 cells of pass 2 for inspection.
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

        rec = {
            "n": i,
            "query": q,
            "expanded_query": expanded_query,
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
                "extracted_entities": entities,
                "wall_seconds": pass1_seconds,
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
            "total_wall_seconds": pass1_seconds + pass2_seconds,
        }
        records.append(rec)

        tag = "OK" if (outcome == "grounded" and kw_hit) else "MISS"
        ent_preview = ", ".join(entities[:3]) if entities else "(none)"
        print(
            f"  [{i}/{len(MULTIHOP_QUERIES)}] [{tag}] {outcome:<14} "
            f"kw={kw_hit}  m={d['gate']['margin']:+.3f}  "
            f"p1={pass1_seconds:.2f}s p2={pass2_seconds:.2f}s  "
            f"ents=[{ent_preview}]",
            flush=True,
        )
        print(f"        Q: {q[:90]}", flush=True)
        if outcome == "grounded":
            print(f"        A: {d['answer'][:160]}", flush=True)

    pipeline.close()

    n = len(records)
    grounded = sum(1 for r in records if r["outcome"] == "grounded")
    correct = sum(1 for r in records if r["outcome"] == "grounded" and r["kw_hit"])
    silence_gate = sum(1 for r in records if r["outcome"] == "silence_gate")
    silence_drift = sum(1 for r in records if r["outcome"] == "silence_drift")
    walls = [r["total_wall_seconds"] for r in records]

    summary = {
        "produced_by": "experiments/exp21_iterative_multihop.py",
        "bank_path": str(bank_path),
        "top_k": int(args.top_k),
        "margin_threshold": float(args.margin),
        "n_feedback_cells": int(args.n_feedback_cells),
        "max_entities": int(args.max_entities),
        "n_queries": n,
        "by_outcome": {
            "grounded": grounded,
            "grounded_with_kw_hit": correct,
            "silence_gate": silence_gate,
            "silence_drift": silence_drift,
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
    print("V1 iterative-retrieval multi-hop summary (vs exp20 baseline)")
    print("=" * 72)
    print(f"  n={n}")
    print(f"  grounded            : {grounded}/{n}    (exp20: 0/{n})")
    print(f"  grounded + kw-hit   : {correct}/{n}    <- end-to-end correct")
    print(f"  silence_gate        : {silence_gate}/{n}")
    print(f"  silence_drift       : {silence_drift}/{n}")
    lat = summary["latency_seconds_total"]
    print(f"  total latency       : p50={lat['p50']:.2f}s  p95={lat['p95']:.2f}s  mean={lat['mean']:.2f}s")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
