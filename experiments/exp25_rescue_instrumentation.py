"""exp25: rescue instrumentation harness (Step A — measurement only).

Runs the same 4 surveys as exp23 at margin=0 and emits one JSON record per
query containing every signal needed to evaluate a future direct-evidence
rescue gate offline:

    - retrieval order (top_k_cell_ids, activations)
    - generator answer
    - margin
    - full verifier verdict (grounded, reason, coverage, stage_e_log,
      unanchored_proper_nouns, etc.)
    - per-rank cell analysis: which novel answer entities appear in this
      cell, which question anchors appear in this cell, whether this single
      cell supports the whole answer (= all novel entities + at least one
      anchor co-occur in the same cell)

No pipeline change. The rescue floor and rank gate are not applied here —
that is offline analysis (Step B).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp17_v1_pipeline_eval import (  # noqa: E402
    SOURCE_QUERIES as PARAGRAPH_PROBES,
    DEFAULT_BANK,
    _classify,
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
from src.agent.v1_answer_verifier import _normalise_for_anchor  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_rescue_instrumentation.json"


def _per_rank_analysis(
    pipeline,
    retrieval: dict,
    verification: dict | None,
) -> tuple[list[dict], str | None, list[str], bool]:
    """For each retrieved cell, record which novel entities and anchors
    appear in it. Returns (ranks, primary_entity, anchors, fallback_used).
    """
    cell_ids = [int(c) for c in retrieval.get("top_k_cell_ids", [])]
    activations = [float(a) for a in retrieval.get("activations", [])]

    log = (verification or {}).get("stage_e_log") or []
    # Novel non-skip entities from Stage E v2: original-case spans + their
    # normalised match form.
    novel_entities: list[tuple[str, str]] = []
    for rec in log:
        if rec.get("decision") == "skip":
            continue
        ent = rec.get("answer_entity")
        if not ent:
            continue
        norm = _normalise_for_anchor(ent)
        if norm:
            novel_entities.append((ent, norm))

    # Question anchors are the same across log entries for one verify call.
    anchors: list[str] = []
    fallback_used = False
    if log:
        anchors = list(log[0].get("question_anchors") or [])
        fallback_used = bool(log[0].get("fallback_used"))

    primary_entity: str | None = novel_entities[0][0] if novel_entities else None

    # Re-fetch source texts in raw retrieval order so rank indices line up
    # with the activations array.
    try:
        texts = pipeline.bank.fetch_source_texts(cell_ids)
    except Exception:
        texts = [None] * len(cell_ids)

    ranks: list[dict] = []
    for i, (cid, act, txt) in enumerate(zip(cell_ids, activations, texts)):
        norm_text = _normalise_for_anchor(txt or "")
        ents_in_cell = [orig for (orig, n) in novel_entities if n and n in norm_text]
        anchors_in_cell = [a for a in anchors if a and a in norm_text]
        supports_all = bool(novel_entities) and len(ents_in_cell) == len(novel_entities)
        contains_any_anchor = len(anchors_in_cell) > 0
        ranks.append({
            "rank": i,
            "cell_id": int(cid),
            "activation": act,
            "has_text": bool(txt),
            "novel_entities_present": ents_in_cell,
            "anchors_present": anchors_in_cell,
            "supports_all_novel_entities": supports_all,
            "contains_any_anchor": contains_any_anchor,
            "single_cell_validates_answer": supports_all and contains_any_anchor,
        })

    return ranks, primary_entity, anchors, fallback_used


def _capture(pipeline, survey: str, query: str, *, sub_question: str | None = None,
             expected_keywords: list[str] | None = None) -> dict:
    asked = sub_question if sub_question is not None else query
    t0 = time.time()
    result = pipeline.ask(asked)
    wall = time.time() - t0
    d = result.as_dict()
    ver = d.get("verification") or {}
    ranks, primary_entity, anchors, fallback_used = _per_rank_analysis(
        pipeline, d.get("retrieval") or {}, ver,
    )
    rec: dict[str, Any] = {
        "survey": survey,
        "query": query,
        "wall_seconds": wall,
        "margin": float((d.get("gate") or {}).get("margin", 0.0)),
        "outcome_at_zero": _classify(d),
        "answer": d.get("answer"),
        "silence": d.get("silence"),
        "silence_reason": d.get("silence_reason"),
        "verifier_grounded": ver.get("grounded"),
        "verifier_reason": ver.get("reason"),
        "verifier_coverage": ver.get("coverage"),
        "stage_e_log": ver.get("stage_e_log") or [],
        "unanchored_proper_nouns": ver.get("unanchored_proper_nouns") or [],
        "primary_entity": primary_entity,
        "question_anchors": anchors,
        "fallback_used": fallback_used,
        "ranks": ranks,
    }
    if sub_question is not None:
        rec["sub_question"] = sub_question
    if expected_keywords is not None:
        rec["expected_keywords"] = list(expected_keywords)
        rec["kw_hit"] = _check_keywords(d.get("answer") or "", expected_keywords)
    return rec


def _decompose(pipeline, decompose_tok, item: dict, n_pass: int = 3) -> str:
    q = item["query"]
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
    prompt = _build_decompose_prompt(decompose_tok, q, cells_for_prompt)
    try:
        raw_subq = pipeline._generate(prompt)
    except Exception:
        raw_subq = ""
    return _clean_subquestion(raw_subq, q)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-path", default=str(DEFAULT_BANK))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    print("[exp25] loading pipeline (margin=0)...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline, PHI3_MODEL
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=0.0,
        use_4bit=not args.no_4bit,
    )
    print(f"[exp25] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    from transformers import AutoTokenizer
    decompose_tok = AutoTokenizer.from_pretrained(PHI3_MODEL)

    paragraph_data = json.loads(PARAGRAPH_PROBES.read_text(encoding="utf-8"))
    unknowns = [e["query"] for e in paragraph_data["unknown"]]
    noise = [e["query"] for e in paragraph_data["noise"]]

    records: list[dict] = []

    print(f"\n[exp25] survey 1/4: known questions ({len(KNOWN_QUESTIONS)} q)",
          flush=True)
    for i, q in enumerate(KNOWN_QUESTIONS, 1):
        rec = _capture(pipeline, "known", q)
        records.append(rec)
        print(f"  [{i:>2}/{len(KNOWN_QUESTIONS)}] m=+{rec['margin']:.3f}  "
              f"vg={rec['verifier_grounded']}  {q[:60]}", flush=True)

    print(f"\n[exp25] survey 2/4: unknown queries ({len(unknowns)} q)", flush=True)
    for i, q in enumerate(unknowns, 1):
        rec = _capture(pipeline, "unknown", q)
        records.append(rec)
        print(f"  [{i:>2}/{len(unknowns)}] m=+{rec['margin']:.3f}  "
              f"vg={rec['verifier_grounded']}  {q[:60]}", flush=True)

    print(f"\n[exp25] survey 3/4: noise queries ({len(noise)} q)", flush=True)
    for i, q in enumerate(noise, 1):
        rec = _capture(pipeline, "noise", q)
        records.append(rec)
        print(f"  [{i:>2}/{len(noise)}] m=+{rec['margin']:.3f}  "
              f"vg={rec['verifier_grounded']}  {q[:60]}", flush=True)

    print(f"\n[exp25] survey 4/4: multi-hop via decomposer "
          f"({len(MULTIHOP_QUERIES)} q)", flush=True)
    for i, item in enumerate(MULTIHOP_QUERIES, 1):
        sub_q = _decompose(pipeline, decompose_tok, item)
        rec = _capture(pipeline, "multihop", item["query"],
                       sub_question=sub_q,
                       expected_keywords=item["expected_keywords"])
        records.append(rec)
        print(f"  [{i}/{len(MULTIHOP_QUERIES)}] m=+{rec['margin']:.3f}  "
              f"vg={rec['verifier_grounded']}  kw={rec['kw_hit']}  "
              f"SQ={sub_q[:60]}", flush=True)

    out = {
        "produced_by": "experiments/exp25_rescue_instrumentation.py",
        "bank_path": str(bank_path),
        "top_k": int(args.top_k),
        "margin_threshold": 0.0,
        "n_records": len(records),
        "records": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[exp25] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
