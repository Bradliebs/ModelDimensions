"""Experiment 13: Knowledge retrieval quality — deterministic vs semantic.

Question: v1.3 shipped imported-knowledge retrieval on the **deterministic**
encoder, which is reproducible but non-semantic — it nails exact-wording
queries and is essentially chance on paraphrases. v1.4 adds an optional
**semantic** backend (MiniLM). Does the semantic backend actually retrieve by
meaning, and does the deterministic backend stay honest as the offline default?

This experiment imports a small, controlled set of coding/project docs through
the same WorkbenchService the workbench app uses, then runs five query
categories against each available backend:

  1. exact        : verbatim chunk wording (deterministic should score high)
  2. paraphrase   : same meaning, different words (semantic should win)
  3. wrong-topic  : unrelated query, e.g. "how to bake sourdough bread"
  4. version      : version-sensitive coding query
  5. deleted      : query a topic whose source was deleted (must not be cited)

Metrics (per backend):
  recall_at_1 / recall_at_3      : over queries with a known relevant chunk,
                                    matched by source_section
  false_retrieval_rate           : over wrong-topic queries, fraction whose
                                    top-1 activation >= --relevance-threshold
  deleted_source_citation_rate   : rate a deleted source_id is cited (expect 0)
  knowledge_used_rate            : fraction of relevance queries that retrieved
  memory_pollution_rate          : ledger writes caused by knowledge queries
                                    (expect 0 — knowledge never touches memory)
  source_metadata_present_rate   : fraction of candidates carrying
                                    source_name/domain/authority + backend_name

Honest caveats this experiment is designed to surface:
  - Knowledge retrieval has NO firing gate: top-k is always returned. A high
    false_retrieval_rate on the deterministic backend at a low threshold is
    expected and is exactly why source metadata + a threshold matter.
  - The deterministic backend is non-semantic: expect strong exact recall and
    weak paraphrase recall. Retrieval is not truth.

If the semantic backend is unavailable (no model), its section is written as
``{"skipped": true, "reason": ...}`` and the deterministic run still completes.

To run:
    python -m experiments.exp13_knowledge_retrieval_quality
    python -m experiments.exp13_knowledge_retrieval_quality --relevance-threshold 0.35
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.workbench_service import WorkbenchService


# ---------- Controlled corpus ----------

# Each doc is a single coding/project source. Sections are the relevance anchors.
_DOCS: Dict[str, Dict[str, object]] = {
    "pydantic.md": {
        "source_name": "Pydantic Notes",
        "domain": "coding",
        "authority": "official",
        "version": "2.6",
        "text": (
            "# Validation\n\n"
            "Pydantic validates data against a typed model when the model is "
            "constructed and raises a ValidationError on invalid input.\n\n"
            "# Serialization\n\n"
            "Pydantic serializes a model to a dictionary with model_dump and "
            "to a JSON string with model_dump_json.\n"
        ),
    },
    "jsonl.md": {
        "source_name": "Ledger Format Notes",
        "domain": "coding",
        "authority": "reputable",
        "version": "1.0",
        "text": (
            "# JSONL\n\n"
            "JSONL stores one JSON object per line, which makes it ideal for "
            "append-only ledgers that are written incrementally.\n\n"
            "# Recovery\n\n"
            "A JSONL ledger can be replayed line by line to rebuild state after "
            "a crash because each line is independently parseable.\n"
        ),
    },
    "retention.md": {
        "source_name": "Retention Policy",
        "domain": "project_docs",
        "authority": "official",
        "version": "3.1",
        "text": (
            "# Archival\n\n"
            "Project decisions are archived after ninety days but never deleted "
            "so the audit trail stays intact.\n"
        ),
    },
}

# Relevance queries: (text, expected source_section, category).
_RELEVANCE_QUERIES = [
    # exact — verbatim wording from the chunk
    ("Pydantic validates data against a typed model when the model is "
     "constructed and raises a ValidationError on invalid input.",
     "Validation", "exact"),
    ("JSONL stores one JSON object per line, which makes it ideal for "
     "append-only ledgers that are written incrementally.",
     "JSONL", "exact"),
    # paraphrase — same meaning, different words
    ("how does pydantic check that incoming data matches the schema",
     "Validation", "paraphrase"),
    ("why is one-object-per-line good for an append log",
     "JSONL", "paraphrase"),
    ("how long before old project choices get moved to the archive",
     "Archival", "paraphrase"),
    # version-sensitive coding query
    ("which pydantic version are these validation notes for",
     "Validation", "version"),
]

# Wrong-topic queries: nothing in the corpus should genuinely match.
_WRONG_TOPIC_QUERIES = [
    "how to bake sourdough bread at home",
    "best hiking trails near the coast",
    "what is the capital of a fictional country",
]


def _import_corpus(service: WorkbenchService, work_dir: Path,
                   skip: Optional[str] = None) -> Dict[str, str]:
    """Import the controlled corpus; return {source_name: source_id}."""
    name_to_id: Dict[str, str] = {}
    for filename, meta in _DOCS.items():
        if skip is not None and filename == skip:
            continue
        path = work_dir / filename
        path.write_text(str(meta["text"]), encoding="utf-8")
        src = service.import_knowledge(
            path,
            domain=str(meta["domain"]),
            authority=str(meta["authority"]),
            source_name=str(meta["source_name"]),
            version=str(meta["version"]),
        )
        name_to_id[str(meta["source_name"])] = src.source_id
    return name_to_id


def _build_service(work_dir: Path, backend: str, tag: str) -> WorkbenchService:
    return WorkbenchService(
        ledger_path=str(work_dir / f"ledger_{tag}.jsonl"),
        queue_path=str(work_dir / f"queue_{tag}.jsonl"),
        knowledge_path=str(work_dir / f"knowledge_{tag}.jsonl"),
        knowledge_backend=backend,
        fresh=True,
    )


def _run_backend(backend: str, work_dir: Path,
                 threshold: float) -> Dict[str, object]:
    service = _build_service(work_dir, backend, tag=backend)
    active = service.knowledge_backend_name()
    if active != backend:
        return {"skipped": True,
                "reason": f"requested '{backend}' but resolved to '{active}'"}

    _import_corpus(service, work_dir)

    # --- relevance queries ---
    hits_at_1 = 0
    hits_at_3 = 0
    knowledge_used = 0
    metadata_ok = 0
    metadata_total = 0
    per_category: Dict[str, Dict[str, int]] = {}
    for text, expected_section, category in _RELEVANCE_QUERIES:
        audit = service.query_knowledge(text)
        cat = per_category.setdefault(category, {"n": 0, "r1": 0, "r3": 0})
        cat["n"] += 1
        if audit.knowledge_used:
            knowledge_used += 1
        sections = [c["section"] for c in audit.candidates]
        if sections[:1] == [expected_section]:
            hits_at_1 += 1
            cat["r1"] += 1
        if expected_section in sections[:3]:
            hits_at_3 += 1
            cat["r3"] += 1
        for cand in audit.candidates:
            metadata_total += 1
            if (cand["source_name"] and cand["domain"] and cand["authority"]
                    and cand["backend_name"]):
                metadata_ok += 1

    n_rel = len(_RELEVANCE_QUERIES)

    # --- wrong-topic queries (false retrieval at threshold) ---
    false_fires = 0
    for text in _WRONG_TOPIC_QUERIES:
        audit = service.query_knowledge(text)
        if audit.candidates and audit.candidates[0]["activation"] >= threshold:
            false_fires += 1
    n_wrong = len(_WRONG_TOPIC_QUERIES)

    # --- deleted-source query ---
    name_to_id = {row["source_name"]: row["source_id"]
                  for row in service.list_knowledge_sources()}
    deleted_id = name_to_id["Pydantic Notes"]
    service.delete_knowledge_source(deleted_id)
    deleted_audit = service.query_knowledge(
        "Pydantic validates data against a typed model")
    deleted_citation_rate = (
        1.0 if deleted_id in deleted_audit.cited_source_ids else 0.0)

    # --- memory pollution: knowledge queries must never write to the ledger ---
    memory_pollution_rate = 1.0 if service.export_ledger() else 0.0

    return {
        "skipped": False,
        "backend": active,
        "recall_at_1": round(hits_at_1 / n_rel, 4),
        "recall_at_3": round(hits_at_3 / n_rel, 4),
        "knowledge_used_rate": round(knowledge_used / n_rel, 4),
        "false_retrieval_rate": round(false_fires / n_wrong, 4),
        "relevance_threshold": threshold,
        "deleted_source_citation_rate": deleted_citation_rate,
        "memory_pollution_rate": memory_pollution_rate,
        "source_metadata_present_rate": (
            round(metadata_ok / metadata_total, 4) if metadata_total else 1.0),
        "by_category": {
            cat: {
                "recall_at_1": round(v["r1"] / v["n"], 4),
                "recall_at_3": round(v["r3"] / v["n"], 4),
            }
            for cat, v in per_category.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knowledge retrieval quality: deterministic vs semantic.")
    parser.add_argument("--relevance-threshold", type=float, default=0.35,
                        help="top-1 activation at/above which a wrong-topic "
                             "query counts as a false retrieval")
    args = parser.parse_args()

    started = time.time()
    results: Dict[str, object] = {
        "experiment": "exp13_knowledge_retrieval_quality",
        "relevance_threshold": args.relevance_threshold,
    }

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        # Deterministic backend always runs (offline default).
        results["deterministic"] = _run_backend(
            "deterministic", _mkdir(work_dir, "det"), args.relevance_threshold)

        # Semantic backend: the service falls back to deterministic if the
        # model is unavailable, in which case _run_backend reports it skipped.
        results["semantic"] = _run_backend(
            "semantic", _mkdir(work_dir, "sem"), args.relevance_threshold)

    results["elapsed_sec"] = round(time.time() - started, 3)

    out_path = ROOT / "results" / "exp13_summary.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    det = results["deterministic"]
    print(f"Deterministic: recall@1={det['recall_at_1']} "
          f"recall@3={det['recall_at_3']} "
          f"false_retrieval={det['false_retrieval_rate']} "
          f"deleted_citation={det['deleted_source_citation_rate']} "
          f"memory_pollution={det['memory_pollution_rate']}")
    sem = results["semantic"]
    if sem.get("skipped"):
        print(f"Semantic: skipped ({sem.get('reason')})")
    else:
        print(f"Semantic: recall@1={sem['recall_at_1']} "
              f"recall@3={sem['recall_at_3']} "
              f"false_retrieval={sem['false_retrieval_rate']}")
    print(f"Wrote {out_path}")


def _mkdir(parent: Path, name: str) -> Path:
    path = parent / name
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main()
