#!/usr/bin/env python
"""Run the v2.1.1 Value Sprint against the M365 / Coding assistant pack.

Builds the existing pack into a registry (no frozen builder semantics change),
opens its service, runs the fixed sprint queries through the **frozen** assistant
path (template composer, offline), and writes an honest report: which queries
grounded and cited, which were refused, which flagged a stale source, and which
are pack gaps (the knowledge exists but natural-language retrieval cannot reach
it with the deterministic exact-match backend).

This is read-only instrumentation. It adds no query path and needs no SLM.

Usage:
    python scripts/run_value_sprint.py
    python scripts/run_value_sprint.py --out results/value_sprint_report.md --jsonl results/value_sprint_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.pack_builder import (  # noqa: E402
    PackBuildPlan,
    build_pack,
    open_pack_service,
)
from agent.project_packs import PackRegistry  # noqa: E402
from agent.value_sprint import (  # noqa: E402
    load_queries,
    render_markdown,
    run_value_sprint,
    summarize,
)

MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
QUERIES = ROOT / "demos" / "value_sprint_queries.jsonl"


def _absolutise(plan: PackBuildPlan) -> None:
    for spec in plan.sources:
        p = Path(spec.path_or_url)
        if not p.is_absolute():
            spec.path_or_url = str((ROOT / spec.path_or_url).resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--pack-root", default=str(ROOT / "packs" / "built"))
    parser.add_argument("--queries", default=str(QUERIES))
    parser.add_argument(
        "--out", default=str(ROOT / "results" / "value_sprint_report.md"),
        help="Markdown report output path")
    parser.add_argument(
        "--jsonl", default=None,
        help="optional JSONL output of every per-query audit row")
    args = parser.parse_args(argv)

    plan = PackBuildPlan.from_file(Path(args.manifest))
    _absolutise(plan)
    registry = PackRegistry(args.pack_root)
    report = build_pack(plan, registry)
    service = open_pack_service(registry, plan.pack_name)

    queries = load_queries(args.queries)
    rows = run_value_sprint(service, queries)
    summary = summarize(rows)
    markdown = render_markdown(rows, summary, pack_label=report.pack_name)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row.to_dict()) + "\n")

    print(f"Value Sprint on pack {report.pack_name!r} ({report.pack_id})")
    print(f"  queries          : {summary.query_count}")
    print(f"  grounded + cited : {summary.grounded_count}")
    print(f"  refused          : {summary.refused_count}")
    print(f"  honest refusals  : {summary.honest_refusal_count}")
    print(f"  false groundings : {summary.false_grounding_count}")
    print(f"  pack gaps        : {summary.pack_gap_count}")
    print(f"  stale-flagged    : {summary.stale_flagged_count}")
    print(f"  model-prior used : {summary.model_prior_count}")
    print(f"\nReport written to {out_path}")
    if args.jsonl:
        print(f"Rows written to {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
