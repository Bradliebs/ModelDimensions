#!/usr/bin/env python
"""Build the v2.1 M365 / Coding assistant pack from its manifest.

Reads ``packs/m365_coding_assistant/pack.yaml``, validates the declared policy
block (local-only sources, allowed knowledge types, no medical/legal sources),
builds the pack into a registry, and — when the eval file is present — runs the
retrieval/provenance eval and reports the pass rate honestly.

This script changes no frozen builder semantics: it calls the existing
``pack_builder.build_pack`` and ``pack_evaluator.evaluate_pack_file``. The
policy validation is an additional guard, not a relaxation of any rule.

Usage:
    python scripts/build_m365_coding_pack.py
    python scripts/build_m365_coding_pack.py --pack-root packs/built --no-eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402  (PyYAML; manifest is YAML)

from agent.pack_builder import (  # noqa: E402
    PackBuildPlan,
    build_pack,
    open_pack_service,
)
from agent.pack_evaluator import evaluate_pack_file  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402

MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
EVAL_FILE = ROOT / "evals" / "m365_coding_pack_eval.jsonl"
_BLOCKED_DOMAINS = {"medical", "legal"}


def _validate_policy(plan: PackBuildPlan, raw: dict) -> list[str]:
    """Check the manifest's sources against its declared policy block.

    Returns a list of human-readable violations (empty means conformant).
    """
    problems: list[str] = []
    policy = raw.get("policy", {}) or {}

    if policy.get("source_policy") == "local_only" and plan.allow_url_ingestion:
        problems.append(
            "policy.source_policy is local_only but allow_url_ingestion is true")

    allowed = set(policy.get("allowed_knowledge_types") or [])
    for spec in plan.sources:
        if spec.domain in _BLOCKED_DOMAINS:
            problems.append(
                f"source {spec.source_name!r} has blocked domain "
                f"{spec.domain!r} (medical/legal is disabled for this pack)")
        if allowed and spec.domain not in allowed:
            problems.append(
                f"source {spec.source_name!r} domain {spec.domain!r} is not in "
                f"the allowed knowledge types {sorted(allowed)}")
        if spec.is_url or spec.is_hf:
            problems.append(
                f"source {spec.source_name!r} is not a local file "
                "(local_only policy)")
    return problems


def _absolutise(plan: PackBuildPlan) -> None:
    """Resolve each source path relative to the repo root, in place."""
    for spec in plan.sources:
        p = Path(spec.path_or_url)
        if not p.is_absolute():
            spec.path_or_url = str((ROOT / spec.path_or_url).resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(MANIFEST),
        help="path to the pack manifest YAML")
    parser.add_argument(
        "--pack-root", default=str(ROOT / "packs" / "built"),
        help="registry root the pack is built into")
    parser.add_argument(
        "--eval-file", default=str(EVAL_FILE),
        help="JSONL eval questions (retrieval/provenance)")
    parser.add_argument(
        "--no-eval", action="store_true",
        help="skip the eval gate (build only)")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    plan = PackBuildPlan.from_file(manifest_path)

    violations = _validate_policy(plan, raw)
    if violations:
        print("Pack policy validation FAILED:")
        for v in violations:
            print(f"  - {v}")
        return 2
    print(f"Policy OK: {len(plan.sources)} local source(s), "
          "no medical/legal, local_only.")

    _absolutise(plan)

    registry = PackRegistry(args.pack_root)
    report = build_pack(plan, registry)

    print(f"\nBuilt pack {report.pack_name!r} ({report.pack_id})")
    print(f"  sources imported: {report.source_count}")
    print(f"  total chunks:     {report.total_chunks}")
    for src in report.sources:
        print(f"    - {src['source_name']} [{src['domain']}/"
              f"{src['staleness_policy']}] {src['chunks']} chunk(s)")
    if report.skipped:
        print("  skipped:")
        for sk in report.skipped:
            print(f"    - {sk['source_name']}: {sk['reason']}")

    if report.source_count != len(plan.sources):
        print("\nBuild FAILED: not every declared source was imported.")
        return 3

    if args.no_eval:
        print("\nEval skipped (--no-eval).")
        return 0

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        print(f"\nNo eval file at {eval_path}; skipping eval.")
        return 0

    service = open_pack_service(registry, plan.pack_name)
    results = evaluate_pack_file(service, eval_path)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nEval: {passed}/{total} passed "
          f"({(100.0 * passed / total) if total else 0:.1f}%).")
    for r in results:
        if not r.passed:
            print(f"  FAIL {r.question_id}: {r.reason}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
