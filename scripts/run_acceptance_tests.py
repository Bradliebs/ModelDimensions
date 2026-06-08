"""Run the three reading-engine acceptance gates over a losses JSON file.

Input JSON shape:
    {"none": <float>, "random": <float>, "real": <float>}

(matches the output of ``src/retro/eval_retro_heldout.py``'s ``evaluate``).

Usage:
    python scripts/run_acceptance_tests.py --losses results/heldout_losses.json
    python scripts/run_acceptance_tests.py --losses ... --out-json reports/acceptance.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retro.diagnostics.acceptance import evaluate_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--losses", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.losses.exists():
        parser.error(f"losses not found: {args.losses}")
    losses = json.loads(args.losses.read_text(encoding="utf-8"))
    report = evaluate_all(losses)

    print("# Reading-engine acceptance gates")
    print()
    for gate in report.gates.values():
        verdict = "PASS" if gate.passed else "FAIL"
        print(
            f"- [{verdict}] {gate.name}: {gate.detail} "
            f"(target {gate.direction} {gate.threshold_nats:.4f})"
        )
    print()
    print(f"Overall: {'PASS' if report.all_passed else 'FAIL'}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0 if report.all_passed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
