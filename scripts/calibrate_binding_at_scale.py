"""Calibrate Hebbian binding success at scale.

Loads an embeddings .npy matrix, runs ``run_binding_trials`` for one or
more values of m, and reports pass/fail against the roadmap targets
(>= 92% success, <= 0.10 false-fires/trial at m=8).

Usage:
    python scripts/calibrate_binding_at_scale.py \\
        --embeddings results/bank_embeddings_sample.npy \\
        --m 4 --m 6 --m 8 \\
        --trials 200 --distractors 5000 \\
        --out-json reports/binding_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.concept_cells.binding_calibration import (  # noqa: E402
    render_calibration_markdown,
    run_binding_trials,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--m", action="append", type=int, default=None,
                        help="Repeat for each binding cardinality (default: 4 6 8).")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--distractors", type=int, default=2000)
    parser.add_argument("--theta-write", type=float, default=0.30)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.embeddings.exists():
        parser.error(f"embeddings not found: {args.embeddings}")
    emb = np.load(args.embeddings)
    if emb.ndim != 2:
        parser.error(f"embeddings must be 2D; got shape {emb.shape}")
    print(f"[binding-cal] loaded {emb.shape[0]:,} x {emb.shape[1]} embeddings")

    ms = args.m or [4, 6, 8]
    all_passed = True
    out_blocks_json: list[dict] = []
    out_blocks_md: list[str] = []
    for m in ms:
        report = run_binding_trials(
            embeddings=emb,
            m=m,
            n_trials=args.trials,
            distractor_sample_size=args.distractors,
            theta_write=args.theta_write,
            safety_margin=args.safety_margin,
            seed=args.seed + m,
        )
        md = render_calibration_markdown(report)
        print(md)
        out_blocks_md.append(md)
        out_blocks_json.append(report.to_dict())
        if not report.passed:
            all_passed = False

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps({"runs": out_blocks_json, "all_passed": all_passed},
                       indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n---\n".join(out_blocks_md), encoding="utf-8")

    return 0 if all_passed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
