"""CLI for the Chinchilla scaling plan.

This is the deliverable for the roadmap's first command:

    "Identify a high-quality, 8-billion-token dataset that can be split to
    ensure zero overlap with our existing 1.8M-cell Wikipedia bank, and
    outline the compute requirements for 400,000 iterations using 8-bit
    Adam."

Usage:
    python scripts/plan_chinchilla_scaling.py
    python scripts/plan_chinchilla_scaling.py --out-json reports/scaling_plan.json
    python scripts/plan_chinchilla_scaling.py --out-md reports/scaling_plan.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retro.diagnostics.compute_plan import (  # noqa: E402
    build_training_plan,
    render_plan_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-params", type=int, default=404_000_000)
    parser.add_argument("--iters", type=int, default=400_000)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--unique-tokens-target", type=int, default=8_000_000_000)
    parser.add_argument(
        "--device",
        action="append",
        default=None,
        help="Repeat to add multiple devices (default: a100_80gb_bf16 + h100_80gb_bf16).",
    )
    parser.add_argument(
        "--hourly-cost",
        action="append",
        default=None,
        help="device=usd_per_hour; repeat for each device.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    devices = tuple(args.device or ("a100_80gb_bf16", "h100_80gb_bf16"))
    costs: dict[str, float] | None = None
    if args.hourly_cost:
        costs = {}
        for entry in args.hourly_cost:
            key, _, value = entry.partition("=")
            if not value:
                parser.error(f"--hourly-cost must be device=usd; got {entry!r}")
            costs[key] = float(value)

    plan = build_training_plan(
        n_params=args.n_params,
        iters=args.iters,
        effective_batch=args.effective_batch,
        block_size=args.block_size,
        unique_tokens_target=args.unique_tokens_target,
        devices=devices,
        hourly_costs_usd=costs,
    )

    md = render_plan_markdown(plan)
    print(md)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
