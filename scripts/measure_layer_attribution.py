"""Compute per-CCA-layer attribution from suppressed-loss measurements.

Input JSON shape:
    {
      "loss_without": 4.000,
      "loss_real_full": 3.800,
      "loss_real_with_layer_suppressed": {"1": 3.81, "3": 3.83, "5": 3.93, "7": 3.85}
    }

Usage:
    python scripts/measure_layer_attribution.py \\
        --losses results/layer_attribution_losses.json \\
        --out-json reports/layer_attribution.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retro.diagnostics.layer_attribution import (  # noqa: E402
    compute_layer_attribution,
    render_attribution_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--losses", type=Path, required=True)
    parser.add_argument("--dominant-share-target", type=float, default=0.50)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.losses.exists():
        parser.error(f"losses not found: {args.losses}")
    payload = json.loads(args.losses.read_text(encoding="utf-8"))

    try:
        loss_without = float(payload["loss_without"])
        loss_real_full = float(payload["loss_real_full"])
        suppressed_raw = payload["loss_real_with_layer_suppressed"]
    except KeyError as exc:
        parser.error(f"losses JSON missing required key: {exc}")

    suppressed = {int(k): float(v) for k, v in suppressed_raw.items()}
    report = compute_layer_attribution(
        loss_without=loss_without,
        loss_real_full=loss_real_full,
        loss_real_with_layer_suppressed=suppressed,
        dominant_share_target=args.dominant_share_target,
    )
    md = render_attribution_markdown(report)
    print(md)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    return 0 if report.dominant_layer_passes else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
