"""Verify ZCA whitening restores effective dim on real bank embeddings.

Usage:
    python scripts/verify_zca_isotropy.py \\
        --fit-embeddings results/bank_fit_emb.npy \\
        --eval-embeddings results/bank_eval_emb.npy \\
        --out-json reports/zca_verification.json
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

from src.concept_cells.zca_verification import (  # noqa: E402
    render_verification_markdown,
    verify_whitening,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-embeddings", type=Path, required=True)
    parser.add_argument("--eval-embeddings", type=Path, required=True)
    parser.add_argument("--effective-dim-target", type=float, default=0.95)
    parser.add_argument("--pairwise-cos-max", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    for p in (args.fit_embeddings, args.eval_embeddings):
        if not p.exists():
            parser.error(f"file not found: {p}")
    fit = np.load(args.fit_embeddings)
    ev = np.load(args.eval_embeddings)
    print(f"[zca] fit {fit.shape}, eval {ev.shape}")

    report = verify_whitening(
        fit_embeddings=fit,
        eval_embeddings=ev,
        effective_dim_target=args.effective_dim_target,
        pairwise_cos_max=args.pairwise_cos_max,
        seed=args.seed,
    )
    md = render_verification_markdown(report)
    print(md)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    return 0 if report.passed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
