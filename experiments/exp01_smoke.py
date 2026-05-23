"""Smoke test: run experiment 01 logic on synthetic controls only.

This skips the sentence-transformers download and the wikitext fetch. It only
exercises the UniformBallEncoder and RandomGaussianEncoder so you can verify
the measurement pipeline works in seconds, on CPU, with no network.

If this passes, the real experiment will work too (modulo model downloads).

    python -m experiments.exp01_smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from concept_cells.encoders import UniformBallEncoder, RandomGaussianEncoder
from concept_cells.geometry import (
    isotropy_report, separation_test, scale_to_unit_ball
)


def main():
    n = 500
    dim = 384
    texts = [f"item_{i}" for i in range(n)]

    print(f"[smoke] n={n}, dim={dim}\n")

    for enc_cls, name in [(UniformBallEncoder, "uniform_ball"),
                          (RandomGaussianEncoder, "gaussian")]:
        enc = enc_cls(dim=dim, seed=0)
        batch = enc.encode(texts)

        # Raw isotropy
        rpt = isotropy_report(batch.embeddings)
        print(f"[{name}] |cos|={rpt.abs_mean_pairwise_cosine:.4f}  "
              f"eff_dim={rpt.participation_ratio:.1f}/{rpt.dim}  "
              f"({rpt.effective_dim_fraction:.2%})")

        # Separation, after scaling into unit ball
        emb = scale_to_unit_ball(batch.embeddings)
        sep = separation_test(emb, epsilon=0.05)
        print(f"[{name}] selectivity={sep.perfect_selectivity_rate:.4f}  "
              f"fp_rate={sep.false_positive_rate:.4e}  "
              f"theory_bound={sep.theoretical_lower_bound:.4f}\n")

    # Sanity check: uniform_ball should be ~perfectly selective
    print("[smoke] Expected: uniform_ball selectivity ~1.0, "
          "|cos| close to 0, eff_dim close to ambient.")
    print("[smoke] If you see those, the measurement pipeline works.")


if __name__ == "__main__":
    main()
