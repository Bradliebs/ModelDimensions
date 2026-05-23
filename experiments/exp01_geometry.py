"""Experiment 01: Geometry verification.

Question: Do the Tyukin/Gorban separation theorems hold on real embeddings?

For three encoders (one real, two synthetic controls) we measure:
  1. Isotropy of the embedding distribution.
  2. Whether one-shot concept cells achieve perfect selectivity.
  3. Whether ZCA whitening + ball-scaling improves things.

Outputs:
  - results/exp01_summary.txt   : numeric summary table
  - results/exp01_isotropy.png  : pairwise cosine distributions
  - results/exp01_separation.png: FP rate vs. number of memorized items

To run:
    cd concept_cells
    python -m experiments.exp01_geometry --n-items 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

# Make src importable when running from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from concept_cells.data import diverse_sample_texts
from concept_cells.encoders import (
    TextEncoder, RandomGaussianEncoder, UniformBallEncoder, EncodedBatch
)
from concept_cells.geometry import (
    isotropy_report, separation_test, zca_whiten, scale_to_unit_ball,
    SeparationResult,
)


def encode_all(texts: List[str], device: str = None) -> Dict[str, EncodedBatch]:
    """Encode the same texts with each candidate encoder."""
    out: Dict[str, EncodedBatch] = {}

    # Control: ideal case the theorems were proved for
    print("[encode] uniform-ball control...")
    out["uniform_ball"] = UniformBallEncoder(dim=384).encode(texts)

    # Control: Gaussian (close to ball in high dim)
    print("[encode] gaussian control...")
    out["gaussian"] = RandomGaussianEncoder(dim=384).encode(texts)

    # The real encoder we actually care about
    print("[encode] sentence-transformers (all-MiniLM-L6-v2)...")
    out["minilm"] = TextEncoder("all-MiniLM-L6-v2", device=device).encode(texts)

    return out


def run_separation_sweep(emb: np.ndarray, sizes: List[int],
                          epsilon: float = 0.05) -> List[SeparationResult]:
    """Test separation at multiple memory sizes."""
    results = []
    for n in sizes:
        sub = emb[:n] if n <= emb.shape[0] else emb
        r = separation_test(sub, epsilon=epsilon,
                            threshold_scheme="norm_minus_eps")
        results.append(r)
        print(f"   n={n:5d}  selectivity={r.perfect_selectivity_rate:6.3f}  "
              f"fp_rate={r.false_positive_rate:.4e}  "
              f"theory_bound={r.theoretical_lower_bound:6.3f}")
    return results


def plot_isotropy(batches: Dict[str, EncodedBatch],
                  variants: Dict[str, np.ndarray],
                  outpath: Path) -> None:
    """Pairwise cosine similarity distributions for each (encoder, variant)."""
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4),
                              sharey=True)
    if len(variants) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    for ax, (variant_name, _) in zip(axes, variants.items()):
        for enc_name, batch in batches.items():
            emb = variants[variant_name].__call__(batch.embeddings) \
                if callable(variants[variant_name]) else variants[variant_name]
            # We need per-encoder transformed embeddings - re-compute here:
            transform = variants[variant_name]
            emb_t = transform(batch.embeddings)
            # Sample pairs
            n = emb_t.shape[0]
            idx_i = rng.integers(0, n, size=20_000)
            idx_j = rng.integers(0, n, size=20_000)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
            norms = np.linalg.norm(emb_t, axis=1, keepdims=True)
            unit = emb_t / np.maximum(norms, 1e-12)
            cos = np.einsum("ij,ij->i", unit[idx_i], unit[idx_j])
            ax.hist(cos, bins=80, alpha=0.45, label=enc_name, density=True)
        ax.set_title(f"variant: {variant_name}")
        ax.set_xlabel("pairwise cosine similarity")
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="k", linestyle=":", alpha=0.4)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Pairwise cosine distribution by encoder and preprocessing",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


def plot_separation_sweep(sweeps: Dict[str, List[SeparationResult]],
                          outpath: Path) -> None:
    """Selectivity vs number of memorized items, per encoder/variant."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in sweeps.items():
        sizes = [r.n_items for r in results]
        sel = [r.perfect_selectivity_rate for r in results]
        ax.plot(sizes, sel, marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("number of memorized items (M)")
    ax.set_ylabel("perfect selectivity rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Concept-cell selectivity vs memory size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=2000,
                        help="Number of texts to embed")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu / mps (auto-detected if None)")
    parser.add_argument("--results-dir", type=str,
                        default=str(ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp01] Loading {args.n_items} diverse texts...")
    texts = diverse_sample_texts(n=args.n_items)
    print(f"[exp01] Got {len(texts)} texts. Sample: {texts[0][:80]!r}")

    t0 = time.time()
    batches = encode_all(texts, device=args.device)
    print(f"[exp01] All encoded in {time.time() - t0:.1f}s")

    # Variants: (1) raw, (2) ball-scaled, (3) ZCA whitened + ball-scaled
    variants = {
        "raw":              lambda x: x,
        "ball_scaled":      lambda x: scale_to_unit_ball(x),
        "whitened_scaled":  lambda x: scale_to_unit_ball(zca_whiten(x)),
    }

    # --- Isotropy report ---
    summary: Dict = {"isotropy": {}, "separation": {}}
    print("\n[exp01] === ISOTROPY ===")
    for enc_name, batch in batches.items():
        summary["isotropy"][enc_name] = {}
        for v_name, transform in variants.items():
            rpt = isotropy_report(transform(batch.embeddings))
            summary["isotropy"][enc_name][v_name] = asdict(rpt)
            print(f" {enc_name:14s} | {v_name:16s}  "
                  f"|cos|={rpt.abs_mean_pairwise_cosine:.4f}  "
                  f"eff_dim={rpt.participation_ratio:6.1f}/{rpt.dim}  "
                  f"({rpt.effective_dim_fraction:.2%})")

    # --- Separation sweep ---
    print("\n[exp01] === SEPARATION SWEEP ===")
    sizes = [50, 100, 250, 500, 1000, args.n_items]
    sizes = [s for s in sizes if s <= args.n_items]
    sweeps: Dict[str, List[SeparationResult]] = {}
    for enc_name, batch in batches.items():
        for v_name, transform in variants.items():
            key = f"{enc_name}::{v_name}"
            print(f"\n -> {key}")
            transformed = transform(batch.embeddings)
            sweeps[key] = run_separation_sweep(transformed, sizes=sizes)
            summary["separation"][key] = [asdict(r) for r in sweeps[key]]

    # --- Save text summary ---
    summary_path = results_dir / "exp01_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[exp01] Wrote summary to {summary_path}")

    # --- Plots ---
    plot_isotropy(batches, variants, results_dir / "exp01_isotropy.png")
    plot_separation_sweep(sweeps, results_dir / "exp01_separation.png")

    # --- Verdict ---
    print("\n[exp01] === VERDICT ===")
    for key, results in sweeps.items():
        final = results[-1]
        verdict = "PASS" if final.perfect_selectivity_rate > 0.95 else (
            "MARGINAL" if final.perfect_selectivity_rate > 0.50 else "FAIL"
        )
        print(f"  {key:40s}  M={final.n_items}  "
              f"selectivity={final.perfect_selectivity_rate:.3f}  [{verdict}]")


if __name__ == "__main__":
    main()
