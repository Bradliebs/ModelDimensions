"""Experiment 02: Confusion-pair analysis.

Question: Are the ~6-7% selectivity 'failures' in Experiment 01 actual bugs,
or are they semantically meaningful? If cell i fires for query j, is that
because items i and j are about the same thing?

This re-frames the whole project. If most confusions are semantic, the
architecture isn't broken — it's exhibiting Theorem 2 behaviour (selectivity
to a *group* of related stimuli) rather than Theorem 1 behaviour (selectivity
to a *single* stimulus). That's an architectural feature, not a failure.

Method:
  1. Reproduce Experiment 01 to get the confusion matrix (which cells fire
     for which queries).
  2. For each confusion pair (i, j) where cell i fires for query j != i,
     compute the *semantic similarity* between items i and j via:
        - Cosine similarity of raw MiniLM embeddings (an independent measure
          from our concept-cell construction)
        - Visual inspection of the texts themselves (we save them).
  3. Compare the semantic-similarity distribution of CONFUSED pairs against
     the distribution of RANDOM pairs. If confused pairs are systematically
     more similar than random, the failures are semantic.
  4. Build a 'top confusions' table showing the actual confused text pairs,
     so we can read them and decide whether the architecture is doing
     something sensible.

Outputs:
  results/exp02_confusion_similarity.png  : histogram comparison
  results/exp02_top_confusions.txt        : human-readable confused pairs
  results/exp02_summary.json              : numeric summary

Verdict structure:
  - If confused pairs have mean similarity >> random pairs: confusions are
    semantic. The architecture is fine; we just need a better evaluation
    metric. (And Theorem 2 is the relevant theorem, not Theorem 1.)
  - If confused pairs look like random pairs: confusions are noise. We do
    need Direction C (compound cells) or similar.
  - In between: mixed. Need to engineer around both.

To run:
    python -m experiments.exp02_confusion --n-items 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from concept_cells.data import diverse_sample_texts
from concept_cells.encoders import TextEncoder
from concept_cells.geometry import (
    build_concept_cells, scale_to_unit_ball, zca_whiten
)


def find_confusion_pairs(emb: np.ndarray, epsilon: float = 0.05
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays (cell_idx, query_idx) for every off-diagonal firing.

    For each cell i, returns all queries j != i for which cell i fires.
    """
    n = emb.shape[0]
    w, theta = build_concept_cells(emb, epsilon=epsilon,
                                    threshold_scheme="norm_minus_eps")
    activations = w @ emb.T
    fires = activations > theta[:, None]
    # Mask diagonal
    np.fill_diagonal(fires, False)
    cell_idx, query_idx = np.nonzero(fires)
    return cell_idx, query_idx


def cosine_similarity_pairs(emb: np.ndarray, idx_a: np.ndarray,
                             idx_b: np.ndarray) -> np.ndarray:
    """Cosine similarity for paired indices."""
    a = emb[idx_a]
    b = emb[idx_b]
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_u = a / np.maximum(a_norm, 1e-12)
    b_u = b / np.maximum(b_norm, 1e-12)
    return np.einsum("ij,ij->i", a_u, b_u)


def random_baseline_pairs(n: int, n_pairs: int, seed: int = 0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample random index pairs (i != j) for baseline comparison."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs * 2)
    j = rng.integers(0, n, size=n_pairs * 2)
    mask = i != j
    return i[mask][:n_pairs], j[mask][:n_pairs]


def plot_similarity_distributions(confused_sims: np.ndarray,
                                   random_sims: np.ndarray,
                                   outpath: Path, variant: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(-0.2, 1.0, 80)
    ax.hist(random_sims, bins=bins, alpha=0.55, density=True,
            label=f"random pairs (n={len(random_sims)})", color="#888")
    ax.hist(confused_sims, bins=bins, alpha=0.65, density=True,
            label=f"confused pairs (n={len(confused_sims)})", color="#d62728")

    ax.axvline(random_sims.mean(), color="#444", linestyle="--",
                label=f"random mean = {random_sims.mean():.3f}")
    ax.axvline(confused_sims.mean(), color="#a01818", linestyle="--",
                label=f"confused mean = {confused_sims.mean():.3f}")

    ax.set_xlabel("cosine similarity (raw MiniLM embeddings)")
    ax.set_ylabel("density")
    ax.set_title(f"Are confusions semantic? — variant: {variant}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


def save_top_confusions(texts: List[str], cell_idx: np.ndarray,
                         query_idx: np.ndarray, sims: np.ndarray,
                         outpath: Path, top_k: int = 50) -> None:
    """Save the most-similar confusion pairs for human inspection.

    These are the cases most likely to be 'legitimate' semantic confusions.
    """
    order = np.argsort(-sims)[:top_k]
    lines = [
        "TOP CONFUSION PAIRS (highest semantic similarity)",
        "=" * 80,
        "These are off-diagonal firings: cell i fires for query j != i.",
        "Sorted by cosine similarity of the two items' embeddings.",
        "If these look semantically related, the 'failures' are features.",
        "=" * 80,
        "",
    ]
    for rank, k in enumerate(order, 1):
        i, j = int(cell_idx[k]), int(query_idx[k])
        s = float(sims[k])
        lines.append(f"#{rank:2d}  sim={s:+.4f}  cell={i}  query={j}")
        lines.append(f"     cell item  : {texts[i][:180]}")
        lines.append(f"     query item : {texts[j][:180]}")
        lines.append("")

    # Also save the BOTTOM confusions — these are the genuinely surprising
    # ones, where the architecture confuses unrelated items.
    lines.extend([
        "",
        "=" * 80,
        "BOTTOM CONFUSION PAIRS (lowest semantic similarity)",
        "These are confusions that aren't explained by semantic similarity.",
        "If these dominate, the architecture has a real noise problem.",
        "=" * 80,
        "",
    ])
    order_low = np.argsort(sims)[:top_k]
    for rank, k in enumerate(order_low, 1):
        i, j = int(cell_idx[k]), int(query_idx[k])
        s = float(sims[k])
        lines.append(f"#{rank:2d}  sim={s:+.4f}  cell={i}  query={j}")
        lines.append(f"     cell item  : {texts[i][:180]}")
        lines.append(f"     query item : {texts[j][:180]}")
        lines.append("")

    outpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] wrote {outpath}  ({top_k} top + {top_k} bottom)")


def analyze_variant(texts: List[str], raw_emb: np.ndarray,
                     transformed: np.ndarray, variant_name: str,
                     results_dir: Path, epsilon: float = 0.05) -> dict:
    """Run the full confusion analysis on one preprocessing variant.

    We always measure semantic similarity on the *raw* MiniLM embeddings
    (raw_emb), because that's the standard, encoder-defined notion of
    semantic similarity. The 'transformed' embeddings are what we build
    cells from — but whether two items are 'semantically similar' is a
    property of the encoder, not of how we preprocess for memory.
    """
    print(f"\n[exp02] --- variant: {variant_name} ---")
    cell_idx, query_idx = find_confusion_pairs(transformed, epsilon=epsilon)
    n_confusions = len(cell_idx)
    n_items = transformed.shape[0]
    n_possible = n_items * (n_items - 1)
    fp_rate = n_confusions / max(n_possible, 1)
    print(f"  confusion pairs found: {n_confusions} "
          f"(fp rate {fp_rate:.4e})")

    if n_confusions == 0:
        print("  no confusions — variant is perfectly selective, skipping.")
        return {
            "variant": variant_name,
            "n_confusions": 0,
            "fp_rate": 0.0,
        }

    # Semantic similarity of confused pairs (always measured on raw embeddings)
    confused_sims = cosine_similarity_pairs(raw_emb, cell_idx, query_idx)

    # Random baseline — same number of pairs for fair comparison
    rand_i, rand_j = random_baseline_pairs(n_items,
                                             n_pairs=max(n_confusions, 5000))
    random_sims = cosine_similarity_pairs(raw_emb, rand_i, rand_j)

    # Statistical comparison
    confused_mean = float(confused_sims.mean())
    random_mean = float(random_sims.mean())
    confused_median = float(np.median(confused_sims))
    random_median = float(np.median(random_sims))

    # Effect size: how many standard deviations of the random distribution
    # is the confused mean shifted by? (Cohen's d style)
    pooled_std = float(np.sqrt(
        (confused_sims.var() + random_sims.var()) / 2
    ))
    cohens_d = (confused_mean - random_mean) / max(pooled_std, 1e-9)

    # What fraction of confused pairs have similarity above the 95th
    # percentile of random pairs? If most do, confusions are clearly semantic.
    random_p95 = float(np.percentile(random_sims, 95))
    frac_above_random_p95 = float((confused_sims > random_p95).mean())

    print(f"  confused mean sim   = {confused_mean:+.4f}  "
          f"(median {confused_median:+.4f})")
    print(f"  random   mean sim   = {random_mean:+.4f}  "
          f"(median {random_median:+.4f})")
    print(f"  Cohen's d           = {cohens_d:+.3f}  "
          f"(>0.8 = large effect)")
    print(f"  % confused > random p95 = {frac_above_random_p95:.1%}")

    # Plots and tables
    plot_similarity_distributions(
        confused_sims, random_sims,
        results_dir / f"exp02_confusion_similarity_{variant_name}.png",
        variant_name,
    )
    save_top_confusions(
        texts, cell_idx, query_idx, confused_sims,
        results_dir / f"exp02_top_confusions_{variant_name}.txt",
        top_k=30,
    )

    return {
        "variant": variant_name,
        "n_confusions": int(n_confusions),
        "fp_rate": fp_rate,
        "confused_mean_sim": confused_mean,
        "confused_median_sim": confused_median,
        "random_mean_sim": random_mean,
        "random_median_sim": random_median,
        "cohens_d": cohens_d,
        "random_p95": random_p95,
        "frac_confused_above_random_p95": frac_above_random_p95,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=str,
                        default=str(ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp02] Loading {args.n_items} diverse texts...")
    texts = diverse_sample_texts(n=args.n_items)
    print(f"[exp02] First text: {texts[0][:80]!r}")

    print("[exp02] Encoding with MiniLM...")
    t0 = time.time()
    batch = TextEncoder("all-MiniLM-L6-v2", device=args.device).encode(texts)
    print(f"[exp02] Encoded in {time.time() - t0:.1f}s")

    raw = batch.embeddings  # used for semantic-similarity ground truth

    # The three preprocessing variants — same as Experiment 01.
    variants = {
        "raw": raw,
        "ball_scaled": scale_to_unit_ball(raw),
        "whitened_scaled": scale_to_unit_ball(zca_whiten(raw)),
    }

    summary = {"epsilon": args.epsilon, "n_items": args.n_items, "variants": {}}

    for vname, transformed in variants.items():
        result = analyze_variant(
            texts=texts, raw_emb=raw, transformed=transformed,
            variant_name=vname, results_dir=results_dir, epsilon=args.epsilon,
        )
        summary["variants"][vname] = result

    summary_path = results_dir / "exp02_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[exp02] Wrote {summary_path}")

    # --- Verdict ---
    print("\n[exp02] === VERDICT ===")
    for vname, r in summary["variants"].items():
        if r.get("n_confusions", 0) == 0:
            print(f"  {vname:18s}  no confusions to analyze.")
            continue
        d = r["cohens_d"]
        frac = r["frac_confused_above_random_p95"]
        if d > 0.8 and frac > 0.5:
            verdict = "SEMANTIC  — confusions are mostly features, not bugs"
        elif d > 0.3 and frac > 0.25:
            verdict = "MIXED     — semantic and noise both contribute"
        else:
            verdict = "NOISE     — confusions look like random failures"
        print(f"  {vname:18s}  d={d:+.2f}  "
              f"frac>p95={frac:.1%}   [{verdict}]")


if __name__ == "__main__":
    main()
