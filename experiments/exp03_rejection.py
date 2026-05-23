"""Experiment 03: Out-of-distribution rejection.

Question: Does the concept-cell architecture correctly stay silent for
items it has never seen?

Experiment 02 showed that items IN memory correctly activate the right
cell (and absorb their paraphrases — Theorem 2 group selectivity). But
this is only half of useful memory. The other half is: items NOT in memory
must produce NO activation. A memory that fires on everything is useless.

Method:
  1. Split a corpus into TRAIN (cells are built from these) and TEST
     (queries that were NEVER used to build cells).
  2. Encode both sets with the same encoder, apply the same preprocessing.
  3. Build cells from train embeddings.
  4. Query with test embeddings. Measure:
       - For each test query, how many cells fire? Should be 0 if the
         architecture rejects novel items.
       - Maximum activation per test query: how close did the closest cell
         get to firing? This tells us the "margin" of rejection.
       - For comparison, do the same with train queries (where we know
         exactly one cell SHOULD fire).

Metrics:
  - test_silence_rate: fraction of test queries that activated zero cells.
    1.0 = perfect rejection, the architecture says "I don't know."
    0.0 = every novel input falsely matches something.
  - test_activation_count: distribution of how many cells fired per test
    query. Should be a dirac at 0 in the ideal case.
  - max_activation_margin: how confident are correct rejections? The
    larger the gap between max-activation-on-test and threshold, the more
    robust the rejection.

Verdict structure:
  - PERFECT_REJECTION : silence_rate > 0.99
  - GOOD_REJECTION    : silence_rate in [0.95, 0.99]
  - LEAKY_REJECTION   : silence_rate in [0.80, 0.95]  (worrying)
  - BROKEN_REJECTION  : silence_rate < 0.80           (fundamental problem)

If rejection works for free from the geometry, this is the missing piece.
If it doesn't, we need an explicit rejection mechanism — and the
architecture story gets considerably more complicated.

To run:
    python -m experiments.exp03_rejection --n-train 1500 --n-test 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from concept_cells.data import load_corpus_split
from concept_cells.encoders import TextEncoder
from concept_cells.geometry import (
    build_concept_cells, scale_to_unit_ball, zca_whiten
)


def query_cells(w: np.ndarray, theta: np.ndarray,
                queries: np.ndarray) -> Dict[str, np.ndarray]:
    """Run queries against a cell bank.

    Returns:
      activations: (n_queries, n_cells) — raw <w_i, q_j> values
      fires:       (n_queries, n_cells) — boolean, did cell i fire for query j
      margins:     (n_queries,) — max activation - threshold per query
                                  (>0 means at least one cell fired)
    """
    # activations[j, i] = <w_i, q_j>
    activations = queries @ w.T   # (n_queries, n_cells)
    fires = activations > theta[None, :]
    # margin = how much the best cell exceeded its threshold for this query
    # (positive = something fired; negative = nothing fired and this is the gap)
    margin_per_cell = activations - theta[None, :]
    best_margin = margin_per_cell.max(axis=1)
    return {
        "activations": activations,
        "fires": fires,
        "best_margin": best_margin,
    }


def classify_rejection(silence_rate: float) -> str:
    if silence_rate > 0.99:
        return "PERFECT_REJECTION"
    if silence_rate > 0.95:
        return "GOOD_REJECTION"
    if silence_rate > 0.80:
        return "LEAKY_REJECTION"
    return "BROKEN_REJECTION"


def analyze_variant(train_emb: np.ndarray, test_emb: np.ndarray,
                     variant: str, results_dir: Path,
                     epsilon: float = 0.05) -> Dict:
    """Build cells from train, query with both train and test, compare."""
    print(f"\n[exp03]   variant: {variant}")
    w, theta = build_concept_cells(train_emb, epsilon=epsilon,
                                    threshold_scheme="norm_minus_eps")
    print(f"   cells built: {w.shape[0]}, dim={w.shape[1]}")

    # --- Query with TRAIN items (sanity: cells should fire for their own item) ---
    train_q = query_cells(w, theta, train_emb)
    train_fires_per_query = train_q["fires"].sum(axis=1)
    # Each train query should activate at least its own cell:
    train_silence_rate = float((train_fires_per_query == 0).mean())
    train_one_or_more = float((train_fires_per_query >= 1).mean())

    # --- Query with TEST items (the key measurement) ---
    test_q = query_cells(w, theta, test_emb)
    test_fires_per_query = test_q["fires"].sum(axis=1)

    test_silence_rate = float((test_fires_per_query == 0).mean())
    test_mean_fires = float(test_fires_per_query.mean())
    test_max_fires = int(test_fires_per_query.max())
    test_any_fired = float((test_fires_per_query >= 1).mean())

    # Margin distribution: for test queries, how far were activations
    # from crossing threshold? Negative margin = silent rejection.
    test_margin = test_q["best_margin"]
    test_margin_mean = float(test_margin.mean())
    test_margin_median = float(np.median(test_margin))
    train_margin = train_q["best_margin"]

    print(f"   train sanity: {train_one_or_more:.1%} of train queries "
          f"activated >=1 cell (should be ~100%)")
    print(f"   test silence: {test_silence_rate:.1%} of test queries "
          f"activated 0 cells")
    print(f"   test false-fire rate: {test_any_fired:.1%}")
    print(f"   test mean cells fired/query: {test_mean_fires:.3f}")
    print(f"   test best margin (median): {test_margin_median:+.4f}")

    return {
        "variant": variant,
        "n_train": int(train_emb.shape[0]),
        "n_test": int(test_emb.shape[0]),
        "epsilon": epsilon,
        "train_silence_rate": train_silence_rate,
        "train_one_or_more_rate": train_one_or_more,
        "test_silence_rate": test_silence_rate,
        "test_any_fired_rate": test_any_fired,
        "test_mean_fires_per_query": test_mean_fires,
        "test_max_fires_per_query": test_max_fires,
        "test_margin_mean": test_margin_mean,
        "test_margin_median": test_margin_median,
        "train_margin_mean": float(train_margin.mean()),
        "train_margin_median": float(np.median(train_margin)),
        "verdict": classify_rejection(test_silence_rate),
        # Save the raw margin arrays for plotting
        "_train_margin": train_margin,
        "_test_margin": test_margin,
        "_test_fires_per_query": test_fires_per_query,
    }


def plot_margin_distributions(results: Dict[str, Dict], outpath: Path) -> None:
    """For each variant, plot the train vs test margin distributions.

    The key visual signal: are TEST margins clearly negative (rejected)
    while TRAIN margins are clearly positive (accepted)? If the two
    distributions overlap, the architecture can't distinguish known from
    novel.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (variant, r) in zip(axes, results.items()):
        train_m = r["_train_margin"]
        test_m = r["_test_margin"]

        # Symmetric bins around 0
        all_m = np.concatenate([train_m, test_m])
        lo, hi = np.percentile(all_m, [1, 99])
        bins = np.linspace(lo, hi, 60)

        ax.hist(train_m, bins=bins, alpha=0.55, density=True,
                label=f"train queries (n={len(train_m)})", color="#2a7a2a")
        ax.hist(test_m, bins=bins, alpha=0.55, density=True,
                label=f"test queries (n={len(test_m)})", color="#c04040")
        ax.axvline(0.0, color="k", linestyle="--", alpha=0.7,
                    label="firing threshold")
        ax.set_title(f"{variant}\nsilence={r['test_silence_rate']:.1%}  "
                      f"[{r['verdict']}]", fontsize=10)
        ax.set_xlabel("best margin (max activation − threshold)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("density")
    fig.suptitle("Exp03: train vs test query margins — does novelty get rejected?",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


def plot_fires_per_query(results: Dict[str, Dict], outpath: Path) -> None:
    """How many cells fire per test query? Ideal: a dirac at 0."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (variant, r) in zip(axes, results.items()):
        fires = r["_test_fires_per_query"]
        max_fires = max(int(fires.max()), 5)
        ax.hist(fires, bins=np.arange(max_fires + 2) - 0.5,
                density=True, color="#c04040", alpha=0.75)
        ax.set_title(f"{variant}\nmean={fires.mean():.2f}, max={fires.max()}",
                      fontsize=10)
        ax.set_xlabel("# cells firing per test query")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("density")
    fig.suptitle("Exp03: how many cells fire per novel query? (ideal: 0)",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--corpus", type=str, default="wikitext",
                        help="wikitext | ag_news | synthetic")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=str,
                        default=str(ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp03] Loading {args.corpus}: {args.n_train} train + "
          f"{args.n_test} test (disjoint)...")
    train_texts, test_texts = load_corpus_split(
        args.corpus, n_train=args.n_train, n_test=args.n_test, seed=0,
    )
    print(f"[exp03] First train text: {train_texts[0][:80]!r}")
    print(f"[exp03] First test text : {test_texts[0][:80]!r}")
    # Verify no overlap
    overlap = set(train_texts) & set(test_texts)
    assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} items!"

    enc = TextEncoder("all-MiniLM-L6-v2", device=args.device)
    t0 = time.time()
    train_batch = enc.encode(train_texts)
    test_batch = enc.encode(test_texts)
    print(f"[exp03] Encoded both splits in {time.time() - t0:.1f}s")

    raw_train, raw_test = train_batch.embeddings, test_batch.embeddings

    # Variants. NOTE: ZCA whitening must be fitted on TRAIN ONLY and applied
    # to TEST — otherwise we leak the test distribution. Same for ball scaling.
    def fit_and_apply_scaling(train: np.ndarray, test: np.ndarray):
        max_norm = float(np.linalg.norm(train, axis=1).max())
        return (train / (max_norm + 1e-8)).astype(np.float32), \
               (test / (max_norm + 1e-8)).astype(np.float32)

    def fit_and_apply_zca(train: np.ndarray, test: np.ndarray, eps: float = 1e-5):
        mu = train.mean(axis=0, keepdims=True)
        centered_train = train - mu
        cov = (centered_train.T @ centered_train) / max(centered_train.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, eps)
        w_mat = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return ((centered_train @ w_mat).astype(np.float32),
                ((test - mu) @ w_mat).astype(np.float32))

    print("\n[exp03] Building variants (fit on TRAIN, apply to TEST)...")
    # Raw
    var_raw = (raw_train, raw_test)
    # Ball scaled
    var_ball = fit_and_apply_scaling(raw_train, raw_test)
    # Whitened + ball scaled
    z_train, z_test = fit_and_apply_zca(raw_train, raw_test)
    var_white = fit_and_apply_scaling(z_train, z_test)

    variants = {
        "raw": var_raw,
        "ball_scaled": var_ball,
        "whitened_scaled": var_white,
    }

    summary = {
        "corpus": args.corpus,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "epsilon": args.epsilon,
        "variants": {},
    }
    plot_data: Dict[str, Dict] = {}

    for vname, (tr, te) in variants.items():
        r = analyze_variant(tr, te, vname, results_dir, epsilon=args.epsilon)
        plot_data[vname] = r
        # Strip raw arrays before JSON-serialising
        summary["variants"][vname] = {k: v for k, v in r.items()
                                       if not k.startswith("_")}

    summary_path = results_dir / "exp03_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[exp03] Wrote {summary_path}")

    plot_margin_distributions(plot_data,
                               results_dir / "exp03_margin_distributions.png")
    plot_fires_per_query(plot_data,
                          results_dir / "exp03_fires_per_query.png")

    # --- Verdict table ---
    print("\n[exp03] === VERDICT TABLE ===")
    print(f"  {'variant':18s}  {'silence':>9s}  {'any_fired':>10s}  "
          f"{'mean_fires':>11s}  {'verdict':>20s}")
    for vname, r in plot_data.items():
        print(f"  {vname:18s}  "
              f"{r['test_silence_rate']:>9.1%}  "
              f"{r['test_any_fired_rate']:>10.1%}  "
              f"{r['test_mean_fires_per_query']:>11.3f}  "
              f"{r['verdict']:>20s}")

    print("\n[exp03] Sanity check (train queries should mostly fire):")
    for vname, r in plot_data.items():
        print(f"  {vname:18s}  train activated >=1 cell: "
              f"{r['train_one_or_more_rate']:.1%}  "
              f"(should be ~100%)")


if __name__ == "__main__":
    main()
