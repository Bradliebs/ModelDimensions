"""Experiment 03b: Paragraph-leak audit.

Question: Was the PERFECT_REJECTION result from Exp03 inflated by paragraphs
from the same Wikipedia article appearing on both sides of the train/test
split?

Exp03 split wikitext at the paragraph level. If two paragraphs from the
same article happened to land on opposite sides of that split, they would
share topic, style, and probably named entities — and an honest semantic
encoder would correctly produce near-identical embeddings. The architecture
would then either correctly fire (matching the train paragraph's cell) OR
correctly reject (if the embeddings just aren't close enough). Either way,
those cases muddy the rejection measurement, because the "novel" test item
isn't really novel.

Method:
  Run the same rejection experiment twice:
    1. Paragraph-level split (the Exp03 method) — control.
    2. Article-level split — every article is wholly train OR wholly test.
       No paragraph in the test set comes from an article that contributed
       any paragraph to the train set.

  Compare silence rates. If they're the same, paragraph-level leak is not
  a problem and Exp03's result stands. If article-level rejection is worse,
  the original number was inflated and the architecture is more permissive
  on genuine novelty than we thought.

What "worse" would mean architecturally:
  Some performance gap is expected and acceptable — paragraphs from the
  same article SHOULD be more semantically similar than paragraphs from
  random other articles. The question is how much. If article-level
  silence drops from 99.8% to e.g. 99.0%, the architecture is still
  excellent. If it drops to 85%, we have a real problem.

To run:
    python -m experiments.exp03b_paragraph_leak --n-train 1500 --n-test 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from concept_cells.data import (
    load_corpus_split, load_corpus_grouped, article_level_split
)
from concept_cells.encoders import TextEncoder
from concept_cells.geometry import build_concept_cells

# Reuse the rejection-analysis helpers from exp03 verbatim
from experiments.exp03_rejection import (
    query_cells, classify_rejection, analyze_variant,
    plot_margin_distributions, plot_fires_per_query,
)


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


def build_variants(raw_train: np.ndarray, raw_test: np.ndarray
                    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    ball = fit_and_apply_scaling(raw_train, raw_test)
    z_train, z_test = fit_and_apply_zca(raw_train, raw_test)
    white = fit_and_apply_scaling(z_train, z_test)
    return {
        "raw": (raw_train, raw_test),
        "ball_scaled": ball,
        "whitened_scaled": white,
    }


def run_one_split(label: str, train_texts: List[str], test_texts: List[str],
                   encoder: TextEncoder, epsilon: float,
                   results_dir: Path) -> Dict:
    """Encode + run all three variants on a given train/test split."""
    print(f"\n[exp03b] ===== SPLIT: {label} =====")
    print(f"   n_train={len(train_texts)}, n_test={len(test_texts)}")

    t0 = time.time()
    raw_train = encoder.encode(train_texts).embeddings
    raw_test = encoder.encode(test_texts).embeddings
    print(f"   encoded in {time.time() - t0:.1f}s")

    variants = build_variants(raw_train, raw_test)
    variant_results: Dict[str, Dict] = {}
    for vname, (tr, te) in variants.items():
        r = analyze_variant(tr, te, vname, results_dir, epsilon=epsilon)
        variant_results[vname] = r

    return variant_results


def plot_comparison(paragraph_results: Dict[str, Dict],
                     article_results: Dict[str, Dict],
                     outpath: Path) -> None:
    """Side-by-side: silence rate by variant under each split scheme."""
    variants = list(paragraph_results.keys())
    x = np.arange(len(variants))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: silence rate
    ax = axes[0]
    para_sil = [paragraph_results[v]["test_silence_rate"] for v in variants]
    art_sil = [article_results[v]["test_silence_rate"] for v in variants]
    ax.bar(x - width / 2, para_sil, width, label="paragraph-level split",
            color="#888")
    ax.bar(x + width / 2, art_sil, width, label="article-level split",
            color="#2a7a8a")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("test silence rate")
    ax.axhline(0.99, color="k", linestyle=":", alpha=0.5,
                label="PERFECT_REJECTION threshold")
    ax.set_title("Silence rate by split method")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    # Annotate bars with exact %
    for xi, (p, a) in enumerate(zip(para_sil, art_sil)):
        ax.text(xi - width / 2, p + 0.003, f"{p:.1%}",
                ha="center", fontsize=8)
        ax.text(xi + width / 2, a + 0.003, f"{a:.1%}",
                ha="center", fontsize=8)

    # Panel 2: median test margin (negative = silent; closer to 0 = leaky)
    ax = axes[1]
    para_mar = [paragraph_results[v]["test_margin_median"] for v in variants]
    art_mar = [article_results[v]["test_margin_median"] for v in variants]
    ax.bar(x - width / 2, para_mar, width, label="paragraph-level split",
            color="#888")
    ax.bar(x + width / 2, art_mar, width, label="article-level split",
            color="#2a7a8a")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylabel("test margin (median)")
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.7,
                label="firing threshold")
    ax.set_title("Median test margin (more negative = safer rejection)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp03b: Did paragraph-level splits leak same-article content?",
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
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=str,
                        default=str(ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    enc = TextEncoder("all-MiniLM-L6-v2", device=args.device)

    # ---- Split 1: paragraph-level (Exp03's method) ----
    print("[exp03b] Loading wikitext at paragraph level...")
    para_train, para_test = load_corpus_split(
        "wikitext", n_train=args.n_train, n_test=args.n_test, seed=0
    )
    para_overlap = len(set(para_train) & set(para_test))
    assert para_overlap == 0, f"text overlap: {para_overlap}"

    # ---- Split 2: article-level ----
    print("\n[exp03b] Loading wikitext at article level...")
    # We need a larger pool to find enough disjoint articles
    grouped = load_corpus_grouped(
        "wikitext", n=args.n_train + args.n_test + 1000, seed=0
    )
    art_train, art_test, split_stats = article_level_split(
        grouped, n_train=args.n_train, n_test=args.n_test, seed=0
    )
    print(f"   article split stats: {split_stats}")

    # ---- Encode + analyse both splits ----
    para_results = run_one_split(
        "paragraph_level", para_train, para_test,
        encoder=enc, epsilon=args.epsilon, results_dir=results_dir,
    )
    art_results = run_one_split(
        "article_level", art_train, art_test,
        encoder=enc, epsilon=args.epsilon, results_dir=results_dir,
    )

    # ---- Summary ----
    def strip_arrays(d):
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in d.items()}

    summary = {
        "n_train": args.n_train,
        "n_test": args.n_test,
        "epsilon": args.epsilon,
        "article_split_stats": split_stats,
        "paragraph_level": strip_arrays(para_results),
        "article_level": strip_arrays(art_results),
    }
    summary_path = results_dir / "exp03b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[exp03b] Wrote {summary_path}")

    plot_comparison(para_results, art_results,
                     results_dir / "exp03b_split_comparison.png")

    # Also plot the article-split margin distributions for direct visual
    # comparison with the exp03 figure.
    plot_margin_distributions(
        art_results,
        results_dir / "exp03b_article_margin_distributions.png",
    )
    plot_fires_per_query(
        art_results,
        results_dir / "exp03b_article_fires_per_query.png",
    )

    # ---- Verdict ----
    print("\n[exp03b] === VERDICT TABLE ===")
    print(f"  {'variant':18s}  {'para_silence':>13s}  {'art_silence':>13s}  "
          f"{'delta':>8s}  {'verdict':>22s}")

    big_drop = False
    for v in para_results:
        p = para_results[v]["test_silence_rate"]
        a = art_results[v]["test_silence_rate"]
        delta = a - p
        # A drop of > 2 percentage points is worth flagging
        if delta < -0.02:
            tag = "MEANINGFUL_DROP"
            big_drop = True
        elif delta < -0.005:
            tag = "SMALL_DROP"
        else:
            tag = "NO_LEAK_DETECTED"
        print(f"  {v:18s}  {p:>13.1%}  {a:>13.1%}  {delta:>+8.1%}  "
              f"{tag:>22s}")

    print("\n[exp03b] FINAL VERDICT:")
    if big_drop:
        print("  Article-level rejection is meaningfully worse than paragraph-level.")
        print("  Exp03's PERFECT_REJECTION was at least partly inflated by")
        print("  same-article paragraph leakage. The architecture still works,")
        print("  but the rejection numbers should be interpreted at the")
        print("  article-level values, which are lower.")
    else:
        print("  Article-level and paragraph-level rejection are equivalent.")
        print("  The Exp03 PERFECT_REJECTION result is robust to the leakage")
        print("  concern. Rejection genuinely works for novel content.")


if __name__ == "__main__":
    main()
