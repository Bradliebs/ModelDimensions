"""Experiment 02b: Confusion analysis across corpora.

Question: Was the SEMANTIC verdict from Exp02 an artifact of the ag_news
corpus (which contains many syndicated near-duplicate articles), or does
it hold for a more general corpus without that structure?

Method:
  Run the same confusion-pair analysis from Exp02 on wikitext (Wikipedia
  paragraphs — diverse content, no syndication duplicates by construction)
  and compare against ag_news.

What we expect to see:
  - If the SEMANTIC verdict was real: wikitext shows the same pattern —
    confused pairs are systematically more similar than random pairs,
    Cohen's d > 0.8, the bottom-of-list confusions look semantically related.
    The architecture generalises.
  - If it was an ag_news artifact: wikitext shows few confusions, OR the
    confusions look like noise (mixed-similarity, no clear semantic link).
    The architecture only works on corpora with structural duplicates.

If wikitext PASSES the SEMANTIC test, we have much stronger evidence that
concept cells exhibit Theorem 2 group-selectivity behaviour on real
semantic content, not just on artificially-duplicated content.

To run:
    python -m experiments.exp02b_corpus_compare --n-items 2000
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

from concept_cells.data import load_corpus
from concept_cells.encoders import TextEncoder
from concept_cells.geometry import scale_to_unit_ball, zca_whiten

# Reuse the analysis helpers from exp02 rather than re-implementing them
from experiments.exp02_confusion import (
    find_confusion_pairs, cosine_similarity_pairs, random_baseline_pairs,
    save_top_confusions,
)


def analyze_one(texts: List[str], raw_emb: np.ndarray,
                 transformed: np.ndarray, corpus: str, variant: str,
                 results_dir: Path, epsilon: float = 0.05) -> Dict:
    cell_idx, query_idx = find_confusion_pairs(transformed, epsilon=epsilon)
    n_conf = len(cell_idx)
    n_items = transformed.shape[0]
    fp_rate = n_conf / max(n_items * (n_items - 1), 1)

    result: Dict = {
        "corpus": corpus,
        "variant": variant,
        "n_items": n_items,
        "n_confusions": int(n_conf),
        "fp_rate": fp_rate,
    }

    if n_conf == 0:
        result["verdict"] = "ZERO_CONFUSIONS"
        return result

    confused_sims = cosine_similarity_pairs(raw_emb, cell_idx, query_idx)
    rand_i, rand_j = random_baseline_pairs(n_items,
                                             n_pairs=max(n_conf, 5000))
    random_sims = cosine_similarity_pairs(raw_emb, rand_i, rand_j)

    pooled_std = float(np.sqrt(
        (confused_sims.var() + random_sims.var()) / 2
    ))
    cohens_d = (confused_sims.mean() - random_sims.mean()) / max(pooled_std, 1e-9)
    p95 = float(np.percentile(random_sims, 95))
    frac_above = float((confused_sims > p95).mean())

    if cohens_d > 0.8 and frac_above > 0.5:
        verdict = "SEMANTIC"
    elif cohens_d > 0.3 and frac_above > 0.25:
        verdict = "MIXED"
    else:
        verdict = "NOISE"

    result.update({
        "confused_mean_sim": float(confused_sims.mean()),
        "confused_median_sim": float(np.median(confused_sims)),
        "random_mean_sim": float(random_sims.mean()),
        "random_median_sim": float(np.median(random_sims)),
        "cohens_d": float(cohens_d),
        "frac_above_random_p95": frac_above,
        "verdict": verdict,
    })

    # Save the 20 top + 20 bottom for human inspection
    save_top_confusions(
        texts, cell_idx, query_idx, confused_sims,
        results_dir / f"exp02b_top_confusions_{corpus}_{variant}.txt",
        top_k=20,
    )
    return result


def plot_corpus_comparison(results_by_corpus: Dict[str, Dict],
                            outpath: Path) -> None:
    """Side-by-side bar chart comparing corpora on each metric."""
    corpora = list(results_by_corpus.keys())
    variants = ["raw", "ball_scaled", "whitened_scaled"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: confusion count per corpus/variant
    ax = axes[0]
    x = np.arange(len(variants))
    width = 0.35
    for i, corpus in enumerate(corpora):
        counts = [results_by_corpus[corpus][v]["n_confusions"] for v in variants]
        ax.bar(x + i * width, counts, width, label=corpus)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylabel("# confusion pairs")
    ax.set_title("Confusion count by corpus")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Cohen's d
    ax = axes[1]
    for i, corpus in enumerate(corpora):
        ds = [results_by_corpus[corpus][v].get("cohens_d", 0.0) for v in variants]
        ax.bar(x + i * width, ds, width, label=corpus)
    ax.axhline(0.8, color="k", linestyle=":", alpha=0.6, label="d=0.8 (large)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect size (confused vs random)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: mean similarity gap
    ax = axes[2]
    for i, corpus in enumerate(corpora):
        gaps = [
            results_by_corpus[corpus][v].get("confused_mean_sim", 0.0)
            - results_by_corpus[corpus][v].get("random_mean_sim", 0.0)
            for v in variants
        ]
        ax.bar(x + i * width, gaps, width, label=corpus)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylabel("mean similarity gap")
    ax.set_title("Confused mean − random mean")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp02b: corpus comparison — does the SEMANTIC verdict generalise?",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--corpora", nargs="+",
                        default=["wikitext", "ag_news"],
                        help="Corpora to compare. Default: wikitext ag_news")
    parser.add_argument("--results-dir", type=str,
                        default=str(ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    enc = TextEncoder("all-MiniLM-L6-v2", device=args.device)
    all_results: Dict[str, Dict] = {}

    for corpus in args.corpora:
        print(f"\n[exp02b] ============ CORPUS: {corpus} ============")
        try:
            texts = load_corpus(corpus, n=args.n_items)
        except Exception as e:
            print(f"[exp02b] Could not load {corpus}: {e}")
            continue
        print(f"[exp02b] First text: {texts[0][:100]!r}")

        t0 = time.time()
        batch = enc.encode(texts)
        print(f"[exp02b] Encoded {len(texts)} texts in {time.time() - t0:.1f}s")

        raw = batch.embeddings
        variants = {
            "raw": raw,
            "ball_scaled": scale_to_unit_ball(raw),
            "whitened_scaled": scale_to_unit_ball(zca_whiten(raw)),
        }

        all_results[corpus] = {}
        for vname, transformed in variants.items():
            r = analyze_one(
                texts=texts, raw_emb=raw, transformed=transformed,
                corpus=corpus, variant=vname,
                results_dir=results_dir, epsilon=args.epsilon,
            )
            all_results[corpus][vname] = r
            if r["n_confusions"] == 0:
                print(f"  {vname:18s}  ZERO CONFUSIONS")
            else:
                print(f"  {vname:18s}  n={r['n_confusions']:4d}  "
                      f"d={r['cohens_d']:+6.2f}  "
                      f"frac>p95={r['frac_above_random_p95']:.1%}  "
                      f"[{r['verdict']}]")

    summary_path = results_dir / "exp02b_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[exp02b] Wrote {summary_path}")

    plot_corpus_comparison(all_results,
                            results_dir / "exp02b_corpus_comparison.png")

    # --- Final verdict ---
    print("\n[exp02b] === FINAL VERDICT ===")
    print("Does the SEMANTIC verdict generalise beyond ag_news?\n")
    for corpus, vmap in all_results.items():
        verdicts = [v["verdict"] for v in vmap.values()]
        print(f"  {corpus:12s}  variants: " + "  ".join(
            f"{k}={v['verdict']}" for k, v in vmap.items()
        ))


if __name__ == "__main__":
    main()
