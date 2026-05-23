"""Experiment 04: Hebbian binding via Oja's rule.

THIS is the actually-novel experiment. Experiments 01–03b validated the
static memory: high-dim concept cells separate items cleanly, absorb
paraphrases (Theorem 2 group-selectivity), and reject novel content. None
of that is unique to the Tyukin/Gorban framework — Hopfield networks and
modern attention-based memories do versions of those things.

What's unique to this paper is DYNAMIC binding: two stimuli arriving
together cause the cell's weight vector to rotate (via Oja's rule) toward
their mean direction. After binding, the cell fires for either stimulus
ALONE. This is one-shot association without gradient descent.

If this works on real embeddings, we have a memory system that composes —
a fundamentally different capability from static lookup. If it fails, we
have an interesting memory but not a new architecture.

Method:
  Two phases.

  PHASE A — Synthetic sanity (uniform-ball samples in 384d).
    This is the ideal setting the paper's theorems assume. We verify that
    Oja-rule binding produces post-binding selective cells with high
    probability for m = 2, 3, 5, 8 items per cell. The synthetic baseline
    serves as the upper bound — real embeddings cannot do better than this.

  PHASE B — Real embeddings (MiniLM on wikitext).
    For each m in {2, 3, 5, 8}, run K binding trials:
      1. Sample m+1 items: m to bind into one cell, 1 anchor for init.
      2. Sample a large distractor set (200 items).
      3. Build initial cell aligned with the anchor.
      4. Verify pre-binding state: cell fires for anchor, silent for the
         others.
      5. Co-present all m bound items (anchor + m-1 others) and run Oja.
      6. Test: does the cell fire for EACH bound item alone? Stay silent
         for the distractors?
      7. Record success/failure and the alignment of w_final with the
         mean direction of the bound items.

Metrics:
  - success_rate(m): fraction of trials where ALL bound items fire AND
    NO distractor fires. This is the binding analog of "perfect selectivity."
  - mean_alignment(m): how well w_final aligned to x̄. ~1.0 means Oja
    converged as theorem 3 predicts.
  - false_fire_rate(m): how often a distractor crept through.

Outputs:
  results/exp04_summary.json
  results/exp04_binding_success.png        : success vs m, synthetic vs real
  results/exp04_alignment.png              : alignment-to-mean vs m
  results/exp04_margin_dynamics.png        : per-step margins during binding
                                              (for one example trial)

Verdict structure (per m):
  - WORKS         : real success_rate within 10pp of synthetic
  - DEGRADED      : real success_rate 10-30pp below synthetic
  - BROKEN        : real success_rate > 30pp below synthetic OR < 0.5

To run:
    python -m experiments.exp04_binding --n-trials 50 --m-values 2 3 5 8
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

from concept_cells.data import load_corpus
from concept_cells.encoders import TextEncoder, UniformBallEncoder
from concept_cells.binding import (
    initialize_cell_for_anchor, bind_items, test_binding,
    BindingResult, theoretical_theta_star,
)
from concept_cells.geometry import scale_to_unit_ball, zca_whiten


# ---------- One trial ----------

def run_one_trial(items: np.ndarray, distractors: np.ndarray,
                   m: int, theta: float, eps: float,
                   alpha: float, dt: float, n_steps: int,
                   rng: np.random.Generator,
                   record_dynamics: bool = False
                   ) -> Tuple[BindingResult, Dict]:
    """One binding trial: sample m items + distractors, bind, test.

    items: (N_items, D) pool of items to draw from (all train embeddings)
    distractors: (N_dist, D) pool of distractors to test against

    Returns the BindingResult and optionally a dict of step-by-step
    diagnostics for plotting.
    """
    n_total = items.shape[0]
    chosen_idx = rng.choice(n_total, size=m, replace=False)
    chosen = [items[i] for i in chosen_idx]
    anchor = chosen[0]

    # Sample distractors disjoint from chosen
    dist_pool_idx = np.array([i for i in range(distractors.shape[0])])
    distractor_sample_n = min(200, len(dist_pool_idx))
    dist_idx = rng.choice(dist_pool_idx, size=distractor_sample_n, replace=False)
    dist = [distractors[i] for i in dist_idx]

    # Build initial cell aligned with anchor
    w0 = initialize_cell_for_anchor(anchor, theta=theta, eps=eps)

    # Pre-binding sanity: cell should fire for anchor, not for chosen[1:] or dist
    pre_anchor = (w0 @ anchor) > theta
    pre_other_fires = sum(int((w0 @ x) > theta) for x in chosen[1:])
    if not pre_anchor:
        # Something is wrong with the initialisation; skip this trial
        return BindingResult(
            n_items=m, n_steps=0, success=False,
            fires_per_item=[False] * m, margins_per_item=[0.0] * m,
            final_alignment_to_mean=0.0,
            n_distractor_false_fires=0, distractor_margin_max=0.0,
            w_norm_final=0.0,
        ), {"skipped": True, "reason": "pre_anchor_failed"}

    # Record dynamics if requested
    dynamics: Dict = {}
    if record_dynamics:
        s_bar = np.sum(chosen, axis=0)
        w = w0.copy().astype(np.float64)
        margins_at_step: List[List[float]] = []
        norms: List[float] = [float(np.linalg.norm(w))]
        for _ in range(n_steps):
            from concept_cells.binding import oja_step
            w = oja_step(w, s_bar, theta, alpha, dt)
            norms.append(float(np.linalg.norm(w)))
            margins_at_step.append([float(w @ x) - theta for x in chosen])
        w_final = w.astype(np.float32)
        dynamics = {
            "norms": norms,
            "margins": np.array(margins_at_step),  # (n_steps, m)
        }
    else:
        w_final, norms = bind_items(w0, chosen, theta=theta,
                                      alpha=alpha, dt=dt, n_steps=n_steps)

    # Test
    result = test_binding(items=chosen, distractors=dist,
                           w_initial=w0, w_final=w_final,
                           theta_initial=theta)
    result.n_steps = n_steps
    return result, dynamics


# ---------- One m, K trials ----------

def run_m_trials(items: np.ndarray, distractors: np.ndarray, m: int,
                  n_trials: int, theta: float, eps: float,
                  alpha: float, dt: float, n_steps: int,
                  seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    results: List[BindingResult] = []
    for k in range(n_trials):
        r, _ = run_one_trial(
            items=items, distractors=distractors, m=m,
            theta=theta, eps=eps, alpha=alpha, dt=dt, n_steps=n_steps,
            rng=rng, record_dynamics=False,
        )
        results.append(r)

    success_rate = float(np.mean([r.success for r in results]))
    mean_alignment = float(np.mean([r.final_alignment_to_mean for r in results]))
    n_false_fires = [r.n_distractor_false_fires for r in results]
    trials_with_any_false_fire = float(np.mean([n > 0 for n in n_false_fires]))
    trials_all_items_fired = float(np.mean(
        [all(r.fires_per_item) for r in results]
    ))

    return {
        "m": m,
        "n_trials": n_trials,
        "success_rate": success_rate,
        "mean_alignment_to_mean": mean_alignment,
        "trials_all_items_fired": trials_all_items_fired,
        "trials_with_any_false_fire": trials_with_any_false_fire,
        "mean_false_fires_per_trial": float(np.mean(n_false_fires)),
        "theoretical_theta_star": theoretical_theta_star(m, eps=eps),
        "raw_results": results,
    }


# ---------- Plots ----------

def plot_success(synthetic: Dict, real: Dict, outpath: Path) -> None:
    ms = sorted(synthetic.keys())
    syn_succ = [synthetic[m]["success_rate"] for m in ms]
    real_succ = [real[m]["success_rate"] for m in ms]
    syn_items = [synthetic[m]["trials_all_items_fired"] for m in ms]
    real_items = [real[m]["trials_all_items_fired"] for m in ms]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax = axes[0]
    ax.plot(ms, syn_succ, "o-", color="#2a7a8a", label="synthetic (uniform ball)")
    ax.plot(ms, real_succ, "s-", color="#c04040", label="real (MiniLM wikitext)")
    ax.set_xlabel("number of items bound (m)")
    ax.set_ylabel("full success rate")
    ax.set_title("Full binding success\n(all m fire AND zero distractors fire)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(ms, syn_items, "o-", color="#2a7a8a", label="synthetic")
    ax.plot(ms, real_items, "s-", color="#c04040", label="real")
    ax.set_xlabel("number of items bound (m)")
    ax.set_title("Per-item recall\n(all m bound items fire, ignoring distractors)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("Exp04: Does Hebbian binding work on real embeddings?",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


def plot_alignment(synthetic: Dict, real: Dict, outpath: Path) -> None:
    ms = sorted(synthetic.keys())
    syn_a = [synthetic[m]["mean_alignment_to_mean"] for m in ms]
    real_a = [real[m]["mean_alignment_to_mean"] for m in ms]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ms, syn_a, "o-", color="#2a7a8a", label="synthetic (uniform ball)")
    ax.plot(ms, real_a, "s-", color="#c04040", label="real (MiniLM wikitext)")
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5,
                label="perfect alignment")
    ax.set_xlabel("number of items bound (m)")
    ax.set_ylabel("cos(w_final, mean direction of bound items)")
    ax.set_title("Did the cell rotate to the predicted direction?")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


def plot_dynamics(dynamics: Dict, theta: float, outpath: Path) -> None:
    """Per-step margins for the m bound items during one trial.

    The visual story: at step 0, only the anchor (item 0) is above zero.
    As Oja steps proceed, the other items' margins should rise above zero
    while the anchor's margin drops slightly — w is rotating away from
    pure-anchor toward the mean.
    """
    margins = dynamics["margins"]  # (n_steps, m)
    n_steps, m = margins.shape
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(m):
        ax.plot(range(1, n_steps + 1), margins[:, i],
                label=f"item {i}" + (" (anchor)" if i == 0 else ""))
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.7,
                label="firing threshold")
    ax.set_xlabel("Oja-rule step")
    ax.set_ylabel("margin (w · x_i − θ)")
    ax.set_title(f"Per-item firing margin during binding (m={m})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


# ---------- Main ----------

def classify(real_succ: float, syn_succ: float) -> str:
    gap = syn_succ - real_succ
    if real_succ < 0.5:
        return "BROKEN"
    if gap > 0.30:
        return "BROKEN"
    if gap > 0.10:
        return "DEGRADED"
    return "WORKS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--m-values", type=int, nargs="+",
                        default=[2, 3, 5, 8])
    parser.add_argument("--n-pool", type=int, default=1500,
                        help="size of item pool to sample bindings from")
    parser.add_argument("--n-distractors", type=int, default=500)
    parser.add_argument("--theta", type=float, default=0.30,
                        help="firing threshold (must be < theoretical_theta_star "
                             "for the largest m to even be feasible)")
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=str,
                        default=str(ROOT / "results"))
    parser.add_argument("--skip-real", action="store_true",
                        help="run only the synthetic sanity phase")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Sanity: is the firing threshold below theta_star for the largest m?
    largest_m = max(args.m_values)
    theta_star = theoretical_theta_star(largest_m, eps=args.eps)
    print(f"[exp04] theta={args.theta}, theta_star(m={largest_m})={theta_star:.4f}")
    if args.theta >= theta_star and theta_star > 0:
        print(f"[exp04] WARNING: theta is above the Theorem-3 bound for m={largest_m}. "
              "Binding for the largest m may fail even on synthetic data.")

    # ---------------- PHASE A: synthetic ----------------
    print("\n[exp04] === PHASE A: synthetic (uniform-ball, 384d) ===")
    synth_enc = UniformBallEncoder(dim=384, seed=0)
    n_pool = args.n_pool + args.n_distractors
    items_synth = synth_enc.encode([f"item_{i}" for i in range(n_pool)]).embeddings
    pool_synth = items_synth[: args.n_pool]
    dist_synth = items_synth[args.n_pool:]

    synth_results: Dict[int, Dict] = {}
    for m in args.m_values:
        print(f"\n[exp04]   m={m}:")
        r = run_m_trials(
            items=pool_synth, distractors=dist_synth, m=m,
            n_trials=args.n_trials,
            theta=args.theta, eps=args.eps,
            alpha=args.alpha, dt=args.dt, n_steps=args.n_steps,
            seed=1000 + m,
        )
        synth_results[m] = r
        print(f"     success_rate={r['success_rate']:.3f}  "
              f"all_items_fired={r['trials_all_items_fired']:.3f}  "
              f"alignment={r['mean_alignment_to_mean']:.3f}  "
              f"false_fires/trial={r['mean_false_fires_per_trial']:.2f}")

    if args.skip_real:
        print("\n[exp04] Skipping real-embedding phase as requested.")
        return

    # ---------------- PHASE B: real embeddings ----------------
    print("\n[exp04] === PHASE B: real embeddings (MiniLM on wikitext) ===")
    n_real_needed = args.n_pool + args.n_distractors
    print(f"[exp04] Loading {n_real_needed} wikitext paragraphs...")
    texts = load_corpus("wikitext", n=n_real_needed)
    enc = TextEncoder("all-MiniLM-L6-v2", device=args.device)
    t0 = time.time()
    raw = enc.encode(texts).embeddings
    print(f"[exp04] Encoded in {time.time() - t0:.1f}s")

    # Preprocess: whiten + scale into unit ball (the winning recipe from exp01-03)
    z = zca_whiten(raw)
    z_scaled = scale_to_unit_ball(z)
    pool_real = z_scaled[: args.n_pool]
    dist_real = z_scaled[args.n_pool:]

    real_results: Dict[int, Dict] = {}
    for m in args.m_values:
        print(f"\n[exp04]   m={m}:")
        r = run_m_trials(
            items=pool_real, distractors=dist_real, m=m,
            n_trials=args.n_trials,
            theta=args.theta, eps=args.eps,
            alpha=args.alpha, dt=args.dt, n_steps=args.n_steps,
            seed=2000 + m,
        )
        real_results[m] = r
        print(f"     success_rate={r['success_rate']:.3f}  "
              f"all_items_fired={r['trials_all_items_fired']:.3f}  "
              f"alignment={r['mean_alignment_to_mean']:.3f}  "
              f"false_fires/trial={r['mean_false_fires_per_trial']:.2f}")

    # ---------------- Record dynamics on one example real-embedding trial ----------------
    print("\n[exp04] Recording margin dynamics on one real-embedding trial (m=3)...")
    rng = np.random.default_rng(9999)
    _, dyn = run_one_trial(
        items=pool_real, distractors=dist_real, m=3,
        theta=args.theta, eps=args.eps, alpha=args.alpha, dt=args.dt,
        n_steps=args.n_steps, rng=rng, record_dynamics=True,
    )
    if "margins" in dyn:
        plot_dynamics(dyn, theta=args.theta,
                       outpath=results_dir / "exp04_margin_dynamics.png")

    # ---------------- Summary + plots ----------------
    def strip_raw(d):
        return {m: {k: v for k, v in d[m].items() if k != "raw_results"}
                for m in d}

    summary = {
        "params": {
            "n_trials": args.n_trials, "m_values": args.m_values,
            "n_pool": args.n_pool, "n_distractors": args.n_distractors,
            "theta": args.theta, "eps": args.eps,
            "alpha": args.alpha, "dt": args.dt, "n_steps": args.n_steps,
        },
        "synthetic": strip_raw(synth_results),
        "real": strip_raw(real_results),
    }
    summary_path = results_dir / "exp04_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[exp04] Wrote {summary_path}")

    plot_success(synth_results, real_results,
                  results_dir / "exp04_binding_success.png")
    plot_alignment(synth_results, real_results,
                    results_dir / "exp04_alignment.png")

    # ---------------- Verdict ----------------
    print("\n[exp04] === VERDICT TABLE ===")
    print(f"  {'m':>3s}  {'syn_succ':>10s}  {'real_succ':>10s}  "
          f"{'syn_align':>10s}  {'real_align':>11s}  {'verdict':>12s}")
    overall_verdicts = []
    for m in args.m_values:
        s = synth_results[m]["success_rate"]
        r = real_results[m]["success_rate"]
        sa = synth_results[m]["mean_alignment_to_mean"]
        ra = real_results[m]["mean_alignment_to_mean"]
        v = classify(r, s)
        overall_verdicts.append(v)
        print(f"  {m:>3d}  {s:>10.3f}  {r:>10.3f}  "
              f"{sa:>10.3f}  {ra:>11.3f}  {v:>12s}")

    print("\n[exp04] FINAL VERDICT:")
    if all(v == "WORKS" for v in overall_verdicts):
        print("  Hebbian binding works on real embeddings across all tested m.")
        print("  This is the novel architectural mechanism — the project has")
        print("  produced its core result. Move to writing up the architecture.")
    elif any(v == "BROKEN" for v in overall_verdicts):
        broken_ms = [m for m, v in zip(args.m_values, overall_verdicts)
                     if v == "BROKEN"]
        print(f"  Binding BROKEN at m={broken_ms}. The architecture binds in")
        print("  principle but not at the scales we tested. Need to investigate")
        print("  whether this is an encoder-anisotropy issue, a hyperparameter")
        print("  issue (theta, n_steps), or a fundamental limit.")
    else:
        print("  Mixed: binding works at low m, degrades at higher m. This")
        print("  matches the Theorem 3 prediction (theta_star decays with m)")
        print("  but the degradation may be steeper on real embeddings than")
        print("  the theory predicts.")


if __name__ == "__main__":
    main()
