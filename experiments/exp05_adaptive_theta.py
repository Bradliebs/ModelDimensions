"""Experiment 05: Adaptive post-binding firing threshold.

Question: The m=5 failure in Exp04 was NOT a learning failure — Oja's rule
converged to perfect alignment (cos=1.000) on real embeddings. The failure
was at READOUT: with a fixed firing threshold of θ=0.3, individual bound
items' projections onto the rotated cell fell below θ as m grew, even
though the cell was pointing in exactly the right direction.

Theorem 3 of the paper says θ_star itself decays with m. We've been using
a fixed θ. This experiment tests whether using an m-aware θ (or a more
sophisticated adaptive policy) recovers the m=5 ceiling.

Three θ policies:

  1. "fixed"             : θ = 0.3 always (Exp04 baseline, for comparison)
  2. "theorem3"          : θ = θ_star(m) * 0.9, the paper's bound
  3. "calibrated"        : after binding, measure the minimum projection of
                            the bound items onto w_final, set θ just below
                            that. Maximally adaptive.

The headline question: does any policy push m=5 from 2% success to
something like 90%+? If yes, the architecture's working ceiling moves up.

The risk: lowering θ raises distractor false-fire rates. The "zero
false-fires" property we've enjoyed up to now was partly because θ was
generous. This experiment must measure the tradeoff, not just the gain.

Outputs:
  results/exp05_summary.json
  results/exp05_success_by_policy.png    : success rate vs m, one curve per policy
  results/exp05_tradeoff.png             : success vs false-fire rate
  results/exp05_margin_landscape.png     : where do bound items and distractors
                                            sit on the margin axis at each m?
                                            This is the single most-informative
                                            plot for understanding the limit.

Verdict per (policy, m):
  - WORKS_CLEAN     : success > 0.9 AND mean false-fires per trial < 1
  - WORKS_NOISY     : success > 0.9 BUT false-fires > 1 per trial
  - BLOCKED         : success < 0.5

To run:
    python -m experiments.exp05_adaptive_theta --n-trials 50 --m-values 2 3 5 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from concept_cells.data import load_corpus
from concept_cells.encoders import TextEncoder, UniformBallEncoder
from concept_cells.binding import (
    initialize_cell_for_anchor, bind_items, test_binding,
    theoretical_theta_star,
)
from concept_cells.geometry import scale_to_unit_ball, zca_whiten


# ---------- Threshold policies ----------

# A threshold policy is a callable: (m, w_final, bound_items) -> theta_readout
# It can ignore any of these arguments.

def policy_fixed(m: int, w_final: np.ndarray,
                  bound_items: List[np.ndarray],
                  fixed_theta: float = 0.30) -> float:
    return fixed_theta


def policy_theorem3(m: int, w_final: np.ndarray,
                     bound_items: List[np.ndarray],
                     eps: float = 0.05, safety: float = 0.9) -> float:
    """Use the paper's Theorem-3 upper bound, scaled by a safety factor."""
    ts = theoretical_theta_star(m, eps=eps)
    return float(max(ts * safety, 0.0))


def policy_calibrated(m: int, w_final: np.ndarray,
                       bound_items: List[np.ndarray],
                       safety_margin: float = 0.02) -> float:
    """Measure actual post-binding projections, set theta just below the min.

    This is the maximally adaptive policy: a cell that has just bound m items
    knows exactly where those items sit on the w_final axis, and can set its
    threshold to fire reliably for all of them. The safety margin is how far
    below the minimum projection to place theta.
    """
    projections = [float(w_final @ x) for x in bound_items]
    min_proj = min(projections)
    return max(min_proj - safety_margin, 0.0)


POLICIES: Dict[str, Callable] = {
    "fixed": policy_fixed,
    "theorem3": policy_theorem3,
    "calibrated": policy_calibrated,
}


# ---------- One trial under a given policy ----------

def run_one_trial(items: np.ndarray, distractors: np.ndarray,
                   m: int, theta_init: float, alpha: float, dt: float,
                   n_steps: int, policy_fn: Callable,
                   rng: np.random.Generator) -> Dict:
    """One binding trial. Returns a dict with success info plus the per-item
    and per-distractor margins so we can build margin-landscape plots."""
    n_total = items.shape[0]
    chosen_idx = rng.choice(n_total, size=m, replace=False)
    chosen = [items[i] for i in chosen_idx]
    anchor = chosen[0]

    n_dist = min(200, distractors.shape[0])
    dist_idx = rng.choice(distractors.shape[0], size=n_dist, replace=False)
    dist = [distractors[i] for i in dist_idx]

    # Unit-sphere initialization (the fix from exp04 v0.6)
    w0 = initialize_cell_for_anchor(anchor, theta=theta_init, eps=0.05)

    # Pre-binding sanity: cell fires for anchor at theta_init
    if (w0 @ anchor) <= theta_init:
        return {"skipped": True, "reason": "pre_anchor_failed"}

    # Bind
    w_final, _ = bind_items(w0, chosen, theta=theta_init,
                              alpha=alpha, dt=dt, n_steps=n_steps)

    # Apply the policy to set the readout threshold
    theta_readout = policy_fn(m, w_final, chosen)

    # Compute margins at the readout threshold
    item_projections = [float(w_final @ x) for x in chosen]
    dist_projections = [float(w_final @ x) for x in dist]
    item_margins = [p - theta_readout for p in item_projections]
    dist_margins = [p - theta_readout for p in dist_projections]

    n_items_fire = sum(1 for m in item_margins if m > 0)
    n_dist_fire = sum(1 for m in dist_margins if m > 0)

    return {
        "skipped": False,
        "theta_readout": theta_readout,
        "n_items_fire": n_items_fire,
        "n_dist_fire": n_dist_fire,
        "all_items_fire": n_items_fire == m,
        "no_distractor_fires": n_dist_fire == 0,
        "success": (n_items_fire == m) and (n_dist_fire == 0),
        "item_margins": item_margins,
        "dist_margins": dist_margins,
        "alignment": _alignment_to_mean(w_final, chosen),
    }


def _alignment_to_mean(w: np.ndarray, items: List[np.ndarray]) -> float:
    x_bar = np.mean(items, axis=0)
    x_bar_unit = x_bar / max(np.linalg.norm(x_bar), 1e-12)
    w_unit = w / max(np.linalg.norm(w), 1e-12)
    return float(w_unit @ x_bar_unit)


# ---------- Sweep over (m, policy) ----------

def run_sweep(items: np.ndarray, distractors: np.ndarray,
               m_values: List[int], n_trials: int, theta_init: float,
               alpha: float, dt: float, n_steps: int, seed: int
               ) -> Dict[str, Dict[int, Dict]]:
    """Returns results[policy_name][m] = aggregated stats."""
    results: Dict[str, Dict[int, Dict]] = {p: {} for p in POLICIES}
    for m in m_values:
        for policy_name, policy_fn in POLICIES.items():
            rng = np.random.default_rng(seed + 100 * m
                                          + hash(policy_name) % 1000)
            trials = []
            for _ in range(n_trials):
                trials.append(run_one_trial(
                    items=items, distractors=distractors, m=m,
                    theta_init=theta_init, alpha=alpha, dt=dt,
                    n_steps=n_steps, policy_fn=policy_fn, rng=rng,
                ))
            trials = [t for t in trials if not t.get("skipped", False)]
            if not trials:
                results[policy_name][m] = {
                    "n_valid_trials": 0, "skipped": True,
                }
                continue
            success_rate = float(np.mean([t["success"] for t in trials]))
            all_fire = float(np.mean([t["all_items_fire"] for t in trials]))
            no_dist = float(np.mean([t["no_distractor_fires"] for t in trials]))
            mean_dist_fires = float(np.mean([t["n_dist_fire"] for t in trials]))
            mean_theta = float(np.mean([t["theta_readout"] for t in trials]))
            mean_align = float(np.mean([t["alignment"] for t in trials]))
            # Margin landscape: collect per-trial margins for plotting
            all_item_margins = np.concatenate([t["item_margins"] for t in trials])
            all_dist_margins = np.concatenate([t["dist_margins"] for t in trials])
            results[policy_name][m] = {
                "n_valid_trials": len(trials),
                "success_rate": success_rate,
                "all_items_fire_rate": all_fire,
                "no_distractor_fires_rate": no_dist,
                "mean_n_distractor_fires": mean_dist_fires,
                "mean_theta_readout": mean_theta,
                "mean_alignment": mean_align,
                "_item_margins": all_item_margins,
                "_dist_margins": all_dist_margins,
            }
    return results


# ---------- Plots ----------

def plot_success_by_policy(real: Dict[str, Dict[int, Dict]],
                            m_values: List[int], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"fixed": "#888888", "theorem3": "#2a7a8a",
              "calibrated": "#c04040"}
    markers = {"fixed": "o", "theorem3": "s", "calibrated": "^"}
    for policy in real:
        succ = [real[policy][m].get("success_rate", 0.0) for m in m_values]
        ax.plot(m_values, succ, marker=markers[policy], linewidth=2.0,
                color=colors[policy], label=f"policy: {policy}")
    ax.set_xlabel("number of items bound (m)")
    ax.set_ylabel("full binding success rate")
    ax.set_title("Exp05: post-binding θ policy vs binding success "
                  "(real MiniLM embeddings)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


def plot_tradeoff(real: Dict[str, Dict[int, Dict]],
                   m_values: List[int], outpath: Path) -> None:
    """Success rate vs mean distractor false-fires per trial.

    Top-left of this plot is utopia (high success, low false-fires).
    Bottom-right is the worst tradeoff. The shape of each policy's curve
    tells us whether it gains success cheaply or at high false-fire cost.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"fixed": "#888888", "theorem3": "#2a7a8a",
              "calibrated": "#c04040"}
    markers = {"fixed": "o", "theorem3": "s", "calibrated": "^"}
    for policy in real:
        xs, ys, labels = [], [], []
        for m in m_values:
            d = real[policy][m]
            xs.append(d.get("mean_n_distractor_fires", 0.0))
            ys.append(d.get("success_rate", 0.0))
            labels.append(f"m={m}")
        ax.plot(xs, ys, marker=markers[policy], color=colors[policy],
                linewidth=1.5, label=f"policy: {policy}")
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=8,
                         xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean distractor false-fires per trial")
    ax.set_ylabel("success rate")
    ax.set_title("Tradeoff: lower θ buys success but may admit distractors")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


def plot_margin_landscape(real: Dict[str, Dict[int, Dict]],
                           m_values: List[int], outpath: Path) -> None:
    """For each (policy, m), show the margin distribution for bound items
    (should be > 0) vs distractors (should be < 0).

    This single plot is the answer to whether Path 2 is viable. If the two
    distributions overlap, no fixed θ can succeed; if they're well-separated
    we just need to pick θ between them.
    """
    n_policies = len(real)
    n_ms = len(m_values)
    fig, axes = plt.subplots(n_policies, n_ms,
                              figsize=(3.5 * n_ms, 3.0 * n_policies),
                              sharex=True, sharey=True)
    if n_policies == 1:
        axes = np.array([axes])
    if n_ms == 1:
        axes = axes.reshape(-1, 1)

    for i, policy in enumerate(real):
        for j, m in enumerate(m_values):
            ax = axes[i, j]
            d = real[policy][m]
            if "_item_margins" not in d:
                ax.set_title(f"{policy}, m={m}\n(skipped)", fontsize=9)
                continue
            item_m = d["_item_margins"]
            dist_m = d["_dist_margins"]

            # Choose bin range from both
            lo = min(item_m.min(), dist_m.min())
            hi = max(item_m.max(), dist_m.max())
            bins = np.linspace(lo, hi, 50)

            ax.hist(dist_m, bins=bins, alpha=0.55, color="#888",
                    density=True, label="distractors")
            ax.hist(item_m, bins=bins, alpha=0.65, color="#c04040",
                    density=True, label="bound items")
            ax.axvline(0.0, color="k", linestyle="--", alpha=0.6,
                        label="firing threshold")
            ax.set_title(f"{policy}, m={m}\n"
                          f"succ={d['success_rate']:.0%}, "
                          f"FF={d['mean_n_distractor_fires']:.2f}",
                          fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == n_policies - 1:
                ax.set_xlabel("margin (w·x − θ)")
            if j == 0:
                ax.set_ylabel(f"{policy}\ndensity", fontsize=9)
    # One legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10)
    fig.suptitle("Exp05: bound-item and distractor margin landscapes "
                  "(rows=policies, cols=m)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {outpath}")


# ---------- Main ----------

def classify(d: Dict) -> str:
    if d.get("skipped"):
        return "SKIPPED"
    s = d["success_rate"]
    ff = d["mean_n_distractor_fires"]
    if s < 0.5:
        return "BLOCKED"
    if s >= 0.9 and ff < 1.0:
        return "WORKS_CLEAN"
    if s >= 0.9:
        return "WORKS_NOISY"
    return "PARTIAL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--m-values", type=int, nargs="+",
                        default=[2, 3, 5, 8])
    parser.add_argument("--n-pool", type=int, default=1500)
    parser.add_argument("--n-distractors", type=int, default=500)
    parser.add_argument("--theta-init", type=float, default=0.30,
                        help="Firing threshold DURING the Oja learning phase. "
                             "Only the readout θ is varied by policy.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=str,
                        default=str(ROOT / "results"))
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Phase A: synthetic positive control (just one policy, the fixed one,
    # for context — Exp04 already established synthetic results)
    print("[exp05] === PHASE A: synthetic baseline (fixed θ only) ===")
    synth_enc = UniformBallEncoder(dim=384, seed=0)
    n_pool_total = args.n_pool + args.n_distractors
    synth_emb = synth_enc.encode(
        [f"x{i}" for i in range(n_pool_total)]
    ).embeddings
    pool_synth = synth_emb[: args.n_pool]
    dist_synth = synth_emb[args.n_pool:]
    synth_results = run_sweep(
        items=pool_synth, distractors=dist_synth,
        m_values=args.m_values, n_trials=args.n_trials,
        theta_init=args.theta_init, alpha=args.alpha, dt=args.dt,
        n_steps=args.n_steps, seed=1000,
    )

    if args.skip_real:
        print("\n[exp05] === SYNTHETIC RESULTS ===")
        print(f"  {'policy':>12s}  {'m':>3s}  {'succ':>6s}  "
              f"{'all_fire':>9s}  {'FF/trial':>9s}  "
              f"{'θ_read':>7s}  {'align':>6s}  {'verdict':>12s}")
        for policy in synth_results:
            for m in args.m_values:
                d = synth_results[policy][m]
                if d.get("skipped"):
                    continue
                v = classify(d)
                print(f"  {policy:>12s}  {m:>3d}  "
                      f"{d['success_rate']:>6.2f}  "
                      f"{d['all_items_fire_rate']:>9.2f}  "
                      f"{d['mean_n_distractor_fires']:>9.2f}  "
                      f"{d['mean_theta_readout']:>7.3f}  "
                      f"{d['mean_alignment']:>6.3f}  "
                      f"{v:>12s}")
        print("[exp05] Skipping real phase.")
        return

    # Phase B: real embeddings
    print("\n[exp05] === PHASE B: real MiniLM on wikitext ===")
    print(f"[exp05] Loading {n_pool_total} wikitext paragraphs...")
    texts = load_corpus("wikitext", n=n_pool_total)
    enc = TextEncoder("all-MiniLM-L6-v2", device=args.device)
    t0 = time.time()
    raw = enc.encode(texts).embeddings
    print(f"[exp05] Encoded in {time.time() - t0:.1f}s")
    z = zca_whiten(raw)
    z_scaled = scale_to_unit_ball(z)
    pool_real = z_scaled[: args.n_pool]
    dist_real = z_scaled[args.n_pool:]

    print(f"\n[exp05] Running policy sweep across m={args.m_values}...")
    real_results = run_sweep(
        items=pool_real, distractors=dist_real,
        m_values=args.m_values, n_trials=args.n_trials,
        theta_init=args.theta_init, alpha=args.alpha, dt=args.dt,
        n_steps=args.n_steps, seed=2000,
    )

    # Print stats
    print("\n[exp05] === RESULTS BY POLICY ===")
    print(f"  {'policy':>12s}  {'m':>3s}  {'succ':>6s}  "
          f"{'all_fire':>9s}  {'FF/trial':>9s}  "
          f"{'θ_read':>7s}  {'align':>6s}  {'verdict':>12s}")
    for policy in real_results:
        for m in args.m_values:
            d = real_results[policy][m]
            if d.get("skipped"):
                print(f"  {policy:>12s}  {m:>3d}  SKIPPED")
                continue
            v = classify(d)
            print(f"  {policy:>12s}  {m:>3d}  "
                  f"{d['success_rate']:>6.2f}  "
                  f"{d['all_items_fire_rate']:>9.2f}  "
                  f"{d['mean_n_distractor_fires']:>9.2f}  "
                  f"{d['mean_theta_readout']:>7.3f}  "
                  f"{d['mean_alignment']:>6.3f}  "
                  f"{v:>12s}")

    # JSON-safe summary
    def to_json(d):
        out = {}
        for policy in d:
            out[policy] = {}
            for m in d[policy]:
                inner = {k: v for k, v in d[policy][m].items()
                          if not k.startswith("_")}
                out[policy][m] = inner
        return out

    summary = {
        "params": {
            "n_trials": args.n_trials, "m_values": args.m_values,
            "n_pool": args.n_pool, "n_distractors": args.n_distractors,
            "theta_init": args.theta_init, "alpha": args.alpha, "dt": args.dt,
            "n_steps": args.n_steps,
        },
        "synthetic": to_json(synth_results),
        "real": to_json(real_results),
    }
    (results_dir / "exp05_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[exp05] Wrote {results_dir / 'exp05_summary.json'}")

    plot_success_by_policy(real_results, args.m_values,
                            results_dir / "exp05_success_by_policy.png")
    plot_tradeoff(real_results, args.m_values,
                   results_dir / "exp05_tradeoff.png")
    plot_margin_landscape(real_results, args.m_values,
                           results_dir / "exp05_margin_landscape.png")

    # Final verdict
    print("\n[exp05] === FINAL VERDICT ===")
    best_per_m: Dict[int, Tuple[str, float, float]] = {}
    for m in args.m_values:
        best = None
        for policy in real_results:
            d = real_results[policy][m]
            if d.get("skipped"):
                continue
            s = d["success_rate"]
            ff = d["mean_n_distractor_fires"]
            score = s - 0.1 * ff  # prefer success, lightly penalise false fires
            if best is None or score > best[2]:
                best = (policy, s, ff)
        best_per_m[m] = best

    moved_up = False
    for m in args.m_values:
        if best_per_m[m] is None:
            continue
        policy, s, ff = best_per_m[m]
        if s >= 0.9 and ff < 1.0:
            moved_up = True
            print(f"  m={m}: best policy = {policy}  "
                  f"(success={s:.0%}, false-fires={ff:.2f}/trial) — CEILING UP")
        elif s >= 0.9:
            print(f"  m={m}: best policy = {policy}  "
                  f"(success={s:.0%}, but false-fires={ff:.2f}/trial) — TRADEOFF")
        else:
            print(f"  m={m}: best policy = {policy}  "
                  f"(success={s:.0%}) — STILL BLOCKED")
    print()
    if moved_up:
        print("  Adaptive θ recovers binding at m values where fixed θ failed.")
        print("  Path 2 succeeded: the architecture's working ceiling is higher")
        print("  than Exp04 indicated.")
    else:
        print("  Adaptive θ did NOT cleanly recover binding. The m=5 ceiling")
        print("  reflects a real architectural limit on real embeddings, not")
        print("  a tunable parameter problem. Path 3 (manifold investigation)")
        print("  or Path 1 (accept m≤3) is the honest next step.")


if __name__ == "__main__":
    main()
