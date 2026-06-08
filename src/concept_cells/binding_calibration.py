"""Hebbian binding calibration at production bank scale.

The roadmap claims a single concept cell can store up to 8 bound concepts
with a success rate of >= 92%. The historical experiment (``exp04_binding``)
established this on a synthetic 2000-item bank. Production banks now have
millions of cells, which means the distractor pool that ought to stay
silent under the bound cell is three orders of magnitude larger. The
selectivity claim must be re-verified at that scale before the binding
mechanism is operationalised in any user-facing flow.

This module is the pure measurement harness: given a corpus of embeddings,
it samples binding trials, runs the existing ``concept_cells.binding`` code,
and returns a pass/fail report against the >= 92% success / <= 0.10
false-fire targets. It accepts arbitrary embedding matrices so unit tests
can drive it with synthetic, deterministic data; the CLI wrapper feeds real
embeddings sampled from a bank.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .binding import bind_items, test_binding


# Roadmap targets at m = 8.
DEFAULT_MIN_SUCCESS_RATE = 0.92
DEFAULT_MAX_FALSE_FIRES_PER_TRIAL = 0.10


@dataclass(frozen=True)
class BindingTrialOutcome:
    trial_index: int
    m: int
    success: bool
    items_fired: int
    n_distractor_false_fires: int
    final_alignment_to_mean: float


@dataclass(frozen=True)
class BindingCalibrationReport:
    m: int
    n_trials: int
    distractor_pool_size: int
    items_per_trial: int
    success_rate: float
    mean_false_fires_per_trial: float
    success_target: float
    false_fires_target: float
    success_passes: bool
    false_fires_passes: bool
    trial_outcomes: List[BindingTrialOutcome]
    notes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.success_passes and self.false_fires_passes

    def to_dict(self) -> Dict[str, object]:
        d = {
            "m": self.m,
            "n_trials": self.n_trials,
            "distractor_pool_size": self.distractor_pool_size,
            "items_per_trial": self.items_per_trial,
            "success_rate": self.success_rate,
            "mean_false_fires_per_trial": self.mean_false_fires_per_trial,
            "success_target": self.success_target,
            "false_fires_target": self.false_fires_target,
            "success_passes": self.success_passes,
            "false_fires_passes": self.false_fires_passes,
            "passed": self.passed,
            "trial_outcomes": [asdict(o) for o in self.trial_outcomes],
            "notes": list(self.notes),
        }
        return d


def _unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        raise ValueError("cannot unit-normalise a zero vector")
    return vec / n


def _calibrated_threshold(
    w_final: np.ndarray,
    items: Sequence[np.ndarray],
    safety_margin: float,
) -> float:
    """Per ARCHITECTURE.md §Policies: theta_readout = min(w·x) - safety."""
    projections = [float(w_final @ x) for x in items]
    return min(projections) - safety_margin


def run_binding_trials(
    *,
    embeddings: np.ndarray,
    m: int,
    n_trials: int,
    distractor_sample_size: int,
    theta_write: float = 0.30,
    safety_margin: float = 0.02,
    alpha: float = 1.0,
    dt: float = 0.05,
    n_steps: int = 300,
    seed: int = 0,
    min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    max_false_fires_per_trial: float = DEFAULT_MAX_FALSE_FIRES_PER_TRIAL,
) -> BindingCalibrationReport:
    """Run ``n_trials`` of m-way binding against a distractor sample.

    The function is pure on its inputs. ``embeddings`` is the full bank-like
    matrix; each trial samples ``m`` items to bind plus
    ``distractor_sample_size`` items to test for false fires.

    Numeric trace:
      embeddings.shape = (10000, 384), m = 8
      Trial 0: bind items[idx0..idx7]; with calibrated threshold the cell
        fires for all 8 -> success=True. Distractor sample has 1 false fire.
      Across 50 trials: 47 successes -> success_rate = 47/50 = 0.94
      0.94 >= 0.92 -> success_passes = True
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D; got shape {embeddings.shape}")
    n_emb, _ = embeddings.shape
    if m < 1:
        raise ValueError(f"m must be >= 1; got {m}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1; got {n_trials}")
    if distractor_sample_size < 0:
        raise ValueError(
            f"distractor_sample_size must be >= 0; got {distractor_sample_size}"
        )
    if n_emb < m + distractor_sample_size + 1:
        raise ValueError(
            f"embeddings pool of size {n_emb} cannot supply {m} bound + "
            f"{distractor_sample_size} distractors without overlap"
        )

    rng = np.random.default_rng(seed)
    outcomes: List[BindingTrialOutcome] = []
    successes = 0
    false_fire_total = 0

    for trial in range(n_trials):
        perm = rng.permutation(n_emb)
        bind_idx = perm[:m]
        dist_idx = perm[m : m + distractor_sample_size]

        items = [embeddings[i] for i in bind_idx]
        distractors = [embeddings[i] for i in dist_idx]
        anchor = items[0]
        w0 = _unit(anchor.astype(np.float64))

        w_final, _ = bind_items(
            w0=w0,
            items=items,
            theta=theta_write,
            alpha=alpha,
            dt=dt,
            n_steps=n_steps,
        )
        theta_final = _calibrated_threshold(w_final, items, safety_margin)
        result = test_binding(
            items=items,
            distractors=distractors,
            w_initial=w0,
            w_final=w_final,
            theta_initial=theta_write,
            theta_final=theta_final,
        )
        outcomes.append(
            BindingTrialOutcome(
                trial_index=trial,
                m=m,
                success=bool(result.success),
                items_fired=sum(result.fires_per_item),
                n_distractor_false_fires=int(result.n_distractor_false_fires),
                final_alignment_to_mean=float(result.final_alignment_to_mean),
            )
        )
        if result.success:
            successes += 1
        false_fire_total += int(result.n_distractor_false_fires)

    success_rate = successes / n_trials
    mean_false_fires = false_fire_total / n_trials
    notes: List[str] = []
    if distractor_sample_size == 0:
        notes.append(
            "distractor_sample_size=0 means false-fire rate cannot be measured; "
            "selectivity is unconstrained."
        )
    if n_emb < 10_000:
        notes.append(
            f"distractor pool drawn from only {n_emb:,} embeddings; "
            "production bank is 5.7M cells, so this run is a lower-bound stress test."
        )
    return BindingCalibrationReport(
        m=m,
        n_trials=n_trials,
        distractor_pool_size=distractor_sample_size,
        items_per_trial=m,
        success_rate=success_rate,
        mean_false_fires_per_trial=mean_false_fires,
        success_target=min_success_rate,
        false_fires_target=max_false_fires_per_trial,
        success_passes=success_rate >= min_success_rate,
        false_fires_passes=mean_false_fires <= max_false_fires_per_trial,
        trial_outcomes=outcomes,
        notes=notes,
    )


def render_calibration_markdown(report: BindingCalibrationReport) -> str:
    lines: List[str] = []
    lines.append("# Binding calibration report")
    lines.append("")
    lines.append(
        f"- m = **{report.m}**, trials = **{report.n_trials}**, "
        f"distractor pool per trial = **{report.distractor_pool_size:,}**"
    )
    lines.append(
        f"- success rate: **{report.success_rate:.3f}** "
        f"(target >= {report.success_target:.2f}) "
        f"-> **{'PASS' if report.success_passes else 'FAIL'}**"
    )
    lines.append(
        f"- mean false-fires per trial: **{report.mean_false_fires_per_trial:.3f}** "
        f"(target <= {report.false_fires_target:.2f}) "
        f"-> **{'PASS' if report.false_fires_passes else 'FAIL'}**"
    )
    lines.append("")
    lines.append(f"## Overall: **{'PASS' if report.passed else 'FAIL'}**")
    if report.notes:
        lines.append("")
        lines.append("## Notes")
        for n in report.notes:
            lines.append(f"- {n}")
    return "\n".join(lines) + "\n"
