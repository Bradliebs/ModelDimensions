"""Opt-in threshold policies for concept-cell firing decisions.

These policies are EXPERIMENTAL and additive. They do **not** replace the firing
rule used by the v0.8/v0.9 core: ``MemoryBank`` and
``geometry.build_concept_cells`` still decide firing by the single rule
``activation > theta``. Experiments (Exp 09) import these to explore calibrated,
quantile, abstain-band, and two-stage policies without modifying the frozen
core.

Contract for every policy here:

  * Pure function of ``(activations, parameters)``.
  * Never mutates its inputs, and never touches cell weight vectors.
  * Deterministic: identical inputs produce identical outputs.

An "activation" is the dot product ``<w_i, q>`` between a cell weight and a
query vector. Policies operate on these scalars, so they are agnostic to how the
vectors were produced (deterministic encoder, MiniLM, whitened MiniLM, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray, float]


class Decision(str, Enum):
    """Per-cell firing decision. ``str`` mixin keeps it JSON-friendly."""

    FIRE = "fire"
    AMBIGUOUS = "ambiguous"
    SILENT = "silent"


@dataclass(frozen=True)
class ThresholdDecision:
    """Outcome of a threshold policy over a set of cell activations.

    ``decisions[i]`` is the call for cell ``i``; ``margins[i]`` is that cell's
    signed distance from the relevant firing boundary (positive => above it).
    The object is frozen so results cannot be edited after the fact.
    """

    decisions: tuple
    margins: tuple

    @property
    def fired_indices(self) -> List[int]:
        return [i for i, d in enumerate(self.decisions) if d is Decision.FIRE]

    @property
    def ambiguous_indices(self) -> List[int]:
        return [i for i, d in enumerate(self.decisions)
                if d is Decision.AMBIGUOUS]

    @property
    def silent_indices(self) -> List[int]:
        return [i for i, d in enumerate(self.decisions) if d is Decision.SILENT]

    @property
    def any_fired(self) -> bool:
        return any(d is Decision.FIRE for d in self.decisions)

    @property
    def any_ambiguous(self) -> bool:
        return any(d is Decision.AMBIGUOUS for d in self.decisions)


# ---------- helpers (copy inputs; never mutate) ----------

def _activations(activations: ArrayLike) -> np.ndarray:
    """Coerce to a 1-D float64 array, copied so the caller's data is safe."""
    return np.asarray(activations, dtype=np.float64).reshape(-1).copy()


def _broadcast_theta(theta: ArrayLike, n: int) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(
            f"theta has {arr.size} entries but there are {n} activations"
        )
    return arr.copy()


def compute_activations(weights: np.ndarray, query: ArrayLike) -> np.ndarray:
    """Read-only ``W @ q`` convenience that never mutates ``weights``.

    Provided so experiments and tests can derive activations from cell vectors
    without any risk of the policy layer altering the (frozen) cell matrix.
    """
    w = np.asarray(weights, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    return (w @ q).astype(np.float64)


# ---------- policies ----------

def fixed_threshold(activations: ArrayLike, theta: float) -> ThresholdDecision:
    """Single global threshold: FIRE iff ``activation > theta``, else SILENT."""
    acts = _activations(activations)
    thetas = _broadcast_theta(theta, acts.size)
    margins = acts - thetas
    decisions = tuple(
        Decision.FIRE if m > 0.0 else Decision.SILENT for m in margins
    )
    return ThresholdDecision(decisions, tuple(float(m) for m in margins))


def per_cell_threshold(activations: ArrayLike,
                       thetas: ArrayLike) -> ThresholdDecision:
    """Per-cell calibrated thresholds: FIRE iff ``activation_i > theta_i``.

    ``thetas`` must have one entry per activation. Only activations and
    thresholds are read; no cell weight vector is referenced or modified.
    """
    acts = _activations(activations)
    th = _broadcast_theta(thetas, acts.size)
    margins = acts - th
    decisions = tuple(
        Decision.FIRE if m > 0.0 else Decision.SILENT for m in margins
    )
    return ThresholdDecision(decisions, tuple(float(m) for m in margins))


def quantile_threshold(activations: ArrayLike,
                       calibration_negatives: ArrayLike,
                       q: float = 0.95,
                       theta_floor: Optional[float] = None) -> ThresholdDecision:
    """Threshold set at the ``q``-quantile of a negative calibration set.

    The idea: pick the firing bar so that at most ``1 - q`` of known negatives
    would have crossed it. ``theta_floor`` optionally clamps the bar from below.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    acts = _activations(activations)
    neg = np.asarray(calibration_negatives, dtype=np.float64).reshape(-1)
    thr = float(np.quantile(neg, q)) if neg.size else 0.0
    if theta_floor is not None:
        thr = max(thr, float(theta_floor))
    margins = acts - thr
    decisions = tuple(
        Decision.FIRE if m > 0.0 else Decision.SILENT for m in margins
    )
    return ThresholdDecision(decisions, tuple(float(m) for m in margins))


def abstain_band_threshold(activations: ArrayLike, low: float,
                           high: float) -> ThresholdDecision:
    """Three-way band: FIRE above ``high``, SILENT below ``low``, else AMBIGUOUS.

    Margins are reported relative to the FIRE boundary (``activation - high``).
    """
    if high < low:
        raise ValueError("high must be >= low")
    acts = _activations(activations)
    decisions: List[Decision] = []
    for a in acts:
        if a >= high:
            decisions.append(Decision.FIRE)
        elif a <= low:
            decisions.append(Decision.SILENT)
        else:
            decisions.append(Decision.AMBIGUOUS)
    margins = tuple(float(a - high) for a in acts)
    return ThresholdDecision(tuple(decisions), margins)


def two_stage_threshold(activations: ArrayLike, theta_silent: float,
                        theta_fire: float,
                        min_margin: float = 0.0) -> ThresholdDecision:
    """Two-stage gate.

    Stage 1 (coarse reject): any activation ``<= theta_silent`` is SILENT.
    Stage 2 (confident accept): of the survivors, FIRE only if the activation
    clears ``theta_fire`` by at least ``min_margin``; otherwise AMBIGUOUS.

    This separates confident hits (FIRE) from borderline matches (AMBIGUOUS)
    from clear misses (SILENT). Margins are reported relative to ``theta_fire``.
    """
    if theta_fire < theta_silent:
        raise ValueError("theta_fire must be >= theta_silent")
    acts = _activations(activations)
    decisions: List[Decision] = []
    for a in acts:
        if a <= theta_silent:
            decisions.append(Decision.SILENT)
        elif a - theta_fire >= min_margin:
            decisions.append(Decision.FIRE)
        else:
            decisions.append(Decision.AMBIGUOUS)
    margins = tuple(float(a - theta_fire) for a in acts)
    return ThresholdDecision(tuple(decisions), margins)


# ---------- threshold selection (for calibration experiments) ----------

@dataclass(frozen=True)
class ThresholdSweepResult:
    """Best threshold found over a positive/negative score sweep."""

    threshold: float
    recall: float
    false_fire_rate: float
    youden_j: float
    feasible: bool  # met the false-fire-rate constraint


def sweep_best_threshold(positive_scores: ArrayLike,
                         negative_scores: ArrayLike,
                         max_false_fire_rate: float = 0.1,
                         n_grid: int = 256) -> ThresholdSweepResult:
    """Pick the firing threshold that maximises recall under a false-fire cap.

    ``positive_scores`` are activations of queries against the cell that SHOULD
    fire; ``negative_scores`` are activations of queries that should fire
    nothing (typically the max activation over all cells). The sweep scans a
    grid spanning both distributions and returns the threshold with the highest
    recall whose false-fire rate is ``<= max_false_fire_rate``. If no threshold
    satisfies the cap, it returns the one maximising Youden's J with
    ``feasible=False``.
    """
    pos = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    neg = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if pos.size == 0:
        return ThresholdSweepResult(float("inf"), 0.0, 0.0, 0.0, False)

    lo = float(min(pos.min(), neg.min() if neg.size else pos.min()))
    hi = float(max(pos.max(), neg.max() if neg.size else pos.max()))
    if hi <= lo:
        hi = lo + 1e-6
    grid = np.linspace(lo - 1e-9, hi + 1e-9, n_grid)

    best_feasible: Optional[ThresholdSweepResult] = None
    best_overall: Optional[ThresholdSweepResult] = None

    for t in grid:
        recall = float((pos > t).mean())
        ffr = float((neg > t).mean()) if neg.size else 0.0
        youden = recall - ffr
        cand = ThresholdSweepResult(float(t), round(recall, 4),
                                    round(ffr, 4), round(youden, 4),
                                    ffr <= max_false_fire_rate)
        if best_overall is None or youden > best_overall.youden_j:
            best_overall = cand
        if cand.feasible:
            # Prefer higher recall, then lower false-fire, then higher threshold.
            if (best_feasible is None
                    or recall > best_feasible.recall
                    or (recall == best_feasible.recall
                        and ffr < best_feasible.false_fire_rate)):
                best_feasible = cand

    return best_feasible if best_feasible is not None else best_overall
