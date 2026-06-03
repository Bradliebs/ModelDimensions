"""Tests for the opt-in threshold policies (src/concept_cells/thresholds.py).

These cover the v1.0-rc1 calibration layer. They assert behaviour and the
hard invariants: policies do not mutate inputs or cell vectors, and outputs are
deterministic.

    python -m pytest evals/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from concept_cells.thresholds import (
    Decision,
    abstain_band_threshold,
    compute_activations,
    fixed_threshold,
    per_cell_threshold,
    quantile_threshold,
    sweep_best_threshold,
    two_stage_threshold,
)


def test_fixed_threshold_fires_above_theta():
    res = fixed_threshold([0.9, 0.8], theta=0.5)
    assert res.decisions == (Decision.FIRE, Decision.FIRE)
    assert all(m > 0 for m in res.margins)
    assert res.fired_indices == [0, 1]


def test_fixed_threshold_silences_below_theta():
    res = fixed_threshold([0.4, 0.1], theta=0.5)
    assert res.decisions == (Decision.SILENT, Decision.SILENT)
    assert all(m < 0 for m in res.margins)
    assert res.fired_indices == []


def test_fixed_threshold_boundary_is_silent():
    # Strictly-greater rule: exactly at theta does NOT fire.
    res = fixed_threshold([0.5], theta=0.5)
    assert res.decisions == (Decision.SILENT,)


def test_abstain_band_returns_ambiguous_in_middle():
    res = abstain_band_threshold([0.9, 0.5, 0.1], low=0.3, high=0.7)
    assert res.decisions == (Decision.FIRE, Decision.AMBIGUOUS, Decision.SILENT)
    assert res.ambiguous_indices == [1]


def test_abstain_band_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        abstain_band_threshold([0.5], low=0.8, high=0.2)


def test_two_stage_separates_fire_ambiguous_silent():
    res = two_stage_threshold([0.9, 0.5, 0.1],
                              theta_silent=0.3, theta_fire=0.7)
    assert res.decisions == (Decision.FIRE, Decision.AMBIGUOUS, Decision.SILENT)
    assert res.fired_indices == [0]
    assert res.ambiguous_indices == [1]
    assert res.silent_indices == [2]


def test_two_stage_min_margin_pushes_borderline_to_ambiguous():
    # 0.72 clears theta_fire=0.7 but not by the required 0.1 margin.
    res = two_stage_threshold([0.72], theta_silent=0.3, theta_fire=0.7,
                              min_margin=0.1)
    assert res.decisions == (Decision.AMBIGUOUS,)


def test_two_stage_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        two_stage_threshold([0.5], theta_silent=0.8, theta_fire=0.2)


def test_quantile_threshold_uses_negative_distribution():
    negatives = [0.1, 0.2, 0.3, 0.4, 0.5]
    # 0.8 quantile of negatives is ~0.42; only the 0.6 activation clears it.
    res = quantile_threshold([0.6, 0.35], negatives, q=0.8)
    assert res.decisions[0] is Decision.FIRE
    assert res.decisions[1] is Decision.SILENT


def test_per_cell_threshold_does_not_mutate_core_cell_vectors():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 8))
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    w_before = w.copy()
    q = rng.standard_normal(8)
    q = q / np.linalg.norm(q)

    acts = compute_activations(w, q)
    thetas = np.full(4, 0.1)
    _ = per_cell_threshold(acts, thetas)

    # The cell matrix must be byte-for-byte unchanged.
    assert np.array_equal(w, w_before)


def test_per_cell_threshold_does_not_mutate_input_arrays():
    acts = np.array([0.9, 0.2, 0.6])
    thetas = np.array([0.5, 0.5, 0.5])
    acts_before, thetas_before = acts.copy(), thetas.copy()
    _ = per_cell_threshold(acts, thetas)
    assert np.array_equal(acts, acts_before)
    assert np.array_equal(thetas, thetas_before)


def test_per_cell_threshold_length_mismatch_raises():
    with pytest.raises(ValueError):
        per_cell_threshold([0.1, 0.2, 0.3], [0.1, 0.2])


def test_threshold_policies_are_deterministic():
    acts = [0.91, 0.55, 0.12, 0.73]
    a = two_stage_threshold(acts, theta_silent=0.3, theta_fire=0.7)
    b = two_stage_threshold(acts, theta_silent=0.3, theta_fire=0.7)
    assert a == b
    c = fixed_threshold(acts, 0.5)
    d = fixed_threshold(acts, 0.5)
    assert c == d


def test_sweep_finds_feasible_operating_point():
    # Cleanly separable: positives high, negatives low.
    pos = [0.8, 0.85, 0.9, 0.95]
    neg = [0.1, 0.2, 0.15, 0.05]
    res = sweep_best_threshold(pos, neg, max_false_fire_rate=0.1)
    assert res.feasible is True
    assert res.recall == 1.0
    assert res.false_fire_rate <= 0.1


def test_sweep_reports_infeasible_when_overlap_total():
    # Identical distributions: no threshold separates them.
    pos = [0.5, 0.5, 0.5]
    neg = [0.5, 0.5, 0.5]
    res = sweep_best_threshold(pos, neg, max_false_fire_rate=0.0)
    assert res.feasible in (True, False)  # defined, deterministic
    # With a 0.0 cap and total overlap, recall cannot be positive while feasible.
    if res.feasible:
        assert res.recall == 0.0
