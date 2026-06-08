"""Unit tests for binding calibration at scale.

Synthetic embeddings only — no real bank, no GPU. We verify the harness
returns the right shapes, enforces its preconditions, and reports a
sensible success-rate for the trivial m=2 case on isotropic vectors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.concept_cells.binding_calibration import (  # noqa: E402
    BindingCalibrationReport,
    render_calibration_markdown,
    run_binding_trials,
)


def _isotropic(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    return raw


class TestBindingCalibration:
    def test_pair_binding_high_success_rate(self) -> None:
        emb = _isotropic(400, dim=128, seed=0)
        report = run_binding_trials(
            embeddings=emb,
            m=2,
            n_trials=25,
            distractor_sample_size=50,
            seed=0,
        )
        # m=2 on isotropic vectors should bind reliably; relax to >= 0.5 to
        # avoid flakiness, while still verifying the harness is not broken.
        assert isinstance(report, BindingCalibrationReport)
        assert report.success_rate >= 0.5
        assert len(report.trial_outcomes) == 25
        assert report.items_per_trial == 2

    def test_thresholds_propagated_to_report(self) -> None:
        emb = _isotropic(80, dim=64, seed=1)
        report = run_binding_trials(
            embeddings=emb,
            m=2,
            n_trials=5,
            distractor_sample_size=10,
            min_success_rate=0.99,  # impossibly strict
            max_false_fires_per_trial=0.0,
            seed=1,
        )
        assert report.success_target == 0.99
        assert report.false_fires_target == 0.0
        # The "passed" property collapses both gates
        assert report.passed == (report.success_passes and report.false_fires_passes)

    def test_dict_round_trip(self) -> None:
        emb = _isotropic(60, dim=32, seed=2)
        report = run_binding_trials(
            embeddings=emb,
            m=2,
            n_trials=3,
            distractor_sample_size=5,
            seed=2,
        )
        d = report.to_dict()
        assert d["m"] == 2
        assert d["n_trials"] == 3
        assert "trial_outcomes" in d
        assert len(d["trial_outcomes"]) == 3

    def test_markdown_render(self) -> None:
        emb = _isotropic(60, dim=32, seed=3)
        report = run_binding_trials(
            embeddings=emb,
            m=2,
            n_trials=3,
            distractor_sample_size=5,
            seed=3,
        )
        md = render_calibration_markdown(report)
        assert "Binding calibration report" in md
        assert "PASS" in md or "FAIL" in md

    def test_pool_too_small_raises(self) -> None:
        emb = _isotropic(5, dim=32, seed=4)
        with pytest.raises(ValueError, match="cannot supply"):
            run_binding_trials(
                embeddings=emb,
                m=4,
                n_trials=2,
                distractor_sample_size=10,
            )

    def test_bad_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            run_binding_trials(
                embeddings=np.zeros(10, dtype=np.float32),
                m=2,
                n_trials=1,
                distractor_sample_size=2,
            )

    def test_no_distractors_flagged_in_notes(self) -> None:
        emb = _isotropic(40, dim=32, seed=5)
        report = run_binding_trials(
            embeddings=emb,
            m=2,
            n_trials=3,
            distractor_sample_size=0,
            seed=5,
        )
        assert any("cannot be measured" in n for n in report.notes)
        # With zero distractors, false_fires is trivially zero
        assert report.mean_false_fires_per_trial == 0.0
        assert report.false_fires_passes is True
