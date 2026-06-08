"""Unit tests for ZCA whitening verification.

Constructs deliberately anisotropic synthetic embeddings, applies ZCA, and
verifies effective-dim climbs and pairwise cosine drops. No bank required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.concept_cells.zca_verification import (  # noqa: E402
    ZcaVerificationReport,
    render_verification_markdown,
    verify_whitening,
)


def _anisotropic(n: int, dim: int, *, axis_scale: float, seed: int) -> np.ndarray:
    """Random Gaussians, then squash all but a few axes to make the data
    nearly-low-rank."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    scales = np.ones(dim, dtype=np.float32) * axis_scale
    scales[: max(1, dim // 8)] = 1.0  # keep ~12.5% of axes at full scale
    raw *= scales
    return raw


class TestZcaVerification:
    def test_whitening_raises_effective_dim(self) -> None:
        # The production target of 0.95 is calibrated for 5.7M real bank embeddings,
        # which is impractical to materialise in a unit test. We instead verify the
        # *direction* of the change: ZCA must demonstrably raise effective-dim and
        # lower pairwise cosine on held-out data drawn from the same distribution.
        fit = _anisotropic(4_000, dim=64, axis_scale=0.05, seed=0)
        ev = _anisotropic(1_000, dim=64, axis_scale=0.05, seed=1)
        report = verify_whitening(
            fit_embeddings=fit,
            eval_embeddings=ev,
            effective_dim_target=0.95,
            sample_pairs=5_000,
            seed=0,
        )
        assert isinstance(report, ZcaVerificationReport)
        # Anisotropic data must show low raw effective dim
        assert report.raw_report["effective_dim_fraction"] < 0.5
        # ZCA must raise effective dim by a substantial margin on held-out data
        improvement = (
            report.whitened_report["effective_dim_fraction"]
            - report.raw_report["effective_dim_fraction"]
        )
        assert improvement > 0.3, (
            f"ZCA improvement only {improvement:.3f}; "
            f"raw={report.raw_report['effective_dim_fraction']:.3f}, "
            f"whitened={report.whitened_report['effective_dim_fraction']:.3f}"
        )

    def test_whitening_lowers_pairwise_cosine(self) -> None:
        fit = _anisotropic(800, dim=64, axis_scale=0.05, seed=2)
        ev = _anisotropic(400, dim=64, axis_scale=0.05, seed=3)
        report = verify_whitening(
            fit_embeddings=fit,
            eval_embeddings=ev,
            sample_pairs=2_000,
            seed=2,
        )
        assert (
            report.whitened_report["abs_mean_pairwise_cosine"]
            <= report.raw_report["abs_mean_pairwise_cosine"] + 1e-6
        )

    def test_dim_mismatch_raises(self) -> None:
        fit = _anisotropic(100, dim=32, axis_scale=0.1, seed=4)
        ev = _anisotropic(100, dim=16, axis_scale=0.1, seed=5)
        with pytest.raises(ValueError, match="share dim"):
            verify_whitening(fit_embeddings=fit, eval_embeddings=ev)

    def test_insufficient_samples_raises(self) -> None:
        fit = np.random.default_rng(6).standard_normal((10, 32)).astype(np.float32)
        ev = np.random.default_rng(7).standard_normal((10, 32)).astype(np.float32)
        with pytest.raises(ValueError, match="need at least"):
            verify_whitening(fit_embeddings=fit, eval_embeddings=ev)

    def test_dict_round_trip(self) -> None:
        fit = _anisotropic(800, dim=64, axis_scale=0.05, seed=8)
        ev = _anisotropic(200, dim=64, axis_scale=0.05, seed=9)
        report = verify_whitening(
            fit_embeddings=fit, eval_embeddings=ev, sample_pairs=1_000, seed=8
        )
        d = report.to_dict()
        assert d["passed"] == report.passed
        assert "raw_report" in d and "whitened_report" in d

    def test_markdown_render(self) -> None:
        fit = _anisotropic(800, dim=64, axis_scale=0.05, seed=10)
        ev = _anisotropic(200, dim=64, axis_scale=0.05, seed=11)
        report = verify_whitening(
            fit_embeddings=fit, eval_embeddings=ev, sample_pairs=1_000, seed=10
        )
        md = render_verification_markdown(report)
        assert "ZCA whitening verification" in md
        assert "PASS" in md or "FAIL" in md
