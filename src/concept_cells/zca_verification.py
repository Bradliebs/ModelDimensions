"""ZCA whitening verification on real bank embeddings.

The Stochastic Separation Theorems (Tyukin & Gorban) require near-isotropic
embeddings. The roadmap target is to recover ~99% effective dimensionality
after whitening. This module wraps the existing geometry primitives in a
*measurement* layer: fit whitening on one slice of embeddings, apply it to
another, then compare effective-dimension and mean-cosine before vs after.

The function only does measurement. The CLI in
``scripts/verify_zca_isotropy.py`` is the one place that loads embeddings
from the production bank.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np

from .geometry import IsotropyReport, isotropy_report


# Roadmap target: effective dim / ambient dim >= 0.95 after whitening.
DEFAULT_EFFECTIVE_DIM_TARGET = 0.95
DEFAULT_PAIRWISE_COS_MAX = 0.05


def _fit_zca(embeddings: np.ndarray, *, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Fit ZCA whitening on the provided embeddings.

    Returns ``(mean, W)`` such that ``apply(x) = (x - mean) @ W`` is whitened.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D; got shape {embeddings.shape}")
    n, d = embeddings.shape
    if n < d + 1:
        raise ValueError(
            f"need at least d+1 samples to fit ZCA; got n={n} for d={d}"
        )
    mean = embeddings.mean(axis=0)
    centred = embeddings - mean
    cov = (centred.T @ centred) / (n - 1)
    # symmetric eigendecomposition; clamp negatives that arise from float noise
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, eps, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return mean.astype(np.float64), W.astype(np.float64)


def _apply(embeddings: np.ndarray, mean: np.ndarray, W: np.ndarray) -> np.ndarray:
    return ((embeddings - mean) @ W).astype(np.float32)


@dataclass(frozen=True)
class ZcaVerificationReport:
    n_fit_samples: int
    n_eval_samples: int
    raw_report: Dict[str, float]
    whitened_report: Dict[str, float]
    effective_dim_target: float
    pairwise_cos_max: float
    effective_dim_pass: bool
    pairwise_cos_pass: bool
    notes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.effective_dim_pass and self.pairwise_cos_pass

    def to_dict(self) -> Dict[str, object]:
        d = {
            "n_fit_samples": self.n_fit_samples,
            "n_eval_samples": self.n_eval_samples,
            "raw_report": dict(self.raw_report),
            "whitened_report": dict(self.whitened_report),
            "effective_dim_target": self.effective_dim_target,
            "pairwise_cos_max": self.pairwise_cos_max,
            "effective_dim_pass": self.effective_dim_pass,
            "pairwise_cos_pass": self.pairwise_cos_pass,
            "passed": self.passed,
            "notes": list(self.notes),
        }
        return d


def _report_to_dict(r: IsotropyReport) -> Dict[str, float]:
    return {
        "n": r.n,
        "dim": r.dim,
        "mean_norm": r.mean_norm,
        "std_norm": r.std_norm,
        "mean_pairwise_cosine": r.mean_pairwise_cosine,
        "std_pairwise_cosine": r.std_pairwise_cosine,
        "abs_mean_pairwise_cosine": r.abs_mean_pairwise_cosine,
        "participation_ratio": r.participation_ratio,
        "effective_dim_fraction": r.effective_dim_fraction,
    }


def verify_whitening(
    *,
    fit_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    effective_dim_target: float = DEFAULT_EFFECTIVE_DIM_TARGET,
    pairwise_cos_max: float = DEFAULT_PAIRWISE_COS_MAX,
    sample_pairs: int = 50_000,
    seed: int = 0,
) -> ZcaVerificationReport:
    """Fit ZCA on ``fit_embeddings``, apply it to ``eval_embeddings``, and
    report effective-dim and pairwise-cosine before vs after.

    Numeric trace:
      raw effective_dim_fraction = 0.25 (anisotropic transformer embeddings)
      after ZCA: effective_dim_fraction = 0.992
      0.992 >= 0.95  ->  effective_dim_pass = True
    """
    if fit_embeddings.ndim != 2 or eval_embeddings.ndim != 2:
        raise ValueError("fit and eval embeddings must be 2D")
    if fit_embeddings.shape[1] != eval_embeddings.shape[1]:
        raise ValueError(
            "fit and eval embeddings must share dim; got "
            f"{fit_embeddings.shape[1]} vs {eval_embeddings.shape[1]}"
        )

    raw = isotropy_report(eval_embeddings, sample_pairs=sample_pairs, seed=seed)
    mean, W = _fit_zca(fit_embeddings)
    whitened = _apply(eval_embeddings, mean, W)
    after = isotropy_report(whitened, sample_pairs=sample_pairs, seed=seed)

    notes: List[str] = []
    if fit_embeddings.shape[0] == eval_embeddings.shape[0]:
        notes.append(
            "fit and eval samples share size; consider supplying a held-out "
            "eval slice to avoid optimistic measurements."
        )
    if after.effective_dim_fraction < raw.effective_dim_fraction:
        notes.append(
            f"effective dim DECREASED after whitening "
            f"({raw.effective_dim_fraction:.3f} -> {after.effective_dim_fraction:.3f}); "
            "small sample size relative to dim likely overfits the whitening matrix."
        )

    return ZcaVerificationReport(
        n_fit_samples=int(fit_embeddings.shape[0]),
        n_eval_samples=int(eval_embeddings.shape[0]),
        raw_report=_report_to_dict(raw),
        whitened_report=_report_to_dict(after),
        effective_dim_target=effective_dim_target,
        pairwise_cos_max=pairwise_cos_max,
        effective_dim_pass=after.effective_dim_fraction >= effective_dim_target,
        pairwise_cos_pass=after.abs_mean_pairwise_cosine <= pairwise_cos_max,
        notes=notes,
    )


def render_verification_markdown(report: ZcaVerificationReport) -> str:
    lines: List[str] = []
    lines.append("# ZCA whitening verification")
    lines.append("")
    lines.append(
        f"- Fit samples: **{report.n_fit_samples:,}**, eval samples: **{report.n_eval_samples:,}**"
    )
    lines.append("")
    lines.append("|  | Raw | Whitened |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| Effective dim fraction | {report.raw_report['effective_dim_fraction']:.3f} | "
        f"**{report.whitened_report['effective_dim_fraction']:.3f}** |"
    )
    lines.append(
        f"| Mean |cosine| | {report.raw_report['abs_mean_pairwise_cosine']:.3f} | "
        f"**{report.whitened_report['abs_mean_pairwise_cosine']:.3f}** |"
    )
    lines.append(
        f"| Mean norm | {report.raw_report['mean_norm']:.3f} | "
        f"{report.whitened_report['mean_norm']:.3f} |"
    )
    lines.append("")
    lines.append(
        f"Effective-dim gate (>= {report.effective_dim_target:.2f}): "
        f"**{'PASS' if report.effective_dim_pass else 'FAIL'}**"
    )
    lines.append(
        f"Pairwise-cosine gate (<= {report.pairwise_cos_max:.2f}): "
        f"**{'PASS' if report.pairwise_cos_pass else 'FAIL'}**"
    )
    lines.append(f"Overall: **{'PASS' if report.passed else 'FAIL'}**")
    if report.notes:
        lines.append("")
        lines.append("## Notes")
        for n in report.notes:
            lines.append(f"- {n}")
    return "\n".join(lines) + "\n"
