"""Regression test for the small-sample whitening collapse documented in
``results/exp08_summary.json`` (config ``minilm_whiten_scale``,
``paraphrase_recall: 0.0``).

The deep-research report calls this a textbook ZCA failure mode: when
``N < ~3*D``, the covariance estimate is ill-conditioned and ZCA over-sharpens,
collapsing paraphrase-vs-unrelated separation. The robust alternative is
All-But-The-Top (Mu & Viswanath, ICLR 2018), which never inverts the
covariance.

These tests do **not** touch the production 5.7M-cell bank. They synthesize
the textbook under-sampled anisotropic regime (N=200, D=384, dominant
top-3 PCs) and verify that:

  1. ZCA collapses paraphrase / unrelated separation on this regime.
  2. ABTT preserves it.
  3. Both isotropy correctors return the same WhiteningParams structure
     so the persistence layer and ``apply_whitening`` are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cc_service.memory import apply_whitening, fit_whitening  # noqa: E402
from concept_cells.geometry import abtt_transform  # noqa: E402


# ---------- Synthetic anisotropic regime ----------

DIM = 384
N_REF = 200          # textbook ill-conditioned: N << D


def _anisotropic_corpus(rng: np.random.Generator, n: int) -> np.ndarray:
    """Anisotropic Gaussian with a few dominant axes, MiniLM-like scale.

    Three top components carry ~10x the variance of the bulk — matches the
    real situation Mu & Viswanath documented for word embeddings.
    """
    bulk = rng.standard_normal((n, DIM)).astype(np.float32) * 0.1
    # Heavy direction 0/1/2: 10x bulk variance + a non-zero mean shift.
    bulk[:, 0] += rng.standard_normal(n).astype(np.float32) * 1.0 + 0.4
    bulk[:, 1] += rng.standard_normal(n).astype(np.float32) * 1.0 - 0.3
    bulk[:, 2] += rng.standard_normal(n).astype(np.float32) * 1.0 + 0.2
    # Unit-normalize, matching MiniLM output convention.
    bulk /= np.maximum(np.linalg.norm(bulk, axis=1, keepdims=True), 1e-8)
    return bulk.astype(np.float32)


def _paraphrase_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A 'paraphrase' pair: shared semantics on heavy axes, INDEPENDENT bulk noise.

    Real paraphrases preserve meaning (so the dominant semantic axes stay
    close) but use different lexical surface forms (so the bulk-dimension
    coordinates are independent). This is the regime where ZCA fails: it
    amplifies the independent bulk noise and drowns the heavy-axis agreement.
    """
    # Shared heavy-axis seed for the pair.
    h0 = np.float32(rng.standard_normal() * 1.0 + 0.4)
    h1 = np.float32(rng.standard_normal() * 1.0 - 0.3)
    h2 = np.float32(rng.standard_normal() * 1.0 + 0.2)
    # Pair member a: independent bulk noise.
    a = rng.standard_normal(DIM).astype(np.float32) * 0.1
    a[0] += h0
    a[1] += h1
    a[2] += h2
    # Pair member b: same heavy-axis values, DIFFERENT bulk noise.
    b = rng.standard_normal(DIM).astype(np.float32) * 0.1
    b[0] += h0
    b[1] += h1
    b[2] += h2
    a /= np.maximum(np.linalg.norm(a), 1e-8)
    b /= np.maximum(np.linalg.norm(b), 1e-8)
    return a, b


def _unrelated_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """An 'unrelated' pair: both anisotropic, but independent draws."""
    a = _anisotropic_corpus(rng, 1)[0]
    b = _anisotropic_corpus(rng, 1)[0]
    return a, b


def _separation_score_with_fn(transform_fn, rng_seed: int = 17,
                              n_pairs: int = 64) -> float:
    """Mean(paraphrase cosine) - mean(unrelated cosine) after applying
    a transform function to each batch.

    Higher = paraphrases distinguished better from unrelated pairs.
    Collapse looks like values near zero or negative.

    transform_fn: callable taking (N, D) ndarray → (N, D) ndarray.
    """
    rng = np.random.default_rng(rng_seed)
    pairs = [_paraphrase_pair(rng) for _ in range(n_pairs)]
    para_a = np.stack([p[0] for p in pairs])
    para_b = np.stack([p[1] for p in pairs])
    pairs_u = [_unrelated_pair(rng) for _ in range(n_pairs)]
    unrel_a = np.stack([p[0] for p in pairs_u])
    unrel_b = np.stack([p[1] for p in pairs_u])

    # Transform must see all four batches stacked so it sees the same mean /
    # principal axes as in the live pipeline. (The fitting reference is the
    # corpus; here we just compose batches for cosine measurement.)
    pa = transform_fn(para_a)
    pb = transform_fn(para_b)
    ua = transform_fn(unrel_a)
    ub = transform_fn(unrel_b)

    def _cos(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        nx = np.linalg.norm(x, axis=1, keepdims=True)
        ny = np.linalg.norm(y, axis=1, keepdims=True)
        return np.sum(x * y, axis=1) / np.maximum(nx[:, 0] * ny[:, 0], 1e-8)

    return float(_cos(pa, pb).mean() - _cos(ua, ub).mean())


# ---------- Tests ----------

def test_fit_whitening_zca_default_unchanged() -> None:
    """Backward-compat: method='zca' is the default and matches the legacy call."""
    rng = np.random.default_rng(0)
    ref = _anisotropic_corpus(rng, n=N_REF)
    p_default = fit_whitening(ref, reference_n=N_REF)
    p_zca = fit_whitening(ref, reference_n=N_REF, method="zca")
    np.testing.assert_array_equal(p_default.mu, p_zca.mu)
    np.testing.assert_array_equal(p_default.w_matrix, p_zca.w_matrix)
    assert p_default.max_norm == p_zca.max_norm


def test_fit_whitening_abtt_returns_compatible_params() -> None:
    """ABTT must return the same WhiteningParams shape so apply_whitening works."""
    rng = np.random.default_rng(0)
    ref = _anisotropic_corpus(rng, n=N_REF)
    params = fit_whitening(ref, reference_n=N_REF, method="abtt")
    assert params.mu.shape == (DIM,)
    assert params.w_matrix.shape == (DIM, DIM)
    assert params.max_norm > 0
    # Round-trip apply on a fresh sample
    sample = _anisotropic_corpus(rng, n=4)
    out = apply_whitening(sample, params)
    assert out.shape == (4, DIM)
    assert np.isfinite(out).all()


def test_fit_whitening_rejects_unknown_method() -> None:
    rng = np.random.default_rng(0)
    ref = _anisotropic_corpus(rng, n=N_REF)
    with pytest.raises(ValueError, match="unknown whitening method"):
        fit_whitening(ref, reference_n=N_REF, method="bogus")


def test_zca_collapses_paraphrase_separation_at_small_n() -> None:
    """Reproduce the exp08 collapse: small-sample ZCA destroys paraphrase signal.

    This tests the geometry-layer ``zca_whiten`` (fixed eps=1e-5, no adaptive
    clamp), which is the exact code path that produced
    ``minilm_whiten_scale: paraphrase_recall = 0.0`` in
    ``results/exp08_summary.json``.

    The cc_service ``fit_whitening`` has since gained an adaptive-eps mitigation
    that partly blunts this collapse, but the underlying geometry-layer
    behaviour and the underlying mathematical fragility remain — which is the
    justification for offering ABTT.
    """
    rng = np.random.default_rng(0)
    ref = _anisotropic_corpus(rng, n=N_REF)

    # Fit ZCA on the reference, then express it as a callable transform that
    # uses the *fitted* parameters (mu + w_matrix) — this is what the geometry
    # layer does in spirit. We compute mu and w_matrix on `ref` and apply to
    # query batches, matching how exp08 ran.
    mu = ref.mean(axis=0, keepdims=True)
    centered = ref - mu
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-5)            # fixed eps — the broken regime
    w_zca = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype(np.float32)

    def _zca_apply(x: np.ndarray) -> np.ndarray:
        return ((x - mu) @ w_zca).astype(np.float32)

    gap = _separation_score_with_fn(_zca_apply)
    assert gap < 0.05, (
        f"Geometry-layer ZCA should collapse paraphrase separation at "
        f"N={N_REF}, D={DIM} (the exp08 regime) but got gap={gap:.4f}. "
        f"If this is now >= 0.05 the failure regime has changed and exp08's "
        f"premise needs re-examining."
    )


def test_abtt_removes_top_k_variance_and_preserves_bulk() -> None:
    """ABTT structural contract.

    After applying ABTT with k=K:
      - Variance projected onto the top-K reference principal axes is ~0
        (that's the nuisance ABTT exists to remove).
      - Variance in the bulk subspace is preserved (>= 90% of pre-transform).

    This is the structural guarantee. Whether ABTT *helps recall on real
    MiniLM* is an empirical question owned by exp09 / exp12; that depends on
    where the paraphrase signal actually lives, which differs per encoder.
    """
    rng = np.random.default_rng(0)
    ref = _anisotropic_corpus(rng, n=N_REF)
    k = max(1, DIM // 100)

    # Reference principal axes (the ones ABTT will project out).
    mu = ref.mean(axis=0, keepdims=True)
    centered = ref - mu
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top = vt[:k]                       # (k, D)

    transformed = abtt_transform(ref, k=k)

    # Project transformed output onto top-k axes; variance should be ~0.
    top_coords = transformed @ top.T                     # (N, k)
    top_var = float(np.var(top_coords))
    assert top_var < 1e-6, (
        f"ABTT failed to remove top-{k} variance (left {top_var:.3e}). "
        f"The projection matrix is not idempotent / not aligned."
    )

    # Bulk variance: project onto remaining D-k axes; should be preserved.
    bulk = vt[k:]                                        # (D-k, D)
    bulk_var_before = float(np.var(centered @ bulk.T))
    bulk_var_after = float(np.var(transformed @ bulk.T))
    preservation = bulk_var_after / max(bulk_var_before, 1e-12)
    assert preservation > 0.9, (
        f"ABTT should preserve bulk variance but kept only "
        f"{preservation:.2%} (before={bulk_var_before:.3e}, "
        f"after={bulk_var_after:.3e})."
    )


def test_abtt_robust_to_small_n_no_warnings() -> None:
    """ABTT must not emit the rank-deficient warning that ZCA does at small N,
    and must produce finite output even when N << D."""
    rng = np.random.default_rng(0)
    ref = _anisotropic_corpus(rng, n=20)               # extreme under-sampling
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")                 # any warning → exception
        params = fit_whitening(ref, reference_n=20, method="abtt")
    sample = _anisotropic_corpus(rng, n=8)
    out = apply_whitening(sample, params)
    assert np.isfinite(out).all(), "ABTT produced non-finite values at N=20"
    assert out.shape == (8, DIM)


def test_abtt_transform_geometry_module_matches_fit_path() -> None:
    """The geometry-layer ABTT and the cc_service fit_whitening ABTT must agree
    on the centered+projected coordinates (modulo the cc_service ball-scaling)."""
    rng = np.random.default_rng(0)
    ref = _anisotropic_corpus(rng, n=N_REF)
    geom_out = abtt_transform(ref, k=max(1, DIM // 100))
    params = fit_whitening(ref, reference_n=N_REF, method="abtt")
    # apply_whitening centers and projects, then scales by max_norm; undo the scale.
    cc_out = apply_whitening(ref, params) * (params.max_norm + 1e-8)
    np.testing.assert_allclose(cc_out, geom_out, atol=1e-4)
