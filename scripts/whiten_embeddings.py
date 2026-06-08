"""Apply ZCA whitening from a fit slice to an eval slice of embeddings.

Fits ZCA whitening on ``--fit`` (mean + decorrelating matrix) and applies
it to ``--apply-to``. Optionally re-unit-normalises each row of the
whitened output so downstream consumers that assume a unit-sphere bank
(e.g. ``concept_cells.binding``) keep working.

This is the bridge step between
``scripts/verify_zca_isotropy.py`` (which only measures isotropy) and
``scripts/calibrate_binding_at_scale.py`` (which expects an embedding
matrix). Numeric trace::

    fit.shape   = (20000, 384)
    eval.shape  = (20000, 384)
    fit covariance -> eigen-decomp -> W = V diag(1/sqrt(lam)) V.T
    whitened_eval = (eval - mean) @ W
    --unit-norm: each row rescaled to ||row|| = 1

Usage:
    python scripts/whiten_embeddings.py \\
        --fit results/bank_fit_emb.npy \\
        --apply-to results/bank_eval_emb.npy \\
        --out results/bank_eval_emb_zca.npy --unit-norm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fit_zca(embeddings: np.ndarray, *, eps: float = 1e-5):
    if embeddings.ndim != 2:
        raise SystemExit(f"fit embeddings must be 2D; got {embeddings.shape}")
    n, d = embeddings.shape
    if n < d + 1:
        raise SystemExit(f"need >= d+1 fit samples; got n={n} d={d}")
    mean = embeddings.mean(axis=0)
    centred = embeddings - mean
    cov = (centred.T @ centred) / (n - 1)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, eps, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return mean.astype(np.float64), W.astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--apply-to", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--unit-norm", action="store_true",
                        help="Rescale each whitened row to unit length")
    args = parser.parse_args(argv)

    for p in (args.fit, args.apply_to):
        if not p.exists():
            parser.error(f"file not found: {p}")

    fit = np.load(args.fit)
    target = np.load(args.apply_to)
    print(f"[whiten] fit {fit.shape}, apply-to {target.shape}")
    if fit.shape[1] != target.shape[1]:
        parser.error(
            f"dim mismatch: fit d={fit.shape[1]} vs apply-to d={target.shape[1]}"
        )

    mean, W = _fit_zca(fit, eps=args.eps)
    out = ((target - mean) @ W).astype(np.float32)

    # Verify the whitening on the fit set (sanity) and report on the apply set.
    fit_white = ((fit - mean) @ W).astype(np.float32)
    fit_cov = (fit_white - fit_white.mean(0)).T @ (fit_white - fit_white.mean(0)) / (fit.shape[0] - 1)
    fit_eig = np.linalg.eigvalsh(fit_cov)
    print(f"[whiten] fit  whitened eigvals: min={fit_eig.min():.3e} max={fit_eig.max():.3e}")
    out_cov = (out - out.mean(0)).T @ (out - out.mean(0)) / (out.shape[0] - 1)
    out_eig = np.linalg.eigvalsh(out_cov)
    print(f"[whiten] eval whitened eigvals: min={out_eig.min():.3e} max={out_eig.max():.3e}")

    raw_norm = np.linalg.norm(out, axis=1)
    print(f"[whiten] eval whitened row norm: mean={raw_norm.mean():.4f} "
          f"min={raw_norm.min():.4f} max={raw_norm.max():.4f}")

    if args.unit_norm:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        zero_rows = int((norms.squeeze() <= 1e-12).sum())
        if zero_rows:
            print(f"[whiten] WARNING: {zero_rows} zero-norm rows after whitening; "
                  f"leaving them un-normalised")
        safe = np.where(norms > 1e-12, norms, 1.0)
        out = (out / safe).astype(np.float32)
        check = np.linalg.norm(out, axis=1)
        print(f"[whiten] after unit-norm: mean norm={check.mean():.4f} "
              f"min={check.min():.4f} max={check.max():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f"[whiten] wrote {args.out}  shape={out.shape}  dtype={out.dtype}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
