"""MemoryBank: the running bank.

Brings together:
  - the encoder (loaded once)
  - whitening parameters (fitted once at init time, applied to every embedding)
  - the SQLite-backed cell store (persistence.BankStore)
  - the architecture primitives (write, query, bind)

The MemoryBank is stateful within a process — it caches the (W, theta, cell_ids)
matrix between writes so query doesn't have to re-read from SQLite. The cache
is invalidated on any write or delete.

This is single-process. Multiple workers would race on writes; use a single
uvicorn worker for now.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .encoder import EncoderSingleton
from .reranker import RerankerSingleton
from .persistence import BankStore, CellRow, WhiteningParams


# ---------- Bind & threshold helpers (vendored from concept_cells.binding) ----------
# We re-implement the small core here rather than depending on the experimental
# `concept_cells` package layout — keeps the service standalone.

def _oja_step(w: np.ndarray, s: np.ndarray, theta: float,
              alpha: float, dt: float) -> np.ndarray:
    y = float(w @ s)
    v = max(y - theta, 0.0)
    if v <= 0.0:
        return w
    dw = alpha * v * y * (s - w * y)
    return w + dt * dw


def _bind_items(w0: np.ndarray, items: List[np.ndarray], theta: float,
                alpha: float, dt: float, n_steps: int) -> np.ndarray:
    w = w0.copy().astype(np.float64)
    s_bar = np.sum(items, axis=0).astype(np.float64)
    for _ in range(n_steps):
        w = _oja_step(w, s_bar, theta, alpha, dt)
    return w.astype(np.float32)


# ---------- Whitening fitting ----------

def fit_whitening(raw_embeddings: np.ndarray,
                   reference_n: int,
                   eps: float = 1e-5,
                   method: str = "zca",
                   abtt_k: Optional[int] = None) -> WhiteningParams:
    """Fit an isotropy-correction transform from a reference corpus.

    The returned ``WhiteningParams`` always has shape ``(mu, w_matrix, max_norm)``
    so the existing :func:`apply_whitening` and persistence layer do not need
    to know which method was used.

    method:
      - ``"zca"`` (default, legacy): ZCA whitening. ``w_matrix = cov^(-1/2)``.
        Requires a well-conditioned covariance estimate; if ``N < ~3*D`` it
        over-sharpens (textbook small-sample failure, observed in exp08 as
        ``paraphrase_recall = 0.0``).
      - ``"abtt"``: All-But-The-Top (Mu & Viswanath, ICLR 2018). Subtract
        global mean, project out the top ``abtt_k`` principal components.
        Default ``abtt_k = max(1, D // 100)``. Robust to small ``N`` because
        it never inverts the covariance — only its top-k eigenspace is used.

    The reference set must be large enough to give a well-conditioned
    covariance estimate for ZCA. As a rule of thumb, you want N >= 3 * D
    samples; below that the rank-deficient directions get whitening factors
    that explode for out-of-reference vectors. We warn but don't refuse below
    that threshold. ABTT does not suffer this issue.
    """
    N, D = raw_embeddings.shape
    mu = raw_embeddings.mean(axis=0)
    centered = raw_embeddings - mu

    if method == "zca":
        if N < D:
            import warnings
            warnings.warn(
                f"Whitening fitted on {N} samples in {D} dims; covariance is "
                f"rank-deficient. New vectors with components along null "
                f"directions will be amplified ~{1/np.sqrt(eps):.0f}x. "
                f"Recommend at least {3*D} samples for stable whitening, "
                f"or use method='abtt' which is robust to small N.",
                stacklevel=2,
            )
        cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Adaptive eps: clamp to a meaningful fraction of the largest eigenvalue,
        # not an absolute number. This prevents null directions from getting
        # whitening factors that explode for out-of-distribution components.
        eps_effective = max(eps, float(eigvals.max()) * 1e-3)
        eigvals = np.maximum(eigvals, eps_effective)
        w_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    elif method == "abtt":
        k = abtt_k if abtt_k is not None else max(1, D // 100)
        k = min(k, D, max(1, N - 1))
        # Top-k right singular vectors of centered = top-k eigenvectors of cov.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        top = vt[:k]                                   # (k, D)
        w_matrix = np.eye(D, dtype=np.float64) - top.T @ top
    else:
        raise ValueError(f"unknown whitening method: {method!r} (use 'zca' or 'abtt')")

    w_matrix = w_matrix.astype(np.float32)

    # Compute max_norm after transform so we can scale into the unit ball.
    # We include a small safety margin so that out-of-reference vectors are
    # unlikely to exceed unit norm catastrophically.
    transformed = centered @ w_matrix
    max_norm_observed = float(np.linalg.norm(transformed, axis=1).max())
    max_norm = max_norm_observed * 1.5  # 50% headroom for OOD samples

    return WhiteningParams(
        mu=mu.astype(np.float32),
        w_matrix=w_matrix,
        max_norm=max_norm,
        reference_n=int(reference_n),
        fitted_at=time.time(),
    )


def apply_whitening(raw: np.ndarray, params: WhiteningParams) -> np.ndarray:
    """Apply the frozen whitening + ball scaling.

    raw can be (D,) or (N, D). Output has same shape.
    """
    single = (raw.ndim == 1)
    if single:
        raw = raw[None, :]
    centered = raw - params.mu[None, :]
    whitened = centered @ params.w_matrix
    scaled = whitened / (params.max_norm + 1e-8)
    out = scaled.astype(np.float32)
    return out[0] if single else out


# ---------- Results ----------

@dataclass
class QueryHit:
    cell_id: int
    label: Optional[str]
    kind: str
    activation: float
    margin: float          # activation - theta_at_query (positive = fired)
    source_text: Optional[str]
    source_cell_ids: Optional[List[int]]
    rerank_score: Optional[float] = None  # stage-2 cross-encoder score, if reranked


@dataclass
class BindResult:
    bound_cell_id: int
    source_cell_ids: List[int]
    theta_readout: float
    alignment_to_mean: float
    items_fire_after_binding: List[bool]


# ---------- MemoryBank ----------

class MemoryBank:
    """The running memory.

    Lifecycle:
      bank = MemoryBank(db_path="bank.db")
      bank.init_from_corpus(reference_texts)        # one-time
      bank.write("some note")                       # repeatable
      hits = bank.query("some question")            # repeatable
      bank.bind([id1, id2, id3], label="topic")     # optional
    """

    # Bank parameters with reasonable defaults from the experiments.
    # Theta_write is the per-cell firing gate: a query hit is "silent" (and
    # suppressed by default) when activation <= theta. Calibrated on the BGE
    # bank against measured activation distributions (calibrate_theta.py):
    # foreign/no-answer queries peak at ~0.19, true partial-mention matches sit
    # at p10~0.26 / median~0.34. 0.22 lies in the clean band between them, so
    # genuine hits fire by default while unrelated queries still return empty.
    # (Was 0.30, inherited from MiniLM geometry, which gated out ~25% of real
    # matches.) See theta_calibration.json / ARCHITECTURE.md.
    DEFAULT_THETA_WRITE = 0.22
    # Oja's-rule binding init is a separate concern, validated at 0.30; kept
    # decoupled from the query gate so recalibration here doesn't shift binding.
    DEFAULT_THETA_BIND = 0.30
    DEFAULT_ALPHA = 1.0
    DEFAULT_DT = 0.05
    DEFAULT_N_STEPS = 300
    DEFAULT_CALIBRATION_MARGIN = 0.02

    # When reranking, how many stage-1 candidates to rescore by default.
    DEFAULT_CANDIDATE_POOL = 50

    # Rerank rejection gate: cross-encoder scores are sigmoid-squashed to [0, 1].
    # Measured on this bank (probe_rerank.py): relevant query/passage pairs score
    # ~0.99-1.00, unrelated/nonsense ~0.00-0.01. 0.10 sits in the wide empty band
    # between them, so the rerank path returns [] for queries with no real match
    # (preserving the bank's rejection property at stage 2 instead of via theta).
    DEFAULT_RERANK_MIN_SCORE = 0.10

    def __init__(self, db_path: str | Path,
                 encoder_model: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 reranker_model: Optional[str] = None,
                 encoder: Optional[object] = None):
        # `encoder=` lets tests (and future alternate runtimes) inject an
        # encoder instance directly, skipping the default EncoderSingleton.
        # The injected object only needs .model_name, .dim, .encode(),
        # .encode_one() — the same duck-typed surface MemoryBank already uses.
        self.store = BankStore(db_path)
        self.encoder = encoder if encoder is not None else EncoderSingleton(
            model_name=encoder_model, device=device,
        )
        self.reranker = (
            RerankerSingleton(model_name=reranker_model, device=device)
            if reranker_model else None
        )
        _min_env = os.environ.get("CCMEM_RERANK_MIN_SCORE")
        self.rerank_min_score = (
            float(_min_env) if _min_env not in (None, "")
            else self.DEFAULT_RERANK_MIN_SCORE
        )
        self._whitening: Optional[WhiteningParams] = None
        self._cache_valid = False
        self._W: Optional[np.ndarray] = None
        self._theta: Optional[np.ndarray] = None
        self._cell_ids: Optional[List[int]] = None
        self._load_whitening_if_present()
        self._verify_encoder_identity()

    # ---- Lifecycle ----

    def _load_whitening_if_present(self) -> None:
        params = self.store.load_whitening()
        if params is not None:
            self._whitening = params

    def _verify_encoder_identity(self) -> None:
        """Refuse to operate on a bank built with a different encoder.

        The stored weight vectors are specific to the encoder + whitening that
        produced them. Pointing a different encoder at an existing bank yields
        silent geometric garbage (queries land in a foreign space). We compare
        the model name recorded at init time and fail loudly on mismatch. Fresh
        banks (no recorded encoder yet) pass through untouched.
        """
        stored = self.store.get_meta("encoder_model")
        if stored is not None and stored != self.encoder.model_name:
            raise RuntimeError(
                f"Encoder mismatch: bank was built with '{stored}' but the "
                f"service is configured with '{self.encoder.model_name}'. "
                f"Querying would return garbage. Either set "
                f"CCMEM_ENCODER='{stored}' or rebuild the bank with the new "
                f"encoder into a fresh database."
            )

    @property
    def is_initialized(self) -> bool:
        return self._whitening is not None

    def init_from_corpus(self, reference_texts: List[str]) -> WhiteningParams:
        """One-time initialization: fit whitening from a reference corpus.

        Once fitted, the parameters are frozen and stored. Subsequent calls
        will *overwrite* the parameters — but if the bank already has cells,
        their stored weights become geometrically incompatible. Don't do this
        unless you're starting over.
        """
        if self.store.count() > 0:
            raise RuntimeError(
                "Cannot re-initialize whitening: bank already has "
                f"{self.store.count()} cells. Delete the database file first."
            )
        if len(reference_texts) < 200:
            raise ValueError(
                f"Need at least 200 reference texts for stable whitening, "
                f"got {len(reference_texts)}."
            )

        raw = self.encoder.encode(reference_texts)
        params = fit_whitening(raw, reference_n=len(reference_texts))
        # Record the encoder dim in meta BEFORE saving whitening (load needs it)
        self.store.set_meta("dim", int(self.encoder.dim))
        self.store.set_meta("encoder_model", self.encoder.model_name)
        self.store.set_meta("theta_write_default", self.DEFAULT_THETA_WRITE)
        self.store.save_whitening(params)
        self._whitening = params
        return params

    def _require_init(self) -> None:
        if self._whitening is None:
            raise RuntimeError(
                "Memory bank not initialized. Run `init_from_corpus` first."
            )

    # ---- Write ----

    def write(self, text: str, label: Optional[str] = None,
              theta: Optional[float] = None) -> int:
        """Write a single item as a new concept cell.

        Returns the cell_id.
        """
        self._require_init()
        raw = self.encoder.encode_one(text)
        x = apply_whitening(raw, self._whitening)
        # Unit-sphere init (the validated architectural decision from Exp 04 v0.6)
        norm = float(np.linalg.norm(x))
        if norm < 1e-6:
            raise ValueError("Encoded item has near-zero norm; cannot write.")
        w = (x / norm).astype(np.float32)
        theta = theta if theta is not None else self.DEFAULT_THETA_WRITE
        cell_id = self.store.insert_cell(
            label=label, weight=w, theta=theta, kind="single",
            source_text=text,
        )
        # Incremental cache update — append instead of full reload
        if self._cache_valid and self._W is not None:
            self._W = np.vstack([self._W, w.reshape(1, -1)])
            self._theta = np.append(self._theta, theta)
            self._cell_ids.append(cell_id)
        else:
            self._cache_valid = False
        return cell_id

    def write_many(self, texts: List[str],
                   labels: Optional[List[Optional[str]]] = None,
                   theta: Optional[float] = None) -> List[int]:
        """Write many items in one batch. Returns list of cell_ids."""
        self._require_init()
        if labels is None:
            labels = [None] * len(texts)
        if len(labels) != len(texts):
            raise ValueError("labels must be same length as texts")
        raw = self.encoder.encode(texts)
        x = apply_whitening(raw, self._whitening)
        theta = theta if theta is not None else self.DEFAULT_THETA_WRITE
        cell_ids = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            norm = float(np.linalg.norm(x[i]))
            if norm < 1e-6:
                continue
            w = (x[i] / norm).astype(np.float32)
            cell_ids.append(self.store.insert_cell(
                label=label, weight=w, theta=theta, kind="single",
                source_text=text,
            ))
        self._cache_valid = False
        return cell_ids

    # ---- Query ----

    def _refresh_cache(self) -> None:
        self._W, self._theta, self._cell_ids = self.store.all_weights_and_thresholds()
        self._cache_valid = True

    def query(self, text: str, top_k: int = 10,
              include_silent: bool = False,
              rerank: Optional[bool] = None,
              candidate_pool: Optional[int] = None,
              rerank_min_score: Optional[float] = None) -> List[QueryHit]:
        """Query the bank with a text. Returns the matching cells.

        Stage 1 (bi-encoder): the bank computes activations and selects a
        candidate set. By default returns only cells that *fired*
        (activation > theta); with include_silent=True returns top by
        activation regardless.

        Stage 2 (cross-encoder, optional): when a reranker is configured and
        ``rerank`` is not False, a recall pool of ``candidate_pool`` cells (by
        activation) is rescored by the cross-encoder and the top_k are returned
        ordered by that score. This bypasses the bi-encoder margin ceiling.
        """
        self._require_init()
        if not self._cache_valid:
            self._refresh_cache()
        if self._W.shape[0] == 0:
            return []

        raw = self.encoder.encode_one(text, is_query=True)
        q = apply_whitening(raw, self._whitening)

        activations = self._W @ q  # (N,)
        margins = activations - self._theta

        do_rerank = (self.reranker is not None) if rerank is None else bool(rerank)
        if do_rerank and self.reranker is None:
            do_rerank = False  # asked for it but none configured

        if do_rerank:
            return self._query_reranked(
                text, activations, margins, top_k,
                candidate_pool or self.DEFAULT_CANDIDATE_POOL,
                rerank_min_score if rerank_min_score is not None
                else self.rerank_min_score,
            )

        if include_silent:
            order = np.argsort(-activations)[:top_k]
        else:
            firing = np.where(margins > 0)[0]
            # Among firing, sort by margin descending
            order = firing[np.argsort(-margins[firing])][:top_k]

        hits: List[QueryHit] = []
        for idx in order:
            cell_id = self._cell_ids[idx]
            row = self.store.get_cell(cell_id)
            hits.append(QueryHit(
                cell_id=cell_id,
                label=row.label,
                kind=row.kind,
                activation=float(activations[idx]),
                margin=float(margins[idx]),
                source_text=row.source_text,
                source_cell_ids=row.source_cell_ids,
            ))
        return hits

    def _query_reranked(self, text: str, activations: np.ndarray,
                        margins: np.ndarray, top_k: int,
                        candidate_pool: int,
                        min_score: float) -> List[QueryHit]:
        """Stage-2 rerank: rescore the top stage-1 candidates by activation.

        The candidate pool is taken by raw activation (recall, not firing) so a
        relevant-but-low-margin cell still gets a chance at the cross-encoder.
        Candidates without source text (e.g. bound cells) fall back to label.
        Candidates scoring below ``min_score`` are dropped, so a query with no
        genuine match returns [] rather than its noisiest neighbours.
        """
        pool = min(max(candidate_pool, top_k), activations.shape[0])
        cand_idx = np.argsort(-activations)[:pool]

        rows = [self.store.get_cell(self._cell_ids[i]) for i in cand_idx]
        cand_texts = [
            (r.source_text or r.label or "") for r in rows
        ]
        scores = self.reranker.score(text, cand_texts)
        rerank_order = np.argsort(-scores)[:top_k]

        hits: List[QueryHit] = []
        for j in rerank_order:
            if float(scores[j]) < min_score:
                continue
            idx = cand_idx[j]
            row = rows[j]
            hits.append(QueryHit(
                cell_id=self._cell_ids[idx],
                label=row.label,
                kind=row.kind,
                activation=float(activations[idx]),
                margin=float(margins[idx]),
                source_text=row.source_text,
                source_cell_ids=row.source_cell_ids,
                rerank_score=float(scores[j]),
            ))
        return hits

    # ---- Bind ----

    def bind(self, source_cell_ids: List[int],
             label: Optional[str] = None,
             n_steps: int = None,
             safety_margin: float = None) -> BindResult:
        """Bind multiple existing cells into one composite cell.

        The new cell's weight vector is the result of running Oja's rule
        starting from the first source cell and co-presenting the rest.
        Threshold is calibrated post-hoc to fire for each source item.

        Returns the bound cell's ID and diagnostic info.
        """
        self._require_init()
        if len(source_cell_ids) < 2:
            raise ValueError("Need at least 2 source cells to bind.")
        if len(source_cell_ids) > 8:
            # Architecture validated up to m=8; beyond that we're extrapolating.
            # We allow it but caller should know.
            pass

        n_steps = n_steps if n_steps is not None else self.DEFAULT_N_STEPS
        safety_margin = (safety_margin if safety_margin is not None
                          else self.DEFAULT_CALIBRATION_MARGIN)

        # Reconstruct the items (which means re-encoding their source texts and
        # re-applying whitening — we don't store the post-whitening embeddings
        # separately because they're recoverable from text + whitening params).
        # For "bound" source cells we use the stored weight directly.
        items = []
        for cid in source_cell_ids:
            row = self.store.get_cell(cid)
            if row is None:
                raise ValueError(f"Cell {cid} does not exist")
            if row.kind == "single" and row.source_text is not None:
                raw = self.encoder.encode_one(row.source_text)
                x = apply_whitening(raw, self._whitening)
                items.append(x)
            else:
                # Bound cell or single without source text — use the stored
                # weight as a stand-in for the item direction. Less precise
                # but functional.
                items.append(row.weight)

        # Initialize w on the unit sphere aligned with the first item.
        anchor = items[0]
        anchor_norm = float(np.linalg.norm(anchor))
        if anchor_norm < 1e-6:
            raise ValueError("Anchor item has near-zero norm.")
        w0 = (anchor / anchor_norm).astype(np.float32)

        # Run binding with the validated binding init (theta_init = 0.30
        # throughout the architecture validation), independent of the query gate.
        theta_init = self.DEFAULT_THETA_BIND
        w_final = _bind_items(
            w0, items, theta=theta_init,
            alpha=self.DEFAULT_ALPHA, dt=self.DEFAULT_DT, n_steps=n_steps,
        )

        # Calibrated readout threshold: just below the minimum projection of
        # the bound items onto w_final.
        projections = np.array([float(w_final @ x) for x in items])
        theta_readout = float(max(projections.min() - safety_margin, 0.0))

        # Diagnostics
        x_bar = np.mean(items, axis=0)
        x_bar_unit = x_bar / max(np.linalg.norm(x_bar), 1e-12)
        w_unit = w_final / max(np.linalg.norm(w_final), 1e-12)
        alignment = float(w_unit @ x_bar_unit)
        items_fire = [(p > theta_readout) for p in projections]

        bound_cell_id = self.store.insert_cell(
            label=label, weight=w_final, theta=theta_readout, kind="bound",
            source_text=None, source_cell_ids=source_cell_ids,
        )
        self._cache_valid = False

        return BindResult(
            bound_cell_id=bound_cell_id,
            source_cell_ids=source_cell_ids,
            theta_readout=theta_readout,
            alignment_to_mean=alignment,
            items_fire_after_binding=items_fire,
        )

    # ---- Misc ----

    def get_cell(self, cell_id: int) -> Optional[CellRow]:
        return self.store.get_cell(cell_id)

    def list_cells(self, limit: int = 100, offset: int = 0,
                   kind: Optional[str] = None) -> List[CellRow]:
        return self.store.list_cells(limit=limit, offset=offset, kind=kind)

    def delete_cell(self, cell_id: int) -> bool:
        ok = self.store.delete_cell(cell_id)
        if ok and self._cache_valid and self._cell_ids is not None:
            # Incremental cache update — remove instead of full reload
            try:
                idx = self._cell_ids.index(cell_id)
                self._W = np.delete(self._W, idx, axis=0)
                self._theta = np.delete(self._theta, idx)
                self._cell_ids.pop(idx)
            except ValueError:
                self._cache_valid = False
        elif ok:
            self._cache_valid = False
        return ok

    def count(self) -> int:
        return self.store.count()

    def info(self) -> dict:
        return {
            "db_path": str(self.store.db_path),
            "encoder_model": self.encoder.model_name,
            "dim": self.encoder.dim,
            "initialized": self.is_initialized,
            "n_cells": self.count(),
            "whitening_fitted_at": (self._whitening.fitted_at
                                      if self._whitening else None),
            "whitening_reference_n": (self._whitening.reference_n
                                        if self._whitening else None),
        }
