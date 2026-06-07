"""Experiment 30: does ABTT recover paraphrase recall on real MiniLM?

Phase 6 shipped ABTT as an alternative whitening method based on Mu &
Viswanath (ICLR 2018) and exp08's documented ZCA-collapse on small-N
reference corpora. The Phase 6 unit tests are structural
(``evals/test_abtt_no_collapse.py``): they prove the math behaves as
specified on synthetic anisotropic data, not that ABTT actually rescues
paraphrase recall on real text embeddings.

This experiment closes that loop. It reuses the exact corpus from
``experiments.exp08_encoder_semantic_recall`` (5 memories, 20 queries
across {exact, paraphrase, near_miss, unrelated}) and feeds it through
three transforms over the same real all-MiniLM-L6-v2 vectors:

  1. ``minilm_raw``         : reference, no whitening.
  2. ``minilm_zca_scale``   : exp08's ZCA + unit-ball scale (the failing
                              regime).
  3. ``minilm_abtt_scale``  : ABTT + unit-ball scale (Phase 6 proposal).

For each config we record exact_recall, paraphrase_recall,
false_fire_rate, and silent_refusal_rate at exp08's epsilon=0.25 cosine
bar. The headline question is whether ``paraphrase_recall`` for
``minilm_abtt_scale`` is meaningfully greater than for
``minilm_zca_scale`` while keeping ``false_fire_rate`` at or below the
ZCA baseline. If yes, Phase 6's claim survives empirical contact with
real embeddings; if no, the RUNBOOK 14 evidence list needs a correction.

Writes ``results/exp30_abtt_minilm_validation.json``. If MiniLM cannot
be loaded the MiniLM configs are recorded as skipped, identical to
exp08's degradation path.

    python -m experiments.exp30_abtt_minilm_validation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.orchestrator import MemoryBank  # noqa: E402
from experiments.exp08_encoder_semantic_recall import (  # noqa: E402
    EPSILON,
    MEMORIES,
    PrecomputedEncoder,
    QUERIES,
    RADIUS,
    _fit_whiten_scale,
    _seed_bank,
    _summary_stats,
)


def _fit_abtt_scale(
    mem_emb: np.ndarray,
    k: Optional[int] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return an ABTT + unit-ball-scale transform fit on ``mem_emb``.

    Mirrors ``exp08._fit_whiten_scale`` so the two configs differ in one
    thing only: the projection step. Subtract the mean, project out the
    top-k principal components of the centered memories, then divide by
    the max post-projection memory norm so memories sit inside the unit
    ball. The transform closes over (mu, projection_matrix, max_norm)
    and is applied identically to memories and queries.

    Default ``k = max(1, D // 100)`` matches the paper's heuristic; we
    clamp it by ``min(k, D, max(1, N-1))`` so it stays well-defined at
    the N=5 memory regime.
    """
    n, d = mem_emb.shape
    if k is None:
        k = max(1, d // 100)
    k = min(k, d, max(1, n - 1))

    mu = mem_emb.mean(axis=0, keepdims=True)
    centered = mem_emb - mu
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top = vt[:k]                                   # (k, d)
    # I - V V^T applied to a centered row vector is (x - x @ V^T @ V).
    # We store V (top) and reuse it per call rather than materializing
    # the full (d, d) projector — this keeps the transform readable and
    # matches src/concept_cells/geometry.abtt_transform.
    proj_mem = centered - centered @ top.T @ top
    max_norm = float(np.linalg.norm(proj_mem, axis=1).max()) + 1e-8

    def transform(x: np.ndarray) -> np.ndarray:
        centered_x = x[None, :] - mu
        projected = centered_x - centered_x @ top.T @ top
        return (projected[0] / max_norm).astype(np.float32)

    return transform


def _build_mapping_minilm_with(
    transform_factory: Optional[Callable[[np.ndarray], Callable]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Encode every text once with MiniLM, optionally apply a transform
    fit on the memory embeddings, and return text -> vector.

    ``transform_factory`` mirrors the exp08 pattern: it takes the memory
    embeddings (np.float64) and returns a callable applied to each
    individual vector (memory or query). When ``None`` we return the raw
    MiniLM vectors. Returns ``None`` if MiniLM cannot be loaded — the
    caller records that config as skipped.
    """
    try:
        from concept_cells.encoders import TextEncoder  # noqa: WPS433
    except Exception:  # pragma: no cover - import guard
        return None
    try:
        encoder = TextEncoder("all-MiniLM-L6-v2")
        mem_texts = list(MEMORIES.values())
        query_texts = [q for _, _, q in QUERIES]
        all_texts = mem_texts + query_texts
        emb = encoder.encode(all_texts).embeddings.astype(np.float64)
    except Exception:
        return None

    mem_emb = emb[: len(mem_texts)]
    if transform_factory is None:
        vectors = emb
    else:
        transform = transform_factory(mem_emb)
        vectors = np.array([transform(emb[i]) for i in range(len(emb))])

    mapping: Dict[str, np.ndarray] = {}
    for text, vec in zip(list(MEMORIES.values()) + [q for _, _, q in QUERIES], vectors):
        mapping[text] = vec.astype(np.float32)
    return mapping


def run_config(name: str, encoder) -> dict:
    """Score one encoder config on the exp08 query set.

    Reuses exp08's bank seeding and the same epsilon/radius so results
    are directly comparable to ``results/exp08_summary.json``.
    """
    bank, id_for = _seed_bank(encoder)

    counts = {k: 0 for k in ("exact", "paraphrase", "near_miss", "unrelated")}
    exact_hits = paraphrase_hits = 0
    false_fires = false_fire_total = 0
    unrelated_total = unrelated_silent = 0
    fired_margins: List[float] = []
    margins_by_kind: Dict[str, List[float]] = {
        "exact": [], "paraphrase": [], "near_miss": [], "unrelated": [],
    }

    for cid, kind, qtext in QUERIES:
        counts[kind] += 1
        result = bank.query(qtext)
        target_id = id_for[cid]
        fired = result.fired_memory_ids
        fired_target = target_id in fired
        if fired:
            fired_margins.extend(result.margins)
            margins_by_kind[kind].extend(result.margins)
        if kind == "exact":
            exact_hits += int(fired_target)
        elif kind == "paraphrase":
            paraphrase_hits += int(fired_target)
        else:
            false_fire_total += 1
            if fired:
                false_fires += 1
            if kind == "unrelated":
                unrelated_total += 1
                if not fired:
                    unrelated_silent += 1

    def ratio(a: int, b: int) -> Optional[float]:
        return round(a / b, 4) if b else None

    per_kind_mean = {
        k: (round(float(np.mean(v)), 4) if v else None)
        for k, v in margins_by_kind.items()
    }
    return {
        "config": name,
        "epsilon": EPSILON,
        "cosine_fire_threshold": round(1.0 - EPSILON / RADIUS, 4),
        "exact_recall": ratio(exact_hits, counts["exact"]),
        "paraphrase_recall": ratio(paraphrase_hits, counts["paraphrase"]),
        "false_fire_rate": ratio(false_fires, false_fire_total),
        "silent_refusal_rate": ratio(unrelated_silent, unrelated_total),
        "margin_distribution": {
            **_summary_stats(fired_margins),
            "per_kind_mean": per_kind_mean,
        },
        "n_queries": len(QUERIES),
    }


def main() -> None:
    configs: List[dict] = []

    raw_map = _build_mapping_minilm_with(None)
    if raw_map is not None:
        configs.append(run_config("minilm_raw", PrecomputedEncoder(raw_map)))
    else:
        configs.append({
            "config": "minilm_raw",
            "skipped": "all-MiniLM-L6-v2 unavailable offline",
        })

    zca_map = _build_mapping_minilm_with(_fit_whiten_scale)
    if zca_map is not None:
        configs.append(
            run_config("minilm_zca_scale", PrecomputedEncoder(zca_map))
        )
    else:
        configs.append({
            "config": "minilm_zca_scale",
            "skipped": "all-MiniLM-L6-v2 unavailable offline",
        })

    abtt_map = _build_mapping_minilm_with(_fit_abtt_scale)
    if abtt_map is not None:
        configs.append(
            run_config("minilm_abtt_scale", PrecomputedEncoder(abtt_map))
        )
    else:
        configs.append({
            "config": "minilm_abtt_scale",
            "skipped": "all-MiniLM-L6-v2 unavailable offline",
        })

    # Headline: ABTT minus ZCA paraphrase recall, false-fire delta.
    by_name = {c["config"]: c for c in configs}
    zca = by_name.get("minilm_zca_scale", {})
    abtt = by_name.get("minilm_abtt_scale", {})
    delta: Dict[str, Optional[float]] = {}
    if "paraphrase_recall" in zca and "paraphrase_recall" in abtt:
        zr = zca["paraphrase_recall"]
        ar = abtt["paraphrase_recall"]
        delta["paraphrase_recall_abtt_minus_zca"] = (
            round(ar - zr, 4) if (zr is not None and ar is not None) else None
        )
    if "false_fire_rate" in zca and "false_fire_rate" in abtt:
        zf = zca["false_fire_rate"]
        af = abtt["false_fire_rate"]
        delta["false_fire_rate_abtt_minus_zca"] = (
            round(af - zf, 4) if (zf is not None and af is not None) else None
        )

    pr_delta = delta.get("paraphrase_recall_abtt_minus_zca")
    ff_delta = delta.get("false_fire_rate_abtt_minus_zca")
    if pr_delta is None or ff_delta is None:
        verdict = "indeterminate (a config was skipped or empty)"
    elif pr_delta > 0 and ff_delta <= 0:
        verdict = "ABTT improves paraphrase recall without raising false fires"
    elif pr_delta > 0 and ff_delta > 0:
        verdict = "ABTT improves paraphrase recall but raises false fires"
    elif pr_delta == 0 and ff_delta <= 0:
        verdict = "no paraphrase-recall change; ABTT is at least as safe as ZCA"
    elif pr_delta < 0:
        verdict = "ABTT REGRESSES paraphrase recall vs ZCA in this regime"
    else:
        verdict = "neutral"

    summary = {
        "experiment": "exp30_abtt_minilm_validation",
        "description": (
            "Real-MiniLM head-to-head of ZCA and ABTT on the exp08 corpus. "
            "Closes the loop the Phase 6 structural tests left open: does "
            "ABTT actually rescue paraphrase recall on real embeddings, "
            "not just synthetic anisotropic data?"
        ),
        "epsilon": EPSILON,
        "radius": RADIUS,
        "n_memories": len(MEMORIES),
        "n_queries": len(QUERIES),
        "configs": configs,
        "headline_delta": delta,
        "verdict": verdict,
    }

    out_path = ROOT / "results" / "exp30_abtt_minilm_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[exp30] wrote {out_path}")
    print(f"[exp30] verdict: {verdict}")


if __name__ == "__main__":
    main()
