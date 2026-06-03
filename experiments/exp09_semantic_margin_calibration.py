"""Experiment 09: semantic margin calibration.

The v1.0-rc1 question: can MiniLM-based concept cells support *paraphrase*
recall while keeping false fires acceptably low? v0.9 (Exp 08) showed raw MiniLM
lifts paraphrase recall but false-fires too often, and that whitening fit on a
handful of memories over-sharpens and kills paraphrase recall. Exp 09 widens the
search over encoders, preprocessing, and -- critically -- threshold choice.

It does NOT change the frozen core. Cell construction still goes through
``concept_cells.geometry.build_concept_cells``; Oja binding, write/query, and the
grounding policy are untouched. The whitening transforms here are fit locally and
applied to memories and queries alike; they mirror the core math but never
mutate it. Threshold sweeping uses the opt-in ``concept_cells.thresholds``
helpers.

Encoder / preprocessing variants:
  - deterministic                : content-addressed offline encoder (control).
  - minilm_raw                   : all-MiniLM-L6-v2, no preprocessing.
  - minilm_whiten_small          : ZCA whiten+scale fit on the MEMORIES only
                                   (reproduces the v0.9 small-sample setting).
  - minilm_whiten_large_calib    : ZCA whiten+scale fit on a larger background
                                   calibration corpus.
  - minilm_pca_whiten            : PCA whitening fit on the calibration corpus.
  - minilm_shrinkage_whiten      : shrinkage-covariance whitening on the corpus.
  - mpnet768_raw                 : all-mpnet-base-v2 (768d) raw, IF available.

Query categories: exact, paraphrase, near_miss, unrelated, adversarial.

Per variant, a threshold is chosen by sweeping for maximum recall under a
false-fire cap, then these metrics are reported AT that threshold:
  exact_recall, paraphrase_recall, near_miss_false_fire_rate,
  unrelated_false_fire_rate, adversarial_false_fire_rate, silence_rate,
  median_positive_margin, p05_positive_margin, p95_negative_margin,
  positive_negative_margin_overlap, best_threshold (cosine).

Writes results/exp09_summary.json and names an operating point if one variant
meets the acceptance bar, or states that none did.

    python -m experiments.exp09_semantic_margin_calibration
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.orchestrator import DeterministicEncoder
from concept_cells.geometry import build_concept_cells
from concept_cells.thresholds import sweep_best_threshold

# Sweep constraint and the acceptance bar for a usable operating point.
SWEEP_MAX_FALSE_FIRE = 0.10  # combined negative false-fire cap during the sweep
ACCEPT = {
    "min_paraphrase_recall": 0.60,
    "max_near_miss_ffr": 0.20,
    "max_unrelated_ffr": 0.05,
    "max_adversarial_ffr": 0.30,
}

# ---------- corpus ----------
# memory_id -> stored sentence (the cell's content).
MEMORIES: Dict[str, str] = {
    "launch": "the product launch is scheduled for Friday afternoon",
    "budget": "the project budget is forty thousand dollars",
    "server": "the primary server is located in the Dublin data center",
    "meeting": "the client meeting is on Tuesday morning",
    "rotation": "the database password rotates every ninety days",
    "ceo": "the chief executive will visit the London office next month",
    "release": "version two of the mobile app ships in September",
    "backup": "nightly backups are stored in the Frankfurt region",
}
MEM_ORDER = list(MEMORIES.keys())
MEM_INDEX = {cid: i for i, cid in enumerate(MEM_ORDER)}

# (memory_id, kind, query_text). kind in
# {exact, paraphrase, near_miss, unrelated, adversarial}.
QUERIES: List[Tuple[str, str, str]] = [
    # launch
    ("launch", "exact", "the product launch is scheduled for Friday afternoon"),
    ("launch", "paraphrase", "we are releasing the product on Friday afternoon"),
    ("launch", "paraphrase", "the product goes live on Friday after lunch"),
    ("launch", "near_miss", "the product launch is scheduled for Monday afternoon"),
    ("launch", "adversarial", "the product recall is scheduled for Friday afternoon"),
    ("launch", "unrelated", "the cat slept on the warm windowsill all day"),
    # budget
    ("budget", "exact", "the project budget is forty thousand dollars"),
    ("budget", "paraphrase", "we have forty thousand dollars allocated for the project"),
    ("budget", "paraphrase", "the project has a forty-thousand-dollar budget"),
    ("budget", "near_miss", "the project budget is ninety thousand dollars"),
    ("budget", "adversarial", "the project deficit is forty thousand dollars"),
    ("budget", "unrelated", "the train departs from platform nine every hour"),
    # server
    ("server", "exact", "the primary server is located in the Dublin data center"),
    ("server", "paraphrase", "our main server lives in the Dublin data centre"),
    ("server", "paraphrase", "the primary server runs out of the Dublin facility"),
    ("server", "near_miss", "the primary server is located in the Frankfurt data center"),
    ("server", "adversarial", "the primary server is relocated from the Dublin data center"),
    ("server", "unrelated", "she painted the fence a bright shade of green"),
    # meeting
    ("meeting", "exact", "the client meeting is on Tuesday morning"),
    ("meeting", "paraphrase", "we are meeting the client on Tuesday in the morning"),
    ("meeting", "paraphrase", "there is a morning client meeting on Tuesday"),
    ("meeting", "near_miss", "the client meeting is on Thursday morning"),
    ("meeting", "adversarial", "the client meeting was cancelled on Tuesday morning"),
    ("meeting", "unrelated", "the recipe needs two cups of flour and one egg"),
    # rotation
    ("rotation", "exact", "the database password rotates every ninety days"),
    ("rotation", "paraphrase", "the db credentials are rotated on a ninety day cycle"),
    ("rotation", "paraphrase", "we change the database password every ninety days"),
    ("rotation", "near_miss", "the database password rotates every thirty days"),
    ("rotation", "adversarial", "the database password never rotates after ninety days"),
    ("rotation", "unrelated", "the marathon route passes three city parks"),
    # ceo
    ("ceo", "exact", "the chief executive will visit the London office next month"),
    ("ceo", "paraphrase", "the CEO is coming to the London office next month"),
    ("ceo", "paraphrase", "next month our chief executive visits the London branch"),
    ("ceo", "near_miss", "the chief executive will visit the Paris office next month"),
    ("ceo", "adversarial", "the chief executive will leave the London office next month"),
    ("ceo", "unrelated", "the orchestra tuned their instruments before the concert"),
    # release
    ("release", "exact", "version two of the mobile app ships in September"),
    ("release", "paraphrase", "the mobile app's second version launches in September"),
    ("release", "paraphrase", "v2 of the app is due out in September"),
    ("release", "near_miss", "version three of the mobile app ships in September"),
    ("release", "adversarial", "version two of the mobile app slips past September"),
    ("release", "unrelated", "the bakery sells out of croissants by noon"),
    # backup
    ("backup", "exact", "nightly backups are stored in the Frankfurt region"),
    ("backup", "paraphrase", "we keep nightly backups in the Frankfurt region"),
    ("backup", "paraphrase", "backups run each night and live in Frankfurt"),
    ("backup", "near_miss", "nightly backups are stored in the Dublin region"),
    ("backup", "adversarial", "nightly backups failed to reach the Frankfurt region"),
    ("backup", "unrelated", "the hikers reached the summit before sunrise"),
]

# Background calibration corpus: in-domain but unrelated to the memories. Used
# only to FIT whitening statistics for the large-calibration / PCA / shrinkage
# variants (never used as memories or queries).
CALIBRATION_BACKGROUND: List[str] = [
    "the quarterly report is due at the end of the month",
    "please update the shared spreadsheet before noon",
    "the printer on the third floor is out of toner",
    "our flight to the conference leaves early on Wednesday",
    "the new intern starts next Monday in the design team",
    "remember to submit your expenses by Friday",
    "the wifi in the lobby has been unreliable lately",
    "the contractor will repaint the meeting rooms this weekend",
    "sales figures rose slightly in the second quarter",
    "the support queue is shorter than it was last week",
    "the parking garage closes at eleven at night",
    "we ordered new monitors for the engineering desks",
    "the fire drill is scheduled for Thursday afternoon",
    "the coffee machine needs descaling again",
    "her presentation ran ten minutes over the slot",
    "the warehouse inventory was counted last Tuesday",
    "the legal team reviewed the new vendor contract",
    "the elevator inspection happens twice a year",
    "the cafeteria added a new vegetarian option",
    "the security badges were reissued company wide",
    "the marketing email had a higher open rate this time",
    "the staircase lights flicker on the second floor",
    "the onboarding documents were translated into French",
    "the shipment arrived a day earlier than expected",
    "the team retrospective surfaced several small issues",
    "the audit found nothing of concern this cycle",
    "the projector bulb burned out during the demo",
    "the recruiter scheduled five interviews for Friday",
    "the office plants need watering twice a week",
    "the newsletter goes out on the first of each month",
    "the customer left a detailed review of the service",
    "the maintenance window is planned for Sunday night",
]


# ---------- preprocessing transforms (mirror core math; core untouched) ----------

def _identity(x: np.ndarray) -> np.ndarray:
    return x


def _fit_zca(fit_emb: np.ndarray, eps: float = 1e-5,
             scale: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    """ZCA whitening + unit-ball scaling, fit on ``fit_emb``.

    Mirrors ``geometry.zca_whiten`` + ``scale_to_unit_ball`` but returns a
    reusable transform so queries get the same mapping as memories.
    """
    mu = fit_emb.mean(axis=0, keepdims=True)
    centered = fit_emb - mu
    cov = (centered.T @ centered) / max(len(fit_emb) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, eps)
    w_mat = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    whitened = centered @ w_mat
    max_norm = (float(np.linalg.norm(whitened, axis=1).max()) + 1e-8
                if scale else 1.0)

    def transform(x: np.ndarray) -> np.ndarray:
        return (((x[None, :] - mu) @ w_mat) / max_norm)[0]

    return transform


def _fit_pca_whiten(fit_emb: np.ndarray, eps: float = 1e-5,
                    n_components: Optional[int] = None
                    ) -> Callable[[np.ndarray], np.ndarray]:
    """PCA whitening: project onto eigenbasis and scale by 1/sqrt(eigval).

    Unlike ZCA this leaves the data in the eigenbasis (no rotation back), which
    can decorrelate more aggressively. Fit on ``fit_emb``.
    """
    mu = fit_emb.mean(axis=0, keepdims=True)
    centered = fit_emb - mu
    cov = (centered.T @ centered) / max(len(fit_emb) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if n_components is not None:
        evals = evals[:n_components]
        evecs = evecs[:, :n_components]
    evals = np.maximum(evals, eps)
    w_mat = evecs @ np.diag(1.0 / np.sqrt(evals))  # (D, k)
    whitened = centered @ w_mat
    max_norm = float(np.linalg.norm(whitened, axis=1).max()) + 1e-8

    def transform(x: np.ndarray) -> np.ndarray:
        return (((x[None, :] - mu) @ w_mat) / max_norm)[0]

    return transform


def _fit_shrinkage_whiten(fit_emb: np.ndarray, alpha: float = 0.2,
                          eps: float = 1e-5
                          ) -> Callable[[np.ndarray], np.ndarray]:
    """ZCA whitening on a shrinkage-regularised covariance.

    The covariance is shrunk toward a scaled identity:
    ``cov_s = (1 - alpha) * cov + alpha * (trace(cov)/D) * I``. Shrinkage keeps
    the whitening matrix well-conditioned when the fit set is small relative to
    the dimension -- the failure mode that wrecked the small-sample variant.
    """
    mu = fit_emb.mean(axis=0, keepdims=True)
    centered = fit_emb - mu
    d = fit_emb.shape[1]
    cov = (centered.T @ centered) / max(len(fit_emb) - 1, 1)
    target = (np.trace(cov) / d) * np.eye(d)
    cov_s = (1.0 - alpha) * cov + alpha * target
    evals, evecs = np.linalg.eigh(cov_s)
    evals = np.maximum(evals, eps)
    w_mat = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    whitened = centered @ w_mat
    max_norm = float(np.linalg.norm(whitened, axis=1).max()) + 1e-8

    def transform(x: np.ndarray) -> np.ndarray:
        return (((x[None, :] - mu) @ w_mat) / max_norm)[0]

    return transform


# ---------- encoding ----------

def _all_texts() -> List[str]:
    texts = list(MEMORIES.values())
    texts += [q for _, _, q in QUERIES]
    texts += CALIBRATION_BACKGROUND
    # de-duplicate, preserve order
    seen = set()
    out = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _encode_deterministic() -> Dict[str, np.ndarray]:
    enc = DeterministicEncoder(dim=64)
    return {t: enc.encode_one(t).astype(np.float64) for t in _all_texts()}


def _encode_with_model(model_name: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        from concept_cells.encoders import TextEncoder
    except Exception:
        return None
    try:
        enc = TextEncoder(model_name)
        texts = _all_texts()
        emb = enc.encode(texts).embeddings.astype(np.float64)
    except Exception:
        return None
    return {t: emb[i] for i, t in enumerate(texts)}


# ---------- evaluation ----------

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values), p)), 4)


def evaluate_variant(name: str, text_to_vec: Dict[str, np.ndarray],
                     method: str, fit_texts: List[str]) -> dict:
    fit_emb = np.array([text_to_vec[t] for t in fit_texts], dtype=np.float64)
    if method == "none":
        transform = _identity
    elif method == "zca":
        transform = _fit_zca(fit_emb)
    elif method == "pca":
        transform = _fit_pca_whiten(fit_emb)
    elif method == "shrinkage":
        transform = _fit_shrinkage_whiten(fit_emb)
    else:
        raise ValueError(f"unknown method {method!r}")

    processed = {t: _unit(transform(np.asarray(v, dtype=np.float64)))
                 for t, v in text_to_vec.items()}

    mem_unit = np.array([processed[MEMORIES[cid]] for cid in MEM_ORDER])
    # Frozen-core cell construction. With unit memories, w == mem_unit.
    w, _theta = build_concept_cells(mem_unit)

    pos_scores: List[float] = []
    # Sweep negatives are the *realistic* false-fire sources (near_miss +
    # unrelated). Adversarial near-duplicates (one-word semantic flips) are a
    # separate stress category: they are reported but do not gate the chosen
    # threshold, since defending against them is a distinct, harder problem.
    sweep_neg_scores: List[float] = []
    by_kind: Dict[str, List[float]] = {
        "exact": [], "paraphrase": [], "near_miss": [],
        "unrelated": [], "adversarial": [],
    }
    all_max_scores: List[float] = []

    for cid, kind, text in QUERIES:
        q = processed[text]
        acts = w @ q  # cosine vs every cell (w rows are unit)
        target_score = float(acts[MEM_INDEX[cid]])
        max_score = float(acts.max())
        all_max_scores.append(max_score)
        if kind in ("exact", "paraphrase"):
            pos_scores.append(target_score)
            by_kind[kind].append(target_score)
        else:
            by_kind[kind].append(max_score)
            if kind in ("near_miss", "unrelated"):
                sweep_neg_scores.append(max_score)

    # Margin/overlap reporting uses all non-adversarial negatives too.
    neg_scores = sweep_neg_scores
    sweep = sweep_best_threshold(pos_scores, sweep_neg_scores,
                                 max_false_fire_rate=SWEEP_MAX_FALSE_FIRE)
    t = sweep.threshold

    def recall(scores: List[float]) -> Optional[float]:
        return round(float(np.mean(np.asarray(scores) > t)), 4) if scores else None

    exact_recall = recall(by_kind["exact"])
    paraphrase_recall = recall(by_kind["paraphrase"])
    near_miss_ffr = recall(by_kind["near_miss"])
    unrelated_ffr = recall(by_kind["unrelated"])
    adversarial_ffr = recall(by_kind["adversarial"])
    silence_rate = round(float(np.mean(np.asarray(all_max_scores) <= t)), 4)

    pos_margins = [s - t for s in pos_scores]
    neg_margins = [s - t for s in neg_scores]
    p05_pos_value = (float(np.percentile(pos_scores, 5))
                     if pos_scores else float("inf"))
    overlap = (round(float(np.mean(np.asarray(neg_scores) >= p05_pos_value)), 4)
               if neg_scores else None)

    accepted = bool(
        paraphrase_recall is not None
        and paraphrase_recall >= ACCEPT["min_paraphrase_recall"]
        and (near_miss_ffr or 0.0) <= ACCEPT["max_near_miss_ffr"]
        and (unrelated_ffr or 0.0) <= ACCEPT["max_unrelated_ffr"]
        and (adversarial_ffr or 0.0) <= ACCEPT["max_adversarial_ffr"]
    )

    return {
        "config": name,
        "method": method,
        "fit_set_size": len(fit_texts),
        "exact_recall": exact_recall,
        "paraphrase_recall": paraphrase_recall,
        "near_miss_false_fire_rate": near_miss_ffr,
        "unrelated_false_fire_rate": unrelated_ffr,
        "adversarial_false_fire_rate": adversarial_ffr,
        "silence_rate": silence_rate,
        "median_positive_margin": _pct(pos_margins, 50),
        "p05_positive_margin": _pct(pos_margins, 5),
        "p95_negative_margin": _pct(neg_margins, 95),
        "positive_negative_margin_overlap": overlap,
        "best_threshold": round(float(t), 4),
        "sweep_feasible": sweep.feasible,
        "sweep_recall": sweep.recall,
        "sweep_false_fire_rate": sweep.false_fire_rate,
        "accepted_operating_point": accepted,
    }


def main():
    configs: List[dict] = []

    # 1. deterministic control.
    det_vecs = _encode_deterministic()
    configs.append(evaluate_variant(
        "deterministic", det_vecs, "none", list(MEMORIES.values())))

    # MiniLM family.
    minilm = _encode_with_model("all-MiniLM-L6-v2")
    mem_texts = list(MEMORIES.values())
    calib_texts = mem_texts + CALIBRATION_BACKGROUND
    if minilm is not None:
        configs.append(evaluate_variant(
            "minilm_raw", minilm, "none", mem_texts))
        configs.append(evaluate_variant(
            "minilm_whiten_small", minilm, "zca", mem_texts))
        configs.append(evaluate_variant(
            "minilm_whiten_large_calib", minilm, "zca", calib_texts))
        configs.append(evaluate_variant(
            "minilm_pca_whiten", minilm, "pca", calib_texts))
        configs.append(evaluate_variant(
            "minilm_shrinkage_whiten", minilm, "shrinkage", calib_texts))
    else:
        for nm in ("minilm_raw", "minilm_whiten_small",
                   "minilm_whiten_large_calib", "minilm_pca_whiten",
                   "minilm_shrinkage_whiten"):
            configs.append({"config": nm,
                            "skipped": "all-MiniLM-L6-v2 unavailable offline"})

    # Optional 768d model.
    mpnet = _encode_with_model("all-mpnet-base-v2")
    if mpnet is not None:
        configs.append(evaluate_variant(
            "mpnet768_raw", mpnet, "none", mem_texts))
        configs.append(evaluate_variant(
            "mpnet768_shrinkage_whiten", mpnet, "shrinkage", calib_texts))
    else:
        configs.append({"config": "mpnet768_raw",
                        "skipped": "all-mpnet-base-v2 unavailable locally"})

    # Pick an operating point: the accepted variant with the highest paraphrase
    # recall (tie-break: lowest combined false-fire).
    accepted = [c for c in configs if c.get("accepted_operating_point")]

    def combined_ffr(c: dict) -> float:
        return ((c.get("near_miss_false_fire_rate") or 0.0)
                + (c.get("unrelated_false_fire_rate") or 0.0)
                + (c.get("adversarial_false_fire_rate") or 0.0))

    if accepted:
        best = sorted(
            accepted,
            key=lambda c: (-(c.get("paraphrase_recall") or 0.0), combined_ffr(c)),
        )[0]
        operating_point = {
            "found": True,
            "config": best["config"],
            "best_threshold": best["best_threshold"],
            "paraphrase_recall": best["paraphrase_recall"],
            "near_miss_false_fire_rate": best["near_miss_false_fire_rate"],
            "unrelated_false_fire_rate": best["unrelated_false_fire_rate"],
            "adversarial_false_fire_rate": best["adversarial_false_fire_rate"],
            "note": "meets acceptance bar: "
                    f"paraphrase_recall>={ACCEPT['min_paraphrase_recall']}, "
                    f"near_miss_ffr<={ACCEPT['max_near_miss_ffr']}, "
                    f"unrelated_ffr<={ACCEPT['max_unrelated_ffr']}, "
                    f"adversarial_ffr<={ACCEPT['max_adversarial_ffr']}.",
        }
    else:
        operating_point = {
            "found": False,
            "note": "No variant met the acceptance bar. Either paraphrase "
                    "recall stayed below "
                    f"{ACCEPT['min_paraphrase_recall']} or a false-fire rate "
                    "exceeded its cap. See per-variant metrics.",
        }

    summary = {
        "experiment": "exp09_semantic_margin_calibration",
        "description": (
            "Semantic calibration of MiniLM concept cells: can paraphrase "
            "recall be raised while keeping false fires low, across encoder, "
            "whitening, and threshold choices? Core geometry, write/query, "
            "Oja binding, and grounding are unchanged."
        ),
        "acceptance_criteria": ACCEPT,
        "sweep_max_false_fire_rate": SWEEP_MAX_FALSE_FIRE,
        "n_memories": len(MEMORIES),
        "n_queries": len(QUERIES),
        "calibration_background_size": len(CALIBRATION_BACKGROUND),
        "operating_point": operating_point,
        "configs": configs,
    }

    out_path = ROOT / "results" / "exp09_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[exp09] wrote {out_path}")


if __name__ == "__main__":
    main()
