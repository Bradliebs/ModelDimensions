"""Experiment 08: encoder semantic-recall comparison.

Asks one question: how does the *encoder* change what the (frozen) concept-cell
memory can recall? The geometry, whitening, scaling, write/query and binding
are all unchanged -- only the vectors feeding the bank differ.

Three encoder configs over the same small corpus and the same query set:

  1. deterministic        : the offline content-addressed encoder (exact match
                            only; a paraphrase is a different hash).
  2. minilm_raw           : all-MiniLM-L6-v2 sentence embeddings, untouched.
  3. minilm_whiten_scale  : the same MiniLM embeddings passed through the SAME
                            math as concept_cells.geometry (ZCA whitening fit on
                            the memories, then unit-ball scaling) before storage.

MiniLM configs require the model; if it cannot be loaded the experiment still
runs the deterministic config and records the others as skipped.

Each query is one of four kinds:
  - exact     : the stored sentence verbatim          -> should fire its memory
  - paraphrase: same meaning, different words         -> should fire its memory
  - near_miss : same topic, a changed fact            -> should stay silent
  - unrelated : a different topic entirely            -> should stay silent

A moderate epsilon (0.25, cosine bar ~= 0.72) is used so paraphrase recall is
measurable at all; the near-duplicate default (0.05) would make every encoder
look identical. This is an experiment parameter, not a change to the core.

Metrics per config:
  - exact_recall        : exact queries that fired the correct memory.
  - paraphrase_recall   : paraphrase queries that fired the correct memory.
  - false_fire_rate     : near_miss + unrelated queries that fired anything.
  - silent_refusal_rate : unrelated queries correctly left silent.
  - margin_distribution : count/mean/std/min/max of fired margins + per-kind mean.

Writes results/exp08_summary.json.

    python -m experiments.exp08_encoder_semantic_recall
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.orchestrator import DeterministicEncoder, MemoryBank

EPSILON = 0.25
RADIUS = 0.9

# corpus_id -> stored sentence
MEMORIES: Dict[str, str] = {
    "launch": "the product launch is scheduled for Friday afternoon",
    "budget": "the project budget is forty thousand dollars",
    "server": "the primary server is located in the Dublin data center",
    "meeting": "the client meeting is on Tuesday morning",
    "rotation": "the database password rotates every ninety days",
}

# (corpus_id, kind, query_text). kind in {exact, paraphrase, near_miss, unrelated}.
QUERIES: List[tuple] = [
    ("launch", "exact", "the product launch is scheduled for Friday afternoon"),
    ("launch", "paraphrase", "we will release the product on Friday afternoon"),
    ("launch", "near_miss", "the product launch is scheduled for Monday afternoon"),
    ("launch", "unrelated", "the cat slept on the warm windowsill all day"),

    ("budget", "exact", "the project budget is forty thousand dollars"),
    ("budget", "paraphrase", "we have forty thousand dollars allocated for the project"),
    ("budget", "near_miss", "the project budget is ninety thousand dollars"),
    ("budget", "unrelated", "the train departs from platform nine every hour"),

    ("server", "exact", "the primary server is located in the Dublin data center"),
    ("server", "paraphrase", "our main server lives in the Dublin data centre"),
    ("server", "near_miss", "the primary server is located in the Frankfurt data center"),
    ("server", "unrelated", "she painted the fence a bright shade of green"),

    ("meeting", "exact", "the client meeting is on Tuesday morning"),
    ("meeting", "paraphrase", "we are meeting the client on Tuesday in the morning"),
    ("meeting", "near_miss", "the client meeting is on Thursday morning"),
    ("meeting", "unrelated", "the recipe needs two cups of flour and one egg"),

    ("rotation", "exact", "the database password rotates every ninety days"),
    ("rotation", "paraphrase", "the db credentials are rotated on a ninety day cycle"),
    ("rotation", "near_miss", "the database password rotates every thirty days"),
    ("rotation", "unrelated", "the marathon route passes three city parks"),
]


class PrecomputedEncoder:
    """Encoder that returns pre-computed vectors by exact text lookup.

    Lets the bank consume MiniLM (and whitened/scaled MiniLM) vectors through
    the normal per-text ``encode_one`` interface without any change to the
    bank or the core geometry.
    """

    def __init__(self, mapping: Dict[str, np.ndarray]):
        self._mapping = mapping

    def encode_one(self, text: str) -> np.ndarray:
        return self._mapping[text]


def _fit_whiten_scale(mem_emb: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a transform mirroring geometry.zca_whiten + scale_to_unit_ball.

    Fit on the MEMORIES only (no query leakage): mean-center, ZCA-whiten with
    the eigendecomposition of the covariance, then divide by the max whitened
    memory norm so memories sit inside the unit ball. The exact same transform
    is then applied to queries. This duplicates the *core* math only to fit a
    reusable transform; the core module itself is untouched.
    """
    mu = mem_emb.mean(axis=0, keepdims=True)
    centered = mem_emb - mu
    n = max(len(mem_emb) - 1, 1)
    cov = (centered.T @ centered) / n
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-5)
    w_mat = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    whitened_mem = centered @ w_mat
    max_norm = float(np.linalg.norm(whitened_mem, axis=1).max()) + 1e-8

    def transform(x: np.ndarray) -> np.ndarray:
        return (((x[None, :] - mu) @ w_mat) / max_norm)[0].astype(np.float32)

    return transform


def _build_mapping_minilm(raw: bool):
    """Return text->vector mapping for MiniLM raw or whitened+scaled, or None."""
    try:
        from concept_cells.encoders import TextEncoder
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

    mem_emb = emb[:len(mem_texts)]
    if raw:
        vectors = emb
    else:
        transform = _fit_whiten_scale(mem_emb)
        vectors = np.array([transform(emb[i]) for i in range(len(emb))])

    mapping = {}
    texts = list(MEMORIES.values()) + [q for _, _, q in QUERIES]
    for text, vec in zip(texts, vectors):
        mapping[text] = vec.astype(np.float32)
    return mapping


def _seed_bank(encoder) -> Dict[str, str]:
    """Write every memory; return corpus_id -> minted memory_id."""
    bank = MemoryBank(encoder, epsilon=EPSILON, radius=RADIUS)
    id_for = {}
    for cid, sentence in MEMORIES.items():
        rec = bank.write(sentence)
        id_for[cid] = rec.memory_id
    return bank, id_for


def _summary_stats(values: List[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def run_config(name: str, encoder) -> dict:
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
        else:  # near_miss, unrelated -> should be silent
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
        # Raw fired-margin lists per query kind, exported so Exp 09 can reuse
        # or compare these distributions without re-running the bank. Additive:
        # the summary metrics above are unchanged.
        "margins_by_kind_raw": {
            k: [round(float(x), 6) for x in v]
            for k, v in margins_by_kind.items()
        },
        "n_queries": len(QUERIES),
    }


def main():
    configs: List[dict] = []
    configs.append(run_config("deterministic", DeterministicEncoder(dim=64)))

    raw_map = _build_mapping_minilm(raw=True)
    if raw_map is not None:
        configs.append(run_config("minilm_raw", PrecomputedEncoder(raw_map)))
    else:
        configs.append({"config": "minilm_raw",
                        "skipped": "all-MiniLM-L6-v2 unavailable offline"})

    ws_map = _build_mapping_minilm(raw=False)
    if ws_map is not None:
        configs.append(run_config("minilm_whiten_scale",
                                  PrecomputedEncoder(ws_map)))
    else:
        configs.append({"config": "minilm_whiten_scale",
                        "skipped": "all-MiniLM-L6-v2 unavailable offline"})

    summary = {
        "experiment": "exp08_encoder_semantic_recall",
        "description": (
            "How the encoder changes recall over the frozen concept-cell "
            "memory: exact vs paraphrase recall, false fires, and firing "
            "margins for deterministic / MiniLM / MiniLM+whiten+scale."
        ),
        "epsilon": EPSILON,
        "radius": RADIUS,
        "note": (
            "Geometry, whitening, scaling, write/query and binding are "
            "unchanged; only the encoder feeding the bank differs."
        ),
        "configs": configs,
    }

    out_path = ROOT / "results" / "exp08_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[exp08] wrote {out_path}")


if __name__ == "__main__":
    main()
