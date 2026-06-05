"""Measure the activation distribution of the bank to choose a firing threshold.

The bank gates query results by ``margin = activation - theta`` and, by default,
suppresses any cell with ``margin <= 0`` (a "silent" hit). The write-time default
``theta = 0.30`` was inherited from the original MiniLM geometry; on the BGE bank
real matches activate lower than that, so genuine hits get suppressed and a plain
query looks empty.

This script characterises three activation distributions so theta can be set from
evidence rather than guessed:

  * POSITIVE   - activation of the *correct* cell for a partial-mention query
                 (first ~half of the cell's own source text). We want theta to
                 sit *below* these so true matches fire.
  * DISTRACTOR - for that same query, the highest activation among all *other*
                 cells (the nearest wrong neighbor). Theta below these means junk
                 leaks in alongside the true hit.
  * FOREIGN    - activation ceiling for queries that have no answer in the bank
                 (random word salad). We want theta *above* these so unrelated
                 queries correctly return empty (the bank's rejection property).

A good theta lies above the FOREIGN ceiling and below the POSITIVE floor. The gap
between FOREIGN-high and POSITIVE-low is the usable operating band.

Usage (from h:\\ai_mem_bank, cc_service venv python):
    python ..\\MiniLM\\cc_service\\calibrate_theta.py \
        --db h:\\ai_mem_bank\\bank_bge.db --encoder BAAI/bge-base-en-v1.5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# pyarrow before torch (Windows segfault guard); see eval_retrieval.py.
try:
    import pyarrow  # noqa: F401
except Exception:
    pass

import numpy as np


def _pin_hf_cache_to_h() -> None:
    cache = Path(r"h:\ai_mem_bank\.cache")
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache / "sentence-transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))


_TAG_RE = re.compile(r"^\[([^\]]+)\]")


def domain_of(label: str | None) -> str:
    if label and label.startswith("["):
        m = _TAG_RE.match(label)
        if m:
            return m.group(1)
    return "NONE"


def half_query(text: str) -> str:
    toks = text.split()
    k = max(8, len(toks) // 2)
    return " ".join(toks[:k])


def sample_probes(db: str, per_domain: int, seed: int):
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT c.id, c.label, s.text "
            "FROM cells c JOIN source_texts s ON s.cell_id = c.id "
            "WHERE c.kind='single' AND s.text IS NOT NULL AND length(s.text) > 40"
        ).fetchall()
    finally:
        con.close()
    by_dom = defaultdict(list)
    for cid, label, text in rows:
        by_dom[domain_of(label)].append((cid, text))
    rng = random.Random(seed)
    probes = []
    for dom in sorted(by_dom):
        pool = by_dom[dom]
        rng.shuffle(pool)
        for cid, text in pool[:per_domain]:
            probes.append((cid, dom, text))
    return probes


_WORDS = (
    "quantum velvet harbor sprocket lantern meadow cobalt thistle ferment "
    "glacier puzzle marigold tangent walrus ember nimbus driftwood quartz "
    "lavender chronicle saffron pebble zephyr trellis cinder maroon "
    "obsidian willow gilded tundra cactus opal vellum brindle"
).split()


def foreign_queries(n: int, seed: int):
    rng = random.Random(seed + 99)
    out = []
    for _ in range(n):
        k = rng.randint(8, 16)
        out.append(" ".join(rng.choice(_WORDS) for _ in range(k)))
    return out


def pct(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if a.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--n-foreign", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: db not found: {args.db}", file=sys.stderr)
        return 1

    _pin_hf_cache_to_h()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from cc_service.memory import MemoryBank, apply_whitening

    bank = MemoryBank(db_path=args.db, encoder_model=args.encoder,
                      device=args.device)
    bank._require_init()
    if not bank._cache_valid:
        bank._refresh_cache()
    W = bank._W                       # (N, dim), unit-norm weights
    cell_ids = bank._cell_ids
    id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    probes = sample_probes(args.db, args.per_domain, args.seed)

    positives = []
    distractors = []
    for cid, _dom, text in probes:
        if cid not in id_to_idx:
            continue
        raw = bank.encoder.encode_one(half_query(text), is_query=True)
        q = apply_whitening(raw, bank._whitening)
        acts = W @ q
        true_i = id_to_idx[cid]
        positives.append(float(acts[true_i]))
        acts[true_i] = -np.inf
        distractors.append(float(acts.max()))

    foreign = []
    for fq in foreign_queries(args.n_foreign, args.seed):
        raw = bank.encoder.encode_one(fq, is_query=True)
        q = apply_whitening(raw, bank._whitening)
        acts = W @ q
        foreign.append(float(acts.max()))

    bank.store.close()

    pos = np.array(positives)
    dis = np.array(distractors)
    frn = np.array(foreign)

    report = {
        "db": args.db,
        "encoder": args.encoder,
        "n_positive_probes": int(pos.size),
        "n_foreign_probes": int(frn.size),
        "positive_activation": {
            "p05": pct(pos, 5), "p10": pct(pos, 10), "p25": pct(pos, 25),
            "p50": pct(pos, 50), "mean": float(pos.mean()),
        },
        "distractor_activation": {
            "p50": pct(dis, 50), "p75": pct(dis, 75), "p90": pct(dis, 90),
            "p95": pct(dis, 95),
        },
        "foreign_activation": {
            "p50": pct(frn, 50), "p90": pct(frn, 90), "p95": pct(frn, 95),
            "p99": pct(frn, 99), "max": float(frn.max()),
        },
    }

    # Recommendation: theta just above the foreign ceiling (p99) but capped below
    # the positive floor (p10) so true matches still fire. Midpoint when both
    # constraints are satisfiable; otherwise flag the overlap.
    foreign_hi = report["foreign_activation"]["p99"]
    pos_lo = report["positive_activation"]["p10"]
    if foreign_hi < pos_lo:
        rec = round((foreign_hi + pos_lo) / 2, 3)
        note = "clean band: theta set midway between foreign-p99 and positive-p10"
    else:
        rec = round(foreign_hi, 3)
        note = ("OVERLAP: foreign-p99 >= positive-p10. theta pinned to foreign-p99; "
                "some true matches will still be gated. Consider a reranker or "
                "returning top-k by activation for low-confidence queries.")
    report["recommended_theta"] = rec
    report["recommendation_note"] = note

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
