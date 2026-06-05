"""
exp15_bank_selectivity_at_scale.py
==================================

Measures how well the concept-cell bank discriminates known vs absent vs
noise queries at deployed scale. Addresses paper §9a's open
threat-to-validity: "selectivity at 1.8M cells has not been measured
on out-of-distribution queries."

The production bank at H:\\MiniLM\\cc_service\\bank.db has grown beyond
1.8M cells since the paper. The artifact records the measured count.

Uses a *streaming* loader (not src/agent/sqlite_bank.SqliteBank) because
the latter does fetchall() then np.stack, which peaks at ~25 GB for
5.7M cells. The streaming loader pre-allocates the (N, 384) weight
array and fills it row-by-row from the SQLite cursor, capping peak
memory near the final ~9 GB.

Three query sets:
  - known   (20): first sentence of a randomly sampled source_text row.
  - unknown (20): plausible-but-absent English (made-up entities/facts).
  - noise   (10): gibberish, repetitions, mojibake.

For each query: encode (MiniLM) → whiten → score all cells → top-k.
Recorded per query: top-1 activation, top-1 margin (activation − theta),
fired count, rank of source cell (known only), elapsed_ms,
above_rerank_floor at the bank's configured 0.10.

Writes results/bank_selectivity_at_5p7m.json. No assertions —
measurement only, per the paper's framing of §9a as an open question.
"""

from __future__ import annotations

# Import torch FIRST so its DLLs are loaded before anything else touches
# CUDA-adjacent libraries. Skipping this caused Windows access violations
# (0xC0000005) when the encoder was instantiated later in the boot path.
import torch  # noqa: F401

import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cc_service.encoder import EncoderSingleton  # noqa: E402


BANK_PATH = r"H:\MiniLM\cc_service\bank.db"
SEED = 42

N_KNOWN = 20
N_UNKNOWN = 20
N_NOISE = 10

TOP_K = 10
RERANK_MIN_SCORE = 0.10

UNKNOWN_QUERIES = [
    "The Treaty of Vulkanos was signed in 1834 by the Margravate of Helmar.",
    "Quintuple-helix DNA was first observed by Petra Voss in 1962.",
    "The lost continent of Zuralia sank beneath the Caspian Sea in 800 BC.",
    "The K-117 lunar module reached the far side of Mars in 1981.",
    "Penguins of the Karkanos archipelago hibernate underground for nine months.",
    "Wolfgang Heintz invented the binary semaphore relay in 1893.",
    "The Battle of Three Bridges (1247) ended the Aragonese-Wallachian wars.",
    "Glyphochromism is the study of color changes in obsidian mosses.",
    "Catherine the Slight ruled the duchy of Pelloria from 1612 to 1618.",
    "The Vrindavan Crystal Cipher was decoded by Brian Halford in 1947.",
    "Tuvalu's deep-sea railway connects Funafuti to the Marshall Islands.",
    "The asteroid 3712 Kepler-Bach was reclassified as a dwarf comet in 2009.",
    "Hoffmann's third law of magnetohydrodynamics describes torsional vacuum drag.",
    "The Republic of Aldobrand declared independence from France in 1798.",
    "Maple sugar fermentation produces a beverage called bakhraan in rural Vermont.",
    "The HMS Calliope sank off the coast of Trincomalee on March 4, 1701.",
    "Bismuth-209 transmutes to lead-205 via alpha decay over a 4-hour half-life.",
    "Octavia Brooks won the Hugo Award in 1962 for The Listening Mountain.",
    "The Erlangen Schism of 1574 split the German Reformed church into seven sects.",
    "Anomalous diffraction patterns in lutetium-yttrium crystals were first reported by Klimov in 1957.",
]

NOISE_QUERIES = [
    "asdf qwerty zxcv hjkl",
    "blah blah blah blah blah",
    "xyzzy plugh nikto fred",
    "qwoeiruty pasldkfj zmxncbv",
    "1234567890 !@#$%^&*()",
    "aaaaa bbbbb ccccc ddddd eeeee",
    "Lorem ipsum dolor sit amet consectetur",
    "hjklhjklhjkl mnbvcxz poiuytrew",
    "test test test test test test test",
    "the the the the the the the the",
]


class StreamingBank:
    """Memory-efficient read-only bank for the selectivity probe.

    Loads weights/thetas/cell_ids by streaming the cursor into
    pre-allocated arrays. Does NOT load source_texts up front —
    fetches them on demand for the small set of top-k hits per query.
    Peak memory ~= 4 × N × dim bytes (weights) plus tiny overhead.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)

        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

        # ---- meta ----
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='dim'"
        ).fetchone()
        self.dim = int(json.loads(row[0]))

        # ---- whitening ----
        row = self._conn.execute(
            "SELECT mu, w_matrix, max_norm FROM whitening WHERE id=1"
        ).fetchone()
        if row is None:
            self.mu = None
            self.w_matrix = None
            self.max_norm = None
        else:
            mu_bytes, w_bytes, max_norm = row
            self.mu = np.frombuffer(mu_bytes, dtype=np.float32).copy()
            self.w_matrix = np.frombuffer(
                w_bytes, dtype=np.float32
            ).reshape((self.dim, self.dim)).copy()
            self.max_norm = float(max_norm)

        # ---- count cells ----
        n = self._conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
        self.n_cells = int(n)

        # ---- stream weights + thetas + ids into pre-allocated arrays ----
        print(f"  [bank] allocating {self.n_cells:,} × {self.dim} float32 "
              f"= {self.n_cells * self.dim * 4 / 1e9:.2f} GB")
        self.weights = np.empty((self.n_cells, self.dim), dtype=np.float32)
        self.thetas = np.empty((self.n_cells,), dtype=np.float32)
        self.cell_ids = np.empty((self.n_cells,), dtype=np.int64)

        cur = self._conn.execute(
            "SELECT id, weight, theta FROM cells ORDER BY id ASC"
        )
        t0 = time.time()
        i = 0
        report_every = 500_000
        for cid, wbytes, theta in cur:
            self.weights[i] = np.frombuffer(
                wbytes, dtype=np.float32
            ).reshape((self.dim,))
            self.thetas[i] = theta
            self.cell_ids[i] = cid
            i += 1
            if i % report_every == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (self.n_cells - i) / rate
                print(f"  [bank] streamed {i:,}/{self.n_cells:,} "
                      f"({rate:,.0f}/s, eta {eta:.0f}s)")
        if i != self.n_cells:
            # cells table is the source of truth; trim arrays to actual count
            self.weights = self.weights[:i]
            self.thetas = self.thetas[:i]
            self.cell_ids = self.cell_ids[:i]
            self.n_cells = i
        print(f"  [bank] streamed {self.n_cells:,} cells in "
              f"{time.time() - t0:.1f}s")

    def whiten(self, raw: np.ndarray) -> np.ndarray:
        if self.mu is None:
            return raw.astype(np.float32, copy=False)
        x = raw.astype(np.float32, copy=False)
        centered = x - self.mu
        whitened = centered @ self.w_matrix
        scaled = whitened / (self.max_norm + 1e-8)
        return scaled.astype(np.float32)

    def query(self, vector: np.ndarray, top_k: int = 10) -> list[dict]:
        if vector.shape != (self.dim,):
            raise ValueError(f"vector shape {vector.shape} != ({self.dim},)")
        q = vector.astype(np.float32, copy=False)
        activations = self.weights @ q
        margins = activations - self.thetas
        # Top-k by margin overall (include silent so we can report top-1
        # even when nothing fires — important for the selectivity story).
        idx = np.argpartition(-margins, min(top_k, len(margins) - 1))[:top_k]
        order = idx[np.argsort(-margins[idx])]
        return [
            {
                "cell_id": int(self.cell_ids[i]),
                "activation": float(activations[i]),
                "theta": float(self.thetas[i]),
                "margin": float(margins[i]),
            }
            for i in order
        ]

    def fetch_source_text(self, cell_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT text FROM source_texts WHERE cell_id = ?", (cell_id,)
        ).fetchone()
        return row[0] if row else None


def sample_known_queries(db_path: str, n: int, rng: random.Random) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    max_id = conn.execute("SELECT MAX(cell_id) FROM source_texts").fetchone()[0]
    seen = set()
    picks = []
    attempts = 0
    while len(picks) < n and attempts < n * 50:
        attempts += 1
        cid = rng.randint(1, max_id)
        if cid in seen:
            continue
        seen.add(cid)
        row = conn.execute(
            "SELECT cell_id, text FROM source_texts WHERE cell_id = ?", (cid,)
        ).fetchone()
        if row is None:
            continue
        cell_id, text = row
        first = text.split(". ")[0].strip()
        if len(first) < 20 or len(first) > 400:
            continue
        picks.append({"source_cell_id": int(cell_id), "query": first,
                      "full_text_head": text[:200]})
    conn.close()
    if len(picks) < n:
        raise RuntimeError(f"only sampled {len(picks)} known queries, need {n}")
    return picks


def probe(bank: StreamingBank, encoder: EncoderSingleton,
          query: str) -> tuple[list, float]:
    raw = encoder.encode_one(query, is_query=True)
    whitened = bank.whiten(raw)
    t0 = time.perf_counter()
    hits = bank.query(whitened, top_k=TOP_K)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return hits, elapsed_ms


def summarise_hits(bank: StreamingBank, hits: list[dict],
                   source_cell_id: int | None) -> dict:
    if not hits:
        return {
            "top1_activation": None, "top1_margin": None,
            "top1_cell_id": None, "top1_source_head": None,
            "fired_count": 0, "rank_of_source": None,
            "above_rerank_floor": False,
        }
    top1 = hits[0]
    fired = sum(1 for h in hits if h["margin"] > 0)
    above_floor = top1["margin"] >= RERANK_MIN_SCORE

    rank = None
    if source_cell_id is not None:
        for i, h in enumerate(hits):
            if h["cell_id"] == source_cell_id:
                rank = i + 1
                break

    src_head = bank.fetch_source_text(top1["cell_id"]) or ""
    return {
        "top1_activation": top1["activation"],
        "top1_margin": top1["margin"],
        "top1_cell_id": top1["cell_id"],
        "top1_source_head": src_head[:120],
        "fired_count": fired,
        "rank_of_source": rank,
        "above_rerank_floor": above_floor,
    }


def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)

    # Load encoder FIRST (on GPU) before the bank fills RAM. Loading the
    # encoder after the bank caused Windows access violations when CUDA
    # init/HF cache hydration competed for memory.
    print("[boot] loading MiniLM encoder on GPU...")
    t0 = time.time()
    # Force CUDA context initialization explicitly. Without this, the lazy
    # init triggered by SentenceTransformer.__init__ crashes with a Windows
    # access violation (0xC0000005) in this script's import context.
    if torch.cuda.is_available():
        _ = torch.cuda.mem_get_info()
    encoder = EncoderSingleton(model_name="all-MiniLM-L6-v2", device="cuda")
    print("[boot] encoder ctor returned, warming up...", flush=True)
    _ = encoder.encode_one("warmup", is_query=True)
    print(f"[boot] encoder ready in {time.time() - t0:.1f}s")

    print(f"[boot] streaming bank from {BANK_PATH}...")
    t0 = time.time()
    bank = StreamingBank(BANK_PATH)
    print(f"[boot] bank ready in {time.time() - t0:.1f}s "
          f"(n_cells={bank.n_cells:,} dim={bank.dim})")

    print(f"[known] sampling {N_KNOWN} queries from source_texts...")
    known = sample_known_queries(BANK_PATH, N_KNOWN, rng)

    print(f"[probe] running {N_KNOWN + N_UNKNOWN + N_NOISE} queries...")
    all_latencies = []

    known_results = []
    for q in known:
        hits, ms = probe(bank, encoder, q["query"])
        all_latencies.append(ms)
        summary = summarise_hits(bank, hits, q["source_cell_id"])
        known_results.append({
            "query": q["query"],
            "source_cell_id": q["source_cell_id"],
            "elapsed_ms": ms,
            **summary,
        })

    unknown_results = []
    for q in UNKNOWN_QUERIES[:N_UNKNOWN]:
        hits, ms = probe(bank, encoder, q)
        all_latencies.append(ms)
        summary = summarise_hits(bank, hits, None)
        unknown_results.append({"query": q, "elapsed_ms": ms, **summary})

    noise_results = []
    for q in NOISE_QUERIES[:N_NOISE]:
        hits, ms = probe(bank, encoder, q)
        all_latencies.append(ms)
        summary = summarise_hits(bank, hits, None)
        noise_results.append({"query": q, "elapsed_ms": ms, **summary})

    def hit_rate(results, predicate) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if predicate(r)) / len(results)

    known_top1_match = hit_rate(
        known_results,
        lambda r: r["top1_cell_id"] == r["source_cell_id"],
    )
    known_in_top10 = hit_rate(
        known_results, lambda r: r["rank_of_source"] is not None,
    )
    known_above_floor = hit_rate(
        known_results, lambda r: r["above_rerank_floor"],
    )
    unknown_false_fire = hit_rate(
        unknown_results, lambda r: r["above_rerank_floor"],
    )
    noise_false_fire = hit_rate(
        noise_results, lambda r: r["above_rerank_floor"],
    )

    latencies = np.array(all_latencies)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    out_path = REPO_ROOT / "results" / "bank_selectivity_at_5p7m.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "produced_by": "experiments/exp15_bank_selectivity_at_scale.py",
        "bank_path": BANK_PATH,
        "n_cells_measured": bank.n_cells,
        "paper_documented_n_cells": 1817204,
        "encoder_model": encoder.model_name,
        "encoder_dim": encoder.dim,
        "rerank_min_score": RERANK_MIN_SCORE,
        "top_k_probe": TOP_K,
        "seed": SEED,
        "summary": {
            "known_top1_exact_match_rate": known_top1_match,
            "known_source_in_top10_rate": known_in_top10,
            "known_above_rerank_floor_rate": known_above_floor,
            "unknown_false_fire_rate": unknown_false_fire,
            "noise_false_fire_rate": noise_false_fire,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
        },
        "known": known_results,
        "unknown": unknown_results,
        "noise": noise_results,
    }

    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nwrote {out_path}")

    print("\n" + "=" * 72)
    print("SELECTIVITY SUMMARY")
    print("=" * 72)
    print(f"  bank size: {bank.n_cells:,} cells "
          f"(paper documented 1,817,204)")
    print(f"  known queries  ({N_KNOWN}):")
    print(f"    top-1 exact source match : {known_top1_match:.1%}")
    print(f"    source in top-10         : {known_in_top10:.1%}")
    print(f"    top-1 above floor (≥0.10): {known_above_floor:.1%}")
    print(f"  unknown queries ({N_UNKNOWN}):")
    print(f"    false-fire above floor   : {unknown_false_fire:.1%}")
    print(f"  noise queries  ({N_NOISE}):")
    print(f"    false-fire above floor   : {noise_false_fire:.1%}")
    print(f"  latency p50={p50:.1f}ms  p95={p95:.1f}ms")
    print("=" * 72)


if __name__ == "__main__":
    main()
