"""Step 2.5 — gate signal analysis on the 5.7M cell bank.

exp15 showed that the current 0.10 rerank floor admits 70% of gibberish
queries. This script re-runs the same 50 queries (known / unknown / noise)
captured by exp15 and records richer per-query signals so we can choose a
better silence gate:

    - top-10 activations (so we can compute top1-top2, top1-mean(top2..top10))
    - self-reference round-trip score: cosine of the query vector against
      a re-encoding of the top-1 cell's source text. Captures whether the
      retrieved cell semantically reflects the query, not just its vector
      proximity.

Output: results/gate_signals_at_5p7m.json
Also prints, at the end, a sweep over candidate gates with their
known-pass / unknown-fire / noise-fire rates so we can pick one.
"""
from __future__ import annotations

# Pre-import torch so DLLs settle before the encoder is built.
import torch  # noqa: F401

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cc_service.encoder import EncoderSingleton  # noqa: E402

BANK_PATH = r"H:\MiniLM\cc_service\bank.db"
SOURCE_JSON = REPO_ROOT / "results" / "bank_selectivity_at_5p7m.json"
OUTPUT_JSON = REPO_ROOT / "results" / "gate_signals_at_5p7m.json"
TOP_K = 10


class StreamingBank:
    """Same loader pattern as exp15: pre-allocate, fill row-by-row."""

    def __init__(self, db_path: str):
        uri = f"file:{db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True)
        cur = self._conn.cursor()
        meta = {k: v for k, v in cur.execute("SELECT key, value FROM meta")}
        self.dim = int(meta["dim"])
        self.encoder_model = meta.get("encoder_model", "?")

        row = cur.execute(
            "SELECT mu, w_matrix, max_norm FROM whitening WHERE id = 1"
        ).fetchone()
        if row is not None:
            mu_blob, w_blob, max_norm = row
            self.mu = np.frombuffer(mu_blob, dtype=np.float32).copy()
            self.w_matrix = np.frombuffer(w_blob, dtype=np.float32).reshape(
                self.dim, self.dim
            ).copy()
            self.max_norm = float(max_norm)
        else:
            self.mu = None
            self.w_matrix = None
            self.max_norm = 1.0

        self.n_cells = int(cur.execute("SELECT COUNT(*) FROM cells").fetchone()[0])
        print(f"  [bank] allocating {self.n_cells:,} x {self.dim} float32 = "
              f"{(self.n_cells * self.dim * 4) / 1e9:.2f} GB", flush=True)

        self.weights = np.empty((self.n_cells, self.dim), dtype=np.float32)
        self.thetas = np.empty(self.n_cells, dtype=np.float32)
        self.cell_ids = np.empty(self.n_cells, dtype=np.int64)

        t0 = time.time()
        cur2 = self._conn.cursor()
        cur2.execute("SELECT id, weight, theta FROM cells ORDER BY id ASC")
        report_every = 500_000
        for i, (cell_id, weight_blob, theta) in enumerate(cur2):
            self.weights[i] = np.frombuffer(weight_blob, dtype=np.float32)
            self.thetas[i] = float(theta)
            self.cell_ids[i] = int(cell_id)
            if (i + 1) % report_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (self.n_cells - i - 1) / rate
                print(f"  [bank] streamed {i + 1:,}/{self.n_cells:,} "
                      f"({rate:,.0f}/s, eta {eta:.0f}s)", flush=True)
        print(f"  [bank] streamed {self.n_cells:,} cells in "
              f"{time.time() - t0:.1f}s", flush=True)

    def whiten(self, raw: np.ndarray) -> np.ndarray:
        if self.mu is None:
            return raw.astype(np.float32, copy=False)
        x = raw.astype(np.float32, copy=False)
        centered = x - self.mu
        whitened = centered @ self.w_matrix
        scaled = whitened / (self.max_norm + 1e-8)
        return scaled.astype(np.float32)

    def query_topk_activations(self, vector: np.ndarray, top_k: int = 10):
        """Returns (cell_ids, activations, thetas) for the top_k by activation."""
        activations = self.weights @ vector.astype(np.float32, copy=False)
        idx = np.argpartition(-activations, top_k)[:top_k]
        order = idx[np.argsort(-activations[idx])]
        return (
            self.cell_ids[order].copy(),
            activations[order].copy(),
            self.thetas[order].copy(),
        )

    def fetch_source_text(self, cell_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT text FROM source_texts WHERE cell_id = ?", (cell_id,)
        ).fetchone()
        return row[0] if row else None


def load_query_set() -> dict[str, list[dict]]:
    """Pull the 50 queries exp15 used so we measure on identical inputs."""
    data = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    return {
        "known": [{"query": e["query"],
                   "source_cell_id": e.get("source_cell_id")}
                  for e in data["known"]],
        "unknown": [{"query": e["query"], "source_cell_id": None}
                    for e in data["unknown"]],
        "noise": [{"query": e["query"], "source_cell_id": None}
                  for e in data["noise"]],
    }


def run_query(bank: StreamingBank, encoder: EncoderSingleton,
              query: str) -> dict:
    raw_q = encoder.encode_one(query, is_query=True)
    whitened = bank.whiten(raw_q)
    t0 = time.perf_counter()
    cell_ids, acts, thetas = bank.query_topk_activations(whitened, top_k=TOP_K)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    top1_cid = int(cell_ids[0])
    top1_text = bank.fetch_source_text(top1_cid) or ""

    # Self-reference round-trip: re-encode top1's source text, cosine
    # against the query. Captures semantic reflection, not just vector
    # proximity in the bank space.
    self_ref = None
    if top1_text:
        # MiniLM is normalized; cosine == dot product in encoder space.
        # Compare in the encoder's raw space (no whitening), since whitening
        # is bank-fit and is not appropriate for a query-vs-text round trip.
        ref_vec = encoder.encode_one(top1_text[:512], is_query=False)
        q_norm = np.linalg.norm(raw_q) + 1e-9
        r_norm = np.linalg.norm(ref_vec) + 1e-9
        self_ref = float(np.dot(raw_q, ref_vec) / (q_norm * r_norm))

    return {
        "query": query,
        "elapsed_ms": elapsed_ms,
        "top1_cell_id": top1_cid,
        "top1_source_head": top1_text[:160],
        "top_k_activations": [float(a) for a in acts],
        "top_k_cell_ids": [int(c) for c in cell_ids],
        "top_k_thetas": [float(t) for t in thetas],
        "self_ref_cosine": self_ref,
    }


def evaluate_gate(records: dict[str, list[dict]], gate_fn) -> dict:
    """Return {known_pass, unknown_fire, noise_fire} percentages."""
    out = {}
    for cat, items in records.items():
        if not items:
            out[cat] = None
            continue
        passes = sum(1 for r in items if gate_fn(r))
        out[cat] = passes / len(items)
    return out


def main():
    print(f"[boot] loading MiniLM encoder on GPU...", flush=True)
    if torch.cuda.is_available():
        _ = torch.cuda.mem_get_info()
    encoder = EncoderSingleton(model_name="all-MiniLM-L6-v2", device="cuda")
    _ = encoder.encode_one("warmup", is_query=True)
    print(f"[boot] encoder ready", flush=True)

    print(f"[boot] streaming bank from {BANK_PATH}...", flush=True)
    bank = StreamingBank(BANK_PATH)
    print(f"[boot] bank ready (n_cells={bank.n_cells:,})", flush=True)

    queries = load_query_set()
    records: dict[str, list[dict]] = {}
    for cat, items in queries.items():
        print(f"[probe] {cat}: {len(items)} queries...", flush=True)
        records[cat] = []
        for i, item in enumerate(items):
            rec = run_query(bank, encoder, item["query"])
            rec["source_cell_id"] = item.get("source_cell_id")
            if item.get("source_cell_id") is not None:
                try:
                    rec["source_rank"] = rec["top_k_cell_ids"].index(
                        item["source_cell_id"]) + 1
                except ValueError:
                    rec["source_rank"] = None
            records[cat].append(rec)

    out = {
        "produced_by": "experiments/exp16_gate_signals.py",
        "bank_path": BANK_PATH,
        "n_cells_measured": bank.n_cells,
        "encoder_model": bank.encoder_model,
        "top_k": TOP_K,
        "source_queries_from": str(SOURCE_JSON.relative_to(REPO_ROOT)),
        "records": records,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUTPUT_JSON}\n", flush=True)

    # --- Gate sweep ---
    print("=" * 72)
    print("GATE SWEEP (known should pass, unknown+noise should fail)")
    print("=" * 72)

    def show(name: str, results: dict):
        k = results.get("known")
        u = results.get("unknown")
        n = results.get("noise")
        fmt = lambda x: f"{x * 100:5.1f}%" if x is not None else "  n/a"
        print(f"  {name:<46}  known={fmt(k)}  unknown={fmt(u)}  "
              f"noise={fmt(n)}")

    # Absolute floor on top-1 activation.
    print("\n[abs floor on top-1 activation]")
    for thr in (0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        show(f"top1 >= {thr:.2f}",
             evaluate_gate(records, lambda r, t=thr: r["top_k_activations"][0] >= t))

    # Margin top1 - top2.
    print("\n[margin: top1 - top2]")
    for thr in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30):
        show(f"(top1 - top2) >= {thr:.2f}",
             evaluate_gate(records,
                           lambda r, t=thr: (r["top_k_activations"][0]
                                             - r["top_k_activations"][1]) >= t))

    # Margin top1 - mean(top2..top10).
    print("\n[margin: top1 - mean(top2..top10)]")
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        show(
            f"(top1 - mean_rest) >= {thr:.2f}",
            evaluate_gate(
                records,
                lambda r, t=thr: (r["top_k_activations"][0]
                                  - float(np.mean(r["top_k_activations"][1:]))) >= t,
            ),
        )

    # Self-reference round trip.
    print("\n[self-ref cosine: cos(query, encode(top1_text))]")
    for thr in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        show(f"self_ref >= {thr:.2f}",
             evaluate_gate(records,
                           lambda r, t=thr: (r["self_ref_cosine"] or 0.0) >= t))

    # Combined: absolute floor + self-ref.
    print("\n[combined: top1 >= A  AND  self_ref >= B]")
    for a, b in [(0.30, 0.40), (0.35, 0.45), (0.40, 0.50),
                 (0.40, 0.55), (0.45, 0.55), (0.50, 0.55)]:
        show(
            f"top1>={a:.2f} & self_ref>={b:.2f}",
            evaluate_gate(
                records,
                lambda r, A=a, B=b: (r["top_k_activations"][0] >= A
                                     and (r["self_ref_cosine"] or 0.0) >= B),
            ),
        )

    # Per-category raw distributions (so we can see where to draw lines).
    print("\n" + "=" * 72)
    print("RAW DISTRIBUTIONS (min / p25 / p50 / p75 / max)")
    print("=" * 72)
    for signal_name, fn in [
        ("top1_activation", lambda r: r["top_k_activations"][0]),
        ("top1 - top2", lambda r: r["top_k_activations"][0]
                                  - r["top_k_activations"][1]),
        ("top1 - mean(rest)", lambda r: r["top_k_activations"][0]
                                        - float(np.mean(r["top_k_activations"][1:]))),
        ("self_ref_cosine", lambda r: r["self_ref_cosine"] or 0.0),
    ]:
        print(f"\n  {signal_name}:")
        for cat in ("known", "unknown", "noise"):
            vals = np.array([fn(r) for r in records[cat]], dtype=np.float64)
            q = np.percentile(vals, [0, 25, 50, 75, 100])
            print(f"    {cat:<8}  min={q[0]:+.3f}  p25={q[1]:+.3f}  "
                  f"p50={q[2]:+.3f}  p75={q[3]:+.3f}  max={q[4]:+.3f}")


if __name__ == "__main__":
    main()
