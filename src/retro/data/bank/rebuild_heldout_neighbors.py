"""
rebuild_heldout_neighbors.py — recompute heldout_neighbors.npy against
the current bank, WITHOUT re-downloading or re-tokenizing held-out
articles.

Reads existing heldout.bin (immutable token stream) and current bank
weights from bank.db, runs MIPS to find top-N_NEIGHBORS for each
held-out chunk. Same algorithm as prepare_heldout.py but:
  - skips the dataset/tokenize step (heldout.bin already exists)
  - chunks W_bank across GPU passes so it fits on small cards
    (5M cells x 384 float32 = 7.5 GB; tiles to MIPS_TILE_CELLS at a time)

Usage:
    cd H:\\MiniLM\\nanogpt
    python data/bank/rebuild_heldout_neighbors.py
"""

import json
import sqlite3
import time
from pathlib import Path

# datasets-free import (just transformers for MiniLM)
from transformers import AutoTokenizer, AutoModel

import numpy as np
import tiktoken
import torch


BANK_DB = r"H:\MiniLM\cc_service\bank.db"
OUT_DIR = Path(__file__).parent

CHUNK_SIZE = 64
N_NEIGHBORS = 2
ENCODE_BATCH = 256
MIPS_TILE_CELLS = 1_500_000   # how many cells to push to GPU per pass
                              # ~1.5M x 384 x 4B = 2.3 GB, leaves room

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MiniLMEncoder:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str):
        self.device = device
        print(f"loading MiniLM on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(device).eval()

    @torch.no_grad()
    def encode(self, texts, max_length=128):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        attn = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (outputs.last_hidden_state * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1e-9)
        return summed / counts


def load_whitening(conn):
    row = conn.execute(
        "SELECT mu, w_matrix, max_norm FROM whitening WHERE id=1"
    ).fetchone()
    mu = np.frombuffer(row[0], dtype=np.float32).reshape(384).copy()
    W = np.frombuffer(row[1], dtype=np.float32).reshape(384, 384).copy()
    max_norm = float(row[2])
    return mu, W, max_norm


def load_bank_weights(conn):
    n_total = conn.execute(
        "SELECT COUNT(*) FROM cells c JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single'"
    ).fetchone()[0]
    print(f"loading {n_total:,} cell weights from bank...")
    W_bank = np.zeros((n_total, 384), dtype=np.float32)
    cur = conn.execute(
        "SELECT c.weight FROM cells c "
        "JOIN source_texts s ON c.id=s.cell_id "
        "WHERE c.kind='single' ORDER BY c.id"
    )
    for i, (w_blob,) in enumerate(cur):
        W_bank[i] = np.frombuffer(w_blob, dtype=np.float32)
        if (i + 1) % 500_000 == 0:
            print(f"  loaded {i+1:>9,}/{n_total:,}")
    print(f"  W_bank: {W_bank.nbytes/1e6:.0f} MB ({n_total:,} cells)")
    return W_bank


@torch.no_grad()
def topk_chunked(q_gpu, W_bank_cpu, k, tile_cells):
    """MIPS top-k where W_bank is too large to fit on GPU.

    Returns (B, k) int64 of global indices into W_bank.
    Streams W_bank in tiles of tile_cells rows; for each tile computes
    activations + local top-k, then merges across tiles.
    """
    n_total, dim = W_bank_cpu.shape
    B = q_gpu.shape[0]
    # Running best-k across tiles (per query).
    best_vals = torch.full((B, k), float("-inf"), device=DEVICE)
    best_idx = torch.zeros((B, k), dtype=torch.int64, device=DEVICE)

    for start in range(0, n_total, tile_cells):
        end = min(start + tile_cells, n_total)
        # Pull tile to GPU (float32).
        tile = torch.from_numpy(W_bank_cpu[start:end]).to(DEVICE)  # (T, D)
        # Activations: (B, T)
        act = q_gpu @ tile.T
        # Local top-k within tile.
        local_k = min(k, end - start)
        loc_vals, loc_idx = act.topk(local_k, dim=1)              # (B, k)
        # Convert local idx -> global idx.
        glob_idx = loc_idx + start
        # Merge with running best.
        cat_vals = torch.cat([best_vals, loc_vals], dim=1)        # (B, 2k)
        cat_idx = torch.cat([best_idx, glob_idx], dim=1)
        # Take overall top-k.
        new_vals, sel = cat_vals.topk(k, dim=1)
        best_vals = new_vals
        best_idx = torch.gather(cat_idx, 1, sel)
        del tile, act, cat_vals, cat_idx

    return best_idx.cpu().numpy()


@torch.no_grad()
def topk_fullgpu(q_gpu, W_bank_gpu, k):
    """MIPS top-k when entire bank fits on GPU."""
    act = q_gpu @ W_bank_gpu.T
    return act.topk(k, dim=1).indices.cpu().numpy()


def main():
    t0 = time.time()
    tok_enc = tiktoken.get_encoding("gpt2")

    # ---- Load token stream (already on disk) ----
    heldout_path = OUT_DIR / "heldout.bin"
    if not heldout_path.exists():
        raise FileNotFoundError(
            f"{heldout_path} missing. Run prepare_heldout.py once to create it."
        )
    tokens_arr = np.memmap(heldout_path, dtype=np.uint16, mode="r")
    n_tokens = len(tokens_arr)
    n_chunks = n_tokens // CHUNK_SIZE
    print(f"heldout.bin: {n_tokens:,} tokens -> {n_chunks:,} chunks")

    # ---- Bank: whitening + cell weights ----
    print("\nopening bank.db (read-only)...")
    conn = sqlite3.connect(f"file:{BANK_DB}?mode=ro", uri=True)
    mu, W_white, max_norm = load_whitening(conn)
    print(f"whitening: max_norm={max_norm:.4f}")
    W_bank = load_bank_weights(conn)
    conn.close()
    n_cells = W_bank.shape[0]

    # ---- Decide GPU strategy ----
    bank_gb = W_bank.nbytes / 1e9
    fits_on_gpu = bank_gb < 5.5  # leave headroom for MiniLM + activations
    print(f"\nbank size: {bank_gb:.2f} GB -> "
          f"{'full-GPU MIPS' if fits_on_gpu else 'chunked-GPU MIPS'}")

    mu_gpu = torch.from_numpy(mu).to(DEVICE).float()
    W_white_gpu = torch.from_numpy(W_white).to(DEVICE).float()
    if fits_on_gpu:
        W_bank_gpu = torch.from_numpy(W_bank).to(DEVICE).float()
    else:
        W_bank_gpu = None   # tiled access from CPU memory

    # ---- MiniLM encoder on GPU ----
    encoder = MiniLMEncoder(DEVICE)

    # ---- MIPS for each chunk ----
    print(f"\nprecomputing neighbors for {n_chunks:,} chunks "
          f"(N_NEIGHBORS={N_NEIGHBORS})...")
    out = np.zeros((n_chunks, N_NEIGHBORS), dtype=np.uint32)
    t_enc = time.time()

    for batch_start in range(0, n_chunks, ENCODE_BATCH):
        batch_end = min(batch_start + ENCODE_BATCH, n_chunks)
        texts_batch = []
        for ci in range(batch_start, batch_end):
            s = ci * CHUNK_SIZE
            e = s + CHUNK_SIZE
            texts_batch.append(tok_enc.decode(tokens_arr[s:e].tolist()))

        raw = encoder.encode(texts_batch).float()
        centered = raw - mu_gpu[None, :]
        whitened = centered @ W_white_gpu
        q = whitened / (max_norm + 1e-8)                           # (B, 384)

        if fits_on_gpu:
            top_idx = topk_fullgpu(q, W_bank_gpu, N_NEIGHBORS)
        else:
            top_idx = topk_chunked(q, W_bank, N_NEIGHBORS, MIPS_TILE_CELLS)
        out[batch_start:batch_end] = top_idx

        if batch_start % (ENCODE_BATCH * 10) == 0:
            elapsed = time.time() - t_enc
            done = batch_end
            rate = done / max(elapsed, 0.01)
            eta = (n_chunks - done) / max(rate, 1)
            print(f"  {done:>7,}/{n_chunks:,} "
                  f"({100*done/n_chunks:5.1f}%)  "
                  f"{rate:.0f} chunks/s  ETA {eta:.0f}s")

    elapsed = time.time() - t_enc
    print(f"  done in {elapsed:.0f}s")

    np.save(OUT_DIR / "heldout_neighbors.npy", out)
    print(f"  wrote heldout_neighbors.npy: shape {out.shape}")

    # Update meta (best effort).
    meta_path = OUT_DIR / "heldout_meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    meta["n_bank_cells_at_recompute"] = int(n_cells)
    meta["recomputed_at"] = time.time()
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\ntotal time: {time.time()-t0:.0f}s, bank had {n_cells:,} cells")


if __name__ == "__main__":
    main()
