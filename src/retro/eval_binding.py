"""
================================================================================
eval_binding.py - Hebbian binding through the RETRO retrieval path
================================================================================

For each of N held-out chunks, measure whether *binding* the top-1 and top-2
retrieved cells into a single composite cell helps the model (by freeing up a
neighbor slot for top-3) -- or hurts (by mangling the fused text).

Per sample:
  1. Live-query the bank for the target chunk's text -> top-3 single cells
  2. /bind top1+top2 -> bound cell (re-query: where does it rank?)
  3. Build 3 neighbor sets for the target chunk:
       A (baseline):    [top1.text, top2.text]
       B (bound-concat):[concat(top1.text, top2.text)[:64 tok], top3.text]
       C (top1+top3):   [top1.text, top3.text]      (control)
  4. Compute per-token CE loss over the TARGET chunk only
  5. /delete the bound cell

The other K-1 chunks in the block use precomputed neighbors so all conditions
share the same surrounding context.

Outcomes:
  B < A:  binding adds compositional value (information density wins)
  B ~= A: binding is neutral (no scarcity-of-slots benefit)
  B > A:  binding loses signal (truncated concat hurts)
  C vs A: confirms top-2 has marginal value

USAGE:
  cd H:\\MiniLM\\nanogpt
  H:\\MiniLM\\cc_service\\.venv\\Scripts\\python.exe eval_binding.py \\
      --ckpt out-retro-kfree-large/ckpt_best.pt --n-samples 200 --verbose

REQUIRES:
  - cc_service running at 127.0.0.1:8766 with full Wikipedia bank
  - data/bank/heldout.bin + heldout_neighbors.npy + cell_tokens.npy
================================================================================
"""

import argparse
import math
import time
from pathlib import Path

import httpx
import numpy as np
import tiktoken
import torch

from model_retro import RetroConfig, RetroGPT


_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data" / "bank"
CKPT_PATH = _SCRIPT_DIR / "out-retro-kfree-large" / "ckpt_best.pt"
SERVICE_URL = "http://127.0.0.1:8766"
EOT_TOKEN = 50256
SEED = 1357

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
DTYPE = (torch.bfloat16
         if (DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported())
         else torch.float16)
CTX = (torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
       if DEVICE_TYPE == "cuda"
       else torch.amp.autocast(device_type="cpu", enabled=False))


# ---- Bank client -----------------------------------------------------------
_client: httpx.Client | None = None


def bank_query(text: str, top_k: int = 10,
               include_silent: bool = True) -> list[dict]:
    """Query the bank. include_silent=True returns top-K by raw similarity
    regardless of per-cell firing threshold (needed for retrieval-style use)."""
    r = _client.post("/query", json={
        "text": text, "top_k": top_k, "include_silent": include_silent,
    })
    r.raise_for_status()
    return r.json().get("hits", [])


def bank_bind(source_cell_ids: list[int], label: str | None = None) -> dict:
    r = _client.post("/bind", json={
        "source_cell_ids": source_cell_ids,
        "label": label,
    })
    r.raise_for_status()
    return r.json()


def bank_delete(cell_id: int) -> bool:
    r = _client.delete(f"/cells/{cell_id}")
    if r.status_code == 404:
        return False
    r.raise_for_status()
    return True


# ---- Neighbor construction -------------------------------------------------
def text_to_neighbor_toks(text: str, enc: tiktoken.Encoding,
                          neighbor_len: int) -> np.ndarray:
    """Tokenize text, truncate, right-pad with EOT (matches training distribution)."""
    toks = enc.encode(text or "", allowed_special={"<|endoftext|>"})
    toks = toks[:neighbor_len]
    if len(toks) < neighbor_len:
        toks = toks + [EOT_TOKEN] * (neighbor_len - len(toks))
    return np.asarray(toks, dtype=np.int64)


def interleaved_concat_toks(text_a: str, text_b: str, enc: tiktoken.Encoding,
                            neighbor_len: int) -> np.ndarray:
    """Build a single neighbor by taking the first half of each source text.
    Guarantees both sources are represented even when each text alone exceeds
    neighbor_len. Pads with EOT if either half is short.
    """
    half = neighbor_len // 2
    toks_a = enc.encode(text_a or "", allowed_special={"<|endoftext|>"})[:half]
    toks_b = enc.encode(text_b or "", allowed_special={"<|endoftext|>"})[:half]
    combined = toks_a + toks_b
    combined = combined[:neighbor_len]
    if len(combined) < neighbor_len:
        combined = combined + [EOT_TOKEN] * (neighbor_len - len(combined))
    return np.asarray(combined, dtype=np.int64)


def build_neighbors(
    bg_neighbors: np.ndarray,
    target_chunk_idx: int,
    target_nbr_toks: list[np.ndarray],
    K: int,
    n_neighbors: int,
    neighbor_len: int,
) -> torch.Tensor:
    """Stitch background neighbors and target chunk's neighbors into (1,K,k,L_n).
    target_nbr_toks: list[n_neighbors] of pre-built (neighbor_len,) int64 arrays.
    """
    out = np.full((1, K, n_neighbors, neighbor_len), EOT_TOKEN, dtype=np.int64)
    bg_i = 0
    for k in range(K):
        if k == target_chunk_idx:
            for ni in range(n_neighbors):
                if ni < len(target_nbr_toks):
                    out[0, k, ni] = target_nbr_toks[ni]
        else:
            out[0, k] = bg_neighbors[bg_i]
            bg_i += 1
    return torch.from_numpy(out).to(DEVICE)


@torch.no_grad()
def target_chunk_loss(model, x, y, neighbors, target_chunk_idx, chunk_size):
    """CE loss restricted to the target chunk's positions via ignore_index=-1."""
    y_masked = y.clone()
    s = target_chunk_idx * chunk_size
    e = s + chunk_size
    y_masked[:, :s] = -1
    y_masked[:, e:] = -1
    with CTX:
        _, loss = model(x, targets=y_masked, neighbors=neighbors)
    return loss.item()


# ---- Main ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT_PATH))
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--target-chunk", type=int, default=3,
                   help="Which chunk in the block to manipulate (0..K-1). "
                        "Default 3 = last chunk (most preceding context).")
    p.add_argument("--service-url", default=SERVICE_URL)
    p.add_argument("--verbose", action="store_true",
                   help="Print details for first sample")
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    global _client
    _client = httpx.Client(base_url=args.service_url, timeout=60)

    # ---- Verify bank --------------------------------------------------------
    info = _client.get("/info").json()
    print(f"bank: {info['n_cells']:,} cells, dim={info['dim']}, "
          f"db={info['db_path']}")

    # ---- Load model ---------------------------------------------------------
    print(f"loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    config: RetroConfig = ckpt["config"]
    K = config.block_size // config.chunk_size
    print(f"  n_layer={config.n_layer}, n_embd={config.n_embd}, "
          f"chunk_size={config.chunk_size}, K={K}, "
          f"n_neighbors={config.n_neighbors}, neighbor_len={config.neighbor_len}")
    model = RetroGPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    assert 0 <= args.target_chunk < K, f"--target-chunk must be in [0,{K})"

    enc = tiktoken.get_encoding("gpt2")

    # ---- Load held-out data -------------------------------------------------
    dd = Path(args.data_dir)
    tokens = np.memmap(dd / "heldout.bin", dtype=np.uint16, mode="r")
    bg_nbr_idx = np.load(dd / "heldout_neighbors.npy", mmap_mode="r")
    cell_tokens = np.load(dd / "cell_tokens.npy", mmap_mode="r")
    max_token_start = len(tokens) - config.block_size - 1
    max_chunk_idx = min(max_token_start // config.chunk_size,
                        len(bg_nbr_idx) - K)
    print(f"  heldout: {len(tokens):,} tokens, "
          f"{len(bg_nbr_idx):,} chunk-neighbors, "
          f"{len(cell_tokens):,} bank cells (precomputed)")
    print(f"  target chunk index within block = {args.target_chunk}\n")

    # ---- Run experiment -----------------------------------------------------
    losses_a, losses_b, losses_c = [], [], []
    bound_ranks: list[int | None] = []
    n_skipped = 0
    n_bound_in_top3 = 0
    t0 = time.time()

    for si in range(args.n_samples):
        chunk_idx = int(np.random.randint(0, max_chunk_idx))
        tstart = chunk_idx * config.chunk_size
        x_np = tokens[tstart:tstart + config.block_size].astype(np.int64)
        y_np = tokens[tstart + 1:tstart + 1 + config.block_size].astype(np.int64)
        x = torch.from_numpy(x_np[None]).to(DEVICE)
        y = torch.from_numpy(y_np[None]).to(DEVICE)

        # Target chunk text within the block
        tc_s = args.target_chunk * config.chunk_size
        tc_e = tc_s + config.chunk_size
        target_text = enc.decode(x_np[tc_s:tc_e].tolist())

        # Background neighbors: every chunk EXCEPT the target one
        bg = np.empty((K - 1, config.n_neighbors, config.neighbor_len),
                      dtype=np.int64)
        bg_i = 0
        for k in range(K):
            if k == args.target_chunk:
                continue
            nbr_idx = bg_nbr_idx[chunk_idx + k]
            bg[bg_i] = cell_tokens[nbr_idx].astype(np.int64)
            bg_i += 1

        # Live retrieval for target chunk
        try:
            hits = bank_query(target_text, top_k=10)
        except Exception as e:
            print(f"  WARN: query failed at sample {si}: {e}")
            n_skipped += 1
            continue

        single_hits = [h for h in hits
                       if h.get("kind") == "single" and h.get("source_text")]
        if len(single_hits) < 3:
            n_skipped += 1
            continue
        top1, top2, top3 = single_hits[0], single_hits[1], single_hits[2]

        # Bind top1 + top2
        try:
            bind_res = bank_bind([top1["cell_id"], top2["cell_id"]],
                                 label=f"eval_binding_{si}")
        except Exception as e:
            print(f"  WARN: bind failed at sample {si}: {e}")
            n_skipped += 1
            continue
        bound_cell_id = bind_res["bound_cell_id"]

        try:
            # Re-query: where does bound cell rank now?
            rehits = bank_query(target_text, top_k=10)
            rank: int | None = None
            for ri, h in enumerate(rehits):
                if h["cell_id"] == bound_cell_id:
                    rank = ri + 1
                    break
            bound_ranks.append(rank)
            if rank is not None and rank <= 3:
                n_bound_in_top3 += 1

            t1_text = top1.get("source_text") or ""
            t2_text = top2.get("source_text") or ""
            t3_text = top3.get("source_text") or ""

            # Pre-tokenize each candidate neighbor sequence.
            t1_toks = text_to_neighbor_toks(t1_text, enc, config.neighbor_len)
            t2_toks = text_to_neighbor_toks(t2_text, enc, config.neighbor_len)
            t3_toks = text_to_neighbor_toks(t3_text, enc, config.neighbor_len)
            # B's bound-cell text = first half of top1 + first half of top2.
            # Guarantees both sources are represented in a single neighbor slot.
            bound_toks = interleaved_concat_toks(
                t1_text, t2_text, enc, config.neighbor_len)

            nbrs_a = build_neighbors(bg, args.target_chunk, [t1_toks, t2_toks],
                                     K, config.n_neighbors, config.neighbor_len)
            nbrs_b = build_neighbors(bg, args.target_chunk, [bound_toks, t3_toks],
                                     K, config.n_neighbors, config.neighbor_len)
            nbrs_c = build_neighbors(bg, args.target_chunk, [t1_toks, t3_toks],
                                     K, config.n_neighbors, config.neighbor_len)

            la = target_chunk_loss(model, x, y, nbrs_a,
                                   args.target_chunk, config.chunk_size)
            lb = target_chunk_loss(model, x, y, nbrs_b,
                                   args.target_chunk, config.chunk_size)
            lc = target_chunk_loss(model, x, y, nbrs_c,
                                   args.target_chunk, config.chunk_size)
            losses_a.append(la)
            losses_b.append(lb)
            losses_c.append(lc)

            if args.verbose and len(losses_a) == 1:
                print("=" * 72)
                print(f"SAMPLE 0  (chunk_idx={chunk_idx})")
                print("=" * 72)
                print(f"target text:\n  {target_text[:240]}")
                print(f"top1 [{top1['cell_id']}] m={top1['margin']:.3f}:\n  "
                      f"{t1_text[:160]}")
                print(f"top2 [{top2['cell_id']}] m={top2['margin']:.3f}:\n  "
                      f"{t2_text[:160]}")
                print(f"top3 [{top3['cell_id']}] m={top3['margin']:.3f}:\n  "
                      f"{t3_text[:160]}")
                print(f"bound cell [{bound_cell_id}] "
                      f"theta_readout={bind_res['theta_readout']:.4f} "
                      f"alignment={bind_res['alignment_to_mean']:.4f}")
                print(f"  rank after binding (similarity, top-10): {rank}")
                bound_decoded = enc.decode(bound_toks.tolist())
                print(f"  bound-concat text [first/second half]:\n  "
                      f"{bound_decoded[:200]}")
                print(f"  loss A (baseline, top1+top2)       = {la:.4f}")
                print(f"  loss B (bound-concat, concat+top3) = {lb:.4f}")
                print(f"  loss C (control, top1+top3)        = {lc:.4f}")
                print("=" * 72 + "\n")

        finally:
            try:
                bank_delete(bound_cell_id)
            except Exception as e:
                print(f"  WARN: failed to delete bound cell {bound_cell_id}: {e}")

        if (si + 1) % 20 == 0 and len(losses_a) > 0:
            ma = float(np.mean(losses_a))
            mb = float(np.mean(losses_b))
            mc = float(np.mean(losses_c))
            n_in = sum(1 for r in bound_ranks if r is not None and r <= 3)
            print(f"  [{si+1:>4d}/{args.n_samples}] "
                  f"A={ma:.4f}  B={mb:.4f}  C={mc:.4f}  "
                  f"B-A={mb-ma:+.4f}  C-A={mc-ma:+.4f}  "
                  f"bound_in_top3={n_in}/{len(bound_ranks)}  "
                  f"skipped={n_skipped}")

    elapsed = time.time() - t0
    n = len(losses_a)
    if n == 0:
        print("\nERROR: no successful samples. Check bank service and data.")
        return

    def stats(vals):
        m = float(np.mean(vals))
        se = (float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
              if len(vals) > 1 else 0.0)
        return m, se

    ma, sa = stats(losses_a)
    mb, sb = stats(losses_b)
    mc, sc = stats(losses_c)
    diffs_ba = np.array(losses_b) - np.array(losses_a)
    diffs_ca = np.array(losses_c) - np.array(losses_a)
    mba = float(diffs_ba.mean())
    sba = (float(diffs_ba.std(ddof=1) / math.sqrt(n))
           if n > 1 else 0.0)
    mca = float(diffs_ca.mean())
    sca = (float(diffs_ca.std(ddof=1) / math.sqrt(n))
           if n > 1 else 0.0)

    print()
    print("=" * 72)
    print("HEBBIAN BINDING THROUGH RETRO - RESULTS")
    print("=" * 72)
    print(f"  n samples:   {n}  (skipped {n_skipped})")
    print(f"  eval time:   {elapsed:.1f}s ({elapsed/max(n,1):.2f}s/sample)")
    print(f"  target chunk index in block: {args.target_chunk}")
    print()
    print(f"  A baseline       (top1, top2)        : "
          f"{ma:.4f}  (+/-{sa:.4f})")
    print(f"  B bound-concat   (concat[:64], top3) : "
          f"{mb:.4f}  (+/-{sb:.4f})")
    print(f"  C control        (top1, top3)        : "
          f"{mc:.4f}  (+/-{sc:.4f})")
    print()
    print(f"  delta B-A: {mba:+.4f} (+/-{sba:.4f})  "
          f"-- binding effect vs baseline (paired)")
    print(f"  delta C-A: {mca:+.4f} (+/-{sca:.4f})  "
          f"-- cost of dropping top2 without replacement (paired)")
    print()
    print(f"  bound cell ranked in top-3 after binding: "
          f"{n_bound_in_top3}/{n} ({100*n_bound_in_top3/max(n,1):.1f}%)")
    rank_counts: dict[object, int] = {}
    for r in bound_ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    rd = sorted(rank_counts.items(),
                key=lambda x: (x[0] is None, x[0] if x[0] is not None else 99))
    print(f"  bound cell rank distribution: {rd}")

    print()
    threshold = 2.0  # ~95% paired-t one-sided
    if mba < 0 and abs(mba) > threshold * sba:
        print(f"  -> B < A (B-A significantly negative). "
              f"Binding ADDS value through RETRO.")
    elif mba > 0 and abs(mba) > threshold * sba:
        print(f"  -> B > A (B-A significantly positive). "
              f"Binding LOSES information through RETRO.")
    else:
        print(f"  -> B ~= A (B-A not significant at 2-sigma). "
              f"Binding is neutral at this neighbor count.")
    print("=" * 72)


if __name__ == "__main__":
    main()
