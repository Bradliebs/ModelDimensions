"""
================================================================================
train_retro_kfree_large.py — Knowledge-free RETRO at 404M scale
================================================================================

Same knowledge-free training as train_retro_kfree.py but with a 404M param
model (24 layers, 16 heads, 1024 embd). Designed for ~300M token corpus.

GPU memory: ~8.2 GB peak at batch=2, fits RTX 3070 (8.59 GB).

USAGE:
  cd H:\\MiniLM\\nanogpt
  H:\\MiniLM\\cc_service\\.venv\\Scripts\\python.exe train_retro_kfree_large.py
================================================================================
"""

import os
import time
import math
import json
from pathlib import Path

import numpy as np
import torch

from model_retro import RetroConfig, RetroGPT


# ---------------- Paths ----------------
DATA_DIR = Path("data") / "kfree"
BANK_DATA_DIR = Path("data") / "bank"
OUT_DIR = Path("out-retro-kfree-large")
OUT_DIR.mkdir(exist_ok=True)

RESUME_FILE = OUT_DIR / "ckpt_resume.pt"     # full state (preferred)
WARMSTART_FILE = OUT_DIR / "ckpt_best.pt"    # weights only (fallback)


# ---------------- Model config ----------------
config = RetroConfig(
    block_size=256,
    vocab_size=50304,
    n_layer=24,
    n_head=16,
    n_embd=1024,
    dropout=0.0,
    bias=False,
    chunk_size=64,
    n_neighbors=2,
    neighbor_len=64,
    cca_every=2,
)


# ---------------- Training ----------------
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 16           # effective batch = 32
MAX_ITERS = 15000               # ~123M token-presentations
WARMUP_ITERS = 300
LR_MAX = 2e-4                   # slightly lower for larger model
LR_MIN = 2e-5
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

EVAL_INTERVAL = 250
EVAL_BATCHES = 20
LOG_INTERVAL = 50

SEED = 1337

# 8-bit AdamW via bitsandbytes: quantizes optimizer state to free ~2.4GB VRAM.
# Cost: tiny precision hit (~0.1-0.3% absolute loss), affects val_with and val_without
# equally so the retrieval gap is preserved. Resume checkpoints are tagged with
# the optimizer type and rejected on mismatch.
USE_8BIT_OPTIM = True
OPTIM_TAG = "adamw8bit" if USE_8BIT_OPTIM else "adamw_fp32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
DTYPE = torch.bfloat16 if (DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
CTX = (
    torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
    if DEVICE_TYPE == "cuda"
    else torch.amp.autocast(device_type="cpu", enabled=False)
)


# ---------------- Data ----------------
class KFreeDataset:
    def __init__(self, split: str):
        self.split = split
        self.tokens = np.memmap(DATA_DIR / f"{split}.bin", dtype=np.uint16, mode="r")
        self.neighbors = np.load(DATA_DIR / f"{split}_neighbors.npy", mmap_mode="r")

        if not hasattr(KFreeDataset, "_cell_tokens"):
            ct_path = BANK_DATA_DIR / "cell_tokens.npy"
            KFreeDataset._cell_tokens = np.load(str(ct_path), mmap_mode="r")
        self.cell_tokens = KFreeDataset._cell_tokens

        self.K = config.block_size // config.chunk_size
        max_token_start = len(self.tokens) - config.block_size - 1
        self.max_chunk_idx = max_token_start // config.chunk_size
        assert self.max_chunk_idx >= self.K, f"{split} too short for one batch"
        self.max_chunk_idx = min(self.max_chunk_idx, len(self.neighbors) - self.K)

        print(f"  [{split}] {len(self.tokens):,} tokens, "
              f"{len(self.neighbors):,} chunks-with-neighbors, "
              f"sampling range chunk_idx [0, {self.max_chunk_idx})")

    def get_batch(self, batch_size: int):
        K = self.K
        Ln = config.neighbor_len
        chunk_starts = np.random.randint(0, self.max_chunk_idx, size=batch_size)

        x = np.empty((batch_size, config.block_size), dtype=np.int64)
        y = np.empty((batch_size, config.block_size), dtype=np.int64)
        nbrs = np.empty((batch_size, K, config.n_neighbors, Ln), dtype=np.int64)
        for b, c in enumerate(chunk_starts):
            tstart = int(c) * config.chunk_size
            x[b] = self.tokens[tstart : tstart + config.block_size].astype(np.int64)
            y[b] = self.tokens[tstart + 1 : tstart + 1 + config.block_size].astype(np.int64)
            nbr_idx = self.neighbors[int(c) : int(c) + K]
            nbrs[b] = self.cell_tokens[nbr_idx].astype(np.int64)
        return (
            torch.from_numpy(x).to(DEVICE),
            torch.from_numpy(y).to(DEVICE),
            torch.from_numpy(nbrs).to(DEVICE),
        )


# ---------------- LR schedule ----------------
def get_lr(it: int) -> float:
    if it < WARMUP_ITERS:
        return LR_MAX * (it + 1) / WARMUP_ITERS
    if it >= MAX_ITERS:
        return LR_MIN
    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LR_MIN + coeff * (LR_MAX - LR_MIN)


# ---------------- Optimizer ----------------
def build_optimizer(model):
    """Build AdamW or AdamW8bit using the same decay/no-decay split as the model."""
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    if USE_8BIT_OPTIM:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(optim_groups, lr=LR_MAX, betas=BETAS)
        print(f"  optimizer: bitsandbytes AdamW8bit ({len(decay_params)} decay, {len(nodecay_params)} no-decay)")
    else:
        opt = model.configure_optimizers(
            weight_decay=WEIGHT_DECAY,
            learning_rate=LR_MAX,
            betas=BETAS,
            device_type=DEVICE_TYPE,
        )
        print(f"  optimizer: torch.optim.AdamW (fp32)")
    return opt


# ---------------- Eval ----------------
@torch.no_grad()
def evaluate(model, ds: KFreeDataset, n_batches: int) -> tuple[float, float]:
    model.eval()
    losses_with, losses_without = [], []
    for _ in range(n_batches):
        x, y, nbrs = ds.get_batch(BATCH_SIZE)
        with CTX:
            _, lw = model(x, targets=y, neighbors=nbrs)
            _, lo = model(x, targets=y, neighbors=None)
        losses_with.append(lw.item())
        losses_without.append(lo.item())
    model.train()
    return sum(losses_with) / len(losses_with), sum(losses_without) / len(losses_without)


# ---------------- Train ----------------
def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"device={DEVICE}  dtype={DTYPE}")
    print(f"loading data from {DATA_DIR.resolve()}...")
    train_ds = KFreeDataset("train")
    val_ds = KFreeDataset("val")

    print(f"\nbuilding model: n_layer={config.n_layer}, n_head={config.n_head}, "
          f"n_embd={config.n_embd}, vocab={config.vocab_size}")
    model = RetroGPT(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.2f}M")

    optimizer = build_optimizer(model)

    scaler = torch.amp.GradScaler(device=DEVICE, enabled=(DTYPE == torch.float16))

    start_iter = 0
    best_val = float("inf")
    history = []

    if RESUME_FILE.exists():
        resume_path = RESUME_FILE
        try:
            _ck_peek = torch.load(RESUME_FILE, map_location="cpu", weights_only=False)
            ck_peek_type = _ck_peek.get("optim_type", "adamw_fp32")
            del _ck_peek
        except Exception as e:
            print(f"[resume] could not peek {RESUME_FILE}: {e}")
            ck_peek_type = None
            resume_path = None
        if ck_peek_type is not None and ck_peek_type != OPTIM_TAG:
            print(f"[resume] SKIPPING {RESUME_FILE.name}: saved with optim_type={ck_peek_type!r} "
                  f"but current run uses {OPTIM_TAG!r}; falling back to warm-start")
            resume_path = None
    else:
        resume_path = None

    if resume_path is not None:
        print(f"\n[resume] loading full state from {resume_path}")
        ck = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scaler.load_state_dict(ck["scaler_state"])
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["np_rng"])
        start_iter = ck["iter"]
        best_val = ck["best_val"]
        history = ck["history"]
        print(f"  resumed at iter {start_iter}  best_val={best_val:.4f}  history={len(history)} pts")
    elif WARMSTART_FILE.exists():
        print(f"\n[warmstart] loading weights only from {WARMSTART_FILE}")
        ck = torch.load(WARMSTART_FILE, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        start_iter = ck["iter"]
        best_val = ck["val_with"]
        hist_path = OUT_DIR / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
        print(f"  warm-started from iter {start_iter}  best_val={best_val:.4f}  history={len(history)} pts")
        print(f"  (optimizer/scaler/RNG fresh \u2014 expect a brief loss bump as Adam moments rebuild)")

    print(f"\ntraining: max_iters={MAX_ITERS}, batch={BATCH_SIZE} x grad_accum={GRAD_ACCUM_STEPS} "
          f"(effective {BATCH_SIZE*GRAD_ACCUM_STEPS}), warmup={WARMUP_ITERS}, "
          f"lr {LR_MAX} -> {LR_MIN} cosine")
    print(f"eval every {EVAL_INTERVAL} iters on {EVAL_BATCHES} batches")
    if start_iter > 0:
        print(f"continuing from iter {start_iter} \u2192 {MAX_ITERS}")
    print("=" * 78)

    t0 = time.time()
    running = []

    for it in range(start_iter, MAX_ITERS):
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(GRAD_ACCUM_STEPS):
            x, y, nbrs = train_ds.get_batch(BATCH_SIZE)
            with CTX:
                _, loss = model(x, targets=y, neighbors=nbrs)
                loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running.append(loss_accum)

        if (it + 1) % LOG_INTERVAL == 0:
            avg = sum(running) / len(running)
            running = []
            dt = time.time() - t0
            t0 = time.time()
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"iter {it+1:>5d}  lr {lr:.2e}  loss {avg:.4f}  "
                  f"({LOG_INTERVAL} iters in {dt:.1f}s)  peak_gpu={mem:.2f}GB")

        if (it + 1) % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            t_eval = time.time()
            val_with, val_without = evaluate(model, val_ds, EVAL_BATCHES)
            gap = val_without - val_with
            print(f"  >> [val]  with-nbrs {val_with:.4f}  no-nbrs {val_without:.4f}  "
                  f"gap {gap:+.4f}  ({EVAL_BATCHES} batches in {time.time()-t_eval:.1f}s)")
            history.append({
                "iter": it + 1,
                "val_with": val_with,
                "val_without": val_without,
                "gap": gap,
                "lr": lr,
            })
            if val_with < best_val:
                best_val = val_with
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "iter": it + 1,
                    "val_with": val_with,
                    "val_without": val_without,
                }, OUT_DIR / "ckpt_best.pt")
                print(f"  >> saved ckpt_best.pt  (val_with={val_with:.4f})")

            # Full resume checkpoint for crash protection (model + optimizer + scaler + RNG)
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "np_rng": np.random.get_state(),
                "config": config,
                "iter": it + 1,
                "best_val": best_val,
                "history": history,
                "optim_type": OPTIM_TAG,
            }, OUT_DIR / "ckpt_resume.pt")
            t0 = time.time()

            # Save history after each eval for crash protection
            with open(OUT_DIR / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    print("=" * 78)
    print("training done. saving final checkpoint.")
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "iter": MAX_ITERS,
    }, OUT_DIR / "ckpt.pt")

    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, val_ds, history


def main():
    model, val_ds, history = train()

    if history:
        last = history[-1]
        print("\n" + "=" * 78)
        print("FINAL VALIDATION (knowledge-free RETRO — 404M)")
        print("=" * 78)
        print(f"  iter {last['iter']}")
        print(f"  loss with bank neighbors    : {last['val_with']:.4f}")
        print(f"  loss without neighbors      : {last['val_without']:.4f}")
        print(f"  gap (no-nbrs - with-nbrs)   : {last['gap']:+.4f}")
        print("-" * 78)
        if last["gap"] > 0.3:
            print("[STRONG] Model heavily relies on bank retrieval.")
        elif last["gap"] > 0.1:
            print("[MODERATE] Model benefits from retrieval but also internalized patterns.")
        elif last["gap"] > 0.05:
            print("[WEAK] Small retrieval benefit.")
        else:
            print("[NULL] No measurable benefit from bank neighbors.")


if __name__ == "__main__":
    main()
