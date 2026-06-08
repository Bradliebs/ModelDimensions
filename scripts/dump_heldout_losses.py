"""Dump 3-condition held-out losses to JSON.

Wraps the existing ``src/retro/eval_retro_heldout.evaluate`` function and
writes ``{"none": ..., "random": ..., "real": ...}`` to a file consumable
by ``scripts/run_acceptance_tests.py``.

The underlying script's defaults are kept: 100 batches x 8 samples on the
held-out shard at H:\\MiniLM\\nanogpt\\data\\bank. Override via flags.

Usage:
    python scripts/dump_heldout_losses.py --out results/heldout_losses.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RETRO_DIR = ROOT / "src" / "retro"
for p in (str(ROOT), str(RETRO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Importing src.retro.eval_retro_heldout brings in DEVICE/CTX globals and
# ensures the script's own sys.path setup runs before model_retro imports.
from src.retro import eval_retro_heldout as ehe  # noqa: E402
from src.retro.model_retro import RetroGPT  # noqa: E402

# The upstream script's CKPT_PATH / DATA_DIR resolve relative to its own
# directory (a leftover from when it was run from H:\MiniLM\nanogpt). The
# real artifacts live in H:\MiniLM\nanogpt; default there so the wrapper
# is runnable from anywhere without --ckpt / --data-dir flags.
DEFAULT_NANOGPT_ROOT = Path(r"H:\MiniLM\nanogpt")
DEFAULT_CKPT = DEFAULT_NANOGPT_ROOT / "out-retro-bank" / "ckpt_best.pt"
DEFAULT_DATA_DIR = DEFAULT_NANOGPT_ROOT / "data" / "bank"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=ehe.SEED)
    parser.add_argument("--out", type=Path, required=True,
                        help="Where to write the {none, random, real} JSON")
    args = parser.parse_args(argv)

    if not args.ckpt.exists():
        parser.error(f"checkpoint not found: {args.ckpt}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[heldout] device={ehe.DEVICE} dtype={ehe.DTYPE}")
    print(f"[heldout] loading checkpoint {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location=ehe.DEVICE, weights_only=False)
    config = ckpt["config"]
    print(
        f"[heldout] config: n_layer={config.n_layer} n_head={config.n_head} "
        f"n_embd={config.n_embd} vocab={config.vocab_size} "
        f"block={config.block_size} chunk={config.chunk_size} "
        f"n_neighbors={config.n_neighbors} neighbor_len={config.neighbor_len}"
    )
    print(
        f"[heldout] checkpoint iter={ckpt.get('iter', '?')} "
        f"val_with={ckpt.get('val_with', float('nan')):.4f} "
        f"val_without={ckpt.get('val_without', float('nan')):.4f}"
    )

    model = RetroGPT(config).to(ehe.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[heldout] model params: {n_params/1e6:.2f}M")

    ds = ehe.HeldoutDataset(config, data_dir=args.data_dir)

    print(
        f"[heldout] running 3-condition eval: "
        f"{args.n_batches} batches x {args.batch_size} samples"
    )
    t0 = time.time()
    losses = ehe.evaluate(model, ds, args.n_batches, args.batch_size)
    elapsed = time.time() - t0
    print(f"[heldout] eval done in {elapsed:.1f}s")
    for mode in ("none", "random", "real"):
        print(f"  {mode:>6s}: {losses[mode]:.4f} nats")
    benefit = losses["none"] - losses["real"]
    hurt = losses["random"] - losses["none"]
    print(f"  semantic gap (none - real)    : {benefit:+.4f}")
    print(f"  ignorance hurt (random - none): {hurt:+.4f}")

    payload = {
        "none": float(losses["none"]),
        "random": float(losses["random"]),
        "real": float(losses["real"]),
        "_meta": {
            "ckpt": str(args.ckpt),
            "n_batches": args.n_batches,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "elapsed_seconds": elapsed,
            "model_params_millions": n_params / 1e6,
            "ckpt_iter": ckpt.get("iter"),
            "ckpt_val_with": float(ckpt.get("val_with", float("nan"))),
            "ckpt_val_without": float(ckpt.get("val_without", float("nan"))),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[heldout] wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
