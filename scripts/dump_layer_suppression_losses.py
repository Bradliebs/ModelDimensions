"""Dump per-CCA-layer suppression losses for layer-attribution analysis.

For each CCA layer k in the model, runs the held-out 'real' condition
with that layer's CCA disabled (block.use_cca toggled to False, no
permanent modification). All conditions share the same chunk indices
per batch so the comparison is paired.

Output JSON shape matches what scripts/measure_layer_attribution.py expects:

    {
      "loss_without": float,
      "loss_real_full": float,
      "loss_real_with_layer_suppressed": {"<k>": float, ...},
      "_meta": {...}
    }

Usage:
    python scripts/dump_layer_suppression_losses.py \\
        --out results/layer_suppression_losses.json
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

from src.retro import eval_retro_heldout as ehe  # noqa: E402
from src.retro.model_retro import RetroGPT  # noqa: E402

DEFAULT_NANOGPT_ROOT = Path(r"H:\MiniLM\nanogpt")
DEFAULT_CKPT = DEFAULT_NANOGPT_ROOT / "out-retro-bank" / "ckpt_best.pt"
DEFAULT_DATA_DIR = DEFAULT_NANOGPT_ROOT / "data" / "bank"


def _run_condition(model, ds, chunk_indices_per_batch, batch_size, mode) -> float:
    """Return mean loss across pre-sampled batches for one mode."""
    losses = []
    for chunk_indices in chunk_indices_per_batch:
        x, y, nbrs = ds.get_batch(batch_size, chunk_indices, mode=mode)
        with ehe.CTX:
            _, loss = model(x, targets=y, neighbors=nbrs)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=ehe.SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.ckpt.exists():
        parser.error(f"checkpoint not found: {args.ckpt}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[suppress] device={ehe.DEVICE} dtype={ehe.DTYPE}")
    print(f"[suppress] loading checkpoint {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location=ehe.DEVICE, weights_only=False)
    config = ckpt["config"]

    model = RetroGPT(config).to(ehe.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cca_layers = list(model.cca_layers)
    print(f"[suppress] CCA layers under test: {cca_layers}")

    ds = ehe.HeldoutDataset(config, data_dir=args.data_dir)

    # Pre-sample chunk indices once so every condition sees the same batches.
    rng = np.random.RandomState(args.seed)
    chunk_indices_per_batch = [
        rng.randint(0, ds.max_chunk_idx, size=args.batch_size)
        for _ in range(args.n_batches)
    ]

    t0 = time.time()
    with torch.no_grad():
        print("[suppress] running mode=none baseline")
        loss_without = _run_condition(
            model, ds, chunk_indices_per_batch, args.batch_size, mode="none"
        )
        print(f"  loss_without = {loss_without:.4f}")

        print("[suppress] running mode=real full")
        loss_real_full = _run_condition(
            model, ds, chunk_indices_per_batch, args.batch_size, mode="real"
        )
        print(f"  loss_real_full = {loss_real_full:.4f}")

        suppressed: dict[int, float] = {}
        for k in cca_layers:
            block = model.transformer.h[k]
            # Sanity: this block actually owns a CCA module.
            assert block.use_cca, f"layer {k} expected to have CCA"
            block.use_cca = False
            try:
                loss = _run_condition(
                    model, ds, chunk_indices_per_batch, args.batch_size, mode="real"
                )
            finally:
                block.use_cca = True
            suppressed[k] = loss
            delta = loss - loss_real_full
            print(f"  layer {k} suppressed: real={loss:.4f}  (delta vs full {delta:+.4f})")

    elapsed = time.time() - t0
    print(f"[suppress] done in {elapsed:.1f}s")

    payload = {
        "loss_without": float(loss_without),
        "loss_real_full": float(loss_real_full),
        "loss_real_with_layer_suppressed": {str(k): float(v) for k, v in suppressed.items()},
        "_meta": {
            "ckpt": str(args.ckpt),
            "n_batches": args.n_batches,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "elapsed_seconds": elapsed,
            "cca_layers": cca_layers,
            "ckpt_iter": ckpt.get("iter"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[suppress] wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
