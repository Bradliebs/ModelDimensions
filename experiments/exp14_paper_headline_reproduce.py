"""
exp14_paper_headline_reproduce.py
=================================

Drives `src/retro/eval_retro_heldout.py` against the two checkpoints named in
the paper and writes `results/paper_headline_reproduce.json` with measured
gaps alongside the paper's claimed numbers.

Paper reference: docs/Knowledge_Free_RETRO_Paper.docx §4.2, §5.4.

This script is the *producer* for the JSON artifact that
`evals/test_paper_headline_reproduces.py` asserts against.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "src" / "retro"
sys.path.insert(0, str(EVAL_DIR))

from eval_retro_heldout import (  # noqa: E402
    HeldoutDataset,
    evaluate_with_stats,
)
from model_retro import RetroGPT  # noqa: E402


# Defaults match the paper's §4.2 reported numbers (100 batches × 8 samples).
N_BATCHES = 100
BATCH_SIZE = 8
SEED = 42

CHECKPOINTS = [
    {
        "label": "55M_bank_trained",
        "path": r"H:\MiniLM\nanogpt\out-retro-bank\ckpt_best.pt",
    },
    {
        "label": "404M_kfree",
        "path": r"H:\MiniLM\nanogpt\out-retro-kfree-large\ckpt_best.pt",
    },
]

DATA_DIR = Path(r"H:\MiniLM\nanogpt\data\bank")

# From paper §4.2 — what we're trying to reproduce.
PAPER_CLAIMS = {
    "55M_bank_trained": {
        "semantic_gap": 0.164,   # val_with_random - val_with_real
        "random_harm": -0.003,   # val_without - val_with_random
    },
    "404M_kfree": {
        "semantic_gap": 0.167,
        "random_harm": -0.041,
    },
}

TOLERANCE_NATS = 0.05


def run_checkpoint(label: str, path: str) -> dict:
    print(f"\n{'=' * 72}\n{label}  ←  {path}\n{'=' * 72}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = RetroGPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    del ckpt

    ds = HeldoutDataset(config, data_dir=DATA_DIR)

    t0 = time.time()
    results = evaluate_with_stats(model, ds, N_BATCHES, BATCH_SIZE)
    elapsed = time.time() - t0

    val_without = results["none"]["mean"]
    val_with_random = results["random"]["mean"]
    val_with_real = results["real"]["mean"]

    semantic_gap = val_with_random - val_with_real
    total_gap = val_without - val_with_real
    random_harm = val_without - val_with_random

    out = {
        "checkpoint_path": path,
        "val_without": val_without,
        "val_with_random": val_with_random,
        "val_with_real": val_with_real,
        "val_with_real1": results["real1"]["mean"],
        "stderr_none": results["none"]["stderr"],
        "stderr_random": results["random"]["stderr"],
        "stderr_real": results["real"]["stderr"],
        "semantic_gap_real_vs_random": semantic_gap,
        "total_gap_real_vs_without": total_gap,
        "random_harm_none_vs_random": random_harm,
        "n_batches": N_BATCHES,
        "batch_size": BATCH_SIZE,
        "elapsed_sec": elapsed,
        "device": device,
    }

    # Free GPU before next checkpoint.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return out


def agreement(measured: float, claimed: float, tol: float) -> str:
    delta = abs(measured - claimed)
    return "match" if delta <= tol else "drift"


def main():
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "paper_headline_reproduce.json"

    per_ckpt = {}
    for spec in CHECKPOINTS:
        per_ckpt[spec["label"]] = run_checkpoint(spec["label"], spec["path"])

    agreement_block = {}
    for label, claims in PAPER_CLAIMS.items():
        m = per_ckpt[label]
        sem = agreement(
            m["semantic_gap_real_vs_random"], claims["semantic_gap"], TOLERANCE_NATS
        )
        harm = agreement(
            m["random_harm_none_vs_random"], claims["random_harm"], TOLERANCE_NATS
        )
        agreement_block[label] = {
            "semantic_gap": sem,
            "random_harm": harm,
        }

    artifact = {
        "produced_by": "experiments/exp14_paper_headline_reproduce.py",
        "config": {
            "n_batches": N_BATCHES,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "tolerance_nats": TOLERANCE_NATS,
            "data_dir": str(DATA_DIR),
        },
        "paper_claims": PAPER_CLAIMS,
        "agreement": agreement_block,
        "measurements": per_ckpt,
    }

    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nwrote {out_path}")

    # Echo decision summary to stdout for the operator.
    print("\n" + "=" * 72)
    print("AGREEMENT SUMMARY")
    print("=" * 72)
    for label, claims in PAPER_CLAIMS.items():
        m = per_ckpt[label]
        a = agreement_block[label]
        print(
            f"  {label:>20s}: "
            f"semantic_gap measured={m['semantic_gap_real_vs_random']:+.4f} "
            f"claimed={claims['semantic_gap']:+.4f} [{a['semantic_gap']}]"
        )
        print(
            f"  {' ':>20s}: "
            f"random_harm  measured={m['random_harm_none_vs_random']:+.4f} "
            f"claimed={claims['random_harm']:+.4f} [{a['random_harm']}]"
        )


if __name__ == "__main__":
    main()
