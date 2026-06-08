"""exp31 — Binding through the CCA path.

Question: when a single retrieved neighbor token cell packs m=6 atomic facts
together, can the 55M RETRO model's Chunked Cross-Attention extract the
specific fact needed for next-token prediction?

The bank-side binding diagnostic (m=6 PASS at safety_margin=0.02) measures
whether the MiniLM bank can retrieve a bound vector cleanly. It does NOT
measure whether CCA can use a bound cell once retrieved -- CCA attends to
neighbor token sequences (shape B,K,k,Ln = B,4,2,64), not to the 384-d
MiniLM vectors that drive retrieval. This experiment closes that gap.

Method (synthetic, additive, read-only on the checkpoint):
  - Two bundles of 6 atomic fictional facts each (no parametric leakage).
  - One probe per fact in bundle A: a short prefix whose correct continuation
    requires that fact.
  - For each probe, evaluate model NLL on the continuation tokens under
    four neighbor conditions:
        none        : neighbors=None (CCA off)
        single      : 1 fact (the relevant one) in the 64-token cell
        bound       : all 6 facts of bundle A packed into the 64-token cell
        distractor  : all 6 facts of bundle B (unrelated) packed similarly
    All K=4 chunk slots and both k=2 neighbor slots receive the same cell so
    no chunk gets "lucky".

Verdict:
  extraction_ratio = (nll_distractor - nll_bound) / (nll_distractor - nll_single)
  PASS if >= 0.5 (CCA extracts at least half the per-fact signal from a
  bound cell).
  INCONCLUSIVE if single_benefit < 0.05 nats (CCA not engaging on a single
  fact either -- can't test extraction from a bundle).

Usage:
    python experiments/exp31_binding_through_cca.py \
        --out-json reports/binding_through_cca.json \
        --out-md reports/binding_through_cca.md
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
import torch.nn.functional as F  # noqa: E402
import tiktoken  # noqa: E402

from src.retro import eval_retro_heldout as ehe  # noqa: E402
from src.retro.model_retro import RetroGPT  # noqa: E402


DEFAULT_CKPT = Path(r"H:\MiniLM\nanogpt\out-retro-bank\ckpt_best.pt")
PASS_RATIO = 0.5
INCONCLUSIVE_SINGLE_BENEFIT = 0.05  # nats per token

# Each fact pairs a unique fictional proper noun (the answer) with a unique
# property cue (matched by the probe prefix). All answers are rare
# multi-token names the model cannot guess from base distribution, so any
# NLL reduction on the answer tokens must come from the neighbor cell.
#
# Telegraphic "cue: name" format with newline separator so all 6 facts fit
# inside the 64-token neighbor cell with margin -- truncation would bias
# the bound condition unfairly (a missing fact would force CCA to fail).
FACTS_A = [
    "Mordwin capital: Zorath",
    "Sapphire lake: Tellaris",
    "Druid sacred forest: Riftwood",
    "Telnar minted coins: Brennix",
    "Ninety-meter watchtower: Korvath",
    "Ferroglass forging town: Daelin",
]

PROBES = [
    ("In the kingdom of Mordwin, the capital is the city of",       " Zorath",   0),
    ("The lake renowned for its sapphire water is named",            " Tellaris", 1),
    ("The druids honor the sacred forest known as",                  " Riftwood", 2),
    ("The official coins minted in Telnar are called",               " Brennix",  3),
    ("The ninety-meter watchtower on the cliff is named",            " Korvath",  4),
    ("The town renowned for forging ferroglass is",                  " Daelin",   5),
]

# Distractor bundle: zero lexical overlap with FACTS_A answers or probe cues.
FACTS_B = [
    "Dawn bell: Hesht-Var",
    "Wyvern caverns: Skarn",
    "Sea-mage lighthouse: Velnor",
    "Captain's vessel: Marigold",
    "Red wine vineyard: Asher",
    "Ancient ruler: Pelthorn",
]

# Neutral statements: topic-orthogonal to FACTS_A (no shared cue or answer
# words). Used to pad the "single-fact" condition so it has the same cell
# structure as bound (6 short statements) -- isolates the binding question
# from a structural confound about EOT-padding density.
NEUTRALS = [
    "Rain falls each spring.",
    "Birds migrate by night.",
    "Stones polish over time.",
    "Rivers carve deep canyons.",
    "Echoes fade in caves.",
    "Clouds gather before storms.",
]

BUNDLE_SEPARATOR = "\n"

FILLER = (
    "The following observations were recorded during the survey. "
    "Researchers documented several routine findings on the expedition. "
    "Notes from the journal were copied carefully into the ledger. "
)


def pack_bundle(facts: list[str], tok: tiktoken.Encoding, target_len: int = 64,
                pad_token: int | None = None) -> np.ndarray:
    """Pack facts into a target_len token cell.

    Raises ValueError if facts overflow target_len -- a truncated bound cell
    would bias the experiment by removing a fact that the probe needs.
    """
    if pad_token is None:
        pad_token = tok.eot_token
    text = BUNDLE_SEPARATOR.join(facts)
    ids = tok.encode(text)
    if len(ids) > target_len:
        raise ValueError(
            f"bundle has {len(ids)} tokens > target_len={target_len}; "
            f"shorten facts or reduce bundle size to keep all facts present"
        )
    if len(ids) < target_len:
        ids = ids + [pad_token] * (target_len - len(ids))
    return np.asarray(ids, dtype=np.int64)


def build_probe_input(prefix: str, answer: str, tok: tiktoken.Encoding,
                      block_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (idx, targets, n_answer_tokens) for one probe.

    The prefix+answer is placed at the END of the block. Filler fills the rest.
    Targets are -1 (ignore) everywhere except the answer positions, where they
    are the expected token ids. We score only how confidently the model
    predicts each answer token given the prefix + neighbor.
    """
    pre_ids = tok.encode(prefix)
    ans_ids = tok.encode(answer)
    prompt_ids = pre_ids + ans_ids

    # Build enough filler to fill (block_size + 1) tokens (one extra for shift).
    filler_ids = tok.encode(FILLER)
    while len(filler_ids) + len(prompt_ids) < block_size + 1:
        filler_ids = filler_ids + filler_ids

    full_ids = (filler_ids + prompt_ids)[-(block_size + 1):]
    idx = np.asarray(full_ids[:block_size], dtype=np.int64)
    tgt_full = np.asarray(full_ids[1:block_size + 1], dtype=np.int64)

    tgt = np.full(block_size, -1, dtype=np.int64)
    # Predicting position p uses logits at position p-1. The answer tokens
    # occupy the last len(ans_ids) positions of full_ids, i.e. tgt_full[-len(ans_ids):].
    ans_start_in_tgt = block_size - len(ans_ids)
    tgt[ans_start_in_tgt:] = tgt_full[ans_start_in_tgt:]
    return idx, tgt, len(ans_ids)


def broadcast_cell(cell_64: np.ndarray, K: int, n_neighbors: int) -> np.ndarray:
    """Tile a single (Ln,) cell across all K chunks and n_neighbors slots."""
    Ln = cell_64.shape[0]
    nbrs = np.broadcast_to(cell_64, (1, K, n_neighbors, Ln)).copy()
    return nbrs


@torch.no_grad()
def score(model: RetroGPT, idx: np.ndarray, tgt: np.ndarray,
          nbrs: np.ndarray | None, device: str) -> tuple[float, int]:
    """Return (sum_nll_nats, n_answer_tokens) for one probe under one condition."""
    idx_t = torch.from_numpy(idx).unsqueeze(0).to(device)
    tgt_t = torch.from_numpy(tgt).unsqueeze(0).to(device)
    if nbrs is not None:
        nbrs_t = torch.from_numpy(nbrs).to(device)
    else:
        nbrs_t = None
    with ehe.CTX:
        _, loss = model(idx_t, targets=tgt_t, neighbors=nbrs_t)
    n_ans = int((tgt != -1).sum())
    # model returns mean CE over non-ignored positions => * n to get sum NLL
    return float(loss.item()) * n_ans, n_ans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=ehe.SEED)
    args = parser.parse_args(argv)

    if not args.ckpt.exists():
        parser.error(f"checkpoint not found: {args.ckpt}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[exp31] device={ehe.DEVICE} dtype={ehe.DTYPE}")
    print(f"[exp31] loading checkpoint {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location=ehe.DEVICE, weights_only=False)
    config = ckpt["config"]
    print(
        f"[exp31] config: n_layer={config.n_layer} n_head={config.n_head} "
        f"n_embd={config.n_embd} vocab={config.vocab_size} "
        f"block={config.block_size} chunk={config.chunk_size} "
        f"n_neighbors={config.n_neighbors} neighbor_len={config.neighbor_len}"
    )
    model = RetroGPT(config).to(ehe.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[exp31] model params: {n_params/1e6:.2f}M")

    tok = tiktoken.get_encoding("gpt2")

    # Diagnostic: how many tokens does each bundle take?
    text_a = BUNDLE_SEPARATOR.join(FACTS_A)
    text_b = BUNDLE_SEPARATOR.join(FACTS_B)
    n_a = len(tok.encode(text_a))
    n_b = len(tok.encode(text_b))
    print(f"[exp31] bundle A: {n_a} tokens (target <= {config.neighbor_len})")
    print(f"[exp31] bundle B: {n_b} tokens (target <= {config.neighbor_len})")
    if n_a > config.neighbor_len or n_b > config.neighbor_len:
        raise SystemExit(
            f"[exp31] bundle overflow: A={n_a} B={n_b} > {config.neighbor_len}; "
            f"shorten facts so the bound bundle is not truncated"
        )

    bound_cell = pack_bundle(FACTS_A, tok, target_len=config.neighbor_len)
    distractor_cell = pack_bundle(FACTS_B, tok, target_len=config.neighbor_len)

    # Sanity check the neutrals also fit in a 6-statement bundle.
    if len(tok.encode(BUNDLE_SEPARATOR.join(NEUTRALS))) > config.neighbor_len:
        raise SystemExit("[exp31] neutrals bundle overflows neighbor_len")

    K = config.block_size // config.chunk_size

    bound_nbrs = broadcast_cell(bound_cell, K, config.n_neighbors)
    distractor_nbrs = broadcast_cell(distractor_cell, K, config.n_neighbors)

    per_probe: list[dict] = []
    sums = {"none": 0.0, "single": 0.0, "bound": 0.0, "distractor": 0.0}
    tokens_total = 0

    t0 = time.time()
    for prefix, answer, fact_idx in PROBES:
        idx, tgt, n_ans = build_probe_input(prefix, answer, tok, config.block_size)
        # "single" condition: target fact + 5 neutral statements. Same cell
        # structure as bound (6 statements, similar EOT-padding density) so the
        # comparison isolates binding extraction rather than padding density.
        single_facts = [FACTS_A[fact_idx]] + NEUTRALS[:5]
        single_cell = pack_bundle(single_facts, tok, target_len=config.neighbor_len)
        single_nbrs = broadcast_cell(single_cell, K, config.n_neighbors)

        nll_none, _ = score(model, idx, tgt, None, ehe.DEVICE)
        nll_single, _ = score(model, idx, tgt, single_nbrs, ehe.DEVICE)
        nll_bound, _ = score(model, idx, tgt, bound_nbrs, ehe.DEVICE)
        nll_distractor, _ = score(model, idx, tgt, distractor_nbrs, ehe.DEVICE)

        # Per-token nats
        per_token = {
            "none": nll_none / n_ans,
            "single": nll_single / n_ans,
            "bound": nll_bound / n_ans,
            "distractor": nll_distractor / n_ans,
        }

        # Per-probe extraction ratio (guard against division by ~0)
        denom = per_token["distractor"] - per_token["single"]
        numer = per_token["distractor"] - per_token["bound"]
        if abs(denom) < 1e-6:
            ratio = float("nan")
        else:
            ratio = numer / denom

        print(
            f"  probe[{fact_idx}] '{prefix[:40]}...' -> '{answer.strip()}' "
            f"({n_ans} tok)  "
            f"none={per_token['none']:.3f}  single={per_token['single']:.3f}  "
            f"bound={per_token['bound']:.3f}  distractor={per_token['distractor']:.3f}  "
            f"ratio={ratio:.3f}"
        )

        per_probe.append({
            "fact_index": fact_idx,
            "fact": FACTS_A[fact_idx],
            "prefix": prefix,
            "answer": answer,
            "n_answer_tokens": n_ans,
            "nll_per_token": per_token,
            "extraction_ratio": ratio,
        })

        sums["none"] += nll_none
        sums["single"] += nll_single
        sums["bound"] += nll_bound
        sums["distractor"] += nll_distractor
        tokens_total += n_ans

    elapsed = time.time() - t0
    print(f"[exp31] done in {elapsed:.1f}s")

    overall = {k: v / tokens_total for k, v in sums.items()}
    overall_denom = overall["distractor"] - overall["single"]
    overall_numer = overall["distractor"] - overall["bound"]
    if abs(overall_denom) < 1e-6:
        overall_ratio = float("nan")
    else:
        overall_ratio = overall_numer / overall_denom

    single_benefit_vs_none = overall["none"] - overall["single"]
    if single_benefit_vs_none < INCONCLUSIVE_SINGLE_BENEFIT:
        verdict = "INCONCLUSIVE"
        verdict_reason = (
            f"single-fact neighbor improves over no-neighbor by only "
            f"{single_benefit_vs_none:+.3f} nats/token (< {INCONCLUSIVE_SINGLE_BENEFIT}); "
            f"CCA is not extracting from single facts either, so binding extraction "
            f"cannot be measured."
        )
    elif overall_ratio >= PASS_RATIO:
        verdict = "PASS"
        verdict_reason = (
            f"CCA extracts {overall_ratio*100:.1f}% of per-fact signal from a "
            f"bound m=6 cell (>= {PASS_RATIO*100:.0f}%)."
        )
    else:
        verdict = "FAIL"
        verdict_reason = (
            f"CCA extracts only {overall_ratio*100:.1f}% of per-fact signal from a "
            f"bound m=6 cell (< {PASS_RATIO*100:.0f}%)."
        )

    payload = {
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "overall_nll_per_token": overall,
        "overall_extraction_ratio": overall_ratio,
        "single_benefit_over_none": single_benefit_vs_none,
        "bound_benefit_over_distractor": overall["distractor"] - overall["bound"],
        "single_benefit_over_distractor": overall["distractor"] - overall["single"],
        "per_probe": per_probe,
        "_meta": {
            "ckpt": str(args.ckpt),
            "seed": args.seed,
            "elapsed_seconds": elapsed,
            "model_params_millions": n_params / 1e6,
            "ckpt_iter": ckpt.get("iter"),
            "block_size": config.block_size,
            "chunk_size": config.chunk_size,
            "n_neighbors": config.n_neighbors,
            "neighbor_len": config.neighbor_len,
            "n_facts_bound": len(FACTS_A),
            "n_facts_distractor": len(FACTS_B),
            "pass_ratio_threshold": PASS_RATIO,
            "inconclusive_single_benefit_threshold": INCONCLUSIVE_SINGLE_BENEFIT,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[exp31] wrote {args.out_json}")

    md_lines = [
        "# Binding through the CCA path (exp31)",
        "",
        f"- Checkpoint: `{args.ckpt}` ({n_params/1e6:.1f}M params, iter={ckpt.get('iter', '?')})",
        f"- m=6 bound concepts per cell; {len(PROBES)} probes",
        f"- Threshold: extraction ratio >= {PASS_RATIO} -> PASS",
        "",
        "## Overall NLL per answer token (nats)",
        "",
        "| Condition  | NLL/token |",
        "| ---------- | --------: |",
        f"| none       | {overall['none']:.3f} |",
        f"| single     | {overall['single']:.3f} |",
        f"| bound (6)  | {overall['bound']:.3f} |",
        f"| distractor | {overall['distractor']:.3f} |",
        "",
        f"- single benefit over none       : **{single_benefit_vs_none:+.3f}**",
        f"- single benefit over distractor : **{overall['distractor'] - overall['single']:+.3f}**",
        f"- bound benefit over distractor  : **{overall['distractor'] - overall['bound']:+.3f}**",
        f"- **extraction ratio = {overall_ratio:.3f}**",
        "",
        f"## Verdict: **{verdict}**",
        "",
        verdict_reason,
        "",
        "## Per-probe",
        "",
        "| Fact | Answer | n_tok | none | single | bound | distractor | ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in per_probe:
        pt = r["nll_per_token"]
        md_lines.append(
            f"| {r['fact']} | `{r['answer'].strip()}` | {r['n_answer_tokens']} | "
            f"{pt['none']:.3f} | {pt['single']:.3f} | {pt['bound']:.3f} | "
            f"{pt['distractor']:.3f} | {r['extraction_ratio']:.3f} |"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[exp31] wrote {args.out_md}")

    print(f"[exp31] VERDICT: {verdict}  (ratio={overall_ratio:.3f})")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
