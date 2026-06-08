"""exp32 -- CCA position-bias diagnostic.

Followup to exp31. exp31 found that on the 55M RETRO checkpoint the
overall CCA binding-extraction ratio for an m=6 bound cell is 0.485 (FAIL),
with 3/6 probes extracting cleanly (positions 0, 4, 5 in the bundle) and
2/6 actively hurt (positions 2, 3). That hints at attention position bias
but at N=6 it is a one-trial observation, not a result.

This experiment rotates each of the 6 target facts through all 6 possible
positions in the bound bundle (36 probes total). For each (fact, position)
we measure NLL on the answer tokens under three neighbor conditions:

    none        : neighbors=None (CCA off; reference)
    single      : target fact at `position` + 5 NEUTRAL statements at the
                  other 5 positions (matches bound's position layout but
                  the other slots are inert -- this is the per-position
                  upper bound on extraction)
    bound       : target fact at `position` + the other 5 FACTS_A facts at
                  the other 5 positions (the interference condition)
    distractor  : 6 unrelated FACTS_B facts in fixed order (the floor)

Per-probe metric (same as exp31):
    extraction_ratio = (nll_distractor - nll_bound) / (nll_distractor - nll_single)

Position effect is reported as the mean extraction_ratio at each position p
in 0..5, averaged over the 6 target facts.

Verdict:
    POSITION_BIAS_CONFIRMED  -- both end positions (0, 5) mean ratio >= 0.5
                                AND both middle positions (2, 3) mean ratio < 0.3
    PARTIAL_POSITION_BIAS    -- mean(end positions) - mean(middle positions) > 0.2
                                but the strict threshold above is not met
    NO_POSITION_BIAS         -- otherwise (ratios are flat across positions)
    INCONCLUSIVE             -- overall single-vs-none benefit < 0.05 nats

Caveats (carried over from exp31):
- 55M debug ckpt, not the 404M roadmap target.
- Synthetic fictional facts; results may not transfer to natural retrieval.
- N=6 facts and N=6 positions is still small; per-position means are
  averages of 6 numbers each.

Usage:
    python experiments/exp32_cca_position_bias.py \
        --out-json reports/cca_position_bias.json \
        --out-md reports/cca_position_bias.md
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
import tiktoken  # noqa: E402

from src.retro import eval_retro_heldout as ehe  # noqa: E402
from src.retro.model_retro import RetroGPT  # noqa: E402


DEFAULT_CKPT = Path(r"H:\MiniLM\nanogpt\out-retro-bank\ckpt_best.pt")
PASS_RATIO = 0.5
INCONCLUSIVE_SINGLE_BENEFIT = 0.05  # nats per token

# Bias thresholds for verdict (calibrated to be conservative)
BIAS_END_MIN = 0.5      # end positions must extract cleanly
BIAS_MIDDLE_MAX = 0.3   # middle positions must be substantially worse
BIAS_PARTIAL_GAP = 0.2  # end-vs-middle gap for "partial" verdict

# --- Stimuli (identical to exp31) -----------------------------------------

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

FACTS_B = [
    "Dawn bell: Hesht-Var",
    "Wyvern caverns: Skarn",
    "Sea-mage lighthouse: Velnor",
    "Captain's vessel: Marigold",
    "Red wine vineyard: Asher",
    "Ancient ruler: Pelthorn",
]

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

# --- Helpers (copied unchanged from exp31) --------------------------------


def pack_bundle(facts: list[str], tok: tiktoken.Encoding, target_len: int = 64,
                pad_token: int | None = None) -> np.ndarray:
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
    pre_ids = tok.encode(prefix)
    ans_ids = tok.encode(answer)
    prompt_ids = pre_ids + ans_ids

    filler_ids = tok.encode(FILLER)
    while len(filler_ids) + len(prompt_ids) < block_size + 1:
        filler_ids = filler_ids + filler_ids

    full_ids = (filler_ids + prompt_ids)[-(block_size + 1):]
    idx = np.asarray(full_ids[:block_size], dtype=np.int64)
    tgt_full = np.asarray(full_ids[1:block_size + 1], dtype=np.int64)

    tgt = np.full(block_size, -1, dtype=np.int64)
    ans_start_in_tgt = block_size - len(ans_ids)
    tgt[ans_start_in_tgt:] = tgt_full[ans_start_in_tgt:]
    return idx, tgt, len(ans_ids)


def broadcast_cell(cell_64: np.ndarray, K: int, n_neighbors: int) -> np.ndarray:
    Ln = cell_64.shape[0]
    nbrs = np.broadcast_to(cell_64, (1, K, n_neighbors, Ln)).copy()
    return nbrs


@torch.no_grad()
def score(model: RetroGPT, idx: np.ndarray, tgt: np.ndarray,
          nbrs: np.ndarray | None, device: str) -> tuple[float, int]:
    idx_t = torch.from_numpy(idx).unsqueeze(0).to(device)
    tgt_t = torch.from_numpy(tgt).unsqueeze(0).to(device)
    if nbrs is not None:
        nbrs_t = torch.from_numpy(nbrs).to(device)
    else:
        nbrs_t = None
    with ehe.CTX:
        _, loss = model(idx_t, targets=tgt_t, neighbors=nbrs_t)
    n_ans = int((tgt != -1).sum())
    return float(loss.item()) * n_ans, n_ans


# --- New: position-rotated bundle construction ---------------------------


def place_at(other_items: list[str], target_item: str, position: int,
             total: int = 6) -> list[str]:
    """Build a list of `total` items with `target_item` at `position` and
    `other_items` filling the remaining slots in order.

    Requires len(other_items) >= total - 1.
    """
    if not (0 <= position < total):
        raise ValueError(f"position={position} not in [0, {total})")
    if len(other_items) < total - 1:
        raise ValueError(
            f"need {total - 1} other_items, got {len(other_items)}"
        )
    out = list(other_items[:total - 1])
    out.insert(position, target_item)
    assert len(out) == total
    return out


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

    print(f"[exp32] device={ehe.DEVICE} dtype={ehe.DTYPE}")
    print(f"[exp32] loading checkpoint {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location=ehe.DEVICE, weights_only=False)
    config = ckpt["config"]
    print(
        f"[exp32] config: n_layer={config.n_layer} n_head={config.n_head} "
        f"n_embd={config.n_embd} vocab={config.vocab_size} "
        f"block={config.block_size} chunk={config.chunk_size} "
        f"n_neighbors={config.n_neighbors} neighbor_len={config.neighbor_len}"
    )
    model = RetroGPT(config).to(ehe.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[exp32] model params: {n_params/1e6:.2f}M")

    tok = tiktoken.get_encoding("gpt2")
    K = config.block_size // config.chunk_size
    Ln = config.neighbor_len

    # Pre-check: every rotated bound bundle must fit (FACTS_A in any order).
    # Since FACTS_A has the same content under any permutation modulo the
    # one-newline separator, encoded length is permutation-invariant if no
    # BPE merges cross the newline boundary -- check the worst case anyway
    # by encoding bundle A in its original order.
    n_a_check = len(tok.encode(BUNDLE_SEPARATOR.join(FACTS_A)))
    n_b_check = len(tok.encode(BUNDLE_SEPARATOR.join(FACTS_B)))
    print(f"[exp32] bundle A original-order tokens: {n_a_check} (limit {Ln})")
    print(f"[exp32] bundle B original-order tokens: {n_b_check} (limit {Ln})")
    if n_a_check > Ln or n_b_check > Ln:
        raise SystemExit("[exp32] bundle overflow at base order; aborting")

    # Distractor (fixed) and per-probe single bundle (rotated) -----------
    distractor_cell = pack_bundle(FACTS_B, tok, target_len=Ln)
    distractor_nbrs = broadcast_cell(distractor_cell, K, config.n_neighbors)

    per_probe: list[dict] = []
    sums = {"none": 0.0, "single": 0.0, "bound": 0.0, "distractor": 0.0}
    tokens_total = 0

    t0 = time.time()
    for prefix, answer, fact_idx in PROBES:
        target_fact = FACTS_A[fact_idx]
        other_facts = [f for i, f in enumerate(FACTS_A) if i != fact_idx]
        # other_facts has 5 entries; NEUTRALS already has 6 entries

        idx, tgt, n_ans = build_probe_input(prefix, answer, tok, config.block_size)
        nll_none, _ = score(model, idx, tgt, None, ehe.DEVICE)

        for position in range(6):
            bound_facts = place_at(other_facts, target_fact, position)
            single_facts = place_at(NEUTRALS[:5], target_fact, position)

            # Rotated bundle must still fit: assert on every probe, not just
            # the base order, because BPE merges can shift token counts when
            # the surrounding strings change. Hard-fail if a probe would be
            # invalidated by truncation.
            bound_cell = pack_bundle(bound_facts, tok, target_len=Ln)
            single_cell = pack_bundle(single_facts, tok, target_len=Ln)

            bound_nbrs = broadcast_cell(bound_cell, K, config.n_neighbors)
            single_nbrs = broadcast_cell(single_cell, K, config.n_neighbors)

            nll_single, _ = score(model, idx, tgt, single_nbrs, ehe.DEVICE)
            nll_bound, _ = score(model, idx, tgt, bound_nbrs, ehe.DEVICE)
            nll_distractor, _ = score(model, idx, tgt, distractor_nbrs, ehe.DEVICE)

            per_token = {
                "none": nll_none / n_ans,
                "single": nll_single / n_ans,
                "bound": nll_bound / n_ans,
                "distractor": nll_distractor / n_ans,
            }

            denom = per_token["distractor"] - per_token["single"]
            numer = per_token["distractor"] - per_token["bound"]
            if abs(denom) < 1e-6:
                ratio = float("nan")
            else:
                ratio = numer / denom

            print(
                f"  fact[{fact_idx}] '{answer.strip():<10}' pos={position}  "
                f"single={per_token['single']:.3f}  bound={per_token['bound']:.3f}  "
                f"distractor={per_token['distractor']:.3f}  ratio={ratio:+.3f}"
            )

            per_probe.append({
                "fact_index": fact_idx,
                "fact": target_fact,
                "position": position,
                "answer": answer,
                "n_answer_tokens": n_ans,
                "nll_per_token": per_token,
                "extraction_ratio": ratio,
            })

            # Sum over all (fact, position) for the overall mean.
            # nll_none is counted only ONCE per fact (it does not depend on
            # position) to avoid inflating its weight.
            if position == 0:
                sums["none"] += nll_none
            sums["single"] += nll_single
            sums["bound"] += nll_bound
            sums["distractor"] += nll_distractor
            if position == 0:
                tokens_total += n_ans

    elapsed = time.time() - t0
    print(f"[exp32] done in {elapsed:.1f}s ({len(per_probe)} probes)")

    # --- Aggregate -------------------------------------------------------
    # tokens_total counts each fact once (for the "none" reference).
    # Each per-condition sum (single, bound, distractor) was added 6 times
    # per fact (once per position), so divide by 6 * tokens_total for the
    # per-token mean over all (fact, position) combinations.
    n_positions = 6
    overall = {
        "none": sums["none"] / tokens_total,
        "single": sums["single"] / (n_positions * tokens_total),
        "bound": sums["bound"] / (n_positions * tokens_total),
        "distractor": sums["distractor"] / (n_positions * tokens_total),
    }
    overall_denom = overall["distractor"] - overall["single"]
    overall_numer = overall["distractor"] - overall["bound"]
    if abs(overall_denom) < 1e-6:
        overall_ratio = float("nan")
    else:
        overall_ratio = overall_numer / overall_denom

    # Mean ratio per position (averaged across the 6 facts at that position).
    per_position_ratios: dict[int, list[float]] = {p: [] for p in range(6)}
    for rec in per_probe:
        r = rec["extraction_ratio"]
        if not (isinstance(r, float) and np.isnan(r)):
            per_position_ratios[rec["position"]].append(r)
    per_position_mean = {
        p: (float(np.mean(rs)) if rs else float("nan"))
        for p, rs in per_position_ratios.items()
    }
    per_position_std = {
        p: (float(np.std(rs, ddof=1)) if len(rs) >= 2 else float("nan"))
        for p, rs in per_position_ratios.items()
    }

    # --- Verdict ---------------------------------------------------------
    single_benefit = overall["none"] - overall["single"]
    if single_benefit < INCONCLUSIVE_SINGLE_BENEFIT:
        verdict = "INCONCLUSIVE"
        verdict_reason = (
            f"single-fact neighbor improves over no-neighbor by only "
            f"{single_benefit:+.3f} nats/token (< {INCONCLUSIVE_SINGLE_BENEFIT}); "
            f"CCA is not extracting single facts so position bias cannot be measured."
        )
    else:
        # Numeric trace for verdict (per user-memory rule on directional
        # diagnostics): position-bias means END positions extract well and
        # MIDDLE positions extract poorly. Numerically:
        #   confirmed: per_position_mean[0] >= 0.5 AND per_position_mean[5] >= 0.5
        #              AND per_position_mean[2] < 0.3 AND per_position_mean[3] < 0.3
        #   partial:   mean(ends) - mean(middles) > 0.2 (but strict not met)
        ends = [per_position_mean[0], per_position_mean[5]]
        middles = [per_position_mean[2], per_position_mean[3]]
        end_mean = float(np.mean(ends))
        middle_mean = float(np.mean(middles))
        gap = end_mean - middle_mean

        strict_confirmed = (
            ends[0] >= BIAS_END_MIN and ends[1] >= BIAS_END_MIN
            and middles[0] < BIAS_MIDDLE_MAX and middles[1] < BIAS_MIDDLE_MAX
        )
        if strict_confirmed:
            verdict = "POSITION_BIAS_CONFIRMED"
            verdict_reason = (
                f"end positions extract cleanly (pos0={ends[0]:.3f}, "
                f"pos5={ends[1]:.3f}; both >= {BIAS_END_MIN}) while middle "
                f"positions fail (pos2={middles[0]:.3f}, pos3={middles[1]:.3f}; "
                f"both < {BIAS_MIDDLE_MAX}). end-vs-middle gap = {gap:+.3f}."
            )
        elif gap > BIAS_PARTIAL_GAP:
            verdict = "PARTIAL_POSITION_BIAS"
            verdict_reason = (
                f"end positions extract better than middle (gap = {gap:+.3f} > "
                f"{BIAS_PARTIAL_GAP}) but the strict thresholds "
                f"(ends >= {BIAS_END_MIN}, middles < {BIAS_MIDDLE_MAX}) "
                f"are not met. ends mean={end_mean:.3f}, middles mean={middle_mean:.3f}."
            )
        else:
            verdict = "NO_POSITION_BIAS"
            verdict_reason = (
                f"no clear position effect: end-vs-middle gap = {gap:+.3f} "
                f"(< {BIAS_PARTIAL_GAP}). ends mean={end_mean:.3f}, "
                f"middles mean={middle_mean:.3f}. exp31's per-probe variation "
                f"is likely fact-specific rather than position-driven."
            )

    payload = {
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "overall_nll_per_token": overall,
        "overall_extraction_ratio": overall_ratio,
        "per_position_mean_ratio": per_position_mean,
        "per_position_std_ratio": per_position_std,
        "per_position_n": {p: len(rs) for p, rs in per_position_ratios.items()},
        "per_probe": per_probe,
        "meta": {
            "experiment": "exp32_cca_position_bias",
            "ckpt_path": str(args.ckpt),
            "ckpt_params_million": n_params / 1e6,
            "n_facts": len(FACTS_A),
            "n_positions": n_positions,
            "n_probes": len(per_probe),
            "pass_ratio": PASS_RATIO,
            "bias_thresholds": {
                "end_min": BIAS_END_MIN,
                "middle_max": BIAS_MIDDLE_MAX,
                "partial_gap": BIAS_PARTIAL_GAP,
            },
            "block_size": config.block_size,
            "chunk_size": config.chunk_size,
            "neighbor_len": config.neighbor_len,
            "n_neighbors": config.n_neighbors,
            "seed": args.seed,
            "elapsed_sec": elapsed,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp32] wrote {args.out_json}")

    # --- Markdown report -------------------------------------------------
    md_lines = [
        "# exp32 -- CCA position-bias diagnostic",
        "",
        f"- Checkpoint: `{args.ckpt}` ({n_params/1e6:.2f}M params)",
        f"- {len(FACTS_A)} facts x {n_positions} positions = {len(per_probe)} probes",
        f"- Block {config.block_size}, chunk {config.chunk_size}, "
        f"neighbor_len {config.neighbor_len}, n_neighbors {config.n_neighbors}",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        verdict_reason,
        "",
        "## Overall (averaged over all 36 probes)",
        "",
        f"- nll/token none        = {overall['none']:.3f}",
        f"- nll/token single      = {overall['single']:.3f}",
        f"- nll/token bound       = {overall['bound']:.3f}",
        f"- nll/token distractor  = {overall['distractor']:.3f}",
        f"- overall extraction_ratio = (dist - bound) / (dist - single) "
        f"= {overall_ratio:+.3f}",
        f"- single_benefit_vs_none = {overall['none'] - overall['single']:+.3f}",
        "",
        "## Mean extraction ratio by position",
        "",
        "(mean over the 6 different target facts placed at each position; "
        "1.0 = full single-fact extraction, 0.0 = no better than distractor, "
        "< 0 = bound actively hurts vs distractor)",
        "",
        "| position | mean ratio | std | n |",
        "|---:|---:|---:|---:|",
    ]
    for p in range(6):
        m = per_position_mean[p]
        s = per_position_std[p]
        n = len(per_position_ratios[p])
        md_lines.append(f"| {p} | {m:+.3f} | {s:.3f} | {n} |")
    md_lines += [
        "",
        "## Per-probe detail",
        "",
        "| fact | answer | position | nll(single) | nll(bound) | nll(distractor) | ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for rec in per_probe:
        nll = rec["nll_per_token"]
        md_lines.append(
            f"| {rec['fact_index']} | {rec['answer'].strip()} | {rec['position']} | "
            f"{nll['single']:.3f} | {nll['bound']:.3f} | {nll['distractor']:.3f} | "
            f"{rec['extraction_ratio']:+.3f} |"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[exp32] wrote {args.out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
