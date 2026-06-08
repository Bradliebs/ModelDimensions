"""scripts/eval_pad_attractor.py -- Compare baseline vs mitigated smoke runs.

Reads two JSON files produced by scripts/ask_retro.py (one with
``--skip-low-info-fraction 1.01`` = mitigation off, one with the default 0.5
= mitigation on) and computes the pad-attractor metric across every
chunk-boundary crossing in every prompt, then writes a markdown delta
report.

The metric the Step 4 summary called out is "fraction of low-info pad
chunks that retrieve the same top-1 attractor cell". The baseline number
was 100% (15 / 15). The mitigation target was <=25%.

Honest measurement: the mitigation does not lower the attractor's
top-1 dominance on the chunks it skips -- it just stops feeding those
retrievals into CCA. So the natural way to report it is two numbers
side by side:

  baseline: <low_info_chunks_retrieved> of <low_info_chunks_total> chunks
            retrieved => <fraction_same_top1>% hit the same top-1 cell.
  mitigated: <low_info_chunks_retrieved> of <low_info_chunks_total> chunks
             retrieved => the CCA-pollution surface area drops to
             whatever survives the skip threshold.

Usage:
    python scripts/eval_pad_attractor.py \\
        --baseline reports/live_generation_smoke_baseline.json \\
        --mitigated reports/live_generation_smoke_mitigated.json \\
        --out-md reports/step4_mitigation_eval.md \\
        --out-json reports/step4_mitigation_eval.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def collect_chunk_stats(payload: dict, reference_cutoff: float) -> dict:
    """Walk every retrieval_log entry across every prompt and tally chunks.

    ``reference_cutoff`` is a single fixed value used to classify a chunk
    as low-info across both runs. It is independent of the run's actual
    ``skip_low_info_fraction`` -- the run's threshold only determines
    whether the chunk was *skipped*; the property "is this chunk
    low-info?" is a measurement that should not change between runs.

    Returns a flat summary:
      total_chunks_observed       — across every boundary crossing
      low_info_chunks             — low_info_fraction >= reference_cutoff
      low_info_chunks_retrieved   — low-info AND retrieved (skipped == False)
      low_info_top1_cells         — Counter over top1_source_preview for
                                    low-info chunks that were retrieved
      hi_info_chunks              — total chunks below the reference cutoff
      hi_info_chunks_retrieved    — hi-info AND retrieved
    """
    threshold = float(payload["meta"]["skip_low_info_fraction"])
    total_chunks = 0
    low_info_total = 0
    low_info_retrieved = 0
    low_info_top1: Counter[str] = Counter()
    hi_info_total = 0
    hi_info_retrieved = 0
    hi_info_top1: Counter[str] = Counter()
    for rec in payload["results"]:
        log = rec.get("with_retrieval", {}).get("retrieval_log") or []
        for boundary in log:
            for pc in boundary.get("per_chunk", []):
                total_chunks += 1
                low_frac = float(pc["low_info_fraction"])
                skipped = bool(pc["skipped"])
                top1 = pc.get("top1_source_preview")
                is_low = low_frac >= reference_cutoff
                if is_low:
                    low_info_total += 1
                    if not skipped and top1 is not None:
                        low_info_retrieved += 1
                        low_info_top1[top1] += 1
                else:
                    hi_info_total += 1
                    if not skipped and top1 is not None:
                        hi_info_retrieved += 1
                        hi_info_top1[top1] += 1
    return {
        "run_threshold": threshold,
        "reference_cutoff": reference_cutoff,
        "total_chunks": total_chunks,
        "low_info_total": low_info_total,
        "low_info_retrieved": low_info_retrieved,
        "low_info_top1": low_info_top1,
        "hi_info_total": hi_info_total,
        "hi_info_retrieved": hi_info_retrieved,
        "hi_info_top1": hi_info_top1,
    }


def _safe_fraction(num: int, den: int) -> float:
    return (num / den) if den > 0 else 0.0


def _format_top_cells(counter: Counter[str], n: int = 3) -> list[str]:
    if not counter:
        return ["(none)"]
    return [f"  - {count}x  {text[:80]!r}" for text, count in counter.most_common(n)]


def render_markdown(baseline: dict, mitigated: dict) -> str:
    base_lo_ret = baseline["low_info_retrieved"]
    base_lo_tot = baseline["low_info_total"]
    base_hi_ret = baseline["hi_info_retrieved"]
    base_hi_tot = baseline["hi_info_total"]
    mit_lo_ret = mitigated["low_info_retrieved"]
    mit_lo_tot = mitigated["low_info_total"]
    mit_hi_ret = mitigated["hi_info_retrieved"]
    mit_hi_tot = mitigated["hi_info_total"]

    # Top-1 dominance: of low-info chunks that were retrieved, what fraction
    # hit the single most common top-1 cell?
    base_top1_dominance = _safe_fraction(
        baseline["low_info_top1"].most_common(1)[0][1] if baseline["low_info_top1"] else 0,
        base_lo_ret,
    )
    mit_top1_dominance = _safe_fraction(
        mitigated["low_info_top1"].most_common(1)[0][1] if mitigated["low_info_top1"] else 0,
        mit_lo_ret,
    )

    lines = [
        "# Step 4 mitigation -- pad-chunk attractor",
        "",
        "Compares two runs of `scripts/ask_retro.py` on the same 5 prompts,",
        "same seed (1337), same 50K subset bank, same 55M ckpt. The only",
        "difference is `--skip-low-info-fraction`: 1.01 (off) vs 0.5 (on).",
        "Each prompt generates up to 80 tokens, so each prompt produces",
        "multiple chunk-boundary crossings (one boundary every chunk_size=64",
        "generated tokens). At each boundary, all K=4 chunks of the current",
        "rolling window are inspected.",
        "",
        f"Chunks are classified as low-info using a fixed reference cutoff of",
        f"{baseline['reference_cutoff']:.2f} (independent of either run's own",
        f"`skip_low_info_fraction`), so the same chunk is classified the same",
        f"way in both runs and only the *skipped* column differs.",
        "",
        "## Headline numbers",
        "",
        f"| Metric                                              | Baseline (off) | Mitigated (on) |",
        f"|-----------------------------------------------------|---------------:|---------------:|",
        f"| Total chunks observed across all boundary crossings | {baseline['total_chunks']:>14d} | {mitigated['total_chunks']:>14d} |",
        f"| Low-info chunks (fraction >= threshold)             | {base_lo_tot:>14d} | {mit_lo_tot:>14d} |",
        f"| Low-info chunks that hit the bank                   | {base_lo_ret:>14d} | {mit_lo_ret:>14d} |",
        f"| Low-info CCA-pollution rate                         | {_safe_fraction(base_lo_ret, base_lo_tot):>14.1%} | {_safe_fraction(mit_lo_ret, mit_lo_tot):>14.1%} |",
        f"| Top-1 dominance on retrieved low-info chunks        | {base_top1_dominance:>14.1%} | {mit_top1_dominance:>14.1%} |",
        f"| Hi-info chunks (real content)                       | {base_hi_tot:>14d} | {mit_hi_tot:>14d} |",
        f"| Hi-info chunks that hit the bank                    | {base_hi_ret:>14d} | {mit_hi_ret:>14d} |",
        "",
        "## What the rows mean",
        "",
        "- **Low-info CCA-pollution rate** is the metric the Step 4 summary",
        "  flagged: out of chunks dominated by pad tokens (`{0, EOT_TOKEN}`),",
        "  what fraction reach CCA carrying attractor cells from the bank.",
        "  Baseline 100% means every pad chunk pollutes CCA. The target was",
        "  <=25%. Mitigated should be ~0% (skip means the neighbor slot stays",
        "  at the all-EOT default and CCA sees no signal for that chunk).",
        "",
        "- **Top-1 dominance** is the secondary signal. On the baseline run",
        "  the same Hangul attractor cell was the top-1 retrieved cell for",
        "  every pad chunk -- so dominance approached 100%. After mitigation",
        "  almost no pad chunks are retrieved, so the metric is computed on",
        "  whatever survives the skip threshold (usually 0 entries).",
        "",
        "- **Hi-info chunks** are the chunks with real prompt content. The",
        "  mitigation must not change how they are retrieved. If",
        "  `Hi-info chunks that hit the bank` matches between baseline and",
        "  mitigated runs, the mitigation is surgical. If both rows show 0,",
        "  the smoke prompts EOT-ed before any chunk filled with > 50% real",
        "  content (i.e. the rolling window was always pad-dominated), so",
        "  this sanity check is uninformative for this specific smoke -- it",
        "  requires longer continuations to exercise.",
        "",
        "## Numeric trace (per the diagnostics rule)",
        "",
        "Baseline:",
        f"  low_info_retrieved = {base_lo_ret}, low_info_total = {base_lo_tot}",
        f"  pollution_rate = {base_lo_ret} / {base_lo_tot} = {_safe_fraction(base_lo_ret, base_lo_tot):.4f}",
        "",
        "Mitigated:",
        f"  low_info_retrieved = {mit_lo_ret}, low_info_total = {mit_lo_tot}",
        f"  pollution_rate = {mit_lo_ret} / {mit_lo_tot} = {_safe_fraction(mit_lo_ret, mit_lo_tot):.4f}",
        "",
        f"  target was <= 0.25.",
        f"  delta = {_safe_fraction(base_lo_ret, base_lo_tot):.4f} -> {_safe_fraction(mit_lo_ret, mit_lo_tot):.4f}",
        "",
        "## Most common top-1 cells on low-info chunks",
        "",
        "Baseline:",
    ]
    lines.extend(_format_top_cells(baseline["low_info_top1"], n=3))
    lines.append("")
    lines.append("Mitigated:")
    lines.extend(_format_top_cells(mitigated["low_info_top1"], n=3))
    lines.append("")
    lines.append("## Most common top-1 cells on hi-info chunks (sanity)")
    lines.append("")
    lines.append("Baseline:")
    lines.extend(_format_top_cells(baseline["hi_info_top1"], n=3))
    lines.append("")
    lines.append("Mitigated:")
    lines.extend(_format_top_cells(mitigated["hi_info_top1"], n=3))
    lines.append("")
    lines.append("## Honest characterization")
    lines.append("")
    lines.append("The mitigation is an inference-time filter only. It does not")
    lines.append("change the bank, the encoder, the ckpt, or training. It changes")
    lines.append("what the CCA layers see during generation: low-info pad chunks")
    lines.append("now contribute zero retrieved tokens instead of contributing")
    lines.append("the encoder's attractor cells. The 55M ckpt is still incoherent")
    lines.append("on these prompts -- that is a model-scale problem, not a")
    lines.append("retrieval problem, and is documented in reports/step4_summary.md.")
    lines.append("This eval only validates that the pad-attractor failure mode is")
    lines.append("no longer present in the retrieval path.")
    lines.append("")
    lines.append("Caveat: the mitigation introduces a train/inference mismatch.")
    lines.append("The 55M ckpt was trained with CCA always receiving retrieved")
    lines.append("tokens; zeroing them at inference shifts the activations out of")
    lines.append("the training distribution and can degrade surface output on")
    lines.append("individual prompts (some baseline continuations become more")
    lines.append("incoherent under mitigation). That is expected, is not measured")
    lines.append("by this eval, and is the reason the 404M training plan (\u00a75.2)")
    lines.append("names this mitigation as a precondition for promoting any future")
    lines.append("ckpt -- the larger model should be trained with the same")
    lines.append("retrieval discipline that inference will use.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="JSON from ask_retro.py with --skip-low-info-fraction 1.01")
    parser.add_argument("--mitigated", type=Path, required=True,
                        help="JSON from ask_retro.py with the default 0.5")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--reference-cutoff", type=float, default=0.5,
        help=(
            "Single fixed cutoff used to classify a chunk as low-info "
            "across both runs. Defaults to 0.5 (matches the mitigated "
            "run's threshold)."
        ),
    )
    args = parser.parse_args(argv)

    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    mitigated_payload = json.loads(args.mitigated.read_text(encoding="utf-8"))

    baseline_stats = collect_chunk_stats(baseline_payload, args.reference_cutoff)
    mitigated_stats = collect_chunk_stats(mitigated_payload, args.reference_cutoff)

    # Sanity check the threshold flag actually flipped.
    base_thr = baseline_payload["meta"]["skip_low_info_fraction"]
    mit_thr = mitigated_payload["meta"]["skip_low_info_fraction"]
    if base_thr <= 1.0:
        print(f"WARNING: baseline threshold={base_thr} should be > 1.0 to disable mitigation")
    if mit_thr > 1.0:
        print(f"WARNING: mitigated threshold={mit_thr} should be <= 1.0 to enable mitigation")

    md = render_markdown(baseline_stats, mitigated_stats)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"[eval_pad_attractor] wrote {args.out_md}")

    # JSON: keep Counter values as plain dicts for json
    def _serializable(stats: dict) -> dict:
        out = dict(stats)
        out["low_info_top1"] = dict(stats["low_info_top1"])
        out["hi_info_top1"] = dict(stats["hi_info_top1"])
        return out

    payload = {
        "baseline": _serializable(baseline_stats),
        "mitigated": _serializable(mitigated_stats),
        "inputs": {
            "baseline_json": str(args.baseline),
            "mitigated_json": str(args.mitigated),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[eval_pad_attractor] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
