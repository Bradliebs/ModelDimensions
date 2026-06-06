"""Inspect the known-class misses from exp17 to decide which dial to turn.

Reads results/v1_pipeline_eval.json and prints per-query detail for every
gate-silence and drift-silence in the known class. Goal: tell whether we
need to (a) lower the gate margin, (b) soften Stage A coverage, or
(c) loosen Stage B numerics, or some mix.

This is read-only diagnostics. No fix yet.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "results" / "v1_pipeline_eval.json"


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    known = [r for r in data["records"] if r["class"] == "known"]
    grounded = [r for r in known if r["outcome"] == "grounded"]
    gate_miss = [r for r in known if r["outcome"] == "silence_gate"]
    drift_miss = [r for r in known if r["outcome"] == "silence_drift"]

    print(f"KNOWN n={len(known)}  grounded={len(grounded)}  "
          f"gate-miss={len(gate_miss)}  drift-miss={len(drift_miss)}")
    print()

    print("=" * 78)
    print(f"GROUNDED ({len(grounded)}) -- gate margins (so we know floor):")
    print("=" * 78)
    margins = sorted(r["gate_margin"] for r in grounded)
    print(f"  margins min={margins[0]:+.3f}  median={margins[len(margins)//2]:+.3f}  max={margins[-1]:+.3f}")
    print(f"  full: {[round(m, 3) for m in margins]}")
    print()

    print("=" * 78)
    print(f"GATE-SILENCED ({len(gate_miss)}) -- top1-top2 < 0.05:")
    print("=" * 78)
    for r in sorted(gate_miss, key=lambda x: -x["gate_margin"]):
        print(f"  margin={r['gate_margin']:+.3f}  wall={r['wall_time_sec']:.2f}s")
        print(f"    q: {r['query_head']!r}")
    print()

    print("=" * 78)
    print(f"DRIFT-SILENCED ({len(drift_miss)}) -- verifier rejected:")
    print("=" * 78)
    for r in sorted(drift_miss, key=lambda x: -(x.get("verify_coverage") or 0)):
        cov = r.get("verify_coverage")
        cov_s = f"{cov:.2f}" if cov is not None else "None"
        print(f"  margin={r['gate_margin']:+.3f}  coverage={cov_s}  wall={r['wall_time_sec']:.2f}s")
        print(f"    q: {r['query_head']!r}")
    print()

    # Coverage histogram for drift cases.
    covs = [r.get("verify_coverage") or 0 for r in drift_miss]
    print(f"Drift coverage distribution: {sorted(round(c, 2) for c in covs)}")
    if covs:
        below_30 = sum(1 for c in covs if c < 0.30)
        in_30_50 = sum(1 for c in covs if 0.30 <= c < 0.50)
        print(f"  coverage < 0.30: {below_30}  (clearly off-topic; relaxing gate won't help)")
        print(f"  0.30 <= coverage < 0.50: {in_30_50}  (close-but-strict; lowering Stage A would recover)")


if __name__ == "__main__":
    main()
