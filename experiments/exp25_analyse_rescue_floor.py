"""exp25 analyser: offline rescue floor sweep (Step B).

Reads results/v1_rescue_instrumentation.json and projects what the four
class-level outcomes would be under candidate (rank_window, activation_floor)
rescue rules. No pipeline change.

Rescue rule (primary-entity gating, matches the conservative recommendation):

    margin < T_NORMAL                           and   # below normal gate
    verifier_grounded == True                   and   # Stage E v2 passes
    primary_entity is not None                  and
    exists rank r in [0, rank_window):
        primary_entity in ranks[r].novel_entities_present  and
        ranks[r].contains_any_anchor             and
        ranks[r].activation >= activation_floor

When all true, the would-be silenced answer is rescued (treated as the
verifier's answer — grounded if outcome_at_zero=='grounded', else still
silence_drift).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "results" / "v1_rescue_instrumentation.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "v1_rescue_floor_sweep.json"

T_NORMAL = 0.015
RANK_WINDOWS = [1, 2, 3]
FLOORS = [0.30, 0.35, 0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]


def _is_rescue_eligible(rec: dict, rank_window: int, floor: float) -> tuple[bool, dict | None]:
    if rec["margin"] >= T_NORMAL:
        return False, None
    if not rec.get("verifier_grounded"):
        return False, None
    primary = rec.get("primary_entity")
    if not primary:
        return False, None
    for rk in rec.get("ranks", []):
        if rk["rank"] >= rank_window:
            break
        if (primary in rk["novel_entities_present"]
                and rk["contains_any_anchor"]
                and rk["activation"] >= floor):
            return True, {
                "rank": rk["rank"],
                "cell_id": rk["cell_id"],
                "activation": rk["activation"],
                "anchors_present": rk["anchors_present"],
            }
    return False, None


def _effective_decision(rec: dict, rank_window: int, floor: float) -> str:
    """Return one of: grounded, rescued, silence_gate, silence_drift."""
    if rec["margin"] >= T_NORMAL:
        # Above gate -> normal decision.
        return rec["outcome_at_zero"]  # 'grounded' or 'silence_drift'
    # Below gate.
    eligible, _ = _is_rescue_eligible(rec, rank_window, floor)
    if eligible:
        # Rescue applies only when the verifier already accepts at zero.
        if rec["outcome_at_zero"] == "grounded":
            return "rescued"
        return "silence_drift"
    return "silence_gate"


def _is_silenced(decision: str) -> bool:
    return decision in ("silence_gate", "silence_drift")


def _is_answered(decision: str) -> bool:
    return decision in ("grounded", "rescued")


def _summarise(records: list[dict], rank_window: int, floor: float) -> dict:
    by_survey: dict[str, dict] = {
        "known": {"n": 0, "answered": 0, "rescued": 0},
        "unknown": {"n": 0, "silenced": 0, "wrongly_answered": 0, "rescued": 0},
        "noise": {"n": 0, "silenced": 0, "wrongly_answered": 0, "rescued": 0},
        "multihop": {"n": 0, "ok": 0, "wrong": 0, "silenced": 0, "rescued": 0,
                     "ok_rescued": 0, "wrong_rescued": 0},
    }
    rescued_records: list[dict] = []
    for rec in records:
        s = rec["survey"]
        d = _effective_decision(rec, rank_window, floor)
        bucket = by_survey[s]
        bucket["n"] += 1
        if d == "rescued":
            bucket["rescued"] = bucket.get("rescued", 0) + 1
            _, info = _is_rescue_eligible(rec, rank_window, floor)
            rescued_records.append({
                "survey": s,
                "query": rec["query"],
                "sub_question": rec.get("sub_question"),
                "margin": rec["margin"],
                "primary_entity": rec.get("primary_entity"),
                "kw_hit": rec.get("kw_hit"),
                "supporting_cell": info,
            })
        if s == "known":
            if _is_answered(d):
                bucket["answered"] += 1
        elif s in ("unknown", "noise"):
            if _is_silenced(d):
                bucket["silenced"] += 1
            elif _is_answered(d):
                bucket["wrongly_answered"] += 1
        elif s == "multihop":
            kw = bool(rec.get("kw_hit"))
            if _is_answered(d):
                if kw:
                    bucket["ok"] += 1
                    if d == "rescued":
                        bucket["ok_rescued"] += 1
                else:
                    bucket["wrong"] += 1
                    if d == "rescued":
                        bucket["wrong_rescued"] += 1
            else:
                bucket["silenced"] += 1
    return {
        "rank_window": rank_window,
        "activation_floor": floor,
        "by_survey": by_survey,
        "rescued_records": rescued_records,
    }


def _format_table_row(s: dict) -> str:
    rw, fl = s["rank_window"], s["activation_floor"]
    bs = s["by_survey"]
    k = bs["known"]; u = bs["unknown"]; n = bs["noise"]; m = bs["multihop"]
    return (f"  rw={rw} floor={fl:.2f} | "
            f"known {k['answered']}/{k['n']} (resc {k['rescued']}) | "
            f"unk silenced {u['silenced']}/{u['n']} (wrong-rescue {u['rescued']}) | "
            f"noise silenced {n['silenced']}/{n['n']} (wrong-rescue {n['rescued']}) | "
            f"mh OK {m['ok']}/{m['n']} (resc {m['ok_rescued']}) "
            f"WRONG {m['wrong']} (resc {m['wrong_rescued']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        return 2
    data = json.loads(inp.read_text(encoding="utf-8"))
    records: list[dict] = data["records"]

    # Baseline (no rescue).
    baseline = _summarise(records, rank_window=0, floor=10.0)  # never rescues

    print(f"=== exp25 baseline (no rescue) at T={T_NORMAL} ===")
    print(_format_table_row(baseline))
    print()
    print(f"=== rescue floor sweep (rank_window x activation_floor) ===")
    sweep_rows = []
    for rw in RANK_WINDOWS:
        for fl in FLOORS:
            row = _summarise(records, rank_window=rw, floor=fl)
            sweep_rows.append(row)
            print(_format_table_row(row))
        print()

    # Find the most permissive (lowest floor) row at each rank_window where
    # constraints hold: no wrong rescue in unknown/noise/multihop, no
    # multi-hop wrong overall stays at baseline level.
    base_mh_wrong = baseline["by_survey"]["multihop"]["wrong"]
    safe_rows = [r for r in sweep_rows
                 if r["by_survey"]["unknown"]["rescued"] == 0
                 and r["by_survey"]["noise"]["rescued"] == 0
                 and r["by_survey"]["multihop"]["wrong_rescued"] == 0
                 and r["by_survey"]["multihop"]["wrong"] <= base_mh_wrong]
    # Among safe rows, pick the one that maximises mh-OK then known answered,
    # breaking ties by lower rank_window (simpler).
    safe_rows.sort(
        key=lambda r: (
            -r["by_survey"]["multihop"]["ok"],
            -r["by_survey"]["known"]["answered"],
            r["rank_window"],
            -r["activation_floor"],
        )
    )
    recommended = safe_rows[0] if safe_rows else None
    print()
    print("=== recommended (max mh-OK among safe rows) ===")
    if recommended is None:
        print("  no safe row found")
    else:
        print(_format_table_row(recommended))
        print()
        print(f"  rescued records ({len(recommended['rescued_records'])}):")
        for r in recommended["rescued_records"]:
            print(f"    [{r['survey']}] m={r['margin']:.4f}  "
                  f"primary={r['primary_entity']!r}  kw_hit={r['kw_hit']}")
            print(f"      supporting: rank={r['supporting_cell']['rank']}  "
                  f"act={r['supporting_cell']['activation']:.4f}  "
                  f"cell={r['supporting_cell']['cell_id']}  "
                  f"anchors={r['supporting_cell']['anchors_present']}")
            print(f"      query={r['query'][:80]}")

    out = {
        "input": str(inp),
        "T_normal": T_NORMAL,
        "baseline": baseline,
        "sweep": sweep_rows,
        "recommended": recommended,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
