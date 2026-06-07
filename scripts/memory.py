"""CLI: governed memory mutations against the V1 overlay store.

Replaces the brittle ``python -c "from src.agent...; ov.add_cell(...)"``
one-liners in RUNBOOK sections 4–6 with a typed plan / apply flow:

    # Dry-run — see what would happen, mutate nothing.
    python scripts/memory.py plan "remember that Mars has two moons" \
        --bank-path H:\\MiniLM\\cc_service\\bank.db

    # Apply — mutates the overlay; requires --confirm.
    python scripts/memory.py apply "remember that Mars has two moons" \
        --bank-path H:\\MiniLM\\cc_service\\bank.db \
        --overlay-path results/v1_bank/overlay.db \
        --confirm

    # Tombstone — does not need the bank.
    python scripts/memory.py apply "tombstone 123456: wrong attribution" \
        --overlay-path results/v1_bank/overlay.db --confirm

    # Inspect — read-only.
    python scripts/memory.py plan "show provenance" \
        --overlay-path results/v1_bank/overlay.db

Exit codes:
    0  success (plan produced, or apply landed and validator passed)
    2  plan blocked (unknown intent, missing operand, empty reason, …)
    3  apply failed or validator reported a regression
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow `python scripts/memory.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BANK = os.environ.get("MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db")
DEFAULT_OVERLAY = os.environ.get(
    "MD_OVERLAY_PATH", str(REPO_ROOT / "results" / "v1_bank" / "overlay.db")
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V1 governed memory mutation CLI.",
    )
    p.add_argument(
        "mode", choices=("plan", "apply"),
        help="'plan' is dry-run only; 'apply' mutates the overlay "
             "and requires --confirm.",
    )
    p.add_argument(
        "instruction",
        help="Natural-language instruction. Examples: "
             "'remember that Mars has two moons', "
             "'tombstone 123456: wrong attribution', "
             "'show provenance'.",
    )
    p.add_argument(
        "--overlay-path", default=DEFAULT_OVERLAY,
        help=f"Overlay SQLite path (default: {DEFAULT_OVERLAY!r}).",
    )
    p.add_argument(
        "--bank-path", default=DEFAULT_BANK,
        help=f"Base bank path; required for 'add' (default: {DEFAULT_BANK!r}).",
    )
    p.add_argument(
        "--confirm", action="store_true",
        help="Required for 'apply'. Without it apply refuses to run.",
    )
    p.add_argument(
        "--source", default=None,
        help="Optional provenance hint (e.g. 'manual', 'operator-bob').",
    )
    p.add_argument("--label", default=None, help="Optional cell label.")
    p.add_argument(
        "--theta", type=float, default=0.30,
        help="Per-cell threshold (default 0.30, matches base bank).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit the result as JSON instead of human-readable text.",
    )
    return p


def _emit(payload: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    # Human-readable summary.
    print(f"mode    : {payload.get('mode')}")
    print(f"op      : {payload.get('op')}")
    print(f"intent  : {payload.get('intent_kind')}")
    if payload.get("blocked"):
        print(f"BLOCKED : {payload.get('block_reason')}")
    for f in payload.get("preflight", []):
        print(f"  - {f}")
    if "applied" in payload:
        print(f"applied : {payload['applied']}")
        if payload.get("new_cell_id") is not None:
            print(f"new id  : {payload['new_cell_id']}")
        if payload.get("tombstoned_cell_id") is not None:
            print(f"tomb id : {payload['tombstoned_cell_id']}")
        if payload.get("error"):
            print(f"error   : {payload['error']}")
        v = payload.get("validator")
        if v:
            print(f"validator.ok = {v['ok']}")
            for c in v["checks"]:
                mark = "PASS" if c["ok"] else "FAIL"
                print(f"  {mark} {c['name']}: {c['detail']}")
        if payload.get("provenance") is not None:
            print(f"provenance ({len(payload['provenance'])} entries):")
            for row in payload["provenance"]:
                print(f"  {row}")


def main() -> int:
    args = _build_parser().parse_args()

    overlay_path = Path(args.overlay_path)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    # Heavy imports after arg parse so --help is fast.
    from src.agent.bank_admin import OverlayStore
    from src.agent.memory_intent_classifier import classify, MemoryIntentKind
    from src.agent.memory_orchestrator import apply as orch_apply
    from src.agent.memory_orchestrator import plan as orch_plan

    intent = classify(args.instruction)

    # Decide whether we need to open the base bank (only for 'add').
    bank = None
    base_max_id = None
    encoder = None
    whiten_fn = None
    bank_dim = None
    if intent.kind == MemoryIntentKind.ADD:
        bank_path = Path(args.bank_path)
        if not bank_path.exists():
            print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
            return 3
        from src.agent.streaming_bank import StreamingBank
        bank = StreamingBank(str(bank_path))
        base_max_id = int(bank.cell_ids.max())
        bank_dim = int(bank.dim)
        whiten_fn = bank.whiten

    overlay = OverlayStore(overlay_path)
    try:
        plan_obj = orch_plan(intent, overlay=overlay, base_max_id=base_max_id)

        payload: dict = {
            "mode": args.mode,
            "op": plan_obj.op,
            "intent_kind": intent.kind,
            "preflight": list(plan_obj.preflight_findings),
            "blocked": plan_obj.blocked,
            "block_reason": plan_obj.block_reason,
        }

        if args.mode == "plan":
            _emit(payload, as_json=args.json)
            return 0 if not plan_obj.blocked else 2

        # ---- apply ----
        if not args.confirm:
            payload["applied"] = False
            payload["error"] = "apply requires --confirm"
            _emit(payload, as_json=args.json)
            return 2

        if plan_obj.blocked:
            payload["applied"] = False
            payload["error"] = f"refusing to apply blocked plan: {plan_obj.block_reason}"
            _emit(payload, as_json=args.json)
            return 2

        if intent.kind == MemoryIntentKind.ADD:
            from src.cc_service.encoder import EncoderSingleton
            encoder = EncoderSingleton(model_name=bank.encoder_model)

        result = orch_apply(
            plan_obj,
            overlay=overlay,
            encoder=encoder,
            whiten_fn=whiten_fn,
            bank_dim=bank_dim,
            source=args.source,
            label=args.label,
            theta=args.theta,
        )
        payload.update(result.as_dict())
        _emit(payload, as_json=args.json)

        if not result.applied:
            return 3
        if result.validator is not None and not result.validator.ok:
            return 3
        return 0
    finally:
        overlay.close()
        if bank is not None:
            bank.close()


if __name__ == "__main__":
    raise SystemExit(main())
