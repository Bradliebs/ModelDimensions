"""Run the V1.1 independent verification audit.

Default output:
  reports/v1_1_verification_audit.json
  reports/v1_1_verification_audit.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.agent.verification_audit import (  # noqa: E402
    load_candidate_audit_cases,
    render_verification_audit_markdown,
    run_verification_audit,
    write_verification_audit,
)


DEFAULT_CASES = REPO_ROOT / "evals" / "fixtures" / "v1_1_independent_verification_cases.jsonl"
DEFAULT_JSON = REPO_ROOT / "reports" / "v1_1_verification_audit.json"
DEFAULT_MD = REPO_ROOT / "reports" / "v1_1_verification_audit.md"
DEFAULT_THRESHOLDS = "0.35,0.50,0.65,0.80"


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"threshold outside [0,1]: {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one threshold is required")
    return tuple(values)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V1.1 verifier audit.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        thresholds = _parse_thresholds(args.thresholds)
        cases = load_candidate_audit_cases(args.cases)
        report = run_verification_audit(cases, thresholds=thresholds)
        write_verification_audit(report, args.out_json)
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            render_verification_audit_markdown(report), encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        print(f"[verification-audit] ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"[verification-audit] wrote {args.out_json}")
    print(f"[verification-audit] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())