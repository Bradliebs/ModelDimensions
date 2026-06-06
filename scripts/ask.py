"""CLI: ask the V1 pipeline a question against the production bank.

    python scripts/ask.py "What is...?"

Exits 0 on a grounded answer, 0 on honest silence, 2 on error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow `python scripts/ask.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BANK = os.environ.get(
    "MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 cited-answer CLI.")
    parser.add_argument("question", help="Question to ask the bank.")
    parser.add_argument(
        "--bank-path", default=DEFAULT_BANK,
        help=f"SQLite bank path (default: {DEFAULT_BANK!r}).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--margin", type=float, default=0.03,
        help="Silence-gate margin threshold (top1 - top2).",
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Load Phi-3 in bfloat16 instead of 4-bit NF4.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the full PipelineResult as JSON.",
    )
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    # Heavy imports after arg parse so --help is fast.
    print(f"[ask] loading bank: {bank_path}", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=args.margin,
        use_4bit=not args.no_4bit,
    )
    print(f"[ask] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    result = pipeline.ask(args.question)
    print(f"[ask] answered in {time.time() - t0:.2f}s", flush=True)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print()
        print(f"Q: {args.question}")
        print(f"A: {result.answer}")
        if result.silence:
            print(f"(silence: {result.silence_reason})")
        else:
            cites = ", ".join(str(c) for c in result.citations)
            print(f"(citations: [{cites}])")
        if result.closest_topics:
            topics = "; ".join(
                f"{t['topic']!r} ({t['activation']:.3f})"
                for t in result.closest_topics
            )
            print(f"(closest topics: {topics})")
        print(
            f"(gate margin={result.gate['margin']:.3f} >= "
            f"{result.gate['threshold']:.2f}, "
            f"verify coverage="
            f"{(result.verification or {}).get('coverage', 'n/a')})"
        )

    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
