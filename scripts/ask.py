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
DEFAULT_OVERLAY = os.environ.get("MD_OVERLAY_PATH", "")
DEFAULT_LEXICAL_INDEX = os.environ.get("MD_LEXICAL_INDEX", "")


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 cited-answer CLI.")
    parser.add_argument("question", help="Question to ask the bank.")
    parser.add_argument(
        "--bank-path", default=DEFAULT_BANK,
        help=f"SQLite bank path (default: {DEFAULT_BANK!r}).",
    )
    parser.add_argument(
        "--overlay-path", default=DEFAULT_OVERLAY,
        help="Optional overlay SQLite to merge into the bank "
             "(default: env MD_OVERLAY_PATH or none).",
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
        "--lexical-index", default=DEFAULT_LEXICAL_INDEX,
        help="Directory containing a saved LexicalIndex. If set, enables "
             "the BM25 → cosine hybrid retrieval cascade (default: env "
             "MD_LEXICAL_INDEX or none).",
    )
    parser.add_argument(
        "--lexical-k", type=int, default=200,
        help="BM25 candidate pool size before cosine re-rank (default: 200).",
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

    overlay_path: Path | None = None
    if args.overlay_path:
        overlay_path = Path(args.overlay_path)
        if not overlay_path.exists():
            print(
                f"ERROR: overlay not found: {overlay_path}", file=sys.stderr
            )
            return 2

    lexical_path: Path | None = None
    if args.lexical_index:
        lexical_path = Path(args.lexical_index)
        if not lexical_path.exists():
            print(
                f"ERROR: lexical index not found: {lexical_path}",
                file=sys.stderr,
            )
            return 2

    # Heavy imports after arg parse so --help is fast.
    print(f"[ask] loading bank: {bank_path}", flush=True)
    if overlay_path is not None:
        print(f"[ask] merging overlay: {overlay_path}", flush=True)
    if lexical_path is not None:
        print(f"[ask] loading lexical index: {lexical_path}", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline
    bank_arg = None
    if overlay_path is not None:
        from src.agent.bank_admin import OverlayStore
        from src.agent.streaming_bank import StreamingBank
        overlay = OverlayStore(overlay_path)
        bank_arg = StreamingBank(str(bank_path), overlay=overlay)
    lexical_index = None
    if lexical_path is not None:
        from src.agent.lexical_index import LexicalIndex
        lexical_index = LexicalIndex.load(lexical_path)
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        bank=bank_arg,
        top_k=args.top_k,
        margin_threshold=args.margin,
        use_4bit=not args.no_4bit,
        lexical_index=lexical_index,
        lexical_k=args.lexical_k,
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
