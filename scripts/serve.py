"""CLI entrypoint for the V1 ask service.

    python scripts/serve.py [--host 127.0.0.1] [--port 8080]
                            [--bank-path ...] [--top-k 10]
                            [--margin 0.05] [--no-4bit]

Loads the AnswerPipeline once (Phi-3 + bank), then serves an HTTP API
defined by :mod:`src.agent.service`. POST /ask {"question": "..."}.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BANK = os.environ.get(
    "MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 ask HTTP service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bank_path = Path(args.bank_path)
    if not bank_path.exists():
        print(f"ERROR: bank not found: {bank_path}", file=sys.stderr)
        return 2

    print(f"[serve] loading pipeline (bank={bank_path})...", flush=True)
    t0 = time.time()
    from src.agent.answer_pipeline import AnswerPipeline
    from src.agent.service import serve
    pipeline = AnswerPipeline(
        bank_path=bank_path,
        top_k=args.top_k,
        margin_threshold=args.margin,
        use_4bit=not args.no_4bit,
    )
    print(f"[serve] pipeline ready in {time.time() - t0:.1f}s", flush=True)

    try:
        serve(pipeline, host=args.host, port=args.port)
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
