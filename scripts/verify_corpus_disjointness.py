"""Verify a candidate training corpus is disjoint from the concept-cell bank.

Reads paragraph fingerprints from a cc_service SQLite bank and a corpus file
(JSONL, one JSON object per line with a 'text' field; optional 'title').
Reports per-paragraph and per-article overlap, and exits non-zero if any
gate fails.

Usage:
    python scripts/verify_corpus_disjointness.py \\
        --bank H:/MiniLM/cc_service/bank.db \\
        --corpus path/to/corpus_shard.jsonl \\
        --out-json reports/disjointness.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retro.diagnostics.disjointness import (  # noqa: E402
    DisjointnessPolicy,
    build_fingerprint_set,
    build_title_set,
    check_disjointness,
    iter_bank_source_texts,
    render_report_markdown,
)


def _iter_corpus(path: Path) -> Iterator[tuple[str, str | None]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} not valid JSON: {exc}") from exc
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            title = obj.get("title")
            yield text, (title if isinstance(title, str) and title.strip() else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--bank-titles", type=Path, default=None,
                        help="Optional JSONL with {'title': ...} per line.")
    parser.add_argument("--max-paragraph-overlap-ratio", type=float, default=0.001)
    parser.add_argument("--max-title-overlap-ratio", type=float, default=0.0)
    parser.add_argument("--min-paragraph-chars", type=int, default=20)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.bank.exists():
        parser.error(f"bank not found: {args.bank}")
    if not args.corpus.exists():
        parser.error(f"corpus not found: {args.corpus}")

    policy = DisjointnessPolicy(
        max_paragraph_overlap_ratio=args.max_paragraph_overlap_ratio,
        max_title_overlap_ratio=args.max_title_overlap_ratio,
        min_paragraph_chars=args.min_paragraph_chars,
    )

    print(f"[disjointness] reading bank paragraphs from {args.bank}")
    bank_fps = build_fingerprint_set(
        iter_bank_source_texts(args.bank), min_chars=policy.min_paragraph_chars
    )
    print(f"[disjointness] bank fingerprints: {len(bank_fps):,}")

    bank_titles_set: set[str] | None = None
    if args.bank_titles is not None:
        if not args.bank_titles.exists():
            parser.error(f"bank-titles not found: {args.bank_titles}")
        with args.bank_titles.open("r", encoding="utf-8") as fh:
            iter_titles = (
                json.loads(line).get("title", "")
                for line in fh
                if line.strip() and not line.startswith("#")
            )
            bank_titles_set = build_title_set(iter_titles)

    print(f"[disjointness] reading corpus from {args.corpus}")
    corpus_texts: list[str] = []
    corpus_titles_seen: list[str] = []
    for text, title in _iter_corpus(args.corpus):
        corpus_texts.append(text)
        if title:
            corpus_titles_seen.append(title)
    corpus_fps = build_fingerprint_set(corpus_texts, min_chars=policy.min_paragraph_chars)
    print(f"[disjointness] corpus fingerprints: {len(corpus_fps):,}")
    corpus_title_set = build_title_set(corpus_titles_seen) if corpus_titles_seen else None

    report = check_disjointness(
        bank_fingerprints=bank_fps,
        corpus_fingerprints=corpus_fps,
        policy=policy,
        bank_titles=bank_titles_set,
        corpus_titles=corpus_title_set,
    )

    md = render_report_markdown(report)
    print(md)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")

    return 0 if report.passed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
