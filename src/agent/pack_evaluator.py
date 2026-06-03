"""Pack evaluator (v1.8): validate a curated pack's retrieval and provenance.

Before trusting a curated knowledge pack, an operator runs a small set of
*evaluation questions* against it. Each question asserts what the pack's own
retrieval backend should return: the expected domain, the expected source and
its authority, that the retrieved chunk contains a known phrase, that forbidden
terms never appear, and that domain-safety flags (e.g. ``informational_only``
for medical/legal) are set.

This module only *reads* through the frozen knowledge query path
(:meth:`WorkbenchService.query_knowledge`); it changes no geometry, no grounding
policy, and writes nothing. It turns each question into a
:class:`~agent.pack_builder.PackEvalResult` recording pass/fail and a reason.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from agent.pack_builder import PackEvalQuestion, PackEvalResult
from agent.workbench_service import WorkbenchService

# Flags a question may require, mapped to a predicate over the knowledge audit.
# ``informational_only`` is the medical/legal safety flag; ``knowledge_used``
# asserts the pack actually answered from an imported source.
_SUPPORTED_FLAGS = {"informational_only", "knowledge_used"}


def load_eval_questions(path: str | Path) -> List[PackEvalQuestion]:
    """Load evaluation questions from a JSONL file (one question per line)."""
    questions: List[PackEvalQuestion] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            questions.append(PackEvalQuestion.from_dict(json.loads(line)))
    return questions


def evaluate_question(service: WorkbenchService,
                      question: PackEvalQuestion) -> PackEvalResult:
    """Run one question through the pack's knowledge query path and score it."""
    audit = service.query_knowledge(question.query)
    candidates = audit.candidates
    checks: List[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # Which candidates back the expected source (used for authority scoping).
    if question.expected_source:
        scoped = [c for c in candidates
                  if question.expected_source in (c["source_name"],
                                                  c["source_id"])]
    else:
        scoped = candidates

    if question.expected_domain is not None:
        ok = audit.domain == question.expected_domain
        record("expected_domain", ok,
               f"domain={audit.domain!r} expected={question.expected_domain!r}")

    if question.expected_source is not None:
        names = {c["source_name"] for c in candidates}
        ids = {c["source_id"] for c in candidates}
        ok = question.expected_source in names or question.expected_source in ids
        record("expected_source", ok,
               f"retrieved={sorted(names)} expected={question.expected_source!r}")

    if question.expected_authority is not None:
        pool = scoped or candidates
        got = {c["authority"] for c in pool}
        ok = question.expected_authority in got
        record("expected_authority", ok,
               f"authorities={sorted(got)} expected={question.expected_authority!r}")

    if question.expected_chunk_contains is not None:
        needle = question.expected_chunk_contains.lower()
        pool = scoped or candidates
        ok = any(needle in (c["text"] or "").lower() for c in pool)
        record("expected_chunk_contains", ok,
               f"no retrieved chunk contains {question.expected_chunk_contains!r}"
               if not ok else "found")

    for term in question.forbidden_terms:
        needle = term.lower()
        hit = next((c["source_name"] for c in candidates
                    if needle in (c["text"] or "").lower()), None)
        record(f"forbidden:{term}", hit is None,
               "absent" if hit is None
               else f"forbidden term {term!r} present in {hit!r}")

    for flag in question.required_flags:
        if flag not in _SUPPORTED_FLAGS:
            record(f"flag:{flag}", False, f"unknown required flag {flag!r}")
            continue
        present = bool(getattr(audit, flag))
        record(f"flag:{flag}", present,
               "set" if present else f"required flag {flag!r} not set")

    failed = [c for c in checks if not c["ok"]]
    passed = not failed
    if not checks:
        reason = "no assertions declared"
    elif passed:
        reason = "all checks passed"
    else:
        reason = "; ".join(c["detail"] for c in failed)

    return PackEvalResult(
        query=question.query,
        passed=passed,
        reason=reason,
        question_id=question.question_id,
        checks=checks,
    )


def evaluate_pack(service: WorkbenchService,
                  questions: List[PackEvalQuestion]) -> List[PackEvalResult]:
    """Evaluate every question against the pack, returning per-question results."""
    return [evaluate_question(service, q) for q in questions]


def evaluate_pack_file(service: WorkbenchService,
                       path: str | Path) -> List[PackEvalResult]:
    """Load questions from a JSONL file and evaluate them against the pack."""
    return evaluate_pack(service, load_eval_questions(path))
