"""Governed HF eval-pack → retrieval-eval bridge — Phase F (v6.9).

This phase closes the loop between an *imported eval pack* (Phase D) and the
existing read-only retrieval-evaluation harness (``retrieval_eval_harness``),
**without activating anything in retrieval**.

Governance honoured here, by construction:

* **Imported is not activated.** This bridge turns an imported eval pack's
  questions into inert retrieval-eval *cases* — a JSONL artefact. It never loads
  the pack's knowledge into a ``WorkbenchService``, never registers a source,
  never calls ``query_knowledge``, and never mutates retrieval, ranking, or
  source selection. Producing eval cases is not the same as activating a pack;
  running them against a service is a separate, human-initiated step using the
  unchanged harness.
* **The harness is unchanged.** The cases this bridge emits are ordinary
  ``RetrievalEvalCase`` records (gap probes by default: ``minimum_hit_k=0``), so
  the v3.0 harness loads and scores them with no modification. Whether a pack's
  source is even present in the evaluated service is the operator's choice.
* **Lineage travels with the case.** Every emitted case carries the originating
  ``question_id`` as its ``case_id`` and the dataset id as the expected source,
  so a retrieval result is always traceable back to the approved import.
* **Pure transform.** This module reads an eval pack from disk (or in-memory
  questions) and writes a cases file. It imports no writer, no service, no
  network client, and no retrieval activator. The docstring is explicit so the
  import-purity test can strip it before scanning the body.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from agent.hf_eval_pack_importer import HFEvalPackImportResult, HFEvalPackQuestion

BRIDGE_VERSION = "hf-retrieval-bridge-v6.9"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Reading a written eval pack (no activation, no service)
# --------------------------------------------------------------------------- #


def load_eval_pack(
    pack_dir: str | Path,
) -> Tuple[Dict, List[HFEvalPackQuestion]]:
    """Read a written HF eval pack's manifest and questions from disk.

    Returns the manifest dict and the list of questions. This is a plain file
    read: it does not load the pack into retrieval or touch any service.
    """
    base = Path(pack_dir)
    manifest_path = base / "manifest.json"
    questions_path = base / "eval_questions.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in eval pack: {base}")
    if not questions_path.exists():
        raise FileNotFoundError(f"no eval_questions.jsonl in eval pack: {base}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pack_kind") != "eval":
        raise ValueError(
            f"not an eval pack (pack_kind={manifest.get('pack_kind')!r}): {base}")

    questions: List[HFEvalPackQuestion] = []
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        questions.append(HFEvalPackQuestion.from_dict(json.loads(line)))
    return manifest, questions


# --------------------------------------------------------------------------- #
# Building retrieval cases (pure transform)
# --------------------------------------------------------------------------- #


def questions_to_retrieval_cases(
    questions: Sequence[HFEvalPackQuestion],
) -> List[dict]:
    """Transform imported eval questions into retrieval-eval-case dicts.

    Each case is a gap probe by default (``minimum_hit_k=0``): it asserts no
    forbidden bleed and is traceable to its source dataset, but does not force a
    hit, because whether the imported pack's source is present in the evaluated
    service is the operator's decision — not something this bridge can assume.
    """
    return [q.to_retrieval_case() for q in questions]


def eval_pack_to_retrieval_cases(pack_dir: str | Path) -> List[dict]:
    """Read a written eval pack and emit its retrieval-eval cases (no activation)."""
    _, questions = load_eval_pack(pack_dir)
    return questions_to_retrieval_cases(questions)


def result_to_retrieval_cases(result: HFEvalPackImportResult) -> List[dict]:
    """Emit retrieval cases directly from an in-memory import result."""
    return questions_to_retrieval_cases(result.questions)


# --------------------------------------------------------------------------- #
# Writing the inert cases file (the only durable write)
# --------------------------------------------------------------------------- #


def write_retrieval_cases(
    cases: Sequence[dict], path: str | Path, *,
    header: Optional[str] = None, now: Optional[datetime] = None,
) -> str:
    """Atomically write retrieval-eval cases as JSONL; returns the written path.

    The file is an inert eval artefact. Writing it activates nothing; it is fed
    to the unchanged retrieval-eval harness by a human when they choose to score
    a service against these probes.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = (now or _utc_now()).isoformat()
    lines = [
        f"# {header}" if header else
        "# Governed HF-import retrieval-eval cases (v6.9)",
        f"# generated: {stamp} by {BRIDGE_VERSION}",
        "# Inert probes. Loading this file activates nothing in retrieval.",
    ]
    for case in cases:
        lines.append(json.dumps(case, ensure_ascii=False, sort_keys=True))
    content = "\n".join(lines) + "\n"

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return str(target)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_bridge_markdown(
    cases: Sequence[dict], *, pack_label: str = "",
) -> str:
    """Deterministic summary of emitted cases. Never prints expected answers."""
    tags: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    for case in cases:
        for tag in case.get("tags") or []:
            tags[tag] = tags.get(tag, 0) + 1
        for src in case.get("expected_sources") or []:
            sources[src] = sources.get(src, 0) + 1
    lines = [
        "# Governed HF-import retrieval-eval bridge (v6.9)",
        "",
        f"- pack: `{pack_label}`" if pack_label else "- pack: (in-memory)",
        f"- cases: {len(cases)}",
        f"- sources: {', '.join(f'`{s}` ({n})' for s, n in sorted(sources.items()))}"
        if sources else "- sources: (none)",
        f"- tags: {', '.join(f'`{t}` ({n})' for t, n in sorted(tags.items()))}"
        if tags else "- tags: (none)",
        "",
        "> Inert eval cases. Running them against a service is a separate, "
        "human-initiated step using the unchanged retrieval-eval harness.",
    ]
    return "\n".join(lines) + "\n"
