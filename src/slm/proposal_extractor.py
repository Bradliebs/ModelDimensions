"""Optional SLM-assisted memory proposal extraction.

This is a thin, safe wrapper over the deterministic note ingester. It can
optionally ask a local SLM to *suggest* additional candidate memories, but the
SLM is never trusted: every suggestion becomes an ordinary
:class:`~agent.memory_proposals.MemoryProposal` with status ``pending``. Nothing
here writes to the memory bank, and SLM-sourced proposals go through exactly the
same human approval gate as deterministic ones.

If the SLM is unavailable or returns anything malformed, the result is simply
the deterministic extraction — so calling with ``use_slm=True`` can only ever
*add* reviewable suggestions, never remove safety.
"""
from __future__ import annotations

import hashlib
import json
from typing import List, Optional

from agent.memory_proposals import (
    MemoryProposal,
    ProposalBatch,
    ProposalKind,
    ProposalStatus,
)
from agent.note_ingestion import extract_candidate_memories
from slm.local_slm_backend import LocalSLMBackend

# Suggestions are capped and length-limited so a chatty model cannot flood the
# review queue with long, unreviewable items.
_MAX_SLM_SUGGESTIONS = 12
_MAX_LEN = 200


def propose_memories_from_note(
        note_text: str, *,
        source_file: str = "(slm note)",
        use_slm: bool = False,
        slm_backend: Optional[LocalSLMBackend] = None) -> ProposalBatch:
    """Extract candidate memories from a note, optionally aided by a local SLM.

    The deterministic extraction always runs. When ``use_slm`` is set and a
    backend is available, validated SLM suggestions are appended as additional
    pending proposals. Every proposal — deterministic or SLM-sourced — is
    returned as ``pending`` and must be approved before it can be written.
    """
    batch = extract_candidate_memories(note_text, source_file)
    if not use_slm or slm_backend is None or not slm_backend.is_available():
        return batch

    try:
        raw = slm_backend.generate(_build_prompt(note_text))
    except Exception:
        return batch

    suggestions = _parse_suggestions(raw)
    existing = {p.canonical_text.strip().lower() for p in batch.proposals}
    for idx, text in enumerate(suggestions):
        key = text.strip().lower()
        if not key or key in existing:
            continue
        existing.add(key)
        batch.proposals.append(MemoryProposal(
            proposal_id=_slm_proposal_id(source_file, text, idx),
            canonical_text=text,
            source_file=source_file,
            kind=ProposalKind.FACT,
            confidence=0.4,
            reason="slm-suggested; requires human approval like any proposal",
            status=ProposalStatus.PENDING,
        ))
    return batch


def _build_prompt(note_text: str) -> str:
    return (
        "Read the note below and suggest atomic candidate facts worth "
        "remembering. Return ONLY a JSON array of short strings, one fact each. "
        "Do not invent facts that are not supported by the note.\n\n"
        f"Note:\n{note_text}")


def _parse_suggestions(raw: str) -> List[str]:
    """Parse a JSON array of strings from raw model text; tolerate noise."""
    if not raw:
        return []
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    out: List[str] = []
    for item in data:
        if isinstance(item, str):
            text = item.strip()[:_MAX_LEN]
            if text:
                out.append(text)
        if len(out) >= _MAX_SLM_SUGGESTIONS:
            break
    return out


def _slm_proposal_id(source_file: str, text: str, idx: int) -> str:
    digest = hashlib.sha1(
        f"{source_file}|{idx}|{text}".encode("utf-8")).hexdigest()[:10]
    return f"slm-{digest}"
