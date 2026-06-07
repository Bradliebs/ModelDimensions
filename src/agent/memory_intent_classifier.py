"""Phase 3 — lexical intent classifier for governed memory mutations.

The Bank Management Workspace and the V1 admin path both expose narrow
mutations to the overlay store: **add a cell from text**, **tombstone a
cell by id**, or **inspect provenance**. Today operators run brittle
``python -c`` one-liners (see RUNBOOK sections 4–6). Phase 3 replaces
that with a typed orchestration layer; this module is its front door.

Design constraints:

* **Lexical, not LLM.** Operator instructions follow short, stable
  patterns ("remember that …", "tombstone 12345: wrong attribution",
  "show provenance"). A regex parser is deterministic, testable, fast,
  and consistent with the Phase 2 Stage F decomposer (also lexical).
  Loading Phi-3 to classify a six-word imperative is over-engineering.
* **Closed kind set.** ``MemoryIntentKind`` is one of ``add``, ``remove``,
  ``inspect``, ``unknown``. Unknown is *honest*: the orchestrator refuses
  to mutate on an instruction it could not classify. We do not fall back
  to a "best guess" — silent misclassification is the failure mode this
  layer exists to prevent.
* **No side effects.** ``classify`` is pure. The orchestrator decides
  what to do with the intent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


class MemoryIntentKind:
    """Closed set of recognised operator instructions."""

    ADD = "add"
    REMOVE = "remove"
    INSPECT = "inspect"
    UNKNOWN = "unknown"


# Match in priority order (most specific first). Each pattern captures
# the operands the orchestrator needs; ``classify`` populates the
# ``fields`` dict of :class:`MemoryIntent` from those groups.
_REMOVE_RE = re.compile(
    r"""
    ^\s*
    (?:tombstone|remove|delete|drop)
    \s+
    (?:cell\s+)?
    (?P<cell_id>\d+)
    \s*
    (?:
        [:\-,]\s*          # 'remove 42: bad data'
      | \s+because\s+      # 'remove 42 because bad data'
      | \s+reason\s+       # 'remove 42 reason bad data'
    )
    (?P<reason>\S.*?)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Inspect must be checked before add so 'show provenance' isn't read as
# 'remember "provenance"' if the add pattern ever broadens.
_INSPECT_RE = re.compile(
    r"^\s*(?:show|list|inspect|print)\s+provenance\s*$",
    re.IGNORECASE,
)

_ADD_RE = re.compile(
    r"""
    ^\s*
    (?:remember|add|note|record|store)
    (?:\s+that)?
    \s*[:]?\s*
    (?P<text>\S.+?)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class MemoryIntent:
    """Structured operator instruction.

    ``fields`` carries the typed operands the orchestrator will plug into
    :class:`OverlayStore` calls. ``raw_text`` is preserved verbatim so
    audit records keep the operator's original phrasing.
    """

    kind: str
    raw_text: str
    fields: dict = field(default_factory=dict)
    note: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.kind in {
            MemoryIntentKind.ADD,
            MemoryIntentKind.REMOVE,
            MemoryIntentKind.INSPECT,
        }


def classify(instruction: str) -> MemoryIntent:
    """Parse ``instruction`` into a :class:`MemoryIntent`.

    Returns ``MemoryIntentKind.UNKNOWN`` on any string the patterns do
    not recognise — including empty input. Callers MUST treat ``unknown``
    as a refusal to act, not an invitation to guess.
    """
    if not isinstance(instruction, str):
        return MemoryIntent(
            kind=MemoryIntentKind.UNKNOWN,
            raw_text="",
            note="instruction was not a string",
        )

    text = instruction.strip()
    if not text:
        return MemoryIntent(
            kind=MemoryIntentKind.UNKNOWN,
            raw_text="",
            note="empty instruction",
        )

    m = _REMOVE_RE.match(text)
    if m:
        return MemoryIntent(
            kind=MemoryIntentKind.REMOVE,
            raw_text=text,
            fields={
                "cell_id": int(m.group("cell_id")),
                "reason": m.group("reason").strip(),
            },
        )

    m = _INSPECT_RE.match(text)
    if m:
        return MemoryIntent(
            kind=MemoryIntentKind.INSPECT,
            raw_text=text,
            fields={},
        )

    m = _ADD_RE.match(text)
    if m:
        payload = m.group("text").strip().strip('"\u201c\u201d\u2018\u2019')
        if not payload:
            return MemoryIntent(
                kind=MemoryIntentKind.UNKNOWN,
                raw_text=text,
                note="add instruction had no payload after the verb",
            )
        return MemoryIntent(
            kind=MemoryIntentKind.ADD,
            raw_text=text,
            fields={"text": payload},
        )

    return MemoryIntent(
        kind=MemoryIntentKind.UNKNOWN,
        raw_text=text,
        note="no recognised verb pattern matched",
    )


__all__ = ["MemoryIntent", "MemoryIntentKind", "classify"]
