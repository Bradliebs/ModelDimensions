"""JSONL-backed memory ledger for the v1.1 workbench.

The ledger is a *metadata* record of the memories a user has written through the
workbench: it is deliberately separate from the concept-cell substrate. The
concept-cell ``MemoryBank`` owns the geometry (vectors, thresholds, firing); the
ledger owns the human-facing provenance (who wrote it, when, with what tags, and
whether it is still active).

It changes no core geometry and reuses the frozen ``MemoryBank`` minted ids: the
workbench writes a memory into the bank, then records the *same* ``memory_id``
here so the two stay aligned. Deleting a memory marks the ledger entry
``deleted`` (it is never silently dropped) so an audit can still see that the
memory existed and was removed.

The file format is one JSON object per line, fully self-describing and readable
by eye.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


ACTIVE = "active"
DELETED = "deleted"


@dataclass
class LedgerEntry:
    """One memory's provenance record.

    ``status`` is ``"active"`` or ``"deleted"``. A deleted entry keeps its
    ``deleted_at`` timestamp so the removal itself is auditable.
    """

    memory_id: str
    canonical_text: str
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    status: str = ACTIVE
    deleted_at: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "LedgerEntry":
        data = json.loads(line)
        return cls(
            memory_id=data["memory_id"],
            canonical_text=data["canonical_text"],
            source=data.get("source"),
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", _utc_now_iso()),
            status=data.get("status", ACTIVE),
            deleted_at=data.get("deleted_at"),
        )


class MemoryLedger:
    """An in-memory ledger that writes through to a JSONL file.

    Every mutation (add / mark_deleted) is persisted immediately so the file on
    disk is always a faithful copy of the ledger state. ``path`` may be ``None``
    for a purely in-memory ledger (used by tests that supply their own path).
    """

    def __init__(self, path: Optional[str | Path] = None, *,
                 load_existing: bool = True):
        self.path = Path(path) if path is not None else None
        self._entries: Dict[str, LedgerEntry] = {}
        if load_existing and self.path is not None and self.path.exists():
            self.load()

    # -- mutating operations --

    def add(self, memory_id: str, canonical_text: str,
            source: Optional[str] = None,
            tags: Optional[List[str]] = None) -> LedgerEntry:
        """Record a newly written memory as active."""
        if memory_id in self._entries:
            raise ValueError(f"duplicate memory_id in ledger: {memory_id}")
        entry = LedgerEntry(
            memory_id=memory_id,
            canonical_text=canonical_text,
            source=source,
            tags=list(tags) if tags else [],
        )
        self._entries[memory_id] = entry
        self._flush()
        return entry

    def mark_deleted(self, memory_id: str) -> bool:
        """Mark a memory deleted. Returns True if an active entry was changed."""
        entry = self._entries.get(memory_id)
        if entry is None or entry.status == DELETED:
            return False
        entry.status = DELETED
        entry.deleted_at = _utc_now_iso()
        self._flush()
        return True

    # -- read operations --

    def get(self, memory_id: str) -> Optional[LedgerEntry]:
        return self._entries.get(memory_id)

    def entries(self) -> List[LedgerEntry]:
        """All entries in insertion order."""
        return list(self._entries.values())

    def active_entries(self) -> List[LedgerEntry]:
        return [e for e in self._entries.values() if e.status == ACTIVE]

    def export(self) -> List[dict]:
        """Return the full ledger (active and deleted) as plain dicts."""
        return [asdict(e) for e in self._entries.values()]

    # -- persistence --

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for entry in self._entries.values():
                fh.write(entry.to_json() + "\n")

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        self._entries.clear()
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = LedgerEntry.from_json(line)
                self._entries[entry.memory_id] = entry
