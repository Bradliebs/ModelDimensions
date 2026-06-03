"""Minimal, inspectable JSONL persistence for concept-cell memories.

A memory record stores everything needed to (a) reload the bank without any
model download, and (b) audit by eye: the canonical text, the full embedding
vector, the cell threshold, provenance, and an optional binding group.

This is the only addition to the ``concept_cells`` package. It does not touch
the geometry, whitening, scaling, write, query, or Oja-binding code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    """One persisted concept-cell memory.

    The vector is stored as a plain list of floats so the file is fully
    self-describing and reloadable with no embedding model present.
    """

    memory_id: str
    canonical_text: str
    vector: List[float]
    threshold: float
    source: str = "user"
    created_at: str = field(default_factory=_utc_now_iso)
    bound_group_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.vector)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "MemoryRecord":
        data = json.loads(line)
        return cls(
            memory_id=data["memory_id"],
            canonical_text=data["canonical_text"],
            vector=list(data["vector"]),
            threshold=float(data["threshold"]),
            source=data.get("source", "user"),
            created_at=data.get("created_at", _utc_now_iso()),
            bound_group_id=data.get("bound_group_id"),
            tags=list(data.get("tags", [])),
        )


def write_records(path: str | Path, records: List[MemoryRecord]) -> None:
    """Overwrite ``path`` with the given records, one JSON object per line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")


def read_records(path: str | Path) -> List[MemoryRecord]:
    """Load all records from a JSONL file. Returns [] if the file is absent."""
    p = Path(path)
    if not p.exists():
        return []
    out: List[MemoryRecord] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(MemoryRecord.from_json(line))
    return out


def append_record(path: str | Path, record: MemoryRecord) -> None:
    """Append a single record to the JSONL file (creating it if needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(record.to_json() + "\n")
