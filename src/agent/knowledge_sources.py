"""Data model for source-governed knowledge import (v1.3).

Imported knowledge is deliberately **separate** from project memory. A project
memory is something the user/team decided or recorded through the workbench; a
knowledge chunk is a passage from an *external* source (docs, a manual, a
reference page) that the user pulled in for reference. The two are never mixed:
the concept-cell ``MemoryBank`` and the ``MemoryLedger`` own project memory;
these dataclasses and the ``KnowledgeLibrary`` own imported knowledge.

Every chunk carries its source's identity (name, domain, authority, version) so
a query-time answer can cite where it came from and apply domain safety rules
(for example, marking a medical or legal passage as informational only).

The file format used by the library is one JSON object per line, fully
self-describing and readable by eye, mirroring the memory ledger's style.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeDomain(str, Enum):
    """The subject area a source belongs to (drives query-time safety rules)."""

    CODING = "coding"
    MEDICAL = "medical"
    LEGAL = "legal"
    MICROSOFT = "microsoft"
    GENERAL = "general"
    PROJECT_DOCS = "project_docs"


class SourceAuthority(str, Enum):
    """How much weight a source's origin carries.

    ``official`` (the vendor/standard itself) and ``reputable`` (a well-regarded
    secondary source) are trusted; ``community`` and ``unknown`` are flagged at
    query time so the reader knows to verify.
    """

    OFFICIAL = "official"
    REPUTABLE = "reputable"
    COMMUNITY = "community"
    UNKNOWN = "unknown"


# Authorities considered trustworthy enough not to warn about.
TRUSTED_AUTHORITIES = {SourceAuthority.OFFICIAL, SourceAuthority.REPUTABLE}


@dataclass
class KnowledgeSource:
    """One imported external source and its provenance metadata."""

    source_id: str
    source_name: str
    source_type: str  # e.g. "markdown", "url", "manual"
    domain: KnowledgeDomain
    authority: SourceAuthority
    path_or_url: str
    staleness_policy: str  # e.g. "static", "review-quarterly", "volatile"
    licence: Optional[str] = None
    retrieved_at: str = field(default_factory=_utc_now_iso)
    published_at: Optional[str] = None
    version: Optional[str] = None
    active: bool = True

    def to_json(self) -> str:
        import json
        data = asdict(self)
        data["domain"] = self.domain.value
        data["authority"] = self.authority.value
        data["_record"] = "source"
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeSource":
        return cls(
            source_id=data["source_id"],
            source_name=data["source_name"],
            source_type=data["source_type"],
            domain=KnowledgeDomain(data["domain"]),
            authority=SourceAuthority(data["authority"]),
            path_or_url=data["path_or_url"],
            staleness_policy=data.get("staleness_policy", "static"),
            licence=data.get("licence"),
            retrieved_at=data.get("retrieved_at", _utc_now_iso()),
            published_at=data.get("published_at"),
            version=data.get("version"),
            active=bool(data.get("active", True)),
        )


@dataclass
class KnowledgeDocument:
    """One document within a source (a source may hold several documents)."""

    document_id: str
    source_id: str
    title: Optional[str] = None
    path_or_url: Optional[str] = None

    def to_json(self) -> str:
        import json
        data = asdict(self)
        data["_record"] = "document"
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeDocument":
        return cls(
            document_id=data["document_id"],
            source_id=data["source_id"],
            title=data.get("title"),
            path_or_url=data.get("path_or_url"),
        )


@dataclass
class KnowledgeChunk:
    """One retrievable passage, carrying a copy of its source metadata.

    The source fields (``source_name``, ``domain``, ``authority``) are
    denormalised onto the chunk so retrieval can label and guard a result
    without a second lookup. ``active`` is set ``False`` when the owning source
    is deleted, so a deleted source can never be cited again.
    """

    chunk_id: str
    source_id: str
    document_id: str
    chunk_text: str
    domain: KnowledgeDomain
    authority: SourceAuthority
    source_name: str
    source_section: Optional[str] = None
    active: bool = True

    def to_json(self) -> str:
        import json
        data = asdict(self)
        data["domain"] = self.domain.value
        data["authority"] = self.authority.value
        data["_record"] = "chunk"
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeChunk":
        return cls(
            chunk_id=data["chunk_id"],
            source_id=data["source_id"],
            document_id=data["document_id"],
            chunk_text=data["chunk_text"],
            domain=KnowledgeDomain(data["domain"]),
            authority=SourceAuthority(data["authority"]),
            source_name=data["source_name"],
            source_section=data.get("source_section"),
            active=bool(data.get("active", True)),
        )
