"""JSONL-backed store for imported knowledge (v1.3).

The library holds external sources, their documents, and the retrievable chunks
extracted from them. It is intentionally **not** the memory ledger: importing a
document here never writes a project memory. Chunking reuses the note-ingestion
section splitter so a Markdown document is broken at its headings, then into
paragraph-sized passages, each tagged with its source's provenance.

Persistence mirrors the memory ledger: one JSON object per line, with a
``_record`` discriminator (``source`` / ``document`` / ``chunk``); every mutation
flushes the whole file by truncate-and-rewrite so the on-disk state always
matches memory exactly. Deleting a source deactivates the source and all of its
chunks (kept on disk for audit, but never retrievable again).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from .knowledge_sources import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeDomain,
    KnowledgeSource,
    SourceAuthority,
)
from .note_ingestion import load_text_file, split_markdown_sections

_MAX_CHUNK_LEN = 400


def _digest(*parts: str) -> str:
    h = hashlib.sha1("\u0001".join(parts).encode("utf-8"))
    return h.hexdigest()[:8]


def _chunk_paragraphs(text: str) -> List[tuple[Optional[str], str]]:
    """Split a document into (section_heading, paragraph_text) pairs.

    Sections come from the Markdown heading splitter; within each section the
    body is split on blank lines into paragraphs, with bullet/heading markup
    stripped. Empty paragraphs are dropped; over-long paragraphs are truncated
    to keep a chunk to a readable passage.
    """
    lines = text.splitlines()
    pairs: List[tuple[Optional[str], str]] = []
    for heading, start, end in split_markdown_sections(text):
        body = lines[start - 1:end]
        para: List[str] = []
        for raw in body:
            stripped = raw.strip()
            if not stripped:
                if para:
                    pairs.append((heading, " ".join(para)))
                    para = []
                continue
            # Drop list/quote markup so the chunk reads as prose.
            stripped = stripped.lstrip("-*>").strip()
            if stripped:
                para.append(stripped)
        if para:
            pairs.append((heading, " ".join(para)))

    cleaned: List[tuple[Optional[str], str]] = []
    for heading, body_text in pairs:
        body_text = body_text.strip()
        if not body_text:
            continue
        if len(body_text) > _MAX_CHUNK_LEN:
            body_text = body_text[:_MAX_CHUNK_LEN].rstrip() + "..."
        cleaned.append((heading, body_text))
    return cleaned


class KnowledgeLibrary:
    """Owns imported sources and chunks, persisted as one JSONL file."""

    def __init__(self, path: Optional[str | Path] = None, *,
                 load_existing: bool = True):
        self.path = Path(path) if path else None
        self._sources: Dict[str, KnowledgeSource] = {}
        self._documents: Dict[str, KnowledgeDocument] = {}
        self._chunks: Dict[str, KnowledgeChunk] = {}
        if load_existing and self.path and self.path.exists():
            self.load()

    # -- mutation --

    def add_source(self, source: KnowledgeSource) -> KnowledgeSource:
        self._sources[source.source_id] = source
        self._flush()
        return source

    def import_text_file(self, path: str | Path, *,
                         domain: KnowledgeDomain,
                         authority: SourceAuthority,
                         source_name: str,
                         version: Optional[str] = None,
                         licence: Optional[str] = None,
                         staleness_policy: str = "static") -> KnowledgeSource:
        """Import one text/Markdown file as a source with chunked passages.

        Returns the created :class:`KnowledgeSource`. The file is read, split
        into provenance-tagged chunks, and stored; nothing is written to the
        memory ledger.
        """
        path = Path(path)
        text = load_text_file(path)  # raises FileNotFoundError if missing
        source_path = str(path)
        source_id = "src-" + _digest(source_name, source_path)
        document_id = "doc-" + _digest(source_id, source_path)

        source = KnowledgeSource(
            source_id=source_id,
            source_name=source_name,
            source_type="markdown" if path.suffix.lower() in {".md", ".markdown"}
            else "text",
            domain=domain,
            authority=authority,
            path_or_url=source_path,
            staleness_policy=staleness_policy,
            licence=licence,
            version=version,
        )
        document = KnowledgeDocument(
            document_id=document_id,
            source_id=source_id,
            title=source_name,
            path_or_url=source_path,
        )
        self._sources[source_id] = source
        self._documents[document_id] = document

        for idx, (heading, body_text) in enumerate(_chunk_paragraphs(text)):
            chunk_id = "chk-" + _digest(source_id, str(idx), body_text)
            self._chunks[chunk_id] = KnowledgeChunk(
                chunk_id=chunk_id,
                source_id=source_id,
                document_id=document_id,
                chunk_text=body_text,
                domain=domain,
                authority=authority,
                source_name=source_name,
                source_section=heading,
            )
        self._flush()
        return source

    def delete_source(self, source_id: str) -> bool:
        """Deactivate a source and all of its chunks. Returns True if found."""
        source = self._sources.get(source_id)
        if source is None or not source.active:
            return False
        source.active = False
        for chunk in self._chunks.values():
            if chunk.source_id == source_id:
                chunk.active = False
        self._flush()
        return True

    # -- read --

    def list_sources(self, *, active_only: bool = True) -> List[KnowledgeSource]:
        sources = self._sources.values()
        if active_only:
            sources = [s for s in sources if s.active]
        return list(sources)

    def list_chunks(self, *, source_id: Optional[str] = None,
                    active_only: bool = True) -> List[KnowledgeChunk]:
        chunks = list(self._chunks.values())
        if source_id is not None:
            chunks = [c for c in chunks if c.source_id == source_id]
        if active_only:
            chunks = [c for c in chunks if c.active]
        return chunks

    def get_source(self, source_id: str) -> Optional[KnowledgeSource]:
        return self._sources.get(source_id)

    def export_library(self) -> dict:
        return {
            "sources": [s.to_json() and json.loads(s.to_json())
                        for s in self._sources.values()],
            "documents": [json.loads(d.to_json())
                          for d in self._documents.values()],
            "chunks": [json.loads(c.to_json())
                       for c in self._chunks.values()],
        }

    # -- persistence --

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for source in self._sources.values():
                fh.write(source.to_json() + "\n")
            for document in self._documents.values():
                fh.write(document.to_json() + "\n")
            for chunk in self._chunks.values():
                fh.write(chunk.to_json() + "\n")

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        self._sources.clear()
        self._documents.clear()
        self._chunks.clear()
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                kind = data.get("_record")
                if kind == "source":
                    src = KnowledgeSource.from_dict(data)
                    self._sources[src.source_id] = src
                elif kind == "document":
                    doc = KnowledgeDocument.from_dict(data)
                    self._documents[doc.document_id] = doc
                elif kind == "chunk":
                    chunk = KnowledgeChunk.from_dict(data)
                    self._chunks[chunk.chunk_id] = chunk
