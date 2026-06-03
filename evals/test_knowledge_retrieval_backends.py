"""Tests for v1.4 pluggable knowledge retrieval backends.

These cover the backend seam: the deterministic backend stays the default; a
semantic backend can be mocked (no model download); retrieval preserves source
metadata and records which backend produced it; deleted-source chunks are never
retrieved; imported knowledge is never written to the memory ledger; ``query_all``
still separates ``memory_used`` from ``knowledge_used``; and a requested-but-
unavailable semantic backend fails gracefully (falls back to deterministic).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import retrieval_backends  # noqa: E402
from agent.knowledge_sources import (  # noqa: E402
    KnowledgeChunk,
    KnowledgeDomain,
    SourceAuthority,
)
from agent.retrieval_backends import (  # noqa: E402
    BackendUnavailableError,
    DeterministicRetrievalBackend,
    SemanticRetrievalBackend,
    make_backend,
    resolve_backend_name,
)
from agent.workbench_service import WorkbenchService  # noqa: E402

_DELIVERY = "the supplier delivery is on Friday afternoon"

_CODING_DOC = (
    "# Pydantic\n\n"
    "Pydantic validates data against a typed model when the model is "
    "constructed and raises a ValidationError on bad input.\n\n"
    "# JSONL\n\n"
    "JSONL stores one JSON object per line for append-only ledgers.\n"
)


class _FakeEmbedder:
    """Offline bag-of-words embedder so paraphrases share direction.

    Tokens hash into a fixed-dimension vector, so two texts with overlapping
    words get a positive cosine similarity. This stands in for MiniLM in tests
    without any download.
    """

    dim = 64

    def encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in text.lower().split():
            vec[hash(tok) % self.dim] += 1.0
        return vec


def _service(tmp_path, **kwargs) -> WorkbenchService:
    return WorkbenchService(
        ledger_path=str(tmp_path / "ledger.jsonl"),
        queue_path=str(tmp_path / "queue.jsonl"),
        knowledge_path=str(tmp_path / "knowledge.jsonl"),
        **kwargs,
    )


def _write_doc(tmp_path, text: str = _CODING_DOC, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _chunks() -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            chunk_id="chk-1", source_id="src-1", document_id="doc-1",
            chunk_text="Pydantic validates a typed model on construction.",
            domain=KnowledgeDomain.CODING, authority=SourceAuthority.OFFICIAL,
            source_name="Pydantic Notes", source_section="Pydantic"),
        KnowledgeChunk(
            chunk_id="chk-2", source_id="src-1", document_id="doc-1",
            chunk_text="JSONL stores one JSON object per line.",
            domain=KnowledgeDomain.CODING, authority=SourceAuthority.OFFICIAL,
            source_name="Pydantic Notes", source_section="JSONL"),
    ]


# 1. deterministic backend remains the default. -----------------------------

def test_deterministic_backend_is_default(tmp_path, monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_RETRIEVAL_BACKEND", raising=False)
    assert resolve_backend_name() == "deterministic"

    service = _service(tmp_path)
    assert service.knowledge_backend_name() == "deterministic"

    backend = make_backend(None, deterministic_encoder=service.encoder)
    assert isinstance(backend, DeterministicRetrievalBackend)
    assert backend.is_semantic is False


# 2. semantic backend can be mocked (no download). --------------------------

def test_semantic_backend_can_be_mocked():
    backend = SemanticRetrievalBackend(embedder=_FakeEmbedder())
    backend.index_chunks(_chunks())

    results = backend.retrieve("how does pydantic validate a model", top_k=2)
    assert results
    assert backend.is_semantic is True
    assert backend.backend_name == "semantic"
    assert all(c.is_semantic for c in results)
    assert all(c.backend_name == "semantic" for c in results)
    # The Pydantic chunk should outrank the JSONL chunk on word overlap.
    assert results[0].source_section == "Pydantic"


# 3. retrieval preserves source metadata. -----------------------------------

def test_retrieval_preserves_source_metadata(tmp_path):
    service = _service(tmp_path, knowledge_backend="semantic",
                       semantic_embedder=_FakeEmbedder())
    doc = _write_doc(tmp_path)
    service.import_knowledge(doc, domain="coding", authority="official",
                             source_name="Pydantic Notes", version="2.x")

    audit = service.query_knowledge("how does pydantic validate a model")
    assert audit.candidates
    cand = audit.candidates[0]
    assert cand["source_name"] == "Pydantic Notes"
    assert cand["domain"] == "coding"
    assert cand["authority"] == "official"
    assert cand["version"] == "2.x"
    assert cand["backend_name"] == "semantic"
    assert cand["is_semantic"] is True


# 4. deleted source chunks are not retrieved. -------------------------------

def test_deleted_source_chunks_not_retrieved(tmp_path):
    service = _service(tmp_path)
    doc = _write_doc(tmp_path)
    source = service.import_knowledge(
        doc, domain="coding", authority="official",
        source_name="Pydantic Notes", version="2.x")

    assert service.delete_knowledge_source(source.source_id) is True
    audit = service.query_knowledge(
        "Pydantic validates a typed model when constructed")
    assert audit.knowledge_used is False
    assert audit.candidates == []
    assert source.source_id not in audit.cited_source_ids


# 5. knowledge chunks are not written to the MemoryLedger. ------------------

def test_knowledge_not_written_to_ledger(tmp_path):
    service = _service(tmp_path)
    doc = _write_doc(tmp_path)
    service.import_knowledge(doc, domain="coding", authority="official",
                             source_name="Pydantic Notes", version="2.x")
    service.query_knowledge("Pydantic validates a typed model")

    assert service.export_ledger() == []


# 6. query_all still distinguishes memory_used and knowledge_used. ----------

def test_query_all_distinguishes_memory_and_knowledge(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_DELIVERY, source="user")
    doc = _write_doc(tmp_path)
    service.import_knowledge(doc, domain="coding", authority="official",
                             source_name="Pydantic Notes", version="2.x")

    audit = service.query_all("what did we decide about " + _DELIVERY)
    assert audit.memory_used is True
    assert audit.knowledge_used is False
    assert audit.route == "memory_only"


# 7. unavailable semantic backend fails gracefully. -------------------------

def test_unavailable_semantic_backend_fails_gracefully(tmp_path, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise BackendUnavailableError("no model in test")

    monkeypatch.setattr(retrieval_backends, "_load_minilm", _boom)

    # Direct construction surfaces the error...
    import pytest
    with pytest.raises(BackendUnavailableError):
        make_backend("semantic",
                     deterministic_encoder=_FakeEmbedder())

    # ...but the service falls back to deterministic without raising.
    service = _service(tmp_path, knowledge_backend="semantic")
    assert service.knowledge_backend_name() == "deterministic"


# 8. backend name is included in the audit. ---------------------------------

def test_backend_name_in_audit(tmp_path):
    service = _service(tmp_path)
    doc = _write_doc(tmp_path)
    service.import_knowledge(doc, domain="coding", authority="official",
                             source_name="Pydantic Notes", version="2.x")

    k_audit = service.query_knowledge("Pydantic validates a typed model")
    assert k_audit.backend_name == "deterministic"

    c_audit = service.query_all("how do I code with Pydantic")
    assert c_audit.knowledge_backend == "deterministic"
