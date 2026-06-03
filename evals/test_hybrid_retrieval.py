"""Tests for the v2.2 opt-in hybrid knowledge retrieval backend.

The hybrid backend composes the frozen deterministic backend with an optional
semantic component and blends a keyword-overlap signal, an optional semantic
score, and a gentle authority weight. These tests prove the v2.2 contract:

* paraphrase recall improves over the whole-string-hash deterministic backend;
* an exact keyword match still outranks a paraphrase;
* with no semantic embedder the backend falls back safely (keyword + geometry);
* the retrieval route decision is exposed in the audit (scores + rejections);
* safety is unchanged and backend-independent: deleted chunks are never
  returned, stale sources are *labelled* (not silently promoted), the memory
  near-miss verifier still refuses Friday/Monday-style flips, a superseded
  memory is never current, and AnswerGuard still runs after composition.

The semantic component is exercised with a deterministic offline token-hashing
embedder, so the whole suite stays offline and reproducible — no downloads.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.knowledge_sources import (  # noqa: E402
    KnowledgeChunk,
    KnowledgeDomain,
    SourceAuthority,
)
from agent.orchestrator import DeterministicEncoder  # noqa: E402
from agent.retrieval_backends import DeterministicRetrievalBackend  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402
from retrieval.hybrid_backend import HybridRetrievalBackend  # noqa: E402
from slm.assistant_composer import ComposerMode  # noqa: E402

_FRIDAY = "the supplier delivery is on Friday afternoon"
_MONDAY = "the supplier delivery is on Monday afternoon"

_CODING_DOC = (
    "# Rename\n\n"
    "Use the rename method with the columns argument to change a column label "
    "in a pandas DataFrame.\n"
)


# ---------- helpers ---------------------------------------------------------

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


def _chunk(chunk_id: str, text: str, *,
           authority: SourceAuthority = SourceAuthority.OFFICIAL,
           active: bool = True) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id, source_id="src-1", document_id="doc-1",
        chunk_text=text, domain=KnowledgeDomain.CODING, authority=authority,
        source_name="Pandas Notes", source_section=chunk_id, active=active)


def _para_chunks() -> list[KnowledgeChunk]:
    return [
        _chunk(
            "rename",
            "Use the rename method with the columns argument to change a "
            "column label in a pandas DataFrame."),
        _chunk(
            "bytes",
            "Open a file in binary mode and read the raw bytes into a buffer."),
        _chunk(
            "sort",
            "Sort a list of integers in ascending order with the sorted "
            "builtin function."),
    ]


def _rank_of(candidates, chunk_id: str):
    for cand in candidates:
        if cand.chunk_id == chunk_id:
            return cand.rank
    return None


# ---------- 1. paraphrase recall improves over deterministic ---------------

def test_paraphrase_recall_improves_over_deterministic():
    chunks = _para_chunks()
    query = "How can I rename a pandas DataFrame column?"
    enc = DeterministicEncoder()

    hybrid = HybridRetrievalBackend(
        enc, semantic_embedder=OfflineHashingEmbedder())
    hybrid.index_chunks(chunks)
    hybrid_results = hybrid.retrieve(query, top_k=3)

    det = DeterministicRetrievalBackend(enc)
    det.index_chunks(chunks)
    det_results = det.retrieve(query, top_k=3)

    # The relevant chunk shares every meaningful query token, so the hybrid
    # backend ranks it first; the whole-string-hash deterministic backend does
    # not (its rank for the paraphrase is strictly worse).
    assert _rank_of(hybrid_results, "rename") == 1
    assert _rank_of(det_results, "rename") != 1


# ---------- 2. exact keyword match still wins ------------------------------

def test_exact_keyword_match_outranks_paraphrase():
    chunks = [
        _chunk("exact", "To rename a pandas dataframe column use df.rename."),
        _chunk("para", "Change the label of a column in a pandas dataframe."),
    ]
    query = "rename a pandas dataframe column"
    hybrid = HybridRetrievalBackend(
        DeterministicEncoder(), semantic_embedder=OfflineHashingEmbedder())
    hybrid.index_chunks(chunks)
    results = hybrid.retrieve(query, top_k=2)

    assert results[0].chunk_id == "exact"
    selected = {s["chunk_id"]: s for s in hybrid.last_report.selected}
    assert selected["exact"]["exact_match"] is True
    assert selected["para"]["exact_match"] is False
    assert selected["exact"]["combined_score"] > selected["para"]["combined_score"]


# ---------- 3. safe fallback when no semantic backend ----------------------

def test_fallback_when_no_semantic_embedder():
    chunks = _para_chunks()
    hybrid = HybridRetrievalBackend(DeterministicEncoder())  # no embedder
    hybrid.index_chunks(chunks)
    results = hybrid.retrieve("rename a pandas column", top_k=3)

    assert hybrid.is_semantic is False
    assert results  # still returns ranked candidates
    assert all(c.backend_name == "hybrid" for c in results)
    assert all(c.is_semantic is False for c in results)
    # Keyword signal alone still surfaces the relevant chunk first.
    assert results[0].chunk_id == "rename"
    assert hybrid.last_report.semantic_available is False
    # No semantic score is fabricated when the component is absent.
    assert hybrid.last_report.selected[0]["semantic_score"] is None


def test_failing_semantic_embedder_degrades_safely():
    class _BoomEmbedder:
        def encode_one(self, text):  # noqa: D401, ANN001
            raise RuntimeError("model unavailable")

    # An injected embedder that throws at index time must not crash a query: the
    # hybrid backend degrades to keyword + geometry and stops claiming semantics.
    hybrid = HybridRetrievalBackend(
        DeterministicEncoder(), semantic_embedder=_BoomEmbedder())
    hybrid.index_chunks(_para_chunks())
    results = hybrid.retrieve("rename a pandas column", top_k=3)

    assert hybrid.backend_name == "hybrid"
    assert hybrid.is_semantic is False
    assert results
    assert results[0].chunk_id == "rename"
    assert hybrid.last_report.semantic_available is False


# ---------- 4. rejected candidates are recorded ----------------------------

def test_report_records_rejected_candidates():
    chunks = _para_chunks()
    hybrid = HybridRetrievalBackend(
        DeterministicEncoder(), semantic_embedder=OfflineHashingEmbedder())
    hybrid.index_chunks(chunks)
    hybrid.retrieve("rename a pandas column", top_k=1)

    report = hybrid.last_report
    assert len(report.selected) == 1
    assert report.rejected  # the other two chunks are recorded as rejected
    assert all(r["reason"] == "below_top_k" for r in report.rejected)


# ---------- 5. inactive chunks are never returned --------------------------

def test_inactive_chunk_never_returned():
    chunks = _para_chunks()
    chunks.append(_chunk(
        "deleted", "Rename a pandas column quickly.", active=False))
    hybrid = HybridRetrievalBackend(
        DeterministicEncoder(), semantic_embedder=OfflineHashingEmbedder())
    hybrid.index_chunks(chunks)
    results = hybrid.retrieve("rename a pandas column", top_k=10)

    assert all(c.chunk_id != "deleted" for c in results)
    reasons = {r["chunk_id"]: r["reason"] for r in hybrid.last_report.rejected}
    assert reasons.get("deleted") == "inactive_source"


# ---------- 6. service wires the hybrid backend by flag --------------------

def test_service_uses_hybrid_backend(tmp_path):
    service = _service(tmp_path, knowledge_backend="hybrid",
                       semantic_embedder=OfflineHashingEmbedder())
    assert service.knowledge_backend_name() == "hybrid"


def test_query_knowledge_includes_retrieval_audit(tmp_path):
    service = _service(tmp_path, knowledge_backend="hybrid",
                       semantic_embedder=OfflineHashingEmbedder())
    doc = _write_doc(tmp_path)
    service.import_knowledge(doc, domain="coding", authority="official",
                             source_name="Pandas Notes", version="2.x")

    audit = service.query_knowledge("how do I rename a pandas column")
    assert audit.backend_name == "hybrid"
    assert audit.retrieval is not None
    assert audit.retrieval["backend_name"] == "hybrid"
    assert audit.candidates
    cand = audit.candidates[0]
    assert "keyword_score" in cand
    assert "combined_score" in cand
    assert "exact_match" in cand


def test_answer_query_audit_shows_hybrid_and_guard(tmp_path):
    service = _service(tmp_path, knowledge_backend="hybrid",
                       semantic_embedder=OfflineHashingEmbedder())
    doc = _write_doc(tmp_path)
    service.import_knowledge(doc, domain="coding", authority="official",
                             source_name="Pandas Notes", version="2.x")

    result = service.answer_query("how do I rename a pandas column")
    assert result.audit["knowledge_backend"] == "hybrid"
    assert result.audit["knowledge"]["backend_name"] == "hybrid"
    assert result.audit["knowledge"]["retrieval"] is not None
    # AnswerGuard still runs after composition.
    assert "guard" in result.audit
    assert result.audit["guard"]["verdict"] in {"ACCEPT", "REJECT"}


# ---------- 7. stale source is labelled, not silently promoted -------------

def test_stale_source_labelled_not_dropped(tmp_path):
    service = _service(tmp_path, knowledge_backend="hybrid",
                       semantic_embedder=OfflineHashingEmbedder())
    doc = _write_doc(tmp_path)
    # The service wrapper defaults to "static"; set a stale policy directly so
    # the labelling path (backend-independent) engages.
    service.knowledge.import_text_file(
        doc, domain=KnowledgeDomain.CODING, authority=SourceAuthority.OFFICIAL,
        source_name="Pandas Notes", version="2.x", staleness_policy="stale")

    audit = service.query_knowledge("how do I rename a pandas column")
    assert audit.knowledge_used is True  # not dropped
    assert any("stale" in c.lower() for c in audit.cautions)  # labelled


# ---------- 8. deleted source is not citeable with hybrid ------------------

def test_deleted_source_not_citeable_with_hybrid(tmp_path):
    service = _service(tmp_path, knowledge_backend="hybrid",
                       semantic_embedder=OfflineHashingEmbedder())
    doc = _write_doc(tmp_path)
    source = service.import_knowledge(
        doc, domain="coding", authority="official",
        source_name="Pandas Notes", version="2.x")
    assert service.delete_knowledge_source(source.source_id) is True

    audit = service.query_knowledge("how do I rename a pandas column")
    assert audit.knowledge_used is False
    assert audit.candidates == []
    assert source.source_id not in audit.cited_source_ids


# ---------- 9. memory near-miss still refused with hybrid enabled ----------

def test_friday_monday_near_miss_refused_with_hybrid(tmp_path):
    service = _service(tmp_path, knowledge_backend="hybrid",
                       semantic_embedder=OfflineHashingEmbedder())
    service.add_memory(_FRIDAY, source="seed")

    # The memory verifier is on a separate path from knowledge retrieval, so a
    # Monday-asserting query against a Friday memory must never be grounded as
    # current fact (it surfaces as a refusal or an explicit conflict).
    package = service.build_grounding_package(_MONDAY)
    assert package.mode != ComposerMode.GROUNDED


# ---------- 10. superseded memory is not current with hybrid ---------------

def test_superseded_memory_not_current_with_hybrid(tmp_path):
    service = _service(tmp_path, knowledge_backend="hybrid",
                       semantic_embedder=OfflineHashingEmbedder())
    old = service.add_memory(_FRIDAY, source="seed")
    note = tmp_path / "note.md"
    note.write_text(f"# Facts\n\n- {_MONDAY}\n", encoding="utf-8")
    service.import_notes(note)
    proposal = next(
        p for p in service.list_proposals(status=None)
        if p.canonical_text.lower().rstrip(".") == _MONDAY.lower().rstrip("."))
    service.approve_proposal_superseding(proposal.proposal_id, old.memory_id)
    service.write_approved_proposals()

    # The old Friday memory is superseded -> it can no longer be cited as the
    # current answer (knowledge backend choice does not affect this).
    package = service.build_grounding_package(_FRIDAY)
    assert package.mode != ComposerMode.GROUNDED
