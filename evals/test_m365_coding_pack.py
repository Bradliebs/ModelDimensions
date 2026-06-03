"""Tests for the v2.1 M365 / Coding assistant domain pack.

This is the practical assistant pack built from ``packs/m365_coding_assistant/``
via ``scripts/build_m365_coding_pack.py``. These tests cover two layers:

* **Retrieval / provenance** — the pack builds from local Markdown with full
  provenance, both ``microsoft`` and ``coding`` domains survive, and every
  committed evaluation question passes through the frozen knowledge query path.
* **Assistant safety** — the optional composer layer refuses when evidence is
  missing, labels a stale source, labels a model-prior answer only when it is
  explicitly allowed, and an SLM backend can never cite a source that is not in
  the grounding package.

Nothing here touches concept-cell geometry, grounding policy, verifier rules, or
lifecycle semantics. The pack is always built into an isolated ``tmp_path``
registry so the tracked source files stay pristine.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import pack_builder, pack_evaluator  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from slm.assistant_composer import ComposerMode, TemplateComposer  # noqa: E402
from slm.local_slm_backend import MockSLMBackend  # noqa: E402

_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_EVAL = ROOT / "evals" / "m365_coding_pack_eval.jsonl"

_SOURCES = {
    "Microsoft 365 Admin Patterns",
    "Purview Sensitivity Labels and DLP",
    "Power Platform PowerApps Formulas",
    "SharePoint Teams Copilot Studio Notes",
    "Local Coding-Agent Workflow Rules",
}
_STALE_SOURCE = "Power Platform PowerApps Formulas"

# A decision-recall query routes to project memory only, so the knowledge
# library is not consulted. With an empty ledger nothing grounds it, which is
# the only way to exercise refusal / model-prior on a populated pack (the
# deterministic backend always returns a knowledge chunk when asked).
_UNGROUNDED = "What did we decide about the production database rollout plan?"


def _build(tmp_path, monkeypatch):
    """Build the v2.1 pack from local docs into an isolated registry.

    ``chdir`` to the repo root so the manifest's repo-relative source paths
    resolve, but write the built pack under ``tmp_path`` so the test leaves no
    trace on the tracked ``packs/built`` registry.
    """
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    return registry, report


def _service(tmp_path, monkeypatch):
    registry, report = _build(tmp_path, monkeypatch)
    return pack_builder.open_pack_service(registry, report.pack_id)


# 1. the pack builds all five local sources with no skips. ------------------

def test_pack_builds_all_sources(tmp_path, monkeypatch):
    _registry, report = _build(tmp_path, monkeypatch)
    assert report.source_count == 5
    assert not report.skipped
    names = {s["source_name"] for s in report.sources}
    assert names == _SOURCES


# 2. both microsoft and coding domains survive into the library. ------------

def test_both_domains_present(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    by_domain: dict[str, set[str]] = {}
    for src in service.list_knowledge_sources():
        by_domain.setdefault(src["domain"], set()).add(src["source_name"])
    assert by_domain.get("coding") == {"Local Coding-Agent Workflow Rules"}
    assert by_domain.get("microsoft") == _SOURCES - {
        "Local Coding-Agent Workflow Rules"}


# 3. the stale source keeps its stale staleness policy. ---------------------

def test_stale_source_metadata_preserved(tmp_path, monkeypatch):
    _registry, report = _build(tmp_path, monkeypatch)
    power = next(s for s in report.sources
                if s["source_name"] == _STALE_SOURCE)
    assert power["staleness_policy"] == "stale"
    assert power["version"] == "local-notes-2023"


# 4. every committed eval question passes (retrieval / provenance). ---------

def test_eval_questions_all_pass(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    results = pack_evaluator.evaluate_pack_file(service, _EVAL)
    assert len(results) >= 20, "expected at least 20 eval questions"
    failed = [r for r in results if not r.passed]
    assert not failed, "; ".join(f"{r.question_id}: {r.reason}" for r in failed)


# 5. missing evidence produces a refusal, not a hallucination. --------------

def test_missing_evidence_refuses(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    package = service.build_grounding_package(_UNGROUNDED)
    answer = TemplateComposer().compose(package)
    assert package.mode == ComposerMode.REFUSAL
    assert answer.refused is True
    assert answer.citations == []


# 6. a stale source is labelled at query time. ------------------------------

def test_stale_source_is_labelled(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    stale_chunk = next(
        c for c in service.knowledge.list_chunks(active_only=True)
        if c.source_name == _STALE_SOURCE)

    combined = service.query_all(stale_chunk.chunk_text)
    assert combined.knowledge_used is True
    assert any("marked stale" in c for c in combined.cautions), combined.cautions

    # the caution survives into the grounding package the composer renders.
    package = service.build_grounding_package(stale_chunk.chunk_text)
    assert any("marked stale" in c for c in package.cautions)


# 7. a model-prior answer is labelled only when explicitly allowed. ---------

def test_model_prior_labelled_only_when_allowed(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    default = service.build_grounding_package(_UNGROUNDED)
    assert default.mode == ComposerMode.REFUSAL
    assert default.refused is True

    labelled = service.build_grounding_package(
        _UNGROUNDED, allow_model_prior=True)
    assert labelled.mode == ComposerMode.MODEL_PRIOR_LABELLED
    assert labelled.refused is False


# 8. an SLM cannot cite a source absent from the grounding package. ---------

def test_slm_cannot_cite_absent_source(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    backend = MockSLMBackend("Use Purview labels [src:not-a-real-chunk].")
    result = service.answer_query(
        _UNGROUNDED, use_slm=True, slm_backend=backend)

    # The fabricated citation is rejected; the refused package cannot be turned
    # into a grounded, cited answer.
    assert result.refused is True
    assert result.evidence_ids == []
    assert "not-a-real-chunk" not in result.answer.text


# 9. imported knowledge is never written to the memory ledger. --------------

def test_knowledge_is_not_project_memory(tmp_path, monkeypatch):
    from agent.review_service import ReviewService

    registry, report = _build(tmp_path, monkeypatch)
    service = pack_builder.open_pack_service(registry, report.pack_id)
    review = ReviewService(service, registry)

    # Knowledge is populated...
    assert len(review.get_knowledge_table()) == 5
    # ...but a knowledge chunk routes to knowledge only, never to memory...
    chunk = next(c for c in service.knowledge.list_chunks(active_only=True)
                 if c.source_name == "Local Coding-Agent Workflow Rules")
    hit = service.query_all(chunk.chunk_text)
    assert hit.knowledge_used is True
    assert hit.memory_used is False
    # ...and importing knowledge creates no memory ledger entries.
    memory_table = review.get_memory_table()
    assert all(len(rows) == 0 for rows in memory_table.values()), memory_table
