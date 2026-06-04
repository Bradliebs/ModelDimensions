"""Tests for the v3.0 retrieval evaluation harness (measurement only).

These tests prove the harness is a faithful, **read-only** measurement layer:

* it scores cases against the frozen ``query_knowledge`` retrieval path and
  reports the documented metric shape;
* it is **deterministic** — the same pack and cases yield byte-identical
  results across two runs;
* it **mutates no state** — the MemoryLedger, the proposal queue, and the
  knowledge library are unchanged after a full evaluation.

The pack is always built into an isolated ``tmp_path`` registry so the tracked
demo packs stay pristine. Nothing here changes retrieval, ranking, source
selection, grounding, composer, or memory behaviour — that is the whole point of
the slice.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from agent import pack_builder  # noqa: E402
from agent import retrieval_eval_harness as reh  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402

_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_CASES = ROOT / "demos" / "retrieval_eval_cases.jsonl"


def _build_service(tmp_path, monkeypatch) -> WorkbenchService:
    """Build the assistant pack into an isolated registry and bind a service.

    Mirrors the value-sprint build: ``chdir`` to the repo root so the manifest's
    relative source paths resolve, write the pack under ``tmp_path`` so the test
    leaves no trace, and bind the hybrid (lexical) retrieval backend — the path
    the retrieval eval actually measures.
    """
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    pack = registry.get_pack(report.pack_id)
    return WorkbenchService.from_pack(
        pack, registry=registry, knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder())


# 1. the bundled case dataset loads with the documented shape. ---------------

def test_cases_load():
    cases = reh.load_cases(_CASES)
    assert len(cases) == 5
    by_id = {c.case_id: c for c in cases}
    assert "least-privilege-pim" in by_id
    bleed = by_id["least-privilege-pim"]
    assert bleed.expected_sources == ["Microsoft 365 Admin Patterns"]
    assert "Copilot Studio" in bleed.forbidden_topic_terms
    # the gap probe declares nothing to find.
    assert by_id["data-residency-gap"].has_expected is False


# 2. running the harness reports the documented per-case + summary metrics. --

def test_metric_shape(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)
    results = reh.run_eval(service, cases)
    summary = reh.summarize(results)

    assert summary.query_count == len(cases)
    assert summary.pass_count + summary.fail_count == summary.query_count
    assert summary.scored_case_count + summary.gap_probe_count == len(cases)
    # off-topic rate is a fraction; recall/hit are present for scored cases.
    assert 0.0 <= summary.off_topic_inclusion_rate <= 1.0
    assert summary.hit_at_1 is not None
    assert summary.expected_source_recall is not None

    for r in results:
        data = r.to_dict()
        for key in ("hit_at_1", "hit_at_3", "hit_at_5",
                    "expected_source_recall", "wrong_source_rate",
                    "off_topic_inclusion_count", "retrieved_chunk_count",
                    "passed"):
            assert key in data
        assert r.retrieved_chunk_count >= 0
        # gap probe (no expected) has undefined hit/recall.
        if not r.expected_source_recall and r.first_hit_rank is None:
            assert r.hit_at_1 is None


# 3. the harness is deterministic for a fixed pack + cases. ------------------

def test_deterministic(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)

    first = [r.to_dict() for r in reh.run_eval(service, cases)]
    second = [r.to_dict() for r in reh.run_eval(service, cases)]
    assert first == second

    s1 = reh.summarize(reh.run_eval(service, cases)).to_dict()
    s2 = reh.summarize(reh.run_eval(service, cases)).to_dict()
    assert s1 == s2


# 4. evaluation mutates no state (no memory / proposal / knowledge writes). --

def test_no_mutation(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)

    ledger_before = service.export_ledger()
    proposals_before = service.list_proposals()
    audit_before = service.query_knowledge(cases[0].query)
    sources_before = sorted(c["source_name"] for c in audit_before.candidates)

    reh.run_eval(service, cases)

    assert service.export_ledger() == ledger_before
    assert service.list_proposals() == proposals_before
    audit_after = service.query_knowledge(cases[0].query)
    sources_after = sorted(c["source_name"] for c in audit_after.candidates)
    assert sources_after == sources_before


# 5. a known baseline expectation holds (imperfect retrieval is acceptable). --

def test_baseline_expectations(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)
    results = {r.case_id: r for r in reh.run_eval(service, cases)}

    # least-privilege query grounds in the admin-patterns source on the hybrid
    # backend; this is the relevance-bleed probe and currently shows no bleed.
    lp = results["least-privilege-pim"]
    assert lp.hit_at_3 is True
    assert lp.off_topic_inclusion_count == 0

    # the gap probe has no expected source and so no hit/recall to report.
    gap = results["data-residency-gap"]
    assert gap.hit_at_1 is None
    assert gap.expected_source_recall is None
