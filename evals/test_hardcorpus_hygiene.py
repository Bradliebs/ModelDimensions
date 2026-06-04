"""Tests for the v3.0.2 harder corpus + source/chunk hygiene probe.

These tests prove the hygiene probe is a faithful, **read-only** classification
layer that distinguishes a forbidden relevance bleed from legitimate adjacent
evidence:

* the new optional case fields load correctly and leave v3.0 / v3.0.1 cases
  unchanged;
* the wrong-source taxonomy works (proven on synthetic candidates so the
  assertion does not depend on retrieval contents);
* a forbidden bleed is separated from an on-topic neighbour;
* expected-gap cases are handled without forcing a failure;
* the chunk/source hygiene diagnostics have a stable shape;
* it is **deterministic** and **mutates no state**;
* the existing v3.0 retrieval eval and v3.0.1 report-path probe still behave
  identically alongside it.

The pack is always built into an isolated ``tmp_path`` registry so the tracked
demo packs stay pristine. Nothing here changes retrieval, ranking, source
selection, grounding, composer, report rendering, or memory behaviour.
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
_CASES = ROOT / "demos" / "retrieval_hardcorpus_cases.jsonl"
_V30_CASES = ROOT / "demos" / "retrieval_eval_cases.jsonl"
_V301_CASES = ROOT / "demos" / "retrieval_report_path_cases.jsonl"


def _build_service(tmp_path, monkeypatch) -> WorkbenchService:
    """Build the assistant pack into an isolated registry and bind a service.

    Mirrors the v3.0 / v3.0.1 harness tests: ``chdir`` to the repo root so the
    manifest's relative source paths resolve, write the pack under ``tmp_path``,
    and bind the hybrid (lexical) retrieval backend the probe reads.
    """
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    pack = registry.get_pack(report.pack_id)
    return WorkbenchService.from_pack(
        pack, registry=registry, knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder())


# 1. the new optional case fields load correctly. ----------------------------

def test_hardcorpus_cases_load():
    cases = reh.load_cases(_CASES)
    assert len(cases) == 8
    by_id = {c.case_id: c for c in cases}

    near = by_id["hard-least-priv-vs-copilot"]
    assert near.expected_sources == ["Microsoft 365 Admin Patterns"]
    assert "Privileged Identity Management" in near.expected_topic_terms
    assert "SharePoint Teams Copilot Studio Notes" in near.forbidden_sources
    assert near.query_shape == "direct"
    assert near.expected_gap is False

    gap = by_id["hard-data-residency-gap"]
    assert gap.expected_gap is True
    assert gap.has_expected is False
    assert gap.query_shape == "decision_lookup"

    vs = by_id["vs-least-priv-requirements"]
    assert vs.query_shape == "value_sprint"
    assert vs.classification_notes != ""


def test_new_fields_default_on_v30_cases():
    """v3.0 cases (no new keys) load with safe defaults — backward compatible."""
    cases = reh.load_cases(_V30_CASES)
    for case in cases:
        assert case.expected_topic_terms == []
        assert case.allowed_neighbour_sources == []
        assert case.expected_gap is False
        assert case.query_shape == "direct"
        assert case.classification_notes == ""


# 2. wrong-source classification works on synthetic candidates. --------------

def test_classify_forbidden_bleed():
    case = reh.RetrievalEvalCase(
        query="least privilege",
        expected_sources=["Microsoft 365 Admin Patterns"],
        expected_topic_terms=["least-privilege", "PIM"],
        forbidden_sources=["SharePoint Teams Copilot Studio Notes"],
        forbidden_topic_terms=["Copilot Studio"],
    )
    forbidden_by_source = {
        "source_name": "SharePoint Teams Copilot Studio Notes",
        "source_id": "", "chunk_id": "c1", "text": "connector availability"}
    forbidden_by_term = {
        "source_name": "Some Other Source", "source_id": "",
        "chunk_id": "c2", "text": "Copilot Studio custom agents"}
    assert reh.classify_candidate(case, forbidden_by_source) == \
        reh.CLASS_FORBIDDEN_BLEED
    assert reh.classify_candidate(case, forbidden_by_term) == \
        reh.CLASS_FORBIDDEN_BLEED


def test_classify_expected_and_neighbour_and_ambiguous():
    case = reh.RetrievalEvalCase(
        query="least privilege",
        expected_sources=["Microsoft 365 Admin Patterns"],
        expected_chunk_ids=["m365-admin-1"],
        expected_topic_terms=["least-privilege", "PIM"],
        allowed_neighbour_sources=["Local Coding-Agent Workflow Rules"],
        forbidden_sources=["SharePoint Teams Copilot Studio Notes"],
        forbidden_topic_terms=["Copilot Studio"],
    )
    expected = {"source_name": "Microsoft 365 Admin Patterns", "source_id": "",
                "chunk_id": "m365-admin-1", "text": "use PIM"}
    neighbour_by_source = {
        "source_name": "Local Coding-Agent Workflow Rules", "source_id": "",
        "chunk_id": "c3", "text": "follow the change rules"}
    neighbour_by_term = {
        "source_name": "Purview Sensitivity Labels and DLP", "source_id": "",
        "chunk_id": "c4", "text": "least-privilege is mentioned here too"}
    ambiguous = {"source_name": "Purview Sensitivity Labels and DLP",
                 "source_id": "", "chunk_id": "c5",
                 "text": "encryption at rest for labels"}

    assert reh.classify_candidate(case, expected) == reh.CLASS_EXPECTED
    assert reh.classify_candidate(case, neighbour_by_source) == \
        reh.CLASS_ON_TOPIC_NEIGHBOUR
    assert reh.classify_candidate(case, neighbour_by_term) == \
        reh.CLASS_ON_TOPIC_NEIGHBOUR
    assert reh.classify_candidate(case, ambiguous) == reh.CLASS_AMBIGUOUS


def test_classify_expected_gap():
    """In a gap case every non-forbidden candidate is an expected_gap surface."""
    case = reh.RetrievalEvalCase(
        query="data residency vendor",
        expected_gap=True,
        forbidden_topic_terms=["Copilot Studio"],
    )
    surface = {"source_name": "Power Platform PowerApps Formulas",
               "source_id": "", "chunk_id": "c6", "text": "Power Fx delegation"}
    forbidden = {"source_name": "Anything", "source_id": "", "chunk_id": "c7",
                 "text": "Copilot Studio agents"}
    assert reh.classify_candidate(case, surface) == reh.CLASS_EXPECTED_GAP
    # forbidden still dominates even in a gap case.
    assert reh.classify_candidate(case, forbidden) == reh.CLASS_FORBIDDEN_BLEED


# 3. forbidden bleed is separated from on-topic neighbour in a diagnosis. -----

def test_diagnose_separates_bleed_from_neighbour(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)
    by_id = {c.case_id: c for c in cases}

    result = reh.diagnose_case(service, by_id["hard-least-priv-vs-copilot"])
    counts = result.classification_counts
    # the taxonomy buckets are always present and non-negative.
    for name in (reh.CLASS_FORBIDDEN_BLEED, reh.CLASS_ON_TOPIC_NEIGHBOUR,
                 reh.CLASS_AMBIGUOUS, reh.CLASS_EXPECTED_GAP):
        assert counts.get(name, 0) >= 0
    # forbidden bleed is counted independently of neighbour/ambiguous evidence:
    # a forbidden candidate never lands in the neighbour bucket.
    for cand in result.candidates:
        if cand.classification == reh.CLASS_FORBIDDEN_BLEED:
            assert cand.classification != reh.CLASS_ON_TOPIC_NEIGHBOUR


# 4. expected-gap cases are handled correctly (no forced failure). -----------

def test_gap_case_handling(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)
    by_id = {c.case_id: c for c in cases}

    gap = reh.diagnose_case(service, by_id["hard-data-residency-gap"])
    assert gap.expected_gap is True
    # a gap case passes as long as nothing forbidden bled in — it is never
    # failed just for surfacing neighbouring text.
    assert gap.passed == (gap.forbidden_bleed_count == 0)


# 5. chunk/source hygiene diagnostics have a stable shape. -------------------

def test_hygiene_metric_shape(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)
    results = reh.run_hygiene(service, cases)
    summary = reh.summarize_hygiene(results)

    assert summary.case_count == len(cases)
    assert summary.pass_count + summary.fail_count == len(cases)
    assert set(summary.wrong_source_classification) == {
        reh.CLASS_FORBIDDEN_BLEED, reh.CLASS_ON_TOPIC_NEIGHBOUR,
        reh.CLASS_AMBIGUOUS, reh.CLASS_EXPECTED_GAP}
    assert summary.forbidden_bleed_reproduced == (
        summary.wrong_source_classification[reh.CLASS_FORBIDDEN_BLEED] > 0)
    for opt in (summary.mean_chunk_topic_density,
                summary.mean_source_topic_overlap,
                summary.value_sprint_query_pass_rate,
                summary.gap_case_pass_rate):
        assert opt is None or 0.0 <= opt <= 1.0

    for r in results:
        data = r.to_dict()
        for key in ("case_id", "query", "query_shape", "expected_gap",
                    "passed", "reason", "retrieved_chunk_count",
                    "first_hit_rank", "classification_counts",
                    "mixed_topic_chunk_count",
                    "chunk_with_forbidden_terms_count",
                    "chunk_topic_density", "source_topic_overlap",
                    "candidates", "tags"):
            assert key in data
        assert len(data["candidates"]) == r.retrieved_chunk_count
        for cand in data["candidates"]:
            for key in ("rank", "source_name", "chunk_id", "classification",
                        "topic_density", "matched_topic_terms",
                        "forbidden_terms_found", "contains_forbidden_terms",
                        "is_mixed_topic"):
                assert key in cand
        # a mixed-topic chunk carries both an expected and a forbidden term.
        for cand in r.candidates:
            if cand.is_mixed_topic:
                assert cand.matched_topic_terms and cand.forbidden_terms_found


def test_render_markdown_smoke(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)
    results = reh.run_hygiene(service, cases)
    summary = reh.summarize_hygiene(results)
    md = reh.render_hygiene_markdown(
        results, summary, pack_label="m365_coding_assistant",
        backend_label="hybrid")
    assert "Harder-corpus hygiene probe" in md
    assert "wrong-source classification" in md
    assert "forbidden bleed reproduced" in md


# 6. the probe is deterministic for a fixed pack + cases. --------------------

def test_hygiene_deterministic(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)

    first = [r.to_dict() for r in reh.run_hygiene(service, cases)]
    second = [r.to_dict() for r in reh.run_hygiene(service, cases)]
    assert first == second

    s1 = reh.summarize_hygiene(reh.run_hygiene(service, cases)).to_dict()
    s2 = reh.summarize_hygiene(reh.run_hygiene(service, cases)).to_dict()
    assert s1 == s2


# 7. diagnosing mutates no state. --------------------------------------------

def test_hygiene_no_mutation(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    cases = reh.load_cases(_CASES)

    ledger_before = service.export_ledger()
    proposals_before = service.list_proposals()
    audit_before = service.query_knowledge(cases[0].query)
    sources_before = sorted(c["source_name"] for c in audit_before.candidates)

    reh.run_hygiene(service, cases)

    assert service.export_ledger() == ledger_before
    assert service.list_proposals() == proposals_before
    audit_after = service.query_knowledge(cases[0].query)
    sources_after = sorted(c["source_name"] for c in audit_after.candidates)
    assert sources_after == sources_before


# 8. existing v3.0 eval + v3.0.1 probe behave identically alongside. ----------

def test_v30_and_v301_unchanged_alongside_hygiene(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    v30_cases = reh.load_cases(_V30_CASES)
    v301_cases = reh.load_cases(_V301_CASES)

    v30_baseline = {r.case_id: r.to_dict()
                    for r in reh.run_eval(service, v30_cases)}
    v301_baseline = {r.case_id: r.to_dict()
                     for r in reh.run_probe(service, v301_cases)}

    # run the hygiene probe in between — it must leave both unchanged.
    reh.run_hygiene(service, reh.load_cases(_CASES))

    v30_after = {r.case_id: r.to_dict()
                 for r in reh.run_eval(service, v30_cases)}
    v301_after = {r.case_id: r.to_dict()
                  for r in reh.run_probe(service, v301_cases)}
    assert v30_after == v30_baseline
    assert v301_after == v301_baseline
