"""Tests for the v4.0 Source Registry Foundation (read-only metadata).

The source registry records *metadata about sources* — authority, freshness,
ownership, review status, supersession, notes — without touching the source text
or any retrieval behaviour. These tests pin down that contract:

* valid entries load with the documented shape, and sparse entries default
  safely (only ``source_id`` is required);
* an invalid ``status`` or ``authority_level`` fails cleanly with ``ValueError``;
* supersession links can be represented and inconsistencies are *reported*, never
  silently repaired;
* an effective freshness status is **computed**, never stored — computing it
  mutates neither the entry nor the file;
* the registry is inert next to the rest of the system: the v3.0 retrieval eval
  baseline and the v2.7 report citation contract are unchanged when a registry is
  loaded alongside them, no MemoryLedger entry is written, and no knowledge
  source is mutated;
* the CLI ``list`` / ``inspect`` rendering is deterministic.

Nothing here changes retrieval, ranking, source selection, grounding, composer,
or memory behaviour — the registry is read-only metadata in this slice.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import source_registry as sr  # noqa: E402
from agent import pack_builder  # noqa: E402
from agent import retrieval_eval_harness as reh  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402
from slm.assistant_composer import (  # noqa: E402
    ComposerMode,
    ConsultantReportComposer,
    EvidenceItem,
    GroundingPackage,
    TemplateComposer,
)

_DEMO_REGISTRY = ROOT / "demos" / "source_registry.jsonl"
_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"
_CASES = ROOT / "demos" / "retrieval_eval_cases.jsonl"

# A fixed "now" so every freshness computation in the suite is deterministic.
_NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)


def _build_service(tmp_path, monkeypatch) -> WorkbenchService:
    """Build the assistant pack into an isolated registry (mirrors other suites)."""
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    pack = registry.get_pack(report.pack_id)
    return WorkbenchService.from_pack(
        pack, registry=registry, knowledge_backend="hybrid",
        semantic_embedder=OfflineHashingEmbedder())


def _grounded_package() -> GroundingPackage:
    return GroundingPackage(
        query="how does pydantic validate input",
        mode=ComposerMode.GROUNDED,
        memory_used=False,
        knowledge_used=True,
        model_prior_used=False,
        informational_only=False,
        refused=False,
        route="knowledge",
        evidence=[
            EvidenceItem(citation_id="src:a", kind="knowledge",
                         text="Pydantic validates a typed model on construction.",
                         source_name="doc-a"),
            EvidenceItem(citation_id="src:b", kind="knowledge",
                         text="FastAPI uses Pydantic models to validate bodies.",
                         source_name="doc-b"),
        ],
    )


# 1. the demo registry loads valid entries with the documented shape. --------

def test_registry_loads_valid_entries():
    entries = sr.load_registry(_DEMO_REGISTRY)
    assert len(entries) == 6
    by_id = sr.index_by_id(entries)
    assert "src:least-privilege" in by_id
    lp = by_id["src:least-privilege"]
    assert lp.authority_level is sr.AuthorityLevel.OFFICIAL
    assert lp.status is sr.SourceStatus.ACTIVE
    assert "pim" in lp.topics
    assert lp.linked_decisions == ["dec:pim-rollout"]


# 2. missing optional fields default safely (only source_id is required). ----

def test_missing_optional_fields_default_safely():
    entry = sr.SourceRegistryEntry.from_dict({"source_id": "src:bare"})
    assert entry.source_id == "src:bare"
    assert entry.title == ""
    assert entry.source_type == ""
    assert entry.authority_level is sr.AuthorityLevel.UNKNOWN
    assert entry.owner == ""
    assert entry.created_at is None
    assert entry.last_reviewed_at is None
    assert entry.freshness_policy == "static"
    assert entry.stale_after_days is None
    assert entry.topics == []
    assert entry.linked_decisions == []
    assert entry.supersedes == []
    assert entry.superseded_by == []
    assert entry.status is sr.SourceStatus.ACTIVE
    assert entry.notes == ""


# 3. invalid status / authority values fail cleanly. -------------------------

def test_invalid_status_fails_cleanly():
    with pytest.raises(ValueError) as excinfo:
        sr.SourceRegistryEntry.from_dict(
            {"source_id": "src:x", "status": "retired"})
    assert "status" in str(excinfo.value)


def test_invalid_authority_fails_cleanly():
    with pytest.raises(ValueError) as excinfo:
        sr.SourceRegistryEntry.from_dict(
            {"source_id": "src:x", "authority_level": "vendor"})
    assert "authority_level" in str(excinfo.value)


def test_missing_source_id_fails_cleanly():
    with pytest.raises(ValueError):
        sr.SourceRegistryEntry.from_dict({"title": "no id"})


# 4. supersedes / superseded_by relationships can be represented. ------------

def test_supersession_relationships_representable():
    entries = sr.load_registry(_DEMO_REGISTRY)
    by_id = sr.index_by_id(entries)
    v1 = by_id["src:power-platform-governance-v1"]
    v2 = by_id["src:power-platform-governance-v2"]
    assert v1.superseded_by == ["src:power-platform-governance-v2"]
    assert v2.supersedes == ["src:power-platform-governance-v1"]
    # The bundled demo links are symmetric and resolvable -> no warnings.
    assert sr.supersession_warnings(entries) == []


def test_supersession_warnings_flag_dangling_and_asymmetric():
    entries = [
        sr.SourceRegistryEntry.from_dict(
            {"source_id": "src:a", "supersedes": ["src:ghost"]}),
        sr.SourceRegistryEntry.from_dict(
            {"source_id": "src:b", "supersedes": ["src:c"]}),
        sr.SourceRegistryEntry.from_dict({"source_id": "src:c"}),  # no back-link
    ]
    warnings = sr.supersession_warnings(entries)
    assert any("unknown source src:ghost" in w for w in warnings)
    assert any("does not record superseded_by src:b" in w for w in warnings)


# 5. stale status is computed without mutating the entry or the file. --------

def test_stale_status_computed_without_mutation(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_bytes(_DEMO_REGISTRY.read_bytes())
    before = registry_path.read_bytes()

    entries = sr.load_registry(registry_path)
    by_id = sr.index_by_id(entries)

    # An old community source whose review window has long expired computes as
    # stale, while its *stored* status stays "active".
    stale_src = by_id["src:copilot-studio-connectors"]
    assert stale_src.status is sr.SourceStatus.ACTIVE
    assert sr.compute_effective_status(stale_src, now=_NOW) is sr.SourceStatus.STALE
    assert stale_src.status is sr.SourceStatus.ACTIVE  # unchanged (frozen)

    # A current source stays active.
    fresh_src = by_id["src:least-privilege"]
    assert sr.compute_effective_status(fresh_src, now=_NOW) is sr.SourceStatus.ACTIVE

    # Terminal lifecycle states are returned as-is, never aged.
    assert sr.compute_effective_status(
        by_id["src:power-platform-governance-v1"], now=_NOW
    ) is sr.SourceStatus.DEPRECATED
    assert sr.compute_effective_status(
        by_id["src:data-residency-draft"], now=_NOW
    ) is sr.SourceStatus.DRAFT

    # The file on disk is byte-for-byte unchanged by load + compute.
    assert registry_path.read_bytes() == before


def test_compute_is_noop_without_window():
    entry = sr.SourceRegistryEntry.from_dict(
        {"source_id": "src:x", "last_reviewed_at": "2000-01-01"})
    # No stale_after_days -> nothing to age against -> stored status preserved.
    assert sr.compute_effective_status(entry, now=_NOW) is sr.SourceStatus.ACTIVE


# 6. the v3.0 retrieval-eval baseline is unchanged alongside a loaded registry.

def test_retrieval_eval_baseline_unchanged_alongside_registry(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    # Load the registry; it must not perturb the retrieval signal at all.
    sr.load_registry(_DEMO_REGISTRY)
    cases = reh.load_cases(_CASES)
    summary = reh.summarize(reh.run_eval(service, cases))
    # Documented v3.0 baseline: no off-topic bleed at the raw retrieval layer.
    assert summary.off_topic_inclusion_rate == 0.0
    assert summary.fail_count == 0


# 7. the v2.7 report citation contract holds alongside a loaded registry. ----

def test_report_citation_contract_unchanged_alongside_registry():
    sr.load_registry(_DEMO_REGISTRY)
    pkg = _grounded_package()
    template = TemplateComposer().compose(pkg)
    report = ConsultantReportComposer().compose(pkg)

    def _citations(answer) -> set:
        import re
        return set(re.findall(r"\[(?:mem|src):[^\]]+\]", answer.text))

    # The report cites exactly what the template cites — registry presence does
    # not add, drop, or alter a single citation.
    assert _citations(report) == _citations(template)
    assert report.report is not None
    assert template.report is None


# 8. no MemoryLedger entry is written by any registry operation. --------------

def test_no_memory_ledger_writes(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    before_count = len(list(service.ledger.entries()))

    entries = sr.load_registry(_DEMO_REGISTRY)
    sr.render_registry_markdown(entries, now=_NOW)
    for e in entries:
        sr.compute_effective_status(e, now=_NOW)
        sr.render_entry_markdown(e, entries, now=_NOW)

    assert len(list(service.ledger.entries())) == before_count


# 9. no knowledge source is mutated by any registry operation. ---------------

def test_no_source_file_mutation(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    before_sources = service.list_knowledge_sources()

    entries = sr.load_registry(_DEMO_REGISTRY)
    sr.render_registry_markdown(entries, now=_NOW)
    for e in entries:
        sr.render_entry_markdown(e, entries, now=_NOW)

    assert service.list_knowledge_sources() == before_sources


# 10. CLI list / inspect rendering is deterministic. -------------------------

def test_cli_list_and_inspect_deterministic():
    entries = sr.load_registry(_DEMO_REGISTRY)

    list_a = sr.render_registry_markdown(entries, now=_NOW)
    list_b = sr.render_registry_markdown(entries, now=_NOW)
    assert list_a == list_b
    # Deterministic ordering: entries are sorted by source_id.
    body_ids = [ln.split("|")[1].strip() for ln in list_a.splitlines()
                if ln.startswith("| src:")]
    assert body_ids == sorted(body_ids)

    entry = sr.index_by_id(entries)["src:power-platform-governance-v2"]
    inspect_a = sr.render_entry_markdown(entry, entries, now=_NOW)
    inspect_b = sr.render_entry_markdown(entry, entries, now=_NOW)
    assert inspect_a == inspect_b
    assert "status (effective): active" in inspect_a
    assert "supersedes: src:power-platform-governance-v1" in inspect_a


def test_cli_main_list_and_inspect(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    assert workbench.main(["source-registry", "list"]) == 0
    out = capsys.readouterr().out
    assert "# Source registry" in out
    assert "src:least-privilege" in out

    assert workbench.main(
        ["source-registry", "inspect", "src:data-residency-draft"]) == 0
    out = capsys.readouterr().out
    assert "status (effective): draft" in out

    # Unknown id fails cleanly without raising.
    assert workbench.main(["source-registry", "inspect", "src:nope"]) == 1
