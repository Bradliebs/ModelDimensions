"""Tests for the v4.1 Source Registry Audit / Stale Source Detector.

The audit walks the read-only source registry and *reports* lifecycle and
metadata risk — stale/deprecated/draft sources, missing owner/review/topics/
authority metadata, and dangling/asymmetric/inconsistent supersession links —
without ever mutating an entry, writing a file, or touching retrieval. These
tests pin that contract:

* fresh active sources produce no stale finding; aged sources produce
  ``stale_by_policy``; an explicit ``stale`` status produces ``stale_by_status``;
* deprecated and draft sources are surfaced; missing metadata is surfaced;
* dangling, asymmetric, active-superseded, and deprecated-without-successor
  supersession problems are each reported;
* the audit is deterministic and read-only: it writes no file, never calls
  ``save_registry``, writes no MemoryLedger entry, and leaves
  ``list_knowledge_sources`` byte-for-byte unchanged;
* the v3.0 retrieval-eval baseline and v2.7 report citation contract are
  unchanged when an audit runs alongside them.

Nothing here changes retrieval, ranking, source selection, grounding, composer,
or memory behaviour — the audit only reports metadata risk. Source text remains
the evidence; registry metadata remains metadata.
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


def _codes_for(report, source_id) -> set:
    return {f.code for f in report.findings_for(source_id)}


# 1. fresh active sources produce no stale finding. --------------------------

def test_fresh_active_source_has_no_stale_finding():
    entry = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:fresh",
        "owner": "team",
        "authority_level": "official",
        "last_reviewed_at": "2026-05-01",
        "stale_after_days": 120,
        "topics": ["x"],
        "status": "active",
    })
    report = sr.audit_registry([entry], now=_NOW)
    codes = _codes_for(report, "src:fresh")
    assert sr.FindingCode.STALE_BY_POLICY not in codes
    assert sr.FindingCode.STALE_BY_STATUS not in codes
    # A complete, current source produces no findings at all.
    assert report.findings == []


# 2. stale_after_days + old last_reviewed_at produces stale_by_policy. -------

def test_aged_review_window_produces_stale_by_policy():
    entry = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:aged",
        "owner": "team",
        "authority_level": "official",
        "last_reviewed_at": "2025-01-01",
        "stale_after_days": 90,
        "topics": ["x"],
        "status": "active",
    })
    report = sr.audit_registry([entry], now=_NOW)
    codes = _codes_for(report, "src:aged")
    assert sr.FindingCode.STALE_BY_POLICY in codes
    assert sr.FindingCode.STALE_BY_STATUS not in codes
    finding = next(f for f in report.findings
                   if f.code == sr.FindingCode.STALE_BY_POLICY)
    assert finding.severity is sr.FindingSeverity.WARNING


# 3. explicit status 'stale' produces stale_by_status. -----------------------

def test_explicit_stale_status_produces_stale_by_status():
    entry = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:flagged",
        "owner": "team",
        "authority_level": "official",
        "last_reviewed_at": "2026-05-01",
        "stale_after_days": 120,
        "topics": ["x"],
        "status": "stale",
    })
    report = sr.audit_registry([entry], now=_NOW)
    codes = _codes_for(report, "src:flagged")
    # Stored 'stale' wins; we do not also emit stale_by_policy for the same entry.
    assert sr.FindingCode.STALE_BY_STATUS in codes
    assert sr.FindingCode.STALE_BY_POLICY not in codes


# 4. deprecated sources are reported. ----------------------------------------

def test_deprecated_source_reported():
    entry = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:old",
        "owner": "team",
        "authority_level": "reputable",
        "last_reviewed_at": "2024-01-01",
        "topics": ["x"],
        "status": "deprecated",
        "superseded_by": ["src:new"],
    })
    successor = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:new",
        "owner": "team",
        "authority_level": "reputable",
        "last_reviewed_at": "2026-05-01",
        "stale_after_days": 180,
        "topics": ["x"],
        "supersedes": ["src:old"],
        "status": "active",
    })
    report = sr.audit_registry([entry, successor], now=_NOW)
    codes = _codes_for(report, "src:old")
    assert sr.FindingCode.DEPRECATED_SOURCE in codes
    # It records a successor, so it is NOT flagged as orphaned.
    assert sr.FindingCode.DEPRECATED_SOURCE_WITHOUT_SUCCESSOR not in codes
    finding = next(f for f in report.findings
                   if f.code == sr.FindingCode.DEPRECATED_SOURCE)
    assert finding.severity is sr.FindingSeverity.INFO


# 5. draft sources are reported. ---------------------------------------------

def test_draft_source_reported():
    entry = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:wip",
        "owner": "team",
        "authority_level": "reputable",
        "last_reviewed_at": "2026-05-01",
        "topics": ["x"],
        "status": "draft",
    })
    report = sr.audit_registry([entry], now=_NOW)
    codes = _codes_for(report, "src:wip")
    assert sr.FindingCode.DRAFT_SOURCE in codes
    finding = next(f for f in report.findings
                   if f.code == sr.FindingCode.DRAFT_SOURCE)
    assert finding.severity is sr.FindingSeverity.INFO


# 6. missing owner/review/topics/authority metadata is reported. -------------

def test_missing_metadata_reported():
    entry = sr.SourceRegistryEntry.from_dict({"source_id": "src:bare"})
    report = sr.audit_registry([entry], now=_NOW)
    codes = _codes_for(report, "src:bare")
    assert sr.FindingCode.MISSING_OWNER in codes
    assert sr.FindingCode.MISSING_LAST_REVIEWED_AT in codes
    assert sr.FindingCode.MISSING_TOPICS in codes
    assert sr.FindingCode.UNKNOWN_AUTHORITY_LEVEL in codes


# 7. dangling supersedes / superseded_by is reported. ------------------------

def test_dangling_supersession_reported():
    a = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:a", "supersedes": ["src:ghost"]})
    b = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:b", "superseded_by": ["src:phantom"]})
    report = sr.audit_registry([a, b], now=_NOW)
    assert sr.FindingCode.DANGLING_SUPERSEDES in _codes_for(report, "src:a")
    assert sr.FindingCode.DANGLING_SUPERSEDED_BY in _codes_for(report, "src:b")
    for code in (sr.FindingCode.DANGLING_SUPERSEDES,
                 sr.FindingCode.DANGLING_SUPERSEDED_BY):
        finding = next(f for f in report.findings if f.code == code)
        assert finding.severity is sr.FindingSeverity.ERROR


# 8. asymmetric supersession is reported. ------------------------------------

def test_asymmetric_supersession_reported():
    a = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:a", "supersedes": ["src:b"]})
    # b exists but does NOT record superseded_by src:a -> asymmetric.
    b = sr.SourceRegistryEntry.from_dict({"source_id": "src:b"})
    report = sr.audit_registry([a, b], now=_NOW)
    codes = _codes_for(report, "src:a")
    assert sr.FindingCode.ASYMMETRIC_SUPERSESSION in codes
    finding = next(f for f in report.findings
                   if f.code == sr.FindingCode.ASYMMETRIC_SUPERSESSION)
    assert finding.severity is sr.FindingSeverity.ERROR


# 9. an active source superseded by another source is reported. --------------

def test_active_source_superseded_reported():
    a = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:a",
        "owner": "team",
        "authority_level": "official",
        "last_reviewed_at": "2026-05-01",
        "topics": ["x"],
        "status": "active",
        "superseded_by": ["src:b"],
    })
    b = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:b",
        "owner": "team",
        "authority_level": "official",
        "last_reviewed_at": "2026-05-01",
        "topics": ["x"],
        "status": "active",
        "supersedes": ["src:a"],
    })
    report = sr.audit_registry([a, b], now=_NOW)
    codes = _codes_for(report, "src:a")
    assert sr.FindingCode.ACTIVE_SOURCE_SUPERSEDED in codes
    finding = next(f for f in report.findings
                   if f.code == sr.FindingCode.ACTIVE_SOURCE_SUPERSEDED)
    assert finding.severity is sr.FindingSeverity.WARNING


# 10. a deprecated source without successor is reported. ---------------------

def test_deprecated_without_successor_reported():
    entry = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:orphan",
        "owner": "team",
        "authority_level": "reputable",
        "last_reviewed_at": "2024-01-01",
        "topics": ["x"],
        "status": "deprecated",
    })
    report = sr.audit_registry([entry], now=_NOW)
    codes = _codes_for(report, "src:orphan")
    assert sr.FindingCode.DEPRECATED_SOURCE_WITHOUT_SUCCESSOR in codes
    finding = next(f for f in report.findings
                   if f.code == sr.FindingCode.DEPRECATED_SOURCE_WITHOUT_SUCCESSOR)
    assert finding.severity is sr.FindingSeverity.WARNING


# 11. audit output is deterministic (ordering + render). ---------------------

def test_audit_is_deterministic():
    entries = sr.load_registry(_DEMO_REGISTRY)
    report_a = sr.audit_registry(entries, now=_NOW)
    report_b = sr.audit_registry(entries, now=_NOW)
    assert report_a.to_dict() == report_b.to_dict()
    assert sr.render_audit_markdown(report_a) == sr.render_audit_markdown(report_b)
    # Findings are ordered by (severity rank, code, source_id): errors first.
    keys = [(sr._SEVERITY_RANK[f.severity], f.code, f.source_id)
            for f in report_a.findings]
    assert keys == sorted(keys)


# 12. the CLI audit is read-only (no file written, exit 0). ------------------

def test_cli_audit_is_read_only(tmp_path, monkeypatch, capsys):
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_bytes(_DEMO_REGISTRY.read_bytes())
    before = registry_path.read_bytes()

    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    assert workbench.main(
        ["source-registry", "audit", "--registry", str(registry_path)]) == 0
    out = capsys.readouterr().out
    assert "# Source registry audit" in out
    assert "## Findings" in out
    # The registry file is byte-for-byte unchanged by the audit.
    assert registry_path.read_bytes() == before


# 13. save_registry is never called by the audit path. -----------------------

def test_audit_never_calls_save_registry(monkeypatch):
    calls = {"n": 0}
    real_save = sr.save_registry

    def _spy(*args, **kwargs):  # pragma: no cover - should never run
        calls["n"] += 1
        return real_save(*args, **kwargs)

    monkeypatch.setattr(sr, "save_registry", _spy)
    entries = sr.load_registry(_DEMO_REGISTRY)
    report = sr.audit_registry(entries, now=_NOW)
    sr.render_audit_markdown(report)
    assert calls["n"] == 0


# 14. the v3.0 retrieval-eval baseline is unchanged alongside an audit. ------

def test_retrieval_eval_baseline_unchanged_alongside_audit(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    sr.audit_registry(sr.load_registry(_DEMO_REGISTRY), now=_NOW)
    cases = reh.load_cases(_CASES)
    summary = reh.summarize(reh.run_eval(service, cases))
    assert summary.off_topic_inclusion_rate == 0.0
    assert summary.fail_count == 0


# 15. the v2.7 report citation contract holds alongside an audit. ------------

def test_report_citation_contract_unchanged_alongside_audit():
    sr.audit_registry(sr.load_registry(_DEMO_REGISTRY), now=_NOW)
    pkg = _grounded_package()
    template = TemplateComposer().compose(pkg)
    report = ConsultantReportComposer().compose(pkg)

    def _citations(answer) -> set:
        import re
        return set(re.findall(r"\[(?:mem|src):[^\]]+\]", answer.text))

    assert _citations(report) == _citations(template)
    assert report.report is not None
    assert template.report is None


# 16. no MemoryLedger entry is written by the audit. -------------------------

def test_audit_writes_no_memory_ledger(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    before_count = len(list(service.ledger.entries()))

    report = sr.audit_registry(sr.load_registry(_DEMO_REGISTRY), now=_NOW)
    sr.render_audit_markdown(report)

    assert len(list(service.ledger.entries())) == before_count


# 17. list_knowledge_sources is unchanged before/after the audit. ------------

def test_knowledge_sources_unchanged_around_audit(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    before_sources = service.list_knowledge_sources()

    report = sr.audit_registry(sr.load_registry(_DEMO_REGISTRY), now=_NOW)
    sr.render_audit_markdown(report)

    assert service.list_knowledge_sources() == before_sources


# baseline: the bundled demo registry audits to a known, error-free result. --

def test_demo_registry_baseline_findings():
    entries = sr.load_registry(_DEMO_REGISTRY)
    report = sr.audit_registry(entries, now=_NOW)

    # No registry-integrity errors in the curated demo (links are consistent).
    assert report.error_count == 0

    # Lifecycle/metadata findings the demo is designed to surface.
    assert sr.FindingCode.STALE_BY_POLICY in _codes_for(
        report, "src:copilot-studio-connectors")
    assert sr.FindingCode.DEPRECATED_SOURCE in _codes_for(
        report, "src:power-platform-governance-v1")
    assert _codes_for(report, "src:data-residency-draft") == {
        sr.FindingCode.DRAFT_SOURCE,
        sr.FindingCode.MISSING_LAST_REVIEWED_AT,
        sr.FindingCode.UNKNOWN_AUTHORITY_LEVEL,
    }
    # The fully-specified current source is clean.
    assert _codes_for(report, "src:least-privilege") == set()

    # The frozen entries are untouched by auditing (no mutation).
    assert sr.load_registry(_DEMO_REGISTRY)[0].to_dict() == entries[0].to_dict()
