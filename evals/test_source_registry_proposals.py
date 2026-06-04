"""Tests for the v4.2 Source Update Proposal Generator.

The generator turns read-only audit findings (v4.1) into structured
*maintenance proposals* — tasks a human reviews and applies by hand. These
tests pin that contract:

* each mapped audit finding becomes its expected proposal type, and a clean
  registry produces no proposals;
* the documented non-mapped findings (``deprecated_source`` as a settled state,
  ``active_source_superseded`` as advisory) produce no proposal;
* every proposal requires human approval and has ``status="proposed"`` — nothing
  is applied automatically;
* output is deterministic and proposal ids are stable;
* generation is read-only: it never calls ``save_registry``, writes no
  MemoryLedger entry, leaves ``list_knowledge_sources`` and the registry file
  byte-for-byte unchanged;
* the CLI prints to stdout writing no file, and ``--out`` writes only the one
  proposal file;
* the v4.1 audit baseline and the v3.0 retrieval / v2.7 report contracts are
  unchanged when proposal generation runs alongside them.

Nothing here changes retrieval, ranking, source selection, grounding, composer,
or memory behaviour. Proposals are tasks, not truth claims.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

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


def _ptypes_for(proposals, source_id) -> set:
    return {p.proposal_type for p in proposals if p.source_id == source_id}


def _snapshot_tree(root: Path) -> set:
    """Set of file paths under ``root`` (to detect any file written)."""
    return {p for p in root.rglob("*") if p.is_file()}


# 1. each major audit finding maps to its expected proposal type. ------------

def test_each_finding_maps_to_expected_proposal_type():
    # The mapping table itself is the contract for lifecycle/metadata codes.
    assert sr.PROPOSAL_TYPE_BY_FINDING[sr.FindingCode.STALE_BY_POLICY] == \
        sr.ProposalType.REVIEW_STALE_SOURCE
    assert sr.PROPOSAL_TYPE_BY_FINDING[sr.FindingCode.STALE_BY_STATUS] == \
        sr.ProposalType.REVIEW_STALE_SOURCE
    assert sr.PROPOSAL_TYPE_BY_FINDING[sr.FindingCode.MISSING_OWNER] == \
        sr.ProposalType.ADD_MISSING_OWNER
    assert sr.PROPOSAL_TYPE_BY_FINDING[sr.FindingCode.MISSING_LAST_REVIEWED_AT] == \
        sr.ProposalType.ADD_LAST_REVIEWED_AT
    assert sr.PROPOSAL_TYPE_BY_FINDING[sr.FindingCode.MISSING_TOPICS] == \
        sr.ProposalType.ADD_TOPICS
    assert sr.PROPOSAL_TYPE_BY_FINDING[sr.FindingCode.UNKNOWN_AUTHORITY_LEVEL] == \
        sr.ProposalType.REVIEW_UNKNOWN_AUTHORITY
    assert sr.PROPOSAL_TYPE_BY_FINDING[sr.FindingCode.DRAFT_SOURCE] == \
        sr.ProposalType.CLARIFY_DRAFT_SOURCE
    assert sr.PROPOSAL_TYPE_BY_FINDING[
        sr.FindingCode.DEPRECATED_SOURCE_WITHOUT_SUCCESSOR] == \
        sr.ProposalType.SET_SUCCESSOR_FOR_DEPRECATED_SOURCE
    assert sr.PROPOSAL_TYPE_BY_FINDING[sr.FindingCode.DANGLING_SUPERSEDES] == \
        sr.ProposalType.RESOLVE_DANGLING_SUPERSESSION
    assert sr.PROPOSAL_TYPE_BY_FINDING[sr.FindingCode.DANGLING_SUPERSEDED_BY] == \
        sr.ProposalType.RESOLVE_DANGLING_SUPERSESSION
    assert sr.PROPOSAL_TYPE_BY_FINDING[sr.FindingCode.ASYMMETRIC_SUPERSESSION] == \
        sr.ProposalType.RESOLVE_ASYMMETRIC_SUPERSESSION

    # And end-to-end: each finding produces a proposal of that type.
    aged = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:aged", "owner": "team", "authority_level": "official",
        "last_reviewed_at": "2024-01-01", "stale_after_days": 90,
        "topics": ["x"], "status": "active"})
    assert sr.ProposalType.REVIEW_STALE_SOURCE in _ptypes_for(
        sr.propose_source_updates([aged], now=_NOW), "src:aged")

    flagged = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:flagged", "owner": "team", "authority_level": "official",
        "last_reviewed_at": "2026-05-01", "topics": ["x"], "status": "stale"})
    assert sr.ProposalType.REVIEW_STALE_SOURCE in _ptypes_for(
        sr.propose_source_updates([flagged], now=_NOW), "src:flagged")

    no_owner = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:no-owner", "authority_level": "official",
        "last_reviewed_at": "2026-05-01", "topics": ["x"], "status": "active"})
    assert sr.ProposalType.ADD_MISSING_OWNER in _ptypes_for(
        sr.propose_source_updates([no_owner], now=_NOW), "src:no-owner")

    no_review = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:no-review", "owner": "team",
        "authority_level": "official", "topics": ["x"], "status": "active"})
    assert sr.ProposalType.ADD_LAST_REVIEWED_AT in _ptypes_for(
        sr.propose_source_updates([no_review], now=_NOW), "src:no-review")

    no_topics = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:no-topics", "owner": "team",
        "authority_level": "official", "last_reviewed_at": "2026-05-01",
        "status": "active"})
    assert sr.ProposalType.ADD_TOPICS in _ptypes_for(
        sr.propose_source_updates([no_topics], now=_NOW), "src:no-topics")

    unknown = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:unknown", "owner": "team",
        "last_reviewed_at": "2026-05-01", "topics": ["x"], "status": "active"})
    assert sr.ProposalType.REVIEW_UNKNOWN_AUTHORITY in _ptypes_for(
        sr.propose_source_updates([unknown], now=_NOW), "src:unknown")

    draft = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:draft", "owner": "team", "authority_level": "official",
        "last_reviewed_at": "2026-05-01", "topics": ["x"], "status": "draft"})
    assert sr.ProposalType.CLARIFY_DRAFT_SOURCE in _ptypes_for(
        sr.propose_source_updates([draft], now=_NOW), "src:draft")

    deprecated = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:dep", "owner": "team", "authority_level": "official",
        "last_reviewed_at": "2026-05-01", "topics": ["x"], "status": "deprecated"})
    assert sr.ProposalType.SET_SUCCESSOR_FOR_DEPRECATED_SOURCE in _ptypes_for(
        sr.propose_source_updates([deprecated], now=_NOW), "src:dep")

    dangling = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:dangling", "owner": "team",
        "authority_level": "official", "last_reviewed_at": "2026-05-01",
        "topics": ["x"], "status": "active", "supersedes": ["src:ghost"]})
    assert sr.ProposalType.RESOLVE_DANGLING_SUPERSESSION in _ptypes_for(
        sr.propose_source_updates([dangling], now=_NOW), "src:dangling")

    asym_a = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:asym-a", "owner": "team", "authority_level": "official",
        "last_reviewed_at": "2026-05-01", "topics": ["x"], "status": "active",
        "supersedes": ["src:asym-b"]})
    asym_b = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:asym-b", "owner": "team", "authority_level": "official",
        "last_reviewed_at": "2026-05-01", "topics": ["x"], "status": "active"})
    assert sr.ProposalType.RESOLVE_ASYMMETRIC_SUPERSESSION in _ptypes_for(
        sr.propose_source_updates([asym_a, asym_b], now=_NOW), "src:asym-a")


# 1b. the documented non-mapped findings produce no proposal. ----------------

def test_settled_and_advisory_findings_produce_no_proposal():
    # A deprecated source *with* a successor: deprecated_source is info-only and
    # has no proposal type; its lineage is consistent, so no proposal at all.
    dep = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:dep", "owner": "team", "authority_level": "official",
        "last_reviewed_at": "2026-05-01", "topics": ["x"], "status": "deprecated",
        "superseded_by": ["src:new"]})
    new = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:new", "owner": "team", "authority_level": "official",
        "last_reviewed_at": "2026-05-01", "topics": ["x"], "status": "active",
        "supersedes": ["src:dep"]})
    assert sr.propose_source_updates([dep, new], now=_NOW) == []

    # An active source flagged active_source_superseded (advisory) yields no
    # proposal — the human deprecation is what would later trigger one.
    active = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:active", "owner": "team", "authority_level": "official",
        "last_reviewed_at": "2026-05-01", "topics": ["x"], "status": "active",
        "superseded_by": ["src:successor"]})
    successor = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:successor", "owner": "team",
        "authority_level": "official", "last_reviewed_at": "2026-05-01",
        "topics": ["x"], "status": "active", "supersedes": ["src:active"]})
    assert _ptypes_for(
        sr.propose_source_updates([active, successor], now=_NOW),
        "src:active") == set()


# 2. a clean registry produces no proposals. ---------------------------------

def test_clean_registry_produces_no_proposals():
    entry = sr.SourceRegistryEntry.from_dict({
        "source_id": "src:clean", "owner": "team", "authority_level": "official",
        "last_reviewed_at": "2026-05-01", "stale_after_days": 120,
        "topics": ["x"], "status": "active"})
    assert sr.propose_source_updates([entry], now=_NOW) == []
    assert "(no proposals)" in sr.render_proposals_markdown([])
    assert sr.proposals_to_jsonl([]) == ""


# 3. proposal output is deterministic. ---------------------------------------

def test_proposal_output_is_deterministic():
    entries = sr.load_registry(_DEMO_REGISTRY)
    a = sr.propose_source_updates(entries, now=_NOW)
    b = sr.propose_source_updates(entries, now=_NOW)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]
    assert sr.render_proposals_markdown(a) == sr.render_proposals_markdown(b)
    assert sr.proposals_to_jsonl(a) == sr.proposals_to_jsonl(b)
    # Sorted by (severity rank, proposal_type, source_id, proposal_id).
    keys = [(sr._SEVERITY_RANK[p.severity], p.proposal_type, p.source_id,
             p.proposal_id) for p in a]
    assert keys == sorted(keys)


# 4. proposal ids are deterministic and stable. ------------------------------

def test_proposal_ids_are_deterministic_and_stable():
    entries = sr.load_registry(_DEMO_REGISTRY)
    a = sr.propose_source_updates(entries, now=_NOW)
    b = sr.propose_source_updates(entries, now=_NOW)
    assert [p.proposal_id for p in a] == [p.proposal_id for p in b]
    # Stable shape, derived only from (source_id, finding_code, rationale).
    for p in a:
        assert p.proposal_id.startswith("srcprop-")
        assert p.proposal_id == sr._proposal_id(
            p.source_id, p.finding_code, p.rationale)


# 5. proposal generation never calls save_registry. --------------------------

def test_propose_never_calls_save_registry(monkeypatch):
    calls = {"n": 0}
    real_save = sr.save_registry

    def _spy(*args, **kwargs):  # pragma: no cover - should never run
        calls["n"] += 1
        return real_save(*args, **kwargs)

    monkeypatch.setattr(sr, "save_registry", _spy)
    entries = sr.load_registry(_DEMO_REGISTRY)
    proposals = sr.propose_source_updates(entries, now=_NOW)
    sr.render_proposals_markdown(proposals)
    sr.proposals_to_jsonl(proposals)
    assert calls["n"] == 0


# 6. the registry file remains byte-identical after generation. --------------

def test_registry_file_byte_identical_after_propose(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_bytes(_DEMO_REGISTRY.read_bytes())
    before = registry_path.read_bytes()

    entries = sr.load_registry(registry_path)
    sr.propose_source_updates(entries, now=_NOW)

    assert registry_path.read_bytes() == before


# 7. knowledge sources remain unchanged around generation. -------------------

def test_knowledge_sources_unchanged_around_propose(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    before_sources = service.list_knowledge_sources()

    proposals = sr.propose_source_updates(sr.load_registry(_DEMO_REGISTRY),
                                          now=_NOW)
    sr.render_proposals_markdown(proposals)

    assert service.list_knowledge_sources() == before_sources


# 8. no MemoryLedger entry is written by generation. -------------------------

def test_propose_writes_no_memory_ledger(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    before_count = len(list(service.ledger.entries()))

    proposals = sr.propose_source_updates(sr.load_registry(_DEMO_REGISTRY),
                                          now=_NOW)
    sr.render_proposals_markdown(proposals)

    assert len(list(service.ledger.entries())) == before_count


# 9. CLI stdout mode writes no files. ----------------------------------------

def test_cli_stdout_mode_writes_no_files(tmp_path, monkeypatch, capsys):
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_bytes(_DEMO_REGISTRY.read_bytes())
    before_tree = _snapshot_tree(tmp_path)
    before_bytes = registry_path.read_bytes()

    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    assert workbench.main(
        ["source-registry", "propose-updates",
         "--registry", str(registry_path)]) == 0
    out = capsys.readouterr().out
    assert "# Source update proposals" in out
    assert "requires human approval" in out
    # No file created or changed by the stdout path.
    assert _snapshot_tree(tmp_path) == before_tree
    assert registry_path.read_bytes() == before_bytes


# 10. CLI --out writes only the proposal file. -------------------------------

def test_cli_out_writes_only_proposal_file(tmp_path, monkeypatch, capsys):
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_bytes(_DEMO_REGISTRY.read_bytes())
    before_bytes = registry_path.read_bytes()
    before_tree = _snapshot_tree(tmp_path)
    out_path = tmp_path / "out" / "proposals.jsonl"

    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))
    import workbench

    assert workbench.main(
        ["source-registry", "propose-updates",
         "--registry", str(registry_path), "--out", str(out_path)]) == 0

    # The registry is untouched; the only new file is the proposal export.
    assert registry_path.read_bytes() == before_bytes
    new_files = _snapshot_tree(tmp_path) - before_tree
    assert new_files == {out_path}
    assert out_path.read_text(encoding="utf-8") == sr.proposals_to_jsonl(
        sr.propose_source_updates(sr.load_registry(registry_path)))


# 11. every proposal requires human approval and is only 'proposed'. ---------

def test_every_proposal_requires_human_approval():
    proposals = sr.propose_source_updates(sr.load_registry(_DEMO_REGISTRY),
                                          now=_NOW)
    assert proposals  # the demo registry surfaces maintenance work
    for p in proposals:
        assert p.requires_human_approval is True
        assert p.status == "proposed"
        assert p.to_dict()["requires_human_approval"] is True
        assert p.to_dict()["status"] == "proposed"


# 12. the v4.1 audit baseline is unchanged alongside proposal generation. ----

def test_audit_baseline_unchanged_alongside_propose():
    entries = sr.load_registry(_DEMO_REGISTRY)
    before = sr.audit_registry(entries, now=_NOW).to_dict()
    sr.propose_source_updates(entries, now=_NOW)
    after = sr.audit_registry(entries, now=_NOW).to_dict()
    assert before == after


# 13. retrieval / report contracts hold alongside proposal generation. -------

def test_retrieval_and_report_unchanged_alongside_propose(tmp_path, monkeypatch):
    service = _build_service(tmp_path, monkeypatch)
    sr.propose_source_updates(sr.load_registry(_DEMO_REGISTRY), now=_NOW)

    cases = reh.load_cases(_CASES)
    summary = reh.summarize(reh.run_eval(service, cases))
    assert summary.off_topic_inclusion_rate == 0.0
    assert summary.fail_count == 0

    pkg = _grounded_package()
    template = TemplateComposer().compose(pkg)
    report = ConsultantReportComposer().compose(pkg)

    def _citations(answer) -> set:
        import re
        return set(re.findall(r"\[(?:mem|src):[^\]]+\]", answer.text))

    assert _citations(report) == _citations(template)
    assert report.report is not None
    assert template.report is None


# baseline: the bundled demo registry yields known proposal types. -----------

def test_demo_registry_proposals_baseline():
    proposals = sr.propose_source_updates(sr.load_registry(_DEMO_REGISTRY),
                                          now=_NOW)
    by_type = {}
    for p in proposals:
        by_type.setdefault(p.proposal_type, []).append(p.source_id)

    # The demo registry's audit findings (stale x2, missing-review, draft,
    # unknown-authority) convert to these maintenance tasks; the settled
    # deprecated source contributes none.
    assert sr.ProposalType.REVIEW_STALE_SOURCE in by_type
    assert sr.ProposalType.ADD_LAST_REVIEWED_AT in by_type
    assert sr.ProposalType.CLARIFY_DRAFT_SOURCE in by_type
    assert sr.ProposalType.REVIEW_UNKNOWN_AUTHORITY in by_type
    assert sr.ProposalType.SET_SUCCESSOR_FOR_DEPRECATED_SOURCE not in by_type
