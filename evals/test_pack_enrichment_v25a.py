"""Tests for the v2.5A targeted M365 / Coding pack enrichment.

v2.5A appends *claim-shaped*, source-governed paragraphs to the existing pack
sources so the previously-ungrounded value-sprint rows (the "pack gaps") can be
answered from real, cited evidence. The enrichment is purely **additive**: no
frozen scoring module (verifier, grounding, citations, lifecycle, refusal, pack
isolation, EvidenceRanker, SufficiencyGate, AnswerGuard) is touched, the pack
manifest still lists the same five sources, and the deterministic/offline path
remains the default.

These tests assert two things under the **hybrid** (token bag-of-words) retrieval
backend — the meaning-based path the enrichment targets:

* each enriched gap query now grounds on the *correct* source and domain, with
  the stale Power Platform source still flagged; and
* the v2.4 false-grounding protections still hold — an out-of-domain query is
  still refused, and a Purview-threshold question leads on the Purview source
  rather than the Power Platform formulas source.

Why hybrid and not deterministic: the default ``DeterministicEncoder`` hashes the
*whole* chunk string into a random vector, so a natural-language paraphrase only
retrieves its chunk by chance. The hybrid ``OfflineHashingEmbedder`` embeds
*tokens*, so shared vocabulary retrieves reliably and offline. The enrichment is
content, not a change to either backend; deterministic remains the default.

Nothing here touches geometry, verifier rules, lifecycle, or citation semantics.
The pack is built into an isolated ``tmp_path`` registry so tracked sources stay
pristine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import pack_builder  # noqa: E402
from agent.project_packs import PackRegistry  # noqa: E402
from agent.workbench_service import WorkbenchService  # noqa: E402
from retrieval.embedding_backend import OfflineHashingEmbedder  # noqa: E402
from slm.assistant_composer import ComposerMode  # noqa: E402

_MANIFEST = ROOT / "packs" / "m365_coding_assistant" / "pack.yaml"

_PURVIEW = "Purview Sensitivity Labels and DLP"
_ADMIN = "Microsoft 365 Admin Patterns"
_POWER = "Power Platform PowerApps Formulas"
_CODING = "Local Coding-Agent Workflow Rules"

# Each enriched gap query, with the source and domain it must ground on. The
# query strings are verbatim from demos/value_sprint_queries.jsonl (the rows the
# v2.5A spec named: 6, 11, 13, 14, 16, 19, 20).
_GROUNDED_CASES = [
    pytest.param(
        "What does my pack say about Purview sensitivity label encryption "
        "thresholds?",
        _PURVIEW, "microsoft", id="row6-purview-encryption-threshold"),
    pytest.param(
        "How should a Microsoft 365 consultant explain role-scoped tenant "
        "administration and least-privilege admin roles to a new admin?",
        _ADMIN, "microsoft", id="row11-least-privilege-admin"),
    pytest.param(
        "How do I design a Microsoft Purview sensitivity label taxonomy and "
        "choose the encryption threshold?",
        _PURVIEW, "microsoft", id="row13-label-taxonomy"),
    pytest.param(
        "How should sensitivity labels protect Highly Confidential files while "
        "keeping third-party tools working?",
        _PURVIEW, "microsoft", id="row14-highly-confidential-third-party"),
    pytest.param(
        "What is the safest next prompt to give the coding agent before it "
        "edits this repo?",
        _CODING, "coding", id="row19-safe-edit-prompt"),
    pytest.param(
        "Which retrieval backend did the v2.2 milestone add to the repo?",
        _CODING, "coding", id="row20-v22-hybrid-backend"),
]


def _hybrid_service(tmp_path, monkeypatch):
    """Build the real M365 pack and bind a hybrid (token-embedder) service.

    Mirrors ``app/workbench.py:_build_sprint_service(..., backend='hybrid')`` so
    the test exercises the same meaning-based retrieval path the value sprint
    uses, without touching the tracked registry.
    """
    monkeypatch.chdir(ROOT)
    registry = PackRegistry(tmp_path / "packs")
    plan = pack_builder.PackBuildPlan.from_file(_MANIFEST)
    report = pack_builder.build_pack(plan, registry)
    pack = registry.get_pack(report.pack_id)
    return WorkbenchService.from_pack(
        pack, registry=registry,
        knowledge_backend="hybrid", semantic_embedder=OfflineHashingEmbedder())


# 1. each enriched gap query grounds on the correct source and domain. --------

@pytest.mark.parametrize("query, source, domain", _GROUNDED_CASES)
def test_enriched_gap_query_grounds_on_correct_source(
        tmp_path, monkeypatch, query, source, domain):
    service = _hybrid_service(tmp_path, monkeypatch)
    pkg = service.build_grounding_package(query)

    assert pkg.mode == ComposerMode.GROUNDED, \
        f"expected GROUNDED for {query!r}, got {pkg.mode}"
    assert not pkg.refused
    assert pkg.evidence, "a grounded package must carry citable evidence"
    lead = pkg.evidence[0]
    assert lead.kind == "knowledge"
    assert lead.source_name == source, \
        f"{query!r} led on {lead.source_name!r}, expected {source!r}"
    assert lead.domain == domain


# 2. the enriched Power Fx delegation row grounds WITH a stale caution. --------
# Power Platform is a STALE source; enrichment must not erase the stale flag.

def test_powerfx_delegation_grounds_but_flags_stale(tmp_path, monkeypatch):
    service = _hybrid_service(tmp_path, monkeypatch)
    pkg = service.build_grounding_package(
        "What Power Fx delegation limits should I watch for when filtering "
        "large data sources?")

    assert pkg.mode == ComposerMode.GROUNDED
    assert pkg.evidence[0].source_name == _POWER
    assert pkg.evidence[0].domain == "microsoft"
    assert any("stale" in c.lower() for c in pkg.cautions), \
        "Power Platform source is stale; a stale caution was expected"


# 3. false-grounding protection: out-of-domain query is still refused. --------
# Enrichment adds no AWS content, and the v2.4 relevance gate must still refuse.

def test_out_of_domain_query_still_refused(tmp_path, monkeypatch):
    service = _hybrid_service(tmp_path, monkeypatch)
    pkg = service.build_grounding_package(
        "How do I configure AWS IAM roles and S3 bucket policies for a Lambda "
        "function?")

    assert pkg.refused, "an out-of-domain query must not falsely ground"
    assert pkg.mode in (
        ComposerMode.REFUSAL, ComposerMode.CONFLICT_EXPLANATION)
    for term in ("aws", "s3 bucket", "lambda"):
        for item in pkg.evidence:
            assert term not in (item.source_name or "").lower()


# 4. cross-source precision: a Purview-threshold question never leads on the ---
# Power Platform formulas source (the threshold claim lives in Purview).

def test_purview_threshold_does_not_lead_on_powerapps(tmp_path, monkeypatch):
    service = _hybrid_service(tmp_path, monkeypatch)
    pkg = service.build_grounding_package(
        "What does my pack say about Purview sensitivity label encryption "
        "thresholds?")

    assert pkg.mode == ComposerMode.GROUNDED
    assert pkg.evidence[0].source_name == _PURVIEW
    assert pkg.evidence[0].source_name != _POWER
