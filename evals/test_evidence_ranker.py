"""Tests for the v2.4 EvidenceRanker and SufficiencyGate (the canonical split).

These cover the two-module implementation directly — the *ranker* that scores
and orders candidates and records the top rejected one, and the *gate* that
turns the ranker's strongest substantive score into an answerability verdict —
plus the surfaces v2.4 added on top of the existing relevance gate:

* the ranker emits an auditable :class:`RankingReport` with the single most
  relevant candidate it did *not* let lead, and the reason it was held back;
* a project-milestone query prefers seeded project memory over an incidental
  knowledge chunk;
* a weak-match query yields a pack-gap proposal from the value-sprint harness;
* the top rejected candidate reaches the query-assist audit through
  ``build_grounding_package``.

The backward-compatible behaviour of :func:`agent.relevance_gate.assess_relevance`
itself is covered in ``test_relevance_gate.py``; this file proves the new pieces.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.value_sprint_harness import (  # noqa: E402
    EXPECTED_GROUNDED_USEFUL,
    EXPECTED_PACK_GAP,
    SprintQuery,
    extract_pack_gap,
)
from agent.workbench_service import WorkbenchService  # noqa: E402
from retrieval.evidence_ranker import EvidenceRanker  # noqa: E402
from retrieval.sufficiency_gate import (  # noqa: E402
    LABEL_GROUNDED_PARTIAL,
    LABEL_GROUNDED_RELEVANT,
    LABEL_OUT_OF_DOMAIN_FALSE_GROUNDING,
    SufficiencyGate,
    SufficiencyVerdict,
)
from slm.assistant_composer import ComposerMode, EvidenceItem  # noqa: E402

_RANKER = EvidenceRanker()
_GATE = SufficiencyGate()


# ---------- helpers ---------------------------------------------------------

def _know(cid: str, text: str) -> EvidenceItem:
    return EvidenceItem(citation_id=f"src:{cid}", kind="knowledge", text=text,
                        source_name="Pack Notes", domain="microsoft",
                        authority="official")


def _mem(cid: str, text: str) -> EvidenceItem:
    return EvidenceItem(citation_id=f"mem:{cid}", kind="memory", text=text)


def _service(tmp_path) -> WorkbenchService:
    return WorkbenchService(
        ledger_path=str(tmp_path / "ledger.jsonl"),
        queue_path=str(tmp_path / "queue.jsonl"),
        knowledge_path=str(tmp_path / "knowledge.jsonl"),
    )


def _write_doc(tmp_path, text: str, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------- ranker: ordering + top rejected --------------------------------

def test_ranker_orders_substantive_lead_first():
    report = _RANKER.rank(
        "purview sensitivity label encryption threshold",
        [_know("off", "The Microsoft 365 admin center manages user licenses."),
         _know("hit",
               "Sensitivity labels apply encryption to documents across the "
               "tenant when the threshold is met.")],
        route="knowledge_only")
    assert report.best is not None
    assert report.best.citation_id == "src:hit"
    assert report.ordered_evidence[0].citation_id == "src:hit"
    # ranks are assigned in display order, strongest first.
    assert report.assessments[0].rank == 1


def test_ranker_records_top_rejected_with_reason():
    report = _RANKER.rank(
        "purview sensitivity label encryption threshold",
        [_know("hit",
               "Sensitivity labels apply encryption to documents across the "
               "tenant when the threshold is met."),
         _know("off", "The Microsoft 365 admin center manages user licenses.")],
        route="knowledge_only")
    rejected = report.top_rejected
    assert rejected is not None
    assert rejected["citation_id"] == "src:off"
    assert rejected["reason"]
    # the report serialises the rejection for the audit trail.
    assert report.to_dict()["top_rejected"]["citation_id"] == "src:off"


def test_ranker_top_rejected_none_for_single_candidate():
    report = _RANKER.rank(
        "purview sensitivity label encryption",
        [_know("only",
               "Sensitivity labels apply encryption to documents tenant-wide.")],
        route="knowledge_only")
    assert report.top_rejected is None


# ---------- gate: verdict mapping ------------------------------------------

def test_gate_strong_domain_match_is_relevant():
    decision = _GATE.classify(
        best_score=0.8, best_domain_match=True, has_best=True,
        out_marker=None, conflict_signal=False, had_evidence=True,
        had_header=False)
    assert decision.verdict == SufficiencyVerdict.RELEVANT
    assert decision.label == LABEL_GROUNDED_RELEVANT
    assert decision.can_ground is True


def test_gate_moderate_is_partial_with_limitation():
    decision = _GATE.classify(
        best_score=0.35, best_domain_match=True, has_best=True,
        out_marker=None, conflict_signal=False, had_evidence=True,
        had_header=False)
    assert decision.verdict == SufficiencyVerdict.PARTIAL
    assert decision.label == LABEL_GROUNDED_PARTIAL
    assert decision.limitation is not None


def test_gate_out_of_domain_refuses():
    decision = _GATE.classify(
        best_score=0.6, best_domain_match=False, has_best=True,
        out_marker="aws", conflict_signal=False, had_evidence=True,
        had_header=False)
    assert decision.verdict == SufficiencyVerdict.NO_SUPPORT
    assert decision.label == LABEL_OUT_OF_DOMAIN_FALSE_GROUNDING
    assert decision.can_ground is False


def test_gate_conflict_outranks_grounding():
    decision = _GATE.classify(
        best_score=0.9, best_domain_match=True, has_best=True,
        out_marker=None, conflict_signal=True, had_evidence=True,
        had_header=False)
    assert decision.verdict == SufficiencyVerdict.CONFLICT
    assert decision.can_ground is False


# ---------- project-milestone preference -----------------------------------

def test_milestone_query_prefers_seeded_memory():
    report = _RANKER.rank(
        "what milestone did we ship in the v2.2 sprint roadmap",
        [_know("admin",
               "The Microsoft 365 admin center manages user licenses and "
               "mailbox settings for the tenant."),
         _mem("decision",
              "We shipped the hybrid retrieval milestone in the v2.2 sprint.")],
        route="both")
    assert report.milestone_query is True
    assert report.best is not None
    assert report.best.citation_id == "mem:decision"
    assert report.ordered_evidence[0].citation_id == "mem:decision"


def test_non_milestone_query_has_no_memory_boost():
    report = _RANKER.rank(
        "how do I filter a gallery by a dropdown selection",
        [_mem("decision",
              "We shipped the hybrid retrieval milestone in the v2.2 sprint.")],
        route="both")
    assert report.milestone_query is False


# ---------- weak match -> pack-gap proposal --------------------------------

def test_weak_match_yields_pack_gap_proposal():
    # A value-expected query that does not ground should propose a pack update.
    spec = SprintQuery(query="how do I rotate signing keys in Purview",
                       note="key rotation", category="howto",
                       expected_outcome=EXPECTED_GROUNDED_USEFUL)
    proposal = extract_pack_gap(spec, grounded=False)
    assert proposal is not None
    assert proposal.suggested_source_type == "knowledge_source"
    assert "key rotation" in proposal.missing_topic


def test_explicit_pack_gap_probe_proposes_even_when_grounded():
    spec = SprintQuery(query="something the pack cannot cover",
                       note="uncovered topic", category="howto",
                       expected_outcome=EXPECTED_PACK_GAP)
    assert extract_pack_gap(spec, grounded=True) is not None


# ---------- top rejected reaches the audit through grounding ---------------

def test_top_rejected_surfaced_in_grounding_audit(tmp_path):
    service = _service(tmp_path)
    service.add_memory(
        "the supplier delivery is on Friday afternoon", source="seed")
    doc = _write_doc(
        tmp_path,
        "# Notes\n\nThe supplier delivery afternoon slot is documented here.\n")
    service.import_knowledge(doc, domain="microsoft", authority="official",
                             source_name="Schedule Notes", version="1.0")

    query = "when is the supplier delivery afternoon slot"
    package = service.build_grounding_package(query)
    relevance = package.query_audit["relevance"]
    # the audit always carries the top_rejected key (None when nothing rejected).
    assert "top_rejected" in relevance
    if relevance["top_rejected"] is not None:
        assert relevance["top_rejected"]["citation_id"]
        assert relevance["top_rejected"]["reason"]
    assert package.mode in (ComposerMode.GROUNDED, ComposerMode.REFUSAL,
                            ComposerMode.CONFLICT_EXPLANATION)
