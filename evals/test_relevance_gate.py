"""Tests for the v2.4 relevance & sufficiency gate.

The gate sits between retrieval and grounding. Retrieval presence is not
relevance: the frozen deterministic backend returns its top candidates and the
old grounding path grounded an answer whenever *any* chunk came back. These
tests prove the v2.4 contract:

* the gate verdicts (RELEVANT / PARTIAL / WEAK_MATCH / CONFLICT / NO_SUPPORT)
  map onto the lexical overlap of query and evidence;
* the value-sprint false-grounding rows no longer ground — a topic mismatch is
  WEAK_MATCH or NO_SUPPORT, never RELEVANT;
* an out-of-domain (AWS/S3/Lambda) question refuses even when an incidental
  chunk is retrieved;
* a metadata/source header can never be the lead evidence;
* a memory near-miss the verifier rejected surfaces as a CONFLICT and is not
  masked by a knowledge chunk retrieved on the same query (the v2.3 caveat);
* the relevance verdict and sufficiency reason appear in the query-assist audit.

The gate is deterministic and offline. It is downgrade-only — it never invents
evidence or upgrades a refusal — so the frozen verifier, citation, lifecycle,
pack-isolation, and AnswerGuard guarantees are untouched (covered elsewhere).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.relevance_gate import (  # noqa: E402
    LABEL_CONFLICT_MASKED,
    LABEL_METADATA_HEADER_LEAD,
    LABEL_OUT_OF_DOMAIN_FALSE_GROUNDING,
    SufficiencyVerdict,
    assess_relevance,
    is_metadata_header,
    overlap_coefficient,
    content_tokens,
)
from agent.workbench_service import WorkbenchService  # noqa: E402
from slm.assistant_composer import ComposerMode, EvidenceItem  # noqa: E402

_FRIDAY = "the supplier delivery is on Friday afternoon"
_MONDAY = "the supplier delivery is on Monday afternoon"


# ---------- helpers ---------------------------------------------------------

def _know(cid: str, text: str) -> EvidenceItem:
    return EvidenceItem(citation_id=f"src:{cid}", kind="knowledge", text=text,
                        source_name="Pack Notes", domain="microsoft",
                        authority="official")


def _mem(cid: str, text: str) -> EvidenceItem:
    return EvidenceItem(citation_id=f"mem:{cid}", kind="memory", text=text)


def _service(tmp_path, **kwargs) -> WorkbenchService:
    return WorkbenchService(
        ledger_path=str(tmp_path / "ledger.jsonl"),
        queue_path=str(tmp_path / "queue.jsonl"),
        knowledge_path=str(tmp_path / "knowledge.jsonl"),
        **kwargs,
    )


def _write_doc(tmp_path, text: str, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------- lexical primitives ---------------------------------------------

def test_overlap_coefficient_is_length_robust():
    q = set(content_tokens("how do I rename a pandas column"))
    e = set(content_tokens(
        "Use the rename method to change a column label in a pandas DataFrame."))
    # short query whose distinctive terms all appear scores high.
    assert overlap_coefficient(q, e) >= 0.5


def test_metadata_header_detected():
    assert is_metadata_header(
        "Source: Purview Notes Version: 1.0 Domain: microsoft "
        "Authority: official Staleness: current")
    assert not is_metadata_header(
        "Sensitivity labels apply encryption to documents across the tenant.")


# ---------- verdict mapping (unit) -----------------------------------------

def test_strong_match_is_relevant():
    report = assess_relevance(
        _FRIDAY, [_mem("m1", _FRIDAY)], route="memory_only")
    assert report.verdict == SufficiencyVerdict.RELEVANT
    assert report.can_ground is True
    assert report.lead_citation_id == "mem:m1"


def test_moderate_match_is_partial_with_limitation():
    report = assess_relevance(
        "purview sensitivity label encryption threshold",
        [_know("k1",
               "Sensitivity labels apply encryption and watermarks to "
               "documents across the tenant estate.")],
        route="knowledge_only")
    assert report.verdict == SufficiencyVerdict.PARTIAL
    assert report.can_ground is True
    assert report.limitation is not None


# ---------- value-sprint false-grounding rows (must NOT ground) ------------

def test_row6_purview_encryption_does_not_ground_on_powerfx():
    report = assess_relevance(
        "what is the encryption threshold for Purview sensitivity "
        "auto-labelling",
        [_know("powerfx",
               "Use the Filter function in Power Fx to filter a gallery by a "
               "dropdown selection value.")],
        route="knowledge_only")
    assert report.verdict != SufficiencyVerdict.RELEVANT
    assert report.can_ground is False


def test_row7_powerapps_filter_does_not_ground_on_sharepoint():
    report = assess_relevance(
        "how do I filter a PowerApps gallery by a dropdown",
        [_know("sp",
               "SharePoint site governance requires owners to review external "
               "sharing policies quarterly.")],
        route="knowledge_only")
    assert report.verdict != SufficiencyVerdict.RELEVANT
    assert report.can_ground is False


def test_row10_aws_out_of_domain_refuses():
    report = assess_relevance(
        "how do I configure AWS IAM roles for S3 bucket access with Lambda",
        [_know("admin",
               "The Microsoft 365 admin center manages user licenses and "
               "mailbox settings.")],
        route="knowledge_only")
    assert report.verdict == SufficiencyVerdict.NO_SUPPORT
    assert report.label == LABEL_OUT_OF_DOMAIN_FALSE_GROUNDING
    assert report.can_ground is False


def test_row14_purview_protection_does_not_ground_on_breakglass():
    report = assess_relevance(
        "how does Purview protect sensitive documents at rest",
        [_know("bg",
               "Break-glass emergency admin accounts must be excluded from "
               "conditional access policies.")],
        route="knowledge_only")
    assert report.verdict != SufficiencyVerdict.RELEVANT
    assert report.can_ground is False


def test_row16_powerfx_delegation_does_not_ground_on_dlp_audit():
    report = assess_relevance(
        "Power Fx delegation warning on a large data source",
        [_know("dlp",
               "Data loss prevention audit mode logs policy matches without "
               "blocking users.")],
        route="knowledge_only")
    assert report.verdict != SufficiencyVerdict.RELEVANT
    assert report.can_ground is False


def test_row20_prefers_seeded_memory_over_irrelevant_admin():
    report = assess_relevance(
        "what retrieval backend did we choose for v2.2 hybrid search",
        [_know("admin",
               "The Microsoft 365 admin center manages user licenses and "
               "mailbox settings."),
         _mem("decision",
              "We chose the hybrid retrieval backend for v2.2 semantic "
              "search.")],
        route="both")
    assert report.verdict in (SufficiencyVerdict.RELEVANT,
                              SufficiencyVerdict.PARTIAL)
    assert report.lead_citation_id == "mem:decision"
    # the substantive memory leads the re-ordered evidence.
    assert report.ordered_evidence[0].citation_id == "mem:decision"


# ---------- conflict masking (the v2.3 caveat, fixed) ----------------------

def test_conflict_signal_surfaces_even_with_knowledge():
    report = assess_relevance(
        _MONDAY,
        [_know("k1", "The supplier delivery schedule lists Monday afternoon.")],
        route="both", conflict_signal=True)
    assert report.verdict == SufficiencyVerdict.CONFLICT
    assert report.label == LABEL_CONFLICT_MASKED
    assert report.can_ground is False


# ---------- metadata header cannot lead ------------------------------------

def test_metadata_header_cannot_be_lead_evidence():
    report = assess_relevance(
        "what is the sensitivity label encryption policy",
        [_know("hdr",
               "Source: Purview Notes Version: 1.0 Domain: microsoft "
               "Authority: official Staleness: current Section: intro")],
        route="knowledge_only")
    assert report.verdict == SufficiencyVerdict.WEAK_MATCH
    assert report.label == LABEL_METADATA_HEADER_LEAD
    assert report.lead_citation_id is None


# ---------- integration through build_grounding_package --------------------

def test_integration_out_of_domain_refuses_even_when_retrieved(tmp_path):
    service = _service(tmp_path)
    doc = _write_doc(
        tmp_path,
        "# Roles\n\n"
        "Configure access roles for users in the admin center to grant "
        "mailbox and site permissions.\n")
    service.import_knowledge(doc, domain="microsoft", authority="official",
                             source_name="Admin Notes", version="1.0")

    query = "how do I configure AWS IAM roles for S3 bucket access"
    combined = service.query_all(query)
    # evidence WAS retrieved (shared tokens configure/roles/access)...
    assert combined.knowledge_used is True
    # ...but the gate refuses the out-of-domain question rather than grounding.
    package = service.build_grounding_package(query)
    assert package.mode == ComposerMode.REFUSAL
    assert package.refused is True
    assert package.query_audit["relevance"]["verdict"] == "no_support"


def test_integration_conflict_not_masked_by_knowledge(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")
    doc = _write_doc(
        tmp_path,
        "# Schedule\n\n"
        "The supplier delivery schedule lists Monday afternoon as the slot.\n")
    service.import_knowledge(doc, domain="microsoft", authority="official",
                             source_name="Schedule Notes", version="1.0")

    combined = service.query_all(_MONDAY)
    # a knowledge chunk is retrievable on this query (the masking condition)...
    assert combined.knowledge_used is True
    # ...yet the rejected Friday near-miss surfaces as a conflict, not a
    # confident grounding.
    package = service.build_grounding_package(_MONDAY)
    assert package.mode == ComposerMode.CONFLICT_EXPLANATION
    assert package.refused is True


def test_integration_audit_carries_relevance_verdict(tmp_path):
    service = _service(tmp_path)
    service.add_memory(_FRIDAY, source="seed")

    package = service.build_grounding_package(_FRIDAY)
    assert package.mode == ComposerMode.GROUNDED
    relevance = package.query_audit["relevance"]
    assert relevance["verdict"] == "relevant"
    assert relevance["sufficiency_reason"]
