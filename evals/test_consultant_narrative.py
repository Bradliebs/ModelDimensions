"""Tests for the v2.9 consultant narrative layer.

The narrative layer makes the opt-in consultant report read more like a
professional deliverable — executive framing, section-level connective wording,
a factual->judgement transition, and evidence-sufficiency-aware recommendation
wording — **without touching the evidence contract**. It adds structural framing
(fact-free section intros on the carrier) and a pure narrative-metrics function;
it changes no retrieval, ranking, source selection, grounding, memory, or guard
semantics.

These tests pin the safety contract the narrative layer must preserve:

* no citation drift versus the template/report citation set;
* no grounding drift (cited ids stay within the package's allowed ids);
* judgement sections remain labelled and uncited;
* narrative framing introduces no unsupported factual claim (every intro is the
  canonical, fact-free frame, audited by ``report_narrative_metrics``);
* the recommendation degrades safely as evidence sufficiency changes, always a
  labelled placeholder, never a fabricated directive;
* the v2.8 fluency metrics still work unchanged;
* the TemplateComposer stays the default and report mode stays opt-in;
* a narrative report still passes the guard with verdict ACCEPT.

Packages are built directly here so the cases are precise and offline.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from slm.answer_guard import check_answer  # noqa: E402
from slm.assistant_composer import (  # noqa: E402
    JUDGEMENT_LABEL,
    SECTION_FACTUAL,
    SECTION_JUDGEMENT,
    ComposerMode,
    ConsultantReportComposer,
    EvidenceItem,
    GroundingPackage,
    ReportSection,
    ReportStructure,
    TemplateComposer,
    report_fluency_metrics,
    report_narrative_metrics,
)

_CITATION_RE_TEXT = r"\[(mem|src):"


def _evidence(citation_id: str, text: str, source_name: str) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id, kind="knowledge", text=text,
        source_name=source_name)


def _grounded_package(query, evidence, *, cautions=None, conflict=None,
                      historical_note=None) -> GroundingPackage:
    return GroundingPackage(
        query=query,
        mode=ComposerMode.GROUNDED,
        memory_used=False,
        knowledge_used=True,
        model_prior_used=False,
        informational_only=False,
        refused=False,
        route="knowledge",
        evidence=list(evidence),
        conflict_context=list(conflict or []),
        cautions=list(cautions or []),
        historical_note=historical_note,
    )


_DOC_A = (
    "Pydantic validates a typed model on construction. "
    "It raises a ValidationError on bad input. "
    "Unrelated trivia about the weather goes here."
)
_DOC_B = (
    "FastAPI uses Pydantic models to validate request bodies. "
    "The dependency injection system resolves shared resources. "
    "A note about office plants that is off topic."
)


def _factual(answer):
    return [s for s in answer.report.sections if s.kind == SECTION_FACTUAL]


def _judgement(answer):
    return [s for s in answer.report.sections if s.kind == SECTION_JUDGEMENT]


# 1. no citation drift versus the template / non-narrative report. -----------

def test_no_citation_drift_vs_template_and_plain_report():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    template = TemplateComposer().compose(pkg)
    plain = ConsultantReportComposer(narrative=False).compose(pkg)
    narrative = ConsultantReportComposer().compose(pkg)  # narrative on by default

    assert sorted(narrative.citations) == sorted(template.citations)
    assert sorted(narrative.citations) == sorted(plain.citations)


# 2. no grounding drift: cited ids stay within the allowed set. --------------

def test_no_grounding_drift():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    assert set(answer.citations) <= pkg.allowed_citation_ids

    # Factual spans are identical to the non-narrative report — narrative framing
    # does not alter which spans are surfaced.
    plain = ConsultantReportComposer(narrative=False).compose(pkg)

    def span_keys(a):
        return [(sp.citation_id, sp.text)
                for s in _factual(a) for sp in s.spans]

    assert span_keys(answer) == span_keys(plain)


# 3. judgement sections remain labelled. -------------------------------------

def test_judgement_remains_labelled():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    for section in _judgement(answer):
        assert (section.judgement_text or "").startswith(JUDGEMENT_LABEL)


# 4. judgement sections remain uncited. --------------------------------------

def test_judgement_remains_uncited():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    for section in _judgement(answer):
        assert re.search(_CITATION_RE_TEXT, section.judgement_text or "") is None
        assert section.spans == []
    # The narrative transition intro on the first judgement section is also
    # uncited and fact-free.
    risks = next(s for s in answer.report.sections if s.title == "Risks")
    assert risks.intro is not None
    assert re.search(_CITATION_RE_TEXT, risks.intro) is None


# 5. narrative framing introduces no unsupported factual claim. --------------

def test_narrative_framing_introduces_no_unsupported_claim():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    metrics = report_narrative_metrics(answer.report)

    # Every section intro is the canonical, fact-free frame: zero free-text
    # framing leaked in.
    assert metrics.unsupported_narrative_claim_count == 0
    # Framing is actually present (executive + current + evidence + transition).
    assert metrics.narrative_transition_count == 4
    # No intro carries a citation marker.
    for section in answer.report.sections:
        if section.intro:
            assert re.search(_CITATION_RE_TEXT, section.intro) is None


def test_tampered_intro_is_counted_as_unsupported():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )
    answer = ConsultantReportComposer().compose(pkg)
    tampered = []
    for section in answer.report.sections:
        if section.title == "Current state":
            tampered.append(ReportSection(
                title=section.title, kind=section.kind, spans=section.spans,
                intro="Pydantic is the best validation library ever built."))
        else:
            tampered.append(section)
    answer.report = ReportStructure(sections=tampered)

    metrics = report_narrative_metrics(answer.report)
    assert metrics.unsupported_narrative_claim_count == 1


# 6. recommendation degrades safely with evidence sufficiency. ---------------

def test_recommendation_degrades_with_evidence_sufficiency():
    one = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )
    two = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    def rec(answer):
        return next(s for s in answer.report.sections
                    if s.title == "Recommendation")

    rec_one = rec(ConsultantReportComposer().compose(one))
    rec_two = rec(ConsultantReportComposer().compose(two))

    # Both are labelled, uncited placeholders that never fabricate a directive.
    for section in (rec_one, rec_two):
        assert section.is_placeholder is True
        assert section.judgement_text.startswith(JUDGEMENT_LABEL)
        assert re.search(_CITATION_RE_TEXT, section.judgement_text) is None
        assert "author judgement required" in section.judgement_text

    # The wording adapts to how many sources back the report.
    assert "2 evidence-bound sources" in rec_two.judgement_text
    assert rec_one.judgement_text != rec_two.judgement_text


def test_recommendation_names_gap_when_evidence_absent():
    # A degenerate grounded package with no evidence: the recommendation must
    # name the gap rather than fill it.
    empty = _grounded_package("a grounded query with no evidence", [])
    answer = ConsultantReportComposer().compose(empty)
    rec = next(s for s in answer.report.sections if s.title == "Recommendation")

    assert rec.is_placeholder is True
    assert rec.judgement_text.startswith(JUDGEMENT_LABEL)
    assert "gap" in rec.judgement_text.lower()


# 7. existing v2.8 fluency metrics still work. -------------------------------

def test_v28_fluency_metrics_still_work():
    assert report_fluency_metrics(None) == (0, 0, 0, 0)

    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )
    answer = ConsultantReportComposer().compose(pkg)
    duplicate, truncated, overlap, placeholders = report_fluency_metrics(
        answer.report)

    assert duplicate == 0
    assert overlap == 0
    assert placeholders == 6  # unchanged by narrative framing


# 8. narrative metrics are inert outside report mode. ------------------------

def test_narrative_metrics_inert_outside_report_mode():
    empty = report_narrative_metrics(None)
    assert empty.narrative_transition_count == 0
    assert empty.unsupported_narrative_claim_count == 0
    assert empty.labelled_judgement_count == 0
    assert empty.unlabelled_judgement_count == 0
    assert empty.recommendation_placeholder_count == 0
    assert empty.evidence_gap_named_count == 0
    assert empty.consultant_readability_score == 0.0

    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )
    template_answer = TemplateComposer().compose(pkg)
    assert template_answer.report is None
    assert report_narrative_metrics(template_answer.report).to_dict() == \
        empty.to_dict()


# 9. TemplateComposer remains the default; report stays opt-in. --------------

def test_template_is_default_and_report_is_opt_in(tmp_path):
    from agent.workbench_service import WorkbenchService

    service = WorkbenchService(
        ledger_path=str(tmp_path / "ledger.jsonl"),
        queue_path=str(tmp_path / "queue.jsonl"),
        knowledge_path=str(tmp_path / "knowledge.jsonl"),
    )
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    # The default composer (no composer argument) is the template: it produces
    # the template backend and never a report. The consultant report — and its
    # narrative layer — is reachable only when explicitly selected.
    default_answer = service.compose_answer(pkg)
    assert default_answer.composer_backend == "template"
    assert default_answer.report is None

    report_answer = service.compose_answer(pkg, ConsultantReportComposer())
    assert report_answer.composer_backend == "consultant"
    assert report_answer.report is not None


# 10. a full narrative report passes the guard (ACCEPT). ---------------------

def test_narrative_report_passes_guard():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
        cautions=["Source 'doc-a' is marked stale (staleness policy: stale)."],
    )

    answer = ConsultantReportComposer().compose(pkg)
    report = check_answer(pkg, answer)
    assert report.ok, report.to_dict()

    metrics = report_narrative_metrics(answer.report)
    assert metrics.unlabelled_judgement_count == 0
    assert metrics.labelled_judgement_count == 6
    assert metrics.recommendation_placeholder_count == 1
    assert metrics.evidence_gap_named_count >= 1
    assert metrics.consultant_readability_score == 1.0


# 11. narrative framing renders into the prose, not into spans. --------------

def test_framing_renders_but_is_not_a_span():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    # The transition framing appears in the rendered prose.
    assert "author judgement: labelled, uncited" in answer.text
    # But it is not a span (factual sections still carry only cited spans).
    for section in _factual(answer):
        for span in section.spans:
            assert span.text != answer.report.sections[3].intro
