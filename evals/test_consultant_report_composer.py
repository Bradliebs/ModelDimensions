"""Tests for the v2.7 consultant report composer.

The consultant report composer is an opt-in *structuring* layer over the same
frozen ``GroundingPackage`` the template composer renders. It is **not** an
abstractive report writer: it partitions a report into two structurally typed
section kinds and never fabricates a conclusion.

The safety contract these tests pin down:

* factual sections (Executive summary, Current state, Evidence) carry only
  citation-bound, verbatim spans -- each span is a substring of the evidence it
  cites, so the report can never cite a source the template would not;
* judgement sections (Risks, Options, Recommendation, ...) are labelled with the
  mandatory ``[JUDGEMENT ...]`` prefix and contain no citation marker;
* the :class:`ReportStructure` carrier is the source of truth for section kind;
  the guard reads it, never the rendered prose;
* :func:`answer_guard.check_answer` rejects an unlabelled or cited judgement
  section and an unsupported factual claim, and :func:`enforce` recomposes to
  the safe template on rejection;
* an unsupported recommendation degrades to a labelled placeholder, never a
  grounded finding;
* non-grounded modes (refusal, conflict, model-prior) are delegated to the
  template verbatim, so those paths are byte-identical to today.

Packages are built directly here so the cases are precise and offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from slm.answer_guard import (  # noqa: E402
    CITED_JUDGEMENT,
    UNLABELLED_JUDGEMENT,
    UNSUPPORTED_REPORT_CLAIM,
    check_answer,
    enforce,
)
from slm.assistant_composer import (  # noqa: E402
    JUDGEMENT_LABEL,
    SECTION_FACTUAL,
    SECTION_JUDGEMENT,
    AnswerSpan,
    ComposerMode,
    ConsultantReportComposer,
    EvidenceItem,
    GroundingPackage,
    ReportSection,
    ReportStructure,
    TemplateComposer,
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


def _refusal_package(query) -> GroundingPackage:
    return GroundingPackage(
        query=query,
        mode=ComposerMode.REFUSAL,
        memory_used=False,
        knowledge_used=False,
        model_prior_used=False,
        informational_only=False,
        refused=True,
        route="refusal",
    )


def _model_prior_package(query) -> GroundingPackage:
    return GroundingPackage(
        query=query,
        mode=ComposerMode.MODEL_PRIOR_LABELLED,
        memory_used=False,
        knowledge_used=False,
        model_prior_used=True,
        informational_only=False,
        refused=False,
        route="model_prior",
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

_SECTION_TITLES = [
    "Executive summary", "Current state", "Evidence", "Risks", "Options",
    "Recommendation", "Assumptions", "Open questions", "Next actions",
]


# 1. a grounded report has all nine sections, partitioned by kind. -----------

def test_report_has_all_sections_partitioned():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    answer = ConsultantReportComposer().compose(pkg)

    assert answer.mode == ComposerMode.GROUNDED
    assert answer.composer_backend == "consultant"
    assert answer.report is not None
    titles = [s.title for s in answer.report.sections]
    assert titles == _SECTION_TITLES
    factual = {s.title for s in answer.report.sections
               if s.kind == SECTION_FACTUAL}
    judgement = {s.title for s in answer.report.sections
                 if s.kind == SECTION_JUDGEMENT}
    assert factual == {"Executive summary", "Current state", "Evidence"}
    assert judgement == {"Risks", "Options", "Recommendation", "Assumptions",
                         "Open questions", "Next actions"}


# 2. factual sections are 100% citation-bound verbatim spans. ----------------

def test_factual_sections_are_verbatim_and_cited():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    by_id = {e.citation_id: e.text for e in pkg.evidence}

    factual = [s for s in answer.report.sections if s.kind == SECTION_FACTUAL]
    saw_span = False
    for section in factual:
        for span in section.spans:
            saw_span = True
            assert span.citation_id in by_id
            assert span.text in by_id[span.citation_id]
    assert saw_span
    # Evidence section covers every allowed source at least once.
    evidence_section = next(s for s in factual if s.title == "Evidence")
    assert {s.citation_id for s in evidence_section.spans} == {"src:a", "src:b"}
    # No grounding drift: report cites exactly the template's allowed ids.
    template = TemplateComposer().compose(pkg)
    assert sorted(answer.citations) == sorted(template.citations)


# 3. judgement sections are labelled and contain no citation marker. ---------

def test_judgement_sections_labelled_and_uncited():
    import re

    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    answer = ConsultantReportComposer().compose(pkg)

    judgement = [s for s in answer.report.sections
                 if s.kind == SECTION_JUDGEMENT]
    assert judgement
    for section in judgement:
        assert section.judgement_text.startswith(JUDGEMENT_LABEL)
        assert section.spans == []
        assert re.search(_CITATION_RE_TEXT, section.judgement_text) is None


# 4. a clean report passes the guard. ----------------------------------------

def test_report_passes_guard():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    report = check_answer(pkg, answer)

    assert report.ok, report.to_dict()


# 5. an unlabelled judgement section is rejected and enforce falls back. -----

def test_unlabelled_judgement_rejected_and_enforced():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )
    answer = ConsultantReportComposer().compose(pkg)
    # Tamper: drop the mandatory label from a judgement section.
    bad_sections = []
    for section in answer.report.sections:
        if section.title == "Recommendation":
            bad_sections.append(ReportSection(
                title=section.title, kind=SECTION_JUDGEMENT,
                judgement_text="We should rewrite everything immediately."))
        else:
            bad_sections.append(section)
    answer.report = ReportStructure(sections=bad_sections)

    report = check_answer(pkg, answer)
    assert not report.ok
    assert any(v.code == UNLABELLED_JUDGEMENT for v in report.violations)

    safe, enforced_report = enforce(pkg, answer)
    assert not enforced_report.ok
    assert safe.composer_backend == "template"
    assert safe.report is None


# 6. a judgement section that smuggles a citation is rejected. ---------------

def test_cited_judgement_rejected():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )
    answer = ConsultantReportComposer().compose(pkg)
    bad_sections = []
    for section in answer.report.sections:
        if section.title == "Options":
            bad_sections.append(ReportSection(
                title=section.title, kind=SECTION_JUDGEMENT,
                judgement_text=f"{JUDGEMENT_LABEL}\nPrefer option [src:a]."))
        else:
            bad_sections.append(section)
    answer.report = ReportStructure(sections=bad_sections)

    report = check_answer(pkg, answer)
    assert not report.ok
    assert any(v.code == CITED_JUDGEMENT for v in report.violations)


# 7. a fabricated factual claim is rejected as an unsupported report claim. --

def test_unsupported_factual_claim_rejected():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )
    answer = ConsultantReportComposer().compose(pkg)
    bad_sections = []
    for section in answer.report.sections:
        if section.title == "Current state":
            bad_sections.append(ReportSection(
                title=section.title, kind=SECTION_FACTUAL,
                spans=[AnswerSpan(text="Pydantic deletes your database.",
                                  citation_id="src:a", source_name="doc-a")]))
        else:
            bad_sections.append(section)
    answer.report = ReportStructure(sections=bad_sections)

    report = check_answer(pkg, answer)
    assert not report.ok
    assert any(v.code == UNSUPPORTED_REPORT_CLAIM for v in report.violations)


# 8. the recommendation degrades to a labelled placeholder, never fabricated. -

def test_recommendation_is_labelled_placeholder():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    rec = next(s for s in answer.report.sections
               if s.title == "Recommendation")

    assert rec.kind == SECTION_JUDGEMENT
    assert rec.judgement_text.startswith(JUDGEMENT_LABEL)
    assert "author judgement required" in rec.judgement_text


# 9. cautions and conflicts surface as labelled risks / open questions. ------

def test_cautions_and_conflicts_become_labelled_judgement():
    conflict = [_evidence("mem:1", "An older near-miss memory text.", "mem")]
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
        cautions=["This source is stale (last reviewed 2019)."],
        conflict=conflict,
    )

    answer = ConsultantReportComposer().compose(pkg)
    risks = next(s for s in answer.report.sections if s.title == "Risks")
    open_q = next(s for s in answer.report.sections
                  if s.title == "Open questions")

    assert risks.judgement_text.startswith(JUDGEMENT_LABEL)
    assert "stale" in risks.judgement_text.lower()
    assert "near-miss" in risks.judgement_text.lower()
    assert open_q.judgement_text.startswith(JUDGEMENT_LABEL)
    # The whole answer still passes the guard (stale caution surfaced, no
    # citation leaked into judgement).
    assert check_answer(pkg, answer).ok


# 10. non-grounded modes are byte-identical to the template. -----------------

def test_refusal_is_identical_to_template():
    pkg = _refusal_package("the zebra quantum teapot orbits")

    template_answer = TemplateComposer().compose(pkg)
    report_answer = ConsultantReportComposer().compose(pkg)

    assert report_answer.text == template_answer.text
    assert report_answer.mode == ComposerMode.REFUSAL
    assert report_answer.refused is True
    assert report_answer.citations == template_answer.citations
    assert report_answer.report is None
    assert report_answer.spans == []
    assert report_answer.composer_backend == "consultant"


def test_model_prior_is_identical_to_template():
    pkg = _model_prior_package("speculate about something ungrounded")

    template_answer = TemplateComposer().compose(pkg)
    report_answer = ConsultantReportComposer().compose(pkg)

    assert report_answer.text == template_answer.text
    assert report_answer.model_prior_labelled is True
    assert report_answer.report is None
    assert check_answer(pkg, report_answer).ok


# 11. no grounding drift: grounded citation set equals the template's. -------

def test_no_grounding_drift_vs_template():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    template = TemplateComposer().compose(pkg)
    report = ConsultantReportComposer().compose(pkg)

    assert sorted(report.citations) == sorted(template.citations)
    # Every cited id in the report is an allowed evidence id (no invention).
    assert set(report.citations) <= pkg.allowed_citation_ids
