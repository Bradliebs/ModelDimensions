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
    report_fluency_metrics,
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
    # v2.8: sections are disjoint, but coverage is preserved -- every allowed
    # source is cited at least once across the factual sections (not necessarily
    # in the Evidence section, which now carries only sources not surfaced above).
    cited_ids = {span.citation_id
                 for section in factual for span in section.spans}
    assert cited_ids == {"src:a", "src:b"}
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


# -- v2.7.1: internal diagnostics must not leak into client-facing Risks. -----

def _risks_section(answer):
    return next(s for s in answer.report.sections if s.title == "Risks")


# 12. the relevance-gate verdict trace never appears in Risks. ----------------

def test_risks_excludes_relevance_gate_trace():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
        cautions=[
            "Relevance gate: relevant — strong substantive match (overlap 0.80)",
        ],
    )

    answer = ConsultantReportComposer().compose(pkg)
    risks = _risks_section(answer)

    assert "Relevance gate" not in risks.judgement_text
    # The package/audit still carries the diagnostic (it is a display filter).
    assert any("Relevance gate" in c for c in pkg.cautions)
    assert check_answer(pkg, answer).ok


# 13. an overlap score never appears in Risks. --------------------------------

def test_risks_excludes_overlap_score():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
        cautions=[
            "Relevance gate: partial — partial match (overlap 0.42); answer is "
            "limited to what the evidence directly supports",
        ],
    )

    answer = ConsultantReportComposer().compose(pkg)
    risks = _risks_section(answer)

    assert "overlap" not in risks.judgement_text.lower()
    assert "0.42" not in risks.judgement_text


# 14. a legitimate stale-source caution still surfaces in Risks. --------------

def test_risks_keeps_stale_source_caution():
    stale = ("Source 'doc-a' is marked stale (staleness policy: stale); it may "
             "be out of date.")
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
        cautions=[
            stale,
            "Relevance gate: relevant — strong substantive match (overlap 0.80)",
        ],
    )

    answer = ConsultantReportComposer().compose(pkg)
    risks = _risks_section(answer)

    assert "marked stale" in risks.judgement_text
    assert "Relevance gate" not in risks.judgement_text
    assert risks.judgement_text.startswith(JUDGEMENT_LABEL)


# 15. a legitimate low-authority caution still surfaces in Risks. -------------

def test_risks_keeps_authority_caution():
    authority = ("Source 'doc-a' has community authority; verify before relying "
                 "on it.")
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
        cautions=[authority],
    )

    answer = ConsultantReportComposer().compose(pkg)
    risks = _risks_section(answer)

    assert "community authority" in risks.judgement_text


# 16. a conflict near-miss still surfaces as an open question. ----------------

def test_open_questions_keeps_conflict_signal():
    conflict = [_evidence("mem:1", "An older near-miss memory text.", "mem")]
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
        conflict=conflict,
    )

    answer = ConsultantReportComposer().compose(pkg)
    open_q = next(s for s in answer.report.sections
                  if s.title == "Open questions")

    assert "near-miss" in open_q.judgement_text.lower()
    assert open_q.judgement_text.startswith(JUDGEMENT_LABEL)
    assert check_answer(pkg, answer).ok


# -- v2.8 report fluency layer (readability only; grounding unchanged) --------

# A source whose top-overlap sentence is clean but which also carries a trailing
# truncated fragment that overlaps the query.
_DOC_FRAG_ALT = (
    "Pydantic validates input on construction. "
    "Off-topic trivia about the weather. "
    "Pydantic validate"
)
# A source whose only candidate sentence is a truncated fragment.
_DOC_FRAG_ONLY = "Pydantic validate input"


def _factual_sections(answer):
    return [s for s in answer.report.sections if s.kind == SECTION_FACTUAL]


def _judgement_sections(answer):
    return [s for s in answer.report.sections if s.kind == SECTION_JUDGEMENT]


def _factual_span_keys(answer):
    return [(span.citation_id, " ".join(span.text.split()))
            for section in _factual_sections(answer)
            for span in section.spans]


# 17. the same span is never repeated across the factual sections. -----------

def test_factual_sections_are_disjoint():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    keys = _factual_span_keys(answer)

    assert len(keys) == len(set(keys)), keys
    assert report_fluency_metrics(answer.report)[0] == 0  # duplicate_span_count
    assert report_fluency_metrics(answer.report)[2] == 0  # section_overlap_count


# 18. the executive span is not echoed in Current state or Evidence. ---------

def test_executive_span_excluded_from_other_sections():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    exec_section = next(s for s in answer.report.sections
                        if s.title == "Executive summary")
    assert exec_section.spans, "executive summary should carry one span"
    exec_key = (exec_section.spans[0].citation_id,
                " ".join(exec_section.spans[0].text.split()))

    others = [s for s in _factual_sections(answer)
              if s.title != "Executive summary"]
    other_keys = [(span.citation_id, " ".join(span.text.split()))
                  for s in others for span in s.spans]
    assert exec_key not in other_keys


# 19. every allowed source is still cited at least once (coverage). ----------

def test_every_source_cited_after_dedup():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b"),
         _evidence("src:c", "Pydantic also supports nested model validation.",
                   "doc-c")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    cited = {cid for cid, _ in _factual_span_keys(answer)}

    assert cited == {"src:a", "src:b", "src:c"}
    assert check_answer(pkg, answer).ok


# 20. a clean sentence is preferred over a fragment for the same source. ------

def test_clean_span_preferred_over_fragment():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_FRAG_ALT, "doc-a")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    texts = [span.text for section in _factual_sections(answer)
             for span in section.spans]

    assert "Pydantic validates input on construction." in texts
    assert "Pydantic validate" not in texts  # the fragment is not surfaced
    assert report_fluency_metrics(answer.report)[1] == 0  # truncated_span_count


# 21. coverage beats fluency: a fragment-only source is retained and counted. -

def test_fragment_only_source_retained_for_coverage():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_FRAG_ONLY, "doc-a")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    cited = {cid for cid, _ in _factual_span_keys(answer)}

    assert cited == {"src:a"}  # the only source is still cited
    assert report_fluency_metrics(answer.report)[1] == 1  # one truncated span
    assert check_answer(pkg, answer).ok  # still a verbatim, guard-clean span


# 22. judgement placeholders are distinctly worded, not one boilerplate. ------

def test_judgement_placeholders_are_varied():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    placeholders = [s.judgement_text for s in _judgement_sections(answer)
                    if s.is_placeholder]

    # With no cautions/conflict every judgement section is a placeholder.
    assert len(placeholders) == 6
    assert len(set(placeholders)) == len(placeholders)  # all distinct


# 23. every judgement section carries the label exactly once and is uncited. --

def test_judgement_label_appears_exactly_once():
    import re

    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    answer = ConsultantReportComposer().compose(pkg)
    for section in _judgement_sections(answer):
        assert section.judgement_text.count(JUDGEMENT_LABEL) == 1
        assert re.search(_CITATION_RE_TEXT, section.judgement_text) is None


# 24. fluency metrics are populated in report mode, zero/absent otherwise. ----

def test_fluency_metrics_inert_outside_report_mode():
    assert report_fluency_metrics(None) == (0, 0, 0, 0)

    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )
    template_answer = TemplateComposer().compose(pkg)
    assert template_answer.report is None
    assert report_fluency_metrics(template_answer.report) == (0, 0, 0, 0)

    report_answer = ConsultantReportComposer().compose(pkg)
    metrics = report_fluency_metrics(report_answer.report)
    # placeholder count is populated; duplicate/overlap stay clean.
    assert metrics[3] == 6
    assert metrics[0] == 0 and metrics[2] == 0

