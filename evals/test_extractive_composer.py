"""Tests for the v2.6 extractive multi-chunk composer.

The extractive composer is an opt-in *rendering* layer over the same frozen
``GroundingPackage`` the template composer renders. Where the template echoes
each evidence item whole, the extractive composer quotes the most query-relevant
verbatim span from each allowed source and binds each span to its citation id.

The safety contract these tests pin down:

* it can only render what is already grounded -- it never adds a source, so for
  a given package its citation set equals the template's (no grounding drift);
* every emitted span is a *verbatim substring* of the evidence item it cites --
  no paraphrase, no merged chunks, no bridging text;
* :func:`answer_guard.check_answer` rejects any span that is not substring-
  supported by its cited evidence, and :func:`enforce` recomposes to the safe
  template on rejection;
* non-grounded modes (refusal, conflict, model-prior) are delegated to the
  template verbatim, so those paths are byte-identical to today;
* a single-source grounded answer still yields one span (never empty).

Packages are built directly here so the cases are precise and offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from slm.answer_guard import UNSUPPORTED_SPAN, check_answer, enforce  # noqa: E402
from slm.assistant_composer import (  # noqa: E402
    AnswerSpan,
    ComposedAnswer,
    ComposerMode,
    EvidenceItem,
    ExtractiveMultiChunkComposer,
    GroundingPackage,
    TemplateComposer,
)


def _evidence(citation_id: str, text: str, source_name: str) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id, kind="knowledge", text=text,
        source_name=source_name)


def _grounded_package(query: str, evidence) -> GroundingPackage:
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
    )


def _refusal_package(query: str) -> GroundingPackage:
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


# 1. multi-source grounding yields >=2 verbatim spans bound to distinct ids. -

def test_multi_source_yields_distinct_bound_spans():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    answer = ExtractiveMultiChunkComposer().compose(pkg)

    assert answer.mode == ComposerMode.GROUNDED
    assert answer.refused is False
    assert answer.composer_backend == "extractive"
    # At least one span from each distinct source.
    cited = {s.citation_id for s in answer.spans}
    assert cited == {"src:a", "src:b"}
    assert len(answer.spans) >= 2
    # Every span is a verbatim substring of the evidence it cites.
    by_id = {e.citation_id: e.text for e in pkg.evidence}
    for span in answer.spans:
        assert span.text in by_id[span.citation_id]


# 2. spans select the relevant sentence, not the whole chunk. ---------------

def test_spans_are_narrower_than_whole_chunk():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    answer = ExtractiveMultiChunkComposer().compose(pkg)

    # The off-topic weather sentence is not selected; the validation sentence is.
    text = " ".join(s.text for s in answer.spans)
    assert "validates a typed model" in text
    assert "weather" not in text
    # And the rendered body is shorter than echoing the full chunk.
    assert len(answer.text) < len(_DOC_A) + len("Based on grounded evidence:")


# 3. the extractive answer passes the guard. --------------------------------

def test_extractive_answer_passes_guard():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a"),
         _evidence("src:b", _DOC_B, "doc-b")],
    )

    answer = ExtractiveMultiChunkComposer().compose(pkg)
    report = check_answer(pkg, answer)

    assert report.ok is True
    assert report.violations == []


# 4. a fabricated span is rejected and enforce falls back to the template. --

def test_fabricated_span_rejected_and_recomposed():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )
    rogue = ComposedAnswer(
        text="Based on grounded evidence:\n  - Pydantic also cures diseases. [src:a]",
        mode=ComposerMode.GROUNDED,
        citations=["src:a"],
        composer_backend="extractive",
        model_prior_labelled=False,
        informational_only=False,
        refused=False,
        spans=[AnswerSpan(text="Pydantic also cures diseases.",
                          citation_id="src:a", source_name="doc-a")],
    )

    report = check_answer(pkg, rogue)
    assert report.ok is False
    assert any(v.code == UNSUPPORTED_SPAN for v in report.violations)

    safe, enforced = enforce(pkg, rogue)
    assert enforced.verdict == "REJECT"
    # The recomposed safe answer is the template's, which has no spans.
    assert safe.composer_backend == TemplateComposer().name
    assert safe.spans == []
    assert "cures diseases" not in safe.text


# 5. a span citing an unknown id is rejected. -------------------------------

def test_span_citing_unknown_id_rejected():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )
    answer = ComposedAnswer(
        text="Based on grounded evidence:\n  - It raises a ValidationError on bad input. [src:ghost]",
        mode=ComposerMode.GROUNDED,
        citations=["src:a"],
        composer_backend="extractive",
        model_prior_labelled=False,
        informational_only=False,
        refused=False,
        spans=[AnswerSpan(text="It raises a ValidationError on bad input.",
                          citation_id="src:ghost", source_name="doc-a")],
    )

    report = check_answer(pkg, answer)
    assert report.ok is False
    assert any(v.code == UNSUPPORTED_SPAN for v in report.violations)


# 6. non-grounded modes are byte-identical to the template. -----------------

def test_refusal_is_identical_to_template():
    pkg = _refusal_package("the zebra quantum teapot orbits")

    template_answer = TemplateComposer().compose(pkg)
    extractive_answer = ExtractiveMultiChunkComposer().compose(pkg)

    assert extractive_answer.text == template_answer.text
    assert extractive_answer.mode == ComposerMode.REFUSAL
    assert extractive_answer.refused is True
    assert extractive_answer.citations == template_answer.citations
    assert extractive_answer.spans == []
    # The selected backend is still recorded as extractive.
    assert extractive_answer.composer_backend == "extractive"


# 7. a single grounded source still yields exactly one (non-empty) span. ----

def test_single_source_yields_one_span():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    answer = ExtractiveMultiChunkComposer().compose(pkg)

    assert len(answer.spans) >= 1
    assert all(s.citation_id == "src:a" for s in answer.spans)
    assert answer.text.strip() != ""


# 8. an evidence item with no query overlap still contributes a span. -------

def test_zero_overlap_source_still_cited():
    overlap = _evidence("src:a", _DOC_A, "doc-a")
    off_topic = _evidence(
        "src:b", "Office plants need watering twice a week.", "doc-b")
    pkg = _grounded_package("how does pydantic validate input",
                            [overlap, off_topic])

    answer = ExtractiveMultiChunkComposer().compose(pkg)

    cited = {s.citation_id for s in answer.spans}
    # The off-topic source is never silently dropped: it still gets a span.
    assert cited == {"src:a", "src:b"}
    by_id = {e.citation_id: e.text for e in pkg.evidence}
    for span in answer.spans:
        assert span.text in by_id[span.citation_id]


# 9. the template composer emits no spans (guard no-op preserved). ----------

def test_template_composer_emits_no_spans():
    pkg = _grounded_package(
        "how does pydantic validate input",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    answer = TemplateComposer().compose(pkg)

    assert answer.spans == []
    # With no spans the new guard check is a no-op and the answer is accepted.
    assert check_answer(pkg, answer).ok is True


# 10. max_spans_per_item bounds spans per source. ---------------------------

def test_max_spans_per_item_is_respected():
    pkg = _grounded_package(
        "pydantic validation error input model construction",
        [_evidence("src:a", _DOC_A, "doc-a")],
    )

    answer = ExtractiveMultiChunkComposer(max_spans_per_item=1).compose(pkg)

    assert len([s for s in answer.spans if s.citation_id == "src:a"]) == 1


# 11. an invalid max_spans_per_item is rejected at construction. ------------

def test_invalid_max_spans_rejected():
    try:
        ExtractiveMultiChunkComposer(max_spans_per_item=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for max_spans_per_item=0")
