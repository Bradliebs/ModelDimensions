"""Direct tests for ``src/agent/v1_answer_verifier.py``.

The pipeline-level tests in ``test_answer_pipeline.py`` cover the
verifier indirectly. These tests pin the unit-level contract — Stage A
(token coverage) and Stage B (numeric/date strict-match) — so a future
threshold tweak doesn't silently change behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.v1_answer_verifier import verify


def test_lexical_paraphrase_passes():
    """Reordering and added qualifiers around the cell's nouns must pass.

    The verifier is exact-token-set: morphological variants ("stripes" vs
    "striped", "Africa" vs "African") are NOT considered the same token.
    Phi-3, when prompted to use the FACTS, tends to reuse the same nouns,
    so this is the realistic paraphrase shape we need to admit.
    """
    cells = ["Zebras have black and white stripes and live in Africa."]
    answer = "Zebras live in Africa and have black and white stripes."
    v = verify(answer, cells, question="What do zebras look like?")
    assert v.grounded is True
    assert v.coverage >= 0.5
    assert v.uncited_numerics == []


def test_drifted_answer_fails_stage_a():
    cells = ["Paris is the capital of France and sits on the Seine."]
    answer = (
        "Quantum entanglement involves correlated photon polarizations "
        "across spacelike separated detectors."
    )
    v = verify(answer, cells, question="Tell me about Paris.")
    assert v.grounded is False
    assert v.coverage < 0.5


def test_uncited_year_fails_stage_b():
    """Token coverage is fine but the year is confabulated."""
    cells = ["Paris is the capital of France and sits on the Seine."]
    answer = "Paris is the capital of France and was founded in 1872."
    v = verify(answer, cells, question="When was Paris founded?")
    assert v.grounded is False
    assert "1872" in v.uncited_numerics
    assert "uncited" in v.reason.lower() or "numeric" in v.reason.lower()


def test_year_present_in_cells_passes_stage_b():
    cells = ["The treaty was signed in 1872 by both delegations."]
    answer = "The treaty was signed in 1872."
    v = verify(answer, cells, question="When was the treaty signed?")
    assert v.grounded is True
    assert v.uncited_numerics == []


def test_question_year_does_not_count_as_evidence():
    """A year that appears in the question is filtered out of Stage B."""
    cells = ["Paris hosts many cultural events each year."]
    # Stage A passes (Paris/hosts/cultural/events/year all in cell).
    # Stage B: "1969" in answer also in question -> filtered, no rejection.
    answer = "Paris hosts cultural events each year."
    v = verify(answer, cells, question="What happened in Paris in 1969?")
    assert v.grounded is True
    assert v.uncited_numerics == []


def test_uncited_month_fails_stage_b():
    cells = ["The conference was held in Berlin during 2003."]
    answer = "The conference was held in Berlin in March 2003."
    v = verify(answer, cells, question="Where was the conference held?")
    assert v.grounded is False
    assert "march" in v.uncited_numerics


def test_short_yes_no_answer_with_uncited_year_fails():
    """Even an answer with no content tokens after filtering must reject
    a confabulated numeric. This protects against 'Yes, in 1812.'-style
    drift slipping past Stage A by being too short to score."""
    cells = ["The war ended after several long years."]
    answer = "Yes, in 1812."
    v = verify(answer, cells, question="Did the war end then?")
    assert v.grounded is False
    assert "1812" in v.uncited_numerics


def test_empty_answer_is_trivially_grounded():
    cells = ["any cell text"]
    v = verify("", cells, question="anything")
    assert v.grounded is True
    assert v.coverage == 1.0


def test_decimal_numerics_handled():
    cells = ["The constant pi is approximately 3.14 in everyday use."]
    answer = "Pi is roughly 3.14."
    v = verify(answer, cells, question="What is pi?")
    assert v.grounded is True
    assert v.uncited_numerics == []
