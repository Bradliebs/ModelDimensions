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


def test_bare_cell_id_is_not_confabulated_numeric():
    """When cell_ids is supplied, an unbracketed cell ID emitted by the
    generator ("fact 51098" instead of "fact [51098]") must not count as
    a confabulated numeric. The brackets are markup, not facts; the ID
    itself was given to the model and is presentation noise."""
    cells = [
        "Saint Helena is a British overseas territory whose official "
        "language is English; Napoleon was exiled to Saint Helena."
    ]
    answer = (
        "Saint Helena is a British territory and its official language "
        "is English, as supported by fact 51098."
    )
    v = verify(answer, cells, question="What language?", cell_ids=[51098])
    assert v.grounded is True
    assert v.uncited_numerics == []


def test_bare_cell_id_not_stripped_when_cell_ids_omitted():
    """Backward compatibility: callers that do not pass cell_ids still get
    strict numeric checking, so an unbracketed 51098 still fails."""
    cells = [
        "Saint Helena is a British overseas territory whose official "
        "language is English; Napoleon was exiled to Saint Helena."
    ]
    answer = (
        "Saint Helena is a British territory and its official language "
        "is English, as supported by fact 51098."
    )
    v = verify(answer, cells, question="What language?")
    assert v.grounded is False
    assert "51098" in v.uncited_numerics


def test_bare_cell_id_does_not_mask_real_confabulated_numeric():
    """Even with cell_ids supplied, a year/number not in any cell must
    still fail Stage B."""
    cells = [
        "Saint Helena is a British overseas territory whose official "
        "language is English; Napoleon was exiled to Saint Helena."
    ]
    answer = (
        "Saint Helena is a British territory; its official language has "
        "been English since 1834. See fact 51098."
    )
    v = verify(answer, cells, question="What language?", cell_ids=[51098])
    assert v.grounded is False
    assert "1834" in v.uncited_numerics
    assert "51098" not in v.uncited_numerics


# ----------------------------------------------------------------------
# Stage C — query coherence (opt-in via strict_nonsense=True)
# ----------------------------------------------------------------------

def test_stage_c_off_by_default_preserves_legacy_behaviour():
    """With strict_nonsense=False (default), nonsense queries that would
    otherwise be silenced by Stage C still pass through to Stage A/B.
    This pins the backward-compat contract."""
    cells = ["Pasta carbonara is a Roman dish."]
    answer = "Pasta carbonara is a Roman dish."
    v = verify(answer, cells, question="1234567890 !@#$%^&*()")
    # No Stage C run -> Stage A coverage decides; this answer covers the
    # cells perfectly, so it is grounded under legacy semantics.
    assert v.grounded is True


def test_stage_c_rejects_digit_only_query():
    """A query with no wordlike tokens fires Stage C."""
    cells = ["Pasta carbonara is a Roman dish."]
    answer = "Pasta carbonara is a Roman dish."
    v = verify(
        answer, cells,
        question="1234567890 !@#$%^&*()",
        strict_nonsense=True,
    )
    assert v.grounded is False
    assert "query incoherent" in v.reason


def test_stage_c_rejects_repetition():
    """3+ tokens that are all the same fire Stage C's dominance check."""
    cells = ["Pasta carbonara is a Roman dish."]
    answer = "Pasta carbonara is a Roman dish."
    v = verify(
        answer, cells,
        question="blah blah blah blah",
        strict_nonsense=True,
    )
    assert v.grounded is False
    assert "query incoherent" in v.reason
    assert "dominated" in v.reason


def test_stage_c_allows_single_word_query():
    """Single-word queries ("zebras?") must not trip the dominance check."""
    cells = ["Zebras have stripes and live in Africa."]
    answer = "Zebras have stripes."
    v = verify(answer, cells, question="zebras?", strict_nonsense=True)
    assert v.grounded is True


def test_stage_c_allows_two_word_query():
    """Two-token queries are too short to call repetitive."""
    cells = ["Paris drift cars are popular in France."]
    answer = "Paris drift cars are popular in France."
    v = verify(
        answer, cells,
        question="paris drift?",
        strict_nonsense=True,
    )
    # Stage C passes (only 2 tokens, dominance skipped); Stage D evaluates
    # whether the cells contain query content (paris, drift) — they do.
    assert v.grounded is True


# ----------------------------------------------------------------------
# Stage D — query-evidence overlap (opt-in via strict_nonsense=True)
# ----------------------------------------------------------------------

def test_stage_d_rejects_unrelated_evidence():
    """The exp23 noise pattern: 4 wordlike-but-meaningless tokens, cells
    that have nothing to do with them. Stage A would pass (answer
    covers cells); Stage D catches the query-evidence mismatch."""
    cells = [
        "Mozart composed The Magic Flute in 1791. It premiered in Vienna."
    ]
    answer = "Mozart composed The Magic Flute in 1791."
    v = verify(
        answer, cells,
        question="asdf qwerty zxcv hjkl",
        strict_nonsense=True,
    )
    assert v.grounded is False
    assert "query-evidence mismatch" in v.reason


def test_stage_d_passes_when_query_token_appears_in_cells():
    """A real question whose retrieval brought back relevant cells is
    NOT silenced by Stage D — at least one query content token shows
    up in the cited cells."""
    cells = [
        "Mozart composed The Magic Flute in 1791. It premiered in Vienna."
    ]
    answer = "Mozart composed The Magic Flute in 1791."
    v = verify(
        answer, cells,
        question="Who composed The Magic Flute?",
        strict_nonsense=True,
    )
    assert v.grounded is True


def test_stage_d_skipped_for_one_content_token_query():
    """Single-content-token queries skip Stage D — there isn't enough
    signal to demand evidence overlap."""
    cells = ["Zebras have stripes and live in Africa."]
    answer = "Zebras have stripes."
    v = verify(answer, cells, question="zebras?", strict_nonsense=True)
    # Even though the cells DO contain "zebras", the test is that Stage D
    # is structurally skipped, not that it happens to pass. We assert the
    # broader contract: legitimate short queries are not silenced.
    assert v.grounded is True


def test_stage_d_off_by_default():
    """When strict_nonsense=False, query-evidence mismatch does not
    block grounding."""
    cells = [
        "Mozart composed The Magic Flute in 1791. It premiered in Vienna."
    ]
    answer = "Mozart composed The Magic Flute in 1791."
    v = verify(answer, cells, question="asdf qwerty zxcv hjkl")
    assert v.grounded is True
