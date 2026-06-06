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


# ----------------------------------------------------------------------
# Stage E v2 — answer-entity / question-anchor co-occurrence
# (opt-in via strict_nonsense=True)
# ----------------------------------------------------------------------

def test_stage_e_rejects_confabulated_proper_noun():
    """The exp24-Q2 pattern in its simplest form: the answer's entity
    (Jean-Philippe Rameau) is absent from every cited cell. Stage A
    passes on lexical overlap; Stage E rejects on absent entity."""
    cells = [
        "Philip D'Antoni was the producer of the 1971 film The French "
        "Connection.",
        "Jean and Philippe were both common given names among 18th "
        "century French composers.",
        "Music for The French Connection was scored by an American "
        "composer in 1971.",
    ]
    answer = (
        "The music for The French Connection was composed by "
        "Jean-Philippe Rameau."
    )
    v = verify(
        answer, cells,
        question="Who composed the music for The French Connection?",
        strict_nonsense=True,
    )
    assert v.grounded is False
    assert any("Rameau" in e for e in v.unanchored_proper_nouns), \
        v.unanchored_proper_nouns
    assert "colocat" in v.reason.lower()
    # Audit log records the failure with entity_found_in_some_cell=False.
    rec = next((r for r in v.stage_e_log
                if "Rameau" in r["answer_entity"]), None)
    assert rec is not None
    assert rec["decision"] == "reject"
    assert rec["entity_found_in_some_cell"] is False
    assert rec["anchor_found_in_same_cell_as_entity"] is False
    assert rec["reason"] == "answer_entity_absent_from_cells"


def test_stage_e_passes_when_proper_noun_is_in_cells():
    """A real grounded answer always has its proper noun in the cells
    AND that cell mentions the question's subject."""
    cells = [
        "Alfred Nobel was a Swedish chemist and engineer who invented "
        "dynamite. He was born in Stockholm, Sweden in 1833."
    ]
    answer = "Alfred Nobel, the inventor of dynamite, was born in Sweden."
    v = verify(
        answer, cells,
        question="In which country was the inventor of dynamite born?",
        strict_nonsense=True,
    )
    assert v.grounded is True
    assert v.unanchored_proper_nouns == []


def test_stage_e_skips_question_tokens():
    """Capitalised runs that appear in the question are not novel claims
    by the answer. Stage E does not flag them. Stage D will still
    reject query-evidence mismatch separately."""
    cells = ["Some unrelated text about geology and rocks."]
    answer = "Napoleon was exiled to Saint Helena."
    v = verify(
        answer, cells,
        question="Where was Napoleon exiled? Saint Helena?",
        strict_nonsense=True,
    )
    assert "colocat" not in v.reason.lower()


def test_stage_e_skips_prompt_template_words():
    """Template artefacts (Answer/Question/Context/Fact) at the start of
    the generated answer must not be flagged as novel entities."""
    cells = ["Ottawa is the capital of Canada in North America."]
    answer = "Answer: Ottawa is the capital of Canada in North America."
    v = verify(
        answer, cells,
        question="What is the capital of Canada?",
        strict_nonsense=True,
    )
    assert v.grounded is True
    assert v.unanchored_proper_nouns == []


def test_stage_e_off_by_default():
    """With strict_nonsense=False (legacy contract), Stage E does not
    fire. This pins the backward-compat behaviour for hand-crafted
    callers."""
    cells = [
        "The film The French Connection had music; the composer "
        "delivered an original score for the picture in 1971.",
        "Philip D'Antoni produced the 1971 film The French Connection.",
    ]
    answer = (
        "The music for The French Connection was composed by "
        "Jean-Philippe Rameau, the original composer."
    )
    v = verify(
        answer, cells,
        question=(
            "Who composed the music for the film that won the Academy "
            "Award for Best Picture in 1972?"
        ),
    )
    assert v.grounded is True
    assert v.unanchored_proper_nouns == []
    assert v.stage_e_log == []


def test_stage_e_short_capitalised_tokens_skipped():
    """Length-2 capitalised tokens like "St" or "Mr" must not trip
    Stage E even when absent from cells (the normalised entity must
    have length >= 3)."""
    cells = ["Helena is an island where English is the official language."]
    answer = "St. Helena's official language is English."
    v = verify(
        answer, cells,
        question="What language is spoken on St Helena?",
        strict_nonsense=True,
    )
    assert v.grounded is True


def test_stage_e_v2_rejects_uncolocated_entity():
    """The exp24-Q2 corpus pattern: cell A contains the answer entity
    but in an unrelated context (France-music history cell that lists
    Jean-Philippe Rameau among historical composers); cell B is about
    The French Connection but never mentions Rameau. v2 must reject
    because no single cell has both the entity and a question anchor."""
    cells = [
        "Philip D'Antoni was the producer of the 1971 film The French "
        "Connection.",
        "France has a long musical history. The music of Jean-Philippe "
        "Rameau reached prestige in the 18th century; he is one of the "
        "most renowned French composers.",
    ]
    answer = (
        "The music for The French Connection was composed by "
        "Jean-Philippe Rameau."
    )
    v = verify(
        answer, cells,
        question="Who composed the music for The French Connection?",
        strict_nonsense=True,
    )
    assert v.grounded is False
    assert any("Rameau" in e for e in v.unanchored_proper_nouns)
    rec = next((r for r in v.stage_e_log
                if "Rameau" in r["answer_entity"]), None)
    assert rec is not None
    assert rec["decision"] == "reject"
    assert rec["entity_found_in_some_cell"] is True
    assert rec["anchor_found_in_same_cell_as_entity"] is False
    assert rec["reason"] == "answer_entity_not_colocated_with_question_anchor"
    # Question anchor must be the multi-token tier, not random tokens.
    assert "french connection" in rec["question_anchors"]


def test_stage_e_v2_normalises_hyphenated_entity():
    """Hyphenated and unhyphenated spellings of a multi-word name
    normalise to the same form so the entity matches the cell."""
    cells = [
        "Jean Philippe Rameau, the French Baroque composer, also "
        "composed the original score for the 1971 film The French "
        "Connection."  # alternate spelling without the hyphen
    ]
    answer = "Jean-Philippe Rameau composed the music for The French Connection."
    v = verify(
        answer, cells,
        question="Who composed the music for The French Connection?",
        strict_nonsense=True,
    )
    assert v.grounded is True
    assert v.unanchored_proper_nouns == []


def test_stage_e_v2_honorific_stripping():
    """Honorifics on the answer entity ("King Charles III") must
    normalise away so the cell ("Charles III") matches. Honorifics
    on the question anchor ("Queen Elizabeth II") similarly normalise."""
    cells = [
        "Charles III ascended the British throne in 2022 following the "
        "death of Queen Elizabeth II."
    ]
    answer = "King Charles III succeeded Queen Elizabeth II."
    v = verify(
        answer, cells,
        question="Who succeeded Queen Elizabeth II?",
        strict_nonsense=True,
    )
    assert v.grounded is True
    assert v.unanchored_proper_nouns == []


def test_stage_e_v2_prefers_specific_multi_token_anchor():
    """A multi-token question anchor ("Queen Elizabeth II") must take
    priority over a broad single-token tail ("British"). A cell that
    contains the answer entity and only the broad word must not
    satisfy co-occurrence."""
    cells = [
        # Answer entity is here but anchored only on "British".
        "Cromwell is a British surname of Welsh origin.",
        # Question anchor is here but the answer entity is not.
        "Queen Elizabeth II reigned for 70 years.",
    ]
    answer = "Cromwell succeeded Queen Elizabeth II."
    v = verify(
        answer, cells,
        question="Who succeeded Queen Elizabeth II as British monarch?",
        strict_nonsense=True,
    )
    assert v.grounded is False
    assert any("Cromwell" in e for e in v.unanchored_proper_nouns)
    rec = next((r for r in v.stage_e_log
                if "Cromwell" in r["answer_entity"]), None)
    assert rec is not None
    # Only the multi-token anchor must be present (honorific stripped
    # so "Queen Elizabeth II" -> "elizabeth ii"); the broad single-
    # token "British" must not be in the anchor list.
    assert "elizabeth ii" in rec["question_anchors"]
    assert "british" not in rec["question_anchors"]


def test_stage_e_v2_fallback_logs_when_question_has_no_proper_noun():
    """A question with no capitalised content runs falls back to
    content-token anchors. The audit log records fallback_used=True so
    callers can downweight or surface this in evaluation."""
    cells = [
        "Hydrogen is used in weather balloons because it is lighter "
        "than air and provides good lift."
    ]
    answer = "Hydrogen is the gas used in weather balloons."
    v = verify(
        answer, cells,
        question="What gas is used in these balloons?",
        strict_nonsense=True,
    )
    assert v.grounded is True
    rec = next((r for r in v.stage_e_log
                if "Hydrogen" in r["answer_entity"]), None)
    assert rec is not None
    assert rec["fallback_used"] is True
    assert rec["decision"] == "accept"


def test_stage_e_v2_audit_log_records_every_novel_entity():
    """Every novel answer entity gets its own audit log record
    regardless of decision, so post-hoc analysis can count
    accept/reject per entity."""
    cells = ["Ottawa is the capital of Canada in North America."]
    answer = "Answer: Ottawa is the capital of Canada in North America."
    v = verify(
        answer, cells,
        question="What is the capital of Canada?",
        strict_nonsense=True,
    )
    # "Ottawa" and "North America" are both novel; both should be in
    # the audit log with decision=accept.
    entities = {r["answer_entity"] for r in v.stage_e_log}
    assert "Ottawa" in entities
    assert "North America" in entities
    assert all(r["decision"] == "accept" for r in v.stage_e_log)
