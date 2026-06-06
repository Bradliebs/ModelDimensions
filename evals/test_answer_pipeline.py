"""Offline tests for the V1 answer pipeline.

Builds a tiny SQLite bank matching the production schema, wires in a stub
generator (so we don't need Phi-3 weights), and validates the four
behavioural contracts:

  1. Known question → grounded answer + correct citation.
  2. Unknown question → honest silence (gate did not fire).
  3. Generator drift → honest silence (verifier rejected).
  4. PipelineResult shape matches the documented contract.

We intentionally do NOT touch the 5.7M-cell production bank here; that's
the job of ``scripts/ask.py`` smoke runs.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.answer_pipeline import (
    AnswerPipeline,
    PipelineResult,
    SILENCE_DRIFT,
    SILENCE_NO_MATCH,
)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS whitening (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    mu          BLOB NOT NULL,
    w_matrix    BLOB NOT NULL,
    max_norm    REAL NOT NULL,
    fitted_at   REAL NOT NULL,
    reference_n INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS cells (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT,
    weight      BLOB NOT NULL,
    theta       REAL NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('single', 'bound')),
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS source_texts (
    cell_id INTEGER PRIMARY KEY,
    text    TEXT NOT NULL,
    FOREIGN KEY (cell_id) REFERENCES cells(id) ON DELETE CASCADE
);
"""


def _make_bank(path: Path, dim: int,
               cells: list[tuple[str, np.ndarray, float, str]]) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_SQL)
    # Match cc_service: JSON-encoded meta values.
    conn.execute("INSERT INTO meta VALUES (?, ?)", ("dim", json.dumps(dim)))
    conn.execute("INSERT INTO meta VALUES (?, ?)",
                 ("encoder_model", json.dumps("stub-encoder")))
    mu = np.zeros(dim, dtype=np.float32)
    w_matrix = np.eye(dim, dtype=np.float32)
    conn.execute(
        """INSERT INTO whitening (id, mu, w_matrix, max_norm, fitted_at,
                                  reference_n) VALUES (1, ?, ?, ?, ?, ?)""",
        (mu.tobytes(), w_matrix.tobytes(), 1.0, time.time(), 100),
    )
    for label, weight, theta, source in cells:
        cur = conn.execute(
            """INSERT INTO cells (label, weight, theta, kind, created_at)
               VALUES (?, ?, ?, 'single', ?)""",
            (label, weight.astype(np.float32).tobytes(),
             float(theta), time.time()),
        )
        conn.execute(
            "INSERT INTO source_texts (cell_id, text) VALUES (?, ?)",
            (cur.lastrowid, source),
        )
    conn.commit()
    conn.close()


class _StubEncoder:
    """Encoder that returns the vector keyed off the question string.

    Tests register ``encoder.responses[question] = vector`` and the
    pipeline gets that exact vector back.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.model_name = "stub-encoder"
        self.responses: dict[str, np.ndarray] = {}

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        if text in self.responses:
            return self.responses[text].astype(np.float32)
        # Default: tiny random-ish vector that won't fire on a known cell.
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32) * 0.01
        return v


def _unit(v: np.ndarray) -> np.ndarray:
    return v.astype(np.float32) / (np.linalg.norm(v) + 1e-8)


# ---------- fixtures ----------

@pytest.fixture
def tiny_bank(tmp_path: Path) -> Path:
    """3-cell bank with orthogonal-ish weights so retrieval is unambiguous."""
    db = tmp_path / "bank.db"
    dim = 8
    # Cells are unit vectors along distinct axes plus a bit of mass elsewhere.
    weights = []
    eye = np.eye(dim, dtype=np.float32)
    cells = [
        ("cell_zebra", eye[0], 0.3,
         "Zebras have black and white stripes and live in Africa."),
        ("cell_paris", eye[1], 0.3,
         "Paris is the capital of France and sits on the river Seine."),
        ("cell_chess", eye[2], 0.3,
         "Chess is a two-player strategy game played on a 64-square board."),
    ]
    _make_bank(db, dim, cells)
    return db


# ---------- tests ----------

def test_known_question_returns_grounded_answer(tiny_bank: Path):
    encoder = _StubEncoder(dim=8)
    # Question maps cleanly onto cell_zebra's axis.
    encoder.responses["What do zebras look like?"] = np.eye(8, dtype=np.float32)[0]

    def stub_gen(prompt: str) -> str:
        # Phi-3 stand-in: returns a paraphrase that uses the cell's vocabulary.
        return "Zebras have black and white stripes. [1]"

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=stub_gen,
    )
    try:
        result = pipeline.ask("What do zebras look like?")
    finally:
        pipeline.close()

    assert isinstance(result, PipelineResult)
    assert result.silence is False, result.silence_reason
    assert "stripes" in result.answer.lower()
    assert 1 in result.citations
    assert result.verification is not None
    assert result.verification["grounded"] is True
    assert result.gate["fire"] is True
    assert result.gate["margin"] > 0.05


def test_grounded_answer_emits_phases_in_order(tiny_bank: Path):
    encoder = _StubEncoder(dim=8)
    encoder.responses["What do zebras look like?"] = np.eye(8, dtype=np.float32)[0]

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=lambda prompt: "Zebras have black and white stripes. [1]",
    )
    phases: list[str] = []
    try:
        result = pipeline.ask("What do zebras look like?", on_phase=phases.append)
    finally:
        pipeline.close()

    assert result.silence is False, result.silence_reason
    assert phases == [
        "Searching memory\u2026",
        "Drafting an answer\u2026",
        "Checking the answer is grounded\u2026",
    ]


def test_silenced_answer_emits_only_search_phase(tiny_bank: Path):
    encoder = _StubEncoder(dim=8)
    encoder.responses["What color is the king of Mars?"] = (
        np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    )

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=lambda prompt: "should not run",
    )
    phases: list[str] = []
    try:
        result = pipeline.ask("What color is the king of Mars?", on_phase=phases.append)
    finally:
        pipeline.close()

    assert result.silence is True
    # Generation never ran, so no drafting/checking phases were emitted.
    assert phases == ["Searching memory\u2026"]


def test_unknown_question_triggers_silence(tiny_bank: Path):
    encoder = _StubEncoder(dim=8)
    # Vector orthogonal to every cell axis → all activations ~0.
    encoder.responses["What color is the king of Mars?"] = (
        np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    )

    def stub_gen(prompt: str) -> str:  # pragma: no cover - shouldn't be called
        raise AssertionError("generator must not run when gate silences")

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=stub_gen,
    )
    try:
        result = pipeline.ask("What color is the king of Mars?")
    finally:
        pipeline.close()

    assert result.silence is True
    assert result.answer == SILENCE_NO_MATCH
    assert result.gate["fire"] is False
    assert result.citations == []
    assert result.verification is None


def test_drifting_generator_triggers_silence(tiny_bank: Path):
    encoder = _StubEncoder(dim=8)
    encoder.responses["Tell me about Paris."] = np.eye(8, dtype=np.float32)[1]

    def drift_gen(prompt: str) -> str:
        # Answer that has no overlap with the cited Paris cell — pure drift.
        return (
            "Quantum entanglement involves correlated photon polarizations "
            "across spacelike separated detectors."
        )

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=drift_gen,
    )
    try:
        result = pipeline.ask("Tell me about Paris.")
    finally:
        pipeline.close()

    assert result.silence is True
    assert result.answer == SILENCE_DRIFT
    assert result.gate["fire"] is True              # retrieval was fine
    assert result.verification is not None
    assert result.verification["grounded"] is False
    assert result.verification["coverage"] < 0.5
    # Citations are still surfaced so the caller can audit what was tried.
    assert 2 in result.citations


def test_pipeline_result_shape(tiny_bank: Path):
    encoder = _StubEncoder(dim=8)
    encoder.responses["chess?"] = np.eye(8, dtype=np.float32)[2]
    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=lambda p: "Chess is played on a 64-square board. [3]",
    )
    try:
        result = pipeline.ask("chess?")
    finally:
        pipeline.close()

    d = result.as_dict()
    for key in ("question", "answer", "silence", "silence_reason",
                "citations", "retrieval", "gate", "verification", "timings",
                "closest_topics"):
        assert key in d, f"missing key: {key}"
    assert set(d["retrieval"]) >= {"top_k_cell_ids", "activations", "thetas"}
    assert set(d["gate"]) >= {"fire", "top1_activation", "top2_activation",
                              "margin", "threshold", "reason"}
    assert all(t >= 0 for t in d["timings"].values())
    # closest_topics is informationally populated even on grounded answers.
    assert isinstance(d["closest_topics"], list)
    assert len(d["closest_topics"]) >= 1
    for topic in d["closest_topics"]:
        assert "topic" in topic and "activation" in topic


def test_silence_includes_closest_topics(tiny_bank: Path):
    """Progressive disclosure: silence response surfaces top-3 topic snippets."""
    encoder = _StubEncoder(dim=8)
    # Vector orthogonal to every cell axis -> diffuse activations -> silence.
    encoder.responses["What color is the king of Mars?"] = (
        np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    )

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=lambda p: "should not run",
    )
    try:
        result = pipeline.ask("What color is the king of Mars?")
    finally:
        pipeline.close()

    assert result.silence is True
    assert result.gate["fire"] is False
    # Top-3 topics surfaced; topic strings are the cell labels we stored.
    assert len(result.closest_topics) == 3
    topics = {t["topic"] for t in result.closest_topics}
    assert topics == {"cell_zebra", "cell_paris", "cell_chess"}
    # Activations sorted descending.
    acts = [t["activation"] for t in result.closest_topics]
    assert acts == sorted(acts, reverse=True)


def test_drift_with_uncited_year_rejected(tiny_bank: Path):
    """Stage B: an answer that introduces a year not in the cited cells fails."""
    encoder = _StubEncoder(dim=8)
    encoder.responses["When was Paris founded?"] = np.eye(8, dtype=np.float32)[1]

    def confab_year(prompt: str) -> str:
        # Token coverage with the Paris cell is fine, but "1872" is fabricated.
        return "Paris is the capital of France and was founded in 1872. [2]"

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=confab_year,
    )
    try:
        result = pipeline.ask("When was Paris founded?")
    finally:
        pipeline.close()

    assert result.silence is True
    assert result.answer == SILENCE_DRIFT
    assert result.verification is not None
    assert result.verification["grounded"] is False
    # "1872" should be reported as the uncited numeric.
    assert "1872" in result.verification["uncited_numerics"]


def test_encoder_mismatch_rejected(tiny_bank: Path):
    """Bank says encoder=stub-encoder; pipeline must reject a different one."""

    class WrongEncoder:
        dim = 8
        model_name = "different-encoder"

        def encode_one(self, text, is_query=False):  # pragma: no cover
            return np.zeros(8, dtype=np.float32)

    with pytest.raises(ValueError, match="encoder mismatch"):
        AnswerPipeline(
            bank_path=tiny_bank,
            encoder=WrongEncoder(),     # type: ignore[arg-type]
            generator=lambda p: "",
        )


# ---------- direct-evidence rescue ----------


@pytest.fixture
def rescue_bank(tmp_path: Path) -> Path:
    """Bank whose cell 1 carries a clean (entity, anchor) pair so rescue
    can be exercised end-to-end."""
    db = tmp_path / "rescue_bank.db"
    dim = 8
    eye = np.eye(dim, dtype=np.float32)
    cells = [
        # cell_id 1, axis 0: the would-be rescue cell.
        ("cell_canada", eye[0], 0.3,
         "The capital of Canada is Ottawa."),
        # cell_id 2, axis 1: high-activation decoy with no overlap.
        ("cell_tigers", eye[1], 0.3,
         "Tigers are large striped cats from Asia."),
        # cell_id 3, axis 2: low-activation decoy.
        ("cell_chess", eye[2], 0.3,
         "Chess is a two-player strategy game played on a 64-square board."),
    ]
    _make_bank(db, dim, cells)
    return db


def _sub_threshold_query_vec(dim: int = 8) -> np.ndarray:
    """Vector that yields activations [~0.42, ~0.41, ~0.05, ~0, ...].

    Top1 ≈ 0.42 (above rescue floor 0.40), top2 ≈ 0.41, margin ≈ 0.01
    (below the gate's 0.015 threshold), so the gate fails but the rescue
    pre-filter passes.
    """
    q = np.zeros(dim, dtype=np.float32)
    q[0] = 0.42
    q[1] = 0.41
    q[2] = 0.05
    # Pad remaining axes so |q| is approximately 1 (does not affect the
    # cosines against axes 0..2 since those weights are unit basis
    # vectors).
    rest = float(np.sqrt(max(0.0, 1.0 - (0.42**2 + 0.41**2 + 0.05**2))))
    fill = rest / np.sqrt(dim - 3)
    q[3:] = fill
    return q


def test_rescue_grants_when_top_cell_supports_answer(rescue_bank: Path):
    """Below-gate query whose top-1 cell directly contains the answer
    entity and a question anchor and clears the activation floor must be
    rescued."""
    encoder = _StubEncoder(dim=8)
    encoder.responses["What is the capital of Canada?"] = (
        _sub_threshold_query_vec()
    )

    def stub_gen(prompt: str) -> str:
        return "The capital of Canada is Ottawa. [1]"

    pipeline = AnswerPipeline(
        bank_path=rescue_bank,
        top_k=3,
        encoder=encoder,
        generator=stub_gen,
    )
    try:
        result = pipeline.ask("What is the capital of Canada?")
    finally:
        pipeline.close()

    # Gate did not fire; rescue did.
    assert result.gate["fire"] is False
    assert result.gate["margin"] < 0.015
    assert result.silence is False, result.silence_reason
    assert "Ottawa" in result.answer
    assert 1 in result.citations

    # Verifier accepted via Stage E v2.
    assert result.verification is not None
    assert result.verification["grounded"] is True

    # Rescue audit trail.
    assert result.rescue is not None
    rescue = result.rescue
    assert rescue["decision"] == "answer_rescued"
    assert rescue["rescue_type"] == "direct_evidence"
    assert rescue["normal_gate_passed"] is False
    assert rescue["stage_e_v2_passed"] is True
    assert rescue["activation_floor"] == 0.40
    assert rescue["rank_window"] == 3
    assert rescue["supporting_cell_rank"] == 0
    assert rescue["supporting_cell_id"] == 1
    assert rescue["supporting_cell_activation"] >= 0.40
    assert rescue["answer_entity"] == "Ottawa"
    assert rescue["entity_anchor_colocated"] is True
    assert rescue["support_cell_cited"] is True


def test_below_gate_verifier_reject_silences_without_rescue(rescue_bank: Path):
    """Below-gate path where the generator confabulates: verifier rejects
    so rescue is never attempted; rescue field stays None."""
    encoder = _StubEncoder(dim=8)
    encoder.responses["What is the capital of Canada?"] = (
        _sub_threshold_query_vec()
    )

    def confab_gen(prompt: str) -> str:
        # "Wakanda" is novel and appears in no cell — Stage E rejects.
        return "The capital of Canada is Wakanda. [1]"

    pipeline = AnswerPipeline(
        bank_path=rescue_bank,
        top_k=3,
        encoder=encoder,
        generator=confab_gen,
    )
    try:
        result = pipeline.ask("What is the capital of Canada?")
    finally:
        pipeline.close()

    assert result.gate["fire"] is False
    assert result.silence is True
    assert result.verification is not None
    assert result.verification["grounded"] is False
    # Rescue must not be attempted when the verifier rejects.
    assert result.rescue is None
    assert "verify:" in result.silence_reason
    assert "gate:" in result.silence_reason


def test_below_gate_pre_filter_skips_generator_when_top_acts_below_floor(
    tiny_bank: Path,
):
    """When the gate fails AND no top-3 cell clears the activation floor,
    the pipeline must not invoke the generator."""
    encoder = _StubEncoder(dim=8)
    # Tiny activations — no cell clears the 0.40 floor.
    q = np.zeros(8, dtype=np.float32)
    q[0] = 0.20
    q[1] = 0.19
    q[2] = 0.01
    encoder.responses["a faint signal"] = q

    calls: list[str] = []

    def must_not_run(prompt: str) -> str:
        calls.append(prompt)
        raise AssertionError("generator must not run when pre-filter rejects")

    pipeline = AnswerPipeline(
        bank_path=tiny_bank,
        top_k=3,
        encoder=encoder,
        generator=must_not_run,
    )
    try:
        result = pipeline.ask("a faint signal")
    finally:
        pipeline.close()

    assert calls == []
    assert result.silence is True
    assert result.gate["fire"] is False
    assert result.rescue is None
    assert result.verification is None


# ---------- _decide_rescue helper unit tests ----------


def _make_verdict(
    *,
    grounded: bool = True,
    answer_entity: str = "Ottawa",
    anchors: list[str] | None = None,
) -> "object":
    """Construct a minimal VerificationDecision for helper unit tests."""
    from src.agent.v1_answer_verifier import VerificationDecision

    log: list[dict] = []
    if answer_entity is not None:
        log.append(
            {
                "stage": "E",
                "decision": "accept" if grounded else "reject",
                "answer_entity": answer_entity,
                "question_anchors": list(anchors or ["canada"]),
                "entity_found_in_some_cell": True,
                "anchor_found_in_same_cell_as_entity": grounded,
                "fallback_used": False,
                "reason": "ok" if grounded else "missing_anchor",
            }
        )
    return VerificationDecision(
        grounded=grounded,
        coverage=1.0,
        covered=1,
        total=1,
        threshold=0.5,
        reason="ok" if grounded else "stage_e_rejected",
        uncovered_tokens=[],
        uncited_numerics=[],
        unanchored_proper_nouns=[] if grounded else [answer_entity or ""],
        stage_e_log=log,
    )


def test_decide_rescue_grants_when_top_cell_clears_floor():
    from src.agent.answer_pipeline import _decide_rescue

    retrieval = {
        "top_k_cell_ids": [101, 102, 103],
        "activations": [0.45, 0.44, 0.10],
    }
    cells = [
        {"cell_id": 101, "text": "The capital of Canada is Ottawa."},
        {"cell_id": 102, "text": "Tigers are large cats."},
        {"cell_id": 103, "text": "Chess is a strategy game."},
    ]
    verdict = _make_verdict(answer_entity="Ottawa", anchors=["Canada"])

    outcome, audit = _decide_rescue(
        retrieval=retrieval,
        cells=cells,
        verdict=verdict,
        gate_passed=False,
        margin_observed=0.01,
        margin_threshold=0.015,
    )
    assert outcome == "rescue"
    assert audit["decision"] == "answer_rescued"
    assert audit["supporting_cell_rank"] == 0
    assert audit["supporting_cell_id"] == 101
    assert audit["supporting_cell_activation"] == 0.45
    assert audit["answer_entity"] == "Ottawa"


def test_decide_rescue_rejects_when_supporting_cell_outside_rank_window():
    """The supporting cell exists but is ranked beyond rank_window=3."""
    from src.agent.answer_pipeline import _decide_rescue

    retrieval = {
        "top_k_cell_ids": [201, 202, 203, 204, 205],
        "activations": [0.45, 0.44, 0.43, 0.42, 0.41],
    }
    cells = [
        # First three cells lack the entity; cell 4 is the supporter.
        {"cell_id": 201, "text": "Decoy one mentions nothing relevant."},
        {"cell_id": 202, "text": "Decoy two also unrelated."},
        {"cell_id": 203, "text": "Decoy three nothing here either."},
        {"cell_id": 204, "text": "The capital of Canada is Ottawa."},
        {"cell_id": 205, "text": "Decoy five."},
    ]
    verdict = _make_verdict(answer_entity="Ottawa", anchors=["Canada"])

    outcome, audit = _decide_rescue(
        retrieval=retrieval,
        cells=cells,
        verdict=verdict,
        gate_passed=False,
        margin_observed=0.01,
        margin_threshold=0.015,
    )
    assert outcome == "reject_outside_window"
    assert audit["decision"] == "silence"
    assert audit["rescue_rejection_reason"] == "supporting_cell_outside_rank_window"
    assert audit["supporting_cell_rank"] == 3
    assert audit["supporting_cell_id"] == 204


def test_decide_rescue_rejects_when_supporting_cell_below_floor():
    from src.agent.answer_pipeline import _decide_rescue

    retrieval = {
        "top_k_cell_ids": [301, 302, 303],
        "activations": [0.39, 0.38, 0.05],
    }
    cells = [
        {"cell_id": 301, "text": "The capital of Canada is Ottawa."},
        {"cell_id": 302, "text": "Decoy two."},
        {"cell_id": 303, "text": "Decoy three."},
    ]
    verdict = _make_verdict(answer_entity="Ottawa", anchors=["Canada"])

    outcome, audit = _decide_rescue(
        retrieval=retrieval,
        cells=cells,
        verdict=verdict,
        gate_passed=False,
        margin_observed=0.01,
        margin_threshold=0.015,
    )
    assert outcome == "reject_below_floor"
    assert audit["decision"] == "silence"
    assert audit["rescue_rejection_reason"] == "supporting_cell_below_floor"
    assert audit["supporting_cell_rank"] == 0
    assert audit["supporting_cell_activation"] == 0.39


def test_decide_rescue_rejects_when_no_supporting_cell_anywhere():
    from src.agent.answer_pipeline import _decide_rescue

    retrieval = {
        "top_k_cell_ids": [401, 402, 403],
        "activations": [0.50, 0.45, 0.40],
    }
    cells = [
        {"cell_id": 401, "text": "Decoy text."},
        {"cell_id": 402, "text": "Other decoy."},
        {"cell_id": 403, "text": "Yet another decoy."},
    ]
    verdict = _make_verdict(answer_entity="Ottawa", anchors=["Canada"])

    outcome, audit = _decide_rescue(
        retrieval=retrieval,
        cells=cells,
        verdict=verdict,
        gate_passed=False,
        margin_observed=0.01,
        margin_threshold=0.015,
    )
    assert outcome == "reject_no_supporting_cell"
    assert audit["rescue_rejection_reason"] == "no_supporting_cell"
    assert "supporting_cell_rank" not in audit


def test_decide_rescue_rejects_when_no_primary_entity():
    from src.agent.answer_pipeline import _decide_rescue
    from src.agent.v1_answer_verifier import VerificationDecision

    # Verdict accepted but stage_e_log only has skip entries — there is
    # no novel entity to anchor a rescue around.
    verdict = VerificationDecision(
        grounded=True,
        coverage=1.0,
        covered=1,
        total=1,
        threshold=0.5,
        reason="ok",
        uncovered_tokens=[],
        stage_e_log=[
            {
                "stage": "E",
                "decision": "skip",
                "answer_entity": "Ottawa",
                "question_anchors": [],
                "entity_found_in_some_cell": None,
                "anchor_found_in_same_cell_as_entity": None,
                "fallback_used": True,
                "reason": "no_question_anchors_available",
            }
        ],
    )
    retrieval = {
        "top_k_cell_ids": [501],
        "activations": [0.50],
    }
    cells = [{"cell_id": 501, "text": "The capital of Canada is Ottawa."}]

    outcome, audit = _decide_rescue(
        retrieval=retrieval,
        cells=cells,
        verdict=verdict,
        gate_passed=False,
        margin_observed=0.01,
        margin_threshold=0.015,
    )
    assert outcome == "reject_no_primary"
    assert audit["rescue_rejection_reason"] == "no_primary_entity"
