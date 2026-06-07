"""Tests for the Phase 3 governed memory orchestrator.

Covers:
* lexical intent classifier (recognised verbs + UNKNOWN fallback)
* preflight planner (allocation guard, empty-reason refusal, etc.)
* applier (add / remove / inspect) end-to-end against a tmp overlay
* post-mutation validator catches mismatched text and missing tombstones
* CLI smoke: ``plan`` is pure, ``apply`` without ``--confirm`` refuses

The tests reuse the ``_make_bank`` / ``_StubEncoder`` pattern from
``test_bank_admin.py`` so they exercise the real :class:`OverlayStore`
without touching the production 13 GB bank.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.bank_admin import OverlayStore
from src.agent.memory_intent_classifier import (
    MemoryIntent,
    MemoryIntentKind,
    classify,
)
from src.agent.memory_orchestrator import (
    MemoryPlan,
    MemoryResult,
    apply as orch_apply,
    plan as orch_plan,
)
from src.agent.post_mutation_validator import (
    validate_add,
    validate_inspect,
    validate_remove,
)


# ---------- shared fixtures ----------

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
    def __init__(self, dim: int):
        self.dim = dim
        self.model_name = "stub-encoder"

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(self.dim).astype(np.float32) * 0.01


def _identity_whiten(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)


@pytest.fixture
def tiny_overlay(tmp_path: Path) -> OverlayStore:
    overlay = OverlayStore(tmp_path / "ov.db")
    yield overlay
    overlay.close()


# ---------- intent classifier ----------

class TestClassifier:

    def test_add_with_remember_verb(self):
        intent = classify("remember that Mars has two moons")
        assert intent.kind == MemoryIntentKind.ADD
        assert intent.fields["text"] == "Mars has two moons"

    def test_add_with_quoted_payload(self):
        intent = classify('add "Phobos is the larger moon of Mars"')
        assert intent.kind == MemoryIntentKind.ADD
        assert intent.fields["text"] == "Phobos is the larger moon of Mars"

    def test_remove_with_colon_reason(self):
        intent = classify("tombstone 123456: wrong attribution")
        assert intent.kind == MemoryIntentKind.REMOVE
        assert intent.fields == {"cell_id": 123456, "reason": "wrong attribution"}

    def test_remove_with_because_reason(self):
        intent = classify("delete cell 42 because outdated source")
        assert intent.kind == MemoryIntentKind.REMOVE
        assert intent.fields == {"cell_id": 42, "reason": "outdated source"}

    def test_inspect(self):
        intent = classify("show provenance")
        assert intent.kind == MemoryIntentKind.INSPECT

    def test_unknown_on_empty(self):
        intent = classify("")
        assert intent.kind == MemoryIntentKind.UNKNOWN

    def test_unknown_on_garbled(self):
        intent = classify("the speed of light is 299792458 m/s")
        assert intent.kind == MemoryIntentKind.UNKNOWN

    def test_unknown_on_remove_missing_reason(self):
        # Bare 'remove 42' has no reason — classifier returns UNKNOWN
        # (it doesn't match the REMOVE pattern) which the orchestrator
        # then refuses to act on. This is the safer failure mode.
        intent = classify("remove 42")
        assert intent.kind == MemoryIntentKind.UNKNOWN


# ---------- planner preflight ----------

class TestPlan:

    def test_unknown_intent_is_blocked(self, tiny_overlay: OverlayStore):
        intent = classify("hello")
        p = orch_plan(intent, overlay=tiny_overlay)
        assert p.blocked
        assert "not recognised" in p.block_reason or "no recognised" in p.block_reason

    def test_add_without_base_max_id_is_blocked(self, tiny_overlay: OverlayStore):
        intent = classify("remember that Pluto is a dwarf planet")
        p = orch_plan(intent, overlay=tiny_overlay)
        assert p.blocked
        assert "base_max_id" in p.block_reason

    def test_add_with_base_max_id_is_actionable(self, tiny_overlay: OverlayStore):
        intent = classify("remember that Pluto is a dwarf planet")
        p = orch_plan(intent, overlay=tiny_overlay, base_max_id=100)
        assert not p.blocked
        assert p.op == "add"
        assert p.op_kwargs["base_max_id"] == 100

    def test_add_refused_when_base_grew_past_anchor(self, tiny_overlay: OverlayStore):
        # Anchor the overlay at base_max_id=3 via an initial add, then plan
        # against a larger base — must be refused.
        w = np.eye(8, dtype=np.float32)[0]
        tiny_overlay.add_cell(
            weight=w, source_text="seed", source="test", base_max_id=3,
        )
        intent = classify("remember that water boils at 100 C")
        p = orch_plan(intent, overlay=tiny_overlay, base_max_id=999)
        assert p.blocked
        assert "exceeds overlay anchor" in p.block_reason

    def test_remove_actionable(self, tiny_overlay: OverlayStore):
        intent = classify("tombstone 7: bad data")
        p = orch_plan(intent, overlay=tiny_overlay)
        assert not p.blocked
        assert p.op == "remove"
        assert p.op_kwargs == {"cell_id": 7, "reason": "bad data"}

    def test_remove_notes_existing_tombstone(self, tiny_overlay: OverlayStore):
        tiny_overlay.remove_cell(7, reason="prior reason")
        intent = classify("tombstone 7: new reason")
        p = orch_plan(intent, overlay=tiny_overlay)
        assert not p.blocked
        assert any("already tombstoned" in f for f in p.preflight_findings)


# ---------- applier ----------

class TestApply:

    def test_add_round_trip(self, tiny_overlay: OverlayStore):
        intent = classify("remember that Mars has two moons")
        p = orch_plan(intent, overlay=tiny_overlay, base_max_id=10)
        encoder = _StubEncoder(dim=8)
        result = orch_apply(
            p, overlay=tiny_overlay,
            encoder=encoder, whiten_fn=_identity_whiten, bank_dim=8,
        )
        assert result.applied
        assert result.new_cell_id == 11  # base_max_id + 1
        assert result.validator is not None and result.validator.ok
        # Text round-trips through the overlay.
        assert tiny_overlay.fetch_overlay_text(11) == "Mars has two moons"

    def test_apply_refuses_blocked_plan(self, tiny_overlay: OverlayStore):
        intent = classify("nonsense input")
        p = orch_plan(intent, overlay=tiny_overlay)
        result = orch_apply(
            p, overlay=tiny_overlay,
            encoder=_StubEncoder(dim=8), whiten_fn=_identity_whiten, bank_dim=8,
        )
        assert not result.applied
        assert "blocked" in result.error.lower()

    def test_add_apply_needs_encoder(self, tiny_overlay: OverlayStore):
        intent = classify("remember that the sky is blue")
        p = orch_plan(intent, overlay=tiny_overlay, base_max_id=10)
        result = orch_apply(p, overlay=tiny_overlay)
        assert not result.applied
        assert "encoder" in result.error

    def test_remove_round_trip(self, tiny_overlay: OverlayStore):
        intent = classify("tombstone 99: misattributed quote")
        p = orch_plan(intent, overlay=tiny_overlay)
        result = orch_apply(p, overlay=tiny_overlay)
        assert result.applied
        assert result.tombstoned_cell_id == 99
        assert result.validator is not None and result.validator.ok
        # Tombstone is persisted.
        row = tiny_overlay._conn.execute(
            "SELECT reason FROM tombstones WHERE base_cell_id = 99"
        ).fetchone()
        assert row[0] == "misattributed quote"

    def test_inspect(self, tiny_overlay: OverlayStore):
        # Seed some provenance.
        w = np.eye(8, dtype=np.float32)[0]
        tiny_overlay.add_cell(
            weight=w, source_text="seed", source="test", base_max_id=0,
        )
        tiny_overlay.remove_cell(5, reason="example")

        intent = classify("show provenance")
        p = orch_plan(intent, overlay=tiny_overlay)
        result = orch_apply(p, overlay=tiny_overlay)
        assert result.applied
        assert result.provenance is not None
        assert len(result.provenance) == 2


# ---------- post-mutation validator ----------

class TestValidator:

    def test_validate_add_detects_missing_row(self, tiny_overlay: OverlayStore):
        report = validate_add(tiny_overlay, new_cell_id=999, expected_text="x")
        assert not report.ok
        assert report.checks[0][0] == "row_exists"

    def test_validate_add_detects_text_mismatch(self, tiny_overlay: OverlayStore):
        w = np.eye(8, dtype=np.float32)[0]
        new_id = tiny_overlay.add_cell(
            weight=w, source_text="actual text", source="test", base_max_id=0,
        )
        report = validate_add(
            tiny_overlay, new_cell_id=new_id, expected_text="different text",
        )
        assert not report.ok
        names = [c[0] for c in report.checks]
        assert "text_matches" in names

    def test_validate_remove_detects_missing_tombstone(self, tiny_overlay: OverlayStore):
        report = validate_remove(
            tiny_overlay, cell_id=42, expected_reason="x",
        )
        assert not report.ok

    def test_validate_remove_detects_reason_mismatch(self, tiny_overlay: OverlayStore):
        tiny_overlay.remove_cell(42, reason="actual reason")
        report = validate_remove(
            tiny_overlay, cell_id=42, expected_reason="different reason",
        )
        assert not report.ok

    def test_validate_inspect_passes_on_empty_log(self, tiny_overlay: OverlayStore):
        report = validate_inspect(tiny_overlay)
        assert report.ok


# ---------- CLI smoke ----------

class TestCLI:

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        cli = ROOT / "scripts" / "memory.py"
        return subprocess.run(
            [sys.executable, str(cli), *args],
            capture_output=True, text=True, cwd=str(ROOT),
        )

    def test_plan_unknown_returns_exit_2(self, tmp_path: Path):
        overlay_path = tmp_path / "ov.db"
        cp = self._run(
            "plan", "nonsense", "--overlay-path", str(overlay_path), "--json",
        )
        assert cp.returncode == 2
        payload = json.loads(cp.stdout)
        assert payload["blocked"] is True

    def test_apply_without_confirm_returns_exit_2(self, tmp_path: Path):
        overlay_path = tmp_path / "ov.db"
        cp = self._run(
            "apply", "tombstone 7: bad data",
            "--overlay-path", str(overlay_path), "--json",
        )
        assert cp.returncode == 2
        # No tombstone was written.
        ov = OverlayStore(overlay_path)
        try:
            row = ov._conn.execute(
                "SELECT base_cell_id FROM tombstones WHERE base_cell_id = 7"
            ).fetchone()
            assert row is None
        finally:
            ov.close()

    def test_apply_remove_with_confirm_mutates(self, tmp_path: Path):
        overlay_path = tmp_path / "ov.db"
        cp = self._run(
            "apply", "tombstone 7: bad data",
            "--overlay-path", str(overlay_path), "--confirm", "--json",
        )
        assert cp.returncode == 0, cp.stderr
        payload = json.loads(cp.stdout)
        assert payload["applied"] is True
        assert payload["tombstoned_cell_id"] == 7
        assert payload["validator"]["ok"] is True
