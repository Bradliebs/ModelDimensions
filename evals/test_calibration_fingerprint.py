"""Tests for V1 calibration fingerprint integrity checks."""
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

from src.agent.calibration_fingerprint import (
    CalibrationFingerprint,
    CalibrationMismatchError,
    fingerprint_from_bank_path,
    load_expected_fingerprint,
    validate_calibration,
    write_expected_fingerprint,
)


_SCHEMA_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE whitening (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    mu BLOB NOT NULL,
    w_matrix BLOB NOT NULL,
    max_norm REAL NOT NULL,
    fitted_at REAL NOT NULL,
    reference_n INTEGER NOT NULL
);
CREATE TABLE cells (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    weight BLOB NOT NULL,
    theta REAL NOT NULL,
    kind TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def _make_bank(path: Path, dim: int = 3, encoder: str = "stub-encoder") -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_SQL)
    conn.execute("INSERT INTO meta VALUES (?, ?)", ("dim", json.dumps(dim)))
    conn.execute("INSERT INTO meta VALUES (?, ?)", ("encoder_model", json.dumps(encoder)))
    mu = np.zeros(dim, dtype=np.float32)
    w_matrix = np.eye(dim, dtype=np.float32)
    conn.execute(
        """INSERT INTO whitening (id, mu, w_matrix, max_norm, fitted_at, reference_n)
           VALUES (1, ?, ?, ?, ?, ?)""",
        (mu.tobytes(), w_matrix.tobytes(), 1.0, time.time(), 10),
    )
    conn.execute(
        "INSERT INTO cells (label, weight, theta, kind, created_at) VALUES (?, ?, ?, ?, ?)",
        ("unused", np.ones(dim, dtype=np.float32).tobytes(), 0.3, "single", time.time()),
    )
    conn.commit()
    conn.close()


def test_given_bank_path_when_fingerprinted_then_materials_are_bound(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)

    # Act
    fp = fingerprint_from_bank_path(bank_path)

    # Assert
    assert fp.encoder_model == "stub-encoder"
    assert fp.embedding_dim == 3
    assert fp.whitening_checksum.startswith("sha256:")
    assert len(fp.fingerprint) == 64


def test_given_written_fingerprint_when_loaded_then_round_trips(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    out_path = tmp_path / "calibration.json"
    _make_bank(bank_path)
    fp = fingerprint_from_bank_path(bank_path)

    # Act
    write_expected_fingerprint(out_path, fp)
    loaded = load_expected_fingerprint(out_path)

    # Assert
    assert loaded == fp
    assert json.loads(out_path.read_text(encoding="utf-8"))["fingerprint"] == fp.fingerprint


def test_given_matching_fingerprints_when_validated_then_ok(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    fp = fingerprint_from_bank_path(bank_path)

    # Act
    result = validate_calibration(fp, fp)

    # Assert
    assert result.ok is True
    assert result.mismatched_fields == ()


def test_given_mismatched_fingerprint_when_validated_then_hard_fails(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    current = fingerprint_from_bank_path(bank_path)
    expected = CalibrationFingerprint(
        **{**current.materials_dict(), "encoder_model": "other-encoder"}
    )

    # Act & Assert
    with pytest.raises(CalibrationMismatchError, match="RECALIBRATION_REQUIRED"):
        validate_calibration(current, expected)


def test_given_mismatched_fingerprint_when_non_throwing_then_reports_field(tmp_path: Path):
    # Arrange
    bank_path = tmp_path / "bank.db"
    _make_bank(bank_path)
    current = fingerprint_from_bank_path(bank_path)
    expected = CalibrationFingerprint(
        **{**current.materials_dict(), "gate_threshold": current.gate_threshold + 0.1}
    )

    # Act
    result = validate_calibration(current, expected, fail_closed=False)

    # Assert
    assert result.ok is False
    assert result.mismatched_fields == ("gate_threshold",)