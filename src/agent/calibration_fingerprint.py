"""V1 calibration fingerprinting.

The V1 thresholds are calibrated against a specific geometry: encoder identity,
embedding dimension, whitening parameters, scoring mode, quantisation, and gate
constants. This module binds those materials into a deterministic SHA-256
fingerprint and provides a fail-closed validator.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.answer_pipeline import RESCUE_ACTIVATION_FLOOR, RESCUE_RANK_WINDOW
from src.agent.v1_silence_gate import DEFAULT_MARGIN_THRESHOLD


DEFAULT_SCORING_MODE = "whitened_dot_activation"
DEFAULT_QUANTISATION = "phi3_nf4_4bit"
DEFAULT_CALIBRATION_VERSION = "v1-grounded-answer-pipeline"


class CalibrationMismatchError(RuntimeError):
    """Raised when current calibration material does not match expected."""


def _decode_meta_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _sha256_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


@dataclass(frozen=True)
class CalibrationFingerprint:
    encoder_model: str
    encoder_revision: str
    embedding_dim: int
    whitening_checksum: str
    scoring_mode: str
    quantisation: str
    gate_threshold: float
    rescue_floor: float
    rescue_rank_window: int
    calibration_version: str

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.materials_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def materials_dict(self) -> dict:
        return {
            "encoder_model": self.encoder_model,
            "encoder_revision": self.encoder_revision,
            "embedding_dim": self.embedding_dim,
            "whitening_checksum": self.whitening_checksum,
            "scoring_mode": self.scoring_mode,
            "quantisation": self.quantisation,
            "gate_threshold": self.gate_threshold,
            "rescue_floor": self.rescue_floor,
            "rescue_rank_window": self.rescue_rank_window,
            "calibration_version": self.calibration_version,
        }

    def as_dict(self) -> dict:
        out = self.materials_dict()
        out["fingerprint"] = self.fingerprint
        out["_record"] = "v1_calibration_fingerprint"
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> "CalibrationFingerprint":
        return cls(
            encoder_model=str(raw["encoder_model"]),
            encoder_revision=str(raw.get("encoder_revision", "")),
            embedding_dim=int(raw["embedding_dim"]),
            whitening_checksum=str(raw["whitening_checksum"]),
            scoring_mode=str(raw["scoring_mode"]),
            quantisation=str(raw["quantisation"]),
            gate_threshold=float(raw["gate_threshold"]),
            rescue_floor=float(raw["rescue_floor"]),
            rescue_rank_window=int(raw["rescue_rank_window"]),
            calibration_version=str(raw["calibration_version"]),
        )


@dataclass(frozen=True)
class CalibrationValidation:
    ok: bool
    current_fingerprint: str
    expected_fingerprint: str
    mismatched_fields: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "current_fingerprint": self.current_fingerprint,
            "expected_fingerprint": self.expected_fingerprint,
            "mismatched_fields": list(self.mismatched_fields),
        }


def whitening_checksum(mu_blob: bytes | None, w_matrix_blob: bytes | None, max_norm: float | None) -> str:
    if mu_blob is None or w_matrix_blob is None:
        return "none"
    max_norm_bytes = repr(float(max_norm or 1.0)).encode("ascii")
    return "sha256:" + _sha256_bytes(mu_blob, w_matrix_blob, max_norm_bytes)


def fingerprint_from_bank_path(
    bank_path: Path,
    *,
    encoder_revision: str = "",
    scoring_mode: str = DEFAULT_SCORING_MODE,
    quantisation: str = DEFAULT_QUANTISATION,
    gate_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    rescue_floor: float = RESCUE_ACTIVATION_FLOOR,
    rescue_rank_window: int = RESCUE_RANK_WINDOW,
    calibration_version: str = DEFAULT_CALIBRATION_VERSION,
) -> CalibrationFingerprint:
    """Read only meta + whitening rows; never loads the cell matrix."""
    uri = f"file:{bank_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        meta = {
            str(k): _decode_meta_value(v)
            for k, v in conn.execute("SELECT key, value FROM meta")
        }
        row = conn.execute(
            "SELECT mu, w_matrix, max_norm FROM whitening WHERE id = 1"
        ).fetchone()
    if row is None:
        checksum = "none"
    else:
        checksum = whitening_checksum(row[0], row[1], row[2])
    return CalibrationFingerprint(
        encoder_model=str(meta.get("encoder_model", "")),
        encoder_revision=encoder_revision,
        embedding_dim=int(meta["dim"]),
        whitening_checksum=checksum,
        scoring_mode=scoring_mode,
        quantisation=quantisation,
        gate_threshold=float(gate_threshold),
        rescue_floor=float(rescue_floor),
        rescue_rank_window=int(rescue_rank_window),
        calibration_version=calibration_version,
    )


def load_expected_fingerprint(path: Path) -> CalibrationFingerprint:
    return CalibrationFingerprint.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_expected_fingerprint(path: Path, fingerprint: CalibrationFingerprint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint.as_dict(), indent=2, sort_keys=True), encoding="utf-8")


def validate_calibration(
    current: CalibrationFingerprint,
    expected: CalibrationFingerprint,
    *,
    fail_closed: bool = True,
) -> CalibrationValidation:
    current_materials = current.materials_dict()
    expected_materials = expected.materials_dict()
    mismatches = tuple(
        key for key in sorted(expected_materials)
        if current_materials.get(key) != expected_materials.get(key)
    )
    result = CalibrationValidation(
        ok=not mismatches and current.fingerprint == expected.fingerprint,
        current_fingerprint=current.fingerprint,
        expected_fingerprint=expected.fingerprint,
        mismatched_fields=mismatches,
    )
    if fail_closed and not result.ok:
        fields = ", ".join(result.mismatched_fields) or "fingerprint"
        raise CalibrationMismatchError(f"RECALIBRATION_REQUIRED: {fields}")
    return result
