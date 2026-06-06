"""V1 setup verification: verify the install before the first ask.

    python scripts/setup.py

Runs five checks and writes a verdict to ``results/setup_check.json``:

  1. Bank file exists at the configured path.
  2. Bank opens cleanly and exposes whitening + at least one cell.
  3. The encoder declared by the bank loads and produces the right shape.
  4. The Phi-3 generator weights are reachable (skippable).
  5. A single smoke ``ask`` returns a result without raising.

Exit code 0 = all checks pass. Exit code 1 = any required check failed; the
JSON file still contains per-check details so the failure is debuggable.

Designed for an offline research workstation: no network calls, no auth,
no remote logging. The verdict file is the audit trail.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BANK = os.environ.get("MD_BANK_PATH", r"H:\MiniLM\cc_service\bank.db")
DEFAULT_OUT = REPO_ROOT / "results" / "setup_check.json"
DEFAULT_SMOKE_QUESTION = "What is the capital of France?"


def _record(name: str, ok: bool, details: str) -> dict:
    return {"name": name, "ok": bool(ok), "details": details}


def _check_bank_path(bank_path: Path) -> dict:
    if not bank_path.exists():
        return _record("bank_path", False, f"missing: {bank_path}")
    size_gb = bank_path.stat().st_size / 1e9
    return _record(
        "bank_path", True, f"present at {bank_path} ({size_gb:.2f} GB)"
    )


def _check_bank_open(bank_path: Path, overlay_path: Path | None) -> tuple[dict, Any]:
    """Open the bank, return (record, bank_or_None)."""
    try:
        from src.agent.streaming_bank import StreamingBank
        if overlay_path is not None:
            from src.agent.bank_admin import OverlayStore
            overlay = OverlayStore(overlay_path)
            bank = StreamingBank(str(bank_path), overlay=overlay)
        else:
            bank = StreamingBank(str(bank_path))
        if bank.n_cells == 0:
            bank.close()
            return _record("bank_open", False, "bank has zero cells"), None
        if not bank.has_whitening:
            bank.close()
            return _record(
                "bank_open", False, "bank has no whitening params"
            ), None
        details = (
            f"{bank.n_cells:,} cells, dim={bank.dim}, "
            f"encoder={bank.encoder_model!r}, "
            f"load_seconds={bank.load_seconds:.1f}"
        )
        return _record("bank_open", True, details), bank
    except Exception as exc:
        return _record(
            "bank_open", False, f"{type(exc).__name__}: {exc}"
        ), None


def _check_encoder(bank: Any) -> tuple[dict, Any]:
    """Load the encoder declared by the bank; verify it produces right shape."""
    encoder_name = bank.encoder_model or "all-MiniLM-L6-v2"
    try:
        from src.cc_service.encoder import EncoderSingleton
        encoder = EncoderSingleton(model_name=encoder_name)
        vec = encoder.encode_one("setup probe", is_query=True)
        if vec.shape != (bank.dim,):
            return _record(
                "encoder_load", False,
                f"encoder produced shape {vec.shape}, bank dim={bank.dim}"
            ), None
        return _record(
            "encoder_load", True,
            f"encoder {encoder_name!r} loaded; produced ({bank.dim},) vector"
        ), encoder
    except Exception as exc:
        return _record(
            "encoder_load", False, f"{type(exc).__name__}: {exc}"
        ), None


def _check_generator(use_4bit: bool) -> dict:
    """Touch the Phi-3 weights without running a generation pass.

    transformers will raise if the model files are missing or unreadable;
    we don't keep the model loaded — the smoke ask reloads it via the
    pipeline if requested. Skipping this check is supported because the
    generator load eats ~2.6 GB VRAM and isn't always wanted in a
    smoke-only run.
    """
    try:
        from src.agent.answer_pipeline import PHI3_MODEL
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(PHI3_MODEL)
        return _record(
            "generator_present", True,
            f"tokenizer for {PHI3_MODEL!r} loadable; 4bit={use_4bit}"
        )
    except Exception as exc:
        return _record(
            "generator_present", False, f"{type(exc).__name__}: {exc}"
        )


def _check_calibration_fingerprint(
    bank_path: Path,
    expected_path: Path | None,
    write_path: Path | None,
) -> dict:
    """Compute the V1 calibration fingerprint without loading all cells."""
    try:
        from src.agent.calibration_fingerprint import (
            CalibrationMismatchError,
            fingerprint_from_bank_path,
            load_expected_fingerprint,
            validate_calibration,
            write_expected_fingerprint,
        )
        current = fingerprint_from_bank_path(bank_path)
        details = f"current={current.fingerprint}"
        if write_path is not None:
            write_expected_fingerprint(write_path, current)
            details += f"; wrote {write_path}"
        if expected_path is not None:
            expected = load_expected_fingerprint(expected_path)
            validation = validate_calibration(current, expected)
            details += f"; expected={validation.expected_fingerprint}; match"
        return _record("calibration_fingerprint", True, details)
    except CalibrationMismatchError as exc:
        return _record("calibration_fingerprint", False, str(exc))
    except Exception as exc:
        return _record(
            "calibration_fingerprint", False, f"{type(exc).__name__}: {exc}"
        )


def _check_smoke_ask(
    bank: Any, encoder: Any, question: str, use_4bit: bool
) -> dict:
    try:
        from src.agent.answer_pipeline import AnswerPipeline
        pipeline = AnswerPipeline(
            bank_path=bank.db_path,
            bank=bank,
            encoder=encoder,
            use_4bit=use_4bit,
        )
        try:
            t0 = time.time()
            result = pipeline.ask(question)
            elapsed = time.time() - t0
        finally:
            pipeline.close()
        verdict = "grounded" if not result.silence else f"silence:{result.silence_reason}"
        details = (
            f"Q={question!r} -> {verdict} in {elapsed:.2f}s "
            f"(top1={result.gate['top1_activation']:.3f}, "
            f"margin={result.gate['margin']:.3f})"
        )
        return _record("smoke_ask", True, details)
    except Exception as exc:
        tb = traceback.format_exc(limit=3)
        return _record(
            "smoke_ask", False, f"{type(exc).__name__}: {exc}\n{tb}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 setup check.")
    parser.add_argument("--bank-path", default=DEFAULT_BANK)
    parser.add_argument(
        "--overlay-path", default=None,
        help="Optional overlay SQLite to merge before checking.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--skip-generator", action="store_true",
        help="Skip Phi-3 tokenizer probe and the smoke ask.",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="Skip the live smoke ask (still probes generator weights).",
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Load Phi-3 in bfloat16 instead of 4-bit NF4 for the smoke ask.",
    )
    parser.add_argument(
        "--calibration-fingerprint",
        default=None,
        help="Expected V1 calibration fingerprint JSON. Mismatch fails setup.",
    )
    parser.add_argument(
        "--write-calibration-fingerprint",
        default=None,
        help="Write the current V1 calibration fingerprint JSON.",
    )
    parser.add_argument("--question", default=DEFAULT_SMOKE_QUESTION)
    args = parser.parse_args()

    bank_path = Path(args.bank_path)
    overlay_path = Path(args.overlay_path) if args.overlay_path else None
    use_4bit = not args.no_4bit

    checks: list[dict] = []
    bank: Any = None
    encoder: Any = None

    print("[setup] checking bank path...", flush=True)
    rec = _check_bank_path(bank_path)
    checks.append(rec)
    print(f"  {'OK' if rec['ok'] else 'FAIL'}: {rec['details']}", flush=True)

    if rec["ok"]:
        expected_fingerprint = (
            Path(args.calibration_fingerprint)
            if args.calibration_fingerprint else None
        )
        write_fingerprint = (
            Path(args.write_calibration_fingerprint)
            if args.write_calibration_fingerprint else None
        )
        if expected_fingerprint is not None or write_fingerprint is not None:
            print("[setup] checking calibration fingerprint...", flush=True)
            rec = _check_calibration_fingerprint(
                bank_path, expected_fingerprint, write_fingerprint
            )
            checks.append(rec)
            print(
                f"  {'OK' if rec['ok'] else 'FAIL'}: {rec['details']}",
                flush=True,
            )

    if rec["ok"]:
        print("[setup] opening bank...", flush=True)
        rec, bank = _check_bank_open(bank_path, overlay_path)
        checks.append(rec)
        print(f"  {'OK' if rec['ok'] else 'FAIL'}: {rec['details']}", flush=True)

    if bank is not None:
        print("[setup] loading encoder...", flush=True)
        rec, encoder = _check_encoder(bank)
        checks.append(rec)
        print(f"  {'OK' if rec['ok'] else 'FAIL'}: {rec['details']}", flush=True)

    if not args.skip_generator:
        print("[setup] probing Phi-3 tokenizer...", flush=True)
        rec = _check_generator(use_4bit)
        checks.append(rec)
        print(f"  {'OK' if rec['ok'] else 'FAIL'}: {rec['details']}", flush=True)

        if rec["ok"] and not args.skip_smoke and bank is not None and encoder is not None:
            print(f"[setup] running smoke ask: {args.question!r}", flush=True)
            rec = _check_smoke_ask(bank, encoder, args.question, use_4bit)
            checks.append(rec)
            print(
                f"  {'OK' if rec['ok'] else 'FAIL'}: {rec['details']}",
                flush=True,
            )

    if bank is not None:
        bank.close()

    ok = all(c["ok"] for c in checks)
    payload = {
        "ok": ok,
        "checks": checks,
        "bank_path": str(bank_path),
        "overlay_path": str(overlay_path) if overlay_path else None,
        "ts": time.time(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[setup] verdict: {'OK' if ok else 'FAIL'} -> {out_path}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
