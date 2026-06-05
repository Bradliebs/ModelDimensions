"""v7.1 active-pack monitoring CLI: read-only guarantees.

The monitor CLI never changes active-pack state: it writes nothing in stdout
mode, writes only a baseline under ``baseline --write``, writes only a report
under ``run --out``, appends history only under ``--history``, and never creates
the governed activation state or audit files. Output is deterministic for fixed
inputs and labels its recommendation advisory only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import workbench  # noqa: E402

_CASES = ROOT / "demos" / "active_pack_monitor_cases.jsonl"
_BASE_ARGS = ["active-pack-monitor", "--deterministic"]


def _make_baseline(tmp_path, capsys) -> Path:
    out = tmp_path / "baseline.json"
    capsys.readouterr()
    rc = workbench.main(_BASE_ARGS + [
        "baseline", "--cases", str(_CASES), "--out", str(out), "--write"])
    capsys.readouterr()
    assert rc == 0
    return out


def test_baseline_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    capsys.readouterr()
    out = tmp_path / "baseline.json"
    rc = workbench.main(_BASE_ARGS + [
        "baseline", "--cases", str(_CASES), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "dry-run" in text
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_baseline_write_creates_only_the_baseline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    out = _make_baseline(tmp_path, capsys)
    created = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert created == [out]
    assert '"_record": "monitoring_baseline"' in out.read_text(encoding="utf-8")


def test_run_stdout_mode_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    base = _make_baseline(tmp_path, capsys)
    before = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    capsys.readouterr()
    rc = workbench.main(_BASE_ARGS + [
        "run", "--baseline", str(base), "--cases", str(_CASES)])
    out = capsys.readouterr().out
    assert rc in (0, 1)  # 1 only signals a critical finding (advisory)
    assert "ADVISORY ONLY" in out
    after = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    assert after == before  # no new files


def test_run_out_writes_only_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    base = _make_baseline(tmp_path, capsys)
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    report = tmp_path / "report.md"
    capsys.readouterr()
    workbench.main(_BASE_ARGS + [
        "run", "--baseline", str(base), "--cases", str(_CASES),
        "--out", str(report)])
    capsys.readouterr()
    created = {p for p in tmp_path.rglob("*") if p.is_file()} - before
    assert created == {report}
    assert "monitoring report" in report.read_text(encoding="utf-8").lower()


def test_run_does_not_create_governed_state_or_audit(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    base = _make_baseline(tmp_path, capsys)
    capsys.readouterr()
    workbench.main(_BASE_ARGS + [
        "run", "--baseline", str(base), "--cases", str(_CASES)])
    capsys.readouterr()
    # The monitor never writes the activation state or audit log.
    assert not (tmp_path / "config" / "active_knowledge_packs.jsonl").exists()
    assert not (tmp_path / "reports"
                / "knowledge_pack_activation_audit.jsonl").exists()


def test_run_output_is_deterministic(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    base = _make_baseline(tmp_path, capsys)
    capsys.readouterr()
    workbench.main(_BASE_ARGS + [
        "run", "--baseline", str(base), "--cases", str(_CASES)])
    out_a = capsys.readouterr().out
    workbench.main(_BASE_ARGS + [
        "run", "--baseline", str(base), "--cases", str(_CASES)])
    out_b = capsys.readouterr().out
    assert out_a == out_b


def test_history_appends_only_under_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    base = _make_baseline(tmp_path, capsys)
    history = tmp_path / "history.jsonl"
    capsys.readouterr()
    workbench.main(_BASE_ARGS + [
        "run", "--baseline", str(base), "--cases", str(_CASES),
        "--history", "--history-path", str(history)])
    capsys.readouterr()
    assert history.exists()
    line_count = len(history.read_text(encoding="utf-8").strip().splitlines())
    assert line_count == 1
    # A second run appends (append-only), never rewrites.
    workbench.main(_BASE_ARGS + [
        "run", "--baseline", str(base), "--cases", str(_CASES),
        "--history", "--history-path", str(history)])
    capsys.readouterr()
    assert len(history.read_text(encoding="utf-8").strip().splitlines()) == 2
