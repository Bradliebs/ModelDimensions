"""v7.1 active-pack monitoring: immutable baselines.

A baseline is created explicitly (separate from a run), is immutable (a new
baseline needs a new identity and path — overwriting fails closed), and is
comparable to a run only when corpus, retrieval config and evaluator match. The
active-state hash is deliberately *not* part of compatibility, because the
baseline is captured pre-activation and the active state is expected to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.active_pack_monitor as apm  # noqa: E402
from agent.knowledge_pack_activation import (  # noqa: E402
    ActivationStateManifest,
    ActivePackState,
    PackLifecycleState,
)

FIXED_NOW = "2024-01-01T00:00:00+00:00"


def _snapshot(state_hash_seed: str = "a"):
    rec = ActivePackState(
        pack_id="p1", pack_version="1.0",
        pack_fingerprint=f"packfp-{state_hash_seed}",
        source_type="hf", source_id="src1", source_revision="r1",
        status=PackLifecycleState.ACTIVE, activated_at=FIXED_NOW,
        activation_approval_id="ap1", evaluation_report_id="ev1")
    return apm.snapshot_from_state(
        ActivationStateManifest(records=(rec,)), captured_at=FIXED_NOW)


def _results():
    return [apm.MonitoringCaseResult(
        case_id=f"c{i}", case_class=apm.MonitoringCaseClass.PACK_RELEVANT,
        severity=apm.MonitoringSeverity.MEDIUM, hit=True,
        expected_source_recall=1.0, wrong_source_rate=0.0, passed=True)
        for i in range(4)]


def _baseline(snapshot=None, corpus_fp="moncorpus-x", rcfg="retrcfg-y"):
    return apm.create_baseline(
        baseline_id="b1", baseline_type=apm.BaselineType.PRE_ACTIVATION,
        snapshot=snapshot or _snapshot(), corpus_fingerprint=corpus_fp,
        retrieval_config_fingerprint=rcfg, results=_results(), created_at=FIXED_NOW)


def test_baseline_hash_is_stable_and_immutable():
    a = _baseline()
    b = _baseline()
    assert a.baseline_hash == b.baseline_hash
    assert a.baseline_hash.startswith(apm.BASELINE_HASH_PREFIX)


def test_baseline_round_trips_through_dict():
    a = _baseline()
    restored = apm.MonitoringBaseline.from_dict(a.to_dict())
    assert restored.baseline_hash == a.baseline_hash
    assert restored.metrics.to_dict() == a.metrics.to_dict()


def test_writing_baseline_twice_fails_closed(tmp_path):
    path = tmp_path / "baseline.json"
    apm.write_baseline(_baseline(), path)
    assert path.exists()
    with pytest.raises(FileExistsError):
        apm.write_baseline(_baseline(), path)


def test_loaded_baseline_matches_written(tmp_path):
    path = tmp_path / "baseline.json"
    apm.write_baseline(_baseline(), path)
    loaded = apm.load_baseline(path)
    assert loaded.baseline_hash == _baseline().baseline_hash


def test_compatibility_ignores_active_state_but_requires_corpus_and_config():
    base = _baseline(snapshot=_snapshot("a"), corpus_fp="moncorpus-1",
                     rcfg="retrcfg-1")
    # Same corpus/config but a different active state -> still compatible.
    assert apm.baseline_compatible(
        base, corpus_fingerprint="moncorpus-1",
        retrieval_config_fingerprint="retrcfg-1")
    # Different corpus -> incompatible.
    assert not apm.baseline_compatible(
        base, corpus_fingerprint="moncorpus-2",
        retrieval_config_fingerprint="retrcfg-1")
    # Different retrieval config -> incompatible.
    assert not apm.baseline_compatible(
        base, corpus_fingerprint="moncorpus-1",
        retrieval_config_fingerprint="retrcfg-2")


def test_from_dict_rejects_non_baseline_record():
    with pytest.raises(ValueError):
        apm.MonitoringBaseline.from_dict({"_record": "not_a_baseline"})


def test_written_baseline_is_valid_json(tmp_path):
    path = tmp_path / "baseline.json"
    apm.write_baseline(_baseline(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["_record"] == "monitoring_baseline"
    assert data["baseline_type"] == "pre_activation"
