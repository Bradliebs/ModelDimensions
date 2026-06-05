"""v7.1 active-pack monitoring: conservative regression attribution.

A regression is attributed to a specific pack ONLY when it appears with that pack
enabled, disappears with it disabled, configuration and corpus are unchanged, and
repeated runs agree. Every other situation is classified more conservatively —
never as pack-specific by assumption.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent.active_pack_monitor as apm  # noqa: E402


def _attribute(**kw):
    base = dict(
        regression_with_pack=True, regression_without_pack=False,
        retrieval_config_changed=False, corpus_changed=False,
        evaluator_changed=False, stable=True, candidate_pack_ids=["p1"])
    base.update(kw)
    return apm.attribute_regression(**base)


def test_pack_specific_only_when_disablement_removes_regression():
    result = _attribute(regression_with_pack=True, regression_without_pack=False)
    assert result.attribution == apm.AttributionClass.PACK_SPECIFIC
    assert result.confidence == apm.ConfidenceBand.HIGH


def test_persisting_regression_is_active_set_interaction():
    result = _attribute(regression_with_pack=True, regression_without_pack=True)
    assert result.attribution == apm.AttributionClass.ACTIVE_SET_INTERACTION


def test_unstable_runs_are_undetermined_not_pack_specific():
    result = _attribute(regression_with_pack=True, regression_without_pack=False,
                        stable=False)
    assert result.attribution == apm.AttributionClass.UNSTABLE_RESULT
    assert result.attribution != apm.AttributionClass.PACK_SPECIFIC


def test_corpus_change_prevents_pack_specific():
    result = _attribute(corpus_changed=True)
    assert result.attribution == apm.AttributionClass.CORPUS_CHANGE


def test_retrieval_config_change_prevents_pack_specific():
    result = _attribute(retrieval_config_changed=True)
    assert result.attribution == apm.AttributionClass.RETRIEVAL_CONFIGURATION


def test_evaluator_change_prevents_pack_specific():
    result = _attribute(evaluator_changed=True)
    assert result.attribution == apm.AttributionClass.EVALUATOR_CHANGE


def test_no_regression_either_way_is_undetermined():
    result = _attribute(regression_with_pack=False, regression_without_pack=False)
    assert result.attribution == apm.AttributionClass.UNDETERMINED
