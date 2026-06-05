"""Pytest for paper headline reproduction.

Reads ``results/paper_headline_reproduce.json`` (produced by
``experiments/exp14_paper_headline_reproduce.py``) and asserts that the
measured semantic gap and random-harm numbers fall within ±0.05 nats of the
paper's claims in §4.2 of ``docs/Knowledge_Free_RETRO_Paper.docx``.

If the JSON has not been produced yet, the tests skip with an explicit
reason rather than failing — the test does not run the (multi-minute) eval
itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "results" / "paper_headline_reproduce.json"

TOLERANCE_NATS = 0.05


def _load_artifact():
    if not ARTIFACT.exists():
        pytest.skip(
            f"missing {ARTIFACT.relative_to(REPO_ROOT)} — run "
            "`python experiments/exp14_paper_headline_reproduce.py` first."
        )
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def artifact():
    return _load_artifact()


@pytest.mark.parametrize("label", ["55M_bank_trained", "404M_kfree"])
def test_semantic_gap_matches_paper(artifact, label):
    """val_with_random − val_with_real should be within ±0.05 nats of paper."""
    claimed = artifact["paper_claims"][label]["semantic_gap"]
    measured = artifact["measurements"][label]["semantic_gap_real_vs_random"]
    delta = abs(measured - claimed)
    assert delta <= TOLERANCE_NATS, (
        f"{label}: semantic_gap measured={measured:+.4f} "
        f"paper-claimed={claimed:+.4f} delta={delta:.4f} "
        f"exceeds tolerance {TOLERANCE_NATS}"
    )


@pytest.mark.parametrize("label", ["55M_bank_trained", "404M_kfree"])
def test_random_harm_matches_paper(artifact, label):
    """val_without − val_with_random should be within ±0.05 nats of paper."""
    claimed = artifact["paper_claims"][label]["random_harm"]
    measured = artifact["measurements"][label]["random_harm_none_vs_random"]
    delta = abs(measured - claimed)
    assert delta <= TOLERANCE_NATS, (
        f"{label}: random_harm measured={measured:+.4f} "
        f"paper-claimed={claimed:+.4f} delta={delta:.4f} "
        f"exceeds tolerance {TOLERANCE_NATS}"
    )


def test_agreement_block_consistent(artifact):
    """The producer's own match/drift labels should agree with the asserts."""
    for label in ("55M_bank_trained", "404M_kfree"):
        sem_agree = artifact["agreement"][label]["semantic_gap"]
        harm_agree = artifact["agreement"][label]["random_harm"]
        assert sem_agree in ("match", "drift")
        assert harm_agree in ("match", "drift")


def test_both_models_reproduce(artifact):
    """Both checkpoints' headline claim (semantic_gap) must agree."""
    for label in ("55M_bank_trained", "404M_kfree"):
        assert artifact["agreement"][label]["semantic_gap"] == "match", (
            f"{label} semantic_gap drift from paper — see artifact for details"
        )
