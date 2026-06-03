"""Tests for the v1.0-rc2 candidate-recall + fact-verifier layer.

Covers the deterministic verifier (the one-word flips that fooled Exp 09) and
the rule that the optional SLM judge can never override a deterministic REJECT.

    python -m pytest evals/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.verifier import (
    detect_antonym_flip_basic,
    detect_date_or_weekday_mismatch,
    detect_entity_mismatch,
    detect_negation_flip,
    detect_number_mismatch,
    verify_candidate,
)
from slm.equivalence_judge import EquivalenceJudge
from slm.schemas import VerificationVerdict


# ---------- deterministic REJECTs (the Exp 09 failure cases) ----------

def test_weekday_flip_friday_vs_monday_rejects():
    q = "the product launch is scheduled for Monday afternoon"
    c = "the product launch is scheduled for Friday afternoon"
    assert detect_date_or_weekday_mismatch(q, c) is True
    assert verify_candidate(q, c) is VerificationVerdict.REJECT


def test_number_flip_100_vs_1000_rejects():
    q = "the project budget is 1000 dollars"
    c = "the project budget is 100 dollars"
    assert detect_number_mismatch(q, c) is True
    assert verify_candidate(q, c) is VerificationVerdict.REJECT


def test_number_word_flip_forty_vs_ninety_rejects():
    q = "the project budget is ninety thousand dollars"
    c = "the project budget is forty thousand dollars"
    assert detect_number_mismatch(q, c) is True
    assert verify_candidate(q, c) is VerificationVerdict.REJECT


def test_antonym_approved_vs_rejected_rejects():
    q = "the request was rejected by the board"
    c = "the request was approved by the board"
    assert detect_antonym_flip_basic(q, c) is True
    assert verify_candidate(q, c) is VerificationVerdict.REJECT


def test_antonym_safe_vs_unsafe_rejects():
    q = "the configuration is unsafe for production"
    c = "the configuration is safe for production"
    assert detect_antonym_flip_basic(q, c) is True
    assert verify_candidate(q, c) is VerificationVerdict.REJECT


def test_antonym_increase_vs_decrease_rejects():
    q = "the latency will decrease next quarter"
    c = "the latency will increase next quarter"
    assert detect_antonym_flip_basic(q, c) is True
    assert verify_candidate(q, c) is VerificationVerdict.REJECT


def test_negation_flip_rejects():
    q = "the database password never rotates after ninety days"
    c = "the database password rotates every ninety days"
    assert detect_negation_flip(q, c) is True
    assert verify_candidate(q, c) is VerificationVerdict.REJECT


def test_entity_substitution_rejects():
    q = "the primary server is located in the Frankfurt data center"
    c = "the primary server is located in the Dublin data center"
    assert detect_entity_mismatch(q, c) is True
    assert verify_candidate(q, c) is VerificationVerdict.REJECT


# ---------- ACCEPT / AMBIGUOUS ----------

def test_exact_match_accepts():
    text = "the client meeting is on Tuesday morning"
    assert verify_candidate(text, text) is VerificationVerdict.ACCEPT


def test_fixture_paraphrase_accepts():
    # Defined in the verifier's _SAFE_PARAPHRASES fixture.
    q = "there is a morning client meeting on Tuesday"
    c = "the client meeting is on Tuesday morning"
    assert verify_candidate(q, c) is VerificationVerdict.ACCEPT


def test_uncertain_paraphrase_is_ambiguous():
    # Genuine paraphrase, not in the fixture, no mismatch, not contained.
    q = "we are releasing the product on Friday afternoon"
    c = "the product launch is scheduled for Friday afternoon"
    assert verify_candidate(q, c) is VerificationVerdict.AMBIGUOUS


def test_unrelated_text_is_not_accepted():
    q = "the cat slept on the warm windowsill all day"
    c = "the product launch is scheduled for Friday afternoon"
    assert verify_candidate(q, c) is not VerificationVerdict.ACCEPT


# ---------- SLM judge interaction ----------

class _FixedBackend:
    """Backend that always returns the same canned text."""

    def __init__(self, text: str):
        self.text = text

    def generate(self, prompt: str) -> str:
        return self.text


def test_slm_judge_cannot_override_deterministic_reject():
    # Model screams ACCEPT, but the deterministic layer already said REJECT.
    backend = _FixedBackend('{"verdict": "accept", "reason": "looks fine"}')
    judge = EquivalenceJudge(backend)
    out = judge.judge(
        "the launch is on Monday", "the launch is on Friday",
        deterministic_verdict=VerificationVerdict.REJECT,
    )
    assert out.verdict is VerificationVerdict.REJECT
    assert judge.stats.deterministic_reject_shortcircuit == 1


def test_malformed_slm_output_falls_back_to_ambiguous():
    backend = _FixedBackend("not json at all, sorry")
    judge = EquivalenceJudge(backend)
    out = judge.judge(
        "a paraphrase here", "a slightly different paraphrase",
        deterministic_verdict=VerificationVerdict.AMBIGUOUS,
    )
    assert out.verdict is VerificationVerdict.AMBIGUOUS
    assert judge.stats.malformed_fallback == 1


def test_slm_judge_accept_on_ambiguous_is_allowed():
    backend = _FixedBackend('{"verdict": "accept", "reason": "same fact"}')
    judge = EquivalenceJudge(backend)
    out = judge.judge(
        "the app ships in September", "version two ships in September",
        deterministic_verdict=VerificationVerdict.AMBIGUOUS,
    )
    assert out.verdict is VerificationVerdict.ACCEPT


def test_verifier_functions_are_pure_and_deterministic():
    q = "the project budget is ninety thousand dollars"
    c = "the project budget is forty thousand dollars"
    assert verify_candidate(q, c) == verify_candidate(q, c)
    assert detect_number_mismatch(q, c) == detect_number_mismatch(q, c)
