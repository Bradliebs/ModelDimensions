"""Release invariants for the v1.0 candidate-recall + fact-verifier path.

These tests pin the safety guarantees the release depends on. They are written
against the public surface only and must keep passing for any change that claims
to preserve v1.0 behaviour. Each test maps to one invariant:

  1. candidate retrieval never grounds by itself;
  2. a deterministic REJECT cannot be overridden;
  3. an AMBIGUOUS candidate is not grounded;
  4. only ACCEPT candidates produce ``memory_used=True``;
  5. a deleted memory cannot be cited;
  6. unsupported-answer-after-grounding remains blocked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.candidate_retrieval import (  # noqa: E402
    ground_accepted_candidates,
    retrieve_candidates,
)
from agent.orchestrator import DeterministicEncoder, MemoryBank  # noqa: E402
from agent.verifier import verify_candidate  # noqa: E402
from slm.equivalence_judge import EquivalenceJudge  # noqa: E402
from slm.schemas import (  # noqa: E402
    MemoryCandidate,
    VerificationVerdict,
    VerifiedMemoryCandidate,
)


def _bank() -> MemoryBank:
    return MemoryBank(DeterministicEncoder(dim=64), epsilon=0.05, radius=0.9)


class _FixedBackend:
    """SLM backend that always tries to ACCEPT; used to prove it cannot."""

    def generate(self, prompt: str) -> str:
        return '{"verdict": "accept", "reason": "forced accept"}'


# 1. Candidate retrieval never grounds by itself. ---------------------------

def test_candidate_retrieval_does_not_ground():
    bank = _bank()
    bank.write("the primary server is located in the Dublin data center")

    candidates = retrieve_candidates(bank, "where is the primary server", k=5)

    assert candidates, "expected at least one candidate"
    # Candidates are MemoryCandidate objects with no grounding semantics: they
    # carry no memory_used flag and citing only happens through grounding.
    for cand in candidates:
        assert isinstance(cand, MemoryCandidate)
        assert not hasattr(cand, "memory_used")
        assert not hasattr(cand, "cited_memory_ids")


# 2. A deterministic REJECT cannot be overridden. ---------------------------

def test_deterministic_reject_cannot_be_overridden_by_slm():
    truth = "the budget request was approved by finance"
    flip = "the budget request was rejected by finance"

    deterministic = verify_candidate(flip, truth)
    assert deterministic is VerificationVerdict.REJECT

    judge = EquivalenceJudge(_FixedBackend())
    judgment = judge.judge(flip, truth, deterministic)

    assert judgment.verdict is VerificationVerdict.REJECT
    assert judge.stats.deterministic_reject_shortcircuit == 1
    # The backend was never consulted: no model verdict was parsed.
    assert judge.stats.valid == 0
    assert judge.stats.malformed_fallback == 0


# 3. An AMBIGUOUS candidate is not grounded. --------------------------------

def test_ambiguous_candidate_is_not_grounded():
    cand = MemoryCandidate(
        memory_id="mem-0001",
        canonical_text="nightly backups are stored in the Frankfurt region",
        activation=0.91,
        rank=1,
        threshold_status="above_threshold",
    )
    verified = [VerifiedMemoryCandidate(
        candidate=cand, verdict=VerificationVerdict.AMBIGUOUS,
        source="deterministic")]

    grounded = ground_accepted_candidates(verified)

    assert grounded.memory_used is False
    assert grounded.refused is True
    assert grounded.cited_memory_ids == []


# 4. Only ACCEPT candidates produce memory_used=True. -----------------------

@pytest.mark.parametrize(
    "verdict,expect_used",
    [
        (VerificationVerdict.ACCEPT, True),
        (VerificationVerdict.AMBIGUOUS, False),
        (VerificationVerdict.REJECT, False),
    ],
)
def test_only_accept_grounds(verdict, expect_used):
    cand = MemoryCandidate(
        memory_id="mem-0001",
        canonical_text="the client meeting is on Tuesday morning",
        activation=0.93,
        rank=1,
        threshold_status="above_threshold",
    )
    verified = [VerifiedMemoryCandidate(
        candidate=cand, verdict=verdict, source="deterministic")]

    grounded = ground_accepted_candidates(verified)

    assert grounded.memory_used is expect_used
    if expect_used:
        assert grounded.cited_memory_ids == ["mem-0001"]
    else:
        assert grounded.cited_memory_ids == []


# 5. A deleted memory cannot be cited. --------------------------------------

def test_deleted_memory_cannot_be_cited():
    bank = _bank()
    rec = bank.write("the project budget is forty thousand dollars")
    mem_id = rec.memory_id

    assert bank.delete(mem_id) is True

    candidates = retrieve_candidates(
        bank, "the project budget is forty thousand dollars", k=5)
    cited_ids = [c.memory_id for c in candidates]
    assert mem_id not in cited_ids

    # Even if a stale candidate object survives, grounding the exact match still
    # cannot resurrect the deleted id through the bank.
    assert bank.get(mem_id) is None


# 6. Unsupported-answer-after-grounding remains blocked. --------------------

def test_unsupported_answer_after_grounding_blocked():
    bank = _bank()
    truth = "the primary server is located in the Dublin data center"
    bank.write(truth)
    flip = "the primary server is located in the Frankfurt data center"

    candidates = retrieve_candidates(bank, flip, k=5)
    assert candidates, "near-miss should still retrieve the loud cell"

    verified = [
        VerifiedMemoryCandidate(
            candidate=c,
            verdict=verify_candidate(flip, c.canonical_text),
            source="deterministic",
        )
        for c in candidates
    ]
    grounded = ground_accepted_candidates(verified)

    # The location flip is a deterministic REJECT, so nothing is grounded.
    assert grounded.memory_used is False
    assert grounded.refused is True
    assert grounded.cited_memory_ids == []
