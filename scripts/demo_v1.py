"""Concept Cells v1.0 — minimal demo of explicit candidate memory.

Smallest possible end-to-end walkthrough of the v1.0 thesis:

    Concept cells are not semantic truth machines. They are explicit candidate
    memories. Verification and grounding decide whether a candidate can be used.

The demo is fully offline and deterministic (no model download): retrieval uses
the hash-seeded ``DeterministicEncoder``, and the verifier is encoder-independent
lexical logic. It mutates nothing in the frozen core — it only calls the public
agent surface (``MemoryBank``, ``retrieve_candidates``, ``verify_candidate``,
``ground_accepted_candidates``).

Six actions, each printing the audit trail (candidate retrieved, verifier
verdict, memory_used, refused):

    1. Write a memory
    2. Query the exact memory          -> ACCEPT, grounded
    3. Query a paraphrase              -> ACCEPT (containment), grounded
    4. Query a dangerous near-miss     -> REJECT, NOT grounded   (the key story)
    5. (verdict shown for every query)
    6. Delete the memory and query     -> silent, NOT grounded

    python scripts/demo_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.candidate_retrieval import (  # noqa: E402
    ground_accepted_candidates,
    retrieve_candidates,
)
from agent.orchestrator import DeterministicEncoder, MemoryBank  # noqa: E402
from agent.verifier import verify_candidate  # noqa: E402
from slm.schemas import VerifiedMemoryCandidate  # noqa: E402

# epsilon is large so a single stored memory always offers itself as a
# candidate regardless of the offline encoder's geometry; safety is decided by
# the verifier, not by retrieval.
EPSILON = 0.25
RADIUS = 0.9
K = 5


def _rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def answer(bank: MemoryBank, query: str) -> None:
    """Run retrieve -> verify -> ground for one query and print the audit trail."""
    print(f'\nQuery: "{query}"')

    candidates = retrieve_candidates(bank, query, K)
    if not candidates:
        print("  candidate_retrieved = false")
    for cand in candidates:
        print(
            f"  candidate_retrieved  = true  "
            f"({cand.memory_id}, activation={cand.activation:.4f})"
        )

    verified = [
        VerifiedMemoryCandidate(
            candidate=cand,
            verdict=verify_candidate(query, cand.canonical_text),
        )
        for cand in candidates
    ]
    for v in verified:
        print(f"  verifier_verdict     = {v.verdict.value.upper()}")

    response = ground_accepted_candidates(verified)
    print(f"  memory_used          = {str(response.memory_used).lower()}")
    print(f"  refused              = {str(response.refused).lower()}")
    print(f"  response             = {response.text.replace(chr(10), ' / ')}")


def main() -> None:
    bank = MemoryBank(DeterministicEncoder(dim=64), epsilon=EPSILON,
                      radius=RADIUS)

    _rule("1. WRITE a memory")
    fact = "the supplier delivery is on Friday afternoon"
    rec = bank.write(fact)
    print(f'\nStored {rec.memory_id}: "{fact}"')

    _rule("2. QUERY the exact memory  (expect ACCEPT -> grounded)")
    answer(bank, "the supplier delivery is on Friday afternoon")

    _rule("3. QUERY a paraphrase  (expect ACCEPT via containment -> grounded)")
    answer(bank, "the supplier delivery is Friday")

    _rule("4. QUERY a dangerous near-miss  (expect REJECT -> NOT grounded)")
    print(
        "\nThe stored fact says Friday. This query says Monday — one word "
        "changed.\nMiniLM ranks this near-miss high in cosine space (Exp 09), "
        "so a\nthreshold alone would admit it. The verifier rejects the "
        "weekday flip."
    )
    answer(bank, "the supplier delivery is on Monday afternoon")

    _rule("5. DELETE the memory, then query again  (expect silent refusal)")
    removed = bank.delete(rec.memory_id)
    print(f"\nDeleted {rec.memory_id}: {removed}")
    answer(bank, "the supplier delivery is on Friday afternoon")

    _rule("Summary")
    print(
        "\nA candidate being retrieved is not the same as a fact being true.\n"
        "Exact and paraphrase queries are grounded only after the verifier\n"
        "ACCEPTs them; the Friday->Monday near-miss is retrieved but REJECTed,\n"
        "so it never reaches the user as memory; a deleted memory is silent.\n"
    )


if __name__ == "__main__":
    main()
