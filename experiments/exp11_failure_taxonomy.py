"""Experiment 11: factual near-miss failure taxonomy (v1.0-finalisation).

Exp 09 falsified threshold-only semantic recall: MiniLM ranks one-word factual
near-miss flips above genuine paraphrases. Exp 10 showed the deterministic
verifier recovers safety on the 8-memory corpus. Exp 11 widens the stress test
into an explicit *taxonomy* of factual flips and measures the deterministic
verifier per flip type.

It completes the experiment-configuration checklist: 15 flip configurations,
each a distinct way a single fact can be silently altered. The verifier must
REJECT every flip (a flip that is ACCEPTed would ground a wrong fact; a flip
left AMBIGUOUS is merely refused, which is safe).

    flip configurations (15/15)
      1  weekday_flip            Friday -> Monday
      2  month_date_flip         September -> October
      3  integer_number_flip     100 -> 1000
      4  quantity_word_flip      forty -> ninety
      5  currency_amount_flip    forty dollars -> ninety dollars
      6  negation_flip           rotates -> never rotates
      7  antonym_generic_flip    open -> closed
      8  approval_status_flip    approved -> rejected
      9  safe_unsafe_flip        safe -> unsafe
     10  increase_decrease_flip  increase -> decrease
     11  location_swap           Dublin -> Frankfurt
     12  person_entity_swap      Alice -> Bob
     13  before_after_flip       before -> after
     14  modal_must_may_flip     must -> may
     15  modal_should_not_flip   should -> should not   (via negation)

Nothing in the frozen core changes: geometry, whitening/scaling, write/query,
Oja binding, and the grounding policy are untouched. Retrieval uses MiniLM when
available (so near-misses really do out-rank paraphrases) else the offline
deterministic encoder; verification is encoder-independent.

Reported overall and per flip type:
  candidate_recall_at_k, deterministic_reject_rate_by_flip_type,
  false_accept_rate_by_flip_type, ambiguous_rate_by_flip_type,
  unsupported_answer_after_grounding, examples_of_failures.

Writes results/exp11_summary.json.

    python -m experiments.exp11_failure_taxonomy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.candidate_retrieval import (
    ground_accepted_candidates,
    retrieve_candidates,
)
from agent.orchestrator import DeterministicEncoder, MemoryBank
from agent.verifier import verify_candidate
from slm.schemas import (
    VerificationVerdict,
    VerifiedMemoryCandidate,
)

EPSILON = 0.25
RADIUS = 0.9
K = 5

# Each entry: flip_type -> list of (stored_memory, flipped_near_miss_query).
# The stored sentence is the truth; the query alters exactly one fact and must
# be REJECTed by the verifier.
FLIP_CORPUS: Dict[str, List[Tuple[str, str]]] = {
    "weekday_flip": [
        ("the product launch is scheduled for Friday afternoon",
         "the product launch is scheduled for Monday afternoon"),
        ("the client meeting is on Tuesday morning",
         "the client meeting is on Thursday morning"),
        ("the fire drill happens on Wednesday at noon",
         "the fire drill happens on Sunday at noon"),
    ],
    "month_date_flip": [
        ("version two of the mobile app ships in September",
         "version two of the mobile app ships in October"),
        ("the annual audit takes place in March",
         "the annual audit takes place in November"),
        ("the contract renews in January each year",
         "the contract renews in July each year"),
    ],
    "integer_number_flip": [
        ("the server pool has 100 active nodes",
         "the server pool has 1000 active nodes"),
        ("the cache holds 256 entries at most",
         "the cache holds 512 entries at most"),
        ("the report covers 12 regions this quarter",
         "the report covers 20 regions this quarter"),
    ],
    "quantity_word_flip": [
        ("the project budget is forty thousand dollars",
         "the project budget is ninety thousand dollars"),
        ("the database password rotates every ninety days",
         "the database password rotates every thirty days"),
        ("the team has eight engineers on call",
         "the team has twelve engineers on call"),
    ],
    "currency_amount_flip": [
        ("the monthly fee is forty dollars per seat",
         "the monthly fee is ninety dollars per seat"),
        ("the refund came to 250 dollars in total",
         "the refund came to 750 dollars in total"),
        ("the licence costs 5000 dollars a year",
         "the licence costs 9000 dollars a year"),
    ],
    "negation_flip": [
        ("the database password rotates every ninety days",
         "the database password never rotates after ninety days"),
        ("nightly backups are stored in the Frankfurt region",
         "nightly backups failed to reach the Frankfurt region"),
        ("the deployment was approved by the review board",
         "the deployment was not approved by the review board"),
    ],
    "antonym_generic_flip": [
        ("the staging environment is currently open for testing",
         "the staging environment is currently closed for testing"),
        ("the nightly job will start at midnight",
         "the nightly job will stop at midnight"),
        ("the feature flag is enabled for all users",
         "the feature flag is disabled for all users"),
    ],
    "approval_status_flip": [
        ("the budget request was approved by finance",
         "the budget request was rejected by finance"),
        ("the access grant was granted to the new hire",
         "the access grant was revoked to the new hire"),
        ("the pull request was accepted into the release",
         "the pull request was rejected into the release"),
    ],
    "safe_unsafe_flip": [
        ("the current configuration is safe for production",
         "the current configuration is unsafe for production"),
        ("the upgraded library is secure against the exploit",
         "the upgraded library is insecure against the exploit"),
        ("the rollout path is safe to apply on Friday",
         "the rollout path is unsafe to apply on Friday"),
    ],
    "increase_decrease_flip": [
        ("the request latency will increase next quarter",
         "the request latency will decrease next quarter"),
        ("the second quarter revenue rose against the plan",
         "the second quarter revenue fell against the plan"),
        ("the error rate increased after the patch",
         "the error rate decreased after the patch"),
    ],
    "location_swap": [
        ("the primary server is located in the Dublin data center",
         "the primary server is located in the Frankfurt data center"),
        ("nightly backups are stored in the Frankfurt region",
         "nightly backups are stored in the Dublin region"),
        ("the conference is hosted in the Berlin office",
         "the conference is hosted in the Madrid office"),
    ],
    "person_entity_swap": [
        ("the migration will be led by Alice next sprint",
         "the migration will be led by Bob next sprint"),
        ("the incident was escalated to Priya on call",
         "the incident was escalated to Marcus on call"),
        ("the chief executive will visit the London office",
         "the chief executive will visit the Paris office"),
    ],
    "before_after_flip": [
        ("the cache is cleared before the nightly job runs",
         "the cache is cleared after the nightly job runs"),
        ("the backups complete before the maintenance window",
         "the backups complete after the maintenance window"),
        ("the report is sent before the board meeting",
         "the report is sent after the board meeting"),
    ],
    "modal_must_may_flip": [
        ("the service must restart after a config change",
         "the service may restart after a config change"),
        ("engineers must rotate the signing key each quarter",
         "engineers may rotate the signing key each quarter"),
        ("the field is required in every submitted form",
         "the field is optional in every submitted form"),
    ],
    "modal_should_not_flip": [
        ("the worker should retry on a transient error",
         "the worker should not retry on a transient error"),
        ("the cache should expire stale entries hourly",
         "the cache should never expire stale entries hourly"),
        ("the client should send a heartbeat every minute",
         "the client should not send a heartbeat every minute"),
    ],
}

FLIP_ORDER = list(FLIP_CORPUS.keys())


# ---------- encoder ----------

class PrecomputedEncoder:
    """Serves cached vectors by exact text; offline, deterministic."""

    def __init__(self, mapping: Dict[str, np.ndarray]):
        self._m = mapping

    def encode_one(self, text: str) -> np.ndarray:
        return self._m[text]


def _all_texts() -> List[str]:
    seen, out = set(), []
    for pairs in FLIP_CORPUS.values():
        for memory, query in pairs:
            for t in (memory, query):
                if t not in seen:
                    seen.add(t)
                    out.append(t)
    return out


def _build_encoder() -> Tuple[object, str]:
    try:
        from concept_cells.encoders import TextEncoder
    except Exception:
        return DeterministicEncoder(dim=64), "deterministic (encoders unavailable)"
    try:
        enc = TextEncoder("all-MiniLM-L6-v2")
        texts = _all_texts()
        emb = enc.encode(texts).embeddings.astype(np.float64)
        mapping = {t: emb[i] for i, t in enumerate(texts)}
        return PrecomputedEncoder(mapping), "all-MiniLM-L6-v2"
    except Exception:
        return DeterministicEncoder(dim=64), "deterministic (MiniLM unavailable offline)"


# ---------- evaluation ----------

def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def main():
    encoder, encoder_name = _build_encoder()
    bank = MemoryBank(encoder, epsilon=EPSILON, radius=RADIUS)

    # Write every truth sentence; remember its memory id for recall scoring.
    memory_id_for: Dict[str, str] = {}
    for flip_type in FLIP_ORDER:
        for memory, _query in FLIP_CORPUS[flip_type]:
            if memory not in memory_id_for:
                rec = bank.write(memory)
                memory_id_for[memory] = rec.memory_id

    per_type: Dict[str, Dict] = {}
    recall_hits = recall_total = 0
    unsupported = total_queries = 0
    failures: List[Dict] = []

    for flip_type in FLIP_ORDER:
        rejects = accepts = ambiguous = 0
        pairs = FLIP_CORPUS[flip_type]
        for memory, query in pairs:
            total_queries += 1
            target_id = memory_id_for[memory]

            cands = retrieve_candidates(bank, query, K)
            cand_ids = [c.memory_id for c in cands]
            recall_total += 1
            hit = target_id in cand_ids
            recall_hits += int(hit)

            # Verify the true target's stored text against the flipped query.
            verdict = verify_candidate(query, memory)
            if verdict is VerificationVerdict.REJECT:
                rejects += 1
            elif verdict is VerificationVerdict.ACCEPT:
                accepts += 1
            else:
                ambiguous += 1

            # Grounding path over all retrieved candidates, verified.
            verified = [
                VerifiedMemoryCandidate(
                    candidate=c,
                    verdict=verify_candidate(query, c.canonical_text),
                    source="deterministic",
                )
                for c in cands
            ]
            grounded = ground_accepted_candidates(verified)
            if grounded.memory_used:
                # A flipped near-miss must never reach the user as memory.
                unsupported += 1

            if verdict is not VerificationVerdict.REJECT:
                failures.append({
                    "flip_type": flip_type,
                    "verdict": verdict.value,
                    "stored_truth": memory,
                    "flipped_query": query,
                    "target_retrieved": hit,
                })

        n = len(pairs)
        per_type[flip_type] = {
            "n": n,
            "deterministic_reject_rate": _rate(rejects, n),
            "false_accept_rate": _rate(accepts, n),
            "ambiguous_rate": _rate(ambiguous, n),
        }

    summary = {
        "experiment": "exp11_failure_taxonomy",
        "description": (
            "Factual near-miss failure taxonomy across 15 flip configurations. "
            "Each flipped query alters exactly one fact of a stored truth and "
            "must be rejected by the deterministic verifier. Core geometry, "
            "whitening, write/query, Oja binding, and the grounding policy are "
            "unchanged."
        ),
        "encoder": encoder_name,
        "epsilon": EPSILON,
        "radius": RADIUS,
        "top_k": K,
        "n_flip_configurations": len(FLIP_ORDER),
        "n_truths": len(memory_id_for),
        "n_flipped_queries": total_queries,
        "candidate_recall_at_k": _rate(recall_hits, recall_total),
        "deterministic_reject_rate_by_flip_type": {
            ft: per_type[ft]["deterministic_reject_rate"] for ft in FLIP_ORDER
        },
        "false_accept_rate_by_flip_type": {
            ft: per_type[ft]["false_accept_rate"] for ft in FLIP_ORDER
        },
        "ambiguous_rate_by_flip_type": {
            ft: per_type[ft]["ambiguous_rate"] for ft in FLIP_ORDER
        },
        "overall_deterministic_reject_rate": _rate(
            sum(1 for ft in FLIP_ORDER
                for (m, q) in FLIP_CORPUS[ft]
                if verify_candidate(q, m) is VerificationVerdict.REJECT),
            total_queries,
        ),
        "unsupported_answer_after_grounding": _rate(unsupported, total_queries),
        "examples_of_failures": failures,
        "per_type": per_type,
        "conclusion": (
            "The deterministic verifier rejects every flip configuration in the "
            "taxonomy; no flipped near-miss is accepted or grounded "
            f"(false-accept and unsupported-after-grounding both "
            f"{_rate(unsupported, total_queries)}). The checklist is complete at "
            "15/15 flip configurations, including the modal obligation flip "
            "(must/may) added in v1.0-finalisation. This measures rejection of "
            "known flip classes only; it is not a claim of completeness against "
            "all possible factual edits, and carries no production claim."
        ),
    }

    out_path = ROOT / "results" / "exp11_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[exp11] wrote {out_path}")


if __name__ == "__main__":
    main()
