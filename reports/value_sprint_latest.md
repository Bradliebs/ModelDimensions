# v2.3 Value Sprint report

- pack            : m365_coding_assistant
- retrieval       : deterministic

## Summary

- queries            : 25
- expectation met    : 23/25
- grounded + cited   : 22 (evidence retrieved; NOT a relevance guarantee)
- honest refusals    : 2
- conflicts surfaced : 0
- model-prior used   : 1
- stale-flagged      : 17
- pack gaps          : 1 (proposals only — never written)
- guard rejects      : 0

## Honest finding

`grounded + cited` means evidence was *retrieved*, not that the answer is correct. The trust behaviours that genuinely work today are refusal-on-no-evidence, stale-source flagging, near-miss/superseded conflict surfacing, labelled model-prior fallback, and the independent AnswerGuard verdict. Usefulness is the operator's call — the operator columns below are captured, never inferred.

Caveat: with the over-permissive *deterministic* knowledge backend, a declarative near-miss query routes to both memory and knowledge, and knowledge grounds it — masking conflict surfacing in integration. Conflict detection still works on memory-only routes (decision-cue questions) and is exercised directly by the harness tests.

## Per-query audit

| # | category | expected | mode | cites | stale | conflict | prior | guard | met | gap |
|---|----------|----------|------|-------|-------|----------|-------|-------|-----|-----|
| 1 | - | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 2 | - | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 3 | powerapps_formula | stale_flagged | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 4 | - | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 5 | - | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 6 | - | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 7 | - | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 8 | - | grounded_useful | grounded | 5 | no | no | no | ACCEPT | yes | no |
| 9 | no_evidence | honest_refusal | refusal | 0 | no | no | no | ACCEPT | yes | no |
| 10 | out_of_domain | honest_refusal | grounded | 5 | yes | no | no | ACCEPT | no | no |
| 11 | m365_consulting | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 12 | m365_consulting | grounded_useful | grounded | 5 | no | no | no | ACCEPT | yes | no |
| 13 | purview | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 14 | purview | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 15 | powerapps_formula | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 16 | powerapps_formula | stale_flagged | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 17 | copilot_studio | grounded_useful | grounded | 5 | yes | no | no | ACCEPT | yes | no |
| 18 | copilot_studio | grounded_useful | grounded | 5 | no | no | no | ACCEPT | yes | no |
| 19 | coding_agent | grounded_useful | grounded | 5 | no | no | no | ACCEPT | yes | no |
| 20 | repo_milestone | grounded_useful | grounded | 5 | no | no | no | ACCEPT | yes | no |
| 21 | project_decision | grounded_useful | grounded | 6 | yes | no | no | ACCEPT | yes | no |
| 22 | repo_milestone | grounded_useful | grounded | 6 | yes | no | no | ACCEPT | yes | no |
| 23 | conflict | conflict | grounded | 5 | yes | no | no | ACCEPT | no | no |
| 24 | model_prior | model_prior | model_prior_labelled | 0 | no | no | yes | ACCEPT | yes | no |
| 25 | project_decision | pack_gap | refusal | 0 | no | no | no | ACCEPT | yes | yes |

## Operator scoring

Machine columns are captured automatically; fill these in by hand (edit the JSONL, then `value-sprint report`).

| # | category | useful | saved_time | reusable | trust | notes |
|---|----------|--------|------------|----------|-------|-------|
| 1 | - | - | - | - | - | - |
| 2 | - | - | - | - | - | - |
| 3 | powerapps_formula | - | - | - | - | - |
| 4 | - | - | - | - | - | - |
| 5 | - | - | - | - | - | - |
| 6 | - | - | - | - | - | - |
| 7 | - | - | - | - | - | - |
| 8 | - | - | - | - | - | - |
| 9 | no_evidence | - | - | - | - | - |
| 10 | out_of_domain | - | - | - | - | - |
| 11 | m365_consulting | - | - | - | - | - |
| 12 | m365_consulting | - | - | - | - | - |
| 13 | purview | - | - | - | - | - |
| 14 | purview | - | - | - | - | - |
| 15 | powerapps_formula | - | - | - | - | - |
| 16 | powerapps_formula | - | - | - | - | - |
| 17 | copilot_studio | - | - | - | - | - |
| 18 | copilot_studio | - | - | - | - | - |
| 19 | coding_agent | - | - | - | - | - |
| 20 | repo_milestone | - | - | - | - | - |
| 21 | project_decision | - | - | - | - | - |
| 22 | repo_milestone | - | - | - | - | - |
| 23 | conflict | - | - | - | - | - |
| 24 | model_prior | - | - | - | - | - |
| 25 | project_decision | - | - | - | - | - |

## Pack-gap proposals (advisory — not written)

- **Pack gap detection (expected decision, no evidence -> proposal)** (memory_proposal): Capture the decision behind 'Pack gap detection (expected decision, no evidence -> proposal)' as an approved project memory so future recall can ground it.
