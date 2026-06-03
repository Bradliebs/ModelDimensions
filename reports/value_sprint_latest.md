# v2.3 Value Sprint report

- pack            : m365_coding_assistant
- retrieval       : deterministic

## Summary

- queries            : 25
- expectation met    : 18/25
- grounded + cited   : 13 (evidence retrieved; NOT a relevance guarantee)
- honest refusals    : 10
- conflicts surfaced : 1
- model-prior used   : 1
- stale-flagged      : 17
- pack gaps          : 8 (proposals only — never written)
- guard rejects      : 0

## Honest finding

`grounded + cited` means evidence was *retrieved*, not that the answer is correct. The trust behaviours that genuinely work today are refusal-on-no-evidence, stale-source flagging, near-miss/superseded conflict surfacing, labelled model-prior fallback, and the independent AnswerGuard verdict. Usefulness is the operator's call — the operator columns below are captured, never inferred.

Caveat (resolved in v2.4): with the over-permissive *deterministic* knowledge backend, a declarative near-miss query routes to both memory and knowledge, and knowledge used to ground it — masking conflict surfacing in integration. The v2.4 relevance & sufficiency gate now surfaces a rejected near-miss as a conflict even when a knowledge chunk was retrieved on the same query, and downgrades weak or out-of-domain evidence to a refusal instead of a confident grounding. The `relevance` column below shows the gate verdict per query.

## Per-query audit

| # | category | expected | mode | relevance | cites | stale | conflict | prior | guard | met | gap |
|---|----------|----------|------|-----------|-------|-------|----------|-------|-------|-----|-----|
| 1 | - | grounded_useful | grounded | relevant | 5 | yes | no | no | ACCEPT | yes | no |
| 2 | - | grounded_useful | grounded | relevant | 5 | yes | no | no | ACCEPT | yes | no |
| 3 | powerapps_formula | stale_flagged | grounded | relevant | 5 | yes | no | no | ACCEPT | yes | no |
| 4 | - | grounded_useful | grounded | relevant | 5 | yes | no | no | ACCEPT | yes | no |
| 5 | - | grounded_useful | grounded | relevant | 5 | yes | no | no | ACCEPT | yes | no |
| 6 | - | grounded_useful | refusal | no_support | 0 | yes | no | no | ACCEPT | no | yes |
| 7 | - | grounded_useful | grounded | relevant | 5 | yes | no | no | ACCEPT | yes | no |
| 8 | - | grounded_useful | grounded | partial | 5 | no | no | no | ACCEPT | yes | no |
| 9 | no_evidence | honest_refusal | refusal | - | 0 | no | no | no | ACCEPT | yes | no |
| 10 | out_of_domain | honest_refusal | refusal | no_support | 0 | yes | no | no | ACCEPT | yes | no |
| 11 | m365_consulting | grounded_useful | refusal | weak_match | 0 | yes | no | no | ACCEPT | no | yes |
| 12 | m365_consulting | grounded_useful | grounded | relevant | 5 | no | no | no | ACCEPT | yes | no |
| 13 | purview | grounded_useful | refusal | weak_match | 0 | yes | no | no | ACCEPT | no | yes |
| 14 | purview | grounded_useful | refusal | weak_match | 0 | yes | no | no | ACCEPT | no | yes |
| 15 | powerapps_formula | grounded_useful | grounded | partial | 5 | yes | no | no | ACCEPT | yes | no |
| 16 | powerapps_formula | stale_flagged | refusal | no_support | 0 | yes | no | no | ACCEPT | no | yes |
| 17 | copilot_studio | grounded_useful | grounded | partial | 5 | yes | no | no | ACCEPT | yes | no |
| 18 | copilot_studio | grounded_useful | grounded | relevant | 5 | no | no | no | ACCEPT | yes | no |
| 19 | coding_agent | grounded_useful | refusal | weak_match | 0 | no | no | no | ACCEPT | no | yes |
| 20 | repo_milestone | grounded_useful | refusal | no_support | 0 | no | no | no | ACCEPT | no | yes |
| 21 | project_decision | grounded_useful | grounded | relevant | 6 | yes | no | no | ACCEPT | yes | no |
| 22 | repo_milestone | grounded_useful | grounded | relevant | 6 | yes | no | no | ACCEPT | yes | no |
| 23 | conflict | conflict | conflict_explanation | conflict | 0 | yes | yes | no | ACCEPT | yes | no |
| 24 | model_prior | model_prior | model_prior_labelled | - | 0 | no | no | yes | ACCEPT | yes | no |
| 25 | project_decision | pack_gap | refusal | - | 0 | no | no | no | ACCEPT | yes | yes |

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

## Top rejected evidence (why the gate held it back)

The strongest candidate the ranker did *not* let lead, per query — the audit line for why a verdict was reached.

- **#1** `src:chk-6edcec1b` — stale source — penalised but still citable
- **#2** `src:chk-57856c85` — stale source — penalised but still citable
- **#3** `src:chk-8dbc8eb7` — metadata/source header — cannot be substantive lead
- **#4** `src:chk-d156adaf` — metadata/source header — cannot be substantive lead
- **#5** `src:chk-6edcec1b` — stale source — penalised but still citable
- **#6** `src:chk-f64dc0a9` — stale source — penalised but still citable
- **#7** `src:chk-d663fa0f` — stale source — penalised but still citable
- **#8** `src:chk-a60a378e` — outranked by lead evidence
- **#10** `src:chk-6edcec1b` — out-of-domain for query topic 'lambda'
- **#11** `src:chk-5ce10d00` — stale source — penalised but still citable
- **#12** `src:chk-79df23b9` — outranked by lead evidence
- **#13** `src:chk-6edcec1b` — stale source — penalised but still citable
- **#14** `src:chk-435cf574` — stale source — penalised but still citable
- **#15** `src:chk-6edcec1b` — stale source — penalised but still citable
- **#16** `src:chk-d156adaf` — metadata/source header — cannot be substantive lead
- **#17** `src:chk-6edcec1b` — stale source — penalised but still citable
- **#18** `src:chk-a60a378e` — outranked by lead evidence
- **#19** `src:chk-d663fa0f` — outranked by lead evidence
- **#20** `src:chk-d663fa0f` — outranked by lead evidence
- **#21** `src:chk-0a44d972` — stale source — penalised but still citable
- **#22** `src:chk-9bde9cf0` — stale source — penalised but still citable
- **#23** `src:chk-24d3fb6a` — stale source — penalised but still citable

## Pack-gap proposals (advisory — not written)

- **Natural-language paraphrase of Purview content in the pack (gap probe)** (knowledge_source): Import an authoritative source covering 'Natural-language paraphrase of Purview content in the pack (gap probe)' so the pack can ground this query instead of refusing it.
- **M365 consulting explanation (knowledge question)** (knowledge_source): Import an authoritative source covering 'M365 consulting explanation (knowledge question)' so the pack can ground this query instead of refusing it.
- **Purview / sensitivity label design (knowledge question)** (knowledge_source): Import an authoritative source covering 'Purview / sensitivity label design (knowledge question)' so the pack can ground this query instead of refusing it.
- **Purview / sensitivity label design (variant)** (knowledge_source): Import an authoritative source covering 'Purview / sensitivity label design (variant)' so the pack can ground this query instead of refusing it.
- **PowerApps delegation (stale source expected)** (knowledge_source): Import an authoritative source covering 'PowerApps delegation (stale source expected)' so the pack can ground this query instead of refusing it.
- **Coding-agent next prompt (knowledge question)** (knowledge_source): Import an authoritative source covering 'Coding-agent next prompt (knowledge question)' so the pack can ground this query instead of refusing it.
- **Repo milestone recall (knowledge question)** (memory_proposal): Capture the decision behind 'Repo milestone recall (knowledge question)' as an approved project memory so future recall can ground it.
- **Pack gap detection (expected decision, no evidence -> proposal)** (memory_proposal): Capture the decision behind 'Pack gap detection (expected decision, no evidence -> proposal)' as an approved project memory so future recall can ground it.
