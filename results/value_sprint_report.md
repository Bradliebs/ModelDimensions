# v2.1.1 Value Sprint report

Pack: m365-coding-assistant-v21

## Summary

- queries           : 10
- grounded + cited  : 9 (retrieved evidence; NOT a relevance guarantee — see finding)
- refused           : 1
- honest refusals   : 1 (refusal was the expected, correct outcome)
- false groundings  : 1 (should have had no evidence, but grounded anyway)
- pack gaps         : 0 (expected value, pack could not deliver)
- stale-flagged     : 8
- model-prior used  : 1

## Honest finding

The deterministic retrieval backend is **over-permissive**: it returns a broad fixed set of citations for almost any query — verbatim, paraphrased, or out-of-domain. So `grounded + cited` here means *evidence was retrieved*, **not** *the answer is relevant*. The trust behaviours that genuinely work today are refusal-on-no-evidence (empty memory) and stale-source flagging, plus the grounding guard that stops forbidden/fabricated content leaking even when an irrelevant source is cited. Relevance-ranked retrieval — actually answering the question asked — would need a semantic backend.

## Per-query audit

| # | note | route | grounded | cites | refused | stale | false-ground | gap |
|---|------|-------|----------|-------|---------|-------|--------------|-----|
| 1 | M365 least-privilege admin (verbatim section) | both | yes | 5 | no | yes | no | no |
| 2 | Purview sensitivity labels (verbatim section) | both | yes | 5 | no | yes | no | no |
| 3 | PowerApps Power Fx (verbatim; STALE source -> stale caution expected) | both | yes | 5 | no | yes | no | no |
| 4 | Coding-agent minimum-change rule (verbatim section) | both | yes | 5 | no | yes | no | no |
| 5 | SharePoint external sharing (verbatim section) | both | yes | 5 | no | yes | no | no |
| 6 | Natural-language paraphrase of Purview content in the pack (gap probe) | both | yes | 5 | no | yes | no | no |
| 7 | Natural-language paraphrase of PowerApps content in the pack (gap probe) | both | yes | 5 | no | yes | no | no |
| 8 | Natural-language paraphrase of coding-agent content in the pack (gap probe) | both | yes | 5 | no | no | no | no |
| 9 | Decision recall against an empty ledger (refusal is correct) | general_model_not_grounded | no | 0 | yes | no | no | no |
| 10 | Out-of-domain (refusal is correct; no AWS content in pack) | both | yes | 5 | no | yes | yes | no |

## False groundings — over-permissive retrieval

- Out-of-domain (refusal is correct; no AWS content in pack)
