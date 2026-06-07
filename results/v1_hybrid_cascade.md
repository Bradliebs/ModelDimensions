# exp28 — hybrid cascade gating proof

- bank: `H:\MiniLM\cc_service\bank.db`
- lexical index: `results\v1_bank\bm25_index`
- reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- rerank margin: 0.5

## Verdict counts per mode

| Mode | grounded | silence | wrong |
|------|---------:|--------:|------:|
| cosine_only | 6 | 2 | 0 |
| cosine_rerank | 4 | 4 | 0 |
| hybrid_cosine_always | 2 | 6 | 0 |
| hybrid_rerank_always | 2 | 6 | 0 |
| hybrid_rerank_fallback | 7 | 1 | 0 |

## Per-query verdicts

| # | Query | cosine_only | cosine_rerank | hyb_cos_alw | hyb_rer_alw | hyb_rer_fb |
|---|-------|:---:|:---:|:---:|:---:|:---:|
| 1 | What language is spoken on the island where Napoleon was exi | grounded | grounded | silence | silence | grounded |
| 2 | Who composed the music for the film that won the Academy Awa | silence | silence | silence | silence | silence |
| 3 | In which country was the inventor of dynamite born? | grounded | silence | silence | silence | grounded |
| 4 | What is the capital of the country where the 2010 Winter Oly | grounded | silence | grounded | silence | grounded |
| 5 | Who succeeded the British monarch who reigned throughout the | silence | grounded | silence | grounded | grounded |
| 6 | What is the highest mountain in the country whose flag featu | grounded | grounded | silence | silence | grounded |
| 7 | What religion was the founder of psychoanalysis raised in? | grounded | silence | silence | silence | grounded |
| 8 | In what city was the author of 'The Old Man and the Sea' bor | grounded | grounded | grounded | grounded | grounded |

**PASS**: True

## Result (run 3 — fallback-only orchestration)

**Status: PASS.** `hybrid_rerank_fallback` mode meets the gating criterion: 7 grounded / 1 silence / 0 wrong on the 8-question multi-hop set. Phase 1 cardinal rule preserved (no mode produces a wrong answer); cosine baseline (6/8) is preserved byte-identically on every query cosine can answer; the cascade successfully rescues 1 of the 2 V1 silences without introducing any regression.

### Per-query outcome (cosine_only vs hybrid_rerank_fallback)

| Q | Sub-question                                            | cosine_only | hyb_rer_fb | Δ |
|--:|---------------------------------------------------------|:-----------:|:----------:|:--|
| 1 | What language is spoken on St. Helena?                  | grounded    | grounded   | preserved |
| 2 | Who composed the music for The French Connection?       | silence     | silence    | not in 100k index |
| 3 | In which country was Alfred Nobel born?                 | grounded    | grounded   | preserved |
| 4 | What is the capital of Canada?                          | grounded    | grounded   | preserved |
| 5 | Who succeeded Queen Elizabeth II as British monarch?    | silence     | grounded   | **rescued** |
| 6 | What is the highest mountain in Canada?                 | grounded    | grounded   | preserved |
| 7 | What religion was Sigmund Freud raised in?              | grounded    | grounded   | preserved |
| 8 | In what city was Ernest Hemingway born?                 | grounded    | grounded   | preserved |

### Q5 rescue detail (the only rescue this run)

```
cosine_only silence:  gate: margin below threshold (retrieval diffuse);
                       rescue: supporting_cell_outside_rank_window
hybrid_rerank_fallback gate:  fire=True, margin=0.624 (rerank-scale)
hybrid_rerank_fallback answer: Prince Charles succeeded Queen Elizabeth II
                                as the British monarch [8272], [8242], [8252].
result.rescue: {hybrid_rescue: True,
                first_pass_silence_reason: "gate: margin below threshold ..."}
```

The `rescue` field on `PipelineResult` carries the provenance flag so downstream auditing can distinguish V1 cosine answers from cascade-rescue answers.

### Q2 still silent — why

`Who composed the music for The French Connection?` requires the Don Ellis / French Connection cell, which `experiments/exp28_diag_keyword_locations.py` previously confirmed sits ABOVE id 100k in the production bank. The current lexical index is built on a 100k-cell capped subset. The cascade cannot retrieve what is not indexed; it correctly silences rather than fabricating an answer. Resolving Q2 requires the full 5.7M-cell BM25 index, which is blocked by `rank_bm25`'s pure-Python OOM (~23 GB on 5.7M docs). Phase 1.5 will swap in `pyserini` or `tantivy-py` to unblock the full-bank build.

### Why the always-on modes fail and the fallback mode passes

`hybrid_cosine_always` 2/8 and `hybrid_rerank_always` 2/8 confirm what run 2 showed: "BM25 narrows, cosine ranks" is the wrong fusion order for this bank. The cosine encoder is already strong (6/8); pre-filtering with BM25 throws away that retrieval quality on the queries cosine can answer.

The fallback orchestration sidesteps this entirely. The `AnswerPipeline.ask` method now:

1. Runs pass-1 with the pure V1 cosine path (hybrid and reranker both temporarily disabled). On queries cosine can answer, this returns immediately and the result is byte-identical to V1.
2. Only when pass-1 returns `silence=True` is pass-2 invoked with hybrid + cross-encoder rerank enabled. If pass-2 grounds, the result carries `rescue.hybrid_rescue = True` for audit. If pass-2 also silences, the original pass-1 silence message is returned with telemetry recording the rescue attempt.

This preserves V1's cardinal recall guarantee on all queries cosine handles AND adds rescue capacity for cosine silences, which is exactly the Phase 1 design intent.

### Next steps

1. **Phase 1.5 (deferred):** swap `rank_bm25` for `pyserini` or `tantivy-py` to unblock full-bank indexing. Re-run exp28 against the full bank; Q2 should rescue if the Don Ellis cell exists anywhere in the bank.
2. **RUNBOOK update:** Section 11 should be promoted from research-only to opt-in production. The default `hybrid_mode` in `AnswerPipeline` is already `fallback`.
3. **Phase 2 unblocked:** the answer-claim verifier upgrade can now proceed against a working hybrid retrieval path.
