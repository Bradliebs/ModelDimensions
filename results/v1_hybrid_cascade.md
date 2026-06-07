# exp28 — hybrid cascade gating proof

- bank: `H:\MiniLM\cc_service\bank.db`
- lexical index: `H:\ModelDimensions_SLM\ModelDimensions\results\v1_bank\bm25_index`
- reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- rerank margin: 0.5

## Verdict counts per mode

| Mode | grounded | silence | wrong |
|------|---------:|--------:|------:|
| cosine_only | 0 | 7 | 1 |
| cosine_rerank | 1 | 6 | 1 |
| hybrid_cosine | 0 | 6 | 2 |
| hybrid_rerank | 0 | 7 | 1 |

## Per-query verdicts

| # | Query | cosine_only | cosine_rerank | hybrid_cosine | hybrid_rerank |
|---|-------|:---:|:---:|:---:|:---:|
| 1 | What language is spoken on the island where Napoleon was exi | silence | silence | silence | silence |
| 2 | Who composed the music for the film that won the Academy Awa | silence | silence | wrong | silence |
| 3 | In which country was the inventor of dynamite born? | silence | silence | silence | silence |
| 4 | What is the capital of the country where the 2010 Winter Oly | silence | wrong | wrong | wrong |
| 5 | Who succeeded the British monarch who reigned throughout the | silence | silence | silence | silence |
| 6 | What is the highest mountain in the country whose flag featu | silence | silence | silence | silence |
| 7 | What religion was the founder of psychoanalysis raised in? | wrong | grounded | silence | silence |
| 8 | In what city was the author of 'The Old Man and the Sea' bor | silence | silence | silence | silence |

**PASS**: False
## Failure analysis

**Status: FAILED. Phase 1 cascade must not be enabled by default.**

The hybrid_rerank mode produced 1 wrong answer (Q4: "Vancouver" instead of "Ottawa") and the hybrid_cosine mode produced 2 wrong answers (Q2 hallucinated composer, Q4 "Vancouver"). The cardinal V1 constraint is that wrong-answer count never increases; this run regresses on that constraint.

### Methodological caveats

The baseline cosine_only column also shows 0 grounded / 7 silence / 1 wrong, while V1's documented baseline on the same 8 queries is 6 grounded / 2 silence / 0 wrong. The difference is the **decomposer**: V1's documented baseline runs each multi-hop query through Phi-3 to generate a single-hop sub-question (exp22 pattern, replicated in exp27 `_capture`), then retrieves on that sub-question. exp28 in its current form calls `pipeline.ask(raw_query)` directly with no decomposition.

Without decomposition no mode has a fair shot at the multi-hop queries — the raw question text spans two facts, the encoder produces a diffuse query vector, and the top-k retrieval lands on cells that mention some surface tokens of the question without containing the answer. This is why cosine_only also collapses to silence/wrong.

### What this proves and does not prove

**Proven:**

- The cascade plumbing works end-to-end (40/40 unit tests green).
- BM25 retrieval can promote topically-related-but-factually-wrong cells with margins above the silence-gate threshold. Q2 hybrid_cosine: top-3 cells all contained "Academy Award" tokens but none was the French Connection paragraph, gate margin 0.033 > 0.015, Phi-3 hallucinated "Arthur" as the composer. Q4 (all 3 fired modes): top retrieval surfaces "2010 Winter Olympics held in Vancouver" with strong matches, Phi-3 answers "Vancouver" as the capital because that's what the cell says.
- The current cascade has no mechanism to detect the keyword-match-but-wrong-paragraph failure mode the cross-encoder reranker is supposed to mitigate. The reranker is operating on the BM25-restricted candidate set, which excludes the right paragraph entirely when the right paragraph contains the answer entity but not the question's surface tokens.

**Not proven (requires re-run with decomposition):**

- Whether Phase 1 cascade lifts the V1 documented baseline from 6/8 to >=7/8 grounded.
- Whether Q2 (the original target — Don Ellis recall fault) is actually recoverable.

### Next steps (recommended)

1. **Do not enable the cascade in production** (no env-var default, no documentation promoting it for daily use). Keep it as an opt-in research path until re-validated.
2. **Rewrite exp28 to use the decomposer** the same way exp22 / exp27 do (Phi-3 generates a single-hop sub-question first, then `pipeline.ask(sub_question)`). Re-run all 4 modes.
3. **Investigate the Q4 "Vancouver" failure under decomposition** — even with decomposition, the sub-question "What is the capital of Canada?" should land on the Ottawa cell. If the cell exists in the first 100k bank ids and the dense encoder can find it, then the wrong answer in the current exp28 is purely a decomposition-missing artifact. If it persists with decomposition, the cascade has a real defect.
4. **Consider a 2-hop pre-gate**: only allow BM25-restricted top-1 to surface to the gate if its BM25-restricted cosine activation also clears a minimum dense-similarity floor. Cells that BM25 promotes purely on surface tokens without dense semantic alignment to the query should be filtered.

## Failure analysis

**Status: FAILED. Phase 1 cascade must not be enabled by default.**

The hybrid_rerank mode produced 1 wrong answer (Q4: "Vancouver" instead of "Ottawa") and the hybrid_cosine mode produced 2 wrong answers (Q2 hallucinated composer, Q4 "Vancouver"). The cardinal V1 constraint is that wrong-answer count never increases; this run regresses on that constraint.

### Methodological caveats

The baseline cosine_only column also shows 0 grounded / 7 silence / 1 wrong, while V1's documented baseline on the same 8 queries is 6 grounded / 2 silence / 0 wrong. The difference is the **decomposer**: V1's documented baseline runs each multi-hop query through Phi-3 to generate a single-hop sub-question (exp22 pattern, replicated in exp27 `_capture`), then retrieves on that sub-question. exp28 in its current form calls `pipeline.ask(raw_query)` directly with no decomposition.

Without decomposition no mode has a fair shot at the multi-hop queries — the raw question text spans two facts, the encoder produces a diffuse query vector, and the top-k retrieval lands on cells that mention some surface tokens of the question without containing the answer. This is why cosine_only also collapses to silence/wrong.

### What this proves and does not prove

**Proven:**

- The cascade plumbing works end-to-end (40/40 unit tests green).
- BM25 retrieval can promote topically-related-but-factually-wrong cells with margins above the silence-gate threshold. Q2 hybrid_cosine: top-3 cells all contained "Academy Award" tokens but none was the French Connection paragraph, gate margin 0.033 > 0.015, Phi-3 hallucinated "Arthur" as the composer. Q4 (all 3 fired modes): top retrieval surfaces "2010 Winter Olympics held in Vancouver" with strong matches, Phi-3 answers "Vancouver" as the capital because that's what the cell says.
- The current cascade has no mechanism to detect the keyword-match-but-wrong-paragraph failure mode the cross-encoder reranker is supposed to mitigate. The reranker is operating on the BM25-restricted candidate set, which excludes the right paragraph entirely when the right paragraph contains the answer entity but not the question's surface tokens.

**Not proven (requires re-run with decomposition):**

- Whether Phase 1 cascade lifts the V1 documented baseline from 6/8 to >=7/8 grounded.
- Whether Q2 (the original target — Don Ellis recall fault) is actually recoverable.

### Next steps (recommended)

1. **Do not enable the cascade in production** (no env-var default, no documentation promoting it for daily use). Keep it as an opt-in research path until re-validated.
2. **Rewrite exp28 to use the decomposer** the same way exp22 / exp27 do (Phi-3 generates a single-hop sub-question first, then `pipeline.ask(sub_question)`). Re-run all 4 modes.
3. **Investigate the Q4 "Vancouver" failure under decomposition** — even with decomposition, the sub-question "What is the capital of Canada?" should land on the Ottawa cell. If the cell exists in the first 100k bank ids and the dense encoder can find it, then the wrong answer in the current exp28 is purely a decomposition-missing artifact. If it persists with decomposition, the cascade has a real defect.
4. **Consider a 2-hop pre-gate**: only allow BM25-restricted top-1 to surface to the gate if its BM25-restricted cosine activation also clears a minimum dense-similarity floor. Cells that BM25 promotes purely on surface tokens without dense semantic alignment to the query should be filtered.
