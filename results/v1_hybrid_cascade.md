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
| hybrid_cosine | 2 | 6 | 0 |
| hybrid_rerank | 2 | 6 | 0 |

## Per-query verdicts

| # | Query | cosine_only | cosine_rerank | hybrid_cosine | hybrid_rerank |
|---|-------|:---:|:---:|:---:|:---:|
| 1 | What language is spoken on the island where Napoleon was exi | grounded | grounded | silence | silence |
| 2 | Who composed the music for the film that won the Academy Awa | silence | silence | silence | silence |
| 3 | In which country was the inventor of dynamite born? | grounded | silence | silence | silence |
| 4 | What is the capital of the country where the 2010 Winter Oly | grounded | silence | grounded | silence |
| 5 | Who succeeded the British monarch who reigned throughout the | silence | grounded | silence | grounded |
| 6 | What is the highest mountain in the country whose flag featu | grounded | grounded | silence | silence |
| 7 | What religion was the founder of psychoanalysis raised in? | grounded | silence | silence | silence |
| 8 | In what city was the author of 'The Old Man and the Sea' bor | grounded | grounded | grounded | grounded |

**PASS**: False
## Failure analysis (run 2 — decomposer wired in)

**Status: FAILED. Phase 1 cascade must not be enabled by default.**

Different failure mode than run 1. Run 1 with no decomposer produced 1-2 wrong answers per mode; this run with the decomposer produces 0 wrong answers across all modes (cardinal rule PASSES). The cascade's failure mode is now **silencing cells that pure cosine grounds**, not generating false answers.

### What changed between runs

The `exp28` harness now mirrors the `exp22` / `exp27` flow: one cosine pass-1 retrieval per query feeds a Phi-3 decomposition prompt, all four modes then `pipeline.ask` on the SAME sub-question. This holds the decomposition step constant and isolates the retrieval-mode variable. `cosine_only` now scores 6 grounded / 2 silence / 0 wrong, which exactly matches V1's documented baseline (exp22 / exp24). The harness is fixed.

### What this proves

With decomposition, no mode produces wrong answers. The cascade does not break the V1 wrong-count guarantee.

The cascade does, however, strictly REDUCE recall:

- `cosine_only` 6 grounded
- `cosine_rerank` 4 grounded (rerank drops 2)
- `hybrid_cosine` 2 grounded (hybrid drops 4)
- `hybrid_rerank` 2 grounded (no recovery from rerank when hybrid is on)

Per-query diagnostic (cells silenced under `hybrid_rerank` that `cosine_only` grounded):

- **Q1** "What language is spoken on St. Helena?" — cosine_only top-2 found the Saint Helena page (English). hybrid_rerank top-1 was a Sao Tome page (Portuguese, Forro creole) at gate margin 1.904. Verifier rejected: answer entity ['English', 'Portuguese', 'Greek'] not colocated with question anchor.
- **Q3** "In which country was Alfred Nobel born?" — both modes retrieved the Alfred Nobel page at rank 1. cosine_only verifier accepted "Sweden"; hybrid_rerank verifier rejected (same top-1 text, different supporting cells in citation set apparently). Diagnostic: top-1 text identical between modes — `'Alfred Nobel ... was a Swedish scientist...'`. Suggests the cascade corrupts the verifier's wider citation context, not just top-1.
- **Q6** "What is the highest mountain in Canada?" — cosine_only top-1 was "Mount Logan is the highest mountain in Canada" (perfect). hybrid_rerank top-1 was a Mauna Loa / Hawaii page at margin 0.519. Verifier rejected Mount McKinley/Alaska answer.
- **Q7** "What religion was Sigmund Freud raised in?" — cosine_only top-1 "Sigmund Freud was born to Jewish parents in a heavily Roman Catholic town". hybrid_rerank top-1 "Sigmund Freud ... was an Austrian neurologist" (no religion). Verifier rejected.

### Root cause

BM25 in the cascade is promoting cells that match more **surface tokens** of the sub-question but do not contain the answer entity. The cross-encoder rerank does not correct this; it re-ranks an already-corrupted candidate pool. The silence-gate margin fires very confidently (margins 0.5 to 5.4 — much higher than cosine_only's typical 0.02-0.09) because the BM25-promoted cells are strong lexical matches to the query. The verifier, doing its job correctly, rejects because the answer entity is not colocated with the question anchor in those cells.

This is a fundamental design flaw in the current cascade: **BM25 → cosine → rerank is the wrong fusion order for a memory bank where the encoder is already pretty good.** The cosine baseline is already retrieving the right cells on 6 of 8 queries; using BM25 to narrow the candidate set throws away that retrieval quality before the cosine step ever sees it.

### Recommended redesigns (for next session)

1. **Fallback-only cascade** (lowest risk, surgical). Run cosine first. Only invoke the BM25 + rerank cascade when cosine's top-2 margin is below threshold (i.e. cosine has already silenced). This preserves V1's 6/8 grounded baseline unchanged and only adds rescue capacity on the 2 queries where cosine fails. Target: Q2 (French Connection) and Q5 (Elizabeth II succession), the two queries V1 currently silences.
2. **Reciprocal Rank Fusion** (medium risk). Replace the "BM25 narrows, cosine ranks" pipeline with RRF over BM25 and cosine top-k lists independently, then rerank the union. This avoids the candidate-narrowing problem but adds complexity.
3. **Dense-similarity floor on BM25 promotions** (mitigation only, does not fix root cause). Require BM25-promoted candidates to also clear a minimum cosine activation before they reach the gate. Likely necessary even under option 1 to prevent regression on edge cases.
4. **Wider lexical_k** — currently 200. Probably not the issue; the failures are not at the edge of the candidate pool but in the top-1.

### Recommended next concrete step

Implement option 1 (fallback-only cascade) as a config-flag mode in `HybridRetriever`. Re-run exp28 with that mode. If it preserves 6/8 cosine_only baseline AND lifts Q2 or Q5 to grounded, Phase 1 is salvaged.
