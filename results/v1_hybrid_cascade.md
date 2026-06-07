# exp28 — hybrid cascade gating proof

- bank: `H:\MiniLM\cc_service\bank.db`
- lexical index: `H:\ModelDimensions_SLM\ModelDimensions\results\v1_bank\bm25_tantivy`
- reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- rerank margin: 0.5

## Verdict counts per mode

| Mode | grounded | silence | wrong |
|------|---------:|--------:|------:|
| cosine_only | 6 | 2 | 0 |
| cosine_rerank | 4 | 4 | 0 |
| hybrid_cosine_always | 5 | 3 | 0 |
| hybrid_rerank_always | 5 | 3 | 0 |
| hybrid_rerank_fallback | 7 | 1 | 0 |

## Per-query verdicts

| # | Query | cosine_only | cosine_rerank | hyb_cos_alw | hyb_rer_alw | hyb_rer_fb |
|---|-------|:---:|:---:|:---:|:---:|:---:|
| 1 | What language is spoken on the island where Napoleon was exi | grounded | grounded | silence | silence | grounded |
| 2 | Who composed the music for the film that won the Academy Awa | silence | silence | silence | silence | silence |
| 3 | In which country was the inventor of dynamite born? | grounded | silence | grounded | silence | grounded |
| 4 | What is the capital of the country where the 2010 Winter Oly | grounded | silence | grounded | grounded | grounded |
| 5 | Who succeeded the British monarch who reigned throughout the | silence | grounded | silence | grounded | grounded |
| 6 | What is the highest mountain in the country whose flag featu | grounded | grounded | grounded | grounded | grounded |
| 7 | What religion was the founder of psychoanalysis raised in? | grounded | silence | grounded | grounded | grounded |
| 8 | In what city was the author of 'The Old Man and the Sea' bor | grounded | grounded | grounded | grounded | grounded |

**PASS**: True