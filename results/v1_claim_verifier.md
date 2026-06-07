# exp29 - Stage F (claim verifier) regression + behaviour proof

- bank: `H:\MiniLM\cc_service\bank.db`
- lexical index: `H:\ModelDimensions_SLM\ModelDimensions\results\v1_bank\bm25_tantivy`
- retrieval mode: `hybrid_rerank_fallback` (production)

## Verdict counts

| Stage F | grounded | silence | wrong |
|---------|---------:|--------:|------:|
| off     | 7 | 1 | 0 |
| on      | 7 | 1 | 0 |

## Per-query verdicts (off -> on)

| # | Query | off | on | delta |
|---|-------|:---:|:---:|:---:|
| 1 | What language is spoken on the island where Napoleon wa | grounded | grounded | = |
| 2 | Who composed the music for the film that won the Academ | silence | silence | = |
| 3 | In which country was the inventor of dynamite born? | grounded | grounded | = |
| 4 | What is the capital of the country where the 2010 Winte | grounded | grounded | = |
| 5 | Who succeeded the British monarch who reigned throughou | grounded | grounded | = |
| 6 | What is the highest mountain in the country whose flag  | grounded | grounded | = |
| 7 | What religion was the founder of psychoanalysis raised  | grounded | grounded | = |
| 8 | In what city was the author of 'The Old Man and the Sea | grounded | grounded | = |

**Regressions (grounded->silence)**: 0
**False-positive catches (wrong->silence)**: 0

**PASS**: True