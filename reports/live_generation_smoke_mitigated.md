# Live generation smoke -- RetroGPT 55M + cc_service bank

- Checkpoint: `H:\MiniLM\nanogpt\out-retro-bank\ckpt_best.pt` (55.29M params, n_layer=8, n_embd=512)
- Service: http://127.0.0.1:8765
- Bank: 50,000 cells, dim=384, encoder=all-MiniLM-L6-v2
- Device: cuda / torch.bfloat16
- Max new tokens: 80, temperature: 0.8, top_k: 40, seed: 1337
- skip_low_info_fraction: 0.5 (mitigation ON)
- Total elapsed: 1.9s for 5 prompts

## What this is (and is not)

This is a functional smoke test that the wiring works end-to-end:
RetroGPT 55M -> chunked queries to cc_service -> tokenized neighbors fed
back into CCA layers at generation time. Per prompt we show both the
WITHOUT-retrieval and WITH-retrieval continuations using the same seed.

It is **not** a quality benchmark. The 55M model is small and was trained
against a different bank than the inference-time cc_service bank. The
encoder identity matches (all-MiniLM-L6-v2, dim 384), so retrieval is
semantically meaningful, but source-text distribution differs from training.

## Prompt: The theory of general relativity describes

**WITHOUT retrieval** (0.3s):

>  the end of the future.<|endoftext|>

**WITH retrieval** (0.1s, coarse_overlap=0 words):

> !!!!<|endoftext|>

First-chunk retrieval:

- chunk 0: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 1: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 2: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 3: SKIPPED (low_info_fraction=0.91 ≥ threshold)

## Prompt: Machine learning is a branch of artificial intelligence that

**WITHOUT retrieval** (0.1s):

>  is in the Dark Age.<|endoftext|>

**WITH retrieval** (0.1s, coarse_overlap=0 words):

>  is a part of the people they have?!<|endoftext|>

First-chunk retrieval:

- chunk 0: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 1: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 2: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 3: SKIPPED (low_info_fraction=0.86 ≥ threshold)

## Prompt: The French Revolution began in 1789 when

**WITHOUT retrieval** (0.2s):

>  the end of the 20th century created the new new new 20th century that is a unique way.<|endoftext|>

**WITH retrieval** (0.8s, coarse_overlap=0 words):

>  you are you to get to get your Het get so much to make it. He can make a lot of things, or work he got the lot to try. His good will be for people who have the lot.<|endoftext|>

First-chunk retrieval:

- chunk 0: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 1: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 2: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 3: SKIPPED (low_info_fraction=0.88 ≥ threshold)

## Prompt: In computer science, a hash table is

**WITHOUT retrieval** (0.2s):

>  from any room to make a book to the universe for the first time in the universe.<|endoftext|>

**WITH retrieval** (0.1s, coarse_overlap=0 words):

> !!!!!!<|endoftext|>

First-chunk retrieval:

- chunk 0: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 1: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 2: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 3: SKIPPED (low_info_fraction=0.88 ≥ threshold)

## Prompt: The Great Wall of China was built

**WITHOUT retrieval** (0.0s):

> .<|endoftext|>

**WITH retrieval** (0.1s, coarse_overlap=0 words):

> .<|endoftext|>

First-chunk retrieval:

- chunk 0: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 1: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 2: SKIPPED (low_info_fraction=1.00 ≥ threshold)
- chunk 3: SKIPPED (low_info_fraction=0.89 ≥ threshold)

## Caveats

- 55M debug ckpt, not the 404M roadmap target.
- Inference bank is the 5.7M cc_service production bank, not the cell_tokens.npy the model trained against. Encoder identity matches (all-MiniLM-L6-v2, dim 384) so retrieval is semantically meaningful, but source-text distribution differs from training.
- first_chunk_retrieval shows only the chunks at the FIRST chunk-boundary crossing (one snapshot per prompt) to keep the report compact.
- coarse_overlap_word_count is a crude grounding proxy, not a verifier.
