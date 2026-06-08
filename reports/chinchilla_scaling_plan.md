# Chinchilla scaling plan — Knowledge-Free Reader

## Parameter and token targets

- Model parameters: **404,000,000**
- Chinchilla-optimal tokens (20x params): **8,080,000,000**
- Unique-token target for this plan: **8,000,000,000**
- Iterations: **400,000** at effective batch **32**, block size **256**
- Token-presentations across the run: **3,276,800,000**
- Presentations per unique token: **0.41**
- Coverage ratio (presented / Chinchilla): **0.41**

## Candidate disjoint corpora

| Source | Est. tokens | Licence | Disjointness risk |
| --- | ---: | --- | --- |
| OpenAssistant Conversations v2 (en) | 80,000,000 | Apache-2.0 | low |
| The Stack v2 dedup (permissive subset, 1% sample) | 3,000,000,000 | permissive (per-repo) | low |
| RedPajama v2 (sample, English, head bucket) | 4,500,000,000 | ODC-By 1.0 | medium |
| C4 cleaned-en (slice) | 500,000,000 | ODC-By 1.0 | medium |
| **Total** | **8,080,000,000** | | |

## Compute envelope

| Device | Throughput tok/s (low–high) | Hours (low–high) | USD (low–high) |
| --- | --- | --- | --- |
| a100_80gb_bf16 | 22,000 – 45,000 | 20.2 – 41.4 | $51 – $103 |
| h100_80gb_bf16 | 50,000 – 95,000 | 9.6 – 18.2 | $48 – $91 |

## Notes and warnings

- unique_tokens_target 8,000,000,000 is below the Chinchilla optimum of 8,080,000,000 tokens for 404,000,000 params; expect a small loss penalty.
- presented_tokens 3,276,800,000 is below the unique-token target 8,000,000,000; the model will see less than one epoch. Increase iters or effective_batch to cover the corpus.
