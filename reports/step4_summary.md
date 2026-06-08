# Step 4 -- Wire cc_service into the live RetroGPT loop

**Status**: shipped, with honest caveats.
**Date**: 2026-11-13
**Artifacts**:
- [scripts/ask_retro.py](../scripts/ask_retro.py) -- repo-root entrypoint
- [scripts/make_subset_bank.py](../scripts/make_subset_bank.py) -- carve subset bank for in-memory query
- [scripts/inspect_bank.py](../scripts/inspect_bank.py) -- read-only schema/row inspector
- [reports/live_generation_smoke.md](live_generation_smoke.md) -- 5-prompt raw transcript
- [reports/live_generation_smoke.json](live_generation_smoke.json) -- machine-readable

## What was attempted

End-to-end wiring of the existing `cc_service` retrieval API into RetroGPT's
chunked-CCA generation loop, against the production bank. The script
`scripts/ask_retro.py` runs each prompt twice with the same seed (with and
without retrieval), captures the retrieved sources for the first chunk
boundary, and records the result.

## Findings

### 1. Wiring works end-to-end against a 50K-cell bank

After warmup (21.8s for cache load) all 5 prompts × 80 max-tokens completed
in 2.0s wall. cc_service logs show 22 successful `POST /query` 200s per run.
The CCA path accepts the live-retrieved neighbor token tensor without
shape/dtype trouble (shape `(1, K=4, n_neighbors=2, neighbor_len=64)` int64,
EOT-padded). RetroGPT 55M loads in ~3s, generates in single-digit ms per
token.

### 2. The production bank does not fit the cc_service query path on a 48 GB host

The first attempt against the full 5.7M-cell bank (`H:\MiniLM\cc_service\bank.db`,
13.7 GB) put the cc_service worker at **27.7 GB resident**, dropped system
free RAM from 11.7 GB to **0 GB**, and started paging. The bottleneck is
`BankStore.all_weights_and_thresholds()`, which:

- `SELECT id, weight, theta FROM cells` over 5.7M rows (8.7 GB of BLOB data)
- `fetchall()` materializes the entire result set as Python tuples
- `np.stack(list_of_5.7M_arrays)` allocates a new contiguous `(5.7M, 384)`
  float32 array (8.7 GB) while still holding the input list

Peak transient allocation is ~17-18 GB on top of ~9 GB of model + Python
baseline, exceeding available RAM. Numeric trace:

    cells = 5,698,239
    weight bytes = 384 * 4 = 1,536
    total weight payload = 5,698,239 * 1,536 / 1024**3 = 8.16 GB
    np.stack peak ~ 2 * payload + Python tuple/object overhead ~ 18 GB
    process resident observed: 27.7 GB
    free RAM observed: 0 GB
    -> swap thrash, query never returns

Workaround used for this smoke: subset bank of first 50,000 single-kind
cells (125 MB on disk, ~76 MB matrix). Loads in <22s, queries return in
single-digit ms.

This is an **architectural finding**, not a bug in the smoke. Serving the
full bank requires either: (a) streaming top-k via SQLite + per-batch
dot products instead of one big matmul, (b) an external ANN index (FAISS),
or (c) a larger host. Out of scope for Step 4.

### 3. Pad-only chunks retrieve the same phonetic-alphabet attractor in 100% of cases

For 80-token generations, only the LAST chunk of the 256-token block contains
prompt content; the first 3 chunks are EOT-padding that decodes to
near-empty strings. All such pad chunks retrieve the same top-2 cells:

- act=0.498: "Hangul: ㄱ ㄲ ㄴ ㄷ ㄸ ... ㅢ ㅣ"
- act=0.483: "Bopomofo: ㄅ ㄆ ㄇ ㄈ ... ㄨ ㄩ ㄭ"

Numeric trace:

    prompts = 5
    pad chunks per prompt = 3 (chunks 0, 1, 2; chunk 3 holds the prompt)
    pad chunks total = 15
    pad chunks hitting Hangul+Bopomofo = 15
    attractor rate = 15/15 = 100%

This is the bank's "what does the encoder return for near-empty input?"
attractor leaking into CCA. At training time the model never saw this
because training fills full blocks. At inference with short prompts it
dominates 3 of the 4 chunk slots. Mitigations to consider later: (a) skip
retrieval for chunks below an information threshold; (b) detect repeated
retrieval and drop neighbors; (c) use shorter `block_size` at inference.
None implemented here.

### 4. 55M generation is incoherent regardless of retrieval

Sample (prompt + WITHOUT + WITH at seed 1337):

| prompt | WITHOUT | WITH |
|---|---|---|
| "The theory of general relativity describes" | "the end of the future." | "the time of the future." |
| "Machine learning is a branch of artificial intelligence that" | "is in the Dark Age." | "is in some ways. They can have a different kind so like a person that is a person." |
| "The French Revolution began in 1789 when" | "the end of the 20th century created the new new new 20th century..." | "you are you to get to get to?!" |
| "In computer science, a hash table is" | "from any room to make a book to the universe..." | "." |
| "The Great Wall of China was built" | "." | "." |

Most continuations EOT in <20 tokens. Coarse word-overlap between generated
text and retrieved sources is 0-1 words (sample size too small and outputs
too short for the metric to mean anything). **No claim is made about
grounding effect at this scale.** This is consistent with `docs/PLAN.md`'s
note that the paper's k-free model "is not instruction-tuned and produces
incoherent Wikipedia-style continuations." Meaningful grounding evaluation
needs either a larger ckpt or the V1 `answer_pipeline.py` (Phi-3 + verifier,
already shipped at `v1-grounded-answer-pipeline`).

## How to re-run

Terminal 1 (cc_service):

    $env:CCMEM_DB_PATH = "H:\MiniLM\cc_service\bank_subset_50k.db"
    $env:CCMEM_ENCODER = "all-MiniLM-L6-v2"
    $env:CCMEM_DEVICE = "cuda"
    cd H:\ModelDimensions_SLM\ModelDimensions
    .\.venv\Scripts\python.exe -m uvicorn src.cc_service.main:app `
        --host 127.0.0.1 --port 8765 --log-level info

Terminal 2 (generation):

    cd H:\ModelDimensions_SLM\ModelDimensions
    .\.venv\Scripts\python.exe scripts\ask_retro.py `
        --service-url http://127.0.0.1:8765 `
        --out-json reports\live_generation_smoke.json `
        --out-md reports\live_generation_smoke.md `
        --max-tokens 80 --seed 1337

If the subset bank doesn't exist:

    .\.venv\Scripts\python.exe scripts\make_subset_bank.py `
        --source H:\MiniLM\cc_service\bank.db `
        --dest H:\MiniLM\cc_service\bank_subset_50k.db `
        --n 50000

## What this is and is not

This smoke proves the wiring works -- bytes flow correctly from RetroGPT
chunk decoding, through cc_service `/query`, through tokenized neighbors,
into the CCA layers, and back to a generated token stream. It does **not**
demonstrate that retrieval improves generation quality at this checkpoint
scale; the 55M model is too small for that to be measurable in a 5-prompt
smoke. The honest claim is "the path is live"; the honest non-claim is
"retrieval does or does not help."

For grounded answer generation that does work, use
`src/v1/answer_pipeline.py` (Phi-3 + verifier, shipped at
`v1-grounded-answer-pipeline`).
