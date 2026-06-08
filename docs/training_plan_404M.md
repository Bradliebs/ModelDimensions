# Training Plan — 404M kfree RetroGPT

**Status:** Planning. No training started against this plan.
**Author:** Step 5 of the post-V1 roadmap.
**Audience:** Whoever runs the next training pass (likely future-self).
**Anchor commits:** model + trainer at `35e2485`. V1 inference stack at
`v1-grounded-answer-pipeline` = `9269b6a` (do not touch).

This document is an engineering contract for the next checkpoint, not
marketing for it. It says what would have to be true to ship a 404M
kfree RetroGPT, what gates must pass before that checkpoint is allowed
into the V1 / cc_service inference path, and what is explicitly **not**
in scope.

## TL;DR

| Question | Answer |
|---|---|
| Target arch | n_layer=24, n_head=16, n_embd=1024, block_size=256, chunk_size=64, n_neighbors=2, neighbor_len=64, cca_every=2 — **already defined** in [src/retro/train_retro_kfree_large.py](../src/retro/train_retro_kfree_large.py) |
| Param count | ~404M (paper figure; verify with `n_params/1e6` print at first build) |
| Compute budget for full Chinchilla | ~8B tokens ≈ 8.1B token-presentations; ~62× the current 15K-iter run |
| Minimum viable budget for gate signal | ~1B tokens, see §4 |
| Corpus | Not yet committed. §3 specifies the contract; Step 2 picks the sources |
| Pass/fail gates | semantic_gap ≥ +0.167 nats, ignorance_test ≥ +0.041 nats, loss_ceiling ≤ 3.78 nats — already wired in [scripts/run_acceptance_tests.py](../scripts/run_acceptance_tests.py) |
| Hardware | RTX 3070 (8.59 GB) — fits at batch=2 + grad_accum=16 + AdamW8bit + bf16; verified peak ~8.2 GB |
| Decision after gate | Pass → integrate into cc_service inference path. Fail → keep 55M ckpt + V1 Phi-3 pipeline as production |

The 404M target is the paper's claim, not ours. This plan describes the
work to either reproduce it or honestly report the gap.

## 1. What we have now (the 55M baseline)

| Item | Value |
|---|---|
| Checkpoint | `H:\MiniLM\nanogpt\out-retro-bank\ckpt_best.pt` |
| Arch | n_layer=8, n_head=8, n_embd=512, block_size=256, chunk=64, n_neighbors=2, neighbor_len=64 |
| Params | 55.29 M |
| Iters | 13,000 |
| Token-presentations seen | ~106 M (13000 × 32 effective batch × 256 block) |
| val_with retrieval | 3.6045 nats |
| val_without retrieval | 3.8279 nats |
| Empirical retrieval gap | 0.2234 nats (vs paper claim 0.167) |
| Loss ceiling | 4.24 nats real / 4.42 nats none (vs paper 3.78) |
| Ignorance test | −0.0083 nats (random ≈ none — FAIL vs paper +0.041) |
| Layer-attribution share (best layer) | 0.362 at layer 5 (FAIL vs paper ≥0.50) |

Honest read: the 55M ckpt reproduces the **weak form** of the paper
claim (retrieval contributes a measurable gap) and fails both the
**strong form** (no single layer dominates ⇒ no clean "knowledge
localization") and the **loss-ceiling** form (we are 0.46 nats above the
paper's reported per-token NLL). All three failures are consistent with
"this model is undertrained for its task" rather than "the mechanism
does not exist."

This is the baseline a 404M run has to beat to justify the cost.

## 2. Target architecture — already coded, do not redesign

The 404M target arch is the one already configured in
[src/retro/train_retro_kfree_large.py](../src/retro/train_retro_kfree_large.py):

```python
config = RetroConfig(
    block_size=256,
    vocab_size=50304,           # GPT-2 BPE
    n_layer=24,
    n_head=16,
    n_embd=1024,
    dropout=0.0,
    bias=False,
    chunk_size=64,
    n_neighbors=2,
    neighbor_len=64,
    cca_every=2,                # CCA in layers [1, 3, 5, ..., 23]
)
```

Rationale for not changing it:

- **Width/depth match the paper.** Reusing the paper's shape removes one
  confound when comparing the gate results.
- **chunk_size=64, n_neighbors=2** match what the bank was built around
  and what the 55M ckpt trained on. Changing them invalidates every
  prior comparison.
- **cca_every=2** puts CCA in 12 of 24 layers. The 55M layer-attribution
  finding (no layer dominates) is consistent with the paper's
  observation that CCA contributes distributed signal; doubling the
  layer count gives more attribution surface, not necessarily more
  per-layer share.
- The trainer is already sized to fit an 8.59 GB GPU with batch=2 +
  grad_accum=16 + AdamW8bit + bf16. Verified peak ~8.2 GB.

If the arch needs to change, that is a different plan, not this one.

**Pre-flight check before training**: at first `model.build()`, print
`sum(p.numel() for p in model.parameters()) / 1e6`. Assert it lands in
[395, 415] M. If it does not, the config has drifted and the rest of
this plan does not apply.

## 3. Corpus contract — what Step 2 must deliver

This plan does **not** commit to a corpus source. That belongs to Step 2
of the broader roadmap. What this plan does commit to is the **shape**
of the corpus, so Step 2 can be scoped without re-opening this one.

### 3.1 Two streams, one bank

The kfree training loop needs two parallel streams that share one bank:

| Stream | What it is | Backing file (current 55M run) | Required for 404M |
|---|---|---|---|
| Main token stream | Wikipedia-like prose, tokenised to GPT-2 BPE | `data/kfree/train.bin`, `val.bin` (np.uint16 memmap) | **same format**, ~8B train tokens, ~10M val tokens |
| Neighbour index | One row per chunk in the token stream; each row is a list of `n_neighbors` cell IDs | `data/kfree/train_neighbors.npy`, `val_neighbors.npy` (np.int64) | **same format**, must be aligned to the new token stream chunk-by-chunk |
| Bank cell tokens | One row per bank cell; tokenised text of that cell | `data/bank/cell_tokens.npy` | **same** — the 5.7M cell bank at `H:\MiniLM\cc_service\bank.db` is reusable as-is |

The fact that **the bank does not need to change** is the single biggest
cost saver in this plan. Re-encoding 5.7M cells with a different
encoder would invalidate ZCA, every threshold, and every V1 calibration
number.

### 3.2 Knowledge-free property — non-negotiable

The training corpus must be **disjoint** from the bank corpus in the
sense that fact-bearing spans in the training stream are not present
verbatim or near-verbatim in the cell bank. The 55M run used a
disjointness check in [scripts/verify_corpus_disjointness.py](../scripts/verify_corpus_disjointness.py).
The 404M run **must re-pass this check on the new corpus** before any
training iteration runs. If it does not pass, the retrieval gap is
contaminated and the gates in §5 are meaningless.

Concrete acceptance: `verify_corpus_disjointness.py --train-bin
data/kfree/train.bin --bank-db H:\MiniLM\cc_service\bank.db
--max-13gram-overlap 0.01` returns exit 0.

### 3.3 What Step 2 has to answer

When Step 2 opens, the conversation needs to land:

1. **Source**: FineWeb-Edu, RedPajama-Wiki, ROOTS-en-wiki, or a curated
   mix? (Open question. Cost varies wildly. License varies wildly.)
2. **Disk budget**: 8B GPT-2 tokens × 2 bytes/uint16 = 16 GB for tokens
   alone. Neighbour index at ~125M chunks × 2 neighbours × 8 bytes ≈
   2 GB. Total raw artefacts ≈ 20 GB. Add ~50 GB working space for
   tokenisation + dedup + disjointness check. **Plan for 80 GB.**
3. **Dedup strategy**: minhash + exact-13gram. Reused tooling — do not
   reinvent.
4. **Tokenisation**: must use the same `tiktoken` `cl100k`/`gpt2` config
   the 55M run used. The bank's `cell_tokens.npy` is already in that
   format.

None of these are decided by this plan. This plan only says: until they
are decided, the 404M run cannot start.

## 4. Token budget — Chinchilla math and the honest gap

Chinchilla (Hoffmann et al. 2022) puts compute-optimal tokens per
parameter at ~20:1. For 404M params that is **~8.1B tokens**.

Current 55M run: 13K iters × 32 effective batch × 256 block ≈ 106M
token-presentations. That is 106M / 55M = ~1.9 tokens per parameter,
~10× under Chinchilla. The fact that semantic_gap still hit +0.179 at
that undertraining is the reason there is any case at all for a 404M
run.

Three honest budget tiers:

| Tier | Tokens | Iters at current effective batch=32, block=256 | Wall time (single RTX 3070, ~2.5 tok/s eff per step) | What it proves |
|---|---|---|---|---|
| **Smoke** | 100M | ~12,200 | ~3–5 days | Trainer runs end-to-end at 404M without OOM, val_with < val_without |
| **Gate signal** | 1B | ~122,000 | ~30–50 days | Enough to test whether semantic_gap moves above 55M's +0.179. Cannot prove paper's +0.167 ceiling but can falsify "404M is no better than 55M" |
| **Chinchilla** | 8B | ~976,000 | ~8–14 months | Full claim. Pass all three gates or fail honestly |

The wall-time numbers above are **best-case estimates** with a single
RTX 3070. They will be worse in practice (CCA layers slow down per-step
relative to vanilla nanoGPT). If the budget is "do this on one 3070,"
the realistic plan is **Tier 2 (Gate signal)**, then make a go/no-go
call before committing to Tier 3.

If a real cluster appears (1× A100-80 GB ≈ 8–10× the throughput, 8× A100
≈ 60–80×), the Chinchilla tier becomes 2–4 weeks. This is the
conversation Step 2 has to have alongside corpus selection.

**This plan does not assume any hardware beyond the current 3070.**
Anything that needs a cluster is flagged as such.

## 5. Pass/fail gates — already wired

The acceptance gates from the paper are already implemented and live as
pure pass/fail wrappers in
[scripts/run_acceptance_tests.py](../scripts/run_acceptance_tests.py).
The 404M run is not allowed to be called "shipped" until all three pass
on a held-out eval that **did not contribute to training**.

| Gate | Target | Falsifies | Current 55M result |
|---|---|---|---|
| **semantic_gap** = mean(loss_none − loss_real) | ≥ +0.167 nats | "retrieval helps in expectation" | +0.179 PASS |
| **ignorance_test** = mean(loss_random − loss_none) | ≥ +0.041 nats | "the model penalises *wrong* retrieval, not just *no* retrieval" | −0.008 FAIL |
| **loss_ceiling** = mean(loss_real) | ≤ 3.78 nats | "the model trained to a competitive perplexity on this corpus" | 4.24 FAIL |

Numeric trace example for ignorance_test (per the
directional-diagnostic-numeric-trace rule):

```
loss_random = 4.42
loss_none   = 4.42
delta       = 4.42 − 4.42 = 0.00
target      = 0.041
0.00 < 0.041 → FAIL
```

Diagnostic gate (not blocking, but reported): **layer_attribution_share**
from [scripts/measure_layer_attribution.py](../scripts/measure_layer_attribution.py).
Paper expects ≥0.50 for a dominant layer; current is 0.362. Reporting
this is a sanity check that CCA is wired correctly, not a ship gate.

### 5.1 What "passes" means in practice

If after the Tier 2 (1B token) checkpoint:

- **All three gates pass** → proceed to Tier 3 (Chinchilla). Document
  the gate-tier numbers as an intermediate result.
- **semantic_gap passes, others fail** → same outcome as the 55M
  baseline. Stop. The 404M run did not earn its compute. Write a
  short honest report and keep V1+Phi-3 as production.
- **semantic_gap fails** → something is wrong with corpus disjointness,
  neighbour index alignment, or the CCA wiring. Stop and diagnose.
  Do not "train through it."

### 5.2 The cc_service integration gate (Step 4 follow-on)

Even if all three paper gates pass, the 404M ckpt does not automatically
replace V1 as the production inference path. It must additionally:

1. Pass the same end-to-end smoke that
   [scripts/ask_retro.py](../scripts/ask_retro.py) runs at 55M, on at
   least 20 prompts (not 5), against the **full** cc_service bank with
   whatever streaming-top-k workaround exists at that time (see
   the architectural finding in [reports/step4_summary.md](../reports/step4_summary.md)).
2. Produce coherent continuations — measured by perplexity on a
   held-out instruction-style probe set and by a simple human-graded
   spot check (n=20 prompts × 2 graders).
3. Demonstrate the pad-chunk attractor finding from Step 4 either does
   not appear at 404M, or is mitigated by an inference-time filter
   (skip retrieval below an info threshold). Numeric trace must show
   ≤25% of pad chunks retrieving the same top-1 cell, vs the 55M
   measurement of 100% (15/15).

If any of these three fail, the 404M ckpt ships as a research artefact
and **not** as the V1 generator replacement. V1 stays as the answer
path.

## 6. Checkpoint cadence and abort criteria

Borrowed verbatim from
[src/retro/train_retro_kfree_large.py](../src/retro/train_retro_kfree_large.py):

| Knob | Value | Why |
|---|---|---|
| `EVAL_INTERVAL` | 250 iters | ~5% of the smoke tier, ~0.2% of the full Chinchilla tier — frequent enough to see drift, rare enough to not dominate wall time |
| `EVAL_BATCHES` | 20 | Stable val_with / val_without estimate without spending real training time |
| `LOG_INTERVAL` | 50 | Live throughput / lr / loss readout |
| `MAX_ITERS` | 15000 (today) → 122000 (Tier 2) → 976000 (Tier 3) | Tier-dependent — set explicitly per run |
| `WARMUP_ITERS` | 300 | Standard. Do not change without reason |
| `LR_MAX` | 2e-4 | Slightly lower than 55M (3e-4) because the model is larger and bf16 has less numerical headroom |
| `LR_MIN` | 2e-5 | 10× decay to cosine floor |
| Checkpoint policy | Save `ckpt_resume.pt` (full state) every `EVAL_INTERVAL`; save `ckpt_best.pt` (weights only) only when `val_with` improves | Already implemented |

### 6.1 Abort criteria — pull the plug if

| Condition | Threshold | Why |
|---|---|---|
| Loss NaN | Any iter | Numerical collapse; resume from last good ckpt, lower LR by 2× |
| val_with > val_without for 3 consecutive evals | After iter 1000 | Retrieval is actively hurting — corpus disjointness or neighbour-alignment bug |
| Throughput drops > 30% from steady-state | Any | Disk thrash, thermal throttle, or memory leak. Diagnose before continuing |
| Disk free < 20 GB on H: | Any | Checkpoints + memmap will hit the wall. Move old ckpts before resuming |
| Wall-time projection exceeds budget by 2× | Tier-dependent | Recalibrate or stop |

These are not paranoia. Each one is a real failure mode that has
shown up in prior runs in adjacent projects.

## 7. What this plan does NOT commit to

Per the always-on hygiene rules, these items are deliberately excluded
and require their own planning:

- **Corpus source selection.** Step 2 conversation.
- **Tokenisation tooling changes.** Reuse what the 55M run used.
- **Encoder swap.** Bank uses MiniLM-L6-v2; this is fixed for the
  duration of the 404M run.
- **Arch changes.** No grouped-query attention, no rotary, no
  block_size > 256, no chunk_size != 64. If any of these become
  interesting, write a new plan.
- **Multi-GPU training.** The current trainer is single-GPU. Adding
  FSDP / DDP is a separate work item.
- **Instruction tuning.** The 404M model is a base LM. It will not
  produce instruction-style answers without a separate post-training
  step. That is out of scope; V1's Phi-3 generator is the answer path
  for instructions.
- **Evaluating on tasks not in the gate set.** MMLU, HellaSwag, etc.
  are not in this plan. The gate set is the paper's claims, not
  general LM benchmarks.
- **Anything that would modify** `src/retro/*`, `src/cc_service/*`,
  the V1 release artefacts at tag `v1-grounded-answer-pipeline`, or
  the 5.7M-cell production bank. All four are read-only for this
  plan's purposes.

## 8. Execution checklist (when Step 2 lands the corpus)

When the corpus is decided and ready, the actual training run is:

```powershell
# 1. Verify corpus disjointness (must pass before training starts)
.\.venv\Scripts\python.exe scripts\verify_corpus_disjointness.py `
    --train-bin H:\MiniLM\nanogpt\data\kfree\train.bin `
    --bank-db H:\MiniLM\cc_service\bank.db `
    --max-13gram-overlap 0.01

# 2. Smoke (Tier 1, ~3-5 days): trainer end-to-end, gate-1 only
Set-Location H:\MiniLM\nanogpt
H:\MiniLM\cc_service\.venv\Scripts\python.exe train_retro_kfree_large.py
# After ~12K iters, kill with Ctrl+C, copy out-retro-kfree-large/ckpt_best.pt aside

# 3. Tier 2 acceptance (1B tokens, ~30-50 days)
# Edit MAX_ITERS=122000 in train_retro_kfree_large.py, resume from ckpt
H:\MiniLM\cc_service\.venv\Scripts\python.exe train_retro_kfree_large.py

# 4. Acceptance gates against the new ckpt
.\.venv\Scripts\python.exe scripts\run_acceptance_tests.py `
    --ckpt H:\MiniLM\nanogpt\out-retro-kfree-large\ckpt_best.pt `
    --eval-batches 100 `
    --out-json reports\acceptance_404m_tier2.json

# 5. If gates pass, go/no-go on Tier 3 (Chinchilla)
# If not, write reports/training_plan_404M_outcome.md honestly and stop.
```

All five steps already have working tooling. Nothing in this checklist
requires new code.

## 9. References

- Paper: `docs/Knowledge_Free_RETRO_Paper.docx` (in the repo)
- 55M reproduction context: [docs/PLAN.md](PLAN.md) Step 1
- Architecture spec: [ARCHITECTURE.md](../ARCHITECTURE.md)
- Scaling diagnostics: [docs/READING_ENGINE_SCALING.md](READING_ENGINE_SCALING.md)
- V1 inference contract (do not touch): [docs/RUNBOOK.md](RUNBOOK.md), tag `v1-grounded-answer-pipeline`
- Step 4 live-loop smoke + pad-chunk attractor finding: [reports/step4_summary.md](../reports/step4_summary.md)
- Chinchilla: Hoffmann et al. 2022 — "Training Compute-Optimal Large Language Models" (arXiv:2203.15556)

---

**Self-check before this plan is acted on**: re-read §3.3 and §4. If any
of those answers have changed, this plan is stale and must be revised
before training starts.
