# Knowledge-Free Reader: scaling diagnostics

This document covers the six measurable gates the "Knowledge-Free Reader"
roadmap implies, the modules that compute them, and the CLIs that wrap
those modules. Every gate has a numeric pass/fail; nothing here is
qualitative.

The work-in-progress system this document instruments is the 404M
RETRO-style reader at `src/retro/`, the frozen-MiniLM encoder, and the
5.7M-cell production bank at `H:\MiniLM\cc_service\bank.db`. The roadmap
goal is to scale the training corpus to ~8B disjoint tokens, expand the
bank toward 18M+ cells, and verify all of the following claims hold at
that scale.

## The six gates

| # | Gate | Target | Module | CLI |
| - | --- | --- | --- | --- |
| 1 | Ignorance test | `loss(random) − loss(none) ≥ 0.041 nats` | [retro/diagnostics/acceptance.py](../src/retro/diagnostics/acceptance.py) | [scripts/run_acceptance_tests.py](../scripts/run_acceptance_tests.py) |
| 2 | Semantic gap | `loss(none) − loss(real) ≥ 0.167 nats` | acceptance.py | run_acceptance_tests.py |
| 3 | Loss ceiling | `loss(real) ≤ 3.78 nats` | acceptance.py | run_acceptance_tests.py |
| 4 | Layer attribution | one CCA layer carries ≥ 0.50 of the retrieval benefit | [retro/diagnostics/layer_attribution.py](../src/retro/diagnostics/layer_attribution.py) | [scripts/measure_layer_attribution.py](../scripts/measure_layer_attribution.py) |
| 5 | Binding at scale | `m=8` success ≥ 0.92, false-fires ≤ 0.10 per trial | [concept_cells/binding_calibration.py](../src/concept_cells/binding_calibration.py) | [scripts/calibrate_binding_at_scale.py](../scripts/calibrate_binding_at_scale.py) |
| 6 | ZCA whitening | effective-dim fraction ≥ 0.95, mean&#124;cosine&#124; ≤ 0.05 | [concept_cells/zca_verification.py](../src/concept_cells/zca_verification.py) | [scripts/verify_zca_isotropy.py](../scripts/verify_zca_isotropy.py) |

Two upstream gates protect the validity of every measurement above:

| # | Gate | Target | Module | CLI |
| - | --- | --- | --- | --- |
| 0a | Corpus disjointness | training-corpus / bank paragraph-overlap ≤ 0.1%, title-overlap = 0 | [retro/diagnostics/disjointness.py](../src/retro/diagnostics/disjointness.py) | [scripts/verify_corpus_disjointness.py](../scripts/verify_corpus_disjointness.py) |
| 0b | Training plan sanity | iterations × batch × seq covers the unique-token target at the Chinchilla ratio | [retro/diagnostics/compute_plan.py](../src/retro/diagnostics/compute_plan.py) | [scripts/plan_chinchilla_scaling.py](../scripts/plan_chinchilla_scaling.py) |

Gate 0a is hard: if any corpus shard overlaps the bank, the entire
"Reader, not memoriser" claim collapses, because the model can satisfy
the semantic-gap gate using memorised content instead of retrieval. The
verifier hashes paragraph text at SHA-1 of a normalised form and runs
set arithmetic against the bank's `source_texts` table.

Gate 0b is soft: it surfaces when an iteration budget cannot cover the
intended unique-token surface in one epoch and warns about over-coverage
(multi-epoch repetition that breaks the "fresh-reading" assumption).

## Running everything

The CLIs are independent and idempotent. None of them modifies the bank,
the model, or any V1 release artifact. All write to `reports/` if asked.

```pwsh
# 1. Compute and data plan (deliverable for the roadmap's first command)
python scripts\plan_chinchilla_scaling.py `
  --hourly-cost a100_80gb_bf16=2.50 --hourly-cost h100_80gb_bf16=5.00 `
  --out-json reports\chinchilla_scaling_plan.json `
  --out-md   reports\chinchilla_scaling_plan.md

# 2. Disjointness on a candidate corpus shard
python scripts\verify_corpus_disjointness.py `
  --bank H:\MiniLM\cc_service\bank.db `
  --corpus path\to\shard.jsonl `
  --out-json reports\disjointness_shard.json

# 3. Acceptance gates over a 3-condition eval result
#    losses.json must contain {"none": <float>, "random": <float>, "real": <float>}
python scripts\run_acceptance_tests.py `
  --losses results\heldout_losses.json `
  --out-json reports\acceptance.json

# 4. Layer attribution from a per-layer suppressed-loss dict
python scripts\measure_layer_attribution.py `
  --losses results\layer_attribution_losses.json `
  --out-json reports\layer_attribution.json

# 5. Binding calibration on an embeddings .npy sample of the bank
python scripts\calibrate_binding_at_scale.py `
  --embeddings results\bank_embeddings_sample.npy `
  --m 4 --m 6 --m 8 --trials 200 --distractors 5000 `
  --out-json reports\binding_calibration.json

# 6. ZCA verification on fit / eval embedding splits
python scripts\verify_zca_isotropy.py `
  --fit-embeddings  results\bank_fit_emb.npy `
  --eval-embeddings results\bank_eval_emb.npy `
  --out-json reports\zca_verification.json
```

Each CLI exits `0` on pass, `2` on gate failure, and `1` for invalid
arguments / missing inputs. They are wired this way so CI can wrap them.

## Producing the inputs the CLIs need

The CLIs only do the *measurement*. Several inputs require a GPU and the
real model; those production steps live outside this document and are
called out below.

* **Losses for gates 1–3** are produced by `src/retro/eval_retro_heldout.py`'s
  existing `evaluate()` function, which accepts `mode="none"|"random"|"real"`.
  Run it three times against a held-out shard and write the three means
  into a `{none, random, real}` JSON file. No new code is required.

* **Per-layer suppressed losses for gate 4** require an ablation pass:
  for each CCA layer index `L`, run the model with layer `L`'s
  cross-attention output replaced by zero and record the `real`-mode loss.
  The baseline is the un-ablated `real` loss. Submit those into the
  layer-attribution CLI as `loss_real_with_layer_suppressed`.

* **Embedding samples for gates 5 and 6** are random draws from the bank's
  cell weights. Save them as `.npy` arrays of shape `(N, D)` (in practice
  `(50_000, 384)` is enough to characterise both binding and ZCA).

## What "pass" means at scale

The three loss-side gates (1–3) are the headline numbers from the
underlying paper, in nats. A model that passes all three has *demonstrated*
the reading hypothesis: it does its work through retrieval, the retrieval
contents matter, and the absolute floor is met. A model that passes
gates 1 and 3 but fails gate 2 is silently parametric — the retrieval
channel is present but inert. A model that passes gates 2 and 3 but
fails gate 1 is suspiciously brittle — real retrieval helps but random
retrieval doesn't hurt, which usually means the model is averaging away
the noise rather than reading it.

Gates 4–6 are structural integrity checks on the surrounding
infrastructure. Even if the loss-side gates pass on a small bank, the
production bank and the production training corpus must continue to pass
4–6 for the system to keep working.

Gates 0a and 0b are pre-flight checks. They prevent the most expensive
class of mistake (training on a corpus that already lives in the bank)
and the second-most expensive (running 400k iterations only to discover
the run never covered one epoch of the intended data).

## Test coverage

All six modules ship with offline unit tests in `evals/`:

* [test_retro_diagnostics.py](../evals/test_retro_diagnostics.py) — acceptance, compute_plan, disjointness, layer_attribution.
* [test_binding_calibration.py](../evals/test_binding_calibration.py) — synthetic isotropic vectors, no bank required.
* [test_zca_verification.py](../evals/test_zca_verification.py) — synthetic anisotropic vectors, verifies direction-of-change.

The tests do not require a GPU, the real model, the live bank, or any
network access. Run them with:

```pwsh
.venv\Scripts\python.exe -m pytest evals\test_retro_diagnostics.py `
  evals\test_binding_calibration.py evals\test_zca_verification.py -q
```
