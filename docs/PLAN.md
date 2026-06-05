# Plan: reproduce the paper's headline from this repo

This document is the working contract for the next slice of work. The agent
follows it. No drift, no scope creep, no "while I'm here" cleanups.

## Destination

The system described in `docs/Knowledge_Free_RETRO_Paper.docx` must be
demonstrable from this single repo, using the checkpoints and bank already on
disk. "Demonstrable" means three things:

1. The paper's §4.2 / §5.4 held-out triple (`val_with_real`, `val_with_random`,
   `val_without`) is reproduced from `src/retro/` against the 55M
   bank-trained baseline **and** the 404M knowledge-free model. Numbers are
   written to a JSON artifact and asserted by a pytest.
2. The 1.8M-cell concept-cell bank at `H:\MiniLM\cc_service\bank.db` is
   measured for selectivity and rejection at deployed scale, addressing the
   open threat-to-validity flagged in paper §9a. Numbers are written to a JSON
   artifact.
3. One end-to-end script demonstrates the architecture: query → bank
   retrieval → grounded answer or honest refusal, against the real 13 GB
   bank, with no UI and no governance layer.

## Scope lock

### In scope

- `evals/test_paper_headline_reproduces.py` — pytest asserting the held-out
  gaps from step 1's JSON are within tolerance of the paper.
- `experiments/exp14_paper_headline_reproduce.py` — runs
  `src/retro/eval_retro_heldout.py` (or its in-repo replacement) against both
  checkpoints; writes `results/paper_headline_reproduce.json`.
- `experiments/exp15_bank_selectivity_at_scale.py` — selectivity / rejection
  probe against `H:\MiniLM\cc_service\bank.db` via
  `src/agent/sqlite_bank.py`; writes `results/bank_selectivity_at_1p8m.json`.
- `scripts/demo_end_to_end.py` — one CLI script: `python demo_end_to_end.py
  "your question"` → either a cited answer or `[silence: no matching memory]`.
- Minor patches to `src/retro/eval_retro_heldout.py` only as needed to make it
  runnable as a module from the repo root with configurable `--data-dir` and
  `--ckpt` flags. No functional changes to the eval logic itself.
- One paragraph added to `README.md` once results land, pointing at the
  reproduced numbers and how to re-run them.

### Out of scope (do not touch in this slice)

- Training. Every checkpoint exists. No `train_*.py` is run.
- The SLM controller layer beyond what `scripts/demo_end_to_end.py` needs to
  ground an answer. `src/slm/`, `src/agent/orchestrator.py`,
  `src/agent/response_policy.py`, `src/agent/verifier.py` are used as-is.
- Governance, source registry, review queues, lifecycle states, approval
  metadata, RAI/SSSC, or anything else from the deleted workbench.
- The M365 consulting assistant example from §14 of the architecture
  write-up. Reintroducing project packs is rebuilding the workbench.
- Full README rewrite. One paragraph addition only, at the end.
- Investigating the cc_service encoder-singleton design. The DI fix from
  commit `9845de4` already unblocked the tests; further redesign is its own
  slice.
- Touching `H:\MiniLM\`. All reads go via absolute paths or
  `src/agent/sqlite_bank.py`. Copies are non-destructive.
- Creating any markdown beyond this file (`PLAN.md`) and the one README
  paragraph.

### Hard rules

- Each step is an independently committable slice. If a step blocks, commit
  what's done and surface the blocker; do not skip ahead.
- If the measured numbers disagree with the paper, **report the gap
  honestly**. Do not tune thresholds, data ranges, or batch counts to make
  numbers match.
- If a path doesn't exist or a file is missing, stop and ask. Do not
  fabricate fallback data.
- No new modules under `src/` unless a step requires it. New code lives under
  `experiments/`, `scripts/`, or `evals/`.
- No commits without an explicit "do it" from the user. The standing rule
  from the consolidation slice still applies.

## Steps

### Step 1 — Reproduce the held-out triple

**Deliverable:** `results/paper_headline_reproduce.json` containing, for each
of `{55M_bank_trained, 404M_kfree}`:

```json
{
  "checkpoint_path": "...",
  "val_without": <float>,
  "val_with_random": <float>,
  "val_with_real": <float>,
  "semantic_gap_real_vs_random": <float>,
  "total_gap_real_vs_without": <float>,
  "n_batches": <int>,
  "batch_size": <int>,
  "elapsed_sec": <float>,
  "device": "..."
}
```

Plus a top-level block comparing measured vs paper-claimed:

```json
{
  "paper_claims": {
    "55M_baseline": {"semantic_gap": 0.164, "random_harm": -0.003},
    "404M_kfree":  {"semantic_gap": 0.167, "random_harm": -0.041}
  },
  "agreement": {"55M": "match | drift | blocker",
                "404M": "match | drift | blocker"}
}
```

**Steps:**

1. Smoke-load each checkpoint (no eval, just `torch.load` + model construct)
   to confirm the wiring works. If VRAM blows up, stop and report.
2. Run eval with `--n-batches 5` against the 55M checkpoint to confirm the
   data path. Numbers will be noisy; that's fine — we're checking it runs.
3. Full eval (paper's `--n-batches` default) for 55M. Write JSON entry.
4. Same smoke + full eval for the 404M checkpoint.
5. Stop condition: JSON written and the "agreement" field is honestly
   filled (match / drift / blocker), regardless of whether the numbers
   match the paper.

**Pytest:** `evals/test_paper_headline_reproduces.py` re-reads the JSON and
asserts each measured semantic gap is within ±0.05 nats of the paper claim
(tolerance is generous because of batch-sampling variance). If the JSON
doesn't exist yet, the test skips with an explicit reason.

### Step 2 — Measure bank selectivity at 1.8M cells

**Deliverable:** `results/bank_selectivity_at_1p8m.json` containing:

- `n_cells`: confirmed cell count from the bank.
- `known_queries`: ~20 queries whose answer is provably in the bank
  (Wikipedia article titles + first sentences); reports top-1 hit rate,
  median activation, median rank of the correct cell.
- `unknown_queries`: ~20 queries that are syntactically plausible but
  semantically absent (made-up names, scrambled phrases); reports
  false-fire rate at the bank's configured `rerank_min_score` (0.10).
- `noise_queries`: ~10 randomly-sampled non-text tokens / gibberish; reports
  false-fire rate.
- `latency_p50_ms`, `latency_p95_ms` over all queries — addresses the paper's
  "sub-200ms" claim directly.

**Steps:**

1. Open `H:\MiniLM\cc_service\bank.db` via `src/agent/sqlite_bank.py`
   (read-only). Confirm `n_cells == 1_817_204` per the project log; if it
   differs, record actual count, do not assume the log is right.
2. Build the three query sets. The known set is generated by sampling
   `source_texts.text` rows directly from the DB. No fabrication.
3. Run all queries; record activations and timing.
4. Write JSON. No assertions yet — this is measurement, not a regression
   gate. The paper itself flags this as an open question, so honest numbers
   beat any pass/fail outcome.

### Step 3 — End-to-end grounded-answer demo

**Deliverable:** `scripts/demo_end_to_end.py`. Single CLI:

```bash
python scripts/demo_end_to_end.py "Who founded Microsoft?"
python scripts/demo_end_to_end.py "What is the airspeed of a banana?"
```

Behaviour:

1. Encode query with the same encoder the bank was built with (read from
   the bank's `meta.encoder_model` — fail loudly on mismatch).
2. Retrieve top-k from the bank.
3. Apply the bank's `rerank_min_score` (0.10) — if nothing survives, print
   `[silence: no matching memory]` and exit 0.
4. Otherwise, print the top-N retrieved cells with their cell_id and
   source_text. The "grounded answer" is the cited evidence itself, not a
   generated response. **No LLM generation in this demo.** Generation is
   what the 404M model in step 1 does; this script is the inspection layer.
5. Exit code: 0 if either a cited answer or an honest refusal; 1 only on
   actual error (missing bank, encoder mismatch, etc).

**Steps:**

1. Write the script. ~80 lines.
2. Smoke-test with both a known-answer query and a known-unknown query.
3. Confirm both paths behave as designed.

## Definition of done

The slice is done when all three artifacts exist:

- `results/paper_headline_reproduce.json` (real numbers, honestly compared)
- `results/bank_selectivity_at_1p8m.json` (real numbers, no assertions)
- `scripts/demo_end_to_end.py` (works against the real bank)

…and the new pytest `evals/test_paper_headline_reproduces.py` either passes
or fails with a clear "measured X, paper claims Y, gap exceeds tolerance"
message.

## Stop conditions (any of these stops the slice)

- The 404M checkpoint won't load in available VRAM. Commit step 1's 55M
  result and stop. Report the VRAM blocker.
- An eval data file referenced by `eval_retro_heldout.py` is missing. Stop
  and ask. Do not regenerate from a different source.
- The bank's `encoder_model` meta doesn't match the encoder the loaders use.
  Stop and ask before forcing an override.
- More than one test outside `evals/test_paper_headline_reproduces.py`
  starts failing. Stop. Diagnose. Do not "fix" unrelated breakage in this
  slice.

## What this is not

- Not a paper rewrite. The paper stands.
- Not a release. No version bump, no tag.
- Not a new architectural direction. The architecture is what the paper
  describes; this slice makes that architecture *demonstrable from this
  repo*.
- Not a workbench reboot. The deleted workbench stays deleted.
