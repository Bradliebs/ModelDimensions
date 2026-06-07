# V1 Runbook

> **Scope: V1 grounded-answer pipeline only.** This document covers the
> V1 pipeline (`scripts/ask.py`, `scripts/setup.py`), overlay editing,
> the Bank Management Workspace (`run_bank_workspace.bat`), the V1.1
> verification audit, and optional deployment hardening. It does **not**
> cover the post-V1 surfaces shipped later — Consultant Workbench UI,
> governed knowledge-pack lifecycle (`hf-lifecycle`, `pdf-pack`,
> `knowledge-packs`, `active-pack-monitor`, `regression-review`,
> `lifecycle-executor`), source registry, memory-proposal queues, chat
> orchestrator, or document ingest. Those CLI surfaces are documented in
> the top-level [README.md](../README.md) and commit history. An operator
> guide for them is not yet written.

A single-page operator's guide for the V1 cited-answer pipeline.

> Audience: the researcher running this on their own workstation. No
> deployment, no service mode, no auth. The bank is on local disk; the
> generator is local Phi-3-mini-4k-instruct on a single GPU.

## What V1 ships

- **Pipeline:** `encode → whiten → topk → silence_gate → pre-filter → generate (Phi-3) → verify → rescue gate → answer / silence`.
- **Bank:** the production 5.7M-cell concept-cell bank at `H:\MiniLM\cc_service\bank.db` (read-only) plus an optional editable **overlay** at `results/v1_bank/overlay.db`.
- **Defaults:** `top_k=10`, `margin_threshold=0.015`, rescue floor `0.40`, rescue rank window `3`.

The production bank is treated as immutable. All mutations land in the overlay; deleting `results/v1_bank/overlay.db` reverts to a clean base bank.

## 1. Verify the install

```pwsh
python scripts/setup.py
```

Runs five checks (bank path, bank open, encoder identity, generator weights, smoke ask), prints OK/FAIL for each, and writes `results/setup_check.json`. Exits 0 if all checks pass.

Useful flags:
- `--skip-generator` — skip the Phi-3 probe and the smoke ask (fast, no GPU).
- `--skip-smoke` — probe Phi-3 weights but don't run a live ask.
- `--bank-path PATH` — override the default `H:\MiniLM\cc_service\bank.db`.
- `--overlay-path PATH` — verify with the overlay merged in.

## 2. Ask a question

```pwsh
python scripts/ask.py "What is the capital of France?"
```

Output: a one-line answer with citation indices, plus gate / verifier diagnostics. Add `--json` to dump the full `PipelineResult`.

The pipeline returns one of:
- **Grounded answer** with citations — gate fired, verifier accepted.
- **Honest silence: no match** — gate did not fire (top-1 too weak or top-1 vs top-2 margin too thin).
- **Honest silence: drift** — gate fired but the generator's answer wasn't grounded in the cited cell.
- **Rescued answer** — gate failed but a single high-activation cell within the top-3 contained the answer entity verbatim. Logged in `result.rescue.decision == "answer_rescued"`.

## 3. Open the Bank Management Workspace

Double-click `run_bank_workspace.bat`, then open:

```text
http://127.0.0.1:8765
```

The workspace lets you:

- Ask questions against the current bank.
- Preview cells from pasted text, Markdown, text files, or extractable PDFs.
- Add approved cells to the existing overlay store at `results/v1_bank/overlay.db`.
- Search existing base and overlay cells.
- Tombstone incorrect cells with a reason.
- Reload the bank explicitly after overlay edits, with background progress.
- Inspect overlay change history.

The base bank remains read-only. All additions and removals go through
`OverlayStore`; no second knowledge-management layer is introduced.

After adding or tombstoning cells, click **Reload bank** before asking questions
that depend on those edits. Reload runs in the background; the page shows cells
loaded, percent complete, and ETA while the 5.7M-cell bank is read. The header
shows whether a reload is required and whether the bank is ready for questions.

## 4. Add a cell to the overlay from the command line

```pwsh
python -c "from pathlib import Path; from src.cc_service.encoder import EncoderSingleton; from src.agent.streaming_bank import StreamingBank; from src.agent.bank_admin import OverlayStore, add_cell_from_text; bank = StreamingBank(r'H:\MiniLM\cc_service\bank.db'); enc = EncoderSingleton(model_name=bank.encoder_model); ov = OverlayStore(Path('results/v1_bank/overlay.db')); cid, _ = add_cell_from_text(ov, bank_dim=bank.dim, base_max_id=int(bank.cell_ids.max()), encoder=enc, whiten_fn=bank.whiten, text='Mars has two moons, Phobos and Deimos.', source='manual', label='cell_mars_moons'); print('added cell_id=', cid); ov.close(); bank.close()"
```

The new cell is encoded with the same encoder and whitening as the base bank (this is the invariant the rescue floor `0.40` is calibrated against). To exercise the overlay end-to-end:

```pwsh
python scripts/ask.py "What are Mars' moons?" --overlay-path results/v1_bank/overlay.db
```

Or set `MD_OVERLAY_PATH` in your environment so every `ask.py` call merges the overlay automatically.

## 5. Remove a cell from the command line

```pwsh
python -c "from pathlib import Path; from src.agent.bank_admin import OverlayStore; ov = OverlayStore(Path('results/v1_bank/overlay.db')); ov.remove_cell(123456, reason='wrong attribution'); ov.close(); print('tombstoned 123456')"
```

Tombstones a base or overlay cell id. The merged bank suppresses the cell in `topk` from the next process restart onward. Provenance is preserved — the row in `overlay_cells` is not deleted, and `provenance_log` records the removal with reason and timestamp.

## 6. Inspect provenance

```pwsh
python -c "from pathlib import Path; from src.agent.bank_admin import OverlayStore; ov = OverlayStore(Path('results/v1_bank/overlay.db')); [print(r) for r in ov.provenance()]; ov.close()"
```

Lists every `add` and `remove` op against the overlay, oldest first, with timestamps and reasons.

## 7. Re-run validations after changes

After any change to the encoder, whitening, indexing, similarity scoring, or model quantisation, **the rescue floor `0.40` is no longer guaranteed valid** and must be recalibrated:

```pwsh
python experiments/exp25_paragraph_calibration.py
python experiments/exp26_rescue_verification.py
```

Read `results/v1_rescue_verification.json`. The acceptance criteria are documented in `docs/PLAN.md` Step 3.

For the test suite:

```pwsh
python -m pytest evals/
```

Current eval baseline: 165 tests pass.

## 7A. Run the V1.1 verification audit

```pwsh
python scripts/run_verification_audit.py
```

Writes `reports/v1_1_verification_audit.json` and
`reports/v1_1_verification_audit.md` from the independent seed corpus at
`evals/fixtures/v1_1_independent_verification_cases.jsonl`.

The audit reports sample counts, observed rates, Wilson 95% intervals,
rule-of-three upper bounds for zero-event metrics, a threshold risk/coverage
curve, verifier/NLI disagreement candidates, and detector ablations. Treat it
as a credibility check, not as product telemetry.

## 7B. Bind calibration before trusting thresholds

To write the current calibration fingerprint:

```pwsh
python scripts/setup.py --skip-generator --write-calibration-fingerprint results/v1_calibration_fingerprint.json
```

To validate a saved fingerprint:

```pwsh
python scripts/setup.py --skip-generator --calibration-fingerprint results/v1_calibration_fingerprint.json
```

The fingerprint binds encoder identity, embedding dimension, whitening
checksum, scoring mode, quantisation, gate threshold, rescue floor, rescue
rank window, and calibration version. A mismatch fails setup with
`RECALIBRATION_REQUIRED`; recalibrate before using the thresholds with the new
geometry.

## 8. Recovery

The base bank is read-only; nothing the V1 pipeline does can corrupt it. To revert all overlay edits to a clean state:

```pwsh
Remove-Item results/v1_bank/overlay.db
```

The next pipeline construction proceeds without an overlay. To revert a single edit, tombstone its `cell_id` (Section 5); provenance is preserved.

## 9. Known V1 limitations

V1 silences are honest — the pipeline never produces a confident wrong answer. On the canonical 8-question multi-hop set (`experiments/exp20_multihop_probe.py::MULTIHOP_QUERIES`) the V1 baseline scores 6/8 grounded, 0/8 wrong, 2/8 silenced. The two silenced queries are research-known and have measured causes (`results/v1_rerank_diagnostic.json`, exp27):

- **Q2 (Best Picture 1972 → Don Ellis):** recall fault. The keyword-bearing cell is not in the top-50 cosine pool of the production bank; cross-encoder rerank cannot recover it. A future hybrid lexical + dense retrieval pass would be needed.
- **Q5 (WWII monarch succession → Elizabeth II):** decomposer fault. The Phi-3 sub-question generator flips direction (asks "Who succeeded Elizabeth II?" instead of "Who succeeded George VI?"); top retrieval is correct *for the flipped question*, but margin top1−top2 sits at ~0.011 and the gate silences. A future direction-preserving validator on the decomposer output would be needed.

Both are out of scope for V1. The system's contract — "ground or stay silent, never confabulate" — holds on the 8-question set and on the broader 144-test eval suite.

## 10. Optional hardening for non-personal deployments

The workspace is open by default (personal, localhost, single operator). For an
internal/team or public deployment, `app/bank_workspace.py` reads these
environment variables (all off unless set):

| Variable                       | Effect                                                                 |
| ------------------------------ | ---------------------------------------------------------------------- |
| `WORKSPACE_API_TOKEN`          | Require `Authorization: Bearer <token>` on mutating routes (commit, tombstone, reload, preview). |
| `WORKSPACE_REQUIRE_AUTH_READS` | When truthy and a token is set, also require the token on read routes (ask, search, status, history). |
| `WORKSPACE_MAX_BODY_MB`        | Reject requests whose `Content-Length` exceeds this many MB (413). Caps the base64 upload path. |
| `WORKSPACE_RATE_LIMIT_PER_MIN` | Per-client fixed-window limit on the expensive routes (ask, preview); excess returns 429. |

`/health` returns `status: "ok"` only when the bank is loaded, else
`"degraded"` — wire it to your supervisor/load-balancer health probe.

Measure before sizing a public deployment. With the server running, point the
load harness at it (it does not load the bank itself):

```pwsh
.\.venv\Scripts\python.exe scripts\loadtest_workspace.py --url http://127.0.0.1:8765 --concurrency 4 --requests 40 --output results\loadtest.json
```

It reports p50/p95/p99 latency, throughput, and error rate for `/api/ask`.
These controls do not replace a reverse proxy, TLS, or process supervision; they
are the application-level floor. The grounded-answer contract is validated for
the current single-process pipeline only — re-validate if you add multi-worker
serving, batching, or a shared vector store.

### Hosted / GPU inference (escaping the CPU latency wall)

The default generator is local Phi-3 on CPU. A measured load test (12 requests,
concurrency 4) showed ~132 s cold, ~258 s p50 under load, ~0.02 rps, with
timeouts beginning at 4 concurrent requests — generation, not retrieval, is the
bottleneck. To serve more than a single sequential operator, point the workspace
at an OpenAI-compatible inference endpoint (Azure OpenAI, or a vLLM/Ollama host
on a GPU). When `WORKSPACE_LLM_BASE_URL` is set, `app/bank_workspace.py` uses the
hosted generator instead of loading local Phi-3; unset, behaviour is unchanged.

| Variable                  | Effect                                                                 |
| ------------------------- | ---------------------------------------------------------------------- |
| `WORKSPACE_LLM_BASE_URL`  | OpenAI-compatible base, e.g. `https://api.openai.com/v1` or `http://gpu-host:8000/v1`. Enables hosted inference. |
| `WORKSPACE_LLM_MODEL`     | Model name to request (default `gpt-4o-mini`).                          |
| `WORKSPACE_LLM_API_KEY`   | Optional bearer token for the endpoint.                                |
| `WORKSPACE_LLM_TIMEOUT`   | Per-request timeout in seconds (default 60).                           |
| `WORKSPACE_LLM_MAX_TOKENS`| Max generated tokens (default 512).                                    |

The hosted generator is deterministic (`temperature=0`) to match the local Phi-3
path, so the grounded-answer verification sees comparable drafts. It calls
`POST {base_url}/chat/completions` and raises an error rather than returning an
empty or fabricated answer on failure. Switching backends does not change the
grounding contract, but re-run `python -m pytest evals -q` and a smoke ask after
pointing at a new endpoint.

## Where things live

| What                          | Where                                     |
| ----------------------------- | ----------------------------------------- |
| Production bank (read-only)   | `H:\MiniLM\cc_service\bank.db`            |
| Editable overlay              | `results/v1_bank/overlay.db`              |
| Setup verdict                 | `results/setup_check.json`                |
| Rescue calibration            | `results/v1_rescue_verification.json`     |
| Verification audit            | `reports/v1_1_verification_audit.*`       |
| Calibration fingerprint        | `results/v1_calibration_fingerprint.json` |
| Pipeline                      | `src/agent/answer_pipeline.py`            |
| Bank loader                   | `src/agent/streaming_bank.py`             |
| Overlay admin API             | `src/agent/bank_admin.py`                 |
| CLI                           | `scripts/ask.py`, `scripts/setup.py`      |
| Tests                         | `evals/`                                  |

## 11. Hybrid retrieval cascade (fallback-only — gating proof PASSED on runs 3 and 4)

**Status: PASSED on run 3 (100k rank_bm25) and re-confirmed on run 4 (full 5.7M tantivy). Phase 1.5 ships the tantivy backend; cascade interface, orchestration, and cardinal-rule guarantees are unchanged.**

- *Run 1* (harness defective — no decomposer): produced 1-2 wrong answers per mode. Cardinal rule broken. Halt and fix harness.
- *Run 2* (harness fixed — decomposer wired in to mirror `exp22`/`exp27`): 0 wrong across all modes (cardinal rule preserved), but the always-on cascade strictly reduces grounded recall vs cosine_only baseline (6/8 → 2/8). Root cause: BM25 promotes lexically-matching-but-semantically-wrong cells with high gate margins; the cross-encoder rerank cannot fix an already-corrupted candidate pool.
- *Run 3* (orchestrator redesigned — fallback-only, 100k rank_bm25): **PASS.** `hybrid_rerank_fallback` mode scored 7 grounded / 1 silence / 0 wrong. The cosine path is preserved byte-identically on the 6 queries cosine can answer; hybrid+rerank runs as a rescue *only* on cosine silences. Q5 (Elizabeth II → Charles III succession) was the rescue. Q2 (Don Ellis / The French Connection) was the single residual silence; hypothesized at the time to be index-cap-bound and expected to rescue under Phase 1.5.
- *Run 4* (full-bank tantivy, 5,698,239 cells): **PASS.** `hybrid_rerank_fallback` again 7g/1s/0w. All five modes preserved cardinal rule (0 wrong / 40 calls). Verdict deltas vs run 3 are noise within the cosine_rerank/hybrid_cosine_always modes; the production `hybrid_rerank_fallback` mode is identical to run 3 on per-query verdicts. Q2 stayed silent in all 5 modes with `kw_rank=None` everywhere — confirming the supporting cell genuinely isn't ingested in this bank (lexical BM25 over 5.7M cells can't find it either), so it is a corpus-coverage gap, not a retrieval-depth gap. See `results/v1_hybrid_cascade.md` for the verdict table.

Phase 1 of the V1 upgrade adds a BM25 → cosine → (optional) cross-encoder rerank cascade behind the silence gate. The cascade is **opt-in**: with no lexical index supplied, the pipeline is byte-identical to the V1 baseline above. The cascade is wired in two orchestration modes on `AnswerPipeline`:

- `hybrid_mode="fallback"` (default, production-safe): pass-1 runs the pure V1 cosine path; only if pass-1 silences does pass-2 run with hybrid+rerank enabled. Successful rescues attach `result.rescue = {"hybrid_rescue": True, "first_pass_silence_reason": ...}` for audit. Failed rescues return the pass-1 silence with `{"hybrid_rescue_attempted": True, "hybrid_rescue_outcome": "silence", "hybrid_silence_reason": ...}`.
- `hybrid_mode="always"` (research only): pass-1 runs hybrid+rerank for every query. Used by `exp28` to confirm the always-on path is strictly worse than fallback on this bank. Do not use in production.

**Cardinal constraints (must hold across all phases):**

- The silence gate is never replaced by the reranker.
- The verifier is never weakened for recall.
- No margin change ships without an eval harness proving it.
- All mutations go to the overlay, not the base bank.
- The V1 wrong-answer count never increases. A single wrong answer in `exp28` mode 4 halts the rollout.

### Build the BM25 index

Two backends are wired through the same CLI. Pick by what fits in RAM on the host:

| Backend       | When to use                                                                                              | Build flag                  |
|---------------|----------------------------------------------------------------------------------------------------------|-----------------------------|
| `rank_bm25`   | Default. Capped corpora (≤ ~200k cells). Pure Python; index lives in `index.pkl` + `cell_ids.npy`.       | `--backend rank_bm25`       |
| `tantivy`     | Full 5.7M-cell bank or any build where rank_bm25 OOMs. Rust, streams segments to disk; index in `tantivy/` subdir. | `--backend tantivy`         |

Both backends share the same Python tokenizer (`tokenize()`) and the same manifest schema (`corpus_hash`, `tokenizer_version`, `n_docs`, `k1`, `b`, plus a new `backend` field used for back-compat dispatch). The cascade itself (`HybridRetriever`) is backend-agnostic: it consumes the duck-typed `topk(query, k, *, excluded_ids)` surface.

```pwsh
# Phase 1 prove-out (rank_bm25, capped corpus, fast on a workstation).
python scripts/build_lexical_index.py `
    --bank-path H:\MiniLM\cc_service\bank.db `
    --out-dir results/v1_bank/bm25_index `
    --limit 100000

# Phase 1.5 full-bank build (tantivy, streams to disk).
python scripts/build_lexical_index.py `
    --bank-path H:\MiniLM\cc_service\bank.db `
    --out-dir results/v1_bank/bm25_tantivy `
    --backend tantivy `
    --writer-heap-mb 512
```

Loaders use the manifest's `backend` field to dispatch. `src.agent.lexical_index.load_lexical_index(directory)` is the public entry point and returns either `LexicalIndex` (rank_bm25) or `TantivyLexicalIndex`. Phase 1 indices on disk (no `backend` field) default to `rank_bm25` for back-compat. The full-bank rank_bm25 build was attempted in Phase 1 and held off — `rank_bm25.BM25Okapi` is pure Python and peaked at ~23 GB of working set without finishing on a 48 GB workstation. Tantivy 0.26 (pip-installable Rust wheel, no JVM) replaces it for the full bank; segment writes stream to disk inside a bounded writer heap (default 256 MB; bump to 512 MB for the full-bank build).

### Ask with the cascade on

```pwsh
python scripts/ask.py "Who composed The French Connection?" `
    --lexical-index results/v1_bank/bm25_index
```

Or set the environment variable once: `$env:MD_LEXICAL_INDEX = "results/v1_bank/bm25_index"` and call `ask.py` normally.

### Gating proof: exp28

Before promoting the cascade to default, run the 4-mode comparison:

```pwsh
python experiments/exp28_hybrid_cascade.py `
    --lexical-index results/v1_bank/bm25_index
```

Writes `results/v1_hybrid_cascade.json` and `results/v1_hybrid_cascade.md`. **Pass criterion:** mode `hybrid_rerank_fallback` must achieve ≥ 7/8 grounded and exactly 0/8 wrong. The script exits non-zero on failure; treat that as a HALT. The harness runs the Phi-3 decomposer once per query (mirroring `exp22` / `exp27`) so all five modes ask the pipeline on the same sub-question and only the retrieval mode varies. Run 3 on this codebase **passed** (7/1/0) — see `results/v1_hybrid_cascade.md` for the verdict table and run-history analysis. Pass `--no-decompose` to reproduce the original (defective) harness for audit; do not use it for evaluation.

### Recalibrating the gate margin

The default margin (0.015) was calibrated for cosine activations. The reranker uses a different score scale; `--rerank-margin` in `exp28` defaults to 0.5 (conservative for `ms-marco-MiniLM-L-6-v2`). Only change `DEFAULT_MARGIN_THRESHOLD` in `src/agent/v1_silence_gate.py` if exp28 proves drift across ≥ 2 of the 8 questions, and only after re-running the full eval suite to confirm no regression.

### Stale-index protection

When the underlying bank's source texts change, the lexical index's manifest hash no longer matches. `LexicalIndex.load(..., expected_corpus_hash=...)` raises `LexicalIndexStale` rather than returning silently bad rankings. Rebuild the index after bulk overlay edits or any base-bank rewrite.

## 12. Stage F claim verifier (off-by-default)

Stage F is a post-generation **cell-conjunction grounding check** that catches cross-cell splices Stage E v2 can let through (e.g. an answer that sentence-fuses a name from cell 1 with a fact from cell 2 and cites both). It is **off by default** in V1; enabling it is a strict-mode opt-in.

### Cardinal rule

Stage F MUST NOT change production behaviour unless the caller explicitly enables it. Every entry point gates Stage F behind a default-`False` flag:

- `v1_answer_verifier.verify(..., enable_claim_verification=False)` — kwarg.
- `AnswerPipeline(..., enable_claim_verification=False)` — constructor kwarg, stored on instance, forwarded to every internal `verify()` call (including the hybrid fallback rescue path).

When `enable_claim_verification=False`, `VerificationDecision.claim_verifier_report` is `{}` and the decision is byte-equivalent to pre-Phase-2 behaviour.

### What Stage F checks

For each atomic claim in the cleaned answer:

1. **Decompose**: split on sentence terminators `[.!?;]`, then on dashes and on `, and|but|while|whereas|however|though|although`. Drop spans shorter than 8 chars and drop refusal/self-silence boilerplate. Each `ClaimSpan` carries its original-answer offsets.
2. **Signal extraction**: numerics (digit runs, including comma-grouped like `658,000`) and capitalised proper-noun runs. Question numerics and question proper-noun anchors (substring match against question text, normalised) are **never** treated as novel — they are background, not new information.
3. **Skip rule**: if a claim has zero distinctive signals after filtering, it is **skipped** (counted in `n_skipped`, not in `n_verified` or `n_rejected`). This avoids false positives on glue sentences ("Here is what I found").
4. **Cell conjunction**: a claim is **verified** iff every one of its signals appears (as a normalised substring) in a **single** cited cell. The first satisfying cell wins. If no single cell carries all signals, the claim is **rejected** and Stage F overrides the verdict to silence.

Stage F only runs when Stages A/B/E have all passed; earlier rejections retain diagnostic priority.

### Enable for diagnostic runs

```python
from src.agent.answer_pipeline import AnswerPipeline
pipe = AnswerPipeline(..., enable_claim_verification=True)
result = pipe.ask("...")
report = result.verification.get("claim_verifier_report", {})
# report keys: grounded, n_claims, n_skipped, n_verified, n_rejected, failed, per_claim
```

### Gating proof: exp29

```pwsh
python experiments/exp29_claim_verifier.py `
    --lexical-index results/v1_bank/bm25_tantivy
```

Writes `results/v1_claim_verifier.{json,md}`. Runs the 8 exp28 multi-hop queries through the production `hybrid_rerank_fallback` cascade twice — once with Stage F off, once on — and asserts:

- **0 wrong** in both configs.
- **0 grounded→silence regressions** when Stage F flips on.

**Run 1 on this codebase passed**: 7g/1s/0w both configs, 0 regressions, 0 wrong→silence catches (the multihop set has no wrong baselines for Stage F to catch in this run — Stage E v2 already rejects them). Stage F is therefore safe to enable as a strict-mode option without disturbing the default cascade.

### When to enable

Turn Stage F on for: bank-promotion audits, regression sweeps after Stage E changes, and any deployment that values silence-over-splice more strongly than recall. Leave it off for: latency-sensitive interactive use (Stage F adds ~0–1s per call on grounded answers, more if the decomposer tokenizer is cold).

## 13. Governed memory mutations (`scripts/memory.py`)

Phase 3 replaces the `python -c "from src.agent...; ov.add_cell(...)"` one-liners in sections 4–6 with a typed **plan → confirm → apply** flow that refuses to mutate the overlay on an instruction it could not parse.

### Cardinal rule

The CLI mutates the overlay **only** when all three hold: `mode == apply`, `--confirm` is passed, and the planner produced a non-blocked `MemoryPlan`. `mode == plan` is read-only end-to-end and is the default.

### Recognised instructions (lexical, not LLM)

`src/agent/memory_intent_classifier.py` parses short imperatives with regexes. The closed kind set is `add | remove | inspect | unknown`. Unknown is *honest* — the orchestrator refuses to act on it rather than guessing.

| Instruction example | Kind | Fields parsed |
| --- | --- | --- |
| `remember that Mars has two moons` | `add` | `text="Mars has two moons"` |
| `add "Phobos is the larger moon"` | `add` | `text="Phobos is the larger moon"` |
| `tombstone 123456: wrong attribution` | `remove` | `cell_id=123456, reason="wrong attribution"` |
| `delete cell 42 because outdated source` | `remove` | `cell_id=42, reason="outdated source"` |
| `show provenance` | `inspect` | — |
| `the speed of light is 299792458 m/s` | `unknown` | — (refused) |

Loading Phi-3 to classify a six-word imperative is over-engineering; the lexical parser is deterministic, testable, and consistent with the Phase 2 Stage F decomposer.

### Plan (dry-run, default)

```pwsh
python scripts/memory.py plan "remember that Mars has two moons" `
    --overlay-path results/v1_bank/overlay.db `
    --bank-path H:\MiniLM\cc_service\bank.db
```

Exit 0 on a non-blocked plan, exit 2 if the planner blocked (unknown intent, empty reason, allocation guard tripped, etc.). Nothing is written.

### Apply (mutates the overlay)

```pwsh
# Add a cell — requires the base bank for allocation + encoder.
python scripts/memory.py apply "remember that Mars has two moons" `
    --overlay-path results/v1_bank/overlay.db `
    --bank-path H:\MiniLM\cc_service\bank.db `
    --source manual --label cell_mars_moons --confirm

# Tombstone — does NOT need the base bank.
python scripts/memory.py apply "tombstone 123456: wrong attribution" `
    --overlay-path results/v1_bank/overlay.db --confirm
```

Apply without `--confirm` is a deliberate no-op that returns exit 2, so a fat-fingered command never mutates state. Exit 3 indicates the apply itself failed or the post-mutation validator caught a regression.

### Post-mutation validator

After every apply, `src/agent/post_mutation_validator.py` reads the overlay back and confirms:

- **add**: the new `overlay_cells` row exists at the returned id, its `source_text` matches the intent verbatim, and a `provenance_log` `op=add` entry is present.
- **remove**: a `tombstones` row exists for `cell_id` with the intent's reason, plus the `op=remove` provenance entry.
- **inspect**: `provenance()` is callable.

The validator does **not** load the answer pipeline (no ~40 s Phi-3 cold load). A smoke ask through `scripts/ask.py` is the separate, optional next step.

### When to use

Use `scripts/memory.py` for any overlay edit performed by hand. The legacy `python -c` recipes in sections 4–6 remain for headless automation that needs to skip argparse, but every interactive operator action should go through the planner so an unparseable instruction can never silently mutate state.

## 14. Isotropy correction: ABTT alternative to ZCA whitening (opt-in at fit time)

The production bank was fitted with ZCA whitening. ZCA is mathematically fragile when the reference sample count `N` is small relative to the embedding dimension `D` — the covariance estimate is rank-deficient and the resulting `cov^(-1/2)` over-sharpens out-of-reference directions. This is the textbook small-sample whitening failure documented in `results/exp08_summary.json`: config `minilm_whiten_scale` produced `paraphrase_recall: 0.0` (vs `0.8` for raw embeddings).

Phase 6 ships an alternative isotropy corrector — **All-But-The-Top (ABTT)** from Mu & Viswanath (ICLR 2018) — selectable via a new `method=` parameter to `cc_service.memory.fit_whitening`. ABTT subtracts the global mean and projects out the top-k principal components without ever inverting the covariance, so it is robust to small `N`.

### Selecting at fit time

```python
from cc_service.memory import fit_whitening

# Legacy default — ZCA. Unchanged.
params = fit_whitening(raw_embeddings, reference_n=len(refs))

# Opt in to ABTT for the next bank build.
params = fit_whitening(raw_embeddings, reference_n=len(refs), method="abtt")

# Custom k for ABTT (default is max(1, D // 100), the paper's heuristic).
params = fit_whitening(raw_embeddings, reference_n=len(refs),
                       method="abtt", abtt_k=5)
```

The returned `WhiteningParams` has the same `(mu, w_matrix, max_norm)` shape regardless of method, so persistence (`save_whitening` / `load_whitening`) and `apply_whitening` work unchanged. The on-disk schema is unaffected — both `tests/cc_service/test_smoke.py` and `src/agent/sqlite_bank.py` continue to load either kind of fitted matrix.

### What this does NOT do

- The live 5.7M-cell bank at `H:\MiniLM\cc_service\bank.db` was built with ZCA. **It is not modified.** Swapping its `whitening` row in place would invalidate every stored cell's geometry. Adoption requires a deliberate full bank rebuild (re-encode + re-fit + re-calibrate thresholds), which is a Stage 2 / "expensive" decision and is not part of Phase 6.
- The geometry-layer `concept_cells.geometry.zca_whiten` is unchanged; a parallel `abtt_transform` is added so future experiments can A/B the two.

### Regression evidence

`evals/test_abtt_no_collapse.py` covers:

- `test_zca_collapses_paraphrase_separation_at_small_n` — reproduces the exp08 collapse on a synthetic anisotropic regime (N=200, D=384) and asserts the paraphrase / unrelated cosine gap goes below 0.05 (ZCA destroys it).
- `test_abtt_removes_top_k_variance_and_preserves_bulk` — structural contract: ABTT removes ~all variance in the top-k reference axes and preserves ≥ 90% of bulk variance.
- `test_abtt_robust_to_small_n_no_warnings` — ABTT at the textbook bad case (N=20, D=384) emits no warning and produces finite output.
- `test_fit_whitening_abtt_returns_compatible_params` — shape and persistence-compatibility of the ABTT `WhiteningParams`.
- `test_fit_whitening_zca_default_unchanged` — backward compat: `method='zca'` (the default) is byte-identical to the no-argument call.

Whether ABTT *recovers retrieval recall on real MiniLM* is an empirical question owned by `exp09` / `exp12`; that depends on where the paraphrase signal lives in real sentence embeddings (which is encoder-specific) and is out of scope for an isolated unit test. Section 14 ships the option; an evaluation run on a freshly rebuilt small-scale bank would be the natural next step before considering full-bank adoption.

