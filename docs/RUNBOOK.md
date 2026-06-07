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

## 11. Hybrid retrieval cascade (research-only, FAILED prove-out)

**Status as of the first exp28 run: FAILED. Do not enable in production.** See `results/v1_hybrid_cascade.md` for the verdict table and the failure analysis. `hybrid_rerank` produced 1 wrong answer (Q4 → "Vancouver" instead of "Ottawa") and `hybrid_cosine` produced 2 wrong answers (Q2 hallucinated composer, Q4 Vancouver). Wrong-answer count must never increase, so the cascade is gated off pending a re-run with the decomposer wired into the harness and a likely architectural fix (a dense-similarity floor on BM25-promoted candidates so that surface-token matches without semantic alignment are filtered before the silence gate).

Phase 1 of the V1 upgrade adds a BM25 → cosine → (optional) cross-encoder rerank cascade in front of the silence gate. The cascade is **opt-in**: with no lexical index supplied, the pipeline is byte-identical to the V1 baseline above. The cascade was designed to recover recall faults like Q2 (Section 9) without weakening the gate or the verifier; the first prove-out showed it does not yet meet that bar.

**Cardinal constraints (must hold across all phases):**

- The silence gate is never replaced by the reranker.
- The verifier is never weakened for recall.
- No margin change ships without an eval harness proving it.
- All mutations go to the overlay, not the base bank.
- The V1 wrong-answer count never increases. A single wrong answer in `exp28` mode 4 halts the rollout.

### Build the BM25 index

The Phase 1 prove-out uses a capped corpus to validate the cascade end-to-end before scaling to the full 5.7M-cell bank:

```pwsh
# Phase 1: prove the cascade on the 8-question eval set first.
python scripts/build_lexical_index.py `
    --bank-path H:\MiniLM\cc_service\bank.db `
    --out-dir results/v1_bank/bm25_index `
    --limit 100000
```

Drop `--limit` once the prove-out passes to index the full bank. The index lives in `results/v1_bank/bm25_index/` and contains `index.pkl` and `manifest.json` (corpus hash + tokenizer version for staleness checks). The full 5.7M-cell build was attempted and held off — `rank_bm25.BM25Okapi` is pure Python and peaked at ~23 GB of working set without finishing on a 48 GB workstation; the 100k subset finishes in seconds. Phase 1.5 needs an alternative BM25 implementation (`pyserini`, `tantivy-py`, or a custom `numpy`/`scipy.sparse` index) before scaling.

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

Writes `results/v1_hybrid_cascade.json` and `results/v1_hybrid_cascade.md`. **Pass criterion:** mode `hybrid_rerank` must achieve ≥ 7/8 grounded and exactly 0/8 wrong. The script exits non-zero on failure; treat that as a HALT. The first run on this codebase **failed** — see `results/v1_hybrid_cascade.md` for verdict table and failure analysis. The harness in its current form does not run the decomposer that the V1 documented baseline depends on; before drawing conclusions about the cascade, exp28 needs a decomposer pass added (mirror the `_capture` pattern in `experiments/exp27_rerank_diagnostic.py`).

### Recalibrating the gate margin

The default margin (0.015) was calibrated for cosine activations. The reranker uses a different score scale; `--rerank-margin` in `exp28` defaults to 0.5 (conservative for `ms-marco-MiniLM-L-6-v2`). Only change `DEFAULT_MARGIN_THRESHOLD` in `src/agent/v1_silence_gate.py` if exp28 proves drift across ≥ 2 of the 8 questions, and only after re-running the full eval suite to confirm no regression.

### Stale-index protection

When the underlying bank's source texts change, the lexical index's manifest hash no longer matches. `LexicalIndex.load(..., expected_corpus_hash=...)` raises `LexicalIndexStale` rather than returning silently bad rankings. Rebuild the index after bulk overlay edits or any base-bank rewrite.
