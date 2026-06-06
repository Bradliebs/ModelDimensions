# V1 Runbook

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

## 3. Add a cell to the overlay

```pwsh
python -c "from pathlib import Path; from src.cc_service.encoder import EncoderSingleton; from src.agent.streaming_bank import StreamingBank; from src.agent.bank_admin import OverlayStore, add_cell_from_text; bank = StreamingBank(r'H:\MiniLM\cc_service\bank.db'); enc = EncoderSingleton(model_name=bank.encoder_model); ov = OverlayStore(Path('results/v1_bank/overlay.db')); cid, _ = add_cell_from_text(ov, bank_dim=bank.dim, base_max_id=int(bank.cell_ids.max()), encoder=enc, whiten_fn=bank.whiten, text='Mars has two moons, Phobos and Deimos.', source='manual', label='cell_mars_moons'); print('added cell_id=', cid); ov.close(); bank.close()"
```

The new cell is encoded with the same encoder and whitening as the base bank (this is the invariant the rescue floor `0.40` is calibrated against). To exercise the overlay end-to-end:

```pwsh
python scripts/ask.py "What are Mars' moons?" --overlay-path results/v1_bank/overlay.db
```

Or set `MD_OVERLAY_PATH` in your environment so every `ask.py` call merges the overlay automatically.

## 4. Remove a cell

```pwsh
python -c "from pathlib import Path; from src.agent.bank_admin import OverlayStore; ov = OverlayStore(Path('results/v1_bank/overlay.db')); ov.remove_cell(123456, reason='wrong attribution'); ov.close(); print('tombstoned 123456')"
```

Tombstones a base or overlay cell id. The merged bank suppresses the cell in `topk` from the next process restart onward. Provenance is preserved — the row in `overlay_cells` is not deleted, and `provenance_log` records the removal with reason and timestamp.

## 5. Inspect provenance

```pwsh
python -c "from pathlib import Path; from src.agent.bank_admin import OverlayStore; ov = OverlayStore(Path('results/v1_bank/overlay.db')); [print(r) for r in ov.provenance()]; ov.close()"
```

Lists every `add` and `remove` op against the overlay, oldest first, with timestamps and reasons.

## 6. Re-run validations after changes

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

V1 baseline: 144 tests pass.

## 7. Recovery

The base bank is read-only; nothing the V1 pipeline does can corrupt it. To revert all overlay edits to a clean state:

```pwsh
Remove-Item results/v1_bank/overlay.db
```

The next pipeline construction proceeds without an overlay. To revert a single edit, tombstone its `cell_id` (Section 4) — provenance is preserved.

## Where things live

| What                          | Where                                     |
| ----------------------------- | ----------------------------------------- |
| Production bank (read-only)   | `H:\MiniLM\cc_service\bank.db`            |
| Editable overlay              | `results/v1_bank/overlay.db`              |
| Setup verdict                 | `results/setup_check.json`                |
| Rescue calibration            | `results/v1_rescue_verification.json`     |
| Pipeline                      | `src/agent/answer_pipeline.py`            |
| Bank loader                   | `src/agent/streaming_bank.py`             |
| Overlay admin API             | `src/agent/bank_admin.py`                 |
| CLI                           | `scripts/ask.py`, `scripts/setup.py`      |
| Tests                         | `evals/`                                  |
