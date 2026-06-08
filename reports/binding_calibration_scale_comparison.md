# Binding calibration: sample-size scaling check

## Why this exists

The earlier binding diagnostics were run on 40k cells sampled from the 5.7M-cell
production bank (`H:\MiniLM\cc_service\bank.db`, ~0.7% coverage). With 5,000
distractors drawn per trial from a 20k eval pool, each trial covered 25% of the
available embeddings, so the 500 trials were heavily overlapping draws from a
narrow slice of the true bank distribution.

To check whether the m=8 FAIL verdict was an artefact of that sampling, the
pipeline was re-run on a 200k-cell sample (100k fit + 100k eval). At that scale
each trial's 5k distractor pool covers 5% of the eval embeddings rather than
25%, giving substantially more between-trial variance and a closer approximation
of the production distractor distribution.

## Inputs

- Bank: `H:\MiniLM\cc_service\bank.db` (5,698,239 cells, dim=384,
  encoder=all-MiniLM-L6-v2)
- Sample (200k): `scripts/sample_bank_embeddings.py --n 200000 --seed 0`
- Whitening: ZCA fit on the 100k fit slice, applied to the 100k eval slice,
  re-unit-normalised (`scripts/whiten_embeddings.py --unit-norm`)
- Binding: `scripts/calibrate_binding_at_scale.py --trials 500 --distractors
  5000` for m in {4, 6, 8}

## Geometry comparison

| Metric                          | 20k fit | 100k fit | Direction        |
|---------------------------------|--------:|---------:|------------------|
| Raw effective-dim fraction      |   0.630 |    0.636 | ≈ unchanged      |
| Whitened effective-dim fraction |   0.955 |    0.985 | closer to 1.0    |
| Fit whitened eigval max         |   1.000 |    1.000 | identity (OK)    |
| Eval whitened eigval max        |   1.479 |    1.192 | tighter           |

The eval-set max eigenvalue is the honest held-out check on the ZCA matrix:
1.479 (20k fit) shrinking to 1.192 (100k fit) means the ZCA matrix learned from
5× more samples generalises noticeably better to unseen embeddings. The
whitened effective-dim gate (≥0.95) was already passing at 20k but tightens
from 0.955 → 0.985 at 100k.

## Binding comparison

|  m | sample | success rate | false-fires / trial | verdict |
|---:|:------:|-------------:|--------------------:|:-------:|
|  4 | 20k    |        0.990 |               0.014 | PASS    |
|  4 | 100k   |        0.996 |               0.006 | PASS    |
|  6 | 20k    |        0.970 |               0.044 | PASS    |
|  6 | 100k   |        0.974 |               0.036 | PASS    |
|  8 | 20k    |        0.896 |               0.182 | FAIL    |
|  8 | 100k   |        0.906 |               0.148 | FAIL    |

Targets: success ≥ 0.92, false-fires ≤ 0.10/trial. 500 trials each.

### Numeric trace (direction check)

For m=8 at 100k:

```
success rate = 0.906   target = 0.92    diff = -0.014   → FAIL  (correct)
false-fires  = 0.148   target = 0.10    diff = +0.048   → FAIL  (correct)
```

### 95% confidence intervals on success

Wilson normal approximation, 500 trials:

| m=8 sample | point | 95% CI         |
|:-----------|------:|:---------------|
| 20k        | 0.896 | [0.869, 0.923] |
| 100k       | 0.906 | [0.880, 0.932] |

CIs overlap heavily. The 100k point is inside the 20k CI. The two runs are
statistically consistent — the m=8 FAIL is not a sample-size artefact.

## Conclusions

1. **m=4 and m=6 verdicts are robust.** Numbers shift in the third decimal and
   both gates still pass comfortably.

2. **m=8 FAIL survives at 5× scale.** Both gates fail by similar margins at 20k
   and 100k. False-fires drop from 0.182 → 0.148 (per-distractor rate
   ~3.6e-5 → ~3.0e-5), and success improves from 0.896 → 0.906, but neither
   clears its target. With 500 trials the CIs overlap; the m=8 ceiling at
   `safety_margin=0.02` is real, not a sampling artefact.

3. **Whitening generalises better with more fit data.** Eval-set max eigval
   improved from 1.479 → 1.192 and whitened effective-dim from 0.955 → 0.985.
   This is the honest reason the m=8 numbers improved slightly: a better ZCA
   matrix, not a different binding regime.

4. **The methodology lesson.** Future binding sweeps that probe the gate
   boundary should use the 200k sample as the default. The 20k sample is fine
   for quick smoke tests but slightly pessimistic at the m=8 boundary because
   the ZCA matrix is itself estimated noisily from 20k samples.

## Files

- `results/bank_fit_emb_200k.npy` (gitignored, ~146 MB)
- `results/bank_eval_emb_200k.npy` (gitignored, ~146 MB)
- `results/bank_eval_emb_zca_200k.npy` (gitignored, ~146 MB)
- `reports/zca_verification_200k.{json,md}`
- `reports/binding_calibration_zca_200k_n500.{json,md}`

Reproduce with seed=0; all scripts deterministic on CPU.
