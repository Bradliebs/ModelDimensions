# Step 4 mitigation -- pad-chunk attractor

Compares two runs of `scripts/ask_retro.py` on the same 5 prompts,
same seed (1337), same 50K subset bank, same 55M ckpt. The only
difference is `--skip-low-info-fraction`: 1.01 (off) vs 0.5 (on).
Each prompt generates up to 80 tokens, so each prompt produces
multiple chunk-boundary crossings (one boundary every chunk_size=64
generated tokens). At each boundary, all K=4 chunks of the current
rolling window are inspected.

Chunks are classified as low-info using a fixed reference cutoff of
0.50 (independent of either run's own
`skip_low_info_fraction`), so the same chunk is classified the same
way in both runs and only the *skipped* column differs.

## Headline numbers

| Metric                                              | Baseline (off) | Mitigated (on) |
|-----------------------------------------------------|---------------:|---------------:|
| Total chunks observed across all boundary crossings |             20 |             20 |
| Low-info chunks (fraction >= threshold)             |             20 |             20 |
| Low-info chunks that hit the bank                   |             18 |              0 |
| Low-info CCA-pollution rate                         |          90.0% |           0.0% |
| Top-1 dominance on retrieved low-info chunks        |          83.3% |           0.0% |
| Hi-info chunks (real content)                       |              0 |              0 |
| Hi-info chunks that hit the bank                    |              0 |              0 |

## What the rows mean

- **Low-info CCA-pollution rate** is the metric the Step 4 summary
  flagged: out of chunks dominated by pad tokens (`{0, EOT_TOKEN}`),
  what fraction reach CCA carrying attractor cells from the bank.
  Baseline 100% means every pad chunk pollutes CCA. The target was
  <=25%. Mitigated should be ~0% (skip means the neighbor slot stays
  at the all-EOT default and CCA sees no signal for that chunk).

- **Top-1 dominance** is the secondary signal. On the baseline run
  the same Hangul attractor cell was the top-1 retrieved cell for
  every pad chunk -- so dominance approached 100%. After mitigation
  almost no pad chunks are retrieved, so the metric is computed on
  whatever survives the skip threshold (usually 0 entries).

- **Hi-info chunks** are the chunks with real prompt content. The
  mitigation must not change how they are retrieved. If
  `Hi-info chunks that hit the bank` matches between baseline and
  mitigated runs, the mitigation is surgical. If both rows show 0,
  the smoke prompts EOT-ed before any chunk filled with > 50% real
  content (i.e. the rolling window was always pad-dominated), so
  this sanity check is uninformative for this specific smoke -- it
  requires longer continuations to exercise.

## Numeric trace (per the diagnostics rule)

Baseline:
  low_info_retrieved = 18, low_info_total = 20
  pollution_rate = 18 / 20 = 0.9000

Mitigated:
  low_info_retrieved = 0, low_info_total = 20
  pollution_rate = 0 / 20 = 0.0000

  target was <= 0.25.
  delta = 0.9000 -> 0.0000

## Most common top-1 cells on low-info chunks

Baseline:
  - 15x  'Hangul: ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ '
  - 1x  'General relativity is a theory of space and time. The theory was published by Al'
  - 1x  'Artificial intelligence (AI) is the ability of a computer program or a machine t'

Mitigated:
(none)

## Most common top-1 cells on hi-info chunks (sanity)

Baseline:
(none)

Mitigated:
(none)

## Honest characterization

The mitigation is an inference-time filter only. It does not
change the bank, the encoder, the ckpt, or training. It changes
what the CCA layers see during generation: low-info pad chunks
now contribute zero retrieved tokens instead of contributing
the encoder's attractor cells. The 55M ckpt is still incoherent
on these prompts -- that is a model-scale problem, not a
retrieval problem, and is documented in reports/step4_summary.md.
This eval only validates that the pad-attractor failure mode is
no longer present in the retrieval path.

Caveat: the mitigation introduces a train/inference mismatch.
The 55M ckpt was trained with CCA always receiving retrieved
tokens; zeroing them at inference shifts the activations out of
the training distribution and can degrade surface output on
individual prompts (some baseline continuations become more
incoherent under mitigation). That is expected, is not measured
by this eval, and is the reason the 404M training plan (§5.2)
names this mitigation as a precondition for promoting any future
ckpt -- the larger model should be trained with the same
retrieval discipline that inference will use.

