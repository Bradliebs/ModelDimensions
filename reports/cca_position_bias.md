# exp32 -- CCA position-bias diagnostic

- Checkpoint: `H:\MiniLM\nanogpt\out-retro-bank\ckpt_best.pt` (55.29M params)
- 6 facts x 6 positions = 36 probes
- Block 256, chunk 64, neighbor_len 64, n_neighbors 2

## Verdict

**NO_POSITION_BIAS**

no clear position effect: end-vs-middle gap = +0.133 (< 0.2). ends mean=0.691, middles mean=0.558. exp31's per-probe variation is likely fact-specific rather than position-driven.

## Overall (averaged over all 36 probes)

- nll/token none        = 7.435
- nll/token single      = 6.889
- nll/token bound       = 7.078
- nll/token distractor  = 7.454
- overall extraction_ratio = (dist - bound) / (dist - single) = +0.665
- single_benefit_vs_none = +0.546

## Mean extraction ratio by position

(mean over the 6 different target facts placed at each position; 1.0 = full single-fact extraction, 0.0 = no better than distractor, < 0 = bound actively hurts vs distractor)

| position | mean ratio | std | n |
|---:|---:|---:|---:|
| 0 | +0.696 | 0.600 | 6 |
| 1 | +0.631 | 0.627 | 6 |
| 2 | +0.560 | 0.720 | 6 |
| 3 | +0.556 | 0.938 | 6 |
| 4 | +0.680 | 0.959 | 6 |
| 5 | +0.686 | 0.934 | 6 |

## Per-probe detail

| fact | answer | position | nll(single) | nll(bound) | nll(distractor) | ratio |
|---|---|---:|---:|---:|---:|---:|
| 0 | Zorath | 0 | 4.062 | 4.432 | 5.229 | +0.683 |
| 0 | Zorath | 1 | 4.214 | 4.510 | 5.229 | +0.708 |
| 0 | Zorath | 2 | 4.333 | 4.656 | 5.229 | +0.640 |
| 0 | Zorath | 3 | 4.396 | 4.755 | 5.229 | +0.569 |
| 0 | Zorath | 4 | 4.396 | 4.760 | 5.229 | +0.563 |
| 0 | Zorath | 5 | 4.526 | 4.755 | 5.229 | +0.674 |
| 1 | Tellaris | 0 | 10.156 | 9.938 | 10.500 | +1.636 |
| 1 | Tellaris | 1 | 10.094 | 9.969 | 10.500 | +1.308 |
| 1 | Tellaris | 2 | 10.094 | 9.969 | 10.500 | +1.308 |
| 1 | Tellaris | 3 | 10.094 | 9.938 | 10.500 | +1.385 |
| 1 | Tellaris | 4 | 10.094 | 9.938 | 10.500 | +1.385 |
| 1 | Tellaris | 5 | 10.094 | 9.906 | 10.500 | +1.462 |
| 2 | Riftwood | 0 | 8.828 | 9.203 | 9.172 | -0.091 |
| 2 | Riftwood | 1 | 8.922 | 9.219 | 9.172 | -0.188 |
| 2 | Riftwood | 2 | 8.938 | 9.234 | 9.172 | -0.267 |
| 2 | Riftwood | 3 | 8.984 | 9.297 | 9.172 | -0.667 |
| 2 | Riftwood | 4 | 8.969 | 9.234 | 9.172 | -0.308 |
| 2 | Riftwood | 5 | 9.016 | 9.203 | 9.172 | -0.200 |
| 3 | Brennix | 0 | 5.760 | 6.484 | 6.682 | +0.215 |
| 3 | Brennix | 1 | 5.990 | 6.745 | 6.682 | -0.090 |
| 3 | Brennix | 2 | 6.057 | 6.911 | 6.682 | -0.367 |
| 3 | Brennix | 3 | 6.214 | 6.922 | 6.682 | -0.511 |
| 3 | Brennix | 4 | 6.260 | 6.901 | 6.682 | -0.519 |
| 3 | Brennix | 5 | 6.260 | 6.953 | 6.682 | -0.642 |
| 4 | Korvath | 0 | 6.844 | 6.906 | 7.656 | +0.923 |
| 4 | Korvath | 1 | 6.938 | 6.932 | 7.656 | +1.007 |
| 4 | Korvath | 2 | 7.094 | 6.984 | 7.656 | +1.194 |
| 4 | Korvath | 3 | 7.193 | 7.010 | 7.656 | +1.393 |
| 4 | Korvath | 4 | 7.312 | 6.995 | 7.656 | +1.924 |
| 4 | Korvath | 5 | 7.266 | 6.995 | 7.656 | +1.693 |
| 5 | Daelin | 0 | 6.328 | 6.469 | 7.073 | +0.811 |
| 5 | Daelin | 1 | 6.479 | 6.453 | 7.073 | +1.044 |
| 5 | Daelin | 2 | 6.443 | 6.536 | 7.073 | +0.851 |
| 5 | Daelin | 3 | 6.609 | 6.531 | 7.073 | +1.169 |
| 5 | Daelin | 4 | 6.609 | 6.594 | 7.073 | +1.034 |
| 5 | Daelin | 5 | 6.667 | 6.615 | 7.073 | +1.128 |

## Interpretation and relationship to exp31

The strict position-bias hypothesis (end positions extract, middle positions
fail) is not confirmed. Per-position mean ratios are:

- pos 0: +0.696
- pos 1: +0.631
- pos 2: +0.560
- pos 3: +0.556
- pos 4: +0.680
- pos 5: +0.686

There is a small monotonic dip toward the middle (~0.13 nats end-vs-middle)
but it is dwarfed by the per-position standard deviation (0.60-0.96, driven
entirely by fact-to-fact variation). exp31's per-probe variation was
*fact*-specific, not *position*-specific.

The more important finding: this experiment's overall extraction ratio is
**+0.665 at N=36**, materially different from exp31's **+0.485 at N=6**.

Why the change: exp31's "single" reference always placed the target fact at
position 0 of the neighbor cell, while "bound" placed it at its natural
position (0..5 depending on which fact). That asymmetry inflated
`(nll_distractor - nll_single)` (the denominator of the ratio) and
deflated the ratio. exp32's "single" reference places the target fact at
the *same* position as "bound", isolating the binding interference cost
from the position cost. With the matched reference, m=6 extraction passes
the 0.5 gate at +0.665.

This **does not retract exp31** -- that measurement is what it is at its
own controls. It does show exp31's headline was driven by a reference-side
asymmetry rather than a genuine extraction failure.

## Fact-specific patterns (per-fact behaviour, not per-position)

| fact | answer    | ratio range over 6 positions | notes |
|---|---|---|---|
| 0 | Zorath    | +0.56 .. +0.71 | consistent mid extraction |
| 1 | Tellaris  | +1.31 .. +1.64 | very high; small absolute signal (single 10.09, dist 10.50) |
| 2 | Riftwood  | -0.67 .. -0.09 | **fails at every position** -- probe/fact issue, not position |
| 3 | Brennix   | -0.64 .. +0.22 | monotonic decline with position (only fact that shows position effect) |
| 4 | Korvath   | +0.92 .. +1.92 | strong over-extraction (>1.0); bound beats single |
| 5 | Daelin    | +0.81 .. +1.17 | strong extraction |

Two facts (Riftwood, Brennix at later positions) actively hurt; four extract
cleanly. The Riftwood failure at *every* position rules out position as the
explanation -- likely the probe prefix ("The druids honor the sacred forest
known as") competes with strong base-distribution priors for forest-name
continuations.

## Caveats

- Same 55M debug ckpt as exp31; not the 404M roadmap target.
- N=6 facts is still small; per-position means each summarise only 6
  numbers and per-position std is large.
- Synthetic fictional facts; results may not transfer to natural-language
  bank cells retrieved at inference time.
- The reversal of exp31's verdict says nothing about whether CCA binding
  extraction scales -- it only fixes an exp31 measurement artifact.
