# Binding through the CCA path (exp31)

- Checkpoint: `H:\MiniLM\nanogpt\out-retro-bank\ckpt_best.pt` (55.3M params, iter=13000)
- m=6 bound concepts per cell; 6 probes
- Threshold: extraction ratio >= 0.5 -> PASS

## Overall NLL per answer token (nats)

| Condition  | NLL/token |
| ---------- | --------: |
| none       | 7.435 |
| single     | 6.685 |
| bound (6)  | 7.081 |
| distractor | 7.454 |

- single benefit over none       : **+0.750**
- single benefit over distractor : **+0.770**
- bound benefit over distractor  : **+0.373**
- **extraction ratio = 0.485**

## Verdict: **FAIL**

CCA extracts only 48.5% of per-fact signal from a bound m=6 cell (< 50%).

## Per-probe

| Fact | Answer | n_tok | none | single | bound | distractor | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mordwin capital: Zorath | `Zorath` | 3 | 4.979 | 4.062 | 4.432 | 5.229 | 0.683 |
| Sapphire lake: Tellaris | `Tellaris` | 2 | 9.875 | 10.156 | 9.969 | 10.500 | 1.545 |
| Druid sacred forest: Riftwood | `Riftwood` | 2 | 9.031 | 8.828 | 9.234 | 9.172 | -0.182 |
| Telnar minted coins: Brennix | `Brennix` | 3 | 6.891 | 5.760 | 6.922 | 6.682 | -0.260 |
| Ninety-meter watchtower: Korvath | `Korvath` | 3 | 7.354 | 6.844 | 6.995 | 7.656 | 0.814 |
| Ferroglass forging town: Daelin | `Daelin` | 3 | 7.823 | 6.328 | 6.615 | 7.073 | 0.615 |

## Caveats and interpretation

- **Verdict is at the threshold edge** (0.485 vs 0.500). The overall ratio
  masks high per-probe variance: 3 probes show clean extraction (0.62-0.81),
  2 probes show that bound actively HURTS vs distractor (-0.18, -0.26), and
  1 probe (Tellaris) has near-zero single-vs-distractor signal making its
  ratio uninterpretable.
- **Position-in-bundle effect**: Facts at bundle positions 0, 4, 5 (Zorath,
  Korvath, Daelin) extracted cleanly. Facts at positions 2, 3 (Riftwood,
  Brennix) failed -- the bound cell was *worse than the distractor*. This is
  consistent with attention-pattern bias toward earlier and later items;
  middle-positioned facts in a 6-fact cell appear to be effectively masked.
  This is a property of THIS 55M model's learned CCA, not necessarily of
  CCA in general.
- **Confounders ruled out in this version**: bundle truncation (asserted
  no overflow), distractor lexical contamination (zero answer-word overlap
  with FACTS_B), and cell structural density (single = 1 target fact + 5
  neutral statements, same 6-item structure as bound).
- **What CCA does extract**: when extraction works, bound recovers
  60-80% of the per-fact signal that a single-fact cell delivers. So CCA
  *can* bind in this model; it just fails on 2/6 of these probes.
- **Scope**: this measures the 55M debug model trained on the held-out
  bank, not the 404M roadmap target. A FAIL here does not predict a FAIL
  at scale; a PASS here would have been a stronger signal than the FAIL is.
- **No statistical significance test was run** -- N=6 probes is too small
  for a meaningful CI. The verdict reflects sample means only.

