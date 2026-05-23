# Concept Cells: Architecture Specification

**Status:** Preliminary. Six experiments completed; mechanism demonstrated at small scale with characterised limits.
**Version:** 0.7 (matches code release `concept_cells_v0.7.tar.gz`)
**Last updated:** 2026-05-23

---

## TL;DR

A memory architecture in which each remembered item is stored by a single neuron-like cell, written in one shot without gradient descent, and queried by a single matmul-plus-threshold. Cells can be *bound* in groups via a Hebbian update rule, producing one cell that fires for any of the bound items.

On real semantic text embeddings (MiniLM 384d on Wikipedia paragraphs), the architecture demonstrates:

- **Static memory at 2000+ items**: writes correctly recognised; novel content rejected at >99% silence rate; near-paraphrases automatically absorbed into the same cell.
- **Hebbian binding up to m=8**: a single cell can be made to fire for any of up to 8 co-presented items, with 100% success and zero false positives at m≤5 and 92% success with 0.08 distractor-false-fires-per-trial at m=8.
- **All of the above without any training of the memory itself**: the encoder is frozen MiniLM; memory writes and bindings are closed-form constructions and dynamical-system updates respectively.

The architecture is not a replacement for transformers. It is a candidate **memory subsystem** for problems where transformers do badly: rapid knowledge acquisition, clean unlearning, explicit concept composition, and transparent rejection of out-of-distribution queries.

Validated scope: M ≤ 2000 cells, m ≤ 8 binding cardinality, single encoder (MiniLM), two corpora (Wikipedia, AG News). Anything beyond that is untested.

---

## Theoretical foundation

The architecture is based on the stochastic separation theorems of Tyukin, Gorban, Calvo, Makarova, and Makarov (2018), *High-Dimensional Brain: A Tool for Encoding and Rapid Learning of Memories by Single Neurons*. The relevant facts from that paper:

- **Theorem 1**: In high dimension, a random item drawn from the uniform distribution on the unit ball is, with high probability, linearly separable from any *fixed* set of other such items by a single hyperplane aligned with the item itself. Translation: one neuron with weights `w = x / ||x||` can selectively fire for item `x` and reject all others, with no training.

- **Theorem 2**: The same construction works for a *group* of related items: a single neuron whose weights align with the mean direction of the group selectively fires for all group members.

- **Theorem 3**: Group binding can be *learned* dynamically via Oja's rule. When two stimuli are co-presented and one is already "known" to the cell, the cell's weight vector rotates toward the mean of the two stimuli. After convergence, the cell fires for either.

The capacity grows exponentially in dimension. The mechanism does not require any iterative training of the cells themselves.

The mathematics assumes uniform-on-ball samples. Real semantic embeddings are not uniform-on-ball. The experiments below characterise the gap.

---

## Architecture

### Components

The architecture consists of three components plus a small set of policies:

```
┌─────────────────────────┐
│   Encoder (frozen)      │   Maps inputs → embedding space R^D
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Preprocessing layer   │   Whitening + ball scaling for isotropy
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Concept-cell bank     │   Each cell = (w_i ∈ R^D, θ_i ∈ R)
│                         │   Writes: O(1). Reads: O(N) matmul.
└─────────────────────────┘
             ▲
             │ optional binding step
             │
┌─────────────────────────┐
│   Oja-rule binding      │   Co-present multiple items, rotate w
└─────────────────────────┘
```

### Component 1: Encoder

Any encoder that maps inputs to a fixed-dimension embedding space. The architecture is encoder-agnostic in principle but has only been validated with one:

**Tested**: `sentence-transformers/all-MiniLM-L6-v2`, output dimension D=384.

The encoder is frozen. No backpropagation flows through it. The architecture treats the encoder as an opaque function `text → R^D`.

### Component 2: Preprocessing

Raw transformer embeddings are anisotropic — they cluster in narrow cones with average pairwise cosine similarity far from zero. The separation theorems require near-isotropic embeddings. We apply two transformations in sequence:

1. **ZCA whitening**: Decorrelates dimensions while staying close to the original embedding. Fitted on a reference corpus, then applied to all embeddings.
2. **Ball scaling**: Divides all vectors by `max(||x||)` so the most-distant point sits on the unit ball boundary; everything else is inside.

When the bank is used in a train/test setting (as in the rejection experiment), the whitening and scaling parameters are fitted on the train embeddings only and *applied* to the test embeddings, to avoid leakage.

**Effect**, measured on MiniLM/Wikipedia (Exp 02):
- Raw embedding pairwise cosine: mean |cos| = 0.083, effective dim = 94/384 (25%)
- After whitening + scaling: mean |cos| = 0.035, effective dim = 381/384 (99%)

### Component 3: Concept-cell bank

The bank stores `N` cells, each represented by a pair `(w_i, θ_i)`:
- `w_i ∈ R^D` is the cell's synaptic weight vector, lying on the unit sphere.
- `θ_i ∈ R` is the cell's firing threshold.

**Write** (one-shot, no training):

```python
def write(x):                # x is a preprocessed embedding
    w = x / ||x||            # unit-sphere alignment with the item
    θ = θ_default            # see "Parameters" below
    bank.append((w, θ))
```

**Read** (single matmul plus elementwise threshold):

```python
def query(q):                # q is a preprocessed query embedding
    activations = W @ q      # W is the stacked weight matrix (N, D)
    fires = activations > θ  # (N,) boolean: which cells fired
    return fires
```

Cost: O(N·D) per query, fully vectorisable. No iteration, no training.

### Component 4: Hebbian binding

Multiple items can be associated into a single cell via an Oja-rule update. Given an existing cell `(w_0, θ)` for which item `x_A` is "known" (fires above threshold), and additional items `x_B, x_C, ...` to bind:

1. Form the sum stimulus `s̄ = x_A + x_B + x_C + ...`
2. Run Euler steps of the gated Oja rule:

```python
def oja_step(w, s, θ, α, dt):
    y = w · s                           # membrane potential
    v = max(y - θ, 0)                   # threshold gate (no update if silent)
    if v == 0: return w
    dw = α · v · y · (s - w · y)        # Oja rule with gating
    return w + dt · dw
```

3. After convergence (~300 steps with α=1.0, dt=0.05), `w` aligns with the mean direction of the bound items.
4. Set a new firing threshold `θ_readout` via one of the policies below.

The cell now fires for each bound item alone, while rejecting unrelated items.

### Policies for readout threshold

After binding, the firing threshold must be set lower than the write-time threshold, because each individual bound item projects onto `w_final` with smaller magnitude than it would onto its own item-aligned cell.

Three policies were tested (Exp 05):

- **Fixed** (`θ = 0.30`): No adaptation. Works only at small m.
- **Theorem-3** (`θ = 0.9 · θ*(m)`): Uses the paper's analytical upper bound. Works at large m, fails at small m.
- **Calibrated** (`θ = min(w_final · x_i) − safety_margin`, taken over the bound items): Measures actual post-binding projections of bound items and sets θ just below their minimum. Dominates the other policies at every m.

**Default recommendation**: Calibrated, with `safety_margin = 0.02`.

---

## Parameters

| Parameter | Default | Justified by | Notes |
|---|---|---|---|
| Embedding dim D | 384 | MiniLM choice | Architecture tolerates D≥~100; D=384 well above the synthetic threshold of ~20 |
| Write threshold θ_write | 0.30 | Exp 03 | Below `min(||x||)` over the corpus after preprocessing; above zero by margin |
| Pre-binding initialization | `w = x/||x||` (unit sphere) | Exp 04 v0.6 | Critical: do NOT use `(θ+ε)·x/||x||` from the paper directly; fails for ||x||<1 |
| Oja learning rate α | 1.0 | Exp 04 | Higher values unstable; lower converge too slowly |
| Oja integration step dt | 0.05 | Exp 04 | Larger destabilises; smaller wastes compute |
| Oja steps to convergence | 300 | Exp 04 dynamics plot | Bound items above threshold by step 80; safety factor 4× |
| Readout policy | Calibrated | Exp 05 | Dominates fixed and theorem3 at every m tested |
| Calibration safety margin | 0.02 | Exp 05 default | Trades a tiny bit of recall margin for distractor rejection |

The biggest "gotcha" in the parameter list is the **unit-sphere initialization**. A literal reading of the paper (equation 13) suggests `w_0 = (θ + ε) · unit(x)`, which works fine for uniform-ball samples (where `||x|| ≈ 1`) but silently fails for real embeddings post-scaling (where most items have `||x|| < 1`). The cell never fires for its own anchor, the Oja gate stays shut, and binding does nothing. We hit this bug, diagnosed it, and the fix is the line "w = x / ||x||" with no further scaling. Future implementations should not rederive this construction without checking.

---

## Validated properties

Each property is paired with the experiment that established it and the numerical result.

### V1. One-shot writes give perfect self-recognition

Every item written into the bank correctly activates its own cell. (Trivially true by construction, but verified: across 1500 train items in Exp 03, 100% activated their own cell.)

### V2. Selectivity to single items, scaling exponentially with dimension

A cell built for item `x` does not fire for unrelated items, and capacity grows exponentially in D. Demonstrated on synthetic uniform-ball samples (Exp 01); on real embeddings (whitened MiniLM, M=2000), the selectivity rate is 93.6%, with all "failures" being semantic near-duplicates rather than spurious activations (see V3).

### V3. Automatic absorption of paraphrases (Theorem-2 behaviour)

On real semantic embeddings, the cell built for item `x` *also* fires for items that are paraphrases or near-duplicates of `x`. This is not a bug — it's the architecture exhibiting Theorem-2 group selectivity spontaneously. Verified on two corpora:

| Corpus | Confusions (whitened) | Cohen's d | Fraction above random p95 | Verdict |
|---|---|---|---|---|
| Wikipedia | 16 / 2000 items | +14.23 | 100% | SEMANTIC |
| AG News | 84 / 2000 items | +12.30 | 100% | SEMANTIC |

All confused pairs were inspected manually. Wikipedia confusions were structurally identical text (e.g. consecutive papal-conclave ballot tallies). AG News confusions were syndicated news stories with minor formatting differences. Zero noise confusions across both corpora.

### V4. Rejection of novel content

Items not in the bank do not activate any cell, with high reliability. Tested with strict train/test separation:

| Split method | Variant | Test silence rate | Median test margin |
|---|---|---|---|
| Paragraph-level | Whitened | 99.8% | −0.38 |
| Article-level (strict) | Whitened | 100.0% | −0.44 |

The "strict" article-level split (no Wikipedia article contributes to both train and test) produces *better* rejection than paragraph-level — confirming the rejection is genuine, not an artifact of split-level leakage.

### V5. Hebbian binding on real embeddings (the novel claim)

Multiple uncorrelated items can be bound into a single cell via Oja's rule. Results on real MiniLM embeddings with the calibrated readout policy:

| m | Success rate | False fires per trial (out of 200 distractors) | Alignment to mean direction |
|---|---|---|---|
| 2 | 100% | 0.00 | 1.000 |
| 3 | 100% | 0.00 | 1.000 |
| 5 | 100% | 0.00 | 1.000 |
| 8 | 92% | 0.08 | 1.000 |

Alignment to the mean direction is 1.000 at every m: Oja's rule converges to the geometrically correct position regardless of m. The 8% degradation at m=8 is a readout-margin effect (bound items spread out around the mean as m grows), not a learning failure.

---

## What was NOT tested

The architecture has clear, characterised limits. The following are explicitly outside what we have evidence for:

- **Scale beyond M=2000 cells.** The paper predicts capacity scales exponentially in D, and we have not seen a failure mode yet, but we also have not measured at M=10,000 or higher. Recommended next test before committing to a production deployment.
- **Binding cardinality beyond m=8.** The dynamics in Exp 05 suggest m=10–15 might still work with the calibrated policy, but this is extrapolation. The Theorem-3 prediction is that θ* eventually decays below the distractor floor, at which point binding becomes impossible.
- **Other encoders.** All experiments used MiniLM (384d). Larger encoders (768d, 1024d) should perform better; smaller ones likely fail. The architecture's geometric properties depend on the encoder's geometric properties.
- **Other content domains.** Tested on Wikipedia paragraphs and AG News headlines. Code, dialogue, scientific abstracts, mathematical notation, and other modalities are untested.
- **Adversarial inputs.** Inputs deliberately crafted to confuse the architecture (e.g. minimally-perturbed strings that produce near-bound-item embeddings) have not been tested. This is a known gap for any memory system based on encoder geometry.
- **Temporal dynamics.** Items written, queried, and re-queried over long horizons (with intervening writes) have not been tested. The bank as currently described has no decay or eviction policy.
- **Multi-bind interference.** Binding two separate groups into two separate cells (each containing different items) and then querying for items from each group has not been tested as a system, only as isolated bindings.

These are not "future work" hand-waves. Each is a specific experiment that could change the architectural picture.

---

## When to use this architecture

The architecture earns its place in problems where the following properties matter:

- **Rapid acquisition**: writes are O(1). Memorising a new fact takes one matmul.
- **Clean unlearning**: deleting a cell removes the memory exactly. No "knowledge spread across parameters" problem.
- **Explicit composition**: binding produces a single readable cell with a known set of bound items.
- **Out-of-distribution rejection**: novel inputs produce silence, not confabulation. The "I don't know" answer is built in.
- **Inspectability**: the bank is a list of (vector, threshold) pairs. Each memory is locatable and modifiable.

The architecture does *not* earn its place when:

- **You need generalisation, not memorisation.** Concept cells are precise. They do not interpolate or extrapolate the way distributed representations do.
- **The encoder is the bottleneck.** No memory architecture is better than its encoder. If the encoder cannot produce embeddings where semantically distinct items are geometrically distinct, no memory built on top can fix that.
- **You need millions of cells.** We have not tested past 2000. Production deployment at millions of cells is speculation.

---

## Experimental record

Six experiments were run, in order. Each addressed a specific falsifiable question.

| # | Question | Result |
|---|---|---|
| 01 | Do separation theorems hold on real text embeddings? | Yes with whitening; raw embeddings show only 25% effective dim, whitened reach 99% |
| 02 | Are the residual "selectivity failures" noise or semantic near-duplicates? | Semantic. Cohen's d = +13 on AG News; zero noise confusions in 254 confused pairs |
| 02b | Does the semantic-grouping behaviour generalise beyond syndicated news? | Yes. Wikipedia shows Cohen's d = +14.23, even cleaner than AG News |
| 03 | Does the architecture correctly reject novel content? | Yes. 99.8% silence on held-out queries, with clean margin separation |
| 03b | Was the rejection result inflated by paragraph-level split leakage? | No. Article-level (strict) split gives 100% silence and *better* margins |
| 04 | Does Hebbian binding work on real embeddings? | Yes, after fixing an initialization bug. Works cleanly at m≤3 with fixed θ |
| 05 | Does adaptive θ extend binding to higher m? | Yes. Calibrated policy gives 100% success up to m=5, 92% at m=8, with negligible false-fires |

All experiments used 50 trials minimum, 384d embeddings, MiniLM as encoder, and whitening + ball-scaling preprocessing. Random seeds are fixed and the code is in `experiments/exp01_*.py` through `experiments/exp05_*.py`. Each experiment generates JSON summaries and PNG plots in `results/`.

The full experimental story, including the diagnostic work that uncovered the initialization bug between Exp 04 v0.5 and v0.6, is in the chat transcript that produced this work.

---

## Implementation notes

The reference implementation is in `src/concept_cells/`:

- `encoders.py` — `TextEncoder` (sentence-transformers wrapper), `UniformBallEncoder` and `RandomGaussianEncoder` (synthetic controls).
- `geometry.py` — `zca_whiten`, `scale_to_unit_ball`, `build_concept_cells`, `separation_test`, `isotropy_report`.
- `binding.py` — `initialize_cell_for_anchor`, `oja_step`, `bind_items`, `test_binding`, `theoretical_theta_star`.
- `data.py` — Corpus loaders with explicit (no-fallback) named sources.

Total core code: roughly 650 lines, no fancy dependencies, runs in 5–10 minutes on a single RTX 3070 for the full Exp 05 sweep.

The code is research-quality, not production-quality. Most notable shortcuts:

- The bank uses a dense `(N, D)` numpy matrix. For large N, this should be a `torch.Tensor` on GPU.
- There is no persistence layer. Writes are in-memory only.
- There is no concurrency control. Concurrent writes will not corrupt anything (append-only) but concurrent binds on the same cell will race.
- The encoder is loaded each time. Production would load once and serve.

---

## Path forward (suggested next steps, in priority order)

1. **Scale-out test**: rerun Exp 05 at M=10,000 cells and m=10. Confirm or refute the assumption that the working ceiling extrapolates. Half a day of work, would substantially strengthen any future claim.

2. **Persistence + service wrapper**: FastAPI service exposing `write`, `query`, `bind`. Single-encoder, single-bank, no scaling concerns yet. Justifies whether the architecture is *practically* useful, not just theoretically working.

3. **Adversarial probe**: deliberately construct near-bound-item embeddings (small perturbations to bound items) and measure whether the cell still discriminates. Tests whether the rejection property survives anything other than i.i.d. natural inputs.

4. **Multi-bind interference**: write two cells with disjoint bound-item sets in the same bank, query each. Measure interference. Tests whether bindings are independent or whether their weights interact in ways the per-cell analysis missed.

5. **Cross-encoder validation**: rerun Exp 02–05 with at least one other encoder (a 768d model). Confirm or weaken the encoder-agnosticism claim.

Items 1–3 are each less than a day's work. Items 4–5 are larger. None are urgent — the architecture as described in this document stands on its current evidence.

---

## Acknowledgement of limits

This document describes preliminary work. The architecture mechanism is demonstrated; the architecture's behaviour at production scale is not. The numbers reported here are repeatable (random seeds are fixed in the code) but represent a single configuration of (encoder, corpus, preprocessing, parameters). They should not be cited as evidence that this architecture will work in other configurations until those configurations have been tested.

The work was developed iteratively with the assistance of Claude, including the diagnostic work that uncovered the initialization bug in Exp 04. The architecture as documented here is the result of six successful experiments and one corrected bug. Every numerical claim points to a specific experiment and a specific output file. Reproducibility is the point.
