# Concept Cells v1.0 — Explicit Candidate Memory with Verified Grounding

A research report on the ModelDimensions concept-cell memory system.

This document tells the full research arc, from the first memory mechanism to
the v1.0 release: what was attempted, what each milestone changed, what the
release claims, and — just as importantly — what it does not claim. It is a
research artifact, not a production readiness statement.

## Central thesis

Three claims, in order, carry the whole project:

- **Similarity is not truth.** Two sentences can sit close in embedding space
  and still state opposite facts.
- **Memory retrieval is not grounding.** Pulling a candidate out of memory is
  not the same as being entitled to answer from it.
- **Verification is the bridge.** A deterministic check between retrieval and
  grounding is what makes a retrieved candidate safe to use.

The simplest illustration is one stored fact and one near-miss query:

> Stored memory: *"the supplier delivery is on Friday afternoon"*
> Query: *"the supplier delivery is on Monday afternoon"*

The query differs by a single word. In embedding space it ranks *high* — higher
than many genuine paraphrases. A similarity threshold would retrieve it and
ground it, answering as if the delivery were on Monday. The v1.0 system instead
retrieves the candidate, runs the verifier, detects the weekday flip, returns
`REJECT`, and refuses to ground. The candidate is retrieved but never reaches
the user as memory. That single example is the entire project in miniature.

---

## 1. What was attempted

The goal was an inspectable memory for a small agent: write facts, recall them
on later queries, and answer *only* from what was actually stored — never
inventing a fact and never grounding a wrong one.

The hard part is not storage or recall. It is the decision in the middle:
*given a query and a retrieved memory, is it safe to answer from that memory?*
The project's arc is the story of getting that decision wrong with a similarity
threshold, then getting it right with a verification step. Each milestone below
added one layer and froze it before the next.

## 2. Concept Cells v0.7 — the mechanism

The substrate. Memories are written as **concept cells** on a whitened,
unit-ball geometry: each stored vector is ZCA-whitened, scaled into the unit
ball, and placed at a fixed radius. A query *fires* a cell when the unit cell
weight dotted with the radius-scaled query exceeds the cell threshold. "Silence"
means no cell fired.

This layer is **frozen**. The geometry, whitening, scaling, and write/query
construction did not change after v0.7 and were not touched in any later
milestone.

What v0.7 established: a deterministic, inspectable memory that fires on close
matches and stays silent otherwise.

## 3. SLM controller and grounding — v0.8

A small-language-model (SLM) controller was placed in front of the substrate to
decide *when* to consult memory and *how* to phrase a grounded answer. Crucially,
the controller never gained authority to invent facts. Grounding draws text
**only** from stored cells, and the response policy
([src/agent/response_policy.py](../src/agent/response_policy.py):
`ground_from_cells`, `enforce_grounding`, `build_response`) is **frozen** from
this point forward.

This is where "memory retrieval is not grounding" first becomes a code-level
boundary: the SLM may phrase an answer and claim which memories it used, but
`enforce_grounding` drops any claimed memory id that did not actually fire. The
language model cannot smuggle in a memory the geometry did not retrieve.

## 4. Stress guardrails — v0.9

Adversarial pressure was added: prompt-injection attempts, silent-refusal
handling, deleted-memory citation attempts, and a "`memory_used` requires a
citation" invariant. These hardened the boundary between *retrieving* a memory
and *grounding* an answer on it, and produced the guard tests the release still
runs. By the end of v0.9 the system could refuse cleanly — but its notion of "is
this query a match?" was still only firing, not fact-checking.

## 5. Semantic threshold failure — v1.0-rc1

rc1 attempted to widen recall with a **semantic threshold** over MiniLM
embeddings: accept a memory when its embedding similarity to the query clears a
tuned threshold. It failed, and the failure was the most instructive result in
the project.

A single-token factual near-miss — a weekday flip, a number flip, a location
swap — embeds *closer* to the stored sentence than a genuine paraphrase does.
Ranking by embedding similarity therefore places wrong-fact near-misses *above*
correct paraphrases. No single threshold can admit the paraphrases without also
admitting the near-misses.

The cost is measurable. On the exp10 corpus, the **threshold-only** system
records:

- false-accept rate **0.58**,
- unsupported-answer-after-grounding rate **0.29**,
- near-miss reject rate **0.0** (it rejects none of them).

See [results/exp10_summary.json](../results/exp10_summary.json),
`systems.threshold_only`. This is "similarity is not truth" stated as a number:
more than half of accepted near-misses were factually wrong. It is the reason
the release does **not** rely on semantic distance for grounding decisions. The
underlying calibration is recorded in exp09.

## 6. Candidate recall + verifier recovery — v1.0-rc2

rc2 replaced the single threshold decision with a **two-stage path**:

1. **Retrieve** the top-k candidates from the concept-cell substrate.
   Retrieval is explicitly *not* grounding — a candidate is a pointer, marked
   used by nothing.
2. **Verify** each candidate against the query with a deterministic fact
   verifier ([src/agent/verifier.py](../src/agent/verifier.py)) *before* any
   grounding.

The verifier is conservative by design. It returns `REJECT` on any material
mismatch (different entity, number, date or weekday, a negation flip, a known
antonym flip, or a modal obligation flip). It returns `ACCEPT` only for an exact
normalised match, strong token containment, or a curated safe-paraphrase
fixture. Everything else is `AMBIGUOUS`. Only `ACCEPT` may be grounded; both
`REJECT` and `AMBIGUOUS` lead to refusal.

The asymmetry is deliberate: a false `ACCEPT` grounds a wrong fact, the failure
we most want to avoid; a false `AMBIGUOUS` merely refuses a correct paraphrase,
which is safe.

**The recovery, measured.** On the same exp10 corpus, the
candidate-recall-plus-deterministic-verifier path (`systems.topk_deterministic`)
records:

| metric | threshold-only | + deterministic verifier |
| --- | --- | --- |
| candidate recall@k | 0.96 | **1.0** |
| near-miss reject rate | 0.0 | **1.0** |
| false-accept rate | 0.58 | **0.0** |
| unsupported-after-grounding | 0.29 | **0.0** |

The verifier rejects **every** near-miss that the threshold grounded, driving
both the false-accept rate and the unsupported-after-grounding rate to **0.0**.
That is the "verification is the bridge" claim as evidence: retrieval recall
went *up* to 1.0 while wrong-fact grounding went *down* to 0.0, because the two
jobs were separated.

An optional **SLM equivalence judge** sits behind the deterministic verifier,
bound by one hard rule: it can never override a deterministic `REJECT`. In the
offline environment it is a conservative stub and recovers no extra paraphrases.

## 7. v1.0-final — reproduction and demo

> **Boundary note.** Everything in sections 2–6 is part of the **frozen v1.0
> baseline**, captured at the annotated tag `v1.0-research-baseline`. The
> reproduction script and demo described in this section are **post-tag
> tooling**: they exercise and explain the baseline but add no architecture and
> change no result. They live outside the tagged commit.

Finalisation did not pivot the architecture. It widened the evidence, pinned the
guarantees, and made the system reproducible and demonstrable by a fresh reader.

**Evidence widened (part of the frozen baseline):**

- **Failure taxonomy — exp11.** The factual-flip taxonomy was enumerated into
  **15 distinct flip configurations** (weekday, month or date, integer, quantity
  word, currency amount, negation, generic antonym, approval status, safe/unsafe,
  increase/decrease, location swap, person or entity swap, before/after, modal
  must/may, and modal should-not). The deterministic verifier already covered 14;
  the single gap — a modal obligation flip such as "must" versus "may" — was
  closed by adding a conservative, symmetric `detect_modal_flip` detector. This
  addition is additive only: it does not touch the geometry, whitening,
  write/query construction, Oja binding, the grounding policy, or the
  REJECT-override rule. All 15 configurations reject at **1.0**, with
  false-accept and unsupported-after-grounding both at **0.0** and candidate
  recall@k at **1.0**. See
  [results/exp11_summary.json](../results/exp11_summary.json).
- **Paraphrase recovery — exp12.** Four configurations compared: deterministic
  only, deterministic plus fixture, deterministic plus SLM judge, and
  deterministic plus cross-encoder/NLI. Paraphrase recall rises monotonically
  (**0.5 → 0.625 → 0.625**) while safety stays at **1.0** throughout — recall is
  bought only by moving `AMBIGUOUS` to `ACCEPT` on proven-safe pairs, never by
  relaxing rejection. The NLI configuration is **deferred** (see below). See
  [results/exp12_summary.json](../results/exp12_summary.json).
- **Release invariant tests.**
  [evals/test_release_invariants.py](../evals/test_release_invariants.py) pins
  the six safety invariants the release depends on (candidate retrieval never
  grounds; a deterministic `REJECT` cannot be overridden; `AMBIGUOUS` is not
  grounded; only `ACCEPT` sets `memory_used`; a deleted memory cannot be cited;
  unsupported-after-grounding is blocked).

**Reproducible and demonstrable (post-tag tooling):**

- **One-command reproduction** — [scripts/validate_v1.ps1](../scripts/validate_v1.ps1)
  runs the full test suite, the exp01 smoke test, exp10/exp11/exp12, and the
  demo, then verifies the expected result artifacts exist. It is the fresh-clone
  proof that the system is intact.
- **Minimal demo** — [scripts/demo_v1.py](../scripts/demo_v1.py) is a fully
  offline, deterministic walkthrough of six actions (write, exact query,
  paraphrase query, dangerous near-miss query, verdict display, delete-then-query).
  It prints the audit trail — `candidate_retrieved`, `verifier_verdict`,
  `memory_used`, `refused` — for each query, and its centrepiece is the
  Friday → Monday near-miss: candidate retrieved, verifier `REJECT`, not grounded.

## 8. Claims, non-claims, and next research

### What v1.0 claims

On the tested corpora, the candidate-recall-plus-deterministic-verifier path:

- rejects **every** enumerated factual near-miss (15/15 flip configurations,
  reject rate 1.0);
- never grounds a `REJECT` or `AMBIGUOUS` candidate;
- never lets the optional SLM judge override a deterministic `REJECT`;
- holds false-accept rate and unsupported-after-grounding rate at **0.0** across
  every measured configuration.

### What v1.0 does not claim — and known limitations

- **No production claim.** This is a research prototype. It is not hardened,
  benchmarked at scale, or validated for deployment. The verifier's safety
  results hold on the specific corpora and flip classes tested; they are
  evidence, not a guarantee of behaviour on arbitrary input.
- **Paraphrase recall is partial — correct rephrasings may be refused.** A
  genuine paraphrase that is not an exact match, a strong containment, or a
  curated fixture is refused as `AMBIGUOUS`. This is a deliberate safety trade:
  the system prefers to refuse a correct paraphrase than to risk grounding a
  wrong fact. exp12 records paraphrase recall at 0.625, not 1.0.
- **The offline SLM judge adds no recall.** In this environment it is a
  conservative stub that confirms only exact matches. A real generative backend
  would be required to gain recall, and even then could not override a
  deterministic `REJECT`.
- **The NLI / cross-encoder path is deferred and opt-in.** No cross-encoder NLI
  model is available offline, and attempting to load one faults the native
  runtime. The wiring exists and obeys the same override rule, but no semantic
  recall is claimed from it. It is opt-in via the `EXP12_ENABLE_NLI=1`
  environment variable and is off by default.
- **Modal "may" collides with the month "May".** The modal detector is purely
  lexical, so a sentence using the month "May" alongside an obligation word could
  trigger a modal-flip rejection. This can only cause a *safe* false `REJECT` (a
  refused correct match), never a false `ACCEPT`.
- **The flip taxonomy is a sample, not a completeness proof.** exp11 measures
  rejection of 15 known flip classes. It does not establish that every possible
  factual edit is caught.
- **Corpora are small and English-only.** Results are on hand-built corpora of
  tens of sentences, not at scale and not across languages.

### Recommended next research

- Replace lexical modal and antonym detection with a learned entailment check
  that obeys the same "never override `REJECT`" rule, to recover paraphrase
  recall without relaxing safety.
- Evaluate a real cross-encoder/NLI backend on the exp12 corpus once a model can
  be loaded, measuring the recall it recovers against any safety cost.
- Disambiguate the modal "may" / month "May" collision with light
  part-of-speech context rather than a raw token match.
- Scale the corpora by one to two orders of magnitude and add non-English
  sentences to test whether the false-accept rate stays at 0.0.
- Stress the candidate-recall stage itself: measure recall@k as the memory bank
  grows, where loud near-misses may begin to crowd out true sources.
