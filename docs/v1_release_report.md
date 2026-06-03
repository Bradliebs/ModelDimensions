# ModelDimensions v1.0 Release Report

This report records what the project built, what each milestone changed, what
the final release claims, and — just as importantly — what it does not claim.
It is a research artifact. It is not a production readiness statement.

## v0.7 — concept-cell mechanism

The substrate. Memories are written as concept cells on a whitened, unit-ball
geometry: each stored vector is ZCA-whitened, scaled into the unit ball, and
placed at a fixed radius. A query fires a cell when the unit cell weight dotted
with the radius-scaled query exceeds the cell threshold. This layer is frozen:
the geometry, whitening, scaling, and write/query construction did not change
after v0.7 and were not touched in any later milestone.

What v0.7 established: a deterministic, inspectable memory that fires on close
matches and stays silent otherwise.

## v0.8 — SLM controller

A small-language-model controller was placed in front of the substrate to decide
when to consult memory and how to phrase a grounded answer. The controller never
gained authority to invent facts: grounding draws text only from stored cells,
and the response policy (`ground_from_cells`, `enforce_grounding`,
`build_response`) is frozen from this point forward.

## v0.9 — stress guardrails

Adversarial pressure was added: prompt injection, silent-refusal handling,
deleted-memory citation attempts, and "memory_used requires a citation"
invariants. These hardened the boundary between *retrieving* a memory and
*grounding* an answer on it, and produced the guard tests the release still runs.

## v1.0-rc1 — semantic threshold failure

rc1 attempted to widen recall with a semantic threshold over MiniLM embeddings.
It failed, and the failure was instructive: a single-token factual near-miss
(for example a weekday or location flip) embeds *closer* to the stored sentence
than a genuine paraphrase does. Ranking by embedding similarity therefore places
wrong-fact near-misses above correct paraphrases. A similarity threshold cannot
separate "same fact, different words" from "different fact, similar words." This
result is recorded in the exp09 calibration, and it is the reason the release
does not rely on semantic distance for grounding decisions.

## v1.0-rc2 — candidate verifier recovery

rc2 replaced the threshold decision with a two-stage path: retrieve top-k
candidates, then verify each candidate against the query with a deterministic
fact verifier before any grounding. The verifier is conservative by design —
it accepts only exact matches, strong token containment, or curated
safe-paraphrase fixtures, and it refuses everything else as AMBIGUOUS. A false
ACCEPT grounds a wrong fact, which is the failure we most want to avoid; a false
AMBIGUOUS merely refuses a correct paraphrase, which is safe.

Measured on the exp10 corpus, this moved the threshold-only false-accept rate of
0.58 to 0.0 and the unsupported-answer-after-grounding rate of 0.29 to 0.0, while
rejecting every near-miss. An optional SLM equivalence judge sits behind the
verifier and is bound by one hard rule: it can never override a deterministic
REJECT.

## v1.0-finalisation — scale and release hardening

Finalisation did not pivot the architecture. It widened the evidence and pinned
the guarantees.

- **Experiment-configuration checklist resolved (15/15).** The factual-flip
  taxonomy was enumerated into 15 distinct flip configurations (weekday, month
  or date, integer, quantity word, currency amount, negation, generic antonym,
  approval status, safe/unsafe, increase/decrease, location swap, person or
  entity swap, before/after, modal must/may, and modal should-not). The
  deterministic verifier already covered 14. The single gap — a modal obligation
  flip such as "must" versus "may" — was closed by adding a conservative,
  symmetric `detect_modal_flip` detector. This is additive: it does not touch the
  geometry, whitening, write/query construction, Oja binding, the grounding
  policy, or the REJECT-override rule.
- **exp11 — failure taxonomy.** All 15 configurations reject at 1.0, with
  false-accept and unsupported-after-grounding both at 0.0, and candidate
  recall@k at 1.0. See [results/exp11_summary.json](../results/exp11_summary.json).
- **exp12 — paraphrase recovery.** Four configurations compared: deterministic
  only, deterministic plus fixture, deterministic plus SLM judge, and
  deterministic plus cross-encoder/NLI. Paraphrase recall rises monotonically
  (0.5, then 0.625, then 0.625) while safety stays at 1.0 throughout. See
  [results/exp12_summary.json](../results/exp12_summary.json).
- **Release invariant tests.** [evals/test_release_invariants.py](../evals/test_release_invariants.py)
  pins the six safety invariants the release depends on.

## Current limitations

- **Paraphrase recall is partial.** Genuine paraphrases that are not exact
  matches, strong containments, or curated fixtures are refused as AMBIGUOUS.
  This is a deliberate safety trade, but it means correct rephrasings are
  sometimes turned away.
- **The offline SLM judge adds no recall.** In this environment the SLM judge is
  a conservative offline stub; it confirms only exact matches and recovers no
  extra paraphrases. A real generative backend would be required to gain recall,
  and even then it could not override a deterministic REJECT.
- **The NLI / cross-encoder configuration is deferred.** No cross-encoder NLI
  model is available offline, and attempting to load one faults the native
  runtime. The wiring exists and obeys the same override rule, but no semantic
  recall is claimed from it. It is opt-in via `EXP12_ENABLE_NLI=1`.
- **Modal "may" collides with the month "May".** The modal detector is purely
  lexical, so a sentence using the month "May" alongside an obligation word could
  trigger a modal-flip rejection. This can only cause a *safe* false REJECT
  (a refused correct match), never a false ACCEPT.
- **The flip taxonomy is a sample, not a proof of completeness.** exp11 measures
  rejection of 15 known flip classes. It does not establish that every possible
  factual edit is caught.
- **Corpora are small and English-only.** Results are on hand-built corpora of
  tens of sentences, not at scale and not across languages.

## No production claim

This is a research prototype. It is not hardened, benchmarked at scale, or
validated for deployment, and nothing here should be read as a production
readiness statement. The verifier's safety results hold on the specific corpora
and flip classes tested; they are evidence, not a guarantee of behaviour on
arbitrary input.

What is claimed: on the tested corpora, the candidate-recall-plus-deterministic-
verifier path rejects every enumerated factual near-miss, never grounds a
rejected or ambiguous candidate, and never lets the optional SLM judge override a
deterministic REJECT — at a false-accept and unsupported-after-grounding rate of
0.0.

What is not claimed: completeness against all factual edits, full paraphrase
recall, robustness at scale, multilingual behaviour, or production safety.

## Recommended next research

- Replace lexical modal and antonym detection with a learned entailment check
  that obeys the same "never override REJECT" rule, to recover paraphrase recall
  without relaxing safety.
- Evaluate a real cross-encoder/NLI backend on the exp12 corpus once a model can
  be loaded, measuring the recall it recovers against any safety cost.
- Disambiguate the modal "may" / month "May" collision with light part-of-speech
  context rather than a raw token match.
- Scale the corpora by one to two orders of magnitude and add non-English
  sentences to test whether the false-accept rate stays at 0.0.
- Stress the candidate-recall stage itself: measure recall@k as the memory bank
  grows, where loud near-misses may begin to crowd out true sources.
