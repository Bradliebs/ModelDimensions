# Concept Cells — Weekly Working Note

A short example note for the workbench's ingestion + approval queue. Importing it
produces *candidate* memories only; nothing is written until a human approves.

## Decisions

- Decision: keep the v1.0 concept-cell geometry frozen and build the workbench
  strictly on top of the existing retrieve -> verify -> ground path.
- Decision: approved memories are written through the same `add_memory` path so
  the bank and ledger ids stay aligned.

## Results

- Result: the full offline test suite passes (71 tests) after the v1.1
  workbench was added.
- Result: the Friday -> Monday near-miss is correctly rejected by the verifier,
  validated end to end in the Streamlit UI.

## Limitations

- Limitation: the v1.0 memory bank is in-memory only and cannot be restored from
  the saved ledger, so each session rebuilds the bank from scratch.
- Limitation: paraphrase recall is capped at 0.625; the optional NLI verifier is
  deferred.

## Next Steps

- Next: document the import workflow in the README so reviewers know approved is
  the only path to a written memory.
- Deferred: revisit the opt-in NLI verifier once recall expansion is scheduled.

## Near-Miss Watch

- The supplier delivery is on Monday afternoon, not Friday — a tempting but
  wrong restatement that must be reviewed before it is ever trusted as memory.
