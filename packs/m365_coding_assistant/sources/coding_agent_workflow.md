# Local Coding-Agent Workflow Rules

Source: assistant local field notes
Version: local-notes-v1
Domain: coding
Authority: reputable
Staleness: review_required (project-specific conventions; confirm against the repo before relying on them)

## Key facts

- The coding agent works in small, reviewable steps: read the relevant files
  before editing, make the minimum change that satisfies the request, and verify
  with the test suite before moving on.
- Every change must trace to the request; unrelated refactors, formatting, and
  "while I'm here" cleanups are surfaced separately rather than folded into the
  diff silently.
- The standard verification loop is: run the targeted tests, run the full
  `pytest evals -q` suite, then run the smoke check before reporting done.
- When evidence is missing the agent refuses rather than inventing an answer,
  and any answer drawn from a model prior instead of retrieved evidence is
  explicitly labelled as such.
- Commits use a conventional subject line, the work is pushed only after tests
  pass, and frozen core semantics are never changed without explicit approval.

## Decisions and assumptions

- Decision: the next step after writing code is always to run the verification
  loop, not to start the next feature.
- Assumption: the deterministic, offline backend is the default for tests so the
  suite never depends on a network model.

## Known limitations

- These workflow rules are project conventions, not universal practice, and may
  not apply to other repositories.
- The agent cannot guarantee correctness beyond what the tests assert, so weak
  success criteria produce weak verification.

## Targeted recall claims (v2.5A)

The safest next prompt to give the coding agent before it edits this repo: tell the agent to read the relevant files first, make the minimum change, and run the verification loop, so the safest next prompt scopes the edit narrowly before it edits the repo.

## Project milestone recall (v2.5A)

The v2.2 milestone added the hybrid retrieval backend to this repo: v2.2 is the milestone that introduced hybrid retrieval as a selectable retrieval backend alongside the default deterministic backend, so the retrieval backend the v2.2 milestone added to the repo is the hybrid retrieval backend.
