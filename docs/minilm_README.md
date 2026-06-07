# Concept Cells Service (ccmem)

> **Historical document — frozen research-context narrative.** This is the
> original ccmem framing tying the concept-cell bank to the knowledge-free
> RETRO research arc. It does not track current state (V1 grounded-answer
> pipeline, governed knowledge-pack lifecycle, Consultant Workbench). For
> current operations see [RUNBOOK.md](RUNBOOK.md); for the frozen V1 baseline
> see [v1_release_report.md](v1_release_report.md).

A research-notebook memory service built on the Concept Cells architecture
(see the parent `concept_cells/ARCHITECTURE.md` for the underlying mechanism).

## Research context

ccmem is the deployable front end of a larger research project: building a
**knowledge-free** retrieval-augmented language model from neuroscience first
principles. The full write-up is in *The Reader Beats the Memoriser*
(`Knowledge_Free_RETRO_Paper.docx`); the model code lives under `nanogpt/`.

The chain of ideas:

- **Neuroscience.** The Tyukin–Gorban stochastic separation theorems (2018)
  prove that single neurons in high-dimensional spaces can memorise, group, and
  dynamically bind stimuli, with storage capacity growing exponentially in
  dimension. Concept cells are a direct implementation of that mechanism.
- **Concept-cell bank.** Each cell stores a unit-norm weight vector and a firing
  threshold; writes are one-shot (no gradient descent) and queries are a single
  matrix–vector product followed by a threshold comparison. ZCA whitening, fit
  once from a reference corpus, restores effective dimensionality from 24.5% to
  99.2% so the theorems hold on real MiniLM embeddings. On real text the bank
  shows 93.6% single-item selectivity, 100% rejection of novel content, Hebbian
  binding of up to 8 items, and spontaneous grouping of semantic paraphrases.
- **Deployment.** The bank was deployed as a FastAPI/SQLite service holding
  ~1.8M cells from Simple English and English Wikipedia, with sub-200ms query
  latency on a consumer GPU (RTX 3070). ccmem is the personal-notes version of
  that same service.
- **Knowledge-free RETRO.** The bank is the retrieval backend for a RETRO-style
  language model trained on a corpus with *zero overlap* with the bank, forcing
  it to learn reading rather than memorisation. At 404M parameters the model
  reproduces the pure-semantic retrieval benefit of a knowledge-trained baseline
  on a bank it never trained against (+0.167 vs +0.164 nats) and, in
  distribution, reaches lower loss with retrieval (3.78 nats) than a
  knowledge-trained 55M baseline (4.25 nats). Random neighbours actively harm it,
  confirming it genuinely reads retrieved content rather than using the pathway
  as spare capacity.

The takeaway driving this codebase: **knowing and reading are separable, and
reading scales.** Knowledge can live outside the model — transparent,
correctable, and instantly updatable — while the model supplies only the
comprehension. ccmem is where you edit that external knowledge directly.

## What this does

- **Add notes** (paragraphs of text) — each becomes a concept cell.
- **Query** with a question or topic — returns matching cells with confidence.
- **Bind** related notes into a single composite concept cell. The bound cell
  fires for any of the source notes.
- **Inspect, edit, and delete** cells. Memory is transparent and modifiable.

The architecture validated in `concept_cells/` supports M=2000+ cells and
binding up to m=8 items per bound cell on real text embeddings, with
near-perfect recall and near-zero false positives.

## Setup

```bash
pip install -r requirements.txt

# First-time init: fits whitening parameters from a reference corpus.
# This is a one-time step. Frozen parameters are stored in the bank file.
ccmem init --reference-corpus wikitext --reference-n 2000

# Add a note (becomes one concept cell)
ccmem add "The double-slit experiment shows that electrons exhibit wave-particle duality..."

# Add notes from a file (one paragraph per blank-line-separated section)
ccmem add --file my_notes.txt

# Query
ccmem query "what is wave-particle duality"

# Bind multiple cells into one concept
ccmem bind 3 7 12 --label "wave-particle duality fundamentals"

# List cells, inspect one
ccmem list
ccmem show 7
ccmem delete 7
```

## Architecture

```
┌──────────────┐    HTTP    ┌────────────────────────┐    ┌──────────┐
│  ccmem CLI   │ ─────────► │  FastAPI service       │ ─► │ SQLite   │
└──────────────┘            │  (memory.py, encoder)  │    │  bank.db │
                            └────────────────────────┘    └──────────┘
                                       │
                                       └─► MiniLM encoder (loaded once)
```

The service holds the encoder in memory; the SQLite file holds:
- Whitening parameters (frozen at init time)
- Per-cell `(weight_vector, threshold, label, metadata)` rows
- Per-bind history (which cells were composed into which)

## Files

| Path | Purpose |
|---|---|
| `service/main.py` | FastAPI entry point |
| `service/memory.py` | `MemoryBank`: ties encoder, preprocessing, bank, and SQLite |
| `service/encoder.py` | MiniLM wrapper, loaded once per process |
| `service/persistence.py` | SQLite schema + read/write |
| `service/schema.py` | Pydantic request/response models |
| `cli/ccmem.py` | Command-line client |
| `tests/test_api.py` | Smoke tests |

## Status

v0.1 — single-machine, single-bank, no auth, no concurrency control beyond
SQLite's. Intended for personal use against your own notes corpus.
