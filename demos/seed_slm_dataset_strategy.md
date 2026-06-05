# SLM Dataset and Training Strategy Notes

A governed reference document capturing the project's decisions about free
datasets and local fine-tuning for a small language model (SLM). It is
reference material imported through the knowledge front door so the Ask page can
retrieve and cite it; it is not a record of applied changes.

## Local fine-tuning scope and hardware

The strategy is to fine-tune an existing 0.5B to 3B instruct model locally with
LoRA or QLoRA on an RTX 3070 desktop with 8 GB of VRAM and 64 GB of system RAM.
Full pretraining from scratch is not realistic on this desktop and is out of
scope. The first run should use a curated dataset of 10,000 to 30,000 supervised
fine-tuning rows plus a separate 100 to 500 question evaluation set, because
5,000 strong examples can outperform hundreds of thousands of mediocre ones.

The RTX 3070 is comfortable with 135M to 1B models using LoRA at 512 to 2,048
token sequence lengths. It is practical but slower for 1.5B to 3B models using
4-bit QLoRA with gradient checkpointing and batch size one. A 7B model under
QLoRA is borderline. Desktop pretraining and full fine-tuning of a 7B model are
not sensible locally.

## Recommended first experiment

Start with the Smol-SmolTalk dataset for general assistant behaviour plus a
small number of the project's own governed consultant examples, and fine-tune a
0.5B to 1.7B model with QLoRA. Do not begin with FineWeb, millions of rows, or
pretraining from scratch.

Candidate models for the first experiment are SmolLM2 360M or 1.7B, Qwen2.5 0.5B
or 1.5B, Qwen3 0.6B or 1.7B subject to stack and licence review, or a similarly
sized Phi-family base or instruct model after checking its current model card
and licence.

A concrete starter recipe is 10,000 Smol-SmolTalk examples, 2,000 OpenAssistant
OASST1 examples, 1,000 Dolly examples, 1,000 custom governed consultant
examples, and 200 held-out evaluation examples. A larger balanced curriculum of
22,000 to 25,000 rows is SmolTalk 8,000, OASST1 4,000, Dolly 3,000, SmolTalk
specialised reasoning 3,000, Orca Math 2,000, and 2,000 to 5,000 of the
project's own governed consultant examples.

## Recommended datasets

Smol-SmolTalk (HuggingFaceTB/smol-smoltalk, Apache 2.0) is the recommended first
experiment dataset because it is adapted for sub-1B models with shorter chat
conversations. Start with 10,000 to 20,000 examples rather than the whole set.

SmolTalk (HuggingFaceTB/smoltalk) is the best general-purpose supervised
fine-tuning source with about one million synthetic samples. Select a balanced
curriculum of roughly 15,000 examples rather than importing everything. Its
weakness is that synthetic data carries repeated styles and systematic errors.

Databricks Dolly 15K (databricks/databricks-dolly-15k, CC BY-SA 3.0) is a good
human-generated simple baseline for proving the workflow, but it should not be
the final dataset; preserve attribution and review the share-alike terms.

OpenAssistant OASST1 (OpenAssistant/oasst1) provides human-written multi-turn
conversation trees. Extract 5,000 to 15,000 high-quality English conversation
paths as a supplement, not the only dataset.

UltraChat 200K (HuggingFaceH4/ultrachat_200k) is useful but too large for a first
run and risks a generic voice. Take a carefully filtered 10,000 to 25,000 row
subset rather than the first rows.

Microsoft Orca Math (microsoft/orca-math-word-problems-200k, MIT) is only for
mathematical-reasoning experiments. Cap it at 2,000 to 10,000 examples in a
mixed curriculum because heavy maths distorts the model toward overexplaining
ordinary questions.

FineWeb-Edu (HuggingFaceFW/fineweb-edu) is continued-pretraining material made of
documents, not instruction pairs, and is unsuitable for chat tuning. Use only a
very small streamed subset of 2 to 100 million tokens for experiments, never the
full corpus. FineWeb2 (HuggingFaceFW/fineweb-2, ODC-By 1.0) covers more than
1,000 languages but is still pretraining material rather than an instruction
dataset.

## Behaviour-first architecture decision

For Concept Memory Workbench the SLM should learn behaviour, not become the
truth store. Train it to classify user intent, select the right response
structure, write fluent transitions, obey evidence boundaries, label judgement,
refuse unsupported claims, format citations, and turn retrieved evidence into
consultant-style prose. Keep changing facts in retrieval and governed memory.

Evaluate the model on behaviour rather than knowledge: instruction-following
accuracy, correct response mode, citation-format compliance, unsupported-claim
rate, judgement-labelling accuracy, report-section completeness, verbosity
control, and desktop latency. The counterfactual is that if a good prompt plus
retrieval performs within roughly 5 to 10 percent of the fine-tuned model on the
target behaviours, local fine-tuning is not yet justified and effort should go
into prompts, retrieval, and dataset collection first.

A balanced training mix is Wikipedia-derived text 20 to 40 percent, general
instruction data 25 to 40 percent, the project's own governed examples 20 to 30
percent, specialist Microsoft 365 material 10 to 25 percent, and safety or
refusal examples 5 to 10 percent. Convert every source into one internal
messages format with system, user, and assistant turns plus identifier, source,
licence, quality score, and split fields rather than mixing datasets blindly.

## Wikipedia datasets

Include Wikipedia mainly for continued pretraining, retrieval, and generating
grounded training examples, not as a direct replacement for instruction-tuning
data. The preferred use order is Structured Wikipedia as a governed retrieval
source first, then Wikipedia passages to generate evidence-bound question and
answer examples, then a small subset for continued pretraining, and official
Wikimedia dumps only when production refresh and revision handling are needed.

The wikimedia/wikipedia dataset provides cleaned articles from a November 2023
snapshot of about 6.41 million English articles under CC BY-SA 3.0 and GFDL, but
it is stale and strips tables and maths, so it should be streamed rather than
downloaded in full. The wikimedia/structured-wikipedia dataset preserves
sections, paragraphs, links, references, and metadata in Parquet under CC BY-SA
4.0, and is the preferred Wikipedia source for the governed source library and
citation design. HuggingFaceFW/finewiki preserves headings, lists, code blocks,
tables, and mathematics for technical continued pretraining, but it is a derived
dataset whose extraction method, snapshot date, and licences must be inspected.
Official Wikimedia dumps give the latest snapshots and full parsing control with
revision provenance, but pages-meta-current is roughly 45 GB compressed and is
best reserved for later production refresh workflows.

When generating evidence-bound examples from Wikipedia, take a source passage,
generate a question, answer using only that passage, verify the answer span or
entailment, and reject unsupported examples. Preserve article revision, section,
source URL, and licence on every Wikipedia-derived record to satisfy CC BY-SA
attribution and share-alike obligations. For continued pretraining at desktop
scale, start with 10 to 50 million tokens for an initial test and 50 to 200
million tokens for a serious controlled experiment, using article chunks of 512
to 1,024 tokens and a 135M to 500M model initially, rather than all 6.4 million
English articles.
