# Concept Cells Engineering Notes

A small, controlled reference document used to demonstrate knowledge import.
It is external reference material, not a record of project decisions.

## Pydantic model validation

Pydantic validates data against a typed model when the model is constructed.
A field declared with a type is coerced and checked, and a ValidationError is
raised when the input does not satisfy the declared type or constraints.

Use a Pydantic model at a system boundary so untrusted input is validated once,
then trusted everywhere downstream.

## JSONL persistence

JSONL stores one JSON object per line, which makes append-only logs and ledgers
easy to read by eye and to stream line by line.

The workbench persists its memory ledger, proposal queue, and knowledge library
as JSONL files, flushing the whole file on each mutation so the on-disk state
always matches memory.

## Streamlit optional UI

Streamlit renders a Python script as a local web app when launched with
`streamlit run`. The workbench detects whether it is running under Streamlit and
falls back to a plain command-line REPL when Streamlit is not installed.

Keep the UI optional so the core tool has no hard dependency on a web framework.

## PowerShell launcher example

A PowerShell launcher can activate the virtual environment and start the app in
one step. For example, `& .\.venv\Scripts\python.exe app\workbench.py` runs the
workbench REPL using the project virtual environment on Windows.
