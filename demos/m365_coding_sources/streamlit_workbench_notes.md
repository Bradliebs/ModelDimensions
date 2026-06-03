# Streamlit Workbench Notes

Source: assistant local field notes
Version: local-notes-v1
Domain: coding
Authority: reputable
Staleness: review_required (verify against the Streamlit docs before relying on it)

## Key facts

- Streamlit runs a Python script top to bottom on every interaction, so the whole
  script re-executes each time a widget changes.
- `st.session_state` persists values across reruns, which is how a workbench
  keeps state (loaded packs, query history) between interactions.
- `@st.cache_data` caches the return value of a function keyed on its inputs,
  avoiding repeated expensive work such as loading a model or a corpus.
- Widgets return their current value on each run; there is no callback-only model
  by default, though `on_change` callbacks are available.
- `streamlit run app.py` launches the local server; importing the script as a
  module (as tests do) must not trigger Streamlit UI calls at import time.

## Decisions and assumptions

- Decision: guard Streamlit UI calls so the same module can be imported headless
  by tests without a ScriptRunContext warning becoming an error.
- Assumption: heavy objects (encoder, knowledge library) are built once and kept
  in `st.session_state` rather than rebuilt on every rerun.
- Decision: long-running work is wrapped in `@st.cache_data` or
  `@st.cache_resource` so reruns stay responsive.

## Known limitations

- Because the script reruns fully, code with side effects at module scope runs
  repeatedly unless cached or guarded.
- `st.session_state` is per-session, not shared across users or browser tabs.
- The "missing ScriptRunContext" warning appears when the script is run outside
  `streamlit run` (for example under pytest); it is benign in that context.

## Examples

- Example: `if "service" not in st.session_state: st.session_state.service =
  build_service()` builds the workbench once and reuses it across reruns.
- Example: `@st.cache_resource def load_encoder(): ...` keeps the encoder in
  memory instead of reloading it on every widget change.

## Caution / near-miss

- Near-miss: a workbench rebuilt its knowledge library on every keystroke because
  the build call sat at module scope without caching, making the UI crawl.
  Moving it behind `st.session_state` restored responsiveness.
