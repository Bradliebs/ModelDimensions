"""Local Bank Management Workspace.

Run:
    python app/bank_workspace.py --skip-generator

Then open http://127.0.0.1:8765. The page is intentionally plain HTML and
JSON: no React, no accounts, no remote service assumptions.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Depends, FastAPI, HTTPException, Request, status  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.agent.bank_workspace import (  # noqa: E402
    DEFAULT_BANK,
    DEFAULT_OVERLAY,
    WorkspaceError,
    WorkspaceSession,
    build_generator_from_env,
    build_ingestion_draft,
    candidates_from_payload,
    commit_candidate_cells,
    extract_source,
    extract_source_from_base64,
    history,
    search_cells,
    tombstone_cell,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class PreviewRequest(BaseModel):
    source_name: str = Field(min_length=1)
    text: str = ""
    data_base64: str = ""


class CommitRequest(BaseModel):
    source_name: str = Field(min_length=1)
    candidates: list[dict[str, Any]]
    smoke_question: str = ""
    pages_processed: int = 0
    extracted_char_count: int | None = None
    warnings: list[str] = []


class TombstoneRequest(BaseModel):
    cell_id: int
    reason: str = Field(min_length=1)


@dataclass
class SecurityConfig:
    """Opt-in hardening for non-personal deployments.

    Every field defaults to "off" so the local single-operator tool keeps its
    original open behaviour unless a deployment explicitly enables a control.
    """

    api_token: str | None = None
    require_auth_for_reads: bool = False
    max_body_bytes: int = 0
    rate_limit_per_minute: int = 0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SecurityConfig":
        env = env if env is not None else dict(os.environ)
        token = env.get("WORKSPACE_API_TOKEN") or None

        def _truthy(value: str | None) -> bool:
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        def _int(value: str | None) -> int:
            try:
                return max(0, int(str(value).strip()))
            except (TypeError, ValueError):
                return 0

        max_mb = _int(env.get("WORKSPACE_MAX_BODY_MB"))
        return cls(
            api_token=token,
            require_auth_for_reads=_truthy(env.get("WORKSPACE_REQUIRE_AUTH_READS")),
            max_body_bytes=max_mb * 1024 * 1024,
            rate_limit_per_minute=_int(env.get("WORKSPACE_RATE_LIMIT_PER_MIN")),
        )


class _RateLimiter:
    """Thread-safe fixed-window per-client limiter. No-op when limit <= 0."""

    def __init__(self, per_minute: int) -> None:
        self._per_minute = per_minute
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, client: str) -> bool:
        if self._per_minute <= 0:
            return True
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            bucket = self._hits.setdefault(client, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._per_minute:
                return False
            bucket.append(now)
            return True

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bank Management Workspace</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f7f7f4; color: #1f2933; }
    header { background: #17324d; color: white; padding: 16px 24px; }
    main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
    section { background: white; border: 1px solid #d7dce0; border-radius: 6px; padding: 16px; }
    h1 { margin: 0; font-size: 22px; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    textarea, input { width: 100%; box-sizing: border-box; margin: 6px 0 10px; padding: 8px; border: 1px solid #b8c0c8; border-radius: 4px; }
    textarea { min-height: 90px; }
    button { background: #1f6feb; color: white; border: 0; border-radius: 4px; padding: 8px 12px; cursor: pointer; margin-right: 6px; }
    button:disabled { background: #8c99a6; cursor: wait; }
    button.secondary { background: #51606f; }
    button.danger { background: #b42318; }
    pre { white-space: pre-wrap; background: #f1f4f7; padding: 10px; border-radius: 4px; max-height: 280px; overflow: auto; }
    .full { grid-column: 1 / -1; }
    .muted { color: #536471; }
    .answer-card { background: #f8fafc; border: 1px solid #d7dce0; border-radius: 6px; padding: 10px; margin-top: 10px; }
    .answer-title { font-weight: 600; margin-bottom: 6px; }
    .answer-text { font-size: 15px; line-height: 1.55; white-space: pre-wrap; }
    .answer-meta { color: #536471; font-size: 13px; margin-top: 8px; }
    .cell { border: 1px solid #e1e5e8; border-radius: 4px; padding: 8px; margin: 8px 0; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } .full { grid-column: auto; } }
  </style>
</head>
<body>
  <header><h1>Bank Management Workspace</h1><div id="status" class="muted">Loading status...</div></header>
  <main>
    <section>
      <h2>Ask</h2>
      <textarea id="question" placeholder="Ask a question"></textarea>
      <button id="askButton" onclick="ask()">Ask</button><button id="reloadButton" class="secondary" onclick="reloadBank()">Reload bank</button>
      <pre id="reloadStatus"></pre>
      <div id="answer" class="answer-card muted">Ask a question to get a grounded answer.</div>
      <details><summary>Advanced details</summary><pre id="advanced"></pre></details>
    </section>
    <section>
      <h2>Add Text Or File</h2>
      <input id="sourceName" placeholder="Source name, e.g. notes.md">
      <textarea id="sourceText" placeholder="Paste text here, or choose a .txt, .md, or PDF file"></textarea>
      <input id="fileInput" type="file" accept=".txt,.md,.pdf">
      <button onclick="preview()">Preview cells</button><button onclick="commitCells()">Add approved cells</button>
      <pre id="ingestReport"></pre>
    </section>
    <section class="full">
      <h2>Candidate Cells</h2>
      <div id="candidates" class="muted">No preview yet.</div>
    </section>
    <section>
      <h2>Search Existing Cells</h2>
      <input id="searchQuery" placeholder="Search text or label">
      <button onclick="searchCells()">Search</button>
      <pre id="searchResults"></pre>
      <input id="tombstoneId" placeholder="Cell ID to tombstone">
      <input id="tombstoneReason" placeholder="Reason required">
      <button class="danger" onclick="tombstone()">Tombstone</button>
    </section>
    <section>
      <h2>Change History</h2>
      <button class="secondary" onclick="loadHistory()">Refresh history</button>
      <button class="secondary" onclick="loadQuestionHistory()">Refresh Q&amp;A</button>
      <pre id="history"></pre>
    </section>
  </main>
<script>
let currentCandidates = [];
let currentDraft = null;
let reloadPoll = null;
async function api(path, body) {
  const opts = body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.error || JSON.stringify(data));
  return data;
}
function setBusy(buttonId, busy, label) {
  const button = document.getElementById(buttonId);
  if (!button) return;
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = label;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
  }
}
function showError(targetId, err) {
  document.getElementById(targetId).textContent = `Error: ${err.message || err}`;
}
function escapeHtml(value) {
  return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}
function formatNumber(value) {
  return value === null || value === undefined ? '?' : Number(value).toLocaleString();
}
function formatEta(seconds) {
  if (seconds === null || seconds === undefined) return '?';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.round(seconds / 60)}m`;
}
function renderStatus(s) {
  const reload = s.reload || {state: 'idle'};
  document.getElementById('status').textContent = `Bank loaded: ${s.pipeline_loaded}; overlay loaded: ${s.overlay_loaded}; reload required: ${s.reload_required}; overlay cells: ${s.overlay.cell_count}; tombstones: ${s.overlay.tombstone_count}`;
  if (reload.state === 'running') {
    document.getElementById('reloadButton').disabled = true;
    document.getElementById('reloadButton').textContent = 'Reloading...';
    document.getElementById('reloadStatus').textContent = `Reloading: ${reload.stage}; ${formatNumber(reload.loaded)} / ${formatNumber(reload.total)} cells; ${reload.percent ?? '?'}%; eta ${formatEta(reload.eta_seconds)}`;
    startReloadPoll();
  } else {
    document.getElementById('reloadButton').disabled = false;
    document.getElementById('reloadButton').textContent = 'Reload bank';
    if (reload.state === 'succeeded') {
      document.getElementById('reloadStatus').textContent = `Bank ready: ${formatNumber(s.n_cells_loaded)} cells loaded.`;
    } else if (reload.state === 'failed') {
      document.getElementById('reloadStatus').textContent = `Reload failed: ${reload.error}`;
    }
    stopReloadPoll();
  }
}
function startReloadPoll() {
  if (reloadPoll) return;
  reloadPoll = setInterval(refreshStatus, 1000);
}
function stopReloadPoll() {
  if (!reloadPoll) return;
  clearInterval(reloadPoll);
  reloadPoll = null;
}
async function refreshStatus() {
  try {
    const s = await api('/api/status');
    renderStatus(s);
  } catch (err) {
    document.getElementById('status').textContent = `Server status unavailable: ${err.message || err}`;
  }
}
let askSource = null;
function ask() {
  const q = document.getElementById('question').value;
  if (!q.trim()) {
    document.getElementById('answer').textContent = 'Enter a question first.';
    return;
  }
  if (askSource) { askSource.close(); askSource = null; }
  setBusy('askButton', true, 'Asking...');
  const answer = document.getElementById('answer');
  answer.className = 'answer-card';
  answer.innerHTML = '<div class="answer-text muted">Thinking\u2026 (the first answer after a restart can take a moment to warm up)</div>';
  document.getElementById('advanced').textContent = '';
  let completed = false;
  const es = new EventSource('/api/ask/stream?question=' + encodeURIComponent(q));
  askSource = es;
  function finish() { completed = true; es.close(); if (askSource === es) askSource = null; setBusy('askButton', false); }
  es.addEventListener('phase', (e) => {
    const name = JSON.parse(e.data);
    answer.innerHTML = `<div class="answer-text muted">${escapeHtml(name)}</div>`;
  });
  es.addEventListener('result', (e) => {
    const r = JSON.parse(e.data);
    document.getElementById('advanced').textContent = JSON.stringify({citations: r.citations, retrieval: r.retrieval, gate: r.gate, verification: r.verification, closest_topics: r.closest_topics}, null, 2);
    typeAnswer(r);
    refreshStatus();
  });
  es.addEventListener('failure', (e) => {
    let msg = 'request failed';
    try { msg = JSON.parse(e.data); } catch (_) {}
    showError('answer', {message: msg});
    finish();
  });
  es.addEventListener('done', finish);
  es.onerror = () => { if (!completed) { showError('answer', {message: 'connection lost'}); finish(); } };
}
function typeAnswer(r) {
  const answer = document.getElementById('answer');
  answer.className = 'answer-card';
  if (r.silence) { renderAnswer(r); return; }
  const text = String(r.answer ?? '');
  const count = (r.citations || []).length;
  const meta = count ? `<div class="answer-meta">Answered from ${count} source${count === 1 ? '' : 's'} in memory.</div>` : '';
  answer.innerHTML = `<div class="answer-text"></div>${meta}`;
  const target = answer.querySelector('.answer-text');
  let i = 0;
  const step = () => {
    if (i >= text.length) return;
    i = Math.min(text.length, i + 2);
    target.textContent = text.slice(0, i);
    setTimeout(step, 12);
  };
  step();
}
function renderAnswer(r) {
  const answer = document.getElementById('answer');
  answer.className = 'answer-card';
  if (!r.silence) {
    const count = (r.citations || []).length;
    const source = count ? `<div class="answer-meta">Answered from ${count} source${count === 1 ? '' : 's'} in memory.</div>` : '';
    answer.innerHTML = `<div class="answer-text">${escapeHtml(r.answer)}</div>${source}`;
    return;
  }
  const topics = (r.closest_topics || []).map(t => escapeHtml(t.topic)).slice(0, 5);
  const near = topics.length ? `<div class="answer-meta">Closest things I have notes on: ${topics.join(', ')}.</div>` : '';
  answer.innerHTML = `<div class="answer-text">I don't have a confident answer for that in my memory yet.</div>${near}<div class="answer-meta">Open the Advanced details panel to see the grounding check.</div>`;
}
async function filePayload() {
  const file = document.getElementById('fileInput').files[0];
  if (!file) return null;
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return {source_name: file.name, data_base64: btoa(binary)};
}
async function preview() {
  const payload = await filePayload() || {source_name: document.getElementById('sourceName').value || 'manual.txt', text: document.getElementById('sourceText').value};
  document.getElementById('ingestReport').textContent = 'Previewing candidate cells...';
  try {
    const draft = await api('/api/preview', payload);
    currentDraft = draft;
    currentCandidates = draft.candidates;
    document.getElementById('sourceName').value = draft.source_name;
    document.getElementById('ingestReport').textContent = JSON.stringify(draft, null, 2);
    document.getElementById('candidates').innerHTML = currentCandidates.map((c, i) => `<div class="cell"><label><input type="checkbox" checked onchange="currentCandidates[${i}].accepted=this.checked"> approve</label><b> ${escapeHtml(c.label)}</b><textarea onchange="currentCandidates[${i}].text=this.value">${c.text.replaceAll('&','&amp;').replaceAll('<','&lt;')}</textarea></div>`).join('');
  } catch (err) {
    showError('ingestReport', err);
  }
}
async function commitCells() {
  const sourceName = document.getElementById('sourceName').value || 'manual.txt';
  const question = document.getElementById('question').value;
  document.getElementById('ingestReport').textContent = 'Adding approved cells to the overlay...';
  try {
    const report = await api('/api/commit', {source_name: sourceName, candidates: currentCandidates, smoke_question: question, pages_processed: currentDraft?.pages_processed || 0, extracted_char_count: currentDraft?.text_char_count || null, warnings: currentDraft?.warnings || []});
    document.getElementById('ingestReport').textContent = JSON.stringify(report, null, 2);
    await refreshStatus();
  } catch (err) {
    showError('ingestReport', err);
  }
}
async function reloadBank() {
  document.getElementById('reloadStatus').textContent = 'Starting background reload...';
  try { const s = await api('/api/reload', {}); renderStatus(s); startReloadPoll(); }
  catch (err) { document.getElementById('status').textContent = `Reload failed: ${err.message || err}`; }
}
async function searchCells() {
  document.getElementById('searchResults').textContent = 'Searching...';
  try { document.getElementById('searchResults').textContent = JSON.stringify(await api('/api/search?q=' + encodeURIComponent(document.getElementById('searchQuery').value)), null, 2); }
  catch (err) { showError('searchResults', err); }
}
async function tombstone() {
  try { await api('/api/tombstone', {cell_id: Number(document.getElementById('tombstoneId').value), reason: document.getElementById('tombstoneReason').value}); await refreshStatus(); }
  catch (err) { showError('searchResults', err); }
}
async function loadHistory() {
  document.getElementById('history').textContent = 'Loading history...';
  try { document.getElementById('history').textContent = JSON.stringify(await api('/api/history'), null, 2); }
  catch (err) { showError('history', err); }
}
async function loadQuestionHistory() {
  document.getElementById('history').textContent = 'Loading Q&A history...';
  try { document.getElementById('history').textContent = JSON.stringify(await api('/api/qa-history'), null, 2); }
  catch (err) { showError('history', err); }
}
refreshStatus();
</script>
</body>
</html>
"""


def create_app(session: WorkspaceSession, security: SecurityConfig | None = None) -> FastAPI:
    security = security if security is not None else SecurityConfig()
    app = FastAPI(title="Bank Management Workspace")
    limiter = _RateLimiter(security.rate_limit_per_minute)

    def _client_id(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def _token_ok(request: Request) -> bool:
        if not security.api_token:
            return True
        header = request.headers.get("authorization", "")
        scheme, _, presented = header.partition(" ")
        if scheme.lower() != "bearer" or not presented:
            return False
        return hmac.compare_digest(presented, security.api_token)

    def require_write(request: Request) -> None:
        if security.api_token and not _token_ok(request):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API token.")

    def require_read(request: Request) -> None:
        if security.api_token and security.require_auth_for_reads and not _token_ok(request):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API token.")

    def rate_limit(request: Request) -> None:
        if not limiter.allow(_client_id(request)):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")

    if security.max_body_bytes > 0:
        @app.middleware("http")
        async def _limit_body(request: Request, call_next):
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    if int(declared) > security.max_body_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": "Request body too large."},
                        )
                except ValueError:
                    pass
            return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/health")
    def health() -> dict:
        snapshot = session.reload_snapshot()
        pipeline_loaded = session.pipeline is not None
        return {
            "status": "ok" if pipeline_loaded else "degraded",
            "pipeline_loaded": pipeline_loaded,
            "reload_state": snapshot.get("state"),
            "reload_required": session.reload_required,
        }

    @app.get("/api/status")
    def status_route(_: None = Depends(require_read)) -> dict:
        return session.status()

    @app.post("/api/reload")
    def reload_bank(_: None = Depends(require_write)) -> dict:
        try:
            session.start_reload()
            return session.status()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/ask")
    def ask(req: AskRequest, _r: None = Depends(require_read), _l: None = Depends(rate_limit)) -> dict:
        try:
            return session.ask(req.question)
        except WorkspaceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/ask/stream")
    def ask_stream(question: str, _r: None = Depends(require_read), _l: None = Depends(rate_limit)) -> StreamingResponse:
        # Server-Sent Events variant of /api/ask. It streams honest progress
        # phases (searching / drafting / checking) followed by the final,
        # already-verified result. No answer tokens are streamed before the
        # grounding check runs, so the no-confabulation contract is preserved:
        # a rejected draft still arrives as silence, never as live text.
        q = (question or "").strip()
        if not q:
            raise HTTPException(status_code=422, detail="question is required")

        def event_stream():
            events: "queue.Queue[tuple[str, object]]" = queue.Queue()

            def on_phase(phase: str) -> None:
                events.put(("phase", phase))

            def worker() -> None:
                try:
                    payload = session.ask(q, on_phase=on_phase)
                    events.put(("result", payload))
                except WorkspaceError as exc:
                    events.put(("failure", str(exc)))
                except Exception as exc:  # noqa: BLE001 - surfaced to client
                    events.put(("failure", str(exc)))
                finally:
                    events.put(("done", None))

            threading.Thread(target=worker, daemon=True).start()
            while True:
                kind, data = events.get()
                if kind == "done":
                    yield "event: done\ndata: {}\n\n"
                    break
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/preview")
    def preview(req: PreviewRequest, _w: None = Depends(require_write), _l: None = Depends(rate_limit)) -> dict:
        try:
            if req.data_base64:
                extraction = extract_source_from_base64(req.source_name, req.data_base64)
            else:
                name = req.source_name or "manual.txt"
                extraction = extract_source(name, req.text.encode("utf-8"))
            return build_ingestion_draft(extraction).as_dict()
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/commit")
    def commit(req: CommitRequest, _: None = Depends(require_write)) -> dict:
        try:
            loaded = session.pipeline
            report = commit_candidate_cells(
                bank_path=session.bank_path,
                overlay_path=session.overlay_path,
                candidates=candidates_from_payload(req.candidates),
                source_name=req.source_name,
                bank=loaded.bank if loaded is not None else None,
                encoder=loaded.encoder if loaded is not None else None,
                smoke_question=req.smoke_question,
                pages_processed=req.pages_processed,
                extracted_char_count=req.extracted_char_count,
                warnings=req.warnings,
            )
            if report.reload_required:
                session.mark_reload_required()
            return report.as_dict()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/search")
    def search(q: str = "", limit: int = 20, _: None = Depends(require_read)) -> dict:
        return {"results": [r.as_dict() for r in search_cells(session.bank_path, session.overlay_path, q, limit)]}

    @app.post("/api/tombstone")
    def tombstone(req: TombstoneRequest, _: None = Depends(require_write)) -> dict:
        try:
            result = tombstone_cell(session.overlay_path, req.cell_id, req.reason)
            session.mark_reload_required()
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/history")
    def change_history(_: None = Depends(require_read)) -> dict:
        return {"history": history(session.overlay_path)}

    @app.get("/api/qa-history")
    def question_history(limit: int = 20, _: None = Depends(require_read)) -> dict:
        return {"history": session.question_history(limit)}

    return app


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Bank Management Workspace.")
    parser.add_argument("--bank-path", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--overlay-path", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--skip-generator", action="store_true", help="Use a placeholder generator for UI smoke sessions.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.skip_generator:
        generator = lambda prompt: "I drafted an answer but could not ground it in memory."
    else:
        generator = build_generator_from_env()
    session = WorkspaceSession(
        bank_path=args.bank_path,
        overlay_path=args.overlay_path,
        generator=generator,
        use_4bit=not args.skip_generator,
    )
    app = create_app(session, security=SecurityConfig.from_env())
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())