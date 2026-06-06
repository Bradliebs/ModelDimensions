"""Stdlib HTTP service wrapper around :class:`AnswerPipeline`.

A deliberately minimal service. The pipeline is the contract; this layer
just exposes it over HTTP so a notebook, a curl command, or another
process can hit it without spinning up Phi-3 itself.

Endpoints:

    GET  /health            -> 200 {"status":"ok","bank_path":...,"n_cells":...}
    POST /ask  body=JSON    -> 200 PipelineResult.as_dict()
                              expects {"question": "..."}
                              4xx on missing/empty question

No threading, no concurrency, no auth, no rate limiting. Single-process,
synchronous. Phi-3 generation is single-batch on one GPU; the queue is
the OS socket backlog. If you need concurrency, put a real WSGI/ASGI
server in front.
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from src.agent.answer_pipeline import AnswerPipeline


_LOG = logging.getLogger(__name__)


class _PipelineHTTPRequestHandler(BaseHTTPRequestHandler):
    """Per-request handler. The pipeline is attached to the server."""

    # Set on the server instance, read here.
    pipeline: AnswerPipeline  # type: ignore[assignment]

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # Quiet the default per-request stdout chatter.
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        _LOG.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802 - http.server convention
        if self.path == "/health":
            pipeline: AnswerPipeline = self.server.pipeline  # type: ignore[attr-defined]
            self._send_json(200, {
                "status": "ok",
                "bank_path": str(pipeline.bank.db_path),
                "n_cells": int(pipeline.bank.n_cells),
                "encoder_model": pipeline.bank.encoder_model,
                "generator_model": pipeline._generator_model_name,
            })
            return
        self._send_json(404, {"error": "not found", "path": self.path})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ask":
            self._send_json(404, {"error": "not found", "path": self.path})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_json(400, {"error": "empty body"})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return

        question = body.get("question") if isinstance(body, dict) else None
        if not isinstance(question, str) or not question.strip():
            self._send_json(
                400, {"error": "body must be {\"question\": str (non-empty)}"})
            return

        pipeline: AnswerPipeline = self.server.pipeline  # type: ignore[attr-defined]
        try:
            result = pipeline.ask(question)
        except Exception as exc:  # surfaces as 500 to caller; logged below
            _LOG.exception("pipeline.ask failed")
            self._send_json(500, {"error": str(exc), "type": type(exc).__name__})
            return

        self._send_json(200, result.as_dict())


def make_server(pipeline: AnswerPipeline, host: str = "127.0.0.1",
                port: int = 8080) -> HTTPServer:
    """Build an HTTPServer with the pipeline attached. Caller serves it."""
    server = HTTPServer((host, port), _PipelineHTTPRequestHandler)
    server.pipeline = pipeline  # type: ignore[attr-defined]
    return server


def serve(pipeline: AnswerPipeline, host: str = "127.0.0.1",
          port: int = 8080) -> None:
    """Block-serve until interrupted. Intended for ``scripts/serve.py``."""
    server = make_server(pipeline, host=host, port=port)
    _LOG.info("ask service listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _LOG.info("shutdown requested")
    finally:
        server.server_close()


__all__ = ["make_server", "serve"]
