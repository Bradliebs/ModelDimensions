"""Local SLM backends for the optional assistant-composer layer.

A *backend* is anything that turns a prompt string into raw model text. The
assistant layer never depends on a real model: the default path is a
deterministic template composer (no backend at all), and every test uses
:class:`MockSLMBackend`, which is fully offline and deterministic.

Two optional real backends are provided for opt-in local use:

* :class:`OllamaBackend` talks to a locally running Ollama daemon.
* :class:`OpenAICompatibleBackend` talks to a local OpenAI-compatible endpoint
  (for example llama.cpp's server or LM Studio).

Both are *gracefully unavailable*: ``is_available()`` returns ``False`` unless
the relevant endpoint is explicitly configured, and they import their network
dependency lazily inside ``generate`` only. Nothing here makes a network call
or downloads a model at import time, so importing this module is always safe in
tests and CI.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class LocalSLMBackend(Protocol):
    """Anything that can turn a prompt into raw model text, optionally."""

    name: str

    def is_available(self) -> bool:
        """Return ``True`` only when this backend can actually generate."""
        ...

    def generate(self, prompt: str) -> str:
        """Return raw model text for ``prompt`` (may raise if unavailable)."""
        ...


class MockSLMBackend:
    """A deterministic, offline backend for tests and demos.

    It never touches the network and never loads a model. The reply is either a
    fixed string or a callable applied to the prompt, so a test can make the
    "model" return grounded text, an invented citation, or malformed output on
    demand. ``available`` lets a test exercise the gracefully-unavailable path.
    """

    def __init__(self, response: str | Callable[[str], str] = "",
                 *, available: bool = True, name: str = "mock-slm"):
        self._response = response
        self._available = available
        self.name = name

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt: str) -> str:
        if not self._available:
            raise RuntimeError(f"{self.name} backend is not available")
        if callable(self._response):
            return self._response(prompt)
        return self._response


class OllamaBackend:
    """Optional backend for a locally running Ollama daemon.

    Unavailable unless an Ollama host is configured (constructor arg or the
    ``OLLAMA_HOST`` environment variable). The ``requests`` dependency and the
    network call live inside ``generate`` only, so importing or constructing
    this backend never reaches out anywhere.
    """

    def __init__(self, model: str = "llama3.2",
                 host: Optional[str] = None,
                 *, timeout: float = 30.0):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def is_available(self) -> bool:
        return bool(self.host)

    def generate(self, prompt: str) -> str:  # pragma: no cover - needs a daemon
        if not self.is_available():
            raise RuntimeError("OllamaBackend has no configured host")
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "OllamaBackend requires the 'requests' package.") from exc
        url = self.host.rstrip("/") + "/api/generate"
        resp = requests.post(
            url,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


class OpenAICompatibleBackend:
    """Optional backend for a local OpenAI-compatible chat endpoint.

    Unavailable unless a base URL is configured (constructor arg or the
    ``OPENAI_COMPAT_BASE_URL`` environment variable). Like the Ollama backend,
    the network dependency is imported lazily inside ``generate`` and there is
    no model download.
    """

    def __init__(self, model: str = "local-model",
                 base_url: Optional[str] = None,
                 *, api_key: Optional[str] = None, timeout: float = 30.0):
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_COMPAT_BASE_URL", "")
        self.api_key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY", "")
        self.timeout = timeout
        self.name = f"openai-compat:{model}"

    def is_available(self) -> bool:
        return bool(self.base_url)

    def generate(self, prompt: str) -> str:  # pragma: no cover - needs a server
        if not self.is_available():
            raise RuntimeError("OpenAICompatibleBackend has no configured base URL")
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "OpenAICompatibleBackend requires the 'requests' package."
            ) from exc
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            url,
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        choices = resp.json().get("choices") or [{}]
        return choices[0].get("message", {}).get("content", "")


def make_default_slm_backend() -> LocalSLMBackend:
    """Return the configured local backend, or an unavailable one.

    Resolution order is environment-driven and offline-safe:

    * ``OLLAMA_HOST`` set -> :class:`OllamaBackend`,
    * ``OPENAI_COMPAT_BASE_URL`` set -> :class:`OpenAICompatibleBackend`,
    * otherwise an unavailable :class:`MockSLMBackend` so the composer falls
      back to deterministic templates.
    """
    if os.environ.get("OLLAMA_HOST"):
        return OllamaBackend(
            model=os.environ.get("OLLAMA_MODEL", "llama3.2"))
    if os.environ.get("OPENAI_COMPAT_BASE_URL"):
        return OpenAICompatibleBackend(
            model=os.environ.get("OPENAI_COMPAT_MODEL", "local-model"))
    return MockSLMBackend(available=False, name="unconfigured-slm")
