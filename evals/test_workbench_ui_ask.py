"""UI-render tests for the evidence-bound Ask workspace (Streamlit page).

These exercise the real Streamlit render path for the Ask page through the
``AppTest`` harness: the page opens, offers the governed answer styles, shows
the evidence scope being searched, holds an explicit empty state, and — when a
question is asked — renders a status and records a read-only result that
attempted no durable mutation. Asking is read-only by construction; export is a
browser download that writes no backend state, so no test here mutates packs,
memory, the registry, or activation.

``streamlit`` is an optional dependency, so the module is skipped when it is not
installed (the same posture as the rest of the suite).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import agent.console_model as cm  # noqa: E402

APP = ROOT / "app" / "workbench.py"

# A question the governed pipeline recognises as an evidence query; whether it
# grounds or is refused, the page must handle it read-only and without error.
_QUERY = "What is least-privilege access control?"


def _ask_page() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Ask").run()
    return at


def _ask_button(at: AppTest):
    for button in at.button:
        if button.label == "Ask":
            return button
    return None


def test_ask_page_renders() -> None:
    at = _ask_page()
    assert not at.exception
    assert at.title[0].value == "Ask"


def test_ask_offers_governed_answer_styles() -> None:
    at = _ask_page()
    labels = [label for _, label in cm.ANSWER_MODES]
    assert list(at.selectbox[0].options) == labels


def test_ask_shows_evidence_scope() -> None:
    at = _ask_page()
    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "pack(s)" in captions
    assert "source(s)" in captions


def test_ask_has_explicit_empty_state() -> None:
    at = _ask_page()
    info = " ".join(el.value for el in at.info)
    assert "Ask a question" in info


def test_ask_answers_and_records_read_only_result() -> None:
    at = _ask_page()
    at.text_area[0].set_value(_QUERY).run()
    button = _ask_button(at)
    assert button is not None
    button.click().run()
    assert not at.exception
    # A status was rendered for the answer.
    assert any("Status:" in el.value for el in at.info)
    # The governed result was recorded and attempted no durable mutation.
    result = at.session_state["ask_result"]
    assert result.state_mutation_attempted is False


def test_ask_clear_resets_the_page() -> None:
    at = _ask_page()
    at.text_area[0].set_value(_QUERY).run()
    _ask_button(at).click().run()
    assert "ask_result" in at.session_state
    for button in at.button:
        if button.label == "Clear":
            button.click().run()
            break
    assert not at.exception
    assert "ask_result" not in at.session_state
