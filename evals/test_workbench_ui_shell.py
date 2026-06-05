"""UI-render tests for the Consultant Workbench Streamlit shell.

These exercise the real Streamlit render path (``_render_streamlit`` in
``app/workbench.py``) through Streamlit's ``AppTest`` harness, so the shell is
no longer untested. They assert shell behaviour only — the app opens as the
Consultant Workbench, every navigation page renders without raising, the global
"Add data" entry point is present, and clicking it navigates to the governed
Imports workflow. No test here mutates packs, memory, the registry, or
activation state; rendering a page is read-only by construction.

``streamlit`` is an optional dependency, so the whole module is skipped when it
is not installed (the same posture as the rest of the suite).
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


def _app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    return at.run()


def _add_data_button(at: AppTest):
    for button in at.sidebar.button:
        if "Add data" in button.label:
            return button
    return None


def test_app_opens_as_consultant_workbench() -> None:
    at = _app()
    assert not at.exception
    assert at.sidebar.title[0].value == cm.PRODUCT_NAME
    assert at.sidebar.title[0].value == "Consultant Workbench"


def test_default_page_is_home() -> None:
    at = _app()
    assert at.title[0].value == "Home"


def test_navigation_lists_every_section() -> None:
    at = _app()
    radio = at.sidebar.radio[0]
    expected = [s.label for s in cm.navigation()]
    assert list(radio.options) == expected


def test_add_data_button_is_visible() -> None:
    at = _app()
    assert _add_data_button(at) is not None


def test_add_data_navigates_to_imports() -> None:
    at = _app()
    assert at.title[0].value != "Imports"
    button = _add_data_button(at)
    assert button is not None
    button.click().run()
    assert not at.exception
    assert at.title[0].value == "Imports"


@pytest.mark.parametrize("label", [s.label for s in cm.navigation()])
def test_every_page_renders_without_error(label: str) -> None:
    at = _app()
    at.sidebar.radio[0].set_value(label).run()
    assert not at.exception
    assert at.title[0].value == label
