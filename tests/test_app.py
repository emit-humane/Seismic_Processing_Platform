"""Smoke tests: both app modes run end to end without raising —
pre-stack (synthetic gather, semblance, auto-pick, NMO, stack) and
post-stack (synthetic volume, attribute display)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_prestack_mode_runs():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception


def test_poststack_mode_runs():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Post-stack (volume)").run()
    assert not at.exception


def test_poststack_attribute_switch():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Post-stack (volume)").run()
    at.sidebar.selectbox[0].set_value("Envelope").run()
    assert not at.exception
