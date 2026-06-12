"""Smoke test: the Streamlit app runs end to end on synthetic data
(generation, semblance, auto-pick, NMO, stack) without raising."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_runs_without_exception():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
