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


def test_poststack_coherence_overlay():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Post-stack (volume)").run()
    at.sidebar.selectbox[0].set_value("Coherence (semblance)").run()
    assert not at.exception
    # time-slice view with the fault overlay active (radio[0]=Mode, [1]=Source, [2]=Slice)
    at.sidebar.radio[2].set_value("Time slice").run()
    assert not at.exception


def test_poststack_segy_export():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Post-stack (volume)").run()
    at.sidebar.selectbox[0].set_value("Envelope").run()
    # the only non-sidebar button is the export-prepare button
    prepare = [b for b in at.button if "Prepare SEG-Y" in (b.label or "")]
    assert prepare, "export button not found"
    prepare[0].set_value(True).run()
    assert not at.exception
    assert "export" in at.session_state
    name, payload = at.session_state["export"]
    assert name == "Envelope"
    assert len(payload) > 1000  # non-trivial SEG-Y bytes
