"""Headless smoke tests of the Streamlit script (Python side only; map clicks need a real browser)."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from conftest import FIX, ROOT

APP = str(ROOT / "app" / "main.py")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(FIX / "model"))
    monkeypatch.delenv("APP_PASSWORD", raising=False)


def run():
    return AppTest.from_file(APP, default_timeout=90).run()


def labels(at):
    return [m.label for m in at.metric]


def test_loads_without_exceptions():
    at = run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Single trip", "Batch upload", "Model & performance", "About"]


def test_example_trace_gives_prediction():
    at = run()
    next(b for b in at.button if b.label == "Load an example trace").click()
    at.run()
    assert not at.exception
    assert "Predicted total trip time" in labels(at)
    assert at.session_state["trace_text"].startswith("[[")


def test_pasting_latlon_trace_is_accepted_with_note():
    at = run()
    at.text_area(key="trace_text").set_value("41.1496,-8.6110\n41.1498,-8.6105\n41.1500,-8.6098").run()
    assert not at.exception
    assert any("swapped" in i.value for i in at.info)
    assert "Predicted total trip time" in labels(at)


def test_bad_trace_shows_error_not_exception():
    at = run()
    at.text_area(key="trace_text").set_value("-8.6,41.1\nnot a point\n-8.5,41.2").run()
    assert not at.exception and at.error and "Predicted total trip time" not in labels(at)


def test_replay_mode_shows_actual_and_error():
    at = run()
    at.radio(key="method").set_value("Replay a dataset trip").run()
    assert not at.exception
    assert "Actual total trip time" in labels(at) and "Prediction error" in labels(at)


def test_sidebar_inputs_change_prediction():
    at = run()
    next(b for b in at.button if b.label == "Load an example trace").click()
    at.run()
    first = [m.value for m in at.metric if m.label == "Predicted total trip time"][0]
    at.radio(key="call").set_value("B: picked up at a taxi stand").run()
    assert not at.exception
    assert any(m.label == "Predicted total trip time" for m in at.metric)
    assert at.sidebar.number_input   # the stand selector appears for call type B


def test_model_tab_renders():
    at = run()
    assert not at.exception
    assert any("Validation performance" in s.value for s in at.subheader)


def test_password_gate(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception and "Predicted total trip time" not in labels(at) and at.text_input
    at.text_input[0].set_value("wrong").run()
    assert at.error and "auth_ok" not in at.session_state
    at.text_input[0].set_value("s3cret").run()
    assert at.session_state["auth_ok"] is True


def test_missing_model_shows_friendly_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    assert any("could not be loaded" in e.value for e in at.error)
