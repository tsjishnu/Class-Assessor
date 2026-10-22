import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.endswith("Class Quality Assessor")
    labels = [b.label for b in at.button]
    assert "Assess class" in labels
    assert at.button[0].disabled, "button must be disabled until a file or link is provided"
    assert any(s.label == "Provider" for s in at.sidebar.selectbox), "provider selector present"


def test_bad_link_shows_readable_error():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input[0].set_value("ftp://nope/video.mp4").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Could not assess" in e.value for e in at.error), "errors must be shown as messages, not stack traces"


def test_access_code_gate(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "s3cret")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert not any(b.label == "Assess class" for b in at.button), "app content must stay hidden until the code is entered"
    at.text_input[0].set_value("wrong").run()
    assert any("Incorrect" in e.value for e in at.error)
    assert not any(b.label == "Assess class" for b in at.button)
    at.text_input[0].set_value("s3cret").run()
    assert any(b.label == "Assess class" for b in at.button), "correct code unlocks the app"


def test_no_access_code_means_open_app(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_CODE", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any(b.label == "Assess class" for b in at.button)
