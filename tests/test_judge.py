import pytest

from assessor import judge, llm, privacy, transcript
from assessor.speech import Segment, collapse_repeats


def seg(t, text, role="tutor", window="opening", lp=-0.3):
    return Segment(t, t + 3, text, lp, 0.01, window, "en", role)


SEGS = [
    seg(21, "Hello everyone, good evening!"),
    seg(25, "How was your day, Aarav?", "tutor"),
    seg(29, "It was great, we went to the park.", "child"),
    seg(900, "Your homework is to solve five puzzles.", "tutor", "closing"),
    seg(910, "See you next week, bye!", "tutor", "closing"),
]
WINDOWS = [{"name": "opening", "start": 0, "end": 480}, {"name": "closing", "start": 840, "end": 1020}]
FEATS = {"duration_s": 1020, "windows": WINDOWS, "audio": {}, "diarization": {"tutor_share_time": 0.6}, "vision": {}, "asr": {"language_share": {"en": 1.0}},
         "transcript_stats": {"first_speech_s": 21}}


def test_collapse_repeats_only_loops():
    assert collapse_repeats("Hello, good morning. Hello, good morning. Hello, good morning.") == "Hello, good morning."
    assert collapse_repeats("Very good, very good!") == "Very good, very good!"


def test_transcript_format_and_budget():
    text, kept = transcript.build_transcript(SEGS, WINDOWS, 5000)
    assert "## OPENING" in text and "## CLOSING" in text and "[0:21] Hello everyone" in text
    tiny, kept2 = transcript.build_transcript(SEGS, WINDOWS, 120)
    assert len(kept2) < len(kept)
    closing_only = [s for s in kept2 if s.window == "closing"]
    assert not closing_only or closing_only[-1].text.startswith("See you")  # closing keeps its LAST lines


def test_quote_verification_rejects_made_up_and_fixes_timestamp():
    ok = judge.verify_quote("how was your day aarav", 5, SEGS)
    assert ok and abs(ok[0] - 25) < 1, "real quote verified and timestamp corrected to the true one"
    assert judge.verify_quote("let us all celebrate the championship trophy", 25, SEGS) is None
    assert judge.verify_quote("ok", None, SEGS) is None


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, provider, system, user, retries=1, max_tokens=1400):
        self.calls += 1
        if self.calls == 1:
            return {"answers": {
                "opening_rapport.greets": {"a": "yes", "t": "0:21", "q": "Hello everyone, good evening!"},
                "opening_rapport.checkin": {"a": "yes", "t": "5", "q": "How was your day, Aarav?"},
                "opening_rapport.agenda": {"a": "no", "t": "", "q": ""},
                "individual_attention.names": {"a": "yes", "t": "0:25", "q": "totally invented words that never happened"},
                "tone_energy_warmth.harsh": {"a": "no", "t": "", "q": ""},
            }, "transcript_usable": True}
        return {"answers": {
            "structure_closing.follow_up": {"a": "yes", "t": "15:00", "q": "Your homework is to solve five puzzles."},
            "structure_closing.goodbye": {"a": "yes", "t": "15:10", "q": "See you next week, bye!"},
            "instruction_feedback.probing": {"a": "unknown"},
        }, "transcript_usable": True, "technical_issues": {"count": 0, "t": "", "q": ""}}


def _items(out, dim):
    return {i["id"]: i for i in out["dimensions"][dim]["items"]}


def test_judge_checklist_with_fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat_json", fake)
    out = judge.judge(FEATS, SEGS, llm.Provider("fake", "ollama", "", "m", local=True))
    assert fake.calls == 2, "two question groups -> two calls"
    op = _items(out, "opening_rapport")
    assert op["greets"]["answer"] == "yes" and op["greets"]["evidence"], "verified quote retained"
    assert op["checkin"]["evidence"][0]["t"] == pytest.approx(25, abs=1), "timestamp corrected to the true one"
    assert op["agenda"]["answer"] == "no" and op["agenda"]["value"] == 0.0
    assert op["recap"]["answer"] == "unknown" and op["recap"]["value"] is None, "unanswered ids are unknown, not guessed"
    inv = _items(out, "individual_attention")["names"]
    assert inv["answer"] == "partly" and inv["unverified"] and inv["evidence"] == [], "a 'yes' with a made-up quote is down-weighted"
    assert _items(out, "tone_energy_warmth")["harsh"]["value"] == 1.0, "reverse item: 'no' harshness is good"
    assert _items(out, "structure_closing")["follow_up"]["evidence"][0]["t"] == pytest.approx(900, abs=1)


def test_dimension_score_from_answers():
    mk = lambda v: {"value": v}  # noqa: E731
    assert judge.dimension_score([mk(1.0), mk(1.0), mk(None)]) == 100.0
    assert judge.dimension_score([mk(1.0), mk(0.0)]) == 50.0
    assert judge.dimension_score([mk(1.0), mk(None)]) is None, "needs at least two answered items"


def test_cloud_provider_refuses_without_deidentification(monkeypatch):
    monkeypatch.setattr(privacy, "available", lambda: False)
    p = llm.Provider("groq", "openai", "https://x", "m", "k", local=False)
    with pytest.raises(RuntimeError):
        judge.judge(FEATS, SEGS, p)


def test_json_parsing_is_tolerant():
    assert llm._parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm._parse_json('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        llm._parse_json("no json here")


@pytest.mark.skipif(not privacy.available(), reason="spaCy model not installed")
def test_deidentify_masks_person_names():
    out, names = privacy.deidentify(["Hello Gautam, how was your day? Gautam did well."])
    assert "Gautam" in names and "Gautam" not in out[0]
