import pytest

from assessor import llm, moves, nli, rubric
from assessor.speech import Segment


def seg(t, text, window="opening"):
    return Segment(t, t + 3, text, -0.3, 0.01, window, "en", "unknown")


SEGS = [
    seg(10, "Hello everyone, good evening!"),
    seg(14, "How was your day today, Aarav?"),
    seg(20, "Today we will learn about the king."),
    seg(150, "Very good move, you protected the king well."),
    seg(900, "Your homework is to solve five puzzles.", "closing"),
    seg(915, "See you next week, bye everyone!", "closing"),
]
WINDOWS = [{"name": "opening", "start": 0, "end": 480}, {"name": "closing", "start": 840, "end": 1020}]
FEATS = {"duration_s": 1020, "windows": WINDOWS, "transcript_stats": {"first_speech_s": 10}, "audio": {}, "diarization": {}, "vision": {}}


def test_segments_are_grouped_per_window_in_two_minute_buckets():
    sg = moves.build_segments(SEGS, WINDOWS)
    assert [s["window"] for s in sg] == ["opening", "opening", "closing"]
    assert sg[0]["start"] == 0 and sg[1]["start"] == 120 and sg[2]["start"] == 840
    assert [s["id"] for s in sg] == [1, 2, 3]


def test_batching_respects_the_model_budget():
    sg = moves.build_segments(SEGS, WINDOWS)
    assert len(moves._batches(sg, 100000)) == 1
    assert len(moves._batches(sg, 150)) == 3, "a small local model gets several small requests"


def test_clean_events_rejects_invented_quotes_and_fixes_timestamps():
    sg = moves.build_segments(SEGS, WINDOWS)
    raw = [{"id": 1, "activity": "puzzle", "energy": "high", "note": "n", "events": [
        {"type": "greeting", "t": "0:11", "quote": "Hello everyone, good evening!"},
        {"type": "rapport_checkin", "t": "5:00", "quote": "how was your day today aarav"},
        {"type": "praise_generic", "t": "0:30", "quote": "Brilliant work, you are a genius"},
        {"type": "not_a_real_type", "t": "0:30", "quote": "Hello everyone"},
        {"type": "greeting", "t": "0:10", "quote": "Hello everyone, good evening!"},
    ]}]
    events, info = moves.clean_events(raw, sg)
    assert [e["type"] for e in events] == ["greeting", "rapport_checkin"]
    assert events[1]["t"] == pytest.approx(14, abs=1), "claimed 5:00 is corrected to the true time"
    assert info["dropped"] == {"unknown_type": 1, "quote_not_found": 1, "duplicate": 1}
    assert info["segments"][1]["activity"] == "puzzle" and info["segments"][1]["energy"] == "high"


def test_nli_verifier_drops_unsupported_claims(monkeypatch):
    events = [{"type": "praise_generic", "t": 150.0, "text": "Very good move", "window": "opening", "supported": None, "p": None},
              {"type": "praise_generic", "t": 20.0, "text": "Today we will learn about the king.", "window": "opening", "supported": None, "p": None}]
    monkeypatch.setattr(nli, "decide", lambda model, premise, hyp: 0.9 if "Very good" in premise else 0.02)
    out = moves.verify_events(events, SEGS, "fake-model", 0.2)
    assert out["verified"] == 2 and out["dropped"] == 1
    assert events[0]["supported"] is True and events[1]["supported"] is False
    assert moves.verify_events(events, SEGS, None, 0.2)["skipped"], "no local verifier -> skipped, never crashes"


def _ev(t, typ, window="opening", supported=True):
    return {"type": typ, "t": t, "text": f"line at {t}", "segment": 1, "window": window, "supported": supported, "p": 0.9}


def _items(dims, dim):
    return {i["id"]: i for i in dims[dim]}


def test_scores_follow_frequency_not_mere_presence():
    wins = [{"name": "opening", "start": 0, "end": 600}]            # 10 observed minutes
    one = [_ev(30, "praise_generic")]
    many = [_ev(30 + 40 * k, "praise_specific") for k in range(6)]
    meta = {1: {"activity": "puzzle", "energy": "medium", "note": ""}}
    low = _items(moves.score_items(one, meta, wins, FEATS, False), "tone_energy_warmth")["praise"]
    high = _items(moves.score_items(many, meta, wins, FEATS, False), "tone_energy_warmth")["praise"]
    assert low["value"] < 0.35 and low["answer"] == "no", "a single generic 'good' is not a warm class"
    assert high["value"] == 1.0 and high["answer"] == "yes"


def test_unsupported_events_do_not_count():
    wins = [{"name": "opening", "start": 0, "end": 600}]
    evs = [_ev(30 + 40 * k, "praise_specific", supported=False) for k in range(6)]
    item = _items(moves.score_items(evs, {}, wins, FEATS, False), "tone_energy_warmth")["praise"]
    assert item["value"] == 0.0


def test_unobserved_parts_are_unknown_not_bad():
    wins = [{"name": "opening", "start": 0, "end": 480}]            # closing never transcribed
    dims = moves.score_items([], {}, wins, FEATS, False)
    assert dims["structure_closing"] == [], "no closing window -> no closing items (dimension not assessed)"
    assert _items(dims, "tone_energy_warmth")["kind_mistakes"]["answer"] == "unknown", "no mistakes observed -> cannot judge kindness"


def test_negative_signals_are_reverse_items():
    wins = [{"name": "opening", "start": 0, "end": 600}]
    items = _items(moves.score_items([_ev(30, "harsh_tone"), _ev(90, "harsh_tone")], {}, wins, FEATS, False), "tone_energy_warmth")
    assert items["harsh"]["reverse"] and items["harsh"]["answer"] == "yes" and items["harsh"]["value"] == 0.0


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, provider, system, user, retries=1, max_tokens=1400):
        self.calls += 1
        return {"segments": [
            {"id": 1, "note": "greeting and check-in", "activity": "discussion", "energy": "high", "events": [
                {"type": "greeting", "t": "0:11", "quote": "Hello everyone, good evening!"},
                {"type": "rapport_checkin", "t": "0:14", "quote": "How was your day today, Aarav?"},
                {"type": "agenda", "t": "0:20", "quote": "Today we will learn about the king."},
                {"type": "names_learner", "t": "0:14", "quote": "How was your day today, Aarav?"}]},
            {"id": 2, "note": "play", "activity": "puzzle", "energy": "medium", "events": [
                {"type": "praise_specific", "t": "2:30", "quote": "Very good move, you protected the king well."},
                {"type": "made_up", "t": "2:40", "quote": "x"}]},
            {"id": 3, "note": "closing", "activity": "discussion", "energy": "medium", "events": [
                {"type": "homework_assigned", "t": "15:00", "quote": "Your homework is to solve five puzzles."},
                {"type": "goodbye", "t": "15:15", "quote": "See you next week, bye everyone!"}]},
        ], "new_learner_present": False}


def test_end_to_end_with_fake_llm_feeds_the_scorer(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat_json", fake)
    monkeypatch.setattr(llm, "_build", lambda name, model=None: None)   # no local verifier in this test
    out = moves.judge_moves(FEATS, SEGS, llm.Provider("fake", "ollama", "", "m", local=True))
    assert fake.calls == 1 and out["usable"] and out["mode"] == "moves"
    assert out["stats"]["events_kept"] == 7 and out["stats"]["verifier"]["skipped"]
    items = {i["id"]: i for i in out["dimensions"]["structure_closing"]["items"]}
    assert items["follow_up"]["answer"] == "yes" and items["goodbye"]["evidence"], "closing items carry verified evidence"
    report = rubric.assess(FEATS, out)
    assert report["ai"]["mode"] == "moves" and report["ai"]["dimensions_judged"] >= 3
    assert any(o["evidence"] for d in report["dimensions"] for o in d["observations"])


def test_unusable_when_no_transcript():
    out = moves.judge_moves(FEATS, [], llm.Provider("fake", "ollama", "", "m", local=True))
    assert out["usable"] is False


def test_demo_session_is_flagged_not_silently_graded():
    out = {"usable": True, "dimensions": {}, "tech_issues": None, "session_type": "demo_trial", "provider": "x", "model": "m", "local": True}
    flags = rubric.assess(FEATS, out)["flags"]
    assert any("demo_trial" in f and "may not be meaningful" in f for f in flags)
    out["session_type"] = "regular_class"
    assert not any("demo_trial" in f for f in rubric.assess(FEATS, out)["flags"])


def test_skipped_providers_are_diagnostics_not_user_warnings(monkeypatch):
    from assessor import pipeline
    a = llm.Provider("gemini", "openai", "https://x", "model-a", "k", local=False)
    b = llm.Provider("gemini", "openai", "https://x", "model-b", "k", local=False)
    monkeypatch.setattr(llm, "generative_chain", lambda prefer=None: [a, b])
    llm.reset_health()

    def fake(feats, segs, p, say=None):
        if p.model == "model-a":
            raise llm.QuotaExhausted("a: quota", 3600)
        return {"usable": True, "dimensions": {}, "tech_issues": None, "provider": p.name, "model": p.model, "mode": "moves"}

    monkeypatch.setattr(pipeline.moves, "judge_moves", fake)
    feats = {"notes": [], "windows": WINDOWS, "segments": [seg(10 * k, "word " * 12).to_dict() for k in range(20)]}
    j = pipeline.judge_features(feats, a)
    assert j["model"] == "model-b" and j["fallbacks"] == ["gemini/model-a: free quota used up"]
    assert feats["notes"] == [], "a later provider succeeded: no user-facing warning"
    llm.reset_health()
    monkeypatch.setattr(pipeline.moves, "judge_moves", lambda *a_, **k: (_ for _ in ()).throw(llm.QuotaExhausted("q", 60)))
    feats = {"notes": [], "windows": WINDOWS, "segments": [seg(10 * k, "word " * 12).to_dict() for k in range(20)]}
    assert pipeline.judge_features(feats, a) is None and len(feats["notes"]) == 2, "every provider failed: now the user is told"
    llm.reset_health()


def test_unusable_transcripts_are_rejected_before_any_ai_call(monkeypatch):
    from assessor import pipeline, transcript
    called = []
    monkeypatch.setattr(pipeline.moves, "judge_moves", lambda *a, **k: called.append(1))
    a = llm.Provider("gemini", "openai", "https://x", "model-a", "k", local=False)
    garbage = [seg(10 * k, "༼༼༽༽ జ੍੍ " * 6) for k in range(20)]
    for segs, why in [([], "segments"), (garbage, "non-Latin"), ([seg(10 * k, "hi") for k in range(20)], "very little speech")]:
        feats = {"notes": [], "windows": WINDOWS, "segments": [s.to_dict() for s in segs]}
        j = pipeline.judge_features(feats, a)
        assert j["usable"] is False and why in j["reason"], (why, j.get("reason"))
    assert not called, "no AI request (and so no quota) is spent on a transcript that cannot be assessed"
    good = [seg(10 * k, "word " * 12) for k in range(20)]
    assert transcript.unusable_reason({"windows": WINDOWS}, good) is None


def test_translated_class_gets_a_flag_and_reduced_confidence():
    good = {"usable": True, "dimensions": {}, "tech_issues": None, "provider": "x", "model": "m", "local": True, "mode": "moves"}
    feats = dict(FEATS, asr={"language_share": {"ml": 1.0}, "translated_share": 1.0})
    rep = rubric.assess(feats, good)
    assert any("machine-translated" in f for f in rep["flags"]) and rep["confidence"] != "High"


def test_unusable_transcript_never_gets_high_confidence():
    bad = {"usable": False, "reason": "only 3 speech segments were recognised", "dimensions": {}, "tech_issues": None, "provider": "x", "model": "m", "local": True}
    rep = rubric.assess(FEATS, bad)
    assert rep["confidence"] == "Low" and any("NOT assessed" in f and "only 3 speech segments" in f for f in rep["flags"])


def test_speech_model_can_be_released_for_low_memory_hosts():
    from assessor import speech
    speech._model = object()
    speech.unload()
    assert speech._model is None


def test_quotes_are_verified_against_masked_text_but_original_line_is_shown():
    """Cloud models see masked text, so their quotes contain [name]; evidence shown must still be the original line."""
    sg = moves.build_segments(SEGS, WINDOWS)
    mask = lambda t: t.replace("Aarav", "[name]")  # noqa: E731
    raw = [{"id": 1, "activity": "discussion", "energy": "high", "note": "n", "moves": {"rapport_checkin": ["How was your day today, [name]?"]}}]
    events, info = moves.clean_events(raw, sg, mask)
    assert len(events) == 1 and info["dropped"]["quote_not_found"] == 0
    assert "Aarav" in events[0]["text"], "the report shows the real (local) line, not the masked one"
