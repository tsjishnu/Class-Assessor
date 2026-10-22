import json

from assessor import config, llm, probes

GEN = llm.Provider("gemini", "openai", "https://x", "gm", "k", local=False)
ITEMS = {"a.one": "Does the teacher greet?", "a.two": "Does the teacher assign homework?"}


def test_hints_are_generated_once_then_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    calls = []

    def fake(p, system, user, retries=1, max_tokens=900):
        calls.append(user)
        return {"hints": {"a.one": ["Hello everyone!", "Good morning, welcome."], "a.two": ["Please practise at home.", "Your homework is page 4."]}}

    monkeypatch.setattr(llm, "chat_json", fake)
    h, src = probes.get_hints(GEN, ITEMS)
    assert src == "ai-generated" and "Good morning" in h["a.one"] and "homework" in h["a.two"].lower()
    h2, src2 = probes.get_hints(GEN, ITEMS)
    assert h2 == h and len(calls) == 1, "second call is served from the cache"
    assert json.load(open(next((tmp_path / "probes").glob("*.json"))))["a.one"]


def test_cache_is_keyed_by_model_and_questions(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"hints": {"a.one": ["x y z"]}})
    probes.get_hints(GEN, ITEMS)
    other = llm.Provider("gemini", "openai", "https://x", "another-model", "k", local=False)
    n_before = len(list((tmp_path / "probes").glob("*.json")))
    probes.get_hints(other, ITEMS)
    assert len(list((tmp_path / "probes").glob("*.json"))) == n_before + 1


def test_falls_back_to_question_text(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    h, src = probes.get_hints(None, ITEMS)
    assert src == "question text" and h == ITEMS
    jev = llm.Provider("jev", "jev", "https://x", "m", "k", local=False)
    assert probes.get_hints(jev, ITEMS)[1] == "question text", "a decider cannot write hints"

    def boom(*a, **k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(llm, "chat_json", boom)
    h, src = probes.get_hints(GEN, ITEMS)
    assert src == "question text" and h == ITEMS, "LLM failure must not break the pipeline"


def test_partial_model_output_falls_back_per_question(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"hints": {"a.one": "Hello there"}})
    h, src = probes.get_hints(GEN, ITEMS)
    assert h["a.one"] == "Hello there" and h["a.two"] == ITEMS["a.two"]
