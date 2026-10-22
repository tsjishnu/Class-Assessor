import numpy as np
import pytest

from assessor import decider, judge, llm, privacy, probes, retrieval
from assessor.speech import Segment

DIM = 16


def _vec(text: str) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    for w in "".join(c if c.isalnum() else " " for c in text.lower()).split():
        v[sum(map(ord, w)) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def emb(texts):
    return np.array([_vec(t) for t in texts])


def seg(t, text, window="opening"):
    return Segment(t, t + 3, text, -0.3, 0.01, window, "en", "unknown")


SEGS = [
    seg(10, "Hello everyone welcome to class"),
    seg(14, "How was your day today"),
    seg(20, "Let us start with the puzzle"),
    seg(900, "Your homework is to practise these puzzles", "closing"),
    seg(910, "Bye see you next week", "closing"),
]
FEATS = {"duration_s": 1000, "windows": [{"name": "opening", "start": 0, "end": 480}, {"name": "closing", "start": 880, "end": 1000}],
         "transcript_stats": {}, "audio": {}, "diarization": {}, "vision": {}}


@pytest.fixture(autouse=True)
def no_network_hints(monkeypatch):
    """Retrieval hints would call a real LLM; use the plain question text unless a test overrides this."""
    monkeypatch.setattr(probes, "get_hints", lambda gen, items, progress=None: (dict(items), "question text"))


def jev(local=True):
    return llm.Provider("jev", "jev", "https://api.typesafe.ai", "jev-latest", "k", local=local, path="/v1/systemone")


@pytest.fixture
def index():
    return retrieval.Index(SEGS, emb, emb, chunk=2, stride=1)


def test_retrieval_scope_and_best_line(index):
    hits = index.search("homework practise puzzles", k=2, scope="closing")
    assert hits and all(h.lines[0][0].window == "closing" for h in hits)
    assert "homework" in index.best_line(hits).text.lower()
    assert index.search("anything", k=3, scope="opening")[0].lines[0][0].window == "opening"


def test_decide_request_and_response_shape(monkeypatch):
    seen = {}

    class R:
        status_code = 200
        headers = {}
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {"answers": {"q1": {"type": "noul", "noul": 0.93}}}

    def post(url, headers=None, json=None, timeout=None):  # noqa: A002
        seen.update(url=url, headers=headers, body=json)
        return R()

    monkeypatch.setattr(llm.requests, "post", post)
    out = llm.decide(jev(), "some text", {"q1": "Is it?"})
    assert out == {"q1": 0.93}
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert seen["body"]["questions"]["q1"] == {"type": "noul", "instructions": "Is it?"}
    assert seen["body"]["model"] == "jev-latest" and seen["body"]["state"] == "some text"


def test_thresholds_evidence_scope_and_escalation(monkeypatch, index):
    def fake_decide(p, state, questions):
        ((qid, _),) = questions.items()
        table = {"tech": 0.1, "opening_rapport.greets": 0.95, "opening_rapport.checkin": 0.5, "tone_energy_warmth.harsh": 0.05}
        return {qid: table.get(qid, 0.1)}

    monkeypatch.setattr(llm, "decide", fake_decide)
    calls = {}

    def fake_judge(feats, segs, provider, progress=None, only=None):
        calls["only"] = only
        items = judge._build_items("opening_rapport", {"opening_rapport.checkin": {"a": "yes", "t": "0:14", "q": "How was your day today"}}, segs)
        return {"dimensions": {"opening_rapport": {"items": items}}}

    monkeypatch.setattr(judge, "judge", fake_judge)
    gen = llm.Provider("gemini", "openai", "https://x", "gm", "k", local=False)
    out = decider.judge_with_decider(FEATS, SEGS, jev(local=True), escalate=gen, index=index)
    items = {f"{d}.{i['id']}": i for d, v in out["dimensions"].items() for i in v["items"]}

    assert items["opening_rapport.greets"]["answer"] == "yes" and items["opening_rapport.greets"]["evidence"], "a confident yes carries a retrieved evidence line"
    assert items["opening_rapport.greets"]["source"] == "jev"
    assert calls["only"] == {"opening_rapport.checkin"}, "only the unsure question is escalated"
    assert items["opening_rapport.checkin"]["answer"] == "yes" and items["opening_rapport.checkin"]["source"] == "gemini"
    assert items["tone_energy_warmth.harsh"]["answer"] == "no" and items["tone_energy_warmth.harsh"]["value"] == 1.0, "reverse item: confident 'no' is good"
    assert items["individual_attention.names"]["answer"] == "no"
    assert out["escalated"] == 1 and out["escalated_to"] == "gemini/gm" and out["uncertain"] == 1
    assert out["tech_issues"] == {"count": 0, "evidence": []}


def test_unsure_without_escalation_provider_becomes_partly(monkeypatch, index):
    monkeypatch.setattr(llm, "decide", lambda p, s, q: {k: 0.5 for k in q})
    out = decider.judge_with_decider(FEATS, SEGS, jev(), escalate=None, index=index)
    items = [i for v in out["dimensions"].values() for i in v["items"]]
    assert all(i["answer"] in ("partly", "unknown") for i in items) and out["escalated"] == 0


def test_closing_questions_unknown_when_end_not_transcribed(monkeypatch):
    opening_only = [s for s in SEGS if s.window == "opening"]
    idx = retrieval.Index(opening_only, emb, emb, chunk=2, stride=1)
    monkeypatch.setattr(llm, "decide", lambda p, s, q: {k: 0.9 for k in q})
    out = decider.judge_with_decider(FEATS, opening_only, jev(), index=idx)
    closing = {i["id"]: i["answer"] for i in out["dimensions"]["structure_closing"]["items"]}
    assert closing["follow_up"] == "unknown" and closing["goodbye"] == "unknown" and closing["flow"] != "unknown"


def test_cloud_decider_requires_deidentification(monkeypatch, index):
    monkeypatch.setattr(privacy, "available", lambda: False)
    with pytest.raises(RuntimeError):
        decider.judge_with_decider(FEATS, SEGS, jev(local=False), index=index)


def test_jev_presets_and_escalation_resolution(monkeypatch):
    for k in ("JEV_API_KEY", "OPENROUTER_API_KEY", "JEV_ENDPOINT", "JEV_MODEL", "GEMINI_API_KEY", "GROQ_API_KEY", "ALLOW_EXTERNAL_LLM",
              "JEV_ESCALATE_TO", "LLM_PROVIDER", "LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "_ollama_models", lambda host: None)
    monkeypatch.setenv("JEV_API_KEY", "k")
    assert llm.resolve_provider("jev") is None, "needs explicit consent"
    monkeypatch.setenv("ALLOW_EXTERNAL_LLM", "1")
    p = llm.resolve_provider("jev")
    assert (p.kind, p.base_url, p.path, p.model) == ("jev", "https://api.typesafe.ai", "/v1/systemone", "jev-latest")
    monkeypatch.setenv("JEV_ENDPOINT", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.delenv("JEV_API_KEY")
    p = llm.resolve_provider("jev")
    assert (p.base_url, p.path, p.model) == ("https://openrouter.ai/api/alpha", "/decisions", "typesafe/jev-1.13")
    assert llm.resolve_escalation(p) is None, "no generative provider configured"
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert llm.resolve_escalation(p).name == "gemini"
    monkeypatch.setenv("JEV_ESCALATE_TO", "none")
    assert llm.resolve_escalation(p) is None
    monkeypatch.delenv("JEV_ESCALATE_TO")
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    assert llm.resolve_provider().name == "gemini", "jev is opt-in and never auto-selected"
    with pytest.raises(RuntimeError):
        llm.chat_json(p, "s", "u")


def test_ai_generated_hints_drive_retrieval(monkeypatch, index):
    seen = []
    monkeypatch.setattr(probes, "get_hints", lambda gen, items, progress=None: ({k: "AI HINT " + k for k in items}, "ai-generated"))
    monkeypatch.setattr(retrieval.Index, "search", lambda self, query, k=3, scope="any": seen.append(query) or [])
    monkeypatch.setattr(llm, "decide", lambda p, s, q: {k: 0.9 for k in q})
    gen = llm.Provider("gemini", "openai", "https://x", "gm", "k", local=False)
    out = decider.judge_with_decider(FEATS, SEGS, jev(), escalate=gen, index=index)
    assert out["retrieval_hints"] == "ai-generated"
    assert seen and all(q.startswith("AI HINT ") for q in seen if not q.startswith("audio or video")), "retrieval queries come from the AI hints"
