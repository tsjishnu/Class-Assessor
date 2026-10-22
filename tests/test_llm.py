import pytest

from assessor import llm

ENV = ["LLM_PROVIDER", "LLM_AUTO_ORDER", "GEMINI_MODELS", "GEMINI_REASONING_EFFORT", "ESCALATE_TO", "JEV_ESCALATE_TO", "LLM_FALLBACK", "ALLOW_EXTERNAL_LLM", "GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "GROQ_MODEL",
       "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "OLLAMA_MODEL", "OLLAMA_HOST"]


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "_ollama_models", lambda host: None)  # no real Ollama in unit tests


def test_nothing_configured_means_no_provider():
    assert llm.resolve_provider() is None
    generative = [n for n in llm.available_providers() if n != "hf-nli"]  # the local decider needs no configuration
    assert generative == []


def test_cloud_provider_needs_key_and_explicit_consent(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert llm.resolve_provider("gemini") is None, "key alone is not consent"
    assert "ALLOW_EXTERNAL_LLM" in llm.explain_unavailable("gemini")
    monkeypatch.setenv("ALLOW_EXTERNAL_LLM", "1")
    p = llm.resolve_provider("gemini")
    assert p.name == "gemini" and not p.local and p.base_url.endswith("/v1beta/openai")


def test_switching_provider_is_one_setting(monkeypatch):
    monkeypatch.setenv("ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "g"); monkeypatch.setenv("GROQ_API_KEY", "q")
    assert llm.resolve_provider().name == "gemini", "auto prefers gemini by default order"
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert llm.resolve_provider().name == "groq"
    monkeypatch.setenv("LLM_PROVIDER", "auto"); monkeypatch.setenv("LLM_AUTO_ORDER", "groq,gemini")
    assert llm.resolve_provider().name == "groq"
    monkeypatch.setenv("LLM_PROVIDER", "none")
    assert llm.resolve_provider() is None


def test_model_and_endpoint_overridable(monkeypatch):
    monkeypatch.setenv("ALLOW_EXTERNAL_LLM", "1"); monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    assert llm.resolve_provider("gemini").model == "gemini-test-model"
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1"); monkeypatch.setenv("LLM_MODEL", "m")
    p = llm.resolve_provider("openai")
    assert p.base_url == "https://example.test/v1" and p.model == "m"


def test_local_ollama_needs_no_consent(monkeypatch):
    monkeypatch.setattr(llm, "_ollama_models", lambda host: ["qwen2.5:3b-instruct"])
    p = llm.resolve_provider()
    assert p.name == "ollama" and p.local
    monkeypatch.setenv("OLLAMA_MODEL", "other:7b")
    assert llm.resolve_provider("ollama") is None, "model must actually be pulled"


class Resp:
    def __init__(self, status, body="", headers=None):
        self.status_code, self._b, self.headers, self.text = status, body, headers or {}, str(body)

    def json(self):
        return self._b

    def raise_for_status(self):
        if self.status_code >= 400:
            raise llm.requests.HTTPError(f"HTTP {self.status_code}")


def _ok(txt='{"ok": true}'):
    return Resp(200, {"choices": [{"message": {"content": txt}}]})


P = llm.Provider("gemini", "openai", "https://x/v1beta/openai", "m", "k", local=False)


def test_rate_limit_is_retried_with_backoff(monkeypatch):
    calls = []
    seq = [Resp(429, "slow down", {"retry-after": "1"}), Resp(503), _ok()]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: calls.append(1) or seq.pop(0))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    assert llm.chat_json(P, "s", "u", retries=0) == {"ok": True}
    assert len(calls) == 3


def test_json_mode_rejected_falls_back(monkeypatch):
    payloads = []
    def post(url, headers=None, json=None, timeout=None):  # noqa: A002
        payloads.append(dict(json))
        return Resp(400, "response_format is not supported") if "response_format" in json else _ok()
    monkeypatch.setattr(llm.requests, "post", post)
    assert llm.chat_json(P, "s", "u", retries=0) == {"ok": True}
    assert "response_format" in payloads[0] and "response_format" not in payloads[1]


def test_check_reports_ok(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _ok())
    assert llm.check(P) == "OK"


def test_circuit_breaker_moves_failed_provider_to_the_back(monkeypatch):
    llm.reset_health()
    monkeypatch.setenv("ALLOW_EXTERNAL_LLM", "1"); monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GEMINI_MODELS", "m-a,m-b,m-c")
    first = [p.model for p in llm.generative_chain("gemini")]
    assert first == ["m-a", "m-b", "m-c"] and all(p.tries == 2 for p in llm.generative_chain("gemini")), "fail fast when alternatives exist"
    llm.mark_unhealthy(llm.generative_chain("gemini")[0], 600, "quota")
    assert [p.model for p in llm.generative_chain("gemini")] == ["m-b", "m-c", "m-a"], "a failed model drops to the end, not out of the chain"
    llm.reset_health()
    assert [p.model for p in llm.generative_chain("gemini")] == ["m-a", "m-b", "m-c"]


def test_http_errors_are_not_retried_twice(monkeypatch):
    calls = []
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: calls.append(1) or Resp(503))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    p = llm.Provider("gemini", "openai", "https://x/v1beta/openai", "m", "k", local=False, tries=2)
    with pytest.raises(Exception):
        llm.chat_json(p, "s", "u", retries=1)
    assert len(calls) == 2, "2 HTTP attempts total (not 2 x tries): the chain handles the rest"
