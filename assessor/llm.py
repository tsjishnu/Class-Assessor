"""Pluggable LLM access for the transcript judge. Switch providers with ONE setting (LLM_PROVIDER) or in the UI.

Presets (all free tiers / local):
  gemini  - Google Gemini via its OpenAI-compatible endpoint.        needs GEMINI_API_KEY
  groq    - Groq (open models such as Llama 3.3 70B).                 needs GROQ_API_KEY
  openai  - any OpenAI-compatible API.                                needs LLM_BASE_URL, LLM_MODEL (+ LLM_API_KEY)
  ollama  - local, private, free; nothing leaves the machine.        needs `ollama serve` + a pulled model
  jev     - TypeSafe Jev: a fast, cheap yes/no "decider" returning probabilities (no text, no quotes). Used with
            retrieval; uncertain answers are escalated to the generative provider above. needs JEV_API_KEY
            (direct) or OPENROUTER_API_KEY (JEV_ENDPOINT=openrouter). Paid per use (very small).
  hf-nli  - open Hugging Face decision model (DeBERTa-v3 zero-shot NLI), runs LOCALLY on CPU, free and private. Same
            decider role as Jev: confident answers accepted, unsure ones escalated. needs `pip install transformers torch`.

Reliability: generative providers form a fallback CHAIN (the preferred one first, then the rest of LLM_AUTO_ORDER, with
GEMINI_MODELS rotated because free-tier quotas are per model). A quota/rate error moves on to the next link.

Cloud providers receive de-identified transcript TEXT ONLY (never audio, video or frames) and only when
ALLOW_EXTERNAL_LLM=1 is set (an explicit operator consent switch). LLM_PROVIDER=auto picks the first configured provider
in LLM_AUTO_ORDER (default: gemini,groq,openai,ollama)."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests

from . import config, perf  # noqa: F401  (config loads .env)

DEFAULT_ORDER = "gemini,groq,openai,ollama"  # jev is never auto-selected: it is a decider, opt in with LLM_PROVIDER=jev


@dataclass
class Provider:
    name: str
    kind: str            # "ollama" | "openai" (generative) | "jev" | "nli" (deciders)
    base_url: str
    model: str
    api_key: str = ""
    local: bool = True
    max_chars: int = 9000   # transcript budget per call (small local models need less)
    timeout: int = 900
    path: str = "/chat/completions"   # request path (the Jev decisions endpoint differs between hosts)
    extra: Optional[dict] = None       # extra request params, e.g. {"reasoning_effort": "low"} for thinking models
    tries: int = 4                     # HTTP attempts on 429/5xx; lowered to 2 when other providers can take over (fail fast)


class QuotaExhausted(RuntimeError):
    """The provider's (daily) quota is used up; retrying the same provider is pointless."""

    def __init__(self, msg: str, retry_after_s: Optional[float] = None):
        super().__init__(msg)
        self.retry_after_s = retry_after_s


# Circuit breaker: a provider that just failed (quota spent, overloaded) is skipped for a cooldown instead of being retried
# for every class. Unhealthy providers stay in the chain as a last resort.
_cooldown: dict[tuple[str, str], float] = {}


def mark_unhealthy(p: Provider, seconds: float, reason: str) -> None:
    _cooldown[(p.name, p.model)] = time.time() + seconds
    perf.log.warning("circuit breaker: skipping %s/%s for %.0fs (%s)", p.name, p.model, seconds, reason)


def is_healthy(p: Provider) -> bool:
    return time.time() >= _cooldown.get((p.name, p.model), 0.0)


def reset_health() -> None:
    _cooldown.clear()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def consent() -> bool:
    return _env("ALLOW_EXTERNAL_LLM", "0") == "1"


def _ollama_models(host: str) -> Optional[list[str]]:
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def _cloud(name: str, base: str, model: str, key: str, kind: str = "openai", path: str = "/chat/completions",
           extra: Optional[dict] = None) -> Optional[Provider]:
    """Cloud presets need a key, a model and the operator's explicit consent."""
    if not (key and model and base and consent()):
        return None
    return Provider(name, kind, base.rstrip("/"), model, key, local=False,
                    max_chars=int(_env("LLM_MAX_CHARS", "30000")), timeout=120, path=path, extra=extra)


def gemini_models() -> list[str]:
    """Models to try in order. Free-tier quotas are per model, so extra models extend the daily capacity."""
    listed = [m.strip() for m in _env("GEMINI_MODELS").split(",") if m.strip()]
    return listed or [_env("GEMINI_MODEL", "gemini-3.5-flash")]


def _build(name: str, model: Optional[str] = None) -> Optional[Provider]:
    if name == "hf-nli":
        import importlib.util

        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
            return None
        return Provider("hf-nli", "nli", "", _env("HF_NLI_MODEL", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"), local=True, max_chars=0)
    if name == "gemini":
        effort = _env("GEMINI_REASONING_EFFORT", "low")  # thinking models spend output tokens on reasoning; keep it small
        return _cloud("gemini", _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
                      model or gemini_models()[0], _env("GEMINI_API_KEY"),
                      extra={"reasoning_effort": effort} if effort else None)
    if name == "groq":
        return _cloud("groq", _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                      _env("GROQ_MODEL", "llama-3.3-70b-versatile"), _env("GROQ_API_KEY"))
    if name == "openai":
        return _cloud("openai-compatible", _env("LLM_BASE_URL"), _env("LLM_MODEL"), _env("LLM_API_KEY", "none") if _env("LLM_BASE_URL") else "")
    if name == "jev":
        if _env("JEV_ENDPOINT", "typesafe").lower() == "openrouter":
            base, path, model = _env("JEV_BASE_URL", "https://openrouter.ai/api/alpha"), _env("JEV_PATH", "/decisions"), _env("JEV_MODEL", "typesafe/jev-1.13")
            key = _env("JEV_API_KEY") or _env("OPENROUTER_API_KEY")
        else:
            base, path, model = _env("JEV_BASE_URL", "https://api.typesafe.ai"), _env("JEV_PATH", "/v1/systemone"), _env("JEV_MODEL", "jev-latest")
            key = _env("JEV_API_KEY")
        return _cloud("jev", base, model, key, kind="jev", path=path)
    if name == "ollama":
        host = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
        host = host if host.startswith("http") else "http://" + host
        want = _env("OLLAMA_MODEL", "qwen2.5:3b-instruct")
        models = _ollama_models(host)
        if models is None or want not in models:
            return None
        # A small local model on a weak CPU cannot label a whole class in reasonable time: cap the wait, then degrade gracefully.
        return Provider("ollama", "ollama", host, want, local=True, max_chars=int(_env("LLM_MAX_CHARS", "9000")), timeout=int(_env("OLLAMA_TIMEOUT", "300")))
    return None


def available_providers() -> dict[str, Provider]:
    """Every provider that is currently usable, keyed by preset name."""
    out = {}
    for n in ("gemini", "groq", "openai", "ollama", "jev", "hf-nli"):
        p = _build(n)
        if p:
            out[n] = p
    return out


def generative_chain(prefer: Optional[str] = None) -> list[Provider]:
    """Generative providers in fallback order: the preferred preset first (all its models), then the rest of
    LLM_AUTO_ORDER. LLM_FALLBACK=0 keeps only the preferred preset."""
    order = [x.strip() for x in _env("LLM_AUTO_ORDER", DEFAULT_ORDER).split(",") if x.strip()]
    first = (prefer or _env("LLM_PROVIDER", "auto")).lower()
    names = ([first] if first in ("gemini", "groq", "openai", "ollama") else []) + [n for n in order if n != first]
    if _env("LLM_FALLBACK", "1") == "0" and first in ("gemini", "groq", "openai", "ollama"):
        names = [first]
    chain: list[Provider] = []
    for n in names:
        if n == "gemini":
            for m in gemini_models():
                p = _build("gemini", m)
                if p:
                    chain.append(p)
        else:
            p = _build(n)
            if p and p.kind in ("ollama", "openai"):
                chain.append(p)
    if len(chain) > 1:  # other providers can take over, so do not spend long retrying any single one
        for p in chain:
            p.tries = 2
    chain.sort(key=lambda p: not is_healthy(p))  # stable: healthy providers first, cooling-down ones last
    return chain


def resolve_escalation(decider: Optional[Provider] = None) -> Optional[Provider]:
    """Generative provider that takes over the questions a decider (Jev / hf-nli) is unsure about.
    ESCALATE_TO (alias JEV_ESCALATE_TO) = auto (default: first configured generative provider) | <preset> | none."""
    choice = (_env("ESCALATE_TO") or _env("JEV_ESCALATE_TO", "auto")).lower()
    if choice == "none":
        return None
    if choice != "auto":
        p = _build(choice)
        return p if p and p.kind in ("ollama", "openai") else None
    chain = generative_chain("auto")
    return chain[0] if chain else None


def resolve_hint_provider() -> Optional[Provider]:
    """Provider that writes the (cached, generic) retrieval hints. Prefers a LOCAL model so it never spends cloud quota."""
    chain = generative_chain("auto")
    local = [p for p in chain if p.local]
    return (local or chain or [None])[0]


def resolve_provider(prefer: Optional[str] = None) -> Optional[Provider]:
    choice = (prefer or _env("LLM_PROVIDER", "auto")).lower()
    if choice == "none":
        return None
    if choice != "auto":
        return _build(choice)
    for n in [x.strip() for x in _env("LLM_AUTO_ORDER", DEFAULT_ORDER).split(",") if x.strip()]:
        p = _build(n)
        if p:
            return p
    return None


def explain_unavailable(name: str) -> str:
    """Human-readable reason a preset is not usable (for the CLI check / UI)."""
    if name == "hf-nli":
        return "install `pip install transformers torch` (the model downloads on first use, ~370 MB)."
    if name in ("gemini", "groq", "jev"):
        key = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "jev": "JEV_API_KEY (or OPENROUTER_API_KEY with JEV_ENDPOINT=openrouter)"}[name]
        if name == "jev":
            if not (_env("JEV_API_KEY") or (_env("JEV_ENDPOINT").lower() == "openrouter" and _env("OPENROUTER_API_KEY"))):
                return f"{key} is not set."
            if not consent():
                return "ALLOW_EXTERNAL_LLM=1 is not set (explicit consent is required to send transcript text to a cloud API)."
            return "not configured."
        if not _env(key):
            return f"{key} is not set."
        if not consent():
            return "ALLOW_EXTERNAL_LLM=1 is not set (explicit consent is required to send transcript text to a cloud API)."
    if name == "openai" and not _env("LLM_BASE_URL"):
        return "LLM_BASE_URL is not set."
    if name == "ollama":
        return "Ollama is not running or the model is not pulled (try `ollama serve` and `ollama pull qwen2.5:3b-instruct`)."
    return "not configured."


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in model output")
    return json.loads(text[a:b + 1])


def _retry_delay_s(text: str) -> Optional[float]:
    """Seconds from a provider message like 'Please retry in 14h3m42.5s' (None if absent)."""
    m = re.search(r"retry in\s+((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)", text, re.IGNORECASE)
    if not m or not m.group(1):
        return None
    h = re.search(r"(\d+)h", m.group(1))
    mi = re.search(r"(\d+)m", m.group(1))
    sec = re.search(r"([\d.]+)s", m.group(1))
    return (int(h.group(1)) * 3600 if h else 0) + (int(mi.group(1)) * 60 if mi else 0) + (float(sec.group(1)) if sec else 0.0)


def _is_quota_exhausted(r: requests.Response) -> bool:
    """A 429 whose suggested wait is long (a spent daily quota), as opposed to a short per-minute rate limit."""
    if r.status_code != 429:
        return False
    delay = _retry_delay_s(getattr(r, "text", "") or "")
    return delay is not None and delay > 120


def _post_with_backoff(url: str, headers: dict, payload: dict, timeout: int, tries: int = 4) -> requests.Response:
    """POST with exponential backoff on rate limits / transient errors (free tiers return 429 and 503).
    A spent daily quota is not retried: the caller falls through to the next provider."""
    delay = 5.0 if tries > 2 else 3.0
    for attempt in range(tries):
        t0 = time.perf_counter()
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        dt = time.perf_counter() - t0
        perf.LATENCY.add("http." + url.split("/")[2], dt)
        perf.log.info("http POST %s status=%s sec=%.2f attempt=%d/%d", url, r.status_code, dt, attempt + 1, tries)
        if _is_quota_exhausted(r):
            perf.log.warning("quota exhausted (provider suggests retry in %.0fs): %s", _retry_delay_s(r.text) or 0, url)
            return r
        if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            wait = float(r.headers.get("retry-after", delay)) if str(r.headers.get("retry-after", "")).replace(".", "").isdigit() else delay
            perf.log.warning("HTTP %s from %s: backing off %.0fs before retry", r.status_code, url, min(wait, 60))
            time.sleep(min(wait, 60))
            delay *= 2
            continue
        return r
    return r  # pragma: no cover


def decide(p: Provider, state: str, questions: dict[str, str]) -> dict[str, float]:
    """Ask a decider (Jev) yes/no questions about `state`. Returns {question_id: probability of yes in 0..1}."""
    body = {"model": p.model, "state": state,
            "questions": {qid: {"type": "noul", "instructions": text} for qid, text in questions.items()}}
    r = _post_with_backoff(f"{p.base_url}{p.path}", {"Authorization": f"Bearer {p.api_key}", "Content-Type": "application/json"}, body, p.timeout)
    r.raise_for_status()
    answers = r.json().get("answers", {})
    return {qid: float(answers[qid]["noul"]) for qid in questions if qid in answers and "noul" in answers[qid]}


def _chat_json_impl(p: Provider, system: str, user: str, retries: int = 1, max_tokens: int = 1400) -> dict:
    if p.kind in ("jev", "nli"):
        raise RuntimeError(f"{p.name} is a yes/no decider, not a text generator; use it via assessor.decider")
    last: Exception = RuntimeError("no attempt")
    for attempt in range(retries + 1):
        try:
            if p.kind == "ollama":
                r = requests.post(
                    f"{p.base_url}/api/chat",
                    json={"model": p.model, "stream": False, "format": "json",
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                          "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": max_tokens}},
                    timeout=p.timeout,
                )
                r.raise_for_status()
                return _parse_json(r.json()["message"]["content"])
            headers = {"Authorization": f"Bearer {p.api_key}"}
            # Thinking models count reasoning tokens against the limit, so cloud calls get a generous budget.
            payload = {"model": p.model, "temperature": 0.1, "max_tokens": max(max_tokens, 6000), "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **(p.extra or {})}
            r = _post_with_backoff(f"{p.base_url}/chat/completions", headers, payload, p.timeout, p.tries)
            if r.status_code == 400 and any(k in r.text for k in ("response_format", "reasoning_effort")):
                for k in ("response_format", "reasoning_effort"):  # some endpoints reject these optional params: retry without
                    if k in r.text:
                        payload.pop(k, None)
                r = _post_with_backoff(f"{p.base_url}/chat/completions", headers, payload, p.timeout, p.tries)
            if _is_quota_exhausted(r):
                raise QuotaExhausted(f"{p.name}/{p.model}: free quota used up", _retry_delay_s(r.text))
            r.raise_for_status()
            return _parse_json(r.json()["choices"][0]["message"]["content"])
        except (QuotaExhausted, requests.HTTPError, requests.ConnectionError, requests.Timeout):
            raise  # HTTP-level failures were already retried with backoff; the caller falls through to the next provider
        except Exception as e:  # retry once on malformed / non-JSON model output
            last = e
    raise RuntimeError(f"LLM call failed: {last}")


def chat_json(p: Provider, system: str, user: str, retries: int = 1, max_tokens: int = 1400) -> dict:
    """Ask a generative provider for a JSON object. Timed and logged (prompt size, latency, outcome)."""
    t0 = time.perf_counter()
    status = "ok"
    try:
        return _chat_json_impl(p, system, user, retries, max_tokens)
    except Exception as e:
        status = type(e).__name__
        raise
    finally:
        dt = time.perf_counter() - t0
        perf.LATENCY.add(f"llm.{p.name}", dt)
        perf.log.info("llm call provider=%s model=%s prompt_chars=%d sec=%.1f status=%s", p.name, p.model, len(system) + len(user), dt, status)


def check(p: Provider) -> str:
    """Round-trip test used by scripts/check_llm.py: verifies key, model name and JSON output."""
    if p.kind == "jev":
        prob = decide(p, "The sky is blue and the grass is green.", {"q": "Does the text say the sky is blue?"}).get("q")
        return "OK" if prob is not None and prob > 0.5 else f"unexpected probability: {prob}"
    data = chat_json(p, "Reply with JSON only.", 'Return {"ok": true}', retries=0, max_tokens=30)
    return "OK" if data.get("ok") in (True, "true") else f"unexpected reply: {data}"
