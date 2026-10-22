"""Decider-based judging (Jev or the local Hugging Face NLI model): retrieve the most relevant transcript lines for each
checklist question, ask the decider for a yes/no probability, accept confident answers, and escalate unsure ones to a
generative LLM.

  question -> retrieval (local embeddings, AI-generated hints) -> short excerpts -> decider probability
     p >= JEV_YES_AT  -> yes     p <= JEV_NO_AT -> no     otherwise -> escalate (or 'partly' if no escalation provider)

Evidence for a 'yes' is the most relevant retrieved transcript line (verifiable by construction). Output has the same shape
as judge.judge(), so scoring and the UI are unchanged. Every stage is timed (see "timings" in the result)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from . import judge, llm, nli, perf, privacy, probes, retrieval, transcript
from .judge import ANSWER_VALUE, CHECKLIST

log = perf.log

# Parts of the class a question can only be answered from (everything else searches the whole transcript).
SCOPE = {
    **{f"opening_rapport.{i}": "opening" for i in CHECKLIST["opening_rapport"]["items"]},
    "individual_attention.new_learner": "opening",
    **{f"structure_closing.{i}": "closing" for i in ("summary", "follow_up", "preview", "goodbye")},
}
HEADER = "Excerpts from an online class for children (automatic transcript, speakers not labelled):\n"
TECH_QUERY = "audio or video problems interrupting the class: can you hear me, I can't see, microphone or camera not working, connection issues"
TECH_QUESTION = "Does this show an audio, video or connection problem interrupting the class (for example asking whether people can hear or see)?"
TECH_HYPOTHESIS = "An audio, video or connection problem interrupted the class."


def _thresholds() -> tuple[float, float]:
    return float(os.environ.get("JEV_YES_AT", "0.65")), float(os.environ.get("JEV_NO_AT", "0.35"))


def _probability(provider: llm.Provider, excerpts: str, qid: str, question: str, hypothesis: str) -> Optional[float]:
    """Probability of 'yes' from whichever decider is configured (hosted Jev or local NLI)."""
    if provider.kind == "nli":
        return nli.decide(provider.model, excerpts, hypothesis)
    return llm.decide(provider, HEADER + excerpts, {qid: question + " Answer yes only if the excerpts clearly show it."}).get(qid)


def _item(dim: str, iid: str, answer: str, evidence: Optional[list] = None, prob: Optional[float] = None, source: str = "jev") -> dict:
    _q, pos, neg, reverse = CHECKLIST[dim]["items"][iid]
    value = ANSWER_VALUE.get(answer)
    if value is not None and reverse:
        value = 1.0 - value
    return {"id": iid, "answer": answer, "value": value, "reverse": reverse, "positive_text": pos, "negative_text": neg,
            "evidence": evidence or [], "unverified": False, "prob": prob, "source": source}


def judge_with_decider(feats: dict, segs: list, provider: llm.Provider, escalate=None,
                       progress: Optional[Callable[[str], None]] = None, index: Optional[retrieval.Index] = None,
                       workers: int = 4, hint_provider: Optional[llm.Provider] = None) -> dict:
    say = progress or (lambda m: None)
    tm: dict[str, float] = {}
    lat = perf.Latency()
    segs = transcript.usable(segs)
    meta = {"provider": provider.name, "model": provider.model, "local": provider.local, "escalated_to": None}
    if not segs:
        log.warning("decider: no usable transcript segments")
        return {"usable": False, "dimensions": {}, "tech_issues": None, **meta}
    names: set[str] = set()
    if not provider.local:  # local deciders (hf-nli) never send text anywhere
        if not privacy.available():
            raise RuntimeError("De-identification model unavailable; refusing to send transcript to a cloud service.")
        with perf.stage("deidentify", tm):
            names = privacy.collect_names([s.text for s in segs])
    mask = (lambda t: privacy.mask(t, names)) if names else (lambda t: t)

    say("Indexing the transcript for retrieval...")
    with perf.stage("index", tm, segments=len(segs)):
        idx = index or retrieval.Index(segs)
    wins = idx.windows()
    yes_at, no_at = _thresholds()
    all_ids = judge.question_ids(list(CHECKLIST))
    questions = {qid: CHECKLIST[qid.split(".", 1)[0]]["items"][qid.split(".", 1)[1]][0] for qid in all_ids}
    with perf.stage("hints", tm):
        hints, hint_source = probes.get_hints(hint_provider or escalate, questions, say)
    meta["retrieval_hints"] = hint_source

    def ask(qid: str):
        dim, iid = qid.split(".", 1)
        scope = SCOPE.get(qid, "any")
        if scope != "any" and scope not in wins and "full" not in wins:
            log.info("question=%s skipped (scope '%s' not transcribed)", qid, scope)
            return qid, None, [], "scope"
        question = CHECKLIST[dim]["items"][iid][0]
        with lat.time("retrieve"):
            hits = idx.search(hints.get(qid, question), k=4, scope=scope)
        if not hits:
            return qid, None, [], "scope"
        excerpts = mask("\n".join(h.text for h in hits))
        hypothesis = CHECKLIST[dim]["items"][iid][1]  # the declarative "observed" statement, e.g. "Homework or practice was assigned."
        with lat.time("decide"):
            p = _probability(provider, excerpts, qid, question, hypothesis)
        log.debug("question=%s p=%s excerpt_chars=%d", qid, None if p is None else round(p, 3), len(excerpts))
        return qid, p, hits, "ok"

    say(f"{provider.name} ({provider.model}): deciding {len(all_ids)} checklist questions...")
    # A local model already uses every CPU core: parallel requests would only fight over them. Hosted deciders parallelise.
    with perf.stage("decisions", tm, questions=len(all_ids), workers=1 if provider.kind == "nli" else workers):
        with ThreadPoolExecutor(max_workers=1 if provider.kind == "nli" else workers) as ex:
            results = list(ex.map(ask, all_ids))

    items: dict[str, dict] = {}
    uncertain: list[str] = []
    for qid, prob, hits, status in results:
        dim, iid = qid.split(".", 1)
        if status == "scope" or prob is None:
            items[qid] = _item(dim, iid, "unknown", prob=prob)
            continue
        line = idx.best_line(hits)
        ev = [{"t": round(line.start, 1), "text": line.text if len(line.text) <= 160 else line.text[:157] + "..."}] if line else []
        if prob >= yes_at:
            items[qid] = _item(dim, iid, "yes", ev, prob)
        elif prob <= no_at:
            items[qid] = _item(dim, iid, "no", [], prob)
        else:
            items[qid] = _item(dim, iid, "partly", ev, prob)  # provisional; replaced if escalation answers
            uncertain.append(qid)
    counts = {a: sum(1 for i in items.values() if i["answer"] == a) for a in ("yes", "partly", "no", "unknown")}
    log.info("decider answers %s | uncertain=%s", counts, uncertain)

    n_escalated = 0
    chain = [escalate] if isinstance(escalate, llm.Provider) else list(escalate or [])
    chain = [c for c in chain if c.kind in ("ollama", "openai")]
    if uncertain and chain:
        with perf.stage("escalation", tm, unsure=len(uncertain)):
            for gen in chain:  # first provider that still has quota answers the unsure questions
                say(f"Escalating {len(uncertain)} unsure answers to {gen.name}/{gen.model}...")
                try:
                    sub = judge.judge(feats, segs, gen, say, only=set(uncertain))
                except Exception as e:
                    log.warning("escalation to %s/%s failed: %s", gen.name, gen.model, e)
                    say(f"Escalation to {gen.name}/{gen.model} failed ({e}); trying the next provider.")
                    continue
                for qid in uncertain:
                    dim, iid = qid.split(".", 1)
                    got = {i["id"]: i for i in (sub.get("dimensions", {}).get(dim) or {}).get("items", [])}.get(iid)
                    if got and got["answer"] != "unknown":
                        got["prob"], got["source"] = items[qid]["prob"], gen.name
                        items[qid] = got
                        n_escalated += 1
                meta["escalated_to"] = f"{gen.name}/{gen.model}"
                break
    elif uncertain:
        log.info("decider: %d unsure answers kept as 'partly' (no generative provider to escalate to)", len(uncertain))

    dims: dict[str, dict] = {}
    for qid in all_ids:
        dim, _ = qid.split(".", 1)
        dims.setdefault(dim, {"items": []})["items"].append(items[qid])

    # Technical problems that interrupted the class: judge the few most relevant lines on their own.
    tech: Optional[dict] = {"count": 0, "evidence": []}
    with perf.stage("tech_check", tm):
        try:
            hits = idx.search(TECH_QUERY, k=5)
            lines = sorted([(sim, s) for h in hits for s, sim in h.lines], key=lambda x: -x[0])[:5]
            if lines:
                probs = [_probability(provider, mask(s.text), "tech", TECH_QUESTION, TECH_HYPOTHESIS) or 0.0 for _, s in lines]
                yes = [(p, s) for p, (_, s) in zip(probs, lines) if p >= 0.7]
                tech = {"count": len(yes), "evidence": [{"t": round(yes[0][1].start, 1), "text": yes[0][1].text}] if yes else []}
        except Exception as e:
            log.warning("tech check failed: %s", e)
            tech = None

    tm["per_question"] = lat.summary()
    log.info("decider timings %s", tm)
    return {"usable": True, "dimensions": dims, "tech_issues": tech, "decided": len(all_ids) - len(uncertain),
            "uncertain": len(uncertain), "escalated": n_escalated, "timings": tm, **meta}
