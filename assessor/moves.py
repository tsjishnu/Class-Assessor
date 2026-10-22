"""Teaching-moves judge (hybrid architecture inspired by ClassMind / TalkMoves research).

  1. LABEL    An LLM reads the transcript in short segments (~2 min) and labels *teaching-move events* (greeting, praise,
              probing question, homework ...) each with a verbatim quote, plus per-segment activity and energy.
              One batched request per class for large models; a few small requests for small local models.
  2. CHECK    Every quote must exist in the transcript (hallucinated evidence is dropped, timestamps are corrected), and a
              local NLI decision model verifies that the quoted lines really support the claimed move.
  3. COUNT    Dimension scores come from FREQUENCIES per observed minute (e.g. praise per minute against what a strong class
              shows), not from "did it ever happen" yes/no answers, which saturate: one "No problem" is not a warm class.

Output has the same item structure as judge.judge(), so rubric.assess() and the UI are unchanged."""
from __future__ import annotations

import math
import os
from typing import Callable, Optional

from . import judge, llm, nli, perf, privacy, transcript
from .transcript import fmt_t, parse_t

log = perf.log
VERSION = "v3-dense-events"  # bump when prompts or scoring change: it is part of the judged-result cache key

# type id -> (definition shown to the labelling model, declarative hypothesis for the NLI verifier)
TYPES: dict[str, tuple[str, str]] = {
    "greeting": ("Teacher welcomes or greets the learners.", "The teacher greets the learners."),
    "rapport_checkin": ("Teacher asks learners something personal or social (how they are, their day, hobbies), not about the lesson.", "The teacher asks the learners something personal, like how they are or how their day was."),
    "agenda": ("Teacher states what the class will cover or the plan for the session.", "The teacher says what the class will cover."),
    "recap_previous": ("Teacher links to earlier learning (last class, a quick revision of what was done).", "The teacher refers back to what was learned in an earlier class."),
    "names_learner": ("Teacher addresses a specific learner by name.", "The teacher addresses a learner by name."),
    "question_directed": ("Teacher asks a question aimed at one specific learner.", "The teacher asks a specific learner a question."),
    "question_open": ("Teacher asks an open/probing question (why, how, what if) that makes learners think.", "The teacher asks a why or how question that makes learners think."),
    "question_closed": ("Teacher asks a short question with a single expected answer.", "The teacher asks a short question that has one correct answer."),
    "praise_specific": ("Teacher praises a learner and says what was good about it.", "The teacher praises a learner and says what was good."),
    "praise_generic": ("Teacher gives clear praise without detail (very good, well done, excellent).", "The teacher praises a learner."),
    "learner_mistake": ("A learner answers wrongly or makes a mistake.", "A learner gives a wrong answer or makes a mistake."),
    "kind_response_to_mistake": ("Teacher responds to a mistake kindly and constructively (not just 'no').", "The teacher responds kindly to a learner's mistake."),
    "explanation_reasoning": ("Teacher explains WHY something is so or HOW it works, not just the answer.", "The teacher explains why something is true or how it works."),
    "feedback_specific": ("Teacher gives specific feedback on a learner's answer beyond right/wrong.", "The teacher gives specific feedback on a learner's answer."),
    "check_understanding": ("Teacher checks understanding (any doubts? did you get it?).", "The teacher asks whether the learners understand or have doubts."),
    "new_learner_welcome": ("Teacher welcomes a newly joined learner or asks them to introduce themselves.", "The teacher welcomes a new learner or asks them to introduce themselves."),
    "learner_response_long": ("A learner answers in a full sentence or explains their thinking.", "A learner answers in a full sentence or explains their thinking."),
    "summary_recap": ("Teacher summarises what was learned (usually near the end).", "The teacher summarises what was learned."),
    "homework_assigned": ("Teacher assigns homework, practice or next steps.", "The teacher assigns homework or practice."),
    "next_class_preview": ("Teacher says what the next class covers or when it happens.", "The teacher says what the next class will cover or when it is."),
    "goodbye": ("Teacher says goodbye or formally ends the class.", "The teacher says goodbye."),
    "harsh_tone": ("Teacher sounds irritated, dismissive, sarcastic or discouraging.", "The teacher is irritated, dismissive or discouraging."),
    "tech_problem": ("Audio, video or connection problem interrupts the class (can you hear me? screen not shared).", "There is an audio, video or connection problem."),
    "off_task": ("Time is lost to confusion, long waiting or off-topic talk.", "Time is being wasted on waiting, confusion or off-topic talk."),
}
ACTIVITIES = ["puzzle", "game", "video", "lecture", "discussion", "waiting", "other"]

SYSTEM = (
    "You label teaching moves in transcripts of live online classes for children (any subject). The transcript is automatic "
    "speech recognition: it has errors, mangled names and no speaker labels, so infer who is speaking (teacher or learners) "
    "from the dialogue. Be STRICT and precise: label an event only when it is a clear instance. Routine acknowledgements "
    "('okay', 'yes', 'right', 'no problem', 'next') are NOT praise, encouragement or feedback. A generic 'good' counts only as "
    "praise_generic. Narration from a pre-recorded video or lesson being played is not the live teacher: label such "
    "segments activity 'video' and do not count its sentences as teacher moves. Each event needs a quote copied word-for-word "
    "from the transcript line that shows it. Answer with JSON only."
)


def _types_text() -> str:
    return "\n".join(f"- {k}: {v[0]}" for k, v in TYPES.items())


# ----------------------------------------------------------------------------------------------- segmentation

def build_segments(segs: list, windows: list[dict], seg_len: float = 120.0) -> list[dict]:
    """Group usable transcript segments into ~seg_len buckets inside each analysed window."""
    segs = transcript.usable(segs)
    out: list[dict] = []
    for w in windows:
        in_w = [s for s in segs if s.window == w["name"]]
        if not in_w:
            continue
        t = w["start"]
        while t < w["end"]:
            part = [s for s in in_w if t <= s.start < t + seg_len]
            if part:
                out.append({"id": len(out) + 1, "start": t, "end": min(t + seg_len, w["end"]), "window": w["name"], "segs": part})
            t += seg_len
    return out


def _render(seg: dict, masker=None) -> str:
    lines = "\n".join(f"[{fmt_t(s.start)}] {masker(s.text) if masker else s.text}" for s in seg["segs"])
    return f"### SEGMENT {seg['id']} ({seg['window']}, {fmt_t(seg['start'])}-{fmt_t(seg['end'])})\n{lines}"


def _batches(segments: list[dict], max_chars: int, masker=None) -> list[list[dict]]:
    batches, cur, size = [], [], 0
    for sg in segments:
        n = len(_render(sg, masker)) + 2
        if cur and size + n > max_chars:
            batches.append(cur)
            cur, size = [], 0
        cur.append(sg)
        size += n
    if cur:
        batches.append(cur)
    return batches


# ----------------------------------------------------------------------------------------------- labelling

def _prompt(batch: list[dict], masker=None) -> str:
    empty = ", ".join(f'"{k}": []' for k in list(TYPES)[:3]) + ", ..."
    return (
        "TRANSCRIPT SEGMENTS:\n" + "\n\n".join(_render(s, masker) for s in batch) +
        "\n\nMOVE TYPES (use these ids only):\n" + _types_text() +
        "\n\nGo through EVERY segment and consider EVERY move type for it. Per segment write a short note FIRST (what is happening, max 15 "
        f"words), its activity ({'|'.join(ACTIVITIES)}), the teacher's energy (low|medium|high), then a 'moves' object that has ALL move type ids as "
        "keys. For each type give a list of verbatim quotes, one per clear instance (max 4), or an empty list [] if there is none. "
        "Do not skip a type just because it is rare: check each segment line by line.\n"
        'Return JSON exactly like: {"segments": [{"id": 1, "note": "...", "activity": "puzzle", "energy": "medium", '
        '"moves": {"greeting": ["Hello everyone"], ' + empty + '}}], "new_learner_present": false, '
        '"session_type": "regular_class"}. session_type is regular_class (a normal lesson with a teacher and learners), demo_trial (a trial / '
        'enrolment / sales conversation with a parent or prospective student) or other.'
    )


def label(feats: dict, segments: list[dict], provider: llm.Provider, masker, say) -> tuple[list[dict], dict]:
    """Run the LLM labeller. Returns (raw per-segment results, meta). One call per batch."""
    results: list[dict] = []
    new_learner = False
    session_type = None
    batches = _batches(segments, provider.max_chars if provider.max_chars > 0 else 30000, masker)
    for bi, batch in enumerate(batches):
        say(f"AI labelling teaching moves ({provider.name}/{provider.model}), batch {bi + 1}/{len(batches)}...")
        prompt = _prompt(batch, masker)
        log.info("moves label call provider=%s/%s batch=%d/%d segments=%d prompt_chars=%d", provider.name, provider.model, bi + 1, len(batches), len(batch), len(prompt))
        data = llm.chat_json(provider, SYSTEM, prompt, retries=1, max_tokens=8000)
        got = data.get("segments") if isinstance(data.get("segments"), list) else []
        results.extend(g for g in got if isinstance(g, dict))
        new_learner = new_learner or bool(data.get("new_learner_present"))
        session_type = session_type or (str(data.get("session_type", "")).lower() or None)
    return results, {"new_learner_present": new_learner, "batches": len(batches), "session_type": session_type}


# ----------------------------------------------------------------------------------------------- checking

def clean_events(results: list[dict], segments: list[dict], masker=None) -> tuple[list[dict], dict]:
    """Keep only well-formed events whose quote really appears in the transcript (timestamp corrected to the true one).
    When the model saw MASKED text (cloud providers), quotes are matched against the masked text but the evidence shown is the original line."""
    by_id = {s["id"]: s for s in segments}
    all_segs = [x for s in segments for x in s["segs"]]
    _masked: dict[int, object] = {}

    def find(quote, t, seglist):
        if masker is None:
            return judge.verify_quote(quote, t, seglist)
        from dataclasses import replace

        view = []
        for s in seglist:
            if id(s) not in _masked:
                _masked[id(s)] = replace(s, text=masker(s.text))
            view.append(_masked[id(s)])
        v = judge.verify_quote(quote, t, view)
        if v is None:
            return None
        orig = next((s for s in seglist if abs(s.start - v[0]) < 1e-6), None)
        return (v[0], orig.text if orig else v[1])

    events, seg_meta = [], {}
    dropped = {"unknown_type": 0, "quote_not_found": 0, "duplicate": 0}
    seen = set()
    for r in results:
        sid = r.get("id")
        sg = by_id.get(sid)
        if sg is None:
            continue
        act = str(r.get("activity", "other")).lower()
        eng = str(r.get("energy", "")).lower()
        seg_meta[sid] = {"activity": act if act in ACTIVITIES else "other", "energy": eng if eng in ("low", "medium", "high") else None, "note": str(r.get("note", ""))[:120]}
        listed = [{"type": k, "quote": q} for k, qs in (r.get("moves") or {}).items() if isinstance(qs, list) for q in qs[:4] if isinstance(q, str)]
        for e in (listed + list(r.get("events") or []))[:60]:
            if not isinstance(e, dict) or e.get("type") not in TYPES:
                dropped["unknown_type"] += 1
                continue
            v = find(str(e.get("quote") or ""), parse_t(e.get("t")), sg["segs"] or all_segs)
            if v is None:
                v = find(str(e.get("quote") or ""), parse_t(e.get("t")), all_segs)
            if v is None:
                dropped["quote_not_found"] += 1
                continue
            key = (e["type"], round(v[0]))
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
            events.append({"type": e["type"], "t": round(v[0], 1), "text": v[1], "segment": sid, "window": sg["window"], "supported": None, "p": None})
    return events, {"dropped": dropped, "segments": seg_meta}


def verify_events(events: list[dict], segs: list, model: Optional[str], min_p: float, max_checks: int = 60, say=None) -> dict:
    """Local NLI check: does the quoted context entail the claimed move? Unsupported events are dropped from scoring."""
    if not model:
        return {"verified": 0, "dropped": 0, "skipped": "no local verifier"}
    todo = events[:max_checks]
    ordered = sorted(segs, key=lambda s: s.start)
    dropped = 0
    t0 = __import__("time").perf_counter()
    for i, ev in enumerate(todo):
        ctx = " ".join(s.text for s in ordered if ev["t"] - 12 <= s.start <= ev["t"] + 12)[:1500]
        try:
            ev["p"] = round(nli.decide(model, ctx, TYPES[ev["type"]][1]), 3)
        except Exception as e:  # never let verification break the run
            log.warning("verify failed: %s", e)
            return {"verified": i, "dropped": dropped, "skipped": f"verifier error: {e}"}
        ev["supported"] = ev["p"] >= min_p
        dropped += int(not ev["supported"])
        if not ev["supported"]:
            log.info("verify DROP type=%s p=%.2f t=%s text=%r", ev["type"], ev["p"], fmt_t(ev["t"]), ev["text"][:90])
        if say and i % 15 == 0:
            say(f"Verifying claimed teaching moves {i}/{len(todo)}...")
    log.info("verify: checked=%d unsupported=%d min_p=%.2f sec=%.1f", len(todo), dropped, min_p, __import__("time").perf_counter() - t0)
    return {"verified": len(todo), "dropped": dropped, "skipped": None}


# ----------------------------------------------------------------------------------------------- scoring
# "Target" = the rate per observed minute (or count) at which a strong class would earn full marks for that component.
# A priori values from classroom-observation practice, overridable; they are NOT fitted to any sample recordings.
TARGETS = {
    "names_learner": 0.45, "question_directed": 0.6, "praise": 0.4, "explanation_reasoning": 0.35, "feedback_specific": 0.25,
    "learner_response_long": 0.5, "question_open": 0.25,
}


def _cnt(events, types, window=None):
    return [e for e in events if e["type"] in types and e.get("supported") is not False and (window is None or e["window"] in window)]


def _ev(evs, n=2):
    return [{"t": e["t"], "text": e["text"] if len(e["text"]) <= 160 else e["text"][:157] + "..."} for e in evs[:n]]


def _item(iid: str, value: Optional[float], pos: str, neg: str, evidence=None, reverse: bool = False, answer: Optional[str] = None) -> dict:
    if value is None:
        ans = "unknown"
    elif answer:
        ans = answer
    else:
        ans = "yes" if value >= 0.67 else "partly" if value >= 0.34 else "no"
    return {"id": iid, "answer": ans, "value": None if value is None else round(max(0.0, min(1.0, value)), 3), "reverse": reverse,
            "positive_text": pos, "negative_text": neg, "evidence": evidence or [], "unverified": False, "source": "moves"}


def _rate_item(iid, events, types, minutes, target, label_, window=None):
    ev = _cnt(events, types, window)
    n = len(ev)
    rate = n / max(minutes, 1e-6)
    v = min(1.0, rate / target)
    return _item(iid, v, f"{label_}: {n} instance(s) in {minutes:.0f} observed minutes ({rate:.2f}/min).",
                 f"{label_}: only {n} instance(s) in {minutes:.0f} observed minutes ({rate:.2f}/min; a strong class shows about {target:.2f}/min).", _ev(ev))


def score_items(events: list[dict], seg_meta: dict, windows: list[dict], feats: dict, new_learner: bool) -> dict:
    """Frequency-based component values per dimension (same item shape as the checklist judge)."""
    minutes = max(sum(w["end"] - w["start"] for w in windows) / 60.0, 1.0)
    opening_min = max(sum(w["end"] - w["start"] for w in windows if w["name"] in ("opening", "full")) / 60.0, 1.0)
    has_open = any(w["name"] in ("opening", "full") for w in windows)
    has_close = any(w["name"] in ("closing", "full") for w in windows)
    OPEN, CLOSE = ("opening", "full"), ("closing", "full")
    metas = list(seg_meta.values())

    def present(types, window, pos, neg, iid, need=1):
        evs = _cnt(events, types, window)
        return _item(iid, min(1.0, len(evs) / need), pos, neg, _ev(evs))

    dims: dict[str, list[dict]] = {}

    first = (feats.get("transcript_stats") or {}).get("first_speech_s")
    d = [
        present({"greeting"}, OPEN, "The teacher welcomed the learners at the start.", "No clear welcome or greeting was observed at the start.", "greets"),
        present({"rapport_checkin"}, OPEN, "The teacher built rapport with personal check-in questions before the content.", "The class moved to content without any personal check-in with the learners.", "checkin", need=2),
        present({"agenda"}, OPEN, "The teacher set out what the class would cover.", "No agenda or plan for the class was stated.", "agenda"),
        present({"recap_previous"}, OPEN, "The teacher linked the class to earlier learning.", "There was no link to earlier learning.", "recap"),
        present({"learner_response_long"}, OPEN, "Learners spoke freely in the opening, a sign of a relaxed start.", "Learners gave only minimal responses in the opening.", "learner_voice", need=3),
    ]
    if first is not None:
        d.append(_item("prompt_start", 1.0 if first <= 30 else 0.0 if first >= 180 else 1 - (first - 30) / 150,
                       f"The class started promptly (first speech at {fmt_t(first)}).", f"The class started late: first speech only at {fmt_t(first)}."))
    dims["opening_rapport"] = d if has_open else []

    names = _rate_item("names", events, {"names_learner"}, minutes, TARGETS["names_learner"], "Learners addressed by name")
    dq = _rate_item("direct_q", events, {"question_directed"}, minutes, TARGETS["question_directed"], "Questions directed at individual learners")
    voice = _rate_item("participation", events, {"learner_response_long"}, minutes, TARGETS["learner_response_long"], "Learners answering in full sentences")
    d = [names, dq, voice]
    if new_learner:
        d.append(present({"new_learner_welcome"}, None, "A new learner was welcomed and invited to introduce themselves.", "A new learner was not welcomed or given a chance to introduce themselves.", "new_learner"))
    dims["individual_attention"] = d

    praise_ev = _cnt(events, {"praise_specific", "praise_generic"})
    weighted = sum(1.0 if e["type"] == "praise_specific" else 0.4 for e in praise_ev)
    pr_rate = weighted / minutes
    d = [_item("praise", min(1.0, pr_rate / TARGETS["praise"]),
               f"The teacher praised and encouraged learners ({len(praise_ev)} instances; {sum(e['type'] == 'praise_specific' for e in praise_ev)} specific).",
               f"There was little praise or encouragement ({len(praise_ev)} instance(s) in {minutes:.0f} observed minutes).", _ev(sorted(praise_ev, key=lambda e: e["type"] != "praise_specific")))]
    mistakes, kind = _cnt(events, {"learner_mistake"}), _cnt(events, {"kind_response_to_mistake"})
    d.append(_item("kind_mistakes", None if not mistakes else min(1.0, len(kind) / max(len(mistakes), 1)),
                   f"Mistakes were handled kindly and constructively ({len(kind)} of {len(mistakes)}).",
                   f"Mistakes were not handled constructively ({len(kind)} kind responses to {len(mistakes)} mistakes).", _ev(kind)))
    energies = [{"low": 0.0, "medium": 0.5, "high": 1.0}[m["energy"]] for m in metas if m.get("energy")]
    if energies:
        e = sum(energies) / len(energies)
        d.append(_item("warm_tone", e, "The teacher sounded upbeat and energetic across the class.", "The teacher's tone came across as flat or low-energy."))
    harsh = _cnt(events, {"harsh_tone"})
    d.append(_item("harsh", 1.0 - min(1.0, len(harsh) / 2), "The teacher sounded irritated or discouraging at times.", "", _ev(harsh), reverse=True,
                   answer="yes" if len(harsh) >= 2 else "partly" if harsh else "no"))
    dims["tone_energy_warmth"] = d

    op, cl = _cnt(events, {"question_open"}), _cnt(events, {"question_closed"})
    share = len(op) / max(len(op) + len(cl), 1)
    d = [
        _rate_item("reasoning", events, {"explanation_reasoning"}, minutes, TARGETS["explanation_reasoning"], "Explanations of reasoning (the why)"),
        _item("probing", min(1.0, share / 0.3) if (op or cl) else None, f"The teacher asked probing questions ({len(op)} open vs {len(cl)} closed).",
              f"Questions were almost all closed ({len(op)} open vs {len(cl)} closed): few that make learners think.", _ev(op)),
        _rate_item("feedback", events, {"feedback_specific"}, minutes, TARGETS["feedback_specific"], "Specific feedback on learners' answers"),
        present({"check_understanding"}, None, "The teacher checked understanding.", "The teacher did not check whether learners understood.", "checks", need=2),
    ]
    played = sum(1 for m in metas if m["activity"] == "video")
    d.append(_item("live", 1.0 - (played / max(len(metas), 1)) if metas else None, "Teaching was mostly live and interactive.",
                   f"{played} of {len(metas)} observed segments were a played/pre-recorded video rather than live teaching."))
    dims["instruction_feedback"] = d

    acts = {m["activity"] for m in metas if m["activity"] not in ("waiting", "other")}
    off = _cnt(events, {"off_task"})
    waiting = sum(1 for m in metas if m["activity"] == "waiting")
    tech = _cnt(events, {"tech_problem"})
    d = [
        _rate_item("active", events, {"learner_response_long"}, minutes, TARGETS["learner_response_long"], "Learner responses"),
        _item("variety", 0.2 if len(acts) <= 1 else 0.6 if len(acts) == 2 else 1.0, f"The class used a variety of activities ({', '.join(sorted(acts))}).",
              f"The class relied on one type of activity ({', '.join(sorted(acts)) or 'none identified'})."),
        _item("on_task", 1.0 - min(1.0, (len(off) + waiting) / max(len(metas), 1) * 2), "Time was used productively.",
              f"Time was lost to confusion, waiting or off-topic talk ({len(off)} events, {waiting} waiting segments).", _ev(off)),
        _item("tech", 1.0 - min(1.0, len(tech) / 4), f"Technical problems interrupted the class ({len(tech)} times).", "", _ev(tech), reverse=True,
              answer="yes" if len(tech) >= 3 else "partly" if tech else "no"),
    ]
    dims["engagement_variety"] = d

    d = []
    if has_close:
        d = [
            present({"summary_recap"}, CLOSE, "The teacher summarised what was learned.", "There was no summary of the learning at the end.", "summary"),
            present({"homework_assigned"}, CLOSE, "Homework or practice was assigned.", "No homework, practice or next steps were assigned.", "follow_up"),
            present({"next_class_preview"}, CLOSE, "The teacher previewed the next class.", "The next class was not previewed.", "preview"),
            present({"goodbye"}, CLOSE, "The class ended with a proper goodbye.", "The class ended without a clear wrap-up or goodbye.", "goodbye"),
        ]
    dims["structure_closing"] = d
    return dims


# ----------------------------------------------------------------------------------------------- entry point

def rescore(feats: dict, judged: dict, min_p: Optional[float] = None) -> dict:
    """Re-score stored events offline (no LLM, no quota): optionally apply a verifier threshold, then recompute all items."""
    events = [dict(e) for e in judged["events"]]
    if min_p is not None:
        for e in events:
            e["supported"] = True if (min_p <= 0 or e.get("p") is None) else e["p"] >= min_p
    dims = score_items(events, {int(k): v for k, v in judged["segments"].items()}, feats.get("windows", []), feats, judged.get("new_learner_present", False))
    tech_events = _cnt(events, {"tech_problem"})
    out = dict(judged)
    out["dimensions"] = {k: {"items": v} for k, v in dims.items()}
    out["tech_issues"] = {"count": len(tech_events), "evidence": _ev(tech_events, 1)}
    return out


def judge_moves(feats: dict, segs: list, provider: llm.Provider, progress: Optional[Callable[[str], None]] = None) -> dict:
    say = progress or (lambda m: None)
    tm: dict[str, float] = {}
    windows = feats.get("windows", [])
    segments = build_segments(segs, windows)
    meta = {"provider": provider.name, "model": provider.model, "local": provider.local, "mode": "moves"}
    if not segments:
        return {"usable": False, "dimensions": {}, "tech_issues": None, **meta}
    masker = None
    if not provider.local:
        if not privacy.available():
            raise RuntimeError("De-identification model unavailable; refusing to send transcript to a cloud LLM.")
        names = privacy.collect_names([s.text for s in transcript.usable(segs)])
        masker = lambda t: privacy.mask(t, names)  # noqa: E731

    with perf.stage("moves_label", tm, segments=len(segments)):
        raw, lab = label(feats, segments, provider, masker, say)
    with perf.stage("moves_clean", tm):
        events, info = clean_events(raw, segments, masker)
    log.info("moves: %d segments labelled, %d events kept, dropped=%s", len(raw), len(events), info["dropped"])

    min_p = float(os.environ.get("VERIFY_MIN_P", "0.0"))   # 0 = verifier scores are recorded but never used to drop events
    verifier = llm._build("hf-nli") if os.environ.get("VERIFY", "0") == "1" else None
    with perf.stage("moves_verify", tm, events=len(events)):
        ver = verify_events(events, transcript.usable(segs), verifier.model if verifier else None, min_p, say=say)
    kept = [e for e in events if e.get("supported") is not False]
    by_type: dict[str, int] = {}
    for e in kept:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    log.info("moves: verified=%s dropped=%s events_by_type=%s", ver["verified"], ver["dropped"], by_type)

    dims = score_items(events, info["segments"], windows, feats, lab["new_learner_present"])
    tech_events = _cnt(events, {"tech_problem"})
    return {"usable": True, "dimensions": {k: {"items": v} for k, v in dims.items()}, "timings": tm,
            # raw labelled events (with verifier probabilities) are kept for audit and for offline re-scoring (see rescore())
            "events": events, "new_learner_present": lab["new_learner_present"], "session_type": lab.get("session_type"),
            "tech_issues": {"count": len(tech_events), "evidence": _ev(tech_events, 1)},
            "stats": {"events_labelled": len(events), "events_kept": len(kept), "quote_not_found": info["dropped"]["quote_not_found"],
                      "verifier": ver, "events_by_type": by_type, "segments": len(segments), "batches": lab["batches"]},
            "segments": info["segments"], **meta}
