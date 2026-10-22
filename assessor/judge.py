"""AI transcript judge: an LLM answers a checklist of generic good-teaching behaviours (yes / partly / no / unknown)
with a verbatim, timestamped quote for each positive answer. Every quote is verified against the transcript and
anything made up is discarded; dimension scores are then computed deterministically from the answers.

Why a checklist rather than a holistic 1-5 score: concrete, evidence-backed yes/no questions are far more reliable
across model sizes (small local models tend to copy rubric text when asked for a holistic score), and every score
stays traceable to specific moments. The checklist is subject-agnostic: it applies to any online class for children.
Constructs follow classroom-observation research (e.g. CLASS: emotional support, organisation, instructional support,
student engagement) adapted to live online teaching."""
from __future__ import annotations

import difflib
import re
from typing import Callable, Optional

from . import llm, perf, privacy, transcript
from .transcript import fmt_t, parse_t

# id: (question asked of the model, text if observed, text if not observed, reverse)
# reverse=True items are bad behaviours: "yes" lowers the score.
CHECKLIST: dict[str, dict] = {
    "opening_rapport": {
        "title": "Opening & rapport",
        "items": {
            "greets": ("Does the teacher greet or welcome the learners warmly at the start?",
                       "The teacher welcomed the learners at the start.", "No clear welcome or greeting was observed at the start.", False),
            "checkin": ("Before the lesson content, does the teacher ask learners something personal or social (how they are, their day, interests)?",
                        "The teacher built rapport with personal check-in questions before the content.", "The class moved to content without any personal check-in with the learners.", False),
            "agenda": ("Does the teacher say what the class will cover or what the plan is?",
                       "The teacher set out what the class would cover.", "No agenda or plan for the class was stated.", False),
            "recap": ("Does the teacher link to earlier learning (what was done last class, a quick revision)?",
                      "The teacher linked the class to earlier learning.", "There was no link to earlier learning.", False),
            "learner_voice": ("Do learners speak in full sentences or share things freely (not only one-word answers) in the opening?",
                              "Learners spoke freely in the opening, a sign of a relaxed start.", "Learners gave only minimal responses in the opening.", False),
        },
    },
    "individual_attention": {
        "title": "Individual attention & inclusion",
        "items": {
            "names": ("Does the teacher address several different learners by name?",
                      "The teacher addressed several learners by name.", "Learners were hardly ever addressed by name.", False),
            "direct_q": ("Does the teacher ask questions directed at individual learners?",
                         "The teacher asked questions directed at individual learners.", "Questions were rarely directed at individual learners.", False),
            "inclusion": ("Does the teacher make sure quieter learners take part or checks in with each learner?",
                          "The teacher made sure different learners took part.", "A few learners seemed to dominate; quieter learners were not drawn in.", False),
            "wait_time": ("Does the teacher give learners time to think and answer rather than answering for them?",
                          "The teacher gave learners time to think and answer.", "The teacher tended to answer for the learners or rushed them.", False),
            "new_learner": ("If a new learner appears, does the teacher welcome them or ask them to introduce themselves? (answer unknown if no new learner is evident)",
                            "New learners were welcomed and invited to introduce themselves.", "A new learner was not welcomed or given a chance to introduce themselves.", False),
        },
    },
    "tone_energy_warmth": {
        "title": "Tone, energy & warmth",
        "items": {
            "praise": ("Does the teacher praise or encourage learners' effort or answers?",
                       "The teacher praised and encouraged the learners.", "There was little praise or encouragement.", False),
            "kind_mistakes": ("When a learner is wrong, does the teacher respond kindly and constructively?",
                              "Mistakes were handled kindly and constructively.", "Mistakes were not handled constructively.", False),
            "warm_tone": ("Does the teacher sound friendly, patient and upbeat (enthusiasm, humour)?",
                          "The teacher sounded friendly, patient and upbeat.", "The teacher's tone came across as flat or low-energy.", False),
            "harsh": ("Does the teacher sound irritated, dismissive, sarcastic or discouraging at any point?",
                      "The teacher sounded irritated or discouraging at times.", "", True),
        },
    },
    "instruction_feedback": {
        "title": "Instruction & feedback",
        "items": {
            "reasoning": ("Does the teacher explain the reasons behind answers (the why), not just the answers?",
                          "The teacher explained the reasoning behind answers.", "The teacher mostly gave instructions and answers without explaining why.", False),
            "probing": ("Does the teacher ask open or probing questions (why, how, what if) that make learners think?",
                        "The teacher asked probing questions that made learners think.", "Hardly any probing 'why/how' questions were asked.", False),
            "feedback": ("Does the teacher give specific feedback on learners' answers (beyond just right or wrong)?",
                         "The teacher gave specific feedback on learners' answers.", "Feedback on learners' answers was mostly just right or wrong.", False),
            "checks": ("Does the teacher check understanding (e.g. any doubts? did you get it?)?",
                       "The teacher checked understanding.", "The teacher did not check whether learners understood.", False),
            "clarity": ("Are instructions and explanations clear and pitched at the learners' age?",
                        "Instructions and explanations were clear and age-appropriate.", "Instructions or explanations were unclear or poorly pitched for the age group.", False),
        },
    },
    "engagement_variety": {
        "title": "Engagement & variety",
        "items": {
            "active": ("Are learners actively responding or doing tasks for most of the observed time (not just listening)?",
                       "Learners were actively responding for most of the observed time.", "Learners were mostly passive listeners.", False),
            "variety": ("Is more than one type of activity or resource used (e.g. games, puzzles, videos, discussion, practice)?",
                        "The class used a variety of activities and resources.", "The class relied on a single type of activity.", False),
            "live": ("Is the class mostly live teaching and interaction, rather than long stretches of a pre-recorded video or lesson?",
                     "Teaching was mostly live and interactive.", "Long stretches relied on pre-recorded or played content rather than live teaching.", False),
            "on_task": ("Is the time used productively (little confusion, waiting, technical delays or off-topic talk)?",
                        "Time was used productively.", "Time was lost to confusion, waiting or off-topic talk.", False),
        },
    },
    "structure_closing": {
        "title": "Structure, closing & follow-up",
        "items": {
            "flow": ("Does the class have a clear, organised flow from start to finish?",
                     "The class had a clear, organised flow.", "The class flow was disorganised or unclear.", False),
            "summary": ("Near the end, does the teacher summarise or recap what was learned? (unknown if the end was not observed)",
                        "The teacher summarised what was learned.", "There was no summary of the learning at the end.", False),
            "follow_up": ("Does the teacher assign homework, practice or next steps? (unknown if the end was not observed)",
                          "Homework or practice was assigned.", "No homework, practice or next steps were assigned.", False),
            "preview": ("Does the teacher say what the next class covers or when it is? (unknown if the end was not observed)",
                        "The teacher previewed the next class.", "The next class was not previewed.", False),
            "goodbye": ("Does the class end with a polite goodbye or wrap-up? (unknown if the end was not observed)",
                        "The class ended with a proper goodbye.", "The class ended without a clear wrap-up or goodbye.", False),
        },
    },
}
DIMENSIONS = {k: {"title": v["title"]} for k, v in CHECKLIST.items()}
GROUPS = [["opening_rapport", "individual_attention", "tone_energy_warmth"], ["instruction_feedback", "engagement_variety", "structure_closing"]]
ANSWER_VALUE = {"yes": 1.0, "partly": 0.5, "no": 0.0}

SYSTEM = (
    "You are an experienced, fair observer of online classes for children (any subject). You answer checklist questions "
    "about a class from automatic transcript excerpts. Rules: (1) answer ONLY from the transcript; (2) for 'yes' or "
    "'partly' you MUST give one quote copied word-for-word from a transcript line, with its [m:ss] time; (3) answer "
    "'no' if the excerpts show the behaviour is absent where you would expect it, and 'unknown' if the excerpts do not "
    "cover that part of the class (the transcript only covers selected windows); (4) the transcript is automatic speech "
    "recognition with errors and mangled names: do not penalise mis-heard words; (5) the transcript has no speaker "
    "labels: infer who is speaking (teacher vs children) from the dialogue. Answer with JSON only."
)


def question_ids(keys: list[str], only: Optional[set] = None) -> list[str]:
    return [f"{k}.{iid}" for k in keys for iid in CHECKLIST[k]["items"] if only is None or f"{k}.{iid}" in only]


def _questions(keys: list[str], only: Optional[set] = None) -> str:
    out = []
    for k in keys:
        ids = [i for i in question_ids([k], only)]
        if not ids:
            continue
        out.append(f"{CHECKLIST[k]['title']}:")
        for qid in ids:
            out.append(f"  {qid}: {CHECKLIST[k]['items'][qid.split('.', 1)[1]][0]}")
    return "\n".join(out)


def _json_example(keys: list[str], extra: str = "", only: Optional[set] = None) -> str:
    first = question_ids(keys, only)[0]
    return ('{"answers": {"' + first + '": {"a": "yes", "t": "0:21", "q": "words copied from the transcript"}, "<next id>": {"a": "no", "t": "", "q": ""}, ... one entry for EVERY id listed above},'
            ' "transcript_usable": true' + extra + "}")


def measured_context(feats: dict) -> str:
    v = feats.get("vision") or {}
    ts = feats.get("transcript_stats") or {}
    lines = [f"Recording length {fmt_t(feats.get('duration_s'))}; windows analysed: " + ", ".join(f"{w['name']} {fmt_t(w['start'])}-{fmt_t(w['end'])}" for w in feats.get("windows", []))]
    if ts.get("first_speech_s") is not None:
        lines.append(f"First speech heard at {fmt_t(ts['first_speech_s'])}.")
    mix = v.get("scene_mix")
    if mix:
        lines.append("On-screen content (share of frames): " + ", ".join(f"{k} {x:.0%}" for k, x in sorted(mix.items(), key=lambda kv: -kv[1])) + ".")
    return "\n".join("- " + ln for ln in lines)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def verify_quote(quote: str, t_claim: Optional[float], segs: list) -> Optional[tuple[float, str]]:
    """Return (true timestamp, true text) if the quote really appears in the transcript, else None."""
    q = _norm(quote)
    if len(q) < 6:
        return None
    best, best_s = 0.0, None
    for s in segs:
        st = _norm(s.text)
        if not st:
            continue
        score = 1.0 if (q in st or (len(st) >= 8 and st in q)) else difflib.SequenceMatcher(None, q, st).ratio()
        if t_claim is not None and abs(s.start - t_claim) > 120:
            score *= 0.9
        if score > best:
            best, best_s = score, s
    return (best_s.start, best_s.text) if best_s is not None and best >= 0.7 else None


def _answer(raw) -> tuple[str, str, str]:
    if not isinstance(raw, dict):
        return "unknown", "", ""
    a = str(raw.get("a", "")).strip().lower()
    a = a if a in ("yes", "partly", "no", "unknown") else ("yes" if a.startswith("y") else "no" if a.startswith("n") else "unknown")
    return a, str(raw.get("t", "") or ""), str(raw.get("q", "") or "")


def _build_items(dim: str, answers: dict, segs: list) -> list[dict]:
    items = []
    for iid, (_q, pos, neg, reverse) in CHECKLIST[dim]["items"].items():
        a, t, q = _answer(answers.get(f"{dim}.{iid}"))
        ev, unverified = [], False
        if a in ("yes", "partly") and q:
            v = verify_quote(q, parse_t(t), segs)
            if v:
                ev = [{"t": round(v[0], 1), "text": v[1] if len(v[1]) <= 160 else v[1][:157] + "..."}]
            else:
                unverified = True
        if a in ("yes", "partly") and not ev:
            # An affirmative answer with no verifiable evidence is not trusted at full value.
            a = "partly" if a == "yes" else "no"
            unverified = True
        value = ANSWER_VALUE.get(a)
        if value is not None and reverse:
            value = 1.0 - value
        items.append({"id": iid, "answer": a, "value": value, "reverse": reverse, "positive_text": pos, "negative_text": neg,
                      "evidence": ev, "unverified": unverified})
    return items


def judge(feats: dict, segs: list, provider: llm.Provider, progress: Optional[Callable[[str], None]] = None,
          only: Optional[set] = None) -> dict:
    """`only`: restrict to these checklist ids ("dimension.item"); others stay unknown (used for escalation)."""
    say = progress or (lambda m: None)
    masker = None
    if not provider.local:
        if not privacy.available():
            raise RuntimeError("De-identification model unavailable; refusing to send transcript to a cloud LLM.")
        names = privacy.collect_names([s.text for s in segs])
        masker = lambda t: privacy.mask(t, names)  # noqa: E731
    text, kept = transcript.build_transcript(segs, feats.get("windows", []), provider.max_chars, masker)
    if not text:
        return {"usable": False, "dimensions": {}, "tech_issues": None, "provider": provider.name, "model": provider.model, "local": provider.local}

    ctx = measured_context(feats)
    dims: dict[str, dict] = {}
    usable_votes: list[bool] = []
    tech = None
    # Small local models answer more reliably in two smaller groups; large cloud models take all questions in ONE request
    # (free tiers cap requests per day, so this is also what keeps quota use to about one request per class).
    groups = GROUPS if provider.local else [[k for g in GROUPS for k in g]]
    for gi, keys in enumerate(groups):
        if not question_ids(keys, only):
            continue
        say(f"AI judge ({provider.name}/{provider.model}): {', '.join(CHECKLIST[k]['title'] for k in keys)}...")
        extra = ""
        if gi == len(groups) - 1 and only is None:
            extra =', "technical_issues": {"count": <times audio/video/connection problems interrupted the class, e.g. "can you hear me?">, "t": "", "q": "<one verbatim example or empty>"}'
        user = ("MEASURED CONTEXT:\n" + ctx + "\n\nTRANSCRIPT EXCERPTS (no speaker labels):\n" + text +
                "\n\nQUESTIONS (answer every id with a: yes|partly|no|unknown):\n" + _questions(keys, only) +
                "\n\nReturn JSON exactly like:\n" + _json_example(keys, extra, only))
        perf.log.info("judge call provider=%s/%s group=%d/%d questions=%d transcript_chars=%d", provider.name, provider.model,
                      gi + 1, len(groups), len(question_ids(keys, only)), len(text))
        data = llm.chat_json(provider, SYSTEM, user, retries=1, max_tokens=1500)
        answers = data.get("answers") if isinstance(data.get("answers"), dict) else {}
        if not answers:  # tolerate models that put answers at top level
            answers = {k: v for k, v in data.items() if isinstance(v, dict) and "." in k}
        usable_votes.append(bool(data.get("transcript_usable", True)))
        for k in keys:
            dims[k] = {"items": _build_items(k, answers, kept)}
        ti = data.get("technical_issues")
        if isinstance(ti, dict):
            try:
                cnt = max(int(ti.get("count", 0)), 0)
            except Exception:
                cnt = 0
            v = verify_quote(ti.get("q", "") or "", parse_t(ti.get("t")), kept) if ti.get("q") else None
            tech = {"count": cnt if v or cnt <= 1 else min(cnt, 2), "evidence": [{"t": round(v[0], 1), "text": v[1]}] if v else []}
    return {"usable": all(usable_votes), "dimensions": dims, "tech_issues": tech, "provider": provider.name, "model": provider.model,
            "local": provider.local, "transcript_chars": len(text)}


def dimension_score(items: list[dict]) -> Optional[float]:
    vals = [i["value"] for i in items if i["value"] is not None]
    if len(vals) < 2:
        return None
    return 100.0 * sum(vals) / len(vals)


def summarise(report_digest: str, provider: llm.Provider) -> Optional[str]:
    """Short manager-facing summary from the already-verified findings (no new facts)."""
    try:
        data = llm.chat_json(
            provider,
            "You write concise feedback summaries for managers. Use ONLY the facts provided; add nothing else. JSON only.",
            report_digest + '\n\nWrite 3-4 sentences: the overall verdict, the strongest points and the most important fixes. Return {"summary": "<text>"}',
            retries=0, max_tokens=300,
        )
        s = str(data.get("summary", "")).strip()
        return s or None
    except Exception:
        return None
