"""Aggregation: AI-judge scores (1-5 per dimension) + measured technical signals -> overall score -> grade.

Bands are set a priori from the scale's meaning (3 = adequate/Average, 4 = Good, 5 = Excellent), not fitted to any
sample data. When the AI judge is unavailable, a few measured proxies give a low-confidence estimate instead."""
from __future__ import annotations

from typing import Optional

from .config import TIERS
from . import judge
from .judge import DIMENSIONS
from .transcript import fmt_t

WEIGHTS = {
    "opening_rapport": 0.15,
    "individual_attention": 0.20,
    "tone_energy_warmth": 0.12,
    "instruction_feedback": 0.15,
    "engagement_variety": 0.12,
    "structure_closing": 0.12,
    "technical_setup": 0.14,
}
TITLES = {**{k: v["title"] for k, v in DIMENSIONS.items()}, "technical_setup": "Technical setup & presentation"}
# 0-100 score -> tier. Equivalent to a mean rubric score of 2.5 / 3.5 / 4.4 on the 1-5 scale.
CUTS = [(0, "Poor"), (37.5, "Average"), (62.5, "Good"), (85.0, "Excellent")]


def tier_for(score: float) -> str:
    t = CUTS[0][1]
    for cut, name in CUTS:
        if score >= cut:
            t = name
    return t


def _ev(items: list[dict]) -> list[str]:
    return [f'{fmt_t(e["t"])} "{e["text"]}"' for e in items]


# ---------------------------------------------------------------- technical (measured, deterministic)

def technical(feats: dict, judged: Optional[dict], brand_check: bool) -> dict:
    v, a = feats.get("vision") or {}, feats.get("audio") or {}
    n = v.get("n_frames", 0)
    obs: list[dict] = []
    score = 62.0
    used = 0

    def add(delta, text, ev=None):
        nonlocal score
        score += delta
        obs.append({"impact": "positive" if delta > 0 else "negative", "points": delta, "text": text, "evidence": ev or []})

    if n >= 8:
        if brand_check and v.get("brand_check"):
            used += 1
            lr = v.get("logo_ratio", 0)
            if lr >= 0.4:
                add(+10, f"The branded virtual background was visible in {lr:.0%} of sampled frames.")
            elif lr >= 0.1:
                add(-6, f"The branded background appeared in only {lr:.0%} of sampled frames (missing, cropped or unclear in the rest).")
            else:
                add(-22, "No branded background detected: the tutor appears to use a different or no virtual background.")
            fg = v.get("tutor_face_given_logo")
            if fg is not None and lr >= 0.1:
                if fg >= 0.6:
                    add(+10, "The tutor's face was clearly visible whenever the tutor was on screen.")
                elif fg >= 0.25:
                    add(-8, f"The tutor's face was visible in only {fg:.0%} of the frames where the tutor's tile was shown (turned away, covered or too small).")
                else:
                    add(-28, f"The tutor's face was visible in only {fg:.0%} of the frames where the tutor's tile was shown: camera effectively off, covered or out of frame.")
        else:
            pv = v.get("person_visible_ratio")
            if pv is not None:
                used += 1
                if pv >= 0.5:
                    add(+8, f"A person's face was visible on camera in {pv:.0%} of sampled frames.")
                elif pv < 0.2:
                    add(-20, f"A face was visible in only {pv:.0%} of sampled frames: cameras appear to be off for most of the class.")
        lum = v.get("face_lum_median")
        if lum is not None:
            used += 1
            if lum < 60:
                add(-12, f"The tutor's face was poorly lit (median brightness {lum:.0f}/255): sit facing a light source.")
            elif lum > 215:
                add(-5, f"The tutor's face was over-exposed (median brightness {lum:.0f}/255).")
            elif lum >= 85:
                add(+6, "Good, even lighting on the tutor's face.")
        br = v.get("blank_ratio", 0)
        if br >= 0.3:
            used += 1
            add(-10, f"{br:.0%} of sampled frames were a black/blank screen.")
    if a:
        snr = a.get("snr_db")
        if snr is not None:
            used += 1
            if snr >= 30:
                add(+6, "Clean audio with good separation between speech and background noise.")
            elif snr < 18:
                add(-10, f"Noisy or weak audio (speech only {snr:.0f} dB above the noise floor).")
        if a.get("clipping_ratio", 0) > 0.005:
            used += 1
            add(-5, "Audio was clipping/distorted (microphone too loud or too close).")
    if judged and judged.get("tech_issues"):
        ti = judged["tech_issues"]
        used += 1
        if ti["count"] >= 4:
            add(-14, f"Audio/video problems repeatedly interrupted the class ({ti['count']} instances noted by the AI judge).", _ev(ti["evidence"]))
        elif ti["count"] >= 2:
            add(-5, "A couple of audio/video checks interrupted the class.", _ev(ti["evidence"]))
    return {"score": max(0.0, min(100.0, score)) if used else None, "observations": obs}


# ---------------------------------------------------------------- fallback proxies (no AI judge)

def proxies(feats: dict) -> dict[str, tuple[float, str]]:
    out: dict[str, tuple[float, str]] = {}
    ts, d, v = feats.get("transcript_stats") or {}, feats.get("diarization") or {}, feats.get("vision") or {}
    fs = ts.get("first_speech_s")
    if fs is not None:
        out["opening_rapport"] = (70 if fs <= 30 else 55 if fs <= 90 else 40 if fs <= 180 else 25, f"First speech at {fmt_t(fs)} (measured proxy only).")
    sh = d.get("tutor_share_time")
    if sh is not None:
        c = 1 - sh
        out["individual_attention"] = (75 if c >= 0.4 else 55 if c >= 0.25 else 35, f"Children spoke roughly {c:.0%} of the sampled speaking time (approximate; speaker separation is imperfect).")
    sw = v.get("scene_switches_per_10min")
    if sw is not None:
        out["engagement_variety"] = (65 if sw >= 4 else 55 if sw >= 2 else 40, f"{sw:.1f} on-screen content changes per 10 min (measured proxy only).")
    return out


# ---------------------------------------------------------------- assembly

def assess(feats: dict, judged: Optional[dict] = None, brand_check: bool = True) -> dict:
    flags: list[str] = list(feats.get("notes", []))
    dims_out, strengths, improvements = [], [], []
    scores: dict[str, Optional[float]] = {}
    n_ai = 0
    prox = proxies(feats)
    jd = (judged or {}).get("dimensions", {}) if judged and judged.get("usable", True) else {}
    if judged and not judged.get("usable", True):
        flags.append(f"The transcript could not be used ({judged.get('reason') or 'too garbled or not in a language the AI could follow'}), so transcript-based "
                     "dimensions were NOT assessed; only measured signals (audio, video) contributed to this grade.")
    st = (judged or {}).get("session_type")
    if st and st not in ("regular_class", "regular", "class"):
        flags.append(f"This recording looks like a '{st}' session (e.g. a trial or enrolment conversation) rather than a regular class, so the grade may not be meaningful.")
    if not judged:
        flags.append("No AI judge was available, so transcript-based dimensions use low-confidence measured proxies only. Start Ollama or configure a free cloud LLM for a full assessment.")

    n_unverified = 0
    for key in DIMENSIONS:
        d = jd.get(key) or {}
        items = d.get("items") or []
        ai_score = judge.dimension_score(items) if items else None
        obs, skipped = [], []
        if ai_score is not None:
            n_ai += 1
            score = ai_score
            for it in items:
                ans = it["answer"]
                n_unverified += int(it.get("unverified", False))
                if ans == "unknown":
                    continue
                if it["reverse"]:
                    if ans in ("yes", "partly"):
                        obs.append({"impact": "negative", "points": 0, "text": it["positive_text"], "evidence": _ev(it["evidence"])})
                elif ans == "yes":
                    obs.append({"impact": "positive", "points": 0, "text": it["positive_text"], "evidence": _ev(it["evidence"])})
                elif ans == "partly":
                    obs.append({"impact": "positive", "points": 0, "text": "Partly: " + it["positive_text"][0].lower() + it["positive_text"][1:], "evidence": _ev(it["evidence"])})
                else:
                    obs.append({"impact": "negative", "points": 0, "text": it["negative_text"], "evidence": []})
            skipped += [f"Not observable in the analysed excerpts: {it['id']}" for it in items if it["answer"] == "unknown"]
        elif key in prox:
            score, txt = prox[key]
            obs.append({"impact": "neutral", "points": 0, "text": txt, "evidence": []})
            skipped.append("AI transcript judgement unavailable; low-confidence estimate")
        else:
            score = None
            skipped.append("not assessed (needs the AI transcript judge)")
        scores[key] = score
        dims_out.append((key, score, obs, skipped))

    tech = technical(feats, judged, brand_check)
    scores["technical_setup"] = tech["score"]
    dims_out.append(("technical_setup", tech["score"], tech["observations"], [] if tech["score"] is not None else ["no audio/video evidence"]))

    num = den = 0.0
    for k, s in scores.items():
        if s is not None:
            num += WEIGHTS[k] * s
            den += WEIGHTS[k]
    overall = num / den if den else 50.0
    grade = tier_for(overall)

    v = feats.get("vision") or {}
    fg = v.get("tutor_face_given_logo")
    if brand_check and v.get("brand_check") and fg is not None and v.get("logo_ratio", 0) >= 0.1 and fg < 0.15 and v.get("n_frames", 0) >= 8:
        if TIERS.index(grade) > TIERS.index("Average"):
            flags.append("Grade capped at Average: the tutor's face was almost never visible during the class.")
            grade = "Average"

    tr = (feats.get("asr") or {}).get("translated_share", 0.0) or 0.0
    if tr >= 0.3:
        flags.append(f"About {tr:.0%} of the class was not in English and was machine-translated before assessment, so transcript-based findings are less reliable.")
    edges = [c for c, _ in CUTS[1:]]
    margin = min(abs(overall - e) for e in edges)
    sources = int(bool(feats.get("transcript_stats", {}).get("n_segments"))) + int(bool(v)) + int(bool(feats.get("audio")))
    if n_ai >= 5 and sources == 3 and margin >= 6:
        confidence = "High"
    elif n_ai >= 3 and sources >= 2 and margin >= 2:
        confidence = "Medium"
    else:
        confidence = "Low"
    if tr >= 0.3 and confidence == "High":
        confidence = "Medium"

    if n_unverified:
        flags.append(f"{n_unverified} AI answer(s) came without a verifiable transcript quote and were down-weighted.")
    out_dims = []
    for key, s, obs, skipped in dims_out:
        label = "Not assessed" if s is None else ("Strong" if s >= 70 else "Adequate" if s >= 50 else "Needs work")
        out_dims.append({"key": key, "title": TITLES[key], "score": None if s is None else round(s, 1), "weight": WEIGHTS[key],
                         "label": label, "observations": obs, "skipped": skipped})
        for o in obs:
            if o["impact"] == "neutral" or s is None:
                continue
            item = {"dimension": TITLES[key], "text": o["text"], "evidence": o["evidence"], "rank": s if o["impact"] == "positive" else -s}
            (strengths if o["impact"] == "positive" else improvements).append(item)
    strengths.sort(key=lambda x: (-x["rank"], -len(x["evidence"])))
    improvements.sort(key=lambda x: (-x["rank"], -len(x["evidence"])))

    ts = feats.get("transcript_stats") or {}
    return {
        "grade": grade,
        "score": round(overall, 1),
        "confidence": confidence,
        "dimensions": out_dims,
        "strengths": strengths[:6],
        "improvements": improvements[:6],
        "flags": flags,
        "ai": {"provider": (judged or {}).get("provider"), "model": (judged or {}).get("model"), "local": (judged or {}).get("local"), "dimensions_judged": n_ai,
               "decided": (judged or {}).get("decided"), "uncertain": (judged or {}).get("uncertain"), "escalated": (judged or {}).get("escalated"),
               "escalated_to": (judged or {}).get("escalated_to"), "mode": (judged or {}).get("mode", "checklist"),
               "stats": (judged or {}).get("stats"), "fallbacks": (judged or {}).get("fallbacks", [])},
        "signals": {
            "duration": fmt_t(feats.get("duration_s")),
            "windows_analysed": [f'{w["name"]} {fmt_t(w["start"])}-{fmt_t(w["end"])}' for w in feats.get("windows", [])],
            "frames_analysed": v.get("n_frames", 0),
            "transcribed_segments": ts.get("n_segments", 0),
            "language": (feats.get("asr") or {}).get("language_share"),
            "tutor_share_of_speech": (feats.get("diarization") or {}).get("tutor_share_time"),
            "screen_content": v.get("scene_mix"),
        },
    }
