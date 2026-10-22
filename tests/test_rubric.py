import copy

from assessor import rubric


def feats(good: bool):
    return {
        "duration_s": 3000, "windows": [{"name": "opening", "start": 0, "end": 480}, {"name": "closing", "start": 2640, "end": 3000}],
        "audio": {"snr_db": 35 if good else 12, "clipping_ratio": 0.0},
        "asr": {"language_share": {"en": 1.0}},
        "diarization": {"tutor_share_time": 0.5 if good else 0.85, "tutor_pitch_std_semitones": 4.0 if good else 1.0},
        "transcript_stats": {"n_segments": 300, "first_speech_s": 15 if good else 240},
        "vision": {"n_frames": 100, "brand_check": True, "logo_ratio": 0.6 if good else 0.0, "tutor_face_given_logo": 0.9 if good else None,
                   "person_visible_ratio": 0.6, "face_lum_median": 110 if good else 40, "blank_ratio": 0.0,
                   "scene_mix": {"activity": 0.7}, "scene_switches_per_10min": 5 if good else 0.5},
        "notes": [],
    }


def judged(level: str):
    """level: 'yes' (all good-practice items observed), 'no' (none observed), 'mixed'."""
    from assessor import judge

    dims = {}
    for k, spec in judge.CHECKLIST.items():
        items = []
        for n, (iid, (_q, pos, neg, rev)) in enumerate(spec["items"].items()):
            ans = level if level != "mixed" else ("yes" if n % 2 == 0 else "no")
            val = judge.ANSWER_VALUE[ans]
            val = 1 - val if rev else val
            ev = [{"t": 21.0, "text": "Hello everyone"}] if ans == "yes" else []
            items.append({"id": iid, "answer": ans, "value": val, "reverse": rev, "positive_text": pos, "negative_text": neg or "Not observed.", "evidence": ev, "unverified": False})
        dims[k] = {"items": items}
    return {"usable": True, "dimensions": dims, "tech_issues": {"count": 0, "evidence": []}, "provider": "fake", "model": "m", "local": True}


def test_band_mapping_follows_rubric_meaning():
    assert rubric.tier_for(20) == "Poor"
    assert rubric.tier_for(50) == "Average"
    assert rubric.tier_for(75) == "Good"
    assert rubric.tier_for(92) == "Excellent"


def test_ai_scores_drive_the_grade():
    hi = rubric.assess(feats(True), judged("yes"))
    lo = rubric.assess(feats(False), judged("no"))
    assert hi["grade"] == "Excellent" and lo["grade"] == "Poor"
    assert hi["score"] > lo["score"] + 40
    assert hi["confidence"] in ("High", "Medium")


def test_schema_and_evidence_passthrough():
    r = rubric.assess(feats(True), judged("yes"))
    assert {"grade", "score", "confidence", "dimensions", "strengths", "improvements", "flags", "signals", "ai"} <= set(r)
    assert len(r["dimensions"]) == 7
    assert any(o["evidence"] for d in r["dimensions"] for o in d["observations"])
    assert r["ai"]["dimensions_judged"] == 6


def test_no_ai_judge_gives_low_confidence_estimate_not_crash():
    r = rubric.assess(feats(True), None)
    assert r["confidence"] == "Low"
    assert any("No AI judge" in f for f in r["flags"])
    by = {d["key"]: d for d in r["dimensions"]}
    assert by["instruction_feedback"]["score"] is None, "no proxy exists for instruction quality; it must not be invented"
    assert by["opening_rapport"]["score"] is not None


def test_unusable_transcript_flagged_and_falls_back():
    j = judged("yes")
    j["usable"] = False
    r = rubric.assess(feats(True), j)
    assert any("could not be used" in f for f in r["flags"])
    assert r["ai"]["dimensions_judged"] == 0


def test_tutor_never_visible_caps_grade():
    f = feats(True)
    f["vision"]["tutor_face_given_logo"] = 0.05
    r = rubric.assess(f, judged("yes"))
    assert r["grade"] in ("Poor", "Average")
    assert any("capped" in x.lower() for x in r["flags"])


def test_brand_check_can_be_disabled():
    f = feats(False)
    assert rubric.assess(f, judged("mixed"), brand_check=False)["score"] > rubric.assess(f, judged("mixed"), brand_check=True)["score"]


def test_missing_vision_and_audio_still_grades():
    f = copy.deepcopy(feats(True))
    f.pop("vision"); f.pop("audio")
    r = rubric.assess(f, judged("yes"))
    assert r["grade"] in ("Poor", "Average", "Good", "Excellent")
