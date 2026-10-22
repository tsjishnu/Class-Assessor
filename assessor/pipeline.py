"""End-to-end orchestration: media -> audio stats, windowed ASR, speaker split, visual signals -> AI judge -> report."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Callable, Optional

from . import audio_features, config, decider, judge, llm, media, moves, perf, rubric, speech, transcript, vision

perf.setup_logging()

VISION_VERSION = 3
AUTO = object()  # sentinel: resolve the AI provider from settings; pass None to disable the AI judge


def _key(path: str) -> str:
    st = os.stat(path)
    return hashlib.sha1(f"{os.path.basename(path)}|{st.st_size}|{config.WHISPER_MODEL}|{config.FRAME_EVERY_S}".encode()).hexdigest()[:16]


def _cache_file(kind: str, key: str) -> str:
    d = os.path.join(config.CACHE_DIR, kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, key + ".json")


def segments_of(feats: dict) -> list[speech.Segment]:
    return [speech.Segment(**{k: v for k, v in s.items() if k in speech.Segment.__dataclass_fields__}) for s in feats.get("segments", [])]


def _vision_stage(path: str, work: str, say) -> dict:
    frames = media.extract_frames(path, os.path.join(work, "frames"), config.FRAME_EVERY_S, config.FRAME_WIDTH)
    vis = vision.analyse_frames(frames, progress=say)
    keep = ("t", "scene", "tutor_tile", "tutor_face", "person_visible", "mean_lum")
    return {"summary": vis["summary"], "frames": [{k: v for k, v in f.items() if k in keep} for f in vis["frames"]]}


def extract(path: str, progress: Optional[Callable[[str], None]] = None, use_cache: bool = False) -> dict:
    """Run every measurement stage. Each stage fails independently and is recorded in `notes`."""
    say = progress or (lambda m: None)
    cp = _cache_file("features", _key(path)) if use_cache else None
    cached: dict = {}
    if cp and os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("_vision_v") == VISION_VERSION:
            return cached

    work = media.make_workdir()
    notes: list[str] = []
    timings: dict[str, float] = dict(cached.get("timings", {}))
    feats: dict = {"source": os.path.basename(path)}
    perf.log.info("extract start file=%s size_mb=%.0f cached_partial=%s", os.path.basename(path), os.path.getsize(path) / 1e6, bool(cached))
    t_all = time.perf_counter()
    try:
        duration = media.probe_duration(path)
        feats["duration_s"] = round(duration, 1)
        has_vid = media.has_video(path)
        feats["has_video"] = has_vid
        perf.log.info("media duration=%.0fs has_video=%s", duration, has_vid)

        reuse_audio = all(k in cached for k in ("audio", "asr", "diarization", "segments", "windows"))
        if reuse_audio:
            perf.log.info("reusing cached audio / ASR / diarization stages")
            for k in ("audio", "asr", "diarization", "segments", "windows"):
                feats[k] = cached[k]
        else:
            with perf.stage("audio_extract", timings):
                say("Extracting audio...")
                samples = media.load_wav(media.extract_audio(path, os.path.join(work, "audio.wav")))
            with perf.stage("audio_stats", timings):
                feats["audio"] = audio_features.audio_stats(samples)

            windows = speech.plan_windows(duration)
            feats["windows"] = [{"name": n, "start": round(s, 1), "end": round(e, 1)} for n, s, e in windows]
            segs, asr_meta = [], {}
            with perf.stage("asr", timings, windows=len(windows), audio_min=round(sum(e - s for _, s, e in windows) / 60, 1)):
                try:
                    segs, asr_meta = speech.transcribe_windows(samples, windows, progress=say)
                except Exception as e:
                    perf.log.exception("speech-to-text failed")
                    notes.append(f"Speech-to-text failed: {e}")
            feats["asr"] = asr_meta

            diar = {"method": "none", "tutor_share_time": None, "n_clusters": 0}
            with perf.stage("diarization", timings, segments=len(segs)):
                if segs:
                    say("Separating voices...")
                    try:
                        diar = audio_features.separate_speakers(samples, segs)
                    except Exception as e:
                        perf.log.exception("speaker separation failed")
                        notes.append(f"Speaker separation failed: {e}")
            feats["diarization"] = diar
            feats["segments"] = [s.to_dict() for s in segs]

        feats["transcript_stats"] = transcript.stats(segments_of(feats), feats["windows"])

        if has_vid:
            with perf.stage("vision", timings):
                say("Sampling video frames...")
                try:
                    vis = _vision_stage(path, work, say)
                    feats["vision"], feats["vision_frames"] = vis["summary"], vis["frames"]
                except Exception as e:
                    perf.log.exception("video analysis failed")
                    notes.append(f"Video analysis failed: {e}")
        else:
            notes.append("No video stream found: visual checks skipped (audio-only input).")
    finally:
        media.cleanup(work)
    feats["notes"] = notes
    feats["timings"] = timings
    feats["_vision_v"] = VISION_VERSION
    perf.log.info("extract done wall=%.1fs stages=%s", time.perf_counter() - t_all, timings)
    if cp:
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(feats, f)
    return feats


def judge_mode() -> str:
    """moves (default): LLM labels teaching-move events, frequencies are scored.  checklist: yes/no checklist answers."""
    return os.environ.get("JUDGE_MODE", "moves").lower()


def _judged_cache(cache_key: Optional[str], p: llm.Provider) -> str:
    return _cache_file("judged", f"{cache_key}_{judge_mode()}{("-" + moves.VERSION) if judge_mode() == "moves" else ""}_{p.name}_{p.model.replace(':', '-').replace('/', '-')}")


def judge_features(feats: dict, provider: Optional[llm.Provider], progress=None, cache_key: Optional[str] = None) -> Optional[dict]:
    """Run the AI judge. Generative providers form a fallback chain (preferred first): if one has no quota left or
    fails, the next is tried. Deciders (Jev / hf-nli) escalate unsure answers through the same chain.
    Returns None when nothing worked (the report then says so and uses measured proxies)."""
    if provider is None:
        perf.log.info("judge: no AI provider configured")
        return None
    say = progress or (lambda m: None)
    # Deciders escalate unsure answers through this chain; ESCALATE_TO=none turns escalation off (decider-only runs).
    chain = [] if (os.environ.get("ESCALATE_TO") or os.environ.get("JEV_ESCALATE_TO", "auto")).lower() == "none" else llm.generative_chain("auto")
    if provider.kind in ("ollama", "openai"):
        candidates = [provider] + [c for c in llm.generative_chain(provider.name) if (c.name, c.model) != (provider.name, provider.model)]
    else:
        candidates = [provider]
    if len(candidates) > 1:
        for c in candidates:
            c.tries = 2  # fail fast: another provider can take over
    candidates.sort(key=lambda c: not llm.is_healthy(c))
    reason = transcript.unusable_reason(feats, segments_of(feats))
    if reason:
        perf.log.warning("judge skipped: transcript unusable (%s)", reason)
        return {"usable": False, "reason": reason, "dimensions": {}, "tech_issues": None, "provider": provider.name, "model": provider.model,
                "local": provider.local, "mode": "none", "fallbacks": []}
    perf.log.info("judge chain: %s", " -> ".join(f"{c.name}/{c.model}" + ("" if llm.is_healthy(c) else " [cooling down]") for c in candidates))
    notes = feats.setdefault("notes", [])
    fallbacks: list[str] = []   # skipped providers: diagnostics, only shown to the user if nothing worked
    for cand in candidates:
        cf = _judged_cache(cache_key, cand) if cache_key else None
        if cf and os.path.exists(cf):
            perf.log.info("judge: cache hit for %s/%s", cand.name, cand.model)
            with open(cf, encoding="utf-8") as f:
                return json.load(f)
        t0 = time.perf_counter()
        try:
            if cand.kind in ("jev", "nli"):
                j = decider.judge_with_decider(feats, segments_of(feats), cand, chain, say, hint_provider=llm.resolve_hint_provider())
            elif judge_mode() == "checklist":
                j = judge.judge(feats, segments_of(feats), cand, say)
            else:
                j = moves.judge_moves(feats, segments_of(feats), cand, say)
        except llm.QuotaExhausted as e:
            llm.mark_unhealthy(cand, min(e.retry_after_s or 3600, 6 * 3600), "free quota used up")
            perf.log.warning("judge: %s/%s quota used up after %.1fs; next provider", cand.name, cand.model, time.perf_counter() - t0)
            fallbacks.append(f"{cand.name}/{cand.model}: free quota used up")
            say(f"{e}; trying the next provider...")
            continue
        except Exception as e:
            llm.mark_unhealthy(cand, 300, type(e).__name__)
            perf.log.exception("judge: %s/%s failed after %.1fs", cand.name, cand.model, time.perf_counter() - t0)
            fallbacks.append(f"{cand.name}/{cand.model} failed: {str(e)[:120]}")
            say(f"{cand.name}/{cand.model} failed ({e}); trying the next provider...")
            continue
        j["wall_s"] = round(time.perf_counter() - t0, 1)
        j["fallbacks"] = fallbacks
        perf.log.info("judge done provider=%s/%s wall=%.1fs", cand.name, cand.model, j["wall_s"])
        if cf:
            with open(cf, "w", encoding="utf-8") as f:
                json.dump(j, f)
        return j
    perf.log.error("judge: every provider in the chain failed")
    notes.extend(fallbacks)   # nothing worked: now the user needs to know why
    return None


def summary_digest(report: dict) -> str:
    lines = [f"Grade {report['grade']} (score {report['score']}/100, confidence {report['confidence']})."]
    for d in report["dimensions"]:
        if d["score"] is not None:
            lines.append(f"- {d['title']}: {d['score']:.0f}/100 ({d['label']}).")
    lines += [f"Strength: {s['text']}" for s in report["strengths"][:3]]
    lines += [f"Issue: {s['text']}" for s in report["improvements"][:3]]
    return "\n".join(lines)


def template_summary(report: dict) -> str:
    parts = [f"Overall this class was assessed as {report['grade']} (score {report['score']}/100, {report['confidence'].lower()} confidence)."]
    if report["strengths"]:
        parts.append("Strongest points: " + " ".join(s["text"] for s in report["strengths"][:2]))
    if report["improvements"]:
        parts.append("Main things to improve: " + " ".join(s["text"] for s in report["improvements"][:2]))
    return " ".join(parts)


def run(path: str, progress=None, use_cache: bool = False, brand_check: bool = True,
        provider=AUTO, write_summary: bool = True) -> tuple[dict, str, dict]:
    say = progress or (lambda m: None)
    feats = extract(path, progress=say, use_cache=use_cache)
    if provider is AUTO:
        provider = llm.resolve_provider()
    judged = judge_features(feats, provider, say, cache_key=_key(path) if use_cache else None)
    report = rubric.assess(feats, judged, brand_check=brand_check)
    report["signals"]["timings"] = {"measure_s": feats.get("timings", {}), "judge_s": (judged or {}).get("timings", {}),
                                     "judge_wall_s": (judged or {}).get("wall_s"), "calls": perf.LATENCY.summary()}
    summary = None
    if write_summary and judged and os.environ.get("LLM_SUMMARY", "0") == "1":  # optional: costs one extra request
        gen = provider if provider is not None and provider.kind in ("ollama", "openai") else llm.resolve_escalation()
        summary = judge.summarise(summary_digest(report), gen) if gen else None
    return report, summary or template_summary(report), feats
