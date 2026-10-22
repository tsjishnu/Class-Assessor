"""Transcript helpers: formatting for the LLM and basic (non-linguistic) statistics. No keyword/regex rules here."""
from __future__ import annotations

from typing import Optional


def fmt_t(sec: Optional[float]) -> str:
    if sec is None:
        return "n/a"
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def parse_t(s) -> Optional[float]:
    try:
        parts = [float(x) for x in str(s).strip().split(":")]
    except Exception:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def usable(segs: list) -> list:
    """Drop likely-hallucinated, non-speech segments."""
    return [s for s in segs if not (s.no_speech_prob > 0.85 and s.avg_logprob < -1.5)]


def latin_share(texts: list[str]) -> float:
    """Share of letters that are plain Latin. Garbage output (hallucinated symbols for unsupported speech) scores ~0."""
    letters = [c for t in texts for c in t if c.isalpha()]
    return sum(1 for c in letters if c.isascii()) / len(letters) if letters else 0.0


def unusable_reason(feats: dict, segs: list) -> Optional[str]:
    """Why the transcript cannot support an assessment (None if it can). Checked BEFORE any AI call, so no quota is wasted."""
    segs = usable(segs)
    windows = feats.get("windows") or []
    observed_min = max(sum(w["end"] - w["start"] for w in windows) / 60.0, 1.0)
    words = sum(len(s.text.split()) for s in segs)
    if len(segs) < 10:
        return f"only {len(segs)} speech segments were recognised (silent, unintelligible or unsupported language)"
    if words / observed_min < 12:
        return f"very little speech was recognised ({words} words in {observed_min:.0f} observed minutes)"
    if latin_share([s.text for s in segs]) < 0.7:
        return "the transcript is mostly non-Latin symbols, which indicates unrecognisable speech"
    return None


def stats(segs: list, windows: list[dict]) -> dict:
    segs = usable(segs)
    n = len(segs)
    in_opening = [s for s in segs if s.window in ("opening", "full")]
    first = min((s.start for s in in_opening), default=None)
    return {
        "n_segments": n,
        "first_speech_s": None if first is None else round(first, 1),
        "transcribed_minutes": round(sum(s.end - s.start for s in segs) / 60.0, 1),
        "asr_mean_logprob": round(sum(s.avg_logprob for s in segs) / n, 3) if n else None,
        "asr_low_conf_share": round(sum(1 for s in segs if s.avg_logprob < -1.0) / n, 3) if n else None,
        "words": sum(len(s.text.split()) for s in segs),
    }


def _label(role: str) -> str:
    return {"tutor": "A", "child": "B"}.get(role, "?")


def build_transcript(segs: list, windows: list[dict], budget_chars: int, masker=None) -> tuple[str, list]:
    """Readable transcript grouped by analysed window, trimmed to `budget_chars`.
    Openings keep their first lines, closings their last lines (what matters most in each)."""
    segs = usable(segs)
    by_win: dict[str, list] = {}
    for s in segs:
        by_win.setdefault(s.window, []).append(s)
    wins = [w for w in windows if w["name"] in by_win]
    if not wins:
        return "", []
    weight = {"opening": 3.0, "closing": 2.5, "full": 1.0}
    total_w = sum(weight.get(w["name"], 1.0) for w in wins)
    out, kept_all = [], []
    for w in wins:
        budget = int(budget_chars * weight.get(w["name"], 1.0) / total_w)
        items = by_win[w["name"]]
        lines = [f"[{fmt_t(s.start)}] {masker(s.text) if masker else s.text}" for s in items]
        if w["name"] == "closing":
            keep, used = [], 0
            for ln, s in reversed(list(zip(lines, items))):
                if used + len(ln) > budget:
                    break
                keep.append((ln, s)); used += len(ln) + 1
            keep.reverse()
        else:
            keep, used = [], 0
            for ln, s in zip(lines, items):
                if used + len(ln) > budget:
                    break
                keep.append((ln, s)); used += len(ln) + 1
        title = f"## {w['name'].upper()} ({fmt_t(w['start'])}-{fmt_t(w['end'])})"
        if keep:
            out.append(title + "\n" + "\n".join(k[0] for k in keep))
            kept_all += [k[1] for k in keep]
    return "\n\n".join(out), kept_all
