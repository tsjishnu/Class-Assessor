"""Windowed speech-to-text with faster-whisper (free, local, CPU int8)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional

import os
import re
import time

import numpy as np

from . import config, perf

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SR = 16000
CHESS_PROMPT = "A chess class for children on Zoom. Words: chess, pawn, rook, knight, bishop, queen, king, checkmate, check, puzzle, Lichess, castling, en passant."

_model = None


def unload() -> None:
    """Release the speech model (and its memory). It is reloaded on next use."""
    global _model
    _model = None
    import gc

    gc.collect()


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=config.WHISPER_THREADS)
        perf.log.info("whisper model loaded name=%s threads=%d sec=%.1f", config.WHISPER_MODEL, config.WHISPER_THREADS, time.perf_counter() - t0)
    return _model


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    window: str
    lang: str = "en"
    role: str = "unknown"
    translated: bool = False   # True when this text was machine-translated to English from `lang`

    def to_dict(self):
        return asdict(self)


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def collapse_repeats(text: str, min_repeats: int = 3, max_ngram: int = 12) -> str:
    """Whisper sometimes loops on a phrase ("Hello, good morning. Hello, good morning. ..."); keep one copy.
    Only runs of >= min_repeats identical consecutive word n-grams are collapsed, so normal speech is untouched."""
    words = text.split()
    norm = [_norm(w) for w in words]
    for n in range(1, max_ngram + 1):
        i, out_w, out_n = 0, [], []
        while i < len(words):
            j = i + n
            reps = 1
            while j + n <= len(words) and norm[j:j + n] == norm[i:i + n] and any(norm[i:i + n]):
                reps += 1
                j += n
            if reps >= min_repeats:
                out_w += words[i:i + n]
                out_n += norm[i:i + n]
                i = j
            else:
                out_w.append(words[i])
                out_n.append(norm[i])
                i += 1
        words, norm = out_w, out_n
    return " ".join(words).strip()


def plan_windows(duration: float) -> list[tuple[str, float, float]]:
    """Opening, closing, and a few mid-class slices; whole file if it is short."""
    total_planned = config.OPENING_S + config.CLOSING_S + config.MID_SLICE_S * len(config.MID_SLICE_POS)
    if duration <= total_planned * 1.1:
        return [("full", 0.0, duration)]
    wins = [("opening", 0.0, float(config.OPENING_S))]
    for i, p in enumerate(config.MID_SLICE_POS):
        s = duration * p - config.MID_SLICE_S / 2
        s = max(s, config.OPENING_S)
        e = min(s + config.MID_SLICE_S, duration - config.CLOSING_S)
        if e - s > 30:
            wins.append((f"mid{i + 1}", s, e))
    wins.append(("closing", duration - config.CLOSING_S, duration))
    return wins


def transcribe_windows(
    samples: np.ndarray,
    windows: list[tuple[str, float, float]],
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[list[Segment], dict]:
    model = get_model()
    segs: list[Segment] = []
    langs: dict[str, float] = {}
    translated_s = 0.0
    for name, s, e in windows:
        if progress:
            progress(f"Transcribing {name} ({int(s // 60)}:{int(s % 60):02d} - {int(e // 60)}:{int(e % 60):02d})...")
        chunk = samples[int(s * SR): int(e * SR)]
        if len(chunk) < SR:
            continue
        t_win = time.perf_counter()
        n_before = len(segs)
        # Language is detected per window (classes can switch language). For non-English speech, plain transcription by a small
        # model can be garbage (measured on Malayalam: hallucinated symbols), while Whisper's translate-to-English task gives
        # usable text in the same time, so non-English windows are translated (NON_EN_TASK=transcribe disables this).
        try:
            det_lang, det_p, _ = model.detect_language(chunk, vad_filter=True, language_detection_segments=3)
        except Exception as ex:  # detection is a hint; fall back to automatic detection inside transcribe()
            perf.log.warning("language detection failed for window %s: %s", name, ex)
            det_lang, det_p = None, 0.0
        translate = bool(det_lang and det_lang != "en" and det_p >= 0.5 and os.environ.get("NON_EN_TASK", "translate") == "translate")
        it, info = model.transcribe(
            chunk,
            language=(det_lang if det_lang and det_p >= 0.5 else None),
            task="translate" if translate else "transcribe",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
            initial_prompt=CHESS_PROMPT,
            temperature=0.0,
        )
        langs[info.language] = langs.get(info.language, 0.0) + (e - s) * info.language_probability
        if translate:
            translated_s += e - s
        for sg in it:
            txt = collapse_repeats(sg.text.strip())
            if not txt:
                continue
            segs.append(Segment(s + sg.start, s + sg.end, txt, float(sg.avg_logprob), float(sg.no_speech_prob), name, info.language, "unknown", translate))
        wall = time.perf_counter() - t_win
        perf.log.info("asr window=%s audio=%.0fs wall=%.1fs rtf=%.2f segments=%d lang=%s(%.2f) task=%s", name, e - s, wall, wall / max(e - s, 1e-6),
                      len(segs) - n_before, info.language, info.language_probability, "translate" if translate else "transcribe")
    total = sum(langs.values()) or 1.0
    lang_share = {k: round(v / total, 3) for k, v in sorted(langs.items(), key=lambda kv: -kv[1])}
    audio_s = sum(e - s for _, s, e in windows) or 1.0
    return segs, {"language_share": lang_share, "model": config.WHISPER_MODEL, "translated_share": round(translated_s / audio_s, 3)}
