"""Audio signals: loudness/noise/silence statistics and best-effort tutor-vs-children separation."""
from __future__ import annotations

import numpy as np

SR = 16000


def frame_db(x: np.ndarray, hop: int = 800) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.array([-90.0])
    f = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((f ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def _runs(mask: np.ndarray):
    """Yield (start_idx, length, value) runs of a boolean array."""
    if len(mask) == 0:
        return
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            yield start, i - start, bool(mask[start])
            start = i


def audio_stats(x: np.ndarray) -> dict:
    hop = 800  # 50 ms
    db = frame_db(x, hop)
    noise = float(np.percentile(db, 10))
    top = float(np.percentile(db, 90))
    active = db > max(noise + 10.0, -60.0)
    sil_runs = [(s * 0.05, l * 0.05) for s, l, v in _runs(active) if not v and l * 0.05 >= 20.0]
    first_sound = None
    for s, l, v in _runs(active):
        if v and l >= 20:  # >= 1 s of sustained sound
            first_sound = s * 0.05
            break
    act_db = db[active] if active.any() else db
    return {
        "duration_s": round(len(x) / SR, 1),
        "noise_floor_db": round(noise, 1),
        "speech_level_db": round(float(np.median(act_db)), 1),
        "snr_db": round(top - noise, 1),
        "level_std_db": round(float(act_db.std()), 2),
        "active_ratio": round(float(active.mean()), 3),
        "long_silence_count": len(sil_runs),
        "long_silence_total_s": round(sum(l for _, l in sil_runs), 1),
        "first_sound_s": None if first_sound is None else round(first_sound, 1),
        "clipping_ratio": round(float((np.abs(x) > 0.99).mean()), 5),
    }


# ---------- MFCC + pitch for speaker clustering ----------

def _mel_filterbank(n_filters=26, n_fft=512, sr=SR, fmin=60.0, fmax=7600.0):
    def hz2mel(h): return 2595 * np.log10(1 + h / 700.0)
    def mel2hz(m): return 700 * (10 ** (m / 2595.0) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_filters + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


_FB = None


def mfcc(x: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    global _FB
    from scipy.fft import dct

    if _FB is None:
        _FB = _mel_filterbank()
    win, hop, n_fft = 400, 160, 512
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hamming(win)
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    mel = np.log(spec @ _FB.T + 1e-10)
    return dct(mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]


def pitch_track(x: np.ndarray, max_frames: int = 80) -> np.ndarray:
    """Median-free autocorrelation F0 estimate (Hz) for voiced frames, 70-450 Hz."""
    frame, hop = 640, 1600
    lo, hi = int(SR / 450), int(SR / 70)
    if len(x) < frame:
        return np.array([])
    starts = np.arange(0, len(x) - frame, hop)
    if len(starts) > max_frames:
        starts = starts[np.linspace(0, len(starts) - 1, max_frames).astype(int)]
    f0s = []
    for s in starts:
        fr = x[s: s + frame]
        fr = fr - fr.mean()
        if np.sqrt((fr ** 2).mean()) < 0.003:
            continue
        ac = np.fft.irfft(np.abs(np.fft.rfft(fr, 2048)) ** 2)[:frame]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lo:hi]
        k = int(seg.argmax())
        if seg[k] > 0.45:
            f0s.append(SR / (lo + k))
    return np.array(f0s)


def segment_embedding(x: np.ndarray):
    m = mfcc(x)[:, 1:]
    f0 = pitch_track(x)
    logf0 = float(np.log(np.median(f0))) if len(f0) >= 3 else np.nan
    return np.concatenate([m.mean(0), m.std(0)]), logf0, f0


def separate_speakers(samples: np.ndarray, segments: list, min_dur: float = 0.8) -> dict:
    """Best-effort split into 'tutor' (largest cluster by speaking time) vs 'other' (children).
    Sets seg.role on each segment (as an attribute) and returns summary stats."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    feats, logf, idxs, f0_by_idx = [], [], [], {}
    for i, sg in enumerate(segments):
        d = sg.end - sg.start
        if d < min_dur:
            continue
        a, b = int(sg.start * SR), int(sg.end * SR)
        if b - a < 4000:
            continue
        e, lf, f0 = segment_embedding(samples[a:b])
        feats.append(e)
        logf.append(lf)
        idxs.append(i)
        f0_by_idx[i] = f0
    for sg in segments:
        sg.role = "unknown"
    if len(feats) < 6:
        return {"method": "none", "reason": "too few segments", "tutor_share_time": None, "n_clusters": 0}
    X = np.array(feats)
    lf = np.array(logf)
    lf = np.where(np.isnan(lf), np.nanmedian(lf) if not np.all(np.isnan(lf)) else 0.0, lf)
    X = np.column_stack([X, lf * 4.0])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    best = (None, -1.0, 1)
    for k in (2, 3, 4):
        if len(X) <= k + 2:
            break
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X)
        try:
            sc = silhouette_score(X, km.labels_)
        except Exception:
            continue
        if sc > best[1]:
            best = (km, sc, k)
    km, sil, k = best
    if km is None:
        return {"method": "none", "reason": "clustering failed", "tutor_share_time": None, "n_clusters": 0}
    dur = {c: 0.0 for c in range(k)}
    for j, i in enumerate(idxs):
        dur[km.labels_[j]] += segments[i].end - segments[i].start
    tutor_c = max(dur, key=dur.get)
    for j, i in enumerate(idxs):
        segments[i].role = "tutor" if km.labels_[j] == tutor_c else "child"
    tot = sum(dur.values()) or 1.0
    tutor_f0 = np.concatenate([f0_by_idx[i] for j, i in enumerate(idxs) if km.labels_[j] == tutor_c] or [np.array([])])
    pitch_std_st = None
    if len(tutor_f0) >= 10:
        pitch_std_st = float(np.std(12 * np.log2(tutor_f0 / np.median(tutor_f0))))
    return {
        "method": "mfcc+pitch kmeans",
        "n_clusters": int(k),
        "silhouette": round(float(sil), 3),
        "tutor_share_time": round(dur[tutor_c] / tot, 3),
        "tutor_pitch_std_semitones": None if pitch_std_st is None else round(pitch_std_st, 2),
    }
