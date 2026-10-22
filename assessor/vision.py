"""Visual signals from sampled frames, computed locally (no frames leave the machine).

General-purpose models, no subject-specific rules:
  * YuNet (deep-learning face detector, ONNX via OpenCV) -> is a person's face visible, and how well lit / sharp.
  * CLIP zero-shot (ONNX via fastembed) -> what is on screen: webcam, activity/board, video lesson, slides,
    desktop/browser, blank screen.  This captures "what happens on screen" for any subject.
Optional brand check: if a reference logo image is configured (assets/logo_template.png by default; set
BRAND_TEMPLATE="" to disable), multi-scale template matching finds the operator's branded virtual background and
lets us tell the tutor's tile apart from children's tiles in Zoom-style recordings."""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

import cv2
import numpy as np

from . import perf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YUNET_PATH = os.path.join(ROOT, "assets", "face_detection_yunet_2023mar.onnx")
BRAND_TEMPLATE = os.environ.get("BRAND_TEMPLATE", os.path.join(ROOT, "assets", "logo_template.png"))

# Matching only the two "8" glyphs of the reference logo: robust to the tutor's head occluding the middle and
# (unlike the checker squares) it does not false-match chessboards. Gap between non-logo (<=0.6) and logo (>=0.85).
LOGO_T = 0.70
_PARTS = ((0, 240), (665, 910))

SCENE_LABELS = {
    "webcam": "a webcam view of a person's face and shoulders in a video call",
    "activity": "a screen showing a game, puzzle or interactive exercise",
    "video_lesson": "a screen playing an educational video lesson",
    "slides_doc": "a presentation slide or text document on screen",
    "desktop": "a web browser or computer desktop with tabs and menus",
    "blank": "a black or empty screen with only a name label",
}

_tpl = None
_yunet = None
_clip = None


def brand_enabled() -> bool:
    return bool(BRAND_TEMPLATE) and os.path.exists(BRAND_TEMPLATE)


def _template():
    global _tpl
    if _tpl is None:
        _tpl = cv2.imread(BRAND_TEMPLATE, cv2.IMREAD_GRAYSCALE)
    return _tpl


def _face_detector():
    global _yunet
    if _yunet is None:
        _yunet = cv2.FaceDetectorYN.create(_YUNET_PATH, "", (320, 320), 0.6, 0.3, 500)
    return _yunet


def detect_faces(img_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    det = _face_detector()
    h, w = img_bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img_bgr)
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[14])) for f in faces]


def _best_logo_match(gray: np.ndarray, scales, pre_scale: float = 1.0) -> float:
    tpl = _template()
    best = -1.0
    for a, b in _PARTS:
        part = tpl[:, a:b]
        for s in scales:
            sc = s * pre_scale
            tw, th = int(part.shape[1] * sc), int(part.shape[0] * sc)
            if tw < 10 or th < 6 or tw >= gray.shape[1] or th >= gray.shape[0]:
                continue
            t = cv2.resize(part, (tw, th), interpolation=cv2.INTER_AREA)
            best = max(best, float(cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED).max()))
    return best


def _clip_models():
    global _clip
    if _clip is None:
        from fastembed import ImageEmbedding, TextEmbedding

        img_m = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")
        txt_m = TextEmbedding("Qdrant/clip-ViT-B-32-text")
        T = np.array(list(txt_m.embed(list(SCENE_LABELS.values()))), dtype=np.float32)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        _clip = (img_m, T)
    return _clip


def scene_probs(paths: list[str]) -> np.ndarray:
    """Zero-shot scene probabilities, shape (n_frames, n_labels)."""
    img_m, T = _clip_models()
    E = np.array(list(img_m.embed(paths)), dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    S = 100.0 * E @ T.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return P / P.sum(axis=1, keepdims=True)


def analyse_frame(path: str) -> dict:
    img = cv2.imread(path)
    if img is None:
        return {"ok": False}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out: dict = {"ok": True, "mean_lum": float(gray.mean())}

    faces = detect_faces(img)
    # Zoom-style recordings put the active speaker in a small top-right tile during screen share.
    tx0, ty1 = int(w * 0.74), int(h * 0.24)
    thumb_up = cv2.resize(img[0:ty1, tx0:w], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    thumb_faces = detect_faces(thumb_up)
    out["face_full"] = bool(faces)
    out["face_thumb"] = bool(thumb_faces)
    out["person_visible"] = bool(faces or thumb_faces)

    logo_big = logo_thumb = False
    big_score = thumb_score = 0.0
    if brand_enabled():
        small = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        big_score = _best_logo_match(small, [0.40, 0.55, 0.70, 0.85, 1.0, 1.15], pre_scale=0.25 * w / 1280)
        thumb_score = _best_logo_match(gray[0:ty1, tx0:w], [0.09, 0.115, 0.14, 0.17, 0.20, 0.23], pre_scale=w / 1280)
        logo_big = big_score >= LOGO_T
        logo_thumb = (not logo_big) and thumb_score >= LOGO_T
    out.update(big_logo=round(big_score, 3), thumb_logo=round(thumb_score, 3), logo_big=bool(logo_big), logo_thumb=bool(logo_thumb))

    tutor_tile = logo_big or logo_thumb
    out["tutor_tile"] = bool(tutor_tile)
    out["tutor_face"] = bool((logo_big and faces) or (logo_thumb and thumb_faces))

    # Quality of the best visible face (lighting, sharpness).
    cand = None
    if logo_big and faces:
        cand = (img, max(faces, key=lambda f: f[2] * f[3]))
    elif logo_thumb and thumb_faces:
        cand = (thumb_up, max(thumb_faces, key=lambda f: f[2] * f[3]))
    elif faces:
        cand = (img, max(faces, key=lambda f: f[2] * f[3]))
    elif thumb_faces:
        cand = (thumb_up, max(thumb_faces, key=lambda f: f[2] * f[3]))
    if cand:
        im, (x, y, fw, fh, _) = cand
        x, y = max(x, 0), max(y, 0)
        roi = cv2.cvtColor(im[y:y + fh, x:x + fw], cv2.COLOR_BGR2GRAY)
        if roi.size > 100:
            out["face_lum"] = float(roi.mean())
            out["face_sharp"] = float(cv2.Laplacian(cv2.resize(roi, (96, 96)), cv2.CV_64F).var())

    main = gray[:, : int(w * 0.74)]
    out["_fp"] = cv2.resize(main, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    return out


def analyse_frames(frames: list[tuple[float, str]], progress: Optional[Callable[[str], None]] = None) -> dict:
    per = []
    for i, (t, p) in enumerate(frames):
        if progress and i % 25 == 0:
            progress(f"Analysing video frames {i}/{len(frames)}...")
        t_f = time.perf_counter()
        r = analyse_frame(p)
        perf.LATENCY.add("vision.frame", time.perf_counter() - t_f)
        r["t"] = t
        per.append(r)
    ok_idx = [i for i, r in enumerate(per) if r.get("ok")]
    ok = [per[i] for i in ok_idx]
    n = max(len(ok), 1)

    scene_mix: dict[str, float] = {}
    switches_per_10min = None
    if ok:
        if progress:
            progress("Classifying what is on screen (CLIP)...")
        t_c = time.perf_counter()
        P = scene_probs([frames[i][1] for i in ok_idx])
        perf.log.info("clip scene classification frames=%d sec=%.1f", len(ok_idx), time.perf_counter() - t_c)
        labels = list(SCENE_LABELS)
        top = P.argmax(axis=1)
        for r, k in zip(ok, top):
            r["scene"] = labels[int(k)]
        scene_mix = {lab: round(float((top == j).mean()), 3) for j, lab in enumerate(labels) if (top == j).any()}
        span_min = max((ok[-1]["t"] - ok[0]["t"]) / 60.0, 1.0)
        switches_per_10min = round(float((top[1:] != top[:-1]).sum()) / span_min * 10, 2)

    tutor_tile = [r for r in ok if r["tutor_tile"]]
    tutor_face = [r for r in tutor_tile if r["tutor_face"]]
    lum = [r["face_lum"] for r in ok if "face_lum" in r]
    sharp = [r["face_sharp"] for r in ok if "face_sharp" in r]
    fps = [r["_fp"] for r in ok]
    changes = [float(np.abs(a - b).mean()) for a, b in zip(fps, fps[1:])]

    summary = {
        "n_frames": len(ok),
        "brand_check": brand_enabled(),
        "logo_ratio": round(len(tutor_tile) / n, 3),
        "tutor_face_given_logo": round(len(tutor_face) / len(tutor_tile), 3) if tutor_tile else None,
        "person_visible_ratio": round(sum(r["person_visible"] for r in ok) / n, 3),
        "face_lum_median": round(float(np.median(lum)), 1) if lum else None,
        "face_sharp_median": round(float(np.median(sharp)), 1) if sharp else None,
        "blank_ratio": round(scene_mix.get("blank", 0.0), 3),
        "scene_mix": scene_mix,
        "scene_switches_per_10min": switches_per_10min,
        "screen_change_ratio": round(float(np.mean([c > 6.0 for c in changes])) if changes else 0.0, 3),
    }
    for r in per:
        r.pop("_fp", None)
    return {"summary": summary, "frames": per}
