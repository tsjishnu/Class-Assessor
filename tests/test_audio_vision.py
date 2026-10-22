import os

import cv2
import numpy as np

from assessor import audio_features, vision
from assessor.speech import Segment

SR = 16000
TPL = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo_template.png"), cv2.IMREAD_GRAYSCALE)


def tone(f, dur, amp=0.3):
    t = np.arange(int(SR * dur)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_audio_stats_detects_late_start_and_silences():
    rng = np.random.default_rng(0)
    quiet = (0.0005 * rng.standard_normal(SR * 30)).astype(np.float32)
    x = np.concatenate([quiet, tone(200, 5) + 0.01 * rng.standard_normal(SR * 5).astype(np.float32)])
    st = audio_features.audio_stats(x)
    assert 28 <= st["first_sound_s"] <= 32
    assert st["long_silence_count"] >= 1
    assert st["snr_db"] > 20


def test_speaker_separation_groups_by_voice():
    rng = np.random.default_rng(1)

    def voice(f0, dur):
        t = np.arange(int(SR * dur)) / SR
        sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
        return (0.2 * sig / np.abs(sig).max() + 0.01 * rng.standard_normal(len(t))).astype(np.float32)

    parts, segs, t = [], [], 0.0
    for i in range(14):
        who = "tutor" if i % 3 else "child"
        d = 3.0 if who == "tutor" else 1.5
        parts.append(voice(120 if who == "tutor" else 330, d))
        segs.append(Segment(t, t + d, "x", -0.2, 0.0, "full"))
        t += d
    out = audio_features.separate_speakers(np.concatenate(parts), segs)
    assert out["n_clusters"] >= 2
    assert 0.6 < out["tutor_share_time"] < 0.95


def _frame_with_logo(scale, corner=False):
    img = np.full((720, 1280), 20, np.uint8)
    t = cv2.resize(TPL, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = t.shape
    if corner:
        img[10:10 + h, 1280 - w - 40:1280 - 40] = t
    else:
        img[60:60 + h, 190:190 + w] = t
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_logo_detected_big_and_thumbnail_but_not_noise(tmp_path):
    big = tmp_path / "big.jpg"; cv2.imwrite(str(big), _frame_with_logo(1.0))
    thumb = tmp_path / "thumb.jpg"; cv2.imwrite(str(thumb), _frame_with_logo(0.14, corner=True))
    rng = np.random.default_rng(2)
    noise = tmp_path / "noise.jpg"; cv2.imwrite(str(noise), rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8))
    board = np.kron(((np.indices((8, 8)).sum(0) % 2) * 200 + 30).astype(np.uint8), np.ones((60, 60), np.uint8))
    chess = tmp_path / "chess.jpg"; cv2.imwrite(str(chess), cv2.cvtColor(cv2.resize(board, (1280, 720)), cv2.COLOR_GRAY2BGR))
    assert vision.analyse_frame(str(big))["logo_big"]
    assert vision.analyse_frame(str(thumb))["logo_thumb"]
    assert not vision.analyse_frame(str(noise))["logo_big"]
    r = vision.analyse_frame(str(chess))
    assert not r["logo_big"] and not r["logo_thumb"], "a chessboard must not be mistaken for the brand logo"


def test_blank_frame_has_no_person(tmp_path):
    p = tmp_path / "dark.jpg"; cv2.imwrite(str(p), np.zeros((720, 1280, 3), np.uint8))
    r = vision.analyse_frame(str(p))
    assert r["mean_lum"] < 5 and not r["person_visible"] and not r["tutor_tile"]
