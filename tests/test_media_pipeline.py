import os
import subprocess

import pytest

from assessor import media, pipeline, rubric


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("v") / "t.mp4")
    subprocess.run(
        [media.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=20",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=20", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10", p],
        check=True,
    )
    return p


def test_probe_and_audio_roundtrip(tiny_video, tmp_path):
    assert 19 <= media.probe_duration(tiny_video) <= 21
    assert media.has_video(tiny_video)
    wav = media.extract_audio(tiny_video, str(tmp_path / "a.wav"))
    x = media.load_wav(wav)
    assert 19 * 16000 <= len(x) <= 21 * 16000


def test_invalid_input_gives_clear_error(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(ValueError):
        media.probe_duration(str(bad))


def test_fetch_link_rejects_non_http(tmp_path):
    with pytest.raises(ValueError):
        media.fetch_link("file:///etc/passwd", str(tmp_path))
    with pytest.raises(ValueError):
        media.fetch_link("ftp://x/y.mp4", str(tmp_path))


def test_pipeline_degrades_gracefully_without_speech(tiny_video):
    """A file with no speech must still produce a report (never crash) and say what was missing."""
    feats = pipeline.extract(tiny_video)
    assert feats["duration_s"] > 15
    assert feats["vision"]["n_frames"] >= 1
    report = rubric.assess(feats, None)
    assert report["grade"] in ("Poor", "Average", "Good", "Excellent")
    assert report["confidence"] == "Low"
    assert report["flags"], "should explain the transcript was unusable"


def test_workdir_is_cleaned_up(tiny_video):
    before = {d for d in os.listdir(media.tempfile.gettempdir()) if d.startswith("assessor_")}
    pipeline.extract(tiny_video)
    after = {d for d in os.listdir(media.tempfile.gettempdir()) if d.startswith("assessor_")}
    assert after <= before, "pipeline must delete its temp files"
