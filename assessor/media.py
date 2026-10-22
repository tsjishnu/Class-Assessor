"""ffmpeg helpers: duration, audio extraction, frame sampling, and input fetching (upload or link)."""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Optional

import numpy as np


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.") from e


def _run(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")


def probe_duration(path: str) -> float:
    r = _run([ffmpeg_exe(), "-hide_banner", "-i", path])
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not m:
        raise ValueError("Could not read the media duration. Is this a valid video/audio file?")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def has_video(path: str) -> bool:
    r = _run([ffmpeg_exe(), "-hide_banner", "-i", path])
    return bool(re.search(r"Stream #\d+:\d+.*Video:", r.stderr))


def extract_audio(path: str, out_wav: str) -> str:
    """16 kHz mono PCM wav for the whole recording."""
    r = _run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav])
    if r.returncode != 0 or not os.path.exists(out_wav):
        raise RuntimeError(f"Audio extraction failed: {r.stderr[-300:]}")
    return out_wav


def load_wav(path: str) -> np.ndarray:
    """Return float32 mono samples in [-1, 1] from a 16 kHz PCM wav."""
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    return data


def extract_frames(path: str, out_dir: str, every_s: int, width: int) -> list[tuple[float, str]]:
    """Sample one frame every `every_s` seconds. Keyframe-only decoding keeps this fast on CPU."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "f_%05d.jpg")
    r = _run(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-skip_frame", "nokey", "-i", path,
         "-vf", f"fps=1/{every_s},scale={width}:-2", "-q:v", "3", pattern]
    )
    files = sorted(glob.glob(os.path.join(out_dir, "f_*.jpg")))
    if not files:
        raise RuntimeError(f"Frame extraction failed: {r.stderr[-300:]}")
    return [(i * every_s, f) for i, f in enumerate(files)]


def fetch_link(url: str, dest_dir: str, progress: Optional[Callable[[str], None]] = None) -> str:
    """Download a recording from a direct URL or a (public) Google Drive link into dest_dir."""
    url = url.strip()
    if not re.match(r"^https?://", url):
        raise ValueError("Please paste an http(s) link.")
    out = os.path.join(dest_dir, "input_video")
    if "drive.google.com" in url or "docs.google.com" in url:
        import gdown

        if progress:
            progress("Downloading from Google Drive...")
        path = gdown.download(url=url, output=out + ".mp4", quiet=True, fuzzy=True)
        if not path:
            raise RuntimeError("Google Drive download failed. Make sure the link is shared as 'Anyone with the link'.")
        return path
    import requests

    if progress:
        progress("Downloading recording...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        ext = os.path.splitext(url.split("?")[0])[1] or ".mp4"
        path = out + ext
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return path


def make_workdir() -> str:
    return tempfile.mkdtemp(prefix="assessor_")


def cleanup(workdir: str) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
