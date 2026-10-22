"""Settings. Everything can be overridden with environment variables or a `.env` file in the project root."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path: str) -> None:
    """Minimal .env loader (KEY=VALUE per line); real environment variables win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(ROOT, ".env"))

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "4"))

# Time-windowed analysis (seconds): openings and closings carry most of the signal for class quality.
OPENING_S = 8 * 60
CLOSING_S = 6 * 60
MID_SLICE_S = 3 * 60
MID_SLICE_POS = (0.30, 0.50, 0.70)

FRAME_EVERY_S = int(os.environ.get("FRAME_EVERY_S", "30"))
FRAME_WIDTH = 1280

CACHE_DIR = os.environ.get("ASSESSOR_CACHE", os.path.join(ROOT, ".cache"))

TIERS = ["Poor", "Average", "Good", "Excellent"]
