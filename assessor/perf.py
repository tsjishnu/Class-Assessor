"""Logging and latency instrumentation shared by every stage.

* `log` - the "assessor" logger. `setup_logging()` writes to stderr and to a rotating file (.cache/assessor.log).
* `stage(name, store)` - context manager: logs the wall time of a stage and records it in `store` (surfaced in the report).
* `Latency` - collects many small timings (HTTP calls, per-question decisions) and summarises count / p50 / p95 / max / total."""
from __future__ import annotations

import logging
import logging.handlers
import os
import threading
import time
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("assessor")
_configured = False


def setup_logging(level: Optional[str] = None, file: Optional[str] = None) -> None:
    """Idempotent. Level from ASSESSOR_LOG_LEVEL (default INFO); file from ASSESSOR_LOG_FILE ('' disables the file)."""
    global _configured
    if _configured:
        return
    _configured = True
    from . import config

    lvl = getattr(logging, (level or os.environ.get("ASSESSOR_LOG_LEVEL", "INFO")).upper(), logging.INFO)
    log.setLevel(lvl)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s [%(threadName)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    path = file if file is not None else os.environ.get("ASSESSOR_LOG_FILE", os.path.join(config.CACHE_DIR, "assessor.log"))
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s [%(threadName)s] %(message)s"))
            log.addHandler(fh)
        except OSError:
            pass


@contextmanager
def stage(name: str, store: Optional[dict] = None, **ctx):
    """Time a block. Logs 'stage=<name> sec=<t> k=v...' and stores the seconds in `store[name]` if given."""
    t0 = time.perf_counter()
    log.info("stage=%s start %s", name, " ".join(f"{k}={v}" for k, v in ctx.items()))
    try:
        yield
    except Exception as e:
        log.error("stage=%s FAILED after %.1fs: %s", name, time.perf_counter() - t0, e)
        raise
    finally:
        dt = time.perf_counter() - t0
        if store is not None:
            store[name] = round(store.get(name, 0.0) + dt, 2)
    log.info("stage=%s done sec=%.2f %s", name, dt, " ".join(f"{k}={v}" for k, v in ctx.items()))


class Latency:
    """Thread-safe collector for many small timings."""

    def __init__(self):
        self._v: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def add(self, name: str, seconds: float) -> None:
        with self._lock:
            self._v.setdefault(name, []).append(seconds)

    @contextmanager
    def time(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def summary(self) -> dict:
        out = {}
        with self._lock:
            for k, v in self._v.items():
                s = sorted(v)
                out[k] = {"n": len(s), "total_s": round(sum(s), 2), "p50_s": round(s[len(s) // 2], 3),
                          "p95_s": round(s[min(len(s) - 1, int(len(s) * 0.95))], 3), "max_s": round(s[-1], 3)}
        return out


# Process-wide collector for outbound calls (LLM / decider HTTP requests): inspect with LATENCY.summary().
LATENCY = Latency()
