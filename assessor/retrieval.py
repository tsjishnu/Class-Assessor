"""In-memory semantic retrieval over one recording's transcript (no persistence: children's speech never stored).

Used by the Jev decider: for each checklist question we fetch only the few most relevant transcript lines, which keeps
the decider's input short (where it is most reliable) and gives us a traceable evidence line for each answer.
Embeddings come from a small free local model (fastembed, ONNX, CPU); the embedder is injectable for testing."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import perf

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_model = None


def _fastembed():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(EMBED_MODEL)
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    v = np.array(list(_fastembed().passage_embed(texts)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed_queries(texts: list[str]) -> np.ndarray:
    v = np.array(list(_fastembed().query_embed(texts)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@dataclass
class Hit:
    score: float
    start: float
    lines: list          # [(segment, similarity)] in time order
    text: str


class Index:
    """Chunks of consecutive segments (same analysis window), scored by their mean segment embedding."""

    def __init__(self, segs: list, embed_passages_fn: Callable = embed_passages, embed_queries_fn: Callable = embed_queries,
                 chunk: int = 5, stride: int = 3):
        self.segs = list(segs)
        self._q = embed_queries_fn
        t0 = time.perf_counter()
        self.vecs = embed_passages_fn([s.text for s in self.segs]) if self.segs else np.zeros((0, 1), np.float32)
        perf.log.info("retrieval index: embedded %d segments in %.1fs", len(self.segs), time.perf_counter() - t0)
        self.chunks: list[list[int]] = []
        by_win: dict[str, list[int]] = {}
        for i, s in enumerate(self.segs):
            by_win.setdefault(s.window, []).append(i)
        for idxs in by_win.values():
            for a in range(0, max(len(idxs), 1), stride):
                part = idxs[a:a + chunk]
                if part:
                    self.chunks.append(part)
                if a + chunk >= len(idxs):
                    break

    def windows(self) -> set[str]:
        return {s.window for s in self.segs}

    def search(self, query: str, k: int = 3, scope: str = "any") -> list[Hit]:
        t0 = time.perf_counter()
        try:
            return self._search(query, k, scope)
        finally:
            perf.LATENCY.add("retrieval.search", time.perf_counter() - t0)

    def _search(self, query: str, k: int = 3, scope: str = "any") -> list[Hit]:
        """Top-k chunks for `query`. scope: 'opening' | 'closing' | 'any' restricts to that analysis window ('full' counts as both)."""
        if not self.chunks:
            return []
        qv = self._q([query])[0]
        sims = self.vecs @ qv
        hits = []
        for idxs in self.chunks:
            win = self.segs[idxs[0]].window
            if scope != "any" and win not in (scope, "full"):
                continue
            c = self.vecs[idxs].mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-9)
            score = float(c @ qv) * 0.5 + float(sims[idxs].max()) * 0.5
            hits.append((score, idxs))
        hits.sort(key=lambda x: -x[0])
        out = []
        for score, idxs in hits[:k]:
            lines = [(self.segs[i], float(sims[i])) for i in idxs]
            out.append(Hit(score, self.segs[idxs[0]].start, lines, "\n".join(f"[{int(s.start // 60)}:{int(s.start % 60):02d}] {s.text}" for s, _ in lines)))
        out.sort(key=lambda h: h.start)
        return out

    def best_line(self, hits: list[Hit]):
        """The single most relevant segment across hits (used as the answer's evidence line)."""
        best = max(((sim, s) for h in hits for s, sim in h.lines), key=lambda x: x[0], default=None)
        return None if best is None else best[1]
