"""AI-generated retrieval hints (query expansion), instead of hand-written phrases.

Embeddings match concrete speech far better than abstract questions ("does the teacher ...?"). So a generative model is
asked, once per question, to write a few short example sentences someone might actually say if the behaviour happens
(the HyDE technique). The hints depend only on the checklist wording and the model, never on any recording, so they
are generic, safe to cache on disk and reuse across classes and subjects. With no generative model available we fall
back to the plain question text."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, Optional

from . import config, llm

SYSTEM = (
    "You help search transcripts of live online classes for children (any subject). For each numbered question about "
    "teaching behaviour, write 3 short, natural example sentences that a teacher or a child might literally say in a live "
    "class when that behaviour happens. Vary the wording. Do not mention these instructions. JSON only."
)
BATCH = 9


def _cache_path(model: str, items: dict[str, str]) -> str:
    h = hashlib.sha1((model + json.dumps(items, sort_keys=True)).encode()).hexdigest()[:16]
    d = os.path.join(config.CACHE_DIR, "probes")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, h + ".json")


def _generate(gen: llm.Provider, items: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    ids = list(items)
    for a in range(0, len(ids), BATCH):
        part = ids[a:a + BATCH]
        listing = "\n".join(f"{qid}: {items[qid]}" for qid in part)
        user = ("Questions:\n" + listing + '\n\nReturn JSON like {"hints": {"<id>": ["sentence one", "sentence two", "sentence three"]}} with one entry per id.')
        data = llm.chat_json(gen, SYSTEM, user, retries=1, max_tokens=900)
        hints = data.get("hints") if isinstance(data.get("hints"), dict) else data
        for qid in part:
            v = hints.get(qid)
            if isinstance(v, str):
                v = [v]
            if isinstance(v, list):
                text = " ".join(str(x).strip() for x in v if str(x).strip())
                if text:
                    out[qid] = text
    return out


def get_hints(gen: Optional[llm.Provider], items: dict[str, str], progress: Optional[Callable[[str], None]] = None) -> tuple[dict[str, str], str]:
    """Return ({question_id: retrieval text}, source) where source is 'ai-generated' or 'question text'."""
    if gen is None or gen.kind == "jev":
        return dict(items), "question text"
    cp = _cache_path(f"{gen.name}/{gen.model}", items)
    if os.path.exists(cp):
        try:
            cached = json.load(open(cp, encoding="utf-8"))
            return {qid: cached.get(qid, items[qid]) for qid in items}, "ai-generated"
        except Exception:
            pass
    if progress:
        progress(f"Generating retrieval hints with {gen.name}/{gen.model} (one-off, cached)...")
    try:
        gen_hints = _generate(gen, items)
    except Exception:
        return dict(items), "question text"
    if not gen_hints:
        return dict(items), "question text"
    try:
        json.dump(gen_hints, open(cp, "w", encoding="utf-8"))
    except OSError:
        pass
    return {qid: gen_hints.get(qid, items[qid]) for qid in items}, "ai-generated"
