"""Local open-source decision model from Hugging Face: zero-shot NLI (DeBERTa-v3) as a yes/no decider.

Given a premise (the retrieved transcript excerpts) and a hypothesis (a declarative statement such as "The teacher
assigned homework."), an NLI model estimates the probability that the premise ENTAILS the hypothesis. That is the same
decider role as Jev: a fast, calibrated yes/no probability with no text output, so it is paired with retrieval and an
escalation path. It runs on CPU, is free, MIT-licensed and private (nothing leaves the machine).

Default model: MoritzLaurer/deberta-v3-base-zeroshot-v2.0 (184M parameters, ~512-token limit). Swap it with
HF_NLI_MODEL (e.g. the -large variant, or a bge-m3 multilingual one).

Performance notes (measured on a 2-core CPU, see README):
  * float32 is forced: the checkpoint is stored in float16, which CPUs without half-precision hardware emulate ~17x slower
    (11.9 s vs 0.7 s per decision).
  * NLI_THREADS defaults to the number of physical cores (os.cpu_count()//2): extra threads were slower."""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

from . import perf

_lock = threading.Lock()
_cache: dict = {}


def _threads() -> int:
    return max(1, int(os.environ.get("NLI_THREADS", max(1, (os.cpu_count() or 2) // 2))))


def _load(model_name: str):
    with _lock:
        if model_name not in _cache:
            t0 = time.perf_counter()
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(_threads())
            tok = AutoTokenizer.from_pretrained(model_name)
            try:
                mdl = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float32).eval()
            except TypeError:  # older transformers
                mdl = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float32).eval()
            labels = {int(i): str(v).lower() for i, v in mdl.config.id2label.items()}
            ent = next((i for i, v in labels.items() if v.startswith("entail")), 0)
            _cache[model_name] = (tok, mdl, ent)
            perf.log.info("nli model loaded name=%s params=%.0fM dtype=%s threads=%d sec=%.1f", model_name,
                          sum(p.numel() for p in mdl.parameters()) / 1e6, next(mdl.parameters()).dtype, _threads(), time.perf_counter() - t0)
        return _cache[model_name]


def entailment(model_name: str, premise: str, hypotheses: list[str]) -> list[float]:
    """P(premise entails hypothesis) for each hypothesis, as a binary entailment vs not-entailment decision."""
    import torch

    tok, mdl, ent = _load(model_name)
    out = []
    for h in hypotheses:
        t0 = time.perf_counter()
        enc = tok(premise, h, return_tensors="pt", truncation="only_first", max_length=512)
        with torch.inference_mode():
            logits = mdl(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        dt = time.perf_counter() - t0
        perf.LATENCY.add("nli.decide", dt)
        perf.log.debug("nli decide tokens=%d sec=%.2f p=%.3f", enc["input_ids"].shape[1], dt, float(probs[ent]))
        out.append(float(probs[ent]))
    return out


def decide(model_name: str, premise: str, hypothesis: str) -> Optional[float]:
    return entailment(model_name, premise, [hypothesis])[0]
