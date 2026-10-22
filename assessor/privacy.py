"""De-identification of transcript text before it may leave the machine (cloud LLM providers only).

Uses a small NER model (spaCy) to mask PERSON entities, then masks every other occurrence of any name found.
Recall is not perfect (unusual names can be missed), which is why local processing is the default and cloud
use requires explicit operator consent."""
from __future__ import annotations

import re
from typing import Optional

_nlp = None


def _load():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "parser"])
    return _nlp


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


def collect_names(texts: list[str]) -> set[str]:
    nlp = _load()
    names: set[str] = set()
    for doc in nlp.pipe(texts, batch_size=64):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                for tok in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", ent.text):
                    names.add(tok)
    return names


def mask(text: str, names: set[str], token: str = "[name]") -> str:
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(n)}\b", token, text)
    return text


def deidentify(texts: list[str]) -> tuple[list[str], set[str]]:
    names = collect_names(texts)
    return [mask(t, names) for t in texts], names
