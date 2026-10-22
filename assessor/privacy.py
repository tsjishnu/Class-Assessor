"""De-identification of transcript text before it may leave the machine (cloud LLM providers only).

Speech recognition mangles names ("Midhansh", "Mithansh", "Madanj" for one child), so a plain person-entity recogniser
misses most of them (measured: 1 of 7 names on a test sample). This module therefore masks, with a small local NLP model (spaCy):

  * named entities of type person / place / facility,
  * every proper noun (the tagger recognises unfamiliar names even when the entity model does not),
  * e-mail addresses, links and long digit strings (phone numbers) via the tokenizer's own flags.

Two refinements keep the text useful to the judge:
  * a capitalised sentence-initial word is re-tagged in lowercase: a common word ("Hello", "Okay") changes class, a name does not;
  * a word that also appears in lowercase at least twice in the same transcript ("bishop", "queen") is a common word, not a name
    (unless the entity model explicitly says it is a person).

Masking is consistent: every occurrence of any collected token is replaced. Over-masking is accepted (privacy first); some
under-masking remains possible (e.g. very short names, names that are also frequent words), which is why local
processing is the default and cloud use requires explicit consent."""
from __future__ import annotations

import collections
import re
from typing import Optional

_nlp = None
# Part-of-speech classes of ordinary words: if a sentence-initial "name" becomes one of these in lowercase, it is not a name.
_FUNCTIONAL = {"INTJ", "DET", "ADP", "ADV", "CCONJ", "SCONJ", "PRON", "AUX", "PART", "ADJ", "VERB", "NUM"}
# Closed-class words and interjections: a name is (almost) never tagged as one of these, so a word the tagger sees in this role
# anywhere in the transcript ("Hello", "Okay") is a common word even when it is tagged as a proper noun elsewhere.
_CLOSED = {"INTJ", "DET", "ADP", "ADV", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}
_ENTITY_TYPES = ("PERSON", "GPE", "LOC", "FAC")


def _load():
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "parser"])
        except OSError:  # hosts that only install requirements.txt (e.g. Streamlit Community Cloud): download once
            from spacy.cli import download

            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "parser"])
    return _nlp


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


def collect_names(texts: list[str]) -> set[str]:
    """All surface strings (names, places, contacts) that must be masked in this transcript."""
    nlp = _load()
    docs = list(nlp.pipe(texts, batch_size=64))
    lower_seen = collections.Counter(t.text for d in docs for t in d if t.text.islower())
    closed_seen = {t.text for d in docs for t in d if t.pos_ in _CLOSED}
    found: set[str] = set()
    person: set[str] = set()
    recheck: list[tuple[int, int]] = []
    for di, doc in enumerate(docs):
        for ent in doc.ents:
            if ent.label_ in _ENTITY_TYPES:
                for t in ent:
                    if t.is_alpha and len(t.text) >= 3 and t.pos_ == "PROPN":   # not "Okay" in an entity span like "Okay Adlin"
                        found.add(t.text)
                        if ent.label_ == "PERSON":
                            person.add(t.text)
        for i, t in enumerate(doc):
            if t.like_email or t.like_url:
                found.add(t.text)
            elif t.like_num and t.text.replace(",", "").isdigit() and len(t.text) >= 5:
                found.add(t.text)
            elif t.pos_ == "PROPN" and t.is_alpha and len(t.text) >= 3:
                if i == 0 or doc[i - 1].is_punct:      # sentence-initial: could be a capitalised common word
                    recheck.append((di, i))
                else:
                    found.add(t.text)
    for di, i in recheck:
        doc = docs[di]
        lowered = "".join(tok.text_with_ws if k != i else tok.text.lower() + tok.whitespace_ for k, tok in enumerate(doc))
        d2 = nlp(lowered)
        if i >= len(d2) or d2[i].pos_ not in _FUNCTIONAL:
            found.add(doc[i].text)
    # common words that the tagger sometimes calls proper nouns (bishop, queen, ...): seen in lowercase often enough
    return {w for w in found if not w.isalpha() or (w not in closed_seen and (w in person or lower_seen.get(w.lower(), 0) < 2))}


def mask(text: str, names: set[str], token: str = "[name]") -> str:
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(n)}(?!\w)", token, text)
    return text


def deidentify(texts: list[str]) -> tuple[list[str], set[str]]:
    names = collect_names(texts)
    return [mask(t, names) for t in texts], names
