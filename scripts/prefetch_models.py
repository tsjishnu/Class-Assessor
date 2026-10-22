"""Download every model the app needs, so a deployment image starts fast and works without network access to model hubs.

    python scripts/prefetch_models.py            # run once at image build time
Models: faster-whisper (speech-to-text), CLIP vision+text (what is on screen), spaCy NER (name masking for cloud LLMs)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assessor import config, speech, vision  # noqa: E402


def main():
    print(f"whisper '{config.WHISPER_MODEL}' ...", flush=True)
    speech.get_model()
    print("CLIP (vision + text) ...", flush=True)
    vision._clip_models()
    print("spaCy en_core_web_sm ...", flush=True)
    import spacy

    spacy.load("en_core_web_sm")
    print("all models ready", flush=True)


if __name__ == "__main__":
    main()
