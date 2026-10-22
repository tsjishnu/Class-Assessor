"""Check which AI judge providers are usable and test one end to end.

    python scripts/check_llm.py                 # list providers and test the auto-selected one
    python scripts/check_llm.py --provider gemini
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assessor import config  # noqa: E402,F401  (loads .env)
from assessor import llm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    a = ap.parse_args()
    avail = llm.available_providers()
    print("Configured providers:")
    for n in ("gemini", "groq", "openai", "ollama", "jev"):
        if n in avail:
            p = avail[n]
            print(f"  [x] {n:7s} model={p.model}  {'local/private' if p.local else 'cloud (de-identified text only)'}")
        else:
            print(f"  [ ] {n:7s} {llm.explain_unavailable(n)}")
    p = llm.resolve_provider(a.provider)
    if not p:
        print("\nNo usable provider selected.")
        sys.exit(1)
    print(f"\nTesting {p.name} / {p.model} ...")
    try:
        print("Result:", llm.check(p))
    except Exception as e:
        print("FAILED:", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
