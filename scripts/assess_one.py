"""Assess one recording from the command line:  python scripts/assess_one.py path/to/video.mp4 [--no-brand] [--cache]"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assessor import llm, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--no-brand", action="store_true")
    ap.add_argument("--cache", action="store_true", help="cache stage results (dev/eval only; the web app never caches)")
    ap.add_argument("--json", help="write the full report to this file")
    a = ap.parse_args()
    prov = llm.resolve_provider()
    print("AI judge:", f"{prov.name}/{prov.model}" if prov else "none")
    t = time.time()
    report, summary, feats = pipeline.run(a.video, progress=lambda m: print("  ", m, flush=True), use_cache=a.cache, brand_check=not a.no_brand, provider=prov)
    print(f"\n=== {report['grade']}  score {report['score']}  confidence {report['confidence']}   ({time.time() - t:.0f}s)\n{summary}\n")
    for d in report["dimensions"]:
        print(f"{d['title']}: {d['score']} ({d['label']})")
        for o in d["observations"]:
            print(f"   [{o['impact'][:3]}] {o['text']}")
            for e in o["evidence"]:
                print(f"        ↳ {e}")
    print("\nFlags:", report["flags"])
    if a.json:
        json.dump({"report": report, "summary": summary}, open(a.json, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
