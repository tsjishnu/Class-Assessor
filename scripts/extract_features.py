"""Extract and cache features for the labelled sample classes (slow; run once, then calibrate/evaluate from cache).

    python scripts/extract_features.py --data "D:/Humanity/class recordings" [--only B491 B472]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assessor import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATASET_DIR", r"D:\Humanity\class recordings"))
    ap.add_argument("--labels", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labels.json"))
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    labels = json.load(open(args.labels, encoding="utf-8"))["classes"]
    for c in labels:
        if args.only and c["id"] not in args.only:
            continue
        p = os.path.join(args.data, c["path"])
        t = time.time()
        print(f"[{c['id']}] {c['tier']:9s} extracting...", flush=True)
        f = pipeline.extract(p, progress=lambda m: print("   ", m, flush=True), use_cache=True)
        print(f"[{c['id']}] done in {time.time() - t:.0f}s  timings={f.get('timings')}  notes={f.get('notes')}", flush=True)


if __name__ == "__main__":
    main()
