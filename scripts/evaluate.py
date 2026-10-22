"""Evaluate the assessor on the labelled sample classes (sanity check, NOT used to tune any rule).

    python scripts/extract_features.py        # once (slow): ASR + audio + video features, cached
    python scripts/evaluate.py [--detail B491 B441] [--no-ai] [--provider ollama|groq]

Reports exact-tier accuracy, within-one-tier accuracy, ranking agreement and a confusion matrix, plus per-dimension
scores so disagreements can be inspected against the reviewer's notes."""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from assessor import config, llm, pipeline, rubric  # noqa: E402

TIERS = config.TIERS
KEYS = list(rubric.WEIGHTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATASET_DIR", r"D:\Humanity\class recordings"))
    ap.add_argument("--detail", nargs="*", default=[])
    ap.add_argument("--no-ai", action="store_true", help="skip the AI judge (measured-proxy fallback only)")
    ap.add_argument("--no-brand", action="store_true")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    labels = json.load(open(os.path.join(ROOT, "labels.json"), encoding="utf-8"))["classes"]
    provider = None if args.no_ai else llm.resolve_provider(args.provider)
    print("AI judge:", f"{provider.name}/{provider.model}" if provider else "none (fallback proxies)")

    rows, conf, secs = [], {t: {u: 0 for u in TIERS} for t in TIERS}, {}
    for c in labels:
        if args.only and c["id"] not in args.only:
            continue
        p = os.path.join(args.data, c["path"])
        feats = pipeline.extract(p, use_cache=True)
        t0 = time.time()
        judged = pipeline.judge_features(feats, provider, progress=lambda m: print("   ", m, flush=True), cache_key=pipeline._key(p))
        secs[c["id"]] = time.time() - t0
        rep = rubric.assess(feats, judged, brand_check=not args.no_brand)
        conf[c["tier"]][rep["grade"]] += 1
        rows.append((c, rep))
        if c["id"] in args.detail:
            print("=" * 80, f"\n{c['id']} true={c['tier']} predicted={rep['grade']} score={rep['score']} conf={rep['confidence']}")
            for dim in rep["dimensions"]:
                print(f"  {dim['title']:34s} {dim['score']}")
                for o in dim["observations"]:
                    print(f"      [{o['impact'][:3]}] {o['text']}  {o['evidence'][:1]}")
            print("  flags:", rep["flags"])

    print(f"\n{'id':7s}{'true':10s}{'pred':10s}{'score':>6s}  conf    " + " ".join(f"{k[:5]:>5s}" for k in KEYS) + "   judge-sec")
    exact = within = 0
    for c, rep in rows:
        d = {x["key"]: x["score"] for x in rep["dimensions"]}
        ti, pi = TIERS.index(c["tier"]), TIERS.index(rep["grade"])
        exact += ti == pi
        within += abs(ti - pi) <= 1
        mark = "OK " if ti == pi else ("~1 " if abs(ti - pi) == 1 else "XX ")
        dims = " ".join(f"{'-' if d[k] is None else int(d[k]):>5}" for k in KEYS)
        print(f"{c['id']:7s}{c['tier']:10s}{rep['grade']:10s}{rep['score']:6.1f}  {rep['confidence']:7s}{dims}  {mark} {secs.get(c['id'], 0):7.0f}")
    n = len(rows)
    print(f"\nMean AI-judge time per class: {sum(secs.values()) / max(len(secs), 1):.0f}s (cached results show ~0)")
    print(f"\nExact-tier accuracy: {exact}/{n} = {exact / n:.0%}    Within one tier: {within}/{n} = {within / n:.0%}")
    print("\nConfusion (rows = reviewer, cols = system):")
    print(" " * 11 + "".join(f"{t:>10s}" for t in TIERS))
    for t in TIERS:
        print(f"{t:11s}" + "".join(f"{conf[t][u]:>10d}" for u in TIERS))
    pairs = [(a, b) for i, (a, _) in enumerate(rows) for (b, _) in rows[i + 1:] if a["tier"] != b["tier"]]
    sc = {c["id"]: r["score"] for c, r in rows}
    ok = sum(1 for a, b in pairs if (sc[a["id"]] - sc[b["id"]]) * (TIERS.index(a["tier"]) - TIERS.index(b["tier"])) > 0)
    if pairs:
        print(f"\nRanking agreement (different-tier pairs ordered correctly by score): {ok}/{len(pairs)} = {ok / len(pairs):.0%}")


if __name__ == "__main__":
    main()
