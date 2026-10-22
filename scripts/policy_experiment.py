"""Offline experiment: re-score the stored moves results under different verification policies. No LLM calls, no quota.

    python scripts/policy_experiment.py

Needs `python scripts/evaluate.py --provider <p>` to have been run first (it stores the labelled events with verifier
probabilities). Prints accuracy metrics for each policy so a policy can be chosen on evidence instead of guesswork."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from assessor import config, moves, pipeline, rubric  # noqa: E402

TIERS = config.TIERS


def load(data):
    labels = json.load(open(os.path.join(ROOT, "labels.json"), encoding="utf-8"))["classes"]
    rows = []
    for c in labels:
        p = os.path.join(data, c["path"])
        feats = pipeline.extract(p, use_cache=True)
        fs = sorted(glob.glob(os.path.join(config.CACHE_DIR, "judged", f"{pipeline._key(p)}_moves-{moves.VERSION}_*.json")), key=os.path.getmtime)
        if not fs:
            continue
        rows.append((c, feats, json.load(open(fs[-1], encoding="utf-8"))))
    return rows


def metrics(rows, min_p, brand=True):
    res = []
    for c, feats, j in rows:
        rep = rubric.assess(feats, moves.rescore(feats, j, min_p), brand_check=brand)
        res.append((c, rep))
    exact = sum(TIERS.index(c["tier"]) == TIERS.index(r["grade"]) for c, r in res)
    within = sum(abs(TIERS.index(c["tier"]) - TIERS.index(r["grade"])) <= 1 for c, r in res)
    pairs = [(a, b) for i, (a, _) in enumerate(res) for (b, _) in res[i + 1:] if a["tier"] != b["tier"]]
    sc = {c["id"]: r["score"] for c, r in res}
    rank = sum((sc[a["id"]] - sc[b["id"]]) * (TIERS.index(a["tier"]) - TIERS.index(b["tier"])) > 0 for a, b in pairs) / max(len(pairs), 1)
    by_tier = {t: [r["score"] for c, r in res if c["tier"] == t] for t in TIERS}
    return len(res), exact, within, rank, by_tier


def main():
    data = os.environ.get("DATASET_DIR", r"D:\Humanity\class recordings")
    rows = load(data)
    print(f"{len(rows)} classes with stored events\n")
    print(f"{'policy':34s}{'exact':>7s}{'+-1':>6s}{'rank':>7s}   mean score per tier (Poor/Avg/Good/Exc)")
    for label, mp in [("no verifier (quote check only)", 0.0), ("drop if NLI p < 0.05", 0.05), ("drop if NLI p < 0.10", 0.10),
                      ("drop if NLI p < 0.20", 0.20), ("drop if NLI p < 0.35", 0.35)]:
        n, ex, wi, rk, bt = metrics(rows, mp)
        means = "/".join(f"{sum(v) / len(v):.0f}" if v else "-" for v in bt.values())
        print(f"{label:34s}{ex:>4d}/{n:<2d}{wi:>4d}/{n:<1d}{rk:>7.0%}   {means}")


if __name__ == "__main__":
    main()
