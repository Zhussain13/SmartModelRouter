"""Step 3: compare all routing strategies and draw the cost-vs-accuracy graph.

Every answer was already generated and graded in step 1, so this step makes
NO model calls. It replays each strategy on the held-out test set.

    python evaluate.py
    python evaluate.py --demo
    python evaluate.py --test outputs/real/labels_kanglish.csv --out outputs/real/results_kanglish
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from router import LearnedRouter, RuleRouter, load_tuning

ap = argparse.ArgumentParser()
ap.add_argument("--test", default=config.TEST_PATH, help="labelled questions to evaluate on")
ap.add_argument("--out", default=config.RESULTS_PREFIX,
                help="output name: writes <out>.csv and <out>.png")
ap.add_argument("--demo", action="store_true", help="use demo-mode files")
args = ap.parse_args()

df = pd.read_csv(args.test)
N = len(df)
print(f"Evaluating on {N} held-out questions ({args.test})\n")


def score(use_large, cascade=False):
    """use_large: boolean array, True = this question went to the large model."""
    use_large = np.asarray(use_large, dtype=bool)
    acc = np.where(use_large, df.large_ok, df.small_ok).mean()
    if cascade:   # cascade always pays for the small attempt first
        cost = df.small_cost + np.where(use_large, df.large_cost, 0)
        lat = df.small_latency + np.where(use_large, df.large_latency, 0)
    else:
        cost = np.where(use_large, df.large_cost, df.small_cost)
        lat = np.where(use_large, df.large_latency, df.small_latency)
    return dict(accuracy=float(acc), cost_per_1k=float(cost.sum() / N * 1000),
                pct_large=float(use_large.mean()), avg_latency_s=float(np.mean(lat)))


# Oracle = a perfect router: use the large model only when the small model is
# wrong AND the large model is right (otherwise escalating gains nothing).
oracle = (df.small_ok == 0) & (df.large_ok == 1)

p_rule = RuleRouter().p_hard_many(df["query"])
p_learned = LearnedRouter().p_hard_many(df["query"])

rows = [
    {"strategy": "always small", **score(np.zeros(N))},
    {"strategy": "always large", **score(np.ones(N))},
    {"strategy": "oracle (perfect router)", **score(oracle)},
    {"strategy": "rules @0.5", **score(p_rule >= 0.5)},
    {"strategy": "learned @0.5", **score(p_learned >= 0.5)},
]
for c in [5, 7, 9]:
    rows.append({"strategy": f"cascade conf<{c}",
                 **score(df.small_conf < c, cascade=True)})
tune = load_tuning()
if tune["tuned"]:   # thresholds chosen on the training questions, scored here on unseen ones
    rows += [
        {"strategy": f"rules tuned @{tune['rules']:.2f}", **score(p_rule >= tune["rules"])},
        {"strategy": f"learned tuned @{tune['learned']:.2f}", **score(p_learned >= tune["learned"])},
        {"strategy": f"cascade tuned conf<{tune['cascade']}", **score(df.small_conf < tune["cascade"], cascade=True)},
    ]

res = pd.DataFrame(rows)
pd.set_option("display.width", 120)
print(res.round(4).to_string(index=False))
res.to_csv(f"{args.out}.csv", index=False)

# ---------- Cost vs accuracy curves ----------
thresholds = np.linspace(0, 1.0001, 101)
curves = {
    "Rule router": [score(p_rule >= t) for t in thresholds],
    "Learned router": [score(p_learned >= t) for t in thresholds],
    "Cascade": [score(df.small_conf < c, cascade=True) for c in range(1, 12)],
}

plt.figure(figsize=(8, 5))
for (name, pts), mk in zip(curves.items(), ["o", "o", "s"]):
    pts = sorted(pts, key=lambda x: x["cost_per_1k"])
    plt.plot([x["cost_per_1k"] for x in pts], [x["accuracy"] for x in pts],
             marker=mk, markersize=3, label=name)
for name in ["always small", "always large", "oracle (perfect router)"]:
    r = res[res.strategy == name].iloc[0]
    plt.scatter(r.cost_per_1k, r.accuracy, s=110, marker="*", zorder=5, color="black")
    right = name == "always large"
    plt.annotate(name, (r.cost_per_1k, r.accuracy), textcoords="offset points",
                 xytext=(-8 if right else 6, -14), ha="right" if right else "left")
plt.xlabel("Cost per 1,000 questions (USD, API-equivalent)")
plt.ylabel("Accuracy")
title = "Cost vs accuracy of routing strategies"
plt.title(title + (" (DEMO: simulated data)" if config.DEMO else ""))
plt.grid(alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{args.out}.png", dpi=150)
print(f"\nSaved {args.out}.csv and {args.out}.png")

# ---------- Headline: cheapest way to keep 95% of large-model accuracy ----------
large = res[res.strategy == "always large"].iloc[0]
target = 0.95 * large.accuracy
print(f"\nCheapest setting reaching 95% of large-model accuracy ({target:.1%}):")
for name, pts in curves.items():
    ok = [x for x in pts if x["accuracy"] >= target]
    if not ok or large.cost_per_1k <= 0:
        print(f"  {name:15s}: not reached")
        continue
    best = min(ok, key=lambda x: x["cost_per_1k"])
    print(f"  {name:15s}: {best['cost_per_1k'] / large.cost_per_1k:5.0%} of the cost "
          f"({best['pct_large']:.0%} of questions sent to the large model, "
          f"accuracy {best['accuracy']:.1%})")
