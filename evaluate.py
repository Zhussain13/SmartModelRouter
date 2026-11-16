"""Step 3: compare all routing strategies and draw the cost-vs-accuracy graph.

Every answer was already generated and graded in step 1, so this step makes
NO API calls. It replays each strategy on the held-out test set.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from router import LearnedRouter, RuleRouter

df = pd.read_csv("data/test_split.csv")
N = len(df)
print(f"Evaluating on {N} held-out questions\n")


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
    return dict(accuracy=acc, cost_per_1k=cost.sum() / N * 1000,
                pct_large=use_large.mean(), avg_latency_s=float(np.mean(lat)))


# Oracle = a perfect router: use the large model only when the small model is
# wrong AND the large model is right (otherwise escalating gains nothing).
oracle = (df.small_ok == 0) & (df.large_ok == 1)

rows = []
rows.append({"strategy": "always small", **score(np.zeros(N))})
rows.append({"strategy": "always large", **score(np.ones(N))})
rows.append({"strategy": "oracle (perfect router)", **score(oracle)})

p_rule = RuleRouter().p_hard_many(df["query"])
rows.append({"strategy": "rules @0.5", **score(p_rule >= 0.5)})

p_learned = LearnedRouter().p_hard_many(df["query"])
rows.append({"strategy": "learned @0.5", **score(p_learned >= 0.5)})

for c in [5, 7, 9]:
    rows.append({"strategy": f"cascade conf<{c}",
                 **score(df.small_conf < c, cascade=True)})

res = pd.DataFrame(rows)
pd.set_option("display.width", 120)
print(res.round(4).to_string(index=False))
res.to_csv("results.csv", index=False)

# ---------- Cost vs accuracy curve ----------
plt.figure(figsize=(8, 5))
for name, p in [("Rule router", p_rule), ("Learned router", p_learned)]:
    pts = [score(p >= t) for t in np.linspace(0, 1.0001, 21)]
    plt.plot([x["cost_per_1k"] for x in pts], [x["accuracy"] for x in pts],
             marker="o", label=name)
pts = [score(df.small_conf < c, cascade=True) for c in range(1, 12)]
plt.plot([x["cost_per_1k"] for x in pts], [x["accuracy"] for x in pts],
         marker="s", label="Cascade")
for name in ["always small", "always large", "oracle (perfect router)"]:
    r = res[res.strategy == name].iloc[0]
    plt.scatter(r.cost_per_1k, r.accuracy, s=90, marker="*", zorder=5)
    plt.annotate(name, (r.cost_per_1k, r.accuracy),
                 textcoords="offset points", xytext=(6, -12))
plt.xlabel("Cost per 1,000 questions (USD, at list price)")
plt.ylabel("Accuracy")
plt.title("Cost vs accuracy of routing strategies")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("\nSaved results.csv and results.png")

# ---------- Headline number ----------
large_acc = res[res.strategy == "always large"].accuracy.iloc[0]
large_cost = res[res.strategy == "always large"].cost_per_1k.iloc[0]
target = 0.95 * large_acc
pts = [score(p_learned >= t) for t in np.linspace(0, 1.0001, 101)]
ok = [x for x in pts if x["accuracy"] >= target]
if ok and large_cost > 0:
    best = min(ok, key=lambda x: x["cost_per_1k"])
    print(f"\nHEADLINE: learned router reaches 95% of large-model accuracy "
          f"at {best['cost_per_1k'] / large_cost:.0%} of its cost "
          f"({best['pct_large']:.0%} of questions sent to the large model).")
