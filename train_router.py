"""Step 2: train the learned router (logistic regression on embeddings).

    python train_router.py           # uses outputs/real/labels.csv
    python train_router.py --demo    # uses outputs/demo/labels.csv
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import config
from router import cheapest_threshold, features, rule_score

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default=config.LABELS_PATH)
ap.add_argument("--demo", action="store_true", help="use demo-mode files")
args = ap.parse_args()

df = pd.read_csv(args.labels)
print(f"{len(df)} labelled questions, {df.needs_large.mean():.1%} need the large model")

if df.needs_large.nunique() < 2 or df.needs_large.value_counts().min() < 3:
    raise SystemExit("Need at least 3 'small OK' and 3 'needs large' examples. "
                     "Label more questions (especially harder ones).")

train, test = train_test_split(df, test_size=0.3, random_state=42,
                               stratify=df["needs_large"])

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
X_train = features(train["query"])
clf.fit(X_train, train["needs_large"])

p = clf.predict_proba(features(test["query"]))[:, 1]
print(classification_report(test["needs_large"], (p >= 0.5).astype(int),
                            target_names=["small OK", "needs large"], zero_division=0))
print("ROC-AUC:", round(roc_auc_score(test["needs_large"], p), 3))

joblib.dump(clf, config.ROUTER_PATH)
test.to_csv(config.TEST_PATH, index=False)
print(f"Saved {config.ROUTER_PATH} and {config.TEST_PATH}")

# ---- Threshold tuning, on the TRAINING questions only ----
# The learned router's scores come from cross-validation inside the training set, so each
# question is scored by a model that did not see it. The test set stays untouched.
folds = max(2, min(5, int(train.needs_large.value_counts().min())))
p_oof = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"),
                          X_train, train["needs_large"], method="predict_proba",
                          cv=StratifiedKFold(folds, shuffle=True, random_state=42))[:, 1]
cols = (train.small_ok, train.large_ok, train.small_cost, train.large_cost)
grid = np.round(np.arange(0, 1.0001, 0.05), 2)
scores = {"learned": np.asarray(p_oof), "rules": np.array([rule_score(q) for q in train["query"]]),
          "cascade": train.small_conf.to_numpy()}
defaults = {"learned": config.ROUTER_THRESHOLD, "rules": config.ROUTER_THRESHOLD,
            "cascade": config.CASCADE_MIN_CONF}
tuned = {}
for k, sc in scores.items():
    cascade = k == "cascade"
    t, acc, cost = cheapest_threshold(sc, *cols, range(1, 12) if cascade else grid, cascade=cascade)
    share_large = np.mean(sc < t) if cascade else np.mean(sc >= t)
    if share_large >= 0.99:   # only "send everything to large" reaches the target: not a router
        print(f"  note: the {k} router cannot reach the target without sending everything to the "
              f"large model; keeping the default threshold {defaults[k]}")
        t = defaults[k]
        big = sc < t if cascade else sc >= t
        acc = float(np.where(big, train.large_ok, train.small_ok).mean())
        cost = float((train.small_cost + np.where(big, train.large_cost, 0)).mean() if cascade
                     else np.where(big, train.large_cost, train.small_cost).mean())
    tuned[k] = (t, acc, cost)
print(f"\nTuned thresholds (cheapest that keeps {config.ACCURACY_TARGET:.0%} of "
      "large-model accuracy on the training questions):")
for k, (t, acc, cost) in tuned.items():
    rule = f"confidence < {int(t)}" if k == "cascade" else f"p_hard >= {t:.2f}"
    print(f"  {k:8s}: {rule:18s} train accuracy {acc:.1%}, cost/1k ${cost * 1000:.4f}")
with open(config.TUNING_PATH, "w") as f:
    json.dump({"learned": tuned["learned"][0], "rules": tuned["rules"][0],
               "cascade": int(tuned["cascade"][0]), "target": config.ACCURACY_TARGET}, f, indent=2)
print(f"Saved {config.TUNING_PATH}")
