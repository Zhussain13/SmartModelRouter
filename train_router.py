"""Step 2: train the learned router (logistic regression on embeddings).

    python train_router.py           # uses outputs/real/labels.csv
    python train_router.py --demo    # uses outputs/demo/labels.csv
"""
import argparse

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

import config
from router import features

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
clf.fit(features(train["query"]), train["needs_large"])

p = clf.predict_proba(features(test["query"]))[:, 1]
print(classification_report(test["needs_large"], (p >= 0.5).astype(int),
                            target_names=["small OK", "needs large"], zero_division=0))
print("ROC-AUC:", round(roc_auc_score(test["needs_large"], p), 3))

joblib.dump(clf, config.ROUTER_PATH)
test.to_csv(config.TEST_PATH, index=False)
print(f"Saved {config.ROUTER_PATH} and {config.TEST_PATH}")
