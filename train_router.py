"""Step 2: train the learned router (logistic regression on embeddings)."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from router import features

df = pd.read_csv("data/labels.csv")
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
                            target_names=["small OK", "needs large"]))
print("ROC-AUC:", round(roc_auc_score(test["needs_large"], p), 3))

joblib.dump(clf, "router.joblib")
test.to_csv("data/test_split.csv", index=False)
print("Saved router.joblib and data/test_split.csv")
