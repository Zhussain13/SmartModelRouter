"""Packs labelled results into the live web dashboard (docs/index.html).

    python export_web_data.py --demo    # use outputs/demo/labels.csv (simulated)
    python export_web_data.py           # use outputs/real/labels.csv (your real run)

Run the pipeline first (python run_pipeline.py [--demo]). This script then:
  * scores every labelled question with the rule router,
  * scores every question with the learned router using 5-fold cross-validation,
    so each score comes from a model that never saw that question,
  * finds which FAQ-style questions the retriever would answer (similarity >= threshold),
  * writes everything as JSON inside docs/index.html, which works as a single file
    (double-click it, host it on GitHub Pages, or publish it anywhere).
"""
import argparse
import json
import os
import re
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
from retriever import FAQRetriever
from router import features, load_tuning, rule_score

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default=config.LABELS_PATH)
ap.add_argument("--page", default=os.path.join("docs", "index.html"))
ap.add_argument("--demo", action="store_true", help="use demo-mode labels")
args = ap.parse_args()

if not os.path.exists(args.labels):
    raise SystemExit(f"{args.labels} not found. Run: python run_pipeline.py"
                     + (" --demo" if config.DEMO else ""))

df = pd.read_csv(args.labels)
print(f"Loaded {len(df)} labelled questions from {args.labels}")

# ---- Router scores ----
X = features(df["query"])
y = df["needs_large"].to_numpy()
folds = max(2, min(5, int(np.bincount(y).min())))
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
p_learned = cross_val_predict(clf, X, y, method="predict_proba",
                              cv=StratifiedKFold(folds, shuffle=True, random_state=42))[:, 1]
p_rule = np.array([rule_score(q) for q in df["query"]])

# ---- FAQ traffic: real FAQ questions plus the ways students actually ask them ----
PARAPHRASES = [
    "When is IA-2?", "IA 2 exam date?", "When are the IA-2 exams for 3rd sem?",
    "What is the minimum attendance?", "How much attendance is required?",
    "Minimum attendance needed for exams?",
    "Where is the CSE office?", "Where can I find the CSE department office?",
    "How to apply for internship NOC?", "How do I get an NOC for my internship?",
    "What is the pass mark in VTU?", "What are the passing marks in VTU?",
    "Who is my class mentor?", "Who is the mentor for 3rd semester?",
]
faq = FAQRetriever()
faq_items = []
for q in faq.df["question"].tolist() + PARAPHRASES:
    sim, _, matched = faq.lookup(q)
    if sim >= config.RETRIEVAL_THRESHOLD:
        faq_items.append({"q": q, "m": matched, "s": round(sim, 3)})
print(f"{len(faq_items)} FAQ-style questions are answered by retrieval "
      f"(threshold {config.RETRIEVAL_THRESHOLD})")


# ---- Pack questions compactly ----
def topic(q):
    ql = q.lower()
    for name, keys in [("C", ["printf", "int ", "pointer", "malloc", "scanf", "c?", " in c", "#define", "struct", "char "]),
                       ("DBMS", ["sql", "dbms", "database", "normal", "schema", "join", "transaction", "relation", "key"]),
                       ("OS", ["process", "thread", "deadlock", "page", "schedul", "semaphore", "memory", "kernel", "fork", "disk"]),
                       ("CN", ["tcp", "udp", "ip ", "ipv", "network", "router", "dns", "http", "subnet", "mac ", "protocol", "lan", "crc", "aloha"]),
                       ("DS & Algo", ["tree", "stack", "queue", "list", "graph", "sort", "search", "hash", "heap", "complexity", "algorithm", "dynamic"]),
                       ("Maths", ["prove", "probability", "matrix", "derivative", "binary", "boolean", "prime", "gcd", "mod ", "k-map"]),
                       ("OOP", ["class", "object", "inherit", "polymorph", "java", "python", "design pattern", "interface"])]:
        if any(k in ql for k in keys):
            return name
    return "General"


rows = []
for i, r in df.reset_index(drop=True).iterrows():
    rows.append([
        re.sub(r"\s+", " ", str(r["query"])).strip(),
        topic(str(r["query"])),
        round(float(p_rule[i]), 3),
        round(float(p_learned[i]), 3),
        int(r["small_conf"]),
        int(r["small_ok"]),
        int(r["large_ok"]),
        round(float(r["small_cost"]), 8),
        round(float(r["large_cost"]), 8),
        round(float(r["small_latency"]), 2),
        round(float(r["large_latency"]), 2),
    ])

data = {
    "meta": {
        "mode": "demo" if config.DEMO else "real",
        "small_model": config.SMALL_MODEL,
        "large_model": config.LARGE_MODEL,
        "price_in": config.LARGE_PRICE_IN,
        "price_out": config.LARGE_PRICE_OUT,
        "retrieval_threshold": config.RETRIEVAL_THRESHOLD,
        "labelled": len(df),
        "folds": folds,
        "generated": date.today().isoformat(),
        "small_acc": round(float(df.small_ok.mean()), 4),
        "large_acc": round(float(df.large_ok.mean()), 4),
        "large_avg_cost": round(float(df.large_cost.mean()), 8),
        "thresholds": {k: v for k, v in load_tuning().items() if k in ("learned", "rules", "cascade", "tuned", "target")},
    },
    "cols": ["q", "topic", "p_rule", "p_learned", "conf", "small_ok", "large_ok",
             "small_cost", "large_cost", "small_lat", "large_lat"],
    "questions": rows,
    "faq": faq_items,
}

# ---- Write into the page between the markers ----
with open(args.page, encoding="utf-8") as f:
    page = f.read()
start, end = "<!--SMR-DATA-START-->", "<!--SMR-DATA-END-->"
if start not in page or end not in page:
    raise SystemExit(f"{args.page} has no data markers.")
payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
block = f'{start}<script id="smr-data" type="application/json">{payload}</script>{end}'
page = page[:page.index(start)] + block + page[page.index(end) + len(end):]
with open(args.page, "w", encoding="utf-8") as f:
    f.write(page)
print(f"Wrote {len(rows)} questions + {len(faq_items)} FAQ items into {args.page} "
      f"({len(payload) / 1024:.0f} KB). Open it in a browser.")
