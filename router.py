"""Decides how hard a question is. Each router outputs p_hard in [0, 1].

p_hard >= threshold  -> send to the large model, otherwise the small model.
(The cascade router needs no code here: it uses the small model's own confidence,
see models.ask_small_with_confidence and evaluate.py.)
"""
import re

import joblib
import numpy as np

import config

HARD_WORDS = ["why", "debug", "error", "segfault", "crash", "prove", "derive",
              "compare", "design", "optimi", "complexity", "step by step",
              "analy", "difference between", "trace", "output of"]

CODE_PATTERN = r"[{};]|#include|int main|def |printf\(|\bclass\b|=>"


def rule_score(q: str) -> float:
    """Hand-written difficulty score in [0, 1]."""
    s = 0.0
    ql = q.lower()
    if len(q.split()) > 40:
        s += 0.4
    if re.search(CODE_PATTERN, q):
        s += 0.4
    if any(w in ql for w in HARD_WORDS):
        s += 0.3
    if re.search(r"\d+\s*[-+*/^]\s*\d+", q):
        s += 0.2
    return min(s, 1.0)


def features(queries):
    """Embedding (384 numbers) + 2 hand-made features per query."""
    from retriever import embed   # imported lazily so the rule router needs no model download
    queries = list(queries)
    E = embed(queries)
    extra = np.array([[rule_score(q), min(len(q.split()), 200) / 200]
                      for q in queries])
    return np.hstack([E, extra])


def load_tuning() -> dict:
    """Thresholds chosen by train_router.py, or the config defaults if not tuned yet."""
    import json
    import os
    out = {"learned": config.ROUTER_THRESHOLD, "rules": config.ROUTER_THRESHOLD,
           "cascade": config.CASCADE_MIN_CONF, "tuned": False}
    if os.path.exists(config.TUNING_PATH):
        with open(config.TUNING_PATH) as f:
            out.update(json.load(f), tuned=True)
    return out


def cheapest_threshold(scores, small_ok, large_ok, small_cost, large_cost, grid,
                       target=None, cascade=False):
    """Cheapest threshold whose accuracy is >= target x (always-large accuracy).

    Normal routers send a question to the large model when score >= t.
    The cascade escalates when confidence < t and always pays for the small attempt.
    Returns (threshold, accuracy, cost_per_question); falls back to the most accurate
    setting if none reaches the target.
    """
    target = config.ACCURACY_TARGET if target is None else target
    scores, small_ok, large_ok = map(np.asarray, (scores, small_ok, large_ok))
    small_cost, large_cost = np.asarray(small_cost), np.asarray(large_cost)
    goal = target * large_ok.mean()
    best, most_accurate = None, None
    for t in grid:
        big = scores < t if cascade else scores >= t
        acc = np.where(big, large_ok, small_ok).mean()
        cost = (small_cost + np.where(big, large_cost, 0)).mean() if cascade \
            else np.where(big, large_cost, small_cost).mean()
        cand = (float(t), float(acc), float(cost))
        if most_accurate is None or acc > most_accurate[1] or (acc == most_accurate[1] and cost < most_accurate[2]):
            most_accurate = cand
        if acc >= goal and (best is None or cost < best[2]):
            best = cand
    return best or most_accurate


class RuleRouter:
    name = "rules"

    def p_hard(self, q: str) -> float:
        return rule_score(q)

    def p_hard_many(self, qs) -> np.ndarray:
        return np.array([rule_score(q) for q in qs])


class LearnedRouter:
    name = "learned"

    def __init__(self, path: str | None = None):
        self.clf = joblib.load(path or config.ROUTER_PATH)

    def p_hard(self, q: str) -> float:
        return float(self.p_hard_many([q])[0])

    def p_hard_many(self, qs) -> np.ndarray:
        """Batched prediction: embeds all questions in one go (much faster)."""
        return self.clf.predict_proba(features(qs))[:, 1]
