"""Decides how hard a question is. Each router outputs p_hard in [0, 1]."""
import re

import joblib
import numpy as np

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


class RuleRouter:
    name = "rules"

    def p_hard(self, q: str) -> float:
        return rule_score(q)

    def p_hard_many(self, qs) -> np.ndarray:
        return np.array([rule_score(q) for q in qs])


class LearnedRouter:
    name = "learned"

    def __init__(self, path: str = "router.joblib"):
        self.clf = joblib.load(path)

    def p_hard(self, q: str) -> float:
        return float(self.p_hard_many([q])[0])

    def p_hard_many(self, qs) -> np.ndarray:
        """Batched prediction: embeds all questions in one go (much faster)."""
        return self.clf.predict_proba(features(qs))[:, 1]
