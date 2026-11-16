"""Built-in simulator used in demo mode (--demo or SMR_DEMO=1).

It stands in for the small model, the large model, the judge and the embedding
model so the whole project runs instantly with no downloads, keys or GPU.

The simulation is deterministic (same question -> same result) and loosely realistic:
each question gets a hidden difficulty score from its wording plus random noise.
  * the small model fails more often as difficulty rises,
  * the large model is almost always right,
  * the small model's self-reported confidence drops with difficulty, but noisily.
The noise is deliberate: if difficulty were exactly the rule router's formula,
the rule router would look perfect. DEMO NUMBERS ARE SIMULATED, NOT REAL RESULTS.
"""
import hashlib
import re

import numpy as np

_HARD_HINTS = ["why", "debug", "bug", "error", "crash", "segfault", "prove", "derive",
               "design", "analy", "compare", "trace", "output", "calculate", "compute",
               "solve", "step by step", "minimi", "normalise", "normalize", "explain how"]


def _u(text: str, salt: str) -> float:
    """Deterministic pseudo-random number in [0, 1) for (text, salt)."""
    h = hashlib.md5(f"{salt}|{text}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0x100000000


def difficulty(q: str) -> float:
    """Hidden difficulty in [0, 1] (the simulator's 'ground truth')."""
    ql = q.lower()
    d = 0.15
    d += 0.30 * any(w in ql for w in _HARD_HINTS)
    d += 0.20 * bool(re.search(r"[{};=]|\w\(", q))           # code-like
    d += 0.15 * min(len(q.split()), 40) / 40                   # longer = harder
    d += 0.10 * bool(re.search(r"\d", q))                      # numerical
    d += 0.35 * (_u(q, "noise") - 0.5)                         # things features miss
    return float(min(max(d, 0.0), 1.0))


def small_correct(q: str) -> bool:
    return _u(q, "small") < 1.12 - 1.0 * difficulty(q)


def large_correct(q: str) -> bool:
    return _u(q, "large") < 0.99 - 0.12 * difficulty(q)


def small_confidence(q: str) -> int:
    c = 9.5 - 6 * difficulty(q) + 3 * (_u(q, "conf") - 0.5)
    if not small_correct(q):
        c -= 1.5                 # models are somewhat (not fully) calibrated
    return int(min(max(round(c), 1), 10))


def _strip(prompt: str) -> str:
    """Remove the confidence instruction added by ask_small_with_confidence."""
    return prompt.split("\n\nAfter your answer, write one final line")[0]


def answer(prompt: str, which: str):
    """Returns (text, tokens_in, tokens_out, latency_s) for 'small' or 'large'."""
    q = _strip(prompt)
    d = difficulty(q)
    tin = int(len(prompt.split()) * 1.3) + 25
    if which == "small":
        tout = int(60 + 220 * d)
        text = (f"[DEMO-SMALL] Simulated answer from the small local model to: {q}")
        if prompt != q:
            text += f"\nCONFIDENCE: {small_confidence(q)}"
        lat = 0.8 + 3.0 * d
    else:
        tout = int(150 + 600 * d)          # bigger models write (and think) more
        text = f"[DEMO-LARGE] Simulated answer from the large model to: {q}"
        lat = 1.5 + 4.0 * d
    return text, tin, tout, lat


def judge(question: str, candidate: str) -> bool:
    if candidate.startswith("[DEMO-SMALL]"):
        return small_correct(question)
    if candidate.startswith("[DEMO-LARGE]"):
        return large_correct(question)
    return True


# Filler words carry no meaning for FAQ matching ("what is the ..." made unrelated questions
# look alike). Used only for FAQ lookup, not for the router's features.
_STOP = set("a an the is are was were be what when where who how why which do does did i my me "
            "you your of in on for to and or with can will it this that there at by from as please "
            "tell".split())


def embed(texts, dim: int = 384, drop_stopwords: bool = False) -> np.ndarray:
    """Hashing embedder: word and character-trigram counts folded into `dim` buckets."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
        if drop_stopwords:
            t = " ".join(w for w in t.split() if w not in _STOP)
        grams = t.split() + [t[j:j + 3] for j in range(len(t) - 2) if " " not in t[j:j + 3]]
        for g in grams:
            out[i, int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % dim] += 1
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out
