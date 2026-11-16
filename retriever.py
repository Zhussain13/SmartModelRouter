"""FAQ lookup using sentence embeddings and cosine similarity."""
import numpy as np
import pandas as pd

import config

_embedder = None


def embed(texts):
    """Turns a list of texts into unit-length vectors (384 numbers each)."""
    texts = list(texts)
    if config.DEMO:                      # no download needed in demo mode
        import demo_models
        return demo_models.embed(texts)
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder.encode(texts, normalize_embeddings=True)


class FAQRetriever:
    def __init__(self, path: str = "data/faq.csv"):
        self.df = pd.read_csv(path)
        self.vecs = embed(self.df["question"].tolist())

    def lookup(self, query: str):
        """Returns (similarity, answer, matched_question)."""
        q = embed([query])[0]
        sims = self.vecs @ q            # cosine similarity (vectors are normalised)
        i = int(np.argmax(sims))
        return float(sims[i]), self.df.iloc[i]["answer"], self.df.iloc[i]["question"]
