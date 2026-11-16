"""FAQ lookup using sentence embeddings and cosine similarity."""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config

_embedder = None


def embed(texts):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder.encode(list(texts), normalize_embeddings=True)


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
