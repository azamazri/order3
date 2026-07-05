"""B5 -- Word2Vec accord embeddings, mean-pooled per fragrance, cosine.

Trained on the catalogue itself: each fragrance's accord list is one "sentence".
Stochastic -> evaluated over several seeds (workers=1 for reproducibility)."""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Features, Method, _l2norm_rows


class B5Word2Vec(Method):
    name = "B5_word2vec"
    tier = "T1 baseline"
    stochastic = True

    def __init__(self, dim: int = 64, window: int = 5, epochs: int = 50, min_count: int = 1):
        self.dim, self.window, self.epochs, self.min_count = dim, window, epochs, min_count

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        from gensim.models import Word2Vec

        sentences = [p.accords for p in ds.products if p.accords] + \
                    [q.accords for q in ds.queries if q.accords]
        model = Word2Vec(sentences, vector_size=self.dim, window=self.window,
                         min_count=self.min_count, sg=1, epochs=self.epochs,
                         workers=1, seed=seed)

        def vec(accords):
            v = [model.wv[a] for a in accords if a in model.wv]
            return np.mean(v, axis=0) if v else np.zeros(self.dim)

        P = _l2norm_rows(np.vstack([vec(p.accords) for p in ds.products]))
        Q = _l2norm_rows(np.vstack([vec(q.accords) for q in ds.queries]))
        return Q @ P.T
