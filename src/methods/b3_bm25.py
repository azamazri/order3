"""B3 -- BM25 over accord tokens (product = document, query accords = query)."""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Features, Method


class B3BM25(Method):
    name = "B3_bm25"
    tier = "T1 baseline"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        from rank_bm25 import BM25Okapi

        corpus = [p.accords if p.accords else ["<empty>"] for p in ds.products]
        bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        out = np.zeros((len(ds.queries), ds.n_pool))
        for q in ds.queries:
            out[q.idx] = bm25.get_scores(q.accords if q.accords else ["<empty>"])
        return out
