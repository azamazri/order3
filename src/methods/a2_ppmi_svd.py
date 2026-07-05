"""A2 -- PPMI-SVD accord co-occurrence embedding, mean-pooled, cosine.

Build an accord-accord co-occurrence count matrix over fragments, convert to PPMI,
factor with TruncatedSVD -> dense accord embeddings. Fragrance vector = mean of its
accord embeddings; cosine. Stochastic (randomised SVD) -> several seeds."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..data import Dataset
from .base import Features, Method, _l2norm_rows


class A2PpmiSvd(Method):
    name = "A2_ppmi_svd"
    tier = "T2 structure"
    stochastic = True

    def __init__(self, dim: int = 50):
        self.dim = dim

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD

        V = len(feats.vocab)
        C = np.zeros((V, V))
        for frag in [p.accords for p in ds.products] + [q.accords for q in ds.queries]:
            idxs = [feats.vidx[a] for a in set(frag)]
            for i, j in combinations(idxs, 2):
                C[i, j] += 1
                C[j, i] += 1
            for i in idxs:
                C[i, i] += 1  # self-count anchors isolated accords

        # PPMI
        total = C.sum()
        row = C.sum(axis=1, keepdims=True)
        col = C.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((C * total) / (row * col))
        pmi[~np.isfinite(pmi)] = 0.0
        ppmi = np.maximum(pmi, 0.0)

        k = min(self.dim, V - 1)
        svd = TruncatedSVD(n_components=k, random_state=seed)
        emb = svd.fit_transform(ppmi)         # (V, k)

        def vec(accords):
            v = [emb[feats.vidx[a]] for a in accords if a in feats.vidx]
            return np.mean(v, axis=0) if v else np.zeros(k)

        P = _l2norm_rows(np.vstack([vec(p.accords) for p in ds.products]))
        Q = _l2norm_rows(np.vstack([vec(q.accords) for q in ds.queries]))
        return Q @ P.T
