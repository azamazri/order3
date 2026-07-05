"""A3 -- Signature-subgraph features + logistic LTR.

Per (query, product) we summarise the shared accord-pair "signature": how many rare
edges are shared (IDF-weighted), the single rarest shared edge, asymmetric coverage,
and Jaccard. A logistic ranker is trained out-of-fold (GroupKFold-5 by query)."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..data import Dataset
from .base import Features, Method, groupkfold_oof
from .p2_fusion import _logreg_fit_predict


def _features(feats: Features, qi: int) -> np.ndarray:
    inter = feats.shared_counts(qi)                  # shared unigrams
    sq = feats.QU[qi].sum()
    sp = feats.PU.sum(axis=1)
    cov_q = inter / max(sq, 1.0)
    cov_p = inter / np.maximum(sp, 1.0)

    # shared bigrams (rare-edge signature)
    qb = feats.QB[qi]                                # (1, B) sparse
    shared_b = feats.PB.multiply(qb)                 # (n_pool, B) shared edges
    n_shared_b = np.asarray((shared_b > 0).sum(axis=1)).ravel()
    idf_b = feats.idf_b
    w_shared_b = np.asarray(shared_b.multiply(idf_b).sum(axis=1)).ravel() \
        if feats.bvocab else np.zeros(feats.n_pool)
    # rarest shared edge = max idf among shared bigrams
    if feats.bvocab:
        masked = shared_b.multiply(idf_b).tocsr()
        max_rare = np.zeros(feats.n_pool)
        for r in range(feats.n_pool):
            seg = masked.data[masked.indptr[r]:masked.indptr[r + 1]]
            if seg.size:
                max_rare[r] = seg.max()
    else:
        max_rare = np.zeros(feats.n_pool)

    return np.column_stack([
        inter, n_shared_b, w_shared_b, max_rare,
        cov_q, cov_p, feats.jaccard(qi),
    ])


class A3Signature(Method):
    name = "A3_signature"
    tier = "T2 structure"
    stochastic = True

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        return groupkfold_oof(ds, lambda qi: _features(feats, qi),
                              _logreg_fit_predict, seed=seed)
