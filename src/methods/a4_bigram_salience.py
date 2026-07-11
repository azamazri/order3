"""A4 -- Supervised per-bigram salience (high-dim L2 logistic).

Feature vector for (query, product) = indicator over the full bigram vocabulary,
1 where a specific accord-pair is shared by both. The model learns one salience
weight per bigram (hundreds to thousands of dimensions) from one positive per query.
Trained out-of-fold (GroupKFold-5 by query)."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..data import Dataset
from .base import Features, Method, groupkfold_oof


def _features(feats: Features, qi: int) -> sparse.csr_matrix:
    # shared-bigram indicator: PB AND QB[qi]
    qb = feats.QB[qi]
    shared = feats.PB.multiply(qb)
    return (shared > 0).astype(float).tocsr()        # (n_pool, B)


def _make_l2logreg_fit_predict(C=1.0):
    def _fp(Xtr, ytr, Xte, seed):
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=C, max_iter=2000,
                                 class_weight="balanced", solver="liblinear",
                                 random_state=seed)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    return _fp


class A4BigramSalience(Method):
    name = "A4_bigram_salience"
    tier = "T2 structure"
    stochastic = True

    def __init__(self, C: float = 1.0):
        self.C = C

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        return groupkfold_oof(ds, lambda qi: _features(feats, qi),
                              _make_l2logreg_fit_predict(self.C), seed=seed,
                              sparse_feats=True)
