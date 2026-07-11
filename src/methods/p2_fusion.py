"""P2 -- order-2 + logistic fusion (a VARIANT of the proposed method).

Per (query, product) features: [bigram-cos, unigram-cos, |shared accords|].
Plain logistic regression trained out-of-fold with GroupKFold(5) by query, so no query
is ever in both train and test. Seed variance comes from the fold shuffling."""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Features, Method, groupkfold_oof


def _features(feats: Features, qi: int) -> np.ndarray:
    return np.column_stack([
        feats.bigram_cos(qi),
        feats.unigram_cos(qi),
        feats.shared_counts(qi),
    ])


def _make_logreg_fit_predict(C=1.0):
    def _fp(Xtr, ytr, Xte, seed):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                                 random_state=seed)
        clf.fit(sc.transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]
    return _fp


# default (backward-compatible) closure used by A3 and P2
_logreg_fit_predict = _make_logreg_fit_predict(1.0)


class P2Fusion(Method):
    name = "P2_fusion"
    tier = "T3 proposed"
    stochastic = True   # variance from fold shuffling only

    def __init__(self, C: float = 1.0):
        self.C = C

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        return groupkfold_oof(ds, lambda qi: _features(feats, qi),
                              _make_logreg_fit_predict(self.C), seed=seed)
