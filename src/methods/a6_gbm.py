"""A6 -- GradientBoosting fusion (overfit demo).

Fuses a handful of similarity features with a gradient-boosted tree ensemble, trained
out-of-fold (GroupKFold-5 by query). Included to show that a flexible learner on these
sparse labels does not beat the simple order-2 cosine."""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Features, Method, groupkfold_oof


def _features(feats: Features, qi: int) -> np.ndarray:
    inter = feats.shared_counts(qi)
    sq = feats.QU[qi].sum()
    sp = feats.PU.sum(axis=1)
    return np.column_stack([
        feats.jaccard(qi),
        feats.unigram_cos(qi),
        feats.bigram_cos(qi),
        inter,
        inter / max(sq, 1.0),
        inter / np.maximum(sp, 1.0),
    ])


def _gbm_fit_predict(Xtr, ytr, Xte, seed):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(random_state=seed, n_estimators=200,
                                     max_depth=3, learning_rate=0.05,
                                     subsample=0.8)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


class A6Gbm(Method):
    name = "A6_gbm_fusion"
    tier = "T2 structure"
    stochastic = True

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        return groupkfold_oof(ds, lambda qi: _features(feats, qi),
                              _gbm_fit_predict, seed=seed)
