"""A6 -- GradientBoosting fusion.

Fuses a handful of similarity features (including a bigram cosine, itself an order-2
quantity) with a gradient-boosted tree ensemble (sklearn.GradientBoostingClassifier,
not XGBoost/LightGBM), trained out-of-fold (GroupKFold-5 by query). Class imbalance is
handled with balanced sample_weight on .fit()."""
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


def _make_gbm_fit_predict(n_estimators=200, max_depth=3, learning_rate=0.05):
    def _fp(Xtr, ytr, Xte, seed):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        clf = GradientBoostingClassifier(random_state=seed, n_estimators=n_estimators,
                                         max_depth=max_depth, learning_rate=learning_rate,
                                         subsample=0.8)
        # GradientBoostingClassifier has no class_weight; use balanced sample_weight
        # so imbalance handling matches A3/A4 (class_weight="balanced").
        clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        return clf.predict_proba(Xte)[:, 1]
    return _fp


class A6Gbm(Method):
    name = "A6_gbm_fusion"
    tier = "T2 structure"
    stochastic = True

    def __init__(self, n_estimators: int = 200, max_depth: int = 3,
                 learning_rate: float = 0.05):
        self.n_estimators, self.max_depth, self.learning_rate = \
            n_estimators, max_depth, learning_rate

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        fp = _make_gbm_fit_predict(self.n_estimators, self.max_depth, self.learning_rate)
        return groupkfold_oof(ds, lambda qi: _features(feats, qi), fp, seed=seed)
