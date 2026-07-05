"""B2 -- Cosine over accord TF-IDF (order-1 marginal). Main baseline."""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Features, Method


class B2TfidfCosine(Method):
    name = "B2_tfidf_cos"
    tier = "T1 baseline"

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        return feats.QUt @ feats.PUt.T
