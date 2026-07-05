"""B1 -- Jaccard similarity over accord sets (Nurmuthia/BINUS-style set baseline)."""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Features, Method


class B1Jaccard(Method):
    name = "B1_jaccard"
    tier = "T1 baseline"

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        return np.vstack([feats.jaccard(qi) for qi in range(feats.n_q)])
