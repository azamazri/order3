r"""P1 -- Order-2 co-occurrence TF-IDF.

Token set = accords UNION accord-pairs (unigram + bigram). IDF-weighted, jointly
L2-normalised, cosine. The score decomposes additively into an order-1 (marginal)
term and an order-2 (co-occurrence) term, because the two blocks share one L2 norm:

    cos(q, p) = <q_uni, p_uni> + <q_bi, p_bi>
                \_____________/   \___________/
                  order-1           order-2
"""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Features, Method


class P1Order2(Method):
    name = "P1_order2"
    tier = "T3 proposed"

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        return feats.QC @ feats.PC.T

    def decompose(self, feats: Features):
        """Return (order1, order2) score matrices; their sum == scores()."""
        nu = feats.n_uni
        order1 = feats.QC[:, :nu] @ feats.PC[:, :nu].T
        order2 = feats.QC[:, nu:] @ feats.PC[:, nu:].T
        return order1, order2
