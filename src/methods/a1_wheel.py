"""A1 -- Edwards-Wheel tree-Wasserstein over accord distributions.

Each fragrance becomes a uniform distribution over its lexicon-mapped accords; the
score is the negative closed-form tree-W1 distance on the frozen Edwards wheel.
Perceptual/taxonomic structure (data-independent). The lexicon is
frozen and never touches the labels."""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from ..wheel import WheelTree
from .base import Features, Method


class A1Wheel(Method):
    name = "A1_wheel_treeW"
    tier = "T2 structure"

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        tree = WheelTree()
        Pd = np.vstack([tree.distribution(p.accords) for p in ds.products])
        Qd = np.vstack([tree.distribution(q.accords) for q in ds.queries])

        # vectorised closed-form tree-W1 over all (query, product) pairs
        Pm = Pd @ tree.edge_mask.T            # (n_pool, E) subtree masses
        Qm = Qd @ tree.edge_mask.T            # (n_q, E)
        out = np.zeros((feats.n_q, ds.n_pool))
        for qi in range(feats.n_q):
            d = np.abs(Pm - Qm[qi]) @ tree.edge_w   # (n_pool,)
            out[qi] = -d                            # higher score = closer
        return out
