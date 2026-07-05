"""P3 -- P1 with IDF hub-discriminative weighting (ablation).

Hub accords (e.g. amber, woody) co-occur with very many others and carry little
discriminative signal. On top of standard IDF we multiply each accord weight by a
hub penalty  hub(a) = 1 / (1 + log(1 + deg(a))), where deg(a) is the number of
distinct accords a co-occurs with in the corpus. A bigram weight is penalised by
the product of its two endpoints' hub factors. Then we rebuild the joint order-2
TF-IDF exactly as P1. This is an ablation to test whether hub down-weighting helps.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..data import Dataset
from .base import Features, Method, _l2norm_rows


class P3HubIdf(Method):
    name = "P3_hubidf"
    tier = "T3 proposed"

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        # degree of each accord in the co-occurrence graph
        deg = {a: set() for a in feats.vocab}
        for frag in [p.accords for p in ds.products] + [q.accords for q in ds.queries]:
            for a, b in combinations(sorted(set(frag)), 2):
                deg[a].add(b)
                deg[b].add(a)
        hub_u = np.array([1.0 / (1.0 + np.log1p(len(deg[a]))) for a in feats.vocab])
        hub_b = np.array([hub_u[feats.vidx[a]] * hub_u[feats.vidx[b]]
                          for (a, b) in feats.bvocab]) if feats.bvocab else np.zeros(0)

        wu = feats.idf_u * hub_u
        wb = feats.idf_b * hub_b

        PBd = feats.PB.toarray() * wb if feats.bvocab else np.zeros((ds.n_pool, 0))
        QBd = feats.QB.toarray() * wb if feats.bvocab else np.zeros((feats.n_q, 0))
        PC = _l2norm_rows(np.hstack([feats.PU * wu, PBd]))
        QC = _l2norm_rows(np.hstack([feats.QU * wu, QBd]))
        return QC @ PC.T
