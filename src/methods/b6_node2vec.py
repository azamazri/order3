"""B6 -- node2vec on the accord co-occurrence graph.

Graph: nodes = accords, edge weight = number of fragrances in which two accords
co-occur (computed over products + queries). node2vec random walks -> Skip-gram
node embeddings. A fragrance vector = mean of its accord node vectors; cosine.
Stochastic -> evaluated over several seeds."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..data import Dataset
from .base import Features, Method, _l2norm_rows


class B6Node2Vec(Method):
    name = "B6_node2vec"
    tier = "T1 baseline"
    stochastic = True

    def __init__(self, dim: int = 64, walk_length: int = 20, num_walks: int = 50,
                 window: int = 5):
        self.dim, self.walk_length = dim, walk_length
        self.num_walks, self.window = num_walks, window

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        import networkx as nx
        from node2vec import Node2Vec

        # co-occurrence graph
        G = nx.Graph()
        G.add_nodes_from(feats.vocab)
        cooc = {}
        for frag in [p.accords for p in ds.products] + [q.accords for q in ds.queries]:
            for a, b in combinations(sorted(set(frag)), 2):
                cooc[(a, b)] = cooc.get((a, b), 0) + 1
        for (a, b), w in cooc.items():
            G.add_edge(a, b, weight=w)

        n2v = Node2Vec(G, dimensions=self.dim, walk_length=self.walk_length,
                       num_walks=self.num_walks, weight_key="weight",
                       workers=1, seed=seed, quiet=True)
        model = n2v.fit(window=self.window, min_count=1, sg=1, workers=1, seed=seed)

        def vec(accords):
            v = [model.wv[a] for a in accords if a in model.wv]
            return np.mean(v, axis=0) if v else np.zeros(self.dim)

        P = _l2norm_rows(np.vstack([vec(p.accords) for p in ds.products]))
        Q = _l2norm_rows(np.vstack([vec(q.accords) for q in ds.queries]))
        return Q @ P.T
