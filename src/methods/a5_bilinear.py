"""A5 -- Low-rank bilinear cross-accord affinity, metric-learned.

Score(q, p) = q^T (diag(d) + L L^T) p, with q, p the L2-normalised unigram TF-IDF
vectors over the shared accord space. Parameters d (V,) and L (V x r) are learned by
minimising a class-balanced logistic loss with full-batch gradient descent (Adam-free,
plain GD with L2 reg). Trained out-of-fold (GroupKFold-5 by query). The L L^T term is a
learned low-rank cross-accord affinity."""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Features, Method, grouped_folds


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _train(Q, P, Y, rank, lr, l2, iters, seed):
    """Q:(nq,V) P:(npool,V) Y:(nq,npool) labels. Returns (d, L)."""
    rng = np.random.default_rng(seed)
    V = Q.shape[1]
    d = np.zeros(V)
    L = 0.01 * rng.standard_normal((V, rank))

    pos = Y.sum()
    neg = Y.size - pos
    w_pos = (Y.size / (2 * pos)) if pos else 1.0
    w_neg = (Y.size / (2 * neg)) if neg else 1.0
    W = np.where(Y > 0, w_pos, w_neg)
    Wsum = W.sum()

    for _ in range(iters):
        QL = Q @ L                      # (nq, r)
        PL = P @ L                      # (npool, r)
        diag_term = (Q * d) @ P.T       # (nq, npool)
        S = diag_term + QL @ PL.T       # scores
        G = W * (_sigmoid(S) - Y) / Wsum   # loss grad wrt S

        # grad wrt d_k = sum_ij G_ij * Q_ik * P_jk
        grad_d = np.einsum("ij,ik,jk->k", G, Q, P)
        # grad wrt L: Q^T G P L + P^T G^T Q L
        grad_L = Q.T @ (G @ PL) + P.T @ (G.T @ QL)

        d -= lr * (grad_d + l2 * d)
        L -= lr * (grad_L + l2 * L)
    return d, L


class A5Bilinear(Method):
    name = "A5_bilinear"
    tier = "T2 structure"
    stochastic = True

    def __init__(self, rank: int = 8, lr: float = 0.5, l2: float = 1e-3, iters: int = 300):
        self.rank, self.lr, self.l2, self.iters = rank, lr, l2, iters

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        Q, P = feats.QUt, feats.PUt          # l2-normalised unigram tf-idf
        Y = np.zeros((feats.n_q, ds.n_pool))
        for q in ds.queries:
            for pidx in q.relevant:
                Y[q.idx, pidx] = 1.0

        out = np.zeros((feats.n_q, ds.n_pool))
        for f, test_qs in enumerate(grouped_folds(feats.n_q, 5, seed)):
            train_qs = np.array([qi for qi in range(feats.n_q) if qi not in set(test_qs)])
            d, L = _train(Q[train_qs], P, Y[train_qs], self.rank,
                          self.lr, self.l2, self.iters, seed)
            QL = Q[test_qs] @ L
            PL = P @ L
            out[test_qs] = (Q[test_qs] * d) @ P.T + QL @ PL.T
        return out
