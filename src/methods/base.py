"""Shared feature precompute and the common method interface.

Every retrieval method is a `Method` exposing

    scores(ds, feats, seed=0) -> np.ndarray of shape (n_queries, n_pool)

where a higher score means "more likely to be the dupe". Deterministic methods
ignore `seed`; stochastic ones use it. Supervised (learning-to-rank) methods produce
**out-of-fold** scores via `groupkfold_oof` so a query never appears in both train
and test (GroupKFold(5) grouped by query).

All accord vectors live in a single shared vocabulary = the union of every accord
seen in the local products and the global queries. Only the 56 shared accords can
actually create overlap; the rest contribute zeros to cross terms. IDF is computed
on the candidate corpus (the 340 products) and the query is transformed with that
same IDF -- a documented judgement call so the weighting is defined by the pool.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy import sparse

from ..data import Dataset


# --------------------------------------------------------------------------- #
def _l2norm_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def _idf(binary: np.ndarray) -> np.ndarray:
    """sklearn-style smoothed idf over a (n_docs, n_terms) binary matrix."""
    n_docs = binary.shape[0]
    df = binary.sum(axis=0)
    return np.log((1.0 + n_docs) / (1.0 + df)) + 1.0


def _bigrams(accords: List[str]) -> List[Tuple[str, str]]:
    """Unordered accord pairs of a fragrance (sorted tuples, no duplicates)."""
    uniq = sorted(set(accords))
    return [tuple(sorted(c)) for c in combinations(uniq, 2)]


@dataclass
class Features:
    """Precomputed accord representations for products and queries."""
    # vocabularies
    vocab: List[str]
    vidx: Dict[str, int]
    bvocab: List[Tuple[str, str]]
    bidx: Dict[Tuple[str, str], int]

    # unigram binary matrices
    PU: np.ndarray   # (n_pool, V)
    QU: np.ndarray   # (n_q, V)
    # bigram binary matrices (sparse)
    PB: sparse.csr_matrix   # (n_pool, B)
    QB: sparse.csr_matrix   # (n_q, B)

    # idf
    idf_u: np.ndarray
    idf_b: np.ndarray

    # l2-normalised tf-idf
    PUt: np.ndarray
    QUt: np.ndarray
    PBt: np.ndarray
    QBt: np.ndarray

    # P1 combined (unigram+bigram) joint-normalised tf-idf, plus the column split
    PC: np.ndarray
    QC: np.ndarray
    n_uni: int       # first n_uni columns of PC/QC are unigram dims

    # accord sets (for Jaccard / set ops)
    p_sets: List[frozenset]
    q_sets: List[frozenset]

    @property
    def n_pool(self) -> int:
        return self.PU.shape[0]

    @property
    def n_q(self) -> int:
        return self.QU.shape[0]

    # ---- convenient pairwise primitives (vectorised over the whole pool) ---- #
    def shared_counts(self, qi: int) -> np.ndarray:
        """|shared accords| between query qi and every product."""
        return self.PU @ self.QU[qi]

    def jaccard(self, qi: int) -> np.ndarray:
        inter = self.shared_counts(qi)
        sq = self.QU[qi].sum()
        sp = self.PU.sum(axis=1)
        union = sq + sp - inter
        union[union == 0] = 1.0
        return inter / union

    def unigram_cos(self, qi: int) -> np.ndarray:
        return self.PUt @ self.QUt[qi]

    def bigram_cos(self, qi: int) -> np.ndarray:
        return np.asarray(self.PBt @ self.QBt[qi]).ravel()


def build_features(ds: Dataset) -> Features:
    # ---- unigram vocabulary (union of products + queries) ----
    vocab = sorted({a for p in ds.products for a in p.accords}
                   | {a for q in ds.queries for a in q.accords})
    vidx = {a: i for i, a in enumerate(vocab)}
    V = len(vocab)

    PU = np.zeros((ds.n_pool, V))
    for p in ds.products:
        for a in p.accords:
            PU[p.idx, vidx[a]] = 1.0
    QU = np.zeros((len(ds.queries), V))
    for q in ds.queries:
        for a in q.accords:
            QU[q.idx, vidx[a]] = 1.0

    # ---- bigram vocabulary (union) ----
    bset = set()
    for p in ds.products:
        bset.update(_bigrams(p.accords))
    for q in ds.queries:
        bset.update(_bigrams(q.accords))
    bvocab = sorted(bset)
    bidx = {b: i for i, b in enumerate(bvocab)}
    B = len(bvocab)

    def _bmat(items, get_acc):
        rows, cols = [], []
        for i, it in enumerate(items):
            for b in _bigrams(get_acc(it)):
                rows.append(i)
                cols.append(bidx[b])
        data = np.ones(len(rows))
        return sparse.csr_matrix((data, (rows, cols)),
                                 shape=(len(items), B))

    PB = _bmat(ds.products, lambda p: p.accords)
    QB = _bmat(ds.queries, lambda q: q.accords)

    # ---- idf from the candidate corpus (products) ----
    idf_u = _idf(PU)
    idf_b = _idf((PB > 0).toarray()) if B else np.zeros(0)

    PUt = _l2norm_rows(PU * idf_u)
    QUt = _l2norm_rows(QU * idf_u)
    PBd = PB.toarray() * idf_b if B else np.zeros((ds.n_pool, 0))
    QBd = QB.toarray() * idf_b if B else np.zeros((len(ds.queries), 0))
    PBt = _l2norm_rows(PBd)
    QBt = _l2norm_rows(QBd)

    # ---- P1 combined order-2 tf-idf (joint L2 normalisation) ----
    PCraw = np.hstack([PU * idf_u, PBd])
    QCraw = np.hstack([QU * idf_u, QBd])
    PC = _l2norm_rows(PCraw)
    QC = _l2norm_rows(QCraw)

    p_sets = [frozenset(p.accords) for p in ds.products]
    q_sets = [frozenset(q.accords) for q in ds.queries]

    return Features(vocab=vocab, vidx=vidx, bvocab=bvocab, bidx=bidx,
                    PU=PU, QU=QU, PB=PB, QB=QB,
                    idf_u=idf_u, idf_b=idf_b,
                    PUt=PUt, QUt=QUt, PBt=PBt, QBt=QBt,
                    PC=PC, QC=QC, n_uni=V,
                    p_sets=p_sets, q_sets=q_sets)


# --------------------------------------------------------------------------- #
# Method interface
# --------------------------------------------------------------------------- #
class Method:
    name: str = "base"
    tier: str = ""
    stochastic: bool = False

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        """Return (n_q, n_pool) score matrix; higher = more likely the dupe."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Out-of-fold runner for supervised (LTR) methods
# --------------------------------------------------------------------------- #
def label_matrix(ds: Dataset) -> np.ndarray:
    Y = np.zeros((len(ds.queries), ds.n_pool))
    for q in ds.queries:
        for pidx in q.relevant:
            Y[q.idx, pidx] = 1.0
    return Y


def grouped_folds(n_q: int, n_splits: int, seed: int) -> List[np.ndarray]:
    """Assign queries to folds, grouped by query (a query is one indivisible unit).
    The assignment is shuffled by `seed`, so different seeds give different folds --
    this is the only source of seed variance for the LTR methods."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_q)
    return [np.sort(f) for f in np.array_split(order, min(n_splits, n_q))]


def groupkfold_oof(ds: Dataset,
                   feature_fn: Callable[[int], np.ndarray],
                   fit_predict: Callable,
                   seed: int = 0,
                   n_splits: int = 5,
                   sparse_feats: bool = False) -> np.ndarray:
    """Generic GroupKFold(5)-by-query out-of-fold scorer.

    `feature_fn(qi)` returns the (n_pool, d) feature block for query qi (label-free).
    `fit_predict(Xtr, ytr, Xte, seed)` trains and returns scores for Xte.
    A query never appears in both train and test (it is one group).
    """
    n_q, n_pool = len(ds.queries), ds.n_pool
    Y = label_matrix(ds)

    blocks = [feature_fn(qi) for qi in range(n_q)]
    if sparse_feats:
        blocks = [sparse.csr_matrix(b) for b in blocks]

    scores = np.zeros((n_q, n_pool))
    folds = grouped_folds(n_q, n_splits, seed)
    for f, test_qs in enumerate(folds):
        train_qs = np.concatenate([folds[j] for j in range(len(folds)) if j != f])
        if sparse_feats:
            Xtr = sparse.vstack([blocks[qi] for qi in train_qs]).tocsr()
            Xte = sparse.vstack([blocks[qi] for qi in test_qs]).tocsr()
        else:
            Xtr = np.vstack([blocks[qi] for qi in train_qs])
            Xte = np.vstack([blocks[qi] for qi in test_qs])
        ytr = np.concatenate([Y[qi] for qi in train_qs])
        pred = fit_predict(Xtr, ytr, Xte, seed).reshape(len(test_qs), n_pool)
        for k, qi in enumerate(test_qs):
            scores[qi] = pred[k]
    return scores
