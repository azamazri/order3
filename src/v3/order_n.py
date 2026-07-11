"""Order-N co-occurrence TF-IDF (ditulis ulang untuk v3; tidak mewarisi kode lama).

Spesifikasi (§6.1):
    tokens(A) = semua subset A berukuran 1..N
    x_t       = 1 jika t ⊆ A, else 0
    idf_t     = log( (1 + n_pool) / (1 + df_t) ) + 1      # df_t dari 340 PRODUK saja
    v(A)      = L2_normalize( x ⊙ idf )
    score(q,p)= v(A(q)) · v(A(p))

N=1 harus mereproduksi B2_tfidf_cos persis (assert < 1e-9), else Gate G5.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

from ..data import Dataset
from ..methods.base import Method


def _subsets(accords, N):
    uniq = sorted(set(accords))
    for k in range(1, N + 1):
        if k > len(uniq):
            break
        for c in combinations(uniq, k):
            yield c


def build_order_features(ds: Dataset, N: int):
    """Return (Pmat, Qmat, meta). Pmat/Qmat: l2-normalised idf tf matrices (sparse),
    idf from the 340 products only. meta has vocab, idf, and per-order column ranges."""
    # ---- df over products; build vocab from products+queries ----
    df: Dict[Tuple[str, ...], int] = {}
    prod_subsets: List[List[Tuple[str, ...]]] = []
    for p in ds.products:
        subs = list(_subsets(p.accords, N))
        prod_subsets.append(subs)
        for t in subs:
            df[t] = df.get(t, 0) + 1
    query_subsets: List[List[Tuple[str, ...]]] = [list(_subsets(q.accords, N))
                                                  for q in ds.queries]
    # vocab = union (products already in df; add query-only)
    vocab_set = set(df.keys())
    for subs in query_subsets:
        vocab_set.update(subs)
    vocab = sorted(vocab_set, key=lambda t: (len(t), t))
    vidx = {t: i for i, t in enumerate(vocab)}
    order_of = np.array([len(t) for t in vocab])

    n_pool = ds.n_pool
    idf = np.array([np.log((1.0 + n_pool) / (1.0 + df.get(t, 0))) + 1.0 for t in vocab])

    def _mat(list_of_subs, n_rows):
        rows, cols = [], []
        for i, subs in enumerate(list_of_subs):
            for t in subs:
                rows.append(i)
                cols.append(vidx[t])
        data = idf[cols]
        M = sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, len(vocab)))
        # L2 normalise rows
        norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        D = sparse.diags(1.0 / norms)
        return (D @ M).tocsr()

    Pmat = _mat(prod_subsets, n_pool)
    Qmat = _mat(query_subsets, len(ds.queries))
    meta = {"vocab": vocab, "idf": idf, "order_of": order_of,
            "prod_subsets": prod_subsets, "query_subsets": query_subsets, "df": df}
    return Pmat, Qmat, meta


class OrderN(Method):
    tier = "T3 proposed"
    stochastic = False

    def __init__(self, N: int):
        self.N = N
        self.name = f"order{N}"

    def scores(self, ds: Dataset, feats=None, seed: int = 0) -> np.ndarray:
        Pmat, Qmat, _ = build_order_features(ds, self.N)
        return np.asarray((Qmat @ Pmat.T).todense())

    def scores_and_decompose(self, ds: Dataset):
        """Return (full_scores, per_order_scores dict k->matrix). Sum of per-order == full."""
        Pmat, Qmat, meta = build_order_features(ds, self.N)
        full = np.asarray((Qmat @ Pmat.T).todense())
        order_of = meta["order_of"]
        per = {}
        for k in range(1, self.N + 1):
            colmask = np.where(order_of == k)[0]
            if len(colmask) == 0:
                per[k] = np.zeros_like(full)
                continue
            Pk = Pmat[:, colmask]
            Qk = Qmat[:, colmask]
            per[k] = np.asarray((Qk @ Pk.T).todense())
        return full, per, meta
