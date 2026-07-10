"""Order-N ablation: why order-2, not 3/4/5/6?

Generalizes the P1 token construction to `max_order`: the token set of a fragrance is
the union of ALL accord subsets of size k for k = 1..max_order (k=1 = single accords,
k=2 = unordered pairs, k=3 = triples, ...). Everything else is IDENTICAL to P1:
the same IDF formula (base._idf), the same L2 normalization (base._l2norm_rows), the
same full-pool retrieval harness and pool (evaluate.per_query_metrics with expected-rank
tie breaking), the same 209 queries and 340-product pool.

Sanity: max_order=1 must reproduce B2 (MRR ~0.454); max_order=2 must reproduce P1
(MRR ~0.496). If not, the generalization is wrong -> the script stops.

Also emits a sparsity diagnostic: for each k, the average number of order-k tokens
SHARED between a query and its correct (gold) product, across the 209 queries.
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .data import load_dataset
from .evaluate import per_query_metrics
from .methods.base import _idf, _l2norm_rows

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MAX_ORDERS = [1, 2, 3, 4, 5, 6]

# Reference numbers from results.csv for the two validated points.
B2_MRR = 0.4540      # max_order=1
P1_MRR = 0.4963      # max_order=2
TOL = 1e-3


def tokens_upto(accords, max_order: int) -> List[Tuple[str, ...]]:
    """All accord subsets of size 1..max_order (sorted tuples, unordered)."""
    base = sorted(set(accords))
    out: List[Tuple[str, ...]] = []
    for k in range(1, max_order + 1):
        out.extend(combinations(base, k))
    return out


def tokens_of_order(accords, k: int) -> List[Tuple[str, ...]]:
    return list(combinations(sorted(set(accords)), k))


def score_matrix(ds, max_order: int) -> np.ndarray:
    """Reproduce the P1 pipeline for a given max_order and return (n_q, n_pool) scores."""
    # vocabulary = union of tokens over products + queries
    vocab = sorted({t for p in ds.products for t in tokens_upto(p.accords, max_order)}
                   | {t for q in ds.queries for t in tokens_upto(q.accords, max_order)})
    vidx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    P = np.zeros((ds.n_pool, V))
    for p in ds.products:
        for t in tokens_upto(p.accords, max_order):
            P[p.idx, vidx[t]] = 1.0
    Q = np.zeros((len(ds.queries), V))
    for q in ds.queries:
        for t in tokens_upto(q.accords, max_order):
            Q[q.idx, vidx[t]] = 1.0

    idf = _idf(P)                         # IDF from the product corpus (same as P1)
    Pt = _l2norm_rows(P * idf)            # IDF-weighted, jointly L2-normalized
    Qt = _l2norm_rows(Q * idf)
    return Qt @ Pt.T


def sparsity_by_order(ds) -> List[Tuple[int, float]]:
    """Average number of shared order-k tokens between a query and its gold product(s)."""
    rows = []
    for k in range(1, max(MAX_ORDERS) + 1):
        shared = []
        for q in ds.queries:
            qk = set(tokens_of_order(q.accords, k))
            for pidx in q.relevant:
                pk = set(tokens_of_order(ds.products[pidx].accords, k))
                shared.append(len(qk & pk))
        rows.append((k, float(np.mean(shared)) if shared else 0.0))
    return rows


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = load_dataset()

    print("=" * 60)
    print("Order-N ablation (deterministic, one run per N)")
    print("=" * 60)
    results = []
    metrics_by_n = {}
    for N in MAX_ORDERS:
        r = per_query_metrics(score_matrix(ds, N), ds)
        metrics_by_n[N] = r
        results.append((N, r.mrr, r.hits1, r.hits3))
        print(f"N={N} | MRR {r.mrr:.4f} | Hits@1 {r.hits1:.4f} | Hits@3 {r.hits3:.4f}")

    # ---- mandatory validation ----
    m1, m2 = metrics_by_n[1].mrr, metrics_by_n[2].mrr
    ok1, ok2 = abs(m1 - B2_MRR) <= TOL, abs(m2 - P1_MRR) <= TOL
    print("-" * 60)
    print(f"validate N=1 == B2 : {m1:.4f} vs {B2_MRR}  -> {'OK' if ok1 else 'MISMATCH'}")
    print(f"validate N=2 == P1 : {m2:.4f} vs {P1_MRR}  -> {'OK' if ok2 else 'MISMATCH'}")
    if not (ok1 and ok2):
        raise SystemExit("VALIDATION FAILED: generalization does not reproduce B2/P1. "
                         "Stopping without writing CSVs.")

    # ---- sparsity diagnostic ----
    print("-" * 60)
    print("Sparsity: avg shared order-k tokens (query vs correct product)")
    sp = sparsity_by_order(ds)
    for k, avg in sp:
        print(f"k={k} | avg shared tokens {avg:.3f}")

    # ---- write CSVs ----
    with open(RESULTS_DIR / "order_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["max_order", "MRR", "Hits@1", "Hits@3"])
        for N, mrr, h1, h3 in results:
            w.writerow([N, f"{mrr:.4f}", f"{h1:.4f}", f"{h3:.4f}"])
    with open(RESULTS_DIR / "diagnostic_sparsity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "avg_shared_tokens"])
        for k, avg in sp:
            w.writerow([k, f"{avg:.4f}"])
    print("-" * 60)
    print(f"wrote {RESULTS_DIR/'order_ablation.csv'} and {RESULTS_DIR/'diagnostic_sparsity.csv'}")


if __name__ == "__main__":
    main()
