"""Order-3 as the main method: significance vs every comparator, and per-order
score decomposition.

Part 1 -- significance. Same protocol as the P1-vs-B2 test: paired Wilcoxon
signed-rank across queries on per-query reciprocal ranks, plus a 95% bootstrap CI
(10,000 resamples). Compares order-3 (max_order=3) against each Tier-1/Tier-2
comparator (B1-B6, A1-A6). Stochastic comparators are scored at seed 0, exactly as in
the existing significance table. Output: order3_vs_comparators.csv.

Part 2 -- decomposition. For the order-3 method, the average share of the top-ranked
product's score contributed by order-1, order-2, and order-3 tokens, over the 209
queries. This replaces the old single "81.3%" (order-2) statistic. Output:
order3_decomposition.csv.

The order-N machinery (order_ablation) and all P1 files are reused unchanged.
"""
from __future__ import annotations

import csv

import numpy as np

from .data import load_dataset, leakage_audit
from .evaluate import bootstrap_delta_mrr, per_query_metrics, wilcoxon_rr
from .methods import ALL_METHODS
from .methods.base import _idf, _l2norm_rows, build_features
from .order_ablation import RESULTS_DIR, score_matrix, tokens_upto

MAIN_ORDER = 3
N_BOOT = 10000


def order_vectors(ds, max_order: int):
    """Return (Qt, Pt, order_of_col) for the order-N method: IDF-weighted, L2-normalized
    token vectors plus the order (subset size) of each vocabulary column."""
    vocab = sorted({t for p in ds.products for t in tokens_upto(p.accords, max_order)}
                   | {t for q in ds.queries for t in tokens_upto(q.accords, max_order)})
    vidx = {t: i for i, t in enumerate(vocab)}
    order_of_col = np.array([len(t) for t in vocab])
    P = np.zeros((ds.n_pool, len(vocab)))
    for p in ds.products:
        for t in tokens_upto(p.accords, max_order):
            P[p.idx, vidx[t]] = 1.0
    Q = np.zeros((len(ds.queries), len(vocab)))
    for q in ds.queries:
        for t in tokens_upto(q.accords, max_order):
            Q[q.idx, vidx[t]] = 1.0
    idf = _idf(P)
    return _l2norm_rows(Q * idf), _l2norm_rows(P * idf), order_of_col


def significance(ds, feats, rr3):
    rows = []
    print(f"{'method':<20}{'dMRR':>9}{'CI low':>10}{'CI high':>10}{'Wilcoxon p':>14}")
    for m in ALL_METHODS:
        if not (m.name.startswith("B") or m.name.startswith("A")):
            continue  # comparators only (skip P1/P2/P3)
        S = m.scores(ds, feats, seed=0)
        if np.all(np.isnan(S)):
            print(f"{m.name:<20}{'SKIPPED (optional dep unavailable)':>43}")
            continue
        rr = per_query_metrics(S, ds).rr
        bs = bootstrap_delta_mrr(rr3, rr, n_boot=N_BOOT)   # delta = order3 - comparator
        p = wilcoxon_rr(rr3, rr)
        pstr = "" if p is None else f"{p:.3e}"
        print(f"{m.name:<20}{bs['delta_mrr']:>9.4f}{bs['ci_low']:>10.4f}"
              f"{bs['ci_high']:>10.4f}{(pstr if pstr else 'nan'):>14}")
        rows.append([m.name, f"{bs['delta_mrr']:.4f}", f"{bs['ci_low']:.4f}",
                     f"{bs['ci_high']:.4f}", pstr])
    return rows


def decomposition(ds, max_order: int):
    Qt, Pt, order_of_col = order_vectors(ds, max_order)
    scores = Qt @ Pt.T
    shares = {o: [] for o in range(1, max_order + 1)}
    for q in ds.queries:
        top = int(np.argmax(scores[q.idx]))
        contrib = Qt[q.idx] * Pt[top]            # per-column contribution to the score
        total = contrib.sum()
        if total <= 1e-12:
            continue
        for o in range(1, max_order + 1):
            shares[o].append(contrib[order_of_col == o].sum() / total)
    return {o: float(np.mean(v)) if v else 0.0 for o, v in shares.items()}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = load_dataset()
    leakage_audit(ds, verbose=False)          # fills text_clean for B4
    feats = build_features(ds)

    rr3 = per_query_metrics(score_matrix(ds, MAIN_ORDER), ds).rr
    print("=" * 64)
    print(f"order-{MAIN_ORDER} vs each comparator (Wilcoxon + 95% bootstrap CI)")
    print(f"MRR(order-{MAIN_ORDER}) = {rr3.mean():.4f}")
    print("=" * 64)
    rows = significance(ds, feats, rr3)
    with open(RESULTS_DIR / "order3_vs_comparators.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "delta_mrr", "ci_low", "ci_high", "wilcoxon_p"])
        w.writerows(rows)

    print("-" * 64)
    dec = decomposition(ds, MAIN_ORDER)
    print(f"order-{MAIN_ORDER} score decomposition (avg share of top-ranked score):")
    for o, s in dec.items():
        print(f"  order-{o}: {s:.1%}")
    with open(RESULTS_DIR / "order3_decomposition.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "avg_share"])
        for o, s in dec.items():
            w.writerow([o, f"{s:.4f}"])

    print("-" * 64)
    print(f"wrote {RESULTS_DIR/'order3_vs_comparators.csv'} and "
          f"{RESULTS_DIR/'order3_decomposition.csv'}")


if __name__ == "__main__":
    main()
