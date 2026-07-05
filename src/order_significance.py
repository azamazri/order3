"""Significance on the order ladder (extends the order-N ablation).

Uses the SAME protocol as the P1-vs-B2 test already in the benchmark: paired Wilcoxon
signed-rank across queries on the per-query reciprocal ranks, plus a 95% bootstrap CI
(10,000 resamples over queries) for delta-MRR. Reuses evaluate.wilcoxon_rr and
evaluate.bootstrap_delta_mrr, and order_ablation.score_matrix for the order-N scores,
so nothing about P1 is re-implemented or touched.

Comparisons: order-2 vs order-3, order-2 vs order-4, order-3 vs order-4
(delta = MRR(first) - MRR(second)).

Also (re)writes diagnostic_sparsity.csv: for k=1..6, the average number of order-k
tokens shared between a query and its correct product, over the 209 queries -- the
evidence for why the MRR curve flattens at orders 4-6.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .data import load_dataset
from .evaluate import bootstrap_delta_mrr, per_query_metrics, wilcoxon_rr
from .order_ablation import RESULTS_DIR, score_matrix, sparsity_by_order

PAIRS = [(2, 3), (2, 4), (3, 4)]
N_BOOT = 10000


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = load_dataset()

    # per-query reciprocal ranks for each order needed
    orders = sorted({n for pair in PAIRS for n in pair})
    rr = {n: per_query_metrics(score_matrix(ds, n), ds).rr for n in orders}
    mrr = {n: float(rr[n].mean()) for n in orders}

    print("=" * 66)
    print("Order-ladder significance (Wilcoxon paired RR + 95% bootstrap CI)")
    print("=" * 66)
    print(f"{'comparison':<20}{'dMRR':>9}{'CI low':>10}{'CI high':>10}{'Wilcoxon p':>15}")
    rows = []
    for a, b in PAIRS:
        bs = bootstrap_delta_mrr(rr[a], rr[b], n_boot=N_BOOT)
        p = wilcoxon_rr(rr[a], rr[b])
        name = f"order-{a} vs order-{b}"
        pstr = "nan" if p is None else f"{p:.3e}"
        print(f"{name:<20}{bs['delta_mrr']:>9.4f}{bs['ci_low']:>10.4f}"
              f"{bs['ci_high']:>10.4f}{pstr:>15}")
        rows.append([name, f"{bs['delta_mrr']:.4f}", f"{bs['ci_low']:.4f}",
                     f"{bs['ci_high']:.4f}", "" if p is None else f"{p:.3e}"])

    with open(RESULTS_DIR / "order_significance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "delta_mrr", "ci_low", "ci_high", "wilcoxon_p"])
        w.writerows(rows)

    # sparsity diagnostic (k=1..6)
    print("-" * 66)
    print("Sparsity: avg shared order-k tokens (query vs correct product)")
    sp = sparsity_by_order(ds)
    for k, avg in sp:
        print(f"k={k} | avg shared tokens {avg:.4f}")
    with open(RESULTS_DIR / "diagnostic_sparsity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "avg_shared_tokens"])
        for k, avg in sp:
            w.writerow([k, f"{avg:.4f}"])

    print("-" * 66)
    print(f"MRR: " + ", ".join(f"order-{n}={mrr[n]:.4f}" for n in orders))
    print(f"wrote {RESULTS_DIR/'order_significance.csv'} and "
          f"{RESULTS_DIR/'diagnostic_sparsity.csv'}")


if __name__ == "__main__":
    main()
