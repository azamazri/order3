"""LANGKAH 2.2 -- A3 feature ablation (PEDOMAN Bagian 4/A3, 8.2).

A3's 7 features (a3_signature._features order):
  0 inter | 1 n_shared_b | 2 w_shared_b | 3 max_rare | 4 cov_q | 5 cov_p | 6 jaccard
Features 1,2,3 (n_shared_b, w_shared_b, max_rare) ARE the order-2 co-occurrence
representation of the proposed method. Three variants, nested CV (C in {0.01,0.1,1,10}):

  A3-full  : all 7 features
  A3-noco  : drop 1,2,3  -> [inter, cov_q, cov_p, jaccard]  (order-1 only)
  A3-conly : only 1,2,3  (co-occurrence only)

If A3-noco falls to the B2 level (~0.454), that is direct evidence the working signal is
co-occurrence, not the logistic regression. Reuses the Fase-2 nested-CV machinery.
Output: results/audit/a3_ablation.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

from .data import load_dataset
from .evaluate import per_query_metrics
from .methods.base import build_features
from .methods.a3_signature import _features as a3_features
from .methods.b2_tfidf import B2TfidfCosine
from .order_ablation import score_matrix
from .phase2_tuning import nested_supervised, _fp_a3, _grid, SEEDS

RESULTS = Path(__file__).resolve().parents[1] / "results" / "audit"
GRID = _grid({"C": [0.01, 0.1, 1, 10]})


def feat_full(feats, qi):
    return a3_features(feats, qi)


def feat_noco(feats, qi):
    return a3_features(feats, qi)[:, [0, 4, 5, 6]]


def feat_conly(feats, qi):
    return a3_features(feats, qi)[:, [1, 2, 3]]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(); feats = build_features(ds)

    rows = []
    for name, fn in [("A3-full", feat_full), ("A3-noco", feat_noco), ("A3-conly", feat_conly)]:
        res, _ = nested_supervised(ds, feats, fn, _fp_a3, GRID, SEEDS)
        rows.append([name, res["best_params"], f"{res['mrr']:.4f}", f"{res['mrr_std']:.4f}",
                     f"{res['hits1']:.4f}", f"{res['hits3']:.4f}"])
        print(f"{name:<9} mrr {res['mrr']:.4f} (±{res['mrr_std']:.3f}) "
              f"h1 {res['hits1']:.3f} h3 {res['hits3']:.3f} best={res['best_params']}", flush=True)

    # references
    b2 = per_query_metrics(B2TfidfCosine().scores(ds, feats), ds)
    o3 = per_query_metrics(score_matrix(ds, 3), ds)
    rows.append(["B2_tfidf_cos (ref)", "none", f"{b2.mrr:.4f}", "0.0000", f"{b2.hits1:.4f}", f"{b2.hits3:.4f}"])
    rows.append(["order3 (ref)", "parameter-free", f"{o3.mrr:.4f}", "0.0000", f"{o3.hits1:.4f}", f"{o3.hits3:.4f}"])
    print(f"ref B2 {b2.mrr:.4f} | order3 {o3.mrr:.4f}")

    with open(RESULTS / "a3_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "best_params", "mrr", "mrr_std", "hits1", "hits3"])
        w.writerows(rows)
    print(f"wrote {RESULTS}/a3_ablation.csv")


if __name__ == "__main__":
    main()
