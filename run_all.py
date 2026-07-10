"""Run the full interaction-free dupe-retrieval benchmark and emit the results table.

Usage:
    python run_all.py                 # all methods (downloads S-BERT, trains embeddings)
    python run_all.py --fast          # skip B4/B5/B6 (no downloads, no embedding training)
    python run_all.py --seeds 5       # number of seeds for stochastic methods (default 5)

Outputs (in ./results/):
    results.csv         per-method MRR / Hits@1 / Hits@3 (mean +/- std over seeds)
    significance.csv    Wilcoxon p + bootstrap delta-MRR CI vs each baseline (P1 & P2)
and prints both tables plus the P1 order-1/order-2 score decomposition.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.data import load_dataset
from src.evaluate import (bootstrap_delta_mrr,
                          evaluate_method, per_query_metrics, wilcoxon_rr)
from src.methods import ALL_METHODS, FAST_METHODS
from src.methods.base import build_features
from src.methods.p1_order2 import P1Order2

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip B4/B5/B6")
    ap.add_argument("--seeds", type=int, default=5, help="seeds for stochastic methods")
    ap.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- load ---
    print("=" * 78)
    print("Interaction-free fragrance dupe-retrieval benchmark")
    print("=" * 78)
    ds = load_dataset()
    print(f"pool (products)   : {ds.n_pool}")
    print(f"labeled queries   : {len(ds.queries)}")
    print(f"shared accords    : {len(ds.shared_accords)}")
    print("-" * 78)

    feats = build_features(ds)

    methods = FAST_METHODS if args.fast else ALL_METHODS

    # -------------------------------------------------------------- evaluate -
    reports = []
    for m in methods:
        print(f"running {m.name} ...", flush=True)
        reports.append(evaluate_method(m, ds, feats, seeds))

    # ------------------------------------------------------ results table ----
    print("\n" + "=" * 78)
    print(f"{'method':<22}{'tier':<14}{'MRR':>10}{'Hits@1':>12}{'Hits@3':>12}")
    print("-" * 78)
    with open(RESULTS_DIR / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "tier", "stochastic", "MRR", "MRR_std",
                    "Hits@1", "Hits@1_std", "Hits@3", "Hits@3_std", "skipped"])
        for r in reports:
            if r.skipped:
                print(f"{r.name:<22}{r.tier:<14}{'SKIPPED (optional dep unavailable)':>34}")
                w.writerow([r.name, r.tier, r.stochastic, "", "", "", "", "", "", True])
                continue
            tag = "+/-" if r.stochastic else "   "
            print(f"{r.name:<22}{r.tier:<14}"
                  f"{r.mrr:>10.3f}{r.h1:>12.3f}{r.h3:>12.3f}"
                  + (f"   (std {r.mrr_std:.3f})" if r.stochastic else ""))
            w.writerow([r.name, r.tier, r.stochastic,
                        f"{r.mrr:.4f}", f"{r.mrr_std:.4f}",
                        f"{r.h1:.4f}", f"{r.h1_std:.4f}",
                        f"{r.h3:.4f}", f"{r.h3_std:.4f}", False])
    print("=" * 78)

    # ----------------------------------------- significance vs proposed ------
    by_name = {r.name: r for r in reports if not r.skipped and r.seed0 is not None}
    sig_rows = []
    for proposed in ("P1_order2", "P2_fusion"):
        if proposed not in by_name:
            continue
        ref = by_name[proposed].seed0
        print(f"\nSignificance: {proposed} vs each baseline "
              f"(Wilcoxon paired RR; bootstrap 95% CI delta-MRR)")
        print("-" * 78)
        print(f"{'baseline':<22}{'dMRR':>9}{'CI low':>10}{'CI high':>10}{'Wilcoxon p':>14}")
        for r in reports:
            if r.skipped or r.seed0 is None or r.name == proposed:
                continue
            p = wilcoxon_rr(ref.rr, r.seed0.rr)
            bs = bootstrap_delta_mrr(ref.rr, r.seed0.rr, n_boot=args.n_boot)
            print(f"{r.name:<22}{bs['delta_mrr']:>9.3f}{bs['ci_low']:>10.3f}"
                  f"{bs['ci_high']:>10.3f}{(p if p is not None else float('nan')):>14.2e}")
            sig_rows.append([proposed, r.name, f"{bs['delta_mrr']:.4f}",
                             f"{bs['ci_low']:.4f}", f"{bs['ci_high']:.4f}",
                             ("" if p is None else f"{p:.3e}")])
    with open(RESULTS_DIR / "significance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["proposed", "baseline", "delta_mrr", "ci_low", "ci_high", "wilcoxon_p"])
        w.writerows(sig_rows)

    # ----------------------------------------- one-line B4 verdict -----------
    if "P1_order2" in by_name and "B4_sbert" in by_name:
        ref, b4 = by_name["P1_order2"].seed0, by_name["B4_sbert"].seed0
        d = ref.rr.mean() - b4.rr.mean()
        p = wilcoxon_rr(ref.rr, b4.rr)
        better = "P1 beats B4" if d > 0 else "B4 beats P1"
        print(f"\nVERDICT: {better} -- delta-MRR(P1-B4) = {d:+.3f} "
              f"(Wilcoxon p = {p:.2e}).")

    # ----------------------------------------- P1 order decomposition --------
    p1 = P1Order2()
    o1, o2 = p1.decompose(feats)
    full = p1.scores(ds, feats)
    # fraction of the winning score contributed by order-2, averaged over queries
    contrib = []
    for q in ds.queries:
        best = int(np.argmax(full[q.idx]))
        tot = full[q.idx, best]
        if tot > 1e-9:
            contrib.append(o2[q.idx, best] / tot)
    print("\n" + "-" * 78)
    print(f"P1 decomposition: order-2 (co-occurrence) contributes on average "
          f"{np.mean(contrib):.1%} of the top-ranked score "
          f"(order-1 marginal: {1 - np.mean(contrib):.1%}).")
    print("-" * 78)
    print(f"\nWrote {RESULTS_DIR/'results.csv'} and {RESULTS_DIR/'significance.csv'}")


if __name__ == "__main__":
    main()
