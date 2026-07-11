"""Tahap 8 — tabel utama 15 metode + signifikansi (BH).

- Tuning ulang pembanding pada data bersih (grid modest, seleksi via MRR NON_OP seed 0;
  status yang sama dengan order-N yang N*-nya dipilih di Tahap 6). best_params dibekukan.
- Main table: 15 metode, 5 seed (stokastik), pada ALL dan NON_OP. multi=best, tie=pesimistis.
- order-N pakai N*=2 (Tahap 6) -> metode 'P1_order2' (= order-2) di registry.
- Signifikansi: proposed (order-2) vs 14 pembanding, BH FDR 0.05, terpisah per eval set.
- Train/test gap untuk metode supervised (in-sample vs OOF).

Output: 04_main_table.csv, 04_gap.csv, 05_significance.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_dataset
from ..methods.base import build_features, groupkfold_oof, label_matrix
from ..methods import (B1Jaccard, B2TfidfCosine, B3BM25, B4SBert, B5Word2Vec,
                       B6Node2Vec, A1Wheel, A2PpmiSvd, A3Signature, A4BigramSalience,
                       A5Bilinear, A6Gbm, P1Order2, P2Fusion, P3HubIdf)
from . import protocol as P
from .audit_impl import PLU

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v3"
SEEDS = [0, 1, 2, 3, 4]
PROPOSED = "P1_order2"   # = order-2 (N*=2 from Tahap 6)

# tunable comparators: name -> (constructor, list of param dicts)
TUNE = {
    "B3_bm25": (lambda **k: B3BM25(**k),
                [{"k1": a, "b": b} for a in (1.2, 1.5, 2.0) for b in (0.5, 0.75, 1.0)]),
    "A2_ppmi_svd": (lambda **k: A2PpmiSvd(**k), [{"dim": d} for d in (25, 50, 75)]),
    "A3_signature": (lambda **k: A3Signature(**k), [{"C": c} for c in (0.5, 1.0, 2.0)]),
    "A4_bigram_salience": (lambda **k: A4BigramSalience(**k), [{"C": c} for c in (0.5, 1.0, 2.0)]),
    "A5_bilinear": (lambda **k: A5Bilinear(**k),
                    [{"rank": r, "l2": l} for r in (4, 8) for l in (1e-3, 1e-2)]),
    "A6_gbm_fusion": (lambda **k: A6Gbm(**k),
                      [{"n_estimators": n, "max_depth": d} for n in (100, 200) for d in (2, 3)]),
    "P2_fusion": (lambda **k: P2Fusion(**k), [{"C": c} for c in (0.5, 1.0, 2.0)]),
    "B5_word2vec": (lambda **k: B5Word2Vec(**k),
                    [{"dim": d, "window": w} for d in (32, 64) for w in (3, 5)]),
    "B6_node2vec": (lambda **k: B6Node2Vec(**k),
                    [{"dim": d, "window": w} for d in (32, 64) for w in (3, 5)]),
}
# defaults for non-tuned methods
DEFAULT_CTOR = {
    "B1_jaccard": B1Jaccard, "B2_tfidf_cos": B2TfidfCosine, "B4_sbert": B4SBert,
    "A1_wheel_treeW": lambda: A1Wheel(mode="map"), "P1_order2": P1Order2, "P3_hubidf": P3HubIdf,
}
ORDER = ["B1_jaccard", "B2_tfidf_cos", "B3_bm25", "B4_sbert", "B5_word2vec", "B6_node2vec",
         "A1_wheel_treeW", "A2_ppmi_svd", "A3_signature", "A4_bigram_salience",
         "A5_bilinear", "A6_gbm_fusion", "P1_order2", "P2_fusion", "P3_hubidf"]
SUPERVISED = {"A3_signature", "A4_bigram_salience", "A5_bilinear", "A6_gbm_fusion", "P2_fusion"}


def tune(ds, feats, nonop_mask):
    """Select best config per tunable method by MRR on NON_OP at seed 0 (label-free
    for unsupervised; OOF for supervised). Returns {name: (best_params, method)}."""
    chosen = {}
    for name, (ctor, grid) in TUNE.items():
        best_p, best_mrr, best_m = None, -1.0, None
        for params in grid:
            m = ctor(**params)
            r = P.run_method(m, ds, feats, seeds=[0], tie="pessimistic",
                             multi="best", eval_idx=nonop_mask)
            if r.get("skipped"):
                continue
            if r["MRR"] > best_mrr:
                best_mrr, best_p, best_m = r["MRR"], params, m
        chosen[name] = (best_p, best_m if best_m is not None else ctor(**grid[0]))
        print(f"  tuned {name}: {best_p} (NON_OP MRR seed0={best_mrr:.4f})")
    return chosen


def in_sample_mrr(name, method, ds, feats, seed=0):
    """MRR when trained AND predicted on all queries (in-sample) — for the OOF gap."""
    from ..methods import a3_signature, a4_bigram_salience, a6_gbm, p2_fusion
    Y = label_matrix(ds)
    nq, npool = len(ds.queries), ds.n_pool
    if name in ("A3_signature", "P2_fusion", "A6_gbm_fusion", "A4_bigram_salience"):
        if name == "A3_signature":
            feat_fn = lambda qi: a3_signature._features(feats, qi)
            fp = p2_fusion._make_logreg_fit_predict(method.C); sparse_f = False
        elif name == "P2_fusion":
            feat_fn = lambda qi: p2_fusion._features(feats, qi)
            fp = p2_fusion._make_logreg_fit_predict(method.C); sparse_f = False
        elif name == "A4_bigram_salience":
            feat_fn = lambda qi: a4_bigram_salience._features(feats, qi)
            fp = a4_bigram_salience._make_l2logreg_fit_predict(method.C); sparse_f = True
        else:
            feat_fn = lambda qi: a6_gbm._features(feats, qi)
            fp = a6_gbm._make_gbm_fit_predict(method.n_estimators, method.max_depth,
                                              method.learning_rate); sparse_f = False
        from scipy import sparse as sp
        blocks = [feat_fn(qi) for qi in range(nq)]
        if sparse_f:
            X = sp.vstack([sp.csr_matrix(b) for b in blocks]).tocsr()
        else:
            X = np.vstack(blocks)
        y = np.concatenate([Y[qi] for qi in range(nq)])
        pred = fp(X, y, X, seed).reshape(nq, npool)
        return P.eval_metrics(pred, ds).mrr
    if name == "A5_bilinear":
        from ..methods.a5_bilinear import _train
        Q, Pp = feats.QUt, feats.PUt
        d, L = _train(Q, Pp, Y, method.rank, method.lr, method.l2, method.iters, seed)
        QL = Q @ L; PL = Pp @ L
        pred = (Q * d) @ Pp.T + QL @ PL.T
        return P.eval_metrics(pred, ds).mrr
    return np.nan


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    feats = build_features(ds)
    strata = pd.read_csv(OUT_DIR / "02_strata.csv")
    nonop_q = set(strata.loc[strata["is_NON_OP"] == 1, "q_idx"])
    nonop_mask = np.array([q.idx in nonop_q for q in ds.queries])

    print("Tuning comparators (NON_OP, seed 0)...")
    chosen = tune(ds, feats, nonop_mask)

    # build final method instances + best_params label
    methods, bestparams = {}, {}
    for name in ORDER:
        if name in chosen:
            bp, m = chosen[name]
            methods[name] = m
            bestparams[name] = str(bp)
        else:
            methods[name] = DEFAULT_CTOR[name]()
            bestparams[name] = "N=2 (nested-CV, Tahap 6)" if name == "P1_order2" else "none"

    # ---- run main table on ALL and NON_OP ----
    rr_store = {}   # (name, eval) -> rr_units
    rows = []
    for name in ORDER:
        m = methods[name]
        for label, mask in (("ALL", None), ("NON_OP", nonop_mask)):
            r = P.run_method(m, ds, feats, seeds=SEEDS, tie="pessimistic",
                             multi="best", eval_idx=mask)
            Pf, Lf, Uf, _ = PLU[name]
            if r.get("skipped"):
                rows.append([name, Pf, Lf, Uf, bestparams[name], label, "", "", "", "", "", True])
                continue
            rr_store[(name, label)] = r["rr_units"]
            rows.append([name, Pf, Lf, Uf, bestparams[name], label, r["n_units"],
                         f"{r['MRR']:.4f}", f"{r['MRR_std']:.4f}", f"{r['Hits@1']:.4f}",
                         f"{r['Hits@3']:.4f}", f"{r['Hits@10']:.4f}", False])
        print(f"  ran {name}")

    with open(OUT_DIR / "04_main_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "P", "L", "U", "best_params", "eval_set", "n_queries",
                    "MRR", "MRR_std", "Hits@1", "Hits@3", "Hits@10", "skipped"])
        # reorder columns to spec: MRR after n_queries already; keep
        for row in rows:
            # row currently: name,P,L,U,bp,eval,n,MRR,MRR_std,H1,H3,H10,skipped
            w.writerow(row)

    # ---- significance: proposed vs 14, BH per eval set ----
    sig_rows = []
    for label in ("ALL", "NON_OP"):
        if (PROPOSED, label) not in rr_store:
            continue
        ref = rr_store[(PROPOSED, label)]
        comps = []
        praw_list = []
        for name in ORDER:
            if name == PROPOSED or (name, label) not in rr_store:
                continue
            b = rr_store[(name, label)]
            p_raw, nnz = P.wilcoxon_test(ref, b)
            bs = P.bootstrap_delta(ref, b)
            comps.append((name, bs, nnz, p_raw))
            praw_list.append(p_raw if p_raw is not None else 1.0)
        p_bh = P.benjamini_hochberg(praw_list)
        for (name, bs, nnz, p_raw), padj in zip(comps, p_bh):
            vd = P.verdict(bs["delta_mrr"], bs["ci_low"], bs["ci_high"], padj)
            sig_rows.append([f"{PROPOSED}_vs_{name}", label, f"{bs['delta_mrr']:.4f}",
                             f"{bs['ci_low']:.4f}", f"{bs['ci_high']:.4f}",
                             f"{p_raw:.3e}" if p_raw is not None else "", f"{padj:.3e}",
                             nnz, vd])
    with open(OUT_DIR / "05_significance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "eval_set", "delta_MRR", "ci_low", "ci_high",
                    "wilcoxon_p_raw", "p_adj_BH", "n_nonzero", "verdict"])
        w.writerows(sig_rows)

    # ---- train/test gap for supervised ----
    gap_rows = []
    main_df = pd.read_csv(OUT_DIR / "04_main_table.csv")
    for name in SUPERVISED:
        oof = main_df[(main_df.method == name) & (main_df.eval_set == "ALL")]["MRR"]
        oof_mrr = float(oof.iloc[0]) if len(oof) else np.nan
        ins = in_sample_mrr(name, methods[name], ds, feats, seed=0)
        gap_rows.append([name, f"{ins:.4f}", f"{oof_mrr:.4f}", f"{ins - oof_mrr:.4f}"])
        print(f"  gap {name}: in-sample={ins:.4f} OOF={oof_mrr:.4f}")
    with open(OUT_DIR / "04_gap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "MRR_in_sample", "MRR_oof", "gap"])
        w.writerows(gap_rows)

    print("\nMain table done. Proposed = order-2 (P1_order2).")
    for label in ("ALL", "NON_OP"):
        sub = main_df[main_df.eval_set == label].copy()
        sub = sub[sub.skipped == False].sort_values("MRR", ascending=False)
        print(f"\n{label} top by MRR:")
        for _, r in sub.head(6).iterrows():
            print(f"  {r['method']:<20} MRR={r['MRR']:.4f} H@1={r['Hits@1']:.4f} H@10={r['Hits@10']:.4f}")


if __name__ == "__main__":
    main()
