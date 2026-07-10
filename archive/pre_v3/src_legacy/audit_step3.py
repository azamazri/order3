"""LANGKAH 3 -- final re-run + significance (PEDOMAN Bagian 3, 5, 8.2).

One identical harness for every method, at its FINAL configuration:

  * comparators: evaluated at the params selected by Phase-2 nested CV
    (exp/fair-baselines, results/rerun/tuned_results.csv). NOT re-tuned here --
    the params are read back and the method is evaluated once at that fixed config.
  * B4b (accord-only, symmetric) replaces B4a in the main table (Step 2.3).
  * variants P2, P3, A3-full reported separately from the comparators (Bagian 5).
  * order-N for N=1..6, parameter-free, never tuned; order-3 is the proposed method.

Stochastic methods (A2/A4/A5/A6/B5/B6 and the OOF-fold randomness of A3/P2) are run
over seeds [0,1,2]; the per-query reciprocal rank is averaged over seeds and mrr_std is
the std of the per-seed MRR -- the same convention as Phase 2. Supervised methods are
scored strictly out-of-fold (GroupKFold-5 by query). Everything below is computed from
one set of per-query RR vectors, so the significance deltas are exactly consistent with
the table.

Outputs (results/audit/):
  final_table.csv          method, category, best_params, mrr, mrr_std, hits1, hits3
  final_significance.csv   comparison, delta_mrr, ci_low, ci_high, wilcoxon_p
  final_gap.csv            method, mrr_train, mrr_test, gap, note
  final_decomposition.csv  order, avg_share            (order-3, Step 3.5)
  final_sparsity.csv       k, avg_shared_tokens        (k=1..6,  Step 3.6)

Validation (Step 3.2): N=1 must reproduce B2 and N=2 must reproduce order-2 (P1) to
within 1e-3, else the script stops before writing anything.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path

import numpy as np
from scipy import sparse

from .data import load_dataset, leakage_audit
from .evaluate import bootstrap_delta_mrr, per_query_metrics, wilcoxon_rr
from .methods.base import build_features, grouped_folds, label_matrix
from .methods.a3_signature import _features as a3_features
from .methods.a4_bigram_salience import _features as a4_features
from .methods.a6_gbm import _features as a6_features
from .methods.b1_jaccard import B1Jaccard
from .methods.b2_tfidf import B2TfidfCosine
from .methods.p2_fusion import _features as p2_features, _logreg_fit_predict as p2_fp
from .methods.p3_hubidf import P3HubIdf
from .order_ablation import score_matrix, sparsity_by_order
from .order3_analysis import decomposition
from .phase2_tuning import (
    _rr, _fp_a3, _fp_a4, _fp_a6, _a5_fit,
    _a1_score, _a2_score, _b3_score, _b5_score, _b6_score,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "audit"
TUNED = Path(__file__).resolve().parents[1] / "results" / "rerun" / "tuned_results.csv"
SEEDS = [0, 1, 2]
N_BOOT = 10000
TOL = 1e-3
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# --------------------------------------------------------------------------- #
# result holder
# --------------------------------------------------------------------------- #
class R:
    def __init__(self, name, category, best_params, rr, h1, h3, mrr_std=0.0,
                 mrr_train=None, note=""):
        self.name, self.category, self.best_params = name, category, best_params
        self.rr, self.h1, self.h3 = rr, h1, h3          # per-query arrays (or None if skipped)
        self.mrr_std, self.mrr_train, self.note = mrr_std, mrr_train, note

    @property
    def mrr(self):
        return float(self.rr.mean()) if self.rr is not None else float("nan")

    @property
    def hits1(self):
        return float(self.h1.mean()) if self.h1 is not None else float("nan")

    @property
    def hits3(self):
        return float(self.h3.mean()) if self.h3 is not None else float("nan")


def _qr(scores, ds):
    r = per_query_metrics(scores, ds)
    return r.rr, r.h1, r.h3


def _avg_over_seeds(score_fn, ds, params):
    """Stochastic: mean per-query rr/h1/h3 over seeds; mrr_std over per-seed MRR."""
    rr, h1, h3, mrrs = [], [], [], []
    for s in SEEDS:
        a, b, c = _qr(score_fn(params, s), ds)
        rr.append(a); h1.append(b); h3.append(c); mrrs.append(float(a.mean()))
    from statistics import pstdev
    return (np.mean(rr, 0), np.mean(h1, 0), np.mean(h3, 0), pstdev(mrrs))


# --------------------------------------------------------------------------- #
# supervised OOF at a FIXED config (also returns train MRR for the gap table)
# --------------------------------------------------------------------------- #
def _supervised_oof(ds, feature_fn, fp, seed, sparse_feats=False):
    n_q, n_pool = len(ds.queries), ds.n_pool
    Y = label_matrix(ds); rel = [q.relevant for q in ds.queries]
    blocks = [feature_fn(qi) for qi in range(n_q)]
    if sparse_feats:
        blocks = [sparse.csr_matrix(b) for b in blocks]

    def stack(qs):
        return sparse.vstack([blocks[qi] for qi in qs]).tocsr() if sparse_feats \
            else np.vstack([blocks[qi] for qi in qs])

    def yv(qs):
        return np.concatenate([Y[qi] for qi in qs])

    oof = np.zeros((n_q, n_pool)); train_rr = []
    for test_qs in grouped_folds(n_q, 5, seed):
        train_qs = np.array([qi for qi in range(n_q) if qi not in set(test_qs)])
        pred = fp(stack(train_qs), yv(train_qs), stack(test_qs), seed).reshape(len(test_qs), n_pool)
        for k, qi in enumerate(test_qs):
            oof[qi] = pred[k]
        ptr = fp(stack(train_qs), yv(train_qs), stack(train_qs), seed).reshape(len(train_qs), n_pool)
        train_rr += [_rr(ptr[k], rel[qi]) for k, qi in enumerate(train_qs)]
    return oof, float(np.mean(train_rr))


def _a5_oof(ds, feats, rank, iters, seed):
    Q, P = feats.QUt, feats.PUt
    n_q, n_pool = len(ds.queries), ds.n_pool
    Y = label_matrix(ds); rel = [q.relevant for q in ds.queries]
    oof = np.zeros((n_q, n_pool)); train_rr = []
    for test_qs in grouped_folds(n_q, 5, seed):
        train_qs = np.array([qi for qi in range(n_q) if qi not in set(test_qs)])
        d, L = _a5_fit(Q[train_qs], P, Y[train_qs], rank, 1e-3, seed, iters)
        sc_te = (Q[test_qs] * d) @ P.T + (Q[test_qs] @ L) @ (P @ L).T
        for k, qi in enumerate(test_qs):
            oof[qi] = sc_te[k]
        sc_tr = (Q[train_qs] * d) @ P.T + (Q[train_qs] @ L) @ (P @ L).T
        train_rr += [_rr(sc_tr[k], rel[qi]) for k, qi in enumerate(train_qs)]
    return oof, float(np.mean(train_rr))


def _supervised_over_seeds(ds, oof_fn):
    """oof_fn(seed) -> (oof_scores, train_mrr). Average rr over seeds."""
    from statistics import pstdev
    rr, h1, h3, mrrs, trains = [], [], [], [], []
    for s in SEEDS:
        oof, tr = oof_fn(s)
        a, b, c = _qr(oof, ds)
        rr.append(a); h1.append(b); h3.append(c)
        mrrs.append(float(a.mean())); trains.append(tr)
    return (np.mean(rr, 0), np.mean(h1, 0), np.mean(h3, 0),
            pstdev(mrrs), float(np.mean(trains)))


# --------------------------------------------------------------------------- #
def _b4b_scores(ds):
    """B4b: S-BERT over accords-only, both sides (symmetric). NaN if model absent."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL)
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[B4b] skipped: {e}")
        return None
    ptxt = [" ".join(p.accords) or " " for p in ds.products]
    qtxt = [" ".join(q.accords) or " " for q in ds.queries]
    pe = model.encode(ptxt, normalize_embeddings=True, show_progress_bar=False)
    qe = model.encode(qtxt, normalize_embeddings=True, show_progress_bar=False)
    return qe @ pe.T


def _load_tuned():
    out = {}
    with open(TUNED, newline="") as f:
        for row in csv.DictReader(f):
            out[row["method"]] = ast.literal_eval(row["best_params"]) \
                if row["best_params"] not in ("parameter-free", "none") else row["best_params"]
    return out


# --------------------------------------------------------------------------- #
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    leakage_audit(ds, verbose=False)
    feats = build_features(ds)
    tuned = _load_tuned()
    print(f"pool={ds.n_pool} queries={len(ds.queries)} seeds={SEEDS}\n")

    # ---------------- 3.2 mandatory order-N validation ---------------- #
    b2_mrr = per_query_metrics(B2TfidfCosine().scores(ds, feats), ds).mrr
    from .methods.p1_order2 import P1Order2
    p1_mrr = per_query_metrics(P1Order2().scores(ds, feats), ds).mrr
    n1 = per_query_metrics(score_matrix(ds, 1), ds).mrr
    n2 = per_query_metrics(score_matrix(ds, 2), ds).mrr
    print("== 3.2 validation ==")
    print(f"N=1 {n1:.4f} vs B2 {b2_mrr:.4f}  -> {'OK' if abs(n1-b2_mrr)<=TOL else 'MISMATCH'}")
    print(f"N=2 {n2:.4f} vs P1 {p1_mrr:.4f}  -> {'OK' if abs(n2-p1_mrr)<=TOL else 'MISMATCH'}")
    if abs(n1 - b2_mrr) > TOL or abs(n2 - p1_mrr) > TOL:
        raise SystemExit("VALIDATION FAILED (3.2): order-N does not reproduce B2/P1. "
                         "Stopping before writing any CSV.")

    results = []

    # ---------------- comparators ---------------- #
    print("\n== comparators ==")
    # B1 Jaccard (deterministic, param-free)
    rr, h1, h3 = _qr(B1Jaccard().scores(ds, feats), ds)
    results.append(R("B1_jaccard", "comparator", "parameter-free", rr, h1, h3))
    # B2 TF-IDF cosine (deterministic, param-free) == order-1
    rr, h1, h3 = _qr(B2TfidfCosine().scores(ds, feats), ds)
    results.append(R("B2_tfidf_cos", "comparator", "parameter-free", rr, h1, h3))
    # B3 BM25 (deterministic, tuned k1,b)
    p = tuned["B3_bm25"]
    rr, h1, h3 = _qr(_b3_score(ds)(p), ds)
    results.append(R("B3_bm25", "comparator", str(p), rr, h1, h3))
    # B4b S-BERT accord-only (pretrained, param-free)
    S = _b4b_scores(ds)
    if S is None:
        results.append(R("B4b_sbert_accord", "comparator", "parameter-free",
                         None, None, None, note="skipped: model unavailable"))
    else:
        rr, h1, h3 = _qr(S, ds)
        results.append(R("B4b_sbert_accord", "comparator", "parameter-free", rr, h1, h3))
    # B5 word2vec (stochastic, tuned)
    p = tuned["B5_word2vec"]
    rr, h1, h3, sd = _avg_over_seeds(_b5_score(ds), ds, p)
    results.append(R("B5_word2vec", "comparator", str(p), rr, h1, h3, sd))
    # B6 node2vec (stochastic, tuned)
    p = tuned["B6_node2vec"]
    rr, h1, h3, sd = _avg_over_seeds(_b6_score(ds, feats), ds, p)
    results.append(R("B6_node2vec", "comparator", str(p), rr, h1, h3, sd))
    # A1 wheel tree-Wasserstein (deterministic, tuned)
    p = tuned["A1_wheel_treeW"]
    rr, h1, h3 = _qr(_a1_score(ds, feats)(p), ds)
    results.append(R("A1_wheel_treeW", "comparator", str(p), rr, h1, h3))
    # A2 PPMI-SVD (stochastic, tuned)
    p = tuned["A2_ppmi_svd"]
    rr, h1, h3, sd = _avg_over_seeds(_a2_score(feats, ds), ds, p)
    results.append(R("A2_ppmi_svd", "comparator", str(p), rr, h1, h3, sd))
    # A4 bigram salience (supervised, tuned C)
    p = tuned["A4_bigram_salience"]
    rr, h1, h3, sd, tr = _supervised_over_seeds(
        ds, lambda s: _supervised_oof(ds, lambda qi: a4_features(feats, qi),
                                      _fp_a4(p), s, sparse_feats=True))
    results.append(R("A4_bigram_salience", "comparator", str(p), rr, h1, h3, sd, tr))
    # A5 bilinear (supervised, tuned rank,iters)
    p = tuned["A5_bilinear"]
    rr, h1, h3, sd, tr = _supervised_over_seeds(
        ds, lambda s: _a5_oof(ds, feats, p["rank"], p["iters"], s))
    results.append(R("A5_bilinear", "comparator", str(p), rr, h1, h3, sd, tr))
    # A6 GBM fusion (supervised, tuned)
    p = tuned["A6_gbm_fusion"]
    rr, h1, h3, sd, tr = _supervised_over_seeds(
        ds, lambda s: _supervised_oof(ds, lambda qi: a6_features(feats, qi), _fp_a6(p), s))
    results.append(R("A6_gbm_fusion", "comparator", str(p), rr, h1, h3, sd, tr))

    # ---------------- variants (Bagian 5) ---------------- #
    print("\n== variants ==")
    # A3-full signature logistic (supervised, tuned C)
    p = tuned["A3_signature"]
    rr, h1, h3, sd, tr = _supervised_over_seeds(
        ds, lambda s: _supervised_oof(ds, lambda qi: a3_features(feats, qi), _fp_a3(p), s))
    results.append(R("A3_signature_full", "variant", str(p), rr, h1, h3, sd, tr))
    # P2 order-2 + logistic fusion (supervised, fixed logistic)
    rr, h1, h3, sd, tr = _supervised_over_seeds(
        ds, lambda s: _supervised_oof(ds, lambda qi: p2_features(feats, qi), p2_fp, s))
    results.append(R("P2_fusion", "variant", "fixed (C=1.0 logistic)", rr, h1, h3, sd, tr))
    # P3 order-2 hub-IDF (deterministic, param-free)
    rr, h1, h3 = _qr(P3HubIdf().scores(ds, feats), ds)
    results.append(R("P3_hubidf", "variant", "parameter-free", rr, h1, h3))

    # ---------------- proposed order-N ---------------- #
    print("\n== proposed (order-N) ==")
    order_rr = {}
    for N in [1, 2, 3, 4, 5, 6]:
        rr, h1, h3 = _qr(score_matrix(ds, N), ds)
        order_rr[N] = rr
        results.append(R(f"order{N}", "proposed", "parameter-free", rr, h1, h3))
    rr3 = order_rr[3]

    for r in results:
        tag = f" train {r.mrr_train:.4f}" if r.mrr_train is not None else ""
        print(f"{r.name:<20} [{r.category:<10}] mrr {r.mrr:.4f} (±{r.mrr_std:.3f}) "
              f"h1 {r.hits1:.3f} h3 {r.hits3:.3f}{tag}")

    # ---------------- 3.1 final_table.csv ---------------- #
    with open(RESULTS / "final_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "category", "best_params", "mrr", "mrr_std", "hits1", "hits3"])
        for r in results:
            mrr = "" if r.rr is None else f"{r.mrr:.4f}"
            h1 = "" if r.h1 is None else f"{r.hits1:.4f}"
            h3 = "" if r.h3 is None else f"{r.hits3:.4f}"
            w.writerow([r.name, r.category, r.best_params, mrr, f"{r.mrr_std:.4f}", h1, h3])

    # ---------------- 3.3 final_significance.csv ---------------- #
    print("\n== 3.3 significance (order-3 = reference; delta = MRR(order3) - MRR(other)) ==")
    sig_rows = []

    def add_sig(label, rr_a, rr_b):
        bs = bootstrap_delta_mrr(rr_a, rr_b, n_boot=N_BOOT)
        p = wilcoxon_rr(rr_a, rr_b)
        sig_rows.append([label, f"{bs['delta_mrr']:.4f}", f"{bs['ci_low']:.4f}",
                         f"{bs['ci_high']:.4f}", "" if p is None else f"{p:.3e}"])
        print(f"{label:<28}{bs['delta_mrr']:>9.4f}{bs['ci_low']:>10.4f}"
              f"{bs['ci_high']:>10.4f}{('nan' if p is None else f'{p:.3e}'):>14}")

    by_name = {r.name: r for r in results}
    # (a) order-3 vs each comparator
    for name in ["B1_jaccard", "B2_tfidf_cos", "B3_bm25", "B4b_sbert_accord",
                 "B5_word2vec", "B6_node2vec", "A1_wheel_treeW", "A2_ppmi_svd",
                 "A4_bigram_salience", "A5_bilinear", "A6_gbm_fusion"]:
        r = by_name[name]
        if r.rr is None:
            print(f"order-3 vs {name}: SKIPPED (method unavailable)")
            continue
        add_sig(f"order3 vs {name}", rr3, r.rr)
    # (b) order-3 vs A3-full  (critical)
    add_sig("order3 vs A3_signature_full", rr3, by_name["A3_signature_full"].rr)
    # (c) order-3 vs P2, P3
    add_sig("order3 vs P2_fusion", rr3, by_name["P2_fusion"].rr)
    add_sig("order3 vs P3_hubidf", rr3, by_name["P3_hubidf"].rr)
    # (d) order ladder (delta = MRR(N) - MRR(N+1))
    for a, b in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]:
        add_sig(f"order{a} vs order{b}", order_rr[a], order_rr[b])

    with open(RESULTS / "final_significance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "delta_mrr", "ci_low", "ci_high", "wilcoxon_p"])
        w.writerows(sig_rows)

    # ---------------- 3.4 final_gap.csv ---------------- #
    with open(RESULTS / "final_gap.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mrr_train", "mrr_test", "gap", "note"])
        for name in ["A3_signature_full", "A4_bigram_salience", "A5_bilinear", "A6_gbm_fusion"]:
            r = by_name[name]
            w.writerow([name, f"{r.mrr_train:.4f}", f"{r.mrr:.4f}",
                        f"{r.mrr_train - r.mrr:.4f}", "supervised (out-of-fold)"])
        for name in ["A2_ppmi_svd", "B5_word2vec", "B6_node2vec"]:
            r = by_name[name]
            w.writerow([name, "", f"{r.mrr:.4f}", "",
                        "train==test by construction (label-free) -> cannot overfit"])
        b4 = by_name["B4b_sbert_accord"]
        w.writerow(["B4b_sbert_accord", "", "" if b4.rr is None else f"{b4.mrr:.4f}", "",
                    "pretrained, no fit -> cannot overfit"])

    # ---------------- 3.5 final_decomposition.csv ---------------- #
    dec = decomposition(ds, 3)
    with open(RESULTS / "final_decomposition.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "avg_share"])
        for o, s in dec.items():
            w.writerow([o, f"{s:.4f}"])
    print("\n== 3.5 order-3 decomposition ==")
    for o, s in dec.items():
        print(f"  order-{o}: {s:.1%}")

    # ---------------- 3.6 final_sparsity.csv ---------------- #
    sp = sparsity_by_order(ds)
    with open(RESULTS / "final_sparsity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "avg_shared_tokens"])
        for k, avg in sp:
            w.writerow([k, f"{avg:.4f}"])
    print("\n== 3.6 sparsity (avg shared order-k tokens, query vs gold) ==")
    for k, avg in sp:
        print(f"  k={k}: {avg:.4f}")

    print("\nwrote final_table.csv, final_significance.csv, final_gap.csv, "
          "final_decomposition.csv, final_sparsity.csv")


if __name__ == "__main__":
    main()
