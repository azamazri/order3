"""FASE 1 -- diagnostic only (fair-baselines audit). Read-only w.r.t. old methods.

Determines whether the parametric comparators OVERFIT or UNDERFIT / are misconfigured,
by measuring in-fold (train) vs out-of-fold (test) MRR under the *identical* protocol
(full-pool retrieval, 209 queries, pool 340, expected-rank tie-breaking, GroupKFold-5 by query).

Reuses the existing method feature/fit logic by IMPORT only (no old file is modified).
Outputs: results/rerun/phase1_report.md, train_test_gap.csv, a5_loss_curve.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy import sparse

from .data import load_dataset
from .evaluate import per_query_metrics
from .methods.base import build_features, grouped_folds, label_matrix
from .order_ablation import score_matrix

# feature builders + model params reused from the (unmodified) method modules
from .methods.a3_signature import _features as a3_features
from .methods.a4_bigram_salience import _features as a4_features
from .methods.a6_gbm import _features as a6_features
from .methods.a2_ppmi_svd import A2PpmiSvd
from .methods.b5_word2vec import B5Word2Vec
from .methods.b6_node2vec import B6Node2Vec
from .methods.b1_jaccard import B1Jaccard
from .methods.b2_tfidf import B2TfidfCosine
from .methods.b3_bm25 import B3BM25
from .methods.a1_wheel import A1Wheel

RESULTS = Path(__file__).resolve().parents[1] / "results" / "rerun"
SEEDS = [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
def _rr(s: np.ndarray, rel) -> float:
    """Expected reciprocal rank for one query (same tie rule as evaluate.py)."""
    best = np.max(s[list(rel)])
    g = int(np.sum(s > best))
    e = int(np.sum(s == best))
    return float(np.mean(1.0 / np.arange(g + 1, g + e + 1)))


# ---- fit-once model builders (params copied verbatim from the method modules) ----
def _fit_logreg_std(Xtr, ytr, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=seed).fit(sc.transform(Xtr), ytr)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def _fit_l2logreg(Xtr, ytr, seed):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                             class_weight="balanced", solver="liblinear",
                             random_state=seed).fit(Xtr, ytr)
    return lambda X: clf.predict_proba(X)[:, 1]


def _fit_gbm(Xtr, ytr, seed):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(random_state=seed, n_estimators=200, max_depth=3,
                                     learning_rate=0.05, subsample=0.8).fit(Xtr, ytr)
    return lambda X: clf.predict_proba(X)[:, 1]


def supervised_train_test(ds, feats, feature_fn, fit_builder, seeds, is_sparse=False):
    """Mean in-fold (train) and out-of-fold (test) MRR for a label-supervised method."""
    n_q, n_pool = len(ds.queries), ds.n_pool
    Y = label_matrix(ds)
    blocks = [feature_fn(feats, qi) for qi in range(n_q)]
    if is_sparse:
        blocks = [sparse.csr_matrix(b) for b in blocks]
    rel = [q.relevant for q in ds.queries]

    tr_all, te_all = [], []
    for s in seeds:
        oof = np.zeros((n_q, n_pool))
        train_rr = []
        for f, test_qs in enumerate(grouped_folds(n_q, 5, s)):
            train_qs = np.array([qi for qi in range(n_q) if qi not in set(test_qs)])
            stack = (lambda qs: sparse.vstack([blocks[qi] for qi in qs]).tocsr()) if is_sparse \
                else (lambda qs: np.vstack([blocks[qi] for qi in qs]))
            Xtr = stack(train_qs)
            ytr = np.concatenate([Y[qi] for qi in train_qs])
            predict = fit_builder(Xtr, ytr, s)
            # test (out-of-fold)
            pred_te = predict(stack(test_qs)).reshape(len(test_qs), n_pool)
            for k, qi in enumerate(test_qs):
                oof[qi] = pred_te[k]
            # train (in-fold)
            pred_tr = predict(Xtr).reshape(len(train_qs), n_pool)
            for k, qi in enumerate(train_qs):
                train_rr.append(_rr(pred_tr[k], rel[qi]))
        te_all.append(per_query_metrics(oof, ds).mrr)
        tr_all.append(mean(train_rr))
    return mean(tr_all), mean(te_all)


# --------------------------------------------------------------------------- #
def a5_train_test_and_loss(ds, feats, seeds):
    """A5 bilinear: train/test MRR (per-fold fit) + full-data loss curve (seed 0)."""
    Q, P = feats.QUt, feats.PUt
    n_q, n_pool = len(ds.queries), ds.n_pool
    Y = np.zeros((n_q, n_pool))
    for q in ds.queries:
        for pidx in q.relevant:
            Y[q.idx, pidx] = 1.0
    rel = [q.relevant for q in ds.queries]
    rank, lr, l2, iters = 8, 0.5, 1e-3, 300

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def train(Qtr, Ytr, seed, record=False):
        rng = np.random.default_rng(seed)
        V = Qtr.shape[1]
        d = np.zeros(V); L = 0.01 * rng.standard_normal((V, rank))
        pos = Ytr.sum(); neg = Ytr.size - pos
        w_pos = (Ytr.size / (2 * pos)) if pos else 1.0
        w_neg = (Ytr.size / (2 * neg)) if neg else 1.0
        W = np.where(Ytr > 0, w_pos, w_neg); Wsum = W.sum()
        losses = []
        for _ in range(iters):
            QL = Qtr @ L; PL = P @ L
            S = (Qtr * d) @ P.T + QL @ PL.T
            pr = sigmoid(S)
            if record:
                eps = 1e-12
                bce = -(Ytr * np.log(pr + eps) + (1 - Ytr) * np.log(1 - pr + eps))
                losses.append(float((W * bce).sum() / Wsum))
            G = W * (pr - Ytr) / Wsum
            grad_d = np.einsum("ij,ik,jk->k", G, Qtr, P)
            grad_L = Qtr.T @ (G @ PL) + P.T @ (G.T @ QL)
            d -= lr * (grad_d + l2 * d); L -= lr * (grad_L + l2 * L)
        return d, L, losses

    # loss curve on the full training set (all queries), seed 0
    _, _, loss_curve = train(Q, Y, 0, record=True)

    tr_all, te_all = [], []
    for s in seeds:
        oof = np.zeros((n_q, n_pool)); train_rr = []
        for test_qs in grouped_folds(n_q, 5, s):
            train_qs = np.array([qi for qi in range(n_q) if qi not in set(test_qs)])
            d, L, _ = train(Q[train_qs], Y[train_qs], s)
            def sc(qs):
                return (Q[qs] * d) @ P.T + (Q[qs] @ L) @ (P @ L).T
            for k, qi in enumerate(test_qs):
                oof[qi] = sc([qi])[0]
            tr = sc(train_qs)
            for k, qi in enumerate(train_qs):
                train_rr.append(_rr(tr[k], rel[qi]))
        te_all.append(per_query_metrics(oof, ds).mrr)
        tr_all.append(mean(train_rr))
    return mean(tr_all), mean(te_all), loss_curve


# --------------------------------------------------------------------------- #
def unsupervised_mrr(ds, feats, method, seeds):
    """A2/B5/B6: no label fit, so train == test; average MRR over seeds."""
    ms = []
    for s in seeds:
        S = method.scores(ds, feats, seed=s)
        ms.append(per_query_metrics(S, ds).mrr)
    return mean(ms), mean(ms)


def deterministic_mrr(ds, feats, S):
    m = per_query_metrics(S, ds).mrr
    return m, m


# --------------------------------------------------------------------------- #
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    feats = build_features(ds)

    # ---- 1.1 vocabulary / token stats ----
    V = len(feats.vocab)
    prod_counts = [len(p.accords) for p in ds.products]
    q_counts = [len(q.accords) for q in ds.queries]
    w2v_tokens = sum(len(p.accords) for p in ds.products) + sum(len(q.accords) for q in ds.queries)

    # ---- 1.2 + 1.3 train/test gap ----
    rows = []  # (method, mrr_train, mrr_test)
    print("computing train/test MRR ...", flush=True)
    # deterministic / parameter-free
    rows.append(("B1_jaccard", *deterministic_mrr(ds, feats, B1Jaccard().scores(ds, feats))))
    rows.append(("B2_tfidf_cos", *deterministic_mrr(ds, feats, B2TfidfCosine().scores(ds, feats))))
    rows.append(("B3_bm25", *deterministic_mrr(ds, feats, B3BM25().scores(ds, feats))))
    rows.append(("A1_wheel_treeW", *deterministic_mrr(ds, feats, A1Wheel().scores(ds, feats))))
    rows.append(("order3", *deterministic_mrr(ds, feats, score_matrix(ds, 3))))
    # supervised label-fit (train/test gap meaningful)
    rows.append(("A3_signature", *supervised_train_test(ds, feats, a3_features, _fit_logreg_std, SEEDS)))
    rows.append(("A4_bigram_salience", *supervised_train_test(ds, feats, a4_features, _fit_l2logreg, SEEDS, is_sparse=True)))
    rows.append(("A6_gbm_fusion", *supervised_train_test(ds, feats, a6_features, _fit_gbm, SEEDS)))
    a5_tr, a5_te, a5_loss = a5_train_test_and_loss(ds, feats, SEEDS)
    rows.append(("A5_bilinear", a5_tr, a5_te))
    # unsupervised (no label fit -> train == test)
    rows.append(("A2_ppmi_svd", *unsupervised_mrr(ds, feats, A2PpmiSvd(), SEEDS)))
    rows.append(("B5_word2vec", *unsupervised_mrr(ds, feats, B5Word2Vec(), SEEDS)))
    rows.append(("B6_node2vec", *unsupervised_mrr(ds, feats, B6Node2Vec(), SEEDS)))

    with open(RESULTS / "train_test_gap.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mrr_train", "mrr_test", "gap"])
        for name, tr, te in rows:
            w.writerow([name, f"{tr:.4f}", f"{te:.4f}", f"{tr - te:.4f}"])

    with open(RESULTS / "a5_loss_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "loss"])
        for i, L in enumerate(a5_loss):
            w.writerow([i, f"{L:.6f}"])

    # ---- 1.4 dim sanity ----
    dim_flags = {
        "B5_word2vec dim=64": 64 > V,
        "B6_node2vec dim=64": 64 > V,
        f"A2_ppmi_svd rank=min(50,V-1)={min(50, V-1)}": min(50, V - 1) > V,
    }

    # ---- A5 loss behaviour ----
    l0, lend = a5_loss[0], a5_loss[-1]
    lmin = min(a5_loss)
    diffs = np.diff(a5_loss)
    monotone = bool(np.all(diffs <= 1e-9))
    diverged = bool(a5_loss[-1] > a5_loss[0] * 1.05)
    oscillates = bool((diffs > 1e-6).sum() > 5)

    # ---- write report ----
    with open(RESULTS / "phase1_report.md", "w", encoding="utf-8") as f:
        f.write("# Fase 1 -- Diagnostic Report (fair-baselines)\n\n")
        f.write("Protocol unchanged: full-pool retrieval, 209 queries, pool 340, expected-rank "
                "tie-breaking, GroupKFold-5 by query, 5 seeds for stochastic methods.\n\n")

        f.write("## 1.1 Vocabulary / token stats\n\n")
        f.write(f"- |V| (unique accords over 340 products + 209 queries) = **{V}**\n")
        f.write(f"- accords/product: mean {mean(prod_counts):.2f}, median {median(prod_counts)}, "
                f"min {min(prod_counts)}, max {max(prod_counts)}\n")
        f.write(f"- accords/query: mean {mean(q_counts):.2f}, median {median(q_counts)}, "
                f"min {min(q_counts)}, max {max(q_counts)}\n")
        f.write(f"- total tokens available to train Word2Vec (sum of accords over "
                f"products+queries) = **{w2v_tokens}**\n\n")

        f.write("## 1.2 Train vs test MRR\n\n")
        f.write("See `train_test_gap.csv`. Note: A2/B5/B6 are *unsupervised* (embeddings fit on "
                "the corpus, no labels), so there is no in-fold/out-of-fold label split -- "
                "train == test by construction; a low value there is a representation issue, not "
                "label overfitting. B1/B2/B3/A1/order3 are parameter-free (train == test).\n\n")
        f.write("| method | mrr_train | mrr_test | gap |\n|---|---|---|---|\n")
        for name, tr, te in rows:
            f.write(f"| {name} | {tr:.4f} | {te:.4f} | {tr - te:+.4f} |\n")
        f.write("\n**Interpretation key (do not copy into paper):** large gap + low test = "
                "overfitting (paper claim holds); small gap + low test = UNDERFITTING / "
                "misconfiguration (paper claim would be wrong).\n\n")

        f.write("## 1.3 A5 (bilinear) convergence, lr=0.5, 300 iters, seed 0, full training set\n\n")
        f.write(f"- loss[0] = {l0:.4f}, loss[end] = {lend:.4f}, loss[min] = {lmin:.4f}\n")
        f.write(f"- monotonically non-increasing: **{monotone}**; diverged (end > 1.05*start): "
                f"**{diverged}**; oscillates (>5 up-steps): **{oscillates}**\n")
        f.write("- full curve in `a5_loss_curve.csv`.\n\n")

        f.write("## 1.4 Dimension sanity (dim vs |V|)\n\n")
        for k, flag in dim_flags.items():
            f.write(f"- {k}: dim > |V| ? **{flag}**"
                    + ("  <-- DEGENERATE" if flag else "") + "\n")
        f.write("\n")

        f.write("## 1.5 Data seen at fit-time (information access; not label leakage)\n\n")
        f.write("- **P1/order-N IDF**: computed over the **340 products only** "
                "(`base.py:153` `idf_u=_idf(PU)`, PU = products).\n")
        f.write("- **B5 Word2Vec**: trained on **products + queries** accord lists "
                "(`b5_word2vec.py:24-25`).\n")
        f.write("- **B6 node2vec**: co-occurrence graph built from **products + queries** "
                "(`b6_node2vec.py:35`).\n")
        f.write("- **A2 PPMI**: co-occurrence matrix from **products + queries** "
                "(`a2_ppmi_svd.py:29`).\n")
        f.write("- Query accords are external (Fragrantica); no label is ever used by the "
                "unsupervised fits. This is information *asymmetry to document*, not label leakage.\n")

    # console summary
    print("\n=== Fase 1 summary ===")
    print(f"|V| = {V} | w2v tokens = {w2v_tokens}")
    for name, tr, te in rows:
        print(f"{name:<20} train {tr:.4f} test {te:.4f} gap {tr-te:+.4f}")
    print(f"A5 loss: {l0:.4f} -> {lend:.4f} (min {lmin:.4f}) monotone={monotone} diverged={diverged}")
    for k, flag in dim_flags.items():
        print(f"dim sanity: {k} -> degenerate={flag}")
    print(f"\nwrote {RESULTS}/phase1_report.md, train_test_gap.csv, a5_loss_curve.csv")


if __name__ == "__main__":
    main()
