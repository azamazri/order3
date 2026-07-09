"""FASE 2.1-2.3 -- fair nested-CV tuning of every comparator.

Outer = GroupKFold-5 by query (test). Inner = GroupKFold-3 inside each outer-train to
select hyperparameters. Test score computed once with the selected params. Stochastic
methods use 3 seeds for tuning (documented deviation from 5, for runtime: the A6 and
node2vec grids are expensive); final metrics are the mean over those seeds.

Imbalance consistency (2.1): A3, A4 already use class_weight="balanced"; A5 uses balanced
sample weights; A6 now receives balanced sample_weight on .fit() (added here, old file
untouched). P1/order-3 is parameter-free and NOT tuned.

Never selects hyperparameters on the test fold. Old method modules are imported, not
modified. Outputs: results/rerun/tuned_results.csv, tuning_traces.csv,
train_test_gap_tuned.csv.
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from .data import load_dataset
from .evaluate import per_query_metrics
from .methods.base import build_features, grouped_folds, label_matrix, _l2norm_rows, _idf
from .order_ablation import score_matrix
from .methods.a3_signature import _features as a3_features
from .methods.a4_bigram_salience import _features as a4_features
from .methods.a6_gbm import _features as a6_features
from .wheel import WheelTree, WHEEL

RESULTS = Path(__file__).resolve().parents[1] / "results" / "rerun"
SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------- #
def _rr(s, rel):
    best = np.max(s[list(rel)])
    g = int(np.sum(s > best)); e = int(np.sum(s == best))
    return float(np.mean(1.0 / np.arange(g + 1, g + e + 1)))


def _inner_split(train_qs, seed, k=3):
    rng = np.random.default_rng(seed + 777)
    order = rng.permutation(train_qs)
    return [np.sort(f) for f in np.array_split(order, k)]


def _grid(d):
    keys = list(d)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[d[k] for k in keys])]


# ---- supervised (feature-based) nested CV ----
def _sw_balanced(y):
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def _fp_a3(params):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fp(Xtr, ytr, Xte, seed):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=params["C"], max_iter=1000, class_weight="balanced",
                                 random_state=seed).fit(sc.transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]
    return fp


def _fp_a4(params):
    from sklearn.linear_model import LogisticRegression

    def fp(Xtr, ytr, Xte, seed):
        clf = LogisticRegression(penalty="l2", C=params["C"], max_iter=2000,
                                 class_weight="balanced", solver="liblinear",
                                 random_state=seed).fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    return fp


def _fp_a6(params):
    from sklearn.ensemble import GradientBoostingClassifier

    def fp(Xtr, ytr, Xte, seed):
        clf = GradientBoostingClassifier(random_state=seed, subsample=0.8,
                                         n_estimators=params["n_estimators"],
                                         max_depth=params["max_depth"],
                                         learning_rate=params["lr"])
        clf.fit(Xtr, ytr, sample_weight=_sw_balanced(ytr))   # 2.1 balanced weights
        return clf.predict_proba(Xte)[:, 1]
    return fp


def nested_supervised(ds, feats, feature_fn, fp_factory, grid, seeds, is_sparse=False):
    n_q, n_pool = len(ds.queries), ds.n_pool
    Y = label_matrix(ds); rel = [q.relevant for q in ds.queries]
    blocks = [feature_fn(feats, qi) for qi in range(n_q)]
    if is_sparse:
        blocks = [sparse.csr_matrix(b) for b in blocks]

    def stack(qs):
        return sparse.vstack([blocks[qi] for qi in qs]).tocsr() if is_sparse \
            else np.vstack([blocks[qi] for qi in qs])

    def y(qs):
        return np.concatenate([Y[qi] for qi in qs])

    te_mrr, te_h1, te_h3, tr_mrr = [], [], [], []
    chosen, traces = [], []
    for s in seeds:
        oof = np.zeros((n_q, n_pool)); train_rr = []
        for of, test_qs in enumerate(grouped_folds(n_q, 5, s)):
            train_qs = np.array([qi for qi in range(n_q) if qi not in set(test_qs)])
            inner = _inner_split(train_qs, s)
            best, best_m = None, -1
            for params in grid:
                fp = fp_factory(params); inner_rr = []
                for i in range(len(inner)):
                    itest = inner[i]
                    itrain = np.concatenate([inner[j] for j in range(len(inner)) if j != i])
                    pred = fp(stack(itrain), y(itrain), stack(itest), s).reshape(len(itest), n_pool)
                    inner_rr += [_rr(pred[k], rel[qi]) for k, qi in enumerate(itest)]
                m = mean(inner_rr)
                traces.append([fp_name(fp_factory), s, of, str(params), f"{m:.4f}"])
                if m > best_m:
                    best, best_m = params, m
            chosen.append(str(best))
            fp = fp_factory(best)
            pred_te = fp(stack(train_qs), y(train_qs), stack(test_qs), s).reshape(len(test_qs), n_pool)
            for k, qi in enumerate(test_qs):
                oof[qi] = pred_te[k]
            pred_tr = fp(stack(train_qs), y(train_qs), stack(train_qs), s).reshape(len(train_qs), n_pool)
            train_rr += [_rr(pred_tr[k], rel[qi]) for k, qi in enumerate(train_qs)]
        r = per_query_metrics(oof, ds)
        te_mrr.append(r.mrr); te_h1.append(r.hits1); te_h3.append(r.hits3)
        tr_mrr.append(mean(train_rr))
    return _agg(te_mrr, te_h1, te_h3, tr_mrr, chosen), traces


def fp_name(factory):
    return {id(_fp_a3): "A3_signature", id(_fp_a4): "A4_bigram_salience",
            id(_fp_a6): "A6_gbm_fusion"}.get(id(factory), "sup")


# ---- A5 bilinear (L-BFGS) nested CV ----
def _a5_fit(Q, P, Y, rank, l2, seed, maxiter):
    V = P.shape[1]
    pos = Y.sum(); neg = Y.size - pos
    Wp = (Y.size / (2 * pos)) if pos else 1.0; Wn = (Y.size / (2 * neg)) if neg else 1.0
    W = np.where(Y > 0, Wp, Wn); Wsum = W.sum()

    def fun(theta):
        d = theta[:V]; L = theta[V:].reshape(V, rank)
        QL = Q @ L; PL = P @ L
        S = (Q * d) @ P.T + QL @ PL.T
        pr = 1.0 / (1.0 + np.exp(-np.clip(S, -30, 30)))
        eps = 1e-12
        loss = float((W * -(Y * np.log(pr + eps) + (1 - Y) * np.log(1 - pr + eps))).sum() / Wsum
                     + 0.5 * l2 * (d @ d + (L * L).sum()))
        G = W * (pr - Y) / Wsum
        gd = np.einsum("ij,ik,jk->k", G, Q, P) + l2 * d
        gL = Q.T @ (G @ PL) + P.T @ (G.T @ QL) + l2 * L
        return loss, np.concatenate([gd, gL.ravel()])
    rng = np.random.default_rng(seed)
    x0 = np.concatenate([np.zeros(V), 0.01 * rng.standard_normal(V * rank)])
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x[:V], res.x[V:].reshape(V, rank)


def nested_a5(ds, feats, grid, seeds):
    Q, P = feats.QUt, feats.PUt
    n_q, n_pool = len(ds.queries), ds.n_pool
    Y = np.zeros((n_q, n_pool))
    for q in ds.queries:
        for pidx in q.relevant:
            Y[q.idx, pidx] = 1.0
    rel = [q.relevant for q in ds.queries]

    def score(qs, d, L):
        return (Q[qs] * d) @ P.T + (Q[qs] @ L) @ (P @ L).T

    te_mrr, te_h1, te_h3, tr_mrr, chosen, traces = [], [], [], [], [], []
    for s in seeds:
        oof = np.zeros((n_q, n_pool)); train_rr = []
        for of, test_qs in enumerate(grouped_folds(n_q, 5, s)):
            train_qs = np.array([qi for qi in range(n_q) if qi not in set(test_qs)])
            inner = _inner_split(train_qs, s)
            best, best_m = None, -1
            for params in grid:
                inner_rr = []
                for i in range(len(inner)):
                    itest = inner[i]
                    itrain = np.concatenate([inner[j] for j in range(len(inner)) if j != i])
                    d, L = _a5_fit(Q[itrain], P, Y[itrain], params["rank"], 1e-3, s, params["iters"])
                    sc = score(itest, d, L)
                    inner_rr += [_rr(sc[k], rel[qi]) for k, qi in enumerate(itest)]
                m = mean(inner_rr)
                traces.append(["A5_bilinear", s, of, str(params), f"{m:.4f}"])
                if m > best_m:
                    best, best_m = params, m
            chosen.append(str(best))
            d, L = _a5_fit(Q[train_qs], P, Y[train_qs], best["rank"], 1e-3, s, best["iters"])
            sc_te = score(test_qs, d, L)
            for k, qi in enumerate(test_qs):
                oof[qi] = sc_te[k]
            sc_tr = score(train_qs, d, L)
            train_rr += [_rr(sc_tr[k], rel[qi]) for k, qi in enumerate(train_qs)]
        r = per_query_metrics(oof, ds)
        te_mrr.append(r.mrr); te_h1.append(r.hits1); te_h3.append(r.hits3); tr_mrr.append(mean(train_rr))
    return _agg(te_mrr, te_h1, te_h3, tr_mrr, chosen), traces


# ---- unsupervised: cache per-(param,seed) query metrics, then nested-select ----
def nested_unsupervised(ds, feats, score_fn, grid, seeds):
    """score_fn(params, seed) -> (n_q, n_pool) scores. Cache per-query rr/h1/h3."""
    n_q = len(ds.queries)
    cache = {}  # (pi, seed) -> QueryResult
    for pi, params in enumerate(grid):
        for s in seeds:
            cache[(pi, s)] = per_query_metrics(score_fn(params, s), ds)

    te_mrr, te_h1, te_h3, chosen, traces = [], [], [], [], []
    for s in seeds:
        arr_rr = np.zeros(n_q); arr_h1 = np.zeros(n_q); arr_h3 = np.zeros(n_q)
        for of, test_qs in enumerate(grouped_folds(n_q, 5, s)):
            train_qs = np.array([qi for qi in range(n_q) if qi not in set(test_qs)])
            best, best_m = None, -1
            for pi, params in enumerate(grid):
                m = float(cache[(pi, s)].rr[train_qs].mean())   # inner = outer-train MRR
                traces.append([score_fn.__name__, s, of, str(params), f"{m:.4f}"])
                if m > best_m:
                    best, best_m, best_pi = params, m, pi
            chosen.append(str(best))
            qr = cache[(best_pi, s)]
            arr_rr[test_qs] = qr.rr[test_qs]; arr_h1[test_qs] = qr.h1[test_qs]; arr_h3[test_qs] = qr.h3[test_qs]
        te_mrr.append(float(arr_rr.mean())); te_h1.append(float(arr_h1.mean())); te_h3.append(float(arr_h3.mean()))
    return _agg(te_mrr, te_h1, te_h3, None, chosen), traces


# ---- deterministic: single rr per config, nested-select (no seed) ----
def nested_deterministic(ds, score_fn, grid):
    n_q = len(ds.queries)
    qr = [per_query_metrics(score_fn(p), ds) for p in grid]
    arr_rr = np.zeros(n_q); arr_h1 = np.zeros(n_q); arr_h3 = np.zeros(n_q)
    chosen, traces = [], []
    for of, test_qs in enumerate(grouped_folds(n_q, 5, 0)):
        train_qs = np.array([qi for qi in range(n_q) if qi not in set(test_qs)])
        best_pi, best_m = 0, -1
        for pi, p in enumerate(grid):
            m = float(qr[pi].rr[train_qs].mean())
            traces.append([score_fn.__name__, 0, of, str(p), f"{m:.4f}"])
            if m > best_m:
                best_pi, best_m = pi, m
        chosen.append(str(grid[best_pi]))
        arr_rr[test_qs] = qr[best_pi].rr[test_qs]; arr_h1[test_qs] = qr[best_pi].h1[test_qs]
        arr_h3[test_qs] = qr[best_pi].h3[test_qs]
    return _agg([float(arr_rr.mean())], [float(arr_h1.mean())], [float(arr_h3.mean())], None, chosen), traces


def _agg(te_mrr, te_h1, te_h3, tr_mrr, chosen):
    from collections import Counter
    best = Counter(chosen).most_common(1)[0][0]
    return {
        "best_params": best,
        "mrr": mean(te_mrr), "mrr_std": pstdev(te_mrr) if len(te_mrr) > 1 else 0.0,
        "hits1": mean(te_h1), "hits3": mean(te_h3),
        "mrr_train": (mean(tr_mrr) if tr_mrr else None),
    }


# --------------------------------------------------------------------------- #
# unsupervised score functions
def _a2_score(feats, ds):
    from sklearn.decomposition import TruncatedSVD
    from itertools import combinations
    V = len(feats.vocab)
    C = np.zeros((V, V))
    for frag in [p.accords for p in ds.products] + [q.accords for q in ds.queries]:
        idxs = [feats.vidx[a] for a in set(frag)]
        for i, j in combinations(idxs, 2):
            C[i, j] += 1; C[j, i] += 1
        for i in idxs:
            C[i, i] += 1
    total = C.sum(); row = C.sum(1, keepdims=True); col = C.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C * total) / (row * col))
    pmi[~np.isfinite(pmi)] = 0.0; ppmi = np.maximum(pmi, 0.0)

    def fn(params, seed):
        k = min(params["rank"], V - 1)
        emb = TruncatedSVD(n_components=k, random_state=seed).fit_transform(ppmi)
        def vec(acc):
            v = [emb[feats.vidx[a]] for a in acc if a in feats.vidx]
            return np.mean(v, 0) if v else np.zeros(k)
        Pm = _l2norm_rows(np.vstack([vec(p.accords) for p in ds.products]))
        Qm = _l2norm_rows(np.vstack([vec(q.accords) for q in ds.queries]))
        return Qm @ Pm.T
    fn.__name__ = "A2_ppmi_svd"
    return fn


def _b5_score(ds):
    from gensim.models import Word2Vec
    sents = [p.accords for p in ds.products if p.accords] + [q.accords for q in ds.queries if q.accords]

    def fn(params, seed):
        m = Word2Vec(sents, vector_size=params["dim"], window=params["window"],
                     min_count=1, sg=1, epochs=params["epochs"], workers=1, seed=seed)
        def vec(acc):
            v = [m.wv[a] for a in acc if a in m.wv]
            return np.mean(v, 0) if v else np.zeros(params["dim"])
        P = _l2norm_rows(np.vstack([vec(p.accords) for p in ds.products]))
        Q = _l2norm_rows(np.vstack([vec(q.accords) for q in ds.queries]))
        return Q @ P.T
    fn.__name__ = "B5_word2vec"
    return fn


def _b6_score(ds, feats):
    import networkx as nx
    from node2vec import Node2Vec
    from itertools import combinations
    G = nx.Graph(); G.add_nodes_from(feats.vocab); cooc = {}
    for frag in [p.accords for p in ds.products] + [q.accords for q in ds.queries]:
        for a, b in combinations(sorted(set(frag)), 2):
            cooc[(a, b)] = cooc.get((a, b), 0) + 1
    for (a, b), w in cooc.items():
        G.add_edge(a, b, weight=w)

    def fn(params, seed):
        n2v = Node2Vec(G, dimensions=params["dim"], walk_length=20, num_walks=50,
                       p=params["p"], q=params["q"], weight_key="weight",
                       workers=1, seed=seed, quiet=True)
        m = n2v.fit(window=5, min_count=1, sg=1, workers=1, seed=seed)
        def vec(acc):
            v = [m.wv[a] for a in acc if a in m.wv]
            return np.mean(v, 0) if v else np.zeros(params["dim"])
        P = _l2norm_rows(np.vstack([vec(p.accords) for p in ds.products]))
        Q = _l2norm_rows(np.vstack([vec(q.accords) for q in ds.queries]))
        return Q @ P.T
    fn.__name__ = "B6_node2vec"
    return fn


def _b3_score(ds):
    from rank_bm25 import BM25Okapi
    corpus = [p.accords if p.accords else ["<empty>"] for p in ds.products]

    def fn(params):
        bm = BM25Okapi(corpus, k1=params["k1"], b=params["b"])
        out = np.zeros((len(ds.queries), ds.n_pool))
        for q in ds.queries:
            out[q.idx] = bm.get_scores(q.accords if q.accords else ["<empty>"])
        return out
    fn.__name__ = "B3_bm25"
    return fn


def _a1_score(ds, feats):
    tree = WheelTree()
    n_acc = len(tree.accords)
    n_super = len(WHEEL)
    n_sub = len(tree.edge_w) - n_acc - n_super
    Pd = np.vstack([tree.distribution(p.accords) for p in ds.products])
    Qd = np.vstack([tree.distribution(q.accords) for q in ds.queries])
    Pm = Pd @ tree.edge_mask.T; Qm = Qd @ tree.edge_mask.T

    def fn(params):
        wa, ws, wsup = params["w"]
        ew = np.concatenate([np.full(n_acc, wa), np.full(n_sub, ws), np.full(n_super, wsup)])
        out = np.zeros((feats.n_q, ds.n_pool))
        for qi in range(feats.n_q):
            out[qi] = -(np.abs(Pm - Qm[qi]) @ ew)
        return out
    fn.__name__ = "A1_wheel_treeW"
    return fn


# --------------------------------------------------------------------------- #
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(); feats = build_features(ds)
    results, traces_all = {}, []

    def run(name, res_traces):
        res, tr = res_traces
        results[name] = res; traces_all.extend(tr)
        print(f"{name:<20} mrr {res['mrr']:.4f} (±{res['mrr_std']:.3f}) "
              f"h1 {res['hits1']:.3f} h3 {res['hits3']:.3f} best={res['best_params']}"
              + (f" train {res['mrr_train']:.4f}" if res['mrr_train'] is not None else ""), flush=True)

    print("== supervised ==", flush=True)
    run("A3_signature", nested_supervised(ds, feats, a3_features, _fp_a3, _grid({"C": [0.01, 0.1, 1, 10]}), SEEDS))
    run("A4_bigram_salience", nested_supervised(ds, feats, a4_features, _fp_a4, _grid({"C": [0.001, 0.01, 0.1, 1, 10]}), SEEDS, is_sparse=True))
    run("A6_gbm_fusion", nested_supervised(ds, feats, a6_features, _fp_a6, _grid({"n_estimators": [100, 300], "max_depth": [2, 3, 5], "lr": [0.01, 0.05, 0.1]}), SEEDS))
    run("A5_bilinear", nested_a5(ds, feats, _grid({"rank": [2, 4, 8, 16], "iters": [1000, 5000]}), SEEDS))

    print("== unsupervised ==", flush=True)
    run("A2_ppmi_svd", nested_unsupervised(ds, feats, _a2_score(feats, ds), _grid({"rank": [8, 16, 32, 50]}), SEEDS))
    run("B5_word2vec", nested_unsupervised(ds, feats, _b5_score(ds), _grid({"dim": [8, 16, 32, 64], "window": [2, 5], "epochs": [50, 200]}), SEEDS))
    run("B6_node2vec", nested_unsupervised(ds, feats, _b6_score(ds, feats), _grid({"dim": [8, 16, 32, 64], "p": [0.25, 1, 4], "q": [0.25, 1, 4]}), SEEDS))

    print("== deterministic ==", flush=True)
    run("B3_bm25", nested_deterministic(ds, _b3_score(ds), _grid({"k1": [0.9, 1.2, 1.5, 2.0], "b": [0.0, 0.3, 0.75]})))
    run("A1_wheel_treeW", nested_deterministic(ds, _a1_score(ds, feats), [{"w": (1, 2, 3)}, {"w": (1, 1, 1)}, {"w": (1, 3, 9)}]))

    # order-3 reference (untuned)
    o3 = per_query_metrics(score_matrix(ds, 3), ds)
    results["order3"] = {"best_params": "parameter-free", "mrr": o3.mrr, "mrr_std": 0.0,
                         "hits1": o3.hits1, "hits3": o3.hits3, "mrr_train": o3.mrr}
    print(f"order3 (untuned)     mrr {o3.mrr:.4f} h1 {o3.hits1:.3f} h3 {o3.hits3:.3f}")

    with open(RESULTS / "tuned_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "best_params", "mrr", "mrr_std", "hits1", "hits3"])
        for name, r in results.items():
            w.writerow([name, r["best_params"], f"{r['mrr']:.4f}", f"{r['mrr_std']:.4f}",
                        f"{r['hits1']:.4f}", f"{r['hits3']:.4f}"])
    with open(RESULTS / "tuning_traces.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "seed", "outer_fold", "params", "inner_mrr"])
        w.writerows(traces_all)
    with open(RESULTS / "train_test_gap_tuned.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "mrr_train", "mrr_test", "gap"])
        for name in ["A3_signature", "A4_bigram_salience", "A5_bilinear", "A6_gbm_fusion"]:
            r = results[name]
            w.writerow([name, f"{r['mrr_train']:.4f}", f"{r['mrr']:.4f}",
                        f"{r['mrr_train'] - r['mrr']:.4f}"])

    print(f"\nwrote tuned_results.csv, tuning_traces.csv, train_test_gap_tuned.csv")


if __name__ == "__main__":
    main()
