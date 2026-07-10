"""FASE 2.0 -- A5 bilinear bug fix.

Phase 1 showed A5 does not overfit; it fails to converge (loss 0.693 -> 0.638 over
300 GD iters, train MRR 0.184). Note: the A5 loss ALREADY uses balanced sample weights
(w_pos = |Y|/2n_pos, w_neg = |Y|/2n_neg) -- see src/methods/a5_bilinear.py -- so the
imbalance hypothesis (2.0a) is not the cause. The actual defect is the hand-written GD
(fixed step, no convergence test). Fix (2.0b/c): replace GD with L-BFGS-B (analytic
gradient, real convergence criterion, up to 5000 iters). We keep the balanced weights.

Compares BEFORE (GD, from Phase 1) vs AFTER (L-BFGS-B): loss curve + train/test MRR.
Old A5 file is not modified. Output: results/rerun/a5_bugfix_report.md,
a5_loss_curve_lbfgs.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.optimize import minimize

from .data import load_dataset
from .evaluate import per_query_metrics
from .methods.base import build_features, grouped_folds

RESULTS = Path(__file__).resolve().parents[1] / "results" / "rerun"
SEEDS = [0, 1, 2, 3, 4]
RANK, L2, MAXITER = 8, 1e-3, 5000


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _rr(s, rel):
    best = np.max(s[list(rel)])
    g = int(np.sum(s > best)); e = int(np.sum(s == best))
    return float(np.mean(1.0 / np.arange(g + 1, g + e + 1)))


def _weights(Y):
    pos = Y.sum(); neg = Y.size - pos
    w_pos = (Y.size / (2 * pos)) if pos else 1.0
    w_neg = (Y.size / (2 * neg)) if neg else 1.0
    W = np.where(Y > 0, w_pos, w_neg)
    return W, W.sum()


def _make_objective(Q, P, Y, rank, l2):
    V = P.shape[1]
    W, Wsum = _weights(Y)

    def fun(theta):
        d = theta[:V]; L = theta[V:].reshape(V, rank)
        QL = Q @ L; PL = P @ L
        S = (Q * d) @ P.T + QL @ PL.T
        pr = _sigmoid(S)
        eps = 1e-12
        bce = -(Y * np.log(pr + eps) + (1 - Y) * np.log(1 - pr + eps))
        loss = float((W * bce).sum() / Wsum + 0.5 * l2 * (d @ d + (L * L).sum()))
        G = W * (pr - Y) / Wsum
        grad_d = np.einsum("ij,ik,jk->k", G, Q, P) + l2 * d
        grad_L = Q.T @ (G @ PL) + P.T @ (G.T @ QL) + l2 * L
        return loss, np.concatenate([grad_d, grad_L.ravel()])
    return fun, V


def _fit_lbfgs(Q, P, Y, rank, l2, seed, maxiter=MAXITER, record=False):
    fun, V = _make_objective(Q, P, Y, rank, l2)
    rng = np.random.default_rng(seed)
    x0 = np.concatenate([np.zeros(V), 0.01 * rng.standard_normal(V * rank)])
    curve = []
    cb = (lambda xk: curve.append(fun(xk)[0])) if record else None
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": maxiter * 2}, callback=cb)
    d = res.x[:V]; L = res.x[V:].reshape(V, rank)
    return d, L, curve, res


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    feats = build_features(ds)
    Q, P = feats.QUt, feats.PUt
    n_q, n_pool = len(ds.queries), ds.n_pool
    Y = np.zeros((n_q, n_pool))
    for q in ds.queries:
        for pidx in q.relevant:
            Y[q.idx, pidx] = 1.0
    rel = [q.relevant for q in ds.queries]

    # loss curve AFTER (full training set, seed 0)
    _, _, curve, res0 = _fit_lbfgs(Q, P, Y, RANK, L2, 0, record=True)
    with open(RESULTS / "a5_loss_curve_lbfgs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iter", "loss"])
        for i, L in enumerate(curve):
            w.writerow([i, f"{L:.6f}"])

    # train/test MRR AFTER (per-fold L-BFGS), 5 seeds
    tr_all, te_all = [], []
    for s in SEEDS:
        oof = np.zeros((n_q, n_pool)); train_rr = []
        for test_qs in grouped_folds(n_q, 5, s):
            train_qs = np.array([qi for qi in range(n_q) if qi not in set(test_qs)])
            d, L, _, _ = _fit_lbfgs(Q[train_qs], P, Y[train_qs], RANK, L2, s)
            def sc(qs):
                return (Q[qs] * d) @ P.T + (Q[qs] @ L) @ (P @ L).T
            for k, qi in enumerate(test_qs):
                oof[qi] = sc([qi])[0]
            tr = sc(train_qs)
            for k, qi in enumerate(train_qs):
                train_rr.append(_rr(tr[k], rel[qi]))
        te_all.append(per_query_metrics(oof, ds).mrr)
        tr_all.append(mean(train_rr))
    tr_after, te_after = mean(tr_all), mean(te_all)

    # BEFORE numbers (from Phase 1)
    tr_before, te_before = 0.1844, 0.1498
    loss_before = (0.6931, 0.6378)
    loss_after = (curve[0] if curve else float("nan"), curve[-1] if curve else res0.fun)

    with open(RESULTS / "a5_bugfix_report.md", "w", encoding="utf-8") as f:
        f.write("# Fase 2.0 -- A5 bilinear bug fix\n\n")
        f.write("**Diagnosis:** A5 already used balanced sample weights "
                "(`w_pos=|Y|/2n_pos`), so the imbalance hypothesis is not the cause. "
                "The defect was the hand-written fixed-step gradient descent, which did not "
                "converge. Fix: replace GD with **L-BFGS-B** (analytic gradient, convergence "
                "test, maxiter 5000), keeping balanced weights and rank 8, l2 1e-3.\n\n")
        f.write("## Loss (full training set, seed 0)\n\n")
        f.write(f"- BEFORE (GD, 300 iters): {loss_before[0]:.4f} -> {loss_before[1]:.4f}\n")
        f.write(f"- AFTER (L-BFGS-B, {len(curve)} iters, converged={res0.success}): "
                f"{loss_after[0]:.4f} -> {loss_after[1]:.4f}\n")
        f.write("- full curve in `a5_loss_curve_lbfgs.csv`.\n\n")
        f.write("## Train / test MRR (5 seeds)\n\n")
        f.write("| | mrr_train | mrr_test |\n|---|---|---|\n")
        f.write(f"| BEFORE (GD) | {tr_before:.4f} | {te_before:.4f} |\n")
        f.write(f"| AFTER (L-BFGS-B) | {tr_after:.4f} | {te_after:.4f} |\n\n")
        verdict = ("A5 train MRR rose well above 0.184 -> the defect WAS the optimizer, "
                   "not imbalance." if tr_after > 0.30 else
                   "A5 train MRR did NOT rise much above 0.184 -> the problem is not just the "
                   "optimizer; the low-rank bilinear model may lack capacity for this signal.")
        f.write(f"**Verdict:** {verdict}\n")

    print(f"A5 BEFORE train/test = {tr_before:.4f}/{te_before:.4f}")
    print(f"A5 AFTER  train/test = {tr_after:.4f}/{te_after:.4f}  "
          f"(loss {loss_after[0]:.4f}->{loss_after[1]:.4f}, converged={res0.success})")
    print(f"wrote {RESULTS}/a5_bugfix_report.md, a5_loss_curve_lbfgs.csv")


if __name__ == "__main__":
    main()
