"""Tahap 6 — order ladder N=1..6, sparsity, nested-CV N*, ladder significance.

Metrik dengan protokol default (tie=pesimistis, multi=best). Dijalankan pada ALL dan
NON_OP (mask dari 02_strata.csv). N=1 diverifikasi == B2 (Gate G5, di order_n test).

Output: 03_order_ladder.csv, 03_sparsity.csv, 03_order_significance.csv,
        03_order_selection.md
"""
from __future__ import annotations

import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_dataset
from . import protocol as P
from .order_n import OrderN, build_order_features

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v3"
N_MAX = 6
CV_SEEDS = [0, 1, 2]
N_SPLITS = 5


def _rr_h_per_query(scores, ds, tie="pessimistic", multi="best"):
    r = P.eval_metrics(scores, ds, tie=tie, multi=multi)
    # multi=best => one unit per query, aligned to ds.queries order
    return r.rr, r.h1, r.h3, r.h10


def _folds(idx, n_splits, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(idx)
    return [np.sort(f) for f in np.array_split(order, min(n_splits, len(idx)))]


def nested_cv(rr_byN, query_idx):
    """Pick N per fold by max mean-rr on train queries; return per-fold N, mode, OOF MRR."""
    picks = []
    oof = np.zeros(len(query_idx))
    pos = {q: i for i, q in enumerate(query_idx)}
    for seed in CV_SEEDS:
        folds = _folds(np.array(query_idx), N_SPLITS, seed)
        for f, test_q in enumerate(folds):
            train_q = np.concatenate([folds[j] for j in range(len(folds)) if j != f])
            best_N, best_m = None, -1
            for N in range(1, N_MAX + 1):
                m = rr_byN[N][train_q].mean()
                if m > best_m:
                    best_m, best_N = m, N
            picks.append(best_N)
            for q in test_q:
                oof[pos[q]] = rr_byN[best_N][q]
    mode_N = Counter(picks).most_common(1)[0][0]
    return picks, mode_N, float(oof.mean())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    strata = pd.read_csv(OUT_DIR / "02_strata.csv")
    nonop_q = strata.loc[strata["is_NON_OP"] == 1, "q_idx"].tolist()
    all_q = list(range(len(ds.queries)))

    # ---- scores + per-query rr/h for N=1..6 ----
    rr_byN, h1_byN, h3_byN, h10_byN, metas = {}, {}, {}, {}, {}
    for N in range(1, N_MAX + 1):
        Pmat, Qmat, meta = build_order_features(ds, N)
        sc = np.asarray((Qmat @ Pmat.T).todense())
        rr, h1, h3, h10 = _rr_h_per_query(sc, ds)
        rr_byN[N], h1_byN[N], h3_byN[N], h10_byN[N] = rr, h1, h3, h10
        metas[N] = meta

    # ---- 03_order_ladder.csv ----
    def m(arr, idx):
        return float(np.mean(arr[idx]))
    with open(OUT_DIR / "03_order_ladder.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eval_set", "N", "MRR", "Hits@1", "Hits@3", "Hits@10", "n_queries"])
        for label, idx in (("ALL", all_q), ("NON_OP", nonop_q)):
            for N in range(1, N_MAX + 1):
                w.writerow([label, N, f"{m(rr_byN[N],idx):.4f}", f"{m(h1_byN[N],idx):.4f}",
                            f"{m(h3_byN[N],idx):.4f}", f"{m(h10_byN[N],idx):.4f}", len(idx)])

    # ---- 03_sparsity.csv: avg #shared subsets per order over labeled pairs ----
    with open(OUT_DIR / "03_sparsity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_k", "avg_shared_subsets_true_pairs", "n_pairs"])
        pairs = [(pi, q.idx) for q in ds.queries for pi in q.relevant]
        for k in range(1, N_MAX + 1):
            tot = 0
            for pi, qi in pairs:
                S = set(ds.products[pi].accords) & set(ds.queries[qi].accords)
                if len(S) >= k:
                    tot += sum(1 for _ in combinations(sorted(S), k))
            w.writerow([k, f"{tot/len(pairs):.3f}", len(pairs)])

    # ---- ladder significance (consecutive), Holm within family of 5, per eval set ----
    sig_rows = []
    selection = {}
    for label, idx in (("ALL", all_q), ("NON_OP", nonop_q)):
        idx = np.array(idx)
        raw = []
        recs = []
        for N in range(1, N_MAX):
            a, b = rr_byN[N + 1][idx], rr_byN[N][idx]
            p_raw, nnz = P.wilcoxon_test(a, b)
            bs = P.bootstrap_delta(a, b)
            raw.append(p_raw if p_raw is not None else 1.0)
            recs.append((f"order{N+1}-order{N}", bs, nnz, p_raw))
        p_holm = P.holm(raw)
        for (comp, bs, nnz, p_raw), padj in zip(recs, p_holm):
            vd = P.verdict(bs["delta_mrr"], bs["ci_low"], bs["ci_high"], padj)
            sig_rows.append([comp, label, f"{bs['delta_mrr']:.4f}", f"{bs['ci_low']:.4f}",
                             f"{bs['ci_high']:.4f}", f"{p_raw:.3e}" if p_raw is not None else "",
                             f"{padj:.3e}", nnz, vd])
        # §7.2 N*: smallest N s.t. every M>N step not SIGNIFIKAN on this eval set
        step_sig = {}  # N -> is step N->N+1 significant
        for (comp, bs, nnz, p_raw), padj in zip(recs, p_holm):
            Nlow = int(comp.split("-")[1].replace("order", ""))
            step_sig[Nlow] = P.verdict(bs["delta_mrr"], bs["ci_low"], bs["ci_high"], padj) == "SIGNIFIKAN"
        Nstar = N_MAX
        for N in range(1, N_MAX + 1):
            if all(not step_sig.get(M, False) for M in range(N, N_MAX)):
                Nstar = N
                break
        # nested CV
        picks, mode_N, oof = nested_cv(rr_byN, list(idx))
        selection[label] = {"Nstar_rule": Nstar, "cv_mode": mode_N, "cv_oof_mrr": oof,
                            "cv_picks": Counter(picks)}

    with open(OUT_DIR / "03_order_significance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "eval_set", "delta_MRR", "ci_low", "ci_high",
                    "wilcoxon_p_raw", "p_adj_holm", "n_nonzero", "verdict"])
        w.writerows(sig_rows)

    # ---- 03_order_selection.md ----
    L = ["# 03 — Pemilihan N* (order ladder + nested CV)\n"]
    L.append("Metrik: tie=pesimistis, multi=best. IDF dari 340 produk (transduktif — "
             "IDF melihat seluruh pool termasuk produk di fold uji; konsisten semua metode).\n")
    ladder = pd.read_csv(OUT_DIR / "03_order_ladder.csv")
    for label in ("ALL", "NON_OP"):
        s = selection[label]
        L.append(f"## Eval set: {label}\n")
        sub = ladder[ladder["eval_set"] == label]
        L.append("| N | MRR | H@1 | H@3 | H@10 |")
        L.append("|---|---|---|---|---|")
        for _, r in sub.iterrows():
            L.append(f"| {int(r['N'])} | {r['MRR']:.4f} | {r['Hits@1']:.4f} | "
                     f"{r['Hits@3']:.4f} | {r['Hits@10']:.4f} |")
        L.append("")
        L.append(f"- **N\\* (aturan §7.2, N terkecil yang setiap step M>N tidak signifikan)** "
                 f"= **{s['Nstar_rule']}**")
        L.append(f"- **Nested-CV mode** = **{s['cv_mode']}** (OOF MRR = {s['cv_oof_mrr']:.4f}); "
                 f"picks = {dict(s['cv_picks'])}")
        agree = "SAMA" if s["Nstar_rule"] == s["cv_mode"] else "BERBEDA"
        L.append(f"- aturan §7.2 vs nested-CV mode: **{agree}**")
        L.append("")
    prim = selection["NON_OP"]
    L.append("## Ringkasan\n")
    L.append(f"- **Eval set primer = NON_OP** (verdict Tahap 5 = DERIVATIF sebagian).")
    L.append(f"- N\\* aturan (NON_OP) = {prim['Nstar_rule']}; nested-CV mode (NON_OP) = "
             f"{prim['cv_mode']}.")
    if prim["Nstar_rule"] != prim["cv_mode"]:
        L.append("- **Keduanya berbeda → dilaporkan keduanya, tidak memilih yang menguntungkan** (§7.2).")
    L.append("- Signifikansi tangga: lihat `03_order_significance.csv` (Holm, per eval set).")
    (OUT_DIR / "03_order_selection.md").write_text("\n".join(L), encoding="utf-8")

    print("Order ladder done.")
    for label in ("ALL", "NON_OP"):
        s = selection[label]
        print(f"  {label}: MRR(N=1..6)=", [round(m(rr_byN[N], np.array(all_q if label=='ALL' else nonop_q)),4) for N in range(1,7)])
        print(f"    N*_rule={s['Nstar_rule']} cv_mode={s['cv_mode']} oof={s['cv_oof_mrr']:.4f} picks={dict(s['cv_picks'])}")


if __name__ == "__main__":
    main()
