"""Tahap 9 — grid sensitivitas (yang membuat hasil bisa dipercaya).

Minimum wajib: S1 x S2 x S3 (keputusan cleaning) x S4 (IDF) x S9 (tie) x S11 (eval).
Setiap sel: order ladder N=1..6, N* (aturan §7.2), rq1_significant.
Plus one-at-a-time: S8 (multi-dupe) dan S12 (wheel A1 map vs drop).

Varian cleaning dibuat DI MEMORI dari Excel bersih (transformasi eksplisit yang
dideklarasikan, bukan cleaning di kode). Sumber perturbasi = cleaning_changelog.csv:
  S1 'oriental': drop (default, data apa adanya) vs map->amber pada 6 produk yang
     accord 'oriental'-nya di-drop (Vicaro, Devine, Pixie Petals, Rose of The Vent,
     Gempita, Black Mirage).
  S2 'warm': ->'warm spicy' (default, Blue Hills) vs drop.
  S3 'white floral and tuberose': white floral+tuberose (default, Blooming) vs floral+tuberose.

Output: 07_sensitivity.csv + ringkasan di 07_sensitivity_summary.md
"""
from __future__ import annotations

import copy
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from ..data import load_dataset
from ..methods.base import build_features
from ..methods.a1_wheel import A1Wheel
from . import protocol as P
from .order_n import build_order_features

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v3"
N_MAX = 6
N_BOOT = 5000

S1_MAP = {"vicaro", "devine", "pixie petals", "rose of the vent", "gempita", "black mirage"}
S2_PROD = "blue hills"
S3_PROD = "blooming"


def variant(ds, s1, s2, s3):
    d = copy.deepcopy(ds)
    for p in d.products:
        nm = p.name.strip().lower()
        if s1 == "map" and nm in S1_MAP and "amber" not in p.accords:
            p.accords = p.accords + ["amber"]
        if s2 == "drop" and nm == S2_PROD and "warm spicy" in p.accords:
            p.accords = [a for a in p.accords if a != "warm spicy"]
        if s3 == "floral" and nm == S3_PROD:
            p.accords = ["floral" if a == "white floral" else a for a in p.accords]
    return d


def rr_by_order(d, idf_pool, tie):
    """per-query rr for N=1..6 on dataset d, given idf_pool and tie policy."""
    out = {}
    for N in range(1, N_MAX + 1):
        Pmat, Qmat, _ = build_order_features(d, N, idf_pool=idf_pool)
        sc = np.asarray((Qmat @ Pmat.T).todense())
        out[N] = P.eval_metrics(sc, d, tie=tie, multi="best").rr
    return out


def nstar_and_rq1(rr, idx):
    """Return (N*, rq1_significant, mrr_by_order) on eval subset idx."""
    idx = np.array(idx)
    mrr = {N: float(rr[N][idx].mean()) for N in range(1, N_MAX + 1)}
    raw, recs = [], []
    for N in range(1, N_MAX):
        a, b = rr[N + 1][idx], rr[N][idx]
        p_raw, _ = P.wilcoxon_test(a, b)
        bs = P.bootstrap_delta(a, b, n_boot=N_BOOT)
        raw.append(p_raw if p_raw is not None else 1.0)
        recs.append((N, bs))
    padj = P.holm(raw)
    step_sig = {}
    for (N, bs), pa in zip(recs, padj):
        step_sig[N] = P.verdict(bs["delta_mrr"], bs["ci_low"], bs["ci_high"], pa) == "SIGNIFIKAN"
    Nstar = N_MAX
    for N in range(1, N_MAX + 1):
        if all(not step_sig.get(M, False) for M in range(N, N_MAX)):
            Nstar = N
            break
    return Nstar, step_sig.get(1, False), mrr


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    ds = load_dataset()
    strata = pd.read_csv(OUT_DIR / "02_strata.csv")
    nonop_q = set(strata.loc[strata["is_NON_OP"] == 1, "q_idx"])
    all_idx = list(range(len(ds.queries)))
    nonop_idx = [q.idx for q in ds.queries if q.idx in nonop_q]
    evalsets = {"ALL": all_idx, "NON_OP": nonop_idx}

    rows = []
    cell = 0
    nstars, rq1_flags = [], []
    for s1 in ("drop", "map"):
        for s2 in ("warmspicy", "drop"):
            for s3 in ("wf", "floral"):
                d = variant(ds, s1, s2, s3)
                for s4 in ("products", "products_query"):
                    for s9 in ("pessimistic", "average", "optimistic"):
                        rr = rr_by_order(d, s4, s9)
                        for s11 in ("ALL", "NON_OP"):
                            Nstar, rq1, mrr = nstar_and_rq1(rr, evalsets[s11])
                            cell += 1
                            nstars.append(Nstar)
                            rq1_flags.append(rq1)
                            rows.append([f"C{cell}", s1, s2, s3, s4, s9, s11, Nstar,
                                         *[f"{mrr[N]:.4f}" for N in range(1, N_MAX + 1)],
                                         int(rq1)])
                print(f"  done S1={s1} S2={s2} S3={s3}")

    header = (["cell_id", "S1_oriental", "S2_warm", "S3_wftub", "S4_idf", "S9_tie",
               "S11_eval", "N_star"] + [f"MRR_order{N}" for N in range(1, N_MAX + 1)]
              + ["rq1_significant"])
    with open(OUT_DIR / "07_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # ---- S8 multi-dupe (one-at-a-time, default cleaning/idf, tie pessimistic) ----
    s8_rows = []
    for multi in ("best", "separate", "first"):
        for N in (1, 2):
            Pmat, Qmat, _ = build_order_features(ds, N)
            sc = np.asarray((Qmat @ Pmat.T).todense())
            r = P.eval_metrics(sc, ds, tie="pessimistic", multi=multi)
            s8_rows.append(["S8", multi, f"order{N}", f"{r.mrr:.4f}", r.rr.shape[0]])

    # ---- S12 wheel A1 map vs drop ----
    feats = build_features(ds)
    nonop_mask = np.array([q.idx in nonop_q for q in ds.queries])
    s12_rows = []
    for mode in ("map", "drop"):
        sc = A1Wheel(mode=mode).scores(ds, feats)
        a = P.eval_metrics(sc, ds, tie="pessimistic", multi="best")
        s12_rows.append(["S12", mode, "A1_wheel", f"{a.rr.mean():.4f}",
                         f"{a.rr[nonop_mask].mean():.4f}"])

    # ---- summary ----
    nstar_counts = Counter(nstars)
    mode_nstar = nstar_counts.most_common(1)[0][0]
    pct_same = 100.0 * nstar_counts[mode_nstar] / len(nstars)
    pct_rq1 = 100.0 * sum(rq1_flags) / len(rq1_flags)
    L = ["# 07 — Ringkasan Grid Sensitivitas\n"]
    L.append(f"Total sel (S1×S2×S3×S4×S9×S11) = {len(rows)}\n")
    L.append(f"- **N\\* modus = {mode_nstar}**, muncul di **{pct_same:.1f}%** sel "
             f"(distribusi N\\*: {dict(nstar_counts)})")
    L.append(f"- **RQ1 (order2>order1) signifikan di {pct_rq1:.1f}%** sel")
    L.append(f"- Aturan stabilitas §7.3: N\\* {'STABIL' if pct_same >= 80 else 'TIDAK STABIL'} "
             f"(ambang 80%).")
    L.append(f"- Aturan RQ1 §7.4 (butuh ≥80% sel): "
             f"{'TERPENUHI' if pct_rq1 >= 80 else 'TIDAK TERPENUHI'} pada dimensi grid ini.")
    L.append("")
    L.append("## S8 — query multi-dupe (one-at-a-time)\n")
    L.append("| multi | order | MRR | n_units |")
    L.append("|---|---|---|---|")
    for r in s8_rows:
        L.append(f"| {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    L.append("")
    L.append("## S12 — wheel A1 (map vs drop)\n")
    L.append("| mode | MRR ALL | MRR NON_OP |")
    L.append("|---|---|---|")
    for r in s12_rows:
        L.append(f"| {r[1]} | {r[3]} | {r[4]} |")
    L.append("")
    L.append("## Catatan cakupan\n")
    L.append("- Dijalankan penuh: S1,S2,S3,S4,S9,S11 (minimum wajib) + S8, S12 one-at-a-time.")
    L.append("- Tidak dijalankan (opsional/mahal): S5 (norm), S6 (bobot), S7 (pool), S10 (fold).")
    (OUT_DIR / "07_sensitivity_summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\nSensitivity: {len(rows)} cells. N*_mode={mode_nstar} in {pct_same:.1f}%; "
          f"RQ1 sig in {pct_rq1:.1f}%.")
    print(f"S8: {[(r[1],r[2],r[3]) for r in s8_rows]}")
    print(f"S12: {[(r[1],r[3],r[4]) for r in s12_rows]}")


if __name__ == "__main__":
    main()
