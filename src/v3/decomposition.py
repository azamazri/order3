"""Tahap 10 — dekomposisi skor order-N + null pasangan acak.

share_k = Σ_{|t|=k} v_q[t]·v_p[t] / Σ_t v_q[t]·v_p[t], untuk tiga populasi:
  true   : pasangan (q, gold)
  top1   : pasangan (q, produk peringkat 1)
  random : (q, produk acak non-gold), 20x per query, seed 0

Kalau `true` dan `random` mirip, statistik tidak informatif dan tidak boleh masuk paper.
Memakai representasi metode usulan = order-N dengan N*=2 (Tahap 6).

Output: 08_decomposition.csv (population, order, share, n_pairs)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..data import load_dataset
from .order_n import OrderN

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v3"
N_STAR = 2
N_RANDOM = 20
SEED = 0
EPS = 1e-12


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    rng = np.random.default_rng(SEED)

    full, per, _ = OrderN(N_STAR).scores_and_decompose(ds)  # per: {k: matrix}
    orders = sorted(per.keys())

    pops = {"true": [], "top1": [], "random": []}   # each: list of share-vectors [k...]
    all_pidx = np.arange(ds.n_pool)
    for q in ds.queries:
        qi = q.idx
        # true
        for pi in q.relevant:
            tot = full[qi, pi]
            if tot > EPS:
                pops["true"].append([per[k][qi, pi] / tot for k in orders])
        # top1
        pi_top = int(np.argmax(full[qi]))
        tot = full[qi, pi_top]
        if tot > EPS:
            pops["top1"].append([per[k][qi, pi_top] / tot for k in orders])
        # random non-gold
        choices = np.setdiff1d(all_pidx, list(q.relevant), assume_unique=False)
        pick = rng.choice(choices, size=min(N_RANDOM, len(choices)), replace=False)
        for pi in pick:
            tot = full[qi, pi]
            if tot > EPS:
                pops["random"].append([per[k][qi, int(pi)] / tot for k in orders])

    rows = []
    for pop, vals in pops.items():
        arr = np.array(vals) if vals else np.zeros((0, len(orders)))
        for j, k in enumerate(orders):
            share = float(arr[:, j].mean()) if len(arr) else float("nan")
            rows.append([pop, k, f"{share:.4f}", len(arr)])

    with open(OUT_DIR / "08_decomposition.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["population", "order", "share", "n_pairs"])
        w.writerows(rows)

    print("Decomposition (share by order):")
    for pop in ("true", "top1", "random"):
        s = {r[1]: r[2] for r in rows if r[0] == pop}
        print(f"  {pop:<7} {s}")
    # informativeness check
    tr = {r[1]: float(r[2]) for r in rows if r[0] == "true"}
    rd = {r[1]: float(r[2]) for r in rows if r[0] == "random"}
    diff = {k: round(tr[k] - rd[k], 4) for k in orders}
    print(f"  true - random by order: {diff}")


if __name__ == "__main__":
    main()
