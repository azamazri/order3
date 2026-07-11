"""Tahap A.3 — reproduksi baseline v3 (3 teratas) pada split.test.

Ini angka yang harus dikalahkan (BUKAN angka results/v3 yang di seluruh 208 query).

Output: results/v4/A_baseline_on_test.csv (method, MRR, Hits@1, Hits@3, per-stratum MRR)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_dataset
from ..methods.base import build_features
from ..methods import P1Order2, P2Fusion, A3Signature
from . import _shared as sh

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v4"
V3_DIR = Path(__file__).resolve().parents[2] / "results" / "v3"
SEEDS = [0, 1, 2, 3, 4]
STRATA = ["prefix", "subseq", "contained", "partial"]


def main():
    ds = load_dataset()
    feats = build_features(ds)
    split = json.loads((OUT_DIR / "split.json").read_text())
    test = np.array(split["test"])
    strata = pd.read_csv(V3_DIR / "02_strata.csv").set_index("q_idx")
    q_stratum = np.array([strata.loc[q.idx, "stratum"] for q in ds.queries])

    methods = [P1Order2(), P2Fusion(C=2.0), A3Signature(C=0.5)]
    rows = []
    for m in methods:
        res = sh.per_query(m, ds, feats, SEEDS)
        rr, h1, h3 = res["rr"], res["h1"], res["h3"]
        tmask = np.zeros(len(ds.queries), bool); tmask[test] = True
        row = {"method": m.name, "eval": "test",
               "MRR": f"{rr[tmask].mean():.4f}", "Hits@1": f"{h1[tmask].mean():.4f}",
               "Hits@3": f"{h3[tmask].mean():.4f}", "n_test": int(tmask.sum())}
        for s in STRATA:
            sm = tmask & (q_stratum == s)
            row[f"MRR_{s}"] = f"{rr[sm].mean():.4f}" if sm.any() else ""
            row[f"n_{s}"] = int(sm.sum())
        rows.append(row)
        print(f"  {m.name}: test MRR={row['MRR']} "
              + " ".join(f"{s}={row['MRR_'+s]}" for s in STRATA))

    cols = (["method", "eval", "MRR", "Hits@1", "Hits@3", "n_test"]
            + [f"MRR_{s}" for s in STRATA] + [f"n_{s}" for s in STRATA])
    with open(OUT_DIR / "A_baseline_on_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_DIR/'A_baseline_on_test.csv'}")


if __name__ == "__main__":
    main()
