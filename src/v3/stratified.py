"""Tahap 5.5 (deliverable 06_stratified.csv) — MRR per stratum containment.

Stratum query dari 02_strata.csv (prefix > subseq > contained > partial). MRR per stratum
untuk metode representatif (order-2 usulan + pembanding kunci). tie=pesimistis, multi=best.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_dataset
from ..methods.base import build_features
from ..methods import B2TfidfCosine, B4SBert, A2PpmiSvd, A3Signature, P1Order2, P2Fusion
from . import protocol as P

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v3"
SEEDS = [0, 1, 2, 3, 4]
STRATA = ["prefix", "subseq", "contained", "partial"]


def main():
    ds = load_dataset()
    feats = build_features(ds)
    strata = pd.read_csv(OUT_DIR / "02_strata.csv").set_index("q_idx")
    q_stratum = np.array([strata.loc[q.idx, "stratum"] for q in ds.queries])
    nonop = np.array([bool(strata.loc[q.idx, "is_NON_OP"]) for q in ds.queries])

    methods = [P1Order2(), P2Fusion(C=2.0), A3Signature(C=0.5), A2PpmiSvd(dim=50),
               B2TfidfCosine(), B4SBert()]
    rows = []
    for m in methods:
        r = P.run_method(m, ds, feats, seeds=SEEDS, tie="pessimistic", multi="best")
        if r.get("skipped"):
            continue
        rr = r["rr_units"]  # per-query, aligned to ds.queries
        row = {"method": m.name}
        for s in STRATA:
            mask = q_stratum == s
            row[s] = f"{rr[mask].mean():.4f}" if mask.any() else ""
            row[f"n_{s}"] = int(mask.sum())
        row["ALL"] = f"{rr.mean():.4f}"
        row["NON_OP"] = f"{rr[nonop].mean():.4f}"
        rows.append(row)
        print(f"  {m.name}: " + " ".join(f"{s}={row[s]}" for s in STRATA)
              + f" ALL={row['ALL']} NON_OP={row['NON_OP']}")

    with open(OUT_DIR / "06_stratified.csv", "w", newline="", encoding="utf-8") as f:
        cols = (["method"] + STRATA + [f"n_{s}" for s in STRATA] + ["ALL", "NON_OP"])
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_DIR/'06_stratified.csv'}  (n per stratum: "
          + ", ".join(f"{s}={int((q_stratum==s).sum())}" for s in STRATA) + ")")


if __name__ == "__main__":
    main()
