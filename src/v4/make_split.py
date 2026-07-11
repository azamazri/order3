"""Tahap A.2 — kunci split dev/test SEKALI.

Test = 40% query, Dev = 60%, split acak seed 20260711, distratifikasi menurut stratum
containment (prefix/subseq/contained/partial) supaya proporsi `partial` di dev ≈ test.

Acceptance: test punya >=12 query `partial`, dev punya >=12. Kalau tidak → Gate GV1.

Output: results/v4/split.json  {"dev": [...], "test": [...], "meta": {...}}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v4"
V3_DIR = Path(__file__).resolve().parents[2] / "results" / "v3"
SEED = 20260711
TEST_FRAC = 0.40
STRATA = ["prefix", "subseq", "contained", "partial"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    strata = pd.read_csv(V3_DIR / "02_strata.csv")
    q_stratum = {int(r.q_idx): r.stratum for r in strata.itertuples()}
    assert len(q_stratum) == len(ds.queries), "strata rows != n_queries"

    rng = np.random.default_rng(SEED)
    dev, test = [], []
    per_stratum = {}
    for s in STRATA:
        qs = sorted([q for q, st in q_stratum.items() if st == s])
        rng.shuffle(qs)
        n_test = int(round(TEST_FRAC * len(qs)))
        test_s = sorted(qs[:n_test])
        dev_s = sorted(qs[n_test:])
        test += test_s
        dev += dev_s
        per_stratum[s] = {"total": len(qs), "dev": len(dev_s), "test": len(test_s)}

    dev, test = sorted(dev), sorted(test)
    assert set(dev).isdisjoint(test)
    assert len(dev) + len(test) == len(ds.queries)

    n_part_dev = per_stratum["partial"]["dev"]
    n_part_test = per_stratum["partial"]["test"]

    payload = {"seed": SEED, "test_frac": TEST_FRAC, "dev": dev, "test": test,
               "meta": {"n_dev": len(dev), "n_test": len(test), "per_stratum": per_stratum}}
    (OUT_DIR / "split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"dev={len(dev)} test={len(test)}")
    for s in STRATA:
        print(f"  {s:<10} total={per_stratum[s]['total']:>3} "
              f"dev={per_stratum[s]['dev']:>3} test={per_stratum[s]['test']:>3}")
    print(f"partial: dev={n_part_dev} test={n_part_test}")

    if n_part_dev < 12 or n_part_test < 12:
        (OUT_DIR / "GATE_GV1.md").write_text(
            f"# GATE GV1 — stratum `partial` tak bisa dibelah\n\n"
            f"partial dev={n_part_dev}, test={n_part_test} (butuh >=12 tiap sisi).\n"
            f"Usul: k-fold cross-val alih-alih single split.\n", encoding="utf-8")
        print(f"\n*** GATE GV1: partial dev={n_part_dev} test={n_part_test} < 12 -> stop")
        return False
    print("Acceptance A.2 LULUS (partial >=12 tiap sisi).")
    return True


if __name__ == "__main__":
    main()
