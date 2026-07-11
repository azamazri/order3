"""Helper v4: per-query metrics (aligned to ds.queries), averaged across seeds.

Memakai protokol v3 (tie=pesimistis, multi=best) supaya konsisten dengan baseline.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..v3 import protocol as P


def per_query(method, ds, feats, seeds: List[int]):
    """Return {rr,h1,h3,h10}: arrays (n_q,) averaged across seeds. multi=best =>
    one unit per query, aligned to ds.queries order."""
    use = seeds if getattr(method, "stochastic", False) else [seeds[0]]
    rr, h1, h3, h10 = [], [], [], []
    for s in use:
        sc = method.scores(ds, feats, seed=s)
        r = P.eval_metrics(sc, ds, tie="pessimistic", multi="best")
        rr.append(r.rr); h1.append(r.h1); h3.append(r.h3); h10.append(r.h10)
    return {"rr": np.mean(rr, axis=0), "h1": np.mean(h1, axis=0),
            "h3": np.mean(h3, axis=0), "h10": np.mean(h10, axis=0)}


def per_query_from_scores(scores, ds):
    r = P.eval_metrics(scores, ds, tie="pessimistic", multi="best")
    return {"rr": r.rr, "h1": r.h1, "h3": r.h3, "h10": r.h10}
