"""Full-pool dupe-retrieval evaluation: metrics, significance, bootstrap.

Protocol
--------
* Query = a global perfume with >=1 labeled local dupe. The query is never itself in
  the pool (it is a global perfume; the pool is the local catalog), so nothing is
  "left out" -- this is full-pool retrieval, not leave-one-out.
* Candidate pool = all 340 products (incl. the 97 unlabeled distractors).
* For each query, rank the whole pool by a method's score; relevance is the held-out
  inspired-by edge. With multiple dupes we take the BEST (min) rank.

Ranking with ties (IMPORTANT): scores produce ties, and a naive "1 + #strictly
greater" rule gives tie-heavy methods (e.g. un-weighted Jaccard, which leaves ~16
candidates tied at the top) a free optimistic boost. We therefore use the EXACT
expectation under uniform random tie-breaking:

    let g = #products scoring strictly higher than the best relevant product,
        e = #products tied at exactly that score (>=1, includes the relevant one).
    The relevant item is equally likely to land at any rank in [g+1, g+e], so
        E[RR]    = mean(1/r  for r in g+1..g+e)
        E[Hit@k] = clip(k - g, 0, e) / e

This is unbiased w.r.t. ties and reduces to the plain rank when e == 1.

Metrics: MRR, Hits@1, Hits@3 (one relevant concept per query).
Significance: Wilcoxon signed-rank PAIRED ACROSS QUERIES on reciprocal rank
(proposed vs each baseline). Bootstrap 95% CI over queries for delta-MRR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .data import Dataset


def accord_containment_report(ds: Dataset, verbose: bool = True) -> Dict[str, float]:
    """Report the mean containment of a product's accords in its true query's accord list.
    The Step-1 data audit (results/audit/step1_report.md) measured this at 0.86 for true
    pairs vs 0.35 for random query-product pairs (only 1.8% fully contained at random),
    so the high overlap is a real, pair-specific signal, not a vocabulary artifact.
    IDF weighting (B2) and tie-correct ranking are used for standard IR reasons -- rarity
    weighting and the fact that un-weighted overlap leaves many candidates tied at the top."""
    cover = []
    for q in ds.queries:
        qa = set(q.accords)
        for pi in q.relevant:
            pa = set(ds.products[pi].accords)
            if pa:
                cover.append(len(qa & pa) / len(pa))
    rep = {"mean_product_accords_covered_by_global": float(np.mean(cover))}
    if verbose:
        print(f"[diagnostic] accord containment: on average "
              f"{rep['mean_product_accords_covered_by_global']:.0%} of a dupe's accords "
              f"are contained in its true query's accord list (vs ~35% for random pairs; "
              f"a real pair-specific signal -- see results/audit/step1_report.md).")
    return rep


@dataclass
class QueryResult:
    rr: np.ndarray      # reciprocal rank per query
    h1: np.ndarray      # hit@1 per query (0/1)
    h3: np.ndarray      # hit@3 per query (0/1)

    @property
    def mrr(self) -> float:
        return float(self.rr.mean())

    @property
    def hits1(self) -> float:
        return float(self.h1.mean())

    @property
    def hits3(self) -> float:
        return float(self.h3.mean())


def per_query_metrics(scores: np.ndarray, ds: Dataset) -> QueryResult:
    """scores: (n_q, n_pool). Returns per-query RR / H@1 / H@3."""
    n_q = len(ds.queries)
    rr = np.zeros(n_q)
    h1 = np.zeros(n_q)
    h3 = np.zeros(n_q)
    for q in ds.queries:
        s = scores[q.idx]
        rel = list(q.relevant)
        best = np.max(s[rel])                       # best-scoring relevant product
        g = int(np.sum(s > best))                   # strictly greater
        e = int(np.sum(s == best))                  # tied at best (>=1)
        ranks = np.arange(g + 1, g + e + 1)
        rr[q.idx] = float(np.mean(1.0 / ranks))     # E[RR] under uniform tie break
        h1[q.idx] = float(np.clip(1 - g, 0, e) / e)  # E[Hit@1]
        h3[q.idx] = float(np.clip(3 - g, 0, e) / e)  # E[Hit@3]
    return QueryResult(rr=rr, h1=h1, h3=h3)


def wilcoxon_rr(rr_a: np.ndarray, rr_b: np.ndarray) -> Optional[float]:
    """Paired Wilcoxon signed-rank p-value on reciprocal rank (a vs b)."""
    from scipy.stats import wilcoxon
    diff = rr_a - rr_b
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(wilcoxon(rr_a, rr_b, zero_method="wilcox").pvalue)
    except ValueError:
        return None


def bootstrap_delta_mrr(rr_a: np.ndarray, rr_b: np.ndarray,
                        n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Bootstrap 95% CI over queries for delta-MRR = MRR(a) - MRR(b)."""
    rng = np.random.default_rng(seed)
    n = len(rr_a)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = rr_a[idx].mean() - rr_b[idx].mean()
    return {
        "delta_mrr": float(rr_a.mean() - rr_b.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


@dataclass
class MethodReport:
    name: str
    tier: str
    stochastic: bool
    skipped: bool
    mrr: float
    h1: float
    h3: float
    mrr_std: float
    h1_std: float
    h3_std: float
    seed0: Optional[QueryResult]   # per-query result at seed 0 (for significance)


def evaluate_method(method, ds: Dataset, feats, seeds: List[int]) -> MethodReport:
    """Run a method (1 seed if deterministic, several if stochastic) and aggregate."""
    use_seeds = seeds if method.stochastic else [seeds[0]]
    mrrs, h1s, h3s = [], [], []
    seed0_result = None
    skipped = False
    for s in use_seeds:
        scores = method.scores(ds, feats, seed=s)
        if np.all(np.isnan(scores)):
            skipped = True
            break
        res = per_query_metrics(scores, ds)
        if seed0_result is None:
            seed0_result = res
        mrrs.append(res.mrr)
        h1s.append(res.hits1)
        h3s.append(res.hits3)

    if skipped:
        return MethodReport(method.name, method.tier, method.stochastic, True,
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None)

    return MethodReport(
        name=method.name, tier=method.tier, stochastic=method.stochastic, skipped=False,
        mrr=float(np.mean(mrrs)), h1=float(np.mean(h1s)), h3=float(np.mean(h3s)),
        mrr_std=float(np.std(mrrs)), h1_std=float(np.std(h1s)), h3_std=float(np.std(h3s)),
        seed0=seed0_result,
    )
