"""Tahap 4 — protokol evaluasi v3 (dikunci; jangan diubah setelah lihat hasil).

Task: untuk tiap query (parfum global) rangking seluruh 340 produk; ground truth =
produk yang revolutionize-nya menunjuk query itu.

Metrik: MRR, Hits@1, Hits@3, Hits@10.

Query >1 dupe (§4.3): default **best rank** di antara produk relevan. Alternatif:
  'separate' (tiap pasangan jadi unit), 'first' (dupe pertama by idx).

Tie-break (§4.4): default **pesimistis** (produk relevan di belakang semua yang seri).
  Alternatif: 'optimistic', 'average' (ekspektasi uniform).

Signifikansi (§4.6-4.7):
  - rr dikumpulkan per query per seed lalu **dirata-ratakan lintas seed**, baru diuji.
  - Wilcoxon (zero_method='wilcox') + n_nonzero, DAN bootstrap 10k paired (CI 95%).
  - Koreksi: Holm (keluarga ladder) & Benjamini-Hochberg FDR 0.05 (proposed vs pembanding).
  - Verdict SIGNIFIKAN hanya jika: p_adj<0.05 AND CI tak memuat 0 AND |dMRR|>=0.01.
    Kalau sebagian: AMBIGU + alasan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data import Dataset

TIE_POLICIES = ("pessimistic", "optimistic", "average")
MULTI_POLICIES = ("best", "separate", "first")


# --------------------------------------------------------------------------- #
# Per-query metrics
# --------------------------------------------------------------------------- #
def _rr_from(g: int, e: int, tie: str) -> float:
    if tie == "pessimistic":
        return 1.0 / (g + e)
    if tie == "optimistic":
        return 1.0 / (g + 1)
    if tie == "average":
        return float(np.mean(1.0 / np.arange(g + 1, g + e + 1)))
    raise ValueError(tie)


def _hit_from(g: int, e: int, k: int, tie: str) -> float:
    if tie == "pessimistic":
        return 1.0 if (g + e) <= k else 0.0
    if tie == "optimistic":
        return 1.0 if (g + 1) <= k else 0.0
    if tie == "average":
        return float(np.clip(k - g, 0, e) / e)
    raise ValueError(tie)


@dataclass
class EvalResult:
    rr: np.ndarray      # (n_units,)
    h1: np.ndarray
    h3: np.ndarray
    h10: np.ndarray
    unit_query: np.ndarray   # which query each unit belongs to (for 'separate')

    @property
    def mrr(self) -> float: return float(self.rr.mean())
    @property
    def hits1(self) -> float: return float(self.h1.mean())
    @property
    def hits3(self) -> float: return float(self.h3.mean())
    @property
    def hits10(self) -> float: return float(self.h10.mean())


def eval_metrics(scores: np.ndarray, ds: Dataset,
                 tie: str = "pessimistic", multi: str = "best") -> EvalResult:
    """scores: (n_q, n_pool). Returns per-unit RR/H@1/H@3/H@10.

    A 'unit' is a query (best/first) or a (query, dupe) pair (separate).
    """
    rr, h1, h3, h10, uq = [], [], [], [], []
    for q in ds.queries:
        s = scores[q.idx]
        rel = sorted(q.relevant)
        if multi == "separate":
            targets = rel
        elif multi == "first":
            targets = [rel[0]]
        elif multi == "best":
            best = max(rel, key=lambda j: s[j])
            targets = [best]
        else:
            raise ValueError(multi)
        for j in targets:
            val = s[j]
            g = int(np.sum(s > val))
            e = int(np.sum(s == val))
            rr.append(_rr_from(g, e, tie))
            h1.append(_hit_from(g, e, 1, tie))
            h3.append(_hit_from(g, e, 3, tie))
            h10.append(_hit_from(g, e, 10, tie))
            uq.append(q.idx)
    return EvalResult(np.array(rr), np.array(h1), np.array(h3), np.array(h10),
                      np.array(uq, dtype=int))


def run_method(method, ds: Dataset, feats, seeds: List[int],
               tie: str = "pessimistic", multi: str = "best",
               eval_idx: Optional[np.ndarray] = None):
    """Run a method over seeds. Returns dict with mean metrics (+std) and the
    per-unit RR averaged across seeds (for significance). eval_idx (optional) is a
    boolean/index mask over UNITS to restrict the eval set (e.g. NON_OP).

    A deterministic method uses one seed. Stochastic uses all. Returns skipped=True
    if the method returns all-NaN (e.g. B4 without the model).
    """
    use_seeds = seeds if getattr(method, "stochastic", False) else [seeds[0]]
    per_seed_rr, per_seed_h1, per_seed_h3, per_seed_h10 = [], [], [], []
    unit_query = None
    for s in use_seeds:
        sc = method.scores(ds, feats, seed=s)
        if np.all(np.isnan(sc)):
            return {"skipped": True}
        r = eval_metrics(sc, ds, tie=tie, multi=multi)
        per_seed_rr.append(r.rr); per_seed_h1.append(r.h1)
        per_seed_h3.append(r.h3); per_seed_h10.append(r.h10)
        unit_query = r.unit_query
    RR = np.vstack(per_seed_rr)   # (n_seeds, n_units)
    H1 = np.vstack(per_seed_h1); H3 = np.vstack(per_seed_h3); H10 = np.vstack(per_seed_h10)
    # restrict eval set on units if requested
    if eval_idx is not None:
        RR = RR[:, eval_idx]; H1 = H1[:, eval_idx]; H3 = H3[:, eval_idx]
        H10 = H10[:, eval_idx]; unit_query = unit_query[eval_idx]
    rr_mean_over_seeds = RR.mean(axis=0)   # (n_units,) — averaged across seeds
    # metric = mean over units, per seed; report mean+std across seeds
    mrr_seeds = RR.mean(axis=1)
    return {
        "skipped": False,
        "MRR": float(mrr_seeds.mean()), "MRR_std": float(mrr_seeds.std()),
        "Hits@1": float(H1.mean()), "Hits@3": float(H3.mean()), "Hits@10": float(H10.mean()),
        "rr_units": rr_mean_over_seeds,      # per-unit RR averaged across seeds (for tests)
        "unit_query": unit_query,
        "n_units": int(RR.shape[1]),
        "n_seeds": len(use_seeds),
    }


# --------------------------------------------------------------------------- #
# Significance
# --------------------------------------------------------------------------- #
def wilcoxon_test(rr_a: np.ndarray, rr_b: np.ndarray) -> Tuple[Optional[float], int]:
    """Paired Wilcoxon on RR. Returns (p_value, n_nonzero). Drops zero diffs
    (zero_method='wilcox' behaviour reported via n_nonzero)."""
    from scipy.stats import wilcoxon
    diff = rr_a - rr_b
    n_nonzero = int(np.sum(diff != 0))
    if n_nonzero == 0:
        return 1.0, 0
    try:
        p = float(wilcoxon(rr_a, rr_b, zero_method="wilcox").pvalue)
    except ValueError:
        return None, n_nonzero
    return p, n_nonzero


def bootstrap_delta(rr_a: np.ndarray, rr_b: np.ndarray,
                    n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Bootstrap 95% CI (paired by unit) for delta-MRR = MRR(a) - MRR(b)."""
    rng = np.random.default_rng(seed)
    n = len(rr_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    da = rr_a[idx].mean(axis=1) - rr_b[idx].mean(axis=1)
    return {"delta_mrr": float(rr_a.mean() - rr_b.mean()),
            "ci_low": float(np.percentile(da, 2.5)),
            "ci_high": float(np.percentile(da, 97.5))}


def holm(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (monotone), same order as input."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj.tolist()


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """BH (FDR) adjusted p-values, same order as input."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvals[i] * m / (rank + 1)
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    return adj.tolist()


def verdict(delta_mrr: float, ci_low: float, ci_high: float,
            p_adj: Optional[float]) -> str:
    """§4.7 — SIGNIFIKAN hanya jika ketiganya benar; kalau sebagian: AMBIGU + alasan."""
    checks = {
        "p_adj<0.05": (p_adj is not None and p_adj < 0.05),
        "CI_excludes_0": not (ci_low <= 0.0 <= ci_high),
        "|dMRR|>=0.01": abs(delta_mrr) >= 0.01,
    }
    if all(checks.values()):
        return "SIGNIFIKAN"
    failed = [k for k, v in checks.items() if not v]
    return "AMBIGU(" + ",".join(failed) + ")"


if __name__ == "__main__":
    # self-test of the correction functions
    ps = [0.001, 0.01, 0.04, 0.2]
    print("Holm:", [round(x, 4) for x in holm(ps)])
    print("BH  :", [round(x, 4) for x in benjamini_hochberg(ps)])
    print("verdict:", verdict(0.05, 0.02, 0.08, 0.001))
    print("verdict:", verdict(0.005, 0.001, 0.02, 0.001))
