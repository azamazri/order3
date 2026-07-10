"""Generate `results/kg_paths.json`: for every labeled query, the Top-3 P1
recommendations with their explainable knowledge-graph paths.

A "kg_path" is the per-shared-token decomposition of the P1 (order-2 co-occurrence
TF-IDF) cosine score. Because P1's combined vectors `feats.QC`/`feats.PC` are jointly
L2-normalised, the component product

    QC[q, k] * PC[p, k]  ==  idf(token_k)^2 / (norm_q * norm_p)   (binary indicators = 1)

is exactly the relative contribution of shared token k, and these contributions sum to
the full score s(q, p). Unigram tokens -> `accord_node` (order-1); bigram tokens ->
`co_occurrence_edge` (order-2). No re-implementation of P1: we reuse `P1Order2` and the
precomputed feature vectors.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import load_dataset, leakage_audit
from .methods.base import build_features
from .methods.p1_order2 import P1Order2
from .wheel import WheelTree

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TOP_K = 3
VERSION = "v5.1"


def _accord_super(tree: WheelTree, accord: str) -> str:
    """Superfamily of a single accord, or 'unmapped'."""
    path = tree.accord2path.get(accord)
    return path[0] if path else "unmapped"


def _edge_super(tree: WheelTree, a: str, b: str) -> str:
    """Wheel superfamily of an accord pair: a shared superfamily, else 'cross_family',
    else 'unmapped' when either endpoint is outside the lexicon."""
    sa, sb = _accord_super(tree, a), _accord_super(tree, b)
    if sa == "unmapped" or sb == "unmapped":
        return "unmapped"
    return sa if sa == sb else "cross_family"


def build_kg_paths() -> List[Dict]:
    ds = load_dataset()
    leakage_audit(ds, verbose=False)
    feats = build_features(ds)
    tree = WheelTree()

    p1 = P1Order2()
    scores = p1.scores(ds, feats)           # (n_q, n_pool)
    QC, PC = feats.QC, feats.PC
    nu = feats.n_uni                         # first nu columns are unigram dims

    entries: List[Dict] = []
    for q in ds.queries:
        s = scores[q.idx]
        order = np.argsort(-s)               # descending score
        # gold rank (audit): 1 + #products strictly above the best gold product
        gold_best = float(np.max(s[list(q.relevant)]))
        gold_rank = int(np.sum(s > gold_best)) + 1

        recs = []
        for rank, pidx in enumerate(order[:TOP_K], start=1):
            pidx = int(pidx)
            qv, pv = QC[q.idx], PC[pidx]
            score = float(s[pidx])

            paths = []
            order1_sum = 0.0
            order2_sum = 0.0
            # shared dims = where both vectors are positive
            shared = np.nonzero((qv > 0) & (pv > 0))[0]
            for k in shared:
                w = float(qv[k] * pv[k])      # == idf(token)^2 / (norm_q*norm_p)
                if k < nu:                    # unigram -> accord node (order-1)
                    accord = feats.vocab[k]
                    paths.append({
                        "type": "accord_node",
                        "accord": accord,
                        "weight": w,
                        "wheel_super_family": _accord_super(tree, accord),
                        "relation": "shared_accord",
                    })
                    order1_sum += w
                else:                          # bigram -> co-occurrence edge (order-2)
                    a, b = feats.bvocab[k - nu]
                    paths.append({
                        "type": "co_occurrence_edge",
                        "from": a,
                        "to": b,
                        "weight": w,
                        "wheel_super_family": _edge_super(tree, a, b),
                        "relation": "shared_signature_pair",
                    })
                    order2_sum += w

            paths.sort(key=lambda d: d["weight"], reverse=True)
            recs.append({
                "rank": rank,
                "product_name": ds.products[pidx].name,
                "local_family": ds.products[pidx].family,
                "score": score,
                "is_gold": pidx in q.relevant,
                "kg_paths": paths,
                "order2_contribution": (order2_sum / score) if score > 1e-12 else 0.0,
                "order1_contribution": (order1_sum / score) if score > 1e-12 else 0.0,
            })

        entries.append({
            "query": {
                "name": q.name,
                "global_family": q.family,
                "accords": list(q.accords),
            },
            "recommendations": recs,
            "gold_rank": gold_rank,
            "meta": {"method": "order2_cooccurrence_tfidf", "version": VERSION},
        })
    return entries


def summarize(entries: List[Dict]) -> None:
    o2, n_paths = [], []
    fam = Counter()
    for e in entries:
        for r in e["recommendations"]:
            o2.append(r["order2_contribution"])
            n_paths.append(len(r["kg_paths"]))
            for p in r["kg_paths"]:
                fam[p["wheel_super_family"]] += 1
    print(f"queries                 : {len(entries)}")
    print(f"recommendations/query   : {TOP_K} (total {sum(len(e['recommendations']) for e in entries)})")
    print(f"avg order2_contribution : {np.mean(o2):.3f}  (target ~0.813)")
    print(f"avg paths/recommendation: {np.mean(n_paths):.2f}")
    print("wheel_super_family distribution (over all paths):")
    for k, v in fam.most_common():
        print(f"    {k:<14}{v}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    entries = build_kg_paths()

    full = RESULTS_DIR / "kg_paths.json"
    sample = RESULTS_DIR / "kg_paths_sample.json"
    with open(full, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    with open(sample, "w", encoding="utf-8") as f:
        json.dump(entries[:5], f, ensure_ascii=False, indent=2)

    print("=" * 70)
    summarize(entries)
    print("=" * 70)
    print(f"wrote {full}  ({len(entries)} queries)")
    print(f"wrote {sample} (first 5 queries)")


if __name__ == "__main__":
    main()
