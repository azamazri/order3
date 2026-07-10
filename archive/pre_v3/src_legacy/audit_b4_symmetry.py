"""LANGKAH 2.3 -- B4 symmetric-text ablation (PEDOMAN Bagian 4/B4, 8.2).

Same checkpoint (paraphrase-multilingual-MiniLM-L12-v2). Three text regimes:

  B4a : product = text_clean (stripped Indonesian prose + family) + accords
        query   = family + accords                                (CURRENT; asymmetric)
  B4b : product = accords only ; query = accords only             (SYMMETRIC; comparable)
  B4c : product = family + accords ; query = family + accords     (symmetric + family)

Only B4b puts both sides on the same footing as the other 11 comparators (accords only),
so B4b is the variant for the main table. Deterministic (pretrained, no fit). If the model
cannot load, results are NaN and reported as skipped -- never faked.
Output: results/audit/b4_symmetry.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .data import load_dataset, leakage_audit
from .evaluate import per_query_metrics

RESULTS = Path(__file__).resolve().parents[1] / "results" / "audit"
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    leakage_audit(ds, verbose=False)   # fills text_clean for B4a

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL)
    except Exception as e:  # pragma: no cover
        print(f"[B4] skipped: {e}")
        with open(RESULTS / "b4_symmetry.csv", "w", newline="") as f:
            csv.writer(f).writerow(["variant", "mrr", "hits1", "hits3", "note"])
            csv.writer(f).writerow(["skipped", "", "", "", str(e)])
        return

    acc = lambda x: " ".join(x.accords)
    variants = {
        "B4a_prose_asymmetric": (
            [(p.text_clean or p.family or " ") + " " + acc(p) for p in ds.products],
            [q.family + " " + acc(q) for q in ds.queries]),
        "B4b_accord_only": (
            [acc(p) or " " for p in ds.products],
            [acc(q) or " " for q in ds.queries]),
        "B4c_family_accord": (
            [(p.family + " " + acc(p)).strip() or " " for p in ds.products],
            [(q.family + " " + acc(q)).strip() or " " for q in ds.queries]),
    }

    rows = []
    for name, (ptxt, qtxt) in variants.items():
        pe = model.encode(ptxt, normalize_embeddings=True, show_progress_bar=False)
        qe = model.encode(qtxt, normalize_embeddings=True, show_progress_bar=False)
        r = per_query_metrics(qe @ pe.T, ds)
        rows.append([name, f"{r.mrr:.4f}", f"{r.hits1:.4f}", f"{r.hits3:.4f}"])
        print(f"{name:<24} mrr {r.mrr:.4f} h1 {r.hits1:.3f} h3 {r.hits3:.3f}", flush=True)

    with open(RESULTS / "b4_symmetry.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "mrr", "hits1", "hits3"])
        w.writerows(rows)
    print(f"wrote {RESULTS}/b4_symmetry.csv")


if __name__ == "__main__":
    main()
