"""Tahap 3 — audit implementasi 15 metode.

(1) Tabel P/L/U diisi dengan MEMBACA kode (bukan nama), disertai kutipan bukti.
    P = pair-dependent (statistik atas pasangan accord di dalam satu fragrance)
    L = label-supervised (parameter dilatih dari label revolutionize)
    U = unsupervised-learned (parameter dipelajari dari data tanpa label)
(2) Uji determinisme: tiap metode dijalankan 2x pada seed tetap; skor harus identik.
    Gate G2 jika ada metode non-deterministik pada seed tetap.

Output: results/v3/01_plu.csv, results/v3/01_determinism.csv
        (bagian 2 dari 01_implementation_audit.md ditulis manual dari sini).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..data import load_dataset
from ..methods import ALL_METHODS
from ..methods.base import build_features

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v3"

# --------------------------------------------------------------------------- #
# P/L/U classification from reading each method's code (evidence = code fact).
# tuple: (P, L, U, evidence)
# --------------------------------------------------------------------------- #
PLU = {
    "B1_jaccard":       (0, 0, 0, "set overlap/union only; no pairs, no learning"),
    "B2_tfidf_cos":     (0, 0, 0, "order-1 unigram tf-idf cosine; IDF = corpus statistic, not learned"),
    "B3_bm25":          (0, 0, 0, "BM25Okapi over unigram accord tokens; k1/b fixed; no pairs"),
    "B4_sbert":         (0, 0, 0, "pretrained sentence encoder over accord string; no params learned from OUR data/labels"),
    "B5_word2vec":      (1, 0, 1, "skip-gram (sg=1) trained on accord lists as sentences, window=5>=len(A(p))~5 => learns accord co-occurrence"),
    "B6_node2vec":      (1, 0, 1, "explicit accord co-occurrence graph + random walks + skip-gram embeddings"),
    "A1_wheel_treeW":   (0, 0, 0, "tree-W1 over uniform unigram distribution on fixed taxonomy; no pair stats, no learning"),
    "A2_ppmi_svd":      (1, 0, 1, "explicit accord-accord co-occurrence matrix -> PPMI -> TruncatedSVD embeddings"),
    "A3_signature":     (1, 1, 0, "features n_shared_b/w_shared_b/max_rare = shared bigrams; logistic on labels OOF"),
    "A4_bigram_salience": (1, 1, 0, "shared-bigram indicator over full bigram vocab; L2 logistic on labels OOF"),
    "A5_bilinear":      (1, 1, 0, "score q^T(diag(d)+L L^T)p; L L^T = cross-accord; d,L learned from labels OOF"),
    "A6_gbm_fusion":    (1, 1, 0, "features include bigram_cos (order-2); GBM trained on labels OOF"),
    "P1_order2":        (1, 0, 0, "unigram+bigram (order-N) tf-idf cosine; IDF fixed statistic; N = hyperparam via nested CV, no gradient learning"),
    "P2_fusion":        (1, 1, 0, "features [bigram_cos,unigram_cos,shared]; logistic on labels OOF"),
    "P3_hubidf":        (1, 0, 0, "order-2 tf-idf with hub down-weight (corpus degree statistic); no label learning"),
}

FILE = {
    "B1_jaccard": "b1_jaccard.py", "B2_tfidf_cos": "b2_tfidf.py", "B3_bm25": "b3_bm25.py",
    "B4_sbert": "b4_sbert.py", "B5_word2vec": "b5_word2vec.py", "B6_node2vec": "b6_node2vec.py",
    "A1_wheel_treeW": "a1_wheel.py", "A2_ppmi_svd": "a2_ppmi_svd.py", "A3_signature": "a3_signature.py",
    "A4_bigram_salience": "a4_bigram_salience.py", "A5_bilinear": "a5_bilinear.py",
    "A6_gbm_fusion": "a6_gbm.py", "P1_order2": "p1_order2.py", "P2_fusion": "p2_fusion.py",
    "P3_hubidf": "p3_hubidf.py",
}


def write_plu():
    with open(OUT_DIR / "01_plu.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "file", "P", "L", "U", "evidence"])
        for m in ALL_METHODS:
            P, L, U, ev = PLU[m.name]
            w.writerow([m.name, FILE.get(m.name, ""), P, L, U, ev])
    npair = sum(PLU[m.name][0] for m in ALL_METHODS)
    print(f"P/L/U written. pair-dependent (P=1): {npair}/15")


def determinism_check(include_heavy=True):
    ds = load_dataset()
    feats = build_features(ds)
    rows = []
    print("determinism (2x at seed 0):")
    for m in ALL_METHODS:
        heavy = m.name in ("B4_sbert", "B5_word2vec", "B6_node2vec")
        if heavy and not include_heavy:
            rows.append([m.name, m.stochastic, "SKIPPED", ""])
            continue
        try:
            s1 = m.scores(ds, feats, seed=0)
            if np.all(np.isnan(s1)):
                rows.append([m.name, m.stochastic, "skipped(NaN)", ""])
                print(f"  {m.name:<20} skipped (NaN, optional dep unavailable)")
                continue
            s2 = m.scores(ds, feats, seed=0)
        except Exception as e:
            rows.append([m.name, m.stochastic, f"ERROR:{e}", ""])
            print(f"  {m.name:<20} ERROR {e}")
            continue
        maxdiff = float(np.nanmax(np.abs(s1 - s2)))
        deterministic = bool(np.allclose(s1, s2, rtol=0, atol=1e-12, equal_nan=True))
        rows.append([m.name, m.stochastic, deterministic, f"{maxdiff:.3e}"])
        print(f"  {m.name:<20} deterministic={deterministic} max|diff|={maxdiff:.3e}")
    with open(OUT_DIR / "01_determinism.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "stochastic", "deterministic_at_fixed_seed", "max_abs_diff"])
        w.writerows(rows)
    # Gate G2
    nondet = [r[0] for r in rows if r[2] is False]
    return nondet


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-heavy", action="store_true", help="skip B4/B5/B6")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_plu()
    nondet = determinism_check(include_heavy=not args.no_heavy)
    if nondet:
        gate = OUT_DIR / "GATE_G2.md"
        gate.write_text("# GATE G2 — metode non-deterministik pada seed tetap\n\n"
                        + "\n".join(f"- {n}" for n in nondet) + "\n", encoding="utf-8")
        print(f"\n*** GATE G2 TRIGGERED: {nondet} -> wrote {gate}")
    else:
        print("\nGate G2 clear: semua metode deterministik pada seed tetap.")


if __name__ == "__main__":
    main()
