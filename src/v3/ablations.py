"""Tahap 11 — ablation A3 / B4 symmetry / A4-vs-A3-vs-order-N.

1) A3 ablation: full / tanpa fitur bigram / hanya fitur bigram
   -> apakah kekuatan A3 datang dari co-occurrence atau dari regresi logistiknya.
2) B4 symmetry:
   - B4a_prose: SBERT atas meaning(+visual) sisi produk vs accord sisi query
     (sumber product_text.csv, WAJIB leakage_audit). Asimetris & lintas-bahasa.
   - B4b_accord: accord-only kedua sisi (= metode tabel utama).
   - B4c_family_accord: accord + family kedua sisi.
   Selisih B4a vs B4b = biaya asimetri (bukan kekuatan prosa).
3) A4 vs A3 vs order-N: ketiganya pair-dependent, mekanisme berbeda (dinarasikan).

Output: 09_ablations.csv, 09_b4_leakage.csv
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_dataset, norm_name
from ..methods.base import build_features, groupkfold_oof
from ..methods import a3_signature
from ..methods.p2_fusion import _make_logreg_fit_predict
from . import protocol as Pr

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v3"
ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2, 3, 4]
_WS = re.compile(r"\s+")
_STOP = {"by", "de", "la", "le", "the", "for", "and", "of", "du", "des", "eau",
         "parfum", "perfume", "extreme", "intense", "man", "men", "woman", "women",
         "pour", "homme", "femme", "edp", "edt"}


def _mrr(scores, ds, mask):
    r = Pr.eval_metrics(scores, ds, tie="pessimistic", multi="best")
    return float(np.mean(r.rr[mask])) if mask is not None else float(r.rr.mean())


# --------------------------------------------------------------------------- #
def a3_ablation(ds, feats, nonop_mask):
    variants = {"A3_full": None, "A3_no_bigram": [0, 4, 5, 6], "A3_only_bigram": [1, 2, 3]}
    rows = []
    for vname, cols in variants.items():
        def feat_fn(qi, cols=cols):
            X = a3_signature._features(feats, qi)
            return X if cols is None else X[:, cols]
        per_seed_all, per_seed_nonop = [], []
        for s in SEEDS:
            sc = groupkfold_oof(ds, feat_fn, _make_logreg_fit_predict(1.0), seed=s)
            per_seed_all.append(_mrr(sc, ds, None))
            per_seed_nonop.append(_mrr(sc, ds, nonop_mask))
        rows.append(["A3_ablation", vname, "ALL", f"{np.mean(per_seed_all):.4f}",
                     f"{np.std(per_seed_all):.4f}"])
        rows.append(["A3_ablation", vname, "NON_OP", f"{np.mean(per_seed_nonop):.4f}",
                     f"{np.std(per_seed_nonop):.4f}"])
        print(f"  {vname}: ALL={np.mean(per_seed_all):.4f} NON_OP={np.mean(per_seed_nonop):.4f}")
    return rows


# --------------------------------------------------------------------------- #
def _distinctive(name_norm):
    if not name_norm:
        return set()
    toks = re.findall(r"[a-z0-9']+", name_norm.lower())
    return {t for t in toks if len(t) >= 4 and t not in _STOP}


def b4_symmetry(ds, nonop_mask):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        print(f"  B4 ablation skipped: {e}")
        return [["B4_symmetry", "ALL_variants", "ALL", "SKIPPED", ""]], []

    pt = pd.read_csv(ROOT / "product_text.csv")
    by_name = {norm_name(r["product_name"]): r for _, r in pt.iterrows()}

    # leakage audit on product free text (meaning + visual)
    leak_rows = []
    n_leaky = 0
    prod_text_clean = {}
    for p in ds.products:
        rec = by_name.get(norm_name(p.name))
        meaning = "" if rec is None or pd.isna(rec.get("meaning")) else str(rec["meaning"])
        vis = ""
        if rec is not None:
            vis = " ".join(str(rec.get(c)) for c in ("visual_note", "visual_note_alt")
                           if pd.notna(rec.get(c)))
        raw = f"{meaning} {vis}".lower()
        clean = meaning
        present = {t for t in _distinctive(p.rev_norm) if re.search(rf"\b{re.escape(t)}\b", raw)}
        if present:
            n_leaky += 1
            leak_rows.append([p.name, p.rev_norm, ";".join(sorted(present))])
            for t in present:
                clean = re.sub(rf"\b{re.escape(t)}\b", " ", clean, flags=re.IGNORECASE)
        fam = "" if rec is None or pd.isna(rec.get("olfactory_family")) else str(rec["olfactory_family"])
        prod_text_clean[p.idx] = {"prose": _WS.sub(" ", (clean + " " + fam)).strip() or " ",
                                  "family": fam}
    print(f"  B4a leakage: {n_leaky}/{len(ds.products)} products leaked own global-name tokens")

    def enc(texts):
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    rows = []
    # B4a_prose: product = prose(+family), query = accords
    pa = enc([prod_text_clean[p.idx]["prose"] for p in ds.products])
    qa = enc([", ".join(q.accords) for q in ds.queries])
    sc_a = qa @ pa.T
    # B4b_accord: accord-only both sides
    pb = enc([", ".join(p.accords) for p in ds.products])
    sc_b = qa @ pb.T
    # B4c_family_accord: accord + family both sides
    pc = enc([(", ".join(p.accords) + " " + prod_text_clean[p.idx]["family"]).strip()
              for p in ds.products])
    qc = enc([(", ".join(q.accords) + " " + q.family).strip() for q in ds.queries])
    sc_c = qc @ pc.T

    for vname, sc in (("B4a_prose", sc_a), ("B4b_accord", sc_b), ("B4c_family_accord", sc_c)):
        a = _mrr(sc, ds, None)
        n = _mrr(sc, ds, nonop_mask)
        rows.append(["B4_symmetry", vname, "ALL", f"{a:.4f}", ""])
        rows.append(["B4_symmetry", vname, "NON_OP", f"{n:.4f}", ""])
        print(f"  {vname}: ALL={a:.4f} NON_OP={n:.4f}")
    return rows, leak_rows


# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    feats = build_features(ds)
    strata = pd.read_csv(OUT_DIR / "02_strata.csv")
    nonop_q = set(strata.loc[strata["is_NON_OP"] == 1, "q_idx"])
    nonop_mask = np.array([q.idx in nonop_q for q in ds.queries])

    all_rows = [["ablation", "variant", "eval_set", "MRR", "MRR_std"]]
    print("A3 ablation:")
    all_rows += a3_ablation(ds, feats, nonop_mask)
    print("B4 symmetry:")
    b4_rows, leak_rows = b4_symmetry(ds, nonop_mask)
    all_rows += b4_rows

    with open(OUT_DIR / "09_ablations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(all_rows)
    with open(OUT_DIR / "09_b4_leakage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["product", "target_name_norm", "leaked_tokens"])
        w.writerows(leak_rows)
    print("Ablations written.")


if __name__ == "__main__":
    main()
