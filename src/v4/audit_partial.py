"""Tahap B — audit stratum `partial` SEBELUM membangun apa pun.

Untuk tiap pasangan berlabel non-contained (A(p) ⊄ A(q)):
  missing_accords     = set(A(p)) − set(A(q))
  alt_query_contained = query q' != q dengan set(A(p)) ⊆ set(A(q'))   (indikasi mislink)
  missing_has_close   = tiap accord hilang punya "kerabat" di A(q) menurut PPMI (340 produk)

Kelas (prioritas): likely_mislink > soft_matchable > genuinely_far.
Gate keputusan: soft_matchable < 8 -> improvement tak layak (lompat Tahap E).

Kedekatan PPMI: b∈A(q) "kerabat" accord hilang a jika b termasuk top-5 tetangga PPMI a
(PPMI>0). PPMI dihitung HANYA dari fitur accord 340 produk (nol kebocoran label).

Output: results/v4/B_partial_audit.csv, results/v4/B_partial_audit.md
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from ..data import load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v4"
TOPK = 5


def build_ppmi(ds):
    vocab = sorted({a for p in ds.products for a in p.accords}
                   | {a for q in ds.queries for a in q.accords})
    vidx = {a: i for i, a in enumerate(vocab)}
    V = len(vocab)
    C = np.zeros((V, V))
    for p in ds.products:                       # PRODUCTS ONLY (no labels)
        idxs = [vidx[a] for a in set(p.accords)]
        for i, j in combinations(idxs, 2):
            C[i, j] += 1; C[j, i] += 1
        for i in idxs:
            C[i, i] += 1
    total = C.sum()
    row = C.sum(1, keepdims=True); col = C.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C * total) / (row * col))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(ppmi, 0.0)
    # top-k neighbors per accord (ppmi>0)
    neigh = {}
    for a, i in vidx.items():
        order = np.argsort(-ppmi[i])
        neigh[a] = [(vocab[j], ppmi[i, j]) for j in order[:TOPK] if ppmi[i, j] > 0]
    return vidx, ppmi, neigh


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    vidx, ppmi, neigh = build_ppmi(ds)

    # non-contained labeled pairs
    pairs = []
    for q in ds.queries:
        for pi in sorted(q.relevant):
            ap, aq = set(ds.products[pi].accords), set(q.accords)
            if not ap <= aq:
                pairs.append((pi, q.idx))

    rows = []
    counts = {"likely_mislink": 0, "soft_matchable": 0, "genuinely_far": 0}
    for pi, qi in pairs:
        ap = ds.products[pi].accords
        aq = set(ds.queries[qi].accords)
        missing = [a for a in ap if a not in aq]
        # alt query containment
        alt = [q2.name for q2 in ds.queries
               if q2.idx != qi and set(ap) <= set(q2.accords)]
        # closeness per missing accord
        close_info = []
        all_close = True
        for a in missing:
            nb = {b for b, _ in neigh.get(a, [])}
            hit = [b for b in aq if b in nb]
            if hit:
                # pick highest-ppmi neighbor present
                best = max(hit, key=lambda b: ppmi[vidx[a], vidx[b]])
                close_info.append(f"{a}->{best}({ppmi[vidx[a],vidx[best]]:.2f})")
            else:
                all_close = False
                close_info.append(f"{a}->NONE")
        if alt:
            cls = "likely_mislink"
        elif missing and all_close:
            cls = "soft_matchable"
        else:
            cls = "genuinely_far"
        counts[cls] += 1
        rows.append([ds.products[pi].name, ds.queries[qi].name,
                     ";".join(missing), len(missing),
                     ";".join(alt[:3]), int(bool(alt)),
                     ";".join(close_info), int(all_close), cls])

    with open(OUT_DIR / "B_partial_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["product", "query", "missing_accords", "n_missing",
                    "alt_query_contained", "has_alt", "missing_closeness",
                    "all_missing_have_close", "class"])
        w.writerows(rows)

    warranted = counts["soft_matchable"] >= 8
    L = ["# B — Audit stratum `partial`\n"]
    L.append(f"n pasangan non-contained (partial) = {len(pairs)}\n")
    L.append("| kelas | jumlah |")
    L.append("|---|---|")
    for k in ("likely_mislink", "soft_matchable", "genuinely_far"):
        L.append(f"| {k} | {counts[k]} |")
    L.append("")
    L.append(f"**Gate keputusan:** soft_matchable = {counts['soft_matchable']} "
             f"({'>=8 → LANJUT Tahap C' if warranted else '<8 → improvement TAK LAYAK, lompat Tahap E'}).")
    L.append("")
    L.append("PPMI dihitung hanya dari 340 produk (nol kebocoran label). "
             "`likely_mislink` = accord produk malah cocok penuh ke query lain (bukan "
             "kelemahan metode). `genuinely_far` = accord hilang tak punya kerabat "
             "co-occurrence → tak ada metode leksikal/co-occurrence yang bisa menolong.")
    (OUT_DIR / "B_partial_audit.md").write_text("\n".join(L), encoding="utf-8")

    print(f"partial pairs = {len(pairs)}")
    print(f"  likely_mislink = {counts['likely_mislink']}")
    print(f"  soft_matchable = {counts['soft_matchable']}")
    print(f"  genuinely_far  = {counts['genuinely_far']}")
    print(f"Gate: {'WARRANTED (>=8) -> Tahap C' if warranted else 'NOT WARRANTED (<8) -> Tahap E'}")
    return warranted


if __name__ == "__main__":
    main()
