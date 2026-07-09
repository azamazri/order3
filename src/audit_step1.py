"""LANGKAH 1 -- uji sirkularitas data (PEDOMAN_EKSPERIMEN.md, Bagian 8.1).

Menguji asumsi yang tertanam di src/evaluate.py ("near-circular"): apakah overlap
accord query-produk adalah sinyal nyata atau artefak kosakata. Read-only terhadap data;
menghasilkan CSV + satu kesimpulan berbasis angka: "DATA BOCOR" atau "DATA BERSIH".

Output: results/audit/{containment_null,vocab_overlap,source_url_check,token_quality}.csv
        results/audit/step1_report.md
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd

from .data import load_dataset, GLOBAL_XLSX

RESULTS = Path(__file__).resolve().parents[1] / "results" / "audit"
RNG = np.random.default_rng(0)
N_RAND = 20


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    q_by_norm = {q.name_norm: q for q in ds.queries}
    all_q = ds.queries

    # ---------- 1.1 containment: true vs random ----------
    rows_c = []
    c_true_all, c_rand_all = [], []
    ft_true = ft_rand = 0
    n_true = n_rand = 0
    for p in ds.products:
        Ap = set(p.accords)
        if not Ap or not p.rev_norm or p.rev_norm not in q_by_norm:
            continue
        qt = q_by_norm[p.rev_norm]
        c_true = len(Ap & set(qt.accords)) / len(Ap)
        c_true_all.append(c_true); n_true += 1
        ft_true += (c_true == 1.0)
        rand_qs = RNG.choice(len(all_q), size=min(N_RAND, len(all_q)), replace=False)
        crs = []
        for ri in rand_qs:
            qr = all_q[int(ri)]
            if qr.name_norm == p.rev_norm:
                continue
            cr = len(Ap & set(qr.accords)) / len(Ap)
            crs.append(cr); c_rand_all.append(cr); n_rand += 1
            ft_rand += (cr == 1.0)
        rows_c.append([p.name, f"{c_true:.4f}", f"{mean(crs):.4f}" if crs else "", len(crs)])

    with open(RESULTS / "containment_null.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["product", "c_true", "c_rand_mean", "n_rand"])
        w.writerows(rows_c)

    # ---------- 1.2 vocabulary overlap ----------
    Vp = set(a for p in ds.products for a in p.accords)
    Vq = set(a for q in ds.queries for a in q.accords)
    inter = Vp & Vq
    prod_only = sorted(Vp - Vq)
    query_only = sorted(Vq - Vp)
    with open(RESULTS / "vocab_overlap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["accord", "in_product", "in_query"])
        for a in sorted(Vp | Vq):
            w.writerow([a, int(a in Vp), int(a in Vq)])

    # ---------- 1.3 source_url ----------
    g = pd.read_excel(GLOBAL_XLSX, header=1).dropna(how="all")
    url_col = next((c for c in g.columns if str(c).lower() == "source_url"), None)
    urls = g[url_col] if url_col is not None else pd.Series([], dtype=object)
    n_rows = len(g)
    n_nonnull = int(urls.notna().sum())
    n_frag = int(urls.dropna().astype(str).str.contains("fragrantica.com", case=False).sum())
    uniq = urls.dropna().astype(str).nunique()
    with open(RESULTS / "source_url_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["global_rows", n_rows])
        w.writerow(["source_url_nonnull", n_nonnull])
        w.writerow(["points_to_fragrantica", n_frag])
        w.writerow(["unique_urls", uniq])
        w.writerow(["unique_over_nonnull", f"{uniq / n_nonnull:.4f}" if n_nonnull else ""])

    # ---------- 1.4 accord-list sizes ----------
    szp = [len(p.accords) for p in ds.products]
    szq = [len(q.accords) for q in ds.queries]
    n_pairs = n_q_longer = 0
    for p in ds.products:
        if p.rev_norm in q_by_norm and p.accords:
            n_pairs += 1
            if len(q_by_norm[p.rev_norm].accords) > len(p.accords):
                n_q_longer += 1

    # ---------- 1.5 token quality ----------
    def suspicious(a):
        reasons = []
        if ":" in a:
            reasons.append("prefix-colon")
        if re.search(r"\d", a):
            reasons.append("digit")
        if re.search(r"(.)\1\1", a):
            reasons.append("triple-letter")
        if len(a) > 20:
            reasons.append("very-long")
        if re.search(r"[^a-z '\-]", a):
            reasons.append("nonalpha")
        return reasons

    sus = []
    for a in sorted(Vp | Vq):
        r = suspicious(a)
        if r:
            side = ("product" if a in Vp else "") + ("+query" if a in Vq and a in Vp else
                    ("query" if a in Vq else ""))
            sus.append([a, side, ";".join(r)])
    with open(RESULTS / "token_quality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["accord", "side", "reasons"])
        w.writerows(sus)

    # ---------- conclusion ----------
    mt, mr = mean(c_true_all), mean(c_rand_all)
    gap = mt - mr
    # DATA BOCOR jika containment acak juga tinggi (artefak kosakata) atau Vp subset Vq
    vp_subset = len(prod_only) == 0
    leak = (mr > 0.5) or (ft_rand / max(n_rand, 1) > 0.3) or vp_subset
    verdict = "DATA BOCOR" if leak else "DATA BERSIH"

    with open(RESULTS / "step1_report.md", "w", encoding="utf-8") as f:
        f.write("# Langkah 1 -- Uji Sirkularitas Data\n\n")
        f.write("## 1.1 Containment: benar vs acak\n\n")
        f.write("| distribusi | mean | median | prop fully-contained (c==1) | n |\n|---|---|---|---|---|\n")
        f.write(f"| c_true (pasangan benar) | {mt:.4f} | {median(c_true_all):.4f} | "
                f"{ft_true / n_true:.4f} | {n_true} |\n")
        f.write(f"| c_rand (query acak) | {mr:.4f} | {median(c_rand_all):.4f} | "
                f"{ft_rand / n_rand:.4f} | {n_rand} |\n")
        f.write(f"\nSelisih mean (true - rand) = **{gap:+.4f}**.\n\n")

        f.write("## 1.2 Kosakata\n\n")
        f.write(f"- |V_produk| = {len(Vp)}, |V_query| = {len(Vq)}, "
                f"|V_produk ∩ V_query| = {len(inter)}\n")
        f.write(f"- accord HANYA di produk (tak pernah di query): **{len(prod_only)}**\n")
        f.write(f"- accord HANYA di query: **{len(query_only)}**\n")
        f.write(f"- V_produk ⊆ V_query ? **{vp_subset}**\n")
        f.write(f"- produk-only: {', '.join(prod_only[:40])}"
                + (" ..." if len(prod_only) > 40 else "") + "\n")
        f.write(f"- query-only: {', '.join(query_only)}\n\n")

        f.write("## 1.3 source_url\n\n")
        f.write(f"- baris global = {n_rows}; source_url non-null = {n_nonnull}; "
                f"menunjuk fragrantica.com = {n_frag}; URL unik = {uniq} "
                f"({uniq / n_nonnull:.1%} dari non-null).\n\n")

        f.write("## 1.4 Ukuran daftar accord\n\n")
        f.write(f"- |A(produk)|: mean {mean(szp):.2f}, median {median(szp)}\n")
        f.write(f"- |A(query)|: mean {mean(szq):.2f}, median {median(szq)}\n")
        f.write(f"- pasangan berlabel dengan |A(q)| > |A(p)|: {n_q_longer} / {n_pairs} "
                f"({n_q_longer / n_pairs:.1%})\n\n")

        f.write("## 1.5 Kualitas token\n\n")
        f.write(f"- accord mencurigakan: **{len(sus)}** (lihat token_quality.csv)\n")
        for a, side, r in sus[:20]:
            f.write(f"  - `{a}` [{side}] -> {r}\n")
        f.write("\n")

        f.write("## KESIMPULAN\n\n")
        f.write(f"# {verdict}\n\n")
        f.write("Alasan berbasis angka:\n")
        f.write(f"- containment acak {mr:.3f} vs benar {mt:.3f} (selisih {gap:+.3f}); "
                f"proporsi fully-contained acak {ft_rand / n_rand:.1%} vs benar {ft_true / n_true:.1%}.\n")
        f.write(f"- produk punya {len(prod_only)} accord yang tak pernah muncul di query "
                f"-> kosakata {'TIDAK independen' if vp_subset else 'independen'}.\n")
        f.write(f"- source_url: {n_frag}/{n_rows} menunjuk Fragrantica, {uniq} URL unik.\n")
        f.write(f"- query sistematis lebih panjang ({mean(szq):.1f} vs {mean(szp):.1f} accord) "
                f"-> konsisten dengan Fragrantica mendaftar lebih banyak, bukan penyalinan.\n")

    print(f"c_true mean {mt:.4f} (fully {ft_true/n_true:.3f}) | "
          f"c_rand mean {mr:.4f} (fully {ft_rand/n_rand:.3f}) | gap {gap:+.4f}")
    print(f"Vp {len(Vp)} Vq {len(Vq)} inter {len(inter)} prod_only {len(prod_only)} "
          f"query_only {len(query_only)} Vp_subset_Vq={vp_subset}")
    print(f"source_url: {n_frag}/{n_rows} fragrantica, {uniq} unique")
    print(f"sizes: prod {mean(szp):.2f}, query {mean(szq):.2f}; |A(q)|>|A(p)| in {n_q_longer}/{n_pairs}")
    print(f"suspicious tokens: {len(sus)}")
    print(f"\n>>> VERDICT: {verdict} <<<")
    print(f"wrote {RESULTS}/step1_report.md + 4 csv")


if __name__ == "__main__":
    main()
