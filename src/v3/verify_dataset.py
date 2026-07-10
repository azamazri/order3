"""Tahap 1 — verifikasi dataset bersih. READ-ONLY.

Membaca kedua Excel bersih secara langsung (tidak bergantung pada src.data, yang
baru diperbaiki di Tahap 2). Menghitung angka §1.1 PEDOMAN_EKSPERIMEN_V2, menjalankan
assertion keras §1.2, dan menyilang-periksa cleaning_changelog.csv (§1.3).

Aturan query valid (dideklarasikan, bukan cleaning diam-diam):
  sebuah baris global menjadi QUERY jika (a) ada >=1 produk berlabel yang menunjuk
  ke namanya DAN (b) baris itu punya >=1 accord. Baris global tanpa accord tidak
  bisa dijawab metode apa pun sehingga tidak sah sebagai query.

Interpretasi assertion `product_only_tokens == []` (§1.2):
  yang di-crash adalah accord produk yang berada DI LUAR kosakata global terkontrol
  (V_product \\ V_global_all) — inilah "kebocoran kosakata" / Gate G1. Diff
  product-vs-query dan query-vs-product tetap dilaporkan sebagai informasi (§1.1),
  tapi tidak men-crash: sebuah accord bisa sah ada di global tanpa muncul di subset
  query.

Output: results/v3/00_dataset_verification.md, results/v3/00_vocab.csv
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AROMATIQUE_XLSX = ROOT / "dataset-aromatique.xlsx"
GLOBAL_XLSX = ROOT / "global_reference.xlsx"
CHANGELOG_CSV = ROOT / "cleaning_changelog.csv"
OUT_DIR = ROOT / "results" / "v3"

_WS = re.compile(r"\s+")
_SPLIT = re.compile(r"[;,]")


def norm_name(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    return (_WS.sub(" ", str(s).strip().lower())) or None


def parse_accords(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    out, seen = [], set()
    for tok in _SPLIT.split(str(s)):
        t = _WS.sub(" ", tok.strip().lower())
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_raw():
    a = pd.read_excel(AROMATIQUE_XLSX)
    g = pd.read_excel(GLOBAL_XLSX)
    acols = [c for c in g.columns if str(c).startswith("accord_")]

    # products
    products = []
    for i, row in a.reset_index(drop=True).iterrows():
        products.append({
            "idx": int(i),
            "name": str(row.get("product_name")),
            "accords": parse_accords(row.get("main_accords")),
            "rev_norm": norm_name(row.get("revolutionize")),
            "is_labeled": pd.notna(row.get("revolutionize")),
        })

    # global map (first occurrence per normalised name)
    gmap = {}
    for _, row in g.iterrows():
        nm = norm_name(row.get("perfume_name"))
        if nm is None:
            continue
        accords, seen = [], set()
        for c in acols:
            v = row.get(c)
            if pd.notna(v):
                t = _WS.sub(" ", str(v).strip().lower())
                if t and t not in seen:
                    seen.add(t)
                    accords.append(t)
        if nm not in gmap:
            gmap[nm] = {"name": str(row.get("perfume_name")).strip(), "accords": accords}
    return a, g, products, gmap


def compute():
    a, g, products, gmap = load_raw()

    n_products = len(products)
    n_labeled = sum(p["is_labeled"] for p in products)
    n_unlabeled = n_products - n_labeled

    V_product = set()
    plen = []
    empty_prod = 0
    for p in products:
        V_product.update(p["accords"])
        plen.append(len(p["accords"]))
        if not p["accords"]:
            empty_prod += 1

    V_global_all = set()
    for info in gmap.values():
        V_global_all.update(info["accords"])

    # relevant sets: rev_norm -> {product idx}
    rel = {}
    for p in products:
        if p["rev_norm"] is not None:
            rel.setdefault(p["rev_norm"], set()).add(p["idx"])

    # valid queries: name in gmap AND has >=1 accord
    queries = []
    excluded_empty_accord = []
    labeled_names_without_global = sorted(nm for nm in rel if nm not in gmap)
    for nm in sorted(rel):
        if nm not in gmap:
            continue
        if len(gmap[nm]["accords"]) == 0:
            excluded_empty_accord.append(gmap[nm]["name"])
            continue
        queries.append({"norm": nm, "name": gmap[nm]["name"],
                        "accords": gmap[nm]["accords"], "relevant": rel[nm]})

    n_queries = len(queries)
    n_labeled_pairs = sum(len(q["relevant"]) for q in queries)
    multi = [(q["name"], len(q["relevant"])) for q in queries if len(q["relevant"]) > 1]

    # products whose (labeled) target is not a resolvable global row
    prod_unmatched = [p["name"] for p in products
                      if p["rev_norm"] is not None and p["rev_norm"] not in gmap]

    V_query = set()
    qlen = []
    for q in queries:
        V_query.update(q["accords"])
        qlen.append(len(q["accords"]))
    V_canonical = V_product | V_query
    V_shared = V_product & V_query

    product_tokens_outside_global = sorted(V_product - V_global_all)  # HARD (G1)
    product_only_vs_query = sorted(V_product - V_query)               # info (§1.1)
    query_only_vs_product = sorted(V_query - V_product)               # info (§1.1)

    return dict(
        products=products, gmap=gmap, queries=queries,
        n_products=n_products, n_labeled=n_labeled, n_unlabeled=n_unlabeled,
        n_global_rows=len(g), n_unique_global=len(gmap),
        n_queries=n_queries, n_labeled_pairs=n_labeled_pairs,
        multi=multi, excluded_empty_accord=excluded_empty_accord,
        labeled_names_without_global=labeled_names_without_global,
        prod_unmatched=prod_unmatched,
        V_product=V_product, V_query=V_query, V_global_all=V_global_all,
        V_canonical=V_canonical, V_shared=V_shared,
        product_tokens_outside_global=product_tokens_outside_global,
        product_only_vs_query=product_only_vs_query,
        query_only_vs_product=query_only_vs_product,
        plen=plen, qlen=qlen, empty_prod=empty_prod,
    )


def changelog_crosscheck(R):
    """§1.3 — untuk tiap baris changelog, verifikasi terhadap produk yang benar."""
    if not CHANGELOG_CSV.exists():
        return {"present": False, "violations": [["changelog", "MISSING", "", "", ""]]}
    cl = pd.read_csv(CHANGELOG_CSV)
    prod_by_name = {norm_name(p["name"]): p for p in R["products"]}
    violations = []
    n_checked = 0
    for _, row in cl.iterrows():
        pname = norm_name(row.get("product_name"))
        action = str(row.get("action")).strip()
        before = str(row.get("before") or "").strip().lower()
        after = "" if pd.isna(row.get("after")) else str(row.get("after")).strip().lower()
        p = prod_by_name.get(pname)
        if p is None:
            # product name in changelog not found (e.g. '-' placeholder rows)
            if str(row.get("product_name")).strip() not in ("-", "", "nan"):
                violations.append([action, str(row.get("product_name")), before, after,
                                   "product_name not found in aromatique"])
            continue
        acc = set(p["accords"])
        n_checked += 1
        if action == "drop":
            if before in acc:
                violations.append([action, p["name"], before, after,
                                   "dropped token still present"])
        elif action == "split":
            comps = [c.strip() for c in after.split(",") if c.strip()]
            missing = [c for c in comps if c not in acc]
            if missing:
                violations.append([action, p["name"], before, after,
                                   f"split components missing: {missing}"])
        elif action == "typo_fix":
            corrected = [c.strip() for c in after.split(",") if c.strip()]
            missing = [c for c in corrected if c not in acc]
            if missing:
                violations.append([action, p["name"], before, after,
                                   f"corrected token missing: {missing}"])
            if before in acc:
                violations.append([action, p["name"], before, after,
                                   "typo token still present"])
    return {"present": True, "n_rows": len(cl), "n_checked": n_checked,
            "violations": violations}


def write_outputs(R, cc):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 00_vocab.csv ----
    all_tokens = sorted(R["V_product"] | R["V_query"] | R["V_global_all"])
    pcnt = Counter(t for p in R["products"] for t in p["accords"])
    qcnt = Counter(t for q in R["queries"] for t in q["accords"])
    with open(OUT_DIR / "00_vocab.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "in_product", "in_query", "in_global_all",
                    "count_products", "count_queries"])
        for t in all_tokens:
            w.writerow([t, int(t in R["V_product"]), int(t in R["V_query"]),
                        int(t in R["V_global_all"]), pcnt.get(t, 0), qcnt.get(t, 0)])

    # ---- 00_dataset_verification.md ----
    ph = dict(sorted(Counter(R["plen"]).items()))
    qh = dict(sorted(Counter(R["qlen"]).items()))
    L = []
    L.append("# 00 — Verifikasi Dataset Bersih (Tahap 1)\n")
    L.append("Read-only. Dihitung langsung dari `dataset-aromatique.xlsx` dan "
             "`global_reference.xlsx` (versi bersih, committed).\n")
    L.append("## 1.1 Angka inti\n")
    L.append("| item | nilai |")
    L.append("|---|---|")
    L.append(f"| n_products | {R['n_products']} |")
    L.append(f"| n_labeled (revolutionize terisi) | {R['n_labeled']} |")
    L.append(f"| n_unlabeled (distractor) | {R['n_unlabeled']} |")
    L.append(f"| n_global_rows | {R['n_global_rows']} |")
    L.append(f"| unique_global_names | {R['n_unique_global']} |")
    L.append(f"| n_queries (>=1 dupe & >=1 accord) | {R['n_queries']} |")
    L.append(f"| n_labeled_pairs | {R['n_labeled_pairs']} |")
    L.append(f"| queries_with_multiple_dupes | {len(R['multi'])} |")
    L.append(f"| labeled_products_without_global_row | {len(R['prod_unmatched'])} "
             f"({len(R['labeled_names_without_global'])} nama distinct) |")
    L.append(f"| queries_excluded_empty_accord | {len(R['excluded_empty_accord'])} |")
    L.append(f"| V_product | {len(R['V_product'])} |")
    L.append(f"| V_query | {len(R['V_query'])} |")
    L.append(f"| V_global_all | {len(R['V_global_all'])} |")
    L.append(f"| V_canonical (V_product ∪ V_query) | {len(R['V_canonical'])} |")
    L.append(f"| V_shared (V_product ∩ V_query) | {len(R['V_shared'])} |")
    L.append(f"| sel accord kosong (produk) | {R['empty_prod']} |")
    L.append("")
    L.append("## 1.1 Query multi-dupe\n")
    for nm, k in R["multi"]:
        L.append(f"- `{nm}` → {k} produk dupe")
    L.append("")
    L.append("## 1.1 Distribusi panjang accord\n")
    L.append(f"- `len(A(p))` (produk): {ph}")
    L.append(f"- `len(A(q))` (query): {qh}")
    L.append("")
    L.append("## 1.1 Kosakata (daftar lengkap)\n")
    L.append(f"**product_only vs query** ({len(R['product_only_vs_query'])}): "
             f"`{R['product_only_vs_query']}`")
    L.append("")
    L.append(f"**query_only vs product** ({len(R['query_only_vs_product'])}): "
             f"`{R['query_only_vs_product']}`")
    L.append("")
    L.append(f"**product_tokens_outside_global_vocab** "
             f"({len(R['product_tokens_outside_global'])}): "
             f"`{R['product_tokens_outside_global']}`  ← target assertion §1.2 / Gate G1")
    L.append("")
    L.append("## 1.2 Assertion keras\n")
    L.append("| assertion | hasil |")
    L.append("|---|---|")
    L.append(f"| product_tokens_outside_global == [] | "
             f"{'LULUS' if not R['product_tokens_outside_global'] else 'GAGAL'} |")
    L.append(f"| n_labeled_pairs > 0 | "
             f"{'LULUS' if R['n_labeled_pairs'] > 0 else 'GAGAL'} |")
    L.append(f"| tidak ada produk dengan 0 accord | "
             f"{'LULUS' if R['empty_prod'] == 0 else 'GAGAL'} |")
    L.append(f"| tidak ada query dengan 0 accord | "
             f"{'LULUS' if all(len(q['accords']) > 0 for q in R['queries']) else 'GAGAL'} |")
    L.append("")
    L.append("## 1.3 Silang-periksa cleaning_changelog.csv\n")
    if cc["present"]:
        L.append(f"- baris changelog: {cc['n_rows']}, ter-cek terhadap produk: "
                 f"{cc['n_checked']}")
        if cc["violations"]:
            L.append(f"- **PELANGGARAN: {len(cc['violations'])}**")
            L.append("")
            L.append("| action | product | before | after | masalah |")
            L.append("|---|---|---|---|---|")
            for v in cc["violations"]:
                L.append("| " + " | ".join(str(x) for x in v) + " |")
        else:
            L.append("- **0 pelanggaran** — semua drop hilang, semua split komponennya ada, "
                     "semua typo_fix terkoreksi.")
    else:
        L.append("- changelog TIDAK ADA")
    L.append("")
    L.append("## Catatan\n")
    L.append(f"- Query multi-dupe ({len(R['multi'])}) berarti penggabungan/duplikasi "
             "global membuat >1 produk menunjuk parfum sama — relevan untuk Tahap 4.3 (S8).")
    L.append(f"- 33 labeled products tanpa baris global (S7) — target parfum tidak ada "
             "di `global_reference`; tetap di pool sebagai distractor berlabel.")
    (OUT_DIR / "00_dataset_verification.md").write_text("\n".join(L), encoding="utf-8")


def main():
    R = compute()

    # -------- §1.2 hard assertions --------
    assert R["product_tokens_outside_global"] == [], \
        f"kosakata produk bocor di luar global: {R['product_tokens_outside_global']}"
    assert R["n_labeled_pairs"] > 0
    assert R["empty_prod"] == 0, "ada produk dengan 0 accord"
    assert all(len(q["accords"]) > 0 for q in R["queries"]), "ada query dengan 0 accord"

    cc = changelog_crosscheck(R)
    write_outputs(R, cc)

    print("Tahap 1 OK")
    print(f"  n_products={R['n_products']} n_labeled={R['n_labeled']} "
          f"n_queries={R['n_queries']} n_pairs={R['n_labeled_pairs']}")
    print(f"  V_product={len(R['V_product'])} V_query={len(R['V_query'])} "
          f"V_canonical={len(R['V_canonical'])}")
    print(f"  product_tokens_outside_global={R['product_tokens_outside_global']}")
    print(f"  changelog violations={len(cc['violations']) if cc['present'] else 'NA'}")
    print(f"  wrote {OUT_DIR/'00_dataset_verification.md'} + 00_vocab.csv")


if __name__ == "__main__":
    main()
