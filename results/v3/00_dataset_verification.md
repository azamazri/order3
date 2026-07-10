# 00 — Verifikasi Dataset Bersih (Tahap 1)

Read-only. Dihitung langsung dari `dataset-aromatique.xlsx` dan `global_reference.xlsx` (versi bersih, committed).

## 1.1 Angka inti

| item | nilai |
|---|---|
| n_products | 340 |
| n_labeled (revolutionize terisi) | 243 |
| n_unlabeled (distractor) | 97 |
| n_global_rows | 263 |
| unique_global_names | 263 |
| n_queries (>=1 dupe & >=1 accord) | 208 |
| n_labeled_pairs | 210 |
| queries_with_multiple_dupes | 2 |
| labeled_products_without_global_row | 33 (32 nama distinct) |
| queries_excluded_empty_accord | 0 |
| V_product | 56 |
| V_query | 58 |
| V_global_all | 61 |
| V_canonical (V_product ∪ V_query) | 59 |
| V_shared (V_product ∩ V_query) | 55 |
| sel accord kosong (produk) | 0 |

## 1.1 Query multi-dupe

- `Black Saffron Byredo` → 2 produk dupe
- `Vibrant Leather Eau de Parfum — Zara` → 2 produk dupe

## 1.1 Distribusi panjang accord

- `len(A(p))` (produk): {2: 5, 3: 7, 4: 19, 5: 303, 6: 4, 8: 1, 9: 1}
- `len(A(q))` (query): {3: 1, 5: 14, 6: 11, 7: 19, 8: 26, 9: 33, 10: 104}

## 1.1 Kosakata (daftar lengkap)

**product_only vs query** (1): `['rum']`

**query_only vs product** (3): `['anis', 'mineral', 'spicy']`

**product_tokens_outside_global_vocab** (0): `[]`  ← target assertion §1.2 / Gate G1

## 1.2 Assertion keras

| assertion | hasil |
|---|---|
| product_tokens_outside_global == [] | LULUS |
| n_labeled_pairs > 0 | LULUS |
| tidak ada produk dengan 0 accord | LULUS |
| tidak ada query dengan 0 accord | LULUS |

## 1.3 Silang-periksa cleaning_changelog.csv

- baris changelog: 67, ter-cek terhadap produk: 67
- **0 pelanggaran** — semua drop hilang, semua split komponennya ada, semua typo_fix terkoreksi.

## Catatan

- Query multi-dupe (2) berarti penggabungan/duplikasi global membuat >1 produk menunjuk parfum sama — relevan untuk Tahap 4.3 (S8).
- 33 labeled products tanpa baris global (S7) — target parfum tidak ada di `global_reference`; tetap di pool sebagai distractor berlabel.