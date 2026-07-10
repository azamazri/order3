# Langkah 2 -- Perbaikan Struktural

Acuan: PEDOMAN_EKSPERIMEN.md Bagian 4, 5, 6, 8.2. Dijalankan setelah Langkah 1
dinyatakan **DATA BERSIH**. Semua sub-langkah sudah di-commit ke `exp/audit-v2`.

| Sub | Isi | Commit |
|---|---|---|
| 2.1 | A3 dipindah ke varian; rename `signature logistic` | `b82fac6` |
| 2.2 | Ablation fitur A3 | `ab0dd25` |
| 2.3 | B4 symmetric-text (B4a/B4b/B4c) | `a10c7b6` |
| 2.4 | A6 imbalance -> `sample_weight` seimbang | `b82fac6` |
| 2.5 | Bersihkan docstring bias | `b82fac6` |
| 2.6 | Rename "leave-one-out" -> "full-pool retrieval" | `b82fac6` |

---

## 2.1 A3 dipindah keluar dari blok pembanding

A3 (fitur #2 `n_shared_b`, #3 `w_shared_b`, #4 `max_rare`) adalah representasi
order-2 metode usulan + regresi logistik. Ia **varian**, bukan lawan.

- `signature LTR` -> **`signature logistic`** (ini regresi logistik biasa, bukan
  learning-to-rank dengan objektif ranking).
- Dikategorikan sebagai **varian metode usulan**, sejajar P2/P3 (PEDOMAN Bagian 5).
- `methods/base.py`: label tier "learning-to-rank/LTR" -> "supervised".

## 2.2 Ablation A3 -- yang bekerja adalah co-occurrence, bukan logistiknya

Nested CV, `C in {0.01, 0.1, 1, 10}`. Referensi: B2 = 0.4540, order-3 = 0.5080.

| set fitur | fitur | best C | MRR | mrr_std | Hits@1 | Hits@3 |
|---|---|---|---|---|---|---|
| A3-full | semua 7 | 0.01 | 0.5131 | 0.0029 | 0.4067 | 0.5654 |
| **A3-noco** | buang #2,#3,#4 (order-1 murni) | 0.01 | **0.4221** | 0.0105 | 0.2919 | 0.4868 |
| A3-conly | HANYA #2,#3,#4 | 0.1 | 0.4949 | 0.0022 | 0.3804 | 0.5519 |

**Tafsir:** membuang tiga fitur co-occurrence menjatuhkan A3 ke **0.422 — di
BAWAH baseline marginal B2 (0.454)**. Sebaliknya memakai HANYA tiga fitur itu
hampir memulihkan performa penuh (0.495). Bukti langsung bahwa kekuatan A3 (dan
selisih kecilnya atas order-3) berasal dari **representasi co-occurrence metode
usulan**, bukan dari regresi logistiknya. Menegaskan A3 = varian, bukan pembanding.

## 2.3 B4 symmetric-text

Tiga regime teks, checkpoint sama (`paraphrase-multilingual-MiniLM-L12-v2`),
deterministik (pretrained, tanpa fit).

| variant | teks produk | teks query | MRR | Hits@1 | Hits@3 |
|---|---|---|---|---|---|
| B4a prosa (asimetris, LAMA) | text_clean + family + accord | family + accord | 0.2717 | 0.1627 | 0.3110 |
| **B4b accord-only (setara)** | accord saja | accord saja | **0.3439** | 0.2297 | 0.3780 |
| B4c family+accord | family + accord | family + accord | 0.3293 | 0.2153 | 0.3732 |

**Tafsir:** B4b satu-satunya regime yang setara dengan 11 pembanding lain
(accord di kedua sisi) dan menjadi varian yang masuk tabel utama. Setup lama
(B4a) memaksa S-BERT mengerjakan *cross-lingual prosa-vs-daftar* — itu justru
**merugikan** (+0.072 MRR saat diseragamkan ke accord-only). Perbandingan lama
tidak setara; B4b memperbaikinya.

## 2.4 A6 imbalance

`GradientBoostingClassifier` tidak punya `class_weight`, sementara A3/A4 memakai
`class_weight="balanced"` — tidak konsisten. Diperbaiki: `.fit()` sekarang
menerima `sample_weight = compute_sample_weight("balanced", ytr)`.
Smoke test: A6 MRR 0.457 -> 0.476.

## 2.5 Docstring dibersihkan

Frasa ekspektasi hasil dihapus dari `src/methods/` dan diganti deskripsi
mekanisme netral:

- A1: "expected to LOSE" — dihapus
- A4: "expected to OVERFIT badly (this is the point)" — dihapus
- A6: "overfit demo / Included to show that a flexible learner does not beat..." — dihapus
- P2: "expected BEST" — dihapus

## 2.6 "leave-one-out" -> "full-pool retrieval"

Query = parfum global, tidak pernah ada di pool 340 produk, jadi tidak ada yang
"ditinggalkan". Istilah salah nama diperbaiki di `evaluate.py`,
`order_ablation.py`, `phase1_diag.py`.

---

## KESIMPULAN

Semua enam perbaikan struktural Langkah 2 selesai dan ter-commit.

- **A3 bukan pembanding** — dibuktikan lewat ablation (2.2): tanpa fitur
  co-occurrence ia jatuh di bawah B2. Direkategorikan sebagai varian (2.1).
- **B4 dibuat setara** — B4b (accord-only) menggantikan B4a di tabel utama (2.3).
- **A6 disamakan** perlakuan imbalance-nya dengan A3/A4 (2.4).
- **Bias eksperimen dihapus** dari docstring (2.5) dan **salah-nama LOO**
  diperbaiki (2.6).

Langkah 3 (re-run final + signifikansi) menunggu review Langkah 2 (PEDOMAN §8.2).
