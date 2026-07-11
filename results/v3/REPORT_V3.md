# REPORT_V3 — Accord Co-occurrence untuk Dupe Retrieval

Sintesis eksperimen v3 di atas **dataset bersih**. Setiap angka merujuk CSV di
`results/v3/`. Tidak ada kata "signifikan" tanpa `p_adj` + CI. Metode usulan = **order-2**
(`P1_order2`), `N*=2` dipilih via nested CV (Tahap 6).

Branch: `exp/v3-clean`. Protokol: `PEDOMAN_EKSPERIMEN_V2.md`.

---

## 1. Verifikasi dataset (Tahap 1 — `00_dataset_verification.md`, `00_vocab.csv`)

| item | nilai |
|---|---|
| n_products | 340 (243 labeled, 97 distractor) |
| n_global_rows | 263 |
| n_queries (≥1 dupe & ≥1 accord) | 208 |
| n_labeled_pairs | 210 |
| queries_with_multiple_dupes | 2 (`black saffron byredo`, `vibrant leather ... zara`) |
| labeled_products_without_global_row | 33 (S7) |
| V_product / V_query / V_global_all / V_canonical | 56 / 58 / 61 / 61 |
| product tokens di luar kosakata global | **[] → Gate G1 lolos** |
| changelog cross-check | 0 pelanggaran |

Aturan query valid (dideklarasikan, dikonfirmasi peneliti): query butuh ≥1 produk berlabel
**dan** ≥1 accord. (Satu baris global "not found" sempat ber-accord kosong; peneliti
melengkapi datanya sebelum run final.)

## 2. Perbaikan kode & keputusan desain (Tahap 2 — `01_implementation_audit.md`)

- Loader `data.py`: header global baris 1, kunci `perfume_name`; `meaning`/`text_clean`/
  `visual`/`leakage_audit` dikeluarkan dari load path.
- **`B4_sbert` tabel utama = accord-only, simetris** (input identik dg semua metode). Prosa
  turun jadi ablation `B4a` (Tahap 11). Alasan penuh di §2.2 audit (bukan "B4 lemah").
- **Wheel A1:** `V_canonical \ WHEEL = {orange, mineral, anis}` (persis tabel usulan);
  dipetakan a-priori (S12=map), `drop` jadi sensitivitas. `anis` judgement call (Fresh/
  Aromatic primer; alt Amber). **Gate G3 tidak aktif** (0/548 fragrance kehilangan >20%
  massa pada map maupun drop).
- De-bias docstring (a5/p1/a3); "parameter-free" → "no learned parameters"; SyntaxWarning
  p1 diperbaiki. Acceptance `run_all.py --fast` bersih (warnings-as-errors).

## 3. Tabel P/L/U (Tahap 3 — `01_plu.csv`, `01_determinism.csv`) — apa adanya

Pair-dependent (P=1): **10/15**. Non-pair-dependent (P=0): B1, B2, B3, B4, A1.
Determinisme: **15/15 deterministik** pada seed tetap → **Gate G2 lolos**.

> **Konsekuensi RQ3:** karena B5, B6, A2, A5, A6 (+A3, A4, P2, P3) semuanya pair-dependent,
> narasi "co-occurrence vs non-co-occurrence" tidak bisa dibangun dari pembanding ini.
> Yang membedakan = **bagaimana** co-occurrence dimodelkan (eksplisit tanpa parameter
> terlatih vs dipelajari/terkompresi vs supervised). Framing diserahkan ke peneliti.

## 4. Verdict sirkularitas (Tahap 5 — `02_circularity.md`, `02_*.csv`)

**Verdict: `DERIVATIF (sebagian)`.** subseq obs=100 (p_emp=0.001), prefix obs=89
(p_emp=0.001) vs null 1000-permutasi; K1 semua konvensi <0.50 (uji sah). contained=172/210.
Strata query: prefix 88 / contained 72 / subseq 11 / partial 37; **NON_OP=109**.

- Konsekuensi (§7.1): **eval set primer = NON_OP**; `ALL` jadi sensitivitas. NON_OP=109 ≥
  100 → **Gate G7 tidak aktif**.
- **Gate G8** (17 suspected mislinked > 15) → ditinjau peneliti: ~16 containment kebetulan,
  1 duplikat varian (Baccarat Rouge 540 base/extrait). Diputuskan **lanjut, list di-flag**
  (`02_suspected_mislinked.csv`, hanya review manual).
- **Terbuka (Limitations):** provenance `main_accords` (penciuman vs salin Fragrantica) —
  email ke Aromatique, jawaban masuk Limitations apa pun hasilnya.

## 5. Tangga order + N* (Tahap 6 — `03_order_ladder.csv`, `03_order_significance.csv`)

MRR order N=1..6 — ALL: 0.505 / **0.566** / 0.568 / 0.576 / 0.577 / 0.577;
NON_OP: 0.442 / **0.509** / 0.505 / 0.510 / 0.510 / 0.510.

Signifikansi tangga (Holm, `verdict` §4.7):
- **order2 vs order1: SIGNIFIKAN** — ALL ΔMRR=0.061 (p_adj=1.2e-7), NON_OP ΔMRR=0.067
  (p_adj=7.5e-4).
- Setiap step >2 (order3..6): **tidak signifikan** (AMBIGU) di ALL & NON_OP.

**N\* = 2** (aturan §7.2: N terkecil yang setiap step M>N tidak signifikan). Nested-CV mode
= 5 (greedy-max tanpa filter signifikansi) — dilaporkan berdampingan, tidak dipilih (§7.2).
**RQ2: titik jenuh di order-2.** Verifikasi Gate G5: order-1 mereproduksi B2 (max|diff|=4e-16).

## 6. Tabel utama per kategori, dua eval set (Tahap 8 — `04_main_table.csv`, `05_significance.csv`)

Peringkat MRR **NON_OP** (primer): P2 0.511 · **order2 0.509** · A3 0.492 · A2 0.485 ·
P3 0.461 · B2 0.442 · A6 0.424 · B4 0.414 · B3 0.380 · B1 0.318 · A1 0.266 · A5 0.245 ·
A4 0.212 · B6 0.186 · B5 0.176.

Signifikansi **order-2 vs 14 pembanding** (NON_OP, BH FDR 0.05, `verdict` §4.7):
- **SIGNIFIKAN lebih baik** (11): B1, B2, B3, B4, B5, B6, A1, A4, A5, A6, P3.
- **AMBIGU / setara** (3): A2_ppmi_svd (Δ=0.025, CI memuat 0), A3_signature (Δ=0.017, CI
  memuat 0), P2_fusion (Δ=−0.002). Ketiganya **pair-dependent**.

> **RQ3 (apa adanya):** order-2 (tanpa parameter terlatih) **setara** dengan metode
> co-occurrence terbaik (A2/A3/P2) dan **signifikan mengungguli** semua metode leksikal/
> marginal (B1/B2/B3), neural-pretrained (B4), co-occurrence-terlatih yang overfit di data
> kecil (B5/B6), taksonomik (A1), dan supervised per-pasangan (A4/A5).

**Train/test gap** (`04_gap.csv`): A4 in-sample 0.558 → OOF 0.222 (**gap 0.336**), A5 gap
0.119 → supervised per-bigram **overfit** pada label langka; A3/A6/P2 gap ≤0.02 (teregularisasi).

## 7. Grid sensitivitas (Tahap 9 — `07_sensitivity.csv`, `07_sensitivity_summary.md`)

96 sel (S1×S2×S3×S4×S9×S11):
- **N\* = 2 di 100% sel** → **STABIL** (§7.3, ambang 80%).
- **RQ1 (order2>order1) signifikan di 100% sel** → **§7.4 TERPENUHI** (primer NON_OP ✓,
  ALL ✓, ≥80% sel ✓). **RQ1 DIDUKUNG. Gate G6 tidak aktif.**
- S8 (multi-dupe): order-2 robust (best/separate/first = 0.566/0.563/0.566).
- S12 (wheel map vs drop): A1 0.303/0.302 (ALL) — perbedaan dapat diabaikan.

## 8. Kinerja per stratum containment (Tahap 5.5 — `02_strata.csv`, `06_stratified.csv`)

Stratum query (tertinggi per query): prefix 88, contained 72, subseq 11, partial 37.
MRR order-2 per stratum: prefix 0.629 · subseq 0.631 · contained 0.699 · **partial 0.140**.
Stratum `partial` (accord produk tak sepenuhnya termuat) **sulit untuk semua metode**
(0.10–0.17) — tugas praktis terpecahkan saat containment berlaku. Eval `NON_OP`
(partial+contained, tanpa prefix/subseq) = 109 query = eval primer. Catatan: NON_OP =
"tanpa bukti transkripsi berurutan", **bukan** "subset bersih".

## 9. Dekomposisi + null (Tahap 10 — `08_decomposition.csv`)

Share order-2 dari skor: **true 78.2%**, top1 81.5%, **random 42.4%**. Selisih true−random
pada order-2 = **+0.358** → statistik **informatif** (bukan tautologis), boleh masuk paper.

## 10. Ablation (Tahap 11 — `09_ablations.csv`, `09_b4_leakage.csv`)

- **A3** (NON_OP): full 0.491 · **no_bigram 0.317** · only_bigram 0.471 → kekuatan A3 dari
  **fitur co-occurrence**, bukan regresi logistiknya.
- **B4 symmetry** (NON_OP): **B4a_prose 0.042** (asimetris lintas-bahasa) · **B4b_accord
  0.414** (tabel utama) · B4c_family 0.362. 33/340 produk bocor token nama global (di-strip).
  → "B4 lemah" versi lama = **confound asimetri input**, bukan arsitektur. Selisih B4a vs
  B4b = biaya asimetri.
- **A4 vs A3 vs order-N:** ketiganya pair-dependent; A4 *mempelajari* bobot per-bigram dari
  label langka (overfit, gap 0.336), order-2 memakai IDF tetap (general), A3 hybrid.

## 11. Audit sitasi (Tahap 12 — `10_citation_audit.md`)

- "Fannisa et al." **salah** → seharusnya "Aurora et al." (r28: Aurora, Fannisa Eimin & Baizal).
- BM25 / Word2Vec / node2vec = pembanding **tanpa sitasi**; PPMI-SVD (Levy&Goldberg) uncited.
- `references.bib` **hilang** (`\bibliography{references}` menunjuk file tak ada).
- KGAT/LightGCN/KGIN **dikecualikan** dari perbandingan di teks (tidak diimplementasikan) —
  konsisten dg `src/methods/`.
- Klaim "parameter-free" masih di `.tex` → diperbaiki saat revisi paper.

## 12. Daftar klaim lama yang GUGUR (dengan angka pengganti)

| klaim lama | status v3 | angka pengganti |
|---|---|---|
| "order-3 terbaik" (nama repo `order3`) | **GUGUR** | N\*=2; step order3−order2 tidak signifikan (`03_order_significance.csv`) |
| "order-2 terbaik" (klaim lain) | order-2 = titik jenuh, bukan "menang atas semua" | setara A2/A3/P2 (`05_significance.csv`) |
| "B4 (neural text) lemah → paradigma kalah" | **GUGUR (confound)** | B4a 0.042 vs B4b 0.414 (`09_ablations.csv`) |
| "sebagian besar skor dari co-occurrence" (tanpa null) | kini **terbukti informatif** | true 78% vs random 42% (`08_decomposition.csv`) |
| "containment tinggi = data bersih" (uji lama) | **DIBATALKAN** (tak diagnostik) | verdict order-preservation = DERIVATIF sebagian |
| "A3 varian / A6 comparator" (kategori tak konsisten) | **DIPERBAIKI** | keduanya P=1,L=1 (`01_plu.csv`) |
| "metode parameter-free" | **DICABUT** | N hyperparameter → nested CV |
| A5 "expected weak" (docstring bias) | **DIHAPUS** | gap 0.119 dilaporkan sebagai angka, bukan ekspektasi |

## 13. Ancaman validitas yang tersisa

1. **Provenance accord** (DERIVATIF sebagian): sebagian pasangan menyalin urutan accord
   target → sinyal sebagian derivatif. Dimitigasi dg eval primer NON_OP; jawaban Aromatique
   masuk Limitations.
2. **IDF transduktif:** IDF dari 340 produk termasuk produk di fold uji — konsisten semua
   metode, tapi transduktif (dicatat).
3. **Skala data kecil** (208 query, 1 positif/query): metode supervised (A4/A5) overfit;
   hasil "learned < fixed-IDF" spesifik rezim data langka ini.
4. **Neural text tak ada:** katalog tak punya prosa panjang; paradigma "neural" diwakili
   SBERT atas string accord (B4b), bukan deskripsi produk.
5. **`anis`** placement judgement call (menunggu pembimbing); dampak dapat diabaikan (S12/A1).
6. **Suspected mislinked** 17 (di-flag, belum ditinjau manual penuh); `references.bib` hilang.

---

### Status Gate
G1 lolos · G2 lolos · G3 tidak aktif · G4 tidak aktif (K1 <0.50) · G5 lolos · G6 tidak aktif
(RQ1 didukung) · G7 tidak aktif (NON_OP=109) · **G8 dipicu → di-resolve peneliti (lanjut)**.

### Ringkasan jawaban RQ
- **RQ1** — order-2 > order-1: **DIDUKUNG** (signifikan di NON_OP, ALL, 100% sel grid).
- **RQ2** — titik jenuh: **N\* = 2** (tak ada perbaikan signifikan di order>2).
- **RQ3** — vs paradigma mapan: order-2 (tanpa parameter terlatih) **setara** metode
  co-occurrence terbaik (A2/A3/P2), **unggul** atas leksikal/neural-pretrained/learned-
  co-occurrence/taksonomik/supervised-per-pair. Framing final diserahkan ke peneliti.
