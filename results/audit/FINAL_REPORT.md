# FINAL REPORT — Deep Audit Interaction-Free Perfume Dupe Retrieval

Acuan tunggal: `PEDOMAN_EKSPERIMEN.md`. Branch: `exp/audit-v2`.
Laporan ringkas per langkah: [step1](step1_report.md) · [step2](step2_report.md) ·
[step3](step3_report.md).

---

## 1. Kesimpulan Langkah 1 — Sirkularitas data

# DATA BERSIH

Uji containment benar vs acak (`containment_null.csv`):

| distribusi | mean | median | fully-contained (c=1) |
|---|---|---|---|
| c_true (pasangan benar) | 0.860 | 1.000 | 69.5% |
| c_rand (query acak, 20×) | 0.350 | 0.400 | 1.8% |

Overlap accord tinggi **hanya** untuk pasangan benar (selisih mean +0.509).
Dukungan tambahan: produk punya 55 accord yang **tak pernah** muncul di query
(kosakata katalog independen, bukan subset global); `source_url` 264/266 menunjuk
Fragrantica dengan 256 URL unik; query sistematis lebih panjang (8.7 vs 4.8 accord)
— konsisten dengan Fragrantica mendaftar lebih banyak, **bukan** penyalinan.
Sinyalnya nyata → eksperimen layak dilanjutkan; komentar "near-circular" dihapus
dari `src/evaluate.py` (commit `44b2c24`).

---

## 2. Tabel final (`final_table.csv`)

MRR = full-pool retrieval atas 209 query, 340 kandidat, expected-rank tie-break.
best_params dari Fase 2 nested CV (GroupKFold-5 by query); order-N parameter-free.

### Pembanding independen (11)

| method | best_params | MRR | Hits@1 | Hits@3 |
|---|---|---|---|---|
| A6_gbm_fusion | n_est300,depth5,lr0.01 | 0.4771 | 0.354 | 0.551 |
| B2_tfidf_cos | parameter-free | 0.4540 | 0.335 | 0.507 |
| A2_ppmi_svd | rank=50 | 0.4431 | 0.342 | 0.498 |
| B1_jaccard | parameter-free | 0.4277 | 0.303 | 0.487 |
| B5_word2vec | dim64,win5,ep200 | 0.3934 | 0.277 | 0.450 |
| B3_bm25 | k1=2.0,b=0.75 | 0.3919 | 0.282 | 0.440 |
| B6_node2vec | dim32,p0.25,q4 | 0.3605 | 0.248 | 0.402 |
| A1_wheel_treeW | w=(1,1,1) | 0.3577 | 0.240 | 0.392 |
| B4b_sbert_accord | parameter-free | 0.3439 | 0.230 | 0.378 |
| A4_bigram_salience | C=0.01 | 0.2963 | 0.184 | 0.328 |
| A5_bilinear | rank16,iters1000 | 0.2358 | 0.131 | 0.260 |

### Varian metode usulan (3)

| method | best_params | MRR | Hits@1 | Hits@3 |
|---|---|---|---|---|
| A3_signature_full | C=0.01 | 0.5131 | 0.407 | 0.565 |
| P2_fusion | fixed logistic | 0.4984 | 0.376 | 0.581 |
| P3_hubidf | parameter-free | 0.4476 | 0.335 | 0.490 |

### order-N (usulan; order-3 = metode utama)

| N | MRR | Hits@1 | Hits@3 |
|---|---|---|---|
| 1 | 0.4540 | 0.335 | 0.507 |
| 2 | 0.4963 | 0.378 | 0.562 |
| **3** | **0.5080** | **0.397** | **0.567** |
| 4 | 0.5105 | 0.397 | 0.583 |
| 5 | 0.5109 | 0.397 | 0.583 |
| 6 | 0.5109 | 0.397 | 0.583 |

Validasi (3.2): N=1 = 0.4540 (= B2), N=2 = 0.4963 (= order-2). Lolos toleransi 1e-3.

---

## 3. Signifikansi (`final_significance.csv`)

Wilcoxon signed-rank berpasangan antar-query atas RR + 95% bootstrap CI (10.000
resample). Referensi order-3; ΔMRR = MRR(order3) − MRR(lawan). Signifikan bila
p<0.05 **dan** CI eksklusi 0.

| perbandingan | ΔMRR | CI | Wilcoxon p | verdict |
|---|---|---|---|---|
| order3 vs A5_bilinear | +0.2722 | [0.222, 0.323] | 3.5e-19 | menang |
| order3 vs A4_bigram_salience | +0.2117 | [0.158, 0.265] | 2.5e-12 | menang |
| order3 vs B4b_sbert_accord | +0.1641 | [0.110, 0.218] | 1.9e-08 | menang |
| order3 vs A1_wheel_treeW | +0.1503 | [0.100, 0.200] | 4.6e-09 | menang |
| order3 vs B6_node2vec | +0.1475 | [0.103, 0.193] | 6.6e-09 | menang |
| order3 vs B3_bm25 | +0.1161 | [0.081, 0.153] | 2.2e-12 | menang |
| order3 vs B5_word2vec | +0.1145 | [0.071, 0.160] | 1.4e-06 | menang |
| order3 vs B1_jaccard | +0.0803 | [0.051, 0.111] | 6.9e-04 | menang |
| order3 vs A2_ppmi_svd | +0.0648 | [0.022, 0.109] | 4.3e-04 | menang |
| order3 vs B2_tfidf_cos | +0.0540 | [0.029, 0.080] | 5.6e-09 | menang |
| **order3 vs A6_gbm_fusion** | **+0.0308** | **[0.008, 0.054]** | **0.141** | **CAMPURAN** |
| **order3 vs A3_signature_full** | **−0.0051** | **[−0.020, 0.009]** | **0.891** | **SERI** |
| order3 vs P2_fusion | +0.0096 | [−0.002, 0.023] | 0.066 | seri |
| order3 vs P3_hubidf | +0.0604 | [0.036, 0.086] | 1.1e-10 | menang |
| order1 vs order2 | −0.0424 | [−0.067, −0.019] | 8.7e-08 | order-2 lebih baik |
| order2 vs order3 | −0.0116 | [−0.023, −0.002] | 2.8e-03 | order-3 lebih baik |
| order3 vs order4 | −0.0025 | [−0.010, 0.005] | 1.7e-02 | marginal (CI memuat 0) |
| order4 vs order5 | −0.0005 | [−0.001, 0.000] | 5.3e-02 | jenuh |
| order5 vs order6 | 0.0000 | [0, 0] | 1.000 | identik |

**Baca:** order-3 menang signifikan atas **10 dari 11** pembanding. Terhadap **A6**
hasil **campuran** (CI eksklusi 0 tetapi Wilcoxon p=0.14) → tidak diklaim menang.
Terhadap **varian sendiri** (A3-full, P2) **seri statistik** — selisih +0.005 A3
tidak signifikan (p=0.89). Kurva orde jenuh setelah order-3 (order-3→order-4 sudah
tak kokoh).

---

## 4. Train/test gap (`final_gap.csv`)

| method | train MRR | test MRR | gap | catatan |
|---|---|---|---|---|
| A3_signature_full | 0.5148 | 0.5131 | 0.0017 | out-of-fold |
| A6_gbm_fusion | 0.5810 | 0.4771 | 0.1039 | out-of-fold |
| A4_bigram_salience | 0.4754 | 0.2963 | 0.1792 | overfit (gap besar) |
| A5_bilinear | 0.5022 | 0.2358 | 0.2664 | overfit (gap besar) |
| A2_ppmi_svd / B5 / B6 | — | test-only | — | train==test menurut konstruksi → tak bisa overfit |
| B4b_sbert_accord | — | 0.3439 | — | pretrained, tanpa fit → tak bisa overfit |

---

## 5. Dekomposisi & sparsity (order-3)

Dekomposisi share skor peringkat teratas (`final_decomposition.csv`):
orde-1 **8.7%**, orde-2 **37.2%**, orde-3 **54.2%** → **~91%** dari co-occurrence.

Sparsity token bersama query↔produk benar (`final_sparsity.csv`):
k=1: 4.20 · k=2: 7.80 · k=3: 7.68 · k=4: 4.12 · k=5: 1.22 · k=6: 0.27.
Memuncak di k=2/3, lenyap di k=5/6 → menjelaskan plateau MRR setelah orde-3.

---

## 6. SEMUA perubahan vs run lama, dengan alasannya

Run lama = `main` @ `9ddcfea` (`results/results.csv`, `results/significance.csv`).

| # | perubahan | lama → baru | alasan |
|---|---|---|---|
| 1 | **A5 optimizer** | GD lr=0.5, 300 iter (tak konvergen, loss≈ln2) → L-BFGS-B, gradien analitik, maxiter 5000 | model lama tak bisa memuat data latihnya (train MRR 0.184) — rusak, bukan overfit (`phase1/phase2`) |
| 2 | **Semua pembanding di-tuning** nested CV | default params → nested GroupKFold-5/3 by query | perbandingan lama tak adil (params default tak pernah dinyatakan/dipilih). MRR naik: B5 0.185→0.393, A4 0.139→0.296, A1 0.284→0.358, A3 0.433→0.513, A6 0.457→0.477 |
| 3 | **A6 imbalance** | tanpa class-weight → `sample_weight="balanced"` pada `.fit()` | A3/A4 pakai balanced, A6 tidak — tak konsisten (2.4) |
| 4 | **B4 teks** | B4a prosa Indonesia (produk) vs daftar accord Inggris (query), asimetris, MRR 0.272 → **B4b accord-only kedua sisi**, MRR 0.344 | B4a satu-satunya metode cross-lingual prosa-vs-daftar; tak setara 11 pembanding lain. B4b setara & lebih baik (2.3) |
| 5 | **Metode utama** | P1 order-2 (0.4963) → **order-3** (0.5080) | order-N diperluas N=1..6; uji signifikansi menaruh knee di order-3 (order-2→3 signifikan, 3→4 tidak kokoh) |
| 6 | **A3 rekategori** | pembanding "T2 structure" → **varian metode usulan** | fitur #2/#3/#4 adalah representasi order-2 usulan; ablation 2.2 membuktikan sinyalnya co-occurrence (A3-noco jatuh ke 0.422 < B2) |
| 7 | **Uji signifikansi** | P1/P2 vs pembanding **belum ter-tuning** → order-3 vs pembanding **ter-tuning penuh** | rule PEDOMAN §7: bandingkan lawan versi terkuat. Efek: A3 lama "kalah" (0.433) kini **seri** (0.513) — dilaporkan jujur |
| 8 | **Docstring de-bias** | "expected to LOSE/OVERFIT badly/overfit demo/expected BEST" → deskripsi mekanisme netral | menulis ekspektasi hasil ke kode = bias eksperimen, terbaca di repo publik (2.5) |
| 9 | **Uji sirkularitas** | tidak ada → `step1_report.md` + 5 CSV | asumsi "near-circular" di `evaluate.py` tak pernah diuji; kini diuji → DATA BERSIH (Langkah 1) |
| 10 | **Diagnostik baru** | — → ablation A3, dekomposisi per-orde 1/2/3, sparsity k=1..6, gap final | menjelaskan MEKANISME, bukan hanya angka akhir |

Catatan angka pembanding stokastik pada harness Langkah 3 dievaluasi ulang pada
best_params tetap (3 seed, GroupKFold-5) sehingga berbeda tipis dari nilai nested-CV
Fase 2 (mis. A6 0.4746→0.4771, A4 0.2890→0.2963); best_params tidak di-tuning ulang.

---

## 7. Metode yang berubah nama, dengan alasannya

| lama | baru | alasan |
|---|---|---|
| `signature LTR` / "learning-to-rank" (A3) | **`signature logistic`** | ini regresi logistik biasa (predict_proba), bukan objektif learning-to-rank. Nama lama menyesatkan (2.1) |
| tier "learning-to-rank/LTR" (`base.py`) | **"supervised"** | idem — bukan LTR |
| **"leave-one-out"** (evaluate.py, order_ablation.py, phase1_diag.py) | **"full-pool retrieval"** | query = parfum global, tak pernah ada di pool 340 produk → tak ada yang "ditinggalkan" (2.6, PEDOMAN §2.2) |
| B4a (tabel utama) | **B4b** (accord-only) | varian setara menggantikan varian asimetris (2.3) |

---

## 8. Reproduktibilitas (4.1 / 4.2)

- **Environment lock:** `environment_lock.txt` (`pip freeze`, versi persis). Kunci:
  numpy, scipy, scikit-learn, gensim 4.4.0, node2vec, networkx, rank-bm25,
  sentence-transformers, torch.
- **Python:** 3.13.3 (MSC v.1943, 64-bit AMD64).
- **OS:** Windows 11 (10.0.26200).
- **Seed:** stokastik `[0, 1, 2]`; bootstrap seed=0, 10.000 resample; GroupKFold
  di-shuffle per-seed (satu-satunya sumber variansi fold).
- **Commit hasil final:** `1fd523b` (`exp/audit-v2`).
- **Cara menjalankan ulang:**
  `python -m src.audit_step1` · `src.audit_a3_ablation` · `src.audit_b4_symmetry` ·
  `src.phase2_tuning` · `src.audit_step3`.

---

## 9. Kesimpulan menyeluruh

Memodelkan accord yang **muncul bersamaan** (order-3) meningkatkan retrieval dupe
secara **signifikan dan konsisten** dibanding memperlakukan accord sebagai independen
(order-1/B2) dan dibanding **10 dari 11** pembanding dari 4 paradigma — bahkan setelah
semua pembanding di-tuning adil. Terhadap A6 hasil campuran (tak diklaim menang), dan
terhadap varian metode sendiri (A3-full, P2) seri — karena keduanya memakai
representasi co-occurrence yang sama. Kontribusi co-occurrence terisolasi jelas: ~91%
skor order-3, dan menghilangkannya (A3-noco, order-1) menjatuhkan performa ke level
baseline marginal. Data terbukti bersih, sehingga sinyal ini nyata, bukan artefak.
