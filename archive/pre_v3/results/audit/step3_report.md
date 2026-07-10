# Langkah 3 -- Re-run Final + Signifikansi

Acuan: PEDOMAN_EKSPERIMEN.md Bagian 3, 5, 8.2. Satu harness identik untuk semua
metode (`src/audit_step3.py`), dijalankan setelah Langkah 2 di-review.
Seed stokastik `[0,1,2]`, bootstrap 10.000 resample atas query, Wilcoxon
signed-rank berpasangan antar-query atas reciprocal rank.

**Konfigurasi:** best_params dibawa dari Fase 2 nested CV (`results/rerun/tuned_results.csv`);
metode **tidak di-tuning ulang** — hanya dievaluasi sekali pada config final itu di
harness ini, sehingga ΔMRR pada tabel signifikansi konsisten persis dengan tabel utama.

> Catatan jumlah pembanding: PEDOMAN Langkah 3.1 menyebut "12 pembanding", tetapi
> Langkah 2.1 memindahkan **A3 ke blok varian**. Jadi blok pembanding independen kini
> **11** (B1, B2, B3, B4b, B5, B6, A1, A2, A4, A5, A6); A3-full, P2, P3 dilaporkan
> sebagai varian.

## 3.2 Validasi order-N (WAJIB)

| cek | nilai | referensi | status |
|---|---|---|---|
| N=1 == B2 | 0.4540 | 0.4540 | OK (<=1e-3) |
| N=2 == order-2 (P1) | 0.4963 | 0.4963 | OK (<=1e-3) |

Lolos. Skrip lanjut menulis CSV.

## 3.1 Tabel final (`final_table.csv`)

| method | kategori | best_params | MRR | ±std | Hits@1 | Hits@3 |
|---|---|---|---|---|---|---|
| **order3** | **proposed** | parameter-free | **0.5080** | 0.000 | 0.397 | 0.567 |
| order1 | proposed | parameter-free | 0.4540 | 0.000 | 0.335 | 0.507 |
| order2 | proposed | parameter-free | 0.4963 | 0.000 | 0.378 | 0.562 |
| order4 | proposed | parameter-free | 0.5105 | 0.000 | 0.397 | 0.583 |
| order5 | proposed | parameter-free | 0.5109 | 0.000 | 0.397 | 0.583 |
| order6 | proposed | parameter-free | 0.5109 | 0.000 | 0.397 | 0.583 |
| A3_signature_full | variant | {'C': 0.01} | 0.5131 | 0.003 | 0.407 | 0.565 |
| P2_fusion | variant | fixed logistic | 0.4984 | 0.002 | 0.376 | 0.581 |
| P3_hubidf | variant | parameter-free | 0.4476 | 0.000 | 0.335 | 0.490 |
| A6_gbm_fusion | comparator | n_est300,depth5,lr0.01 | 0.4771 | 0.007 | 0.354 | 0.551 |
| B2_tfidf_cos | comparator | parameter-free | 0.4540 | 0.000 | 0.335 | 0.507 |
| A2_ppmi_svd | comparator | {'rank': 50} | 0.4431 | 0.001 | 0.342 | 0.498 |
| B1_jaccard | comparator | parameter-free | 0.4277 | 0.000 | 0.303 | 0.487 |
| B5_word2vec | comparator | dim64,win5,ep200 | 0.3934 | 0.012 | 0.277 | 0.450 |
| B3_bm25 | comparator | k1=2.0,b=0.75 | 0.3919 | 0.000 | 0.282 | 0.440 |
| B6_node2vec | comparator | dim32,p0.25,q4 | 0.3605 | 0.021 | 0.248 | 0.402 |
| A1_wheel_treeW | comparator | w=(1,1,1) | 0.3577 | 0.000 | 0.240 | 0.392 |
| B4b_sbert_accord | comparator | parameter-free | 0.3439 | 0.000 | 0.230 | 0.378 |
| A4_bigram_salience | comparator | {'C': 0.01} | 0.2963 | 0.011 | 0.184 | 0.328 |
| A5_bilinear | comparator | rank16,iters1000 | 0.2358 | 0.014 | 0.131 | 0.260 |

Pembanding terbaik = **A6 (0.4771)**. order-3 (0.5080) di atas semua pembanding.

## 3.3 Signifikansi (`final_significance.csv`)

Referensi = order-3; ΔMRR = MRR(order3) − MRR(lawan). Signifikan bila
Wilcoxon p < 0.05 **dan** 95% CI tidak memuat 0.

### (a) order-3 vs 11 pembanding

| lawan | ΔMRR | CI | Wilcoxon p | verdict |
|---|---|---|---|---|
| A5_bilinear | +0.2722 | [0.222, 0.323] | 3.5e-19 | order-3 menang |
| A4_bigram_salience | +0.2117 | [0.158, 0.265] | 2.5e-12 | order-3 menang |
| B4b_sbert_accord | +0.1641 | [0.110, 0.218] | 1.9e-08 | order-3 menang |
| A1_wheel_treeW | +0.1503 | [0.100, 0.200] | 4.6e-09 | order-3 menang |
| B6_node2vec | +0.1475 | [0.103, 0.193] | 6.6e-09 | order-3 menang |
| B3_bm25 | +0.1161 | [0.081, 0.153] | 2.2e-12 | order-3 menang |
| B5_word2vec | +0.1145 | [0.071, 0.160] | 1.4e-06 | order-3 menang |
| B1_jaccard | +0.0803 | [0.051, 0.111] | 6.9e-04 | order-3 menang |
| A2_ppmi_svd | +0.0648 | [0.022, 0.109] | 4.3e-04 | order-3 menang |
| B2_tfidf_cos | +0.0540 | [0.029, 0.080] | 5.6e-09 | order-3 menang |
| **A6_gbm_fusion** | **+0.0308** | **[0.008, 0.054]** | **0.141** | **CAMPURAN** |

**A6 dilaporkan apa adanya:** ΔMRR positif dan bootstrap CI **tidak** memuat 0,
tetapi Wilcoxon **tidak signifikan** (p=0.14). Jadi keunggulan order-3 atas A6
**tidak kokoh** — kedua uji tidak sepakat. order-3 tetap unggul angka (0.508 vs
0.477) dan menang telak atas **10 pembanding lain**. (A6 sendiri sebagian
mengonsumsi fitur order-2: cosine bigram — lihat PEDOMAN §4/A6.)

### (b) order-3 vs A3-full (kritis)

ΔMRR = **−0.0051**, CI [−0.020, 0.009], Wilcoxon **p=0.891** → **TIDAK signifikan**.

Selisih 0.005 yang tampak di tabel **bukan** kemenangan A3: secara statistik
**seri**. Digabung dengan ablation Langkah 2.2 (A3 runtuh ke 0.422 tanpa fitur
co-occurrence, di bawah B2), kesimpulannya konsisten: A3 adalah **metode usulan +
regresi logistik**, dan edge kecilnya berasal dari representasi co-occurrence yang
sama — bukan pembanding independen yang mengalahkan order-N.

### (c) order-3 vs P2, P3

| lawan | ΔMRR | CI | Wilcoxon p | verdict |
|---|---|---|---|---|
| P2_fusion (varian) | +0.0096 | [−0.002, 0.023] | 0.066 | seri (tak signifikan) |
| P3_hubidf (varian) | +0.0604 | [0.036, 0.086] | 1.1e-10 | order-3 menang |

### (d) Tangga orde (ΔMRR = MRR(N) − MRR(N+1))

| pasangan | ΔMRR | CI | Wilcoxon p | verdict |
|---|---|---|---|---|
| order1 vs order2 | −0.0424 | [−0.067, −0.019] | 8.7e-08 | order-2 lebih baik (signifikan) |
| order2 vs order3 | −0.0116 | [−0.023, −0.002] | 2.8e-03 | **order-3 lebih baik (signifikan)** |
| order3 vs order4 | −0.0025 | [−0.010, 0.005] | 1.7e-02 | marginal; CI memuat 0 → tak kokoh |
| order4 vs order5 | −0.0005 | [−0.001, 0.000] | 5.3e-02 | dapat diabaikan |
| order5 vs order6 | 0.0000 | [0, 0] | 1.000 | identik |

**Kurva jenuh setelah orde-3.** Lompatan terakhir yang signifikan penuh adalah
order-2 → order-3; order-3 → order-4 sudah marginal (CI memuat 0). order-3 adalah
titik knee yang dibenarkan secara statistik.

## 3.4 Train/test gap (`final_gap.csv`)

| method | train MRR | test MRR | gap | catatan |
|---|---|---|---|---|
| A3_signature_full | 0.5148 | 0.5131 | 0.0017 | supervised (out-of-fold) |
| A6_gbm_fusion | 0.5810 | 0.4771 | 0.1039 | supervised (out-of-fold) |
| A4_bigram_salience | 0.4754 | 0.2963 | 0.1792 | supervised — gap besar (overfit) |
| A5_bilinear | 0.5022 | 0.2358 | 0.2664 | supervised — gap besar (overfit) |
| A2, B5, B6 | — | (test) | — | train==test menurut konstruksi (label-free) → tak bisa overfit |
| B4b | — | 0.3439 | — | pretrained, tanpa fit → tak bisa overfit |

A4 dan A5 punya gap train-test besar (0.18 dan 0.27) → overfit nyata pada ruang
fitur besar dengan label sedikit. A3 hampir tanpa gap. Metode unsupervised/pretrained
tidak bisa overfit.

## 3.5 Dekomposisi order-3 (`final_decomposition.csv`)

Rata-rata share skor produk peringkat teratas:

| orde | share |
|---|---|
| 1 | 8.7% |
| 2 | 37.2% |
| 3 | 54.2% |

Co-occurrence (orde ≥2) menyumbang **~91%** skor. Sinyal marginal (orde-1) hanya 8.7%.

## 3.6 Diagnostik sparsity (`final_sparsity.csv`)

Rata-rata jumlah token orde-k yang dimiliki bersama query dan produk jawaban benar:

| k | avg shared |
|---|---|
| 1 | 4.20 |
| 2 | 7.80 |
| 3 | 7.68 |
| 4 | 4.12 |
| 5 | 1.22 |
| 6 | 0.27 |

Token bersama memuncak di k=2/3 lalu turun tajam; di k=5,6 nyaris tak ada token
bersama → menjelaskan kenapa MRR mendatar setelah orde-3.

---

## KESIMPULAN

- **order-3 mengalahkan 10 dari 11 pembanding secara signifikan** (Wilcoxon p<0.05
  dan CI eksklusi 0). Terhadap **A6** hasilnya **campuran** (CI eksklusi 0 tapi
  Wilcoxon p=0.14) — dilaporkan apa adanya, tidak diklaim menang.
- **Terhadap varian metode usulan (A3-full, P2) order-3 statistik seri.** Selisih
  +0.005 A3 tidak signifikan (p=0.89); digabung ablation 2.2, A3 bukan pembanding
  yang mengalahkan order-N melainkan varian dengan representasi co-occurrence yang sama.
- **order-3 adalah knee yang dibenarkan:** order-2→order-3 signifikan, order-3→order-4
  sudah tak kokoh (CI memuat 0), sisanya jenuh.
- **Co-occurrence menyumbang ~91% skor** order-3; sparsity menjelaskan plateau.
- **A4/A5 overfit** (gap train-test besar); metode label-free/pretrained tak bisa overfit.

Semua angka ter-commit. Langkah 3 berhenti di sini menunggu review sebelum Langkah 4.
