# HANDOFF — Tesis S2: Higher-Order Accord Co-occurrence

Dokumen ini ditulis untuk memindahkan konteks penuh ke percakapan baru.
Lampirkan file ini + `PEDOMAN_EKSPERIMEN.md` + hasil Langkah 1–4 di chat baru.

---

## 0. Cara kerja yang disepakati (BACA DULU)

1. **Paper BUKAN acuan.** Paper yang ada dianggap berisi temuan yang belum tentu benar.
   Acuan tunggal adalah `PEDOMAN_EKSPERIMEN.md` dan hasil eksperimen.
2. **Eksperimen difinalkan dulu, paper ditulis belakangan** mengikuti hasil.
   Bukan sebaliknya.
3. **Jangan mengarang** angka, referensi, venue, atau nama. Kalau tidak terverifikasi,
   katakan "belum terverifikasi".
4. **Selisih angka tidak boleh diklaim tanpa uji signifikansi.**
5. **Peran:** Claude (chat) = analisis, review kritis, menyusun brief.
   Claude Code = menjalankan eksperimen dan menulis `.tex`. Chat tidak menjalankan
   eksperimen.
6. Bahasa: Indonesia + istilah teknis Inggris. Respons ringkas, tanpa validasi kosong.

---

## 1. Konteks penelitian

- **Peneliti:** Azam Azri Ahmad (S2, UGM). Pembimbing: Syukron Abu Ishaq Alfarozi,
  Bimo Sunarfri Hantono. Kolaborator KMITL Thailand: Kuntpong Woraratpanya.
- **Mitra:** Aromatique, brand parfum Indonesia. Katalog 340 produk.
- **Kolaborator:** Raissa — membangun chatbot (HCI + XAI). Sistem rekomendasi ini
  memberi output `{top3_products, kg_paths}` ke chatbot-nya.
- **Masalah nyata:** jam ramai di cabang → staf sedikit, antrean panjang.
  Chatbot + recommender untuk mengurangi bottleneck.

**Pertanyaan penelitian:**
Apakah memodelkan **accord yang muncul bersamaan** (co-occurrence) meningkatkan
kualitas rekomendasi dibanding memperlakukan accord sebagai **independen**?

**RQ final (3):**
- RQ1: Apakah co-occurrence (order ≥ 2) meningkatkan retrieval dibanding order-1?
- RQ2: Sampai order berapa peningkatannya berlanjut — adakah titik jenuh?
- RQ3: Bagaimana dibanding paradigma mapan (marginal, neural, perceptual, learned)?

RQ tentang explainability **sengaja tidak dibuat** — `kg_paths` adalah domain
kolaborator, dan faithfulness-nya belum diukur.

---

## 2. Repo

`https://github.com/azamazri/order3` (publik).
Bisa di-clone penuh: `git clone https://github.com/azamazri/order3.git`
(**jangan** `--depth 1` — nanti branch lain tidak ikut).

Branch:
- `main` — masih berisi hasil order-2 **lama**. Belum di-merge.
- `exp/order-ablation` — kurva order-N, signifikansi tangga order
- `audit/citations` — `verification_report.md`
- `exp/fair-baselines` — Fase 1 (diagnostik) + Fase 2 (bugfix A5, nested-CV tuning)
- `exp/audit-v2` — Langkah 1–4 (baru dijalankan, **belum dianalisis**)

---

## 3. Tugas dan harness (ringkas — detail di PEDOMAN)

- Pool: 340 produk. Query: 209 parfum global berlabel.
- Label: relasi `inspired_by` (produk mana meniru global mana), disembunyikan.
- Tiap query: rank seluruh 340 produk, lihat peringkat jawaban benar.
- **Bukan leave-one-out.** Query tidak pernah ada di pool. Nama yang benar:
  **full-pool retrieval**. Istilah lama salah.
- **Aturan seri wajib**: expected reciprocal rank di bawah pemecahan seri seragam.
  (B1 Jaccard meninggalkan ~16 kandidat seri di puncak; tanpa koreksi ini ia dapat
  bonus palsu.)
- Metrik: MRR, Hits@1, Hits@3.
- Signifikansi: Wilcoxon signed-rank berpasangan **antar-query** + bootstrap 95% CI
  (10.000 resample atas query).
- Supervised (A3, A4, A5, A6) dinilai out-of-fold, GroupKFold-5 by query.

---

## 4. Metode usulan: order-N

Token = semua subset accord ukuran 1..N (tak berurut).
IDF = `log((1+340)/(1+df))+1`, **df dihitung dari 340 produk saja**.
Vektor biner × idf → L2-norm → cosine.
Skor terdekomposisi **eksak**: `s = S₁ + S₂ + ... + S_N` (identitas aljabar).
**Parameter-free**: tidak ada hyperparameter, tidak dilatih, tidak melihat label.

Validasi wajib: `N=1` harus mereproduksi B2 (0.4540); `N=2` harus mereproduksi
order-2 (0.4963). Ada `assert` di kode.

**Implementasi order-N sudah diverifikasi BENAR** terhadap rumus.

---

## 5. Angka yang sudah ada (per akhir Fase 2)

### Order ablation (sebelum tuning pembanding)

| N | MRR | Hits@1 | Hits@3 |
|---|---|---|---|
| 1 | 0.454 | 0.335 | 0.507 |
| 2 | 0.496 | 0.378 | 0.562 |
| **3** | **0.508** | **0.397** | **0.567** |
| 4 | 0.511 | 0.397 | 0.583 |
| 5 | 0.511 | 0.397 | 0.583 |
| 6 | 0.511 | 0.397 | 0.583 |

Signifikansi tangga order:
- order-2 vs 3: ΔMRR −0.0116, CI [−0.023, −0.002], p = 0.0028 → **order-3 menang**
- order-2 vs 4: ΔMRR −0.0142, CI [−0.027, −0.002], p = 0.0005 → order-4 menang
- order-3 vs 4: ΔMRR −0.0025, CI [−0.010, **+0.005**], p = 0.017 → **CI memuat 0 = SERI**

→ **order-3 dipilih** karena setara order-4 tapi lebih sederhana (parsimony).
Kurva **naik lalu jenuh**, bukan naik lalu turun → sinyal nyata yang menjenuh,
bukan overfitting.

Sparsity (rata-rata token orde-k bersama query–target):
k1 4.20 | k2 7.80 | k3 7.68 | **k4 4.12 | k5 1.22 | k6 0.27**
→ plateau di orde tinggi karena tuple langka. Triple **tidak** sparse.

Dekomposisi skor order-3: order-1 **9%**, order-2 **37%**, order-3 **54%**.

### Hasil setelah pembanding di-tuning adil (nested CV, Fase 2)

| Metode | MRR lama | **MRR tuned** |
|---|---|---|
| **A3 signature** | 0.433 | **0.5131** |
| **order-3 (usulan)** | 0.508 | **0.5080** |
| A6 GBM | 0.457 | 0.4746 |
| B2 TF-IDF | 0.454 | 0.454 |
| A2 PPMI–SVD | 0.444 | 0.4431 |
| B5 Word2Vec | 0.185 | **0.4018** |
| B3 BM25 | 0.388 | 0.3919 |
| B6 node2vec | 0.340 | 0.3602 |
| A1 wheel | 0.284 | 0.3577 |
| A4 bigram | 0.139 | 0.2890 |
| A5 bilinear | 0.150 | 0.2358 |

**Klaim lama "order-3 ranks first on every metric" SUDAH MATI.**

Train/test gap setelah tuning:
- A3: 0.515 / 0.513 → gap +0.002, **tidak overfit**
- A6: 0.552 / 0.475 → gap +0.078, ringan
- A4: 0.444 / 0.289 → gap +0.155, overfit
- A5: 0.502 / 0.236 → gap +0.266, overfit parah
- A2, B5, B6 unsupervised; B4 pretrained → **tidak bisa overfit** (train ≡ test)

---

## 6. Temuan kritis tentang implementasi

### A3 BUKAN pembanding independen
Fitur A3 mencakup `n_shared_b`, `w_shared_b`, `max_rare` — semuanya dihitung dari
**pasangan accord bersama ber-IDF**, yaitu **representasi order-2 metode usulan**.
A3 = metode kita + regresi logistik. Ia **varian**, bukan lawan.
Nama "signature LTR" juga salah — ini regresi logistik biasa, bukan learning-to-rank.

**Implikasi:** A3 mengalahkan order-3 **bukan** dengan menyangkal co-occurrence,
melainkan dengan **membobotinya secara terpelajar**.

### A5 rusak, bukan overfit (sudah diperbaiki)
Versi lama: plain GD, `lr=0.5`, 300 iter. Loss `0.6931 → 0.6378`.
`0.6931 = ln(2)` = loss tebakan acak. Train MRR 0.184 — **lebih buruk dari B2 di data
latihnya sendiri**. Model yang tak bisa memuat data latihnya bukan overfit; ia rusak.
Class balancing **sudah ada** — bukan penyebabnya. Penyebabnya optimizer.
Diperbaiki dengan L-BFGS-B → konvergen, train MRR 0.387.

### B4 mengerjakan tugas yang berbeda
Satu-satunya metode yang meng-encode **prosa Indonesia** di sisi produk dan
**daftar accord Inggris** di sisi query. Sebelas lainnya hanya melihat accord.
Perlu varian B4b (accord-only kedua sisi) sebagai perbandingan setara.

### B6 "node2vec" dengan p=q=1 = DeepWalk
Versi lama tak menyetel `p`/`q`. Tuning Fase 2 memilih `p=0.25, q=4` → sudah benar.

### Docstring kode menuliskan ekspektasi hasil
`"expected to OVERFIT badly (this is the point)"` (A4),
`"overfit demo... Included to show that a flexible learner does not beat the
simple order-2 cosine"` (A6), `"expected to LOSE"` (A1).
Ini bias eksperimen dan bisa dibaca reviewer di repo publik. Harus dihapus.

### Asimetri akses data
order-N: IDF dari 340 produk saja.
B5, B6, A2: di-fit atas **produk + query**. Mereka melihat lebih banyak data.
Bukan kebocoran label, tapi harus dinyatakan.

### Asumsi sirkularitas yang belum teruji (PRIORITAS TERTINGGI)
`src/evaluate.py` baris 40 & 55 (docstring **kode**, bukan paper) menyatakan accord
global "evidently derived from the dupes" → overlap mentah "near-circular".
**Belum pernah diuji.** Langkah 1 dari audit menguji ini.
Kalau benar → semua angka mengukur artefak.
Kalau salah → komentar itu harus dihapus dari kode.

---

## 7. Status sekarang

**Langkah 1–4 sudah dijalankan** di branch `exp/audit-v2`. Hasilnya **belum dianalisis**.

Yang harus dibawa ke chat baru:
- `results/audit/step1_report.md` — **bocor atau bersih?** Ini menentukan segalanya.
- `results/audit/a3_ablation.csv` — A3-full / A3-noco / A3-conly
- `results/audit/b4_symmetry.csv` — B4a / B4b / B4c
- `results/audit/final_table.csv`
- `results/audit/final_significance.csv`
- `results/audit/final_gap.csv`
- `results/audit/FINAL_REPORT.md`

---

## 8. Pertanyaan yang harus dijawab di chat baru

1. **Apakah data bocor?** (Langkah 1) Kalau ya, berhenti dan rombak dataset.
2. **A3-noco jatuh ke level B2 (~0.454)?** Kalau ya → bukti langsung bahwa yang bekerja
   adalah co-occurrence, bukan regresi logistiknya. Ini eksperimen terpenting.
3. **Apakah selisih A3 (0.5131) vs order-3 (0.5080) signifikan?**
   Selisih 0.0051. Kalau **tidak signifikan** → order-3 setara tapi **parameter-free**;
   klaim parsimony (pola yang sama seperti P1 vs P2 dan order-3 vs order-4).
   Kalau **signifikan** → A3 memang lebih baik, laporkan apa adanya.
4. **B4b jauh di atas B4a?** Kalau ya → skor rendah B4 lama akibat mismatch teks.
   Kalau setara → embedding memang gagal, dan pembelaan "text mismatch" harus dicabut.
5. Apakah kurva order-N berubah setelah harness final?

---

## 9. Narasi yang kemungkinan besar akan jadi kesimpulan

**Yang lama (sudah tidak benar):**
"Metode parameter-free sederhana mengalahkan semua model kompleks."

**Yang didukung data sejauh ini:**
> Sinyal yang menentukan adalah **co-occurrence accord**, bukan kapasitas model.
> Setelah semua pembanding di-tuning adil, tiga metode teratas semuanya mengonsumsi
> co-occurrence (A3 0.513, order-3 0.508, A6 0.475), sementara baseline order-1
> tertinggal (B2 0.454). Matcher order-3 mencapai **kinerja setara** dengan yang
> terbaik **tanpa parameter terlatih, tanpa label, tanpa training**.
> Model terlatih yang mencoba mempelajari bobot co-occurrence dari label langka
> (A4, A5) justru overfit parah.

Ini klaim **efisiensi dan parsimony**, bukan superioritas. Lebih defensible.

**Catatan penting:** kalau A3 ternyata signifikan lebih baik, itu **tidak membunuh
tesis**, karena A3 memakai fitur order-2 kita. Yang runtuh hanya klaim
"parameter-free lebih baik". Tesis inti (co-occurrence adalah sinyalnya) justru
**menguat** — semua metode teratas memakainya.

---

## 10. Utang teknis yang belum lunas

- `main` masih berisi hasil order-2 lama. Belum di-merge.
- Tidak ada lockfile; `requirements.txt` hanya `>=`.
- P3 dijalankan tapi tak pernah dilaporkan → selective reporting.
- Paper (`conference_101719.tex`) masih memuat:
  - "Fannisa et al. [3]" padahal bibliografi mencetak "Aurora and Baizal"
    ("Fannisa" nama depan; marganya Aurora; hanya 2 penulis)
  - LightGCN & KGIN disebut 3× tapi disitasi ke KR-GCN [4] dan KEGNN [5].
    Verifikasi kode: keduanya **nol kemunculan** — memang tak pernah diimplementasikan.
  - A1 diklaim "following graph-based optimal transport [22],[23]" padahal kode
    memakai **tree-W1 tulisan tangan**, tanpa library OT, tanpa Sobolev/tree-sliced.
  - Istilah "leave-one-out" (seharusnya full-pool retrieval)
- `reference.bib` sudah diperbaiki (33 entri, semua metadata dari publisher).
  Koreksi yang sudah dilakukan: 4 nama penulis salah orang, 2 venue salah,
  5 tahun salah, 8 entri "and others" padahal daftar penulis lengkap.

---

## 11. Kalimat pembuka untuk chat baru

> Lanjutan tesis S2. Baca HANDOFF_V2.md (terlampir) + PEDOMAN_EKSPERIMEN.md untuk
> konteks penuh. Langkah 1–4 audit eksperimen sudah dijalankan; hasilnya saya lampirkan.
> Paper BUKAN acuan — eksperimen difinalkan dulu, paper ditulis belakangan.
> Mulai dari: analisis hasil Langkah 1 (bocor atau bersih).
