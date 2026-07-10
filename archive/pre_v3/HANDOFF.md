# HANDOFF — Tesis S2 Aromatique (Order-2 Co-occurrence Fragrance Retrieval)

Dokumen ini self-contained. Chat/akun baru cukup baca ini + lampiran untuk lanjut tanpa kehilangan konteks. Memori chat lama TIDAK ikut; semua konteks penting ada di sini.

---

## 1. SIAPA & APA

- Peneliti: mahasiswa S2 Magister Teknologi Informasi (UGM). Bahasa: Indonesia + istilah teknis Inggris. Mode `/caveman` sering dipakai (ringkas, padat).
- Produk akhir: **recommender parfum interaction-free** → output **Top-3 produk Aromatique** + **`kg_paths.json`** (untuk layer XAI kolaborator).
- Status: **eksperimen SELESAI, paper sudah didraf (Section I–VI/VII oleh Claude Code di Overleaf via Git)**. Tahap sekarang: revisi paper + finalisasi.

---

## 2. INTI PENELITIAN (yang sudah final, jangan diutak-atik)

**Tugas:** cross-reference dupe retrieval. Query = parfum global; jawaban benar = produk Aromatique yang `inspired_by` (kolom `revolutionize`) menunjuk query. Pool = 340 produk Aromatique (243 berlabel + 97 original = distraktor). Leave-one-out, 209 query berlabel.

**Metode usulan (P1) = Order-2 Co-occurrence TF-IDF:**
- Tiap parfum → graf ko-okurensi (accord = node, pasangan accord = edge).
- Token = accord (unigram) ∪ accord-pair (bigram), bobot IDF, L2-norm, cosine.
- Skor terdekomposisi: orde-1 (marginal) + orde-2 (ko-okurensi). Orde-2 = 81.3% skor top-rank.
- Parameter-free. `kg_paths` = edge bersama dikelompokkan per Edwards Wheel super-family.

**Novelty = FINDING/PROTOCOL, BUKAN model.** Applied/moderate (level S2). Klaim: demonstrasi non-sirkular pertama bahwa orde-2 co-occurrence mengalahkan marginal + neural-embedding + perceptual-taxonomy + learned (15 metode) di fragrance retrieval interaction-free.

---

## 3. ANGKA RESMI (dari results.csv — WAJIB pakai ini, JANGAN angka pilot lama 0.61)

| ID | Metode | MRR | H@1 | H@3 | ΔMRR vs P1 | Wilcoxon p |
|---|---|---|---|---|---|---|
| P1 | order-2 co-occ TF-IDF (usulan) | 0.496 | 0.378 | 0.562 | — | — |
| P2 | P1+LTR fusion | 0.500 | 0.379 | 0.580 | −0.005 | 0.18 n.s. |
| P3 | P1+hub-IDF ablation | 0.448 | 0.335 | 0.491 | +0.049 | 6.2e-10 |
| B1 | Jaccard | 0.428 | 0.303 | 0.487 | +0.069 | 6.6e-3 |
| B2 | TF-IDF cosine (BASELINE utama) | 0.454 | 0.335 | 0.507 | +0.042 | 8.7e-8 |
| B3 | BM25 | 0.388 | 0.278 | 0.433 | +0.108 | 8.3e-12 |
| B4 | Sentence-BERT | 0.272 | 0.163 | 0.311 | +0.225 | 1.6e-11 |
| B5 | Word2Vec | 0.185 | 0.110 | 0.182 | +0.294 | 2.0e-16 |
| B6 | node2vec | 0.340 | 0.233 | 0.371 | +0.181 | 2.7e-11 |
| A1 | wheel tree-Wasserstein | 0.284 | 0.174 | 0.306 | +0.213 | 3.9e-13 |
| A2 | PPMI+SVD | 0.444 | 0.343 | 0.498 | +0.054 | 3.9e-3 |
| A3 | signature LTR | 0.433 | 0.303 | 0.500 | +0.066 | 5.7e-8 |
| A4 | bigram salience | 0.139 | 0.060 | 0.145 | +0.367 | 2.7e-21 |
| A5 | bilinear metric | 0.150 | 0.072 | 0.150 | +0.332 | 2.5e-21 |
| A6 | GBM fusion | 0.457 | 0.325 | 0.533 | +0.052 | 2.3e-2 |

**Headline:** P1 vs B2 → ΔMRR=+0.042, 95% CI [+0.019, +0.067], p=8.7e-8.
**Data:** 340 produk, 209 query, 56 shared accord, 17 family lokal / 49 global, provenance Fragrantica 264/266 source_url.

---

## 4. CAVEAT JUJUR (WAJIB dipertahankan, jangan di-overclaim)

1. Efek **modest** (+0.042 MRR) tapi robust & signifikan. 86% overlap query-target = batas atas perbaikan (sinyal nyata, bukan artefak).
2. **B2 baseline** = lantai tinggi. P1 menang tipis tapi konsisten.
3. **A2 (0.444) match baseline; A6 (0.457) sedikit DI ATAS baseline B2 (0.454)** — keduanya tetap signifikan di bawah P1. Jangan bilang "A6 cuma match baseline" (salah, A6 > B2).
4. **B4 (S-BERT) kalah sebagian karena asimetri teks** (query miskin teks vs produk prosa Indonesia), bukan murni semantik gagal.
5. **Matcher sederhana BY DESIGN** — model kompleks (A4/A5/P2) overfit terbukti. Simplicity = keputusan, bukan kelalaian.
6. **P2 ≈ P1 (p=0.18 n.s.)** → P1 metode resmi, P2 ablation. P2 numeris 0.500 > P1 0.496 tapi tak signifikan.
7. Novelty = finding/protocol, BUKAN model. Generalisasi terbatas (1 katalog proprietary).

---

## 5. KEPUTUSAN PENTING (kenapa begini — biar tak diulang debat)

- **Wheel-transport DIBUANG sebagai ranker** (kalah, 0.284) → wheel hanya dipakai untuk menata `kg_paths` (kelompok per super-family).
- **Path A (data intensitas accord) MATI** — produk Aromatique tak bisa dapat intensitas (katalog riil fixed). Maka novelty model mustahil → ambil Path B (finding/protocol).
- **KGAT/LightGCN/KGIN TIDAK dibandingkan** — butuh user-item interaction, data tak punya. Dinyatakan eksplisit di paper (bukan ditutupi).
- **8 model "advance" sudah diuji & kalah** (wheel, PPMI, signature, supervised salience, bilinear, GBM, dst) → jadi feasibility study / ablation.

---

## 6. STATUS PAPER & TUGAS BERIKUTNYA

- Template: **IEEE Conference (IEEEtran, conference, 2-kolom)**.
- Workflow: **Jalur B** — Claude Code tulis `.tex` di repo Git ter-sync Overleaf premium. Section-by-section, 1 commit+push per section, user pull manual di Overleaf.
- Draf paper Section I–VII sudah ada (judul: "Order-2 Co-occurrence TF-IDF for Interaction-Free Perfume Retrieval", penulis Azam Azri).
- **REVISI yang sudah teridentifikasi (belum dikerjakan):**
  1. **Abstract kontradiksi:** klaim P1 "best MRR (0.496)" padahal P2=0.500. Ganti → "best among parameter-free / statistically tied". (PRIORITAS — cacat kredibilitas abstract)
  2. **A6 (0.457) > B2 (0.454):** revisi kalimat "A2/A6 match baseline" → "A2 matches, A6 marginally exceeds baseline, both significantly below P1".
  3. **"Faithful by construction"** terlalu kuat — lunakkan, faithfulness kuantitatif = future work (counterfactual edge-ablation, ranah kolaborator).
  4. **Placeholder:** affiliation "(to be completed)" + venue ref [3] Fannisa tak jelas → beresin.
  5. Pertimbangan venue: kejujuran ("modest, single catalog, parameter-free") aman untuk SIDANG tapi berisiko untuk conference kompetitif — keputusan user.

---

## 7. FILE YANG HARUS DILAMPIRKAN DI CHAT BARU

- `HANDOFF.md` (file ini)
- `BLUEPRINT_V5.md` (otak: RQ, 9-langkah math, novelty, 28 sitasi + link Consensus)
- `IMPLEMENTATION_REPORT.md` (hasil + kg_paths generation)
- `results.csv`, `significance.csv` (angka tabel)
- `kg_paths_sample.json` (contoh output)
- (opsional) 2 file dataset: `datasetaromatique.xlsx`, `global_reference.xlsx`
- (opsional) draf paper `.tex` terakhir dari repo

---

## 8. ATURAN MAIN (untuk asisten di chat baru)

- Pakai angka results.csv (§3), JANGAN angka pilot lama.
- Pertahankan 7 caveat jujur (§4) — jangan biarkan draf overclaim.
- Mode kritis (reviewer), bukan pemuji. Surface kelemahan dulu.
- Bahasa Indonesia + istilah teknis Inggris. `/caveman` kalau diminta.
- Workflow paper: section-by-section, commit+push per section, tunggu approval, user pull di Overleaf.
- 28 referensi (mayoritas Q1/Q2) + link Consensus ada di BLUEPRINT_V5 §12.
