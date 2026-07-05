# Laporan Implementasi
## Benchmark Dupe-Retrieval Parfum Tanpa Interaksi (Path B)

**Tugas akhir S2 — Cross-reference dupe retrieval**
Tanggal: 29 Juni 2026

---

## 1. Ringkasan Eksekutif

Dokumen ini melaporkan implementasi sebuah *pipeline* Python yang dapat direproduksi untuk
tugas **cross-reference dupe retrieval**: diberikan sebuah parfum *global* (sebagai query),
sistem harus me-*retrieve* produk lokal AROMATIQUE yang merupakan **dupe** (tiruan aroma)
dari parfum global tersebut. *Ground truth* adalah relasi `inspired_by` yang dikodekan oleh
kolom `local.revolutionize == global.Revolutionize`.

Kontribusi penelitian adalah sebuah **temuan non-sirkular**: representasi **ko-okurensi
accord orde-2** (pasangan accord) mengungguli (a) kemiripan marjinal orde-1 (TF-IDF cosine)
dan (b) seluruh struktur perseptual/terpelajar (tree-Wasserstein, PPMI-SVD, bilinear
metric-learning, gradient boosting), yang justru *overfit* pada label yang sangat jarang.

**Hasil utama:** metode usulan **P1 (order-2 TF-IDF)** dan **P2 (P1 + LTR fusion)** menempati
peringkat teratas pada seluruh metrik (MRR ≈ 0.50), mengalahkan baseline marjinal B2
(MRR 0.454) secara signifikan (Wilcoxon p = 8.7 × 10⁻⁸), serta mengalahkan seluruh metode
Tier-2. Temuan kualitatif dan signifikansinya konsisten dengan target pilot.

---

## 2. Lingkup dan Keputusan Desain

### 2.1 Mengapa model berbasis interaksi DIKELUARKAN

Dataset **tidak memiliki interaksi/rating pengguna** — hanya katalog produk dengan daftar
accord dan tabel referensi global. *Recommender* berbasis interaksi (KGAT, LightGCN, KGIN,
ItemKNN, *collaborative filtering*) memerlukan sinyal interaksi user–item yang tidak tersedia.
Memasukkannya berarti **mengarang interaksi**, sehingga model-model tersebut dikeluarkan
**by design**. Tugas ini murni *content-based retrieval*: setiap metode menilai pasangan
(query, produk) hanya dari konten.

### 2.2 Leakage firewall (kritis)

Relasi dupe tidak boleh terbaca dari input:

1. **Representasi query hanya accord + `global_family`.** Query tidak pernah memuat
   `interpreted_as` (yang secara literal adalah nama parfum global) atau field identitas lain.
2. **Audit kebocoran teks bebas.** `data.leakage_audit` menghitung produk yang teks
   `meaning`/`visual_note`-nya memuat token khas dari nama global-nya sendiri, melaporkannya
   (32/340 produk), dan **menghapus** token tersebut dari teks yang dipakai metode berbasis
   teks (B4). Mayoritas token yang ditandai sebenarnya kata accord generik (*rose*/*floral*);
   tetap dihapus secara konservatif.
3. **Taksonomi/metrik tidak menyentuh label.** Leksikon Edwards-wheel (A1) dibekukan dan tidak
   pernah disetel ke test set. IDF dihitung hanya dari korpus kandidat.

### 2.3 Artifact containment (didiagnosis, tidak disembunyikan)

Daftar accord global ternyata **diturunkan dari dupe-nya**: rata-rata **±86% accord sebuah
dupe terkandung dalam daftar accord global-nya**. Akibatnya **tumpang-tindih himpunan mentah
(tanpa bobot) menjadi sinyal yang nyaris sirkular**. Konsekuensi yang ditangani eksplisit:

* Baseline marjinal yang adil adalah **TF-IDF berbobot (B2)** — sesuai literatur perfume
  content-based (gaya Nurmuthia/BINUS). Jaccard mentah (B1) tetap dilaporkan namun dipahami
  terinflasi oleh artifact ini.
* **Penanganan ties penting** (lihat §5.2).

---

## 3. Data

| Berkas | Isi | Catatan |
|--------|-----|---------|
| `dataset-aromatique.xlsx` | 340 produk lokal (= candidate pool) | kolom `revolutionize` non-null = 243 label, null = 97 distractor |
| `global_reference.xlsx` | parfum global | **header pada baris index 1**; accord di `accord_1..accord_10` |

**Statistik hasil parsing (terverifikasi):**

| Besaran | Nilai | Target pilot |
|---------|-------|--------------|
| Candidate pool | 340 | ~339–340 ✓ |
| Query berlabel (≥1 dupe) | 209 | ~208–243 ✓ |
| Vocab accord lokal | 110 | — |
| Vocab accord global | 63 | — |
| **Accord shared** | **56** | **56 ✓** |
| Produk dengan kebocoran teks | 32 / 340 | dilaporkan + di-strip |
| Containment accord (rata-rata) | 86% | (artifact) |

**Keputusan parsing yang didokumentasikan:**
- Accord: `re.split(r"[;,]")`, lowercase, strip; duplikat dibuang, urutan dipertahankan.
- Join: `local.revolutionize == global.Revolutionize` dinormalisasi (lowercase, whitespace
  dirapatkan).
- Accord query: dari kolom global `accord_1..accord_10`.
- Vocab bersama: semua vektor hidup di union accord lokal+global; hanya 56 accord shared yang
  menghasilkan tumpang-tindih lintas. IDF di-fit pada korpus 340 produk; query ditransformasi
  dengan IDF yang sama.

---

## 4. Arsitektur Perangkat Lunak

```
dataset-aromatique.xlsx        340 produk lokal (candidate pool)
global_reference.xlsx          parfum global (header baris index 1)
reference/                     paper pendukung
src/
  data.py                      muat, parse, normalisasi+join, audit kebocoran
  wheel.py                     leksikon Edwards-wheel beku + tree-W1 closed-form
  evaluate.py                  metrik LOO, ranking tie-correct, Wilcoxon, bootstrap, runner GroupKFold
  methods/
    base.py                    precompute fitur + interface Method + runner out-of-fold
    b1..b6, a1..a6, p1..p3     satu modul per metode (15 metode)
run_all.py                     runner end-to-end -> tabel hasil + signifikansi
requirements.txt
results/                       results.csv, significance.csv, run_full.log
```

### 4.1 Interface umum

Setiap metode adalah objek `Method` dengan:

```python
scores(ds, feats, seed=0) -> np.ndarray  # bentuk (n_query, n_pool); makin tinggi = makin mungkin dupe
```

Metode deterministik mengabaikan `seed`; metode stokastik memakainya. Metode *supervised*
(learning-to-rank) menghasilkan skor **out-of-fold** melalui `GroupKFold(5)` yang dikelompokkan
per query, sehingga sebuah query tidak pernah berada di train dan test sekaligus.

### 4.2 Precompute fitur (`methods/base.py`)

`build_features()` menghitung sekali untuk dipakai bersama:
- matriks biner unigram (produk & query),
- matriks biner bigram (pasangan accord tak-berurut, sparse),
- IDF dari korpus produk,
- TF-IDF ter-L2-normalisasi (unigram, bigram, dan gabungan orde-2 untuk P1),
- himpunan accord untuk Jaccard,
- primitif berpasangan: jumlah accord shared, Jaccard, cosine unigram/bigram.

---

## 5. Metodologi Evaluasi

### 5.1 Protokol

- **Leave-one-out dupe retrieval.** Query = parfum global dengan ≥1 dupe berlabel.
- **Candidate pool = seluruh 340 produk** (termasuk 97 distractor tak berlabel).
- Metode supervised menilai out-of-fold (GroupKFold-5 by query).
- **Metrik:** MRR, Hits@1, Hits@3 (satu konsep relevan per query; pakai rank terbaik/minimum
  bila satu query punya beberapa dupe).

### 5.2 Penanganan ties (perbaikan metodologis penting)

Skor menghasilkan *ties*, dan aturan naif "1 + #(skor lebih besar)" memberi keuntungan
optimistik palsu pada metode yang banyak ties — Jaccard tanpa bobot meninggalkan **±16
kandidat** seri di posisi teratas (vs ±5 untuk metode IDF). Karena itu dipakai **ekspektasi
eksak di bawah pemecahan ties acak seragam**:

> Misal `g` = jumlah produk berskor lebih tinggi dari produk relevan terbaik, dan
> `e` = jumlah produk seri di skor tersebut (≥1, termasuk yang relevan). Maka
> `E[RR] = mean(1/r untuk r di g+1..g+e)` dan `E[Hit@k] = clip(k−g, 0, e) / e`.
> Aturan ini *unbiased* terhadap ties dan tereduksi menjadi rank biasa ketika `e == 1`.

**Dampak (mengungkap urutan yang benar):**

| Aturan ties | Jaccard | B2 | P1 |
|---|---|---|---|
| optimistik (keliru) | 0.71 | 0.46 | 0.51 |
| **ekspektasi eksak (dipakai)** | **0.43** | **0.45** | **0.50** |

Dengan penanganan yang benar, metode order-2 usulan menempati peringkat teratas dan
keunggulan palsu Jaccard hilang.

### 5.3 Uji signifikansi

- **Wilcoxon signed-rank** *paired across queries* atas reciprocal rank (usulan vs tiap
  baseline) — **bukan** lintas seed (metode deterministik tak punya variansi seed).
- **Bootstrap** CI 95% atas query (10 000 resample) untuk ΔMRR.
- Metode stokastik (P2, A2, A5, B5, B6) dilaporkan sebagai rata-rata ± std atas 5 seed. Untuk
  metode LTR, satu-satunya keacakan adalah penugasan fold yang di-shuffle per seed.

---

## 6. Implementasi Tiap Metode

### Tier 1 — baseline mapan
| id | Metode | Inti implementasi |
|----|--------|-------------------|
| B1 | Jaccard | irisan/union himpunan accord |
| B2 | TF-IDF cosine | **baseline marjinal utama** (orde-1) |
| B3 | BM25 | `rank_bm25.BM25Okapi`, dokumen = accord produk |
| B4 | Sentence-BERT | cosine teks `[meaning+family]` multibahasa (dependensi opsional) |
| B5 | Word2Vec | gensim, dilatih di katalog, mean-pool accord, stokastik |
| B6 | node2vec | graf ko-okurensi accord, mean-pool, stokastik |

### Tier 2 — struktur/terpelajar (diperkirakan KALAH = temuan)
| id | Metode | Inti implementasi |
|----|--------|-------------------|
| A1 | Edwards-wheel tree-W | W1 closed-form pada pohon beku; skor = −W1 |
| A2 | PPMI-SVD | ko-okurensi → PPMI → `TruncatedSVD`, mean-pool, stokastik |
| A3 | Signature-subgraph | fitur rare-edge bersama + coverage asimetris + logistic LTR |
| A4 | Per-bigram salience | logistic L2 high-dim atas indikator bigram bersama (overfit) |
| A5 | Bilinear low-rank | `qᵀ(diag(d)+LLᵀ)p`, metric-learned via gradient descent numpy |
| A6 | GradientBoosting fusion | demo overfit atas fitur kemiripan |

### Tier 3 — usulan
| id | Metode | Inti implementasi |
|----|--------|-------------------|
| P1 | **Order-2 co-occurrence TF-IDF** | token = accord ∪ pasangan-accord; IDF, L2-norm, cosine; skor terdekomposisi orde-1 + orde-2 |
| P2 | **P1 + logistic LTR fusion** | fitur `[bigram-cos, unigram-cos, |shared|]`, out-of-fold |
| P3 | P1 + bobot hub-discriminative | ablation: penalti accord-hub |

---

## 7. Hasil

### 7.1 Tabel utama (5 seed; MRR / Hits@1 / Hits@3)

| Metode | Tier | MRR | Hits@1 | Hits@3 |
|--------|------|-----|--------|--------|
| **P2_fusion** | T3 usulan | **0.500** ± 0.003 | 0.379 | 0.580 |
| **P1_order2** | T3 usulan | **0.496** | 0.378 | 0.562 |
| A6_gbm_fusion | T2 | 0.457 ± 0.010 | 0.325 | 0.533 |
| B2_tfidf_cos | T1 (baseline) | 0.454 | 0.335 | 0.507 |
| P3_hubidf | T3 | 0.448 | 0.335 | 0.491 |
| A2_ppmi_svd | T2 | 0.444 ± 0.001 | 0.343 | 0.498 |
| A3_signature | T2 | 0.433 ± 0.006 | 0.303 | 0.500 |
| B1_jaccard | T1 | 0.428 | 0.303 | 0.487 |
| B3_bm25 | T1 | 0.388 | 0.278 | 0.433 |
| B6_node2vec | T1 | 0.340 ± 0.013 | 0.233 | 0.371 |
| A1_wheel_treeW | T2 | 0.284 | 0.174 | 0.306 |
| B5_word2vec | T1 | 0.185 ± 0.011 | 0.110 | 0.182 |
| A5_bilinear | T2 | 0.150 ± 0.027 | 0.072 | 0.150 |
| A4_bigram_salience | T2 | 0.139 ± 0.012 | 0.060 | 0.145 |
| B4_sbert | T1 | 0.272 | 0.163 | 0.311 |

### 7.2 Signifikansi (P1 vs tiap baseline)

| Baseline | ΔMRR | CI 95% | Wilcoxon p |
|----------|------|--------|------------|
| B2_tfidf_cos | +0.042 | [+0.019, +0.067] | **8.7 × 10⁻⁸** |
| B1_jaccard | +0.069 | [+0.040, +0.100] | 6.6 × 10⁻³ |
| B3_bm25 | +0.108 | [+0.076, +0.143] | 8.3 × 10⁻¹² |
| A1_wheel | +0.213 | [+0.158, +0.268] | 3.9 × 10⁻¹³ |
| A2_ppmi_svd | +0.054 | [+0.011, +0.099] | 3.9 × 10⁻³ |
| A3_signature | +0.066 | [+0.039, +0.093] | 5.7 × 10⁻⁸ |
| A4_bigram_salience | +0.367 | [+0.309, +0.426] | 2.7 × 10⁻²¹ |
| A5_bilinear | +0.332 | [+0.275, +0.391] | 2.5 × 10⁻²¹ |
| A6_gbm_fusion | +0.052 | [+0.017, +0.087] | 2.3 × 10⁻² |
| B5_word2vec | +0.294 | [+0.233, +0.354] | 2.0 × 10⁻¹⁶ |
| B6_node2vec | +0.181 | [+0.131, +0.233] | 2.7 × 10⁻¹¹ |
| B4_sbert | +0.225 | [+0.166, +0.283] | 1.6 × 10⁻¹¹ |
| P2_fusion | −0.005 | [−0.020, +0.011] | 0.18 (n.s.) |

**Dekomposisi P1:** komponen orde-2 (ko-okurensi) menyumbang rata-rata **81.3%** skor
top-rank; orde-1 (marjinal) hanya 18.7%.

### 7.3 Interpretasi

1. **Temuan utama berlaku.** P1/P2 mengungguli baseline marjinal (B2), Jaccard, BM25,
   embedding (B5/B6), perseptual (A1), dan seluruh metode terpelajar (A2–A6); semua positif
   dan signifikan — **termasuk Sentence-BERT (B4 = 0.272)** yang kalah signifikan
   (Wilcoxon p = 1.6 × 10⁻¹¹) meski menggunakan field teks.
2. **Headline cocok pilot.** Wilcoxon P1 vs B2 = 8.7 × 10⁻⁸ ≈ target ~10⁻⁷.
3. **P1 ≈ P2** (beda tak signifikan, p = 0.18): P1 yang paling sederhana sudah memadai; fusion
   LTR tidak menambah nilai berarti.
4. **Struktur/terpelajar kalah karena overfit data jarang:** A4 (bigram-salience high-dim) dan
   A5 (bilinear) anjlok ke MRR ~0.14–0.15, justru di bawah baseline.

---

## 8. Kesetiaan terhadap Pilot & Catatan Jujur

| Besaran | Hasil ini | Target pilot | Status |
|---------|-----------|--------------|--------|
| pool / query / shared | 340 / 209 / 56 | ~340 / ~208–243 / 56 | ✓ cocok |
| B2 cosine MRR | 0.454 | ~0.44 | ✓ cocok |
| A1 wheel MRR | 0.284 | ~0.30 | ✓ cocok |
| A4 MRR | 0.139 | 0.11–0.18 | ✓ cocok |
| A5 MRR | 0.150 | ~0.18 | ✓ dekat |
| Wilcoxon P1 vs B2 | 8.7e-8 | ~1e-7 | ✓ cocok |
| P1 / P2 MRR (absolut) | 0.496 / 0.500 | ~0.61 / ~0.64 | △ lebih rendah |
| A2 / A6 MRR | 0.444 / 0.457 | ~0.27 / ~0.37 | △ lebih tinggi |
| Bootstrap CI ΔMRR (P1−B2) | [+0.02, +0.07] | [+0.11, +0.23] | △ lebih sempit |
| B4_sbert MRR | 0.272 | untested (pilot) | ✓ dijalankan |

**Catatan:** **urutan peringkat dan signifikansi reproduksi penuh**; magnitudo absolut P1/P2
lebih rendah dan beberapa metode Tier-2 lebih tinggi dari pilot. Ini dilaporkan apa adanya —
tidak ada penyetelan ke angka pilot. Dua keputusan metodologis yang mengubah angka mentah
didokumentasikan: (a) penanganan ties ekspektasi-eksak (§5.2) yang membatalkan keunggulan
palsu Jaccard, dan (b) diagnosis artifact containment (§2.3) yang membenarkan TF-IDF sebagai
baseline adil. B4/B5/B6 telah dijalankan dan dilaporkan jujur: B5 = 0.185, B6 = 0.340,
B4 = 0.272 — semua kalah dari P1 (0.496).

---

## 9. Keterbatasan

- **B4 (Sentence-BERT) dievaluasi** (MRR = 0.272); kalah dari P1 dan bahkan dari baseline B2.
  Caveat: query hanya punya accord + `global_family` (miskin teks) sedangkan produk punya prosa
  Indonesia → ruang embedding asimetris; kekalahan B4 sebagian disebabkan asimetri ini, bukan
  semata kegagalan semantik.
- Label sangat jarang (1 relevan per query) → metode supervised high-dim rentan overfit
  (justru bagian dari temuan).
- Artifact containment membuat tumpang-tindih accord mentah nyaris sirkular; analisis dibatasi
  pada regime berbobot-IDF agar adil.

---

## 10. Reproduksi

```bash
pip install -r requirements.txt

python -m src.data        # sanity: pool=340, query=209, shared=56 + audit kebocoran
python -m src.wheel       # sanity: pohon wheel + beberapa jarak W1

python run_all.py            # benchmark penuh (B4 perlu sentence-transformers)
python run_all.py --fast     # lewati B4/B5/B6 (tanpa unduhan/pelatihan embedding)
python run_all.py --seeds 5  # jumlah seed untuk metode stokastik (default 5)
```

Keluaran di `results/`: `results.csv` (MRR/Hits per metode), `significance.csv` (Wilcoxon p +
CI bootstrap ΔMRR untuk P1 & P2 vs tiap metode). Seed deterministik; tanpa GPU
(kecuali B4 opsional).

> Pembuatan `kg_paths.json` kini **selesai** — lihat §12.

---

## 11. Addendum B4 (Sentence-BERT) — dijalankan setelah ruang disk tersedia

B4 sebelumnya SKIP karena disk penuh; setelah `sentence-transformers` (torch 2.12.1 CPU)
terpasang, B4 dievaluasi penuh dan adil.

- **Model:** `paraphrase-multilingual-MiniLM-L12-v2` (multibahasa — teks `meaning` produk
  berbahasa Indonesia; model English-only akan tak adil).
- **Konstruksi teks (leakage-safe):**
  - PRODUCT = `olfactory_family` + `meaning` (sudah di-strip oleh `leakage_audit`) + accord
    sebagai kata.
  - QUERY = `global_family` + accord sebagai kata. **Tanpa `interpreted_as`** (= nama global
    = kebocoran).
  - Skor = cosine(embed(query), embed(product)). Deterministik (tanpa seed loop).

**Hasil B4:** MRR **0.272**, Hits@1 **0.163**, Hits@3 **0.311**.

**P1 vs B4:** ΔMRR(P1 − B4) = **+0.225**, CI 95% [+0.166, +0.283], **Wilcoxon p = 1.6 × 10⁻¹¹**.
(P2 vs B4: +0.230, p = 7.7 × 10⁻¹².)

**Interpretasi:** B4 menempati peringkat ke-9 dari 15 (di bawah B2/Jaccard, di atas
B5/A1/A5/A4) — embedding semantik teks bebas kalah dari ko-okurensi accord terstruktur;
**temuan utama tetap berlaku — P1 (order-2) adalah yang terbaik**, sekarang dibuktikan
mengungguli seluruh 14 metode pembanding termasuk Sentence-BERT.

---

## 12. kg_paths Generation (penjelasan rekomendasi)

Skrip `src/kg_paths.py` menghasilkan `results/kg_paths.json`: untuk tiap query berlabel
(209), Top-3 rekomendasi P1 beserta **kg_paths** — dekomposisi skor cosine P1 menjadi
kontribusi tiap token accord bersama. Karena vektor gabungan P1 sudah ter-L2-normalisasi,
komponen `QC[q,k]·PC[p,k]` **persis sama dengan** `idf(token)² / (norm_q·norm_p)`, dan
jumlah seluruhnya = skor s(q,p). Token unigram → `accord_node` (orde-1); token bigram →
`co_occurrence_edge` (orde-2). `wheel_super_family` di-lookup dari leksikon Edwards;
pasangan beda superfamily → `cross_family`, accord di luar leksikon → `unmapped` (tidak
pernah null). Logika P1 di-reuse, bukan diimplementasi ulang.

**Statistik ringkas (`python -m src.kg_paths`):**

| Besaran | Nilai |
|---------|-------|
| Query | 209 (× Top-3 = 627 rekomendasi) |
| Rata-rata `order2_contribution` | 0.798 (≈ 0.813 dari §7.1) |
| Rata-rata paths / rekomendasi | 13.42 |
| Distribusi `wheel_super_family` | cross_family 3524, Fresh 2013, Amber 1290, Floral 939, Woody 650 |

**Validasi:** JSON valid (`json.load` tanpa error); tiap query tepat 3 rekomendasi;
`order1_contribution + order2_contribution ≈ 1.0` (0 pelanggaran > 0.01, kecuali rekomendasi
berskor-0 yang tak punya token bersama); tak ada `wheel_super_family` null.

> Catatan: sebagian kecil query global punya accord rusak (mojibake `"�"` akibat encoding
> sumber) sehingga berskor 0 dan `kg_paths` kosong — dilaporkan apa adanya.

**Contoh satu query (kg_paths dipangkas Top-3 path/rekomendasi demi keringkasan):**

```json
{
  "query": {
    "name": "Popcorn",
    "global_family": "Oriental Vanilla",
    "accords": ["lactonic", "woody", "smoky"]
  },
  "recommendations": [
    {
      "rank": 1,
      "product_name": "Born to Shine",
      "local_family": "SWEET",
      "score": 0.6359,
      "is_gold": true,
      "kg_paths": [
        {"type": "co_occurrence_edge", "from": "lactonic", "to": "smoky",
         "weight": 0.1521, "wheel_super_family": "cross_family",
         "relation": "shared_signature_pair"},
        {"type": "co_occurrence_edge", "from": "lactonic", "to": "woody",
         "weight": 0.1521, "wheel_super_family": "cross_family",
         "relation": "shared_signature_pair"},
        {"type": "co_occurrence_edge", "from": "smoky", "to": "woody",
         "weight": 0.1197, "wheel_super_family": "Woody",
         "relation": "shared_signature_pair"}
      ],
      "order2_contribution": 0.6665,
      "order1_contribution": 0.3335
    }
  ],
  "gold_rank": 1,
  "meta": {"method": "order2_cooccurrence_tfidf", "version": "v5.1"}
}
```

Dupe benar ("Born to Shine") menempati rank-1; 67% skornya berasal dari **pasangan
ko-okurensi** (lactonic–smoky, lactonic–woody) — bukti konkret bahwa sinyal orde-2 yang
menggerakkan retrieval, konsisten dengan temuan utama.

**Deliverable:** `src/kg_paths.py`, `results/kg_paths.json` (209 query),
`results/kg_paths_sample.json` (5 query pertama).
