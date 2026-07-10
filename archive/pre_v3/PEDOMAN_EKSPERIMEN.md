# PEDOMAN EKSPERIMEN — Interaction-Free Perfume Dupe Retrieval

Dokumen ini adalah **satu-satunya acuan** untuk implementasi eksperimen.
Paper **bukan** acuan. Kalau ada perbedaan antara dokumen ini dan paper, dokumen ini yang benar.

Ditulis dari pembacaan langsung kode di `github.com/azamazri/order3`,
branch `main` dan `exp/fair-baselines`.

---

## BAGIAN 0 — Ringkasan Ide Penelitian

**Pertanyaan:** Apakah memodelkan **accord yang muncul bersamaan** (co-occurrence)
meningkatkan kualitas rekomendasi dibanding memperlakukan accord sebagai
**berdiri sendiri** (independen)?

**Cara menguji:** Aromatique membuat produk yang meniru parfum global tertentu.
Relasi "produk X meniru parfum global Y" diketahui dan objektif. Sembunyikan relasi
itu, minta sistem menemukan kembali, lalu ukur seberapa tinggi jawaban benar
diletakkan.

**Yang dibandingkan:** metode order-N (usulan) melawan 12 metode pembanding
dari 4 paradigma berbeda.

---

## BAGIAN 1 — Data

### 1.1 Sumber

| Komponen | Sumber | File |
|---|---|---|
| Accord produk | katalog lokal Aromatique | `dataset-aromatique.xlsx`, kolom `main_accords` |
| Accord query | Fragrantica (eksternal) | `global_reference.xlsx`, kolom `main_accord` |
| Label | relasi `inspired_by` | `dataset-aromatique.xlsx`, kolom `revolutionize` |

### 1.2 Ukuran

- **Pool kandidat**: 340 produk Aromatique (tidak berubah, tidak pernah difilter).
- **Query**: 209 parfum global yang punya ≥1 produk berlabel.
- **Kosakata accord** `V`: gabungan accord dari produk + query.
- Accord produk: rata-rata ~4.8 per produk.
- Accord query: rata-rata ~8.7 per query.

### 1.3 Parsing accord

Sama untuk semua metode, tanpa kecuali:

```
split pada ',' atau ';'
lowercase
strip whitespace
buang token kosong
dedup (himpunan, bukan daftar)
```

Accord **tidak berulang** dalam satu fragrance → semua representasi bersifat
**kehadiran biner**, bukan hitungan frekuensi.

### 1.4 Leakage audit (wajib, sebelum metode apa pun jalan)

1. Kolom `interpreted_as` (= nama parfum global) **tidak pernah** diberikan ke metode mana pun.
2. Teks bebas produk (`meaning`, `visual_note`) diperiksa: token khas dari nama
   parfum global yang ditirunya **dihapus**. Hasilnya disimpan di `text_clean`.
3. Hanya `text_clean` yang boleh dilihat metode berbasis teks (B4).

---

## BAGIAN 2 — Tugas dan Harness Evaluasi

Bagian ini **identik untuk semua 15 metode**. Tidak boleh ada satu pun metode
yang memakai harness berbeda.

### 2.1 Tugas

Untuk setiap query `q` (1 parfum global):

1. Metode menghasilkan skor `s(q, p)` untuk **semua 340 produk**.
   Skor lebih tinggi = lebih mungkin jadi jawaban.
2. Urutkan 340 produk menurun berdasarkan skor.
3. Cari peringkat produk jawaban benar. Kalau ada >1 jawaban benar, ambil
   peringkat **terbaik (terkecil)**.

Output tiap metode: matriks skor berukuran `(209, 340)`.

### 2.2 Ini BUKAN leave-one-out

Query adalah parfum global. Pool adalah produk Aromatique. Query **tidak pernah
ada** di dalam pool, jadi tidak ada yang perlu "ditinggalkan".

Istilah yang benar: **full-pool retrieval**.

Nama "leave-one-out" di kode dan dokumen lama adalah **salah nama** dan harus
diperbaiki di semua tempat.

### 2.3 Penanganan seri (WAJIB — jangan diubah)

Banyak metode menghasilkan skor seri. Aturan naif "peringkat = 1 + jumlah yang
lebih besar" memberi **bonus palsu** ke metode yang banyak seri (B1 Jaccard
meninggalkan ~16 kandidat seri di puncak).

Gunakan nilai harapan di bawah pemecahan seri seragam:

```
g = jumlah produk dengan skor > skor jawaban benar terbaik
e = jumlah produk dengan skor == skor jawaban benar terbaik   (e >= 1)

E[RR]     = mean( 1/r  untuk r = g+1 .. g+e )
E[Hit@k]  = clip(k - g, 0, e) / e
```

Kalau `e == 1`, rumus ini kembali jadi peringkat biasa.

### 2.4 Metrik

- **MRR** = rata-rata `E[RR]` atas 209 query.
- **Hits@1** = rata-rata `E[Hit@1]`.
- **Hits@3** = rata-rata `E[Hit@3]`.

### 2.5 Uji statistik

- **Wilcoxon signed-rank**, berpasangan **antar-query**, atas reciprocal rank.
  (Bukan antar-seed. Bukan antar-fold.)
- **Bootstrap 95% CI**, 10.000 resample **atas query**, untuk ΔMRR.

### 2.6 Metode stokastik

Metode yang hasilnya berubah tiap seed dijalankan dengan beberapa seed,
dilaporkan `mean ± std`.

Stokastik: A2, A3, A4, A5, A6, B5, B6.
Deterministik: B1, B2, B3, B4, A1, order-N.

### 2.7 Metode supervised

A3, A4, A5, A6 memakai label saat training. Mereka **wajib** dinilai
**out-of-fold**:

```
GroupKFold-5, dikelompokkan per QUERY.
Satu query tidak boleh muncul di train dan test sekaligus.
```

Skor test dikumpulkan dari fold-fold, lalu dievaluasi sekali atas 209 query.

---

## BAGIAN 3 — Metode Usulan: order-N

### 3.1 Konstruksi token

Dari himpunan accord `A(f)`, ambil **semua subset** berukuran `k = 1..N`:

```
T_k(f) = { S ⊆ A(f) : |S| = k }
T_≤N(f) = union dari T_1(f) .. T_N(f)
```

Subset **tak berurut** — `(a,b)` sama dengan `(b,a)`. Disimpan sebagai tuple terurut.

**Contoh** `A(f) = {amber, sweet, woody}`, N=3:

| orde | token | jumlah |
|---|---|---|
| 1 | `amber`, `sweet`, `woody` | 3 |
| 2 | `(amber,sweet)`, `(amber,woody)`, `(sweet,woody)` | 3 |
| 3 | `(amber,sweet,woody)` | 1 |
| | **total** | **7** |

### 3.2 Kosakata

Kosakata token = gabungan token dari **produk + query**.

Token yang hanya ada di query mendapat `df = 0`. Ia tak pernah cocok dengan produk
mana pun (kontribusi ke pembilang = 0), tapi tetap masuk ke `‖v_q‖`.
Ini keputusan desain — **harus dinyatakan**, bukan disembunyikan.

### 3.3 Bobot IDF

```
df(t)  = jumlah PRODUK (dari 340) yang memuat token t
idf(t) = log( (1 + 340) / (1 + df(t)) ) + 1
```

**IDF dihitung dari 340 produk saja**, tidak dari query. Query di-transformasi
memakai IDF yang sama.

### 3.4 Vektor dan skor

```
v_f[t] = idf(t)   jika t ∈ T_≤N(f)
       = 0        jika tidak
v̂_f    = v_f / ‖v_f‖₂

s(q,p) = ⟨v̂_q, v̂_p⟩
```

### 3.5 Dekomposisi per orde

Karena semua blok orde dinormalisasi dengan **satu norma bersama**, cosine terpecah
secara **eksak**:

```
s(q,p) = S_1 + S_2 + ... + S_N

S_k = (1 / (‖v_q‖ · ‖v_p‖)) · Σ_{t ∈ T_k(q) ∩ T_k(p)} idf(t)²
```

Ini **identitas aljabar**, bukan aproksimasi. Berlaku karena kehadiran biner × idf.

### 3.6 Yang dijalankan

`N = 1, 2, 3, 4, 5, 6`. Deterministik, satu run per N.

### 3.7 Validasi wajib

```
N=1 HARUS mereproduksi B2 (TF-IDF cosine). MRR ≈ 0.4540
N=2 HARUS mereproduksi P1 (order-2).        MRR ≈ 0.4963
```

Kalau meleset > 1e-3, **hentikan skrip**. Generalisasinya salah.

### 3.8 Diagnostik sparsity

Untuk tiap `k`, hitung rata-rata jumlah token orde-`k` yang **dimiliki bersama**
antara query dan produk jawaban benarnya, atas 209 query.

Ini menjelaskan kenapa kurva mendatar di orde tinggi.

### 3.9 Parameter

**Tidak ada.** order-N tidak punya hyperparameter, tidak dilatih, tidak melihat label.
Karena itu ia **tidak boleh** di-tuning.

---

## BAGIAN 4 — 12 Metode Pembanding (DETAIL)

Semua memakai harness Bagian 2. Semua menghasilkan matriks `(209, 340)`.

---

### PARADIGMA 1 — Marginal similarity (accord satu-satu)

#### B1 — Jaccard

**Ide:** kemiripan himpunan mentah, tanpa bobot.

**Mekanisme:**
```
s(q,p) = |A(q) ∩ A(p)| / |A(q) ∪ A(p)|
```
Kalau union = 0, skor = 0.

**Input:** himpunan accord, kedua sisi.
**Parameter:** tidak ada. Deterministik.

**Catatan penting:** B1 menghasilkan **banyak seri** (~16 kandidat seri di puncak).
Ini alasan aturan seri di §2.3 wajib. Tanpa itu B1 dapat bonus palsu.

---

#### B2 — TF-IDF cosine  ← **BASELINE UTAMA**

**Ide:** accord satu-satu, ditimbang kelangkaan.

**Mekanisme:**
```
v_f[a] = idf(a)  jika accord a ∈ A(f), else 0
idf(a) = log((1+340)/(1+df(a))) + 1     # df dari 340 produk
v̂ = v / ‖v‖₂
s(q,p) = ⟨v̂_q, v̂_p⟩
```

**Ini identik dengan order-N pada N=1.** Bukan `sklearn.TfidfVectorizer` — ditulis
manual dengan numpy, tapi rumusnya sama dengan smoothed idf sklearn.

**Input:** accord, kedua sisi.
**Parameter:** tidak ada. Deterministik.

**Kenapa ini baseline utama:** perbedaan tunggal antara B2 dan order-N adalah
**token co-occurrence**. Jadi selisih B2 vs order-N mengisolasi kontribusi
co-occurrence, bukan hal lain.

---

#### B3 — BM25

**Ide:** pembobotan retrieval probabilistik.

**Mekanisme:**
```
BM25Okapi dari library rank_bm25
dokumen  = daftar accord tiap produk (340 dokumen)
query    = daftar accord parfum global
s(q,p)   = bm25.get_scores(accords(q))[p]
```

**Parameter:** `k1`, `b`.
Versi lama: **default library** (`k1=1.5`, `b=0.75`) — tidak pernah dinyatakan.
Harus dilaporkan eksplisit.

**Catatan:** dokumen sangat pendek (~5 token) dan panjangnya seragam. Normalisasi
panjang dokumen (`b`) mungkin justru merugikan. `b=0` layak diuji.

**Deterministik.**

---

### PARADIGMA 2 — Neural / learned embedding

#### B4 — Sentence-BERT

**Ide:** kemiripan semantik dari sentence embedding pretrained.

**Mekanisme:**
```
model = "paraphrase-multilingual-MiniLM-L12-v2"   (pretrained, TIDAK dilatih)

teks_produk = text_clean (prosa Indonesia, sudah di-strip leakage)
              + olfactory_family
              + accord sebagai kata

teks_query  = global_family
              + accord sebagai kata          (Inggris)

embed kedua sisi, normalize_embeddings=True
s(q,p) = dot product (= cosine)
```

**Parameter:** tidak ada (pretrained, tanpa fit). Deterministik.

**⚠️ MASALAH SERIUS:** B4 adalah **satu-satunya** metode yang meng-encode
**prosa Indonesia** di sisi produk sementara sisi query hanya **daftar accord
Inggris**. Sebelas metode lain hanya melihat accord di kedua sisi.

Artinya B4 mengerjakan tugas **cross-lingual prose-vs-list matching**, bukan
tugas yang sama dengan pembanding lain. Perbandingannya tidak setara.

**Perbaikan wajib** — jalankan tiga varian:

| varian | teks produk | teks query |
|---|---|---|
| B4a | prosa + family + accord | family + accord |
| B4b | **accord saja** | **accord saja** |
| B4c | family + accord | family + accord |

**B4b** adalah perbandingan yang setara. Yang masuk tabel utama harus B4b.

**Fallback:** kalau model gagal di-load, metode mengembalikan seluruh NaN dan
dilaporkan sebagai "skipped". **Tidak boleh** dipalsukan dengan skor nol.

---

#### B5 — Word2Vec

**Ide:** embedding accord dipelajari dari ko-okurensi lokal.

**Mekanisme:**
```
kalimat = daftar accord tiap fragrance
          (dari PRODUK + QUERY)

model = gensim.Word2Vec(kalimat, sg=1 (skip-gram), min_count=1, workers=1, seed=s)

vektor fragrance = rata-rata vektor accord-nya
L2-normalize
s(q,p) = cosine
```

**Parameter:** `vector_size` (dim), `window`, `epochs`.
Versi lama: `dim=64, window=5, epochs=50` — tidak pernah di-tuning.

**Stokastik** (bergantung seed).

**⚠️ Asimetri:** dilatih atas **produk + query**. order-N hanya melihat IDF dari
340 produk. B5 melihat lebih banyak data. Bukan kebocoran label, tapi harus
dinyatakan.

**Catatan:** total token latih hanya ~3.400. Skip-gram di data sekecil ini
kemungkinan besar menghasilkan representasi lemah — itu **temuan**, bukan bug,
asalkan sudah di-tuning adil.

---

#### B6 — node2vec

**Ide:** embedding node dari graf ko-okurensi accord.

**Mekanisme:**
```
Graf G:
  node       = accord
  edge (a,b) = ada jika a,b muncul bersama di ≥1 fragrance
  bobot edge = jumlah fragrance yang memuat a dan b bersama
               (dihitung atas PRODUK + QUERY)

node2vec random walk (walk_length=20, num_walks=50, weighted)
  -> skip-gram (window=5, sg=1)
  -> vektor per accord

vektor fragrance = rata-rata vektor accord-nya
L2-normalize
s(q,p) = cosine
```

**Parameter:** `dimensions`, `p` (return), `q` (in-out), `walk_length`, `num_walks`, `window`.

**⚠️ Versi lama tidak menyetel `p` dan `q`** → default `p=1, q=1` = walk tak berbias.
Itu **setara DeepWalk**, bukan kontribusi inti node2vec. Kalau `p=q=1` tetap terpilih
setelah tuning, **harus dinyatakan eksplisit**.

**Stokastik.** Asimetri sama dengan B5 (lihat query saat membangun graf).

---

### PARADIGMA 3 — Perceptual transport

#### A1 — Edwards wheel tree-Wasserstein

**Ide:** jarak berbasis taksonomi persepsi bau, bukan pola data.

**Mekanisme:**
```
Pohon Edwards wheel (di-hardcode, 4 level):
  root
   └── superfamily   (Fresh, Floral, Amber, Woody)
        └── subfamily (Citrus, Green, Aromatic, Water, Fruity,
                       Floral, SoftFloral, Amber, SoftAmber,
                       Woods, MossyWoods, DryWoods)
             └── accord (citrus, green, amber, woody, ...)

Fragrance -> distribusi SERAGAM atas accord-nya yang ada di leksikon wheel.
  (accord di luar leksikon tidak berkontribusi)

Bobot edge:
  accord -> subfamily        = 1.0
  subfamily -> superfamily   = 2.0
  superfamily -> root        = 3.0

Jarak tree-Wasserstein-1 (closed form):
  W1(P,Q) = Σ_edge  w_edge · |massa_subtree_P(edge) - massa_subtree_Q(edge)|

s(q,p) = -W1(q,p)      # negatif, karena skor tinggi = mirip
```

Implementasi ditulis tangan dengan numpy. **Tidak memakai library optimal transport**
(tidak ada `POT`, `pyemd`). Bukan Sobolev transport. Bukan tree-sliced Wasserstein.
Jangan menyitasi metode-metode itu sebagai dasar A1.

**Parameter:** bobot edge `(1, 2, 3)` — dipilih tanpa justifikasi, tidak pernah diuji.
Layak diuji: `(1,1,1)` datar, `(1,3,9)` eksponensial.

**Deterministik.** Leksikon beku, tidak pernah menyentuh label.

---

### PARADIGMA 4 — Learned models

#### A2 — PPMI–SVD

**Ide:** embedding accord dari faktorisasi matriks ko-okurensi.

**Mekanisme:**
```
Matriks ko-okurensi C (V x V):
  untuk tiap fragrance (PRODUK + QUERY):
    untuk tiap pasangan accord (i,j) di fragrance itu: C[i,j] += 1, C[j,i] += 1
    untuk tiap accord i:                                C[i,i] += 1

PMI  = log( (C * total) / (rowsum * colsum) )
PPMI = max(PMI, 0)          # nilai negatif dan non-finite dibuat 0

emb = TruncatedSVD(n_components=k).fit_transform(PPMI)    # (V, k)

vektor fragrance = rata-rata emb accord-nya
L2-normalize
s(q,p) = cosine
```

**Parameter:** `rank` (dim SVD). Versi lama `min(50, V-1)`.
**Stokastik** (randomized SVD).

**⚠️ Asimetri:** matriks ko-okurensi dibangun dari **produk + query**.

**Tidak memakai label.** Jadi A2 **tidak bisa overfit**; `train == test` menurut
konstruksi. Skor rendah = kegagalan representasi, bukan overfitting.

---

#### A3 — Signature (regresi logistik)

**Ide:** ringkas "tanda tangan" pasangan accord bersama jadi beberapa fitur,
lalu pelajari bobotnya.

**Mekanisme:**

Untuk tiap pasangan `(query q, produk p)`, hitung **7 fitur**:

| # | fitur | definisi |
|---|---|---|
| 1 | `inter` | jumlah accord tunggal bersama |
| 2 | `n_shared_b` | jumlah **pasangan accord** bersama |
| 3 | `w_shared_b` | jumlah IDF pasangan accord bersama |
| 4 | `max_rare` | IDF **tertinggi** di antara pasangan bersama |
| 5 | `cov_q` | `inter / |A(q)|` |
| 6 | `cov_p` | `inter / |A(p)|` |
| 7 | `jaccard` | Jaccard accord |

Lalu:
```
StandardScaler -> LogisticRegression(class_weight="balanced", max_iter=1000)
dilatih OUT-OF-FOLD (GroupKFold-5 by query)
s(q,p) = predict_proba(...)[:, 1]
```

**Parameter:** `C` (regularisasi). Versi lama: default `C=1.0`.
**Stokastik** (variasi dari pengacakan fold).

**⚠️ MASALAH STRUKTURAL — A3 BUKAN PEMBANDING INDEPENDEN.**

Fitur #2, #3, #4 **adalah representasi order-2 metode usulan**:
`n_shared_b`, `w_shared_b`, dan `max_rare` semuanya dihitung dari
**pasangan accord bersama ber-IDF** — persis "pasangan langka" yang jadi
tesis penelitian ini.

Jadi A3 = **metode kita + regresi logistik di atasnya**. Ia varian, bukan lawan.

**Konsekuensi:** kalau A3 mengalahkan order-N, itu **bukan** bukti bahwa
co-occurrence kalah. Itu bukti bahwa co-occurrence **yang dibobot secara terpelajar**
sedikit lebih baik daripada co-occurrence tak berbobot.

**Wajib:** pindahkan A3 keluar dari blok "pembanding" ke blok **"varian metode usulan"**.
Nama "signature LTR" juga menyesatkan — ini regresi logistik biasa,
bukan learning-to-rank dengan objektif ranking. Ganti jadi `signature logistic`.

**Ablation yang wajib dijalankan:** A3 **tanpa** fitur #2, #3, #4.
Kalau skornya jatuh ke level B2 (~0.454), itu bukti langsung bahwa yang bekerja
adalah **co-occurrence**, bukan regresi logistiknya.

---

#### A4 — Bigram salience

**Ide:** pelajari satu bobot **untuk setiap pasangan accord**.

**Mekanisme:**
```
Vektor fitur untuk (q,p) = indikator sepanjang SELURUH kosakata bigram B:
   x[b] = 1  jika pasangan accord b dimiliki q DAN p
        = 0  jika tidak

LogisticRegression(penalty="l2", solver="liblinear",
                   class_weight="balanced", max_iter=2000)
dilatih OUT-OF-FOLD (GroupKFold-5 by query)
s(q,p) = predict_proba(...)[:, 1]
```

Dimensi fitur = ukuran kosakata bigram (ratusan sampai ribuan).
Label: 1 positif per query, 339 negatif.

**Parameter:** `C`. Versi lama: default `C=1.0` — **tidak cukup teregularisasi**
untuk ruang sebesar ini dengan label sesedikit itu.

**Stokastik.** Supervised → bisa overfit.

**⚠️ Docstring lama menulis:** *"expected to OVERFIT badly (this is the point)"*.
Menuliskan ekspektasi kalah ke dalam kode itu **bias eksperimen** dan bisa dibaca
reviewer di repo publik. Hapus.

---

#### A5 — Bilinear metric

**Ide:** pelajari matriks afinitas antar-accord berpangkat rendah.

**Mekanisme:**
```
q, p = vektor TF-IDF unigram yang sudah di-L2-normalize

s(q,p) = qᵀ ( diag(d) + L Lᵀ ) p

  d : (V,)      bobot diagonal (self-affinity per accord)
  L : (V, rank) faktor berpangkat rendah (cross-accord affinity)

Loss: logistic loss dengan bobot sampel seimbang
      + regularisasi L2 (l2 = 1e-3)
Dilatih OUT-OF-FOLD (GroupKFold-5 by query)
```

**Parameter:** `rank`, `lr`, `iters`, `l2`.

**⚠️ BUG (versi lama):** optimizer = gradient descent langkah-tetap,
`lr=0.5`, 300 iterasi. **Tidak konvergen.**
Loss bergerak `0.6931 → 0.6378`. Angka `0.6931 = ln(2)` = loss tebakan acak.
Train MRR hanya 0.184 — **lebih buruk dari B2 di data latihnya sendiri**.

Model yang tidak bisa memuat data latihnya **bukan overfit**, melainkan **rusak**.

**Perbaikan:** ganti GD dengan **L-BFGS-B** (gradien analitik, kriteria konvergensi,
maxiter 5000). Bobot sampel seimbang **sudah ada** — imbalance bukan penyebabnya.

Setelah perbaikan: konvergen dalam ~117 iterasi, loss `→ 0.4793`,
train MRR `0.184 → 0.387`.

---

#### A6 — GBM fusion

**Ide:** gabungkan beberapa fitur kemiripan dengan ensemble pohon.

**Mekanisme:**
```
6 fitur untuk (q,p):
  1. jaccard(q,p)
  2. cosine unigram  (= skor B2)
  3. cosine bigram
  4. |accord bersama|
  5. |accord bersama| / |A(q)|
  6. |accord bersama| / |A(p)|

sklearn.GradientBoostingClassifier(subsample=0.8)
dilatih OUT-OF-FOLD (GroupKFold-5 by query)
s(q,p) = predict_proba(...)[:, 1]
```

**Bukan XGBoost. Bukan LightGBM.** `sklearn.GradientBoostingClassifier`.
Sitasi kanoniknya Friedman, bukan Chen & Guestrin.

**Parameter:** `n_estimators`, `max_depth`, `learning_rate`.
Versi lama: `200, 3, 0.05` — tidak di-tuning.

**⚠️ `GradientBoostingClassifier` tidak punya `class_weight`.** Versi lama
**tidak menangani imbalance sama sekali**, sementara A3 dan A4 memakai
`class_weight="balanced"`. Tidak konsisten. Perbaikan: berikan
`sample_weight` seimbang pada `.fit()`.

**⚠️ Docstring lama:** *"overfit demo... Included to show that a flexible learner
does not beat the simple order-2 cosine"*. Hapus.

**Catatan:** fitur #3 (cosine bigram) **juga** turunan order-2. A6 sebagian
mengonsumsi ide metode usulan, walau lebih longgar daripada A3.

**Stokastik.** Supervised → bisa overfit.

---

## BAGIAN 5 — Varian Metode Usulan (bukan pembanding)

Ini **bukan** bagian dari 12 pembanding. Harus dilaporkan terpisah.

| ID | Mekanisme |
|---|---|
| **P2** | order-2 + regresi logistik atas 3 fitur: `[cosine bigram, cosine unigram, |accord bersama|]`. Out-of-fold. |
| **P3** | order-2 dengan penalti hub: `hub(a) = 1/(1 + log(1 + deg(a)))`, `deg(a)` = jumlah accord berbeda yang ber-ko-okurensi dengan `a`. Bobot bigram dikali hasil kali hub kedua ujungnya. Lalu bangun ulang TF-IDF order-2 seperti P1. |
| **A3** | **harus dipindah ke sini** — lihat §4/A3. |

**P3 dijalankan tapi tidak pernah dilaporkan.** Itu selective reporting.
Laporkan, atau hapus dari registry.

---

## BAGIAN 6 — Ringkasan Status Implementasi

| ID | Implementasi sesuai definisinya? | Masalah |
|---|---|---|
| order-N | ✅ benar | vocab termasuk query (dokumentasikan) |
| B1 Jaccard | ✅ benar | — |
| B2 TF-IDF | ✅ benar | — |
| B3 BM25 | ✅ benar | `k1`, `b` default, tidak dilaporkan |
| B4 S-BERT | ⚠️ | **input tidak setara** (prosa vs accord) |
| B5 Word2Vec | ✅ benar | melihat query saat fit; tidak di-tuning |
| B6 node2vec | ⚠️ | `p=q=1` → sebenarnya DeepWalk |
| A1 wheel | ✅ benar | bobot edge tak pernah diuji |
| A2 PPMI–SVD | ✅ benar | melihat query saat fit; tidak bisa overfit |
| A3 signature | ❌ | **bukan pembanding — varian metode usulan** |
| A4 bigram | ✅ benar | `C` default terlalu longgar; docstring bias |
| A5 bilinear | ❌ | **optimizer tidak konvergen** (sudah diperbaiki) |
| A6 GBM | ⚠️ | tanpa penanganan imbalance; docstring bias |

---

## BAGIAN 7 — Aturan yang Tidak Boleh Dilanggar

1. **Semua metode memakai harness Bagian 2.** Tidak ada pengecualian.
2. **Hyperparameter dipilih HANYA di dalam training fold** (nested CV).
   Tidak pernah berdasarkan skor test.
3. **order-N tidak di-tuning.** Ia parameter-free menurut definisi.
4. **Kalau pembanding yang sudah diperbaiki mengalahkan order-N, laporkan.**
   Jangan mencari konfigurasi lain untuk mengalahkannya kembali.
5. **Jangan tulis ekspektasi hasil ke dalam docstring kode.**
   ("expected to lose", "overfit demo", dst.)
6. **Selisih angka tidak boleh diklaim tanpa uji signifikansi.**
   0.513 vs 0.508 harus diuji, bukan dilihat mata.
7. **Semua hasil di-commit ke repo.** Angka yang tidak ter-commit tidak bisa
   direproduksi.

---

## BAGIAN 8 — Yang Belum Diverifikasi

### 8.1 Sirkularitas data (PRIORITAS TERTINGGI)

Kode eksperimen sendiri memuat sebuah **asumsi yang belum pernah diuji**.
Di `src/evaluate.py` baris 40 dan 55, fungsi `accord_containment_report` menyatakan
bahwa overlap accord mentah adalah sinyal "near-circular", dan bahwa daftar accord
global tampaknya diturunkan dari produk dupe-nya.

Asumsi itu **tidak pernah diverifikasi**. Ia hanya ditulis di komentar, lalu dipakai
sebagai alasan memilih pembobotan IDF. Kalau asumsi itu benar, seluruh eksperimen
mengukur artefak, dan tidak ada gunanya melanjutkan apa pun. Kalau salah, komentar
itu harus dihapus dari kode.

**Uji yang menentukan (belum pernah dijalankan):**

```
Untuk tiap produk berlabel p dengan query benar q_benar:
  containment_benar = |A(p) ∩ A(q_benar)| / |A(p)|

  ulangi 20x dengan q_acak diambil acak dari pool query:
  containment_acak  = |A(p) ∩ A(q_acak)| / |A(p)|

Bandingkan kedua distribusi.
```

Tafsir:

- `containment_acak` **juga tinggi** → overlap adalah artefak kosakata, bukan sinyal.
  Data bocor. **Hentikan eksperimen.**
- `containment_acak` **rendah**, `containment_benar` tinggi → overlap tinggi hanya
  untuk pasangan benar. Sinyalnya nyata. Lanjut, dan **hapus komentar "near-circular"**
  dari `src/evaluate.py`.

**Pemeriksaan tambahan yang wajib:**

1. Apakah kosakata accord produk merupakan **subset** dari kosakata accord global?
   Kalau ya, kemungkinan accord produk disalin dari global.
   Kalau produk punya accord yang tidak pernah muncul di global mana pun, berarti
   katalog punya kosakata sendiri.
2. Apakah `source_url` menunjuk halaman Fragrantica yang **spesifik per parfum**,
   atau halaman generik?
3. Apakah jumlah accord query sistematis lebih banyak daripada accord produk?
   Kalau ya, itu konsisten dengan Fragrantica mendaftar lebih banyak accord —
   bukan bukti penyalinan.

Semua ini harus dijalankan oleh Claude Code dan hasilnya dilaporkan sebagai angka,
bukan sebagai komentar di docstring.

### 8.2 Yang harus dijalankan setelah 8.1 bersih

1. Ablation A3 tanpa fitur bigram (#2, #3, #4).
2. B4 symmetric-text (B4a / B4b / B4c).
3. Uji signifikansi ulang dengan **semua angka hasil tuning**:
   order-N vs setiap pembanding, Wilcoxon + bootstrap CI.
4. Uji signifikansi order-N vs A3 (varian), terpisah.
5. Laporkan P3.
6. Ulangi order-ablation (N=1..6) dengan harness identik.
7. `pip freeze` → versi persis, bukan `>=`.
8. Merge semua ke `main`.
