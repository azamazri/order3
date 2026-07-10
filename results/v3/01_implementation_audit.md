# 01 — Audit Implementasi

## Bagian 1 — Perbaikan kode (Tahap 2)

Semua perubahan di bawah wajib sebelum eksperimen apa pun; kode lama tidak jalan di
skema bersih. Acceptance Tahap 2: `python run_all.py --fast --seeds 1` berjalan tanpa
error dan tanpa warning — **LULUS**.

### 2.1 Loader `src/data.py` — skema global bersih
- Pembacaan global: `pd.read_excel(global_xlsx, header=1)` → `pd.read_excel(global_xlsx)`
  (header sekarang di baris 1 / index 0).
- Kunci join & nama laporan: `row.get("Revolutionize")` → `row.get("perfume_name")`.
- `norm_name()` **tidak** diubah.
- **Aturan query valid (dideklarasikan):** query = punya ≥1 produk berlabel **dan**
  ≥1 accord. Baris global tanpa accord di-skip (lihat `00_dataset_verification.md`).

### 2.2 `meaning` dihapus dari load path; B4 = accord-only simetris
- `Product.meaning`, `Product.text_clean`, `Product.visual` **dihapus** dari
  `load_dataset()`; pemanggilan `leakage_audit()` dikeluarkan dari load path.
- **`B4_sbert` (tabel utama) = accord-only, simetris.** Query = `", ".join(A(q))`,
  Produk = `", ".join(A(p))`. Tidak ada teks bebas, tidak ada `olfactory_family` /
  `global_family` di sisi mana pun.
- Prosa turun menjadi ablation `B4a_prose` (Tahap 11), membaca `product_text.csv`;
  `leakage_audit()` pindah ke modul ablation itu dan **wajib** dijalankan di sana.
- **Alasan (apa adanya, bukan "B4 lemah"):** kolom `meaning` bukan prosa (median
  beberapa kata, campuran ID–EN); `visual_note` bukan teks bebas (dua token gaya nota,
  `visual_note_alt` = urutan dibalik). Perbandingan lama (prosa ID sisi produk vs daftar
  accord EN sisi query) **asimetris & lintas-bahasa** — itu confound, bukan hasil
  paradigma. Accord-only memberi SBERT input identik dengan semua metode → satu-satunya
  bentuk yang menjawab RQ3.
- **Limitations:** setelah pembersihan katalog tidak punya teks bebas panjang; paradigma
  "neural text" diwakili sentence encoder atas string accord. `B4a_prose` melaporkan
  kontribusi frasa pendek itu.

### 2.3 `src/wheel.py` — cakupan lexicon (A1)
- `V_canonical \ WHEEL_lexicon` **dihitung sendiri** = **{orange, mineral, anis}**
  (V_canonical = V_product ∪ V_global_all = 61 token). Persis sama dengan tabel usulan
  → tidak ada divergensi, tidak perlu berhenti.
- Pemetaan a-priori (label-free, sebelum metode apa pun, **beku** setelah commit):

  | accord | superfamily / subfamily | keyakinan |
  |---|---|---|
  | `orange` | Fresh / Citrus | tinggi |
  | `mineral` | Fresh / Water | tinggi |
  | `anis` | Fresh / Aromatic | **rendah — judgement call** |

- `anis` judgement call: penempatan alternatif `Amber/Amber` (spicy) tersedia
  (`ANIS_ALT`) dan diuji di sensitivitas A1. Butuh keputusan peneliti/pembimbing;
  sampai ada, primer = Fresh/Aromatic.
- **Sensitivitas S12:** `map` (primer) vs `drop` (perilaku lama: buang + normalisasi
  ulang massa).
- Docstring provenance diperbaiki: hanya **skeleton** superfamily/subfamily yang
  terinspirasi Edwards; pemetaan daun accord→subfamily adalah karya penulis. Klaim
  "frozen a-priori dari Edwards" hanya untuk struktur, bukan daun.
- **Cakupan massa (Tahap 2.3.5):**
  - `map`: `V_canonical \ lexicon = []`; **0/548** fragrance kehilangan >20% massa.
  - `drop`: hanya `anis`, `mineral` tak terpetakan (orange tak muncul di accord
    produk/query); **0/548 (0.0%)** fragrance kehilangan >20% massa.
  - **Gate G3 TIDAK aktif** (jauh di bawah ambang >10% fragrance kehilangan >20%).

### 2.4 De-bias docstring
- `src/methods/a5_bilinear.py`: hapus "Expected to be weak" → deskripsi mekanisme
  (learned low-rank cross-accord affinity).
- `src/methods/p1_order2.py`: hapus "PROPOSED core" dari judul docstring.
- `grep -rniE "parameter-free|expected to be|proposed core|is the winner|should improve"
  src/` → **CLEAN**.

### 2.5 `src/methods/p1_order2.py` — SyntaxWarning
- Docstring dijadikan raw string (`r"""`) → `\_` tidak lagi memicu
  `SyntaxWarning: invalid escape sequence`. Diverifikasi dengan
  `python -W error::SyntaxWarning run_all.py --fast --seeds 1` (lulus).

### 2.6 Klaim "parameter-free" dicabut
- `N` pada order-N adalah **hyperparameter** → dipilih via nested CV (Tahap 6), bukan
  dengan melihat metrik seluruh query. Istilah yang boleh: **"no learned parameters"**.
- Semua string "parameter-free" di `src/` dihapus (file legacy yang memuatnya
  diarsipkan ke `archive/pre_v3/src_legacy/`).

### Repo hygiene tambahan
- Driver pra-v3 yang superseded (mengimpor `leakage_audit`/`order_ablation` lama atau
  ditulis ulang di v3) dipindah ke `archive/pre_v3/src_legacy/`:
  `audit_step1/step3/a3_ablation/b4_symmetry`, `kg_paths`, `order3_analysis`,
  `order_ablation`, `order_significance`, `phase1_diag`, `phase2_a5fix`,
  `phase2_tuning`. `src/` kini: `data.py`, `evaluate.py`, `wheel.py`, `methods/`,
  `v3/`. Tuner comparator v3 ditulis ulang di Tahap 8.

## Bagian 2 — Tabel P/L/U + kebenaran implementasi (Tahap 3)

Diisi dengan **membaca kode**, bukan nama. Sumber CSV: `01_plu.csv`.

- **P** pair-dependent: skor bergantung statistik atas pasangan accord dalam satu fragrance.
- **L** label-supervised: parameter dilatih dari label `revolutionize`.
- **U** unsupervised-learned: parameter dipelajari dari data tanpa label.

| metode | file | P | L | U | bukti (mekanisme) |
|---|---|:-:|:-:|:-:|---|
| B1_jaccard | b1_jaccard.py | 0 | 0 | 0 | set overlap/union; tanpa pasangan, tanpa learning |
| B2_tfidf_cos | b2_tfidf.py | 0 | 0 | 0 | order-1 unigram tf-idf cosine; IDF = statistik korpus |
| B3_bm25 | b3_bm25.py | 0 | 0 | 0 | BM25Okapi atas token unigram; k1/b tetap; tanpa pasangan |
| B4_sbert | b4_sbert.py | 0 | 0 | 0 | sentence encoder pretrained atas string accord; tak ada param dari data/label kita |
| B5_word2vec | b5_word2vec.py | **1** | 0 | **1** | skip-gram (sg=1) atas daftar accord sebagai kalimat, window=5 ≥ len(A(p))≈5 → belajar co-occurrence |
| B6_node2vec | b6_node2vec.py | **1** | 0 | **1** | graf co-occurrence accord eksplisit + random walk + skip-gram |
| A1_wheel_treeW | a1_wheel.py | 0 | 0 | 0 | tree-W1 atas distribusi unigram uniform pada taksonomi tetap |
| A2_ppmi_svd | a2_ppmi_svd.py | **1** | 0 | **1** | matriks co-occurrence accord eksplisit → PPMI → TruncatedSVD |
| A3_signature | a3_signature.py | **1** | **1** | 0 | fitur n_shared_b/w_shared_b/max_rare = bigram bersama; logistic pada label OOF |
| A4_bigram_salience | a4_bigram_salience.py | **1** | **1** | 0 | indikator bigram bersama atas kosakata bigram penuh; L2 logistic pada label OOF |
| A5_bilinear | a5_bilinear.py | **1** | **1** | 0 | q^T(diag(d)+L Lᵀ)p; L Lᵀ = cross-accord; d,L dilatih dari label OOF |
| A6_gbm_fusion | a6_gbm.py | **1** | **1** | 0 | fitur termasuk bigram_cos (order-2); GBM dilatih pada label OOF |
| P1_order2 | p1_order2.py | **1** | 0 | 0 | unigram+bigram (order-N) tf-idf cosine; IDF statistik; N hyperparam via nested CV |
| P2_fusion | p2_fusion.py | **1** | **1** | 0 | fitur [bigram_cos,unigram_cos,shared]; logistic pada label OOF |
| P3_hubidf | p3_hubidf.py | **1** | 0 | 0 | order-2 tf-idf + hub down-weight (statistik derajat korpus); tanpa learning label |

**Pair-dependent (P=1): 10/15.** Non-pair-dependent (P=0): B1, B2, B3, B4, A1.

**Konsekuensi (apa adanya, bukan narasi lama):** karena B5, B6, A2, A5, A6 (dan A3, A4,
P2, P3) semuanya **pair-dependent**, klaim *"metode co-occurrence mengungguli metode
non-co-occurrence"* **tidak bisa** dibangun dari pembanding-pembanding ini. Yang
membedakan bukan "pakai co-occurrence atau tidak", melainkan **bagaimana** co-occurrence
dimodelkan: eksplisit & tanpa parameter terlatih (P1/P3) vs dipelajari & terkompresi
(B5/B6/A2) vs dipelajari-supervised (A3/A4/A5/A6/P2). Framing RQ3 diserahkan ke peneliti;
tabel dilaporkan apa adanya.

**Inkonsistensi lama diperbaiki:** A3 dulu diberi label "variant" sementara A6 "comparator"
padahal keduanya P=1, L=1 (kelas mekanisme sama). Docstring A3 sudah di-de-bias. Kategori
ditentukan tabel P/L/U ini (Tahap 7.5), bukan hasil.

### Kebenaran implementasi (§3.2)

| aspek | temuan |
|---|---|
| Sumber IDF | **340 produk saja** (`base._idf(PU)`, PU = matriks produk). Query ditransform dengan IDF produk. Konsisten untuk B2, P1, P3, A3(idf_b), A4(idf_b), A5. B3 BM25 memakai IDF internalnya atas korpus 340 produk. |
| Normalisasi | L2 row-norm di `build_features` (PUt/QUt/PBt/QBt/PC/QC). A1 = distribusi (jumlah 1). B1 tanpa norm (Jaccard). B3 punya normalisasi BM25 sendiri. |
| Hyperparameter | default modul (dicatat). **Di-tuning ulang di Tahap 8** pada data bersih; `best_params` lama tidak sah. |
| Seed | stokastik (5 seed 0..4): A2, A3, A4, A5, A6, B5, B6, P2. Deterministik: B1, B2, B3, B4, A1, P1, P3. Sumber varian A3/A4/A6/P2 = shuffle fold saja. |
| Out-of-fold | supervised A3/A4/A5/A6/P2 → `groupkfold_oof` / `grouped_folds` by query. Query tak pernah di train & test bersamaan. |
| Skip diam-diam | B4 → NaN bila model gagal dimuat → `evaluate_method` menandai `skipped`, bukan 0. Terverifikasi. |
| Determinisme | **15/15 deterministik** pada seed tetap (2× seed 0, `max|diff|=0`). Sumber: `01_determinism.csv`. **Gate G2 CLEAR.** |

### Verifikasi terhadap paper asli (ringkas; audit penuh Tahap 12)

| metode | parameter aktual | status sumber |
|---|---|---|
| B3_bm25 | `BM25Okapi` (rank_bm25) default k1=1.5, b=0.75; doc=produk, query=accord query; IDF dari 340 produk | Robertson & Zaragoza (PDF ada) |
| B4_sbert | checkpoint `paraphrase-multilingual-MiniLM-L12-v2`, mean-pool, `normalize_embeddings=True`, **simetris accord-only** | Reimers & Gurevych (PDF ada) |
| B5_word2vec | dim=64, window=5, epochs=50, min_count=1, sg=1, workers=1, seed | **SUMBER BELUM DIVERIFIKASI** (Mikolov PDF tak ada) |
| B6_node2vec | dim=64, walk_length=20, num_walks=50, window=5, p=q=1 (default), bobot sisi = #co-occurrence, workers=1 | Grover & Leskovec (PDF ada) |
| A2_ppmi_svd | PPMI (tanpa shift), dim=50, TruncatedSVD (randomized) | **SUMBER BELUM DIVERIFIKASI** (Levy & Goldberg PDF tak ada) |
| A1_wheel | skeleton superfamily/subfamily Edwards-inspired; daun accord = karya penulis; bobot 1/2/3 | **EDWARDS BELUM DIVERIFIKASI** (leaf mapping bukan dari terbitan) |

**Determinisme & determinasi seed dikonfirmasi. Gate G2 tidak aktif.**

