# PEDOMAN_EKSPERIMEN_V2

Protokol eksekusi untuk Claude Code. Menggantikan `PEDOMAN_EKSPERIMEN.md`.
Baca `HANDOFF_V3.md` lebih dulu.

Branch kerja: **`exp/v3-clean`** (dari `main`).
Semua output ke **`results/v3/`**.

---

## CARA PAKAI DOKUMEN INI

Kerjakan Tahap 0 → 12 **berurutan**. Jangan melompat.
Setiap tahap punya **Acceptance** — kalau tidak terpenuhi, berhenti dan tulis
`results/v3/BLOCKED_<tahap>.md`, commit, lapor.

Ada **Gate G1–G8** (§13). Gate memaksa berhenti. Patuhi.

Aturan keputusan di §7 ditulis **sebelum** hasil ada. Jangan diubah setelah melihat angka.
Kalau kamu merasa aturannya salah, tulis alasannya di file terpisah dan minta persetujuan —
jangan diam-diam pakai aturan lain.

---

## TAHAP 0 — Repo hygiene

Folder sudah terlalu penuh. Bersihkan **sebelum** menambah apa pun.

### 0.1 Arsipkan, jangan hapus

```
mkdir -p archive/pre_v3
git mv results/                    archive/pre_v3/results/
git mv IMPLEMENTATION_REPORT.md    archive/pre_v3/
git mv BLUEPRINT_V5.md             archive/pre_v3/
git mv HANDOFF.md                  archive/pre_v3/
git mv HANDOFF_V2.md               archive/pre_v3/
git mv PEDOMAN_EKSPERIMEN.md       archive/pre_v3/
git mv "BRIEF REVISI PAPER.md"     archive/pre_v3/
```

Tulis `archive/pre_v3/README.md`:
> Semua isi folder ini dihitung di atas dataset yang belum bersih (typo accord, sel
> majemuk tak ter-split, kolom nama tidak terkoreksi). **Angka-angkanya tidak sah.**
> Disimpan hanya sebagai jejak audit. Jangan dikutip.

### 0.2 Hapus

```
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
rm -f results/run_full.log            # jika masih ada di root results
```

Tambahkan ke `.gitignore` bila belum ada: `__pycache__/`, `*.pyc`, `.venv/`, `*.log`.

### 0.3 Jangan sentuh

`conference_101719.tex`, `reference/`, `reference.bib`, `dataset-aromatique.xlsx`,
`global_reference.xlsx`, `cleaning_changelog.csv`, `product_text.csv`.

`product_text.csv` adalah file bantu **khusus ablation `B4a_prose`** (Tahap 11).
Tidak memuat accord. Tidak boleh dibaca oleh `load_dataset()` maupun metode lain.

### 0.4 Struktur target

```
.
├── HANDOFF_V3.md
├── PEDOMAN_EKSPERIMEN_V2.md
├── README.md                       (tulis ulang, singkat)
├── run_all.py
├── requirements.txt
├── dataset-aromatique.xlsx
├── global_reference.xlsx
├── cleaning_changelog.csv
├── product_text.csv                (khusus ablation B4a)
├── src/
├── results/v3/
└── archive/pre_v3/
```

**Acceptance:** `git status` bersih. `python -c "import src.data"` berhasil.
Commit: `chore(v3): archive pre-clean artifacts, repo hygiene`

---

## TAHAP 1 — Verifikasi dataset bersih

Buat `src/v3/verify_dataset.py`. **Read-only.**

### 1.1 Yang harus dilaporkan (angka, bukan kalimat)

| item | keterangan |
|---|---|
| `n_products` | baris katalog |
| `n_labeled` | `revolutionize` tidak kosong |
| `n_unlabeled` | distractor |
| `n_global_rows` | baris `global_reference.xlsx` |
| `n_queries` | baris global yang benar-benar jadi query (punya ≥1 produk berlabel) |
| `n_labeled_pairs` | total pasangan (produk, query) |
| `queries_with_multiple_dupes` | **penting**: penggabungan duplikat global bisa membuat >1 produk per query |
| `labeled_products_without_global_row` | produk berlabel yang namanya tidak ada di global |
| `V_product`, `V_query`, `V_shared` | ukuran kosakata accord |
| `product_only_tokens` | daftar lengkap |
| `query_only_tokens` | daftar lengkap |
| distribusi `len(A(p))` dan `len(A(q))` | histogram penuh |
| sel accord kosong | jumlah |

### 1.2 Assertion keras (crash, bukan warning)

```python
assert product_only_tokens == [], f"kosakata produk bocor: {product_only_tokens}"
assert n_labeled_pairs > 0
assert not any(len(p.accords) == 0 for p in ds.products)
assert not any(len(q.accords) == 0 for q in ds.queries)
```

Data sudah dibersihkan di Excel. **Kalau ada token di luar kosakata global, itu bug data —
crash.** Jangan menambahkan typo-fix, fuzzy match, atau normalisasi apa pun di kode.
Aturan `HANDOFF_V3.md` §0.9.

### 1.3 Silang-periksa dengan changelog

Baca `cleaning_changelog.csv`. Untuk setiap baris `action=drop`, pastikan token itu memang
tidak ada lagi di Excel. Untuk `action=split`, pastikan komponennya ada.

**Output:** `results/v3/00_dataset_verification.md` + `00_vocab.csv`
**Acceptance:** semua assertion lulus, `product_only_tokens` kosong.
Commit: `feat(v3): dataset verification`

---

## TAHAP 2 — Perbaikan kode yang wajib sebelum eksperimen apa pun

Ini bukan opsional. Kode sekarang **tidak akan jalan** di dataset baru.

### 2.1 `src/data.py` — pembacaan global_reference

File bersih punya header di **baris 1** dan kolom nama **`perfume_name`**.
Kode sekarang mengasumsikan `header=1` dan kolom `Revolutionize`.

Tiga titik yang harus diubah (verifikasi nomor barisnya sendiri, jangan percaya angka ini):
- pembacaan: `pd.read_excel(global_xlsx, header=1)` → `pd.read_excel(global_xlsx)`
- kunci join: `row.get("Revolutionize")` → `row.get("perfume_name")`
- nama untuk laporan: idem

Jangan mengubah `norm_name()`.

### 2.2 `src/data.py` — kolom `meaning` sudah tidak ada

**KEPUTUSAN SUDAH DIAMBIL. Jangan tanya lagi, jangan pilih sendiri.**

`Product.meaning`, `Product.text_clean`, dan `leakage_audit()` bergantung pada kolom
`meaning` yang kini dihapus dari Excel. `B4_sbert` memakai `text_clean`.

**Yang harus dilakukan:**

1. **`B4_sbert` (tabel utama) = accord-only, simetris.**
   Kedua sisi di-encode SBERT dari string daftar accord.
   Query: `", ".join(A(q))`. Produk: `", ".join(A(p))`.
   Tidak ada teks bebas. Tidak ada `global_family` / `olfactory_family` di sisi mana pun.

2. **`B4a_prose` = ablation saja, bukan tabel utama.** (Tahap 11)
   Sumber teks: file terpisah `product_text.csv` (kolom `product_name, meaning,
   visual_note, visual_note_alt, olfactory_family`). File ini **tidak** dipakai metode
   lain dan **tidak** memuat accord. `leakage_audit()` wajib dijalankan di sana.

3. **Hapus `Product.meaning` dan `Product.text_clean` dari `load_dataset()`.**
   Muat `product_text.csv` hanya di dalam modul ablation.

**Alasan (tulis apa adanya di laporan, jangan diringkas jadi "B4 lemah"):**

- Kolom `meaning` bukan prosa. Panjangnya beberapa kata saja, campuran Indonesia–Inggris.
- Kolom `visual_note` bukan prosa. Isinya dua token bergaya nota (`"Amber / Cacao"`),
  dan pasangannya `visual_note_alt` adalah dua token yang sama dengan urutan dibalik.
- Perbandingan lama (prosa Indonesia sisi produk vs daftar accord Inggris sisi query)
  **asimetris dan lintas-bahasa**. Kalau B4 kalah dalam setup itu, kita tidak tahu apakah
  penyebabnya arsitektur atau asimetris input. Itu confound, bukan paradigma.
- Versi accord-only membuat SBERT menerima **input yang sama** dengan semua metode lain.
  Itu satu-satunya perbandingan yang menjawab RQ3: *apakah sentence encoder terlatih
  mengungguli pembobotan leksikal pada input identik?*

**Konsekuensi yang harus ditulis di Limitations:**
Setelah pembersihan, katalog tidak memiliki teks bebas panjang. Paradigma "neural text"
diwakili oleh sentence encoder atas string accord, bukan atas deskripsi produk.
`B4a_prose` melaporkan berapa besar kontribusi frasa pendek itu.

`leakage_audit()` **tetap wajib dijalankan** untuk `B4a_prose` dan hasilnya dilaporkan
(berapa produk yang teks bebasnya memuat token khas nama global-nya sendiri).

### 2.3 `src/wheel.py` — cakupan lexicon

**KEPUTUSAN SUDAH DIAMBIL.**

A1 (Edwards wheel tree-Wasserstein) membuang accord yang tidak ada di lexicon lalu
menormalisasi ulang massa. Setelah pembersihan, ada accord kanonik yang tidak terpetakan.

**Yang harus dilakukan:**

1. **Hitung dan cetak** `V_canonical \ WHEEL_lexicon` sebelum apa pun.
   Verifikasi sendiri; jangan percaya daftar mana pun dari luar kode.

2. **Petakan accord yang tidak terpetakan ke wheel, SEKALI, sebelum menjalankan metode apa pun.**
   Pemetaan taksonomik dilakukan tanpa melihat label maupun skor. Itu sah.
   Membuang accord diam-diam **tidak** sah — itu menghukum A1 karena bug, bukan karena
   metodenya, dan membuat RQ3 tidak adil.

   Pemetaan yang diusulkan (peneliti sudah menyetujui pendekatannya; **konfirmasikan
   penempatannya sebelum commit**, dan tulis di `01_implementation_audit.md`):

   | accord | superfamily / subfamily | keyakinan |
   |---|---|---|
   | `orange` | Fresh / Citrus | tinggi |
   | `mineral` | Fresh / Water | tinggi |
   | `anis` | Fresh / Aromatic | **rendah — tandai sebagai judgement call** |

   Kalau daftar `V_canonical \ WHEEL_lexicon` yang kamu hitung berbeda dari tiga di atas,
   **berhenti dan lapor.** Jangan memetakan sendiri accord yang tidak ada di tabel ini.

3. **Bekukan pemetaan.** Setelah di-commit, tidak boleh diubah lagi. Dilarang keras
   menyesuaikan penempatan setelah melihat skor A1.

4. **Sensitivitas wajib** (masuk Tahap 9 sebagai dimensi `S12`):
   - `S12=map` — pemetaan di atas (primer)
   - `S12=drop` — perilaku lama: buang + normalisasi ulang massa

   Kalau MRR A1 berbeda material antara keduanya, itu limitasi A1 yang harus dilaporkan
   angkanya.

5. **Laporkan** per fragrance: proporsi massa yang hilang (harus 0 pada `S12=map`),
   dan pada `S12=drop`, berapa fragrance yang kehilangan >20% massa.

**Gate G3** kalau pada `S12=drop` lebih dari 10% fragrance kehilangan >20% massa
**dan** pemetaan `S12=map` tidak dapat menutupinya.

**Jangan menambahkan accord ke `WHEEL` di luar tabel di atas.**

### 2.4 De-bias docstring

`grep -rn "Expected to be\|should improve\|expected to win\|is the winner\|proposed core" src/`

Hapus semua kalimat yang menyatakan ekspektasi hasil. Contoh nyata:
`src/methods/a5_bilinear.py` docstring memuat *"Expected to be weak"*.
`src/methods/p1_order2.py` docstring memuat *"PROPOSED core"*.

Ganti dengan deskripsi mekanisme, tanpa penilaian.

### 2.5 `src/methods/p1_order2.py` — SyntaxWarning

Docstring memuat `\_` yang memicu `SyntaxWarning: invalid escape sequence`.
Jadikan raw string (`r"""`) atau escape.

### 2.6 Klaim "parameter-free" dicabut

`grep -rn "parameter-free" src/ *.py *.md`

`N` pada order-N **adalah hyperparameter**. Metode ini boleh disebut
**"no learned parameters"** — tidak boleh disebut "parameter-free". Ganti semua.
Konsekuensinya: `N` harus dipilih lewat nested CV (Tahap 6), bukan dengan melihat metrik
di seluruh query.

**Output:** `results/v3/01_implementation_audit.md` (bagian 1: perbaikan kode)
**Acceptance:** `python run_all.py --fast --seeds 1` berjalan tanpa error dan tanpa warning.
Commit: `fix(v3): adapt loader to clean schema, B4 text source, wheel coverage, de-bias docstrings`

---

## TAHAP 3 — Audit implementasi 15 metode

**Ini bagian terpenting dan paling sering dilewat.** Perbandingan yang tidak adil membuat
seluruh RQ3 tidak berarti. Baca **kode setiap metode**, bukan namanya.

### 3.1 Klasifikasi mekanisme (isi dengan membaca kode, jangan menebak)

Untuk setiap metode, jawab tiga pertanyaan **ya/tidak**:

- **P (pair-dependent):** apakah skornya bergantung pada statistik yang dihitung atas
  **pasangan accord di dalam satu fragrance**? (bigram indicator, matriks co-occurrence,
  jendela skip-gram atas daftar accord, cross-term bilinear, jalan acak di graf accord)
- **L (label-supervised):** apakah ada parameter yang dilatih dari label `revolutionize`?
- **U (unsupervised-learned):** apakah ada parameter yang dipelajari dari data tanpa label?

| metode | file | P | L | U |
|---|---|---|---|---|
| B1_jaccard | `b1_jaccard.py` | | | |
| B2_tfidf_cos | `b2_tfidf.py` | | | |
| B3_bm25 | `b3_bm25.py` | | | |
| B4_sbert | `b4_sbert.py` | | | |
| B5_word2vec | `b5_word2vec.py` | | | |
| B6_node2vec | `b6_node2vec.py` | | | |
| A1_wheel_treeW | `a1_wheel.py` | | | |
| A2_ppmi_svd | `a2_ppmi_svd.py` | | | |
| A3_signature | `a3_signature.py` | | | |
| A4_bigram_salience | `a4_bigram_salience.py` | | | |
| A5_bilinear | `a5_bilinear.py` | | | |
| A6_gbm | `a6_gbm.py` | | | |
| P1_order2 / order-N | `p1_order2.py`, `order_ablation.py` | | | |
| P2_fusion | `p2_fusion.py` | | | |
| P3_hubidf | `p3_hubidf.py` | | | |

**Peringatan yang sudah diketahui — verifikasi, jangan asumsikan:**

- `a6_gbm.py` memakai `feats.bigram_cos(...)`. `a3_signature.py` memakai fitur bigram.
  Kalau benar, **keduanya pair-dependent** dan tidak boleh dikategorikan berbeda.
  Pada laporan lama, A3 disebut "varian" sementara A6 disebut "comparator" — itu
  **inkonsisten** dan menguntungkan metode usulan. Perbaiki.
- `b5_word2vec.py` melatih skip-gram dengan **daftar accord sebagai kalimat**.
  Skip-gram belajar dari co-occurrence di dalam jendela. Kalau `window ≥ len(accords)`,
  metode ini pair-dependent. Periksa nilai `window` dan panjang daftar accord.
- `b6_node2vec.py` membangun **graf co-occurrence accord** secara eksplisit. Pair-dependent.
- `a2_ppmi_svd.py` membangun **matriks co-occurrence accord**. Pair-dependent.
- `a5_bilinear.py` punya term `L L^T` = cross-accord. Pair-dependent.

**Konsekuensi:** kalau P bernilai "ya" untuk B5, B6, A2, A5, A6, maka narasi
*"metode co-occurrence mengungguli metode non-co-occurrence"* **tidak bisa** dibuat dari
pembanding-pembanding itu. Yang membedakan bukan "pakai co-occurrence atau tidak",
melainkan **bagaimana** co-occurrence dimodelkan (eksplisit & tanpa parameter terlatih vs
dipelajari & terkompresi).

**Jangan memaksakan narasi lama.** Laporkan tabel apa adanya dan biarkan peneliti memutuskan
framing RQ3.

### 3.2 Kebenaran implementasi per metode

Untuk **setiap** metode, verifikasi dan catat:

| yang diperiksa | cara |
|---|---|
| Sumber IDF | dari 340 produk saja, atau produk+query? Harus **konsisten** untuk semua metode. Catat pilihannya. |
| Normalisasi | L2 row-norm diterapkan di mana |
| Hyperparameter | nilai default vs hasil tuning; sumbernya |
| Seed | metode stokastik → `stochastic = True`, dievaluasi ≥5 seed |
| Out-of-fold | metode supervised → GroupKFold-5 by query, **tidak boleh** melihat query test |
| Skip diam-diam | B4 mengembalikan NaN jika model gagal diunduh → harus tercatat sebagai `skipped`, bukan skor 0 |
| Determinisme | jalankan 2× dengan seed sama, skor harus identik |

Untuk pembanding non-trivial, verifikasi terhadap paper aslinya (PDF ada di folder proyek):

| metode | paper referensi | yang harus dicek |
|---|---|---|
| B3_bm25 | Robertson & Zaragoza, *BM25 and Beyond* | `k1`, `b`, definisi dokumen & query, sumber IDF |
| B4_sbert | Reimers & Gurevych, *Sentence-BERT* | model checkpoint persis, pooling, apakah simetris |
| B5_word2vec | Mikolov et al. — **paper belum ada di folder** | `dim`, `window`, `epochs`, `min_count`, `workers=1` |
| B6_node2vec | Grover & Leskovec, *node2vec* | `p`, `q`, `walk_length`, `num_walks`, bobot sisi |
| A2_ppmi_svd | Levy & Goldberg — **paper belum ada di folder** | shift PPMI, `dim` SVD, apakah `randomized` |
| A1_wheel | Edwards Fragrance Wheel — **sumber belum diverifikasi** | struktur pohon, bobot sisi, massa yang hilang |

Untuk yang papernya belum ada: **tulis `SUMBER BELUM DIVERIFIKASI`** di
`10_citation_audit.md`. Jangan mengarang sitasi.

### 3.3 Uji determinisme & reproduksibilitas

```
python run_all.py --fast --seeds 3   # jalankan 2×
```
Bandingkan `04_main_table.csv` dua kali. Metode non-stokastik harus identik bit-per-bit.
Metode stokastik harus identik untuk seed yang sama.

**Output:** `results/v3/01_implementation_audit.md`
**Acceptance:** tabel P/L/U terisi penuh dengan kutipan baris kode sebagai bukti.
**Gate G2** kalau ada metode yang tidak deterministik pada seed tetap.
Commit: `feat(v3): implementation audit of 15 methods`

---

## TAHAP 4 — Protokol evaluasi (kunci ini dulu, jangan diubah lagi)

### 4.1 Task
Untuk setiap query `q` (parfum global), rangking seluruh **340** produk katalog.
Ground truth: produk yang `revolutionize`-nya menunjuk ke `q`.

### 4.2 Metrik
`MRR`, `Hits@1`, `Hits@3`, `Hits@10`. Tambahkan `Hits@10` — belum ada, murah, informatif.

### 4.3 Query dengan >1 dupe
Aturan default: **best rank** di antara semua produk relevan.
Alternatif yang wajib diuji di Tahap 9: (b) setiap pasangan jadi query terpisah,
(c) hanya dupe pertama.

### 4.4 Tie-breaking
Skor identik → **peringkat pesimistis** (produk relevan ditaruh di belakang semua yang seri).
Ini keputusan konservatif. Uji sensitivitasnya di Tahap 9.

### 4.5 Seed
Metode stokastik: **5 seed** (`0..4`). Laporkan mean ± std.
Metode supervised: GroupKFold-5 by query, ulangi untuk 3 shuffle seed.

### 4.6 Signifikansi — perbaiki 4 hal dari harness lama

1. **Jangan pakai seed-0 saja.** Harness lama menyimpan `rr` hanya untuk seed 0 lalu
   menguji dengan itu, sementara tabel MRR adalah rata-rata beberapa seed. Kumpulkan
   `rr` per query per seed, **rata-ratakan lintas seed**, baru uji.
2. **Laporkan Wilcoxon DAN bootstrap.** Keduanya menguji hal berbeda (rank vs mean).
   Kalau tidak sepakat, tulis mekanismenya:
   *"mean RR berbeda, median tidak → distribusi selisih skew"*. Jangan pakai label
   "campuran" tanpa penjelasan.
3. **Laporkan `n_nonzero`.** `scipy.stats.wilcoxon(zero_method="wilcox")` membuang
   selisih nol. Jumlah query yang tersisa harus terlihat.
4. **Koreksi multiple comparison.** Holm–Bonferroni untuk keluarga uji "order-N ladder"
   (uji berurutan, sedikit) dan Benjamini–Hochberg (FDR 0.05) untuk keluarga
   "proposed vs 14 pembanding". Laporkan `p_raw` dan `p_adj` berdampingan.

Bootstrap: 10 000 resample, paired by query, seed 0. Laporkan CI 95%.

### 4.7 Verdict signifikansi

Sebuah selisih disebut **signifikan** hanya jika **ketiganya** benar:
- `p_adj < 0.05`
- CI 95% bootstrap tidak memuat 0
- `|ΔMRR| ≥ 0.01`

Kalau hanya sebagian: tulis `AMBIGU` + jelaskan yang mana yang gagal. Jangan bulatkan
ke "signifikan".

**Output:** `src/v3/protocol.py` + bagian protokol di `REPORT_V3.md`.
Commit: `feat(v3): evaluation protocol (metrics, seeds, significance, corrections)`

---

## TAHAP 5 — Uji sirkularitas (menggantikan §8.1 pedoman lama)

`src/v3/circularity.py`. Jalankan pada dataset bersih.

### 5.1 Kenapa uji lama dibuang

`containment_benar` vs `containment_acak` **tidak diagnostik**. Dua hipotesis berbeda
memprediksi hasil yang sama:
- H-sinyal: accord produk & accord global dibuat independen, mirip karena parfumnya mirip.
- H-salin: accord produk disalin dari halaman Fragrantica parfum target.

Uji itu hanya membuktikan "ada sinyal". **Hapus** verdict "DATA BERSIH" yang lama.

### 5.2 Uji utama: order-preservation

Fragrantica menampilkan main accords **terurut menurun** menurut intensitas.
Kalau anotator menyalin, **urutan ikut tersalin**.

Untuk setiap pasangan berlabel `(p, q)`:
- `contained` = `set(A(p)) ⊆ set(A(q))`
- `subseq` = `A(p)` adalah subsequence **terurut** dari `A(q)`
- `prefix` = `A(q)[:len(A(p))] == A(p)`

**Null:** permutasi acak urutan `A(q)` (set dipertahankan), **1000 permutasi, seed 0**.
Hitung p-value empiris satu sisi untuk `subseq` dan untuk `prefix`.

### 5.3 Kontrol wajib (tanpa ini, 5.2 tidak sah)

- **K1 — konvensi sorting.** Proporsi `A(q)` dan `A(p)` yang terurut menurut
  (a) alfabetis, (b) frekuensi korpus menurun, (c) IDF menaik.
  Kalau salah satu > 50%, order-preservation **tidak informatif**. → **Gate G4**.
- **K2 — kontrol panjang.** Distribusi `len(A(p))`. Kalau terkonsentrasi di satu nilai,
  itu konvensi "ambil top-k"; laporkan.
- **K3 — kontrol negatif.** Ulangi 5.2 pada pasangan `(q, p_acak)` non-label.
  Harus turun ke level null.
- **K4 — jangan pakai distribusi typo/blob sebagai bukti.** Sudah tidak relevan (data bersih),
  dan pada data kotor pun **confounded**: token majemuk secara mekanis tidak bisa cocok,
  sehingga otomatis jatuh ke stratum non-contained. Kausalitasnya terbalik.

### 5.4 Label mislinked

Untuk pasangan yang **tidak** contained, cari `q' ≠ q` dengan `set(A(p)) ⊆ set(A(q'))`
atau `A(q')[:len(A(p))] == A(p)`. Kandidat label salah.

Nama parfum di `global_reference` lama banyak yang ambigu (`LV L'Immensite`,
`JPG Scandal Pour Home`). Kolom `ambiguity_note` sudah dihapus dari file bersih, tapi
masih ada di `archive/pre_v3/` — pakai untuk konteks.

**Daftar ini TIDAK BOLEH dipakai untuk memperbaiki label otomatis.** Hanya untuk review manusia.

### 5.5 Stratum

Definisikan stratum per query (ambil tertinggi kalau ada >1 dupe):
`prefix > subseq > contained > partial`

Definisikan dua eval set:
- **`ALL`** — semua query
- **`NON_OP`** — query tanpa dupe `prefix` maupun `subseq`

**Peringatan yang harus ditulis di laporan:** `NON_OP` bukan "subset bersih". Ia adalah
subset **tanpa bukti transkripsi berurutan**. Anotator yang menyalin lalu mengacak urutan
tidak akan terdeteksi. Frasa yang benar: *"no evidence of order-derived transcription"*.

### 5.6 Verdict

Tepat satu dari tiga, ditentukan **oleh angka**:

| verdict | kondisi |
|---|---|
| `DERIVATIF (sebagian)` | p-value empiris `prefix` atau `subseq` < 0.01 **dan** K1 lulus |
| `TIDAK TERBUKTI DERIVATIF` | p-value ≥ 0.01 |
| `TIDAK DAPAT DIUJI` | K1 gagal |

**Output:** `02_circularity.md`, `02_order_preservation.csv`, `02_null.csv`,
`02_sorting_convention.csv`, `02_suspected_mislinked.csv`, `02_strata.csv`
Commit: `feat(v3): order-preservation circularity test (verdict: <isi>)`

---

## TAHAP 6 — Order ladder + pemilihan `N*`

### 6.1 Spesifikasi order-N (tulis ulang, jangan warisi)

```
tokens(A) = { semua subset A berukuran 1..N }
x_t       = 1 jika t ⊆ A, else 0
idf_t     = log( (1 + n_pool) / (1 + df_t) ) + 1     # df dari 340 PRODUK saja
v(A)      = L2_normalize( x ⊙ idf )
score(q,p)= v(A(q)) · v(A(p))
```

Assertion wajib: **`N=1` harus mereproduksi `B2_tfidf_cos` persis** (selisih < 1e-9).
Kalau tidak, salah satu dari keduanya salah. Cari tahu yang mana. → **Gate G5**.

### 6.2 Kurva
Jalankan `N = 1..6`. Laporkan MRR, H@1, H@3, H@10, dan jumlah token bersama rata-rata
per orde (sparsity).

### 6.3 Pemilihan `N*` via nested CV

Masalah lama: `N` dipilih dengan melihat metrik pada seluruh query yang juga dilaporkan,
sementara pembanding di-tuning nested CV. Itu bias ke arah metode usulan.

- Outer: GroupKFold-5 by query, 3 shuffle seed.
- Inner: pada fold latih, pilih `N ∈ {1..6}` yang memaksimalkan MRR.
- Laporkan `N` terpilih **per fold**, modus, dan MRR out-of-fold agregat.

IDF dihitung dari 340 produk (tanpa label), jadi tidak ada kebocoran label. Tapi **catat**
bahwa IDF melihat seluruh pool, termasuk produk di fold uji — ini transduktif, konsisten
untuk semua metode, dan harus disebutkan.

### 6.4 Uji tangga

Untuk `(1,2), (2,3), (3,4), (4,5), (5,6)`:
Wilcoxon + bootstrap CI + `n_nonzero`, **Holm-corrected** dalam keluarga 5 uji ini.

Jalankan pada `ALL` **dan** `NON_OP`.

**Output:** `03_order_ladder.csv`, `03_order_significance.csv`, `03_order_selection.md`,
`03_sparsity.csv`
Commit: `feat(v3): order ladder + nested-CV selection of N`

---

## TAHAP 7 — ATURAN KEPUTUSAN (ditulis SEBELUM hasil ada)

Ini kontrak. Jangan diubah setelah melihat angka.

### 7.1 Eval set primer

- Jika verdict Tahap 5 = `TIDAK TERBUKTI DERIVATIF` → **primer = `ALL`**
- Jika verdict = `DERIVATIF (sebagian)` → **primer = `NON_OP`**, `ALL` jadi sensitivitas
- Jika verdict = `TIDAK DAPAT DIUJI` → berhenti, **Gate G4**

### 7.2 Aturan memilih `N*`

> `N*` = **N terkecil** sedemikian rupa sehingga untuk **setiap** `M > N`,
> selisih `MRR(M) − MRR(N)` **tidak signifikan** menurut §4.7 pada eval set primer.

Tambahan: `N*` juga harus menjadi modus pilihan nested CV (Tahap 6.3).
Kalau tidak sama → tulis keduanya, jangan pilih yang lebih menguntungkan.

### 7.3 Aturan stabilitas

`N*` harus **sama** pada ≥ 80% sel grid sensitivitas (Tahap 9).
Kalau tidak: laporkan `N*` sebagai **TIDAK STABIL** dan sajikan seluruh grid.
**Jangan** melaporkan satu `N*` yang kebetulan menang.

### 7.4 Aturan RQ1

RQ1 dinyatakan terdukung hanya jika `order-1 → order-2` signifikan (§4.7) pada
eval set primer **dan** pada `ALL` **dan** pada ≥ 80% sel grid sensitivitas.

Kalau RQ1 gugur → **Gate G6**, berhenti, lapor. Seluruh tesis harus dirumuskan ulang.

### 7.5 Aturan RQ3

Kategori metode ditentukan oleh tabel P/L/U (Tahap 3.1), **bukan** oleh hasilnya.
Dilarang memindahkan sebuah metode dari "pembanding" ke "varian" setelah melihat skornya.

---

## TAHAP 8 — Tabel utama & signifikansi

### 8.1 Run

```
python run_all.py --seeds 5
```

- Semua 15 metode.
- **Tuning ulang seluruh pembanding** (`src/phase2_tuning.py`) pada dataset bersih.
  `best_params` lama tidak sah — grid-nya dicari di ruang fitur yang sudah tidak ada.
- order-N memakai `N*` dari Tahap 6, bukan angka hardcoded.
- Jalankan pada `ALL` dan `NON_OP`.

### 8.2 Kolom `04_main_table.csv`

`method, P, L, U, best_params, eval_set, n_queries, MRR, MRR_std, Hits@1, Hits@3, Hits@10, skipped`

### 8.3 Kolom `05_significance.csv`

`comparison, eval_set, delta_MRR, ci_low, ci_high, wilcoxon_p_raw, p_adj_BH, n_nonzero, verdict`

Keluarga uji BH: order-N vs 14 pembanding, per eval set, **terpisah per eval set**.

### 8.4 Train/test gap

Untuk metode supervised (A3, A4, A5, A6, P2): MRR train vs MRR out-of-fold.
`04_gap.csv`. Gap besar = overfit, laporkan.

Commit: `feat(v3): main table + significance with BH correction`

---

## TAHAP 9 — Grid sensitivitas (INI YANG MEMBUAT HASIL BISA DIPERCAYA)

Peneliti tidak akan menerima satu angka dari satu konfigurasi. Jalankan grid.
Setiap sel: order ladder `N=1..6` + `N*` terpilih. Simpan MRR & `N*`.

| ID | dimensi | nilai |
|---|---|---|
| S1 | keputusan `oriental` | dihapus **/** dipetakan ke `amber` |
| S2 | keputusan `warm` | → `warm spicy` **/** dihapus |
| S3 | keputusan `white floral and tuberose` | `white floral`+`tuberose` **/** `floral`+`tuberose` |
| S4 | sumber IDF | 340 produk **/** produk + query |
| S5 | normalisasi vektor | L2 **/** tanpa normalisasi |
| S6 | bobot token | biner × IDF **/** IDF saja (tanpa indikator) |
| S7 | isi pool | 340 **/** 340 − 33 produk berlabel tanpa baris global **/** 243 berlabel saja |
| S8 | query multi-dupe | best-rank **/** pasangan terpisah **/** dupe pertama |
| S9 | tie-break | pesimistis **/** rata-rata **/** optimistis |
| S10 | fold | GroupKFold-5 **/** GroupKFold-10 **/** LOO by query |
| S11 | eval set | `ALL` **/** `NON_OP` **/** `contained` **/** `partial` |
| S12 | lexicon wheel (A1) | `map` (accord tak terpetakan dipetakan a-priori) **/** `drop` (dibuang + massa dinormalisasi ulang) |

**Minimum wajib:** S1, S2, S3 (keputusan cleaning) × S4 × S9 × S11 → jalankan penuh.
**Disarankan:** S5, S6, S7, S8 satu-satu (one-at-a-time dari konfigurasi default).
**Opsional (mahal):** S10.

Untuk S1–S3, buat varian dataset **di memori** dari Excel bersih (jangan menulis Excel baru,
jangan menaruh typo-fix di kode — ini transformasi eksplisit yang dideklarasikan, bukan cleaning).

**Output:** `07_sensitivity.csv` dengan kolom
`cell_id, S1..S11, N_star, MRR_order1..order6, rq1_significant`

Lalu satu ringkasan: **berapa persen sel yang menghasilkan `N*` yang sama?**
**berapa persen sel di mana RQ1 signifikan?**

Ini yang menjawab kekhawatiran "angkanya berubah-ubah".

Commit: `feat(v3): sensitivity grid over cleaning decisions and protocol choices`

---

## TAHAP 10 — Dekomposisi skor + null

Klaim lama "sebagian besar skor berasal dari term co-occurrence" dihitung tanpa pembanding.
Tanpa null, statistik itu tautologis: dokumen 5-accord punya 5 token orde-1, 10 orde-2,
10 orde-3, dan IDF rata-rata naik menurut orde.

Hitung `share_k = Σ_{|t|=k} v_q[t]·v_p[t] / Σ_t v_q[t]·v_p[t]` untuk **tiga** populasi:

| populasi | definisi |
|---|---|
| `true` | pasangan `(q, gold)` |
| `top1` | pasangan `(q, produk peringkat 1)` |
| `random` | `(q, produk acak non-gold)`, 20× per query, seed 0 |

Laporkan ketiganya berdampingan. Kalau `true` dan `random` mirip, statistik ini tidak
informatif dan **tidak boleh masuk paper**.

**Output:** `08_decomposition.csv` (`population, order, share, n_pairs`)
Commit: `feat(v3): score decomposition with random-pair null`

---

## TAHAP 11 — Ablation

1. **A3 ablation** — A3-full / A3 tanpa fitur bigram / A3 hanya fitur bigram.
   Menjawab: apakah kekuatan A3 datang dari co-occurrence atau dari regresi logistiknya.
2. **B4 symmetry** — bandingkan tiga varian, dengan `B4_sbert` accord-only (Tahap 2.2)
   sebagai referensi tabel utama:
   - `B4a_prose` — SBERT atas `meaning` (+`visual_note`) sisi produk vs accord sisi query.
     Sumber: `product_text.csv`. **Wajib `leakage_audit()`.** Asimetris & lintas-bahasa.
   - `B4b_accord` — accord-only kedua sisi. **Ini yang masuk tabel utama.**
   - `B4c_family_accord` — accord + `olfactory_family` / `global_family` kedua sisi.

   Menjawab: apakah B4 kalah karena arsitektur, atau karena input lamanya asimetris.
   Laporkan selisih `B4a` vs `B4b` sebagai biaya asimetri, bukan sebagai kekuatan prosa.
3. **A4 vs A3 vs order-N** — ketiganya pair-dependent tapi berbeda cara.
   Jelaskan mekanismenya: A4 *mempelajari* bobot per-bigram dari label langka;
   order-N memakai bobot IDF tetap. Ini kontras yang penting untuk paper.

**Output:** `09_ablations.csv`
Commit: `feat(v3): A3 / B4 / A4 ablations`

---

## TAHAP 12 — Audit sitasi

Untuk **setiap** metode di tabel utama, tuliskan paper yang benar-benar
diimplementasikan — bukan paper yang kebetulan mirip namanya.

| yang harus diperiksa |
|---|
| Apakah setiap pembanding di `.tex` punya sitasi? |
| Apakah sitasinya menunjuk paper yang benar? |
| Apakah checkpoint / varian model persis sama dengan paper? |

**Masalah yang sudah dilaporkan sebelumnya — verifikasi, jangan percaya begitu saja:**
- `LightGCN` dan `KGIN` diduga disitir ke paper yang salah (KR-GCN dan KEGNN).
- `BM25`, `Word2Vec`, `node2vec` diduga tidak punya sitasi.
- `Aurora and Baizal` diduga disitir sebagai "Fannisa et al." — "Fannisa" adalah nama
  depan, bukan nama keluarga.

Kalau sebuah metode disebut di `.tex` tapi **tidak ada** di `src/methods/`, tandai
`TIDAK DIIMPLEMENTASIKAN` — jangan diam-diam dibiarkan di tabel perbandingan.

**Output:** `10_citation_audit.md`
Commit: `docs(v3): citation audit`

---

## 13. GATE — berhenti dan lapor

Tulis `results/v3/GATE_<n>.md`, commit, berhenti.

| gate | kondisi |
|---|---|
| **G1** | `product_only_tokens` tidak kosong (Tahap 1) → data belum bersih |
| **G2** | ada metode non-deterministik pada seed tetap (Tahap 3.3) |
| **G3** | > 10% fragrance kehilangan > 20% massa di lexicon wheel (Tahap 2.3) |
| **G4** | kontrol sorting K1 gagal (Tahap 5.3) → uji sirkularitas tidak sah |
| **G5** | `order-N` dengan `N=1` tidak mereproduksi `B2_tfidf_cos` (Tahap 6.1) |
| **G6** | RQ1 gugur (Tahap 7.4) |
| **G7** | verdict `DERIVATIF` **dan** `NON_OP` < 100 query → power tidak cukup. Jangan tulis "N\* adalah knee"; tulis "tidak ada bukti N>N\* lebih baik" + laporkan minimum detectable effect |
| **G8** | `02_suspected_mislinked.csv` > 15 baris → integritas label dipertanyakan |

---

## 14. LARANGAN

- Jangan mengubah `dataset-aromatique.xlsx` atau `global_reference.xlsx`.
- Jangan menambahkan typo-fix, fuzzy match, atau normalisasi accord ke dalam kode.
- Jangan memakai angka dari `archive/pre_v3/`.
- Jangan memakai angka yang muncul di percakapan chat.
- Jangan menambah metode baru. Ini audit + re-run, bukan eksperimen baru.
- Jangan menyentuh `conference_101719.tex`.
- Jangan merge ke `main` sebelum §6 `HANDOFF_V3.md` terpenuhi seluruhnya.
- Jangan menulis kesimpulan tanpa CSV pendukung.
- Jangan memindahkan metode antar kategori setelah melihat skornya.
- Jangan mengganti aturan keputusan (§7) setelah melihat hasil.

---

## 15. Urutan commit

```
chore(v3): archive pre-clean artifacts, repo hygiene
feat(v3): dataset verification
fix(v3):  adapt loader to clean schema, B4 text source, wheel coverage, de-bias docstrings
feat(v3): implementation audit of 15 methods
feat(v3): evaluation protocol
feat(v3): order-preservation circularity test (verdict: <isi>)
feat(v3): order ladder + nested-CV selection of N (N*=<isi>)
feat(v3): main table + significance with BH correction
feat(v3): sensitivity grid
feat(v3): score decomposition with random-pair null
feat(v3): A3 / B4 / A4 ablations
docs(v3): citation audit
docs(v3): REPORT_V3
```

Terakhir: `pip freeze > results/v3/environment_lock.txt` (versi persis, bukan `>=`).

---

## 16. REPORT_V3.md — struktur wajib

1. Verifikasi dataset (Tahap 1)
2. Perbaikan kode & keputusan desain (Tahap 2) — termasuk pilihan B4
3. Tabel P/L/U (Tahap 3) — **apa adanya**
4. Verdict sirkularitas (Tahap 5) + apa yang masih terbuka (email ke Aromatique)
5. Tangga order + `N*` (Tahap 6) dengan `p_adj`
6. Tabel utama per kategori, dua eval set (Tahap 8)
7. Grid sensitivitas: berapa persen sel setuju (Tahap 9)
8. Kinerja per stratum containment (Tahap 5.5)
9. Dekomposisi + null (Tahap 10)
10. Ablation (Tahap 11)
11. Audit sitasi (Tahap 12)
12. **Daftar klaim lama yang gugur**, satu per satu, dengan angka pengganti
13. Ancaman validitas yang tersisa

Aturan menulis: setiap angka merujuk file CSV. Tidak ada kata "signifikan" tanpa
`p_adj` dan CI. Tidak ada kalimat evaluatif tanpa angka.

**Kalau hasil bertentangan dengan hipotesis H1–H4 di `HANDOFF_V3.md` §5, laporkan apa adanya.
Hipotesis itu boleh salah. Data yang menang.**
