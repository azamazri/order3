# HANDOFF_V3 — Tesis S2: Accord Co-occurrence untuk Dupe Retrieval

Menggantikan `HANDOFF.md` dan `HANDOFF_V2.md`.
Dokumen acuan eksperimen: `PEDOMAN_EKSPERIMEN_V2.md` (menggantikan `PEDOMAN_EKSPERIMEN.md`).

---

## 0. ATURAN KERJA (BACA DULU, TIDAK BISA DINEGOSIASIKAN)

1. **Paper BUKAN acuan.** `conference_101719.tex` berisi klaim yang belum terverifikasi.
   Jangan dijadikan referensi. Jangan disentuh sampai eksperimen final.
2. **Hasil eksperimen lama BUKAN acuan.** Semua isi `results/`, `results/audit/`,
   `results/rerun/` dihitung di atas dataset yang belum bersih. **Angka-angka itu mati.**
   Arsipkan, jangan dipakai.
3. **Analisis chat BUKAN acuan.** Angka apa pun yang muncul di percakapan — dari siapa pun —
   tidak sah sampai Claude Code menjalankannya di repo dan meng-commit CSV-nya.
   Satu-satunya sumber kebenaran adalah file di `results/v3/`.
4. **Peneliti sudah pernah disesatkan oleh klaim "order-3 terbaik" dan oleh klaim
   "order-2 terbaik".** Keduanya belum pernah diverifikasi pada data bersih.
   **Jangan menuliskan kesimpulan sebelum angkanya ada.**
5. **Aturan keputusan ditulis SEBELUM melihat hasil.** Lihat §3 `PEDOMAN_EKSPERIMEN_V2.md`.
   Kalau hasil tidak memenuhi aturan, jawabannya "tidak ada bukti", bukan mencari uji lain.
6. **Jangan mengarang** angka, referensi, venue, nama, atau p-value.
   Kalau tidak terverifikasi: tulis `NA` + alasan.
7. **Selisih angka tidak boleh diklaim tanpa uji signifikansi + koreksi multiple comparison.**
8. **Jangan menulis ekspektasi hasil ke dalam docstring atau komentar.**
   `src/methods/a5_bilinear.py` saat ini memuat kalimat "Expected to be weak" di docstring.
   Kalimat semacam itu harus dihapus. Cari semuanya.
9. **Data Excel adalah sumber tunggal.** Tidak boleh ada layer cleaning, typo-fix, atau
   normalisasi accord di dalam kode. Kalau ada token yang tidak dikenal → **crash**, jangan
   diam-diam diperbaiki.
10. Bahasa laporan: Indonesia + istilah teknis Inggris. Ringkas, tanpa validasi kosong.

---

## 1. Konteks

- **Peneliti:** Azam Azri Ahmad (S2, UGM). Pembimbing: Syukron Abu Ishaq Alfarozi,
  Bimo Sunarfri Hantono. Kolaborator KMITL: Kuntpong Woraratpanya.
- **Mitra:** Aromatique, brand parfum Indonesia. Katalog 340 produk.
- **Kolaborator:** Raissa — chatbot (HCI + XAI). Konsumen output `{top3_products, kg_paths}`.
- **Masalah nyata:** jam ramai di cabang, staf sedikit, antrean panjang.

**Pertanyaan penelitian:**
Apakah memodelkan **accord yang muncul bersamaan** meningkatkan kualitas rekomendasi
dibanding memperlakukan accord sebagai independen?

**RQ:**
- **RQ1** — Apakah co-occurrence (order ≥ 2) meningkatkan retrieval dibanding order-1?
- **RQ2** — Sampai order berapa peningkatannya berlanjut? Di mana titik jenuhnya?
- **RQ3** — Bagaimana dibanding paradigma mapan (marginal, neural, perceptual, learned)?

RQ explainability sengaja tidak dibuat — `kg_paths` domain kolaborator, faithfulness
belum diukur.

**Status jawaban RQ2: BELUM DIKETAHUI.** Nama repo `order3` adalah peninggalan hipotesis
lama, bukan kesimpulan. Jangan diperlakukan sebagai jawaban.

---

## 2. Repo

`https://github.com/azamazri/order3` (publik). Clone penuh, **jangan** `--depth 1`.

Branch lama — semuanya dihitung di atas data kotor:

| branch | isi | status |
|---|---|---|
| `main` | hasil order-2 lama | **SUPERSEDED** |
| `exp/order-ablation` | kurva order-N | **SUPERSEDED** |
| `exp/fair-baselines` | Fase 1–2 (bugfix A5, nested-CV tuning) | **SUPERSEDED** |
| `exp/audit-v2` | Langkah 1–4 | **SUPERSEDED** |
| `audit/citations` | `verification_report.md` | masih relevan, belum selesai |

Branch kerja baru: **`exp/v3-clean`** dari `main`.

---

## 3. Apa yang berubah sejak HANDOFF_V2

### 3.1 Dataset sudah dibersihkan di file Excel, bukan di kode

Kedua `.xlsx` diganti dengan versi bersih. Perubahan tercatat di `cleaning_changelog.csv`
(kolom: `excel_row, product_name, before, action, after`; `action ∈ {typo_fix, split, drop, dedup}`).

**`dataset-aromatique.xlsx`**
- Kolom `meaning` **dihapus**.
- Kolom `revolutionize` sekarang berisi nama parfum versi terkoreksi (dulu `interpreted_as`).
- `main_accords` bersih: typo diperbaiki, sel majemuk dipecah jadi accord terpisah,
  label piramida nota (`haert:`, `bae:`, `baae:`) dihapus, token `oriental` dan
  `aromatique` dihapus.

**`global_reference.xlsx`**
- Header pindah ke **baris 1** (dulu baris 2).
- Kolom nama sekarang **`perfume_name`** (dulu `Revolutionize` + `interpreted_as`).
- Kolom `Revolutionize`, `source_url`, `ambiguity_note`, `Unnamed: 16` **dihapus**.
- Baris duplikat (nama sama) dihapus setelah diverifikasi accord-nya identik.
- `accord_1..accord_10` bersih dan sudah dirapatkan ke kiri.

**Konsekuensi yang HARUS diverifikasi Claude Code sebelum eksperimen apa pun** (§1
`PEDOMAN_EKSPERIMEN_V2.md`): jumlah produk, jumlah query, jumlah pasangan berlabel,
ukuran kosakata, dan **berapa query yang punya lebih dari satu produk dupe** (penggabungan
baris duplikat global dapat membuat dua produk menunjuk ke satu parfum yang sama).

### 3.2 Tiga keputusan cleaning yang bersifat judgement call

Ini **bukan fakta**, ini keputusan peneliti. Karena itu wajib diuji sensitivitasnya (§6).

| token asli | keputusan | alternatif yang harus diuji |
|---|---|---|
| `oriental` (beberapa produk) | dihapus | dipetakan ke `amber` |
| `warm` | → `warm spicy` | dihapus |
| `white floral and tuberose` | → `white floral`, `tuberose` | → `floral`, `tuberose` |

### 3.3 Bukti sirkularitas yang belum tuntas

Sebagian daftar accord produk tampak identik dengan daftar accord parfum yang ditirunya,
**termasuk urutannya**. Kalau benar, `A(produk)` adalah encoding dari label `revolutionize`,
dan MRR mengukur decoding, bukan retrieval.

Uji lama (`containment_benar` vs `containment_acak`, `PEDOMAN_EKSPERIMEN.md` §8.1)
**tidak bisa membedakan** dua hipotesis ini: sinyal nyata dan penyalinan sama-sama
memprediksi `containment_benar >> containment_acak`. Uji itu **dibatalkan**.
Penggantinya: uji order-preservation (§4 `PEDOMAN_EKSPERIMEN_V2.md`).

**Aksi manusia (paralel, tidak memblokir Claude Code):**
Peneliti mengirim satu pertanyaan ke Aromatique — *"kolom `main_accords` diisi bagaimana:
dari penciuman/brief internal, atau menyalin dari halaman Fragrantica parfum yang ditiru?"*
Jawabannya masuk ke Limitations, bukan ke kode.

---

## 4. Peran

| aktor | tugas |
|---|---|
| Peneliti | keputusan ilmiah, komunikasi dengan Aromatique, review |
| Claude (chat) | analisis, review kritis, menyusun brief. **Tidak menjalankan eksperimen.** |
| Claude Code | menjalankan eksperimen, menulis kode, commit CSV, menulis `.tex` |

Satu-satunya angka yang sah adalah angka yang Claude Code hasilkan dan commit.

---

## 5. Hipotesis yang sedang diuji (bukan kesimpulan)

- **H1** — Order-2 lebih baik daripada order-1. *(RQ1)*
- **H2** — Ada titik jenuh `N*`. Nilai `N*` **tidak diketahui**. Kandidat: 2, 3, 4. *(RQ2)*
- **H3** — Metode yang mengonsumsi representasi co-occurrence mengungguli yang tidak,
  lintas paradigma. *(RQ3)*
- **H4** — Sebagian pasangan berlabel bersifat derivatif (accord produk diturunkan dari
  accord target), dan kinerja pada subset itu berbeda dari subset lain.

H1–H4 semuanya bisa gugur. Kalau H1 gugur, seluruh tesis harus dirumuskan ulang —
lapor segera, jangan lanjut.

---

## 6. Definisi "selesai"

Eksperimen dinyatakan selesai ketika **semua** ini ada di `results/v3/` dan ter-commit:

- [ ] `00_dataset_verification.md` — sanity check dataset bersih
- [ ] `01_implementation_audit.md` — audit implementasi 15 metode
- [ ] `02_circularity.md` — verdict order-preservation
- [ ] `03_order_ladder.csv` + `03_order_selection.md` — `N*` terpilih via nested CV
- [ ] `04_main_table.csv` — 15 metode × ≥2 eval set
- [ ] `05_significance.csv` — Wilcoxon + bootstrap + Holm/BH
- [ ] `06_stratified.csv` — kinerja per stratum containment
- [ ] `07_sensitivity.csv` — grid keputusan cleaning + hyperparameter
- [ ] `08_decomposition.csv` — dekomposisi skor + null pasangan acak
- [ ] `09_ablations.csv` — A3 ablation, B4 symmetry
- [ ] `10_citation_audit.md` — setiap pembanding ↔ paper yang benar
- [ ] `environment_lock.txt` — `pip freeze`, versi persis
- [ ] `REPORT_V3.md` — sintesis, semua angka merujuk CSV di atas

Baru setelah itu paper direvisi.
