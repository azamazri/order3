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

*(Bagian 2 — tabel P/L/U + kebenaran implementasi 15 metode — diisi di Tahap 3.)*
