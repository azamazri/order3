# BRIEF EKSEKUSI — `exp/v4-improve`

Menyambung `exp/v3-clean` (hasil final, terverifikasi). Baca `HANDOFF_V3.md` dan
`PEDOMAN_EKSPERIMEN_V2.md` lebih dulu. Aturan di kedua dokumen itu tetap berlaku penuh.

Tujuan tunggal: **menguji satu improvement terhadap baseline v3 yang dibekukan.**
Bukan menaikkan angka. Menguji hipotesis, lalu memutuskan dengan kriteria yang dikunci
**sebelum** hasil keluar.

---

## 0. PRINSIP (tidak bisa dinegosiasikan)

1. **Baseline v3 dibekukan.** Angka di `results/v3/` adalah titik referensi permanen.
   Jangan sentuh, jangan hitung ulang, jangan overwrite.
2. **Kriteria "lebih baik" dikunci di §2 SEBELUM improvement dijalankan.** Dilarang
   mengubahnya setelah melihat angka improvement. Kalau kriteria terasa salah, tulis
   alasannya, commit, berhenti, minta persetujuan — jangan diam-diam pakai kriteria lain.
3. **Metode lebih sederhana menang saat imbang.** Kalau improvement tidak mengalahkan
   baseline secara **signifikan**, baseline yang dipakai. Ini prinsip yang sama yang
   membuat P1 (nol parameter) dipilih atas P2 (terlatih) di v3.
4. **Hasil negatif adalah hasil yang sah.** Kalau improvement tidak menang, itu bukan
   kegagalan tugas — itu jawaban. Tulis apa adanya.
5. **Jangan tuning ke test set.** Semua pemilihan hyperparameter/desain terjadi di dev
   split. Test split hanya disentuh sekali, di akhir, untuk angka final.
6. **Jangan mengarang.** Token tak dikenal → crash. Angka tak terverifikasi → `NA`.
7. Dataset = file bersih v3 apa adanya. Tidak ada layer cleaning di kode.

---

## TAHAP A — Bekukan baseline & kunci split

Sebelum apa pun.

### A.1 Tag baseline
```
git checkout exp/v3-clean
git tag v3-baseline-final          # menandai commit 6a191af
git checkout -b exp/v4-improve
```

### A.2 Kunci split dev/test — SEKALI, disimpan ke file

`src/v4/make_split.py`:
- Group by query (208 query).
- **Test = 40% query, Dev = 60%**, split acak **seed 20260711**, disimpan ke
  `results/v4/split.json` (`{"dev": [query_idx...], "test": [query_idx...]}`).
- Stratifikasi split berdasarkan stratum containment (prefix/subseq/contained/partial)
  supaya proporsi `partial` di dev ≈ di test. Ini penting: stratum `partial` kecil (37),
  kalau tidak distratifikasi bisa habis di satu sisi.
- Cetak: jumlah query & jumlah `partial` di dev vs test.

**Acceptance:** `split.json` ada, test punya ≥12 query `partial`, dev punya ≥12.
Kalau tidak tercapai (partial terlalu sedikit untuk dibelah) → **Gate GV1**: lapor,
usulkan k-fold cross-val alih-alih single split, berhenti.

### A.3 Reproduksi baseline di split ini
Jalankan **P1_order2, P2_fusion, A3_signature** (tiga teratas v3) pada `split.test`.
Simpan `results/v4/A_baseline_on_test.csv` (`method, MRR, Hits@1, Hits@3, per_stratum_MRR`).

**Ini angka yang harus dikalahkan.** Bukan angka dari `results/v3/` (itu di seluruh 208
query; test split beda populasi). Semua perbandingan v4 memakai `A_baseline_on_test.csv`.

Commit: `chore(v4): freeze v3 baseline, lock dev/test split`

---

## TAHAP B — Audit stratum `partial` SEBELUM membangun apa pun

Alasan: satu-satunya celah mekanis ada di stratum `partial` (MRR baseline ~0.14 vs
0.63–0.70 di stratum lain). Tapi `partial` = 38 pasangan, paling sedikit data, dan paling
mungkin memuat label mislinked (Gate G8 di v3 menandai 17 kandidat). **Mungkin 0.14 itu
batas atas sebenarnya karena sebagian label salah, bukan kegagalan metode.** Harus dicek
dulu, sebelum waktu dihabiskan membangun soft-matching yang mengejar noise.

`src/v4/audit_partial.py`. Untuk tiap pasangan `partial` (`A(p) ⊄ A(q)`):

| kolom | isi |
|---|---|
| product, query | nama |
| missing_accords | `set(A(p)) − set(A(q))` |
| n_missing | jumlahnya |
| alt_query_contained | query lain `q'` yang `set(A(p)) ⊆ set(A(q'))` (indikasi mislink) |
| missing_has_close | apakah accord yang hilang punya "kerabat" di `A(q)` menurut PPMI co-occurrence korpus (mis. `amber`↔`warm spicy`) — ya/tidak + accord terdekat |

Klasifikasikan tiap pasangan ke salah satu:
- **`likely_mislink`** — `alt_query_contained` tidak kosong (accord produk malah cocok penuh ke query lain)
- **`soft_matchable`** — semua accord hilang punya kerabat dekat di query (soft-matching berpotensi menolong)
- **`genuinely_far`** — accord hilang tidak punya kerabat (tak ada metode leksikal/co-occurrence yang bisa menolong)

**Output:** `results/v4/B_partial_audit.csv` + ringkasan `B_partial_audit.md`
(berapa pasangan di tiap kelas).

### Gate keputusan (tulis di `B_partial_audit.md`):
- Jika `soft_matchable` < 8 pasangan → **improvement tidak layak dikejar.** Ceruknya
  terlalu kecil / terlalu banyak noise. **Lompat ke Tahap E**, tulis kesimpulan:
  baseline v3 dipertahankan, batas `partial` adalah label-noise + accord non-koheren,
  bukan kelemahan metode yang bisa diperbaiki. **Ini hasil yang sah dan mungkin.**
- Jika `soft_matchable` ≥ 8 → lanjut Tahap C.

Commit: `feat(v4): partial-stratum audit (soft_matchable=<n>)`

---

## TAHAP C — Improvement (hanya jika Tahap B lolos)

Hipotesis yang diuji (tulis di `REPORT_V4.md` sebelum jalan):

> Mengganti pencocokan accord biner dengan **soft accord-matching** (accord berkerabat
> saling menguatkan sebagian) menaikkan MRR **khusus di stratum `partial`**, tanpa
> menurunkan stratum lain.

### C.1 Definisi metode — `src/methods/p4_soft_order2.py`

Kembangkan dari P1 (`p1_order2.py`), **jangan** dari P2. Alasan: P1 nol parameter; kalau
improvement bekerja tanpa menambah parameter terlatih, klaimnya jauh lebih kuat.

Mekanisme (pilih **satu**, deklarasikan, jangan campur diam-diam):

- **C-a (disarankan, nol parameter terlatih):** soft-expansion vektor accord. Bangun
  matriks kedekatan accord `S` dari **PPMI co-occurrence pada 340 produk** (bukan dari
  label — nol kebocoran). Ganti indikator biner accord `x` dengan `x' = normalize(x + α·Sx)`,
  lalu jalankan pipeline order-2 TF-IDF seperti P1. `α ∈ [0, 1]` satu skalar, dipilih
  **di dev split**.
- **C-b (alternatif):** jarak wheel sebagai kedekatan `S` (bukan PPMI). Sama mekanismenya.

Batasan wajib:
- `α = 0` **harus** mereproduksi P1 persis (assert `max|diff| < 1e-9`). Kalau tidak, salah.
- `S` dihitung **hanya dari fitur accord**, tidak pernah dari label `revolutionize`.
- IDF, normalisasi, sumber pool: **identik** dengan P1. Satu-satunya perubahan adalah `x → x'`.
- Deterministik. Seed hanya relevan kalau ada elemen acak (seharusnya tidak ada di C-a).

### C.2 Pemilihan `α` — di DEV saja
Grid `α ∈ {0, 0.1, 0.2, ..., 1.0}`. Untuk tiap `α`, hitung MRR di `split.dev`
(agregat **dan** stratum `partial`). Pilih `α*` yang memaksimalkan MRR agregat dev.
Simpan seluruh kurva: `results/v4/C_alpha_dev.csv`.

**Test split TIDAK disentuh di tahap ini.**

**Acceptance:** `α=0` mereproduksi P1 (assert lolos). Kurva tersimpan.
Commit: `feat(v4): soft accord-matching order-2 (alpha*=<val> on dev)`

---

## TAHAP D — Evaluasi final di TEST (sekali)

`src/v4/final_eval.py`. Dengan `α*` yang sudah terkunci dari dev:

- Hitung MRR/H@1/H@3 + per-stratum untuk **P4(α\*)** dan tiga baseline di `split.test`.
- Uji signifikansi **P4 vs P1** (baseline utama) dan **P4 vs P2** di test:
  Wilcoxon + bootstrap CI 10000 (paired by query, seed 0) + `n_nonzero`.
- Karena ada 2 perbandingan utama × (agregat + stratum partial), koreksi **Holm** dalam
  keluarga itu.

**Output:** `results/v4/D_final_test.csv`, `results/v4/D_significance.csv`.

---

## TAHAP E — Keputusan (kriteria DIKUNCI, ditulis di sini sebelum hasil ada)

Terapkan **verdict §4.7** yang sama dari v3. Improvement **P4 dipakai** hanya jika,
di `split.test`:

> **MRR(P4) − MRR(P1) signifikan**: p_adj(Holm) < 0.05 **dan** CI 95% bootstrap tidak
> memuat 0 **dan** ΔMRR ≥ 0.01.

Selain aturan utama itu, catat (tidak mengubah keputusan, hanya untuk laporan):
- Apakah kenaikan datang dari stratum `partial` (sesuai hipotesis) atau dari tempat lain
  (kalau agregat naik tapi `partial` tidak bergerak → **curiga overfit**, turunkan
  kepercayaan meski lolos ambang).
- Apakah ada stratum yang **turun** akibat soft-matching (regresi tersembunyi).

Tiga kemungkinan hasil, semuanya sah:

| hasil di test | keputusan |
|---|---|
| P4 menang signifikan **dan** kenaikan di `partial` | **P4 jadi metode usulan.** P1 turun jadi ablation (`α=0`). |
| P4 menang agregat tapi `partial` tak bergerak | **tahan.** Laporkan sebagai kemungkinan overfit; default tetap P1 kecuali ada penjelasan mekanis. |
| P4 tidak menang signifikan | **P1 tetap usulan.** Tulis: soft-matching tidak memberi perbaikan yang dapat dipertahankan; batas `partial` bersifat struktural (label-noise + accord non-koheren). |

**Dilarang** menjalankan ulang dengan `α` lain, split lain, atau kriteria lain untuk
membalik hasil. Satu improvement, satu evaluasi test, satu keputusan.

---

## TAHAP F — Laporan

`results/v4/REPORT_V4.md`:
1. Hipotesis (ditulis sebelum hasil — dari Tahap C).
2. Audit `partial` (Tahap B) + apakah improvement layak dikejar.
3. Kurva `α` di dev (Tahap C).
4. Hasil test + signifikansi (Tahap D).
5. Keputusan menurut kriteria terkunci (Tahap E).
6. Kalau P1 tetap menang: satu paragraf kenapa — ini memperkuat paper, bukan melemahkan.

Setiap angka merujuk CSV di `results/v4/`. Tidak ada "lebih baik" tanpa uji §4.7.

Terakhir: `pip freeze > results/v4/environment_lock.txt`.

---

## GATE

| gate | kondisi |
|---|---|
| **GV1** | stratum `partial` tak bisa dibelah dev/test dengan ≥12 tiap sisi (A.2) → usul k-fold, berhenti |
| **GV2** | `α=0` tidak mereproduksi P1 (C.1) → implementasi salah, berhenti |
| **GV3** | improvement menaikkan MRR agregat tapi menurunkan ≥1 stratum lain >0.02 → regresi tersembunyi, lapor sebelum klaim menang |

---

## LARANGAN

- Jangan overwrite `results/v3/` atau tag `v3-baseline-final`.
- Jangan hitung `S` dari label. Hanya dari fitur accord (340 produk).
- Jangan sentuh `split.test` sebelum Tahap D.
- Jangan ubah kriteria §E setelah melihat angka.
- Jangan menjalankan >1 improvement di brief ini. Kalau C-a gagal, jangan langsung coba
  C-b untuk "menang" — laporkan C-a dulu, itu keputusan peneliti berikutnya.
- Jangan merge ke `main`.
- Jangan sentuh `conference_101719.tex`.

---

## Urutan commit

```
chore(v4): freeze v3 baseline, lock dev/test split
feat(v4):  partial-stratum audit (soft_matchable=<n>)
feat(v4):  soft accord-matching order-2 (alpha*=<val> on dev)      # jika Tahap B lolos
feat(v4):  final test evaluation + significance
docs(v4):  REPORT_V4 (decision: <P4 dipakai | P1 dipertahankan>)
```

Kalau Tahap B menghentikan (ceruk terlalu kecil), commit-nya:
```
chore(v4): freeze v3 baseline, lock dev/test split
feat(v4):  partial-stratum audit — improvement not warranted, v3 baseline retained
docs(v4):  REPORT_V4 (decision: P1 retained, partial limit is structural)
```
