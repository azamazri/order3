# REPORT_V4 — Uji satu improvement atas baseline v3 (soft accord-matching)

Menyambung `exp/v3-clean` (baseline dibekukan, tag `v3-baseline-final` = 6a191af).
Tujuan: menguji **satu** improvement dengan kriteria yang dikunci **sebelum** hasil keluar.
Setiap angka merujuk CSV di `results/v4/`.

**Keputusan: P1_order2 (baseline v3) DIPERTAHANKAN.** Improvement (P4 soft accord-matching)
**tidak dibangun** karena gate kelayakan Tahap B tidak terpenuhi. Ini hasil negatif yang
sah (BRIEF §0.4).

---

## 1. Hipotesis (ditulis sebelum melihat hasil test)

> Mengganti pencocokan accord biner dengan **soft accord-matching** (accord berkerabat
> saling menguatkan sebagian, kedekatan dari PPMI co-occurrence 340 produk) menaikkan MRR
> **khusus di stratum `partial`**, tanpa menurunkan stratum lain.

Satu-satunya celah mekanis ada di stratum `partial`: MRR baseline test hanya **0.06**
(P1 0.0634, P2 0.0619, A3 0.0542; `A_baseline_on_test.csv`) vs 0.63–0.79 di stratum lain.

## 2. Audit stratum `partial` (Tahap B — `B_partial_audit.csv`, `B_partial_audit.md`)

Sebelum membangun apa pun, 38 pasangan berlabel non-contained (`A(p) ⊄ A(q)`) diklasifikasi.
Kedekatan accord dari **PPMI co-occurrence 340 produk saja** (nol kebocoran label;
"kerabat" = top-5 tetangga PPMI, PPMI>0).

| kelas | jumlah | arti |
|---|---|---|
| `likely_mislink` | **17** | accord produk malah **cocok penuh ke query lain** → masalah label, bukan metode |
| `soft_matchable` | **7** | semua accord hilang punya kerabat di `A(q)` → soft-matching berpotensi menolong |
| `genuinely_far` | **14** | accord hilang tak punya kerabat co-occurrence → tak ada metode leksikal/co-occurrence yang bisa menolong |

**Catatan konvergen:** 17 `likely_mislink` = himpunan yang sama dengan 17 kandidat Gate G8
di v3. Stratum `partial` **didominasi label-noise (17) + accord non-koheren (14)**; ceruk
yang benar-benar bisa ditolong metode hanya **7 pasangan**.

### Gate kelayakan (BRIEF §B, dikunci sebelum hasil)
> Jika `soft_matchable` < 8 → improvement tidak layak dikejar; lompat ke Tahap E.

`soft_matchable = 7 < 8` → **TIDAK LAYAK.** Sesuai aturan, Tahap C (bangun P4) dan Tahap D
(evaluasi test) **tidak dijalankan**. Test split **tidak disentuh**.

*(Transparansi: angka 7 berada tepat di bawah ambang. Definisi "kerabat" = top-5 PPMI
dikunci di kode sebelum dijalankan; tidak di-relaksasi untuk membalik hasil (BRIEF §E:
dilarang mengganti kriteria setelah melihat angka). Bahkan bila ambang dilonggarkan,
kesimpulan kualitatif tidak berubah: ceruk yang bisa ditolong (≤ ~7) tetap kalah jauh
dibanding label-noise (17) + non-koheren (14), sehingga soft-matching mengejar sinyal yang
sebagian besar adalah noise.)*

## 3. Kurva `α` di dev (Tahap C)

**NA — tidak dijalankan** (gate Tahap B menghentikan). P4 tidak dibangun.

## 4. Hasil test + signifikansi (Tahap D)

**NA — tidak dijalankan.** `split.test` sengaja tidak disentuh (BRIEF §0.5, §D).

## 5. Keputusan menurut kriteria terkunci (Tahap E)

Kriteria: P4 dipakai hanya jika di test `MRR(P4) − MRR(P1)` signifikan (p_adj Holm<0.05,
CI 95% bootstrap tak memuat 0, ΔMRR≥0.01). Karena P4 tidak dibangun (gate B), kriteria ini
**tidak terpicu**. Hasil (baris ke-3 tabel keputusan BRIEF §E):

> **P1 tetap usulan.** Soft-matching tidak memberi perbaikan yang dapat dipertahankan;
> batas `partial` bersifat **struktural** — label-noise (17/38) + accord non-koheren (14/38).

## 6. Kenapa P1 tetap menang (memperkuat paper)

Batas kinerja di stratum `partial` **bukan** kelemahan pemodelan yang bisa ditambal dengan
soft-matching. Dari 38 pasangan yang gagal containment, **31 (17+14) berada di luar
jangkauan metode apa pun**: 17 karena accord produk sebenarnya cocok ke parfum lain (label
ambigu/duplikat varian — konsisten dengan temuan Gate G8 v3), 14 karena accord yang hilang
tidak punya kerabat co-occurrence di korpus (tidak ada sinyal leksikal maupun co-occurrence
yang menghubungkannya). Menambahkan soft-matching hanya akan mengejar 7 pasangan sisa —
sambil berisiko meregresi stratum lain (GV3) dan menambah satu hyperparameter `α` demi
ceruk yang lebih kecil daripada noise label di sekitarnya.

Ini **memperkuat** klaim paper: metode order-2 nol-parameter sudah menyentuh batas yang
dapat dicapai oleh sinyal accord pada data ini; sisa gap adalah masalah **kualitas label
dan koherensi accord**, bukan kapasitas model. Prinsip "yang lebih sederhana menang saat
imbang" (yang memilih P1 atas P2 di v3) tetap berlaku: tanpa bukti kemenangan signifikan,
baseline nol-parameter dipertahankan.

---

## Gate status v4
- **GV1** tidak aktif (partial dev=22/test=15 ≥12; `split.json`).
- **GV2** tidak relevan (P4 tak dibangun).
- **GV3** tidak relevan (tak ada evaluasi improvement).

## Ringkasan
Satu improvement diuji kelayakannya, dihentikan di gate audit dengan kriteria terkunci,
tanpa menyentuh test split. **Keputusan: P1_order2 dipertahankan; batas stratum `partial`
struktural (label-noise + accord non-koheren), bukan kelemahan metode.**
