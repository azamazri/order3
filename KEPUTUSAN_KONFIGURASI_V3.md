# KEPUTUSAN KONFIGURASI V3

Tiga hal yang tadinya terbuka. Peneliti mendelegasikan keputusannya.
Sudah dimasukkan ke `PEDOMAN_EKSPERIMEN_V2.md` (Tahap 2.2, 2.3, 9, 11).
Dokumen ini menjelaskan **alasannya**, supaya bisa dibantah kalau salah.

Dataset yang dipakai: **file bersih apa adanya.** Tidak ada kolom yang dikembalikan.

---

## 1. Loader — `src/data.py`

Murni mekanis. Tiga titik.

```python
# 1. pembacaan global_reference (header sekarang di baris 1)
g = pd.read_excel(global_xlsx).dropna(how="all")

# 2. kunci join
nm = norm_name(row.get("perfume_name"))

# 3. nama untuk laporan
"name": str(row.get("perfume_name")).strip(),
```

Plus: **hapus** `Product.meaning` dan `Product.text_clean` dari `load_dataset()`,
serta pemanggilan `leakage_audit()` di dalamnya. Keduanya pindah ke modul ablation B4a.

`norm_name()` **tidak** diubah.

---

## 2. B4 — SBERT

**Keputusan: `B4_sbert` di tabel utama = accord-only, simetris.**
Prosa turun jadi ablation `B4a_prose`, memakai file bantu `product_text.csv`.

### Kenapa bukan `meaning` (walau bisa dikembalikan)

Kolom `meaning` **bukan prosa**. Panjang median beberapa kata, campuran Indonesia–Inggris.
Contoh: `"Aroma Cacao Creamy Dan Incense Yang Lembut"`.

### Kenapa bukan `visual_note`

`visual_note` **bukan teks bebas**. Isinya dua token bergaya nota — `"Amber / Cacao"` —
dan `visual_note_alt` adalah dua token yang sama dengan urutan dibalik.
Memberi SBERT input dua kata lalu menyebutnya "paradigma neural text" tidak jujur.

### Kenapa accord-only

Setup lama membandingkan **frasa Indonesia di sisi produk** melawan **daftar accord Inggris
di sisi query**. Asimetris dan lintas-bahasa. Kalau B4 kalah di situ, kita tidak tahu
penyebabnya arsitektur atau asimetri input. Itu **confound**, bukan hasil.

Accord-only memberi SBERT **input yang persis sama** dengan semua metode lain. Itu satu-satunya
bentuk yang menjawab RQ3: *apakah sentence encoder terlatih mengungguli pembobotan leksikal
pada input identik?* B4 tetap mewakili paradigma yang berbeda (semantik padat terlatih vs
leksikal jarang) — cuma tidak lagi mewakili "teks bebas", karena teks bebas itu memang
tidak pernah benar-benar ada di katalog ini.

### Biaya yang harus ditulis di Limitations

Setelah pembersihan, katalog tidak punya teks bebas panjang. Paradigma "neural text"
diwakili sentence encoder atas string accord. `B4a_prose` melaporkan berapa besar
kontribusi frasa pendek tersebut, dan **wajib** lewat `leakage_audit()` karena
`meaning` bisa memuat token khas nama parfum yang ditiru.

**File bantu `product_text.csv`** (340 baris): `product_name, meaning, visual_note,
visual_note_alt, olfactory_family`. Tidak memuat accord. Hanya dibaca oleh modul B4a.

---

## 3. Wheel lexicon — A1

**Keputusan: petakan accord yang belum terpetakan, sekali, sebelum eksperimen apa pun.
Perilaku lama (buang + normalisasi ulang) turun jadi sel sensitivitas `S12=drop`.**

### Kenapa bukan "buang saja"

Perilaku sekarang: accord di luar lexicon dibuang, massa distribusi dinormalisasi ulang.
Itu artinya A1 menerima **input yang lebih miskin** daripada pembanding lain. Kalau A1 kalah,
kita tidak tahu penyebabnya metodenya atau lexicon-nya yang tidak menutupi kosakata.
Pembanding yang dilumpuhkan membuat RQ3 tidak berarti.

### Kenapa memetakan itu sah

Pemetaan accord → wheel adalah keputusan **taksonomik**, dibuat tanpa melihat label maupun
skor. Tidak ada kebocoran. Yang **tidak** sah adalah menyesuaikan penempatan setelah melihat
hasil A1 — dan itu dilarang keras di pedoman.

Perlu ditegaskan: `WHEEL` di `src/wheel.py` **bukan** taksonomi Edwards yang diterbitkan.
Edwards wheel tidak punya simpul di level accord. Peta accord→subfamily di repo itu sudah
merupakan karya penulis sejak awal. Jadi klaim "frozen a-priori dari Edwards" hanya berlaku
untuk struktur superfamily/subfamily, bukan untuk daun accord-nya. Perbaiki kalimat itu di
docstring.

### Pemetaan yang diusulkan

Claude Code **wajib menghitung sendiri** `V_canonical \ WHEEL_lexicon`.
Kalau hasilnya bukan persis tiga token ini → berhenti, lapor.

| accord | superfamily / subfamily | keyakinan |
|---|---|---|
| `orange` | Fresh / Citrus | tinggi |
| `mineral` | Fresh / Water | tinggi |
| `anis` | Fresh / Aromatic | **rendah** |

`anis` adalah judgement call saya. Anise punya karakter aromatik–licorice dan bisa
dibenarkan di Fresh/Aromatic maupun di Amber/Amber (spicy). **Ini perlu keputusan peneliti
atau pembimbing.** Sampai ada keputusan, `anis` ditandai judgement call di laporan, dan
kedua penempatan masuk grid sensitivitas.

Setelah di-commit: **beku**. Dilarang diubah setelah skor A1 terlihat.

---

## 4. Yang tidak saya putuskan

Dua hal ini bukan keputusan konfigurasi, dan tidak bisa diselesaikan dengan kode:

1. **Penempatan `anis`.** Butuh peneliti/pembimbing.
2. **Provenance `main_accords`.** Butuh jawaban Aromatique. Masuk Limitations, apa pun
   jawabannya. Tidak memblokir eksekusi Claude Code.

---

## 5. Yang berubah di pedoman

| bagian | perubahan |
|---|---|
| Tahap 0.3 / 0.4 | `product_text.csv` masuk daftar file, ditandai "khusus ablation B4a" |
| Tahap 2.2 | tiga opsi B4 → satu keputusan + alasan + konsekuensi Limitations |
| Tahap 2.3 | "jangan tambal lexicon" → "petakan sekali, bekukan, ujikan sensitivitasnya" |
| Tahap 9 | dimensi baru `S12` (map vs drop) |
| Tahap 11 | ablation B4 ditulis ulang: `B4a_prose` / `B4b_accord` / `B4c_family_accord` |
| Gate G3 | syaratnya diperketat: hanya memblokir kalau `S12=map` pun tidak menutupi |
