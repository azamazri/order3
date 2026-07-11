# 10 — Audit Sitasi (Tahap 12)

Sumber: `conference_101719.tex` (dibaca, tidak disentuh) dan `reference/reference.txt`
(kumpulan entri bib mentah). **Catatan penting:** `.tex` memakai
`\bibliography{references}` → file **`references.bib` TIDAK ADA di repo** (hanya
`reference.txt` mentah + PDF). Akibatnya pemetaan kunci `r1..r28 → paper` **tidak dapat
diverifikasi penuh** dari repo; verifikasi di bawah berbasis isi `reference.txt` dan teks.

## 1. Pemetaan metode terimplementasi → paper

| metode (src) | paper yang diimplementasikan | status sitasi |
|---|---|---|
| B1_jaccard | Nurmuthia et al. 2025 (cosine & Jaccard, `NURMUTHIA20251097`) | ADA di reference.txt; benar |
| B2_tfidf_cos | Aurora & Baizal 2025 (TF-IDF, `11157374`); Nurmuthia (cosine) | ADA; lihat isu nama di §2 |
| B3_bm25 | Robertson & Zaragoza, *BM25 and Beyond* | **PDF ada di `reference/`, TIDAK ADA entri di reference.txt → sitasi hilang** |
| B4_sbert | Reimers & Gurevych 2019, *Sentence-BERT* (`Reimers2019SentenceBERTSE`); checkpoint `paraphrase-multilingual-MiniLM-L12-v2` | ADA; paper benar |
| B5_word2vec | Mikolov et al., *word2vec* | **TIDAK ADA PDF, TIDAK ADA entri → SUMBER BELUM DIVERIFIKASI / uncited** |
| B6_node2vec | Grover & Leskovec, *node2vec* | **PDF ada, TIDAK ADA entri di reference.txt → sitasi hilang** |
| A1_wheel_treeW | wheel = pemetaan penulis (skeleton Edwards-inspired); tree-Wasserstein: Le et al. Sobolev (`le2024...`), Doan et al. Ordered Tree-Sliced W (`11520243`) | transport ADA; **Edwards wheel BELUM DIVERIFIKASI** (reference.txt hanya punya Zarzo, *sensory wheel feminine fragrances* — wheel BERBEDA) |
| A2_ppmi_svd | Levy & Goldberg (PPMI-SVD/SPPMI) | **TIDAK ADA PDF/entri → uncited.** Catatan: implementasi = PPMI biasa (tanpa shift) |
| A3/A4/A5/A6 | metode struktur/varian usulan (bukan reproduksi paper pembanding) | N/A (bukan comparator eksternal) |
| P1_order2 / P2 / P3 | metode usulan | N/A |

## 2. Isu sitasi di `.tex` (verifikasi masalah yang dilaporkan)

1. **"Fannisa et al." → salah.** `.tex` (baris ~118) menulis *"Fannisa et al. \cite{r28}
   report that a TF-IDF content-based filter outperforms a Word2Vec variant."*
   Entri `r28` (reference.txt `11157374`) beri author **"Aurora, Fannisa Eimin and
   Baizal, Z. K. A."** → "Fannisa Eimin" adalah **nama depan**; nama keluarga penulis
   pertama = **Aurora**. Seharusnya **"Aurora et al."** (atau "Aurora & Baizal").
   **CONFIRMED — salah kutip nama.**

2. **BM25, Word2Vec, node2vec = pembanding tanpa sitasi.** Ketiganya diimplementasikan
   (src/methods + tabel utama) tetapi tidak punya entri di reference.txt. BM25 & node2vec
   punya PDF di `reference/` (belum masuk bib); Word2Vec tidak ada sama sekali.
   **CONFIRMED — comparator uncited.**

3. **`references.bib` hilang.** `\bibliography{references}` menunjuk file yang tidak ada di
   repo → bibliografi paper tidak akan ter-build apa adanya. Perlu dibuat dari
   `reference.txt` + PDF sebelum submit.

4. **KGAT / LightGCN / KGIN.** `.tex` **secara eksplisit mengecualikan** ketiganya dari
   perbandingan (*"Because the catalog studied here has no interactions, KGAT, LightGCN,
   and KGIN cannot be instantiated and are excluded from comparison"*). Ketiganya **tidak
   ada** di `src/methods/` → konsisten: **TIDAK DIIMPLEMENTASIKAN, dan dinyatakan excluded
   di teks** (bukan diam-diam ditaruh di tabel perbandingan). Kekhawatiran lama (LightGCN/
   KGIN disitir salah ke KR-GCN/KEGNN) tidak dapat dikonfirmasi tanpa `references.bib`;
   KR-GCN (`10.1145/3511019`) dan KEGNN (`9681226`) ada di reference.txt sebagai
   "related reasoning models". **Perlu verifikasi pemetaan r-number saat bib dibuat.**

5. **Klaim "parameter-free" masih di `.tex`** (baris ~148: *"motivate the parameter-free
   design"*). Per Tahap 2.6 klaim ini dicabut → ganti "no learned parameters" saat revisi
   paper (paper belum disentuh; revisi setelah eksperimen final).

## 3. Aksi untuk revisi paper (bukan sekarang)
- Buat `references.bib` dari `reference.txt`; verifikasi pemetaan `r1..r28`.
- Tambah sitasi BM25 (Robertson & Zaragoza), node2vec (Grover & Leskovec), Word2Vec
  (Mikolov), PPMI-SVD (Levy & Goldberg); atau tandai `SUMBER BELUM DIVERIFIKASI`.
- Perbaiki "Fannisa et al." → "Aurora et al.".
- Perbaiki provenance wheel (skeleton Edwards, daun = penulis) dan cabut "parameter-free".
