BRIEF REVISI PAPER — untuk Claude Code
Direktif global

Judul: ubah dari "Order-2 Co-occurrence..." → "Higher-Order Accord Co-occurrence for Interaction-Free Perfume Recommendation" (atau serupa; order-3 sebagai operating point disebut di abstract).
Metode utama = ORDER-3. Semua "order-2" lama → higher-order, order-3 sebagai titik operasi.
Batas 6 halaman IEEE 2-kolom. Struktur 5-section ala KGAT.
Sedikit angka di prosa, banyak visual. Angka detail masuk ke gambar/tabel, bukan diulang di teks.
4 gambar + 2 tabel (tidak lebih — batas halaman).
Jangan mengarang angka atau referensi. Pakai HANYA angka di blok "Angka Terkunci" di bawah.
P2/P3 dibuang dari tabel; P2 = satu kalimat di Discussion.
p-value & uji signifikansi disebut secukupnya, bukan diulang tiap section.

Angka terkunci (satu-satunya sumber angka)
Order ablation (MRR / Hits@1 / Hits@3):
order1: 0.454 / 0.335 / 0.507 (= baseline TF-IDF, B2)
order2: 0.496 / 0.378 / 0.562
order3: 0.508 / 0.397 / 0.567 ← METODE UTAMA
order4: 0.511 / 0.397 / 0.583
order5: 0.511 / 0.397 / 0.583
order6: 0.511 / 0.397 / 0.583

Signifikansi tangga order (ΔMRR, 95% CI, Wilcoxon p):
order2 vs 3: -0.0116, [-0.023,-0.002], p=0.0028 → order3 sig. lebih baik
order2 vs 4: -0.0142, [-0.027,-0.002], p=0.0005 → order4 sig. lebih baik
order3 vs 4: -0.0025, [-0.010,+0.005], p=0.017 → CI memuat 0 → SERI

Order-3 vs 12 comparator (ΔMRR order3 di atas comparator; semua p<0.01, semua sig. menang):
B1 Jaccard +0.080 | B2 TF-IDF +0.054 | B3 BM25 +0.120 | B4 SBERT +0.236
B5 Word2Vec +0.305 | B6 node2vec +0.193 | A1 wheel-treeW +0.224
A2 PPMI-SVD +0.066 | A3 signature +0.077 | A4 bigram-salience +0.379
A5 bilinear +0.344 | A6 GBM +0.063

Dekomposisi skor order-3 (ganti "81.3%" lama):
order-1: 9% | order-2: 37% | order-3: 54%

Sparsity (rata-rata token orde-k yang di-share query–target):
k1:4.20 k2:7.80 k3:7.68 k4:4.12 k5:1.22 k6:0.27

Dataset: 340 produk (243 berlabel + 97 non-target), 209 query, 56 shared accord.
P2 (LTR fusion) vs P1: p=0.18 (tidak signifikan).
I. Introduction (~1 kolom)

Konteks Aromatique: brand parfum Indonesia, banyak cabang; jam ramai → bottleneck staf; butuh recommender interaction-free (tak ada data klik/rating).
Paradigma dominan: recommender perfume berbasis konten menganggap accord independen (order-1: cosine/Jaccard/TF-IDF).
Fig. 1 dirujuk di sini (contoh masalah: order-1 mengikat produk benar & pengecoh, higher-order memisahkan).
Perfume work yang ada (satu baris masing-masing, sebagai penanda gap domain): Lutan (survey), Nurmuthia (cosine/Jaccard), Fannisa (TF-IDF) — semua order-1, tak ada yang uji co-occurrence.
Kenapa metode canggih tak dipakai (taksonomi singkat, sitasi Q1/Q2): KG/collaborative (KGAT dst.) butuh interaksi yang tak ada; embedding/learned overfit di katalog kecil.
Pendekatan kami: perluas accord TF-IDF dengan token co-occurrence higher-order, parameter-free, dievaluasi di bawah protokol non-circular.
Kontribusi (bullet): (1) temuan — co-occurrence meningkatkan retrieval, menguat dengan order lalu jenuh di order-3; (2) protokol evaluasi non-circular bebas-kebocoran. Bukan model baru.
Related work dilebur di sini (gaya KGAT), tidak jadi section terpisah.

II. Problem Formulation & Non-circular Protocol (~1 kolom)

Formalkan retrieval task: Input = reference perfume (accord set); Output = ranking 340 produk; target = produk yang dibangun untuk meniru referensi (held-out inspired_by edge sebagai ground truth).
Definisikan interaction order sebagai konsep kunci (padanan "high-order connectivity" KGAT): order-k = token subset accord ukuran k.
Bedakan deployment vs evaluation (INI kritikal — jelaskan eksplisit): mesin skor identik; deployment melayani preferensi bebas & bisa merekomendasikan produk apa pun (dupe atau original); evaluation memakai query parfum global karena hanya di situ ada ground-truth. 97 produk original = kandidat non-target dalam pool (bukan "tak penting").
Fig. 1 panel-b atau prosa: deployment vs evaluation.
Tabel I: three-source firewall (item accords lokal | query accords Fragrantica eksternal, 264/266 source_url | label inspired_by held-out) + statistik dataset.
Sebut leakage audit singkat (buang token nama-diri, dst.) — satu kalimat.

III. Method (~1.25 kolom)

Overview: token = gabungan subset accord ukuran k=1..N; IDF-weight; L2-norm; cosine; skor terdekomposisi additif per orde.
Rumus (nomori): (1) token set order-N {S ⊆ A(f) : |S|=k, k≤N}; (2) IDF idf(t)=log((1+M)/(1+df(t)))+1, M=340; (3) vektor + L2-norm; (4) cosine; (5) dekomposisi skor = S₁+S₂+S₃.
Fig. 2: pipeline (accord → token order-1/2/3 → IDF → cosine → dekomposisi). Tandai: jalur order-1 = TF-IDF biasa; jalur order-2/3 = tambahan kami.
Dekomposisi: sebutkan order-3 menyumbang 54% skor top, order-2 37%, order-1 9% (ganti 81.3%). Satu kalimat: higher-order structure memikul mayoritas sinyal.
Parameter-free: satu kalimat — tak ada bobot terlatih → tak bisa overfit (alasan prinsipil di data kecil).
kg_paths: satu kalimat — triple accord bersama dikelompokkan per super-family wheel, sebagai byproduct penjelasan (domain kolaborator). Bukan klaim faithfulness.

IV. Experiments (~2.5 kolom — bagian terbesar)
Buka dengan 3 RQ:

RQ1: Apakah co-occurrence (order≥2) meningkatkan retrieval dibanding order-1?
RQ2: Sampai order berapa peningkatannya berlanjut — adakah titik jenuh?
RQ3: Bagaimana order-3 dibanding paradigma mapan (marginal, neural, perceptual-transport, learned)?

Setup: dataset (Tabel I sudah di §II); 12 comparator lintas 4 paradigma; metrik MRR/Hits@1/Hits@3; leave-one-out 209 query; uji Wilcoxon signed-rank + bootstrap CI disebut sekali di sini sebagai metode.
RQ1 + RQ2 (jadikan satu subsection dengan Fig. 3):

Fig. 3 (performance = kurva order-ablation MRR vs max order 1→6): naik tajam 1→2, naik 2→3, mendatar 3→6.
Naskah: co-occurrence menaikkan MRR 0.454→0.508 (signifikan). Naik signifikan hingga order-3; order-4 tidak signifikan lebih baik dari order-3.
Caption Fig. 3 WAJIB memuat peringatan (biar pembaca tak salah simpul order-4 terbaik):

"Although order four attains a marginally higher point estimate, it is not significantly better than order three (ΔMRR = 0.003, 95% CI [−0.010, +0.005]); order three is adopted as the operating point."

Sparsity: order ≥5 mendatar karena token orde tinggi menjadi langka (k5=1.22, k6=0.27 shared) — sebut sebagai penjelasan plateau, dan sebagai caveat (plateau dataset-dependent).

RQ3 (Tabel II):

Tabel II: 12 comparator + order-3 (3 metrik). Bold = method of record (bukan "nilai tertinggi"). Sertakan ±std untuk baris stochastic.
Narasi margin (dua kelompok): menang tipis tapi signifikan atas metode sederhana/order-1 (B2 +0.054, A2 +0.066, A6 +0.063, B1 +0.080); menang telak atas model kompleks yang overfit (word2vec +0.305, bigram-salience +0.379, bilinear +0.344, SBERT +0.236).
Poin kunci: metode parametrik yang mencoba memanfaatkan co-occurrence (bigram-salience, bilinear) justru overfit paling parah → mendukung "simplicity as design decision".
Signifikansi headline disebut sekali: order-3 vs baseline B2 (p=5.6e-9).

RQ3 lanjut — case study (Fig. 4):

Fig. 4 (contoh nyata dari dataset, padanan Fig.4 KGAT): pilih 1 pasangan real (parfum global → produk Aromatique benar) dari datasetaromatique.xlsx+global_reference.xlsx; tampilkan triple accord bersama yang mendorong match (kg_paths). Claude Code pilih contoh yang shared-triple-nya jelas.

V. Conclusion & Future Work (~0.5 kolom)

Rekap: temuan (kurva co-occurrence + kejenuhan order-3) + protokol non-circular; scope applied/moderate.
Caveat jujur — pertahankan 7 dari draf lama (Threats to Validity): (1) efek modest; (2) satu katalog proprietary; (3) accord biner tanpa intensitas; (4) imitation = proxy perseptual, bukan kepuasan; (5) overlap ~86% = ceiling; (6) klaim "to our knowledge, first" (jangan telanjang "first"); (7) plateau order tinggi = artefak ukuran data.
P2 = satu kalimat: "Adding a learning-to-rank layer over the order-3 representation yields no significant improvement (p = 0.18), so we retain the parameter-free form."
Future work: intensity-weighted co-occurrence, symmetric-text eval, kuantifikasi faithfulness kg_paths (counterfactual), replikasi lintas katalog.

Abstract (tulis TERAKHIR, setelah isi terkunci)

4 angka saja (340, 209, 0.454→0.508). Frasa kejenuhan order-4 tidak di abstract (taruh di Results). Draf acuan sudah tersedia (versi order-3) — Claude Code boleh pakai/rapikan.

Referensi

Jangan mengarang venue/judul. Pakai .bib yang ada; reorganisasi agar Q1/Q2 memikul klaim metodologis, perfume papers hanya penanda gap (satu baris).
Q1/Q2 anchor per klaim: co-occurrence/higher-order membantu (recsys Q1) · struktur perseptual odor (Nature/Science) · overfitting small-data (Nature Methods / IEEE TMI / AI Review).
Pertahankan conference refs kanonik yang metodenya benar-benar dipakai/diuji: KGAT (justifikasi interaction-based tak dipakai), Sentence-BERT, node2vec.
Referensi baru apa pun harus diverifikasi (Consensus, exclude_preprints), bukan ditebak.

Anggaran halaman (target)
I ≈ 1 kol · II ≈ 1 kol · III ≈ 1.25 kol · IV ≈ 2.5 kol · V ≈ 0.5 kol · Ref ≈ 1 kol. Gambar 4 + Tabel 2 ≈ 5 kol. Total ≈ 6 halaman. Kalau meluber: pangkas prosa RQ3, jadikan Fig. 1 dua-panel (masalah + deploy/eval) untuk hemat satu gambar.
