# BLUEPRINT V5 — Aromatique
## Order-2 Co-occurrence TF-IDF untuk Interaction-Free Fragrance Recommendation
### A Non-Circular Study: Co-occurrence > Marginal, and Why Perceptual/Learned Smoothing Fails on Sparse Catalogs

**Versi 5.1 — Angka rigor post-implementasi (14 metode, B4 teruji)**
**Status:** Dikunci. Implementasi selesai & terreproduksi. Siap penulisan bab tesis.

---

## 0. Ringkasan Eksekutif

Penelitian Tesis S2 ini membangun **recommender parfum interaction-free** yang menghasilkan **Top-3 produk Aromatique** + **`kg_paths` JSON** untuk layer XAI kolaborator. Kontribusinya adalah **temuan + protokol** yang divalidasi non-sirkular dari 14 metode lintas paradigma:

> Untuk fragrance retrieval **interaction-free** dievaluasi **non-sirkular**, representasi **ko-okurensi accord (orde-2)** mengalahkan (a) paradigma **marginal** dominan di perfume recsys (cosine/Jaccard/BM25), (b) **embedding neural** (Word2Vec, node2vec, Sentence-BERT), (c) **perceptual-taxonomy transport** (Edwards Wheel tree-Wasserstein), dan (d) metode **learned** high-dim (supervised bigram-salience, bilinear affinity) — karena smoothing/over-parameterisasi **overfit** pada katalog sparse; sementara learned moderat (PPMI-SVD, GBM-fusion) hanya setara baseline, tidak melampaui.

Klaim dikalibrasi level S2: **finding/protocol novelty (applied/moderate)**, kuat karena rigor empiris + hasil counterintuitive + grounding Q1/Q2. Preseden: Ahlmann-Eltze et al. (Nature Methods, 2025) [R14].

**Koreksi dari V5.0:** (a) angka diperbarui ke implementasi rigor (P1=0.496, bukan pilot 0.61); (b) P2 fusion diturunkan (P1≈P2, p=0.18 n.s.) → **P1 adalah metode**; (c) narasi RQ2 dilunakkan (A2/A6 setara baseline, bukan kalah telak); (d) B4 Sentence-BERT (MRR 0.272) ditambahkan ke tabel.

---

## 1. Dua Konteks yang Harus Dipisahkan

| | **A. SISTEM (dipakai user)** | **B. EVALUASI (pembuktian ilmiah)** |
|---|---|---|
| Sumber query | Teks bebas user / parfum global | Parfum global (`global_reference.xlsx`) |
| Pool kandidat | **Seluruh 340 produk Aromatique** | 340 produk (97 original = distraktor) |
| Output | Top-3 + `kg_paths` | Top-3 + `kg_paths` |
| Kunci jawaban | Tidak ada (deploy nyata) | Ada: edge `inspired_by` di-hold-out |

Mesin penilai **identik**. Parfum global **tidak pernah** jadi output.

**Provenance data (terverifikasi):** 264/266 `source_url` di `global_reference.xlsx` → `www.fragrantica.com` (ID spesifik per parfum). Accord global diambil **independen** dari Fragrantica, bukan diisi-balik dari produk Aromatique. Benchmark non-sirkular by construction.

**Batas proxy (wajib ditulis):** uji dupe mengukur "kecocokan perseptual", bukan "kepuasan user". Overlap accord tinggi (~86%) adalah sinyal nyata (dupe memang dibuat mirip aslinya), bukan artefak konstruksi — justru menjelaskan mengapa efek P1 vs B2 modest (+0.042 MRR).

---

## 2. Masalah & Posisi Ilmiah

### 2.1 Aplikasi
Free-text fragrance recommender → Top-3 Aromatique + `kg_paths`.

### 2.2 Tugas evaluasi
Cross-reference dupe retrieval (leave-one-out): query = parfum global; jawaban benar = produk Aromatique yang `revolutionize`-nya menunjuk query. Pool = 340 (243 berlabel + 97 distraktor).

### 2.3 Kesenjangan (di-ground Q1/Q2)

**G1 — Perfume recsys masih di paradigma marginal.**
Sistem perfume dominan memakai note/accord-similarity marginal (orde-1), diakui terbatas: Lutan (RecSys 2025) [R23]; Nurmuthia et al. (Procedia CS 2025) [R24]; Fannisa (2025) [R28]. Belum ada yang uji apakah orde-2 mengalahkannya secara non-sirkular.

**G2 — Co-occurrence/higher-order terbukti unggul tapi hanya berbasis interaksi.**
HEM-GNN (Inf. Process. Manag. 2024) [R1]; IC-GAR (Neural Comput. Appl. 2022) [R2]; CoGraph (Mathematics 2025) [R3]; MMSBR (IEEE TKDE 2023) [R4]; Haddou et al. (IEEE Access 2024) [R5]. Semua butuh interaksi user-item/sesi → **gap: interaction-free, content-only, non-sirkular**.

**G3 — "Struktur/smoothing menolong" adalah asumsi belum diuji di katalog fragrance sparse.**
Asumsi ini ada di KG-rec (KGAT [R20]) dan sains perseptual (Zarzo [R10], POM [R13], Ravia [R12], Kumari [R11]). Pelajaran "sederhana > kompleks di data kecil/imbalanced" sudah Q1 [R14,R15,R16,R17] tapi belum diuji di domain fragrance interaction-free.

### 2.4 Yang TIDAK diklaim
Bukan model/arsitektur baru. Bukan mengalahkan model berbasis interaksi (LightGCN/KGAT/KGIN — butuh interaksi, tak dapat dijalankan; dinyatakan eksplisit di README). Bukan teori OT baru.

---

## 3. Fondasi Empiris (terverifikasi)

| Properti | Nilai |
|---|---|
| Produk Aromatique (pool) | 340 |
| Produk berlabel `revolutionize` | 243 (97 original tanpa label = distraktor) |
| Query berlabel (LOO) | 209 |
| Accord shared lokal∩global | 56 |
| Family lokal / global | 17 / 49 |
| Source URL Fragrantica | 264/266 (≥99% terverifikasi independen) |

### 3.1 Hasil implementasi rigor (14 metode, LOO, OOF GroupKFold, 5 seed)

| Metode | Tier | MRR | Hits@1 | Hits@3 | Vs P1 Wilcoxon p |
|---|---|---|---|---|---|
| **P1_order2** | **T3 usulan** | **0.496** | **0.378** | **0.562** | — |
| P2_fusion | T3 (n.s. vs P1) | 0.500 ±0.003 | 0.379 | 0.580 | 0.18 n.s. |
| A6_gbm_fusion | T2 | 0.457 ±0.010 | 0.325 | 0.533 | 2.3e-2 |
| B2_tfidf_cos *(main baseline)* | T1 | 0.454 | 0.335 | 0.507 | **8.7e-8** |
| P3_hubidf | T3 ablation | 0.448 | 0.335 | 0.491 | — |
| A2_ppmi_svd | T2 | 0.444 ±0.001 | 0.343 | 0.498 | 3.9e-3 |
| A3_signature | T2 | 0.433 ±0.006 | 0.303 | 0.500 | 5.7e-8 |
| B1_jaccard | T1 | 0.428 | 0.303 | 0.487 | 6.6e-3 |
| B3_bm25 | T1 | 0.388 | 0.278 | 0.433 | 8.3e-12 |
| B6_node2vec | T1 | 0.340 ±0.013 | 0.233 | 0.371 | 2.7e-11 |
| B4_sbert | T1 | 0.272 | 0.163 | 0.311 | — |
| A1_wheel_treeW | T2 | 0.284 | 0.174 | 0.306 | 3.9e-13 |
| B5_word2vec | T1 | 0.185 ±0.011 | 0.110 | 0.182 | 2.0e-16 |
| A4_bigram_salience | T2 | 0.139 ±0.012 | 0.060 | 0.145 | 2.7e-21 |
| A5_bilinear | T2 | 0.150 ±0.027 | 0.072 | 0.150 | 2.5e-21 |

**P1 vs B2 (main baseline):** ΔMRR = **+0.042**, bootstrap 95% CI = **[+0.019, +0.067]**, Wilcoxon p = **8.7×10⁻⁸**.

**P1 vs B4 (Sentence-BERT):** ΔMRR = **+0.224**, p < 1e-10. B4 kalah signifikan meski menggunakan field teks (`meaning`). Caveat wajib ditulis: query tak punya teks kaya (hanya accord + global_family) → ruang embedding asimetris; B4 kalah sebagian karena asimetri ini, bukan semata karena semantik tak berguna.

**Dekomposisi P1:** suku orde-2 (ko-okurensi) menyumbang rata-rata **81.3%** skor top-rank; orde-1 (marginal) hanya 18.7%.

**Catatan P2:** P2 (fusion LTR) tidak signifikan vs P1 (p=0.18 n.s.) → **P1 adalah metode yang dipakai**, P2 dilaporkan sebagai ablation fusion (tidak menambah nilai).

**Catatan A2/A6:** PPMI-SVD (0.444) dan GBM-fusion (0.457) setara baseline B2 (0.454) — bukan kalah telak. Narasi RQ2 yang tepat: *"learned moderat setara baseline tapi tidak melampaui P1; learned high-dim (A4/A5) dan perceptual-taxonomy (A1) overfit/kalah signifikan."*

---

## 4. Metode (matematika runtun)

Notasi: `f` = parfum; `A(f)` = himpunan accord; `N` = jumlah produk katalog.

**L1 — Parfum → graf ko-okurensi.**
```
G_f = (A(f), E_f),    E_f = {(a,b) : a,b ∈ A(f), a < b}
```
Node = accord; edge = pasangan accord yang muncul bersama dalam satu parfum.

**L2 — Kosakata token.**
```
T(f) = A(f)  ∪  E_f        (unigram accord  ∪  bigram pasangan accord)
```

**L3 — IDF per token.**
```
df(t) = |{f : t ∈ T(f)}|
idf(t) = log( (1+N) / (1+df(t)) ) + 1
```
Pasangan langka (signature unik) → idf tinggi. Accord hub (woody/citrus, df besar) → idf rendah. Penalti hub otomatis, tanpa parameter.

**L4 — Vektor dan normalisasi.**
```
v_f[t] = idf(t)  jika t ∈ T(f),  selain itu 0
v_f  ←  v_f / ‖v_f‖₂
```

**L5 — Skor kemiripan (dekomposisi).**
```
s(q,p) = ⟨v_q, v_p⟩

       = [ Σ_{a ∈ A(q)∩A(p)} idf(a)²  +  Σ_{e ∈ E_q∩E_p} idf(e)² ]
         ─────────────────────────────────────────────────────────────
                            ‖v_q‖ · ‖v_p‖

         └─── suku orde-1 (marginal) ───┘  └─── suku orde-2 (ko-okurensi) ───┘
```
Suku orde-2 menyumbang **81.3%** skor top-rank (terverifikasi implementasi).

**L6 — Ranking → Top-3.**
Urutkan 340 produk by `s(q,p)` descending → ambil 3 teratas.

**L7 — kg_paths.**
Untuk tiap produk Top-3: ekstrak edge bersama `E_q ∩ E_p`, urutkan by `idf(e)²` descending, kelompokkan per super-family Edwards Wheel:
```
path = {from: a, to: b, weight: idf(a,b)², wheel_family: superfamily(a,b),
        relation: "co_occurrence_edge"}
```

**L8 — Evaluasi.**
```
RR_q = 1 / rank(gold_q)
MRR  = (1/|Q|) Σ_q RR_q
Hits@k = (1/|Q|) Σ_q 𝟙[rank(gold_q) ≤ k]

Signifikansi: Wilcoxon signed-rank berpasangan antar-query {RR_q^proposed} vs {RR_q^baseline}
CI: bootstrap 10.000 resample atas query untuk ΔMRR
Catatan: metode deterministik → multi-seed bukan untuk Wilcoxon,
         hanya untuk metode stokastik (A2/A3/A5/A6/B5/B6) → mean±std
```

---

## 5. Struktur Framework

```
INPUT (query bebas / parfum global)
    ↓
[1] Query grounding: teks/accord → A(q) → G_q = (A(q), E_q)
    ↓
[2] Co-occurrence graph representation ★         ← titik novelty #1
    node = accord, edge = pasangan, bobot = idf rarity
    ↓
[3] Matcher: s(q,p) = ⟨v_q, v_p⟩ (parameter-free, deterministik)
    ↓
[4] Ranking 340 produk → Top-3
    ↓
[5] kg_paths: edge bersama dikelompokkan per wheel super-family ★  ← titik novelty #2
    ↓
OUTPUT: Top-3 produk + kg_paths JSON

[6] Harness non-sirkular ★                       ← titik novelty #3
    leave-one-out + leakage firewall (Fragrantica provenance verified)
    + baseline ladder (14 metode) + Wilcoxon/bootstrap
```

★ = letak novelty. Matcher sengaja parameter-free: menambah parameter → overfit (P2 n.s., A4/A5 MRR 0.14–0.15, terbukti empiris).

---

## 6. Skema `kg_paths` JSON (kontrak kolaborator XAI)

```json
{
  "query": {
    "name": "Bad Boy Extreme By Carolina Herrera",
    "global_family": "Oriental",
    "accords": ["warm spicy","amber","cacao","woody","aromatic","balsamic","patchouli","earthy","smoky"]
  },
  "recommendations": [
    {
      "rank": 1,
      "product_name": "Berrylicious",
      "local_family": "AMBER",
      "score": 0.69,
      "kg_paths": [
        {
          "type": "co_occurrence_edge",
          "from": "amber", "to": "warm spicy",
          "weight": 0.41, "wheel_super_family": "Amber",
          "relation": "shared_signature_pair"
        },
        {
          "type": "co_occurrence_edge",
          "from": "amber", "to": "cacao",
          "weight": 0.38, "wheel_super_family": "Amber",
          "relation": "shared_signature_pair"
        },
        {
          "type": "accord_node",
          "from": "amber",
          "weight": 0.12, "wheel_super_family": "Amber",
          "relation": "shared_accord"
        }
      ],
      "order2_contribution": 0.813,
      "order1_contribution": 0.187
    }
  ],
  "meta": {
    "method": "order2_cooccurrence_tfidf",
    "version": "v5.1",
    "pool_size": 340,
    "query_accords_mapped": 9
  }
}
```

Tiap path auditable: `from_accord → to_accord` (co-occurrence edge) atau `accord` (node), berbobot idf², dikelompokkan per wheel super-family. **Otomatis konsisten dengan skor** (diturunkan dari plan yang sama menghasilkan ranking).

---

## 7. Titik Novelty vs Penelitian Sebelumnya

| Lini penelitian | Yang mereka lakukan | Delta kita |
|---|---|---|
| Perfume recsys [R23,R24,R28] | marginal similarity orde-1 (cosine/Jaccard/TF-IDF accord) | **naik ke orde-2** (ko-okurensi accord-pair sebagai fitur) |
| Co-occurrence recsys [R1–R5] | ko-okurensi/higher-order, **berbasis interaksi** user-item/sesi | **interaction-free, content-only, non-sirkular** |
| KG-explainable [R18,R19,R20,R21] | path explanation, butuh interaksi, path indiscriminate | edge ko-okurensi **diskriminatif** (bobot idf²) sebagai path, tanpa interaksi |
| Perceptual/odor structure [R10–R13] | mengasumsikan geometri perseptual menolong retrieval | **membuktikan**: wheel/OT kalah signifikan (A1=0.284) di katalog sparse |
| Embedding neural [R22,R28] | Word2Vec/fastText/S-BERT accord atau teks | **kalah** dari exact co-occurrence (B4=0.272, B5=0.185, B6=0.340) |
| Learned/complex [R14–R17 konteks] | model kapasitas besar; "complex > simple" | **dibantah empiris**: A4/A5 overfit (0.139–0.150); A2/A6 hanya setara baseline |

**Pernyataan novelty (satu kalimat):**
> Demonstrasi **non-sirkular pertama** bahwa untuk fragrance retrieval **interaction-free**, **ko-okurensi accord orde-2** mengalahkan seluruh paradigma pembanding — marginal, neural-embedding, perceptual-taxonomy, dan learned — dari 14 metode, dengan penjelasan path ko-okurensi yang auditable dan faithful-by-construction.

**Irisan yang belum ditempati:**
`interaction-free × non-sirkular × orde-2 × fragrance × bantahan-smoothing-neural-perceptual`

---

## 8. Research Questions & Hipotesis (dengan hasil)

**RQ1 (utama) — Ko-okurensi vs marginal.**
Apakah ko-okurensi accord (orde-2) mengalahkan similarity marginal untuk interaction-free dupe retrieval?
→ **H1:** P1 > B2 pada MRR/Hits@k (Wilcoxon p<0.05).
→ **Hasil: TERBUKTI.** ΔMRR=+0.042, CI=[+0.019,+0.067], p=8.7×10⁻⁸. Orde-2 menyumbang 81.3% skor.
→ *Grounding: [R1–R5] (interaksi) + [R23,R24] (gap fragrance).*

**RQ2 (utama) — Apakah struktur/smoothing/neural memperbaiki atau overfit?**
Apakah perceptual-taxonomy, neural embedding, atau learned model memperbaiki di atas P1?
→ **H2a (kuat):** perceptual-taxonomy (A1) dan learned high-dim (A4/A5) kalah signifikan vs P1.
→ **Hasil: TERBUKTI.** A1=0.284 (p=3.9e-13), A4=0.139, A5=0.150, B4=0.272, B5=0.185 — semua < B2.
→ **H2b (moderat):** learned moderat (A2/A6) setara baseline, tidak melampaui P1.
→ **Hasil: TERBUKTI.** A2=0.444, A6=0.457 ≈ B2=0.454; keduanya < P1=0.496 (signifikan).
→ *Grounding: [R14,R15,R16,R17].*
→ *Caveat B4: asimetri teks query-product berkontribusi pada kekalahan S-BERT; bukan murni kegagalan semantik.*

**RQ3 (pelengkap, non-utama) — Faithfulness kg_paths.**
Apakah edge ko-okurensi yang diklaim path berkontribusi sesuai dengan kontribusi marginal sejati?
→ **H3:** counterfactual edge ablation menunjukkan korelasi kontribusi.
→ Validasi mendalam = ranah kolaborator XAI. Tesis melaporkan ringan (dekomposisi orde-1/orde-2 = proxy faithfulness awal).

---

## 9. Protokol Evaluasi (non-sirkular)

**Firewall tiga-sumber:**

| Komponen | Sumber | Independen dari |
|---|---|---|
| Fitur item | accord katalog lokal | label |
| Fitur query | accord Fragrantica (external) | katalog lokal & label |
| Label | edge `inspired_by` hold-out | accord-similarity |

**Desain:** leave-one-out; pool=340 (97 original tanpa label = distraktor nyata).
**Metrik:** MRR (primary), Hits@1, Hits@3.
**Leakage audit:** `interpreted_as` (= nama global) dan `meaning` dengan token bocor (32/340 produk) → di-strip sebelum dipakai baseline teks (B4).
**Statistik:**
- Deterministik (P1, B1, B2, B3, A1, B4): Wilcoxon signed-rank antar-query, bootstrap 95% CI.
- Stokastik (P2,A2,A3,A5,A6,B5,B6): mean±std 5 seed + Wilcoxon.
- **Bukan** multi-seed Wilcoxon untuk metode deterministik.

---

## 10. Feasibility Study (kontribusi pendukung resmi)

8 eksperimen awal + 14 metode implementasi membentuk **karakterisasi empiris** yang dilaporkan sebagai "ablation & negative results":
- Smooth/perceptual (wheel) → menekan sinyal exact-overlap → kalah.
- Neural embedding (Word2Vec, node2vec, S-BERT) → representasi accord sebagai vektor laten → kehilangan kekhasan pasangan → kalah.
- Learned high-dim (A4/A5) → overfit ekstrem (1 positif/query, imbalance parah [R16]) → kalah.
- Learned moderat (A2/A6) → setara baseline, tidak melampaui → tak cukup untuk klaim "struktur menolong".
- P1 (paling sederhana dari Tier-3) = terbaik → sejalan [R14].

Ini bukan sekadar tuning; ini **karakterisasi kondisi data** yang menjadi justifikasi desain P1.

---

## 11. Limitasi (wajib ditulis di tesis)

1. **Accord biner** (tanpa intensitas) → produk dengan set accord identik tak terbedakan; intensitas item tidak tersedia (katalog riil, fixed).
2. **N kecil** (209 query, 1 relevant/query) → power terbatas; CI dilaporkan; efek P1 vs B2 modest (+0.042).
3. **Validitas proxy** → dupe ≠ kepuasan user; overlap accord tinggi karena dupe memang dibuat mirip aslinya.
4. **Novelty di finding/protocol**, bukan model → matcher sederhana by design, bukan keterbatasan yang tidak disadari.
5. **Caveat B4** → kekalahan S-BERT sebagian karena asimetri teks query (miskin) vs produk (prosa Indonesia); bukan klaim "semantik selamanya tak berguna".
6. **Generalisasi** terbatas: satu katalog proprietary, satu domain.

---

## 12. Daftar Pustaka (Q1/Q2 prioritas)

*Kuartil dari Consensus sjr_max≤2; konfirmasi final per tahun di Scimago sebelum sidang.*

**Co-occurrence / higher-order recommendation**
- [R1] Hao, Q. dkk. (2024). Simplices-based higher-order enhancement GNN for multi-behavior recommendation. *Information Processing & Management* (Q1). [Consensus](https://consensus.app/papers/details/7ad87bb1f18f51eda95fdbcef78e193d/)
- [R2] Gwadabe, T. dkk. (2022). IC-GAR: item co-occurrence graph augmented session-based recommendation. *Neural Computing and Applications* (Q1/Q2). [Consensus](https://consensus.app/papers/details/1e2f2e6111d85312b8455ee3c73d8694/)
- [R3] Cai, C. dkk. (2025). Graph-Based Feature Crossing to Enhance Recommender Systems. *Mathematics* (Q1/Q2). [Consensus](https://consensus.app/papers/details/3bb0885f16f15c00910b12dfac0473c6/)
- [R4] Zhang, X. dkk. (2023). Beyond Co-Occurrence: Multi-Modal Session-Based Recommendation. *IEEE TKDE* (Q1). [Consensus](https://consensus.app/papers/details/a5eadc377ee952d88a753fef92a197ae/)
- [R5] Haddou, K. dkk. (2024). Leveraging Negative Co-Occurrence in Recommender Systems. *IEEE Access* (Q1/Q2). [Consensus](https://consensus.app/papers/details/28dcf0dd1a1e56ccb4b4397b0eab367b/)

**Optimal Transport / tree-Wasserstein**
- [R7] Le, T., Nguyen, T., Fukumizu, K. (2022). Generalized Sobolev Transport for Probability Measures on a Graph. *NeurIPS* (conference). [Consensus](https://consensus.app/papers/details/8512f157fdf45a719aa079264d68229f/)
- [R8] Yamada, M. dkk. (2023). Empirical Study of Self-Supervised Learning with Wasserstein Distance. *Entropy* (Q2). [Consensus](https://consensus.app/papers/details/9ba61bbd0b3053d5b62b67c9408d375a/)
- [R9] Doan, K. dkk. (2026). Ordered Tree-Sliced Wasserstein Distance for Sequential Data. *IEEE Access* (Q1/Q2). [Consensus](https://consensus.app/papers/details/9a8b9619d215569b964daac5161ef0f3/)
- [R9b] Zhou, F. dkk. (2020). Domain generalization via optimal transport with metric similarity learning. *Neurocomputing* (Q1). [Consensus](https://consensus.app/papers/details/621650b30abc5ea99a1bc3be53db296f/)

**Fragrance / perceptual odor**
- [R10] Zarzo, M. (2020). Multivariate Analysis / Standard Sensory Wheel of Fragrances. *Cosmetics* (Q2). [Consensus](https://consensus.app/papers/details/5625a4a9121358fcb9f6045ac2733e3f/)
- [R11] Kumari, S. dkk. (2023). Perceptual metrics for odorants: non-expert similarity feedback. *PLOS ONE* (Q1/Q2). [Consensus](https://consensus.app/papers/details/8853a2e354105e2d86ffb6a8f3a535b2/)
- [R12] Ravia, A. dkk. (2020). A measure of smell enables the creation of olfactory metamers. *Nature* (Q1). [Consensus](https://consensus.app/papers/details/5f1843a289df54a2a9d70608ae2a0652/)
- [R13] Lee, B.K. dkk. (2023). A principal odor map unifies diverse tasks in olfactory perception. *Science* (Q1). [Consensus](https://consensus.app/papers/details/ca4e70ae2b29579f82e3d1aac031c8bf/)

**Small-data / simple-beats-complex**
- [R14] Ahlmann-Eltze, C. dkk. (2025). Deep-learning-based gene perturbation prediction does not yet outperform simple linear baselines. *Nature Methods* (Q1). [Consensus](https://consensus.app/papers/details/a48c95b391f25de2813ae2d2bc322524/)
- [R15] Yao, P. dkk. (2021). Single Model Deep Learning on Imbalanced Small Datasets. *IEEE TMI* (Q1). [Consensus](https://consensus.app/papers/details/7ecf0d6abea35712a9a835da9569602c/)
- [R16] Li, Z. dkk. (2020). Analyzing Overfitting Under Class Imbalance. *IEEE TMI* (Q1). [Consensus](https://consensus.app/papers/details/e203f66d844e5b2db5e38d2e5c913347/)
- [R17] Rather, I.H. dkk. (2024). A review of deep learning for small datasets. *Artificial Intelligence Review* (Q1). [Consensus](https://consensus.app/papers/details/4e8e568f1f795def81a2965f05c0b413/)

**KG-explainable / cold-start recommendation**
- [R18] Ma, T. dkk. (2022). KR-GCN: Knowledge-Aware Reasoning for Explainable Recommendation. *ACM TOIS* (Q1). [Consensus](https://consensus.app/papers/details/f58087adb90f54a2bfc7192348401e90/)
- [R19] Lyu, Z. dkk. (2023). Knowledge Enhanced GNN for Explainable Recommendation. *IEEE TKDE* (Q1). [Consensus](https://consensus.app/papers/details/3b52919336a857d2a2e62491444230fd/)
- [R20] Wang, X. dkk. (2019). KGAT: Knowledge Graph Attention Network. *KDD* (conference). [Consensus](https://consensus.app/papers/details/8515b3d3c4f151ca9d2c4231ebb0efe7/)
- [R21] Huang, X. dkk. (2021). Path-enhanced explainable recommendation (PeRN). *World Wide Web* (Q1/Q2). [Consensus](https://consensus.app/papers/details/8857b574c4d355689a2ccfb2134539c8/)
- [R25] Bernardis, C. dkk. (2022). NFC: deep hybrid item-based model for item cold-start. *UMUAI* (Q1). [Consensus](https://consensus.app/papers/details/d5630fd8d4025adb8fc6d3528964ba89/)
- [R26] Kuznetsov, S. dkk. (2023). Ontology-based algorithm for cold-start. *Int. J. Data Sci. Anal.* (Q2). [Consensus](https://consensus.app/papers/details/994aa4bb63675fd3b320888cd79d33dd/)
- [R27] Deldjoo, Y. dkk. (2019). Movie genome: alleviating new item cold start. *UMUAI* (Q1). [Consensus](https://consensus.app/papers/details/846e02f29dc65b2caa05dc2181797240/)

**Representasi teks / domain perfume**
- [R22] Reimers, N., Gurevych, I. (2019). Sentence-BERT. *EMNLP* (conference). [Consensus](https://consensus.app/papers/details/f7a80d897e4a55c48497e98a2c640cfa/)
- [R23] Lutan, E.-R. (2025). Challenges in Perfume Recommender Systems. *RecSys* (conference). [Consensus](https://consensus.app/papers/details/98c5a7189d18534ba0f131d817a254bf/)
- [R24] Nurmuthia dkk. (2025). Perfume Recommendations Using Cosine and Jaccard. *Procedia CS* (conference proc.). [Consensus](https://consensus.app/papers/details/1ca68e574e305a458544e87b3a1ab733/)
- [R28] Fannisa dkk. (2025). Perfume CBF: TF-IDF mengalahkan Word2Vec. [Consensus](https://consensus.app/papers/details/11255306a8b05e20a8b2ce51d4ab05fb/)

---

## 13. Status Implementasi & Roadmap

**Selesai:**
- ✅ `src/data.py` — load, parse, join, leakage audit
- ✅ `src/wheel.py` — leksikon wheel + tree-W closed-form
- ✅ `src/methods/` — 14 metode (B1–B6, A1–A6, P1–P3)
- ✅ `src/evaluate.py` — LOO, MRR/Hits@k, Wilcoxon, bootstrap, GroupKFold
- ✅ `run_all.py` → `results/results.csv` + `results/significance.csv`
- ✅ `IMPLEMENTATION_REPORT.md`

**Berikutnya:**
- ⬜ `src/kg_paths.py` — generate `kg_paths.json` per query (deliverable kolaborator XAI)
- ⬜ Penulisan bab tesis: Metodologi (§4) → Eksperimen (§3.1) → Diskusi (RQ1/RQ2 + caveat)

---

*Angka §3.1 dari `results/results.csv` (implementasi Claude Code, terreproduksi). Sitasi §12 dari Consensus + paper yang dibaca langsung di project. Final-verify bibliografi di Scopus/Scimago sebelum sidang.*
