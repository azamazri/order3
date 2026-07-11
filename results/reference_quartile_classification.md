# Klasifikasi Kuartil Referensi (folder `reference/`)

Venue diverifikasi dari `reference/reference.txt`, halaman-1 tiap PDF, dan Consensus
(mis. principal odor map → Science; IndoSBERT → ICAICTA 2023). **Kuartil = Scimago SJR**
(Consensus hanya mencetak nama venue, bukan kuartil); kuartil bersifat per-kategori
subjek/tahun, jadi kasus batas ditandai `Q1/Q2`. Untuk klaim resmi tesis, verifikasi
angka SJR terakhir di scimagojr.com.

Total: **33 PDF** referensi.

## A. Jurnal Q1 (16)

| # | Paper | Venue | Kuartil |
|---|---|---|---|
| 1 | A measure of smell … olfactory metamers | Nature | Q1 |
| 2 | A principal odor map … olfactory perception | Science | Q1 |
| 3 | Deep-learning gene perturbation … linear baselines | Nature Methods | Q1 |
| 4 | Beyond Co-Occurrence: Multi-Modal Session-Based Rec | IEEE TKDE | Q1 |
| 5 | Knowledge Enhanced GNN for Explainable Rec | IEEE TKDE | Q1 |
| 6 | KR-GCN: Knowledge-Aware Reasoning GCN | ACM TOIS | Q1 |
| 7 | Simplices-based higher-order enhancement GNN | Information Processing & Management | Q1 |
| 8 | Analyzing Overfitting Under Class Imbalance … Segmentation | IEEE Trans. Medical Imaging | Q1 |
| 9 | Single Model Deep Learning … Skin Lesion | IEEE Trans. Medical Imaging | Q1 |
| 10 | Breaking the data barrier … small datasets | Artificial Intelligence Review | Q1 |
| 11 | Domain generalization via optimal transport … | Neurocomputing | Q1 |
| 12 | Movie genome … new item cold start | User Modeling & User-Adapted Interaction | Q1 |
| 13 | NFC: deep hybrid item-based … cold-start | User Modeling & User-Adapted Interaction | Q1 |
| 14 | Perceptual metrics for odorants … | PLOS ONE | Q1 |
| 15 | BM25 and Beyond | Foundations & Trends in Information Retrieval | Q1 |
| 16 | Path-enhanced explainable recommendation with KG | World Wide Web (Springer) | Q1/Q2 |

## B. Jurnal Q2 (7)

| # | Paper | Venue | Kuartil |
|---|---|---|---|
| 17 | Graph-Based Feature Crossing to Enhance Rec | Mathematics (MDPI) | Q1/Q2 |
| 18 | IC-GAR: item co-occurrence graph … session rec | Neural Computing & Applications | Q2 |
| 19 | An Empirical Study of SSL with Wasserstein Distance | Entropy (MDPI) | Q2 |
| 20 | Beyond Positive Similarity … Negative Co-Occurrence | IEEE Access | Q1/Q2 |
| 21 | Ordered Tree-Sliced Wasserstein Distance | IEEE Access | Q1/Q2 |
| 22 | Improving recommendation diversity & serendipity … cold start | Int. J. of Data Science and Analytics | Q2 |
| 23 | Sentiment-driven community detection … perfume | Applied Network Science (SpringerOpen) | Q2 |

## C. Jurnal Q3 (1)

| # | Paper | Venue | Kuartil |
|---|---|---|---|
| 24 | Multivariate Analysis … Sensory Wheel (140 perfumes) | Cosmetics (MDPI) | Q3 |

## D. Conference paper (8)

| # | Paper | Venue |
|---|---|---|
| 25 | Challenges in Perfume Recommender Systems | ACM RecSys 2025 |
| 26 | Perfume Product Selection … Content-based (Aurora & Baizal) | IEEE ICoDSA 2025 |
| 27 | Perfume Recommendation … BERT/RoBERTa/DistilBERT | IEEE conf. (BINUS) |
| 28 | IndoSBERT … Siamese Networks Fine-tuning | IEEE ICAICTA 2023 |
| 29 | KGAT: Knowledge Graph Attention Network | ACM SIGKDD (KDD) 2019 |
| 30 | node2vec: Scalable Feature Learning for Networks | ACM SIGKDD (KDD) 2016 |
| 31 | Sentence-BERT … Siamese BERT-Networks | EMNLP 2019 |
| 32 | Personalized Perfume Rec … Cosine & Jaccard (Nurmuthia) | Procedia Computer Science (proceedings) |

## E. Preprint / lainnya (1)

| # | Paper | Venue |
|---|---|---|
| 33 | Generalized Sobolev Transport … Graph | arXiv 2024 (diterima ICML 2024) |

## Ringkasan
- **Jurnal Q1: 15** (+1 batas Q1/Q2 = World Wide Web) → 15–16
- **Jurnal Q2: 7** (3 batas Q1/Q2: Mathematics MDPI, IEEE Access ×2)
- **Jurnal Q3: 1** (Cosmetics)
- **Conference: 8** (termasuk Procedia CS = proceedings; Sentence-BERT versi terbit = EMNLP 2019 meski `reference.txt` menulis "ArXiv")
- **Preprint: 1** (Generalized Sobolev Transport)

## Catatan verifikasi
- Kuartil batas (Q1/Q2): World Wide Web, Mathematics MDPI, IEEE Access — SJR berbeda antar
  kategori subjek/tahun; cek Scimago untuk kategori spesifik yang diklaim.
- **Sentence-BERT**: di `reference.txt` venue tertulis "ArXiv"; paper terbit di EMNLP 2019
  (konferensi) — perbaiki di bib.
- **Procedia Computer Science** (Nurmuthia): seri prosiding ber-ISSN; sering tidak dihitung
  sebagai jurnal Q1/Q2.
- **BM25 and Beyond** dan **node2vec** ada di folder tetapi tidak punya entri di
  `reference.txt` (uncited) — lihat `results/v3/10_citation_audit.md`.
