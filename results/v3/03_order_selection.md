# 03 — Pemilihan N* (order ladder + nested CV)

Metrik: tie=pesimistis, multi=best. IDF dari 340 produk (transduktif — IDF melihat seluruh pool termasuk produk di fold uji; konsisten semua metode).

## Eval set: ALL

| N | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| 1 | 0.5049 | 0.3654 | 0.5817 | 0.7692 |
| 2 | 0.5663 | 0.4279 | 0.6490 | 0.8125 |
| 3 | 0.5681 | 0.4279 | 0.6587 | 0.8077 |
| 4 | 0.5763 | 0.4423 | 0.6635 | 0.8125 |
| 5 | 0.5765 | 0.4423 | 0.6635 | 0.8173 |
| 6 | 0.5765 | 0.4423 | 0.6635 | 0.8173 |

- **N\* (aturan §7.2, N terkecil yang setiap step M>N tidak signifikan)** = **2**
- **Nested-CV mode** = **5** (OOF MRR = 0.5762); picks = {5: 12, 4: 3}
- aturan §7.2 vs nested-CV mode: **BERBEDA**

## Eval set: NON_OP

| N | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| 1 | 0.4423 | 0.3211 | 0.4954 | 0.6789 |
| 2 | 0.5093 | 0.3945 | 0.5688 | 0.7064 |
| 3 | 0.5052 | 0.3945 | 0.5688 | 0.6881 |
| 4 | 0.5097 | 0.4037 | 0.5596 | 0.6881 |
| 5 | 0.5100 | 0.4037 | 0.5596 | 0.6881 |
| 6 | 0.5100 | 0.4037 | 0.5596 | 0.6881 |

- **N\* (aturan §7.2, N terkecil yang setiap step M>N tidak signifikan)** = **2**
- **Nested-CV mode** = **5** (OOF MRR = 0.5043); picks = {5: 9, 2: 4, 4: 2}
- aturan §7.2 vs nested-CV mode: **BERBEDA**

## Ringkasan

- **Eval set primer = NON_OP** (verdict Tahap 5 = DERIVATIF sebagian).
- N\* aturan (NON_OP) = 2; nested-CV mode (NON_OP) = 5.
- **Keduanya berbeda → dilaporkan keduanya, tidak memilih yang menguntungkan** (§7.2).
- Signifikansi tangga: lihat `03_order_significance.csv` (Holm, per eval set).