# 07 — Ringkasan Grid Sensitivitas

Total sel (S1×S2×S3×S4×S9×S11) = 96

- **N\* modus = 2**, muncul di **100.0%** sel (distribusi N\*: {2: 96})
- **RQ1 (order2>order1) signifikan di 100.0%** sel
- Aturan stabilitas §7.3: N\* STABIL (ambang 80%).
- Aturan RQ1 §7.4 (butuh ≥80% sel): TERPENUHI pada dimensi grid ini.

## S8 — query multi-dupe (one-at-a-time)

| multi | order | MRR | n_units |
|---|---|---|---|
| best | order1 | 0.5049 | 208 |
| best | order2 | 0.5663 | 208 |
| separate | order1 | 0.5010 | 210 |
| separate | order2 | 0.5626 | 210 |
| first | order1 | 0.5049 | 208 |
| first | order2 | 0.5663 | 208 |

## S12 — wheel A1 (map vs drop)

| mode | MRR ALL | MRR NON_OP |
|---|---|---|
| map | 0.3027 | 0.2658 |
| drop | 0.3024 | 0.2641 |

## Catatan cakupan

- Dijalankan penuh: S1,S2,S3,S4,S9,S11 (minimum wajib) + S8, S12 one-at-a-time.
- Tidak dijalankan (opsional/mahal): S5 (norm), S6 (bobot), S7 (pool), S10 (fold).