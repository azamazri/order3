# 02 — Uji Sirkularitas (order-preservation)

n pasangan berlabel = 210

## Observasi (§5.2)

| statistik | observed | null mean | null p95 | p empiris |
|---|---|---|---|---|
| contained | 172 | (set-based, permutation-invariant) | | |
| subseq | 100 | 3.37 | 6.0 | 0.0010 |
| prefix | 89 | 0.19 | 1.0 | 0.0010 |

## K1 — konvensi sorting (Gate G4 jika ada > 0.50)

| konvensi | proporsi |
|---|---|
| Aq_alpha_asc | 0.0000 |
| Aq_freq_desc | 0.0000 |
| Aq_idf_asc | 0.0000 |
| Ap_alpha_asc | 0.0143 |
| Ap_freq_desc | 0.0143 |
| Ap_idf_asc | 0.0143 |

K1 fail = False

## K2 — kontrol panjang len(A(p))

{2: 3, 3: 5, 4: 10, 5: 188, 6: 2, 8: 1, 9: 1}

> len(A(p)) terkonsentrasi → konvensi 'ambil top-k'; dilaporkan.

## K3 — kontrol negatif (q, produk acak non-label)

- subseq = 1, prefix = 0 (harus ~ null mean: 3.4/0.2)

## §5.4 — suspected mislinked (non-contained dgn kandidat q' lain)

- jumlah = 17 (Gate G8 jika > 15). Lihat `02_suspected_mislinked.csv`. **Hanya untuk review manusia — jangan auto-fix.**

## §5.5 — strata & eval set

- distribusi stratum query: {'partial': 37, 'prefix': 88, 'contained': 72, 'subseq': 11}
- NON_OP (query tanpa dupe prefix/subseq) = 109 dari 208
- **NON_OP bukan 'subset bersih'**: ia subset *tanpa bukti transkripsi berurutan*. Anotator yang menyalin lalu mengacak urutan tak terdeteksi.

## VERDICT (§5.6)

**DERIVATIF (sebagian)**

**GATE: G8**