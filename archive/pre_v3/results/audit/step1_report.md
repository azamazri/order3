# Langkah 1 -- Uji Sirkularitas Data

## 1.1 Containment: benar vs acak

| distribusi | mean | median | prop fully-contained (c==1) | n |
|---|---|---|---|---|
| c_true (pasangan benar) | 0.8597 | 1.0000 | 0.6952 | 210 |
| c_rand (query acak) | 0.3505 | 0.4000 | 0.0184 | 4185 |

Selisih mean (true - rand) = **+0.5092**.

## 1.2 Kosakata

- |V_produk| = 110, |V_query| = 60, |V_produk ∩ V_query| = 55
- accord HANYA di produk (tak pernah di query): **55**
- accord HANYA di query: **5**
- V_produk ⊆ V_query ? **False**
- produk-only: aldehydes, amber powdery, aromatique, aroomatic, baae: warm spicy, bae: powdery musky, citrus caramel, citrus floral, citrus white floral, citrus woody, coffe, floral fruity, freahspicy, freesh, fresh spicy. citrus, fruity aromatic, fruity caramel, fruity fresh, fruity iris, fruity musky, fruity rose, fruity vanilla rose, fuity, greeen, haert: white floral warm spicy, iris woody, metalic, musky rose, nuity, oriental, oriental: vanilla, ozonoc, powderry, powdery sweet, pzonic, rose sweet, rum, sweet floral, sweet fresh, sweet vanilla ...
- query-only: anis, aquaticp, mineral, spicy, —

## 1.3 source_url

- baris global = 266; source_url non-null = 266; menunjuk fragrantica.com = 264; URL unik = 256 (96.2% dari non-null).

## 1.4 Ukuran daftar accord

- |A(produk)|: mean 4.79, median 5.0
- |A(query)|: mean 8.69, median 9
- pasangan berlabel dengan |A(q)| > |A(p)|: 198 / 210 (94.3%)

## 1.5 Kualitas token

- accord mencurigakan: **8** (lihat token_quality.csv)
  - `baae: warm spicy` [product] -> prefix-colon;nonalpha
  - `bae: powdery musky` [product] -> prefix-colon;nonalpha
  - `fresh spicy. citrus` [product] -> nonalpha
  - `greeen` [product] -> triple-letter
  - `haert: white floral warm spicy` [product] -> prefix-colon;very-long;nonalpha
  - `oriental: vanilla` [product] -> prefix-colon;nonalpha
  - `white floral and tuberose` [product] -> very-long
  - `—` [query] -> nonalpha

## KESIMPULAN

# DATA BERSIH

Alasan berbasis angka:
- containment acak 0.350 vs benar 0.860 (selisih +0.509); proporsi fully-contained acak 1.8% vs benar 69.5%.
- produk punya 55 accord yang tak pernah muncul di query -> kosakata independen.
- source_url: 264/266 menunjuk Fragrantica, 256 URL unik.
- query sistematis lebih panjang (8.7 vs 4.8 accord) -> konsisten dengan Fragrantica mendaftar lebih banyak, bukan penyalinan.
