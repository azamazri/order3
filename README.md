# Accord Co-occurrence untuk Dupe Retrieval (v3)

Apakah memodelkan **accord yang muncul bersamaan** (co-occurrence) meningkatkan kualitas
retrieval parfum "dupe" dibanding memperlakukan accord sebagai independen? Tugas:
untuk tiap parfum global (query), rangking 340 produk katalog Aromatique; ground truth =
`revolutionize`.

Metode usulan: **order-N co-occurrence TF-IDF**, `N*=2` (dipilih via nested CV).

## Menjalankan (branch `exp/v3-clean`)

```bash
pip install -r requirements.txt
python -m src.v3.verify_dataset     # Tahap 1  -> 00_*
python -m src.v3.audit_impl         # Tahap 3  -> 01_plu/determinism
python -m src.v3.circularity        # Tahap 5  -> 02_*
python -m src.v3.order_ladder       # Tahap 6  -> 03_*
python -m src.v3.main_table         # Tahap 8  -> 04_/05_
python -m src.v3.stratified         # Tahap 5.5 -> 06_stratified
python -m src.v3.sensitivity        # Tahap 9  -> 07_*
python -m src.v3.decomposition      # Tahap 10 -> 08_
python -m src.v3.ablations          # Tahap 11 -> 09_*
```

Semua hasil sah ada di `results/v3/` dan ter-commit. Sintesis: **`results/v3/REPORT_V3.md`**.

## Aturan (lihat `HANDOFF_V3.md`, `PEDOMAN_EKSPERIMEN_V2.md`)
- Data Excel adalah sumber tunggal; **tidak ada** cleaning/typo-fix/normalisasi accord di kode.
- Angka sah hanya yang ada di `results/v3/*`. `archive/pre_v3/` = pra-bersih, **tidak sah**.
- `conference_101719.tex` tidak disentuh sampai eksperimen final.

## Layout
```
dataset-aromatique.xlsx  global_reference.xlsx  cleaning_changelog.csv
product_text.csv                     # khusus ablation B4a
src/{data,evaluate,wheel}.py  src/methods/  src/v3/   # harness v3
results/v3/                           # deliverable (00..10, REPORT_V3, environment_lock)
archive/pre_v3/                       # artefak pra-bersih (jangan dikutip)
```

## Temuan ringkas (RQ)
- **RQ1** order-2 > order-1: didukung (signifikan di NON_OP, ALL, 100% sel sensitivitas).
- **RQ2** titik jenuh: **N\*=2** (tak ada perbaikan signifikan di order>2).
- **RQ3** order-2 (tanpa parameter terlatih) **setara** metode co-occurrence terbaik
  (A2/A3/P2), **unggul** atas leksikal/neural-pretrained/learned-co-occurrence/taksonomik/
  supervised-per-pair. Detail + p_adj/CI: `REPORT_V3.md`.
