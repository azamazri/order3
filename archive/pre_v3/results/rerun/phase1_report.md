# Fase 1 -- Diagnostic Report (fair-baselines)

Protocol unchanged: leave-one-out, 209 queries, pool 340, expected-rank tie-breaking, GroupKFold-5 by query, 5 seeds for stochastic methods.

## 1.1 Vocabulary / token stats

- |V| (unique accords over 340 products + 209 queries) = **115**
- accords/product: mean 4.79, median 5.0, min 1, max 9
- accords/query: mean 8.69, median 9, min 1, max 10
- total tokens available to train Word2Vec (sum of accords over products+queries) = **3447**

## 1.2 Train vs test MRR

See `train_test_gap.csv`. Note: A2/B5/B6 are *unsupervised* (embeddings fit on the corpus, no labels), so there is no in-fold/out-of-fold label split -- train == test by construction; a low value there is a representation issue, not label overfitting. B1/B2/B3/A1/order3 are parameter-free (train == test).

| method | mrr_train | mrr_test | gap |
|---|---|---|---|
| B1_jaccard | 0.4277 | 0.4277 | +0.0000 |
| B2_tfidf_cos | 0.4540 | 0.4540 | +0.0000 |
| B3_bm25 | 0.3879 | 0.3879 | +0.0000 |
| A1_wheel_treeW | 0.2836 | 0.2836 | +0.0000 |
| order3 | 0.5080 | 0.5080 | +0.0000 |
| A3_signature | 0.4359 | 0.4331 | +0.0028 |
| A4_bigram_salience | 0.5321 | 0.1393 | +0.3929 |
| A6_gbm_fusion | 0.6785 | 0.4513 | +0.2273 |
| A5_bilinear | 0.1844 | 0.1498 | +0.0347 |
| A2_ppmi_svd | 0.4435 | 0.4435 | +0.0000 |
| B5_word2vec | 0.1848 | 0.1848 | +0.0000 |
| B6_node2vec | 0.3396 | 0.3396 | +0.0000 |

**Interpretation key (do not copy into paper):** large gap + low test = overfitting (paper claim holds); small gap + low test = UNDERFITTING / misconfiguration (paper claim would be wrong).

## 1.3 A5 (bilinear) convergence, lr=0.5, 300 iters, seed 0, full training set

- loss[0] = 0.6931, loss[end] = 0.6378, loss[min] = 0.6378
- monotonically non-increasing: **True**; diverged (end > 1.05*start): **False**; oscillates (>5 up-steps): **False**
- full curve in `a5_loss_curve.csv`.

## 1.4 Dimension sanity (dim vs |V|)

- B5_word2vec dim=64: dim > |V| ? **False**
- B6_node2vec dim=64: dim > |V| ? **False**
- A2_ppmi_svd rank=min(50,V-1)=50: dim > |V| ? **False**

## 1.5 Data seen at fit-time (information access; not label leakage)

- **P1/order-N IDF**: computed over the **340 products only** (`base.py:153` `idf_u=_idf(PU)`, PU = products).
- **B5 Word2Vec**: trained on **products + queries** accord lists (`b5_word2vec.py:24-25`).
- **B6 node2vec**: co-occurrence graph built from **products + queries** (`b6_node2vec.py:35`).
- **A2 PPMI**: co-occurrence matrix from **products + queries** (`a2_ppmi_svd.py:29`).
- Query accords are external (Fragrantica); no label is ever used by the unsupervised fits. This is information *asymmetry to document*, not label leakage.
