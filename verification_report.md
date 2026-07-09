# Verification Report — What the Code Actually Runs

**Scope:** read-only audit of the benchmark code in this repository (`fix/`, the
`interaction-free fragrance dupe-retrieval benchmark`). No `.tex`, `.bib`, or code was
modified. Every claim below cites the exact file and line. Where something could not be
found in the repo it is marked **TIDAK DITEMUKAN**.

**Method registry (ground truth):** `src/methods/__init__.py:19-23`
```python
ALL_METHODS = [
    B1Jaccard(), B2TfidfCosine(), B3BM25(), B4SBert(), B5Word2Vec(), B6Node2Vec(),
    A1Wheel(), A2PpmiSvd(), A3Signature(), A4BigramSalience(), A5Bilinear(), A6Gbm(),
    P1Order2(), P2Fusion(), P3HubIdf(),
]
```

---

## Baselines (Table II)

### B1 — Jaccard
- **(a)** Manual Jaccard over accord sets (no library).
- **(b)** Pure `numpy`.
- **(c)** `src/methods/b1_jaccard.py:15` → `src/methods/base.py:98-105`
  ```python
  # b1_jaccard.py:15
  return np.vstack([feats.jaccard(qi) for qi in range(feats.n_q)])
  # base.py:98-105
  def jaccard(self, qi: int) -> np.ndarray:
      inter = self.shared_counts(qi)
      sq = self.QU[qi].sum(); sp = self.PU.sum(axis=1)
      union = sq + sp - inter
  ```
- **(d)** none.

### B2 — TF-IDF cosine  *(primary baseline)*
- **(a)** **Manual** TF-IDF cosine — **NOT** `sklearn.TfidfVectorizer`.
- **(b)** Pure `numpy`. IDF is a hand-written smoothed idf.
- **(c)** `src/methods/b2_tfidf.py:15`; weights in `src/methods/base.py:37-41,156-157`
  ```python
  # b2_tfidf.py:15
  return feats.QUt @ feats.PUt.T
  # base.py:37-41  (idf definition)
  def _idf(binary):
      n_docs = binary.shape[0]; df = binary.sum(axis=0)
      return np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
  # base.py:156-157
  PUt = _l2norm_rows(PU * idf_u); QUt = _l2norm_rows(QU * idf_u)
  ```
- **(d)** idf = `log((1+N)/(1+df))+1` (sklearn-style smoothing, `smooth_idf`+`sublinear=False`
  equivalent); term presence is **binary** (accord sets are deduplicated), vectors L2-normalized,
  score = cosine. No `TfidfVectorizer` parameters exist because the vectorizer is not used.

### B3 — BM25
- **(a)** **BM25Okapi** (the Okapi variant), from `rank_bm25`.
- **(b)** `rank-bm25` (requirements: `rank-bm25>=0.2.2`).
- **(c)** `src/methods/b3_bm25.py:15,17-18,21`
  ```python
  from rank_bm25 import BM25Okapi
  corpus = [p.accords if p.accords else ["<empty>"] for p in ds.products]
  bm25 = BM25Okapi(corpus)
  out[q.idx] = bm25.get_scores(q.accords if q.accords else ["<empty>"])
  ```
- **(d)** **k1 and b are NOT set → library defaults** (`BM25Okapi`: `k1=1.5`, `b=0.75`,
  `epsilon=0.25`). Documents = per-product accord token lists; query = query accord tokens.

### B4 — Sentence-BERT
- **(a)** Checkpoint = **`paraphrase-multilingual-MiniLM-L12-v2`** (exact). The paper's phrase
  "multilingual Sentence-BERT" is **CORRECT**. It is **NOT** IndoSBERT / IndoBERT.
- **(b)** `sentence-transformers` (requirements: `sentence-transformers>=2.2`).
- **(c)** `src/methods/b4_sbert.py:20,33,41-46,48-50`
  ```python
  _MODEL = "paraphrase-multilingual-MiniLM-L12-v2"           # line 20
  model = SentenceTransformer(self.model_name)                # line 33
  prod_text = [(p.text_clean or p.family or " ") + " " + " ".join(p.accords) ...]  # 41-44
  q_text = [q.family + " " + " ".join(q.accords) for q in ds.queries]              # 46
  pe = model.encode(prod_text, normalize_embeddings=True, ...)                     # 48
  ```
- **(d)** **Text encoded (important):** PRODUCT = leakage-stripped `meaning`
  (**Indonesian prose**) + `olfactory_family` + accord words; QUERY = `global_family` +
  accord words (English). So product text **does** include Indonesian prose; query text is
  accords+family only (asymmetric). `normalize_embeddings=True`, score = dot product (cosine).

### B5 — Word2Vec
- **(a)** `gensim` `Word2Vec`, **trained from scratch** on this catalog — **not** pretrained.
- **(b)** `gensim` (requirements: `gensim>=4.3`).
- **(c)** `src/methods/b5_word2vec.py:18,22,24-28`
  ```python
  def __init__(self, dim=64, window=5, epochs=50, min_count=1):                    # 18
  from gensim.models import Word2Vec                                                # 22
  sentences = [p.accords ...] + [q.accords ...]                                     # 24-25
  model = Word2Vec(sentences, vector_size=64, window=5, min_count=1, sg=1,
                   epochs=50, workers=1, seed=seed)                                 # 26-28
  ```
- **(d)** `vector_size=64`, `window=5`, `epochs=50`, `min_count=1`, `sg=1` (**skip-gram**),
  `workers=1`, `seed`. "Sentences" = accord lists of products **and** queries; fragrance
  vector = mean of accord vectors; cosine.

### B6 — node2vec
- **(a)** `node2vec` package (`Node2Vec`) over a `networkx` accord co-occurrence graph.
- **(b)** `node2vec` (`>=0.4`) + `networkx` (`>=3.0`); Skip-gram fit via gensim internally.
- **(c)** `src/methods/b6_node2vec.py:22-23,28-29,41-44`
  ```python
  def __init__(self, dim=64, walk_length=20, num_walks=50, window=5):              # 22-23
  import networkx as nx; from node2vec import Node2Vec                              # 28-29
  n2v = Node2Vec(G, dimensions=64, walk_length=20, num_walks=50,
                 weight_key="weight", workers=1, seed=seed, quiet=True)             # 41-43
  model = n2v.fit(window=5, min_count=1, sg=1, workers=1, seed=seed)               # 44
  ```
- **(d)** `dimensions=64`, `walk_length=20`, `num_walks=50`, `window=5`, weighted walks
  (`weight_key="weight"`). **The return parameters `p` and `q` are NOT set → `node2vec`
  package defaults `p=1, q=1`** (i.e. unbiased walks, DeepWalk-equivalent). Graph edge weight
  = number of fragrances in which two accords co-occur.

---

## Advanced comparators (Table II)

### A1 — Edwards tree-Wasserstein
- **(a)** **Hand-written closed-form tree-Wasserstein-1 (W1)** on a frozen Edwards-wheel tree.
- **(b)** Pure `numpy`. **No optimal-transport library** (no `POT`/`ot`, no `pyemd`).
- **(c)** taxonomy + distance are in `src/wheel.py`; used by `src/methods/a1_wheel.py:20-31`.
  The **wheel taxonomy is hard-coded** (not an external file): `src/wheel.py:30-56` dict `WHEEL`.
  ```python
  # wheel.py:118  (closed-form W1, custom)
  """Closed-form tree-Wasserstein-1 between two distributions ..."""
  # a1_wheel.py:25-30
  Pm = Pd @ tree.edge_mask.T; Qm = Qd @ tree.edge_mask.T
  d = np.abs(Pm - Qm[qi]) @ tree.edge_w
  ```
- **(d)** tree edge weights accord→subfamily=1, subfamily→superfamily=2, superfamily→root=3
  (`wheel.py:58-60`, `W_ACCORD/W_SUB/W_SUPER`).
- **CRITICAL — citation mismatch:** there is **no** reference in code to **Kusner / Word Mover /
  WMD**, to **Sobolev transport**, or to **tree-sliced / OTSW** methods. The implementation is a
  self-contained closed-form tree-W1 on a fixed taxonomy. Any paper text citing Sobolev
  transport or a tree-sliced Wasserstein paper as the basis of A1 does **not** match the code
  (see Q2).

### A2 — PPMI-SVD
- **(a)** Manual PPMI + `sklearn` `TruncatedSVD`.
- **(b)** `scikit-learn` (`>=1.3`).
- **(c)** `src/methods/a2_ppmi_svd.py:21,25,37-47`
  ```python
  def __init__(self, dim=50):                                       # 21
  from sklearn.decomposition import TruncatedSVD                    # 25
  ppmi = np.maximum(pmi, 0.0)                                       # 44
  svd = TruncatedSVD(n_components=k, random_state=seed)             # 47  (k=min(50,V-1))
  ```
- **(d)** SVD rank = `min(50, V-1)`; PPMI from an accord–accord co-occurrence matrix; fragrance
  vector = mean accord embedding; cosine. `random_state=seed`.

### A3 — signature LTR
- **(a)** **Logistic regression** on signature-subgraph features — **NOT LambdaMART** or any
  learning-to-rank tree model.
- **(b)** `scikit-learn` `LogisticRegression` (+ `StandardScaler`).
- **(c)** `src/methods/a3_signature.py` uses `_logreg_fit_predict` from
  `src/methods/p2_fusion.py:22-27`
  ```python
  clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
  ```
- **(d)** `class_weight="balanced"`, `max_iter=1000`, default `lbfgs` solver; features standardized;
  trained out-of-fold (GroupKFold-5 by query). "LTR" here means logistic scoring, not a
  ranking-specific objective.

### A4 — bigram salience
- **(a)** **L2-regularized logistic regression** over high-dimensional shared-bigram indicators.
- **(b)** `scikit-learn` `LogisticRegression`.
- **(c)** `src/methods/a4_bigram_salience.py:23-27`
  ```python
  clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                           class_weight="balanced", solver="liblinear", random_state=seed)
  ```
- **(d)** `penalty="l2"`, `C=1.0`, `solver="liblinear"`, `class_weight="balanced"`, `max_iter=2000`.

### A5 — bilinear metric
- **(a)** **Custom low-rank bilinear metric**, learned by hand-written gradient descent —
  **no ML framework** (no torch/sklearn estimator for the model itself).
- **(b)** Pure `numpy`.
- **(c)** `src/methods/a5_bilinear.py:56,60`
  ```python
  def __init__(self, rank=8, lr=0.5, l2=1e-3, iters=300):     # 56
  Q, P = feats.QUt, feats.PUt   # l2-normalised unigram tf-idf # 60
  ```
- **(d)** **rank = 8**, `lr=0.5`, `l2=1e-3`, `iters=300`; score = `q^T (diag(d)+L L^T) p`;
  trained out-of-fold.

### A6 — GBM fusion
- **(a)** `sklearn` **GradientBoostingClassifier** — **NOT XGBoost, NOT LightGBM**.
- **(b)** `scikit-learn`.
- **(c)** `src/methods/a6_gbm.py:29-32`
  ```python
  from sklearn.ensemble import GradientBoostingClassifier
  clf = GradientBoostingClassifier(random_state=seed, n_estimators=200,
                                   max_depth=3, learning_rate=0.05, subsample=0.8)
  ```
- **(d)** `n_estimators=200`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`.

---

## Specific questions

### Q1 — LightGCN / KGIN ever implemented, tested, or referenced?
**Zero occurrences in code** (`src/`). Grep of the whole repo finds them only in prose docs and
the paper, never in an executable path:
- `BLUEPRINT_V5.md:59` — "Bukan mengalahkan model berbasis interaksi (LightGCN/KGAT/KGIN — ...)".
- `HANDOFF.md:70` — "KGAT/LightGCN/KGIN TIDAK dibandingkan — butuh user-item interaction ...".
- `IMPLEMENTATION_REPORT.md:34` — "*Recommender* berbasis interaksi (KGAT, LightGCN, KGIN, ...)".
- `README.md:18` — "recommenders (KGAT, LightGCN, KGIN, ItemKNN, ...)".
- `conference_101719.tex` (paper, other repo): lines 69, 97, 133, 207, 534.

**Conclusion:** LightGCN and KGIN were **never implemented, trained, or run**. This is consistent
with the paper's explicit exclusion argument.

### Q2 — Kusner / Word Mover / WMD / wasserstein / sobolev / tree_sliced / OTSW
- `Kusner`, `Word Mover`, `WMD`, `OTSW`, `sobolev`, `tree_sliced`: **zero occurrences anywhere in
  `src/`**. In the repo they appear only in `reference/reference.txt` (a BibTeX dump) and
  `BLUEPRINT_V5.md` (the reading list, e.g. `BLUEPRINT_V5.md:347-350`).
- `wasserstein` in code appears only for the **custom** tree-W1: `src/wheel.py:1,118` and
  `src/methods/a1_wheel.py:1,25` ("vectorised closed-form tree-W1").
- **No external optimal-transport library is imported** (`import ot`, `pyemd`, `POT` all return
  **zero** matches in `src/`).

**Conclusion:** A1 is a hand-rolled closed-form tree-Wasserstein-1; it does **not** use, follow,
or reference Kusner-WMD, Sobolev transport, or tree-sliced/ordered-tree-sliced transport in code.

### Q3 — IndoSBERT / indo / multilingual
- `IndoSBERT`, `indobert`: **zero occurrences** in `src/`.
- `multilingual`: only in `src/methods/b4_sbert.py:19` (comment) and the model name on line 20
  (`paraphrase-multilingual-MiniLM-L12-v2`).

**Conclusion:** B4 uses the multilingual MiniLM checkpoint, not any Indonesian-specific model.

### Q4 — Exact pinned versions
There is **no lockfile / environment.yml / Pipfile / pip-freeze** in the repo (**TIDAK
DITEMUKAN**). The only version record is `requirements.txt`, which pins **minimums, not exact
versions**, verbatim:
```
scikit-learn>=1.3
rank-bm25>=0.2.2            # B3
gensim>=4.3                 # B5 (Word2Vec), node2vec walks
node2vec>=0.4              # B6
networkx>=3.0              # B6 graph
sentence-transformers>=2.2
```
`xgboost` / `lightgbm` / `POT` are **not listed and not imported** (they are not used).
Exact resolved versions are **TIDAK DITEMUKAN** in the repo.

### Q5 — Number of comparators
- **12 comparators confirmed** in Table II: B1–B6 and A1–A6.
- The code registry (`src/methods/__init__.py:19-23`) runs **15** methods: the 12 comparators
  plus **P1** (method of record), **P2** (LTR fusion), and **P3** (hub-discriminative IDF).
- **Methods run but not in Table II:** **P2** (reported in the Conclusion as one sentence) and
  **P3** (present in `results/results.csv` but **not shown in the paper at all**). P3 is therefore
  an extra run not surfaced in the paper — minor, but noted.
- Additional **ablation** experiments exist (`src/order_ablation.py`, `src/order3_analysis.py`,
  `src/order_significance.py`) producing the order-N curve and order-3 significance; these are
  ablations of the proposed method, not additional comparators.

---

## Critical findings (summary)
1. **A1 citation mismatch.** The code implements a **custom closed-form tree-W1** on a hard-coded
   Edwards wheel; it imports **no OT library** and references **no** Kusner-WMD / Sobolev /
   tree-sliced method. Citing Sobolev transport or a tree-sliced Wasserstein paper as the basis
   of A1 is not supported by the code.
2. **A6 is sklearn GradientBoosting**, not XGBoost/LightGBM.
3. **A3 "LTR" is plain logistic regression**, not LambdaMART or a ranking-objective model.
4. **B2 is a manual TF-IDF** (numpy), not `sklearn.TfidfVectorizer` — mathematically a smoothed
   idf with binary term presence; no vectorizer parameters exist.
5. **B3 BM25 uses library defaults** (`k1=1.5, b=0.75`); these are not stated anywhere in the repo.
6. **B6 node2vec uses `p=q=1`** (defaults) — i.e. unbiased/DeepWalk-style walks, not tuned.
7. **No exact dependency versions are pinned** (requirements are `>=` only; no lockfile).
8. **P3 is run but never reported** in the paper; **P2** appears only as a one-line mention.
9. **LightGCN / KGIN / KGAT are never in code** — the exclusion is genuine.
