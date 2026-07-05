# Interaction-free fragrance dupe-retrieval benchmark (Path B)

A reproducible Python pipeline for an S2 thesis. The task is **cross-reference dupe
retrieval**: given a *global* perfume (the query), retrieve the *AROMATIQUE* local
product that is its **dupe**. Ground truth is the held-out `inspired_by` edge encoded
by `local.revolutionize == global.Revolutionize`.

The benchmark compares three tiers of methods and reports a **non-circular finding**:
*order-2 accord co-occurrence beats order-1 marginal similarity, and beats
perceptual/learned structure*, which overfit on these sparse labels.

---

## Why interaction-based recommenders are OUT OF SCOPE

There are **no user interactions or ratings** in this dataset — only a product
catalogue with accord lists and a global reference table. Interaction-driven
recommenders (KGAT, LightGCN, KGIN, ItemKNN, and collaborative filtering in general)
require a user–item interaction signal that simply does not exist here. Including them
would mean fabricating interactions, so they are **excluded by design**. The task is a
pure **content-based cross-reference retrieval** problem, and every method below scores
a (query, product) pair from content alone.

---

## Leakage firewall (critical)

The dupe relationship must never be readable from the inputs:

1. **Query representation uses ONLY accords + `global_family`.** The query NEVER
   exposes `interpreted_as` (it is literally the global perfume name) or any other
   identity-bearing field.
2. **Free-text leakage audit.** `data.leakage_audit` counts every product whose
   `meaning`/`visual_note` text contains a distinctive token of its *own* inspired-by
   global name, reports them, and **strips** those tokens from the text handed to the
   text-based method (B4). (Most flagged tokens are generic accord words such as
   *rose*/*floral*; they are stripped anyway, conservatively.)
3. **No taxonomy or metric touches the label.** The Edwards-wheel lexicon (A1) is
   frozen a-priori and never tuned on the test set. IDF is computed on the candidate
   corpus only.

### The accord-containment artifact (diagnosed, not hidden)

The global accord lists were evidently *derived from the dupes*: on average **~86% of a
dupe's accords are contained in its global's accord list**. This makes **raw
(un-weighted) set overlap a near-circular signal** — and it has two consequences the
pipeline handles explicitly:

* The fair *marginal* baseline is **IDF-weighted** cosine (B2), as used in the cited
  content-based perfume work (Nurmuthia / BINUS-style). Raw Jaccard is reported (B1) but
  understood to be inflated by this artifact.
* **Tie handling matters.** Un-weighted overlap leaves *many* candidates tied at the top
  score (~16 on average for Jaccard, vs ~5 for IDF methods). A naive "1 + #strictly
  greater" rank hands those methods a free optimistic boost. We instead use the **exact
  expectation under uniform random tie-breaking** (see Evaluation), which removes the
  artifact. Under correct tie handling, the proposed order-2 methods rank first.

---

## Methods

Common interface (`src/methods/base.py`): every method is a `Method` exposing
`scores(ds, feats, seed) -> (n_queries, n_pool)`; higher = more likely the dupe.
Supervised (learning-to-rank) methods produce **out-of-fold** scores via
`GroupKFold(5)` grouped by query, so a query is never in both train and test.

### Tier 1 — established baselines
| id | method |
|----|--------|
| B1 | Jaccard over accord sets |
| B2 | Cosine over accord TF-IDF (order-1 marginal) — **main baseline** |
| B3 | BM25 over accord tokens |
| B4 | Sentence-BERT cosine over [meaning + family] text (multilingual; optional dep) |
| B5 | Word2Vec accord embeddings, mean-pooled, cosine |
| B6 | node2vec on the accord co-occurrence graph, mean-pooled, cosine |

### Tier 2 — structure / learned (expected to LOSE — this is the finding)
| id | method |
|----|--------|
| A1 | Edwards-wheel tree-Wasserstein (closed-form W1 on a frozen tree) |
| A2 | PPMI-SVD accord co-occurrence embedding, mean-pooled, cosine |
| A3 | Signature-subgraph features (shared rare edges, asymmetric coverage) + logistic LTR |
| A4 | Supervised per-bigram salience (high-dim L2 logistic over shared-bigram indicators) |
| A5 | Low-rank bilinear cross-accord affinity `qᵀ(diag(d)+LLᵀ)p`, metric-learned |
| A6 | GradientBoosting fusion (overfit demo) |

### Tier 3 — proposed
| id | method |
|----|--------|
| P1 | **Order-2 co-occurrence TF-IDF** (unigram ∪ bigram, IDF-weighted, L2-norm, cosine); score decomposes into order-1 + order-2 |
| P2 | **P1 + logistic LTR fusion** of [bigram-cos, unigram-cos, |shared accords|] — expected best |
| P3 | P1 with IDF hub-discriminative weighting (ablation) |

---

## Evaluation

* **Protocol:** leave-one-out dupe retrieval. Query = global perfume with ≥1 labeled
  local dupe. Candidate pool = all 340 products (incl. the 97 unlabeled distractors).
  Supervised methods score out-of-fold (GroupKFold-5 by query).
* **Metrics:** MRR, Hits@1, Hits@3 (one relevant concept per query; best/min rank if a
  query has several dupes).
* **Ties:** exact expectation under uniform random tie-breaking. With `g` products
  strictly above the best relevant one and `e` tied at its score,
  `E[RR] = mean(1/r for r in g+1..g+e)`, `E[Hit@k] = clip(k-g,0,e)/e`. Reduces to the
  plain rank when `e == 1`.
* **Significance:** Wilcoxon signed-rank **paired across queries** on reciprocal rank
  (proposed vs each baseline) — *not* across seeds (deterministic methods have no seed
  variance).
* **Bootstrap:** 95% CI over queries (10 000 resamples) for ΔMRR.
* **Stochastic methods** (P2, A2, A5, B5, B6) are reported as mean ± std over 5 seeds.
  For the LTR methods the only randomness is the seed-shuffled fold assignment.

---

## How to run

```bash
pip install -r requirements.txt

python -m src.data        # sanity: pool=340, queries=209, shared accords=56 + leakage audit
python -m src.wheel       # sanity: wheel tree + a couple of W1 distances

python run_all.py            # full benchmark (B4 needs sentence-transformers)
python run_all.py --fast     # skip B4/B5/B6 (no downloads, no embedding training)
python run_all.py --seeds 5  # seeds for stochastic methods (default 5)
```

Outputs land in `results/`:
* `results.csv` — per-method MRR / Hits@1 / Hits@3 (mean ± std over seeds)
* `significance.csv` — Wilcoxon p + bootstrap ΔMRR CI for P1 and P2 vs every method

B4 (Sentence-BERT) requires `sentence-transformers`; if it cannot be installed/loaded
the method is reported as **SKIPPED** rather than silently faked.

---

## Repository layout

```
dataset-aromatique.xlsx     340 local products (the candidate pool)
global_reference.xlsx       global perfumes (header on row index 1)
reference/                  background papers
src/
  data.py                   load, parse, normalise+join, leakage audit
  wheel.py                  frozen Edwards-wheel lexicon + closed-form tree-W1
  evaluate.py               LOO metrics, tie-correct ranking, Wilcoxon, bootstrap, GroupKFold runner
  methods/                  one module per method (B1–B6, A1–A6, P1–P3) + base.py
run_all.py                  end-to-end runner -> results table + significance
requirements.txt
```

---

## Documented judgement calls

* **Accord parsing:** `re.split(r"[;,]")`, lowercase, strip; duplicates dropped,
  order preserved.
* **Join:** `local.revolutionize == global.Revolutionize`, normalised (lowercase,
  whitespace-collapsed).
* **Query accords:** taken from the global `accord_1..accord_10` columns.
* **Shared vocabulary:** all vectors live in the union of local+global accords; only the
  56 shared accords create cross overlap. IDF is fit on the 340-product corpus and the
  query is transformed with that same IDF.
* **Wheel lexicon:** frozen; accords absent from the lexicon are dropped and the
  remaining distribution mass renormalised.
* **Tie handling:** exact uniform-tie expectation (above) — chosen because un-weighted
  overlap is extremely tie-heavy and a naive rank would reward it spuriously.

> `kg_paths.json` generation is intentionally **deferred** to a separate later task.
