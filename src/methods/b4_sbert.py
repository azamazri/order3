"""B4 -- Sentence-BERT cosine over free text.

Product text  = leakage-stripped [meaning + olfactory_family]  (Indonesian).
Query text    = [accords + global_family]                       (English).
The leakage firewall (data.leakage_audit) has already removed any distinctive token
of a product's own global name from `Product.text_clean`.

Requires `sentence-transformers`. If the model cannot be loaded (e.g. no network for
the first download) the method degrades gracefully and returns all-NaN, which the
evaluator reports as "skipped" -- it is never silently faked.
"""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Features, Method

# Multilingual model: product text is Indonesian, query text is English accords.
_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class B4SBert(Method):
    name = "B4_sbert"
    tier = "T1 baseline"

    def __init__(self, model_name: str = _MODEL):
        self.model_name = model_name

    def scores(self, ds: Dataset, feats: Features, seed: int = 0) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name)
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[B4] skipped: could not load '{self.model_name}' ({e}).")
            return np.full((len(ds.queries), ds.n_pool), np.nan)

        # PRODUCT text = family + leakage-stripped meaning + accords-as-words.
        # p.text_clean is the leakage-audited "meaning + family" (own global-name tokens
        # already removed); we append the accord words so query/product share a space.
        prod_text = [
            (p.text_clean or p.family or " ") + " " + " ".join(p.accords)
            for p in ds.products
        ]
        # QUERY text = global_family + accords-as-words. NEVER interpreted_as (= leakage).
        q_text = [q.family + " " + " ".join(q.accords) for q in ds.queries]

        pe = model.encode(prod_text, normalize_embeddings=True, show_progress_bar=False)
        qe = model.encode(q_text, normalize_embeddings=True, show_progress_bar=False)
        return qe @ pe.T
