"""B4 -- Sentence-BERT cosine over the accord list (accord-only, symmetric).

Both sides are encoded from the SAME input every other method sees: the comma-joined
accord list. Query text = ", ".join(A(q)); product text = ", ".join(A(p)). No free
text, no olfactory/global family on either side. This is the only form that answers
RQ3 on identical input: does a trained sentence encoder beat lexical weighting when
both receive the same accord string? (The prose/family variants are ablations B4a/B4c
in src/v3, not the main-table method.)

Requires `sentence-transformers`. If the model cannot be loaded (e.g. no network for
the first download) the method returns all-NaN, which the evaluator reports as
"skipped" -- it is never silently faked.
"""
from __future__ import annotations

import numpy as np

from ..data import Dataset
from .base import Features, Method

# Multilingual checkpoint (accords are English tokens; kept for parity with ablations).
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

        prod_text = [", ".join(p.accords) for p in ds.products]
        q_text = [", ".join(q.accords) for q in ds.queries]

        pe = model.encode(prod_text, normalize_embeddings=True, show_progress_bar=False)
        qe = model.encode(q_text, normalize_embeddings=True, show_progress_bar=False)
        return qe @ pe.T
