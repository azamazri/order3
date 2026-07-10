"""Data loading, parsing and join for the dupe-retrieval benchmark.

Task = cross-reference dupe retrieval. A GLOBAL perfume is the QUERY; the candidate
pool is the 340 AROMATIQUE local products. Ground truth = the inspired-by edge
encoded by `local.revolutionize == global.perfume_name`.

The dataset is cleaned in the Excel files (source of truth). No typo-fix, fuzzy
match or accord normalisation happens here (HANDOFF_V3 §0.9). Both sides are
represented by ACCORDS ONLY. Free product text lives in `product_text.csv` and is
read only by the B4a ablation module, never by `load_dataset()`.

A query is kept only if it (a) has >=1 labeled local dupe AND (b) has >=1 accord.
A global row with zero accords cannot be answered by any method and is not a query
(declared rule; see results/v3/00_dataset_verification.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths (repo root is the parent of this file's parent: <root>/src/data.py)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
AROMATIQUE_XLSX = ROOT / "dataset-aromatique.xlsx"
GLOBAL_XLSX = ROOT / "global_reference.xlsx"

# --------------------------------------------------------------------------- #
# Parsing helpers (frozen; documented judgement calls)
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
_SPLIT = re.compile(r"[;,]")


def norm_name(s) -> Optional[str]:
    """Normalise a perfume name for the local<->global join: lowercase, collapse
    whitespace, strip. Returns None for NaN/empty."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = _WS.sub(" ", str(s).strip().lower())
    return s or None


def parse_accords(s) -> List[str]:
    """Split an accord list on ',' or ';', lowercase + strip each token.
    Duplicates removed but order preserved."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    out, seen = [], set()
    for tok in _SPLIT.split(str(s)):
        t = _WS.sub(" ", tok.strip().lower())
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Product:
    idx: int                      # position in the candidate pool (0..n_pool-1)
    name: str
    accords: List[str]
    family: str                   # olfactory_family (local taxonomy; not fed to methods)
    rev_norm: Optional[str]       # normalised inspired-by global name (None => distractor)
    is_labeled: bool


@dataclass
class Query:
    idx: int                      # position in the query list
    name: str                     # global perfume_name (kept only for reporting)
    name_norm: str
    accords: List[str]            # from global accord_1..accord_10
    family: str                   # global_family
    relevant: Set[int] = field(default_factory=set)   # product idxs that are dupes


@dataclass
class Dataset:
    products: List[Product]
    queries: List[Query]
    local_vocab: Set[str]
    global_vocab: Set[str]

    @property
    def n_pool(self) -> int:
        return len(self.products)

    @property
    def shared_accords(self) -> Set[str]:
        return self.local_vocab & self.global_vocab


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dataset(aromatique_xlsx: Path = AROMATIQUE_XLSX,
                 global_xlsx: Path = GLOBAL_XLSX) -> Dataset:
    """Load both Excel files, parse accords, and build products + queries.

    Query accords come from the global `accord_1..accord_10` columns (clean,
    pre-split). A query is kept only if it has >=1 labeled local dupe AND >=1 accord.
    """
    # ---- local products (candidate pool = all rows) ----
    a = pd.read_excel(aromatique_xlsx)
    products: List[Product] = []
    local_vocab: Set[str] = set()
    for i, row in a.reset_index(drop=True).iterrows():
        accords = parse_accords(row.get("main_accords"))
        local_vocab.update(accords)
        products.append(Product(
            idx=int(i),
            name=str(row.get("product_name")),
            accords=accords,
            family=("" if pd.isna(row.get("olfactory_family"))
                    else str(row.get("olfactory_family")).strip()),
            rev_norm=norm_name(row.get("revolutionize")),
            is_labeled=pd.notna(row.get("revolutionize")),
        ))

    # ---- global reference (header on row 1; name column = perfume_name) ----
    g = pd.read_excel(global_xlsx).dropna(how="all")
    accord_cols = [c for c in g.columns if str(c).startswith("accord_")]
    global_vocab: Set[str] = set()

    # map normalised global name -> (accords, family); keep first occurrence
    gmap: Dict[str, Dict[str, object]] = {}
    for _, row in g.iterrows():
        nm = norm_name(row.get("perfume_name"))
        if nm is None:
            continue
        accords, seen = [], set()
        for c in accord_cols:
            v = row.get(c)
            if pd.notna(v):
                t = _WS.sub(" ", str(v).strip().lower())
                if t and t not in seen:
                    seen.add(t)
                    accords.append(t)
        global_vocab.update(accords)
        if nm not in gmap:
            gmap[nm] = {
                "name": str(row.get("perfume_name")).strip(),
                "accords": accords,
                "family": ("" if pd.isna(row.get("global_family"))
                           else str(row.get("global_family")).strip()),
            }

    # ---- relevant sets: for each global name, which products are its dupes ----
    rel_by_name: Dict[str, Set[int]] = {}
    for p in products:
        if p.rev_norm is not None:
            rel_by_name.setdefault(p.rev_norm, set()).add(p.idx)

    # ---- build queries: global names that (a) exist in gmap, (b) have a dupe, ----
    #      and (c) have >=1 accord (declared valid-query rule) ----
    queries: List[Query] = []
    for nm, rel in sorted(rel_by_name.items()):
        if nm not in gmap or not rel:
            continue
        info = gmap[nm]
        if not info["accords"]:                       # empty-accord global row: not a query
            continue
        queries.append(Query(
            idx=len(queries),
            name=str(info["name"]),
            name_norm=nm,
            accords=list(info["accords"]),
            family=str(info["family"]),
            relevant=set(rel),
        ))

    return Dataset(products=products, queries=queries,
                   local_vocab=local_vocab, global_vocab=global_vocab)


if __name__ == "__main__":
    ds = load_dataset()
    print(f"pool (products)      : {ds.n_pool}")
    print(f"labeled products     : {sum(p.is_labeled for p in ds.products)}")
    print(f"queries (>=1 dupe)   : {len(ds.queries)}")
    print(f"local accord vocab   : {len(ds.local_vocab)}")
    print(f"global accord vocab  : {len(ds.global_vocab)}")
    print(f"shared accords       : {len(ds.shared_accords)}")
