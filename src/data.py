"""Data loading, parsing, join and leakage audit for the dupe-retrieval benchmark.

Task = cross-reference dupe retrieval. A GLOBAL perfume is the QUERY; the candidate
pool is the 340 AROMATIQUE local products. Ground truth = the held-out `inspired_by`
edge encoded by `local.revolutionize == global.Revolutionize`.

LEAKAGE FIREWALL (see README):
  * A query is represented with ONLY accords + global_family. We NEVER expose
    `interpreted_as` (it is the global name) or any identity-bearing field.
  * Product free text (meaning / visual_note) is leakage-audited: any distinctive
    token of a product's own inspired-by global name is reported and stripped before
    it is handed to a text-based method (B4).
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
    family: str                   # olfactory_family (local taxonomy)
    meaning: str                  # free text (Indonesian) -- leakage stripped copy below
    visual: str                   # visual_note + visual_note_alt joined
    rev_norm: Optional[str]       # normalised inspired-by global name (None => distractor)
    is_labeled: bool
    text_clean: str = ""          # meaning+family with own-name leakage stripped (for B4)


@dataclass
class Query:
    idx: int                      # position in the query list
    name: str                     # global Revolutionize name (kept only for reporting)
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
# Leakage audit
# --------------------------------------------------------------------------- #
# Common words that are NOT distinctive of a brand/perfume identity. Used to decide
# which tokens of a global name could leak identity into product free text.
_STOP = {
    "by", "de", "la", "le", "the", "for", "and", "of", "du", "des", "eau",
    "parfum", "perfume", "extreme", "intense", "man", "men", "woman", "women",
    "pour", "homme", "femme", "edp", "edt",
}


def _distinctive_tokens(global_name: str) -> Set[str]:
    """Tokens of a global perfume name that would betray its identity if they
    appeared verbatim in a product's free text."""
    toks = re.findall(r"[a-z0-9']+", global_name.lower())
    return {t for t in toks if len(t) >= 4 and t not in _STOP}


def leakage_audit(dataset: Dataset, verbose: bool = True) -> Dict[str, object]:
    """Count products whose meaning/visual free text contains a distinctive token
    of their own inspired-by global name. Returns a report dict. Side effect: fills
    `Product.text_clean` with the leakage-stripped meaning+family text for B4."""
    norm2name = {q.name_norm: q.name for q in dataset.queries}
    hits = []
    for p in dataset.products:
        raw_text = f"{p.meaning} {p.visual}".lower()
        clean_meaning = f"{p.meaning} {p.family}"
        if p.rev_norm and p.rev_norm in norm2name:
            distinctive = _distinctive_tokens(norm2name[p.rev_norm])
            present = {t for t in distinctive if re.search(rf"\b{re.escape(t)}\b", raw_text)}
            if present:
                hits.append((p.idx, p.name, norm2name[p.rev_norm], sorted(present)))
                # strip leaked tokens from the text handed to text-based methods
                for t in present:
                    clean_meaning = re.sub(rf"\b{re.escape(t)}\b", " ", clean_meaning,
                                           flags=re.IGNORECASE)
        p.text_clean = _WS.sub(" ", clean_meaning).strip()

    report = {
        "n_products": len(dataset.products),
        "n_leaky_products": len(hits),
        "leaky": hits,
    }
    if verbose:
        print(f"[leakage audit] {len(hits)}/{len(dataset.products)} products contain a "
              f"distinctive token of their own global name in meaning/visual text.")
        for idx, name, gname, toks in hits[:20]:
            print(f"    - #{idx} {name!r} (dupe of {gname!r}) leaks {toks}")
        if len(hits) > 20:
            print(f"    ... and {len(hits) - 20} more")
        print("[leakage audit] such tokens are stripped from text handed to B4 (S-BERT).")
    return report


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dataset(aromatique_xlsx: Path = AROMATIQUE_XLSX,
                 global_xlsx: Path = GLOBAL_XLSX) -> Dataset:
    """Load both Excel files, parse accords, and build products + queries.

    Query accords come from the global `accord_1..accord_10` columns (clean,
    pre-split). A query is kept only if it has >=1 labeled local dupe in the pool.
    """
    # ---- local products (candidate pool = all rows) ----
    a = pd.read_excel(aromatique_xlsx)
    products: List[Product] = []
    local_vocab: Set[str] = set()
    for i, row in a.reset_index(drop=True).iterrows():
        accords = parse_accords(row.get("main_accords"))
        local_vocab.update(accords)
        visual = " ".join(
            str(row.get(c)) for c in ("visual_note", "visual_note_alt")
            if pd.notna(row.get(c))
        )
        products.append(Product(
            idx=int(i),
            name=str(row.get("product_name")),
            accords=accords,
            family=("" if pd.isna(row.get("olfactory_family")) else str(row.get("olfactory_family")).strip()),
            meaning=("" if pd.isna(row.get("meaning")) else str(row.get("meaning")).strip()),
            visual=visual,
            rev_norm=norm_name(row.get("revolutionize")),
            is_labeled=pd.notna(row.get("revolutionize")),
        ))

    # ---- global reference (header lives in file row index 1) ----
    g = pd.read_excel(global_xlsx, header=1).dropna(how="all")
    accord_cols = [c for c in g.columns if str(c).startswith("accord_")]
    global_vocab: Set[str] = set()

    # map normalised global name -> (accords, family); keep first occurrence
    gmap: Dict[str, Dict[str, object]] = {}
    for _, row in g.iterrows():
        nm = norm_name(row.get("Revolutionize"))
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
                "name": str(row.get("Revolutionize")).strip(),
                "accords": accords,
                "family": ("" if pd.isna(row.get("global_family")) else str(row.get("global_family")).strip()),
            }

    # ---- relevant sets: for each global name, which products are its dupes ----
    rel_by_name: Dict[str, Set[int]] = {}
    for p in products:
        if p.rev_norm is not None:
            rel_by_name.setdefault(p.rev_norm, set()).add(p.idx)

    # ---- build queries: global names that (a) exist in gmap and (b) have a dupe ----
    queries: List[Query] = []
    for nm, rel in sorted(rel_by_name.items()):
        if nm not in gmap or not rel:
            continue
        info = gmap[nm]
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
    leakage_audit(ds)
