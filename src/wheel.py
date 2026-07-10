"""Fragrance wheel as a weighted tree, and the closed-form tree-Wasserstein (W1)
distance between two accord distributions on that tree (method A1).

IMPORTANT ON PROVENANCE: only the superfamily/subfamily SKELETON is inspired by the
published Edwards fragrance wheel. Edwards' wheel has no accord-level nodes; the
accord->subfamily leaf mapping below is the author's own taxonomic assignment. So
"frozen a-priori" applies to the tree structure, not to the leaf placements.

Lexicon coverage (Tahap 2.3): three canonical accords are not in the original leaf
mapping -- {orange, mineral, anis}. They are mapped a-priori ONCE, before any method
runs, without looking at labels or scores (WHEEL_EXTRA). `anis` is a low-confidence
judgement call (aromatic/licorice) and its alternative placement is tested in the
A1 sensitivity. The old behaviour (drop unmapped accord + renormalise mass) is kept
as sensitivity dimension S12=drop.

Tree structure:  root -> superfamily -> subfamily -> accord
Edge weights:    accord->subfamily = 1, subfamily->superfamily = 2, superfamily->root = 3

For a tree, W1 has the closed form
    W1(p, q) = sum_e  w_e * | mass_p(subtree below e) - mass_q(subtree below e) |
where the sum is over every edge e and mass_*(subtree) is the total probability mass
the distribution places on accords in the subtree hanging below e.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Base lexicon: superfamily -> subfamily -> [accords]  (author mapping, frozen)
# --------------------------------------------------------------------------- #
WHEEL: Dict[str, Dict[str, List[str]]] = {
    "Fresh": {
        "Citrus": ["citrus", "sour"],
        "Green": ["green", "herbal", "conifer", "bitter"],
        "Aromatic": ["aromatic", "fresh", "fresh spicy", "lavender"],
        "Water": ["aquatic", "marine", "ozonic", "metallic", "salty"],
        "Fruity": ["fruity", "cherry", "tropical", "apple", "berry"],
    },
    "Floral": {
        "Floral": ["floral", "white floral", "rose", "tuberose", "yellow floral"],
        "SoftFloral": ["aldehydic", "iris", "powdery", "violet", "lactonic"],
    },
    "Amber": {
        "Amber": ["amber", "warm spicy", "cinnamon", "cannabis", "soft spicy",
                  "spicy", "incense", "resinous"],
        "SoftAmber": ["vanilla", "sweet", "balsamic", "honey", "caramel", "almond",
                      "coconut", "nutty", "cacao", "chocolate", "coffee", "musky",
                      "rum", "vodka", "whiskey"],
    },
    "Woody": {
        "Woods": ["woody", "oud", "cedar", "sandalwood"],
        "MossyWoods": ["patchouli", "mossy", "earthy"],
        "DryWoods": ["leather", "animalic", "smoky", "tobacco"],
    },
}

# A-priori coverage mapping for accords absent from the base leaf mapping.
# accord -> (superfamily, subfamily). Frozen once, pre-experiment, label-free.
WHEEL_EXTRA: Dict[str, Tuple[str, str]] = {
    "orange": ("Fresh", "Citrus"),       # high confidence
    "mineral": ("Fresh", "Water"),       # high confidence
    "anis": ("Fresh", "Aromatic"),       # LOW -- judgement call; alt tested in sensitivity
}
# Alternative placement of `anis` for the A1 sensitivity (aromatic vs amber/spicy).
ANIS_ALT: Tuple[str, str] = ("Amber", "Amber")

W_ACCORD = 1.0   # accord  -> subfamily
W_SUB = 2.0      # subfamily -> superfamily
W_SUPER = 3.0    # superfamily -> root


def build_lexicon(mode: str = "map",
                  anis_placement: Tuple[str, str] = WHEEL_EXTRA["anis"]) -> Dict:
    """Return a superfamily->subfamily->[accords] lexicon.

    mode="map"  : include WHEEL_EXTRA (orange/mineral/anis mapped a-priori). Nothing
                  in the canonical vocabulary is dropped.
    mode="drop" : base lexicon only; unmapped accords are dropped + mass renormalised
                  (the old behaviour), tested as sensitivity S12=drop.
    """
    lex = copy.deepcopy(WHEEL)
    if mode == "map":
        extra = dict(WHEEL_EXTRA)
        extra["anis"] = anis_placement
        for accord, (sup, sub) in extra.items():
            lex.setdefault(sup, {}).setdefault(sub, [])
            if accord not in lex[sup][sub]:
                lex[sup][sub].append(accord)
    elif mode != "drop":
        raise ValueError(f"unknown wheel mode: {mode!r}")
    return lex


class WheelTree:
    """Precomputes, for every tree edge, the set of accord indices in its subtree
    and the edge weight, so that W1 is a couple of vector ops per pair."""

    def __init__(self, mode: str = "map",
                 anis_placement: Tuple[str, str] = WHEEL_EXTRA["anis"]):
        self.mode = mode
        lex = build_lexicon(mode, anis_placement)

        # accord -> (superfamily, subfamily)
        self.accord2path: Dict[str, tuple] = {}
        for sup, subs in lex.items():
            for sub, accords in subs.items():
                for ac in accords:
                    self.accord2path[ac] = (sup, sub)
        self.accords: List[str] = sorted(self.accord2path)
        self.idx: Dict[str, int] = {a: i for i, a in enumerate(self.accords)}
        n = len(self.accords)

        edges_w: List[float] = []
        edges_mask: List[np.ndarray] = []

        # 1) accord-leaf edges (weight 1): subtree = {that accord}
        for a in self.accords:
            m = np.zeros(n)
            m[self.idx[a]] = 1.0
            edges_w.append(W_ACCORD)
            edges_mask.append(m)

        # 2) subfamily edges (weight 2): subtree = accords in that subfamily
        for sup, subs in lex.items():
            for sub, accords in subs.items():
                m = np.zeros(n)
                for a in accords:
                    m[self.idx[a]] = 1.0
                edges_w.append(W_SUB)
                edges_mask.append(m)

        # 3) superfamily edges (weight 3): subtree = accords in that superfamily
        for sup, subs in lex.items():
            m = np.zeros(n)
            for sub, accords in subs.items():
                for a in accords:
                    m[self.idx[a]] = 1.0
            edges_w.append(W_SUPER)
            edges_mask.append(m)

        self.edge_w = np.asarray(edges_w)               # (E,)
        self.edge_mask = np.asarray(edges_mask)          # (E, n)

    # ------------------------------------------------------------------ #
    def distribution(self, accords: Sequence[str]) -> np.ndarray:
        """Uniform distribution over the lexicon-mapped accords of a fragrance.
        Unmapped accords are dropped; remaining mass renormalised. Empty -> zeros."""
        v = np.zeros(len(self.accords))
        for a in accords:
            j = self.idx.get(a)
            if j is not None:
                v[j] += 1.0
        s = v.sum()
        if s > 0:
            v /= s
        return v

    def mass_kept(self, accords: Sequence[str]) -> float:
        """Fraction of a fragrance's accords that map into the lexicon (1.0 on map)."""
        if not accords:
            return 1.0
        kept = sum(1 for a in accords if a in self.idx)
        return kept / len(accords)

    def w1(self, p: np.ndarray, q: np.ndarray) -> float:
        """Closed-form tree-Wasserstein-1 between two distributions."""
        mp = self.edge_mask @ p
        mq = self.edge_mask @ q
        return float(np.sum(self.edge_w * np.abs(mp - mq)))


if __name__ == "__main__":
    t = WheelTree()
    print(f"lexicon accords: {len(t.accords)}, edges: {len(t.edge_w)}, mode={t.mode}")
    for tok in ("orange", "mineral", "anis"):
        print(f"  {tok} -> {t.accord2path.get(tok)}")
