"""Edwards Fragrance Wheel as a weighted tree, and the closed-form tree-Wasserstein
(W1) distance between two accord distributions on that tree (method A1).

The lexicon is FROZEN: it is fixed a-priori from the published Edwards wheel and is
NEVER tuned on the test labels. Accords absent from the lexicon are dropped from the
distribution and the remaining mass is renormalised (a documented judgement call).

Tree structure:  root -> superfamily -> subfamily -> accord
Edge weights:    accord->subfamily = 1, subfamily->superfamily = 2, superfamily->root = 3

For a tree, W1 has the closed form
    W1(p, q) = sum_e  w_e * | mass_p(subtree below e) - mass_q(subtree below e) |
where the sum is over every edge e and mass_*(subtree) is the total probability mass
the distribution places on accords in the subtree hanging below e.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Frozen lexicon: superfamily -> subfamily -> [accords]
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

W_ACCORD = 1.0   # accord  -> subfamily
W_SUB = 2.0      # subfamily -> superfamily
W_SUPER = 3.0    # superfamily -> root


class WheelTree:
    """Precomputes, for every tree edge, the set of accord indices in its subtree
    and the edge weight, so that W1 is a couple of vector ops per pair."""

    def __init__(self):
        # accord -> (superfamily, subfamily)
        self.accord2path: Dict[str, tuple] = {}
        for sup, subs in WHEEL.items():
            for sub, accords in subs.items():
                for ac in accords:
                    self.accord2path[ac] = (sup, sub)
        self.accords: List[str] = sorted(self.accord2path)
        self.idx: Dict[str, int] = {a: i for i, a in enumerate(self.accords)}
        n = len(self.accords)

        # Build one (weight, membership-mask) per edge.
        # 1) accord-leaf edges (weight 1): subtree = {that accord}
        # 2) subfamily edges (weight 2): subtree = accords in that subfamily
        # 3) superfamily edges (weight 3): subtree = accords in that superfamily
        edges_w: List[float] = []
        edges_mask: List[np.ndarray] = []

        for a in self.accords:
            m = np.zeros(n)
            m[self.idx[a]] = 1.0
            edges_w.append(W_ACCORD)
            edges_mask.append(m)

        for sup, subs in WHEEL.items():
            for sub, accords in subs.items():
                m = np.zeros(n)
                for a in accords:
                    m[self.idx[a]] = 1.0
                edges_w.append(W_SUB)
                edges_mask.append(m)

        for sup, subs in WHEEL.items():
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

    def w1(self, p: np.ndarray, q: np.ndarray) -> float:
        """Closed-form tree-Wasserstein-1 between two distributions (vectors over
        self.accords)."""
        # mass of each distribution under every edge's subtree
        mp = self.edge_mask @ p
        mq = self.edge_mask @ q
        return float(np.sum(self.edge_w * np.abs(mp - mq)))


if __name__ == "__main__":
    t = WheelTree()
    print(f"lexicon accords: {len(t.accords)}, edges: {len(t.edge_w)}")
    a = t.distribution(["amber", "warm spicy", "woody"])
    b = t.distribution(["amber", "vanilla", "sweet"])
    c = t.distribution(["citrus", "marine"])
    print("W1(amber-woody, amber-sweet) =", round(t.w1(a, b), 4))
    print("W1(amber-woody, citrus-marine) =", round(t.w1(a, c), 4))
