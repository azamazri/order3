"""Method registry. Each method implements the common `Method.scores` interface."""
from .b1_jaccard import B1Jaccard
from .b2_tfidf import B2TfidfCosine
from .b3_bm25 import B3BM25
from .b4_sbert import B4SBert
from .b5_word2vec import B5Word2Vec
from .b6_node2vec import B6Node2Vec
from .a1_wheel import A1Wheel
from .a2_ppmi_svd import A2PpmiSvd
from .a3_signature import A3Signature
from .a4_bigram_salience import A4BigramSalience
from .a5_bilinear import A5Bilinear
from .a6_gbm import A6Gbm
from .p1_order2 import P1Order2
from .p2_fusion import P2Fusion
from .p3_hubidf import P3HubIdf

# Order = reporting order. B4/B5/B6 may be slow / require optional deps.
ALL_METHODS = [
    B1Jaccard(), B2TfidfCosine(), B3BM25(), B4SBert(), B5Word2Vec(), B6Node2Vec(),
    A1Wheel(), A2PpmiSvd(), A3Signature(), A4BigramSalience(), A5Bilinear(), A6Gbm(),
    P1Order2(), P2Fusion(), P3HubIdf(),
]

# Fast subset (no S-BERT download, no embedding training) -- handy for quick runs.
FAST_METHODS = [
    B1Jaccard(), B2TfidfCosine(), B3BM25(),
    A1Wheel(), A2PpmiSvd(), A3Signature(), A4BigramSalience(), A5Bilinear(), A6Gbm(),
    P1Order2(), P2Fusion(), P3HubIdf(),
]

__all__ = ["ALL_METHODS", "FAST_METHODS"]
