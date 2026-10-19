"""tinyllm.info.pmi (M11.4): mutual information, pointwise mutual
information, and positive PMI.

Mutual information I(X; Y) is how many nats knowing X saves you about Y: the
KL divergence from the joint distribution to the product of its marginals,
zero exactly when the two are independent. Pointwise mutual information is
the log ratio inside that sum for one pair, log P(w, c) / (P(w) P(c)):
positive when w and c occur together more often than chance. Clipping it at
zero (PPMI) gives the classic count-based word vectors that L2.3 factors
with an SVD, the baseline that word2vec is compared with.

Contract: contracts/py/tinyllm/info/pmi.pyi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.info.entropy import kl


def _table(x: ArrayLike, who: str) -> NDArray:
    """A validated float64 copy of a 2-D non-negative table with a positive sum."""
    t = np.array(x, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError(f"{who}: need a 2-D table, got shape {t.shape}")
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError(f"{who}: entries must be finite and non-negative")
    if not t.sum() > 0:
        raise ValueError(f"{who}: the table sums to 0")
    return t


def mutual_information(joint: ArrayLike) -> float:
    P = _table(joint, "mutual_information")  # our own copy
    P /= P.sum()
    px = P.sum(axis=1)
    py = P.sum(axis=0)
    # I(X; Y) = KL(P_XY || P_X P_Y): flatten both tables into one
    # distribution over pairs. Wherever P(x, y) > 0 the product is > 0 too.
    return float(kl(P.ravel(), np.outer(px, py).ravel()))


def pmi_matrix(cooc: ArrayLike, cds_alpha: float = 0.75) -> NDArray:
    C = _table(cooc, "pmi_matrix")
    if not 0.0 < cds_alpha <= 1.0:
        raise ValueError(f"pmi_matrix: need 0 < cds_alpha <= 1, got {cds_alpha}")
    D = C.sum()
    nw = C.sum(axis=1, keepdims=True)  # #(w), [W, 1]
    nc = C.sum(axis=0, keepdims=True)  # #(c), [1, C]
    # Smooth the CONTEXT distribution only, and renormalize over contexts:
    # P_alpha(c) = #(c)^alpha / sum_c' #(c')^alpha.
    nca = nc**cds_alpha
    pc = nca / nca.sum()
    with np.errstate(divide="ignore"):
        # Logs of the three probabilities, kept apart: log 0 = -inf only for
        # the joint, and the marginals of a seen pair are positive.
        logp = (
            np.log(C / D)
            - np.log(np.where(nw > 0, nw / D, 1.0))
            - np.log(np.where(pc > 0, pc, 1.0))
        )
    return np.where(C > 0, logp, -np.inf)


def ppmi(cooc: ArrayLike, cds_alpha: float = 0.75) -> NDArray:
    return np.maximum(pmi_matrix(cooc, cds_alpha), 0.0)
