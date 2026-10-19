# contracts/py/tinyllm/info/pmi.pyi (M11.4): mutual information and PMI, in nats
# chapter: content/courses/mathematics-for-computing/11-information-theory/04-mutual-information-pmi-ppmi.md
#
# A co-occurrence table cooc[w, c] counts how often word w was seen with
# context c (any non-negative numbers: counts, or weighted counts). With
# D = sum of all entries, #(w) = row sums, #(c) = column sums, the maximum
# likelihood probabilities are P(w, c) = cooc[w, c] / D, P(w) = #(w) / D,
# P(c) = #(c) / D (M07.2). Every log is natural. Inputs are not modified;
# results are float64.
from numpy.typing import ArrayLike, NDArray

def mutual_information(joint: ArrayLike) -> float:
    """I(X; Y) = KL(P(x, y) || P(x) P(y)) in nats, computed with M11.1's kl,
    for a 2-D table of counts or probabilities (normalized by its sum
    first). 0 exactly when the table is an outer product (X and Y
    independent); 0 <= I <= min(H(X), H(Y)); symmetric under transposing.
    ValueError unless joint is 2-D, has no negative or non-finite entry,
    and has a positive sum."""

def pmi_matrix(cooc: ArrayLike, cds_alpha: float = 0.75) -> NDArray:
    """PMI with context distribution smoothing (Levy, Goldberg, Dagan 2015):
        PMI(w, c) = log P(w, c) - log P(w) - log P_alpha(c),
        P_alpha(c) = #(c)^alpha / sum over c' of #(c')^alpha,
    the same shape as cooc. cds_alpha = 1 is plain PMI; alpha < 1 raises
    the probability of rare contexts, which lowers their PMI. An entry with
    cooc[w, c] == 0 is -inf (log 0). ValueError unless cooc is 2-D with no
    negative or non-finite entry and a positive sum, and 0 < cds_alpha <= 1."""

def ppmi(cooc: ArrayLike, cds_alpha: float = 0.75) -> NDArray:
    """Positive PMI: max(pmi_matrix(cooc, cds_alpha), 0), so unseen pairs
    (-inf) and pairs seen less often than chance become exactly 0. Finite
    and non-negative everywhere; the matrix L2.3 factors with M03.5's SVD.
    ValueError as pmi_matrix."""
