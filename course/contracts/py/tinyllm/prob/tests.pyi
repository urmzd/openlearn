# contracts/py/tinyllm/prob/tests.pyi (M07.5)
# chapter: content/courses/mathematics-for-computing/07-probability-statistics/05-hypothesis-tests.md
#
# Hypothesis tests for "is model B really better than model A?": a p-value is
# the probability, if the null hypothesis H0 is true, of a statistic at least
# as extreme as the one observed. Small p-values are evidence against H0.
#
# Permutation tests build the null distribution by relabelling the data the
# way H0 says is harmless. Every one is two-sided (it compares |statistic|),
# counts a permuted statistic as "at least as extreme" when it is >= the
# observed one up to a relative 1e-9 (so rounding never splits a tie), and
# returns the add-one estimate p = (1 + count) / (1 + n_perm), which is never
# 0 and never below 1 / (1 + n_perm). Randomness comes in through an `rng`
# with a `uniform() -> float` method (the PCG32 of spec/pcg32.md: M06.3 in
# your repo, the frozen copy in course tests), consumed in the documented
# order so load.02's Go port gets bit-identical p-values from the same seed.
# All arithmetic is float64.
from typing import Callable, Protocol

from numpy.typing import ArrayLike, NDArray

class UniformSource(Protocol):
    def uniform(self) -> float:
        """One float64 uniform in [0, 1) (spec/pcg32.md uniform_f64)."""

def mean_difference(a: NDArray, b: NDArray) -> float:
    """mean(a) - mean(b): the default two-sample statistic."""

def paired_permutation_test(
    a: ArrayLike, b: ArrayLike, n_perm: int, rng: UniformSource
) -> float:
    """Two-sided sign-flip test of H0 "a and b are exchangeable within each
    pair" (the same prompts scored by two models). With d = a - b and
    T(s) = |sum_i s_i d_i|, the observed statistic is T(1, ..., 1). For each
    of n_perm permutations, in order, draw one rng.uniform() per pair i =
    0 .. n - 1: s_i = -1 when u < 0.5, else +1. count = #{T(s) >= T_obs *
    (1 - 1e-9)}; returns (1 + count) / (1 + n_perm). Draws exactly
    n_perm * n uniforms. All-zero differences give 1.0.
    ValueError unless a and b are 1-D, the same length n >= 1, finite, and
    n_perm is an integer >= 1."""

def permutation_test(
    a: ArrayLike,
    b: ArrayLike,
    stat: Callable[[NDArray, NDArray], float],
    n_perm: int,
    rng: UniformSource,
) -> float:
    """Two-sided test of H0 "a and b come from the same distribution": pool
    z = concat(a, b) (float64, a first), observed T_obs = |stat(a, b)|. For
    each of n_perm permutations, start from the identity order idx = 0 .. N-1
    (N = len(a) + len(b)) and shuffle it by Fisher-Yates from the end: for
    i = N - 1 down to 1, u = rng.uniform(), j = min(floor(u * (i + 1)), i),
    swap idx[i] and idx[j]. The permuted samples are z[idx[:len(a)]] and
    z[idx[len(a):]]; T = |stat(...)|; count = #{T >= T_obs * (1 - 1e-9)}.
    Returns (1 + count) / (1 + n_perm). Draws exactly n_perm * (N - 1)
    uniforms. stat is typically mean_difference.
    ValueError unless a and b are 1-D, non-empty, finite, and n_perm is an
    integer >= 1."""

def mcnemar(b01: int, b10: int, exact: bool = True) -> float:
    """Two-sided McNemar test for paired binary outcomes (model A right and
    B wrong: b01; A wrong and B right: b10; the concordant pairs carry no
    information). Under H0 each discordant pair is a fair coin, so with
    n = b01 + b10 and k = min(b01, b10):
      exact:     p = min(1, 2 * sum_{i=0..k} C(n, i) / 2^n)   (binomial, in
                 exact integer arithmetic)
      not exact: chi2 = max(|b01 - b10| - 1, 0)^2 / n with Edwards' continuity
                 correction, p = erfc(sqrt(chi2 / 2)) (chi-square, 1 df)
    n = 0 gives 1.0. ValueError unless b01 and b10 are integers >= 0."""

def holm(pvalues: ArrayLike, alpha: float = 0.05) -> NDArray:
    """Holm's step-down procedure, controlling the family-wise error rate at
    alpha over m tests: sort the p-values ascending (ties by input position),
    reject the k-th smallest (k = 0, 1, ...) while p_(k) <= alpha / (m - k),
    and stop at the first that is not. Returns bool [m], True = rejected, in
    the input order. An empty input gives an empty array.
    ValueError unless pvalues is 1-D with every value in [0, 1] and
    0 < alpha < 1."""

def holm_adjust(pvalues: ArrayLike) -> NDArray:
    """Holm-adjusted p-values, float64 [m] in the input order: in ascending
    order, adj_(k) = max over j <= k of min(1, (m - j) * p_(j)). Then
    holm(p, alpha) == (holm_adjust(p) <= alpha) for every alpha.
    ValueError unless pvalues is 1-D with every value in [0, 1]."""
