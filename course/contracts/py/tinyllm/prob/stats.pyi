# contracts/py/tinyllm/prob/stats.pyi (M07.4)
# chapter: content/courses/mathematics-for-computing/08-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md
#
# Confidence intervals: the error bars every metric in the course carries
# (L4.5 sequence metrics, L6.7 evaluation, L8.5 quantization verdicts, L10.7).
# The law of large numbers says a sample mean converges to the true mean; the
# central limit theorem says how fast, with a normal shape and standard
# deviation sigma / sqrt(n). An interval is "estimate +- quantile * standard
# error", or, when no formula exists, the spread of the statistic over
# bootstrap resamples.
#
# All arithmetic is float64. Quantiles of the normal and Student t
# distributions are computed here from first principles (erfc and the
# closed-form t series, then bisection), so no statistics library is needed.
# Randomness comes in through an `rng` with a `uniform() -> float` method (the
# PCG32 of spec/pcg32.md: M06.3 in your repo, the frozen copy in course tests).
from typing import Callable, Protocol

from numpy.typing import ArrayLike, NDArray

class UniformSource(Protocol):
    def uniform(self) -> float:
        """One float64 uniform in [0, 1) (spec/pcg32.md uniform_f64)."""

def normal_cdf(z: float) -> float:
    """Phi(z) = P(Z <= z) for Z ~ N(0, 1), as 0.5 * erfc(-z / sqrt(2))
    (math.erfc), which keeps full relative precision in the lower tail."""

def normal_ppf(p: float) -> float:
    """The z with Phi(z) = p (the quantile, Phi^{-1}). For p < 0.5, bisection on
    normal_cdf over [-40, 0] until the bracket stops shrinking; for p > 0.5,
    -normal_ppf(1 - p) (1 - p is exact in float64 there); 0.0 at p = 0.5.
    ValueError unless 0 < p < 1."""

def t_cdf(t: float, df: int) -> float:
    """P(T <= t) for Student's t with df degrees of freedom (an integer >= 1),
    by the closed-form series of Abramowitz and Stegun 26.7.3 and 26.7.4 with
    theta = atan(|t| / sqrt(df)): A = P(|T| < |t|) is
      df odd:  (2 / pi) * (theta + sin(theta) * (cos(theta) + (2/3) cos^3(theta)
               + ... + [2 4 ... (df-3)] / [1 3 ... (df-2)] cos^(df-2)(theta)))
               (just 2 theta / pi for df = 1)
      df even: sin(theta) * (1 + (1/2) cos^2(theta) + (1 3)/(2 4) cos^4(theta)
               + ... + [1 3 ... (df-3)] / [2 4 ... (df-2)] cos^(df-2)(theta))
    and the result is 0.5 + 0.5 * sign(t) * A. Cost O(df).
    ValueError unless df is an integer >= 1 and t is not NaN."""

def t_ppf(p: float, df: int) -> float:
    """The t with t_cdf(t, df) = p, by bisection: for p > 0.5 the bracket
    [0, b] with b doubled from 1 until t_cdf(b, df) >= p, then halved until it
    stops shrinking; for p < 0.5, -t_ppf(1 - p, df); 0.0 at p = 0.5. Tends to
    normal_ppf(p) as df grows. ValueError unless 0 < p < 1 and df >= 1."""

def standard_error(x: ArrayLike) -> float:
    """s / sqrt(n): the estimated standard deviation of the sample mean, with
    s the sample standard deviation (divisor n - 1, Bessel's correction).
    ValueError unless x is 1-D, finite, with n >= 2."""

def mean_ci(x: ArrayLike, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi): the Student t interval mean -+ t_ppf(1 - alpha/2, n - 1)
    * standard_error(x). Exact coverage 1 - alpha for normal data, and by the
    CLT approximately 1 - alpha for any finite-variance data as n grows.
    ValueError unless 0 < alpha < 1 (and the checks of standard_error)."""

def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """The Wilson score interval for a proportion with k successes in n trials,
    z = normal_ppf(1 - alpha/2), phat = k / n:
      center = (phat + z^2 / (2n)) / (1 + z^2 / n)
      half   = z * sqrt(phat (1 - phat) / n + z^2 / (4 n^2)) / (1 + z^2 / n)
    returned as (max(0, center - half), min(1, center + half)), except that
    the interval starts at exactly 0.0 when k = 0 and ends at exactly 1.0
    when k = n (where the exact formula lands, rounding aside). Unlike the
    Wald interval phat -+ z sqrt(phat (1 - phat) / n), it is not empty at
    k = 0 or k = n. ValueError unless k, n are integers with 0 <= k <= n,
    n >= 1, and 0 < alpha < 1."""

def quantile(x: ArrayLike, q: float) -> float:
    """The q-quantile of the values in x by linear interpolation between order
    statistics (Hyndman and Fan type 7, numpy's default): with s = sorted(x),
    h = (n - 1) q, i = floor(h), the result is s[i] + (h - i) (s[i+1] - s[i])
    (just s[n-1] when i = n - 1). ValueError for an empty x, a NaN in x, or q
    outside [0, 1]."""

def bootstrap_ci(
    x: ArrayLike,
    stat: Callable[[NDArray], float],
    n_boot: int,
    alpha: float,
    rng: UniformSource,
) -> tuple[float, float, float]:
    """(stat(x), lo, hi): the percentile bootstrap interval. For b = 0 ..
    n_boot - 1, draw a resample of n indices, each from exactly one
    rng.uniform() in order, i = min(floor(u * n), n - 1); theta_b =
    float(stat(x[indices])). lo = quantile(thetas, alpha / 2), hi =
    quantile(thetas, 1 - alpha / 2). Draws exactly n_boot * n uniforms, so
    the same seed gives the same interval in every language (ag.12 re-implements
    it in Go). ValueError for an empty or non-1-D x, n_boot < 1, or alpha
    outside (0, 1)."""
