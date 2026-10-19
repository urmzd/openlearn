"""LLN, CLT, confidence intervals, and the bootstrap (M07.4).

The law of large numbers says the sample mean of n independent draws converges
to the true mean mu. The central limit theorem says how: the error
(xbar - mu) is approximately normal with standard deviation sigma / sqrt(n).
So an interval "estimate +- quantile * standard error" covers mu with a known
probability. Student's t replaces the normal quantile when sigma itself is
estimated from the same n values; the Wilson interval handles proportions near
0 and 1; the percentile bootstrap handles statistics with no formula (BLEU,
median, a pass@k) by resampling the data.

Every quantile is computed here from erfc and the closed-form t series, then
bisection, in float64.

Contract: contracts/py/tinyllm/prob/stats.pyi.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class UniformSource(Protocol):
    def uniform(self) -> float: ...


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    return a


def _bisect(f: Callable[[float], float], target: float, lo: float, hi: float) -> float:
    """The x in [lo, hi] where the increasing f crosses target, to the last bit:
    keeps f(lo) < target <= f(hi) and halves until the midpoint is an end."""
    for _ in range(2000):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-float(z) / math.sqrt(2.0))


def normal_ppf(p: float) -> float:
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"normal_ppf needs 0 < p < 1, got {p!r}")
    if p == 0.5:
        return 0.0
    if p > 0.5:
        # 1 - p is exact for p in [0.5, 1) (Sterbenz), so the upper tail keeps
        # its precision by symmetry.
        return -normal_ppf(1.0 - p)
    return _bisect(normal_cdf, p, -40.0, 0.0)


def _t_abs_prob(t: float, df: int) -> float:
    """A = P(|T| < |t|) by Abramowitz and Stegun 26.7.3 (odd df) and 26.7.4 (even df)."""
    theta = math.atan(abs(t) / math.sqrt(df))
    s, c = math.sin(theta), math.cos(theta)
    c2 = c * c
    if df % 2 == 0:
        # 1 + (1/2) c^2 + (1 3)/(2 4) c^4 + ... : k-th ratio (2k - 1) / (2k) c^2
        m = (df - 2) // 2
        if m == 0:
            return s
        k = np.arange(1, m + 1, dtype=np.float64)
        terms = np.cumprod((2.0 * k - 1.0) / (2.0 * k) * c2)
        return s * (1.0 + float(np.sum(terms)))
    if df == 1:
        return 2.0 * theta / math.pi
    # c + (2/3) c^3 + (2 4)/(3 5) c^5 + ... : k-th ratio (2k) / (2k + 1) c^2
    m = (df - 3) // 2
    k = np.arange(1, m + 1, dtype=np.float64)
    terms = c * np.cumprod((2.0 * k) / (2.0 * k + 1.0) * c2) if m else np.zeros(0)
    return 2.0 / math.pi * (theta + s * (c + float(np.sum(terms))))


def _check_df(df: int) -> int:
    if isinstance(df, bool) or int(df) != df or df < 1:
        raise ValueError(f"df must be an integer >= 1, got {df!r}")
    return int(df)


def t_cdf(t: float, df: int) -> float:
    df = _check_df(df)
    t = float(t)
    if math.isnan(t):
        raise ValueError("t_cdf of NaN")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    a = _t_abs_prob(t, df)
    return 0.5 + 0.5 * a if t >= 0 else 0.5 - 0.5 * a


def t_ppf(p: float, df: int) -> float:
    df = _check_df(df)
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"t_ppf needs 0 < p < 1, got {p!r}")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    hi = 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    return _bisect(lambda v: t_cdf(v, df), p, 0.0, hi)


def _sample(x: ArrayLike, min_n: int) -> NDArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {a.shape}")
    if a.size < min_n:
        raise ValueError(f"x needs at least {min_n} values, got {a.size}")
    if not np.isfinite(a).all():
        raise ValueError("x must be finite")
    return a


def standard_error(x: ArrayLike) -> float:
    a = _sample(x, 2)
    n = a.size
    mean = math.fsum(a.tolist()) / n
    ss = math.fsum(((a - mean) ** 2).tolist())
    # Bessel: the deviations are measured from the sample mean, which sits
    # closer to the data than mu does, so divide by n - 1, not n.
    return math.sqrt(ss / (n - 1)) / math.sqrt(n)


def mean_ci(x: ArrayLike, alpha: float = 0.05) -> tuple[float, float, float]:
    a = _sample(x, 2)
    alpha = _check_alpha(alpha)
    mean = math.fsum(a.tolist()) / a.size
    half = t_ppf(1.0 - alpha / 2.0, a.size - 1) * standard_error(a)
    return mean, mean - half, mean + half


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    for name, v in (("k", k), ("n", n)):
        if isinstance(v, bool) or int(v) != v:
            raise ValueError(f"{name} must be an integer, got {v!r}")
    k, n = int(k), int(n)
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k}, n={n}")
    z = normal_ppf(1.0 - _check_alpha(alpha) / 2.0)
    z2 = z * z
    phat = k / n
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    # In exact arithmetic the interval ends at 0 when k = 0 and at 1 when
    # k = n; rounding can leave it a hair inside or outside, so pin the ends.
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return lo, hi


def quantile(x: ArrayLike, q: float) -> float:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError("quantile of an empty sample")
    if np.isnan(a).any():
        raise ValueError("quantile of a sample with NaN")
    q = float(q)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must lie in [0, 1], got {q!r}")
    s = np.sort(a)
    h = (s.size - 1) * q
    i = int(math.floor(h))
    if i >= s.size - 1:
        return float(s[-1])
    return float(s[i] + (h - i) * (s[i + 1] - s[i]))


def bootstrap_ci(
    x: ArrayLike,
    stat: Callable[[NDArray], float],
    n_boot: int,
    alpha: float,
    rng: UniformSource,
) -> tuple[float, float, float]:
    a = _sample(x, 1)
    alpha = _check_alpha(alpha)
    if isinstance(n_boot, bool) or int(n_boot) != n_boot or n_boot < 1:
        raise ValueError(f"n_boot must be an integer >= 1, got {n_boot!r}")
    n = a.size
    thetas = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = np.empty(n, dtype=np.int64)
        for i in range(n):
            # One uniform per index, in order: the same seed resamples the same
            # rows in Python and in Go (ag.12).
            idx[i] = min(int(rng.uniform() * n), n - 1)
        thetas[b] = float(stat(a[idx]))
    return (
        float(stat(a)),
        quantile(thetas, alpha / 2.0),
        quantile(thetas, 1.0 - alpha / 2.0),
    )
