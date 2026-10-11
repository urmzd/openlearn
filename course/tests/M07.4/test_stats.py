"""Course tests for M07.4: LLN, CLT, confidence intervals, and the bootstrap
(tinyllm/prob/stats.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M07.4), and the chapter section it comes from.

The chapter's worked example (section 3): x = [2, 4, 4, 5, 7, 8] has mean 5,
squared deviations summing to 24, s^2 = 24 / 5, standard error
sqrt(4.8 / 6) = 0.894427, t_{0.975, 5} = 2.570582, so the 95% interval is
[2.700802, 7.299198]; t_cdf(1, 3) = 1/2 + 1/6 + sqrt(3) / (4 pi) = 0.804499;
and the Wilson interval for 0 successes in 10 trials is [0, 0.277533].

Golden values come from scipy 1.17.1 (course/fixtures/M07.4/scipy_golden.json,
written by course/oracle/M07.4/scipy_golden.py). Statistical tests draw from
the frozen PCG32 at OL_SEED.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.prob.stats import (
    bootstrap_ci,
    mean_ci,
    normal_cdf,
    normal_ppf,
    quantile,
    standard_error,
    t_cdf,
    t_ppf,
    wilson_interval,
)

FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "M07.4"
    / "scipy_golden.json"
)
HAND = [2.0, 4.0, 4.0, 5.0, 7.0, 8.0]


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def golden() -> dict:
    return json.loads(FIX.read_text())


class Counting:
    """The frozen PCG32 behind `uniform()`, counting the draws."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)
        self.calls = 0

    def uniform(self) -> float:
        self.calls += 1
        return self.g.uniform()


class Script:
    """A uniform source that returns fixed values in order."""

    def __init__(self, values) -> None:
        self.values = list(values)
        self.calls = 0

    def uniform(self) -> float:
        v = self.values[self.calls % len(self.values)]
        self.calls += 1
        return v


# --- the worked example ---------------------------------------------------------


def test_hand_example_mean_ci():
    # WHY: the chapter's worked example, number for number: Bessel's n - 1,
    #      the standard error s / sqrt(n), and Student's t quantile with
    #      n - 1 = 5 degrees of freedom (2.570582, not the normal 1.959964,
    #      which would give the too-narrow [3.246955, 6.753045]).
    # KIND: unit
    # CATCHES: s01, s02, s03, s11, m08
    # CHAPTER: M07.4 section 3, Worked example by hand
    assert_close(standard_error(HAND), math.sqrt(0.8), dtype="float64")
    mean, lo, hi = mean_ci(HAND)
    assert_close(
        [mean, lo, hi],
        [5.0, 2.700801709516404, 7.299198290483596],
        rtol=1e-9,
        atol=1e-12,
    )
    assert_close(
        t_cdf(1.0, 3), 0.5 + 1 / 6 + math.sqrt(3) / (4 * math.pi), dtype="float64"
    )


def test_hand_example_wilson():
    # WHY: zero successes in ten trials. The Wald interval phat -+ z se is
    #      [0, 0]: it claims certainty after ten tries. Wilson's center is
    #      pulled toward 1/2 by z^2 / (2n), giving [0, 0.277533], an honest
    #      upper bound on a failure rate nobody has observed yet.
    # KIND: unit
    # CATCHES: s04, s13
    # CHAPTER: M07.4 section 3, Worked example by hand
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert_close(hi, 0.2775327998628892, rtol=1e-9, atol=1e-12)


# --- against scipy ------------------------------------------------------------------


def test_normal_matches_scipy():
    # WHY: normal_cdf is 0.5 erfc(-z / sqrt 2) and normal_ppf inverts it by
    #      bisection, to about the last bit, including the far tails
    #      (p = 1e-12 and 1 - 1e-12) where computing 1 - cdf would lose every
    #      digit. Every z interval in the course (Wilson, L6.7) starts here.
    # KIND: golden
    # CATCHES: s09, m01
    # CHAPTER: M07.4 section 2, Principles (quantiles)
    g = golden()
    for z, want in g["normal_cdf"]:
        assert_close(
            normal_cdf(z), want, rtol=1e-12, atol=1e-300, msg=f"normal_cdf({z})"
        )
    for p, want in g["normal_ppf"]:
        assert_close(
            normal_ppf(p), want, rtol=1e-11, atol=1e-14, msg=f"normal_ppf({p})"
        )


def test_t_matches_scipy():
    # WHY: Student's t over 13 degrees of freedom from 1 (Cauchy) to 1000,
    #      negative and positive t, odd and even df (two different series),
    #      and quantiles out to p = 0.9995, where df = 1 needs t = 636.6: a
    #      bracket that stops growing at a fixed bound returns the wrong value.
    # KIND: golden
    # CATCHES: s08, s10, s12
    # CHAPTER: M07.4 section 2, Principles (Student's t)
    g = golden()
    for t, df, want in g["t_cdf"]:
        assert_close(t_cdf(t, df), want, rtol=1e-9, atol=1e-13, msg=f"t_cdf({t}, {df})")
    for p, df, want in g["t_ppf"]:
        assert_close(t_ppf(p, df), want, rtol=1e-9, atol=1e-12, msg=f"t_ppf({p}, {df})")


def test_mean_ci_matches_scipy():
    # WHY: standard_error equals scipy.stats.sem and mean_ci equals
    #      scipy.stats.t.interval on five samples (n from 2 to 200, normal,
    #      skewed, uniform) at alpha 0.05, 0.1, and 0.01. This is the interval
    #      every L6.7 metric reports.
    # KIND: golden
    # CATCHES: s01, s02, s03, m03, m08
    # CHAPTER: M07.4 section 4, The interface
    g = golden()
    for name, want in g["standard_error"].items():
        assert_close(
            standard_error(g["samples"][name]), want, rtol=1e-10, atol=1e-14, msg=name
        )
    for name, alpha, mean, lo, hi in g["mean_ci"]:
        got = mean_ci(g["samples"][name], alpha)
        assert_close(
            got, [mean, lo, hi], rtol=1e-9, atol=1e-12, msg=f"{name} alpha={alpha}"
        )


def test_wilson_matches_scipy():
    # WHY: the Wilson interval of scipy's binomtest at the extremes (k = 0,
    #      k = n, n = 1) and in the middle, at three confidence levels.
    # KIND: golden
    # CATCHES: s04, m05
    # CHAPTER: M07.4 section 2, Principles (proportions)
    for k, n, alpha, lo, hi in golden()["wilson"]:
        assert_close(
            wilson_interval(k, n, alpha),
            [lo, hi],
            rtol=1e-9,
            atol=1e-12,
            msg=f"k={k} n={n} alpha={alpha}",
        )


def test_quantile_matches_numpy_type7():
    # WHY: the bootstrap's percentile rule is linear interpolation between
    #      order statistics (numpy's default, type 7). Go's re-implementation
    #      (ag.12) must use the same rule, or the two CIs differ in the third
    #      digit on small n_boot.
    # KIND: golden
    # CATCHES: s07
    # CHAPTER: M07.4 section 2, Principles (the bootstrap)
    g = golden()
    for name, q, want in g["quantile"]:
        assert_close(
            quantile(g["quantile_samples"][name], q),
            want,
            dtype="float64",
            msg=f"{name} q={q}",
        )


def test_bootstrap_matches_independent_resampling():
    # WHY: with the same PCG32 seed, one uniform per index in order, your
    #      percentile interval equals an independent implementation's
    #      (numpy statistics and quantiles over the same resamples), for the
    #      mean and for the median, which has no standard-error formula.
    # KIND: golden
    # CATCHES: s05, s06, s14, m06
    # CHAPTER: M07.4 section 4, The interface
    g = golden()
    for name, statname, n_boot, alpha, s, point, lo, hi in g["bootstrap"]:
        stat = np.mean if statname == "mean" else np.median
        got = bootstrap_ci(g["samples"][name], stat, n_boot, alpha, PCG32(seed=s))
        assert_close(
            got, [point, lo, hi], rtol=1e-10, atol=1e-12, msg=f"{name} {statname}"
        )


# --- the CLT at work ------------------------------------------------------------------


def _coverage(draw, mu: float, n: int, sims: int) -> float:
    hits = 0
    for _ in range(sims):
        _, lo, hi = mean_ci(draw(n))
        hits += lo <= mu <= hi
    return hits / sims


def test_mean_ci_coverage_normal_small_n():
    # WHY: what "95%" promises: over 2000 repeated samples of n = 10 normal
    #      values, about 95% of the intervals contain the true mean. With
    #      Student's t the coverage is exactly 95% for normal data; with the
    #      normal quantile it is only about 92%, outside the +-2% band.
    # KIND: statistical
    # CATCHES: s01, s12
    # CHAPTER: M07.4 section 2, Principles (the CLT and Student's t)
    g = PCG32(seed=seed())
    cov = _coverage(
        lambda n: np.array([1.0 + 2.0 * g.normal() for _ in range(n)]), 1.0, 10, 2000
    )
    assert 0.93 <= cov <= 0.97, f"coverage {cov:.3f} outside [0.93, 0.97]"


def test_mean_ci_coverage_skewed_data_by_clt():
    # WHY: the CLT at work: exponential data is far from normal, yet with
    #      n = 200 the sample mean is close enough to normal that the t
    #      interval covers the true mean (1.5) about 95% of the time. This is
    #      why per-example eval scores, which are never normal, still get
    #      honest error bars.
    # KIND: statistical
    # CATCHES: s03, s11
    # CHAPTER: M07.4 section 2, Principles (the central limit theorem)
    g = PCG32(seed=seed() + 1)

    def draw(n: int) -> np.ndarray:
        return np.array([-1.5 * math.log1p(-g.uniform()) for _ in range(n)])

    cov = _coverage(draw, 1.5, 200, 2000)
    assert 0.93 <= cov <= 0.97, f"coverage {cov:.3f} outside [0.93, 0.97]"


def test_standard_error_shrinks_like_root_n():
    # WHY: the law of large numbers with a rate: repeating the same data four
    #      times keeps s about the same and halves the standard error,
    #      s' / sqrt(4n). Quadrupling an eval set halves its error bars; it
    #      does not quarter them.
    # KIND: property
    # CATCHES: s02, s11
    # CHAPTER: M07.4 section 2, Principles (the law of large numbers)
    x = np.array(HAND)
    x4 = np.tile(x, 4)
    s4 = math.sqrt(float(((x4 - x4.mean()) ** 2).sum()) / (x4.size - 1))
    assert_close(standard_error(x4), s4 / math.sqrt(x4.size), dtype="float64")
    assert 0.45 < standard_error(x4) / standard_error(x) < 0.55


# --- quantile functions ------------------------------------------------------------------


def test_ppf_inverts_cdf_and_t_tends_to_normal():
    # WHY: the quantile is the inverse of the CDF, both distributions are
    #      symmetric around 0, and t's quantile falls toward the normal one as
    #      df grows (the extra width is the price of estimating sigma).
    # KIND: property
    # CATCHES: s09, s08, m01
    # CHAPTER: M07.4 section 2, Principles (quantiles)
    for p in [1e-9, 0.01, 0.2, 0.5, 0.8, 0.975, 0.999]:
        assert_close(normal_cdf(normal_ppf(p)), p, rtol=1e-12, atol=1e-15)
        assert_close(normal_ppf(1 - p), -normal_ppf(p), rtol=1e-9, atol=1e-12)
        for df in (1, 2, 7, 30):
            assert_close(t_cdf(t_ppf(p, df), df), p, rtol=1e-9, atol=1e-13)
    widths = [t_ppf(0.975, df) for df in (1, 2, 5, 30, 1000)]
    assert all(a > b for a, b in zip(widths, widths[1:]))
    assert widths[-1] > normal_ppf(0.975)
    assert abs(t_ppf(0.975, 10000) - normal_ppf(0.975)) < 1e-3


def test_t_closed_forms():
    # WHY: two distributions you can check by hand: df = 1 is the Cauchy,
    #      P(T <= t) = 1/2 + atan(t) / pi, and df = 2 is
    #      1/2 + t / (2 sqrt(2 + t^2)). They are the shortest cases of the odd
    #      and the even series.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: M07.4 section 2, Principles (Student's t)
    for t in [-3.0, -0.4, 0.0, 0.7, 5.0]:
        assert_close(t_cdf(t, 1), 0.5 + math.atan(t) / math.pi, dtype="float64")
        assert_close(t_cdf(t, 2), 0.5 + t / (2 * math.sqrt(2 + t * t)), dtype="float64")
    assert t_cdf(math.inf, 4) == 1.0 and t_cdf(-math.inf, 4) == 0.0


# --- the bootstrap ------------------------------------------------------------------------


def test_bootstrap_draws_and_determinism():
    # WHY: exactly n_boot * n uniforms, in order, and the point estimate is
    #      stat(x) on the original data (not the mean of the resampled
    #      statistics). That is what lets Go (ag.12) reproduce the interval
    #      from the same seed.
    # KIND: unit
    # CATCHES: s06, s14
    # CHAPTER: M07.4 section 4, The interface
    x = np.array([0.2, 0.9, 0.4, 0.4, 0.1, 0.7, 0.3])
    a, b = Counting(seed()), Counting(seed())
    r1 = bootstrap_ci(x, np.median, 40, 0.1, a)
    r2 = bootstrap_ci(x, np.median, 40, 0.1, b)
    assert a.calls == 40 * 7 and b.calls == 40 * 7
    assert r1 == r2
    assert r1[0] == float(np.median(x))
    assert r1[1] <= r1[2]


def test_bootstrap_index_rule():
    # WHY: index i = min(floor(u n), n - 1). With n = 4, u = 0.74 is index 2,
    #      not round(2.96) = 3, and u just below 1 is the last index. A
    #      resample of all one index makes the statistic that value.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M07.4 section 5, Pitfalls
    x = np.array([10.0, 20.0, 30.0, 40.0])
    point, lo, hi = bootstrap_ci(x, np.mean, 1, 0.5, Script([0.74]))
    assert (point, lo, hi) == (25.0, 30.0, 30.0)
    _, lo, hi = bootstrap_ci(x, np.mean, 1, 0.5, Script([0.9999999999]))
    assert (lo, hi) == (40.0, 40.0)
    _, lo, hi = bootstrap_ci(
        x, np.mean, 2, 0.5, Script([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    )
    assert (lo, hi) == (15.0, 25.0)


def test_bootstrap_shift_equivariance():
    # WHY: adding a constant to every value moves the mean's percentile
    #      interval by that constant, draw for draw, with the same seed: the
    #      resampled indices do not depend on the values.
    # KIND: property
    # CATCHES: s06, m06
    # CHAPTER: M07.4 section 2, Principles (the bootstrap)
    x = np.array([1.0, 3.0, 2.0, 8.0, 5.0, 4.0])
    p0, lo0, hi0 = bootstrap_ci(x, np.mean, 60, 0.1, PCG32(seed=seed()))
    p1, lo1, hi1 = bootstrap_ci(x + 100.0, np.mean, 60, 0.1, PCG32(seed=seed()))
    assert_close([p1 - 100, lo1 - 100, hi1 - 100], [p0, lo0, hi0], rtol=1e-9, atol=1e-9)
    assert lo0 < p0 < hi0


# --- boundaries ---------------------------------------------------------------------------


def test_quantile_edges():
    # WHY: q = 0 is the minimum, q = 1 the maximum (no index past the end),
    #      a single value is every quantile, and the input order does not
    #      matter.
    # KIND: boundary
    # CATCHES: s07, m04
    # CHAPTER: M07.4 section 2, Principles (the bootstrap)
    x = [3.0, -1.0, 2.0, 9.0]
    assert quantile(x, 0.0) == -1.0
    assert quantile(x, 1.0) == 9.0
    assert quantile([4.0], 0.3) == 4.0
    assert_close(quantile(x, 0.5), 2.5, dtype="float64")
    assert_close(quantile(x, 0.9), 9.0 - 0.3 * 6.0, dtype="float64")


def test_wilson_stays_inside_zero_one():
    # WHY: at k = n the interval ends at exactly 1 and at k = 0 it starts at
    #      exactly 0, never outside [0, 1]; in the middle it contains phat.
    # KIND: boundary
    # CATCHES: s04, s13
    # CHAPTER: M07.4 section 5, Pitfalls
    for n in (1, 3, 10, 57, 1000):
        lo0, hi0 = wilson_interval(0, n)
        lon, hin = wilson_interval(n, n)
        assert lo0 == 0.0 and 0.0 < hi0 < 1.0
        assert hin == 1.0 and 0.0 < lon < 1.0
        for k in range(1, n, max(1, n // 7)):
            lo, hi = wilson_interval(k, n)
            assert 0.0 < lo < k / n < hi < 1.0


def test_rejects_bad_arguments():
    # WHY: an interval from one value, a level outside (0, 1), or k > n is a
    #      bug upstream; raising at the call site beats a NaN in a report.
    # KIND: boundary
    # CATCHES: m03, m09
    # CHAPTER: M07.4 section 4, The interface
    bad = [
        lambda: standard_error([1.0]),
        lambda: mean_ci([1.0, 2.0], 0.0),
        lambda: mean_ci([1.0, 2.0], 1.0),
        lambda: mean_ci([1.0, float("nan")]),
        lambda: mean_ci([[1.0, 2.0], [3.0, 4.0]]),
        lambda: normal_ppf(0.0),
        lambda: normal_ppf(1.0),
        lambda: t_ppf(0.5, 0),
        lambda: t_cdf(1.0, 2.5),
        lambda: wilson_interval(5, 4),
        lambda: wilson_interval(0, 0),
        lambda: quantile([], 0.5),
        lambda: quantile([1.0], 1.5),
        lambda: bootstrap_ci([1.0, 2.0], np.mean, 0, 0.05, Script([0.5])),
        lambda: bootstrap_ci([], np.mean, 10, 0.05, Script([0.5])),
    ]
    for i, f in enumerate(bad):
        with pytest.raises(ValueError):
            f()
            pytest.fail(f"case {i} did not raise")
    assert mean_ci([1.0, 2.0])[0] == 1.5
