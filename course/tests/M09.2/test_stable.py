"""Course tests for M09.2: stable numerics (tinyllm/num/stable.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M09.2), and the chapter section it comes from.

The worked example of the chapter (section 3) is the row x = [1, 2, 3]
for log-sum-exp and softmax, and the float32 sum
[2^24, 1, 1, -2^24] = 2 for the compensated sums.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from _lib.close import assert_close, assert_close_bounded
from _lib.pcg32 import PCG32
from tinyllm.num.stable import (
    kahan_sum,
    log_softmax,
    logsumexp,
    pairwise_sum,
    softmax,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M09.2" / "scipy_golden.npz"

# Section 3: x = [1, 2, 3], m = 3, sum exp(x - m) = e^-2 + e^-1 + 1.
HAND_X = [1.0, 2.0, 3.0]
HAND_S = math.exp(-2) + math.exp(-1) + 1.0  # 1.5032147244...
HAND_LSE = 3.0 + math.log(HAND_S)  # 3.4076059644...
HAND_SOFTMAX = [math.exp(-2) / HAND_S, math.exp(-1) / HAND_S, 1.0 / HAND_S]
HAND_LOG_SOFTMAX = [v - HAND_LSE for v in HAND_X]

U32 = 2.0**-24  # unit roundoff of float32


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def exact_sum(x: np.ndarray) -> float:
    """The correctly rounded sum of the float values in x (math.fsum)."""
    return math.fsum(float(v) for v in np.asarray(x).ravel())


def sequential_f32(x: np.ndarray) -> float:
    """Plain left-to-right float32 summation (np.cumsum adds in order)."""
    return float(np.cumsum(np.asarray(x, dtype=np.float32))[-1])


# --- log-sum-exp, softmax, log-softmax -----------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example, number for number: m = 3, the shifted
    #      exponentials e^-2, e^-1, 1 sum to 1.50321, so logsumexp = 3.40761,
    #      softmax = [0.09003, 0.24473, 0.66524], log_softmax = x - 3.40761.
    #      You and the test agree on the definitions before any edge case.
    # KIND: unit
    # CATCHES: s05
    # CHAPTER: M09.2 section 3, Worked example by hand
    x = np.array(HAND_X)
    assert_close(logsumexp(x), HAND_LSE)
    assert_close(softmax(x), HAND_SOFTMAX)
    assert_close(log_softmax(x), HAND_LOG_SOFTMAX)


def test_hand_example_sums():
    # WHY: the worked example of section 3 for sums: in float32, 2^24 + 1
    #      rounds back to 2^24 (the gap between floats there is 2). Left to
    #      right the sum of [2^24, 1, 1, -2^24] is 0; Kahan recovers the lost
    #      ones and returns the exact 2; pairwise gives (2^24 + 1) + (1 - 2^24)
    #      = 2^24 - 16777215 = 1, better than 0 but not exact.
    # KIND: unit
    # CATCHES: s06, s07, s09, m02
    # CHAPTER: M09.2 section 3, Worked example by hand
    x = np.array([2.0**24, 1.0, 1.0, -(2.0**24)], dtype=np.float32)
    assert sequential_f32(x) == 0.0  # the problem, shown by plain addition
    assert kahan_sum(x) == 2.0
    assert pairwise_sum(x) == 1.0
    assert type(kahan_sum(x)) is float and type(pairwise_sum(x)) is float


def test_matches_scipy_golden():
    # WHY: scipy.special's logsumexp, softmax, and log_softmax on rows of
    #      several shapes, axes, and scales (near +-9000, with -inf masks),
    #      recorded by course/oracle/M09.2/scipy_golden.py. An independent
    #      implementation agrees with yours to float64 tolerance.
    # KIND: golden
    # CATCHES: s01, s05, s11, s12
    # CHAPTER: M09.2 section 2, Principles
    data = np.load(GOLDEN, allow_pickle=False)
    names = [k[: -len("/x")] for k in data.files if k.endswith("/x")]
    assert len(names) == 7
    for name in names:
        x = data[f"{name}/x"]
        axis = int(data[f"{name}/axis"])
        for fn, key in (
            (logsumexp, "logsumexp"),
            (softmax, "softmax"),
            (log_softmax, "log_softmax"),
        ):
            assert_close(
                fn(x, axis=axis), data[f"{name}/{key}"], msg=f"{key} on {name}"
            )


def test_shift_invariance():
    # WHY: softmax(x + c) = softmax(x) and log-sum-exp(x + c) = lse(x) + c
    #      for any constant c. This identity is the whole trick (subtract the
    #      max), and the online softmax of L9.2 relies on it to rescale a
    #      running sum when the max changes.
    # KIND: property
    # CATCHES: s01, s05
    # CHAPTER: M09.2 section 2, Principles (shift invariance)
    rng = PCG32(seed=seed())
    for _ in range(20):
        x = rng.normal_array((3, 9), scale=5.0)
        c = 2000.0 * rng.uniform() - 1000.0
        assert_close_bounded(softmax(x + c), softmax(x), k=9, dtype="float64")
        assert_close_bounded(logsumexp(x + c), logsumexp(x) + c, k=9, dtype="float64")
        assert_close_bounded(log_softmax(x + c), log_softmax(x), k=9, dtype="float64")


def test_no_overflow():
    # WHY: e^1000 overflows float64 (the limit is about 709.8) and e^89
    #      overflows float32. A softmax that exponentiates first returns
    #      inf / inf = nan. Logits this large appear after training and in
    #      masked attention scores; the answer is still finite.
    # KIND: boundary
    # CATCHES: s01, m01
    # CHAPTER: M09.2 section 5, Pitfalls, item 1
    assert_close(softmax(np.array([1000.0, 1000.0])), [0.5, 0.5])
    assert_close(logsumexp(np.array([1000.0, 1000.0])), 1000.0 + math.log(2))
    big = np.array([1e4, -1e4, 0.0])
    assert_close(softmax(big), [1.0, 0.0, 0.0])
    assert_close(logsumexp(big), 1e4)
    f32 = np.array([100.0, 99.0], dtype=np.float32)
    assert_close(
        softmax(f32),
        [1 / (1 + math.exp(-1)), math.exp(-1) / (1 + math.exp(-1))],
        dtype="float32",
    )


def test_no_underflow_in_log_softmax():
    # WHY: softmax([0, -1e4]) is [1, 0] because e^-10000 underflows to 0, so
    #      log(softmax(x)) gives -inf where the answer is -10000. The cross-
    #      entropy of L0.3 is -log_softmax at the target; a -inf there is an
    #      infinite loss and a nan gradient.
    # KIND: boundary
    # CATCHES: s02, m01
    # CHAPTER: M09.2 section 5, Pitfalls, item 2
    out = log_softmax(np.array([0.0, -1e4]))
    assert np.isfinite(out).all()
    assert_close(out, [0.0, -1e4])
    out32 = log_softmax(np.array([0.0, -200.0], dtype=np.float32))
    assert np.isfinite(out32).all()
    assert_close(out32, [0.0, -200.0], dtype="float32")


def test_fully_masked_row():
    # WHY: a causal mask writes -inf over the future; a padded row can be
    #      masked entirely. Then the max is -inf and x - max is -inf - -inf =
    #      nan. The contract defines the row instead: softmax gives zeros,
    #      logsumexp and log_softmax give -inf, and the other rows are
    #      untouched. A partly masked row puts exactly 0 on the masked entries.
    # KIND: boundary
    # CATCHES: s03, s04, s14
    # CHAPTER: M09.2 section 5, Pitfalls, item 3
    ninf = -np.inf
    x = np.array([[ninf, ninf, ninf], [0.0, ninf, 0.0]])
    with np.errstate(all="raise"):
        p = softmax(x)
        lse = logsumexp(x)
        ls = log_softmax(x)
    assert not np.isnan(p).any() and not np.isnan(lse).any() and not np.isnan(ls).any()
    assert p.tolist() == [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
    assert lse[0] == ninf
    assert_close(lse[1], math.log(2))
    assert np.isneginf(ls[0]).all()
    assert_close(ls[1], [-math.log(2), ninf, -math.log(2)])


def test_axis_and_keepdims():
    # WHY: attention normalizes over the last axis of [B, H, T, T], a
    #      classifier over axis 1 of [N, C]; every reduction must follow
    #      `axis`, and keepdims must keep the reduced axis as size 1 so the
    #      result broadcasts back against x.
    # KIND: unit
    # CATCHES: s11, s12
    # CHAPTER: M09.2 section 4, The interface
    rng = PCG32(seed=seed())
    x = rng.normal_array((2, 3, 4))
    for axis in (0, 1, 2, -1, -2):
        p = softmax(x, axis=axis)
        assert p.shape == x.shape
        assert_close_bounded(
            p.sum(axis=axis), np.ones(np.delete(x.shape, axis)), k=4, dtype="float64"
        )
        lse = logsumexp(x, axis=axis)
        assert lse.shape == tuple(np.delete(x.shape, axis))
        lse_k = logsumexp(x, axis=axis, keepdims=True)
        expect = list(x.shape)
        expect[axis] = 1
        assert lse_k.shape == tuple(expect)
        assert_close(log_softmax(x, axis=axis), x - lse_k)
    assert np.ndim(logsumexp(np.array([1.0, 2.0]))) == 0


def test_dtype_preserved():
    # WHY: float32 logits stay float32 (the kernels of L9 and the engine
    #      compare against float32 references); integer input is computed in
    #      float64 instead of failing or truncating.
    # KIND: unit
    # CATCHES: s13
    # CHAPTER: M09.2 section 4, The interface
    x32 = np.array([[0.5, 1.5, -2.0]], dtype=np.float32)
    for fn in (softmax, log_softmax, logsumexp):
        assert fn(x32).dtype == np.float32
    assert softmax(np.array([1, 2, 3])).dtype == np.float64
    assert_close(softmax(np.array([1, 2, 3])), HAND_SOFTMAX)


def test_softmax_rows_sum_to_one():
    # WHY: every unmasked softmax row is a probability distribution, for
    #      rows of any scale. The sampler of L8.1 inverts its CDF and assumes
    #      the last entry of the CDF is 1.
    # KIND: property
    # CATCHES: s01, m01
    # CHAPTER: M09.2 section 2, Principles
    rng = PCG32(seed=seed())
    for scale in (0.1, 1.0, 30.0, 700.0):
        x = rng.normal_array((16, 64), scale=scale)
        p = softmax(x)
        assert (p >= 0).all()
        assert_close_bounded(p.sum(axis=-1), np.ones(16), k=64, dtype="float64")


# --- compensated sums --------------------------------------------------------


def test_kahan_error_bound():
    # WHY: Kahan's error is about 2u|S| no matter how many terms, so 65536
    #      float32 values sum as if in higher precision. The data are
    #      near-constant positive values (like per-token losses), where plain
    #      left-to-right rounding errors all point the same way and miss the
    #      same bound by a factor over 1000 (asserted first, so the test cannot
    #      pass vacuously). L6.7 sums 10^7 per-token losses with it (M11.2).
    # KIND: property
    # CATCHES: s06, s07, m02
    # CHAPTER: M09.2 section 2, Principles (Kahan summation)
    rng = PCG32(seed=seed())
    x = (0.1 + 1e-4 * rng.uniform_array(65536)).astype(np.float32)
    exact = exact_sum(x)
    bound = 2 * U32 * abs(exact) + 2 * x.size * U32**2 * float(
        np.abs(x).astype(np.float64).sum()
    )
    assert abs(sequential_f32(x) - exact) > 100 * bound
    assert abs(kahan_sum(x) - exact) <= bound


def test_pairwise_error_bound():
    # WHY: pairwise summation's error grows with log2(n), not n: each value
    #      takes part in only about log2(n) additions. On near-constant
    #      data, where plain left-to-right float32 summation misses the bound
    #      by a factor over 100 (asserted first), pairwise stays inside
    #      ceil(log2 n) u sum|x|.
    # KIND: property
    # CATCHES: s09
    # CHAPTER: M09.2 section 2, Principles (pairwise summation)
    rng = PCG32(seed=seed())
    for n in (65536, 50001):
        x = (0.1 + 1e-4 * rng.uniform_array(n)).astype(np.float32)
        exact = exact_sum(x)
        bound = (
            math.ceil(math.log2(n)) * U32 * float(np.abs(x).astype(np.float64).sum())
        )
        assert abs(sequential_f32(x) - exact) > 10 * bound
        assert abs(pairwise_sum(x) - exact) <= bound, f"n = {n}"


def test_sums_cover_every_element():
    # WHY: odd lengths, one element, and the empty array: a recursion that
    #      drops the odd element or reads x[0] of an empty array is off by a
    #      whole term, which no error bound forgives. Small integers sum
    #      exactly in any order, so the answer is exact.
    # KIND: boundary
    # CATCHES: s08, s10
    # CHAPTER: M09.2 section 4, The interface
    for n in (0, 1, 2, 3, 5, 7, 8, 9, 31, 100):
        x = np.arange(1, n + 1, dtype=np.float32)
        assert kahan_sum(x) == n * (n + 1) / 2, f"kahan n = {n}"
        assert pairwise_sum(x) == n * (n + 1) / 2, f"pairwise n = {n}"
    m = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert kahan_sum(m) == 66.0 and pairwise_sum(m) == 66.0
    assert kahan_sum(np.array([3, 4])) == 7.0


def test_sums_work_in_the_input_dtype():
    # WHY: the same example one precision up. In float64 the gap between
    #      floats at 1e16 is 2, so 1e16 + 1 rounds back to 1e16 and so does
    #      1 - 1e16: plain and pairwise summation both return 0, Kahan the
    #      exact 2. Every operation happens in the input's dtype.
    # KIND: unit
    # CATCHES: s06, s07
    # CHAPTER: M09.2 section 2, Principles (Kahan summation)
    x = np.array([1e16, 1.0, 1.0, -1e16])
    assert float(np.cumsum(x)[-1]) == 0.0
    assert kahan_sum(x) == 2.0
    assert pairwise_sum(x) == 0.0
