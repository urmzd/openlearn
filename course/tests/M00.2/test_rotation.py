"""Course tests for M00.2: rotations and Euler's formula (tinyllm/num/rotation.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M00.2), and the chapter section it comes from.

The chapter's worked example (section 3) rotates x = [3, 4, 1, 0]: pair 0,
(3, 4), by the angle t0 whose cosine is 3/5 and sine 4/5, giving (-1.4, 4.8);
pair 1, (1, 0), by a quarter turn, giving (0, 1).
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
from tinyllm.num.rotation import as_complex, as_real, rotate_pairs, rotation_matrix

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M00.2" / "rotation.json"

T0 = math.atan2(4.0, 3.0)  # cos T0 = 3/5, sin T0 = 4/5
HAND_X = [3.0, 4.0, 1.0, 0.0]
HAND_THETA = [T0, math.pi / 2]
HAND_OUT = [-1.4, 4.8, 0.0, 1.0]


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def rand(rng: PCG32, shape, lo=-2.0, hi=2.0) -> np.ndarray:
    return np.asarray(rng.uniform_array(shape, lo, hi), dtype=np.float64)


# --- the worked example ----------------------------------------------------------


def test_hand_example_rotation():
    # WHY: the chapter's worked example, number for number:
    #      (3, 4) -> (3*0.6 - 4*0.8, 3*0.8 + 4*0.6) = (-1.4, 4.8), and a
    #      quarter turn sends (1, 0) to (0, 1).
    # KIND: unit
    # CATCHES: s01, s02, s03, s08, m01
    # CHAPTER: M00.2 section 3, Worked example by hand
    got = rotate_pairs(np.array(HAND_X), np.array(HAND_THETA))
    assert_close(got, HAND_OUT, dtype="float64")


def test_hand_example_as_complex_product():
    # WHY: the same example as complex numbers: (3 + 4i)(0.6 + 0.8i) =
    #      1.8 + 2.4i + 2.4i + 3.2 i^2 = -1.4 + 4.8i. Multiplying by
    #      e^{i t} is the rotation (Euler's formula).
    # KIND: unit
    # CATCHES: s05, s10
    # CHAPTER: M00.2 section 3, Worked example by hand
    z = as_complex(np.array(HAND_X))
    assert z.dtype == np.complex128 and z.shape == (2,)
    assert_close(z.real, [3.0, 1.0], dtype="float64")
    assert_close(z.imag, [4.0, 0.0], dtype="float64")
    w = z * np.exp(1j * np.array(HAND_THETA))
    assert_close(as_real(w), HAND_OUT, dtype="float64")


def test_rotation_matrix_hand_values():
    # WHY: R(t) = [[cos t, -sin t], [sin t, cos t]]: its columns are where
    #      (1, 0) and (0, 1) land. A quarter turn is [[0, -1], [1, 0]].
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: M00.2 section 2.3, Rotations of the plane
    assert_close(
        rotation_matrix(math.pi / 2), [[0.0, -1.0], [1.0, 0.0]], dtype="float64"
    )
    assert_close(rotation_matrix(T0), [[0.6, -0.8], [0.8, 0.6]], dtype="float64")
    assert_close(rotation_matrix(0.0), np.eye(2), dtype="float64")


# --- laws of rotations --------------------------------------------------------------


def test_rotate_pairs_matches_rotation_matrix():
    # WHY: rotating pair i by theta[i] is R(theta[i]) @ (x[2i], x[2i+1]).
    #      Two of your functions must agree on the same definition.
    # KIND: differential
    # CATCHES: s01, s02, s03, s04, s06, s08, m01
    # CHAPTER: M00.2 section 2.3, Rotations of the plane
    rng = PCG32(seed=seed())
    x = rand(rng, (5, 6))
    theta = rand(rng, (5, 3), -7.0, 7.0)
    got = rotate_pairs(x, theta)
    for t in range(5):
        for i in range(3):
            want = rotation_matrix(theta[t, i]) @ x[t, 2 * i : 2 * i + 2]
            assert_close(got[t, 2 * i : 2 * i + 2], want, dtype="float64")


def test_length_is_preserved():
    # WHY: a rotation never stretches: the length of every pair, and so of
    #      the whole vector, is unchanged. RoPE relies on this to move
    #      positions into queries and keys without changing their norms.
    # KIND: property
    # CATCHES: s02, s08
    # CHAPTER: M00.2 section 2.3, Rotations of the plane
    rng = PCG32(seed=seed())
    x = rand(rng, (8, 10))
    theta = rand(rng, (8, 5), -20.0, 20.0)
    y = rotate_pairs(x, theta)
    pair_len = lambda v: np.hypot(v[..., 0::2], v[..., 1::2])  # noqa: E731
    assert_close(pair_len(y), pair_len(x), dtype="float64")


def test_angles_add():
    # WHY: R(a) R(b) = R(a + b): rotating by b then by a is one rotation by
    #      a + b. This is the law that turns absolute positions into relative
    #      ones in RoPE (section 2.5).
    # KIND: property
    # CATCHES: s04, s08, m01
    # CHAPTER: M00.2 section 2.4, Composition: angles add
    rng = PCG32(seed=seed())
    for _ in range(20):
        a, b = rand(rng, (2,), -10.0, 10.0)
        assert_close(
            rotation_matrix(a) @ rotation_matrix(b),
            rotation_matrix(a + b),
            dtype="float64",
        )
    x = rand(rng, (4, 8))
    a = rand(rng, (4, 4), -5.0, 5.0)
    b = rand(rng, (4, 4), -5.0, 5.0)
    assert_close(
        rotate_pairs(rotate_pairs(x, b), a), rotate_pairs(x, a + b), dtype="float64"
    )


def test_negative_angle_undoes():
    # WHY: R(-t) R(t) = I: rotating back by the same angle returns the input.
    # KIND: property
    # CATCHES: s04, s08, m01
    # CHAPTER: M00.2 section 2.4, Composition: angles add
    rng = PCG32(seed=seed())
    x = rand(rng, (3, 12))
    theta = rand(rng, (3, 6), -10.0, 10.0)
    assert_close(rotate_pairs(rotate_pairs(x, theta), -theta), x, dtype="float64")


def test_dot_product_sees_only_relative_angle():
    # WHY: <R(m w) q, R(n w) k> depends only on m - n. Shift both positions by
    #      the same amount and every dot product stays the same: the RoPE
    #      property L7.3 is built on (section 2.5).
    # KIND: property
    # CATCHES: s04, s08
    # CHAPTER: M00.2 section 2.5, Why RoPE rotates pairs
    rng = PCG32(seed=seed())
    q, k = rand(rng, (8,)), rand(rng, (8,))
    w = np.array([1.0, 0.1, 0.01, 0.001])
    for m, n in [(5, 2), (17, 3), (0, 9)]:
        base = rotate_pairs(q, m * w) @ rotate_pairs(k, n * w)
        for shift in (1, 10, 123):
            moved = rotate_pairs(q, (m + shift) * w) @ rotate_pairs(k, (n + shift) * w)
            assert_close(moved, base, rtol=1e-10, atol=1e-10)


# --- complex numbers ------------------------------------------------------------------


def test_euler_formula_is_the_rotation():
    # WHY: as_complex(rotate_pairs(x, t)) == as_complex(x) * e^{i t}.
    #      Euler's formula e^{it} = cos t + i sin t makes the two views one.
    # KIND: differential
    # CATCHES: s01, s02, s03, s04, s05, s08, m01
    # CHAPTER: M00.2 section 2.6, Complex numbers and Euler's formula
    rng = PCG32(seed=seed())
    x = rand(rng, (6, 10))
    theta = rand(rng, (6, 5), -10.0, 10.0)
    lhs = as_complex(rotate_pairs(x, theta))
    rhs = as_complex(x) * np.exp(1j * theta)
    assert_close(lhs.real, rhs.real, dtype="float64")
    assert_close(lhs.imag, rhs.imag, dtype="float64")


def test_layout_is_interleaved():
    # WHY: pair i is (x[2i], x[2i+1]), never (x[i], x[i + k]). The "half"
    #      layout of HF's rotate_half is a different model: same shapes, wrong
    #      numbers, and nothing crashes (pitfall 2).
    # KIND: boundary
    # CATCHES: s02, s03, s05, s10, m01
    # CHAPTER: M00.2 section 5, Pitfalls, item 2
    z = as_complex(np.array([1.0, 2.0, 3.0, 4.0]))
    assert_close(z.real, [1.0, 3.0], dtype="float64")
    assert_close(z.imag, [2.0, 4.0], dtype="float64")
    assert_close(
        as_real(np.array([1 + 2j, 3 + 4j])), [1.0, 2.0, 3.0, 4.0], dtype="float64"
    )
    got = rotate_pairs(np.array([1.0, 2.0, 3.0, 4.0]), np.array([math.pi, 0.0]))
    assert_close(got, [-1.0, -2.0, 3.0, 4.0], rtol=1e-12, atol=1e-12)


def test_as_real_roundtrip():
    # WHY: as_real undoes as_complex exactly (no arithmetic happens), and keeps
    #      the precision: float32 <-> complex64, float64 <-> complex128.
    # KIND: property
    # CATCHES: s05, s10
    # CHAPTER: M00.2 section 4, The interface
    rng = PCG32(seed=seed())
    x = rand(rng, (2, 3, 8))
    assert np.array_equal(as_real(as_complex(x)), x)
    x32 = x.astype(np.float32)
    z32 = as_complex(x32)
    assert z32.dtype == np.complex64
    back = as_real(z32)
    assert back.dtype == np.float32 and np.array_equal(back, x32)


def test_golden_torch_polar():
    # WHY: the oracle is torch's complex arithmetic, the way Meta's Llama code
    #      applies RoPE: view_as_complex(x) * polar(1, theta). Three cases,
    #      one in float32 (course/oracle/M00.2/rotation_golden.py).
    # KIND: golden
    # CATCHES: s01, s02, s03, s07, s08, m01
    # CHAPTER: M00.2 section 2.6, Complex numbers and Euler's formula
    for case in json.loads(GOLDEN.read_text())["cases"]:
        dt = np.dtype(case["dtype"])
        x = np.array(case["x"], dtype=dt)
        theta = np.array(case["theta"], dtype=dt)
        got = rotate_pairs(x, theta)
        assert got.dtype == dt, case["name"]
        assert_close(got, np.array(case["out"]), dtype=case["dtype"], msg=case["name"])


# --- shapes, dtypes, and errors ----------------------------------------------------


def test_theta_broadcasts():
    # WHY: L7.3 passes one row of angles per position ([T, k]) and L5.4 one
    #      angle per pair shared by a batch ([k]); both are broadcasting.
    # KIND: unit
    # CATCHES: s04
    # CHAPTER: M00.2 section 4, The interface
    rng = PCG32(seed=seed())
    x = rand(rng, (2, 3, 4))
    theta = rand(rng, (2,), -3.0, 3.0)
    got = rotate_pairs(x, theta)
    assert got.shape == (2, 3, 4)
    for b in range(2):
        for t in range(3):
            assert_close(got[b, t], rotate_pairs(x[b, t], theta), dtype="float64")


def test_dtype_follows_input():
    # WHY: float32 activations stay float32 (L7.3 rotates float32 queries),
    #      everything else is float64; integers are promoted, not truncated.
    # KIND: boundary
    # CATCHES: s01, s03, s07, s08, m01
    # CHAPTER: M00.2 section 4, The interface
    x32 = np.array([3.0, 4.0], dtype=np.float32)
    assert rotate_pairs(x32, np.array([T0])).dtype == np.float32
    got = rotate_pairs(np.array([3, 4]), np.array([T0]))
    assert got.dtype == np.float64
    assert_close(got, [-1.4, 4.8], dtype="float64")


def test_input_not_modified():
    # WHY: callers reuse x (the same keys are rotated for every query in
    #      L7.3); an in-place rotation corrupts the next use.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: M00.2 section 5, Pitfalls, item 4
    x = np.array(HAND_X)
    keep = x.copy()
    rotate_pairs(x, np.array(HAND_THETA))
    assert np.array_equal(x, keep)


def test_odd_or_bad_shapes_rejected():
    # WHY: a vector of odd length has no last pair; dropping the leftover
    #      element silently changes the model. Mismatched theta is an error
    #      too, not a broadcast you did not intend.
    # KIND: boundary
    # CATCHES: s09
    # CHAPTER: M00.2 section 4, The interface
    with pytest.raises(ValueError):
        rotate_pairs(np.zeros(5), np.zeros(2))
    with pytest.raises(ValueError):
        rotate_pairs(np.zeros((2, 6)), np.zeros(4))
    with pytest.raises(ValueError):
        as_complex(np.zeros(3))
    with pytest.raises(ValueError):
        rotate_pairs(np.float64(1.0), np.zeros(1))
