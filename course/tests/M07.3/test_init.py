"""Course tests for M07.3: variance propagation and initialization
(tinyllm/nn/init.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M07.3), and the chapter section it comes from.

The worked example of the chapter (section 3) is a Linear weight of shape
(4, 3): 3 inputs, 4 outputs.

Normals come from your M07.0 `normal(rng, n)`; the tests recompute them from
the frozen PCG32 with Box-Muller (cosine first, the last sine dropped for odd
n), so a wrong M07.0 shows up here too.
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
from tinyllm.nn.init import (
    calculate_gain,
    fans,
    kaiming_normal,
    normal_init,
    scaled_residual_std,
    xavier_normal,
    xavier_uniform,
)


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def fixture() -> dict:
    root = Path(
        os.environ.get(
            "TINYLLM_FIXTURES", Path(__file__).resolve().parents[2] / "fixtures"
        )
    )
    return json.loads((root / "M07.3" / "init_torch.json").read_text())


def spec_normals(rng: PCG32, n: int) -> np.ndarray:
    """n standard normals in spec/pcg32.md pair order, as M07.0 defines them."""
    out = []
    while len(out) < n:
        u1, u2 = rng.uniform(), rng.uniform()
        r = math.sqrt(-2.0 * math.log(1.0 - u1))
        out += [r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)]
    return np.array(out[:n])


def check_variance(x: np.ndarray, var: float, mu4_over_var2: float, what: str) -> None:
    """Sample mean and variance within 4 standard errors of 0 and var (a
    correct initializer fails one check with probability about 6e-5).
    The standard error of a sample variance is sqrt((mu4 - var^2) / n)."""
    x = x.astype(np.float64).ravel()
    n = x.size
    se_mean = math.sqrt(var / n)
    se_var = math.sqrt((mu4_over_var2 - 1.0) * var * var / n)
    assert abs(x.mean()) <= 4 * se_mean, (
        f"{what}: mean {x.mean():.4g} is not 0 (4 se = {4 * se_mean:.3g})"
    )
    v = float(((x - x.mean()) ** 2).sum() / (n - 1))
    assert abs(v - var) <= 4 * se_var, (
        f"{what}: variance {v:.6g}, want {var:.6g} +- {4 * se_var:.3g}"
    )


# --- the worked example -------------------------------------------------------


def test_hand_example_linear_4x3():
    # WHY: the chapter's worked example. A Linear weight (4, 3) has fan_in 3
    #      and fan_out 4. Xavier: a = sqrt(6/7), std sqrt(2/7); Kaiming for
    #      ReLU: std sqrt(2)/sqrt(3) = sqrt(2/3); GPT-2's residual projections
    #      with 12 layers: 0.02 / sqrt(24).
    # KIND: unit
    # CATCHES: s01, s05, s08, s09, s10, m01, m03
    # CHAPTER: M07.3 section 3, Worked example by hand
    assert fans((4, 3)) == (3, 4)
    assert_close(calculate_gain("relu"), math.sqrt(2.0), dtype="float64")
    assert_close(scaled_residual_std(0.02, 12), 0.02 / math.sqrt(24.0), dtype="float64")
    rng, ref = PCG32(seed=seed()), PCG32(seed=seed())
    w = kaiming_normal((4, 3), "fan_in", "relu", rng)
    assert_close(
        w, math.sqrt(2.0 / 3.0) * spec_normals(ref, 12).reshape(4, 3), dtype="float32"
    )
    rng, ref = PCG32(seed=seed()), PCG32(seed=seed())
    w = xavier_uniform((4, 3), 1.0, rng)
    a = math.sqrt(6.0 / 7.0)
    u = np.array([ref.uniform() for _ in range(12)]).reshape(4, 3)
    assert_close(w, -a + 2 * a * u, dtype="float32")


def test_fans_and_gains_match_torch():
    # WHY: torch.nn.init's fan rule (receptive field for conv shapes) and its
    #      gain table, including leaky_relu's default slope 0.01. L0.4 copies
    #      torch weights in its oracle tests, so the conventions must agree.
    # KIND: golden
    # CATCHES: s01, s02, s05, s07, s12, m02
    # CHAPTER: M07.3 section 2.3, Fans and gains
    doc = fixture()
    for g in doc["gains"]:
        assert_close(
            calculate_gain(g["nonlinearity"], g["param"]),
            g["gain"],
            dtype="float64",
            msg=g["nonlinearity"],
        )
    for f in doc["fans"]:
        assert fans(tuple(f["shape"])) == (f["fan_in"], f["fan_out"]), f["shape"]


# --- draws -----------------------------------------------------------------------


def test_normal_inits_draw_spec_normals_in_c_order():
    # WHY: element k of every normal initializer is std times the k-th spec
    #      normal, in row-major order, whatever the shape (odd sizes drop the
    #      last sine). A seed must give the same weights in Python and in the
    #      Rust port (L10.1).
    # KIND: unit
    # CATCHES: s04, s06, s09, s10, s11
    # CHAPTER: M07.3 section 2.4, Drawing the weights
    for shape in ((3, 5), (2, 3, 2, 2)):
        n = math.prod(shape)
        fi, fo = fans(shape)
        for name, make, std in (
            ("normal_init", lambda r: normal_init(shape, 0.02, r), 0.02),
            (
                "xavier_normal",
                lambda r: xavier_normal(shape, 2.0, r),
                2.0 * math.sqrt(2.0 / (fi + fo)),
            ),
            (
                "kaiming_normal",
                lambda r: kaiming_normal(shape, "fan_out", "tanh", r),
                (5 / 3) / math.sqrt(fo),
            ),
        ):
            got = make(PCG32(seed=seed() + 7))
            want = std * spec_normals(PCG32(seed=seed() + 7), n).reshape(shape)
            assert got.shape == shape and got.dtype == np.float32, name
            assert_close(got, want, dtype="float32", msg=f"{name} {shape}")


def test_uniform_uses_one_draw_per_element():
    # WHY: xavier_uniform consumes exactly one rng.uniform() per element, in
    #      C order. L0.4 initializes layer after layer from one generator, so a
    #      layer that draws extra (or fewer) values shifts every later layer.
    # KIND: unit
    # CATCHES: s03, s09, s13
    # CHAPTER: M07.3 section 2.4, Drawing the weights
    rng, ref = PCG32(seed=seed()), PCG32(seed=seed())
    w = xavier_uniform((5, 7), 0.5, rng)
    a = 0.5 * math.sqrt(6.0 / 12.0)
    u = np.array([ref.uniform() for _ in range(35)])
    assert_close(w, (-a + 2 * a * u).reshape(5, 7), dtype="float32")
    assert rng.state == ref.state, (
        "the generator advanced by a different number of draws"
    )


def test_same_seed_same_weights():
    # WHY: initialization is a pure function of the generator's state: equal
    #      seeds give identical bytes, different seeds give different weights.
    #      MS-L0 resumes and replays runs bit for bit.
    # KIND: property
    # CATCHES: s13
    # CHAPTER: M07.3 section 2.4, Drawing the weights
    for make in (
        lambda r: xavier_uniform((16, 8), 1.0, r),
        lambda r: kaiming_normal((16, 8), "fan_in", "relu", r),
    ):
        a, b, c = make(PCG32(seed=5)), make(PCG32(seed=5)), make(PCG32(seed=6))
        assert a.tobytes() == b.tobytes()
        assert a.tobytes() != c.tobytes()


# --- variance -----------------------------------------------------------------


def test_empirical_variance_matches_the_formula():
    # WHY: over 32768 weights, the sample mean is 0 and the sample variance is
    #      the formula's within 4 standard errors: gain^2 * 2 / (fan_in +
    #      fan_out) for Xavier (uniform: mu4 = 9/5 var^2; normal: 3 var^2) and
    #      gain^2 / fan for Kaiming, with fan_out chosen when asked.
    # KIND: statistical
    # CATCHES: s03, s04, s05, s06, s10
    # CHAPTER: M07.3 section 2.2, Choosing the weight variance
    rng = PCG32(seed=seed() + 3)
    shape = (128, 256)
    fi, fo = 256, 128
    check_variance(
        xavier_uniform(shape, 1.5, rng), 1.5**2 * 2 / (fi + fo), 9 / 5, "xavier_uniform"
    )
    check_variance(
        xavier_normal(shape, 1.5, rng), 1.5**2 * 2 / (fi + fo), 3.0, "xavier_normal"
    )
    check_variance(
        kaiming_normal(shape, "fan_in", "relu", rng), 2.0 / fi, 3.0, "kaiming fan_in"
    )
    check_variance(
        kaiming_normal(shape, "fan_out", "relu", rng), 2.0 / fo, 3.0, "kaiming fan_out"
    )


def test_relu_signal_survives_20_layers():
    # WHY: the point of the module. Through 20 ReLU layers of width 128,
    #      Kaiming (fan_in, relu) keeps the mean square of the pre-activations
    #      where it started, up to the random walk of a finite width (a factor
    #      of 16 is 4 standard deviations of that walk); Xavier's variance,
    #      half of what ReLU needs, shrinks it by about 2^19.
    # KIND: property
    # CATCHES: s05, s10, m01
    # CHAPTER: M07.3 section 2.2, Choosing the weight variance
    rng = PCG32(seed=seed() + 4)
    x0 = rng.normal_array((256, 128))

    def run(make):
        h, ms = x0, []
        for _ in range(20):
            z = h @ make().astype(np.float64).T
            ms.append(float((z * z).mean()))
            h = np.maximum(z, 0.0)
        return ms

    kaiming = run(lambda: kaiming_normal((128, 128), "fan_in", "relu", rng))
    assert 1 / 16 < kaiming[-1] / kaiming[0] < 16, (
        f"Kaiming: mean square went {kaiming[0]:.3g} -> {kaiming[-1]:.3g}"
    )
    xavier = run(lambda: xavier_normal((128, 128), 1.0, rng))
    assert xavier[-1] / xavier[0] < 1e-4, (
        f"Xavier on ReLU: {xavier[0]:.3g} -> {xavier[-1]:.3g}"
    )


def test_tanh_signal_survives_with_xavier():
    # WHY: for a near-linear activation the right variance is 1 / fan
    #      (Xavier with equal fans). Through 20 tanh layers the mean square
    #      stays between 0.1 and 1 of its start with gain 5/3, instead of
    #      collapsing toward 0 as with gain 1.
    # KIND: property
    # CATCHES: s04, s07
    # CHAPTER: M07.3 section 2.3, Fans and gains
    rng = PCG32(seed=seed() + 5)
    x0 = rng.normal_array((256, 128))

    def run(gain):
        h = x0
        for _ in range(20):
            h = np.tanh(h @ xavier_normal((128, 128), gain, rng).astype(np.float64).T)
        return float((h * h).mean())

    good = run(calculate_gain("tanh"))
    assert 0.1 < good < 1.0, f"gain 5/3 ends at mean square {good:.3g}"
    assert run(1.0) < good / 2


def test_scaled_residual_keeps_the_stream_bounded():
    # WHY: a residual stream that adds 2 * n_layers outputs of std s grows to
    #      variance 2 n s^2. With s = base / sqrt(2 n) it ends at base^2 for
    #      every depth, which is why GPT-2 (and your L7.9) scale those
    #      projections.
    # KIND: property
    # CATCHES: s08, m03
    # CHAPTER: M07.3 section 2.5, Residual streams
    for base, n in ((0.02, 1), (0.02, 12), (1.0, 48)):
        s = scaled_residual_std(base, n)
        assert_close(2 * n * s * s, base * base, dtype="float64")


# --- shapes and arguments -------------------------------------------------------


def test_shapes_and_dtype():
    # WHY: every initializer returns a new float32 array of exactly the shape
    #      asked; normal_init also takes 1-D shapes (biases, a position table
    #      row) where fans are undefined.
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: M07.3 section 4, The interface
    rng = PCG32(seed=seed())
    for w in (
        xavier_uniform((3, 2), 1.0, rng),
        xavier_normal((3, 2), 1.0, rng),
        kaiming_normal((3, 2, 2), "fan_in", "linear", rng),
        normal_init((7,), 0.5, rng),
        normal_init((2, 3, 4), 0.5, rng),
    ):
        assert w.dtype == np.float32 and w.flags.c_contiguous
    assert normal_init((7,), 0.5, rng).shape == (7,)
    assert kaiming_normal((3, 2, 2), "fan_in", "linear", rng).shape == (3, 2, 2)
    assert normal_init((4,), 0.0, rng).tolist() == [0.0] * 4


def test_rejects_bad_arguments():
    # WHY: a 1-D shape has no fan_in, an unknown nonlinearity has no gain, and
    #      a negative std or depth is a configuration error. Guessing a default
    #      trains a model at the wrong scale with no error.
    # KIND: boundary
    # CATCHES: s14, m04
    # CHAPTER: M07.3 section 4, The interface
    rng = PCG32(seed=seed())
    for bad in (
        lambda: fans((5,)),
        lambda: fans((0, 3)),
        lambda: xavier_uniform((5,), 1.0, rng),
        lambda: xavier_uniform((2, 2), -1.0, rng),
        lambda: kaiming_normal((4, 4), "fan_avg", "relu", rng),
        lambda: kaiming_normal((4, 4), "fan_in", "gelu", rng),
        lambda: calculate_gain("swish"),
        lambda: calculate_gain("leaky_relu", "0.2"),
        lambda: normal_init((3,), -0.1, rng),
        lambda: scaled_residual_std(0.02, 0),
    ):
        with pytest.raises(ValueError):
            bad()
