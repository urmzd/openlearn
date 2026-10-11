"""Course tests for M01.3: activation functions and their derivatives
(tinyllm/num/activations.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M01.3), and the chapter section it comes from.

The worked example of the chapter (section 3) is x = 1: sigmoid 0.7310586,
dsigmoid 0.1966119, silu 0.7310586, dsilu 0.9276705, tanh 0.7615942,
dtanh 0.4199743, softplus 1.3132617, and the GELUs.

The golden values in course/fixtures/M01.3/activations_torch.npz were
computed by PyTorch autograd in float64 at 1000 points from -40 to 40
(course/oracle/M01.3/activations_torch.py).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32

import tinyllm.num.activations as act

SEED = int(os.environ.get("OL_SEED", "0"))
GOLDEN = (
    Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M01.3" / "activations_torch.npz"
)
NAMES = ["sigmoid", "tanh", "relu", "softplus", "gelu_tanh", "gelu_erf", "silu"]
SMOOTH = [n for n in NAMES if n != "relu"]


def fn(name: str):
    return getattr(act, name)


def golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "__meta__"}


# --- the worked example -------------------------------------------------------


def test_hand_example_at_one():
    # WHY: the chapter's worked example at x = 1, digit for digit:
    #      sigmoid(1) = 1 / (1 + e^-1) = 0.7310586, its derivative
    #      s (1 - s) = 0.1966119, silu = 1 * s = 0.7310586 and
    #      dsilu = s (1 + 1 * (1 - s)) = 0.9276705; tanh 0.7615942 with
    #      derivative 1 - tanh^2 = 0.4199743; softplus ln(1 + e) = 1.3132617.
    # KIND: unit
    # CATCHES: s01, s02, s08
    # CHAPTER: M01.3 section 3, Worked example by hand
    s = 1 / (1 + math.exp(-1))
    want = {
        "sigmoid": s,
        "dsigmoid": s * (1 - s),
        "silu": s,
        "dsilu": s * (2 - s),
        "tanh": math.tanh(1),
        "dtanh": 1 - math.tanh(1) ** 2,
        "softplus": math.log(1 + math.e),
        "dsoftplus": s,
        "relu": 1.0,
        "drelu": 1.0,
    }
    for name, w in want.items():
        assert_close(fn(name)(1.0), w, rtol=1e-15, atol=0, msg=name)
    assert_close(act.dsilu(1.0), 0.9276705, rtol=1e-7, atol=0)
    assert_close(act.softplus(1.0), 1.3132617, rtol=1e-7, atol=0)


def test_hand_example_gelus():
    # WHY: the two GELUs at x = 1. Exact: 1 * Phi(1) = 0.8413447; its
    #      derivative Phi(1) + phi(1) = 0.8413447 + 0.2419707 = 1.0833155.
    #      The tanh approximation: u = sqrt(2/pi) * 1.044715 = 0.8335620,
    #      0.5 (1 + tanh u) = 0.8411920. They differ in the 4th digit: two
    #      different functions, never interchangeable in a checkpoint.
    # KIND: unit
    # CATCHES: s05, s07, m03
    # CHAPTER: M01.3 section 3, Worked example by hand
    phi = 0.5 * (1 + math.erf(1 / math.sqrt(2)))
    pdf = math.exp(-0.5) / math.sqrt(2 * math.pi)
    assert_close(act.gelu_erf(1.0), phi, rtol=1e-14, atol=0)
    assert_close(act.dgelu_erf(1.0), phi + pdf, rtol=1e-14, atol=0)
    u = math.sqrt(2 / math.pi) * (1 + 0.044715)
    assert_close(act.gelu_tanh(1.0), 0.5 * (1 + math.tanh(u)), rtol=1e-15, atol=0)
    assert_close(act.gelu_erf(1.0), 0.8413447, rtol=1e-7, atol=0)
    assert_close(act.gelu_tanh(1.0), 0.8411920, rtol=1e-7, atol=0)


# --- against PyTorch ----------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_matches_torch_golden(name):
    # WHY: L0.2's op library is checked against PyTorch forward and backward
    #      (DESIGN `O`); these are the elementwise pieces, at 1000 points
    #      including 0 (relu's kink: torch's gradient there is 0) and +-40,
    #      where naive formulas overflow. Tolerance: the course float64 row.
    # KIND: golden
    # CATCHES: s04, s05, s06, s07, s08, m02, m06, m07
    # CHAPTER: M01.3 section 2, Principles
    g = golden()
    x = g["x"]
    assert_close(fn(name)(x), g[name], dtype="float64", msg=name)
    assert_close(fn("d" + name)(x), g["d" + name], dtype="float64", msg="d" + name)


# --- derivatives against central differences ----------------------------------


@pytest.mark.parametrize("name", SMOOTH)
def test_derivatives_pass_gradcheck(name):
    # WHY: each d<name> must be the derivative of <name>: the frozen
    #      gradcheck compares sum(f(x)) differentiated by central differences
    #      with d<name>(x) at seeded points (DESIGN `G`, float64). A derivative
    #      that matches torch only by coincidence of formulas still has to
    #      agree with the function you wrote.
    # KIND: gradcheck
    # CATCHES: s01, s02, s05, s06, m03
    # CHAPTER: M01.3 section 2, Principles
    rng = PCG32(seed=SEED)
    x = rng.uniform_array((40,), -6.0, 6.0)
    f, df = fn(name), fn("d" + name)
    gradcheck(lambda v: float(np.sum(f(v))), [x], [df(x)], names=[name])


def test_relu_gradcheck_away_from_zero():
    # WHY: relu has a kink at 0 where no derivative exists; away from it the
    #      derivative is 0 or 1 and central differences agree exactly.
    # KIND: gradcheck
    # CATCHES: m06
    # CHAPTER: M01.3 section 2, Principles
    x = np.array([-3.0, -0.5, -1e-3, 1e-3, 0.5, 3.0])
    gradcheck(lambda v: float(np.sum(act.relu(v))), [x], [act.drelu(x)])


# --- identities ---------------------------------------------------------------


def test_identities():
    # WHY: algebraic laws that hold for every x (section 2): sigmoid(x) +
    #      sigmoid(-x) = 1; tanh(x) = 2 sigmoid(2x) - 1; softplus(x) -
    #      softplus(-x) = x; silu(x) - silu(-x) = x; gelu(x) - gelu(-x) = x for
    #      both GELUs. Each catches a sign or a symmetry slip that a few
    #      spot values miss.
    # KIND: property
    # CATCHES: m07
    # CHAPTER: M01.3 section 2, Principles
    rng = PCG32(seed=SEED)
    x = rng.uniform_array((500,), -30.0, 30.0)
    assert_close(
        act.sigmoid(x) + act.sigmoid(-x), np.ones_like(x), rtol=0, atol=2.3e-16
    )
    assert_close(act.tanh(x), 2 * act.sigmoid(2 * x) - 1, rtol=0, atol=5e-16)
    for name in ("softplus", "silu", "gelu_tanh", "gelu_erf"):
        assert_close(fn(name)(x) - fn(name)(-x), x, rtol=1e-14, atol=1e-14, msg=name)


def test_derivative_tails_keep_relative_accuracy():
    # WHY: sigmoid'(30) = 9.36e-14, but s (1 - s) with s = 1 - 9.36e-14
    #      rounded keeps only 3 correct digits; s(x) s(-x) never subtracts
    #      from 1 and keeps all 16. Same for softplus(-40) = 4.25e-18 through
    #      log1p. Tiny gradients still feed products in the chain rule.
    # KIND: boundary
    # CATCHES: s03, s09
    # CHAPTER: M01.3 section 5, Pitfalls, item 2
    for x in (20.0, 30.0, 35.0):
        exact = math.exp(-x) / (1 + math.exp(-x)) ** 2
        assert_close(act.dsigmoid(x), exact, rtol=1e-14, atol=0)
        assert_close(act.dsigmoid(-x), exact, rtol=1e-14, atol=0)
    assert_close(act.softplus(-40.0), math.exp(-40.0), rtol=1e-14, atol=0)


# --- robustness ---------------------------------------------------------------


ALL = [n for name in NAMES for n in (name, "d" + name)]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_no_overflow_for_any_finite_input(dtype):
    # WHY: after a bad optimizer step a logit can be -1000 or 1e30. Every
    #      function must return a finite value with no overflow, invalid, or
    #      divide warning (raised here as errors): exp(1000) = inf in a naive
    #      sigmoid or softplus, x^3 in the GELU's tanh argument, x^2 in the
    #      normal density.
    # KIND: boundary
    # CATCHES: s10, s11, s12, m04
    # CHAPTER: M01.3 section 5, Pitfalls, item 1
    big = np.finfo(dtype).max
    x = np.array(
        [-big, -1e30, -1000, -100, -40, -1e-30, 0, 1e-30, 40, 100, 1000, 1e30, big],
        dtype=dtype,
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for name in ALL:
            y = fn(name)(x)
            assert np.isfinite(y).all(), (name, y)
    assert act.softplus(np.float64(1000.0)) == 1000.0
    assert (
        act.sigmoid(np.float64(-1000.0)) == 0.0
        and act.sigmoid(np.float64(1000.0)) == 1.0
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_and_shape_preserved(dtype):
    # WHY: L0.2 runs float32 and float64 tensors through these; a function
    #      that silently promotes float32 to float64 doubles memory and breaks
    #      the C kernel comparisons later. Integer input computes in float64.
    # KIND: unit
    # CATCHES: s13, m01, m05
    # CHAPTER: M01.3 section 4, The interface
    x = np.linspace(-3, 3, 12, dtype=dtype).reshape(3, 4)
    for name in ALL:
        y = fn(name)(x)
        assert y.dtype == dtype and y.shape == (3, 4), (name, y.dtype, y.shape)
        assert_close(
            y, fn(name)(x.astype(np.float64)), dtype=np.dtype(dtype).name, msg=name
        )
    yi = act.relu(np.array([-2, 0, 3]))
    assert yi.dtype == np.float64 and yi.tolist() == [0.0, 0.0, 3.0]


def test_relu_derivative_at_zero_is_zero():
    # WHY: the derivative of relu does not exist at 0; PyTorch (and so every
    #      checkpoint trained with it) uses 0 there. drelu(0) = 1 would make
    #      gradients differ from the reference exactly at the inputs that are 0.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: M01.3 section 5, Pitfalls, item 3
    assert act.drelu(np.array([0.0, -0.0])).tolist() == [0.0, 0.0]
    assert act.relu(np.array([-0.0, 0.0, 2.5])).tolist() == [0.0, 0.0, 2.5]
