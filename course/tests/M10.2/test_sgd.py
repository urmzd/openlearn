"""Course tests for M10.2: the Optimizer protocol and SGD (tinyllm/optim/sgd.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M10.2), and the chapter section it comes from.

The worked example of the chapter (section 3) is f(x) = x^2 / 2 (so the
gradient is x) from x = 1 with lr = 0.1: momentum 0.9 gives 0.9, 0.72,
0.486; Nesterov gives 0.81, 0.5751; weight decay 0.1 without momentum
gives 0.89.

One test calls M10.1's gradient_descent through the overlay.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.optim.sgd import SGD, Optimizer, Param

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M10.2" / "torch_sgd.npz"


class P:
    """The smallest parameter: data and grad, nothing else."""

    def __init__(self, data, grad=None):
        self.data = np.array(data, dtype=np.float64)
        self.grad = grad


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def run_on_half_square(opt_kwargs, steps: int, x0: float = 1.0) -> list[float]:
    """Minimize x^2 / 2 (gradient x) and return the iterates after each step."""
    p = P([x0])
    opt = SGD([p], **opt_kwargs)
    out = []
    for _ in range(steps):
        opt.zero_grad()
        p.grad = p.data.copy()
        opt.step()
        out.append(float(p.data[0]))
    return out


# --- the worked example --------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example on f(x) = x^2/2 from x = 1, lr 0.1.
    #      Momentum 0.9: buf = 1, 0.9 + 0.9 = 1.8, 1.62 + 0.72 = 2.34, so
    #      x = 0.9, 0.72, 0.486. Nesterov steps along g + 0.9 buf: 1.9, then
    #      0.81 + 0.9 * 1.71 = 2.349, so x = 0.81, 0.5751. Weight decay 0.1
    #      adds 0.1 x to the gradient: x = 1 - 0.1 * 1.1 = 0.89.
    # KIND: unit
    # CATCHES: s01, s02, s03, s12
    # CHAPTER: M10.2 section 3, Worked example by hand
    assert_close(run_on_half_square(dict(lr=0.1, momentum=0.9), 3), [0.9, 0.72, 0.486])
    assert_close(
        run_on_half_square(dict(lr=0.1, momentum=0.9, nesterov=True), 2), [0.81, 0.5751]
    )
    assert_close(run_on_half_square(dict(lr=0.1, weight_decay=0.1), 1), [0.89])


def test_lr_change_takes_effect_on_the_next_step():
    # WHY: schedules (M10.4) set opt.lr before every step. The buffer holds
    #      gradients and lr multiplies it when the step is taken, so a new
    #      lr scales the whole next step: buf = 0.9 + 0.9 = 1.8 and
    #      x = 0.9 - 0.01 * 1.8 = 0.882. A buffer that absorbed lr (u = 0.1,
    #      then 0.09 + 0.009) matches PyTorch only while lr is constant and
    #      lands on 0.801 here.
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: M10.2 section 3, Worked example by hand (changing the learning rate)
    p = P([1.0])
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = p.data.copy()
    opt.step()
    opt.lr = 0.01
    p.grad = p.data.copy()
    opt.step()
    assert_close(p.data, [0.882])
    assert_close(opt.state_dict()["state"][0]["momentum_buffer"], [1.8])


def test_matches_torch_golden():
    # WHY: 20-step trajectories of torch.optim.SGD on a two-parameter problem
    #      for five settings (plain, momentum, Nesterov, momentum with weight
    #      decay, weight decay alone), recorded in float64 by
    #      course/oracle/M10.2/torch_sgd_golden.py. Your SGD is the one
    #      PyTorch users expect, so recipes transfer.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s10, s12, m02
    # CHAPTER: M10.2 section 2, Principles
    d = np.load(GOLDEN, allow_pickle=False)
    H, b, C, D = d["H"], d["b"], d["C"], d["D"]
    configs = {
        "plain": dict(lr=0.05),
        "momentum": dict(lr=0.05, momentum=0.9),
        "nesterov": dict(lr=0.05, momentum=0.9, nesterov=True),
        "momentum_wd": dict(lr=0.05, momentum=0.8, weight_decay=0.01),
        "wd": dict(lr=0.1, weight_decay=0.1),
    }
    for name, kw in configs.items():
        p1, p2 = P(d["p1_0"]), P(d["p2_0"])
        opt = SGD([p1, p2], **kw)
        for t in range(20):
            opt.zero_grad()
            p1.grad = H @ p1.data - b
            p2.grad = C * (p2.data - D)
            opt.step()
            assert_close(p1.data, d[f"{name}/p1"][t], msg=f"{name} p1 step {t}")
            assert_close(p2.data, d[f"{name}/p2"][t], msg=f"{name} p2 step {t}")


def test_plain_sgd_equals_gradient_descent():
    # WHY: with momentum 0 and no weight decay, SGD with the full gradient is
    #      M10.1's gradient descent: the protocol is a stateful, in-place
    #      wrapper around the same update. Bit for bit on a quadratic.
    # KIND: differential
    # CATCHES: m02
    # CHAPTER: M10.2 section 2, Principles (from gradient descent to SGD)
    from tinyllm.optim.gd import gradient_descent

    rng = PCG32(seed=seed())
    A = rng.normal_array((4, 4))
    H = A @ A.T + np.eye(4)

    def f(x):
        return 0.5 * float(x @ H @ x)

    def g(x):
        return H @ x

    x0 = rng.normal_array(4)
    xs = gradient_descent(f, g, x0, lr=0.05, steps=15)
    p = P(x0)
    opt = SGD([p], lr=0.05)
    for t in range(15):
        p.grad = g(p.data)
        opt.step()
        assert np.array_equal(p.data, xs[t + 1]), f"step {t + 1}"


def test_updates_in_place():
    # WHY: the model holds references to its parameter arrays (layers, tied
    #      embeddings, the checkpoint writer). step must change those arrays,
    #      not rebind p.data to a new one, and keep their dtype: a float32
    #      parameter stays float32.
    # KIND: unit
    # CATCHES: s05
    # CHAPTER: M10.2 section 5, Pitfalls, item 3
    p = P([1.0, 2.0])
    alias = p.data
    opt = SGD([p], lr=0.5, momentum=0.5)
    p.grad = np.array([1.0, 1.0])
    opt.step()
    assert p.data is alias and alias.tolist() == [0.5, 1.5]
    q = P([1.0])
    q.data = q.data.astype(np.float32)
    q.grad = np.array([0.25], dtype=np.float32)
    SGD([q], lr=1.0, weight_decay=0.5).step()
    assert q.data.dtype == np.float32 and q.data.tolist() == [0.25]


def test_none_grad_is_skipped_and_zero_grad_clears():
    # WHY: a parameter with no gradient this step (an unused embedding row
    #      block, a frozen layer) must not move, not even by weight decay,
    #      and its momentum buffer must not advance. zero_grad sets grads to
    #      None, so the next backward assigns instead of accumulating into
    #      stale values.
    # KIND: boundary
    # CATCHES: s06, s07
    # CHAPTER: M10.2 section 4, The interface
    a, b = P([1.0]), P([1.0])
    opt = SGD([a, b], lr=0.1, momentum=0.9, weight_decay=0.5)
    a.grad = np.array([1.0])
    opt.step()
    assert b.data.tolist() == [1.0]
    assert 1 not in opt.state_dict()["state"]
    opt.zero_grad()
    assert a.grad is None and b.grad is None
    opt.step()  # every grad is None: nothing moves
    assert_close(a.data, [1.0 - 0.1 * 1.5])
    assert b.data.tolist() == [1.0]


def test_state_dict_resume_bitwise():
    # WHY: a training run that stops and resumes (L0.5 checkpoints, C1 on a
    #      laptop) must continue exactly as if it never stopped: 10 steps,
    #      save, a fresh optimizer loads the state, 10 more steps, bit for
    #      bit equal to 20 uninterrupted steps.
    # KIND: property
    # CATCHES: s09, m04
    # CHAPTER: M10.2 section 5, Pitfalls, item 4
    rng = PCG32(seed=seed())
    W0 = rng.normal_array((3, 2))
    target = rng.normal_array((3, 2))

    def train(opt, p, steps):
        for _ in range(steps):
            opt.zero_grad()
            p.grad = 2.0 * (p.data - target)
            opt.step()

    kw = dict(lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-3)
    p = P(W0)
    straight = SGD([p], **kw)
    train(straight, p, 20)

    q = P(W0)
    first = SGD([q], **kw)
    train(first, q, 10)
    saved = first.state_dict()
    second = SGD([q], lr=999.0)
    second.load_state_dict(saved)
    train(second, q, 10)
    assert np.array_equal(q.data, p.data)


def test_state_dict_is_a_copy():
    # WHY: state_dict feeds a checkpoint writer; if it hands out the live
    #      buffers, the next step silently changes the "saved" state.
    #      load_state_dict copies in for the same reason.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: M10.2 section 4, The interface
    p = P([1.0, 1.0])
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = np.array([1.0, 2.0])
    opt.step()
    sd = opt.state_dict()
    before = sd["state"][0]["momentum_buffer"].copy()
    opt.step()
    assert np.array_equal(sd["state"][0]["momentum_buffer"], before)
    assert sd["param_groups"][0]["params"] == [0]
    other = SGD([P([0.0, 0.0])], lr=0.1, momentum=0.9)
    other.load_state_dict(sd)
    sd["state"][0]["momentum_buffer"][:] = 123.0
    assert other.state_dict()["state"][0]["momentum_buffer"].tolist() == before.tolist()
    with pytest.raises(ValueError):
        SGD([P([0.0]), P([0.0])], lr=0.1).load_state_dict(sd)


def test_rejects_bad_hyperparameters():
    # WHY: PyTorch's rules: lr, momentum, and weight decay are nonnegative,
    #      and Nesterov needs momentum (with momentum 0 it would silently be
    #      plain SGD).
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: M10.2 section 4, The interface
    for kw in (
        dict(lr=-0.1),
        dict(lr=0.1, momentum=-0.5),
        dict(lr=0.1, weight_decay=-1.0),
        dict(lr=0.1, nesterov=True),
    ):
        with pytest.raises(ValueError):
            SGD([P([1.0])], **kw)
    SGD([P([1.0])], lr=0.0)


def test_momentum_beats_plain_on_ill_conditioned():
    # WHY: the reason momentum exists: on a quadratic with kappa = 100, plain
    #      gradient descent (lr = 1/L) shrinks the slow direction by 1 - 1/kappa
    #      = 0.99 per step, while heavy-ball momentum with
    #      lr = 4 / (sqrt(L) + sqrt(mu))^2 and beta = ((sqrt(kappa) - 1) /
    #      (sqrt(kappa) + 1))^2 shrinks it by about 0.82. After 150 steps the
    #      gap differs by orders of magnitude.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: M10.2 section 2, Principles (momentum)
    L, mu = 100.0, 1.0
    H = np.diag([mu, L])

    def gap_after(**kw):
        p = P([1.0, 1.0])
        opt = SGD([p], **kw)
        for _ in range(150):
            p.grad = H @ p.data
            opt.step()
        return 0.5 * float(p.data @ H @ p.data)

    plain = gap_after(lr=1 / L)
    k = L / mu
    heavy = gap_after(
        lr=4 / (L**0.5 + mu**0.5) ** 2, momentum=((k**0.5 - 1) / (k**0.5 + 1)) ** 2
    )
    assert plain > 0.02
    assert heavy < 1e-8


def test_weight_decay_shrinks_toward_zero():
    # WHY: with a zero gradient, coupled weight decay alone multiplies the
    #      parameter by (1 - lr wd) per step: p_t = (1 - lr wd)^t p_0. This is
    #      the L2 penalty wd p^2 / 2 that M10.3 contrasts with AdamW's
    #      decoupled decay.
    # KIND: property
    # CATCHES: s12, m02
    # CHAPTER: M10.2 section 2, Principles (weight decay)
    p = P([2.0, -4.0])
    opt = SGD([p], lr=0.1, weight_decay=0.5)
    for _ in range(10):
        p.grad = np.zeros(2)
        opt.step()
    assert_close(p.data, np.array([2.0, -4.0]) * 0.95**10)


def test_buffers_are_per_parameter():
    # WHY: each parameter has its own momentum buffer. One shared buffer
    #      leaks the velocity of one tensor into another; with gradients of
    #      different shapes it does not even broadcast.
    # KIND: unit
    # CATCHES: s10, m03
    # CHAPTER: M10.2 section 5, Pitfalls, item 2
    a, b = P([0.0]), P([0.0])
    opt = SGD([a, b], lr=1.0, momentum=0.5)
    for _ in range(2):
        a.grad, b.grad = np.array([1.0]), np.array([-1.0])
        opt.step()
    assert_close(a.data, [-2.5])  # buffers 1, 1.5
    assert_close(b.data, [2.5])
    sd = opt.state_dict()["state"]
    assert sorted(sd) == [0, 1]
    assert_close(sd[0]["momentum_buffer"], [1.5])
    assert_close(sd[1]["momentum_buffer"], [-1.5])


def test_sgd_is_an_optimizer():
    # WHY: the training loops of L0.5, L2.2, and L3.6 accept any Optimizer;
    #      SGD must satisfy the runtime-checkable protocol, and the course's
    #      minimal parameter satisfies Param.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: M10.2 section 4, The interface
    opt = SGD([P([1.0])], lr=0.1)
    assert isinstance(opt, Optimizer)
    assert isinstance(P([1.0]), Param)
    p = P([1.0], grad=np.array([3.0]))
    SGD([p], lr=0.1).zero_grad()
    assert p.grad is None
