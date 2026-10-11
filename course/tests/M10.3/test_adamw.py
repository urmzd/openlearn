"""Course tests for M10.3: Adam and AdamW (tinyllm/optim/adamw.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M10.3), and the chapter section it comes from.

The worked example of the chapter (section 3) is one scalar parameter x = 1,
gradients 2 then -1, lr = 0.1, betas (0.9, 0.999), eps = 0.
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
from tinyllm.optim.adamw import Adam, AdamW


class P:
    """A parameter as the optimizer sees it: data (updated in place) and grad."""

    def __init__(self, data, grad=None):
        self.data = data
        self.grad = grad


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def fixture() -> dict:
    root = Path(
        os.environ.get(
            "TINYLLM_FIXTURES", Path(__file__).resolve().parents[2] / "fixtures"
        )
    )
    return json.loads((root / "M10.3" / "adam_torch.json").read_text())


def replay(case: dict, steps: int | None = None):
    """Run the learner's optimizer on a golden case's gradients; return the
    parameters after every step and the optimizer."""
    dt = np.dtype(case["dtype"])
    ps = [P(np.array(x, dtype=dt)) for x in case["init"]]
    cls = AdamW if case["optimizer"] == "AdamW" else Adam
    opt = cls(
        ps,
        lr=case["lr"][0],
        betas=tuple(case["betas"]),
        eps=case["eps"],
        weight_decay=case["weight_decay"],
    )
    out = []
    for t in range(steps or len(case["grads"])):
        opt.lr = case["lr"][t]
        for p, g in zip(ps, case["grads"][t]):
            p.grad = np.array(g, dtype=dt)
        opt.step()
        out.append([p.data.copy() for p in ps])
    return out, opt


def check_case(name: str) -> None:
    case = next(c for c in fixture()["cases"] if c["name"] == name)
    traj, opt = replay(case)
    for t, (got, want) in enumerate(zip(traj, case["trajectory"])):
        for i, (g, w) in enumerate(zip(got, want)):
            assert g.dtype == np.dtype(case["dtype"]), (
                f"parameter {i} changed dtype to {g.dtype}"
            )
            assert_close(
                g,
                np.array(w),
                dtype=case["dtype"],
                msg=f"{name}: step {t + 1}, parameter {i}",
            )
    sd = opt.state_dict()
    for i in range(len(case["init"])):
        assert_close(
            sd["exp_avg"][i], np.array(case["final_exp_avg"][i]), dtype=case["dtype"]
        )
        assert_close(
            sd["exp_avg_sq"][i],
            np.array(case["final_exp_avg_sq"][i]),
            dtype=case["dtype"],
        )


def scalar_run(cls, grads, x0=1.0, **kw) -> list[float]:
    p = P(np.array([x0]))
    opt = cls([p], **kw)
    xs = []
    for g in grads:
        p.grad = np.array([g])
        opt.step()
        xs.append(float(p.data[0]))
    return xs


# --- the worked example -------------------------------------------------------


def test_hand_example_two_steps():
    # WHY: the chapter's worked example, number for number. Step 1: m = 0.2,
    #      v = 0.004, and bias correction turns them into exactly 2 and 4, so
    #      x moves by lr * 2 / sqrt(4) = 0.1. Step 2: the gradient flips to -1
    #      but m is still positive (0.08), so x keeps going down, by
    #      0.1 * (8/19) / sqrt(4996/1999).
    # KIND: unit
    # CATCHES: s01, s02, s03, s09, s10, m01, m02, m03, m04
    # CHAPTER: M10.3 section 3, Worked example by hand
    xs = scalar_run(Adam, [2.0, -1.0], lr=0.1, betas=(0.9, 0.999), eps=0.0)
    assert_close(xs[0], 0.9, dtype="float64")
    assert_close(xs[1], 0.9 - 0.1 * (8 / 19) / math.sqrt(4996 / 1999), dtype="float64")


def test_hand_example_adamw_and_adam_l2_differ():
    # WHY: the same two steps with weight_decay = 0.1. AdamW shrinks x by
    #      (1 - lr * wd) = 0.99 before each update: 0.99 - 0.1 = 0.89, then
    #      0.89 * 0.99 - 0.0266337... Adam adds wd * x to the gradient instead
    #      (2.1, then -0.91), and the normalization washes most of it out.
    # KIND: unit
    # CATCHES: s04, s05, s06, s11, s12
    # CHAPTER: M10.3 section 3, Worked example by hand
    upd2 = 0.1 * (8 / 19) / math.sqrt(4996 / 1999)
    xw = scalar_run(
        AdamW, [2.0, -1.0], lr=0.1, betas=(0.9, 0.999), eps=0.0, weight_decay=0.1
    )
    assert_close(xw, [0.89, 0.89 * 0.99 - upd2], dtype="float64")
    xa = scalar_run(
        Adam, [2.0, -1.0], lr=0.1, betas=(0.9, 0.999), eps=0.0, weight_decay=0.1
    )
    g1, g2 = 2.0 + 0.1 * 1.0, -1.0 + 0.1 * 0.9
    m2 = 0.9 * 0.1 * g1 + 0.1 * g2
    v2 = 0.999 * 0.001 * g1 * g1 + 0.001 * g2 * g2
    assert_close(
        xa,
        [0.9, 0.9 - 0.1 * (m2 / 0.19) / math.sqrt(v2 / (1 - 0.999**2))],
        dtype="float64",
    )


# --- torch trajectories ------------------------------------------------------------


def test_matches_torch_adamw_trajectory():
    # WHY: 20 steps of torch.optim.AdamW (single-tensor path, float64) on two
    #      parameters of shapes (3,) and (2, 3), and the final moments. Every
    #      later training run (L0.5, C1) assumes your AdamW is torch's AdamW.
    # KIND: golden
    # CATCHES: s01, s02, s03, s05, s06, s08, s09, s10, s11, s14, m01, m02, m03, m04
    # CHAPTER: M10.3 section 2, Principles
    check_case("adamw")


def test_matches_torch_adam_l2_trajectory():
    # WHY: torch.optim.Adam with weight_decay = 0.1: the penalty is added to the
    #      gradient before both moments see it (coupled L2).
    # KIND: golden
    # CATCHES: s04, s08, s12, s14
    # CHAPTER: M10.3 section 2.4, Decoupled weight decay
    check_case("adam_l2")


def test_lr_set_between_steps_is_used():
    # WHY: a schedule (M10.4) writes opt.lr before every step. The golden case
    #      changes lr on every one of 20 steps (and uses betas (0.9, 0.95), the
    #      C1 setting); an optimizer that cached lr at construction drifts away.
    # KIND: golden
    # CATCHES: s13
    # CHAPTER: M10.3 section 4, The interface
    check_case("adamw_lr_per_step")


def test_eps_is_added_after_bias_correction():
    # WHY: with gradients of size 1e-6 and eps = 1e-6, sqrt(v_hat) and eps are
    #      the same size, so where eps goes changes every step. torch adds it to
    #      sqrt(v) / sqrt(1 - beta2^t), after the correction.
    # KIND: golden
    # CATCHES: s08, s14
    # CHAPTER: M10.3 section 5, Pitfalls
    check_case("adamw_tiny_grads")


def test_float32_parameters_stay_float32():
    # WHY: models train in float32 (L0.4 parameters). The moments live in the
    #      parameter's dtype, the update happens in place, and the trajectory
    #      matches torch's float32 AdamW within float32 tolerance.
    # KIND: golden
    # CATCHES: s15
    # CHAPTER: M10.3 section 4, The interface
    check_case("adamw_float32")
    case = next(c for c in fixture()["cases"] if c["name"] == "adamw_float32")
    _, opt = replay(case, steps=2)
    sd = opt.state_dict()
    assert all(m.dtype == np.float32 for m in sd["exp_avg"] + sd["exp_avg_sq"])


# --- properties ------------------------------------------------------------------


def test_first_step_moves_each_coordinate_by_lr():
    # WHY: bias correction makes the first update exactly lr * g / |g| per
    #      coordinate when eps = 0, whatever the size of g (m_hat = g,
    #      v_hat = g^2). Without it the first step is lr * (1 - b1) / sqrt(1 - b2),
    #      3.16 times too large with the defaults.
    # KIND: property
    # CATCHES: s01, s02, s03, s09, m03
    # CHAPTER: M10.3 section 2.3, Bias correction
    rng = PCG32(seed=seed())
    for scale in (1e-4, 1.0, 1e3):
        g = scale * rng.normal_array((4, 5))
        g[g == 0] = 1.0
        p = P(rng.normal_array((4, 5)))
        x0 = p.data.copy()
        opt = Adam([p], lr=0.01, betas=(0.9, 0.999), eps=0.0)
        p.grad = g
        opt.step()
        assert_close(p.data, x0 - 0.01 * np.sign(g), dtype="float64")


def test_update_ignores_gradient_scale():
    # WHY: Adam divides m by sqrt(v), so multiplying every gradient by c > 0
    #      leaves the whole trajectory unchanged (eps = 0). This is why one lr
    #      works across layers whose gradients differ by orders of magnitude.
    # KIND: property
    # CATCHES: s09, s10
    # CHAPTER: M10.3 section 2.2, Normalizing by the second moment
    rng = PCG32(seed=seed() + 1)
    grads = [rng.normal_array((6,)) for _ in range(8)]
    x0 = rng.normal_array((6,))
    runs = []
    for c in (1.0, 1e-3, 250.0):
        p = P(x0.copy())
        opt = AdamW([p], lr=0.05, betas=(0.8, 0.99), eps=0.0, weight_decay=0.0)
        for g in grads:
            p.grad = c * g
            opt.step()
        runs.append(p.data)
    assert_close(runs[1], runs[0], rtol=1e-9, atol=1e-12)
    assert_close(runs[2], runs[0], rtol=1e-9, atol=1e-12)


def test_decoupled_decay_with_zero_gradient():
    # WHY: with g = 0 AdamW's moments stay 0 and only the decay acts:
    #      x = 4 * (1 - 0.1 * 0.5) = 3.8. Adam's L2 turns the penalty into a
    #      gradient (2.0) and normalizes it away: 4 - 0.1 = 3.9. Decoupling
    #      is the point of AdamW.
    # KIND: unit
    # CATCHES: s05, s06
    # CHAPTER: M10.3 section 2.4, Decoupled weight decay
    assert_close(
        scalar_run(AdamW, [0.0], x0=4.0, lr=0.1, weight_decay=0.5),
        [3.8],
        dtype="float64",
    )
    assert_close(
        scalar_run(Adam, [0.0], x0=4.0, lr=0.1, eps=0.0, weight_decay=0.5),
        [3.9],
        dtype="float64",
    )


def test_parameter_without_grad_is_untouched():
    # WHY: a parameter the loss did not reach has grad None (L0.1). It is not
    #      decayed and its moments do not move; the step counter still
    #      advances, so a later grad is corrected for the global t.
    # KIND: boundary
    # CATCHES: s07, s17
    # CHAPTER: M10.3 section 5, Pitfalls
    a, b = P(np.array([1.0, 2.0])), P(np.array([3.0]))
    opt = AdamW([a, b], lr=0.1, weight_decay=0.5)
    a.grad = np.array([1.0, -1.0])
    opt.step()
    assert b.data.tolist() == [3.0]
    sd = opt.state_dict()
    assert sd["step"] == 1
    assert sd["exp_avg"][1].tolist() == [0.0] and sd["exp_avg_sq"][1].tolist() == [0.0]


def test_update_is_in_place():
    # WHY: the model holds its parameter arrays (L0.4 state_dict names them,
    #      L11 views them inside flat buffers). Rebinding p.data to a new array
    #      leaves every other holder with stale weights.
    # KIND: unit
    # CATCHES: s16
    # CHAPTER: M10.3 section 4, The interface
    buf = np.arange(6, dtype=np.float32)
    view = buf[2:5]
    p = P(view, grad=np.ones(3, dtype=np.float32))
    opt = AdamW([p], lr=0.5, weight_decay=0.0)
    opt.step()
    assert p.data is view
    assert buf[2:5].tolist() == p.data.tolist() and buf[2] != 2.0
    assert buf[0] == 0.0 and buf[5] == 5.0


def test_zero_grad_sets_none():
    # WHY: L0.1 accumulates into grad, so the train loop clears it every step.
    #      None (not zeros) is what keeps an unused parameter from being decayed
    #      on the next step.
    # KIND: unit
    # CATCHES: s18
    # CHAPTER: M10.3 section 4, The interface
    ps = [P(np.ones(2), grad=np.ones(2)), P(np.ones(3), grad=np.ones(3))]
    opt = AdamW(ps)
    opt.zero_grad()
    assert all(p.grad is None for p in ps)


# --- state_dict ---------------------------------------------------------------


def test_resume_is_bitwise():
    # WHY: 5 steps, save, a fresh optimizer loads the state, 5 more steps,
    #      equals 10 uninterrupted steps bit for bit. dur.11 kills training
    #      workers and resumes from checkpoints; any drift there is a bug.
    # KIND: property
    # CATCHES: s13, s20, s22
    # CHAPTER: M10.3 section 2.5, Saving and restoring the state
    rng = PCG32(seed=seed() + 2)
    x0 = [rng.normal_array((3,)), rng.normal_array((2, 2))]
    grads = [[rng.normal_array((3,)), rng.normal_array((2, 2))] for _ in range(10)]

    def run(ps, opt, gs):
        for g in gs:
            for p, gi in zip(ps, g):
                p.grad = gi
            opt.step()

    full = [P(x.copy()) for x in x0]
    run(full, AdamW(full, lr=0.03, weight_decay=0.1), grads)

    first = [P(x.copy()) for x in x0]
    opt1 = AdamW(first, lr=0.03, weight_decay=0.1)
    run(first, opt1, grads[:5])
    sd = opt1.state_dict()
    second = [P(p.data.copy()) for p in first]
    opt2 = AdamW(second, lr=1.0, betas=(0.5, 0.5), eps=1.0, weight_decay=0.0)
    opt2.load_state_dict(sd)
    run(second, opt2, grads[5:])
    for a, b in zip(second, full):
        assert a.data.tobytes() == b.data.tobytes()


def test_state_dict_is_a_snapshot():
    # WHY: a checkpoint writer (L0.6) may serialize the dict after training
    #      continues. Returning the live moment arrays makes the saved state
    #      the state of a later step.
    # KIND: unit
    # CATCHES: s19
    # CHAPTER: M10.3 section 2.5, Saving and restoring the state
    p = P(np.array([1.0, -1.0]), grad=np.array([0.5, 0.25]))
    opt = AdamW([p], lr=0.1)
    opt.step()
    sd = opt.state_dict()
    saved = [sd["exp_avg"][0].copy(), sd["exp_avg_sq"][0].copy()]
    p.grad = np.array([4.0, -4.0])
    opt.step()
    assert sd["step"] == 1
    assert sd["exp_avg"][0].tolist() == saved[0].tolist()
    assert sd["exp_avg_sq"][0].tolist() == saved[1].tolist()


def test_state_dict_layout():
    # WHY: formats/checkpoint.md stores <name>.exp_avg and <name>.exp_avg_sq per
    #      parameter plus the step; L0.6 reads exactly these keys, and the
    #      hyperparameters come back on load.
    # KIND: unit
    # CATCHES: s20
    # CHAPTER: M10.3 section 4, The interface
    ps = [P(np.zeros(3)), P(np.zeros((2, 2)))]
    opt = AdamW(ps, lr=0.02, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.1)
    sd = opt.state_dict()
    assert {
        "step",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "exp_avg",
        "exp_avg_sq",
    } <= set(sd)
    assert sd["step"] == 0 and sd["lr"] == 0.02 and list(sd["betas"]) == [0.9, 0.95]
    assert sd["eps"] == 1e-6 and sd["weight_decay"] == 0.1
    assert [m.shape for m in sd["exp_avg"]] == [(3,), (2, 2)]
    other = AdamW([P(np.zeros(3)), P(np.zeros((2, 2)))])
    other.load_state_dict(sd)
    back = other.state_dict()
    assert back["lr"] == 0.02 and list(back["betas"]) == [0.9, 0.95]
    assert back["eps"] == 1e-6 and back["weight_decay"] == 0.1
    with pytest.raises(ValueError):
        AdamW([P(np.zeros(3))]).load_state_dict(sd)
    with pytest.raises(ValueError):
        AdamW([P(np.zeros(3)), P(np.zeros(4))]).load_state_dict(sd)


# --- arguments -------------------------------------------------------------------


def test_rejects_bad_hyperparameters():
    # WHY: a negative lr or a beta of 1 trains silently in the wrong direction
    #      or never forgets; torch rejects them at construction, and so do you.
    # KIND: boundary
    # CATCHES: s21, m05
    # CHAPTER: M10.3 section 4, The interface
    ok = [P(np.zeros(2))]
    for kw in (
        {"lr": -1e-3},
        {"eps": -1.0},
        {"weight_decay": -0.1},
        {"betas": (1.0, 0.999)},
        {"betas": (0.9, 1.0)},
        {"betas": (-0.1, 0.999)},
    ):
        with pytest.raises(ValueError):
            AdamW(ok, **kw)
    with pytest.raises(ValueError):
        AdamW([])
    with pytest.raises(ValueError):
        AdamW([P(np.zeros(2, dtype=np.int64))])
    AdamW(ok, lr=0.0, betas=(0.0, 0.0), eps=0.0, weight_decay=0.0)  # all legal
