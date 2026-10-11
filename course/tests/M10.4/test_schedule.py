"""Course tests for M10.4: learning-rate schedules and gradient clipping
(tinyllm/optim/schedule.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M10.4), and the chapter section it comes from.

The worked examples of the chapter (section 3): cosine with warmup 2, total 6,
lr 1 down to 0.1; WSD with warmup 2, stable 2, decay 4, lr 1 down to 0.2; Noam
with d_model 4 and warmup 4; clipping the gradients [3, 4] and [12] (norm 13)
to 6.5.
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
from tinyllm.optim.schedule import clip_grad_norm_, cosine_with_warmup, noam, wsd


class P:
    """A parameter as clipping sees it: only grad matters."""

    def __init__(self, grad):
        self.grad = grad


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def fixture() -> dict:
    root = Path(
        os.environ.get(
            "TINYLLM_FIXTURES", Path(__file__).resolve().parents[2] / "fixtures"
        )
    )
    return json.loads((root / "M10.4" / "schedule_hf.json").read_text())


# --- the worked examples --------------------------------------------------------


def test_hand_example_cosine():
    # WHY: the chapter's cosine table, step by step: a linear ramp 0, 0.5, then
    #      the half cosine from 1 at step 2 to the floor 0.1 at step 6, and the
    #      floor after.
    # KIND: unit
    # CATCHES: s01, s02, s03, m01
    # CHAPTER: M10.4 section 3, Worked example by hand
    c = math.cos(math.pi / 4)
    want = [
        0.0,
        0.5,
        1.0,
        0.1 + 0.9 * (1 + c) / 2,
        0.55,
        0.1 + 0.9 * (1 - c) / 2,
        0.1,
        0.1,
    ]
    got = [
        cosine_with_warmup(t, warmup=2, total=6, lr_max=1.0, lr_min=0.1)
        for t in range(8)
    ]
    assert_close(got, want, dtype="float64")


def test_hand_example_wsd():
    # WHY: the chapter's WSD table: 0, 0.5, then 1 for steps 2 to 4, a straight
    #      line 0.8, 0.6, 0.4 down to the floor 0.2 at step 8, and 0.2 after.
    # KIND: unit
    # CATCHES: s05, s06, s07, m02
    # CHAPTER: M10.4 section 3, Worked example by hand
    got = [
        wsd(t, warmup=2, stable=2, decay=4, lr_max=1.0, lr_min=0.2) for t in range(10)
    ]
    assert_close(
        got, [0.0, 0.5, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.2], dtype="float64"
    )


def test_hand_example_noam():
    # WHY: the chapter's Noam numbers with d_model = 4 and warmup = 4:
    #      0 at step 0, 1/16 at step 1, 1/8 at step 2, the peak 1/4 at step 4,
    #      and 1/8 again at step 16 (decay like 1/sqrt(step)).
    # KIND: unit
    # CATCHES: s08, s09, s10, m04
    # CHAPTER: M10.4 section 3, Worked example by hand
    got = [noam(t, d_model=4, warmup=4) for t in (0, 1, 2, 4, 16)]
    assert_close(got, [0.0, 1 / 16, 1 / 8, 1 / 4, 1 / 8], dtype="float64")


def test_hand_example_clip():
    # WHY: gradients [3, 4] and [12] have global norm sqrt(9 + 16 + 144) = 13.
    #      Clipping to 6.5 halves every entry (up to torch's 1e-6 guard) and the
    #      call returns 13, the norm before clipping.
    # KIND: unit
    # CATCHES: s11, s13, s14, m03
    # CHAPTER: M10.4 section 3, Worked example by hand
    a, b = P(np.array([3.0, 4.0])), P(np.array([12.0]))
    total = clip_grad_norm_([a, b], max_norm=6.5)
    assert_close(total, 13.0, dtype="float64")
    coef = 6.5 / (13.0 + 1e-6)
    assert_close(a.grad, [3.0 * coef, 4.0 * coef], dtype="float64")
    assert_close(b.grad, [12.0 * coef], dtype="float64")


# --- oracles ------------------------------------------------------------------


def check_schedule(name: str, fn) -> None:
    case = next(c for c in fixture()["schedules"] if c["name"] == name)
    for t, want in enumerate(case["lr"]):
        if t < case["compare_from"]:
            continue
        assert_close(
            fn(t, **case["args"]), want, dtype="float64", msg=f"{name}: step {t}"
        )


def test_cosine_matches_hf():
    # WHY: HF transformers' get_cosine_with_min_lr_schedule_with_warmup and
    #      get_cosine_schedule_with_warmup, read through LambdaLR at every step
    #      from 0 to total. C1 runs are compared with HF-trained baselines.
    # KIND: golden
    # CATCHES: s01, s02, s03, m01
    # CHAPTER: M10.4 section 2.2, Cosine decay with warmup
    check_schedule("cosine_min_lr", cosine_with_warmup)
    check_schedule("cosine_no_warmup_to_zero", cosine_with_warmup)


def test_wsd_matches_hf():
    # WHY: HF get_wsd_schedule with decay_type="linear", past the end of
    #      training. With a floor, HF's warmup starts at the floor and yours at
    #      0, so that case is compared from the end of warmup on.
    # KIND: golden
    # CATCHES: s05, s06, s07, m02
    # CHAPTER: M10.4 section 2.3, Warmup, stable, decay
    check_schedule("wsd_linear_floor", wsd)
    check_schedule("wsd_linear_to_zero", wsd)


def test_clip_matches_torch():
    # WHY: torch.nn.utils.clip_grad_norm_ on four gradient sets: clipped, below
    #      the limit, with a None grad, and float32. The returned norm and every
    #      gradient after the call match.
    # KIND: golden
    # CATCHES: s11, s13, s14, s16, m03
    # CHAPTER: M10.4 section 2.5, Clipping by the global norm
    for case in fixture()["clip"]:
        dt = case["dtype"]
        ps = [P(None if g is None else np.array(g, dtype=dt)) for g in case["grads"]]
        total = clip_grad_norm_(ps, case["max_norm"])
        assert_close(total, case["norm"], dtype=dt, msg=case["name"])
        for p, want in zip(ps, case["after"]):
            if want is None:
                assert p.grad is None
            else:
                assert p.grad.dtype == np.dtype(dt)
                assert_close(p.grad, np.array(want), dtype=dt, msg=case["name"])


# --- properties and boundaries ----------------------------------------------------


def test_schedules_hit_their_corners():
    # WHY: over random configurations, each schedule is lr_max exactly where
    #      warmup ends, never above it, never below lr_min after warmup, never
    #      increasing after warmup, and exactly lr_min from the end on. Every
    #      off-by-one in a phase boundary breaks one of these.
    # KIND: property
    # CATCHES: s01, s04, s05, s06, s07, m01, m02
    # CHAPTER: M10.4 section 2.1, A schedule is a function of the step
    rng = PCG32(seed=seed())
    for _ in range(40):
        warm, stable, decay = rng.below(6), rng.below(6), 1 + rng.below(8)
        lr_max = 0.5 + rng.uniform()
        lr_min = lr_max * rng.uniform()
        total = warm + stable + decay
        for name, f in (
            ("cosine", lambda t: cosine_with_warmup(t, warm, total, lr_max, lr_min)),
            ("wsd", lambda t: wsd(t, warm, stable, decay, lr_max, lr_min)),
        ):
            lrs = [f(t) for t in range(total + 4)]
            if warm < total:
                assert_close(f(warm), lr_max, dtype="float64", msg=name)
            assert max(lrs) <= lr_max * (1 + 1e-12), name
            assert all(x >= lr_min * (1 - 1e-12) for x in lrs[warm:]), name
            assert all(
                b <= a * (1 + 1e-12) for a, b in zip(lrs[warm:], lrs[warm + 1 :])
            ), name
            assert all(x == lr_min for x in lrs[total:]), name


def test_cosine_after_total_stays_at_floor():
    # WHY: a run that trains past `total` (an extra eval step, a resumed run with
    #      a larger budget) must stay at the floor. The raw cosine keeps going
    #      and climbs back toward lr_max.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: M10.4 section 5, Pitfalls
    for t in (100, 101, 150, 199, 1000):
        assert (
            cosine_with_warmup(t, warmup=10, total=100, lr_max=3e-4, lr_min=3e-5)
            == 3e-5
        )


def test_noam_peak_and_inverse_sqrt_decay():
    # WHY: the Noam rate grows linearly to its peak (d_model * warmup)^-1/2 at
    #      step = warmup, then lr * sqrt(step) is constant. L5.5 trains the 2017
    #      Transformer with d_model = 512 and warmup = 4000.
    # KIND: property
    # CATCHES: s09, s10, m04
    # CHAPTER: M10.4 section 2.4, The Noam schedule
    d, w = 512, 4000
    peak = noam(w, d, w)
    assert_close(peak, (d * w) ** -0.5, dtype="float64")
    assert max(noam(t, d, w) for t in range(1, 3 * w, 97)) <= peak
    for t in (w, 2 * w, 9 * w, 100 * w):
        assert_close(noam(t, d, w) * math.sqrt(t), peak * math.sqrt(w), dtype="float64")
    for t in (1, 10, w // 2):
        assert_close(noam(t, d, w), t * peak / w, dtype="float64")


def test_noam_step_zero_is_zero():
    # WHY: the train loop asks for the lr of update 0. The formula's limit is
    #      0 (min(inf, 0)); computing 0 ** -0.5 raises ZeroDivisionError in
    #      Python.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: M10.4 section 5, Pitfalls
    assert noam(0, d_model=512, warmup=4000) == 0.0


def test_clip_below_max_is_a_no_op():
    # WHY: clipping only ever shrinks. When the norm is under max_norm the
    #      gradients are left exactly as they were, not scaled up to the limit.
    # KIND: boundary
    # CATCHES: s12
    # CHAPTER: M10.4 section 2.5, Clipping by the global norm
    rng = PCG32(seed=seed() + 1)
    gs = [0.01 * rng.normal_array((5,)), 0.01 * rng.normal_array((2, 3))]
    ps = [P(g.copy()) for g in gs]
    total = clip_grad_norm_(ps, max_norm=10.0)
    assert total < 10.0
    for p, g in zip(ps, gs):
        assert p.grad.tobytes() == g.tobytes()


def test_clip_is_global_and_in_place():
    # WHY: one coefficient for all tensors keeps the gradient's direction; after
    #      clipping, the global norm is max_norm (to 1e-6 relative). The arrays
    #      are scaled in place, so an optimizer holding p.grad sees the change.
    # KIND: property
    # CATCHES: s11, s14
    # CHAPTER: M10.4 section 2.5, Clipping by the global norm
    rng = PCG32(seed=seed() + 2)
    for _ in range(10):
        gs = [
            rng.normal_array((4,)),
            30.0 * rng.normal_array((3, 3)),
            0.1 * rng.normal_array((2,)),
        ]
        ps = [P(g.copy()) for g in gs]
        held = [p.grad for p in ps]
        total = clip_grad_norm_(ps, max_norm=1.0)
        assert_close(
            total, math.sqrt(sum(float((g * g).sum()) for g in gs)), dtype="float64"
        )
        after = math.sqrt(sum(float((p.grad * p.grad).sum()) for p in ps))
        assert_close(after, 1.0, rtol=1e-5, atol=0.0)
        ratio = ps[0].grad[0] / gs[0][0]
        for p, g, h in zip(ps, gs, held):
            assert p.grad is h
            assert_close(p.grad, ratio * g, dtype="float64")


def test_clip_nonfinite_norm_leaves_grads():
    # WHY: an overflowed gradient (inf) gives an infinite norm. Scaling by
    #      max_norm / inf = 0 would silently zero the whole update; the grads
    #      are left alone and the caller sees the inf and skips the step
    #      (L11.1 loss scaling does exactly that).
    # KIND: boundary
    # CATCHES: s15
    # CHAPTER: M10.4 section 5, Pitfalls
    a, b = P(np.array([1.0, np.inf])), P(np.array([2.0, 3.0]))
    total = clip_grad_norm_([a, b], max_norm=1.0)
    assert math.isinf(total)
    assert a.grad.tolist() == [1.0, np.inf] and b.grad.tolist() == [2.0, 3.0]


def test_clip_skips_missing_grads():
    # WHY: parameters the loss did not reach have grad None; they contribute
    #      nothing and stay None. With no grads at all the norm is 0.
    # KIND: boundary
    # CATCHES: s16
    # CHAPTER: M10.4 section 4, The interface
    a, b = P(None), P(np.array([0.6, 0.8]))
    assert_close(clip_grad_norm_([a, b], max_norm=0.5), 1.0, dtype="float64")
    assert a.grad is None
    assert clip_grad_norm_([P(None)], max_norm=1.0) == 0.0


def test_rejects_bad_arguments():
    # WHY: a schedule with lr_min above lr_max, a negative step, or a zero decay
    #      length is a configuration error; say so at the first step, not as a
    #      division by zero or a rising "decay".
    # KIND: boundary
    # CATCHES: m05
    # CHAPTER: M10.4 section 4, The interface
    for args in (
        (-1, 0, 10, 1.0, 0.0),
        (0, 11, 10, 1.0, 0.0),
        (0, 0, 0, 1.0, 0.0),
        (0, 0, 10, 1.0, 2.0),
        (0, 0, 10, 1.0, -0.1),
    ):
        with pytest.raises(ValueError):
            cosine_with_warmup(*args)
    for args in (
        (-1, 0, 0, 1, 1.0, 0.0),
        (0, 0, 0, 0, 1.0, 0.0),
        (0, 1, 1, 1, 0.5, 1.0),
    ):
        with pytest.raises(ValueError):
            wsd(*args)
    for args in ((-1, 512, 4000), (1, 0, 4000), (1, 512, 0)):
        with pytest.raises(ValueError):
            noam(*args)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            clip_grad_norm_([P(np.ones(2))], max_norm=bad)
