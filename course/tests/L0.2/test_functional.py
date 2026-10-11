"""Course tests for L0.2: the op library (tinyllm/autograd/functional.py, `F`).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L0.2), and the chapter section it comes from.

The worked example of the chapter (section 3): softmax of x = [0, ln 2, ln 3]
is [1/6, 2/6, 3/6]; with upstream gradient g = [1, 0, 0] the input gradient
is y * (g - y . g) = [5/36, -2/36, -3/36].

The golden cases in course/fixtures/L0.2/ops_torch.npz were recorded from
torch 2.14.1 by course/oracle/L0.2/ops_torch.py (forward and backward).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor

F64 = np.float64
FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "L0.2"
    / "ops_torch.npz"
)


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Rng:
    """The frozen PCG32 behind the generator API dropout uses (uniforms),
    counting draws so a test can check the draw accounting."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)
        self.draws = 0

    def uniform(self) -> float:
        self.draws += 1
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.uniform() for _ in range(n)], dtype=np.float64)


# --- the golden table -----------------------------------------------------------------


def _golden():
    data = np.load(FIX, allow_pickle=False)
    meta = json.loads(str(data["__meta__"]))
    return data, meta["cases"]


def _axis(v):
    return tuple(v) if isinstance(v, list) else v


def _call(op: str, xs: list[Tensor], kw: dict, aux: dict) -> Tensor:
    if op in ("exp", "log", "tanh", "sigmoid", "relu", "silu"):
        return getattr(F, op)(xs[0])
    if op == "gelu":
        return F.gelu(xs[0], **kw)
    if op in ("sum", "mean", "max"):
        return getattr(F, op)(
            xs[0], axis=_axis(kw.get("axis")), keepdims=kw.get("keepdims", False)
        )
    if op == "var":
        return F.var(
            xs[0], axis=_axis(kw.get("axis")), correction=kw.get("correction", 0)
        )
    if op == "reshape":
        return F.reshape(xs[0], tuple(kw["shape"]))
    if op == "transpose":
        return F.transpose(xs[0], kw["a"], kw["b"])
    if op == "permute":
        return F.permute(xs[0], tuple(kw["dims"]))
    if op in ("concat", "stack"):
        return getattr(F, op)(xs, axis=kw["axis"])
    if op == "where":
        return F.where(aux["cond"], xs[0], xs[1])
    if op == "gather":
        return F.gather(xs[0], aux["idx"], axis=kw["axis"])
    if op == "embedding":
        return F.embedding(xs[0], aux["ids"])
    if op == "masked_fill":
        return F.masked_fill(xs[0], aux["mask"], kw["value"])
    if op in ("softmax", "log_softmax"):
        return getattr(F, op)(xs[0], axis=kw["axis"])
    if op == "logsumexp":
        return F.logsumexp(xs[0], axis=kw["axis"], keepdims=kw.get("keepdims", False))
    if op == "matmul":
        return F.matmul(xs[0], xs[1])
    if op == "add":
        return xs[0] + xs[1]
    if op == "div":
        return xs[0] / xs[1]
    if op == "pow":
        return xs[0] ** kw["exponent"]
    if op == "getitem":
        return xs[0][aux["index"]]
    raise KeyError(op)


def _case_ids():
    try:
        return [c["name"] for c in _golden()[1]]
    except OSError:  # the fixture is missing: one placeholder case fails loudly
        return ["missing-fixture"]


# --- the worked example ---------------------------------------------------------------


def test_hand_example_softmax_backward():
    # WHY: the chapter's worked example, number for number: softmax of
    #      [0, ln 2, ln 3] is [1/6, 2/6, 3/6], and the vjp y * (g - y . g)
    #      with g = [1, 0, 0] is [5/36, -2/36, -3/36]. The gradient sums to
    #      zero because softmax outputs always sum to one.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L0.2 section 3, Worked example by hand
    x = Tensor([0.0, math.log(2.0), math.log(3.0)], requires_grad=True, dtype=F64)
    y = F.softmax(x)
    assert_close(y.data, [1 / 6, 2 / 6, 3 / 6])
    y.backward(np.array([1.0, 0.0, 0.0]))
    assert_close(x.grad, [5 / 36, -2 / 36, -3 / 36])


def test_max_ties_share_the_gradient():
    # WHY: the chapter's second hand example: max([1, 3, 3]) = 3 has two
    #      maximal entries, and torch.amax splits the gradient between them,
    #      [0, 0.5, 0.5]. Sending all of it to both doubles it; sending it to
    #      the first only disagrees with torch.
    # KIND: boundary
    # CATCHES: s03, s04
    # CHAPTER: L0.2 section 5, Pitfalls, item 3
    x = Tensor([[1.0, 3.0, 3.0]], requires_grad=True, dtype=F64)
    F.max(x, axis=-1).backward(np.array([1.0]))
    assert_close(x.grad, [[0.0, 0.5, 0.5]])


# --- against torch ---------------------------------------------------------------------


@pytest.mark.parametrize("name", _case_ids())
def test_matches_torch(name):
    # WHY: forward value and input gradients equal torch 2.14 on every op of
    #      the library (45 cases: each reduction with and without keepdims,
    #      ties, masked softmax rows, logits near 1e3, batched matmul, float32).
    #      Model code ported from torch (L5 to L7) relies on the same semantics.
    # KIND: golden
    # CATCHES: s02, s05, s06, s15, s16, s17, s18, s21, s23, m02, m03, m06, m08
    # CHAPTER: L0.2 section 4, The interface
    data, cases = _golden()
    c = next(c for c in cases if c["name"] == name)
    k, dt = c["key"], np.dtype(c["dtype"])
    xs = [
        Tensor(data[f"{k}_x{i}"], requires_grad=True, dtype=dt)
        for i in range(c["n_inputs"])
    ]
    aux = {a: data[f"{k}_aux_{a}"] for a in c["aux"]}
    y = _call(c["op"], xs, c["kwargs"], aux)
    assert y.dtype == dt, f"{name}: output dtype {y.dtype}, expected {dt}"
    want = data[f"{k}_y"]
    tol = {} if dt == F64 else {"dtype": "float32"}
    assert_close(y.data, want, msg=f"{name} forward", **tol)
    y.backward(data[f"{k}_g"])
    for i, x in enumerate(xs):
        assert_close(x.grad, data[f"{k}_gx{i}"], msg=f"{name} grad of input {i}", **tol)


# --- against central differences --------------------------------------------------------

GRAD_CASES = {
    "exp": (lambda a: F.exp(a), [(2, 3)]),
    "log": (lambda a: F.log(a), [(2, 3)]),
    "tanh": (lambda a: F.tanh(a), [(2, 3)]),
    "sigmoid": (lambda a: F.sigmoid(a), [(2, 3)]),
    "relu": (lambda a: F.relu(a), [(2, 3)]),
    "silu": (lambda a: F.silu(a), [(2, 3)]),
    "gelu": (lambda a: F.gelu(a), [(2, 3)]),
    "gelu_tanh": (lambda a: F.gelu(a, approximate="tanh"), [(2, 3)]),
    "sum": (lambda a: F.sum(a, axis=(0, 2)), [(2, 3, 2)]),
    "mean": (lambda a: F.mean(a, axis=1, keepdims=True), [(2, 3)]),
    "max": (lambda a: F.max(a, axis=0), [(3, 2)]),
    "var": (lambda a: F.var(a, axis=-1, correction=1), [(2, 4)]),
    "var_all": (lambda a: F.var(a), [(2, 3)]),
    "reshape": (lambda a: F.reshape(a, (-1, 2)), [(2, 3)]),
    "transpose": (lambda a: F.transpose(a, 1, 2), [(2, 2, 3)]),
    "permute": (lambda a: F.permute(a, (1, 2, 0)), [(2, 3, 2)]),
    "concat": (lambda a, b, c: F.concat([a, b, c], axis=0), [(1, 2), (3, 2), (2, 2)]),
    "stack": (lambda a, b: F.stack([a, b], axis=1), [(2, 3), (2, 3)]),
    "where": (
        lambda a, b: F.where(np.array([True, False, True]), a, b),
        [(2, 3), (2, 1)],
    ),
    "gather": (lambda a: F.gather(a, np.array([[0, 0], [2, 1]]), axis=1), [(2, 3)]),
    "embedding": (lambda w: F.embedding(w, np.array([3, 1, 3, 0])), [(4, 2)]),
    "masked_fill": (
        lambda a: F.masked_fill(a, np.array([[True], [False]]), 0.5),
        [(2, 3)],
    ),
    "softmax": (lambda a: F.softmax(a, axis=0), [(3, 2)]),
    "log_softmax": (lambda a: F.log_softmax(a, axis=-1), [(2, 4)]),
    "logsumexp": (lambda a: F.logsumexp(a, axis=1, keepdims=True), [(2, 3)]),
    "dropout": (lambda a: F.dropout(a, 0.5, True, Rng(11)), [(3, 4)]),
    "matmul": (lambda a, b: F.matmul(a, b), [(2, 3), (3, 4)]),
}


@pytest.mark.parametrize("name", sorted(GRAD_CASES))
def test_gradcheck_each_op(name):
    # WHY: every vjp against the frozen central differences in float64
    #      (rtol 1e-5). Inputs avoid the kinks (|x| > 0.05 for relu, distinct
    #      values for max) and the log's domain (x > 0.2).
    # KIND: gradcheck
    # CATCHES: s01, s02, s05, s06, s07, s15, s21, s22, m02, m03, m08
    # CHAPTER: L0.2 section 2, Principles (one vjp per op)
    fn, shapes = GRAD_CASES[name]
    rng = PCG32(seed=seed() * 7919 + sorted(GRAD_CASES).index(name))
    lo, hi = (0.2, 2.0) if name == "log" else (-2.0, 2.0)
    xs = [rng.uniform_array(s, lo, hi) for s in shapes]
    xs = [np.where(np.abs(x) < 0.05, x + 0.1, x) for x in xs]
    out = fn(*[Tensor(x, dtype=F64) for x in xs])
    w = rng.uniform_array(out.shape, -1.0, 1.0)

    def f(*arrays):
        return float((fn(*[Tensor(a, dtype=F64) for a in arrays]).data * w).sum())

    ts = [Tensor(x, requires_grad=True, dtype=F64) for x in xs]
    fn(*ts).backward(w)
    gradcheck(
        f, xs, [t.grad if t.grad is not None else np.zeros_like(t.data) for t in ts]
    )


# --- edges ---------------------------------------------------------------------------------


def test_float32_in_float32_out():
    # WHY: models train in float32; an op that returns float64 (a float64
    #      dropout mask, a float64 activation derivative) doubles memory and
    #      makes Python disagree with the float32 C kernels in L9.
    # KIND: boundary
    # CATCHES: m04
    # CHAPTER: L0.2 section 5, Pitfalls, item 7
    x = Tensor(np.linspace(-1, 1, 12).reshape(3, 4), requires_grad=True)
    outs = [
        F.exp(x),
        F.tanh(x),
        F.gelu(x),
        F.silu(x),
        F.softmax(x),
        F.log_softmax(x),
        F.logsumexp(x),
        F.var(x, axis=-1),
        F.mean(x),
        F.dropout(x, 0.5, True, Rng(1)),
        F.masked_fill(x, x.data > 0, 0.0),
        F.where(x.data > 0, x, 1.0),
    ]
    for y in outs:
        assert y.dtype == np.float32, y
    F.sum(F.dropout(x, 0.5, True, Rng(1))).backward()
    assert x.grad.dtype == np.float32


def test_relu_derivative_at_zero_is_zero():
    # WHY: relu has no derivative at 0; the course, like torch, uses 0 there,
    #      so a dead unit initialized at exactly 0 gets no update.
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: L0.2 section 2, Principles (kinks)
    x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    F.relu(x).backward(np.ones(3))
    assert_close(x.grad, [0.0, 0.0, 1.0], dtype="float32")


def test_embedding_repeated_ids_accumulate():
    # WHY: a token that appears twice in a batch gets the sum of both rows of
    #      gradient. `out[ids] = g` keeps only the last occurrence, which
    #      trains frequent tokens slower than rare ones.
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: L0.2 section 5, Pitfalls, item 1
    w = Tensor(np.zeros((3, 2)), requires_grad=True)
    y = F.embedding(w, np.array([[2, 0], [2, 2]]))
    assert y.shape == (2, 2, 2)
    y.backward(np.ones((2, 2, 2)))
    assert_close(w.grad, [[1.0, 1.0], [0.0, 0.0], [3.0, 3.0]], dtype="float32")


def test_gather_repeated_indices_accumulate():
    # WHY: gather picks x[0, 0] twice here, so its gradient is 2; a scatter
    #      that assigns (put_along_axis) keeps 1.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: L0.2 section 5, Pitfalls, item 1
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = F.gather(x, np.array([[0, 0, 2]]), axis=1)
    assert_close(y.data, [[1.0, 1.0, 3.0]], dtype="float32")
    y.backward(np.ones((1, 3)))
    assert_close(x.grad, [[2.0, 0.0, 1.0]], dtype="float32")


def test_embedding_rejects_bad_ids():
    # WHY: numpy reads -1 as "the last row" and float ids as an error deep
    #      inside indexing. An id outside [0, n) is a tokenizer or data bug
    #      and must fail where it enters the model.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: L0.2 section 5, Pitfalls, item 2
    w = Tensor(np.zeros((4, 2)), requires_grad=True)
    for bad in (np.array([0, -1]), np.array([4]), np.array([0.0, 1.0])):
        with pytest.raises(ValueError):
            F.embedding(w, bad)
    with pytest.raises(ValueError):
        F.gather(Tensor(np.zeros((2, 3))), np.array([[0, 3]]), axis=1)


def test_dropout_eval_is_identity():
    # WHY: inverted dropout scales at training time, so evaluation returns the
    #      input unchanged, the same object, and draws no random numbers (a
    #      draw here would shift every later sample of the run).
    # KIND: unit
    # CATCHES: s10
    # CHAPTER: L0.2 section 2, Principles (inverted dropout)
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    r = Rng(0)
    assert F.dropout(x, 0.5, False, r) is x
    assert F.dropout(x, 0.0, True, r) is x
    assert r.draws == 0


def test_dropout_mask_and_scale():
    # WHY: at p = 0.25 about 75% of the units survive (3 standard deviations
    #      of a binomial over 20000), and each survivor is scaled by 1/(1 - p)
    #      so the expected output equals the input. The gradient uses the same
    #      mask and scale.
    # KIND: statistical
    # CATCHES: s09
    # CHAPTER: L0.2 section 2, Principles (inverted dropout)
    n, p = 20000, 0.25
    x = Tensor(np.full(n, 2.0), requires_grad=True)
    y = F.dropout(x, p, True, Rng(seed()))
    kept = y.data != 0
    frac = kept.mean()
    sd = math.sqrt(p * (1 - p) / n)
    assert abs(frac - (1 - p)) < 3 * sd, frac
    assert_close(y.data[kept], np.full(int(kept.sum()), 2.0 / (1 - p)), dtype="float32")
    y.backward(np.ones(n))
    assert_close(x.grad, np.where(kept, 1 / (1 - p), 0.0), dtype="float32")


def test_dropout_draw_accounting():
    # WHY: one uniform per element, in C order, from the generator you pass:
    #      the mask is exactly u >= p for the uniforms of a same-seed generator,
    #      and the generator ends exactly x.size draws further on. Resuming a
    #      run (L0.6) replays dropout only if the draw count is fixed.
    # KIND: unit
    # CATCHES: s11, s12
    # CHAPTER: L0.2 section 4, The interface (dropout)
    x = Tensor(np.ones((3, 5)))
    r = Rng(42)
    y = F.dropout(x, 0.3, True, r)
    assert r.draws == 15
    ref = PCG32(seed=42)
    u = np.array([ref.uniform() for _ in range(15)]).reshape(3, 5)
    assert ((y.data != 0) == (u >= 0.3)).all()
    assert r.g.uniform() == ref.uniform()


def test_dropout_p_bounds():
    # WHY: p is a probability. p = 1 drops everything (all zeros, no division
    #      by zero); p outside [0, 1] is a configuration error.
    # KIND: boundary
    # CATCHES: m09
    # CHAPTER: L0.2 section 4, The interface (dropout)
    x = Tensor(np.ones(4), requires_grad=True)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            F.dropout(x, bad, True, Rng(0))
    y = F.dropout(x, 1.0, True, Rng(0))
    assert_close(y.data, np.zeros(4), dtype="float32")


def test_softmax_fully_masked_row():
    # WHY: a causal mask can mask a whole row (padding). Softmax then gives
    #      zeros (M09.2's rule), and the gradient through that row is zero,
    #      not nan: one nan poisons every parameter after the first step.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: L0.2 section 5, Pitfalls, item 5
    x = Tensor(
        [[-np.inf, -np.inf, -np.inf], [0.0, math.log(3.0), -np.inf]],
        requires_grad=True,
        dtype=F64,
    )
    y = F.softmax(x, axis=-1)
    assert_close(y.data, [[0.0, 0.0, 0.0], [0.25, 0.75, 0.0]])
    y.backward(np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]]))
    assert np.isfinite(x.grad).all()
    assert_close(x.grad, [[0.0, 0.0, 0.0], [0.1875, -0.1875, 0.0]])


def test_logsumexp_no_overflow():
    # WHY: exp(1000) overflows; logsumexp([1000, 1000]) = 1000 + ln 2 and its
    #      gradient is softmax = [0.5, 0.5]. A vjp written as exp(x) / exp(y)
    #      gives inf / inf = nan.
    # KIND: boundary
    # CATCHES: s13
    # CHAPTER: L0.2 section 5, Pitfalls, item 4
    x = Tensor([1000.0, 1000.0], requires_grad=True, dtype=F64)
    y = F.logsumexp(x)
    assert_close(y.data, 1000.0 + math.log(2.0))
    y.backward()
    assert_close(x.grad, [0.5, 0.5])


def test_var_correction():
    # WHY: var([1, 2, 3, 4]) is 1.25 with correction 0 (LayerNorm's
    #      population variance) and 5/3 with correction 1 (the sample
    #      variance); n <= correction has no variance.
    # KIND: unit
    # CATCHES: s05, m05
    # CHAPTER: L0.2 section 3, Worked example by hand
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True, dtype=F64)
    assert_close(F.var(x).data, 1.25)
    v = F.var(x, correction=1)
    assert_close(v.data, 5 / 3)
    v.backward()
    assert_close(x.grad, [-1.0, -1 / 3, 1 / 3, 1.0])  # 2 (x - 2.5) / 3
    with pytest.raises(ValueError):
        F.var(Tensor([1.0]), correction=1)


def test_where_and_masked_fill_route_gradients():
    # WHY: where sends each output's gradient to the operand it came from,
    #      including a broadcast one; masked_fill blocks the gradient at the
    #      filled positions (an attention mask must not train masked scores).
    # KIND: unit
    # CATCHES: s14, s15
    # CHAPTER: L0.2 section 2, Principles (selection)
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, dtype=F64)
    b = Tensor([10.0, 20.0], requires_grad=True, dtype=F64)
    cond = np.array([[True, False], [False, False]])
    y = F.where(cond, a, b)
    assert_close(y.data, [[1.0, 20.0], [10.0, 20.0]])
    y.backward(np.ones((2, 2)))
    assert_close(a.grad, [[1.0, 0.0], [0.0, 0.0]])
    assert_close(b.grad, [1.0, 2.0])
    c = Tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=F64)
    z = F.masked_fill(c, np.array([False, True, False]), 0.5)
    assert_close(z.data, [1.0, 0.5, 3.0])
    z.backward(np.ones(3))
    assert_close(c.grad, [1.0, 0.0, 1.0])
    assert F.masked_fill(c, np.array([True, False, False]), -np.inf).data[0] == -np.inf


def test_gelu_rejects_unknown_approximation():
    # WHY: "none" and "tanh" are the two forms torch and HF configs name; a
    #      typo must not silently select one of them.
    # KIND: boundary
    # CATCHES: s18
    # CHAPTER: L0.2 section 4, The interface
    x = Tensor([1.0])
    with pytest.raises(ValueError):
        F.gelu(x, approximate="fast")
    assert (
        abs(
            float(F.gelu(Tensor([1.0], dtype=F64), approximate="tanh").data[0])
            - 0.8411919906082768
        )
        < 1e-12
    )


def test_ops_build_no_graph_under_no_grad():
    # WHY: every op goes through from_op, so under no_grad none of them keeps
    #      its inputs alive: evaluation of a large batch does not hold every
    #      activation in memory. An ndarray passed where a Tensor goes is a
    #      constant, so it never requires grad either.
    # KIND: property
    # CATCHES: m10
    # CHAPTER: L0.2 section 2, Principles (one vjp per op)
    assert not F.exp(np.zeros(2)).requires_grad
    assert not F.softmax(np.zeros((2, 2))).requires_grad
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    with no_grad():
        for y in (
            F.exp(x),
            F.softmax(x),
            F.sum(x),
            F.embedding(x, np.array([0])),
            F.concat([x, x]),
            F.dropout(x, 0.5, True, Rng(0)),
            F.matmul(x, F.transpose(x, 0, 1)),
        ):
            assert not y.requires_grad


def test_gradcheck_all_reports_every_op():
    # WHY: gradcheck_all is MS-L0's `{tinyllm} gradcheck --suite all`: one
    #      report per op of the library, every one ok on a correct library.
    # KIND: unit
    # CATCHES: s24
    # CHAPTER: L0.2 section 4, The interface (gradcheck_all)
    rep = F.gradcheck_all()
    want = {
        "exp",
        "log",
        "tanh",
        "sigmoid",
        "relu",
        "silu",
        "gelu",
        "gelu_tanh",
        "sum",
        "mean",
        "max",
        "var",
        "reshape",
        "transpose",
        "permute",
        "concat",
        "stack",
        "where",
        "gather",
        "embedding",
        "masked_fill",
        "softmax",
        "log_softmax",
        "logsumexp",
        "dropout",
        "matmul",
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "neg",
        "getitem",
    }
    assert want <= set(rep), sorted(want - set(rep))
    bad = [k for k, r in rep.items() if not r.ok]
    assert not bad, f"not ok: {bad}"


def test_gradcheck_all_catches_a_wrong_op(monkeypatch):
    # WHY: a checker that always says ok is worse than none. With softmax's
    #      backward off by 0.1% gradcheck_all must flag softmax (central
    #      differences resolve 1e-5); with sigmoid's off by 1e-7, only the
    #      forward-mode dual check (1e-9) can see it, and it must.
    # KIND: boundary
    # CATCHES: s19, s20
    # CHAPTER: L0.2 section 5, Pitfalls, item 6
    from tinyllm.autograd.tensor import from_op

    def softmax_bad(x, axis=-1):
        e = np.exp(x.data - x.data.max(axis=axis, keepdims=True))
        y = e / e.sum(axis=axis, keepdims=True)
        return from_op(
            y,
            (x,),
            lambda g: (1.001 * y * (g - (g * y).sum(axis=axis, keepdims=True)),),
            "softmax",
        )

    def sigmoid_bad(x):
        s = 1 / (1 + np.exp(-x.data))
        return from_op(s, (x,), lambda g: (g * s * (1 - s) * (1 + 1e-7),), "sigmoid")

    monkeypatch.setattr(F, "softmax", softmax_bad)
    monkeypatch.setattr(F, "sigmoid", sigmoid_bad)
    rep = F.gradcheck_all()
    assert not rep["softmax"].ok
    assert not rep["sigmoid"].ok
    assert rep["exp"].ok
