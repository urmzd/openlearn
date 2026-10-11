"""Course tests for L0.4: the module system and basic layers
(tinyllm/nn/module.py, tinyllm/nn/layers.py).

Rung R0 for this file: read these before you write code. (Your own graded
tests for this module, rung R3, go in python/tests/l0-4-module/; see the
chapter, section 4.) Each test names why it exists (WHY), what kind of check
it is (KIND), the planted bugs it kills (CATCHES, mutants in
course/mutants/L0.4), and the chapter section it comes from.

The worked example of the chapter (section 3): Linear(2, 3) with
W = [[1, 2], [3, 4], [5, 6]], b = [0.5, -0.5, 1], x = [[1, 1]]: y = x W^T + b
= [[3.5, 6.5, 12]]; with an upstream gradient of ones, dW = g^T x = all ones,
db = [1, 1, 1], dx = g W = [[9, 12]].

The golden cases in course/fixtures/L0.4/layers_torch.npz were recorded from
torch 2.14.1 by course/oracle/L0.4/layers_torch.py.
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
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import (
    GELU,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
    Tanh,
)
from tinyllm.nn.module import Module

FIX = (
    Path(os.environ.get("TINYLLM_FIXTURES", "course/fixtures"))
    / "L0.4"
    / "layers_torch.npz"
)


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Rng:
    """The frozen PCG32 behind the generator API layers use (uniform(),
    uniforms(), substream() is never needed when an rng is passed)."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.g.uniform() for _ in range(n)], dtype=np.float64)


class Net(Module):
    """The nested structure of the oracle's torch Net, attribute for attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = Embedding(7, 4, rng=Rng(1))
        self.blocks = ModuleList(
            [
                Sequential(Linear(4, 8, rng=Rng(2)), ReLU(), Linear(8, 4, rng=Rng(3)))
                for _ in range(2)
            ]
        )
        self.ln = LayerNorm(4)
        self.head = Linear(4, 7, bias=False, rng=Rng(4))


def _golden():
    data = np.load(FIX, allow_pickle=False)
    return data, json.loads(str(data["__meta__"]))


def _build(name: str) -> Module:
    r = Rng(0)
    return {
        "linear": lambda: Linear(3, 4, rng=r),
        "linear_nobias": lambda: Linear(3, 2, bias=False, rng=r),
        "layernorm": lambda: LayerNorm(6),
        "embedding": lambda: Embedding(5, 3, rng=r),
        "mlp": lambda: Sequential(Linear(3, 8, rng=r), Tanh(), Linear(8, 2, rng=r)),
        "mlp_gelu": lambda: Sequential(
            Linear(3, 5, rng=r),
            GELU(),
            Linear(5, 3, rng=r),
            ReLU(),
            Linear(3, 1, rng=r),
        ),
    }[name]()


def _cases():
    try:
        return [c["name"] for c in _golden()[1]["cases"]]
    except OSError:
        return ["missing-fixture"]


# --- the worked example ---------------------------------------------------------------


def test_hand_example_linear():
    # WHY: the chapter's worked example, number for number: torch's layout
    #      stores one row of W per output, so y = x W^T + b, and the three
    #      gradients are g^T x, the column sums of g, and g W.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: L0.4 section 3, Worked example by hand
    lin = Linear(2, 3, rng=Rng(seed()))
    lin.load_state_dict({"weight": [[1, 2], [3, 4], [5, 6]], "bias": [0.5, -0.5, 1.0]})
    x = Tensor([[1.0, 1.0]], requires_grad=True)
    y = lin(x)
    assert_close(y.data, [[3.5, 6.5, 12.0]], dtype="float32")
    y.backward(np.ones((1, 3)))
    assert_close(lin.weight.grad, np.ones((3, 2)), dtype="float32")
    assert_close(lin.bias.grad, [1.0, 1.0, 1.0], dtype="float32")
    assert_close(x.grad, [[9.0, 12.0]], dtype="float32")


# --- against torch -------------------------------------------------------------------------


@pytest.mark.parametrize("name", _cases())
def test_matches_torch(name):
    # WHY: with torch's weights copied in by state_dict name, every layer
    #      (Linear with and without bias, LayerNorm with a learned affine,
    #      Embedding with repeated ids, two MLPs through Sequential) gives
    #      torch's output and torch's gradients for the input and every
    #      parameter. L7.9 loads HF checkpoints exactly this way.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, m01, m02
    # CHAPTER: L0.4 section 4, The interface
    data, meta = _golden()
    c = next(c for c in meta["cases"] if c["name"] == name)
    k = c["key"]
    m = _build(name)
    assert [n for n, _ in m.named_parameters()] == c["params"]
    m.load_state_dict({p: data[f"{k}_p_{p}"] for p in c["params"]})
    x = data[f"{k}_x"] if c["int_input"] else Tensor(data[f"{k}_x"], requires_grad=True)
    y = m(x)
    assert_close(y.data, data[f"{k}_y"], dtype="float32", msg=f"{name} forward")
    y.backward(data[f"{k}_g"])
    if not c["int_input"]:
        assert_close(x.grad, data[f"{k}_gx"], dtype="float32", msg=f"{name} input grad")
    params = dict(m.named_parameters())
    for p in c["params"]:
        assert_close(
            params[p].grad,
            data[f"{k}_gp_{p}"],
            dtype="float32",
            msg=f"{name} grad of {p}",
        )


def test_state_dict_names_match_torch():
    # WHY: state_dict names are the safetensors key contract: the same
    #      attributes registered in the same order give torch's exact keys,
    #      in torch's order, down through ModuleList and Sequential
    #      ("blocks.1.2.bias"), and a Linear without bias has no "bias" key.
    # KIND: golden
    # CATCHES: s07, m03
    # CHAPTER: L0.4 section 2, Principles (names are the contract)
    _, meta = _golden()
    assert list(Net().state_dict()) == meta["net_state_dict_keys"]


# --- properties ------------------------------------------------------------------------------


def test_state_dict_roundtrip():
    # WHY: save, build a fresh model with a different seed, load: the two
    #      models compute the same function. The saved arrays are copies, so
    #      training on after a save does not change what was saved.
    # KIND: property
    # CATCHES: s08, m04
    # CHAPTER: L0.4 section 2, Principles (state_dict)
    a, b = Net(), Net()
    b.load_state_dict({k: np.full_like(v, 3.0) for k, v in b.state_dict().items()})
    sd = a.state_dict()
    snapshot = {k: v.copy() for k, v in sd.items()}
    b.load_state_dict(dict(reversed(list(sd.items()))))  # matched by name, not position
    for k in sd:
        assert_close(b.state_dict()[k], sd[k], dtype="float32", msg=k)
    x = Tensor(PCG32(seed=seed()).uniform_array((2, 4), -1, 1))
    assert_close(b.ln(b.blocks[1](x)).data, a.ln(a.blocks[1](x)).data, dtype="float32")
    a.head.weight.data += 1.0
    for k, v in snapshot.items():
        assert_close(sd[k], v, dtype="float32", msg=k)


def test_load_is_in_place():
    # WHY: an optimizer built before loading holds the parameter Tensors and
    #      updates their data arrays in place; load_state_dict must write into
    #      those same arrays, or a resumed run trains copies nobody reads.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: L0.4 section 5, Pitfalls, item 3
    lin = Linear(2, 2, rng=Rng(seed()))
    held = list(lin.parameters())
    arrays = [p.data for p in held]
    lin.load_state_dict({"weight": np.eye(2), "bias": np.array([1.0, 2.0])})
    assert all(p is q for p, q in zip(held, lin.parameters()))
    assert all(p.data is a for p, a in zip(held, arrays))
    assert_close(held[1].data, [1.0, 2.0], dtype="float32")
    assert held[0].data.dtype == np.float32


def test_load_state_dict_strict():
    # WHY: a checkpoint from a different architecture must fail loudly:
    #      strict mode names every missing and unexpected key and copies
    #      nothing; a wrong shape is a ValueError; strict=False loads what
    #      matches (loading a backbone into a model with a new head).
    # KIND: boundary
    # CATCHES: s10, s11, m05
    # CHAPTER: L0.4 section 5, Pitfalls, item 4
    lin = Linear(2, 2, rng=Rng(seed()))
    before = lin.state_dict()
    with pytest.raises(KeyError, match="bias"):
        lin.load_state_dict({"weight": np.zeros((2, 2))})
    with pytest.raises(KeyError, match="extra"):
        lin.load_state_dict(
            {"weight": np.zeros((2, 2)), "bias": np.zeros(2), "extra": np.zeros(1)}
        )
    assert_close(lin.weight.data, before["weight"], dtype="float32")  # nothing copied
    with pytest.raises(ValueError):
        lin.load_state_dict(
            {"weight": np.zeros((1, 2)), "bias": np.zeros(2)}
        )  # would broadcast
    lin.load_state_dict({"weight": np.zeros((2, 2))}, strict=False)
    assert_close(lin.weight.data, np.zeros((2, 2)), dtype="float32")
    assert_close(lin.bias.data, before["bias"], dtype="float32")


def test_parameters_order_and_tying():
    # WHY: parameters come in registration order, own before children's; a
    #      Tensor registered twice (tied input and output embeddings, L7.9)
    #      appears once, so the optimizer does not step it twice.
    # KIND: unit
    # CATCHES: s06, s12
    # CHAPTER: L0.4 section 2, Principles (registration)

    class Tied(Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = Tensor([1.0], requires_grad=True)
            self.emb = Embedding(5, 3, rng=Rng(1))
            self.out = Linear(3, 5, bias=False, rng=Rng(2))
            self.out.weight = self.emb.weight

    m = Tied()
    names = [n for n, _ in m.named_parameters()]
    assert names == ["scale", "emb.weight"]
    assert len(list(m.parameters())) == 2
    assert list(m.state_dict()) == ["scale", "emb.weight"]


def test_plain_state_is_not_a_parameter():
    # WHY: only Tensors that require grad are parameters; a constant Tensor
    #      (a causal mask, a frequency table) and numbers are plain state, and
    #      re-assigning a parameter name to None unregisters it.
    # KIND: boundary
    # CATCHES: s13
    # CHAPTER: L0.4 section 2, Principles (registration)

    class M(Module):
        def __init__(self) -> None:
            super().__init__()
            self.mask = Tensor(np.tril(np.ones((3, 3))))
            self.w = Tensor(np.ones(2), requires_grad=True)
            self.n = 3

    m = M()
    assert [n for n, _ in m.named_parameters()] == ["w"]
    m.w = None
    assert list(m.named_parameters()) == []


def test_forgetting_super_init_is_explained():
    # WHY: assigning a parameter before super().__init__() would register it
    #      nowhere; the error must say what to do.
    # KIND: boundary
    # CATCHES: m06
    # CHAPTER: L0.4 section 5, Pitfalls, item 1

    class Bad(Module):
        def __init__(self) -> None:
            self.w = Tensor([1.0], requires_grad=True)

    with pytest.raises(AttributeError, match="super"):
        Bad()


def test_train_eval_reaches_every_child():
    # WHY: model.eval() must switch off dropout in every nested block, and
    #      train() must switch it back; both return the model for chaining.
    # KIND: unit
    # CATCHES: s14, m09
    # CHAPTER: L0.4 section 5, Pitfalls, item 2
    m = Sequential(
        Linear(4, 4, rng=Rng(1)), Sequential(Dropout(0.5, rng=Rng(2)), Tanh())
    )
    assert m.eval() is m
    assert all(not mod.training for _, mod in m.named_modules())
    x = Tensor(np.ones((2, 4)))
    assert_close(m[1][0](x).data, x.data, dtype="float32")
    assert m.train() is m
    assert all(mod.training for _, mod in m.named_modules())
    assert (m[1][0](x).data == 0).any()
    with pytest.raises(ValueError):
        Dropout(1.5)


def test_zero_grad():
    # WHY: gradients accumulate (L0.1); zero_grad clears every parameter's
    #      .grad before the next step.
    # KIND: unit
    # CATCHES: s15
    # CHAPTER: L0.4 section 4, The interface
    m = Sequential(Linear(2, 3, rng=Rng(1)), Linear(3, 1, rng=Rng(2)))
    m(Tensor(np.ones((1, 2)))).backward(np.ones((1, 1)))
    assert all(p.grad is not None for p in m.parameters())
    m.zero_grad()
    assert all(p.grad is None for p in m.parameters())


def test_layernorm_normalizes():
    # WHY: with the initial affine (weight 1, bias 0) every row comes out with
    #      mean 0 and population variance 1 (up to eps), whatever its scale.
    # KIND: property
    # CATCHES: s03
    # CHAPTER: L0.4 section 2, Principles (LayerNorm)
    rng = PCG32(seed=seed())
    x = rng.uniform_array((5, 16), -50, 50) + rng.uniform_array((5, 1), -100, 100)
    y = LayerNorm(16)(Tensor(x, dtype=np.float64)).data
    assert_close(y.mean(axis=-1), np.zeros(5), rtol=0, atol=1e-6)
    assert_close(y.var(axis=-1), np.ones(5), rtol=1e-4, atol=0)


def test_layernorm_eps_inside_the_square_root():
    # WHY: the row [0, 0.002] has variance 1e-6, smaller than eps = 1e-5, so
    #      where eps goes matters: (x - mu) / sqrt(var + eps) gives
    #      +-0.001 / sqrt(1.1e-5) = +-0.30151, torch's value; dividing by
    #      sqrt(var) + eps gives +-0.990.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: L0.4 section 5, Pitfalls, item 5
    y = LayerNorm(2)(Tensor([[0.0, 0.002]], dtype=np.float64)).data
    v = 0.001 / math.sqrt(1.1e-5)
    assert_close(y, [[-v, v]], rtol=1e-6, atol=0)


def test_init_statistics_and_seeding():
    # WHY: Linear(in, out) starts with N(0, 1/in) weights and zero bias, so a
    #      unit-variance input gives unit-variance outputs (M07.3). The same
    #      rng seed gives the same weights; another seed gives others.
    # KIND: statistical
    # CATCHES: s16, m07
    # CHAPTER: L0.4 section 2, Principles (initialization)
    lin = Linear(400, 300, rng=Rng(seed()))
    w = lin.weight.data
    assert w.shape == (300, 400) and w.dtype == np.float32
    sd = math.sqrt(1 / 400)
    # The sample std of 120000 normals is within 1% of the true std (about 7 se).
    assert abs(float(w.std()) - sd) / sd < 0.01
    assert abs(float(w.mean())) < 4 * sd / math.sqrt(w.size)
    assert (lin.bias.data == 0).all()
    a = Linear(3, 2, rng=Rng(5)).weight.data
    b = Linear(3, 2, rng=Rng(5)).weight.data
    c = Linear(3, 2, rng=Rng(6)).weight.data
    assert (a == b).all() and not (a == c).all()
    assert (Linear(3, 2).weight.data == Linear(3, 2).weight.data).all()


def test_sequential_and_modulelist():
    # WHY: Sequential applies its children in order and indexes like a list;
    #      ModuleList only holds modules (it has no forward) and append
    #      registers the next index.
    # KIND: unit
    # CATCHES: s17, m08
    # CHAPTER: L0.4 section 4, The interface
    a, b = Linear(2, 3, rng=Rng(1)), Tanh()
    s = Sequential(a, b)
    assert len(s) == 2 and s[0] is a and list(s) == [a, b]
    x = Tensor(np.ones((1, 2)))
    assert_close(s(x).data, np.tanh(a(x).data), dtype="float32")
    ml = ModuleList([Linear(1, 1, rng=Rng(2))]).append(Linear(1, 1, rng=Rng(3)))
    assert len(ml) == 2 and [n for n, _ in ml.named_parameters()] == [
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
    ]
    with pytest.raises(NotImplementedError):
        ml(x)
    with pytest.raises(TypeError):
        Sequential(a, "tanh")
