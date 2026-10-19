"""Model test suite for L0.4: what each test protects. Written test-first; the
chapter section 4 gives the interface and the first test. They import only the contract."""

import numpy as np
import pytest
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Dropout, Embedding, LayerNorm, Linear, ModuleList, Sequential, Tanh
from tinyllm.nn.module import Module


class Gen:
    """A tiny deterministic stand-in generator: uniform() walks a fixed grid."""

    def __init__(self) -> None:
        self.k = 0

    def uniform(self) -> float:
        self.k += 1
        return (self.k * 0.6180339887) % 1.0

    def next_u32(self) -> int:
        return int(self.uniform() * 2**32)

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.uniform() for _ in range(n)])


def test_hand_example_linear():
    """W = [[1, 2], [3, 4], [5, 6]], b = [0.5, -0.5, 1], x = [[1, 1]]: y = [[3.5, 6.5, 12]], dx = [[9, 12]]."""
    lin = Linear(2, 3, rng=Gen())
    lin.load_state_dict({"weight": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), "bias": np.array([0.5, -0.5, 1.0])})
    x = Tensor([[1.0, 1.0]], requires_grad=True)
    y = lin(x)
    np.testing.assert_allclose(y.data, [[3.5, 6.5, 12.0]])
    y.backward(np.ones((1, 3)))
    np.testing.assert_allclose(x.grad, [[9.0, 12.0]])
    np.testing.assert_allclose(lin.weight.grad, np.ones((3, 2)))
    np.testing.assert_allclose(lin.bias.grad, [1.0, 1.0, 1.0])
    seq = Sequential(lin, Tanh())
    np.testing.assert_allclose(seq(Tensor([[1.0, 1.0]])).data, np.tanh([[3.5, 6.5, 12.0]]), rtol=1e-6)
    seq.zero_grad()
    assert lin.weight.grad is None and lin.bias.grad is None


def test_state_dict_names_follow_registration_order():
    """Sequential children are "0", "1", ...; weight before bias; no key for a missing bias."""
    m = Sequential(Linear(2, 3, rng=Gen()), Tanh(), Linear(3, 1, bias=False, rng=Gen()))
    assert list(m.state_dict()) == ["0.weight", "0.bias", "2.weight"]
    ml = ModuleList([Embedding(4, 2, rng=Gen())])
    assert list(ml.state_dict()) == ["0.weight"]
    emb = ml[0]
    np.testing.assert_array_equal(emb(np.array([3, 0])).data, emb.weight.data[[3, 0]])


def test_load_state_dict_roundtrip_by_name():
    """Loading a reordered state dict gives the same parameters (names, not positions)."""
    a = Sequential(Linear(2, 3, rng=Gen()), Linear(3, 2, rng=Gen()))
    b = Sequential(Linear(2, 3, rng=Gen()), Linear(3, 2, rng=Gen()))
    for p in b.parameters():
        p.data[...] = 7.0
    sd = a.state_dict()
    b.load_state_dict(dict(reversed(list(sd.items()))))
    for k, v in b.state_dict().items():
        np.testing.assert_array_equal(v, sd[k])
    a[0].weight.data[...] = -1.0
    assert not (sd["0.weight"] == -1.0).all()  # state_dict is a copy


def test_load_writes_in_place():
    """The Tensors (and arrays) an optimizer holds are still the model's after loading."""
    lin = Linear(2, 2, rng=Gen())
    w, wd = lin.weight, lin.weight.data
    lin.load_state_dict({"weight": np.eye(2), "bias": np.zeros(2)})
    assert lin.weight is w and lin.weight.data is wd
    np.testing.assert_array_equal(wd, np.eye(2))


def test_strict_load_rejects_missing_and_unexpected_keys():
    """Missing or extra keys raise KeyError under strict and change nothing; strict=False loads the rest."""
    lin = Linear(2, 2, rng=Gen())
    before = lin.state_dict()
    with pytest.raises(KeyError):
        lin.load_state_dict({"weight": np.zeros((2, 2))})
    np.testing.assert_array_equal(lin.weight.data, before["weight"])
    with pytest.raises(KeyError):
        lin.load_state_dict({"weight": np.zeros((2, 2)), "bias": np.zeros(2), "junk": np.zeros(1)})
    with pytest.raises(ValueError):
        lin.load_state_dict({"weight": np.zeros(2), "bias": np.zeros(2)})
    lin.load_state_dict({"bias": np.ones(2)}, strict=False)
    np.testing.assert_array_equal(lin.bias.data, [1.0, 1.0])


def test_tied_parameter_listed_once():
    """One Tensor under two names appears once, under the first name."""

    class Tied(Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = Linear(2, 2, rng=Gen())
            self.b = Linear(2, 2, rng=Gen())
            self.b.weight = self.a.weight

    t = Tied()
    t.mask = Tensor(np.ones(2))  # a constant: plain state, not a parameter
    assert [n for n, _ in t.named_parameters()] == ["a.weight", "a.bias", "b.bias"]


def test_eval_turns_off_dropout_everywhere():
    """eval() reaches nested modules (dropout becomes the identity); train() turns it back on."""
    m = Sequential(Sequential(Dropout(0.5, rng=Gen())))
    x = Tensor(np.ones((4, 4)))
    m.eval()
    np.testing.assert_array_equal(m(x).data, x.data)
    m.train()
    assert (m(x).data == 0).any()
    m.zero_grad()


def test_layernorm_matches_formula():
    """(x - mean) / sqrt(var + eps) with the population variance, then weight and bias."""
    x = np.array([[1.0, 2.0, 4.0], [0.0, 0.002, 0.001]])
    ln = LayerNorm(3)
    ln.load_state_dict({"weight": np.array([1.0, 2.0, 0.5]), "bias": np.array([0.0, 1.0, -1.0])})
    want = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * [1.0, 2.0, 0.5] + [0.0, 1.0, -1.0]
    np.testing.assert_allclose(ln(Tensor(x, dtype=np.float64)).data, want, rtol=1e-6, atol=1e-6)
