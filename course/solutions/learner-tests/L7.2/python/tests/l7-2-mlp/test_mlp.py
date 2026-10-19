"""My tests for L7.2 (rung R5: oracles and my own gradient check). The oracle
is the gated MLP written out in numpy float64 with the module's own weights.
They import only the contract."""

import math

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.mlp import GatedMLP, llama_ffn_dim


def silu(z):
    return z / (1 + np.exp(-z))


def gelu_tanh(z):
    return 0.5 * z * (1 + np.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z**3)))


def oracle(m, x, act, bias):
    sd = {k: v.astype(np.float64) for k, v in m.state_dict().items()}
    b = (lambda n: sd[n + ".bias"]) if bias else (lambda n: 0.0)
    gate = x @ sd["gate_proj.weight"].T + b("gate_proj")
    up = x @ sd["up_proj.weight"].T + b("up_proj")
    a = silu(gate) if act == "silu" else gelu_tanh(gate)
    return (a * up) @ sd["down_proj.weight"].T + b("down_proj")


def test_hand_example():
    m = GatedMLP(1, 1)
    m.load_state_dict({"gate_proj.weight": [[1.0]], "up_proj.weight": [[2.0]], "down_proj.weight": [[3.0]]})
    np.testing.assert_allclose(m(Tensor([[1.0]])).data, [[6 * silu(1.0)]], rtol=1e-6)


@pytest.mark.parametrize("act,bias", [("silu", False), ("silu", True), ("gelu_tanh", False)])
def test_matches_the_formula(act, bias):
    r = np.random.Generator(np.random.PCG64(1))
    m = GatedMLP(6, 10, act=act, bias=bias)
    for p in m.parameters():
        p.data = r.uniform(-1, 1, size=p.data.shape).astype(np.float32)
    x = 2 * r.normal(size=(2, 3, 6))
    np.testing.assert_allclose(m(Tensor(x)).data, oracle(m, x, act, bias), rtol=1e-4, atol=1e-5)


def test_names():
    assert [n for n, _ in GatedMLP(2, 3).named_parameters()] == ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]


def test_gradients_by_finite_differences():
    r = np.random.Generator(np.random.PCG64(2))
    m = GatedMLP(3, 4)
    for p in m.parameters():
        p.data = r.normal(size=p.data.shape)
    x0, gy = r.normal(size=(2, 3)), r.normal(size=(2, 3))
    x = Tensor(x0, requires_grad=True, dtype=np.float64)
    F.sum(m(x) * gy).backward()
    for name, p in [("x", None)] + list(m.named_parameters()):
        arr = x0 if p is None else p.data
        got = x.grad if p is None else p.grad
        num = np.zeros_like(arr)
        for i in np.ndindex(arr.shape):
            old = arr[i]
            arr[i] = old + 1e-6
            hi = np.sum(m(Tensor(x0, dtype=np.float64)).data * gy)
            arr[i] = old - 1e-6
            lo = np.sum(m(Tensor(x0, dtype=np.float64)).data * gy)
            arr[i] = old
            num[i] = (hi - lo) / 2e-6
        np.testing.assert_allclose(got, num, rtol=1e-5, atol=1e-8, err_msg=name)


def test_llama_ffn_dim():
    assert llama_ffn_dim(4096) == 11008
    assert llama_ffn_dim(4096, 1024, 1.3) == 14336
