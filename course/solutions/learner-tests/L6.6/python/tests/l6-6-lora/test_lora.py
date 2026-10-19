"""My tests for L6.6 (rung R5: oracles). The oracles are numpy: the adapted
output recomputed as x W^T + b + (alpha / r) x A^T B^T, LAPACK's SVD for
PiSSA, and finite differences for the adapter gradients. They import only
the contract."""

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear, ReLU, Sequential
from tinyllm.num.rng import PCG32
from tinyllm.obj.lora import (
    LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    trainable_fraction,
)


def rand(shape, seed):
    return np.asarray(PCG32(seed).uniforms(int(np.prod(shape)))).reshape(shape) * 2 - 1


def model(seed=0):
    return Sequential(
        Linear(6, 8, rng=PCG32(seed)), ReLU(), Linear(8, 4, rng=PCG32(seed + 1))
    )


def everything(name, mod):
    return True


def test_forward_formula():
    m = LoRALinear(Linear(5, 3, rng=PCG32(1)), r=2, alpha=6.0, rng=PCG32(2))
    m.lora_B.weight.data[...] = rand((3, 2), 3)
    x = rand((4, 5), 4).astype(np.float32)
    W, b = m.weight.data.astype(np.float64), m.bias.data.astype(np.float64)
    A, B = (
        m.lora_A.weight.data.astype(np.float64),
        m.lora_B.weight.data.astype(np.float64),
    )
    want = x @ W.T + b + 3.0 * (x @ A.T) @ B.T
    np.testing.assert_allclose(m(Tensor(x)).data, want, rtol=1e-5, atol=1e-5)


def test_starts_as_the_base():
    net = model()
    x = Tensor(rand((3, 6), 9).astype(np.float32))
    want = net(x).data.copy()
    inject_lora(net, everything, r=2, alpha=4.0, rng=PCG32(5))
    assert np.array_equal(net(x).data, want)


def test_only_adapters_train_and_get_grads():
    net = model()
    inject_lora(net, lambda n, m: n == "0", r=2, alpha=2.0, rng=PCG32(5))
    net[0].lora_B.weight.data[...] = 0.2
    F.sum(net(Tensor(rand((3, 6), 1).astype(np.float32)))).backward()
    for n, p in net.named_parameters():
        assert p.requires_grad == ("lora_" in n), n
        assert (p.grad is not None) == ("lora_" in n), n
    assert trainable_fraction(net) == pytest.approx(
        (2 * 6 + 8 * 2) / (6 * 8 + 8 + 8 * 4 + 4 + 28)
    )


def test_adapter_gradients_by_finite_differences():
    m = LoRALinear(Linear(4, 3, rng=PCG32(1)), r=2, alpha=3.0, rng=PCG32(2))
    m.lora_B.weight.data[...] = rand((3, 2), 3)
    x = rand((2, 4), 5).astype(np.float32)
    F.sum(m(Tensor(x))).backward()
    W, b = m.weight.data.astype(np.float64), m.bias.data.astype(np.float64)
    A0, B0 = (
        m.lora_A.weight.data.astype(np.float64),
        m.lora_B.weight.data.astype(np.float64),
    )

    def f(A, B):
        return np.sum(x @ W.T + b + 1.5 * (x @ A.T) @ B.T)

    eps = 1e-4
    for which, X in (("A", A0), ("B", B0)):
        g = np.zeros_like(X)
        for i in np.ndindex(X.shape):
            hi, lo = X.copy(), X.copy()
            hi[i] += eps
            lo[i] -= eps
            g[i] = (
                (f(hi, B0) - f(lo, B0)) / (2 * eps)
                if which == "A"
                else (f(A0, hi) - f(A0, lo)) / (2 * eps)
            )
        got = m.lora_A.weight.grad if which == "A" else m.lora_B.weight.grad
        np.testing.assert_allclose(got, g, rtol=1e-3, atol=1e-4)


def test_pissa_matches_lapack():
    base = Linear(6, 5, rng=PCG32(7))
    W = base.weight.data.astype(np.float64).copy()
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    m = LoRALinear(base, r=2, alpha=8.0, init="pissa")
    Wr = (U[:, :2] * S[:2]) @ Vt[:2]
    np.testing.assert_allclose(m.delta_weight(), Wr, atol=1e-5)
    np.testing.assert_allclose(m.weight.data, W - Wr, atol=1e-5)
    x = Tensor(rand((3, 6), 2).astype(np.float32))
    np.testing.assert_allclose(m(x).data, x.data @ W.T + base.bias.data, atol=1e-5)


def test_merge_matches_and_removes_adapters():
    net = model(3)
    before = list(net.state_dict())
    inject_lora(net, everything, r=2, alpha=4.0, rng=PCG32(5))
    for _, mod in net.named_modules():
        if isinstance(mod, LoRALinear):
            mod.lora_B.weight.data[...] = rand(mod.lora_B.weight.shape, 8)
    x = Tensor(rand((3, 6), 9).astype(np.float32))
    want = net(x).data.copy()
    assert merge_lora(net) == ["0", "2"]
    assert type(net[0]) is Linear and type(net[2]) is Linear
    np.testing.assert_allclose(net(x).data, want, rtol=1e-5, atol=1e-5)
    assert list(net.state_dict()) == before


def test_peft_names_and_roundtrip():
    a, b = model(), model()
    inject_lora(a, everything, r=1, alpha=1.0, rng=PCG32(1))
    inject_lora(b, everything, r=1, alpha=1.0, rng=PCG32(2))
    sd = lora_state_dict(a)
    assert list(sd) == [
        f"base_model.model.{n}.lora_{w}.weight" for n in ("0", "2") for w in "AB"
    ]
    load_lora_state_dict(b, sd)
    for k, v in lora_state_dict(b).items():
        assert np.array_equal(v, sd[k])


def test_dropout_spares_the_frozen_path():
    m = LoRALinear(
        Linear(5, 3, rng=PCG32(1)), r=2, alpha=2.0, dropout=0.9, rng=PCG32(2)
    )
    x = Tensor(rand((4, 5), 3).astype(np.float32))
    want = x.data @ m.weight.data.T + m.bias.data
    np.testing.assert_allclose(m(x).data, want, rtol=1e-6, atol=1e-6)


def test_rejects_bad_rank():
    with pytest.raises(ValueError):
        LoRALinear(Linear(4, 3, rng=PCG32(0)), r=4, alpha=1.0)
    LoRALinear(Linear(4, 3, rng=PCG32(0)), r=3, alpha=1.0)
