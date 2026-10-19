"""My tests for L7.8 (rung R5). The oracle is a per-token loop in numpy
float64: router weights from the chapter's table, each chosen expert run on
its own row. They import only the contract."""

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.moe import MoE, combine, dispatch, load_balance_loss, topk_ids


def moe(seed=0, **kw):
    args = dict(d_ff_expert=4, n_experts=5, top_k=2)
    args.update(kw)
    m = MoE(6, **args)
    rng = np.random.Generator(np.random.PCG64(seed))
    for _, p in m.named_parameters():
        p.data = rng.normal(size=p.shape) * 0.5
    return m


def oracle(m, x):
    z = x @ m.gate.weight.data.T
    out = np.zeros_like(x)
    for n in range(x.shape[0]):
        if m.router == "sigmoid":
            s = 1 / (1 + np.exp(-z[n]))
            ids = sorted(
                range(len(s)), key=lambda e: (-(s[e] + m.e_score_correction_bias[e]), e)
            )[: m.top_k]
            w = s[ids] / (s[ids].sum() + 1e-20) if m.norm_topk else s[ids]
        elif m.router == "softmax_topk":
            p = np.exp(z[n] - z[n].max()) / np.exp(z[n] - z[n].max()).sum()
            ids = sorted(range(len(p)), key=lambda e: (-p[e], e))[: m.top_k]
            w = p[ids] / p[ids].sum() if m.norm_topk else p[ids]
        else:
            ids = sorted(range(len(z[n])), key=lambda e: (-z[n][e], e))[: m.top_k]
            w = np.exp(z[n][ids]) / np.exp(z[n][ids]).sum()
        for wj, e in zip(w * m.routed_scaling, ids):
            out[n] += wj * m.experts[e](Tensor(x[n : n + 1], dtype=np.float64)).data[0]
        if m.shared_experts is not None:
            out[n] += m.shared_experts(Tensor(x[n : n + 1], dtype=np.float64)).data[0]
    return out


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(norm_topk=False),
        dict(router="topk_softmax"),
        dict(router="sigmoid", n_shared=1, routed_scaling=2.0),
    ],
)
def test_matches_per_token_loop(kw):
    m = moe(1, **kw)
    if m.router == "sigmoid":
        m.e_score_correction_bias = np.array(
            [0.3, -0.2, 0.0, 0.5, -0.4], dtype=np.float32
        )
    x = np.random.Generator(np.random.PCG64(2)).normal(size=(7, 6))
    np.testing.assert_allclose(
        m(Tensor(x, dtype=np.float64)).data, oracle(m, x), atol=1e-10
    )


def test_dispatch_roundtrip():
    idx = np.array([[2, 0], [0, 1], [2, 2]])
    x = Tensor(np.arange(6.0).reshape(3, 2), dtype=np.float64)
    xs, off, inv = dispatch(x, idx, 3)
    assert off.tolist() == [0, 2, 3, 6]
    assert xs.data[:, 0].tolist() == [0.0, 2.0, 2.0, 0.0, 4.0, 4.0]
    np.testing.assert_allclose(
        combine(xs, Tensor(np.full((3, 2), 0.5), dtype=np.float64), inv).data, x.data
    )


def test_ties_lowest_id():
    assert topk_ids(np.array([[1.0, 2.0, 2.0]]), 1).tolist() == [[1]]


def test_balance_loss_counts():
    probs = Tensor(
        np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]),
        requires_grad=True,
        dtype=np.float64,
    )
    loss = load_balance_loss(probs, np.array([[0], [0]]), 3)
    assert loss.data == pytest.approx(3 * (1.0 * 0.65))
    loss.backward()
    np.testing.assert_allclose(probs.grad, [[1.5, 0, 0], [1.5, 0, 0]])


def test_router_gets_gradient():
    m = moe(3)
    x = np.random.Generator(np.random.PCG64(4)).normal(size=(3, 6))
    F.sum(m(Tensor(x, dtype=np.float64))).backward()
    g = m.gate.weight.grad.copy()
    w = m.gate.weight
    old = w.data[1, 2]
    w.data[1, 2] = old + 1e-6
    hi = m(Tensor(x, dtype=np.float64)).data.sum()
    w.data[1, 2] = old - 1e-6
    lo = m(Tensor(x, dtype=np.float64)).data.sum()
    w.data[1, 2] = old
    assert abs((hi - lo) / 2e-6 - g[1, 2]) < 1e-6


def test_bias_rule():
    m = moe(5, router="sigmoid", bias_update_rate=0.5)
    m.update_bias([9, 1, 5, 5, 5])
    np.testing.assert_allclose(m.e_score_correction_bias, [-0.5, 0.5, 0, 0, 0])


def test_rejects_too_many_experts_per_token():
    with pytest.raises(ValueError):
        MoE(6, 4, 3, 4)
    with pytest.raises(ValueError):
        dispatch(Tensor(np.ones((1, 2))), np.array([[3]]), 3)
