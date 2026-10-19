"""My tests for L2.2 (rung R3: tests first, `ol tdd red L2.2`, then the code,
`ol tdd green L2.2`; the chapter's section 3 numbers are the first test).
They import only contract names."""

import json
import math

import numpy as np
import pytest
from tinyllm.autograd.losses import cross_entropy
from tinyllm.lm.nplm import NPLM, load_nplm, save_nplm, train_nplm, windows
from tinyllm.nn.layers import Embedding, Linear
from tinyllm.num.rng import PCG32
from tinyllm.optim.sgd import SGD
from tinyllm.train.loop import DataLoader, train_step

T1 = math.tanh(1.0)


def hand(direct=True):
    m = NPLM(3, 2, 1, 1, direct=direct, rng=PCG32(0))
    sd = {
        "emb.weight": [[1.0], [0.0], [-1.0]],
        "hidden.weight": [[0.5, -0.5]],
        "hidden.bias": [0.0],
        "out.weight": [[1.0], [0.0], [-1.0]],
        "out.bias": [0.0, 0.5, 0.0],
    }
    if direct:
        sd["direct.weight"] = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
    m.load_state_dict(sd)
    return m


def small(seed=3, direct=True):
    return NPLM(7, 2, 3, 4, direct=direct, rng=PCG32(seed))


def test_hand_example_logits():
    """Window (a, c): x = (1, -1), h = tanh 1, logits (1 + tanh 1, 0.5, -1 - tanh 1)."""
    y = hand()(np.array([[0, 2], [2, 0]])).data
    np.testing.assert_allclose(
        y, [[1 + T1, 0.5, -1 - T1], [-1 - T1, 0.5, 1 + T1]], rtol=1e-6
    )
    np.testing.assert_allclose(
        hand(False)(np.array([[0, 2]])).data, [[T1, 0.5, -T1]], rtol=1e-6
    )


def test_windows():
    ctx, tgt = windows(np.array([5, 6, 7, 8]), 2)
    assert ctx.tolist() == [[5, 6], [6, 7]] and tgt.tolist() == [7, 8]
    with pytest.raises(ValueError):
        windows([1, 2], 2)


def test_every_parameter_gets_a_gradient():
    m = hand()
    cross_entropy(m(np.array([[0, 2]])), np.array([0])).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and np.abs(p.grad).sum() > 0, name
    y = np.array([1 + T1, 0.5, -1 - T1])
    p = np.exp(y) / np.exp(y).sum()
    np.testing.assert_allclose(m.out.bias.grad, p - [1, 0, 0], rtol=1e-5)


def test_embedding_gradient_by_finite_differences():
    m = small()
    ctx, tgt = np.array([[0, 1], [1, 1], [6, 2]]), np.array([2, 0, 5])
    cross_entropy(m(ctx), tgt).backward()
    g = m.emb.weight.grad.astype(np.float64)
    W = m.emb.weight.data
    for i, j in ((1, 0), (6, 2), (2, 1)):
        old = W[i, j]
        W[i, j] = old + 1e-2
        hi = float(cross_entropy(m(ctx), tgt).data)
        W[i, j] = old - 1e-2
        lo = float(cross_entropy(m(ctx), tgt).data)
        W[i, j] = old
        assert abs((hi - lo) / 2e-2 - g[i, j]) < 2e-3


def test_state_dict_names_and_one_generator():
    m = NPLM(7, 2, 3, 4, rng=PCG32(9))
    assert list(m.state_dict()) == [
        "emb.weight",
        "hidden.weight",
        "hidden.bias",
        "out.weight",
        "out.bias",
        "direct.weight",
    ]
    r = PCG32(9)
    layers = [
        Embedding(7, 3, rng=r),
        Linear(6, 4, rng=r),
        Linear(4, 7, rng=r),
        Linear(6, 7, bias=False, rng=r),
    ]
    np.testing.assert_array_equal(m.hidden.weight.data, layers[1].weight.data)
    np.testing.assert_array_equal(m.direct.weight.data, layers[3].weight.data)
    a = NPLM(7, 2, 3, 4)
    e = a.emb.weight.data.ravel()[:8]
    h = a.hidden.weight.data.ravel()[:8] * math.sqrt(6)
    assert np.abs(e - h).max() > 1e-3


def test_nll_and_perplexity():
    m = small()
    ids = np.array([0, 1, 2, 6, 5, 4, 3, 3, 2])
    nll = m.nll(ids)
    assert nll.shape == (7,)
    ctx, tgt = windows(ids, 2)
    np.testing.assert_allclose(
        nll.mean(), float(cross_entropy(m(ctx), tgt).data), rtol=1e-5
    )
    np.testing.assert_allclose(m.perplexity(ids), math.exp(nll.mean()), rtol=1e-12)


def test_generate_greedy_ties_and_last_window():
    flat = small()
    for p in flat.parameters():
        p.data[...] = 0.0
    assert flat.generate([4, 5], 3, 0.0, seed=0) == [0, 0, 0]
    m = small(seed=5)
    for p in m.parameters():
        p.data *= 3.0
    long = [0, 1, 2, 3, 4, 5, 6, 2, 5]
    assert m.generate(long, 6, 1.0, seed=3) == m.generate(long[-2:], 6, 1.0, seed=3)
    assert m.generate(long, 6, 0.8, seed=7) == m.generate(long, 6, 0.8, seed=7)
    with pytest.raises(ValueError):
        m.generate([1], 2, 1.0, seed=0)
    # greedy is the step-by-step argmax over the LAST window, history included
    hist = [6, 1, 4]
    for t in m.generate(hist, 6, 0.0, seed=0):
        assert t == int(np.argmax(m(np.array([hist[-2:]])).data[0]))
        hist.append(t)


def test_one_batch_of_data_is_enough():
    ids = np.array([(3 * i + i // 7) % 7 for i in range(10)])  # 8 windows
    assert len(train_nplm(small(2), ids, 3, 8, 0.1, PCG32(0))) == 3


def test_training_is_the_l05_loop_across_epochs():
    ids = np.array([(3 * i + i // 7) % 7 for i in range(38)])
    a, b = small(21), small(21)
    la = train_nplm(a, ids, 10, 8, 0.2, PCG32(4), momentum=0.9, clip=0.5)
    ctx, tgt = windows(ids, 2)
    loader = DataLoader(
        {"x": ctx, "y": tgt}, 8, shuffle=True, rng=PCG32(4), drop_last=True
    )
    opt = SGD(b.parameters(), lr=0.2, momentum=0.9)
    lb = []
    while len(lb) < 10:  # 4 batches per epoch: two epoch boundaries
        for batch in loader:
            lb.append(
                train_step(
                    b,
                    batch,
                    lambda m, x: cross_entropy(m(x["x"]), x["y"]),
                    opt,
                    clip=0.5,
                )["loss"]
            )
            if len(lb) == 10:
                break
    assert la == lb
    for k, v in b.state_dict().items():
        np.testing.assert_array_equal(a.state_dict()[k], v)


def test_loss_drops_on_a_repeating_pattern():
    ids = np.array([0, 1, 2, 3, 4, 5, 6] * 20)
    m = small(8)
    losses = train_nplm(m, ids, 60, 16, 0.5, PCG32(1), momentum=0.9)
    assert losses[-1] < 0.5 * losses[0]
    assert m.perplexity(ids) < 2.0


def test_save_load(tmp_path):
    for direct in (True, False):
        m = NPLM(9, 3, 2, 5, direct=direct, rng=PCG32(6))
        d = tmp_path / str(direct)
        save_nplm(m, str(d))
        cfg = json.loads((d / "config.json").read_text())
        assert (
            cfg["tl_arch"] == "nplm"
            and cfg["tl_direct"] == direct
            and cfg["tl_context"] == 3
        )
        back = load_nplm(str(d))
        x = np.array([[1, 2, 3], [8, 0, 8]])
        np.testing.assert_array_equal(back(x).data, m(x).data)
