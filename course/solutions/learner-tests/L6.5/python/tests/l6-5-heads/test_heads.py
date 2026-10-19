"""My tests for L6.5 (rung R5: oracles). The oracles are numpy: pools,
softmax, cross-entropy, and the logistic fit recomputed by hand or through
M07.7's own functions. They import only the contract."""

import math

import numpy as np
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.obj.bert import BertConfig, BertEncoder
from tinyllm.obj.heads import (
    SequenceClassifier,
    TokenClassifier,
    fit_linear_head,
    head_probs,
    lora_classifier,
    pairwise_reward_loss,
    pool,
)
from tinyllm.prob.metrics import logistic_predict_proba, logistic_regression_fit


class Const(Module):
    def __init__(self, h):
        super().__init__()
        self.h = np.asarray(h, dtype=np.float32)

    def forward(self, ids, token_type_ids=None, attn_mask=None):
        return Tensor(self.h)


H = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
M = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)


def enc():
    return BertEncoder(
        BertConfig(
            vocab=20, max_len=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0
        ),
        PCG32(1),
    )


def test_pools():
    np.testing.assert_allclose(pool(Tensor(H), M, "cls").data, H[:, 0])
    np.testing.assert_allclose(pool(Tensor(H), M, "last").data, H[[0, 1], [2, 1]])
    np.testing.assert_allclose(
        pool(Tensor(H), M, "mean").data, [H[0, :3].mean(0), H[1, :2].mean(0)], rtol=1e-6
    )


def test_encoder_gets_the_mask():
    ids = np.array([[3, 4, 5, 0], [6, 7, 0, 0]])
    clf = SequenceClassifier(enc(), 8, 2, pool="mean", rng=PCG32(2))
    a, _ = clf(ids, None, M)
    ids2 = np.where(M, ids, 9)
    b, _ = clf(ids2, None, M)
    np.testing.assert_allclose(a.data, b.data, rtol=1e-5, atol=1e-6)


def test_token_loss_skips_padding():
    tc = TokenClassifier(Const(H), 3, 3, rng=PCG32(3))
    labels = np.array([[0, 1, 2, 2], [1, 0, 2, 2]])
    logits, loss = tc(np.zeros((2, 4), dtype=np.int64), None, M, labels)
    z = logits.data.astype(np.float64)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    want = -np.mean(
        [lp[b, t, labels[b, t]] for b in range(2) for t in range(4) if M[b, t]]
    )
    assert abs(float(loss.data) - want) < 1e-5


def test_reward_loss_sign():
    loss = pairwise_reward_loss(Tensor(np.array([1.0])), Tensor(np.array([-1.0])))
    assert abs(float(loss.data) - math.log1p(math.exp(-2.0))) < 1e-6


def test_head_matches_irls_on_unit_rows():
    g = PCG32(5)
    X = np.asarray(g.uniforms(60)).reshape(20, 3) * 4 - 2
    y = (X[:, 0] + 0.3 * X[:, 2] > 0).astype(int)
    head = fit_linear_head(X, y, ["a", "b"], l2=1.0)
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    w = logistic_regression_fit(U, y, 1.0, 50)
    np.testing.assert_allclose(
        head_probs(head, X)[:, 1], logistic_predict_proba(U, w), rtol=1e-8
    )
    np.testing.assert_allclose(
        head_probs(head, 10 * X), head_probs(head, X), rtol=1e-10
    )


def test_lora_keeps_the_head_trainable():
    clf = SequenceClassifier(enc(), 8, 2, rng=PCG32(4))
    lora_classifier(clf, r=1, alpha=1.0, rng=PCG32(5))
    train = {n for n, p in clf.named_parameters() if p.requires_grad}
    assert {"classifier.weight", "classifier.bias"} <= train
    assert all("lora_" in n or n.startswith("classifier.") for n in train)


def test_export_reads_back_exactly(tmp_path):
    from tinyllm.obj.heads import export_linear_head, load_linear_head

    g = PCG32(8)
    X = np.asarray(g.uniforms(40)).reshape(10, 4) - 0.5
    y = np.array([0, 1] * 5)
    head = fit_linear_head(X, y, ["safe", "unsafe"], l2=0.3)
    export_linear_head(head, "m", str(tmp_path / "h.json"))
    back = load_linear_head(str(tmp_path / "h.json"))
    assert np.array_equal(back["W"], np.asarray(head["W"]))
    assert np.isfinite(head_probs(head, np.zeros((1, 4)))).all()


def test_metrics_use_the_threshold():
    from tinyllm.obj.heads import head_metrics

    head = {
        "W": [[0.0], [1.0]],
        "b": [0.0, 0.0],
        "threshold": 0.8,
        "classes": ["s", "u"],
        "dim": 1,
    }
    m = head_metrics(head, np.array([[1.0], [-1.0]]), np.array([1, 0]))
    assert m["recall"] == 0.0


def test_reward_shape_and_default_targets():
    from tinyllm.obj.heads import RewardHead

    r = RewardHead(Const(H), 3, rng=PCG32(1))
    assert r(np.zeros((2, 4), dtype=np.int64), M).shape == (2,)
    clf = SequenceClassifier(enc(), 8, 2, rng=PCG32(4))
    names = lora_classifier(clf, r=1, alpha=1.0, rng=PCG32(5))
    assert names and all(n.endswith((".q_proj", ".v_proj")) for n in names)


def test_gpt_classifier_roundtrip_and_pool_check(tmp_path):
    import pytest
    from tinyllm.obj.gpt import GPT, GPTConfig
    from tinyllm.obj.heads import load_classifier, save_classifier

    g = SequenceClassifier(
        GPT(
            GPTConfig(vocab=20, n_ctx=8, d_model=8, n_heads=2, n_layers=1, d_ff=16),
            PCG32(2),
        ),
        8,
        2,
        pool="last",
        rng=PCG32(3),
    )
    save_classifier(g, str(tmp_path / "g"), "gpt")
    back, _ = load_classifier(str(tmp_path / "g"))
    ids = np.array([[3, 4, 5, 0], [6, 7, 0, 0]])
    assert np.array_equal(back(ids, None, M)[0].data, g(ids, None, M)[0].data)
    with pytest.raises(ValueError):
        SequenceClassifier(enc(), 8, 2, pool="max")
