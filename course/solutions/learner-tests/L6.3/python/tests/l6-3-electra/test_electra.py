"""My tests for L6.3 (rung R5: oracles). The oracles: numpy BCE and softmax
written here, the step recomputed from its pieces with a copied generator,
and hand-made logits whose samples I know. They import only the contract."""

import copy
import math

import numpy as np
from tinyllm.autograd.tensor import Tensor
from tinyllm.num.rng import PCG32
from tinyllm.obj.bert import BertConfig, mlm_mask
from tinyllm.obj.electra import ELECTRA, ElectraDiscriminator, replace_tokens, rtd_loss


def cfg(layers=1):
    return BertConfig(
        vocab=16, max_len=8, d_model=8, n_heads=2, n_layers=layers, d_ff=16, dropout=0.0
    )


def batch():
    ids = np.array([[2, 5, 6, 7, 8, 9], [3, 4, 4, 10, 0, 0]])
    real = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=bool)
    return ids, real


class Fixed:
    def __init__(self, us):
        self.us = list(us)

    def uniform(self):
        return self.us.pop(0)


def test_sampled_original_is_not_replaced():
    ids = np.array([[4, 5]])
    labels = np.array([[4, 5]])
    z = np.full((1, 2, 8), -50.0)
    z[0, 0, 4] = 50.0  # certain: the original
    z[0, 1, 7] = 50.0  # certain: something else
    corrupt, rep = replace_tokens(ids, labels, z, Fixed([0.5, 0.5]))
    assert corrupt.tolist() == [[4, 7]]
    assert rep.tolist() == [[0.0, 1.0]]


def test_one_draw_per_masked_position_only():
    ids = np.array([[1, 1, 1, 1]])
    labels = np.array([[1, -100, -100, 1]])
    z = np.zeros((1, 4, 4))
    corrupt, _ = replace_tokens(ids, labels, z, Fixed([0.8, 0.3]))
    assert corrupt.tolist() == [[3, 1, 1, 1]]


def test_rtd_loss_is_mean_bce_over_real_tokens():
    z = np.array([[0.5, -1.0, 2.0], [3.0, 0.0, -9.0]])
    y = np.array([[1, 0, 0], [1, 1, 1]], dtype=np.float32)
    real = np.array([[1, 1, 1], [1, 0, 0]], dtype=bool)
    terms = [
        math.log1p(math.exp(-a)) if b else math.log1p(math.exp(a))
        for a, b in zip(z[real], y[real])
    ]
    assert abs(float(rtd_loss(Tensor(z), y, real).data) - np.mean(terms)) < 1e-6


def test_head_is_dense_gelu_dense():
    d = ElectraDiscriminator(cfg(), rng=PCG32(1))
    ids, real = batch()
    h = d.electra(ids, None, real).data.astype(np.float64)
    W1, b1 = (
        d.discriminator_predictions.dense.weight.data,
        d.discriminator_predictions.dense.bias.data,
    )
    W2, b2 = (
        d.discriminator_predictions.dense_prediction.weight.data,
        d.discriminator_predictions.dense_prediction.bias.data,
    )
    a = h @ W1.T + b1
    g = 0.5 * a * (1 + np.vectorize(math.erf)(a / math.sqrt(2)))
    want = (g @ W2.T + b2)[..., 0]
    np.testing.assert_allclose(d(ids, None, real).data, want, rtol=1e-4, atol=1e-5)


def test_step_recomputed_from_pieces():
    ids, real = batch()
    m = ELECTRA(cfg(), cfg(2), rng=PCG32(3))
    r = PCG32(11)
    r2 = copy.deepcopy(r)
    out = m(ids, r, lam=3.0, mask_id=1, attn_mask=real, p=0.5)
    inputs, labels = mlm_mask(ids, ~real, 1, 16, 0.5, r2)
    assert np.array_equal(out["labels"], labels)
    assert (labels[~real] == -100).all()
    gl, gloss = m.generator(inputs, None, real, labels)
    corrupt, rep = replace_tokens(ids, labels, gl.data, r2)
    assert np.array_equal(out["corrupt"], corrupt)
    dloss = rtd_loss(m.discriminator(corrupt, None, real), rep, real)
    np.testing.assert_allclose(
        out["loss"].data, gloss.data + 3.0 * dloss.data, rtol=1e-5
    )


def test_generator_grads_come_from_mlm_only():
    ids, real = batch()
    a = ELECTRA(cfg(), cfg(), rng=PCG32(4), tie_embeddings=False)
    b = ELECTRA(cfg(), cfg(), rng=PCG32(4), tie_embeddings=False)
    a(ids, PCG32(5), mask_id=1, attn_mask=real, p=0.6)["loss"].backward()
    b(ids, PCG32(5), mask_id=1, attn_mask=real, p=0.6)["gen_loss"].backward()
    for pa, pb in zip(a.generator.parameters(), b.generator.parameters()):
        if pb.grad is not None:
            np.testing.assert_allclose(pa.grad, pb.grad, rtol=1e-5, atol=1e-7)


def test_embeddings_are_shared_and_saved(tmp_path):
    from tinyllm.obj.electra import load_electra, save_electra

    m = ELECTRA(cfg(), cfg(2), rng=PCG32(6))
    assert m.generator.bert.word_emb.weight is m.discriminator.electra.word_emb.weight
    save_electra(m, str(tmp_path / "e"), mask_id=1)
    back, _ = load_electra(str(tmp_path / "e"))
    assert (
        back.generator.bert.word_emb.weight
        is back.discriminator.electra.word_emb.weight
    )


def test_rtd_accuracy_over_real_tokens():
    from tinyllm.obj.electra import rtd_accuracy

    ids, real = batch()
    m = ELECTRA(cfg(), cfg(), rng=PCG32(7))
    head = m.discriminator.discriminator_predictions.dense_prediction
    head.weight.data[...] = 0.0
    head.bias.data[...] = -9.0
    acc = rtd_accuracy(m, ids, PCG32(3), mask_id=1, attn_mask=real, p=0.5)
    out = m(ids, PCG32(3), mask_id=1, attn_mask=real, p=0.5)
    assert acc.shape == (int(real.sum()),)
    assert abs(acc.mean() - (1 - out["is_replaced"][real].astype(float).mean())) < 1e-12
