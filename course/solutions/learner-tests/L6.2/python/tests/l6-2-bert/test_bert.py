"""My tests for L6.2 (rung R5: the masking op order replayed with my own
copy of the generator, a BERT layer written out in float64 numpy with the
exact GELU, and the padding and both-directions laws). They import only the
contract."""

import math

import numpy as np
import pytest
from tinyllm.autograd.tensor import Tensor
from tinyllm.num.rng import PCG32
from tinyllm.obj.bert import (
    BertConfig,
    BertEncoder,
    BertForMLM,
    BertLayer,
    mlm_batch,
    mlm_mask,
)
from tinyllm.tok.wordpiece import WordPieceTokenizer


class Script:
    def __init__(self, u, b):
        self.u, self.b = list(u), list(b)

    def uniform(self):
        return self.u.pop(0)

    def below(self, n):
        return self.b.pop(0)


def test_hand_example():
    rng = Script([0.05, 0.5, 0.9, 0.1, 0.85], [9])
    inp, lab = mlm_mask(
        np.array([[2, 5, 6, 7, 3, 0]]),
        np.array([[1, 0, 0, 0, 1, 1]], dtype=bool),
        4,
        40,
        0.15,
        rng,
    )
    assert inp.tolist() == [[2, 4, 6, 9, 3, 0]] and lab.tolist() == [
        [-100, 5, -100, 7, -100, -100]
    ]
    assert rng.u == [] and rng.b == []


def test_replay_op_order():
    ids = np.random.Generator(np.random.PCG64(1)).integers(5, 500, size=(3, 20))
    sp = np.zeros(ids.shape, dtype=bool)
    sp[:, 0] = True
    got = mlm_mask(ids, sp, 4, 500, 0.4, PCG32(7))
    r = PCG32(7)
    want_in, want_lab = ids.copy(), np.full(ids.shape, -100)
    for i, j in np.ndindex(ids.shape):
        if sp[i, j] or r.uniform() >= 0.4:
            continue
        want_lab[i, j] = ids[i, j]
        a = r.uniform()
        if a < 0.8:
            want_in[i, j] = 4
        elif a < 0.9:
            want_in[i, j] = r.below(500)
    assert np.array_equal(got[0], want_in) and np.array_equal(got[1], want_lab)


def test_keep_still_labelled():
    inp, lab = mlm_mask(
        np.array([[8]]), np.array([[False]]), 4, 40, 0.5, Script([0.0, 0.95], [])
    )
    assert inp.tolist() == [[8]] and lab.tolist() == [[8]]


def test_layer_matches_numpy():
    cfg = BertConfig(
        vocab=10, max_len=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0
    )
    layer = BertLayer(cfg, PCG32(2))
    r = np.random.Generator(np.random.PCG64(3))
    layer.load_state_dict(
        {
            n: r.normal(size=v.shape) * 0.7 + ("norm.weight" in n)
            for n, v in layer.state_dict().items()
        }
    )
    sd = {n: v.astype(np.float64) for n, v in layer.state_dict().items()}
    x = r.normal(size=(2, 4, 8)) * 2
    keep = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)

    def ln(z, k):
        return (z - z.mean(-1, keepdims=True)) / np.sqrt(
            z.var(-1, keepdims=True) + 1e-12
        ) * sd[k + ".weight"] + sd[k + ".bias"]

    def lin(z, k):
        return z @ sd[k + ".weight"].T + sd[k + ".bias"]

    q, k, v = (
        lin(x, f"attn.{c}_proj").reshape(2, 4, 2, 4).transpose(0, 2, 1, 3)
        for c in "qkv"
    )
    e = np.where(keep[:, None, None, :], q @ k.transpose(0, 1, 3, 2) / 2.0, -np.inf)
    a = np.exp(e - e.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    y = ln(
        x + lin((a @ v).transpose(0, 2, 1, 3).reshape(2, 4, 8), "attn.out_proj"),
        "attn_norm",
    )
    u = lin(y, "ff1")
    want = ln(
        y + lin(u * 0.5 * (1 + np.vectorize(math.erf)(u / math.sqrt(2))), "ff2"),
        "ff_norm",
    )
    np.testing.assert_allclose(
        layer(Tensor(x), keep[:, None, None, :]).data, want, rtol=1e-4, atol=1e-4
    )


def test_padding_unread_and_both_directions():
    cfg = BertConfig(
        vocab=20, max_len=8, d_model=8, n_heads=2, n_layers=2, d_ff=16, dropout=0.0
    )
    m = BertEncoder(cfg, PCG32(4))
    x = np.array([[5, 6, 7, 8, 9]])
    am = np.array([[1, 1, 1, 0, 0]], dtype=bool)
    y = x.copy()
    y[0, 3:] = 1
    np.testing.assert_array_equal(
        m(x, None, am).data[0, :3], m(y, None, am).data[0, :3]
    )
    z = x.copy()
    z[0, 2] = 11
    assert not np.allclose(m(x).data[0, 0], m(z).data[0, 0])
    assert not np.allclose(m(x, np.ones_like(x)).data, m(x).data)


def test_mlm_head_and_batch():
    cfg = BertConfig(
        vocab=20, max_len=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0
    )
    m = BertForMLM(cfg, PCG32(5))
    m.decoder_bias.data[:] = 3.0
    m2 = BertForMLM(cfg, PCG32(5))
    np.testing.assert_allclose(
        m(np.array([[5, 6]]))[0].data, m2(np.array([[5, 6]]))[0].data + 3.0, rtol=1e-5
    )
    tok = WordPieceTokenizer(
        {
            t: i
            for i, t in enumerate(
                ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b"]
            )
        }
    )
    b = mlm_batch(tok, ["a b a", "b"], 8, 1.0, PCG32(6))
    assert b["labels"].tolist() == [[-100, 5, 6, 5, -100], [-100, 6, -100, -100, -100]]
    assert b["attention_mask"].tolist() == [[True] * 5, [True] * 3 + [False] * 2]


def hf_named(seed, cfg):
    """A Hugging Face BertForMaskedLM state dict I make up, and the same
    numbers under the course's names."""
    r = np.random.Generator(np.random.PCG64(seed))
    ref = BertForMLM(cfg, PCG32(0))
    ours = {
        n: r.normal(size=v.shape) * 0.5 + ("norm.weight" in n)
        for n, v in ref.state_dict().items()
    }
    names = {
        "word_emb": "embeddings.word_embeddings",
        "pos_emb": "embeddings.position_embeddings",
        "type_emb": "embeddings.token_type_embeddings",
        "emb_norm": "embeddings.LayerNorm",
    }
    layer = {
        "attn.q_proj": "attention.self.query",
        "attn.k_proj": "attention.self.key",
        "attn.v_proj": "attention.self.value",
        "attn.out_proj": "attention.output.dense",
        "attn_norm": "attention.output.LayerNorm",
        "ff1": "intermediate.dense",
        "ff2": "output.dense",
        "ff_norm": "output.LayerNorm",
    }
    hf = {}
    for n, v in ours.items():
        if n == "decoder_bias":
            hf["cls.predictions.bias"] = v
            continue
        mod, leaf = n.rsplit(".", 1)
        if mod.startswith("bert.layers."):
            i, rest = mod[len("bert.layers.") :].split(".", 1)
            hf[f"bert.encoder.layer.{i}.{layer[rest]}.{leaf}"] = v
        elif mod.startswith("bert."):
            hf[f"bert.{names[mod[5:]]}.{leaf}"] = v
        else:
            hf[
                f"cls.predictions.transform.{'dense' if mod == 'transform' else 'LayerNorm'}.{leaf}"
            ] = v
    return ours, hf


def test_load_hf_names_and_mlm_numpy():
    from tinyllm.obj.bert import load_hf_bert

    cfg = BertConfig(
        vocab=20, max_len=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0
    )
    ours, hf = hf_named(8, cfg)
    m, enc = BertForMLM(cfg, PCG32(1)), BertEncoder(cfg, PCG32(2))
    load_hf_bert(m, hf)
    load_hf_bert(enc, hf)
    for n, v in m.state_dict().items():
        np.testing.assert_allclose(v, ours[n], rtol=1e-6, err_msg=n)
    x = np.array([[5, 6, 7]])
    np.testing.assert_array_equal(enc(x).data, m.bert(x).data)
    sd = {n: v.astype(np.float64) for n, v in m.state_dict().items()}

    def ln(z, k):
        return (z - z.mean(-1, keepdims=True)) / np.sqrt(
            z.var(-1, keepdims=True) + 1e-12
        ) * sd[k + ".weight"] + sd[k + ".bias"]

    h0 = ln(
        sd["bert.word_emb.weight"][x]
        + sd["bert.pos_emb.weight"][:3]
        + sd["bert.type_emb.weight"][0],
        "bert.emb_norm",
    )
    np.testing.assert_allclose(
        m.bert.layers[0](Tensor(h0), np.ones((1, 1, 1, 3), dtype=bool)).data,
        m.bert(x).data,
        rtol=1e-4,
        atol=1e-4,
    )
    h = m.bert(x).data.astype(np.float64)
    t = h @ sd["transform.weight"].T + sd["transform.bias"]
    t = ln(t * 0.5 * (1 + np.vectorize(math.erf)(t / math.sqrt(2))), "transform_norm")
    want = t @ sd["bert.word_emb.weight"].T + sd["decoder_bias"]
    logits, loss = m(x, labels=np.array([[-100, 9, -100]]))
    np.testing.assert_allclose(logits.data, want, rtol=1e-4, atol=1e-4)
    z = want[0, 1]
    assert float(loss.data) == pytest.approx(
        np.log(np.exp(z - z.max()).sum()) + z.max() - z[9], rel=1e-4
    )
    assert "decoder_bias" in m.state_dict()


def test_truncation_and_validation():
    tok = WordPieceTokenizer(
        {
            t: i
            for i, t in enumerate(
                ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b"]
            )
        }
    )
    assert mlm_batch(tok, ["a b a b"], 4, 0.0, PCG32(1))["input_ids"].tolist() == [
        [2, 5, 6, 3]
    ]
    with pytest.raises(ValueError):
        mlm_mask(np.array([[5]]), np.array([[False]]), 40, 40, 0.1, PCG32(1))
