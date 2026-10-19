"""My tests for L6.7 (rung R5: oracles). The oracles: a toy model whose NLL
tells how much context each token had, hand-computed log-likelihoods, M07.4
and M07.5 called directly, and checkpoints whose scores I know exactly.
They import only the contract."""

import json
import math
import struct

import numpy as np
import pytest
from tinyllm.eval.lm import (
    ByteTokenizer,
    Result,
    compare,
    eval_ppl,
    score_choices,
    token_nlls,
)
from tinyllm.eval.zoo import run_zoo
from tinyllm.io.safetensors import save_safetensors
from tinyllm.num.rng import PCG32
from tinyllm.prob.stats import mean_ci
from tinyllm.prob.tests import paired_permutation_test
from tinyllm.seq2seq.model import Seq2Seq, save_seq2seq


class Counter:
    """p(next) = 1 / (j + 2) at window position j, ids are 0..n-1."""

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        x = np.asarray(x)[0]
        z = np.full((x.size, self.n + 1), -40.0)
        for j in range(x.size):
            nxt = min(x[0] + j + 1, self.n)
            z[j] = math.log((1 - 1 / (j + 2)) / self.n)
            z[j, nxt] = math.log(1 / (j + 2))
        return z[None]


def test_windows():
    nll = token_nlls(Counter(6), np.arange(6), 4, 2)
    np.testing.assert_allclose(nll, np.log([2, 3, 4, 3, 4]), rtol=1e-9)
    with pytest.raises(ValueError):
        token_nlls(Counter(6), np.arange(6), 4, 4)


def test_ppl_interval():
    ids = np.arange(9)
    r = eval_ppl(Counter(9), ids, 4, 2, n_bytes=8)
    nll = token_nlls(Counter(9), ids, 4, 2)
    _, lo, hi = mean_ci(nll)
    np.testing.assert_allclose([r["ppl_lo"], r["ppl_hi"]], np.exp([lo, hi]))
    np.testing.assert_allclose(r["bpb"], nll.sum() / (8 * math.log(2)))


class Bigram:
    def __init__(self):
        P = np.full((256, 256), 1e-3)
        P[ord("a"), ord("x")] = 0.5
        P[ord("a"), ord("y")] = 0.2
        P[ord("y"), ord("y")] = 0.9
        self.L = np.log(P / P.sum(1, keepdims=True))

    def __call__(self, x):
        return self.L[np.asarray(x)[0]][None]


def test_choices():
    m = Bigram()
    raw = score_choices(m, ByteTokenizer(), "a", ["x", "yy"], normalize=False)
    np.testing.assert_allclose(raw, [m.L[97, 120], m.L[97, 121] + m.L[121, 121]])
    norm = score_choices(m, ByteTokenizer(), "a", ["x", "yy"], normalize=True)
    np.testing.assert_allclose(norm, [raw[0], raw[1] / 2])


def test_compare_is_paired():
    a = np.array([1.0, 0, 1, 1, 0, 1, 1, 1])
    b = np.array([0.0, 0, 1, 0, 0, 1, 0, 1])
    got = compare(Result(0, 0, 0, 8, a), Result(0, 0, 0, 8, b), 300, PCG32(5))
    assert got == paired_permutation_test(a, b, 300, PCG32(5))


def tokens(path, data):
    path.write_bytes(
        struct.pack("<256i", *([20240520, 1, len(data), 256] + [0] * 252))
        + np.array(data, dtype="<u2").tobytes()
    )


def test_zoo_uniform_bigram_and_eos(tmp_path):
    tokens(tmp_path / "t.bin", list(b"abcd"))
    (tmp_path / "b").mkdir()
    save_safetensors(
        str(tmp_path / "b" / "model.safetensors"),
        {"bigram.weight": np.zeros((256, 256), np.float32)},
        {"format": "tinyllm"},
    )
    (tmp_path / "b" / "config.json").write_text(
        json.dumps({"tl_arch": "bigram", "tl_tokenizer": "bytes"})
    )
    s = Seq2Seq(6, 6, 3, 4, rng=PCG32(1))
    s.out.bias.data[2] = 60.0
    save_seq2seq(s, str(tmp_path / "s"))
    (tmp_path / "r.jsonl").write_text(json.dumps({"src": [3, 4], "tgt": []}) + "\n")
    models = [
        {"id": "b", "dir": "b", "task": "t", "data": "t.bin"},
        {
            "id": "s",
            "dir": "s",
            "task": "r",
            "data": "r.jsonl",
            "bos": 1,
            "eos": 2,
            "beam": 2,
            "max_len": 3,
        },
    ]
    (tmp_path / "z.json").write_text(json.dumps({"models": models}))
    rows = run_zoo(str(tmp_path / "z.json"), PCG32(0))["rows"]
    assert abs(rows[0]["value"] - 8.0) < 1e-9
    assert rows[1]["value"] == 1.0


class Calls(Counter):
    def __init__(self, n):
        super().__init__(n)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return super().__call__(x)


def test_window_count():
    m = Calls(10)
    token_nlls(m, np.arange(10), 4, 3)
    assert m.calls == 3  # [0,4) [3,7) [6,10)


def test_ties_go_first(tmp_path):
    from tinyllm.eval.lm import run_task

    t = tmp_path / "t.jsonl"
    t.write_text(json.dumps({"context": "a", "choices": ["x", "x"], "label": 0}) + "\n")
    assert (
        run_task(Bigram(), ByteTokenizer(), str(t), "acc", PCG32(0), n_boot=10).value
        == 1.0
    )


def zoo_bigram(tmp_path, W):
    (tmp_path / "b").mkdir()
    save_safetensors(
        str(tmp_path / "b" / "model.safetensors"),
        {"bigram.weight": W.astype(np.float32)},
        {"format": "tinyllm"},
    )
    (tmp_path / "b" / "config.json").write_text(
        json.dumps({"tl_arch": "bigram", "tl_tokenizer": "bytes"})
    )


def test_zoo_rows_against_my_own_numbers(tmp_path):
    from tinyllm.eval.zoo import save_word2vec
    from tinyllm.obj.gpt import GPT, GPTConfig, save_gpt

    data = list(b"abcabd" * 4)
    tokens(tmp_path / "t.bin", data)
    W = np.zeros((256, 256))
    W[ord("a"), ord("b")] = 3.0
    W[ord("b"), ord("c")] = 1.0
    zoo_bigram(tmp_path, W)
    gpt = GPT(
        GPTConfig(vocab=256, n_ctx=8, d_model=8, n_heads=2, n_layers=1, d_ff=16),
        PCG32(1),
    )
    save_gpt(gpt, str(tmp_path / "g"))
    words = ["cat", "dog", "fox", "pie", "jam"]
    E = np.array([[1, 0], [0.9, 0.1], [0.8, 0.3], [0, 1], [0.2, 0.9]])
    save_word2vec(E, words, str(tmp_path / "w"))
    (tmp_path / "s.tsv").write_text(
        "cat\tdog\t3\ncat\tpie\t0\ndog\tfox\t3\nfox\tjam\t1\npie\tjam\t3\n"
    )
    models = [
        {"id": "b", "dir": "b", "task": "t", "data": "t.bin"},
        {"id": "g", "dir": "g", "task": "t", "data": "t.bin", "stride": 3},
        {"id": "w", "dir": "w", "task": "s", "data": "s.tsv"},
        {"id": "x", "dir": "missing", "task": "t", "data": "t.bin"},
    ]
    (tmp_path / "z.json").write_text(json.dumps({"models": models}))
    rep = run_zoo(str(tmp_path / "z.json"), PCG32(0), seed=7)
    assert rep["seed"] == 7
    rows = rep["rows"]
    lsm = W - np.log(np.exp(W).sum(1, keepdims=True))
    ids = np.array(data)
    nll = -lsm[ids[:-1], ids[1:]]
    _, lo, hi = mean_ci(nll)
    np.testing.assert_allclose(rows[0]["value"], nll.mean() / math.log(2), rtol=1e-6)
    np.testing.assert_allclose(
        rows[0]["ci95"], [lo / math.log(2), hi / math.log(2)], rtol=1e-6
    )
    g = token_nlls(gpt, ids, 8, 3)
    np.testing.assert_allclose(rows[1]["value"], g.mean() / math.log(2), rtol=1e-6)
    rho = rows[2]["value"]
    z = math.atanh(rho)
    np.testing.assert_allclose(
        rows[2]["ci95"],
        [
            math.tanh(z - 1.959963984540054 / math.sqrt(2)),
            math.tanh(z + 1.959963984540054 / math.sqrt(2)),
        ],
        rtol=1e-6,
    )
    assert rows[3]["status"] == "error"


def test_classifier_rows_use_val_split(tmp_path):
    from tinyllm.eval.zoo import encode_batch
    from tinyllm.obj.bert import BertConfig, BertEncoder
    from tinyllm.obj.heads import SequenceClassifier, predict, save_classifier

    clf = SequenceClassifier(
        BertEncoder(
            BertConfig(
                vocab=256,
                max_len=20,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=16,
                dropout=0.0,
            ),
            PCG32(1),
        ),
        8,
        2,
        rng=PCG32(2),
    )
    save_classifier(clf, str(tmp_path / "c"), "bert")
    rows = [("train", "zz", 1), ("val", "good", 1), ("val", "bad", 0), ("val", "ok", 1)]
    (tmp_path / "d.tsv").write_text(
        "split\tsentence\tlabel\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in rows)
    )
    (tmp_path / "z.json").write_text(
        json.dumps({"models": [{"id": "c", "dir": "c", "task": "s", "data": "d.tsv"}]})
    )
    r = run_zoo(str(tmp_path / "z.json"), PCG32(0))["rows"][0]
    ids, real = encode_batch(["good", "bad", "ok"], 20)
    assert r["n"] == 3
    assert r["value"] == np.mean(predict(clf, ids, real) == np.array([1, 0, 1]))


def test_decode_label(tmp_path):
    s = Seq2Seq(6, 6, 3, 4, rng=PCG32(1))
    save_seq2seq(s, str(tmp_path / "s"))
    (tmp_path / "r.jsonl").write_text(json.dumps({"src": [3, 4], "tgt": [3]}) + "\n")
    models = [
        {
            "id": f"s{b}",
            "dir": "s",
            "task": "r",
            "data": "r.jsonl",
            "bos": 1,
            "eos": 2,
            "beam": b,
            "max_len": 3,
        }
        for b in (1, 2)
    ]
    (tmp_path / "z.json").write_text(json.dumps({"models": models}))
    rows = run_zoo(str(tmp_path / "z.json"), PCG32(0))["rows"]
    assert [r["decode"] for r in rows] == ["greedy", "beam"]
