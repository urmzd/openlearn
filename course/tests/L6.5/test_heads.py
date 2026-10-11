"""Course tests for L6.5: fine-tuning heads and the linear-head export
(tinyllm/obj/heads.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L6.5), and the chapter section it comes
from. Backbones are L6.2's BertEncoder and L6.1's GPT; adapters come from
L6.6; the policy head is fitted with M07.7's IRLS.

The worked examples of the chapter (section 3):
  * pooling, one row of hidden states (1, 2), (3, 4), (5, 6) with the third
    position padding: cls = (1, 2), mean = (2, 3), last = (3, 4). With an
    identity classifier and label 1, the mean-pooled logits are (2, 3) and
    the loss is ln(1 + e^-1) = 0.313262.
  * the policy head: e = (3, 4), so u = (0.6, 0.8); W = [[0, 0], [1, 2]],
    b = (0, -1): logits (0, 1.2) and p(unsafe) = sigmoid(1.2) = 0.768525.
    e = (0.03, 0.04) and e = (300, 400) give the same p.
  * a reward pair r_chosen = 2, r_rejected = 0: loss ln(1 + e^-2) = 0.126928.

course/fixtures/L6.5/linear_head.json is the head Python and Go must score
the same within 1e-6 (gw.08 reads it too). The learning test fine-tunes a
tiny BERT on course/fixtures/small-corpora/sst2-2k.tsv (a synthetic stand-in
for SST-2) and compares validation accuracy with
course/fixtures/ref-thresholds.tsv.
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
from _lib.thresholds import check
from schema_lite import errors
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module
from tinyllm.obj.bert import BertConfig, BertEncoder
from tinyllm.obj.gpt import GPT, GPTConfig
from tinyllm.obj.heads import (
    RewardHead,
    SequenceClassifier,
    TokenClassifier,
    export_linear_head,
    fit_linear_head,
    head_metrics,
    head_probs,
    load_classifier,
    load_linear_head,
    lora_classifier,
    pairwise_reward_loss,
    pool,
    predict,
    save_classifier,
    train_classifier,
)
from tinyllm.obj.lora import merge_lora, trainable_fraction
from tinyllm.prob.metrics import logistic_predict_proba, logistic_regression_fit

FIX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "formats"


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


class Rng:
    """The frozen PCG32 behind the generator API of M06.3."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.g.uniform() for _ in range(n)], dtype=np.float64)

    def below(self, n: int) -> int:
        return self.g.below(n)


class Fixed(Module):
    """A backbone that returns given hidden states (the hand examples)."""

    def __init__(self, h) -> None:
        super().__init__()
        self.h = np.asarray(h, dtype=np.float32)

    def forward(self, ids, token_type_ids=None, attn_mask=None):
        return Tensor(self.h[: len(ids)])


def bert(d=8, vocab=30, layers=1, s=1) -> BertEncoder:
    return BertEncoder(
        BertConfig(
            vocab=vocab,
            max_len=16,
            d_model=d,
            n_heads=2,
            n_layers=layers,
            d_ff=16,
            dropout=0.0,
        ),
        Rng(s),
    )


def gpt(d=8, vocab=30, s=1) -> GPT:
    return GPT(
        GPTConfig(vocab=vocab, n_ctx=16, d_model=d, n_heads=2, n_layers=1, d_ff=16),
        Rng(s),
    )


def padded(s=3, B=3, T=7, vocab=30):
    g = PCG32(seed=s)
    ids = np.array(
        [[3 + g.below(vocab - 3) for _ in range(T)] for _ in range(B)], dtype=np.int64
    )
    real = np.ones((B, T), dtype=bool)
    real[1, 4:] = False
    real[2, 2:] = False
    ids[~real] = 0
    return ids, real


# --- the worked examples ---------------------------------------------------------------


def test_hand_example_pooling_and_loss():
    # WHY: the chapter's worked example: the three pools of one padded row,
    #      and the mean-pooled classifier's logits and cross-entropy.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: L6.5 section 3, Worked example by hand
    h = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32))
    m = np.array([[1, 1, 0]])
    assert_close(pool(h, m, "cls").data, [[1.0, 2.0]], dtype="float32")
    assert_close(pool(h, m, "mean").data, [[2.0, 3.0]], dtype="float32")
    assert_close(pool(h, m, "last").data, [[3.0, 4.0]], dtype="float32")
    clf = SequenceClassifier(Fixed(h.data), 2, 2, pool="mean", rng=Rng(0))
    clf.classifier.weight.data[...] = np.eye(2)
    clf.classifier.bias.data[...] = 0.0
    logits, loss = clf(np.zeros((1, 3), dtype=np.int64), None, m, np.array([1]))
    assert_close(logits.data, [[2.0, 3.0]], dtype="float32")
    assert_close(loss.data, math.log1p(math.exp(-1.0)), dtype="float32")


def test_hand_example_linear_head_probs():
    # WHY: the gateway scores p = softmax(W u + b) on the UNIT embedding u;
    #      the hand numbers fix the formula, and rescaling e changes nothing.
    # KIND: unit
    # CATCHES: s05, m01, m05
    # CHAPTER: L6.5 section 3, Worked example by hand
    head = {
        "W": [[0.0, 0.0], [1.0, 2.0]],
        "b": [0.0, -1.0],
        "threshold": 0.5,
        "classes": ["safe", "unsafe"],
        "dim": 2,
    }
    p = head_probs(
        head, np.array([[3.0, 4.0], [0.03, 0.04], [300.0, 400.0], [0.0, 0.0]])
    )
    want = 1.0 / (1.0 + math.exp(-1.2))
    assert_close(p[:3, 1], [want] * 3, dtype="float64")
    assert_close(p.sum(axis=1), np.ones(4), dtype="float64")
    assert_close(
        p[3], [math.e / (1.0 + math.e), 1.0 / (1.0 + math.e)], dtype="float64"
    )  # u = 0: softmax(b)


def test_hand_example_reward_loss():
    # WHY: Bradley-Terry: the loss of a preference pair is -log sigmoid of
    #      the reward margin, so it only depends on r_chosen - r_rejected.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: L6.5 section 3, Worked example by hand
    loss = pairwise_reward_loss(
        Tensor(np.array([2.0, 5.0])), Tensor(np.array([0.0, 3.0]))
    )
    assert_close(loss.data, math.log1p(math.exp(-2.0)), dtype="float32")


# --- pooling and padding --------------------------------------------------------------------


def test_last_pool_reads_the_last_real_token():
    # WHY: with right padding a decoder's summary of row b is at its last REAL
    #      position, not at T - 1: reading the padding gives every short row
    #      the same meaningless vector.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: L6.5 section 5, Pitfalls
    g = gpt()
    ids, real = padded()
    h = g.hidden(ids).data
    clf = SequenceClassifier(g, 8, 3, pool="last", rng=Rng(2))
    got = pool(Tensor(h), real, "last").data
    assert_close(got, h[np.arange(3), [6, 3, 1]], dtype="float32")
    logits, _ = clf(ids, None, real)
    W, b = clf.classifier.weight.data, clf.classifier.bias.data
    assert_close(
        logits.data, h[np.arange(3), [6, 3, 1]] @ W.T + b, rtol=1e-5, atol=1e-6
    )


def test_mean_pool_ignores_padding():
    # WHY: the mean is over real tokens: the padded positions' hidden states
    #      must not enter the sum OR the count.
    # KIND: property
    # CATCHES: s02
    # CHAPTER: L6.5 section 2.1, Pooling
    g = PCG32(seed=4)
    h = g.uniform_array((3, 7, 5), -1, 1)
    _, real = padded()
    want = np.stack([h[b, real[b]].mean(axis=0) for b in range(3)])
    assert_close(pool(Tensor(h), real, "mean").data, want, rtol=1e-6, atol=1e-7)
    h2 = h.copy()
    h2[~real] = 1e3
    assert_close(pool(Tensor(h2), real, "mean").data, want, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError):
        pool(Tensor(h), np.zeros((3, 7), dtype=bool), "mean")
    with pytest.raises(ValueError):
        pool(Tensor(h), real, "max")


def test_bert_classifier_does_not_see_padding():
    # WHY: end to end with L6.2's encoder: a padded row and the same row
    #      without its padding get the same logits, whatever the padding ids.
    # KIND: property
    # CATCHES: s02, s03
    # CHAPTER: L6.5 section 2.1, Pooling
    ids, real = padded(5)
    clf = SequenceClassifier(bert(), 8, 2, pool="mean", rng=Rng(3))
    full, _ = clf(ids, None, real)
    for b in range(3):
        n = int(real[b].sum())
        alone, _ = clf(ids[b : b + 1, :n], None, real[b : b + 1, :n])
        assert_close(full.data[b], alone.data[0], rtol=1e-5, atol=1e-6)
    ids2 = ids.copy()
    ids2[~real] = 7
    assert_close(clf(ids2, None, real)[0].data, full.data, rtol=1e-5, atol=1e-6)


# --- token and reward heads -------------------------------------------------------------------


def test_token_classifier_skips_padding_and_ignored_labels():
    # WHY: a token head's loss averages over real tokens with a label;
    #      padding never counts, whatever label the batch gives it.
    # KIND: unit
    # CATCHES: s04
    # CHAPTER: L6.5 section 2.2, Token and reward heads
    ids, real = padded(6)
    tc = TokenClassifier(bert(s=2), 8, 3, rng=Rng(4))
    labels = np.array(PCG32(seed=1).uniform_array(ids.shape) * 3, dtype=np.int64)
    labels[0, 2] = -100
    logits, loss = tc(ids, None, real, labels)
    assert logits.shape == (3, 7, 3)
    z = logits.data.astype(np.float64)
    lp = z - z.max(-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    keep = real & (labels != -100)
    want = -np.mean(
        np.take_along_axis(lp, np.where(keep, labels, 0)[..., None], -1)[..., 0][keep]
    )
    assert_close(loss.data, want, rtol=1e-5, atol=1e-6)


def test_reward_head_ranks_by_the_margin():
    # WHY: a reward head is a scalar per sequence from its last real token;
    #      one gradient step on the pairwise loss raises r_chosen - r_rejected.
    # KIND: property
    # CATCHES: s01, s06, m02
    # CHAPTER: L6.5 section 2.2, Token and reward heads
    ids, real = padded(7)
    rh = RewardHead(gpt(s=3), 8, rng=Rng(5))
    assert [n for n, _ in rh.named_parameters()][-1] == "score.weight"
    r = rh(ids, real)
    assert r.shape == (3,)
    h = rh.backbone.hidden(ids).data[np.arange(3), [6, 3, 1]]
    assert_close(r.data, h @ rh.score.weight.data[0], rtol=1e-5, atol=1e-6)
    chosen, rejected = ids[[0, 1]], ids[[2, 2]]
    cm, rm = real[[0, 1]], real[[2, 2]]

    def margin():
        return float(np.sum(rh(chosen, cm).data - rh(rejected, rm).data))

    before = margin()
    loss = pairwise_reward_loss(rh(chosen, cm), rh(rejected, rm))
    loss.backward()
    for p in rh.parameters():
        if p.grad is not None:
            p.data -= 0.05 * p.grad
    assert margin() > before


# --- LoRA fine-tuning ------------------------------------------------------------------------


def test_lora_classifier_trains_adapters_and_the_head():
    # WHY: `finetune classify --lora` trains the attention adapters (L6.6)
    #      AND the new head, which has no pretrained value to keep; the rest
    #      of the backbone stays frozen.
    # KIND: unit
    # CATCHES: s07, m03
    # CHAPTER: L6.5 section 2.3, Fine-tuning with LoRA
    clf = SequenceClassifier(bert(d=16, layers=2), 16, 2, rng=Rng(6))
    names = lora_classifier(clf, r=2, alpha=4.0, rng=Rng(7))
    assert names == [
        f"backbone.layers.{i}.attn.{p}" for i in range(2) for p in ("q_proj", "v_proj")
    ]
    trainable = sorted(n for n, p in clf.named_parameters() if p.requires_grad)
    want = sorted(
        [f"{n}.lora_{w}.weight" for n in names for w in "AB"]
        + ["classifier.weight", "classifier.bias"]
    )
    assert trainable == want
    assert trainable_fraction(clf) < 0.15
    ids, real = padded(8, vocab=30)
    losses = train_classifier(
        clf,
        ids,
        np.array([0, 1, 1]),
        steps=3,
        batch_size=3,
        lr=1e-2,
        rng=Rng(1),
        attn_mask=real,
    )
    assert len(losses) == 3 and not clf.training


# --- the linear policy head ------------------------------------------------------------------


def two_clusters(n=40, d=6, s=9):
    g = PCG32(seed=s)
    mu = g.normal_array((d,))
    X = np.concatenate(
        [g.normal_array((n // 2, d)) + mu, g.normal_array((n // 2, d)) - mu]
    )
    return X, np.array([1] * (n // 2) + [0] * (n // 2))


def test_fit_linear_head_is_irls_on_unit_vectors():
    # WHY: the head is M07.7's logistic regression fitted on the UNIT rows
    #      (what the gateway will score), stored as two softmax rows with
    #      class 0 as the reference: p(class 1) is exactly the fitted sigmoid.
    # KIND: differential
    # CATCHES: s05, s08, m04, m05
    # CHAPTER: L6.5 section 2.4, The linear policy head
    X, y = two_clusters()
    head = fit_linear_head(X, y, ["safe", "unsafe"], l2=0.5)
    assert (
        head["classes"] == ["safe", "unsafe"]
        and head["dim"] == 6
        and head["threshold"] == 0.5
    )
    W, b = np.asarray(head["W"]), np.asarray(head["b"])
    assert W.shape == (2, 6) and b.shape == (2,)
    assert not W[0].any() and b[0] == 0.0
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    w = logistic_regression_fit(U, y, 0.5, 50)
    assert_close(
        head_probs(head, X)[:, 1], logistic_predict_proba(U, w), rtol=1e-9, atol=1e-12
    )
    assert_close(head_probs(head, 7.0 * X), head_probs(head, X), rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError):
        fit_linear_head(X, y, ["a", "b", "c"], l2=0.5)


def test_python_scores_equal_the_shared_fixture():
    # WHY: Python (here) and Go (gw.08) must score the exported head the same
    #      within 1e-6; this fixture is the contract between them, with a
    #      zero vector, a huge one, and a negative zero among the probes.
    # KIND: golden
    # CATCHES: s05, m01, m05
    # CHAPTER: L6.5 section 4, The interface
    f = json.loads((FIX / "L6.5" / "linear_head.json").read_text())
    p = head_probs(f["head"], np.array(f["embeddings"]))
    assert_close(p, np.array(f["probs"]), rtol=1e-12, atol=1e-12)


def test_export_validates_against_the_schema(tmp_path):
    # WHY: the gateway loads models/<id>/heads/<name>.json by the schema
    #      formats/linear-head.schema.json; every number must read back as the
    #      same float64, or Go and Python drift apart.
    # KIND: conformance
    # CATCHES: s09
    # CHAPTER: L6.5 section 4, The interface
    X, y = two_clusters(s=3)
    head = fit_linear_head(X, y, ["safe", "unsafe"], l2=1.0, threshold=0.8)
    path = tmp_path / "heads" / "policy.json"
    export_linear_head(head, "smol-135m", str(path), metrics=head_metrics(head, X, y))
    doc = json.loads(path.read_text())
    schema = json.loads((CONTRACTS / "linear-head.schema.json").read_text())
    assert errors(doc, schema) == []
    assert doc["embedding_model"] == "smol-135m" and doc["threshold"] == 0.8
    assert np.array_equal(np.array(doc["W"]), np.asarray(head["W"]))
    assert np.array_equal(np.array(doc["b"]), np.asarray(head["b"]))
    back = load_linear_head(str(path))
    assert np.array_equal(back["W"], np.asarray(head["W"]))
    with pytest.raises(ValueError):
        export_linear_head(head, "", str(tmp_path / "x.json"))
    bad = dict(head, W=np.zeros((2, 5)))
    with pytest.raises(ValueError):
        export_linear_head(bad, "m", str(tmp_path / "y.json"))


def test_head_metrics_at_the_threshold():
    # WHY: the export records held-out precision and recall AT THE
    #      THRESHOLD the gateway will use, plus M07.7's threshold-free AUC
    #      and calibration error.
    # KIND: unit
    # CATCHES: s10, m05
    # CHAPTER: L6.5 section 2.4, The linear policy head
    head = {
        "W": [[0.0], [1.0]],
        "b": [0.0, 0.0],
        "threshold": 0.7,
        "classes": ["s", "u"],
        "dim": 1,
    }
    E = np.array(
        [[1.0], [1.0], [-1.0], [-1.0], [1.0]]
    )  # p1 = sigmoid(+-1) = 0.731 or 0.269
    y = np.array([1, 0, 0, 1, 1])
    m = head_metrics(head, E, y)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(
        2 / 3
    )
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["auc"] == pytest.approx(
        3.5 / 6
    )  # 6 positive-negative pairs: 2 wins, 3 ties
    head["threshold"] = 0.75
    assert head_metrics(head, E, y)["recall"] == 0.0


# --- saving and learning --------------------------------------------------------------------


def test_save_load_classifier_roundtrip(tmp_path):
    # WHY: the zoo loads a classifier directory by tl_arch and its tl_head;
    #      the loaded model predicts exactly like the saved one. A model that
    #      still holds adapters must be merged first (L6.6).
    # KIND: property
    # CATCHES: m06
    # CHAPTER: L6.5 section 4, The interface
    ids, real = padded(9)
    clf = SequenceClassifier(bert(s=4), 8, 2, pool="cls", rng=Rng(8))
    lora_classifier(clf, r=1, alpha=1.0, rng=Rng(9))
    with pytest.raises(ValueError):
        save_classifier(clf, str(tmp_path / "c"), "bert")
    merge_lora(clf)
    save_classifier(clf, str(tmp_path / "c"), "electra", labels=["neg", "pos"])
    back, cfg = load_classifier(str(tmp_path / "c"))
    assert cfg["tl_arch"] == "electra" and cfg["tl_head"]["labels"] == ["neg", "pos"]
    assert cfg["tl_backbone"]["d_model"] == 8 and cfg["tl_head"]["pool"] == "cls"
    assert np.array_equal(clf(ids, None, real)[0].data, back(ids, None, real)[0].data)
    g = SequenceClassifier(gpt(), 8, 3, pool="last", rng=Rng(1))
    save_classifier(g, str(tmp_path / "g"), "gpt")
    gb, _ = load_classifier(str(tmp_path / "g"))
    assert np.array_equal(predict(g, ids, real), predict(gb, ids, real))


def sentiment():
    rows = [
        l.split("\t")
        for l in (FIX / "small-corpora" / "sst2-2k.tsv").read_text().splitlines()[1:]
    ]
    words = sorted({w for _, s, _ in rows for w in s.split()})
    vocab = {
        w: i + 4 for i, w in enumerate(words)
    }  # 0 [PAD], 1 [CLS], 2 [SEP], 3 [MASK]
    T = max(len(s.split()) for _, s, _ in rows) + 2
    X = np.zeros((len(rows), T), dtype=np.int64)
    for i, (_, s, _) in enumerate(rows):
        ids = [1] + [vocab[w] for w in s.split()] + [2]
        X[i, : len(ids)] = ids
    y = np.array([int(r[2]) for r in rows])
    train = np.array([r[0] == "train" for r in rows])
    return X, X != 0, y, train, len(vocab) + 4


def test_classifier_learns_sentiment():
    # WHY: the whole path learns: a tiny BERT classifier on the synthetic
    #      sentiment set (negation and "but" clauses need more than single
    #      words), 150 AdamW steps from the given seed, then validation
    #      accuracy at the reference's calibrated bar.
    # KIND: learning
    # CATCHES: s03
    # CHAPTER: L6.5 section 2.1, Pooling
    X, real, y, train, V = sentiment()
    s = seed()
    cfg = BertConfig(
        vocab=V,
        max_len=X.shape[1],
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    )
    clf = SequenceClassifier(
        BertEncoder(cfg, Rng(100 + s)), 16, 2, pool="cls", rng=Rng(200 + s)
    )
    train_classifier(
        clf, X[train], y[train], 150, 32, 3e-3, Rng(300 + s), attn_mask=real[train]
    )
    acc = float(np.mean(predict(clf, X[~train], real[~train]) == y[~train]))
    check("L6.5/test_classifier_learns_sentiment", "val_acc", acc, direction="min")


def test_validation():
    # WHY: shape and range errors are caller bugs that must fail loudly.
    # KIND: boundary
    # CATCHES: m07
    # CHAPTER: L6.5 section 4, The interface
    with pytest.raises(ValueError):
        SequenceClassifier(bert(), 8, 1)
    with pytest.raises(ValueError):
        SequenceClassifier(bert(), 8, 2, pool="max")
    with pytest.raises(ValueError):
        pairwise_reward_loss(Tensor(np.zeros(2)), Tensor(np.zeros(3)))
    with pytest.raises(ValueError):
        head_probs({"W": [[0.0, 0.0]], "b": [0.0]}, np.zeros((1, 3)))
    with pytest.raises(ValueError):
        train_classifier(
            SequenceClassifier(bert(), 8, 2),
            np.zeros((2, 3), dtype=np.int64),
            np.zeros(3),
            1,
            1,
            0.1,
            Rng(0),
        )
