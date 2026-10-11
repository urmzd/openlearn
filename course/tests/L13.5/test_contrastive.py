"""Course tests for L13.5: clip_loss, siglip_loss, TextEncoder, DualEncoder,
zero_shot_classifier, and zero_shot_predict (tinyllm/vision/contrastive.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L13.5), and the chapter section it comes from.

The chapter's worked example (section 3): two pairs in two dimensions,
images (1, 0) and (0, 1), texts (1, 0) and (0.6, 0.8). The cosine matrix
(texts on rows) is C = [[1, 0], [0.6, 0.8]]. With logit_scale = ln 5 the
CLIP logits are [[5, 0], [3, 4]]: the text rows cost ln(1 + e^-5) and
ln(1 + e^-1), the image columns ln(1 + e^-2) and ln(1 + e^-4), and the loss
is their mean, 0.116264. With t = ln 5 and b = -3 the SigLIP logits are
[[2, -3], [0, 1]]; times the signs [[+, -], [-, +]] they are 2, 3, 0, 1, and
the loss is (ln(1 + e^-2) + ln(1 + e^-3) + ln 2 + ln(1 + e^-1)) / 2 =
0.590962. The golden values in $TINYLLM_FIXTURES/L13.5/contrastive_hf.npz
come from course/oracle/L13.5/contrastive_hf.py (transformers 5.19.0
CLIPModel and SiglipModel losses and their autograd gradients, float64).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from _lib.thresholds import check
from tinyllm.autograd.tensor import Tensor
from tinyllm.vision.contrastive import (
    DualEncoder,
    TextEncoder,
    clip_loss,
    siglip_loss,
    zero_shot_classifier,
    zero_shot_predict,
)
from tinyllm.vision.vit import ViT, ViTConfig

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L13.5" / "contrastive_hf.npz"
LN100 = math.log(100.0)


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


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def t64(a, grad: bool = False) -> Tensor:
    return Tensor(np.asarray(a, dtype=np.float64), requires_grad=grad, dtype=np.float64)


def scalar(x: Tensor) -> float:
    assert x.data.size == 1, f"the loss is one number, got shape {x.shape}"
    return float(x.data.reshape(()))


def unit_rows(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def tiny_model(loss: str = "clip", s: int = 1, **vit_over) -> DualEncoder:
    kw = dict(
        image_size=4, patch_size=2, n_channels=3, d=8, n_layers=1, n_heads=2, d_ff=16
    )
    kw.update(vit_over)
    r = Rng(s)
    vision = ViT(ViTConfig(**kw), rng=r)
    text = TextEncoder(vocab=12, d=8, n_layers=1, n_heads=2, d_ff=16, max_len=6, rng=r)
    return DualEncoder(vision, text, 6, loss, math.log(10.0), -10.0, rng=r)


def no_grad_reached(t: Tensor) -> bool:
    return t.grad is None or not np.any(t.grad)


HAND_IMG = np.array([[1.0, 0.0], [0.0, 1.0]])
HAND_TXT = np.array([[1.0, 0.0], [0.6, 0.8]])


def test_hand_example_two_pairs():
    # WHY: section 3's worked example, both losses on two pairs. The rows are
    #      handed over unnormalized (lengths 2, 0.5, 3, 5) because the losses
    #      score directions: normalizing is their job, not the caller's.
    # KIND: unit, smoke
    # CATCHES: s01, s02, s03, s05, s06, s08, s11
    # CHAPTER: L13.5 section 3
    scale = np.array([[2.0], [0.5]])
    img, txt = t64(HAND_IMG * scale), t64(HAND_TXT * np.array([[3.0], [5.0]]))
    t = t64([math.log(5.0)])
    l1p = lambda x: math.log1p(math.exp(-x))  # noqa: E731
    want_clip = ((l1p(5) + l1p(1)) / 2 + (l1p(2) + l1p(4)) / 2) / 2
    assert_close(scalar(clip_loss(img, txt, t)), want_clip, rtol=1e-9, atol=1e-12)
    assert_close(want_clip, 0.116264, rtol=0.0, atol=1e-6)
    want_sig = (l1p(2) + l1p(3) + math.log(2.0) + l1p(1)) / 2
    assert_close(
        scalar(siglip_loss(img, txt, t, t64([-3.0]))), want_sig, rtol=1e-9, atol=1e-12
    )
    assert_close(want_sig, 0.590962, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("case", ["clip.b4", "clip.b1"])
def test_clip_loss_matches_hf(case):
    # WHY: the loss a published CLIP was trained with: Hugging Face's
    #      CLIPModel normalizes, scales by exp(logit_scale), and averages
    #      the two cross-entropies. Values and all three gradients must
    #      agree, or a checkpoint fine-tunes toward a different optimum. At
    #      B = 1 each softmax has one class, so the loss and every gradient
    #      are 0.
    # KIND: golden
    # CATCHES: s01, s02, s03, s16
    # CHAPTER: L13.5 section 2.2
    z = np.load(GOLDEN)
    img, txt = t64(z[f"{case}.img"], True), t64(z[f"{case}.txt"], True)
    t = t64(z[f"{case}.t"].reshape(1), True)
    loss = clip_loss(img, txt, t)
    assert_close(scalar(loss), float(z[f"{case}.loss"]), rtol=1e-8, atol=1e-10)
    loss.backward()
    if case == "clip.b1":
        assert no_grad_reached(img) and no_grad_reached(txt) and no_grad_reached(t)
        return
    assert_close(img.grad, z[f"{case}.d_img"], rtol=1e-7, atol=1e-10, msg="d img")
    assert_close(txt.grad, z[f"{case}.d_txt"], rtol=1e-7, atol=1e-10, msg="d txt")
    assert_close(
        t.grad.reshape(()), z[f"{case}.d_t"], rtol=1e-7, atol=1e-10, msg="d logit_scale"
    )


@pytest.mark.parametrize("case", ["siglip.b5", "siglip.b3", "siglip.b1"])
def test_siglip_loss_matches_hf(case):
    # WHY: SiglipModel's loss: every one of the B^2 pairs is a yes-or-no
    #      question, summed and divided by B (not B^2). Checked at the
    #      published init (t = ln 10, b = -10), at a trained-looking point,
    #      and at B = 1, with the gradients of t and b.
    # KIND: golden
    # CATCHES: s01, s05, s06, s08, s11, s15
    # CHAPTER: L13.5 section 2.3
    z = np.load(GOLDEN)
    img, txt = t64(z[f"{case}.img"], True), t64(z[f"{case}.txt"], True)
    t, b = t64(z[f"{case}.t"].reshape(1), True), t64(z[f"{case}.b"].reshape(1), True)
    loss = siglip_loss(img, txt, t, b)
    assert_close(scalar(loss), float(z[f"{case}.loss"]), rtol=1e-8, atol=1e-10)
    loss.backward()
    assert_close(img.grad, z[f"{case}.d_img"], rtol=1e-7, atol=1e-10, msg="d img")
    assert_close(txt.grad, z[f"{case}.d_txt"], rtol=1e-7, atol=1e-10, msg="d txt")
    assert_close(t.grad.reshape(()), z[f"{case}.d_t"], rtol=1e-7, atol=1e-10, msg="d t")
    assert_close(b.grad.reshape(()), z[f"{case}.d_b"], rtol=1e-7, atol=1e-10, msg="d b")


def test_gradcheck_clip_loss():
    # WHY: the loss is the last op before backward: its gradient reaches both
    #      embeddings through the normalization and reaches the temperature.
    #      Proven against central differences on unnormalized rows.
    # KIND: gradcheck
    # CATCHES: s16
    # CHAPTER: L13.5 section 2.2
    g = PCG32(seed=seed() + 1)
    i0, x0, s0 = g.normal_array((3, 4)), g.normal_array((3, 4)), np.array([1.2])

    def f(i, x, s):
        return scalar(clip_loss(t64(i), t64(x), t64(s)))

    img, txt, t = t64(i0, True), t64(x0, True), t64(s0, True)
    clip_loss(img, txt, t).backward()
    gradcheck(
        f,
        [i0, x0, s0],
        [img.grad, txt.grad, t.grad],
        names=["img", "txt", "logit_scale"],
    )


def test_gradcheck_siglip_loss():
    # WHY: log sigmoid is the one op this module adds to the op library, so
    #      its backward (sigmoid(-x)) is checked here, through the loss, for
    #      the embeddings, t, and b.
    # KIND: gradcheck
    # CATCHES: s15
    # CHAPTER: L13.5 section 2.3
    g = PCG32(seed=seed() + 2)
    i0, x0 = g.normal_array((3, 4)), g.normal_array((3, 4))
    t0, b0 = np.array([0.9]), np.array([-0.4])

    def f(i, x, t, b):
        return scalar(siglip_loss(t64(i), t64(x), t64(t), t64(b)))

    img, txt, t, b = t64(i0, True), t64(x0, True), t64(t0, True), t64(b0, True)
    siglip_loss(img, txt, t, b).backward()
    gradcheck(
        f,
        [i0, x0, t0, b0],
        [img.grad, txt.grad, t.grad, b.grad],
        names=["img", "txt", "t", "b"],
    )


def test_clip_loss_is_symmetric_in_modalities():
    # WHY: InfoNCE is the mean of two classification problems, texts picking
    #      images and images picking texts, so swapping the two arguments
    #      changes nothing; a loss with one direction only is not symmetric.
    #      Reordering the pairs together also changes nothing.
    # KIND: property
    # CATCHES: s02
    # CHAPTER: L13.5 section 2.2
    g = PCG32(seed=seed() + 3)
    a, b = g.normal_array((5, 6)), g.normal_array((5, 6))
    t = t64([math.log(8.0)])
    ab = scalar(clip_loss(t64(a), t64(b), t))
    assert_close(scalar(clip_loss(t64(b), t64(a), t)), ab, rtol=1e-9, atol=1e-12)
    perm = np.array([3, 0, 4, 1, 2])
    assert_close(
        scalar(clip_loss(t64(a[perm]), t64(b[perm]), t)), ab, rtol=1e-9, atol=1e-12
    )
    one_way = scalar(clip_loss(t64(a), t64(a[perm]), t))
    assert abs(one_way - ab) > 1e-6  # the pairing itself does matter


def test_siglip_defined_at_batch_one():
    # WHY: a softmax over one candidate is constant, so CLIP learns nothing
    #      from a batch of one (loss 0). SigLIP's question "is this a pair?"
    #      still has an answer: the loss is -log sigmoid(e^t cos + b) and its
    #      gradient is not zero. That independence from the batch is what
    #      lets SigLIP train with small or sharded batches.
    # KIND: boundary
    # CATCHES: s01, s08, s11
    # CHAPTER: L13.5 section 2.3
    img, txt = t64([[3.0, 4.0]], True), t64([[4.0, 3.0]], True)  # cosine 24/25
    t, b = t64([math.log(10.0)], True), t64([-10.0], True)
    loss = siglip_loss(img, txt, t, b)
    want = math.log1p(math.exp(-(10.0 * 0.96 - 10.0)))
    assert_close(scalar(loss), want, rtol=1e-9, atol=1e-12)
    loss.backward()
    assert_close(
        b.grad.reshape(()), -1.0 / (1.0 + math.exp(-0.4)), rtol=1e-9, atol=1e-12
    )
    assert np.any(img.grad) and np.any(txt.grad)
    clip = clip_loss(t64([[3.0, 4.0]]), t64([[4.0, 3.0]]), t64([2.0]))
    assert_close(scalar(clip), 0.0, rtol=0.0, atol=1e-12)


def test_temperature_gradient_sign():
    # WHY: logit_scale is log(1 / temperature): logits are cosines TIMES
    #      exp(logit_scale). When every pair already matches best, a sharper
    #      softmax lowers the loss, so d loss / d logit_scale < 0; when the
    #      pairing is wrong, sharpening costs, so it is > 0. Dividing by
    #      exp(logit_scale) flips both signs and training cools forever.
    # KIND: property
    # CATCHES: s03, s08
    # CHAPTER: L13.5 section 2.4
    g = PCG32(seed=seed() + 4)
    img = unit_rows(
        np.eye(6, 8) + 0.02 * g.normal_array((6, 8))
    )  # six separate directions
    txt = unit_rows(img + 0.02 * g.normal_array((6, 8)))  # near-perfect pairs
    wrong = np.roll(txt, 1, axis=0)  # every text paired with the wrong image
    for pairs, sign in ((txt, -1.0), (wrong, +1.0)):
        t = t64([math.log(10.0)], True)
        clip_loss(t64(img), t64(pairs), t).backward()
        assert sign * float(t.grad.reshape(())) > 0, ("clip", sign, t.grad)
        t = t64([math.log(10.0)], True)
        siglip_loss(t64(img), t64(pairs), t, t64([-10.0])).backward()
        assert sign * float(t.grad.reshape(())) > 0, ("siglip", sign, t.grad)


def test_clip_temperature_is_clamped_at_100():
    # WHY: nothing in the loss stops exp(logit_scale) from growing: on a
    #      batch the model already separates, a larger scale always lowers
    #      the loss. CLIP caps the scale at 100; past the cap the loss is the
    #      loss at exactly 100 and logit_scale gets no gradient.
    # KIND: boundary
    # CATCHES: s03, s04, s16
    # CHAPTER: L13.5 section 2.4
    g = PCG32(seed=seed() + 5)
    img, txt = g.normal_array((4, 5)), g.normal_array((4, 5))
    at_cap = scalar(clip_loss(t64(img), t64(txt), t64([LN100])))
    over = t64([LN100 + 1.5], True)
    loss = clip_loss(t64(img, True), t64(txt), over)
    assert_close(scalar(loss), at_cap, rtol=1e-9, atol=1e-10)
    loss.backward()
    assert no_grad_reached(over), over.grad
    under = t64([math.log(50.0)], True)
    low = clip_loss(t64(img), t64(txt), under)
    low.backward()
    assert abs(scalar(low) - at_cap) > 1e-6 and np.any(under.grad)


def test_siglip_loss_is_stable_at_large_logits():
    # WHY: with a scale of 2000 the logits reach +-2000. log(sigmoid(x))
    #      computed as written overflows to -inf at x = -2000; the stable
    #      form min(x, 0) - log1p(exp(-|x|)) gives -2000. Two pairs matched
    #      the wrong way round: loss = ln 2 + 2000, d loss / d t = 2000.
    # KIND: boundary
    # CATCHES: s05, s06, s07, s11
    # CHAPTER: L13.5 section 2.3
    img, txt = t64([[1.0, 0.0], [0.0, 1.0]]), t64([[0.0, 1.0], [1.0, 0.0]])
    t = t64([math.log(2000.0)], True)
    with np.errstate(all="ignore"):
        loss = siglip_loss(img, txt, t, t64([0.0]))
        loss.backward()
    assert_close(scalar(loss), math.log(2.0) + 2000.0, rtol=1e-9, atol=1e-9)
    assert_close(t.grad.reshape(()), 2000.0, rtol=1e-9, atol=1e-9)


def test_dual_encoder_parameters_and_loss():
    # WHY: the model is two towers, two bias-free projections into one
    #      space, and the loss's own parameters: logit_scale always,
    #      logit_bias only for SigLIP. loss() is the chosen loss of the two
    #      encodings, and one backward reaches every parameter of both
    #      towers.
    # KIND: unit
    # CATCHES: s10, m02
    # CHAPTER: L13.5 section 4
    g = PCG32(seed=6)
    x = Tensor(g.normal_array((3, 3, 4, 4)).astype(np.float32))
    ids = np.array([[3, 5, 2], [4, 1, 0], [7, 0, 0]])
    for kind in ("clip", "siglip"):
        m = tiny_model(kind)
        names = [n for n, _ in m.named_parameters()]
        assert "visual_projection.weight" in names and "text_projection.weight" in names
        assert (
            "visual_projection.bias" not in names
            and "text_projection.bias" not in names
        )
        assert (
            "text.token_embedding.weight" in names
            and "text.final_layer_norm.bias" in names
        )
        assert "vision.embeddings.patch_embedding.weight" in names
        assert m.visual_projection.weight.shape == (6, 8)
        assert m.logit_scale.shape == (1,) and m.logit_scale.requires_grad
        assert_close(m.logit_scale.data, [math.log(10.0)], rtol=1e-6, atol=1e-6)
        img, txt = m.encode_image(x), m.encode_text(ids)
        assert img.shape == (3, 6) and txt.shape == (3, 6)
        assert_close(np.linalg.norm(img.data, axis=1), np.ones(3), rtol=1e-5, atol=1e-5)
        assert_close(np.linalg.norm(txt.data, axis=1), np.ones(3), rtol=1e-5, atol=1e-5)
        if kind == "clip":
            assert getattr(m, "logit_bias", None) is None
            want = clip_loss(img, txt, m.logit_scale)
        else:
            assert m.logit_bias.shape == (1,) and m.logit_bias.requires_grad
            assert_close(m.logit_bias.data, [-10.0], rtol=1e-6, atol=1e-6)
            want = siglip_loss(img, txt, m.logit_scale, m.logit_bias)
        loss = m.loss(x, ids)
        assert_close(scalar(loss), scalar(want), rtol=1e-6, atol=1e-6)
        m.zero_grad()
        loss.backward()
        dead = [
            n
            for n, p in m.named_parameters()
            if p.grad is None or not np.all(np.isfinite(p.grad))
        ]
        assert dead == [], dead
    with pytest.raises(ValueError):
        DualEncoder(m.vision, m.text, 6, "infonce", 2.0)


def test_image_pooling_cls_or_mean():
    # WHY: CLIP's tower has a CLS token and the image embedding is its
    #      output; SigLIP's has none, so the patches are averaged. Pooling
    #      the wrong thing still trains, but a loaded checkpoint's
    #      projection then reads a vector it was never fitted to.
    # KIND: unit
    # CATCHES: s10, s13
    # CHAPTER: L13.5 section 2.5
    x = Tensor(PCG32(seed=7).normal_array((2, 3, 4, 4)).astype(np.float32))
    for cls in (True, False):
        m = tiny_model("clip", s=3, cls_token=cls)
        last, _ = m.vision(x)
        assert last.shape == (2, 5 if cls else 4, 8)
        pooled = last.data[:, 0] if cls else last.data.mean(axis=1)
        want = unit_rows(m.visual_projection(Tensor(pooled)).data.astype(np.float64))
        assert_close(
            m.encode_image(x).data, want, rtol=1e-5, atol=1e-5, msg=f"cls_token {cls}"
        )


def test_text_padding_is_invisible():
    # WHY: captions in a batch have different lengths and are right-padded
    #      with id 0. A caption's embedding must not depend on how much
    #      padding its batch gave it: pads are hidden as attention keys and
    #      left out of the mean. A row of only padding has no mean.
    # KIND: property
    # CATCHES: s09, s12, m03
    # CHAPTER: L13.5 section 2.5
    m = tiny_model("siglip", s=4)
    short = np.array([[3, 5, 2], [4, 1, 0]])
    wide = np.zeros((2, 6), dtype=np.int64)
    wide[:, :3] = short
    a, b = m.encode_text(short).data, m.encode_text(wide).data
    assert_close(b, a, rtol=1e-5, atol=1e-5)
    alone = m.encode_text(np.array([[4, 1]])).data
    assert_close(a[1:], alone, rtol=1e-5, atol=1e-5)
    h = m.text(wide)
    assert h.shape == (2, 6, 8)
    with pytest.raises(ValueError):
        m.encode_text(np.array([[3, 5], [0, 0]]))
    with pytest.raises(ValueError):
        m.text(np.zeros((1, 7), dtype=np.int64))  # longer than max_len
    with pytest.raises(ValueError):
        m.text(np.array([3, 5, 2]))


def test_zero_shot_classifier_is_a_prompt_ensemble():
    # WHY: a zero-shot classifier is a weight matrix written in words: one
    #      row per class, the renormalized mean of that class's prompt
    #      embeddings over the templates. Prediction is the nearest row by
    #      cosine. data.10's CLIP-score filter and ethics.07's probe are
    #      built on exactly these two calls.
    # KIND: unit
    # CATCHES: s09, s12, s14
    # CHAPTER: L13.5 section 2.6
    vocab = {
        w: i + 1
        for i, w in enumerate(["a", "photo", "of", "red", "blue", "square", "bar"])
    }

    def tok(text):
        return [vocab[w] for w in text.split()]

    class Tok:
        def encode(self, text):
            return tok(text)

    m = tiny_model("clip", s=5)
    names, templates = (
        ["red square", "blue bar", "red bar"],
        ["a {}", "a photo of a {}"],
    )
    W = zero_shot_classifier(m, tok, names, templates)
    assert W.shape == (3, 6)
    assert_close(np.linalg.norm(W, axis=1), np.ones(3), rtol=1e-5, atol=1e-5)
    for k, name in enumerate(names):
        rows = [
            m.encode_text(np.array([tok(t.format(name))])).data[0].astype(np.float64)
            for t in templates
        ]
        mean = np.mean(rows, axis=0)
        assert (
            np.linalg.norm(mean) < 0.9999
        )  # two different prompts: the mean is shorter than 1
        assert_close(W[k], mean / np.linalg.norm(mean), rtol=1e-5, atol=1e-5, msg=name)
    assert_close(
        zero_shot_classifier(m, Tok(), names, templates), W, rtol=1e-6, atol=1e-6
    )
    x = Tensor(PCG32(seed=8).normal_array((5, 3, 4, 4)).astype(np.float32))
    pred = zero_shot_predict(m, x, W)
    assert pred.shape == (5,) and pred.dtype == np.int64
    assert pred.tolist() == np.argmax(m.encode_image(x).data @ W.T, axis=1).tolist()
    assert (
        zero_shot_predict(m, x, np.stack([W[0], W[0]])).tolist() == [0] * 5
    )  # ties go low


def test_bad_embedding_shapes_are_refused():
    # WHY: row i of the images goes with row i of the texts. Two batches of
    #      different sizes, or embeddings of different widths, have no
    #      diagonal; broadcasting them into a loss would train on noise.
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: L13.5 section 4
    t, b = t64([1.0]), t64([0.0])
    for i_shape, x_shape in (
        ((3, 4), (2, 4)),
        ((3, 4), (3, 5)),
        ((4,), (4,)),
        ((2, 3, 4), (2, 3, 4)),
        ((3, 4), (1, 4)),
    ):
        img, txt = t64(np.ones(i_shape)), t64(np.ones(x_shape))
        with pytest.raises(ValueError):
            clip_loss(img, txt, t)
        with pytest.raises(ValueError):
            siglip_loss(img, txt, t, b)


# The learning test's data: `mm-shapes` in miniature. 6 colors x 4 shapes on
# 8 x 8 images, drawn from PCG32 with a jittered position and brightness and
# a noisy background, captioned by a 13-word vocabulary.
COLORS = {
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "yellow": (1, 1, 0),
    "magenta": (1, 0, 1),
    "cyan": (0, 1, 1),
}
SHAPES = ("square", "bar", "pole", "cross")
WORDS = ["a", "photo", "of", *COLORS, *SHAPES]
VOCAB = {w: i + 1 for i, w in enumerate(WORDS)}
CLASSES = [f"{c} {s}" for c in COLORS for s in SHAPES]
CAPTIONS = ("a {}", "a photo of a {}", "{}")


def shape_tok(text: str) -> list[int]:
    return [VOCAB[w] for w in text.split()]


def shape_mask(shape: str, dy: int, dx: int) -> np.ndarray:
    m = np.zeros((8, 8), dtype=bool)
    if shape == "square":
        m[2 + dy : 6 + dy, 2 + dx : 6 + dx] = True
    if shape in ("bar", "cross"):
        m[3 + dy : 5 + dy, 1 + dx : 7 + dx] = True
    if shape in ("pole", "cross"):
        m[1 + dy : 7 + dy, 3 + dx : 5 + dx] = True
    return m


def mm_shapes(n_per_class: int, g: PCG32) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for k, name in enumerate(CLASSES):
        color, shape = name.split()
        for _ in range(n_per_class):
            mask = shape_mask(shape, g.below(3) - 1, g.below(3) - 1)
            img = 0.1 * g.uniform_array((3, 8, 8))
            img += (
                mask[None]
                * np.array(COLORS[color], dtype=np.float64)[:, None, None]
                * (0.6 + 0.4 * g.uniform())
            )
            xs.append(img)
            ys.append(k)
    return np.stack(xs).astype(np.float32), np.array(ys)


def test_dual_encoder_learns_zero_shot_shapes():
    # WHY: the parts learn together: a one-layer ViT and a one-layer text
    #      tower, 600 Adam steps of the SigLIP loss on 16 image-caption
    #      pairs, then a classifier built from class names alone labels
    #      held-out images of 24 color-shape classes (chance is 1/24; the
    #      reference reaches about 0.89). No label is ever shown: only
    #      captions. A loss with a wrong gradient that still passes shape
    #      checks shows up here as a model that does not learn.
    # KIND: learning
    # CATCHES: s06, s08, s11
    # CHAPTER: L13.5 section 2.6
    n_train = 8
    xtr, ytr = mm_shapes(n_train, PCG32(seed=1350))
    xte, yte = mm_shapes(4, PCG32(seed=1351))
    r = Rng(100 + seed())
    vision = ViT(
        ViTConfig(
            image_size=8,
            patch_size=2,
            n_channels=3,
            d=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
        ),
        rng=r,
    )
    text = TextEncoder(
        vocab=len(WORDS) + 1, d=16, n_layers=1, n_heads=2, d_ff=32, max_len=6, rng=r
    )
    model = DualEncoder(vision, text, 16, "siglip", math.log(10.0), -10.0, rng=r)
    params = list(model.parameters())
    m1 = [np.zeros_like(p.data) for p in params]
    m2 = [np.zeros_like(p.data) for p in params]
    order = PCG32(seed=200 + seed())
    lr, b1, b2, batch = 5e-3, 0.9, 0.999, 16
    for step in range(1, 601):
        classes = list(range(len(CLASSES)))  # 16 different classes: no false negatives
        for i in range(len(classes) - 1, 0, -1):
            j = order.below(i + 1)
            classes[i], classes[j] = classes[j], classes[i]
        idx = np.array([c * n_train + order.below(n_train) for c in classes[:batch]])
        ids = np.zeros((batch, 6), dtype=np.int64)
        for row, i in enumerate(idx):
            words = shape_tok(
                CAPTIONS[order.below(len(CAPTIONS))].format(CLASSES[ytr[i]])
            )
            ids[row, : len(words)] = words
        loss = model.loss(Tensor(xtr[idx]), ids)
        model.zero_grad()
        loss.backward()
        for i, p in enumerate(params):  # Adam, the frozen test copy
            m1[i] = b1 * m1[i] + (1 - b1) * p.grad
            m2[i] = b2 * m2[i] + (1 - b2) * p.grad * p.grad
            mh, vh = m1[i] / (1 - b1**step), m2[i] / (1 - b2**step)
            p.data -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(p.data.dtype)
    W = zero_shot_classifier(model, shape_tok, CLASSES, ["a photo of a {}", "a {}"])
    acc = float(np.mean(zero_shot_predict(model, Tensor(xte), W) == yte))
    assert acc >= 0.5, acc
    check(
        "L13.5/test_dual_encoder_learns_zero_shot_shapes",
        "zero_shot_acc",
        acc,
        direction="min",
    )
