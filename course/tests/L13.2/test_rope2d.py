"""Course tests for L13.2: patch_coords, rope2d_cos_sin, mrope_position_ids,
mrope_cos_sin, and apply_mrope (tinyllm/vision/rope2d.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L13.2), and the chapter section it comes from.

The chapter's worked example (section 3): two text tokens, then one image of
4 x 4 patches with spatial merge 2 (four image tokens), then text. The text
gets (0, 0, 0) and (1, 1, 1); the image tokens get t = 2 and (h, w) =
(2, 2), (2, 3), (3, 2), (3, 3); the next text token resumes at 2 + max(2, 2)
= 4. The golden values in $TINYLLM_FIXTURES/L13.2/mrope_hf.npz come from
course/oracle/L13.2/mrope_hf.py (transformers 5.19.0 Qwen2-VL:
get_rope_index, Qwen2VLRotaryEmbedding, apply_rotary_pos_emb,
get_vision_position_ids, Qwen2VLVisionRotaryEmbedding).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.rope import apply_rope, rope_cos_sin
from tinyllm.vision.rope2d import (
    apply_mrope,
    mrope_cos_sin,
    mrope_position_ids,
    patch_coords,
    rope2d_cos_sin,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L13.2" / "mrope_hf.npz"
IMG = 7


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def ladder(r: int, base: float = 10000.0) -> np.ndarray:
    return (1.0 / base ** (np.arange(0, r, 2) / r)).astype(np.float32)


def test_hand_example():
    # WHY: section 3's worked example: text, a 4 x 4 patch image merged
    #      2 x 2 into four tokens, text again. Block-major patch order puts
    #      one merge block's patches next to each other.
    # KIND: unit, smoke
    # CATCHES: s01, s04, s05
    # CHAPTER: L13.2 section 3
    ids = np.array([5, 6, IMG, IMG, IMG, IMG, 8, 9])
    pos = mrope_position_ids(ids, [[1, 4, 4]], IMG, spatial_merge=2)
    want = np.array(
        [
            [0, 1, 2, 2, 2, 2, 4, 5],
            [0, 1, 2, 2, 3, 3, 4, 5],
            [0, 1, 2, 3, 2, 3, 4, 5],
        ]
    )
    assert pos.tolist() == want.tolist()
    assert patch_coords((2, 4), 2).tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
    ]
    assert patch_coords((2, 3)).tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
    ]


def test_mrope_position_ids_match_hf():
    # WHY: Qwen2-VL's get_rope_index on three images (one tall, one wide,
    #      one two-frame image right after it). L13.7's qwen2vl model feeds
    #      these ids to every attention layer; one off-by-one shifts every
    #      later token's rotation.
    # KIND: golden
    # CATCHES: s01, s05, m03
    # CHAPTER: L13.2 section 2.3
    z = np.load(GOLDEN)
    pos = mrope_position_ids(
        z["text.input_ids"], z["text.grid_thw"], int(z["text.image_token_id"])
    )
    assert pos.dtype.kind == "i" and pos.shape == z["text.position_ids"].shape
    assert pos.tolist() == z["text.position_ids"].tolist()


def test_apply_mrope_matches_hf():
    # WHY: the three axes are spliced into the pair ladder by
    #      mrope_section (t, h, w, t, ...); queries and keys rotated this
    #      way must equal HF's apply_rotary_pos_emb.
    # KIND: golden
    # CATCHES: s02
    # CHAPTER: L13.2 section 2.4
    z = np.load(GOLDEN)
    pos = z["text.position_ids"]
    sec = [int(v) for v in z["text.mrope_section"]]
    for name in ("q", "k"):
        got = apply_mrope(Tensor(z[f"text.{name}"]), pos, z["text.inv_freq"], sec).data
        assert_close(got, z[f"text.{name}_embed"], rtol=1e-5, atol=1e-5, msg=name)


def test_vision_rope_matches_hf():
    # WHY: Qwen2-VL's vision tower: patch positions block-major over 2 x 2
    #      merge blocks, a ladder over half the head, rows then columns.
    #      L13.3 builds its 'rope2d' ViT from these tables.
    # KIND: golden
    # CATCHES: s03, s04, s06
    # CHAPTER: L13.2 section 2.2
    z = np.load(GOLDEN)
    _, h, w = (int(v) for v in z["vision.grid_thw"][0])
    assert patch_coords((h, w), 2).tolist() == z["vision.position_ids"].tolist()
    cos, sin = rope2d_cos_sin((h, w), 16, 10000.0, merge=2)
    assert cos.dtype == np.float32 and cos.shape == (h * w, 8)
    # HF repeats each table to the full head; L7.3 keeps one value per pair.
    assert_close(np.concatenate([cos, cos], -1), z["vision.cos"], rtol=1e-6, atol=1e-6)
    assert_close(np.concatenate([sin, sin], -1), z["vision.sin"], rtol=1e-6, atol=1e-6)
    for name in ("q", "k"):
        x = Tensor(z[f"vision.{name}"].transpose(1, 0, 2))  # [heads, N, dh]
        got = apply_rope(x, cos, sin, "half").data.transpose(1, 0, 2)
        assert_close(got, z[f"vision.{name}_embed"], rtol=1e-5, atol=1e-5, msg=name)


def test_2d_scores_depend_only_on_offsets():
    # WHY: the reason to use axial RoPE for patches: after rotation, the
    #      score of patch (h1, w1) against (h2, w2) depends only on
    #      (h1 - h2, w1 - w2). Shifting both patches by the same offset
    #      leaves the score unchanged.
    # KIND: property
    # CATCHES: s03, s06
    # CHAPTER: L13.2 section 2.2
    g = PCG32(seed=seed() + 1)
    q = g.normal_array((16,))
    k = g.normal_array((16,))
    cos, sin = rope2d_cos_sin((8, 8), 16)
    rq = apply_rope(
        Tensor(np.tile(q, (64, 1)), dtype=np.float64), cos, sin, "half"
    ).data
    rk = apply_rope(
        Tensor(np.tile(k, (64, 1)), dtype=np.float64), cos, sin, "half"
    ).data

    def score(a, b):
        return float(rq[a[0] * 8 + a[1]] @ rk[b[0] * 8 + b[1]])

    for (h1, w1), (h2, w2), (dh, dw) in (
        ((1, 2), (3, 1), (2, 3)),
        ((0, 0), (5, 6), (1, 1)),
        ((4, 1), (4, 1), (3, 5)),
    ):
        a = score((h1, w1), (h2, w2))
        b = score((h1 + dh, w1 + dw), (h2 + dh, w2 + dw))
        assert abs(a - b) < 1e-4, ((h1, w1), (h2, w2), a, b)
    # Rows and columns are different axes: a row step is not a column step.
    assert abs(score((1, 0), (0, 0)) - score((0, 1), (0, 0))) > 1e-3


def test_mrope_reduces_to_1d_rope_on_text():
    # WHY: a text-only sequence has t = h = w at every token, so M-RoPE
    #      must be exactly L7.3's RoPE: a qwen2vl model with no image is a
    #      plain Qwen2 language model.
    # KIND: property
    # CATCHES: s02
    # CHAPTER: L13.2 section 2.4
    ids = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    pos = mrope_position_ids(ids, np.zeros((0, 3)), IMG)
    assert pos.tolist() == [list(range(8))] * 3
    inv = ladder(12)
    x = Tensor(PCG32(seed=seed() + 2).normal_array((1, 2, 8, 12)).astype(np.float32))
    c1, s1 = rope_cos_sin(np.arange(8), inv)
    want = apply_rope(x, c1, s1, "half").data
    got = apply_mrope(x, pos, inv, [1, 2, 3]).data
    assert_close(got, want, rtol=0.0, atol=0.0)
    cos, sin = mrope_cos_sin(pos, inv, [3, 3])
    assert_close(cos, c1, rtol=0.0, atol=0.0)


def test_gradcheck_apply_mrope():
    # WHY: M-RoPE sits inside attention, so its backward must reach x: the
    #      inverse rotation of the upstream gradient, checked against the
    #      frozen central differences.
    # KIND: gradcheck
    # CATCHES: s02
    # CHAPTER: L13.2 section 2.4
    g = PCG32(seed=seed() + 3)
    x0 = g.normal_array((1, 2, 5, 8))
    gout = g.normal_array((1, 2, 5, 8))
    pos = np.array([[0, 1, 2, 2, 4], [0, 1, 2, 3, 4], [0, 1, 3, 2, 4]])
    inv = ladder(8).astype(np.float64)

    def f(x):
        return float(
            np.sum(
                apply_mrope(Tensor(x, dtype=np.float64), pos, inv, [1, 2, 1]).data
                * gout
            )
        )

    x = Tensor(x0, requires_grad=True, dtype=np.float64)
    F.sum(apply_mrope(x, pos, inv, [1, 2, 1]) * gout).backward()
    gradcheck(f, [x0], [x.grad], names=["x"])


def test_rejects_bad_inputs():
    # WHY: a wrong section split or image-token count would otherwise give
    #      silently wrong positions; these fail loudly instead.
    # KIND: boundary
    # CATCHES: m01, m02, m03
    # CHAPTER: L13.2 section 4
    with pytest.raises(ValueError):
        rope2d_cos_sin((2, 2), 6)
    with pytest.raises(ValueError):
        patch_coords((3, 4), 2)
    with pytest.raises(ValueError):
        patch_coords((0, 4))
    pos = np.zeros((3, 4), dtype=np.int64)
    with pytest.raises(ValueError):
        mrope_cos_sin(pos, ladder(8), [1, 1, 1])
    with pytest.raises(ValueError):
        mrope_cos_sin(pos, ladder(8), [2, 2, 2])
    with pytest.raises(ValueError):
        mrope_cos_sin(pos[:2], ladder(8), [2, 1, 1])
    with pytest.raises(ValueError):
        apply_mrope(Tensor(np.zeros((1, 4, 6))), pos, ladder(8), [2, 1, 1])
    with pytest.raises(ValueError):  # 3 image tokens for a grid that needs 4
        mrope_position_ids(np.array([1, IMG, IMG, IMG, 2]), [[1, 4, 4]], IMG)
    with pytest.raises(ValueError):  # merge 2 does not divide 3
        mrope_position_ids(np.array([1, IMG, IMG, 2]), [[1, 3, 4]], IMG)
    with pytest.raises(ValueError):  # a grid without image tokens
        mrope_position_ids(np.array([1, 2]), [[1, 2, 2]], IMG)
