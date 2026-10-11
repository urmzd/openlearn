"""Course tests for L13.3: ViTConfig, PatchEmbed, ViT, interpolate_pos_embed,
and bucketed_position_ids (tinyllm/vision/vit.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L13.3), and the chapter section it comes from.

The chapter's worked example (section 3): Idefics3 positions for a 4 x 4
table (n_side 4) and an image whose real patches are 3 rows by 2 columns.
Row fractions 0, 1/3, 2/3 fall in buckets 0, 1, 2; column fractions 0, 1/2
fall in buckets 0, 2 (1/2 sits ON the boundary 2/4, and a boundary counts
when it is <= x). So the ids are [[0, 2], [4, 6], [8, 10]]. The golden
values in $TINYLLM_FIXTURES/L13.3/vit_hf.npz come from
course/oracle/L13.3/vit_hf.py (transformers 5.19.0 ViTModel,
CLIPVisionModel, SiglipVisionModel, Idefics3VisionTransformer, and the
Qwen2-VL vision blocks, float32, with weights renamed to this contract).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.vision.vit import (
    PatchEmbed,
    ViT,
    ViTConfig,
    bucketed_position_ids,
    interpolate_pos_embed,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L13.3" / "vit_hf.npz"


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


def tiny(**over) -> ViTConfig:
    kw = dict(
        image_size=8, patch_size=2, n_channels=3, d=8, n_layers=2, n_heads=2, d_ff=16
    )
    kw.update(over)
    return ViTConfig(**kw)


def hf_model(z, case: str, weights: str | None = None) -> ViT:
    cfg = ViTConfig.from_hf(json.loads(bytes(z[f"{case}.config"]).decode()))
    m = ViT(cfg, rng=Rng(1))
    pre = f"{weights or case}.w."
    m.load_state_dict({k[len(pre) :]: z[k] for k in z.files if k.startswith(pre)})
    return m


def test_hand_example_bucketed_positions():
    # WHY: section 3's worked example. Idefics3 maps any real-patch grid onto
    #      the fixed 4 x 4 table by fractional coordinates; a boundary hit
    #      exactly (1/2) belongs to the upper bucket. A full grid gives the
    #      identity 0 .. 15, and padded patches get 0.
    # KIND: unit, smoke
    # CATCHES: s03
    # CHAPTER: L13.3 section 3
    mask = np.zeros((4, 4), bool)
    mask[:3, :2] = True
    ids = bucketed_position_ids(mask, 4)
    assert ids.shape == (16,) and ids.dtype.kind == "i"
    assert ids.reshape(4, 4)[:3, :2].tolist() == [[0, 2], [4, 6], [8, 10]]
    assert ids.reshape(4, 4)[3].tolist() == [0, 0, 0, 0]
    assert bucketed_position_ids(np.ones((4, 4), bool), 4).tolist() == list(range(16))
    batch = bucketed_position_ids(np.stack([np.ones((4, 4), bool), mask]), 4)
    assert batch.shape == (2, 16) and batch[1].tolist() == ids.tolist()


@pytest.mark.parametrize("case", ["vit", "clip", "siglip", "idefics3"])
def test_towers_match_hf(case):
    # WHY: one encoder covers four published towers: ViT (CLS, learned
    #      table, final LayerNorm), CLIP (no patch bias, pre-LayerNorm,
    #      quick_gelu, no final norm on the hidden states), SigLIP (no CLS,
    #      tanh GELU) and Idefics3 (bucketed positions, padded patches).
    #      Every hidden state L13.7 might tap must equal Hugging Face's.
    # KIND: golden
    # CATCHES: s01, s02, s06, s07
    # CHAPTER: L13.3 section 2.2
    z = np.load(GOLDEN)
    m = hf_model(z, case)
    mask = z["idefics3.patch_mask"] if case == "idefics3" else None
    last, hidden = m(Tensor(z[f"{case}.pixel_values"]), mask, output_hidden_states=True)
    n = int(z[f"{case}.n_hidden"])
    assert len(hidden) == n
    rows = slice(None)
    if mask is not None:  # padded patches' own outputs are unspecified
        keep = mask.reshape(2, -1)
        for i in range(n):
            assert_close(
                hidden[i].data[keep],
                z[f"{case}.hidden{i}"][keep],
                rtol=1e-5,
                atol=1e-5,
                msg=f"hidden {i}",
            )
        assert_close(last.data[keep], z[f"{case}.last"][keep], rtol=1e-5, atol=1e-5)
        return
    for i in range(n):
        assert_close(
            hidden[i].data[rows],
            z[f"{case}.hidden{i}"],
            rtol=1e-5,
            atol=1e-5,
            msg=f"hidden {i}",
        )
    assert_close(last.data, z[f"{case}.last"], rtol=1e-5, atol=1e-5)


def test_position_interpolation_matches_hf():
    # WHY: a ViT trained at one resolution runs at another by resizing its
    #      position table bicubically (torch's F.interpolate, a = -0.75)
    #      while keeping the CLS row: a 4 x 4 table serves a 6 x 5 grid.
    # KIND: golden
    # CATCHES: s04
    # CHAPTER: L13.3 section 2.3
    z = np.load(GOLDEN)
    m = hf_model(z, "vitinterp", weights="vit")
    last, _ = m(Tensor(z["vitinterp.pixel_values"]))
    assert last.shape == (1, 31, 32)
    assert_close(last.data, z["vitinterp.last"], rtol=1e-5, atol=1e-5)
    table = z["vit.w.embeddings.position_embedding.weight"]
    same = interpolate_pos_embed(table, (4, 4), (4, 4))
    assert_close(same, table, rtol=1e-6, atol=1e-6)
    big = interpolate_pos_embed(table, (4, 4), (6, 5))
    assert big.shape == (31, 32) and big.dtype == np.float32
    assert_close(big[0], table[0], rtol=0.0, atol=0.0)


def test_rope2d_tower_matches_qwen2vl():
    # WHY: pos = 'rope2d' adds nothing to the patches and rotates queries
    #      and keys by (row, column) inside every block; with merge-block
    #      patch order this is Qwen2-VL's vision tower (L13.7's qwen2vl).
    # KIND: golden
    # CATCHES: s05
    # CHAPTER: L13.3 section 2.3
    z = np.load(GOLDEN)
    cfg = tiny(
        image_size=16,
        patch_size=4,
        d=32,
        n_heads=4,
        d_ff=64,
        pos="rope2d",
        act="quick_gelu",
        post_layernorm=False,
        rope_merge=2,
    )
    m = ViT(cfg, rng=Rng(2))
    pre = "qwen2vl.w."
    m.load_state_dict(
        {
            k[len(pre) :]: z[k]
            for k in z.files
            if k.startswith(pre) and not k.endswith("patch_linear")
        },
        strict=False,
    )
    h = Tensor((z["qwen2vl.patches"] @ z["qwen2vl.w.patch_linear"].T)[None])
    last, hidden = m.encode(h, (4, 6), output_hidden_states=True)
    for i in range(int(z["qwen2vl.n_hidden"])):
        assert_close(
            hidden[i].data[0],
            z[f"qwen2vl.hidden{i}"],
            rtol=1e-5,
            atol=1e-5,
            msg=f"hidden {i}",
        )
    assert_close(last.data[0], z["qwen2vl.last"], rtol=1e-5, atol=1e-5)


def test_patch_embed_is_patchify_plus_linear():
    # WHY: a conv whose kernel equals its stride touches each patch once, so
    #      it is one linear map per patch: cut the image into patches,
    #      flatten each in (c, i, j) order, multiply by the flattened kernel.
    # KIND: differential
    # CATCHES: s01
    # CHAPTER: L13.3 section 2.1
    cfg = tiny(n_channels=2, d=5, patch_size=3, image_size=9)
    pe = PatchEmbed(cfg, rng=Rng(3))
    assert pe.weight.shape == (5, 2, 3, 3) and pe.bias.shape == (5,)
    x = PCG32(seed=seed() + 4).normal_array((2, 9, 6)).astype(np.float32)
    patches = x.reshape(2, 3, 3, 2, 3).transpose(1, 3, 0, 2, 4).reshape(6, 18)
    want = patches @ pe.weight.data.reshape(5, -1).T + pe.bias.data
    got = pe(Tensor(x[None])).data[0]  # [d, 3, 2]
    assert_close(got.reshape(5, 6).T, want, rtol=1e-5, atol=1e-5)
    assert PatchEmbed(tiny(patch_bias=False), rng=Rng(3)).bias is None


def test_hidden_states_and_names():
    # WHY: L13.7 taps hidden[vision_feature_layer] (LLaVA uses -2), so the
    #      list holds the L + 1 block inputs and outputs before the final
    #      norm, and parameter names are the HF names checkpoints load by
    #      (a module's own tensors, like class_embedding, list first).
    # KIND: unit
    # CATCHES: s07
    # CHAPTER: L13.3 section 4
    m = ViT(tiny(cls_token=True, pre_norm=True), rng=Rng(5))
    names = [n for n, _ in m.named_parameters()]
    assert names[:5] == [
        "embeddings.class_embedding",
        "embeddings.patch_embedding.weight",
        "embeddings.patch_embedding.bias",
        "embeddings.position_embedding.weight",
        "pre_layrnorm.weight",
    ]
    assert (
        "encoder.layers.1.self_attn.out_proj.bias" in names
        and names[-1] == "post_layernorm.bias"
    )
    assert m.embeddings.position_embedding.weight.shape == (17, 8)
    x = Tensor(PCG32(seed=6).normal_array((1, 3, 8, 8)).astype(np.float32))
    last, hidden = m(x, output_hidden_states=True)
    assert len(hidden) == 3 and last.shape == (1, 17, 8)
    assert_close(hidden[0].data, m.embed(x).data, rtol=0.0, atol=0.0)
    assert m(x)[1] == []


def test_sincos2d_positions():
    # WHY: the fixed table: the first half of the width encodes the row,
    #      the second half the column, each as sines then cosines of the
    #      coordinate times 10000^(-k / (d/4)).
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: L13.3 section 2.3
    cfg = tiny(d=8, n_heads=2, n_layers=0, pos="sincos2d", post_layernorm=False)
    m = ViT(cfg, rng=Rng(7))
    m.embeddings.patch_embedding.weight.data[...] = 0.0
    m.embeddings.patch_embedding.bias.data[...] = 0.0
    h = m.embed(Tensor(np.zeros((1, 3, 8, 8), np.float32))).data[0]  # [16, 8]
    w = np.array([1.0, 0.01])
    for r, c in ((0, 0), (1, 2), (3, 1)):
        want = np.concatenate(
            [np.sin(r * w), np.cos(r * w), np.sin(c * w), np.cos(c * w)]
        )
        assert_close(h[r * 4 + c], want, rtol=1e-6, atol=1e-6, msg=f"{(r, c)}")


def test_outputs_permute_with_patches_without_positions():
    # WHY: attention is permutation-equivariant: without position
    #      information, shuffling the patches only shuffles the outputs. That
    #      is why a ViT needs positions at all.
    # KIND: property
    # CATCHES: s02
    # CHAPTER: L13.3 section 2.3
    m = ViT(tiny(post_layernorm=True), rng=Rng(8))
    m.embeddings.position_embedding.weight.data[...] = 0.0
    x = PCG32(seed=seed() + 9).normal_array((1, 3, 8, 8)).astype(np.float32)
    perm = np.array([5, 0, 15, 3, 9, 12, 1, 2, 4, 6, 7, 8, 10, 11, 13, 14])
    blocks = x.reshape(3, 4, 2, 4, 2).transpose(1, 3, 0, 2, 4).reshape(16, 3, 2, 2)
    shuffled = (
        blocks[perm].reshape(4, 4, 3, 2, 2).transpose(2, 0, 3, 1, 4).reshape(1, 3, 8, 8)
    )
    a = m(Tensor(x))[0].data[0]
    b = m(Tensor(shuffled))[0].data[0]
    assert_close(b, a[perm], rtol=1e-5, atol=1e-5)


def test_padded_patches_never_change_real_outputs():
    # WHY: Idefics3 pads images of different sizes into one batch; a padded
    #      patch is hidden as a key, so the real patches' outputs equal those
    #      of the unpadded image, whatever the padding holds.
    # KIND: property
    # CATCHES: s02, s03
    # CHAPTER: L13.3 section 2.4
    cfg = tiny(bucketed=True, act="gelu_tanh")
    m = ViT(cfg, rng=Rng(10))
    g = PCG32(seed=seed() + 11)
    real = g.normal_array((1, 3, 6, 4)).astype(np.float32)  # 3 x 2 patches
    small, _ = m(Tensor(real))
    mask = np.zeros((1, 4, 4), bool)
    mask[0, :3, :2] = True
    keep = mask.reshape(1, -1)
    for fill in (0.0, 5.0):
        padded = np.full((1, 3, 8, 8), fill, np.float32)
        padded[:, :, :6, :4] = real
        big, _ = m(Tensor(padded), mask)
        assert_close(
            big.data[keep], small.data[0], rtol=1e-5, atol=1e-5, msg=f"fill {fill}"
        )


def test_gradcheck_vit():
    # WHY: the tower is built from autograd ops (L13.1's conv, L5.3's
    #      attention, L0.4's layers), so its backward must reach the pixels
    #      and the position table; proven against central differences.
    # KIND: gradcheck
    # CATCHES: s01
    # CHAPTER: L13.3 section 2.2
    cfg = tiny(
        image_size=4, n_channels=1, d=4, n_layers=1, n_heads=1, d_ff=4, cls_token=True
    )
    m = ViT(cfg, rng=Rng(12))
    for p in m.parameters():
        p.data = p.data.astype(np.float64)
    table = m.embeddings.position_embedding.weight
    g = PCG32(seed=13)
    x0 = g.normal_array((1, 1, 4, 4))
    gout = g.normal_array((1, 5, 4))

    def f(x, t):
        table.data = t
        return float(np.sum(m(Tensor(x, dtype=np.float64))[0].data * gout))

    t0 = table.data.copy()
    x = Tensor(x0, requires_grad=True, dtype=np.float64)
    F.sum(m(x)[0] * gout).backward()
    gradcheck(f, [x0, t0], [x.grad, table.grad], names=["pixels", "position table"])


def test_from_hf_and_bad_configs():
    # WHY: configs come from Hugging Face JSON; each tower kind sets its own
    #      switches, and impossible shapes fail at construction, not deep
    #      inside a forward pass.
    # KIND: boundary
    # CATCHES: m01, m02, m03
    # CHAPTER: L13.3 section 4
    base = dict(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        image_size=8,
        patch_size=2,
    )
    c = ViTConfig.from_hf(
        {**base, "model_type": "clip_vision_model", "layer_norm_eps": 1e-3}
    )
    assert (c.cls_token, c.pre_norm, c.post_layernorm, c.patch_bias, c.act) == (
        True,
        True,
        False,
        False,
        "quick_gelu",
    )
    assert c.ln_eps == 1e-3
    s = ViTConfig.from_hf({**base, "model_type": "idefics3_vision"})
    assert (s.cls_token, s.bucketed, s.act) == (False, True, "gelu_tanh")
    with pytest.raises(ValueError):
        ViTConfig.from_hf({**base, "model_type": "bert"})
    with pytest.raises(ValueError):
        ViTConfig.from_hf({"model_type": "vit", "patch_size": 2})
    with pytest.raises(ValueError):
        ViT(tiny(d=9, n_heads=2), rng=Rng(1))
    with pytest.raises(ValueError):
        ViT(tiny(image_size=9), rng=Rng(1))
    with pytest.raises(ValueError):
        ViT(tiny(d=12, n_heads=2, pos="rope2d"), rng=Rng(1))
    m = ViT(tiny(), rng=Rng(1))
    with pytest.raises(ValueError):
        m(Tensor(np.zeros((1, 3, 7, 8), np.float32)))
    with pytest.raises(ValueError):
        m(Tensor(np.zeros((1, 2, 8, 8), np.float32)))
    with pytest.raises(ValueError):
        m(Tensor(np.zeros((1, 3, 8, 8), np.float32)), np.ones((1, 3, 4), bool))
    with pytest.raises(ValueError):
        interpolate_pos_embed(np.zeros((11, 4)), (3, 3), (4, 4))
