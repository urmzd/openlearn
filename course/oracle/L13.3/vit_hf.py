# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L13.3 golden fixtures (transformers 5.19.0 vision towers).

Tiny random towers (d 32, 4 heads, d_ff 64, 2 layers, patch 4, image 16,
so a 4 x 4 grid), float32, eval mode, output_hidden_states=True:

  vit       ViTModel (add_pooling_layer False) on 2 x 3 x 16 x 16
  vitinterp the same model on 1 x 3 x 24 x 20 with interpolate_pos_encoding
            (a 6 x 5 grid: the table is resized bicubically)
  clip      CLIPVisionModel (quick_gelu, pre_layrnorm)
  siglip    SiglipVisionModel (gelu_pytorch_tanh; the pooling head is not stored)
  idefics3  Idefics3VisionTransformer on 2 images with patch_attention_mask:
            image 0 full, image 1 real on its top-left 3 x 2 patches and
            zero elsewhere (bucketed positions, masked keys)
  qwen2vl   Qwen2VisionTransformerPretrainedModel blocks (2D RoPE,
            quick_gelu, patch 4, merge 2) on one 4 x 6 patch grid; stored are
            the flattened patch inputs, the patch_embed weight as a
            [d, 3 * 2 * 4 * 4] matrix (Conv3d with kernel = stride is one
            matmul), and the block outputs before the merger

Every model's weights are stored under the tinyllm names of the L13.3
contract (<case>.w.<name>); the renames from HF are in RENAMES below.
<case>.config is the HF config as JSON bytes, <case>.pixel_values the input,
<case>.hidden<i> HF's hidden_states[i], <case>.last the last_hidden_state.

    uv run --offline --python 3.12 --script course/oracle/L13.3/vit_hf.py
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    Idefics3VisionConfig,
    SiglipVisionConfig,
    SiglipVisionModel,
    ViTConfig,
    ViTModel,
)
from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionTransformer
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)

OUT = Path("course/fixtures/L13.3/vit_hf.npz")
SIZES = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    image_size=16,
    patch_size=4,
)
RENAMES = {
    "vit": [
        (r"^embeddings\.cls_token$", "embeddings.class_embedding"),
        (r"^embeddings\.position_embeddings$", "embeddings.position_embedding.weight"),
        (r"^embeddings\.patch_embeddings\.projection\.", "embeddings.patch_embedding."),
        (
            r"^layers\.(\d+)\.attention\.o_proj\.",
            r"encoder.layers.\1.self_attn.out_proj.",
        ),
        (r"^layers\.(\d+)\.attention\.", r"encoder.layers.\1.self_attn."),
        (r"^layers\.(\d+)\.layernorm_before\.", r"encoder.layers.\1.layer_norm1."),
        (r"^layers\.(\d+)\.layernorm_after\.", r"encoder.layers.\1.layer_norm2."),
        (r"^layers\.(\d+)\.mlp\.", r"encoder.layers.\1.mlp."),
        (r"^layernorm\.", "post_layernorm."),
    ],
    "clip": [(r"^post_layernorm\..*", None)],
    "siglip": [(r"^head\..*", None)],
    "idefics3": [],
}


def rename(case: str, sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if "position_ids" in k:
            continue
        name = k
        for pat, rep in RENAMES[case]:
            if re.search(pat, name):
                name = None if rep is None else re.sub(pat, rep, name)
                break
        if name is None:
            continue
        a = v.detach().numpy().astype(np.float32)
        if name == "embeddings.class_embedding":
            a = a.reshape(-1)
        if name == "embeddings.position_embedding.weight":
            a = a.reshape(-1, a.shape[-1])
        out[name] = a
    return out


def run(arrs, case, model, cfg_dict, pixel_values, **kw):
    model.eval()
    with torch.no_grad():
        o = model(
            pixel_values=torch.from_numpy(pixel_values), output_hidden_states=True, **kw
        )
    arrs[f"{case}.config"] = np.frombuffer(
        json.dumps(cfg_dict).encode(), dtype=np.uint8
    )
    arrs[f"{case}.pixel_values"] = pixel_values
    arrs[f"{case}.n_hidden"] = np.array(len(o.hidden_states))
    for i, h in enumerate(o.hidden_states):
        arrs[f"{case}.hidden{i}"] = h.numpy()
    arrs[f"{case}.last"] = o.last_hidden_state.numpy()
    return o


def weights(arrs, case, sd):
    for k, v in sd.items():
        arrs[f"{case}.w.{k}"] = v


def main() -> None:
    torch.manual_seed(0)
    gen = np.random.default_rng(20261330)
    arrs: dict[str, np.ndarray] = {}
    px = lambda *s: gen.standard_normal(s).astype(np.float32)  # noqa: E731

    c = ViTConfig(**SIZES, layer_norm_eps=1e-12)
    m = ViTModel(c, add_pooling_layer=False)
    run(arrs, "vit", m, c.to_dict(), px(2, 3, 16, 16))
    run(
        arrs,
        "vitinterp",
        m,
        c.to_dict(),
        px(1, 3, 24, 20),
        interpolate_pos_encoding=True,
    )
    weights(arrs, "vit", rename("vit", m.state_dict()))

    c = CLIPVisionConfig(**SIZES, hidden_act="quick_gelu")
    m = CLIPVisionModel(c)
    run(arrs, "clip", m, c.to_dict(), px(2, 3, 16, 16))
    weights(arrs, "clip", rename("clip", m.state_dict()))

    c = SiglipVisionConfig(**SIZES)
    m = SiglipVisionModel(c)
    run(arrs, "siglip", m, c.to_dict(), px(2, 3, 16, 16))
    weights(arrs, "siglip", rename("siglip", m.state_dict()))

    c = Idefics3VisionConfig(**SIZES)
    m = Idefics3VisionTransformer(c)
    pv = px(2, 3, 16, 16)
    pv[1, :, 12:, :] = 0.0
    pv[1, :, :, 8:] = 0.0
    mask = np.ones((2, 4, 4), dtype=bool)
    mask[1, 3:, :] = False
    mask[1, :, 2:] = False
    arrs["idefics3.patch_mask"] = mask
    run(
        arrs,
        "idefics3",
        m,
        c.to_dict(),
        pv,
        patch_attention_mask=torch.from_numpy(mask),
    )
    weights(arrs, "idefics3", rename("idefics3", m.state_dict()))

    c = Qwen2VLVisionConfig(
        depth=2,
        embed_dim=32,
        hidden_size=48,
        num_heads=4,
        mlp_ratio=2,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
        hidden_act="quick_gelu",
    )
    m = Qwen2VisionTransformerPretrainedModel(c).eval()
    grid = torch.tensor([[1, 4, 6]])
    flat = px(24, 3 * 2 * 4 * 4)
    with torch.no_grad():
        o = m(torch.from_numpy(flat), grid_thw=grid, output_hidden_states=True)
    arrs["qwen2vl.config"] = np.frombuffer(
        json.dumps(c.to_dict()).encode(), dtype=np.uint8
    )
    arrs["qwen2vl.patches"] = flat
    arrs["qwen2vl.grid_thw"] = grid.numpy()
    arrs["qwen2vl.n_hidden"] = np.array(len(o.hidden_states))
    for i, h in enumerate(o.hidden_states):
        arrs[f"qwen2vl.hidden{i}"] = h.numpy()
    arrs["qwen2vl.last"] = o.last_hidden_state.numpy()
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    q = {"patch_linear": sd["patch_embed.proj.weight"].reshape(32, -1)}
    for i in range(2):
        b = f"blocks.{i}."
        o_ = f"encoder.layers.{i}."
        q[o_ + "layer_norm1.weight"], q[o_ + "layer_norm1.bias"] = (
            sd[b + "norm1.weight"],
            sd[b + "norm1.bias"],
        )
        q[o_ + "layer_norm2.weight"], q[o_ + "layer_norm2.bias"] = (
            sd[b + "norm2.weight"],
            sd[b + "norm2.bias"],
        )
        for j, nm in enumerate("qkv"):
            q[o_ + f"self_attn.{nm}_proj.weight"] = sd[b + "attn.qkv.weight"][
                32 * j : 32 * (j + 1)
            ]
            q[o_ + f"self_attn.{nm}_proj.bias"] = sd[b + "attn.qkv.bias"][
                32 * j : 32 * (j + 1)
            ]
        q[o_ + "self_attn.out_proj.weight"], q[o_ + "self_attn.out_proj.bias"] = (
            sd[b + "attn.proj.weight"],
            sd[b + "attn.proj.bias"],
        )
        for f in ("fc1", "fc2"):
            q[o_ + f"mlp.{f}.weight"], q[o_ + f"mlp.{f}.bias"] = (
                sd[b + f"mlp.{f}.weight"],
                sd[b + f"mlp.{f}.bias"],
            )
    weights(arrs, "qwen2vl", q)

    meta = {
        "generator": "course/oracle/L13.3/vit_hf.py",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    b = buf.getvalue()
    OUT.write_bytes(b)
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L13.3/vit_hf.py\t"
        f"torch=={torch.__version__.split('+')[0]},transformers=={transformers.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
