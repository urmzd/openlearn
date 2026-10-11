# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L13.2 golden fixtures (transformers 5.19.0 Qwen2-VL).

A tiny random Qwen2VLConfig (text: head_dim 16, mrope_section [2, 3, 3];
vision: embed_dim 32, 2 heads of 16, spatial_merge_size 2). Nothing is
trained; only the position code runs.

  text.*    an id sequence: 3 text tokens, image 0 (grid t1 x 4 x 6 patches:
            6 image tokens), 2 text tokens, image 1 (grid t1 x 6 x 4: 6
            tokens), image 2 directly after it (grid t2 x 2 x 2: 2 tokens),
            then 4 text tokens. position_ids is Qwen2VLModel.get_rope_index;
            q, k [1, 2, T, 16] random, q_embed and k_embed are
            apply_rotary_pos_emb with Qwen2VLRotaryEmbedding's cos and sin
            for those ids (mrope_section [2, 3, 3], rope_theta 1e4), plus
            inv_freq.
  vision.*  grid t1 x 4 x 6 patches: position_ids from
            transformers.vision_utils.get_vision_position_ids (merge 2), cos
            and sin of Qwen2VLVisionRotaryEmbedding (rope_theta 1e4, head_dim
            16), q, k [24, 2, 16] random, and apply_rotary_pos_emb_vision.

    uv run --offline --python 3.12 --script course/oracle/L13.2/mrope_hf.py
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import Qwen2VLConfig
from transformers.models.qwen2_vl import modeling_qwen2_vl as q2
from transformers.vision_utils import get_vision_position_ids

OUT = Path("course/fixtures/L13.2/mrope_hf.npz")
IMG = 99


def main() -> None:
    torch.manual_seed(0)
    gen = np.random.default_rng(20261320)
    cfg = Qwen2VLConfig(
        text_config=dict(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 3, 3],
            },
        ),
        vision_config=dict(
            depth=1,
            embed_dim=32,
            hidden_size=32,
            num_heads=2,
            mlp_ratio=2,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        image_token_id=IMG,
    )
    model = q2.Qwen2VLModel(cfg).eval()
    grids = torch.tensor([[1, 4, 6], [1, 6, 4], [2, 2, 2]])
    ids = [1, 2, 3] + [IMG] * 6 + [4, 5] + [IMG] * 6 + [IMG] * 2 + [6, 7, 8, 9]
    input_ids = torch.tensor([ids])
    mm = (input_ids == IMG).int()
    # get_rope_index groups runs by modality: two adjacent images form one
    # run, so they are passed as separate runs by marking image 2 as video
    # with the same grid semantics (temporal ids are t-scaled by 1).
    mm[0, 17:19] = 2
    pos, _ = model.get_rope_index(
        input_ids, mm, image_grid_thw=grids[:2], video_grid_thw=grids[2:]
    )
    rot = q2.Qwen2VLRotaryEmbedding(cfg.text_config)
    T = len(ids)
    q = torch.from_numpy(gen.standard_normal((1, 2, T, 16)).astype(np.float32))
    k = torch.from_numpy(gen.standard_normal((1, 2, T, 16)).astype(np.float32))
    cos, sin = rot(q, pos)
    qe, ke = q2.apply_rotary_pos_emb(q, k, cos, sin)

    vgrid = torch.tensor([[1, 4, 6]])
    vpos = get_vision_position_ids(vgrid, 2)
    vrot = q2.Qwen2VLVisionRotaryEmbedding(cfg.vision_config)
    vq = torch.from_numpy(gen.standard_normal((24, 2, 16)).astype(np.float32))
    vk = torch.from_numpy(gen.standard_normal((24, 2, 16)).astype(np.float32))
    vcos, vsin = vrot(vq, vpos)
    vqe, vke = q2.apply_rotary_pos_emb_vision(vq, vk, vcos, vsin)

    arrs = {
        "text.input_ids": np.array(ids, dtype=np.int64),
        "text.grid_thw": grids.numpy(),
        "text.image_token_id": np.array(IMG),
        "text.position_ids": pos[:, 0].numpy(),
        "text.inv_freq": rot.inv_freq.numpy(),
        "text.mrope_section": np.array([2, 3, 3]),
        "text.q": q.numpy(),
        "text.k": k.numpy(),
        "text.q_embed": qe.numpy(),
        "text.k_embed": ke.numpy(),
        "vision.grid_thw": vgrid.numpy(),
        "vision.position_ids": vpos.numpy(),
        "vision.cos": vcos.numpy(),
        "vision.sin": vsin.numpy(),
        "vision.q": vq.numpy(),
        "vision.k": vk.numpy(),
        "vision.q_embed": vqe.numpy(),
        "vision.k_embed": vke.numpy(),
    }
    meta = {
        "generator": "course/oracle/L13.2/mrope_hf.py",
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
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L13.2/mrope_hf.py\t"
        f"torch=={torch.__version__.split('+')[0]},transformers=={transformers.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
