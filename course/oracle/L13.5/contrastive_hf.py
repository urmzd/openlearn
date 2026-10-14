# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L13.5 golden fixtures (transformers 5.19.0 losses, float64).

The losses of Hugging Face's CLIPModel and SiglipModel (forward with
return_loss=True) on given embeddings: each tiny model's towers are replaced
by functions that return the fixture embeddings as pooler_output, so the
model's own normalization, scaling, and loss code runs. Gradients come from
autograd.

  clip.b4    B 4, d 8, logit_scale log(1 / 0.07)
  clip.b1    B 1 (the loss is 0: one text, one image)
  siglip.b5  B 5, d 8, t = log 10, b = -10 (the init)
  siglip.b3  B 3, t = 1.3, b = 0.7
  siglip.b1  B 1

Keys: <case>.img, .txt, .t (and .b), .loss, .d_img, .d_txt, .d_t (.d_b).

    uv run --offline --python 3.12 --script course/oracle/L13.5/contrastive_hf.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from transformers import CLIPConfig, CLIPModel, SiglipConfig, SiglipModel

OUT = Path("course/fixtures/L13.5/contrastive_hf.npz")
TINY = dict(
    hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2
)


def run(arrs, name, kind, b_, t, bias=None):
    gen = np.random.default_rng(20261350 + len(arrs))
    img = torch.tensor(gen.standard_normal((b_, 8)), requires_grad=True)
    txt = torch.tensor(gen.standard_normal((b_, 8)), requires_grad=True)
    if kind == "clip":
        m = CLIPModel(
            CLIPConfig(
                text_config=TINY,
                vision_config={**TINY, "image_size": 8, "patch_size": 4},
                projection_dim=8,
            )
        ).double()
        m.get_image_features = lambda **kw: SimpleNamespace(pooler_output=img)
        m.get_text_features = lambda **kw: SimpleNamespace(pooler_output=txt)
    else:
        m = SiglipModel(
            SiglipConfig(
                text_config=TINY,
                vision_config={**TINY, "image_size": 8, "patch_size": 4},
            )
        ).double()
        del m.vision_model
        del m.text_model
        m.vision_model = lambda **kw: SimpleNamespace(pooler_output=img)
        m.text_model = lambda **kw: SimpleNamespace(pooler_output=txt)
    with torch.no_grad():
        m.logit_scale.fill_(t)
        if bias is not None:
            m.logit_bias.fill_(bias)
    out = m(pixel_values=None, input_ids=None, return_loss=True)
    out.loss.backward()
    arrs.update(
        {
            f"{name}.img": img.detach().numpy(),
            f"{name}.txt": txt.detach().numpy(),
            f"{name}.t": np.array(t),
            f"{name}.loss": out.loss.detach().numpy(),
            f"{name}.d_img": img.grad.numpy(),
            f"{name}.d_txt": txt.grad.numpy(),
            f"{name}.d_t": m.logit_scale.grad.numpy().reshape(()),
        }
    )
    if bias is not None:
        arrs[f"{name}.b"] = np.array(bias)
        arrs[f"{name}.d_b"] = m.logit_bias.grad.numpy().reshape(())


def main() -> None:
    torch.manual_seed(0)
    arrs: dict[str, np.ndarray] = {}
    run(arrs, "clip.b4", "clip", 4, math.log(1 / 0.07))
    run(arrs, "clip.b1", "clip", 1, math.log(1 / 0.07))
    run(arrs, "siglip.b5", "siglip", 5, math.log(10.0), -10.0)
    run(arrs, "siglip.b3", "siglip", 3, 1.3, 0.7)
    run(arrs, "siglip.b1", "siglip", 1, math.log(10.0), -10.0)
    meta = {
        "generator": "course/oracle/L13.5/contrastive_hf.py",
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
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L13.5/contrastive_hf.py\t"
        f"torch=={torch.__version__.split('+')[0]},transformers=={transformers.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
