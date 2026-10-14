# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3", "pillow==12.3.0"]
# ///
"""Maintainer generator for the L13.4 golden fixtures (transformers 5.19.0, Pillow 12.3.0).

table.*   40 image sizes (h, w), from 2 x 2 to 4000 x 3000 and extreme
          aspect ratios (no height of 1 or 3: Hugging Face would read such an
          HWC array as channels-first). idefics3_*: SmolVLM-256M-Instruct's processor
          settings (size longest_edge 2048, max_image_size 512,
          image_seq_len 64): the resized size, rows, cols from
          Idefics3ImageProcessorPil._preprocess (with np_resize patched to
          return zeros of the requested size, so only the size arithmetic
          runs), and the number of <image> placeholders in the prompt
          string of Idefics3Processor.replace_image_token. llava_*:
          LLaVA-v1.6 settings (pinpoints [[336, 672], [672, 336],
          [672, 672], [1008, 336], [336, 1008]], tile 336, patch 14, one
          CLS removed by the 'default' strategy): select_best_resolution
          and LlavaNextProcessor.replace_image_token's token count.
prompt.*  replace_image_token strings for (rows, cols) in (0, 0), (1, 2),
          (2, 3) with image_seq_len 3 (UTF-8 bytes).
idef.*    real tiles: Idefics3ImageProcessorPil(size longest_edge 48,
          max_image_size 16, no rescale, no normalize) on three random
          uint8 images (23 x 37, 40 x 12, 10 x 10): every frame as uint8
          CHW, plus rows and cols.
anyres.*  LlavaNextImageProcessorPil(shortest_edge 16, crop 16, pinpoints
          [[16, 32], [32, 16], [32, 32], [48, 16]], no rescale, no
          normalize) on 20 x 45 and 33 x 30 images: base then tiles.
shuffle.* Idefics3Connector.pixel_shuffle on x [2, 16, 6], r = 2.

  uv run --offline --python 3.12 --script course/oracle/L13.4/tiling_hf.py
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import torch
import transformers
import transformers.image_processing_backends as backends
from transformers.image_processing_utils import select_best_resolution
from transformers.models.idefics3.image_processing_pil_idefics3 import (
    Idefics3ImageProcessorPil,
)
from transformers.models.idefics3.modeling_idefics3 import Idefics3Connector
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
from transformers.models.llava_next.image_processing_pil_llava_next import (
    LlavaNextImageProcessorPil,
)
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessor

OUT = Path("course/fixtures/L13.4/tiling_hf.npz")
SIZES = [
    (2, 2),
    (10, 10),
    (512, 512),
    (513, 512),
    (512, 513),
    (300, 400),
    (400, 300),
    (480, 640),
    (640, 480),
    (720, 1280),
    (1280, 720),
    (1080, 1920),
    (1920, 1080),
    (1000, 1000),
    (1024, 1024),
    (2048, 2048),
    (3000, 4000),
    (4000, 3000),
    (100, 2000),
    (2000, 100),
    (50, 3000),
    (3000, 50),
    (336, 336),
    (336, 672),
    (672, 336),
    (500, 1500),
    (1500, 500),
    (777, 333),
    (333, 777),
    (1023, 1025),
    (1025, 1023),
    (256, 1024),
    (1024, 256),
    (4, 512),
    (512, 1),
    (64, 48),
    (900, 1600),
    (1600, 900),
    (5, 7),
    (2047, 1),
]
PINPOINTS = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
PROC_NS = SimpleNamespace(
    fake_image_token="<fake_token_around_image>",
    image_token="<image>",
    global_image_tag="<global-img>",
    image_seq_len=64,
)


def idefics3_prompt(rows, cols, seq):
    ns = SimpleNamespace(**{**vars(PROC_NS), "image_seq_len": seq})
    return Idefics3Processor.replace_image_token(
        ns, {"rows": [[rows]], "cols": [[cols]]}, 0
    )


def main() -> None:
    gen = np.random.default_rng(20261340)
    arrs: dict[str, np.ndarray] = {}

    # -- the 40-size table: idefics3 sizes from the processor's own code ----------
    real_np_resize = backends.np_resize
    sizes_seen = []

    def fake_resize(image, size, **kw):
        sizes_seen.append(tuple(size))
        return np.zeros((image.shape[0], *size), dtype=image.dtype)

    backends.np_resize = fake_resize
    proc = Idefics3ImageProcessorPil(
        size={"longest_edge": 2048},
        max_image_size={"longest_edge": 512},
        do_rescale=False,
        do_normalize=False,
        do_pad=False,
    )
    rows, cols, rh, rw, idef_tokens = [], [], [], [], []
    for h, w in SIZES:
        sizes_seen.clear()
        enc = proc(np.zeros((h, w, 3), np.uint8), return_row_col_info=True)
        r, c = enc["rows"][0][0], enc["cols"][0][0]
        rows.append(r)
        cols.append(c)
        rh.append(sizes_seen[1][0])
        rw.append(sizes_seen[1][1])
        idef_tokens.append(idefics3_prompt(r, c, 64).count("<image>"))
    backends.np_resize = real_np_resize

    lns = SimpleNamespace(
        image_processor=SimpleNamespace(image_grid_pinpoints=PINPOINTS),
        patch_size=14,
        num_additional_image_tokens=1,
        vision_feature_select_strategy="default",
        image_token="<image>",
    )
    lns._get_number_of_features = lambda *a: LlavaNextProcessor._get_number_of_features(
        lns, *a
    )
    lns._get_unpadded_features = lambda *a: LlavaNextProcessor._get_unpadded_features(
        lns, *a
    )
    best, llava_tokens = [], []
    for h, w in SIZES:
        best.append(select_best_resolution((h, w), PINPOINTS))
        inputs = {"image_sizes": [[h, w]], "pixel_values": [[np.zeros((3, 336, 336))]]}
        llava_tokens.append(
            LlavaNextProcessor.replace_image_token(lns, inputs, 0).count("<image>")
        )
    arrs.update(
        {
            "table.hw": np.array(SIZES),
            "table.idefics3_resized": np.array([rh, rw]).T,
            "table.idefics3_rows": np.array(rows),
            "table.idefics3_cols": np.array(cols),
            "table.idefics3_tokens": np.array(idef_tokens),
            "table.llava_best": np.array(best),
            "table.llava_tokens": np.array(llava_tokens),
            "table.pinpoints": np.array(PINPOINTS),
        }
    )
    for r, c in ((0, 0), (1, 2), (2, 3)):
        arrs[f"prompt.{r}x{c}"] = np.frombuffer(
            idefics3_prompt(r, c, 3).encode(), dtype=np.uint8
        )

    # -- real tiles ---------------------------------------------------------------
    small = Idefics3ImageProcessorPil(
        size={"longest_edge": 48},
        max_image_size={"longest_edge": 16},
        do_rescale=False,
        do_normalize=False,
        do_pad=False,
    )
    for i, (h, w) in enumerate([(23, 37), (40, 12), (10, 10)]):
        img = gen.integers(0, 256, (h, w, 3), dtype=np.uint8)
        enc = small(img, return_row_col_info=True)
        frames = np.stack([np.asarray(f) for f in enc["pixel_values"][0]])
        arrs[f"idef.{i}.img"] = img
        arrs[f"idef.{i}.frames"] = np.rint(frames).astype(np.uint8)
        arrs[f"idef.{i}.grid"] = np.array([enc["rows"][0][0], enc["cols"][0][0]])
    ll = LlavaNextImageProcessorPil(
        size={"shortest_edge": 16},
        crop_size={"height": 16, "width": 16},
        image_grid_pinpoints=[[16, 32], [32, 16], [32, 32], [48, 16]],
        do_rescale=False,
        do_normalize=False,
        do_pad=False,
    )
    for i, (h, w) in enumerate([(20, 45), (33, 30)]):
        img = gen.integers(0, 256, (h, w, 3), dtype=np.uint8)
        enc = ll(img)
        arrs[f"anyres.{i}.img"] = img
        arrs[f"anyres.{i}.frames"] = np.rint(np.asarray(enc["pixel_values"][0])).astype(
            np.uint8
        )
        arrs[f"anyres.{i}.best"] = np.array(
            select_best_resolution((h, w), [[16, 32], [32, 16], [32, 32], [48, 16]])
        )

    x = gen.standard_normal((2, 16, 6)).astype(np.float32)
    conn = SimpleNamespace()
    arrs["shuffle.x"] = x
    arrs["shuffle.y"] = Idefics3Connector.pixel_shuffle(
        conn, torch.from_numpy(x), 2
    ).numpy()

    meta = {
        "generator": "course/oracle/L13.4/tiling_hf.py",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    b = buf.getvalue()
    OUT.write_bytes(b)
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L13.4/tiling_hf.py\t"
        f"torch=={torch.__version__.split('+')[0]},transformers=={transformers.__version__},pillow=={PIL.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
