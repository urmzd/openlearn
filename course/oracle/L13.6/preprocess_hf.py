# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers==5.19.0", "pillow==12.3.0", "numpy==2.5.3"]
# ///
"""Independent HF PIL processor fixtures; no learner imports or model weights."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import transformers
from transformers.models.clip.image_processing_pil_clip import CLIPImageProcessorPil
from transformers.models.idefics3.image_processing_pil_idefics3 import (
    Idefics3ImageProcessorPil,
)
from transformers.models.llava_next.image_processing_pil_llava_next import (
    LlavaNextImageProcessorPil,
)
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessor
from transformers.models.siglip.image_processing_pil_siglip import (
    SiglipImageProcessorPil,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(20261360)
    arrays = {}
    cases = []
    for kernel in range(4):
        cases.extend(
            [
                (
                    f"clip{kernel}",
                    CLIPImageProcessorPil,
                    {
                        "size": {"shortest_edge": 10},
                        "crop_size": {"height": 8, "width": 8},
                        "resample": kernel,
                    },
                ),
                (
                    f"siglip{kernel}",
                    SiglipImageProcessorPil,
                    {"size": {"height": 8, "width": 12}, "resample": kernel},
                ),
                (
                    f"idefics{kernel}",
                    Idefics3ImageProcessorPil,
                    {
                        "size": {"longest_edge": 24},
                        "max_image_size": {"longest_edge": 8},
                        "resample": kernel,
                    },
                ),
            ]
        )
    cases.extend(
        [
            (
                "crop_pad",
                CLIPImageProcessorPil,
                {"do_resize": False, "crop_size": {"height": 12, "width": 14}},
            ),
            (
                "idefics_noresize",
                Idefics3ImageProcessorPil,
                {"do_resize": False, "max_image_size": {"longest_edge": 8}},
            ),
            (
                "idefics_low",
                Idefics3ImageProcessorPil,
                {
                    "size": {"longest_edge": 24},
                    "max_image_size": {"longest_edge": 8},
                    "do_image_splitting": False,
                },
            ),
            (
                "anyres",
                LlavaNextImageProcessorPil,
                {
                    "size": {"shortest_edge": 8},
                    "crop_size": {"height": 8, "width": 8},
                    "image_grid_pinpoints": [[8, 16], [16, 8], [16, 16], [24, 8]],
                },
            ),
        ]
    )
    for name, klass, kwargs in cases:
        proc = klass(**kwargs)
        config = proc.to_dict() | {
            "patch_size": 2,
            "scale_factor": 2 if klass == Idefics3ImageProcessorPil else 1,
            "image_seq_len": 4,
        }
        arrays[f"{name}.config"] = np.frombuffer(
            json.dumps(config, sort_keys=True).encode(), dtype=np.uint8
        )
        for i, hw in enumerate([(9, 17), (19, 6), (5, 5)]):
            key = f"{name}.{i}"
            pixels = rng.integers(0, 256, size=(*hw, 3), dtype=np.uint8)
            extra = (
                {"return_row_col_info": True}
                if klass == Idefics3ImageProcessorPil
                else {}
            )
            output = proc(
                pixels, input_data_format="channels_last", return_tensors="np", **extra
            )
            raw = proc(
                pixels,
                input_data_format="channels_last",
                return_tensors="np",
                do_rescale=False,
                do_normalize=False,
                **extra,
            )
            values = np.asarray(output["pixel_values"])[0]
            resized = np.asarray(raw["pixel_values"])[0]
            if values.ndim == 3:
                values, resized = values[None], resized[None]
            arrays[f"{key}.input"] = pixels
            arrays[f"{key}.values"] = values
            arrays[f"{key}.resized"] = resized.astype(np.uint8)
            if klass == LlavaNextImageProcessorPil:
                geometry = SimpleNamespace(
                    image_processor=proc,
                    patch_size=2,
                    num_additional_image_tokens=1,
                    vision_feature_select_strategy="default",
                    image_token="<image>",
                )
                geometry._get_number_of_features = partial(
                    LlavaNextProcessor._get_number_of_features, geometry
                )
                geometry._get_unpadded_features = partial(
                    LlavaNextProcessor._get_unpadded_features, geometry
                )
                arrays[f"{key}.tokens"] = np.array(
                    LlavaNextProcessor.replace_image_token(geometry, output, 0).count(
                        "<image>"
                    )
                )
            if klass == Idefics3ImageProcessorPil:
                arrays[f"{key}.grid"] = np.array(
                    [output["rows"][0][0], output["cols"][0][0]]
                )
                arrays[f"{key}.mask"] = np.asarray(output["pixel_attention_mask"])[0]
    versions = f"transformers=={transformers.__version__},pillow=={PIL.__version__},numpy=={np.__version__}"
    arrays["__meta__"] = np.frombuffer(
        json.dumps({"versions": versions}).encode(), dtype=np.uint8
    )
    path = Path("course/fixtures/L13.6/preprocess_hf.npz")
    if args.check:
        with np.load(path) as old:
            assert set(old.files) == set(arrays)
            for name, value in arrays.items():
                if name.endswith(".values"):
                    np.testing.assert_allclose(
                        old[name], value, atol=1e-6, rtol=0, err_msg=name
                    )
                else:
                    np.testing.assert_array_equal(old[name], value, err_msg=name)
        print("Image preprocessing oracle matches the committed fixture")
        return
    np.savez_compressed(path, **arrays)
    data = path.read_bytes()
    print(
        f"{path}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L13.6/preprocess_hf.py\t{versions}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
