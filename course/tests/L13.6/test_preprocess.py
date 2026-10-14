"""L13.6 composition: frozen HF processors, geometry, media identity and bounds."""

import json
import os
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from tinyllm.vision.preprocess import (
    ProcessorConfig,
    media_hash,
    preprocess,
    preprocess_pixels,
)

ROOT = Path(os.environ.get("TINYLLM_FIXTURES", ""))
GOLDEN = ROOT / "L13.6/preprocess_hf.npz"
CASES = [
    *(f"{kind}{k}" for k in range(4) for kind in ("clip", "siglip", "idefics")),
    "crop_pad",
    "idefics_noresize",
    "idefics_low",
    "anyres",
]


def load_config(tmp_path, raw):
    path = tmp_path / "preprocessor_config.json"
    path.write_text(json.dumps(raw))
    return ProcessorConfig.from_hf(str(path))


@pytest.mark.parametrize("name", CASES)
def test_hf_processors(name, tmp_path):
    # WHY: crop position, resize filters, tiling order and normalization must match training.
    # KIND: golden
    # CATCHES: s09, s10, s11, s12, s13
    # CHAPTER: L13.6 section 3
    with np.load(GOLDEN) as z:
        cfg = load_config(tmp_path, json.loads(z[name + ".config"].tobytes()))
        for i in range(3):
            key = f"{name}.{i}"
            pixels = z[key + ".input"]
            before = pixels.copy()
            result = preprocess_pixels([pixels], cfg)[0]
            raw = preprocess_pixels(
                [pixels], replace(cfg, do_rescale=False, do_normalize=False)
            )[0]
            assert_close(raw.pixel_values, z[key + ".resized"], atol=1, rtol=0)
            assert_close(
                result.pixel_values,
                z[key + ".values"],
                atol=1 / (255 * min(cfg.image_std)),
                rtol=0,
            )
            assert_close(pixels, before, rtol=0, atol=0)
            assert (
                result.pixel_values.dtype == np.float32
                and result.pixel_values.flags.c_contiguous
            )
            assert result.pixel_mask.shape == (
                len(result.pixel_values),
                *result.pixel_values.shape[-2:],
            )
            assert result.pixel_mask.all() and result.pixel_mask.dtype == np.bool_
            assert result.original_size == pixels.shape[:2]
            if name.startswith("idefics"):
                assert result.grid == tuple(z[key + ".grid"])
                assert_close(result.pixel_mask, z[key + ".mask"], rtol=0, atol=0)
                assert result.n_tokens == len(result.pixel_values) * cfg.image_seq_len
            elif name != "anyres":
                assert result.grid == (0, 0) and result.n_tokens == cfg.image_seq_len
            else:
                assert result.n_tokens == int(z[key + ".tokens"])
                low = preprocess_pixels([pixels], cfg, "low")[0]
                assert low.n_tokens == 17 and low.grid == (0, 0)
                assert len(low.pixel_values) == 1


def test_hand_example_identity(tmp_path):
    # WHY: already-sized pixels need no interpolation, normalization order is observable by hand.
    # KIND: unit, smoke
    # CATCHES: s11, s14
    # CHAPTER: L13.6 section 3
    cfg = load_config(
        tmp_path,
        {
            "image_processor_type": "SiglipImageProcessor",
            "size": {"height": 2, "width": 2},
            "image_seq_len": 1,
            "image_mean": [0.1, 0.2, 0.3],
            "image_std": [0.5, 1.0, 2.0],
        },
    )
    pixels = np.array(
        [[[0, 127, 255], [255, 0, 127]], [[17, 33, 49], [81, 97, 113]]], dtype=np.uint8
    )
    result = preprocess_pixels([pixels], cfg)[0]
    expected = (pixels.astype(np.float64) / 255 - [0.1, 0.2, 0.3]) / [0.5, 1.0, 2.0]
    assert_close(result.pixel_values[0], expected.transpose(2, 0, 1), atol=1e-7, rtol=0)
    raw = preprocess_pixels(
        [pixels], replace(cfg, do_rescale=False, do_normalize=False)
    )[0]
    assert_close(raw.pixel_values[0], pixels.transpose(2, 0, 1), rtol=0, atol=0)
    again = preprocess_pixels(
        [raw.pixel_values[0].transpose(1, 2, 0).astype(np.uint8)], cfg
    )[0]
    assert_close(again.pixel_values, result.pixel_values, rtol=0, atol=0)
    assert result.media_hash == media_hash(
        "image", b"RGB8" + struct.pack("<II", 2, 2) + pixels.tobytes()
    )
    assert preprocess_pixels([], cfg) == []


def test_encoded_identity_and_batch(tmp_path):
    # WHY: encoded-byte identities distinguish media, independent of decode-equivalent pixels.
    # KIND: property
    # CATCHES: s14, s15
    # CHAPTER: L13.6 section 4
    cfg = load_config(
        tmp_path,
        {
            "image_processor_type": "SiglipImageProcessor",
            "size": {"height": 4, "width": 4},
            "image_seq_len": 1,
        },
    )
    with np.load(ROOT / "L13.6/decode_pil.npz") as z:
        images = [z[name + ".bytes"].tobytes() for name in ("rgb", "exif6")]
        decoded = [z[name + ".rgb"] for name in ("rgb", "exif6")]
    got = preprocess(images, cfg)
    direct = preprocess_pixels(decoded, cfg)
    for data, a, b in zip(images, got, direct, strict=True):
        assert_close(a.pixel_values, b.pixel_values, rtol=0, atol=0)
        assert a.media_hash == media_hash("image", data)
        assert a.media_hash != b.media_hash
    assert preprocess([], cfg) == []


def test_detail_overrides(tmp_path):
    # WHY: low detail suppresses tiles and high detail enables them even when the config does not.
    # KIND: unit
    # CATCHES: s12, s16
    # CHAPTER: L13.6 section 4
    cfg = load_config(
        tmp_path,
        {
            "image_processor_type": "Idefics3ImageProcessor",
            "size": {"longest_edge": 24},
            "max_image_size": {"longest_edge": 8},
            "image_seq_len": 4,
        },
    )
    pixels = np.arange(9 * 17 * 3, dtype=np.uint8).reshape(9, 17, 3)
    low = preprocess_pixels([pixels], cfg, "low")[0]
    assert low.grid == (0, 0) and low.n_tokens == 4 and len(low.pixel_values) == 1
    high = preprocess_pixels([pixels], replace(cfg, do_image_splitting=False), "high")[
        0
    ]
    auto = preprocess_pixels([pixels], cfg)[0]
    assert_close(high.pixel_values, auto.pixel_values, rtol=0, atol=0)
    assert high.n_tokens == 28 and high.grid == (2, 3)
    assert (
        preprocess_pixels([pixels], replace(cfg, do_image_splitting=False))[0].n_tokens
        == 4
    )


@pytest.mark.parametrize(
    "override",
    [
        {"resample": 4},
        {"resample": 5},
        {"image_processor_type": "UnknownProcessor"},
        {"image_std": [1, 0, 1]},
        {"image_mean": [1, 2]},
        {"rescale_factor": float("nan")},
        {"do_resize": "false"},
        {"size": {"width": 4}},
        {"size": {"shortest_edge": -1}},
        {"patch_size": 0},
        {"preprocess_rev": 2**32},
        {"image_mean": None},
    ],
)
def test_bad_config(override, tmp_path):
    # WHY: bad model configs must fail before memory allocation or arithmetic.
    # KIND: boundary
    # CATCHES: s17
    # CHAPTER: L13.6 section 4
    with pytest.raises(ValueError):
        load_config(tmp_path, override)


def test_companion_geometry(tmp_path):
    # WHY: HF stores image token geometry separately from pixel transforms.
    # KIND: unit
    # CATCHES: s18
    # CHAPTER: L13.6 section 4
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"image_seq_len": 64, "scale_factor": 4, "patch_size": 16})
    )
    cfg = load_config(
        tmp_path,
        {
            "image_processor_type": "Idefics3ImageProcessor",
            "max_image_size": {"longest_edge": 512},
            "size": {"longest_edge": 2048},
            "unused_future_option": 7,
        },
    )
    assert (cfg.tile_size, cfg.scale_factor, cfg.patch_size, cfg.image_seq_len) == (
        512,
        4,
        16,
        64,
    )
    assert ProcessorConfig.from_hf(str(tmp_path)) == cfg
    square = load_config(
        tmp_path, {"image_processor_type": "CLIPImageProcessor", "image_seq_len": 9}
    )
    assert square.image_seq_len == 9


def test_input_and_allocation_bounds(tmp_path):
    # WHY: malformed arrays and oversized intermediate resizes must fail before allocation.
    # KIND: boundary
    # CATCHES: s19
    # CHAPTER: L13.6 section 4
    cfg = load_config(
        tmp_path,
        {
            "image_processor_type": "SiglipImageProcessor",
            "size": {"height": 8, "width": 8},
            "image_seq_len": 4,
            "max_pixels": 100,
        },
    )
    for a in (
        np.zeros((3, 4)),
        np.zeros((4, 4, 3)),
        np.zeros((0, 4, 3), np.uint8),
        np.zeros((4, 4, 4), np.uint8),
        np.zeros((11, 10, 3), np.uint8),
    ):
        with pytest.raises(ValueError):
            preprocess_pixels([a], cfg)
    with pytest.raises(ValueError, match="max_pixels"):
        preprocess_pixels(
            [np.zeros((2, 2, 3), np.uint8)],
            replace(cfg, size={"height": 100, "width": 100}),
        )
    with pytest.raises(ValueError, match="detail"):
        preprocess_pixels([], cfg, "medium")
    with pytest.raises(ValueError, match="detail"):
        preprocess([], cfg, "medium")
