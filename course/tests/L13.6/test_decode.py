"""L13.6 decode tests use frozen Pillow pixels and encoded synthetic media."""

import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from tinyllm.vision.preprocess import (
    decode_image,
    exif_orientation,
    image_header,
    media_hash,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L13.6/decode_pil.npz"


@pytest.mark.parametrize(
    "name",
    [
        *(f"exif{i}" for i in range(1, 9)),
        "rgb",
        "jpeg",
        "gray",
        "binary",
        "rgba",
        "la",
        "palette",
        "palette_alpha",
        "palette_key",
        "rgb_key",
        "gray_key",
    ],
)
def test_frozen_decode(name):
    # WHY: decode, all EXIF cases, and alpha arithmetic agree with an independent oracle.
    # KIND: golden
    # CATCHES: s01, s02, s03
    # CHAPTER: L13.6 section 3
    with np.load(GOLDEN) as z:
        encoded = z[name + ".bytes"].tobytes()
        got = decode_image(encoded, max_pixels=15)
        want = z[name + ".rgb"]
    assert got.dtype == np.uint8 and got.flags.c_contiguous and got.flags.owndata
    assert_close(got, want, rtol=0, atol=2 if name == "jpeg" else 0)
    assert image_header(encoded) == ("JPEG" if name == "jpeg" else "PNG", 5, 3)
    assert exif_orientation(encoded) == (
        int(name[4:]) if name.startswith("exif") else None
    )


@pytest.mark.parametrize("name", ["cmyk", "png16", "rgb16"])
def test_unsupported_before_load(name, monkeypatch):
    # WHY: unsupported bit depth and color spaces must not be silently converted.
    # KIND: unit
    # CATCHES: s04
    # CHAPTER: L13.6 section 4
    with np.load(GOLDEN) as z:
        encoded = z[name + ".bytes"].tobytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("pixels were decompressed before checking the header")

    monkeypatch.setattr("PIL.Image.Image.load", forbidden)
    with pytest.raises(ValueError, match="unsupported"):
        decode_image(encoded, max_pixels=15)


def test_budget_before_load(monkeypatch):
    # WHY: rejecting a pixel bomb after decompression is too late.
    # KIND: unit
    # CATCHES: s05
    # CHAPTER: L13.6 section 4
    with np.load(GOLDEN) as z:
        encoded = z["rgb.bytes"].tobytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("pixels were decompressed before checking the budget")

    monkeypatch.setattr("PIL.Image.Image.load", forbidden)
    assert image_header(encoded) == ("PNG", 5, 3)
    with pytest.raises(ValueError, match="max_pixels"):
        decode_image(encoded, max_pixels=14)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_pixels"):
            decode_image(encoded, max_pixels=invalid)


def test_corrupt_inputs():
    # WHY: bad containers and truncated pixel streams fail explicitly.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: L13.6 section 4
    with np.load(GOLDEN) as z:
        jpeg = z["jpeg.bytes"].tobytes()
        png = z["rgb.bytes"].tobytes()
    for encoded in (b"", b"not an image", jpeg[:40], jpeg[:-30], png[:45], png[:-35]):
        with pytest.raises(ValueError):
            decode_image(encoded, max_pixels=100)
    with pytest.raises(ValueError):
        image_header(b"broken")
    with pytest.raises(ValueError):
        exif_orientation(b"broken")


def test_invalid_orientation():
    # WHY: invalid orientation tags must not silently produce rotated training data.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: L13.6 section 4
    with np.load(GOLDEN) as z:
        encoded = z["invalid_exif.bytes"].tobytes()
    with pytest.raises(ValueError, match="orientation"):
        exif_orientation(encoded)
    with pytest.raises(ValueError, match="orientation"):
        decode_image(encoded, max_pixels=15)


def test_media_hash_domain():
    # WHY: identical bytes in image and audio namespaces must not share identity.
    # KIND: unit
    # CATCHES: s07
    # CHAPTER: L13.6 section 3
    def independent(data):
        h = 14695981039346656037
        for b in data:
            h = ((h ^ b) * 1099511628211) % (1 << 64)
        return h

    for payload in (b"", b"hello", bytes(range(256))):
        assert media_hash("image", payload) == independent(b"\x01" + payload)
        assert media_hash("audio", payload) == independent(b"\x02" + payload)
        assert media_hash("image", payload) != media_hash("audio", payload)
    with pytest.raises(ValueError):
        media_hash("video", b"")
