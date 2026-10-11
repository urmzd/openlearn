"""tinyllm.io.wav (L14.1): WAV files from the bytes up.

A WAV file is a RIFF container: a 12-byte header ("RIFF", a little-endian
u32 size, "WAVE") followed by chunks, each a 4-byte id, a u32 size, the
payload, and one pad byte when the size is odd. Two chunks matter: "fmt "
says how samples are encoded (PCM integers or IEEE floats, how many bits,
how many channels, the sample rate), and "data" holds the interleaved
frames. Everything else (LIST metadata, fact, cue points) is skipped. The
parser trusts nothing: every size is checked against the bytes present,
because a size field is the easiest place for a broken or hostile file to
lie. The engine's Rust parser (L15.1) follows the same rules.

Contract: contracts/py/tinyllm/io/wav.pyi.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

_PCM = 1
_FLOAT = 3
_EXTENSIBLE = 0xFFFE
_MAX_SR = 384000


@dataclass
class Wav:
    samples: NDArray  # float32 [channels, frames]
    sr: int
    channels: int
    fmt: Literal["s16", "s24", "f32"]


def read_wav(data: bytes) -> Wav:
    # SOLUTION-BEGIN L14.1
    b = bytes(data)
    if len(b) < 12 or b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("read_wav: not a RIFF/WAVE file (header must be RIFF....WAVE)")
    (riff_size,) = struct.unpack_from("<I", b, 4)
    end = 8 + riff_size
    if end > len(b):
        raise ValueError(
            f"read_wav: RIFF size {riff_size} runs past the {len(b)} bytes present (truncated file)"
        )
    pos = 12
    fmt = None
    pcm = None
    while pos + 8 <= end:
        cid = b[pos : pos + 4]
        (size,) = struct.unpack_from("<I", b, pos + 4)
        body = pos + 8
        if body + size > end:
            raise ValueError(
                f"read_wav: chunk {cid!r} of {size} bytes runs past the end of the file (truncated)"
            )
        if cid == b"fmt ":
            fmt = _parse_fmt(b[body : body + size])
        elif cid == b"data":
            if fmt is None:
                raise ValueError("read_wav: data chunk before the fmt chunk")
            pcm = b[body : body + size]
            break
        # Chunks are word-aligned: an odd size is followed by one pad byte.
        pos = body + size + (size & 1)
    if fmt is None:
        raise ValueError("read_wav: no fmt chunk")
    if pcm is None:
        raise ValueError("read_wav: no data chunk")
    kind, channels, sr, block = fmt
    if len(pcm) % block:
        raise ValueError(
            f"read_wav: data size {len(pcm)} is not a whole number of {block}-byte frames"
        )
    frames = len(pcm) // block
    if kind == "s16":
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    elif kind == "s24":
        u = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)  # sign-extend 24 bits
        x = v.astype(np.float32) / np.float32(8388608.0)
    else:
        x = np.frombuffer(pcm, dtype="<f4").astype(np.float32)
    # Interleaved L R L R ... -> [channels, frames].
    samples = np.ascontiguousarray(x.reshape(frames, channels).T)
    return Wav(samples=samples, sr=sr, channels=channels, fmt=kind)
    # SOLUTION-END


def _parse_fmt(f: bytes) -> tuple[str, int, int, int]:
    """(kind, channels, sample_rate, block_align) of a fmt chunk payload."""
    # SOLUTION-BEGIN L14.1
    if len(f) < 16:
        raise ValueError(f"read_wav: fmt chunk of {len(f)} bytes, need at least 16")
    tag, channels, sr, byte_rate, block, bits = struct.unpack_from("<HHIIHH", f, 0)
    if tag == _EXTENSIBLE:
        if len(f) < 40:
            raise ValueError("read_wav: WAVE_FORMAT_EXTENSIBLE needs a 40-byte fmt chunk")
        (tag,) = struct.unpack_from("<H", f, 24)  # first two bytes of the GUID
    if tag == _PCM and bits == 16:
        kind = "s16"
    elif tag == _PCM and bits == 24:
        kind = "s24"
    elif tag == _FLOAT and bits == 32:
        kind = "f32"
    else:
        raise ValueError(
            f"read_wav: unsupported encoding (format {tag}, {bits} bits); "
            "need PCM 16 or 24 bits or IEEE float 32 bits"
        )
    if channels not in (1, 2):
        raise ValueError(f"read_wav: {channels} channels; mono or stereo only")
    if not 1 <= sr <= _MAX_SR:
        raise ValueError(f"read_wav: sample rate {sr} outside [1, {_MAX_SR}]")
    if block != channels * bits // 8 or byte_rate != sr * block:
        raise ValueError(
            f"read_wav: block_align {block} or byte_rate {byte_rate} inconsistent "
            f"with {channels} channels of {bits} bits at {sr} Hz"
        )
    return kind, channels, sr, block
    # SOLUTION-END


def write_wav(samples: NDArray, sr: int) -> bytes:
    # SOLUTION-BEGIN L14.1
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"write_wav: need 1-D samples, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("write_wav: samples must be finite")
    if not 1 <= int(sr) <= _MAX_SR:
        raise ValueError(f"write_wav: sample rate {sr} outside [1, {_MAX_SR}]")
    q = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
    pcm = q.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, _PCM, 1, int(sr), int(sr) * 2, 2, 16)
    return header + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    # SOLUTION-END
