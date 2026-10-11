# contracts/py/tinyllm/io/wav.pyi (L14.1): WAV files from the bytes up
# chapter: ml/08-tinyllm/p14-speech/01-audio-frontend.md
#
# A WAV file is a RIFF container (D47: the only audio input in v1):
#
#     "RIFF" <u32 riff_size> "WAVE" chunk chunk ...
#     chunk = <4-byte id> <u32 size> <size bytes> [one pad byte when size is odd]
#
# All integers are little-endian. The "fmt " chunk (16, 18, or 40 bytes)
# holds <u16 format> <u16 channels> <u32 sample_rate> <u32 byte_rate>
# <u16 block_align> <u16 bits>, and for format 0xFFFE (WAVE_FORMAT_EXTENSIBLE)
# also <u16 cb_size = 22> <u16 valid_bits> <u32 channel_mask> <16-byte
# subformat GUID>, whose first two bytes are the real format. Accepted
# encodings: PCM (format 1) with 16 or 24 bits, IEEE float (format 3) with 32
# bits; 1 or 2 channels. The "data" chunk holds interleaved frames of
# block_align = channels * bits / 8 bytes. Every other chunk (LIST, fact,
# cue, ...) is skipped, honoring its pad byte. Samples are scaled to
# [-1, 1): s16 / 2^15, s24 / 2^23, f32 as stored.
from dataclasses import dataclass
from typing import Literal

from numpy.typing import NDArray

@dataclass
class Wav:
    samples: NDArray  # float32 [channels, frames]
    sr: int
    channels: int
    fmt: Literal["s16", "s24", "f32"]

def read_wav(data: bytes) -> Wav:
    """Parse a whole WAV file. Bytes after the RIFF size are ignored.
    ValueError, naming the problem, for: no "RIFF"/"WAVE" header, a RIFF
    size or a chunk size that runs past the bytes present (a truncated file),
    a missing "fmt " or "data" chunk or "data" before "fmt ", an unsupported
    format or bit depth, channels outside {1, 2}, a sample rate outside
    [1, 384000], block_align or byte_rate inconsistent with the format, or a
    data size that is not a whole number of frames."""

def write_wav(samples: NDArray, sr: int) -> bytes:
    """A 44-byte-header s16le mono WAV of the 1-D float samples:
    clip(round_half_even(x * 2^15), -2^15, 2^15 - 1). read_wav of the result
    gives back samples that are already multiples of 2^-15 in [-1, 1)
    exactly. ValueError for samples that are not 1-D or not finite, or sr
    outside [1, 384000]."""
