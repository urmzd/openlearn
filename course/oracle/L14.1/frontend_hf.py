# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3", "scipy==1.17.1", "soundfile==0.13.1"]
# ///
"""Maintainer generator for the L14.1 fixtures (WAV parsing and Whisper
log-mel).

course/fixtures/L14.1/wav/: hand-built RIFF files covering what read_wav
must handle, written byte by byte here (not by a library, so every corner
is deliberate):

  s16-mono.wav       PCM 16-bit mono 16 kHz, plain 44-byte header
  s24-mono.wav       PCM 24-bit mono 16 kHz
  f32-stereo.wav     IEEE float 32-bit stereo 22050 Hz, an 18-byte fmt
                     chunk (cb_size 0) and a fact chunk
  s16-list-odd.wav   PCM 16-bit stereo 8 kHz with a LIST chunk of odd size
                     (and its pad byte) between fmt and data, and an odd
                     junk chunk before fmt
  ext-s24-stereo.wav WAVE_FORMAT_EXTENSIBLE, 24-bit PCM subformat, stereo
                     48 kHz, with trailing bytes after the RIFF size
  trunc-data.wav     the data chunk claims more bytes than the file has
  riff-too-big.wav   the RIFF size claims about 4 GiB

wav_scipy.npz: scipy.io.wavfile.read of each good file, scaled to [-1, 1)
(int16 / 2^15, int32 left-justified 24-bit / 2^31, float32 as is), as
float32 [channels, frames], and its rate.

course/fixtures/L14.1/clips/en-{0..5}.wav: six LibriSpeech test-clean
utterances of 2 to 4 s (CC BY 4.0), s16 mono 16 kHz, the first six by
utterance id with distinct speakers; clips/clips.json has the attribution
(speaker, chapter, utterance id) and the transcript of each.

hf_features.npz: HF WhisperFeatureExtractor (feature_size 80 and 128,
padding to 30 s as Whisper does) on each clip as read back from its s16 WAV:
`en-{i}.f80`, `en-{i}.f128`, float32 [n_mels, 3000].

    uv run --script course/oracle/L14.1/frontend_hf.py <LibriSpeech dir>

The LibriSpeech dir is the extracted test-clean.tar.gz (ASSETS.tsv row
librispeech-test-clean). Run from the repo root and paste the printed rows
into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.io.wavfile
import soundfile
import torch
import transformers
from transformers import WhisperFeatureExtractor

OUT = Path("course/fixtures/L14.1")
GEN = "course/oracle/L14.1/frontend_hf.py"
VERS = (
    f"torch=={torch.__version__},transformers=={transformers.__version__},"
    f"scipy=={scipy.__version__},soundfile=={soundfile.__version__}"
)
rng = np.random.default_rng(20261011)
ROWS: list[str] = []


def chunk(cid: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) % 2 else b""
    return cid + struct.pack("<I", len(payload)) + payload + pad


def fmt_chunk(
    tag: int, ch: int, sr: int, bits: int, extra: bytes | None = None
) -> bytes:
    block = ch * bits // 8
    body = struct.pack("<HHIIHH", tag, ch, sr, sr * block, block, bits)
    if extra is not None:
        body += extra
    return chunk(b"fmt ", body)


def riff(chunks: list[bytes], size: int | None = None, trailing: bytes = b"") -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return (
        b"RIFF"
        + struct.pack("<I", len(body) if size is None else size)
        + body
        + trailing
    )


def tone(frames: int, ch: int, sr: int) -> np.ndarray:
    t = np.arange(frames) / sr
    x = [
        0.5 * np.sin(2 * np.pi * (220.0 * (c + 1)) * t)
        + 0.05 * rng.standard_normal(frames)
        for c in range(ch)
    ]
    return np.clip(np.stack(x, axis=1), -1.0, 0.999)  # [frames, ch]


def s16(x: np.ndarray) -> bytes:
    return np.round(x * 32767).astype("<i2").tobytes()


def s24(x: np.ndarray) -> bytes:
    v = np.round(x * 8388607).astype(np.int64).ravel()
    v = np.where(v < 0, v + (1 << 24), v).astype(np.uint32)
    b = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=1).astype(np.uint8)
    return b.tobytes()


def write(rel: str, data: bytes, upstream: str = "-", lic: str = "Apache-2.0") -> None:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    ROWS.append(
        f"{p}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\t{GEN}\t{VERS}\t{upstream}\t{lic}"
    )


def savez(rel: str, arrs: dict) -> None:
    arrs = dict(arrs)
    meta = {"generator": GEN, "versions": VERS}
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    write(rel, buf.getvalue())


def wav_cases() -> dict[str, bytes]:
    cases: dict[str, bytes] = {}
    x = tone(1600, 1, 16000)
    cases["s16-mono"] = riff([fmt_chunk(1, 1, 16000, 16), chunk(b"data", s16(x))])
    x = tone(1601, 1, 16000)  # odd frame count: 4803 data bytes, then a pad byte
    cases["s24-mono"] = riff([fmt_chunk(1, 1, 16000, 24), chunk(b"data", s24(x))])
    x = tone(2205, 2, 22050).astype("<f4")
    cases["f32-stereo"] = riff(
        [
            fmt_chunk(3, 2, 22050, 32, struct.pack("<H", 0)),
            chunk(b"fact", struct.pack("<I", 2205)),
            chunk(b"data", x.tobytes()),
        ]
    )
    x = tone(800, 2, 8000)
    info = chunk(b"INAM", b"tinyllm fixture\x00\x00")  # 17 bytes with the NUL: odd
    cases["s16-list-odd"] = riff(
        [
            chunk(b"junk", b"abc"),  # odd size before fmt
            fmt_chunk(1, 2, 8000, 16),
            chunk(b"LIST", b"INFO" + info[:-1]),  # odd LIST payload, padded
            chunk(b"data", s16(x)),
        ]
    )
    x = tone(960, 2, 48000)
    guid_pcm = struct.pack("<H", 1) + bytes.fromhex("000000001000800000aa00389b71")
    ext = struct.pack("<HHI", 22, 24, 0x3) + guid_pcm
    cases["ext-s24-stereo"] = riff(
        [fmt_chunk(0xFFFE, 2, 48000, 24, ext), chunk(b"data", s24(x))],
        trailing=b"TRAILING",
    )
    good = riff([fmt_chunk(1, 1, 16000, 16), chunk(b"data", s16(tone(400, 1, 16000)))])
    # data claims 800 bytes but the file stops after 300 of them.
    hdr = good[:40] + struct.pack("<I", 800)
    trunc = hdr + good[44 : 44 + 300]
    cases["trunc-data"] = trunc[:4] + struct.pack("<I", len(trunc) - 8) + trunc[8:]
    cases["riff-too-big"] = good[:4] + struct.pack("<I", 0xFFFFFFF0) + good[8:]
    return cases


def scipy_read(data: bytes) -> tuple[np.ndarray, int]:
    sr, a = scipy.io.wavfile.read(io.BytesIO(data))
    if a.dtype == np.int16:
        x = a.astype(np.float64) / 2**15
    elif a.dtype == np.int32:
        x = a.astype(np.float64) / 2**31  # scipy left-justifies 24-bit samples
    else:
        x = a.astype(np.float64)
    x = x.reshape(len(x), -1).T
    return x.astype(np.float32), int(sr)


def librispeech_clips(root: Path) -> list[dict]:
    base = root / "test-clean"
    picked, speakers = [], set()
    for flac in sorted(base.rglob("*.flac"), key=lambda p: p.stem):
        spk, chap, _ = flac.stem.split("-")
        if spk in speakers:
            continue
        info = soundfile.info(str(flac))
        if not 2.0 <= info.duration <= 4.0:
            continue
        trans = (flac.parent / f"{spk}-{chap}.trans.txt").read_text().splitlines()
        text = next(t.split(" ", 1)[1] for t in trans if t.startswith(flac.stem + " "))
        picked.append(
            {
                "path": flac,
                "id": flac.stem,
                "speaker": spk,
                "chapter": chap,
                "text": text,
            }
        )
        speakers.add(spk)
        if len(picked) == 6:
            break
    return picked


def main() -> None:
    lib = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if lib is None or not (lib / "test-clean").is_dir():
        sys.exit("usage: frontend_hf.py <extracted LibriSpeech dir with test-clean/>")
    cases = wav_cases()
    arrs = {}
    for name, data in cases.items():
        write(f"wav/{name}.wav", data)
        if not name.startswith(("trunc", "riff")):
            x, sr = scipy_read(data)
            arrs[f"{name}.samples"] = x
            arrs[f"{name}.sr"] = np.array(sr)
    savez("wav_scipy.npz", arrs)

    fe = {n: WhisperFeatureExtractor(feature_size=n) for n in (80, 128)}
    feats = {}
    meta = []
    for i, c in enumerate(librispeech_clips(lib)):
        a, sr = soundfile.read(str(c["path"]), dtype="int16")
        assert sr == 16000 and a.ndim == 1
        data = riff(
            [fmt_chunk(1, 1, 16000, 16), chunk(b"data", a.astype("<i2").tobytes())]
        )
        upstream = f"LibriSpeech test-clean {c['id']} (speaker {c['speaker']}, chapter {c['chapter']}), openslr.org/12"
        write(f"clips/en-{i}.wav", data, upstream, "CC-BY-4.0")
        x = a.astype(np.float32) / 32768.0
        for n, f in fe.items():
            feats[f"en-{i}.f{n}"] = (
                f(x, sampling_rate=16000, return_tensors="np")
                .input_features[0]
                .astype(np.float32)
            )
        meta.append(
            {
                "file": f"en-{i}.wav",
                "librispeech_id": c["id"],
                "speaker": c["speaker"],
                "chapter": c["chapter"],
                "seconds": len(a) / 16000,
                "text": c["text"],
                "license": "CC BY 4.0",
                "source": "https://www.openslr.org/12",
            }
        )
    write(
        "clips/clips.json",
        (json.dumps(meta, indent=2) + "\n").encode(),
        "LibriSpeech test-clean transcripts, openslr.org/12",
        "CC-BY-4.0",
    )
    savez("hf_features.npz", feats)
    for r in ROWS:
        print(r)


if __name__ == "__main__":
    main()
