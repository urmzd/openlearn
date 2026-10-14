# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "soundfile==0.13.1"]
# ///
"""Build licensed ASR assets from the hash-pinned LibriSpeech test-clean archive.

Selection is independent of model predictions: the first six numeric speaker
IDs, first four 2–15 second utterances each. Long-form uses consecutive complete
utterances from the first numeric chapter that fills at least 80 of 90 seconds,
then silence pads to exactly 90 seconds. No word is cut and no clip is repeated.
Outputs are learner-local assets, never published audio downloads.
"""

import argparse
import csv
import hashlib
import io
import json
import struct
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf

RATE = 16000


def numeric_id(name):
    return tuple(int(part) for part in Path(name).stem.split("-"))


def download(path):
    with Path("course/fixtures/ASSETS.tsv").open() as source:
        row = next(
            r
            for r in csv.reader(source, delimiter="\t")
            if r and r[0] == "librispeech-test-clean"
        )
    _, _, url, sha, size, *_ = row
    digest, count = hashlib.sha256(), 0
    with urlopen(url, timeout=60) as response, path.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            digest.update(chunk)
            target.write(chunk)
    if count != int(size) or digest.hexdigest() != sha:
        raise ValueError("LibriSpeech source fingerprint mismatch")
    return sha


def wav(samples):
    data = np.asarray(samples, dtype="<i2").tobytes()
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def build(archive, out, source_sha):
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        clips = sorted((n for n in members if n.endswith(".flac")), key=numeric_id)
        texts = {}
        for name in sorted(members):
            if name.endswith(".trans.txt"):
                for line in (
                    tar.extractfile(members[name]).read().decode("utf-8").splitlines()
                ):
                    key, text = line.split(" ", 1)
                    texts[key] = text

        def read(name):
            samples, rate = sf.read(
                io.BytesIO(tar.extractfile(members[name]).read()), dtype="int16"
            )
            if rate != RATE or samples.ndim != 1:
                raise ValueError(f"unexpected source geometry: {name}")
            return samples

        def attribution(name):
            speaker, chapter, utterance = numeric_id(name)
            return {
                "id": Path(name).stem,
                "speaker": speaker,
                "chapter": chapter,
                "utterance": utterance,
                "text": texts[Path(name).stem],
            }

        selected, counts = [], {}
        speakers = sorted({numeric_id(n)[0] for n in clips})[:6]
        for name in clips:
            speaker = numeric_id(name)[0]
            if speaker not in speakers or counts.get(speaker, 0) == 4:
                continue
            samples = read(name)
            if not 2 * RATE <= len(samples) <= 15 * RATE:
                continue
            rel = f"clips/{Path(name).stem}.wav"
            (out / rel).parent.mkdir(exist_ok=True)
            (out / rel).write_bytes(wav(samples))
            selected.append({"audio": rel, **attribution(name)})
            counts[speaker] = counts.get(speaker, 0) + 1
        if len(selected) != 24:
            raise ValueError(
                "source no longer yields the specified 24-utterance corpus"
            )
        (out / "manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
            encoding="utf-8",
        )
        chapters = sorted({numeric_id(n)[:2] for n in clips})
        for chapter in chapters:
            pieces, sources, frames = [], [], 0
            for name in clips:
                if numeric_id(name)[:2] != chapter:
                    continue
                samples = read(name)
                if frames + len(samples) > 90 * RATE:
                    break
                pieces.append(samples)
                sources.append(
                    {**attribution(name), "start_frame": frames, "frames": len(samples)}
                )
                frames += len(samples)
            if frames >= 80 * RATE:
                break
        else:
            raise ValueError(
                "no chapter supplies 80–90 seconds of consecutive complete utterances"
            )
        samples = np.concatenate(
            [*pieces, np.zeros(90 * RATE - frames, dtype=np.int16)]
        )
        (out / "longform-90s.wav").write_bytes(wav(samples))
        (out / "longform-90s.txt").write_text(
            " ".join(s["text"] for s in sources) + "\n", encoding="utf-8"
        )
    provenance = {
        "source": "https://www.openslr.org/12/",
        "license": "CC-BY-4.0",
        "source_sha256": source_sha,
        "corpus": selected,
        "longform": {
            "assembly": "consecutive complete chapter utterances followed by zero padding",
            "sources": sources,
            "silence_frames": 90 * RATE - frames,
        },
    }
    (out / "attribution.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    for path in sorted(out.rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            print(
                "\t".join(
                    [
                        "asr-mini",
                        path.relative_to(out).as_posix(),
                        "build:course/oracle/MS-L14/build_asr_mini.py",
                        hashlib.sha256(payload).hexdigest(),
                        str(len(payload)),
                        "CC-BY-4.0",
                        "librispeech-test-clean; deterministic selection v1",
                    ]
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="asr-mini-source-") as temporary:
        archive = Path(temporary) / "test-clean.tar.gz"
        source_sha = download(archive)
        build(archive, args.out, source_sha)


if __name__ == "__main__":
    main()
