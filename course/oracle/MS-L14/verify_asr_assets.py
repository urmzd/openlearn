"""Validate real-corpus geometry and complete-utterance long-form assembly."""

import csv
import hashlib
import json
import sys
import wave
from pathlib import Path


def main(root):
    with Path("course/fixtures/ASSETS.tsv").open() as source:
        registered = [
            r for r in csv.reader(source, delimiter="\t") if r and r[0] == "asr-mini"
        ]
    if not registered:
        raise AssertionError("asr-mini asset fingerprints are not registered")
    for _, name, _, sha, size, *_ in registered:
        payload = (root / name).read_bytes()
        assert (
            len(payload) == int(size) and hashlib.sha256(payload).hexdigest() == sha
        ), name
    attribution = json.loads((root / "attribution.json").read_text())
    rows = [
        json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 24
    assert len({r["id"] for r in rows}) == 24
    speakers = {r["speaker"] for r in rows}
    assert len(speakers) == 6
    assert all(sum(r["speaker"] == s for r in rows) == 4 for s in speakers)
    assert rows == attribution["corpus"]
    for row in rows:
        assert row["text"].strip()
        with wave.open(str(root / row["audio"]), "rb") as audio:
            assert (
                audio.getnchannels(),
                audio.getsampwidth(),
                audio.getframerate(),
            ) == (1, 2, 16000)
            assert 32000 <= audio.getnframes() <= 240000
    longform = attribution["longform"]
    sources = longform["sources"]
    assert len({(s["speaker"], s["chapter"]) for s in sources}) == 1
    assert len({s["id"] for s in sources}) == len(sources)
    frames = 0
    for source in sources:
        assert source["start_frame"] == frames
        frames += source["frames"]
    assert 80 * 16000 <= frames <= 90 * 16000
    assert frames + longform["silence_frames"] == 90 * 16000
    assert (root / "longform-90s.txt").read_text().strip() == " ".join(
        s["text"] for s in sources
    )
    with wave.open(str(root / "longform-90s.wav"), "rb") as audio:
        assert (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
            audio.getnframes(),
        ) == (1, 2, 16000, 90 * 16000)
        audio.setpos(frames)
        assert audio.readframes(longform["silence_frames"]) == bytes(
            longform["silence_frames"] * 2
        )
    print(
        "24 attributed utterances, six speakers, and complete 90-second assembly validated"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
