# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "soundfile==0.13.1"]
# ///
"""Extract three attributed MLS excerpts without downloading giant archives.

The first FLAC in each official language archive supplies up to eight seconds.
Selection never consults model predictions. The bounded stream is intentionally
partial: provenance records the source member and its SHA-256, not a claim to
have verified the entire 50–115 GB archive. Audio is licensed CC BY 4.0.
"""

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf

from build_asr_mini import wav

LANGUAGES = {"de": "german", "es": "spanish", "fr": "french"}


class BoundedReader:
    def __init__(self, source):
        self.source = source
        self.count = 0

    def read(self, size=-1):
        if size < 0:
            raise ValueError("unbounded archive read refused")
        if self.count + size > 64 * 1024 * 1024:
            raise ValueError("no early MLS audio within 64 MiB compressed budget")
        data = self.source.read(size)
        self.count += len(data)
        return data


def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for language, name in LANGUAGES.items():
        url = f"https://dl.fbaipublicfiles.com/mls/mls_{name}.tar.gz"
        print(
            f"Reading bounded MLS {language} archive prefix",
            file=sys.stderr,
            flush=True,
        )
        with urlopen(url, timeout=30) as response:
            bounded = BoundedReader(response)
            with tarfile.open(fileobj=bounded, mode="r|gz") as archive:
                for member in archive:
                    if member.offset > 256 * 1024 * 1024:
                        raise ValueError(
                            "no early MLS audio within 256 MiB unpacked budget"
                        )
                    if not member.isfile() or not member.name.endswith(".flac"):
                        continue
                    if member.size > 8 * 1024 * 1024:
                        raise ValueError(
                            f"unexpectedly large first FLAC: {member.name}"
                        )
                    payload = archive.extractfile(member).read()
                    samples, rate = sf.read(io.BytesIO(payload), dtype="int16")
                    if rate != 16000 or samples.ndim != 1 or len(samples) < 16000:
                        raise ValueError(
                            f"unexpected first MLS clip geometry: {member.name}"
                        )
                    frames = min(len(samples), 8 * rate)
                    audio = wav(np.asarray(samples[:frames], dtype=np.int16))
                    relative = f"lang-{language}.wav"
                    (out / relative).write_bytes(audio)
                    cases.append(
                        {
                            "language": language,
                            "audio": relative,
                            "source_url": url,
                            "source_member": member.name,
                            "source_flac_sha256": hashlib.sha256(payload).hexdigest(),
                            "source_frames": len(samples),
                            "start_frame": 0,
                            "excerpt_frames": frames,
                            "rate": rate,
                            "sha256": hashlib.sha256(audio).hexdigest(),
                            "license": "CC-BY-4.0",
                            "attribution": "Multilingual LibriSpeech, Pratap et al. (2020), derived from LibriVox",
                            "license_url": "https://creativecommons.org/licenses/by/4.0/",
                            "dataset_url": "https://www.openslr.org/94/",
                        }
                    )
                    break
                else:
                    raise ValueError(f"MLS {language} archive contains no FLAC")
    (out / "languages.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    generate(args.out)


if __name__ == "__main__":
    main()
