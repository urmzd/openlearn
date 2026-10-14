"""Record pinned HF Whisper-tiny timestamp generation on licensed speech clips.

No learner model or decoder is imported. All checkpoint bytes are verified
against ASSETS.tsv before transformers loads them. Scores preserve near-tie
information rather than silently selecting only matching learner outputs.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperTokenizerFast


def checkpoint(root):
    root.mkdir(parents=True, exist_ok=True)
    with Path("course/fixtures/ASSETS.tsv").open() as source:
        rows = list(csv.reader(source, delimiter="\t"))
    for row in rows:
        if not row or row[0] != "whisper-tiny":
            continue
        _, name, url, sha, size, *_ = row
        path = root / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            continue
        temporary = path.with_suffix(path.suffix + ".partial")
        digest, count = hashlib.sha256(), 0
        with urlopen(url, timeout=60) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
                count += len(chunk)
        if count != int(size) or digest.hexdigest() != sha:
            temporary.unlink()
            raise ValueError(f"checkpoint fingerprint mismatch: {name}")
        temporary.replace(path)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = checkpoint(Path(args.model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        root, local_files_only=True, attn_implementation="eager"
    ).eval()
    tokenizer = WhisperTokenizerFast.from_pretrained(root, local_files_only=True)
    cases = []
    with np.load("course/fixtures/L14.1/hf_features.npz") as features:
        for i in range(6):
            mel = torch.from_numpy(features[f"en-{i}.f80"][None])
            with torch.no_grad():
                result = model.generate(
                    mel,
                    language="en",
                    task="transcribe",
                    do_sample=False,
                    return_timestamps=True,
                    max_new_tokens=128,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            margins = []
            # Timestamp generation retains each underlying decoder result on
            # its segments; several segments can share one decoder invocation.
            decoded = {}
            for segment in result["segments"][0]:
                decoded[id(segment["result"])] = segment["result"]
            for output in decoded.values():
                for score in output["scores"]:
                    top = score.reshape(-1).topk(2).values
                    if torch.isfinite(top).all():
                        margins.append(float(top[0] - top[1]))
            ids = result["sequences"][0].tolist()
            cases.append(
                {
                    "clip": f"en-{i}.wav",
                    "ids": ids,
                    "text": tokenizer.decode(ids, skip_special_tokens=True),
                    "minimum_margin": min(margins) if margins else None,
                    "margins": margins,
                }
            )
    output = {
        "model_revision": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cases": cases,
    }
    if args.check:
        committed = json.loads(
            Path("course/fixtures/whisper-parity/greedy_timestamps.json").read_text()
        )
        if committed["model_revision"] != output["model_revision"]:
            raise ValueError("real Whisper oracle revision drift")
        old_cases = committed["cases"]
        if len(old_cases) != len(cases):
            raise ValueError("real Whisper oracle case count drift")
        for old, new in zip(old_cases, cases):
            for field in ("clip", "ids", "text"):
                if old[field] != new[field]:
                    raise ValueError(
                        f"real Whisper oracle drift: {new['clip']} {field}"
                    )
            if new["minimum_margin"] is None or new["minimum_margin"] < 0.001:
                raise ValueError(f"real Whisper oracle near tie: {new['clip']}")
    Path(args.out).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(cases), "out": args.out}))


if __name__ == "__main__":
    main()
