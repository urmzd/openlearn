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
            for score in result.scores:
                top = score[0].topk(2).values
                if torch.isfinite(top).all():
                    margins.append(float(top[0] - top[1]))
            ids = result.sequences[0].tolist()
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
    Path(args.out).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(cases), "out": args.out}))


if __name__ == "__main__":
    main()
