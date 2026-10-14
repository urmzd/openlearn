# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.4"]
# ///
"""Generate independent Whisper decoder logits from the committed tiny HF model.

Run with the exact torch/transformers/numpy versions recorded in the fixture manifest.
No course reference implementation is imported.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import WhisperForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path("course/fixtures")
    model = WhisperForConditionalGeneration.from_pretrained(
        str(root / "L14.2/tiny-whisper-128"),
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    mel = np.load(root / "L14.2/encoder_hf.npz")["small.mel"]
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    with torch.no_grad():
        logits = model(
            input_features=torch.from_numpy(mel),
            decoder_input_ids=torch.from_numpy(ids),
        ).logits.numpy()
    versions = f"torch=={torch.__version__},transformers=={transformers.__version__},numpy=={np.__version__}"
    path = root / "L14.3/decoder_hf.npz"
    if args.check:
        with np.load(path) as old:
            np.testing.assert_array_equal(old["ids"], ids)
            np.testing.assert_array_equal(
                old["parameter_count"], model.num_parameters()
            )
            # CPU kernels vary across hosted runner architectures. Allow two
            # float32 accumulation error units at the fixture's 1e-5 scale;
            # learner conformance retains its stricter 1e-5 absolute bound.
            np.testing.assert_allclose(old["logits"], logits, atol=2e-5, rtol=0)
        print("Whisper decoder oracle matches the committed fixture")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=ids,
        logits=logits,
        parameter_count=np.array(model.num_parameters()),
        __meta__=np.frombuffer(
            json.dumps({"versions": versions}).encode(), dtype=np.uint8
        ),
    )
    data = path.read_bytes()
    print(
        f"{path}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L14.3/decoder_hf.py\t{versions}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
