"""Generate independent Whisper decoder logits from the committed tiny HF model.

Run with the exact torch/transformers/numpy versions recorded in the fixture manifest.
No course reference implementation is imported.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import WhisperForConditionalGeneration


def main():
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
