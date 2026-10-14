# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the tiny random Whisper models (L14.2, L14.3)
and the L14.2 encoder golden values.

Model directories (transformers 5.19 save_pretrained, float32, random
weights, Apache-2.0):

  course/fixtures/L14.2/tiny-whisper/      WhisperForConditionalGeneration,
      80 mels, max_source_positions 1500 (it reads real 30 s features),
      d_model 32, 4 heads, 2 encoder and 2 decoder layers, ffn 64,
      max_target_positions 64, vocab 100
  course/fixtures/L14.2/tiny-whisper-128/  128 mels, max_source_positions
      8 (16 mel frames), d_model 16, 2 heads, 1 + 1 layers, ffn 48,
      max_target_positions 16, vocab 50

Every weight is redrawn from a seeded numpy generator (norms 1 + 0.1 N,
biases 0.2 N, embeddings N(0, 1), matrices N(0, 1.5^2 / fan_in)); the
encoder's sinusoid table (embed_positions) keeps HF's values.

encoder_hf.npz:
  sinusoids.1500x32, sinusoids.8x16   HF's stored encoder tables
  tiny.state.<name>                   the tiny-whisper encoder weights under
                                      course names ("model.encoder." removed,
                                      no embed_positions)
  tiny.mel, tiny.hidden               HF features of course clip en-0
                                      (course/fixtures/L14.1/hf_features.npz,
                                      [1, 80, 3000]) and the HF encoder's
                                      last_hidden_state [1, 1500, 32]
  small.state.<name>, small.mel, small.hidden
                                      the same for tiny-whisper-128 on two
                                      random [128, 16] inputs

    uv run --python 3.12 --script course/oracle/L14.2/whisper_tiny_hf.py

Run from the repo root (after course/oracle/L14.1/frontend_hf.py) and paste
the printed rows into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import WhisperConfig, WhisperForConditionalGeneration

DIR = Path("course/fixtures/L14.2")
GEN = "course/oracle/L14.2/whisper_tiny_hf.py"
VERS = f"torch=={torch.__version__},transformers=={transformers.__version__}"
rng = np.random.default_rng(20261013)
ROWS: list[str] = []


def row(p: Path, data: bytes) -> None:
    ROWS.append(
        f"{p}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\t{GEN}\t{VERS}\t-\tApache-2.0"
    )


def randomize(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "encoder.embed_positions" in name:
                continue
            if "layer_norm" in name and name.endswith("weight"):
                v = 1.0 + 0.1 * rng.standard_normal(p.shape)
            elif name.endswith("bias"):
                v = 0.2 * rng.standard_normal(p.shape)
            elif "embed" in name:
                v = rng.standard_normal(p.shape)
            else:
                fan_in = int(np.prod(p.shape[1:]))
                v = rng.standard_normal(p.shape) * (1.5 / np.sqrt(fan_in))
            p.copy_(torch.tensor(v, dtype=torch.float32))


def save(model, name: str) -> Path:
    d = DIR / name
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    (d / "generation_config.json").unlink(missing_ok=True)
    for f in sorted(d.iterdir()):
        row(f, f.read_bytes())
    return d


def build(cfg: WhisperConfig, name: str):
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg).eval()
    randomize(model)
    d = save(model, name)
    return WhisperForConditionalGeneration.from_pretrained(
        d, dtype=torch.float32
    ).eval()


def enc_state(model) -> dict[str, np.ndarray]:
    out = {}
    for k, v in model.state_dict().items():
        if k.startswith("model.encoder.") and "embed_positions" not in k:
            out[k[len("model.encoder.") :]] = v.detach().numpy().astype(np.float32)
    return out


def main() -> None:
    specials = dict(
        pad_token_id=99, bos_token_id=98, eos_token_id=98, decoder_start_token_id=97
    )
    tiny = build(
        WhisperConfig(
            vocab_size=100,
            num_mel_bins=80,
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            max_source_positions=1500,
            max_target_positions=64,
            suppress_tokens=None,
            begin_suppress_tokens=None,
            **specials,
        ),
        "tiny-whisper",
    )
    small = build(
        WhisperConfig(
            vocab_size=50,
            num_mel_bins=128,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=48,
            decoder_ffn_dim=48,
            max_source_positions=8,
            max_target_positions=16,
            suppress_tokens=None,
            begin_suppress_tokens=None,
            pad_token_id=49,
            bos_token_id=48,
            eos_token_id=48,
            decoder_start_token_id=47,
        ),
        "tiny-whisper-128",
    )
    feats = np.load("course/fixtures/L14.1/hf_features.npz")
    arrs: dict[str, np.ndarray] = {}
    arrs["sinusoids.1500x32"] = (
        tiny.model.encoder.embed_positions.weight.detach().numpy()
    )
    arrs["sinusoids.8x16"] = small.model.encoder.embed_positions.weight.detach().numpy()
    with torch.no_grad():
        mel = feats["en-0.f80"][None].astype(np.float32)
        arrs["tiny.mel"] = mel
        arrs["tiny.hidden"] = tiny.model.encoder(
            torch.tensor(mel)
        ).last_hidden_state.numpy()
        smel = rng.standard_normal((2, 128, 16)).astype(np.float32)
        arrs["small.mel"] = smel
        arrs["small.hidden"] = small.model.encoder(
            torch.tensor(smel)
        ).last_hidden_state.numpy()
    for k, v in enc_state(tiny).items():
        arrs[f"tiny.state.{k}"] = v
    for k, v in enc_state(small).items():
        arrs[f"small.state.{k}"] = v
    arrs["__meta__"] = np.frombuffer(
        json.dumps({"generator": GEN, "versions": VERS}).encode(), dtype=np.uint8
    )
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    p = DIR / "encoder_hf.npz"
    p.write_bytes(buf.getvalue())
    row(p, buf.getvalue())
    for r in ROWS:
        print(r)


if __name__ == "__main__":
    main()
