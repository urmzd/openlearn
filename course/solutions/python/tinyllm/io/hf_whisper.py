"""Strict loading of local Hugging Face Whisper safetensors (L14.3)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tinyllm.io.safetensors import load_safetensors
from tinyllm.speech.encoder import WhisperConfig, whisper_sinusoids


def whisper_state_from_hf(
    tensors: Mapping[str, NDArray], cfg: WhisperConfig
) -> tuple[dict[str, NDArray], NDArray | None]:
    state = {}
    positions = None
    for name, value in tensors.items():
        value = np.asarray(value, dtype=np.float32)
        if name == "proj_out.weight":
            embedding = tensors.get("model.decoder.embed_tokens.weight")
            if embedding is None or not np.array_equal(
                value, np.asarray(embedding, dtype=np.float32)
            ):
                raise ValueError(
                    "Whisper output projection is not tied to token embeddings"
                )
        elif name == "model.encoder.embed_positions.weight":
            expected = whisper_sinusoids(cfg.n_audio_ctx, cfg.d)
            # Published Whisper checkpoints can store the fixed table after
            # float16 rounding. At unit magnitude its rounding error is up to
            # 2**-12, plus float32 phase differences across math libraries.
            quantized = np.array_equal(
                value, value.astype(np.float16).astype(np.float32)
            )
            tolerance = 5e-4 if quantized else 1e-4
            if (
                value.shape != expected.shape
                or not np.all(np.isfinite(value))
                or np.max(np.abs(value - expected)) > tolerance
            ):
                raise ValueError("stored encoder positions are not Whisper sinusoids")
            positions = value.copy()
        elif name.startswith("model."):
            state[name[len("model.") :]] = value.copy()
        else:
            raise ValueError(f"unexpected Whisper tensor name: {name}")
    return state, positions


def read_whisper(
    dir: str,
) -> tuple[WhisperConfig, dict[str, NDArray], NDArray | None]:
    root = Path(dir).resolve()
    cfg = WhisperConfig.from_hf(str(root))
    index = root / "model.safetensors.index.json"
    weight_map = None
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Whisper shard index has no tensor map")
        names = sorted(set(weight_map.values()))
    else:
        names = ["model.safetensors"]
    tensors = {}
    for name in names:
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError("Whisper shard path escapes the checkpoint directory")
        shard, _ = load_safetensors(str(path))
        for key, value in shard.items():
            if key in tensors or (
                weight_map is not None and weight_map.get(key) != name
            ):
                raise ValueError(f"duplicate or misindexed Whisper tensor: {key}")
            tensors[key] = value
    if weight_map is not None and set(tensors) != set(weight_map):
        raise ValueError("Whisper shard index names missing tensors")
    state, positions = whisper_state_from_hf(tensors, cfg)
    return cfg, state, positions
