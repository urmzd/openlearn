#!/usr/bin/env python3
"""Reference load path: Hub snapshot -> tokenizer + chat template -> model -> generate.

Targets transformers v5 (`dtype=`, not the deprecated `torch_dtype=`) and
huggingface_hub v1+/v2. Not run in CI. Try it without installing globally:

    uv run --with transformers --with torch --with accelerate \
        python load_with_transformers.py

    # any causal LM; pin a commit in production
    python load_with_transformers.py --repo Qwen/Qwen3-0.6B --revision main
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
    )
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"missing dependency ({exc.name}). Run with:\n"
        "  uv run --with transformers --with torch --with accelerate python load_with_transformers.py"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument(
        "--revision",
        default="main",
        help="branch, tag, or commit sha (pin a sha in prod)",
    )
    p.add_argument("--prompt", default="In one sentence, what is a safetensors file?")
    a = p.parse_args()

    # 1. Download only what the loader needs: no .bin pickles, no GGUF, no ONNX.
    t0 = time.perf_counter()
    local = snapshot_download(
        a.repo,
        revision=a.revision,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.jinja"],
    )
    print(f"snapshot: {local}  ({time.perf_counter() - t0:.1f}s)")

    # 2. Config first: this is what decides the model class and dtype.
    cfg = AutoConfig.from_pretrained(local)
    print(
        f"model_type={cfg.model_type} arch={cfg.architectures} tie={cfg.tie_word_embeddings}"
    )

    # 3. Tokenizer + chat template. Look at the exact ids the model will see.
    tok = AutoTokenizer.from_pretrained(local)
    messages = [{"role": "user", "content": a.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("rendered prompt:\n" + text)
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    print(
        f"first ids: {inputs['input_ids'][0, :12].tolist()}  eos={tok.eos_token!r}/{tok.eos_token_id}"
    )

    # 4. Model. dtype="auto" keeps the checkpoint dtype (BF16 for most LLMs);
    #    device_map="auto" lets accelerate place shards on GPU, then CPU, then disk.
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(local, dtype="auto", device_map="auto")
    print(
        f"loaded in {time.perf_counter() - t0:.1f}s dtype={model.dtype} map={getattr(model, 'hf_device_map', None)}"
    )

    # 5. generation_config.json carries eos ids and default sampling params.
    gen = GenerationConfig.from_pretrained(local)
    print(f"generation_config eos={gen.eos_token_id}")

    inputs = inputs.to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    new = out[0, inputs["input_ids"].shape[1] :]
    print("completion:", tok.decode(new, skip_special_tokens=True))


if __name__ == "__main__":
    main()
