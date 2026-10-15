# Model Loading: Code

Runnable companions to [Model Loading: Hub to GPU](../). The first two are stdlib-only Python (3.11+) and need no install.

| File | What it does | Run |
|------|--------------|-----|
| [`safetensors_inspect.py`](safetensors_inspect.py) | Parses the safetensors header by hand (8-byte LE length, JSON, byte buffer). Lists tensors, dtypes, shapes, byte sizes, element totals; validates sizes and holes; handles `model.safetensors.index.json` (shard map vs shard headers); decodes one tensor (F32/F16/BF16/ints) into a Python list; inspects **remote** files with HTTP Range requests (honors `HF_TOKEN`) | `python safetensors_inspect.py --selftest` |
| [`config_explain.py`](config_explain.py) | Reads a `config.json` and prints total and active params, attention variant (MHA/GQA/MQA/MLA), MoE layout, KV bytes per token, max context, RoPE scaling, EOS ids, and a cold-load time estimate per storage path | `python config_explain.py samples/*.json` |
| [`load_with_transformers.py`](load_with_transformers.py) | Reference load path: `snapshot_download(allow_patterns=...)`, `AutoTokenizer.apply_chat_template`, `AutoModelForCausalLM.from_pretrained(dtype="auto", device_map="auto")`, `generate`. Defaults to `HuggingFaceTB/SmolLM2-135M-Instruct` | `uv run --no-project --with transformers --with torch --with accelerate python load_with_transformers.py` |
| [`samples/`](samples/) | Verbatim public `config.json` files: Llama 3.1 8B Instruct (dense, GQA, llama3 RoPE), Qwen3-30B-A3B (MoE, 128 experts top-8), DeepSeek-V3 (MLA, 256 experts + shared, FP8, YaRN) | |

## Examples

```bash
# Inspect a remote checkpoint without downloading the weights
python safetensors_inspect.py --show 5 --tensor model.norm.weight --limit 6 \
  --url https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors

# Check a customer's sharded checkpoint: index vs shard headers
python safetensors_inspect.py /mnt/models/customer-ft/model.safetensors.index.json --show 0

# KV cache sizing with an FP8 cache
python config_explain.py samples/llama-3.1-8b-instruct.config.json --kv-dtype fp8
```

The Llama sample reproduces the official checkpoint exactly: `config_explain.py` computes 8,030,261,248 parameters, and the Hub index reports `total_size = 16,060,522,496` bytes in BF16 (2 bytes each).

Sample sources: [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (gated; values match the public mirror), [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3). Fetched October 2026.
