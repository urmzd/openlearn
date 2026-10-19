# Serving and Load Code

These are the runnable companions to [Serving, Capacity & Load Testing: Math to Code](../README.md). One script turns the memory budget and the decode roofline into numbers for a specific model and GPU; the other measures a real or mock server under open-loop load and reports the latency distributions and goodput that make up an SLO. Both are Python 3.11+ with the standard library only, so they run anywhere without an install, and both can be verified offline:

| File | What it does | Verify offline |
|------|--------------|----------------|
| [`capacity.py`](capacity.py) | Weight memory, KV bytes per token, KV pool, max concurrent sequences, decode roofline bound, $ per 1M tokens | `python capacity.py --preset llama-3.1-8b` |
| [`loadgen.py`](loadgen.py) | Open-loop Poisson load generator for OpenAI-compatible streaming endpoints: TTFT, TPOT, ITL, E2E, tok/s, goodput, p50/p90/p99 | `python loadgen.py --mock --rate 8 --num-requests 100` |

## capacity.py

The capacity calculator works from built-in presets or from any Hugging Face `config.json`. These invocations reproduce the rows of the memory-budget table in the section and extend it to DeepSeek-V3 on H200:

```bash
python capacity.py --list-presets
python capacity.py --preset llama-3.1-8b                                   # BF16, 1x H100
python capacity.py --preset llama-3.1-8b --weight-bytes 1 --kv-bytes 1     # FP8 weights + FP8 KV
python capacity.py --preset llama-3.1-70b --tp 4                           # BF16 70B on 4x H100
python capacity.py --preset llama-3.1-70b --tp 2 --weight-bytes 1 --kv-bytes 1
python capacity.py --preset deepseek-v3 --gpu h200 --tp 8 --weight-bytes 1 --kv-bytes 1 --context 32768
python capacity.py --hf-config ./config.json --tp 2 --context 32768 --mbu 0.7 --gpu-hourly 3.00
```

Three flags deserve explanation. `--mbu` (model bandwidth utilization) scales the bandwidth bound to what engines actually reach; 1.0 is the physical ceiling. `--util` mirrors vLLM `--gpu-memory-utilization`. `--activation-gib` is the per-GPU reserve for activations, CUDA graphs, and NCCL buffers that the engine profiles at startup.

## loadgen.py

The load generator has an in-process mock engine, so you can see the shape of a latency-throughput curve before you have a GPU. Start with a single rate, then sweep rates to get the curve, then point it at a real server:

```bash
# offline self-test (in-process fake engine that slows down as its batch grows)
python loadgen.py --mock --rate 8 --num-requests 100 --slo-ttft-ms 200 --slo-tpot-ms 20

# the curve, not a single number
python loadgen.py --mock --sweep 2,8,32 --num-requests 120 --output-len 48

# a real server
python loadgen.py --base-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint chat --sweep 1,2,4,8,16 --num-requests 400 \
    --input-len 1024 --output-len 256 --len-dist lognormal --ignore-eos \
    --slo-ttft-ms 500 --slo-tpot-ms 40 --json-out run.jsonl
```

Two design choices keep results honest. Every prompt starts with a unique nonce so prefix caching cannot inflate results; pass `--shared-prefix-len 800` to measure the cache on purpose. `--burstiness < 1` draws gamma inter-arrivals (burstier than Poisson), matching `vllm bench serve --burstiness`.

`loadgen.py` is for understanding what a benchmark computes. For production numbers, cross-check with the engine-native tools, which tokenize exactly:

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --model meta-llama/Llama-3.1-8B-Instruct --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --num-prompts 1000 \
    --request-rate 8 --burstiness 1.0 --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 --goodput ttft:500 tpot:40 --save-result

python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
    --model meta-llama/Llama-3.1-8B-Instruct --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --random-range-ratio 0.5 \
    --num-prompts 1000 --request-rate 8 --max-concurrency 256
```

## Example engine launches

The launches below configure Llama-3.1-8B-Instruct on one 80 GB H100 for several tenants: a chat-tuned FP8 deployment, a latency-bound tenant with EAGLE-3 speculative decoding, a multi-LoRA pool, and the SGLang equivalents. Each flag maps to a term of the memory budget or step-time equation explained in the section.

```bash
# vLLM: chat-tuned, FP8 weights + FP8 KV, prefix caching, 16k context
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --port 8000

# vLLM: add EAGLE-3 speculative decoding for a low-concurrency, latency-bound tenant
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 16384 --max-num-seqs 32 \
    --speculative-config '{"method": "eagle3", "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3", "num_speculative_tokens": 3}'

# vLLM: multi-LoRA tenant pool
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora --max-loras 8 --max-lora-rank 16 \
    --lora-modules support=./adapters/support sql=./adapters/sql

# SGLang: same box, RadixAttention on by default, chunked prefill at 8k
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp 1 --context-length 16384 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 8192 \
    --max-running-requests 256 \
    --schedule-policy lpm \
    --kv-cache-dtype fp8_e4m3 \
    --port 30000

# SGLang: EAGLE-3
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

Flag names were checked against [docs.vllm.ai engine args](https://docs.vllm.ai/en/latest/configuration/engine_args.html) and [docs.sglang.io server arguments](https://docs.sglang.io/advanced_features/server_arguments.html) in October 2026. Draft-model repo names change often: confirm the current EAGLE-3 checkpoint for your target model on Hugging Face before shipping.
