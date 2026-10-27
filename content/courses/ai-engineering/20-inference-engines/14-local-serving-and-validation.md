# Local training, serving, and validation lab

Use one local artifact and a shared evaluation set to understand what changes between training, in-process inference, and a serving engine. Start on the hardware you have. CPU arithmetic checks and tiny training runs teach the learning loop; a supported GPU runtime is needed to measure GPU serving behavior.

> **Learning objectives**
> - Run a small training/validation loop before attempting scale.
> - Serve a compatible local artifact through vLLM, SGLang, or llama.cpp.
> - Apply the same API, numerical, task, and load checks across engines.

## The local execution choices

| Route | Artifact and execution | What to learn or validate |
|---|---|---|
| Reference model in-process | Framework-native checkpoint, direct Python calls | Losses, masks, gradients, fixed-prefix logits, and custom training |
| vLLM server or offline API | Supported model/config/tokenizer and weight layout | Continuous batching, cache/scheduling controls, GPU serving and rollout generation |
| SGLang server or offline engine | Supported model/config/tokenizer and weight layout | Radix/prefix reuse, scheduling, structured generation, rollout integration |
| llama.cpp `llama-server` | Supported GGUF model; CPU and optional supported accelerators | Local quantized execution and a lightweight HTTP server |

Hardware support is release-specific. The main SGLang installation examples currently target Linux/NVIDIA; both GPU engines document additional hardware paths with separate constraints. On a laptop without a supported device, use the in-process CPU and llama.cpp routes, then repeat the GPU experiments on a compatible machine. Compare those environments separately.

## First: learn and test the update

From the repository root, run the arithmetic checks with Python 3.11+:

```bash
python3 content/courses/ai-engineering/19-training-methods/code/learning_checks.py
```

With `uv` and the framework dependency available, run the existing tiny training example in an isolated dependency environment:

```bash
uv run --no-project --with torch python \
  content/courses/ai-engineering/19-training-methods/code/train_step_torch.py
```

It trains a character-level decoder and checks loss reduction and SFT target masking. This is a sanity check on a deliberately repetitive training corpus, not a held-out capability evaluation. Inspect the target shift and trainable parameter deltas. For a full local model-building path, use [TinyStories](../capstones/01-tinystories.md), then the [post-trained capstone](../capstones/02-post-trained.md).

For a smaller CPU-only adaptation exercise, the NumPy LoRA example checks manual gradients against finite differences, learns adapters on a synthetic task, and verifies merged/unmerged forward equivalence. The DPO example checks its analytic gradient and separates tiny synthetic preference pairs:

```bash
uv run --no-project --with numpy python \
  content/courses/ai-engineering/19-training-methods/code/lora_from_scratch.py
uv run --no-project --with numpy python \
  content/courses/ai-engineering/19-training-methods/code/dpo_loss.py
```

These exercises establish their small examples and invariants. They do not establish general language-model quality or GPU training throughput.

The toy character model is not automatically a vLLM/SGLang-compatible architecture. Export or select a model those engines explicitly support. For a real fine-tune, preserve the base revision, model configuration, tokenizer, chat template, adapter identity or merged weights, and generation defaults. Run held-out task evaluation in the training framework before exporting, then repeat it after conversion and quantization.

## Prepare a local artifact and an isolated runtime

Select a small supported instruct model whose weights and expected KV/workspace fit the device. Resolve it to an immutable revision and place all required config, weights, and tokenizer files in a local directory. Check its license and chat template as part of artifact selection. A Hugging Face model directory and a GGUF file are different representations; conversion can introduce numerical differences.

Install each runtime in a separate environment or use its official container for the target hardware. Follow [vLLM installation](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/index.html), [SGLang installation](https://docs.sglang.io/docs/get-started/install), or [llama.cpp build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md). Pin the package versions or container digest used, save help output, and record the driver/device. A generic package install does not select the correct accelerator build on every machine.

## Start one engine at a time

The examples assume `./artifacts/local-model` is a complete compatible model directory and `./artifacts/local-model.gguf` is a compatible GGUF artifact. They bind loopback and use distinct ports. Run each command in its own foreground terminal and wait for the engine's readiness output.

```bash
# vLLM: option spelling referenced to the archived 0.11.0 CLI.
vllm serve ./artifacts/local-model \
  --served-model-name local-model --host 127.0.0.1 --port 8000 \
  --max-model-len 2048 --max-num-seqs 4

# SGLang: option spelling checked against the server reference on 2026-10-26.
python -m sglang.launch_server --model-path ./artifacts/local-model \
  --served-model-name local-model --host 127.0.0.1 --port 30000 \
  --context-length 2048 --max-running-requests 4

# llama.cpp: use a built/installed llama-server and a supported GGUF file.
llama-server -m ./artifacts/local-model.gguf \
  --alias local-model --host 127.0.0.1 --port 8080 --ctx-size 2048
```

These values bound a first experiment; they do not promise fit or optimal performance. Confirm supported context, precision, and template, and inspect the resolved configuration. Each runtime's default graph, cache, and sampler behavior can differ. Do not compare default launches and attribute every difference to the engine implementation.

For direct in-process execution, use the reference framework's model forward/generation call with evaluation mode and disabled gradient recording. [Transformers' generation guide](https://huggingface.co/docs/transformers/main/en/llm_tutorial) describes its API. vLLM's `LLM` API and SGLang's `Engine` API also support offline work, avoiding HTTP while retaining engine scheduling. Those serving APIs do not expose a general backward pass for training.

## Use the same client contract

Set `LOCAL_BASE_URL` to the running engine's port. Discover the served model ID rather than assuming the filename is its API name:

```bash
LOCAL_BASE_URL=http://127.0.0.1:8000
curl --fail-with-body "$LOCAL_BASE_URL/v1/models"

curl --fail-with-body "$LOCAL_BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"What is 2 + 2? Give only the number."}],"temperature":0,"max_tokens":32,"stream":false}'

curl --fail-with-body -N "$LOCAL_BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"What is 2 + 2? Give only the number."}],"temperature":0,"max_tokens":32,"stream":true}'
```

Inspect HTTP status, model identity, output content, finish reason, usage where supported, and streaming completion. A reasoning model may need a larger budget or template-specific control before producing visible answer text. Repeat with Unicode, multi-turn input, length caps, stop sequences, malformed input, and client cancellation. OpenAI-compatible APIs share conventions but not every extension, default, error shape, or seed semantic.

## Validate in layers

First compare exact rendered token IDs across the reference model and engine. Teacher-force identical prefixes and compare logits or action log probabilities if the APIs expose them; text-only APIs cannot prove this layer. Then compare greedy generation and a frozen task suite. Run the [batch-invariance matrix](13-batch-invariance-and-reproducibility.md) with concurrent neighbors; sequential repetition alone does not test it.

Next run a short open-loop load test with shared controls. The companion's synthetic prompt length is an estimate, so use the engine's native tokenizer-aware benchmark to cross-check exact lengths:

```bash
python3 content/courses/ai-engineering/20-inference-engines/04-serving-and-load/code/loadgen.py \
  --base-url "$LOCAL_BASE_URL" --model local-model --endpoint chat \
  --rate 1 --num-requests 20 --input-len 128 --output-len 32 \
  --slo-ttft-ms 1000 --slo-tpot-ms 100 --json-out /tmp/local-engine-run.jsonl
```

Use a distinct result file and recorded configuration per engine/run. Increase offered rate gradually and report latency tails, failures, generated lengths, and goodput. A successful API request proves connectivity and response shape; the numerical and task suites establish different properties.

## Train, serve, compare, then tune

Start from a reference checkpoint and held-out suite. Make one small SFT or adapter update using [the training methods](../19-training-methods/README.md), evaluate it, export it, and repeat the service checks. Add PPO/GRPO only when a justified reward signal and a correct rollout/learner contract exist. Use inference engines to generate rollouts, while a trainer owns gradient updates and checkpoints.

Finally sweep one control from [the generic flag guide](10-runtime-controls-and-flags.md). Record the expected mechanism and acceptance budget before each run. Keep a known-correct baseline so a serving optimization, export error, and training regression can be distinguished.

## Key terms

- **Local artifact**: a complete revision of the files needed to construct and tokenize the model.
- **Smoke test**: a small connectivity or basic-behavior check with deliberately limited coverage.
- **Reference execution**: a defined implementation used to compare numerical and task behavior.

## Summary

Learn the update on a small model, serve a supported artifact, and use the same validation layers across runtimes. Engine choice and tuning follow measured constraints.

## Review questions

1. Why is serving the toy character model with vLLM not just a path change?

   <details><summary>Answer</summary>The runtime needs a supported architecture implementation, parameter layout, tokenizer, and model configuration. An arbitrary Python module is not that contract.</details>
2. What can a text-only API comparison leave unverified?

   <details><summary>Answer</summary>Tokenization details, numerical logits, actual action probabilities, and cache/position invariants may remain inaccessible. Report those limits explicitly.</details>
