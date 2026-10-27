# Code for Training and Post-Training

These are standalone, runnable files for the [Training and Post-Training](../README.md) chapter. Each one implements a formula from the chapter and asserts that it holds, so you can check the chapter's numbers rather than take them on trust. The table lists each file, the part of the chapter it supports, what it demonstrates, and what it needs to run.

| File | Chapter section | What it shows | Dependencies |
|------|-----------------|---------------|--------------|
| [`learning_checks.py`](learning_checks.py) | Learning signals, PPO/GAE, GRPO, distillation, and inference numerics | Worked loss/advantage calculations, clipping sign cases, terminal/truncation masks, importance correction, finite-group baseline bias, KL direction, KV accounting, and floating-point batch effects | stdlib only |
| [`memory_calc.py`](memory_calc.py) | The mathematics of pretraining; Memory and parallelism; Parameter-efficient fine-tuning; Scoping a fine-tune | Parameter count from architecture, training memory (full FT vs LoRA vs QLoRA), `6ND` vs `4ND` FLOPs, GPU-hours, the 8B-on-15T pretraining estimate | stdlib only |
| [`lora_from_scratch.py`](lora_from_scratch.py) | Parameter-efficient fine-tuning | LoRA `W0 + (alpha/r) B A` on a frozen 2-layer MLP: `B = 0` init equals the base model, manual backprop with a finite-difference check, Adam on adapters only, merge equals unmerged | numpy |
| [`dpo_loss.py`](dpo_loss.py) | Post-training | DPO loss and gradient through per-token log-softmax, checked against central finite differences; loss is exactly `log 2` at policy = reference | numpy |
| [`train_step_torch.py`](train_step_torch.py) | The same step in three frameworks | Next-token step in PyTorch: shift-by-one targets, `scaled_dot_product_attention(is_causal=True)`, AdamW, grad clipping, BF16 autocast on CUDA, SFT prompt masking with `ignore_index=-100` | torch |
| [`train_step_jax.py`](train_step_jax.py) | Memory and parallelism; The same step in three frameworks | The same step in JAX + Optax: pure `loss_fn`, `jax.value_and_grad`, `jax.jit`, and the batch placed on a `Mesh` with `NamedSharding(mesh, P("data"))` | jax, optax |

## Running the files

The calculator needs only the standard library; the others run cleanly under `uv` with their dependencies supplied on the command line:

```bash
python learning_checks.py
python memory_calc.py
python memory_calc.py --preset 70b --tokens 50e6 --rank 32 --gpu h200

uv run --with numpy python lora_from_scratch.py
uv run --with numpy python dpo_loss.py

uv run --with torch python train_step_torch.py
uv run --with jax --with optax python train_step_jax.py
# simulate 4 devices on CPU to see the batch sharded across a mesh
XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run --with jax --with optax python train_step_jax.py
```

The torch and JAX files exit with an install hint if their framework is missing. They were last verified in October 2026 with torch 2.14.1 and jax 0.10.2 on CPU.

## Notes

`memory_calc.py` is a scoping tool, not a profiler. It ignores allocator fragmentation, NCCL buffers, and the CUDA context, so leave 10-20% headroom and measure with `torch.cuda.max_memory_allocated()` before quoting a number to a customer.

`train_step_jax.py` builds its mesh with `AxisType.Auto`, which gives classic GSPMD propagation. Recent JAX defaults to `Explicit` axes, where shardings are part of each array's type and operations such as embedding gathers need explicit output shardings.
