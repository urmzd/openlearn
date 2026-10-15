# Training & Post-Training: Code

Standalone, runnable files for [Training & Post-Training](../). Each one implements a formula from the topic README and asserts that it holds.

| File | README § | What it shows | Dependencies |
|------|----------|---------------|--------------|
| [`memory_calc.py`](memory_calc.py) | §2, §3, §5, §6 | Parameter count from architecture, training memory (full FT vs LoRA vs QLoRA), `6ND` vs `4ND` FLOPs, GPU-hours, the 8B-on-15T pretraining estimate | stdlib only |
| [`lora_from_scratch.py`](lora_from_scratch.py) | §5 | LoRA `W0 + (alpha/r) B A` on a frozen 2-layer MLP: `B = 0` init equals the base model, manual backprop with a finite-difference check, Adam on adapters only, merge equals unmerged | numpy |
| [`dpo_loss.py`](dpo_loss.py) | §4 | DPO loss and gradient through per-token log-softmax, checked against central finite differences; loss is exactly `log 2` at policy = reference | numpy |
| [`train_step_torch.py`](train_step_torch.py) | §7 | Next-token step in PyTorch: shift-by-one targets, `scaled_dot_product_attention(is_causal=True)`, AdamW, grad clipping, BF16 autocast on CUDA, SFT prompt masking with `ignore_index=-100` | torch |
| [`train_step_jax.py`](train_step_jax.py) | §3, §7 | The same step in JAX + Optax: pure `loss_fn`, `jax.value_and_grad`, `jax.jit`, and the batch placed on a `Mesh` with `NamedSharding(mesh, P("data"))` | jax, optax |

## Run

```bash
python memory_calc.py
python memory_calc.py --preset 70b --tokens 50e6 --rank 32 --gpu h200

uv run --with numpy python lora_from_scratch.py
uv run --with numpy python dpo_loss.py

uv run --with torch python train_step_torch.py
uv run --with jax --with optax python train_step_jax.py
# simulate 4 devices on CPU to see the batch sharded across a mesh
XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run --with jax --with optax python train_step_jax.py
```

The torch and JAX files exit with an install hint if their framework is missing. Last verified October 2026 with torch 2.14.1 and jax 0.10.2 on CPU.

## Notes

- `memory_calc.py` is a scoping tool. It ignores allocator fragmentation, NCCL buffers, and the CUDA context, so leave 10-20% headroom and measure with `torch.cuda.max_memory_allocated()` before quoting a customer.
- `train_step_jax.py` builds its mesh with `AxisType.Auto` (GSPMD propagation). Recent JAX defaults to `Explicit` axes, where shardings are part of each array's type and ops such as embedding gathers need explicit output shardings.
