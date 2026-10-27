# Pretraining compute and scaling

This part of [Choosing Training and Adaptation Methods](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Estimate training compute and distinguish a cost model from a measurement.

## The mathematics of pretraining

The pretraining objective is the cross entropy between the data and the model, in nats per token:

$$L(\theta) = -\frac{1}{T}\sum_t \log p_\theta(x_{t+1} \mid x_{\le t})$$

Its exponential, $e^{L}$, is the perplexity. In code the whole objective is a shift by one: `inputs = x[:, :-1]` and `targets = x[:, 1:]`, as in [`train_step_torch.py`](code/train_step_torch.py).

### Why training costs 6ND

The number to carry everywhere is that training costs about $6ND$ FLOPs for a model with $N$ parameters trained on $D$ tokens. The forward pass costs $2N$ FLOPs per token, one multiply-add per parameter. The backward pass costs $4N$, because it computes two sets of gradients, with respect to the activations and with respect to the weights, at $2N$ each. Attention adds about $12 \cdot L \cdot d \cdot s$ FLOPs per token for $L$ layers of width $d$ at sequence length $s$, which stays small until the context gets long.

### Chinchilla, and why everyone overtrains anyway

Given a fixed compute budget $C = 6ND$, how should you split it between model size and data? [Kaplan et al.](https://arxiv.org/abs/2001.08361) gave the first answer; [Chinchilla](https://arxiv.org/abs/2203.15556) gave the one that stuck. In the Chinchilla study, compute-optimal parameter and token budgets scaled approximately together, motivating a rule of thumb near **$D \approx 20N$** within that experimental setting. It is an empirical fit, not a universal token requirement for every architecture, data mixture, or serving budget. The fitted loss is

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}, \qquad \alpha \approx 0.34,\ \beta \approx 0.28.$$

Yet modern models are trained far past that point on purpose. Chinchilla minimizes *training* cost, but a model's lifetime cost is $6ND + 2N \cdot D_{\text{inference}}$, and for a popular model $D_{\text{inference}}$ dwarfs $D$ ([*Beyond Chinchilla-Optimal*](https://arxiv.org/abs/2401.00448)). A smaller model trained well past $20N$ reaches the same quality at a much lower serving cost. Llama 3 8B saw about 15T tokens, roughly 93 times its Chinchilla-optimal budget.

### Utilization: MFU and HFU

Hardware never runs at its peak. **Model FLOPs utilization** measures how close you get:

$$\text{MFU} = \frac{6N \cdot \text{tokens/sec}}{n_{\text{gpus}} \cdot \text{peak FLOP/s}}$$

MFU depends on shapes, precision, parallelism, topology, attention accounting, and the chosen hardware peak. Treat any assumed percentage as a scoping input and report the measured result separately. **Hardware FLOPs utilization** (HFU) also counts FLOPs that are recomputed, so under activation checkpointing HFU is greater than MFU.

### Working it by hand: an 8B model on 15T tokens

These formulas are enough to cost a real pretraining run. Take the Llama 3 8B shape on H100s, as computed by [`memory_calc.py`](code/memory_calc.py):

```text
N = 8.03e9 (Llama 3 8B shape)       D = 15e12
C = 6 N D = 6 × 8.03e9 × 15e12      = 7.23e23 FLOPs
H100 dense BF16 peak                = 989 TFLOP/s
at 40% MFU                          = 3.96e14 FLOP/s per GPU
GPU-seconds = 7.23e23 / 3.96e14     = 1.83e9 s  →  ~507,000 GPU-hours
on 1,024 GPUs                       ≈ 20.6 days
assuming $2-3 per H100-hour               ≈ $1.0-1.5M of pure compute
```

This is an analytical scenario using a fixed MFU and an illustrative GPU-hour price, not a current quote or a reconstructed training bill. Actual wall time also depends on attention work, achieved utilization, data loading, evaluations, checkpointing, failures, and the scope of the reported run. A difference from a published GPU-hour figure does not identify which of those causes dominated.

## Summary

Estimate training compute and distinguish a cost model from a measurement. State the assumptions before applying the result to a different workload.

## Review questions

1. Does doubling parameter count always double total run time?

   <details><summary>Answer</summary>

   Only with fixed tokens, throughput efficiency, communication, and memory assumptions. A change that forces a different layout changes those assumptions.

   </details>

[Chapter contents](README.md) · [Next: Memory, precision, and parallelism](03-memory-precision-and-parallelism.md)
