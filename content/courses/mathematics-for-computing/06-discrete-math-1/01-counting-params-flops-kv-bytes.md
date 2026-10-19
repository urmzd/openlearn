# Counting: parameters, FLOPs, KV bytes, and memory plans

In *Build Your Own LLM System*, [the modern-block chapter](../../build-an-llm/08-modern-block/) swaps the 2017 transformer for a Llama-family model and, at the end, loads SmolLM2-135M from Hugging Face ([The Llama-Family Model: Config, Weight Loading, and Download](../../build-an-llm/08-modern-block/09-llama-family-model.md)). Before loading 270 MB of weights, three questions need numbers, not guesses.

Does the model have the same parameters as theirs? The `{tinyllm} info` command must report exactly what Hugging Face's `num_parameters()` reports, and a missing norm or a double-counted tied head is off by thousands or by 28 million. How many bytes of KV cache does one generated token cost? The Rust serving engine admits requests by exactly this number ([Continuous batching scheduler with priority and preemption](../../build-an-llm/11-serving/03-continuous-batching.md)). Will a training run fit in a laptop's memory? The [TinyStories capstone](../../build-an-llm/capstones/01-tinystories.md) picks its model size from that answer. This section answers all three by counting.

No code from earlier sections is needed. You will use powers and logs for orders of magnitude from [Exponents, logarithms, and units of information](../01-precalculus/01-exponents-logs-and-units-of-information.md), and the product and sum rules, which [this chapter's problem set](03-problem-set.md) practices on paper. The solution is [`accounting.py`](../../../../course/solutions/python/tinyllm/accounting.py), with `ModelConfig`, `param_count`, `flops_per_token`, `kv_bytes_per_token`, and `memory_plan`; the typed interface is [`course/contracts/py/tinyllm/accounting.pyi`](../../../../course/contracts/py/tinyllm/accounting.pyi).

> **Learning objectives**
> - Count a decoder's parameters per component from its config, exactly matching Hugging Face for dense, GQA, MoE, and MLA models.
> - Compute KV-cache bytes per token and explain why they depend on kv heads, not query heads.
> - Compute forward and training FLOPs per token, including the attention term that "$6N$" drops.
> - Build a training memory plan and explain why AdamW costs 16 bytes per parameter in both fp32 and bf16.
> - Explain why every count must be an exact integer.

## Notation

The counts are all products of a handful of config integers. This table names them:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size | `int` |
| $d$ | model width, `d_model` | `int` |
| $L$ | number of blocks, `n_layers` | `int` |
| $H$, $H_{kv}$ | query heads, key/value heads | `int` |
| $d_h$ | width of one head, `d_head` | `int` |
| $f$ | MLP hidden width, `d_ff` | `int` |
| $E$, $k$, $S$, $f_e$ | experts, experts per token (`top_k`), shared experts, expert width | `int` |
| $r$, $d_r$, $d_v$, $r_q$ | MLA latent width, rope width, value width, query latent width | `int` |
| $T$ | context length, `seq_len` | `int` |
| $P$ | total parameters | `int` |
| $N$ | matmul parameters one token touches | `int` |
| $b$ | bytes per number (`dtype_bytes`: 2 for bf16, 4 for fp32) | `int` |

## The two counting rules

Everything here uses two rules. The **product rule**: a matrix with $m$ rows and $n$ columns holds $m \cdot n$ numbers. The **sum rule**: parts that do not overlap add. A `Linear(in, out)` layer stores a weight of $\text{out} \cdot \text{in}$ numbers, plus $\text{out}$ more with a bias. The rest of the section is these two rules applied carefully to each piece of a model.

## Embeddings and the head

The token embedding is a $V \times d$ table. The lm_head maps the final hidden state back to $V$ logits, another $V \times d$ matrix, unless the model **ties** them. Then the head reuses the embedding and stores nothing. SmolLM2 ties, saving $49152 \cdot 576 = 28{,}311{,}552$ numbers, a fifth of the model. Keep one distinction in mind from the start: tying saves parameters but not work. The head is still a matrix product per token.

## One block

A Llama block is RMSNorm, attention, RMSNorm, MLP, built in *Build Your Own LLM System* from [Pre-LN and RMSNorm](../../build-an-llm/08-modern-block/01-pre-ln-rmsnorm.md) through [MQA/GQA attention](../../build-an-llm/08-modern-block/05-gqa-attention.md). Each part has a closed-form count.

**Attention.** $q$ is $H d_h \times d$, $k$ and $v$ are $H_{kv} d_h \times d$ each, and the output projection is $d \times H d_h$, for a total of $2 d H d_h + 2 d H_{kv} d_h$. Qwen-style biases add $H d_h + 2 H_{kv} d_h$. MHA is the case $H_{kv} = H$; GQA shares each kv head among $H / H_{kv}$ query heads.

**MLA.** Multi-head latent attention (DeepSeek-V2/V3, [built here](../../build-an-llm/08-modern-block/06-multi-head-latent-attention.md)) compresses keys and values into a latent of width $r$. It has a down projection $(r + d_r) \times d$ (the extra $d_r$ is one shared rope key), a norm of $r$, an up projection $H (d_h + d_v) \times r$, the queries $H (d_h + d_r) \times d$ (or through a latent of width $r_q$ with its own norm), and the output $d \times H d_v$.

**Gated MLP.** In a [gated MLP](../../build-an-llm/08-modern-block/02-gated-mlps.md), gate and up are $f \times d$ and down is $d \times f$: $3 d f$.

**Mixture of experts.** A router $E \times d$, then $E$ experts and $S$ shared experts of $3 d f_e$ each. A token is routed to $k$ experts, so it touches only $k + S$ of them: its **active** parameters are the total minus $(E - k) \cdot 3 d f_e$ per MoE layer. DeepSeek-V3 keeps its first 3 layers dense.

**Norms.** Two RMSNorm gains of $d$ per block (plus $r$ and $r_q$ inside MLA), and one final gain of $d$.

## KV bytes

Parameters tell you what a model stores. The next count tells you what generation stores. Generating token $t+1$ needs the keys and values of tokens $1..t$ in every layer. Recomputing them is quadratic work, so the engine caches them. One token adds a key and a value of $d_h$ numbers for each kv head in each layer:

$$\text{kv bytes per token} = L \cdot 2 \cdot H_{kv} \cdot d_h \cdot b .$$

Query heads do not appear, and that is the whole argument for GQA: SmolLM2's 9 query heads read 3 kv heads, so its cache is a third of the MHA cache. MLA caches $r + d_r$ numbers per layer instead. DeepSeek-V3 stores $61 \cdot 576 \cdot 2 = 70{,}272$ bytes per token where an MHA cache of its 128 heads would take about 6 MB.

## FLOPs per token

A multiply-add is 2 floating-point operations. In $y = W x$ each entry of $W$ is used in exactly one multiply-add per token, so a matrix product costs 2 FLOPs per parameter. Let $N$ be the matmul parameters one token touches: the attention projections, the active MLP, and the $V d$ head, tied or not. The embedding lookup is a copy and the norm gains are elementwise, so they are left out.

Attention itself adds work that has no parameters. The token's query is dotted with $T$ keys ($T \cdot d_{qk}$ multiply-adds per head), and the $T$ weights average $T$ values ($T \cdot d_v$). Following PaLM, every token attends to all $T$ positions:

$$\text{forward} = 2N + 2 L H T (d_{qk} + d_v), \qquad \text{training step} = 3 \cdot \text{forward}.$$

The factor 3 comes from the backward pass, which computes two products per forward product: the gradient with respect to the input and with respect to the weight, as [Matrix differentials, the trace trick, and closed-form VJPs](../09-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) shows. The familiar "$6N$" is the training cost with the attention term dropped; for a small model at a long context that term is not small.

## Training memory

The last count is the one that decides whether training fits. With $P$ parameters at $b$ bytes, there are the weights ($bP$), their gradients ($bP$), the optimizer states in fp32 ($4P$ per state: SGD momentum keeps 1, AdamW keeps 2), and, when training in 16-bit, an fp32 **master copy** of the weights ($4P$) so small updates are not rounded away. AdamW in bf16 is $2 + 2 + 4 + 8 = 16$ bytes per parameter, and in fp32 it is $4 + 4 + 0 + 8 = 16$ too.

Then come the **activations**: values saved during the forward pass for the backward pass. Per token per layer, the counting contract here includes the norm inputs and outputs ($2d$ twice), $q$, $k$, $v$, one softmax row of $T$ probabilities per head, the attention output, and the four MLP intermediates ($4f$); plus fp32 logits of $V$ per token. The softmax rows make activations grow with $T^2$ per sequence, which is why [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](../../build-an-llm/12-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md) recomputes them.

## Working it by hand

Take a tiny model: $V = 10$, $d = 4$, $L = 2$, $H = 2$, $H_{kv} = 1$, $d_h = 2$, $f = 6$, untied. Every count follows in eight steps.

1. Embedding: $10 \cdot 4 = 40$. Head (untied): 40.
2. Attention per layer: $q$ is $4 \cdot 4 = 16$, $k$ is $2 \cdot 4 = 8$, $v$ is 8, output $4 \cdot 4 = 16$: 48. Two layers: 96.
3. MLP per layer $3 \cdot 4 \cdot 6 = 72$, two layers 144.
4. Norms: $2 \cdot 4$ per layer, two layers, plus the final 4: 20.
5. Total $40 + 96 + 144 + 20 + 40 = 340$.
6. FLOPs at $T = 3$: $N = 96 + 144 + 40 = 280$, so $2N = 560$; attention $2 \cdot 2 \cdot 2 \cdot 3 \cdot (2 + 2) = 96$. Forward 656, training $3 \cdot 656 = 1968$.
7. KV bytes in fp16: $2$ layers $\cdot\ 2$ (key and value) $\cdot\ 1$ kv head $\cdot\ 2$ wide $\cdot\ 2$ bytes $= 16$.
8. fp32 AdamW, batch 1, $T = 3$: weights 1360, grads 1360, master 0, optimizer $2 \cdot 4 \cdot 340 = 2720$. Saved values per token per layer: $8 + 4 + 2 + 2 + 2 \cdot 3 + 4 = 26$ (attention) plus $8 + 4 \cdot 6 = 32$ (MLP) $= 58$; per token $2 \cdot 58 \cdot 4 + 4 \cdot 10 = 504$ bytes; three tokens 1512. Total 6952.

These eight steps are the first three test cases below.

## Designing the interface

One config dataclass and four functions:

```python
@dataclass
class ModelConfig:
    vocab: int; d_model: int; n_layers: int; n_heads: int; n_kv_heads: int; d_head: int; d_ff: int
    tie_embeddings: bool; attn: Literal["mha", "gqa", "mla"] = "gqa"; kv_lora_rank: int = 0; qk_rope_dim: int = 0
    n_experts: int = 0; top_k: int = 0; n_shared: int = 0
    q_lora_rank: int = 0; v_head_dim: int = 0; d_ff_expert: int = 0; n_dense_layers: int = 0; qkv_bias: bool = False
def param_count(cfg) -> dict[str, int]: ...            # embed, attn, mlp, norm, lm_head, total, active
def flops_per_token(cfg, seq_len: int, training: bool) -> int: ...
def kv_bytes_per_token(cfg, dtype_bytes: int) -> int: ...
def memory_plan(cfg, batch, seq, dtype_bytes, optimizer) -> dict[str, int]: ...   # weights, grads, master, optimizer, activations, total
```

A correct solution must handle the following cases, each tied to a later use:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example, parameters | unit | steps 1 to 5 of the worked example | pins every component |
| Hand example, FLOPs and KV bytes | unit | steps 6 and 7 | the FLOP and cache formulas |
| Hand example, memory plan | unit | step 8 | the memory formula, term by term |
| Hugging Face counts | golden | nine architectures against `num_parameters()` in transformers 5.19.0, per component | `{tinyllm} info` in the Llama-family model |
| Python ints | boundary | every count is an `int`; DeepSeek-V3's 671,026,404,352 | no float rounding in large counts |
| Tying | property | untying adds $Vd$ params, no FLOPs | the head is a product either way |
| KV bytes and kv heads | property | SmolLM2's 23,040 bytes; MHA, GQA, MQA ratios | admission in continuous batching |
| MLA cache | unit | DeepSeek-V3's 70,272 bytes, independent of heads | the MLA layer's cache |
| MoE active parameters | property | Mixtral and DeepSeek-V3 active counts | MoE compute budgets in the capstone |
| SmolLM2 FLOPs | unit | 410,517,504 FLOPs per token at 2048 | throughput estimates for the [matmul kernel](../../build-an-llm/10-kernels/03-tiled-batch-invariant-matmul.md) |
| Attention term | property | the per-position step for GQA and MLA | long-context costs |
| Routed experts only | property | more experts, same FLOPs | MoE ablations |
| 16 bytes per parameter | property | AdamW 16 bytes in fp32 and bf16, SGD 12 | fitting the capstone in memory |
| Activation scaling | property | batch scaling and the softmax rows | recomputation in training at scale |
| Validation | boundary | inconsistent configs raise | wrong counts never pass silently |

## Building it step by step

The design follows from the counting rules in six steps.

1. Every count is a sum of matrix and vector sizes. So the solution counts per component (`embed`, `attn`, `mlp`, `norm`, `lm_head`), each as a product of config integers, and the total is their sum. Per-component counts are also what make a mismatch with Hugging Face diagnosable.
2. Only three things change between dense, GQA, MoE, and MLA models: the attention projections, the MLP of a layer, and a few extra norms. So one helper counts one layer's attention for either attention family, the MLP count distinguishes dense layers from MoE layers, and the MLA norms are added to the per-layer norm count.
3. FLOPs are $2N$ for the matmul parameters a token touches, plus the parameter-free attention term, times 3 for training. $N$ is therefore built from the parameter counts: attention, the *active* MLP, and the $Vd$ head whether tied or not. The embedding and the norms are left out.
4. The KV cache depends on kv heads and head width only, or on the latent width for MLA. It is a one-line product, with no reference to query heads.
5. Training memory is four parameter-sized terms (weights, gradients, master copy, optimizer states) plus activations. The master copy exists only when the training dtype is narrower than fp32, and optimizer states are always fp32.
6. Counts must be exact integers, and a wrong config must not produce a plausible number. So every input is validated as an integer in range, and every formula uses only `int` arithmetic.

Validation comes first in every public function. It rejects floats, booleans (which are `int` in Python), and configs whose shapes are inconsistent:

```python
def _check(cfg: ModelConfig) -> None:
    for name in ("vocab", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_head", "d_ff"):
        v = getattr(cfg, name)
        if isinstance(v, bool) or int(v) != v or v < 1:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    # ... the optional fields must be non-negative integers
    if cfg.attn == "mha":
        if cfg.n_kv_heads != cfg.n_heads:          # MHA is the case H_kv = H
            raise ValueError(...)
    elif cfg.attn == "gqa":
        if cfg.n_heads % cfg.n_kv_heads:            # each kv head serves H / H_kv query heads
            raise ValueError(...)
    elif cfg.attn == "mla":
        if cfg.kv_lora_rank < 1:                    # MLA without a latent is meaningless
            raise ValueError("mla needs kv_lora_rank >= 1")
    # ... top_k in [1, n_experts] for MoE; n_dense_layers <= n_layers
```

Next is one layer's attention, for both families. The standard branch is $2dHd_h + 2dH_{kv}d_h$ written as $q$, $k$ and $v$, and the output projection; the MLA branch follows the shapes given earlier:

```python
def _attn_params(cfg: ModelConfig) -> int:
    """Attention parameters of one layer, its norm weights excluded."""
    d, H, dh = cfg.d_model, cfg.n_heads, cfg.d_head
    if cfg.attn == "mla":
        dr, r, rq = cfg.qk_rope_dim, cfg.kv_lora_rank, cfg.q_lora_rank
        dv = cfg.v_head_dim or dh
        # queries directly from d, or through a latent of width r_q
        q = d * H * (dh + dr) if rq == 0 else d * rq + rq * H * (dh + dr)
        # down to the latent plus one shared rope key, then up to every head's k and v
        kv = d * (r + dr) + r * H * (dh + dv)
        return q + kv + H * dv * d                  # + the output projection
    q, kv = H * dh, cfg.n_kv_heads * dh            # k and v are sized by H_kv, not H
    bias = (q + 2 * kv) if cfg.qkv_bias else 0     # Qwen-style q, k, v biases
    return d * q + 2 * d * kv + q * d + bias
```

`param_count` assembles the components. Two details carry most of the golden matches: the final norm (`+ d`) and the tied head (0):

```python
def param_count(cfg: ModelConfig) -> dict[str, int]:
    _check(cfg)
    d, V, L = cfg.d_model, cfg.vocab, cfg.n_layers
    fe = cfg.d_ff_expert or cfg.d_ff
    n_moe = _moe_layers(cfg)                        # L - n_dense_layers, or 0 without experts
    dense_mlp = 3 * d * cfg.d_ff                    # gate, up, down
    moe_mlp = cfg.n_experts * d + (cfg.n_experts + cfg.n_shared) * 3 * d * fe   # router + experts
    mlp = (L - n_moe) * dense_mlp + n_moe * moe_mlp
    norm_per_layer = 2 * d                          # the two RMSNorm gains of a block
    if cfg.attn == "mla":
        norm_per_layer += cfg.kv_lora_rank + cfg.q_lora_rank   # MLA's latent norms
    out = {
        "embed": V * d,
        "attn": L * _attn_params(cfg),
        "mlp": mlp,
        "norm": L * norm_per_layer + d,             # + the final norm
        "lm_head": 0 if cfg.tie_embeddings else V * d,   # a tied head stores nothing
    }
    out["total"] = sum(out.values())
    # a token touches top_k routed experts (plus the shared ones): drop the rest
    out["active"] = out["total"] - n_moe * (cfg.n_experts - cfg.top_k) * 3 * d * fe
    return out
```

On the hand example this gives `embed` 40, `attn` $2 \cdot 48 = 96$, `mlp` 144, `norm` $2 \cdot 8 + 4 = 20$, `lm_head` 40, total 340.

`flops_per_token` rebuilds $N$ from those counts. The head is always included (tying saves storage, not work), the embedding and norms are not, and only routed experts count:

```python
def flops_per_token(cfg: ModelConfig, seq_len: int, training: bool) -> int:
    _check(cfg)
    # ... ValueError unless seq_len is a positive integer
    p = param_count(cfg)
    n_moe = _moe_layers(cfg)
    fe = cfg.d_ff_expert or cfg.d_ff
    active_mlp = p["mlp"] - n_moe * (cfg.n_experts - cfg.top_k) * 3 * cfg.d_model * fe
    n = p["attn"] + active_mlp + cfg.vocab * cfg.d_model   # the head, tied or not
    if cfg.attn == "mla":
        d_qk, d_v = cfg.d_head + cfg.qk_rope_dim, cfg.v_head_dim or cfg.d_head
    else:
        d_qk = d_v = cfg.d_head
    # 2 FLOPs per touched parameter, plus Q.K^T and the weighted sum of V
    # over all T positions in every head of every layer (the PaLM convention)
    fwd = 2 * n + 2 * cfg.n_layers * cfg.n_heads * int(seq_len) * (d_qk + d_v)
    return 3 * fwd if training else fwd             # backward = two products per forward one
```

On the hand example, $n = 96 + 144 + 40 = 280$, the attention term is $2 \cdot 2 \cdot 2 \cdot 3 \cdot 4 = 96$, the forward is 656, and training is 1968.

The KV cache is a single product, with the factor 2 for keys and values and no query heads anywhere:

```python
def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int) -> int:
    _check(cfg)
    # ... ValueError unless dtype_bytes is a positive integer
    if cfg.attn == "mla":
        per_layer = cfg.kv_lora_rank + cfg.qk_rope_dim    # one latent + one rope key, for all heads
    else:
        per_layer = 2 * cfg.n_kv_heads * cfg.d_head       # a key and a value per kv head
    return cfg.n_layers * per_layer * int(dtype_bytes)
```

The activation count per token per layer is the list from the training-memory discussion written out. The `n_heads * seq` term is the softmax row, the one term that grows with the context:

```python
def _saved_per_token(cfg: ModelConfig, seq: int, moe: bool) -> int:
    """A_l: values one layer keeps per token for its backward."""
    d = cfg.d_model
    q, k, v = _widths(cfg)                          # per-token q, k, v widths
    d_v = (cfg.v_head_dim or cfg.d_head) if cfg.attn == "mla" else cfg.d_head
    # norm in and out (2d), q, k, v, one softmax row per head, the attention output
    attn = 2 * d + q + k + v + cfg.n_heads * seq + cfg.n_heads * d_v
    if cfg.attn == "mla":
        attn += cfg.kv_lora_rank + cfg.qk_rope_dim + cfg.q_lora_rank   # the latents
    if moe:   # only the experts this token visits, plus its router logits
        f_tok = (cfg.top_k + cfg.n_shared) * (cfg.d_ff_expert or cfg.d_ff) + cfg.n_experts
    else:
        f_tok = cfg.d_ff
    return attn + 2 * d + 4 * f_tok                 # the MLP's norm in/out and 4 intermediates
```

`memory_plan` puts the terms together. The master copy and the fp32 optimizer states are what make AdamW cost 16 bytes per parameter in both fp32 and bf16:

```python
def memory_plan(cfg, batch, seq, dtype_bytes, optimizer) -> dict[str, int]:
    _check(cfg)
    # ... ValueError for a bad batch, seq, dtype_bytes (1, 2, 4, or 8), or optimizer
    states = {"sgd": 1, "adamw": 2}.get(optimizer)  # momentum; or first and second moments
    P = param_count(cfg)["total"]
    n_moe = _moe_layers(cfg)
    n_dense = cfg.n_layers - n_moe
    per_token = (
        n_dense * _saved_per_token(cfg, seq, False) + n_moe * _saved_per_token(cfg, seq, True)
    ) * dtype_bytes + 4 * cfg.vocab                 # + fp32 logits
    out = {
        "weights": P * dtype_bytes,
        "grads": P * dtype_bytes,
        "master": 4 * P if dtype_bytes < 4 else 0,  # an fp32 copy only when training narrower
        "optimizer": 4 * P * states,                # states are fp32 whatever the dtype
        "activations": batch * seq * per_token,
    }
    out["total"] = sum(out.values())
    return out
```

On the hand example in fp32: weights and gradients $4 \cdot 340 = 1360$ each, no master copy, optimizer $2 \cdot 4 \cdot 340 = 2720$, activations $3 \cdot (2 \cdot 58 \cdot 4 + 40) = 1512$, total 6952. The full solution is in [`accounting.py`](../../../../course/solutions/python/tinyllm/accounting.py).

### Why this design

Every function is $O(1)$ arithmetic on config integers; the point is exactness, not speed.

Closed forms beat instantiating the model. Counting a live module, as `num_parameters()` does, needs the weights in memory, which is the very question being answered before loading 270 MB or choosing a model that may not fit. Closed forms also explain *where* the parameters are, per component.

`int` arithmetic is used throughout because a double holds integers exactly only up to $2^{53} \approx 9 \times 10^{15}$. DeepSeek-V3's 671,026,404,352 parameters fit, but the products built from such counts do not: training FLOPs over $10^{12}$ tokens are around $10^{25}$, and their last digits would be rounded. A float count also invites formatting such as `6.71e11` that hides digits. Python ints are exact at any size, and a golden comparison with Hugging Face is then an exact equality.

Training is $3 \times$ forward with an explicit attention term, not "$6N$", because "$6N$" drops the attention term, which for SmolLM2-135M at 2048 tokens is about a third of the forward pass (141,557,760 of 410,517,504 FLOPs). The explicit form is what makes long-context and MFU estimates honest.

Validation matters because an MHA config with $H_{kv} \ne H$, or a GQA config with $H$ not divisible by $H_{kv}$, describes no real model. Returning a number for it would let a typo produce a plausible but wrong cache budget.

One tradeoff is accepted. The activation count is one specific contract (what a straightforward implementation saves), not what every framework saves; FlashAttention, for example, does not keep the softmax rows. The model is consistent and testable, and the training-at-scale chapter changes the terms explicitly when it recomputes.

## What can go wrong

Every pitfall here produces a plausible number, which is what makes them dangerous. The table pairs each with its symptom and the line of the solution that prevents it:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| counting a tied head twice | 28,311,552 too many for SmolLM2 | `lm_head` is 0 when `tie_embeddings` is set |
| sizing $k$ and $v$ by $H$ (GQA counted as MHA) | attention too large by $2 d (H - H_{kv}) d_h$ per layer | sizes $k$ and $v$ by `n_kv_heads * d_head` |
| caching keys only (no factor 2) | half the KV bytes; the engine over-admits and runs out of cache | `2 * n_kv_heads * d_head` per layer |
| training as two forwards | FLOPs, and every MFU estimate, a third low | returns `3 * fwd` for training |
| counting every expert as active | Mixtral "uses" 46.7B per token | `active` subtracts the $E - k$ unrouted experts |
| counts as floats | the last digits of a 671B count rounded away | only `int` arithmetic, and `_check` rejects non-integers |
| a tied head left out of the FLOPs | tying appears to save compute | adds `vocab * d_model` to $N$ whether tied or not |
| the attention term halved or dropped | long contexts look cheap | `2 * L * H * T * (d_qk + d_v)` in full |
| a master copy in fp32 training | 20 bytes per parameter instead of 16 | `master` only when `dtype_bytes < 4` |
| the MLA cache counted per head | DeepSeek-V3 looks 85 times more expensive to serve | caches `kv_lora_rank + qk_rope_dim` per layer, no head factor |
| forgetting the final norm | off by $d$ against Hugging Face | `norm` ends with `+ d` |
| optimizer states at the training dtype | bf16 AdamW 12 bytes, not 16 | `optimizer` is `4 * P * states` |
| forgetting the softmax rows | activation memory linear in context | `n_heads * seq` in `_saved_per_token` |
| forgetting MLA's query-latent norm | off by $r_q$ per layer | adds `kv_lora_rank + q_lora_rank` to the per-layer norms |
| FLOPs over every expert | MoE as expensive as dense | $N$ uses `active_mlp` |

> **Warning:** The missing factor 2 in the KV cache is the costliest of these in production. A serving engine that budgets half the real cache admits too many requests and runs out of memory under load.

## Where this leads

This section draws on powers of ten and logs for reading counts like $6.7 \times 10^9$ ([Exponents, logarithms, and units of information](../01-precalculus/01-exponents-logs-and-units-of-information.md)) and on the product and sum rules practiced in [this chapter's problem set](03-problem-set.md). Its functions are used throughout *Build Your Own LLM System*:

- In [Grouped-Query Attention with a Cache Hook, Windows, and Sinks](../../build-an-llm/08-modern-block/05-gqa-attention.md), the bytes the attention layer hands its cache must equal `kv_bytes_per_token`.
- In [Multi-head latent attention](../../build-an-llm/08-modern-block/06-multi-head-latent-attention.md), the MLA latent cache must cost exactly `kv_bytes_per_token` with `attn="mla"`.
- In [The Llama-Family Model: Config, Weight Loading, and Download](../../build-an-llm/08-modern-block/09-llama-family-model.md), `{tinyllm} info` reports `param_count`.
- In [Continuous batching scheduler with priority and preemption](../../build-an-llm/11-serving/03-continuous-batching.md), the Rust engine admits a request when its KV bytes fit.
- In [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](../../build-an-llm/12-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md), mixed precision, accumulation, and recomputation trade the terms of `memory_plan`.
- In [Capstone: TinyStories, owned end to end](../../build-an-llm/capstones/01-tinystories.md), the model size is chosen from `flops_per_token` and `memory_plan`.

### Further reading

- `PreTrainedModel.num_parameters()` in `transformers/modeling_utils.py`: counts the live module, with `exclude_embeddings`.
- MFU accounting in training frameworks measures throughput over the theoretical peak, per hardware; see PaLM appendix B and nanoGPT's `estimate_mfu`.
- vLLM's cache sizing (`CacheEngine.get_cache_block_size`) profiles free memory, then divides by block bytes.
- DeepSpeed and Megatron memory estimators add sharding (ZeRO stages 1 to 3), tensor and pipeline parallelism, and selective recomputation.
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020), section 2.1.
- Chowdhery et al., "PaLM" (2022), appendix B.
- Rajbhandari et al., "ZeRO" (2020), section 3.1.
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022), section 4.

## Key terms

- **Tied embeddings**: a model whose lm_head reuses the token embedding matrix, storing no extra parameters but still doing the $V \times d$ product.
- **Active parameters**: the parameters a single token touches; in an MoE model, the total minus the experts the token is not routed to.
- **KV cache**: the stored keys and values of earlier tokens, $L \cdot 2 \cdot H_{kv} \cdot d_h \cdot b$ bytes per token.
- **FLOPs per token**: $2N$ for the matmul parameters touched plus the parameter-free attention term; three times that for a training step.
- **Master copy**: an fp32 copy of 16-bit weights kept so small updates are not rounded away.
- **Activations**: values saved during the forward pass for use in the backward pass.

## Summary

- A decoder's parameters are a sum of products of a few config integers; counted per component, they match Hugging Face exactly for dense, GQA, MoE, and MLA models.
- The KV cache grows with the number of **kv** heads, not query heads: GQA and MQA shrink it by $H / H_{kv}$, and MLA replaces it with one latent per layer.
- A forward pass costs 2 FLOPs per parameter a token touches plus an attention term linear in the context; a training step costs three forwards.
- AdamW training costs 16 bytes per parameter before activations, in fp32 or bf16 alike.
- Counts are exact Python integers: 671,026,404,352 is not $6.71 \times 10^{11}$.

## Review questions

1. Why does `flops_per_token` include `cfg.vocab * cfg.d_model` even when `tie_embeddings` is true, while `param_count` reports `lm_head = 0`?

   <details><summary>Answer</summary>

   Tying means the head *reuses* the embedding matrix, so no new numbers are stored. But computing the logits is still a $V \times d$ matrix product per token, 2 FLOPs per entry. Parameters measure storage; FLOPs measure work.

   </details>

2. SmolLM2 has 9 query heads and 3 kv heads. Why does `kv_bytes_per_token` never read `n_heads`?

   <details><summary>Answer</summary>

   The cache stores keys and values, and there is one key and one value of width $d_h$ per kv head per layer. Query heads are computed fresh for the new token and never cached. With 3 kv heads the cache is a third of what 9 MHA heads would need, $30 \cdot 2 \cdot 3 \cdot 64 \cdot 2 = 23{,}040$ bytes per token in bf16.

   </details>

3. Why is `master` zero for `dtype_bytes = 4`, and why is `optimizer` always `4 * P * states`?

   <details><summary>Answer</summary>

   A master copy exists to keep small updates from being rounded away in a 16-bit weight; in fp32 the weights already are the fp32 copy. Optimizer states are kept in fp32 for the same reason in every precision. So AdamW is $4 + 4 + 0 + 8 = 16$ bytes per parameter in fp32 and $2 + 2 + 4 + 8 = 16$ in bf16.

   </details>

4. Where does the $T^2$ growth of activation memory come from in `_saved_per_token`?

   <details><summary>Answer</summary>

   From `cfg.n_heads * seq`: each token keeps one softmax row of `seq` probabilities per head. `memory_plan` multiplies the per-token count by `batch * seq`, so that term grows as $H T^2$ per sequence while everything else grows as $T$.

   </details>

5. For an MoE layer, why does `active` subtract $(E - k) \cdot 3 d f_e$ but keep the router and the shared experts?

   <details><summary>Answer</summary>

   Every token runs the router ($E \times d$) and the $S$ shared experts, and is sent to exactly $k$ routed experts. The $E - k$ routed experts it is not sent to do no work for it, so they are the only parameters a token does not touch.

   </details>
