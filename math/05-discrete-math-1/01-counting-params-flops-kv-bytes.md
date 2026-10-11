<!-- ol:module M05.1 -->
# Counting: params, FLOPs, KV bytes, memory plans

## Overview

| | |
|---|---|
| **Module** | `M05.1` · build · Python · Pass 5 · 2 to 3 h |
| **You build** | `python/tinyllm/accounting.py`: `ModelConfig`, `param_count`, `flops_per_token`, `kv_bytes_per_token`, `memory_plan` |
| **Contract** | [`course/contracts/py/tinyllm/accounting.pyi`](../../course/contracts/py/tinyllm/accounting.pyi) |
| **Tests** | `course/tests/M05.1/test_accounting.py` (what they check: section 4), golden counts from Hugging Face transformers 5.19.0 in `course/fixtures/M05.1/hf_param_counts.json` (`course/oracle/M05.1/hf_param_counts.py`) |
| **Needs** | no code from earlier modules · reading: `M00.1` (powers and logs for orders of magnitude), `S-M05` (the product and sum rules) |
| **Used by** | `L7.5` checks its cache hook against `kv_bytes_per_token` · `L7.6` latent cache bytes · `L7.9` `{tinyllm} info` · later: `L10.2` admission by KV bytes, `L11.1` and `C1` memory and compute budgets |
| **Milestone** | `MS-L7` (step 4: `info` reports the parameter count Hugging Face reports) |
| **Optional depth** | Kaplan et al., "Scaling Laws for Neural Language Models" (2020), section 2.1; Chowdhery et al., "PaLM" (2022), appendix B; Rajbhandari et al., "ZeRO" (2020), section 3.1; Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022), section 4 |

## Key Takeaways

- A decoder's parameters are a sum of products of a few config integers; counted per component, they match Hugging Face exactly for dense, GQA, MoE, and MLA models (`test_golden_hf_param_counts`).
- The KV cache grows with the number of **kv** heads, not query heads: GQA and MQA shrink it by $H / H_{kv}$, MLA replaces it with one latent per layer (`test_kv_bytes_scale_with_kv_heads_not_query_heads`, `test_mla_caches_the_latent`).
- A forward pass costs 2 FLOPs per parameter a token touches plus an attention term linear in the context; a training step costs three forwards (`test_smollm2_flops`, `test_flops_attention_term_is_linear_in_context`).
- AdamW training costs 16 bytes per parameter before activations, in fp32 or bf16 alike (`test_memory_plan_16_bytes_per_param`).
- Counts are exact Python integers: 671,026,404,352 is not $6.71 \times 10^{11}$ (`test_counts_are_python_ints`).

## How to work this chapter

```bash
ol start M05.1              # stubs accounting.py into your repo
ol tests M05.1              # read the test catalog first
ol check M05.1              # exit code is the verdict
ol diff  M05.1              # after passing: your code against the reference
```

---

## 1. Why now

Pass 5 swaps your 2017 transformer for a Llama-family model and, at its milestone, loads SmolLM2-135M from Hugging Face. Before you load 270 MB of weights, three questions need numbers, not guesses. Does your model have the same parameters as theirs (MS-L7 step 4 compares `{tinyllm} info` with Hugging Face's `num_parameters()`, and a missing norm or a double-counted tied head is off by thousands or by 28 million)? How many bytes of KV cache does one generated token cost (the Rust engine in Pass 7 admits requests by exactly this number, `L10.2`)? Will a training run fit in your laptop's memory (`C1` picks its model size from it)? This module answers all three by counting.

## 2. Principles

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

### 2.1 The two counting rules

Everything here uses two rules from `S-M05`. **Product rule**: a matrix with $m$ rows and $n$ columns holds $m \cdot n$ numbers. **Sum rule**: parts that do not overlap add. A `Linear(in, out)` layer stores a weight of $\text{out} \cdot \text{in}$ numbers, plus $\text{out}$ more with a bias.

### 2.2 Embeddings and the head

The token embedding is a $V \times d$ table. The lm_head maps the final hidden state back to $V$ logits, another $V \times d$ matrix, unless the model **ties** them: then the head reuses the embedding and stores nothing (SmolLM2 does, saving $49152 \cdot 576 = 28{,}311{,}552$ numbers, a fifth of the model). Tying saves parameters but not work: the head is still a matrix product per token.

### 2.3 One block

A Llama block is RMSNorm, attention, RMSNorm, MLP (`L7.1` to `L7.5`).

- **Attention** (`L7.5`): $q$ is $H d_h \times d$, $k$ and $v$ are $H_{kv} d_h \times d$ each, the output projection is $d \times H d_h$: $2 d H d_h + 2 d H_{kv} d_h$. Qwen-style biases add $H d_h + 2 H_{kv} d_h$. MHA is $H_{kv} = H$; GQA shares each kv head among $H / H_{kv}$ query heads.
- **MLA** (DeepSeek-V2/V3) compresses keys and values into a latent of width $r$: a down projection $(r + d_r) \times d$ (the extra $d_r$ is one shared rope key), a norm of $r$, an up projection $H (d_h + d_v) \times r$, the queries $H (d_h + d_r) \times d$ (or through a latent of width $r_q$ with its own norm), and the output $d \times H d_v$.
- **Gated MLP** (`L7.2`): gate and up are $f \times d$, down is $d \times f$: $3 d f$.
- **Mixture of experts**: a router $E \times d$, then $E$ experts and $S$ shared experts of $3 d f_e$ each. A token is routed to $k$ experts, so it touches only $k + S$ of them: its **active** parameters are the total minus $(E - k) \cdot 3 d f_e$ per MoE layer. DeepSeek-V3 keeps its first 3 layers dense.
- **Norms**: two RMSNorm gains of $d$ per block (plus $r$ and $r_q$ inside MLA), and one final gain of $d$.

### 2.4 KV bytes

Generating token $t+1$ needs the keys and values of tokens $1..t$ in every layer. Recomputing them is quadratic work, so the engine caches them. One token adds a key and a value of $d_h$ numbers for each kv head in each layer:

$$\text{kv bytes per token} = L \cdot 2 \cdot H_{kv} \cdot d_h \cdot b .$$

Query heads do not appear. That is the whole argument for GQA: SmolLM2's 9 query heads read 3 kv heads, so its cache is a third of the MHA cache. MLA caches $r + d_r$ numbers per layer instead: DeepSeek-V3 stores $61 \cdot 576 \cdot 2 = 70{,}272$ bytes per token where an MHA cache of its 128 heads would take about 6 MB.

### 2.5 FLOPs per token

A multiply-add is 2 floating-point operations. In $y = W x$ each entry of $W$ is used in exactly one multiply-add per token, so a matrix product costs 2 FLOPs per parameter. Let $N$ be the matmul parameters one token touches: the attention projections, the active MLP, and the $V d$ head (tied or not; the embedding lookup is a copy and the norm gains are elementwise, so they are left out). Attention itself adds work that has no parameters: the token's query is dotted with $T$ keys ($T \cdot d_{qk}$ multiply-adds per head) and the $T$ weights average $T$ values ($T \cdot d_v$). Following PaLM, every token attends to all $T$ positions:

$$\text{forward} = 2N + 2 L H T (d_{qk} + d_v), \qquad \text{training step} = 3 \cdot \text{forward},$$

because the backward pass computes two products per forward product (the gradient with respect to the input and with respect to the weight, `M08.3`). The familiar "$6N$" is the training cost with the attention term dropped; for a small model at a long context that term is not small.

### 2.6 Training memory

With $P$ parameters at $b$ bytes: the weights ($bP$), their gradients ($bP$), the optimizer states in fp32 ($4P$ per state: SGD momentum keeps 1, AdamW keeps 2), and, when training in 16-bit, an fp32 **master copy** of the weights ($4P$) so small updates are not rounded away. AdamW in bf16 is $2 + 2 + 4 + 8 = 16$ bytes per parameter, and in fp32 $4 + 4 + 0 + 8 = 16$ too. Then the **activations**: backward needs values saved during the forward. Per token per layer this contract counts the norm inputs and outputs ($2d$ twice), $q$, $k$, $v$, one softmax row of $T$ probabilities per head, the attention output, and the four MLP intermediates ($4f$); plus fp32 logits of $V$ per token. The softmax rows make activations grow with $T^2$ per sequence, which is why `L11.1` recomputes them.

## 3. Worked example by hand

$V = 10$, $d = 4$, $L = 2$, $H = 2$, $H_{kv} = 1$, $d_h = 2$, $f = 6$, untied.

1. Embedding: $10 \cdot 4 = 40$. Head (untied): 40.
2. Attention per layer: $q$ $4 \cdot 4 = 16$, $k$ $2 \cdot 4 = 8$, $v$ 8, output $4 \cdot 4 = 16$: 48. Two layers: 96.
3. MLP per layer $3 \cdot 4 \cdot 6 = 72$, two layers 144.
4. Norms: $2 \cdot 4$ per layer, two layers, plus the final 4: 20.
5. Total $40 + 96 + 144 + 20 + 40 = 340$.
6. FLOPs at $T = 3$: $N = 96 + 144 + 40 = 280$, so $2N = 560$; attention $2 \cdot 2 \cdot 2 \cdot 3 \cdot (2 + 2) = 96$. Forward 656, training $3 \cdot 656 = 1968$.
7. KV bytes in fp16: $2$ layers $\cdot\ 2$ (key and value) $\cdot\ 1$ kv head $\cdot\ 2$ wide $\cdot\ 2$ bytes $= 16$.
8. fp32 AdamW, batch 1, $T = 3$: weights 1360, grads 1360, master 0, optimizer $2 \cdot 4 \cdot 340 = 2720$. Saved values per token per layer: $8 + 4 + 2 + 2 + 2 \cdot 3 + 4 = 26$ (attention) plus $8 + 4 \cdot 6 = 32$ (MLP) $= 58$; per token $2 \cdot 58 \cdot 4 + 4 \cdot 10 = 504$ bytes; three tokens 1512. Total 6952.

These are `test_hand_example_params`, `test_hand_example_flops_and_kv_bytes`, and `test_hand_example_memory_plan`.

## 4. The interface

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_params` | unit | section 3, steps 1 to 5 | you and the test agree on every component |
| `test_hand_example_flops_and_kv_bytes` | unit | section 3, steps 6 and 7 | the FLOP and cache formulas |
| `test_hand_example_memory_plan` | unit | section 3, step 8 | the memory formula, term by term |
| `test_golden_hf_param_counts` | golden | nine architectures against `num_parameters()`, per component | `{tinyllm} info` in MS-L7 |
| `test_counts_are_python_ints` | boundary | every count is an `int`; DeepSeek-V3's 671,026,404,352 | no float rounding in large counts |
| `test_tying_saves_exactly_one_matrix_of_params_but_no_flops` | property | untying adds $Vd$ params, no FLOPs | the head is a product either way |
| `test_kv_bytes_scale_with_kv_heads_not_query_heads` | property | SmolLM2's 23,040 bytes; MHA, GQA, MQA ratios | `L10.2` admission |
| `test_mla_caches_the_latent` | unit | DeepSeek-V3's 70,272 bytes, independent of heads | `L7.6` |
| `test_moe_active_parameters` | property | Mixtral and DeepSeek-V3 active counts | MoE compute budgets in `C1` |
| `test_smollm2_flops` | unit | 410,517,504 FLOPs per token at 2048 | throughput estimates in `L9.1` |
| `test_flops_attention_term_is_linear_in_context` | property | the per-position step for GQA and MLA | long-context costs |
| `test_flops_count_only_routed_experts` | property | more experts, same FLOPs | MoE ablations |
| `test_memory_plan_16_bytes_per_param` | property | AdamW 16 bytes in fp32 and bf16, SGD 12 | fitting `C1` in memory |
| `test_activations_linear_in_batch_quadratic_in_context` | property | batch scaling and the softmax rows | recomputation in `L11.1` |
| `test_validation` | boundary | inconsistent configs raise | wrong counts never pass silently |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. counting a tied head twice | 28,311,552 too many for SmolLM2 | `test_golden_hf_param_counts`, `test_tying_saves_exactly_one_matrix_of_params_but_no_flops` (mutant `s01`) |
| 2. sizing $k$ and $v$ by $H$ (GQA counted as MHA) | attention too large by $2 d (H - H_{kv}) d_h$ per layer | `test_hand_example_params` (mutant `s02`) |
| 3. caching keys only (no factor 2) | half the KV bytes; the engine over-admits and runs out of cache | `test_hand_example_flops_and_kv_bytes` (mutant `s03`) |
| 4. training as two forwards | FLOPs, and every MFU estimate, a third low | `test_smollm2_flops` (mutant `s04`) |
| 5. counting every expert as active | Mixtral "uses" 46.7B per token | `test_moe_active_parameters` (mutant `s05`) |
| counts as floats | the last digits of a 671B count rounded away | `test_counts_are_python_ints` (mutant `s14`) |
| a tied head left out of the FLOPs | tying appears to save compute | `test_tying_saves_exactly_one_matrix_of_params_but_no_flops` (mutant `s06`) |
| the attention term halved or dropped | long contexts look cheap | `test_flops_attention_term_is_linear_in_context` (mutant `s07`) |
| a master copy in fp32 training | 20 bytes per parameter instead of 16 | `test_memory_plan_16_bytes_per_param` (mutant `s08`) |
| the MLA cache counted per head | DeepSeek-V3 looks 85 times more expensive to serve | `test_mla_caches_the_latent` (mutant `s09`) |
| forgetting the final norm | off by $d$ against Hugging Face | `test_hand_example_params` (mutant `s10`) |
| optimizer states at the training dtype | bf16 AdamW 12 bytes, not 16 | `test_memory_plan_16_bytes_per_param` (mutant `s11`) |
| forgetting the softmax rows | activation memory linear in context | `test_activations_linear_in_batch_quadratic_in_context` (mutant `s12`) |
| forgetting MLA's query-latent norm | off by $r_q$ per layer | `test_golden_hf_param_counts` (mutant `s13`) |
| FLOPs over every expert | MoE as expensive as dense | `test_flops_count_only_routed_experts` (mutant `s15`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | powers of ten and logs for reading counts like $6.7 \times 10^9$ |
| Back | `S-M05` | the product and sum rules, as pen-and-paper problems |
| Forward | `L7.5` | its course tests check the bytes the attention layer hands its cache against `kv_bytes_per_token` |
| Forward | `L7.6` | the MLA latent cache must cost exactly `kv_bytes_per_token(attn="mla")` |
| Forward | `L7.9` | `{tinyllm} info` reports `param_count` |
| Forward | `L10.2` | the Rust engine admits a request when its KV bytes fit |
| Forward | `L11.1` | mixed precision, accumulation, and recomputation trade the terms of `memory_plan` |
| Forward | `C1` | the capstone model size is chosen from `flops_per_token` and `memory_plan` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `param_count` | `PreTrainedModel.num_parameters()` | counts the live module, with `exclude_embeddings` | `transformers/modeling_utils.py` |
| `flops_per_token` | MFU accounting in training frameworks | measured throughput over the theoretical peak, per hardware | PaLM appendix B; nanoGPT `estimate_mfu` |
| `kv_bytes_per_token` | vLLM's cache sizing | profiles free memory, then divides by block bytes | vLLM `CacheEngine.get_cache_block_size` |
| `memory_plan` | DeepSpeed and Megatron memory estimators | sharding (ZeRO stages 1 to 3), tensor and pipeline parallelism, selective recomputation | ZeRO section 3; Korthikanti et al. section 4 |
