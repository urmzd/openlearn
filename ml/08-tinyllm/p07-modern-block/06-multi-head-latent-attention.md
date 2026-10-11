<!-- ol:module L7.6 -->
# Multi-head latent attention (DeepSeek-V2/V3), weight absorption

## Overview

| | |
|---|---|
| **Module** | `L7.6` · build · Python · Pass 5 · 3 to 4 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/modern/mla.py`: `MLAttention`, `ConcatLatentCache`, `AbsorbedMLA`; and your own oracle tests in `python/tests/l7-6-mla/` |
| **Contract** | [`course/contracts/py/tinyllm/modern/mla.pyi`](../../../course/contracts/py/tinyllm/modern/mla.pyi) |
| **Tests** | `course/tests/L7.6/test_mla.py` (what they check: section 4), golden values from transformers 5.19.0 `DeepseekV3Attention` in `course/fixtures/L7.6/mla_hf.npz` (`course/oracle/L7.6/mla_hf.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L7.3` `apply_rope`, `rope_cos_sin`, `RopeSpec` · `L7.1` `RMSNorm` · `L0.4` `Linear`, `Module` · `L0.2` the op library · `L0.1` `Tensor` · `M06.3` `PCG32` · `M05.1` `kv_bytes_per_token` (the tests compare with it) · reading: `M03.5` low rank, `L7.5` the cache hook, `M09.2` (or `--ref-deps`) |
| **Used by** | `L7.9` builds it for `tl_attention = "mla"` · later: `L8.2` `LatentCache`, C1's MLA-vs-GQA ablation at equal KV bytes |
| **Milestone** | `MS-L7` (your decoder loads and matches Hugging Face checkpoints) |
| **Optional depth** | DeepSeek-AI, "DeepSeek-V2" (2024), section 2.1; "DeepSeek-V3 Technical Report" (2024), section 2.1.1 |

## Key Takeaways

- MLA caches one latent vector $c$ of width $r$ plus one shared rope key of width $d_r$ per token, whatever the number of heads: $r + d_r$ numbers, M05.1's `kv_bytes_per_token` for `mla` (`test_cache_bytes_match_m05_1`).
- Every head's key and value are linear in $c$, so a decode step folds $W_{UK}$ into the query and $W_{UV}$ into `o_proj` and never rebuilds per-head keys (`test_absorbed_decode_equals_naive`).
- RoPE cannot be absorbed (it depends on position), so MLA splits each query and key into a position-free "nope" part and a small rotated part (`test_scores_see_relative_position_only`).
- The cached latent is the normalized one, and the layout of every projection is HF's, so DeepSeek checkpoints load by name (`test_mla_golden`).

## How to work this chapter

```bash
ol start L7.6              # stubs mla.py; prints your test path and rung (R5)
ol tests L7.6              # the course tests
# write your oracle tests in python/tests/l7-6-mla/, then:
ol check L7.6              # course tests and the mutation grade of your tests
ol diff  L7.6              # after passing: your code against the reference
```

---

## 1. Why now

Your GQA layer (`L7.5`) already shrank SmolLM2's cache by three: 3 kv heads instead of 9. The cache still grows by $2 H_{kv} d_h$ numbers per token per layer, and at serving time (`L10.2` admits requests by KV bytes) that is what limits how many conversations fit in memory. DeepSeek-V2 cut it much further with a different idea: cache a low-rank summary of the token and rebuild keys and values from it. Built naively that costs compute at every step; built with weight absorption it costs almost nothing. This module builds both forms and proves they agree, so C1 can compare MLA and GQA at equal KV bytes.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_t \in \mathbb{R}^d$ | the input of token $t$ | `float32[d]` |
| $H$ | query heads | `int` |
| $r$ | `kv_lora_rank`: width of the latent | `int` |
| $r_q$ | `q_lora_rank`: width of the query bottleneck (`None`: none) | `int` |
| $d_n$, $d_r$, $d_v$ | `qk_nope_dim`, `qk_rope_dim`, `v_dim` per head | `int` |
| $c_t \in \mathbb{R}^r$ | the latent of token $t$: what the cache holds | `float32[r]` |
| $k^R_t \in \mathbb{R}^{d_r}$ | the rotated rope key, shared by every head | `float32[d_r]` |
| $W_{UK}^h \in \mathbb{R}^{d_n \times r}$, $W_{UV}^h \in \mathbb{R}^{d_v \times r}$ | head $h$'s rows of `kv_b_proj` | matrices |
| $W_O^h \in \mathbb{R}^{d \times d_v}$ | head $h$'s columns of `o_proj` | matrix |
| $R_p$ | the RoPE rotation at position $p$ (`L7.3`) | $d_r \times d_r$ |
| $s$ | `softmax_scale` $= (d_n + d_r)^{-1/2}$ | `float` |

### 2.1 Low rank, again

`M03.5` showed that a matrix of rank $r$ factors as a product of an $m \times r$ and an $r \times n$ matrix, and that truncating the SVD keeps the best rank-$r$ approximation. MLA bets that the keys and values of all heads, stacked as one $H(d_n + d_v)$-wide vector per token, live near an $r$-dimensional subspace. It learns the factorization directly: a down projection to $c_t$ (`kv_a_proj_with_mqa`, the first $r$ outputs), an RMSNorm (`kv_a_layernorm`, `L7.1`), and an up projection `kv_b_proj` of shape $[H(d_n + d_v), r]$. SmolLM2-sized MHA caches $2 H d_h$ numbers per token; DeepSeek-V3 uses $r = 512$ and $d_r = 64$ for 128 heads.

### 2.2 The latent and the decoupled rope key

Steps of the naive form, for each token:

$$c_t = \mathrm{RMSNorm}(W_{DKV} x_t), \qquad k^{N,h}_t = W_{UK}^h c_t, \qquad v^h_t = W_{UV}^h c_t,$$
$$k^R_t = R_t\, W_{KR} x_t, \qquad q^h_t = [\,q^{N,h}_t \mid R_t\, q^{R,h}_t\,],$$
$$\mathrm{score}^h(t, j) = s\,\big(q^{N,h}_t \cdot k^{N,h}_j + (R_t q^{R,h}_t) \cdot k^R_j\big).$$

Each head's query is `[nope | rope]` with the nope part first, and `kv_b_proj` splits per head into `[k_nope | v]`, nope first. Positions enter only through $R$, so a shift of every position changes nothing (`L7.3`'s law). The rope key is one vector shared by all heads, like MQA, which keeps it to $d_r$ cached numbers.

### 2.3 What the cache holds

The cache receives $c_t$ after the norm and $k^R_t$ after the rotation, $r + d_r$ numbers per token per layer: `ConcatLatentCache.update(layer, c, k_rope)` appends a chunk and returns everything held. A query of a chunk of $T$ new tokens sits at key index $T_k - T + t$, exactly as in `L7.5`, so prefilling in chunks equals one full forward.

### 2.4 Weight absorption

Rewrite the nope score and the output with matrix products moved around:

$$q^{N,h}_t \cdot (W_{UK}^h c_j) = \big((W_{UK}^h)^\top q^{N,h}_t\big) \cdot c_j, \qquad W_O^h \sum_j p_j W_{UV}^h c_j = (W_O^h W_{UV}^h) \sum_j p_j c_j .$$

The query is mapped into latent space once ($q^{lat,h}_t = (W_{UK}^h)^\top q^{N,h}_t$, an $r$-vector), the scores are dot products with the cached $c_j$, the weighted latent $\sum_j p_j c_j$ is formed once per head, and $W_O^h W_{UV}^h$ (precomputed, $d \times r$) maps it out. No per-head key or value is ever built: decoding reads $r + d_r$ numbers per cached token. RoPE's part cannot be absorbed (it depends on $t - j$), which is the reason for the decoupled rope key.

### 2.5 Training the naive form

Absorption is an inference rewrite. Training uses the naive form built from the op library, so gradients reach every projection through the latent, its norm, and the rope key.

## 3. Worked example by hand

$d = 2$, one head, $r = 1$, $d_n = 1$, $d_r = 2$, $d_v = 1$, `inv_freq` $= (\pi/2)$, so position 1 rotates the rope pair by 90 degrees: $(a, b) \mapsto (-b, a)$. Weights: $q = [x_0 \mid x_0, x_1]$; $c_{raw} = x_0 + x_1$ and $k^R = (x_1, x_0)$; `kv_b_proj` gives $k^N = 2c$, $v = 3c$; `o_proj` outputs $(o, 0)$. The latent norm of a single number is $c / |c| = \pm 1$ (the $\epsilon = 10^{-6}$ changes the 7th digit).

- Token 0, $x = (1, 1)$, position 0: $c_{raw} = 2$, so $c = 1$; $k^R = (1, 1)$, not rotated; $v = 3$. It sees only itself: output $(3, 0)$.
- Token 1, $x = (0, -1)$, position 1: $c_{raw} = -1$, so $c = -1$, $v = -3$; $k^R = (-1, 0)$ rotated to $(0, -1)$. Query $q^N = 0$, $q^R = (0, -1)$ rotated to $(1, 0)$.
- Scores of token 1: key 0 is $0 \cdot 2 + (1, 0)\cdot(1, 1) = 1$; key 1 is $0 \cdot (-2) + (1, 0) \cdot (0, -1) = 0$. Scaled by $s = 3^{-1/2} = 0.577350$: softmax gives $p = (0.640457, 0.359543)$.
- Output: $3 \cdot 0.640457 - 3 \cdot 0.359543 = 0.842745$, so $(0.842745, 0)$.

Absorbed: $W_{UK} = 2$, so $q^{lat} = 0$ and the scores are the rope terms alone, the same two numbers. The weighted latent is $0.640457 - 0.359543 = 0.280915$ and $W_O W_{UV} = 1 \cdot 3 = 3$: again $0.842745$. The cache holds latents $(1, -1)$ and rope keys $(1, 1)$, $(0, -1)$: three numbers per token. This is `test_hand_example`.

## 4. The interface

```python
class LatentCacheHook(Protocol):
    def update(self, layer: int, c_new: NDArray, k_rope_new: NDArray) -> tuple[NDArray, NDArray]: ...
class ConcatLatentCache:                                    # update, seq_len
class MLAttention(Module):
    def __init__(self, d, n_heads, q_lora_rank, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_dim, rope: RopeSpec,
                 attention_bias=False, softmax_scale=None, rng=None)
    def forward(self, x, positions, mask=None, cache=None, layer=0) -> Tensor   # [B, T, d] -> [B, T, d]
    def absorb_weights(self) -> AbsorbedMLA
class AbsorbedMLA:                                          # w_uk [H, dn, r], w_ov [H, d, r], o_bias
    def forward(self, x, positions, cache, layer=0, mask=None) -> NDArray
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3: outputs, weights, cache contents, absorbed form | you and the test agree on every split |
| `test_state_dict_keys_and_shapes_are_hf` | unit | HF's parameter names, order, shapes | DeepSeek checkpoints load by name in `L7.9` |
| `test_mla_golden` | golden | `DeepseekV3Attention` outputs and cached latents, plain and low-rank queries | the model in MS-L7 and C1 |
| `test_absorbed_decode_equals_naive` | differential | prefill then decode, absorbed vs naive, float64 | `L8.2` decodes with the cheap form |
| `test_cache_chunks_equal_full_forward` | differential | chunked prefill through the cache | chunked prefill in `L10.3` |
| `test_cache_bytes_match_m05_1` | property | cached arrays are `kv_bytes_per_token` for `mla` | admission by KV bytes, C1's equal-bytes ablation |
| `test_scores_see_relative_position_only` | property | shifting every position changes nothing | the decoupled rope key is RoPE |
| `test_padding_mask_and_fully_masked_rows` | boundary | padded keys hidden in both forms; an empty row gives 0 | batched prompts of unequal length |
| `test_gradcheck_through_the_latent` | gradcheck | float64 central differences through the latent | C1 trains MLA |
| `test_concat_latent_cache` | unit | append, copy, `seq_len`, chunk checks | the hook `L8.2` implements |
| `test_validation` | boundary | rope width, sizes, scale, positions, mask shape | wiring bugs fail loudly |

### Your graded tests (rung R5)

Your oracle is MLA written out in numpy float64 from the equations of section 2 (RoPE as complex multiplication), compared with `MLAttention` on random weights for both query forms and both layouts; the absorbed form against the naive one while decoding; a padding mask in the absorbed form; a central-difference gradient through `kv_a_proj_with_mqa`; the cache's copy; the rope-width check; the state-dict names. Import only `tinyllm.modern.mla`, `tinyllm.modern.rope`, and `tinyllm.autograd`. `ol check L7.6` requires 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. reading `kv_b_proj` per head as v first, then k_nope | a DeepSeek checkpoint loads and attends to garbage | `test_mla_golden`, `test_absorbed_decode_equals_naive` (mutant `s01`) |
| 2. splitting the query rope-first | scores mix position-free and rotated entries | `test_hand_example`, `test_mla_golden` (mutant `s02`) |
| 3. forgetting to rotate the shared rope key | relative position is lost; outputs drift with absolute position | `test_scores_see_relative_position_only` (mutant `s03`) |
| 4. caching the latent before `kv_a_layernorm` | the cached keys and values are scaled wrong | `test_hand_example`, `test_mla_golden` (mutant `s04`) |
| 5. scaling scores by $d_n^{-1/2}$ instead of $(d_n + d_r)^{-1/2}$ | softmax too sharp; HF disagrees | `test_mla_golden`, `test_validation` (mutant `s05`) |
| the rope key in a different layout from the query | relative scores break only for interleaved checkpoints | `test_mla_golden` (mutant `s06`) |
| absorbing with the value rows of `kv_b_proj` | the absorbed decode diverges from the naive one | `test_absorbed_decode_equals_naive` (mutant `s07`) |
| a cache that returns only the new chunk | each decode step forgets the past | `test_cache_chunks_equal_full_forward` (mutant `s08`) |
| the absorbed form dropping the padding mask | padded batches decode wrong only in production | `test_padding_mask_and_fully_masked_rows` (mutant `s09`) |
| pairing `o_proj` columns with the wrong head in $W_O W_{UV}$ | absorbed outputs wrong for $H > 1$ only | `test_absorbed_decode_equals_naive` (mutant `s10`) |
| a detached latent | `kv_a_proj_with_mqa` never learns | `test_gradcheck_through_the_latent` (mutant `s11`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L7.3` | rotates the decoupled rope parts with a `RopeSpec` |
| Back | `L7.1` | `RMSNorm` on the latent and the query bottleneck |
| Back | `L0.4` | `Linear` projections, `Module` registration |
| Back | `L0.2` | the ops that make the naive form trainable |
| Back | `L0.1` | `Tensor` slicing and products |
| Back | `M06.3` | the default initialization stream |
| Back | `M05.1` | `kv_bytes_per_token` for `mla`, checked against the cache |
| Forward | `L7.9` | `tl_attention = "mla"` builds this layer in every block |
| Forward | `L8.2` | `LatentCache` implements the hook and decodes with `AbsorbedMLA` |
| Forward | C1 | the MLA-vs-GQA ablation at equal KV bytes |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `MLAttention` | HF `DeepseekV3Attention` | YaRN `mscale` folded into the softmax scale, interleaved rope weights | `transformers/models/deepseek_v3/modeling_deepseek_v3.py` |
| `AbsorbedMLA` | vLLM MLA backend, FlashMLA | absorbed decode kernels over a paged latent cache | vLLM `vllm/attention/backends/mla/`, DeepSeek FlashMLA |
| `ConcatLatentCache` | SGLang `MLATokenToKVPool` | one latent pool per layer, radix-tree prefix sharing | SGLang `mem_cache/memory_pool.py` |
| the factorization | TransMLA, MHA2MLA | converting a trained GQA model to MLA with an SVD and fine-tuning | Meng et al. 2025, Ji et al. 2025 |
