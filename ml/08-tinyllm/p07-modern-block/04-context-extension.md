<!-- ol:module L7.4 -->
# Context extension: PI, NTK, YaRN, Llama-3 scaling; ALiBi (optional part)

## Overview

| | |
|---|---|
| **Module** | `L7.4` · build · Python · Pass 5 · 3 h, plus your graded tests (rung R5); the ALiBi part (section 2.6) is optional (D31) |
| **You build** | `python/tinyllm/modern/ctxext.py`: `rope_inv_freq_scaled` and, optionally, `alibi_bias`; and your own oracle tests in `python/tests/l7-4-ctxext/` |
| **Contract** | [`course/contracts/py/tinyllm/modern/ctxext.pyi`](../../../course/contracts/py/tinyllm/modern/ctxext.pyi) |
| **Tests** | `course/tests/L7.4/test_ctxext.py` and the optional `test_alibi.py` (skipped while `alibi_bias` raises `NotImplementedError`) (what they check: section 4), golden values from transformers 5.19.0 `ROPE_INIT_FUNCTIONS` and BLOOM's `build_alibi_tensor` in `course/fixtures/L7.4/rope_scaling_hf.json` (`course/oracle/L7.4/rope_scaling_hf.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `M00.3` `rope_inv_freq` and `alibi_slopes` · reading: `L7.3` (what the frequencies do) (or `--ref-deps`) |
| **Used by** | `L7.9` builds RoPE from `config.json`'s `rope_scaling` (the `tiny-llama-yarn` and `tiny-llama-3` fixture models) · later: `C1` evaluates at twice the training context with `rope_scaling = yarn` |
| **Milestone** | `MS-L7` (SmolLM2-135M logits match Hugging Face) |
| **Optional depth** | Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (2023); bloc97, "NTK-Aware Scaled RoPE" (2023); Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (2023), sections 3.1 to 3.4; Meta, "The Llama 3 Herd of Models" (2024); Press et al., "Train Short, Test Long: Attention with Linear Biases" (2022) |

## Key Takeaways

- Every RoPE extension method only changes the frequencies (and sometimes a temperature); the rotation code of `L7.3` is unchanged (`test_golden_hf_rope_init_functions`).
- Position interpolation slows every pair by the factor, so long positions reuse trained angles exactly (`test_linear_maps_long_positions_onto_trained_angles`).
- NTK-aware scaling raises the base so the fastest pair is untouched and the slowest is slowed by exactly the factor (`test_ntk_keeps_fastest_divides_slowest`).
- YaRN and Llama 3 split the pairs into bands by how often they turn over the original context: keep the fast ones, slow the slow ones, blend between; YaRN also sharpens attention by $0.1 \ln s + 1$ (`test_yarn_bands`, `test_llama3_bands_keep_frequencies_ordered`).
- ALiBi (optional) uses no rotation at all: a bias that falls linearly with distance, one slope per head (`test_hand_example_alibi`).

## How to work this chapter

```bash
ol start L7.4              # stubs ctxext.py; prints your test path and rung (R5)
ol tests L7.4              # the course tests (test_alibi.py is optional)
# write your oracle tests in python/tests/l7-4-ctxext/, then:
ol check L7.4              # course tests and the mutation grade of your tests
ol diff  L7.4              # after passing: your code against the reference
```

---

## 1. Why now

A model trained on contexts of $L$ tokens has learned what every pair's angle $p\,\theta_i$ looks like for $p < L$. Feed it position $2L$ and the slow pairs, which never completed a turn during training, sit at angles it has never seen: its attention degrades and perplexity climbs past the training length. Real checkpoints ship a fix in `config.json`: Llama 3.1 has `"rope_scaling": {"rope_type": "llama3", "factor": 8.0, ...}`, Qwen2.5 uses `yarn`. Your loader in `L7.9` must build exactly the frequencies Hugging Face builds from those fields, and the capstone (`C1`) measures what extension buys at twice the training context. This module computes them.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $r$ | `d_rot`, the rotary width | `int`, even |
| $b$ | `base` (`rope_theta`) | `float` $> 1$ |
| $\theta_i$ | default inverse frequency $b^{-2i/r}$, $i = 0 .. r/2 - 1$ | `float64[r/2]` |
| $s$ | `factor`: how many times longer the new context is | `float` $\ge 1$ |
| $L_0$ | `original_max_pos`, the trained context | `int` |
| $\lambda_i$ | wavelength $2\pi / \theta_i$: positions per full turn of pair $i$ | `float` |
| $\theta'_i$ | the scaled inverse frequency | `float64[r/2]` |
| $t$ | attention scaling, multiplies cos and sin (`L7.3`) | `float` |
| $m_h$ | ALiBi slope of head $h$ | `float` |

### 2.1 What breaks

Pair $i$ completes $L_0 / \lambda_i$ turns over the trained context. Fast pairs (small $\lambda_i$) turn hundreds of times: every angle was seen, and they encode fine local order. Slow pairs turn less than once: at position $2 L_0$ they reach angles never seen. Each method below decides which pairs to slow down and by how much.

### 2.2 Position interpolation (`linear`)

Divide every frequency by $s$: $\theta'_i = \theta_i / s$. Position $s p$ then has exactly the angles position $p$ had. Nothing is new, but neighbouring tokens are now $s$ times closer in angle on the fast pairs, so the model must be fine-tuned to tell them apart.

### 2.3 NTK-aware scaling (`ntk`)

Keep the fast pairs, slow the slow ones, by changing the base instead: $b' = b \cdot s^{r / (r - 2)}$. Then $\theta'_0 = 1 = \theta_0$, and for the last pair $i = r/2 - 1$, $\theta'_i = b'^{-(r-2)/r} = b^{-(r-2)/r} / s = \theta_i / s$: exactly the factor. Pairs between are slowed by between 1 and $s$. Hugging Face's `dynamic` type applies this rule at inference with a factor that grows with the current length, $s_\text{eff} = s \cdot T / L_0 - (s - 1)$; at $T = 2 L_0$ and $s = 2$ that is 3, which is how the golden fixture checks `ntk`.

### 2.4 YaRN (`yarn`)

YaRN makes the bands explicit. Pair $i$ turns $L_0 / \lambda_i$ times; the pair index that turns exactly $n$ times is

$$c(n) = \frac{r \ln\left(L_0 / (2\pi n)\right)}{2 \ln b}.$$

With $lo = \lfloor c(\beta_\text{fast}) \rfloor$, $hi = \lceil c(\beta_\text{slow}) \rceil$ (clamped to $[0, r - 1]$; defaults $\beta_\text{fast} = 32$, $\beta_\text{slow} = 1$), the ramp $\rho_i = \operatorname{clamp}((i - lo) / (hi - lo), 0, 1)$ blends

$$\theta'_i = (1 - \rho_i)\, \theta_i + \rho_i\, \theta_i / s .$$

Pairs turning more than 32 times keep $\theta_i$, pairs turning less than once get $\theta_i / s$. Stretching the slow pairs makes attention scores flatter (more entropy) at long range, so YaRN sharpens them: cos and sin are multiplied by $t = 0.1 \ln s + 1$, which multiplies every score by $t^2$.

### 2.5 Llama 3 scaling (`llama3`)

Llama 3.1 uses wavelength bands directly, with $L_0 = 8192$, low factor 1 and high factor 4: $\lambda_i < L_0 / 4$ keeps $\theta_i$, $\lambda_i > L_0$ gets $\theta_i / s$, and between, with $\sigma = (L_0 / \lambda_i - 1) / (4 - 1)$, $\theta'_i = (1 - \sigma)\, \theta_i / s + \sigma\, \theta_i$. No temperature. The frequencies stay strictly decreasing in $i$.

### 2.6 ALiBi (optional)

ALiBi drops rotations entirely. Each head $h$ adds a bias $-m_h \cdot (i - j)$ to the score of query position $i$ and key position $j$: the further back, the lower. The slopes are `M00.3`'s geometric sequence ($2^{-8/n}, 2^{-16/n}, \dots$ for $n$ heads). With $T_q$ queries that are the last of $T_k$ keys, query $i$ sits at key index $T_k - T_q + i$. Softmax ignores a constant added to a whole row, so BLOOM's form $m_h \cdot j$ gives the same weights. ALiBi has no call site in the system you build (every model you load uses RoPE), so this part is optional: `test_alibi.py` skips itself while `alibi_bias` raises `NotImplementedError`.

## 3. Worked example by hand

**Linear and NTK**, $r = 4$, $b = 10^4$: $\theta = (1, 0.01)$. Linear with $s = 2$: $(0.5, 0.005)$. NTK with $s = 2$: $b' = 10^4 \cdot 2^{4/2} = 4 \cdot 10^4$, $\theta' = (1, (4 \cdot 10^4)^{-1/2}) = (1, 0.005)$: the fast pair kept, the slow one halved.

**YaRN**, $r = 8$, $b = 10^4$, $s = 4$, $L_0 = 2048$: $\theta = (1, 0.1, 0.01, 0.001)$. $L_0 / 2\pi = 325.95$; $c(32) = 8 \ln(10.186) / 18.421 = 1.008$, so $lo = 1$; $c(1) = 8 \ln(325.95) / 18.421 = 2.513$, so $hi = 3$. Ramp $\rho = (0, 0, 0.5, 1)$. $\theta' = (1, 0.1, 0.5 \cdot 0.01 + 0.5 \cdot 0.0025, 0.00025) = (1, 0.1, 0.00625, 0.00025)$; $t = 0.1 \ln 4 + 1 = 1.138629$.

**Llama 3**, same $\theta$, $s = 8$, $L_0 = 1024$: wavelengths $(6.28, 62.8, 628.3, 6283)$; the bands are below 256 (keep) and above 1024 (divide). Pairs 0 and 1 keep, pair 3 becomes $0.000125$, pair 2 blends with $\sigma = (1024 / 628.3 - 1) / 3 = 0.209916$: $0.01 \cdot (0.790084 / 8 + 0.209916) = 0.003087$.

**ALiBi**, 2 heads (slopes $1/16$, $1/256$), 2 queries over 3 keys: distances $(1, 0, -1)$ and $(2, 1, 0)$, head 0 biases $(-0.0625, 0, 0.0625)$ and $(-0.125, -0.0625, 0)$.

These are `test_hand_example_linear_and_ntk`, `test_hand_example_yarn`, `test_hand_example_llama3`, and `test_hand_example_alibi`.

## 4. The interface

```python
def rope_inv_freq_scaled(d_rot, base, kind: Literal["default", "linear", "ntk", "yarn", "llama3"], factor=1.0,
                         original_max_pos=0, beta_fast=32, beta_slow=1, low_freq_factor=1, high_freq_factor=4
                         ) -> tuple[NDArray, float]: ...      # (inv_freq float64 [d_rot/2], attention_scaling)
def alibi_bias(n_heads: int, Tq: int, Tk: int) -> NDArray: ...  # float32 [n_heads, Tq, Tk]; optional
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_linear_and_ntk` | unit | section 3, linear, NTK, default | you and the test agree on the rules |
| `test_hand_example_yarn` | unit | section 3, YaRN ramp and temperature | Qwen-style configs |
| `test_hand_example_llama3` | unit | section 3, the three bands | Llama 3.1 configs |
| `test_golden_hf_rope_init_functions` | golden | ten HF cases, every kind | `rope_scaling` in `L7.9` |
| `test_linear_maps_long_positions_onto_trained_angles` | property | angles at $s p$ equal trained angles at $p$ | why interpolation works |
| `test_ntk_keeps_fastest_divides_slowest` | property | fastest kept, slowest divided by $s$, monotone between | the NTK base rule |
| `test_yarn_bands` | property | fast pairs kept, slow divided, temperature, $s = 1$ is default | YaRN's design |
| `test_llama3_bands_keep_frequencies_ordered` | property | bands and strictly decreasing frequencies | no reordered pairs |
| `test_validation` | boundary | unknown kind, odd width, bad base, factor below 1, missing $L_0$, swapped factors | config typos fail loudly |
| `test_hand_example_alibi` | unit | section 3 ALiBi (optional) | ALiBi models |
| `test_golden_bloom_same_softmax` | golden | same weights as BLOOM after the causal mask (optional) | BLOOM's convention |
| `test_decode_row_equals_last_row_of_prefill` | property | decoding rows match prefill (optional) | incremental decoding |
| `test_validation_alibi` | boundary | more queries than keys, zero sizes (optional) | wiring bugs |

### Your graded tests (rung R5)

Your oracles are the rules of section 2 written out pair by pair in plain Python floats: compare `rope_inv_freq_scaled` for linear, NTK, YaRN at several sizes, and Llama 3, check YaRN's temperature, and that bad configs raise. Import only `tinyllm.modern.ctxext`. `ol check L7.4` requires 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. multiplying by the factor in position interpolation | the context shrinks instead of growing | `test_hand_example_linear_and_ntk` (mutant `s01`) |
| 2. scaling the base by $s$, not $s^{r/(r-2)}$ | the slowest pair is not slowed by exactly $s$ | `test_ntk_keeps_fastest_divides_slowest` (mutant `s02`) |
| 3. YaRN's ramp backwards | fast pairs are interpolated, slow ones extrapolated: local order blurs | `test_yarn_bands` (mutant `s03`) |
| 4. forgetting YaRN's temperature | long-range attention too flat; perplexity a little worse everywhere | `test_hand_example_yarn` (mutant `s04`) |
| 5. swapping Llama 3's low and high bands | the wrong pairs are slowed | `test_llama3_bands_keep_frequencies_ordered` (mutant `s05`) |
| rounding the correction range inward | one pair lands in the wrong band | `test_hand_example_yarn` (mutant `s06`) |
| a missing 2 in the correction dimension | the bands move by a factor of two | `test_golden_hf_rope_init_functions` (mutant `s07`) |
| ALiBi bias growing with distance | far tokens dominate (optional part) | `test_golden_bloom_same_softmax` (mutant `s08`) |
| ALiBi queries treated as the first keys | decoding rows wrong (optional part) | `test_decode_row_equals_last_row_of_prefill` (mutant `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.3` | `rope_inv_freq` is every ladder here; `alibi_slopes` gives the ALiBi slopes |
| Back | `L7.3` | the frequencies and temperature go into `rope_cos_sin` |
| Forward | `L7.9` | `LlamaConfig.rope_scaling` becomes a `RopeSpec` through `rope_inv_freq_scaled` |
| Forward | `C1` | the long-context eval: bits per byte at twice the training context with and without YaRN |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rope_inv_freq_scaled` | HF `ROPE_INIT_FUNCTIONS` | `dynamic` (recomputed per length), `longrope` (per-pair learned factors), DeepSeek's `mscale` | `transformers/modeling_rope_utils.py` |
| YaRN | vLLM `YaRNScalingRotaryEmbedding` | precomputed scaled tables with the temperature folded in | vLLM `rotary_embedding.py` |
| context extension | LongRoPE, fine-tuning on long data | searched per-pair factors; most of the gain still comes from continued training at the longer length | Ding et al. 2024; `sq.ctx-extension` |
| `alibi_bias` | BLOOM, MPT | the bias fused into attention kernels | FlashAttention `alibi_slopes` argument |
