# Context extension: PI, NTK, YaRN, Llama-3 scaling; ALiBi (optional part)

## Overview

| | |
|---|---|
| **Module** | `L7.4` · build · Python · Pass 5 · 2 h reading; the ALiBi part (section 2.6) is optional (D31) |
| **Solution** | [`ctxext.py`](../../../../course/solutions/python/tinyllm/modern/ctxext.py): `rope_inv_freq_scaled`, `_yarn_ramp`, and the optional `alibi_bias`; model oracle tests in [`test_ctxext.py`](../../../../course/solutions/learner-tests/L7.4/python/tests/l7-4-ctxext/test_ctxext.py) |
| **Contract** | [`course/contracts/py/tinyllm/modern/ctxext.pyi`](../../../../course/contracts/py/tinyllm/modern/ctxext.pyi) |
| **Needs** | `M00.3` `rope_inv_freq` and `alibi_slopes` · reading: `L7.3` (what the frequencies do) |
| **Used by** | `L7.9` builds RoPE from `config.json`'s `rope_scaling` (the `tiny-llama-yarn` and `tiny-llama-3` fixture models), on the way to SmolLM2-135M logits that match Hugging Face · later: `C1` evaluates at twice the training context with `rope_scaling = yarn` |
| **Optional depth** | Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (2023); bloc97, "NTK-Aware Scaled RoPE" (2023); Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (2023), sections 3.1 to 3.4; Meta, "The Llama 3 Herd of Models" (2024); Press et al., "Train Short, Test Long: Attention with Linear Biases" (2022) |

## Key Takeaways

- Every RoPE extension method only changes the frequencies (and sometimes a temperature); the rotation code of `L7.3` is unchanged.
- Position interpolation slows every pair by the factor, so long positions reuse trained angles exactly.
- NTK-aware scaling raises the base so the fastest pair is untouched and the slowest is slowed by exactly the factor.
- YaRN and Llama 3 split the pairs into bands by how often they turn over the original context: keep the fast ones, slow the slow ones, blend between; YaRN also sharpens attention by $0.1 \ln s + 1$.
- ALiBi (optional) uses no rotation at all: a bias that falls linearly with distance, one slope per head.

## How to read this chapter

Sections 1 to 3 explain why long positions break a trained model and work every scaling rule by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution and shows each worked number falling out of the code. Try to predict each step of the solution before reading it.

---

## 1. Why now

A model trained on contexts of $L$ tokens has learned what every pair's angle $p\,\theta_i$ looks like for $p < L$. Feed it position $2L$ and the slow pairs, which never completed a turn during training, sit at angles it has never seen: its attention degrades and perplexity climbs past the training length. Real checkpoints ship a fix in `config.json`: Llama 3.1 has `"rope_scaling": {"rope_type": "llama3", "factor": 8.0, ...}`, Qwen2.5 uses `yarn`. The loader in `L7.9` must build exactly the frequencies Hugging Face builds from those fields, and the capstone (`C1`) measures what extension buys at twice the training context. This module computes them.

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

ALiBi drops rotations entirely. Each head $h$ adds a bias $-m_h \cdot (i - j)$ to the score of query position $i$ and key position $j$: the further back, the lower. The slopes are `M00.3`'s geometric sequence ($2^{-8/n}, 2^{-16/n}, \dots$ for $n$ heads). With $T_q$ queries that are the last of $T_k$ keys, query $i$ sits at key index $T_k - T_q + i$. Softmax ignores a constant added to a whole row, so BLOOM's form $m_h \cdot j$ gives the same weights. ALiBi has no call site in the system (every model the system loads uses RoPE), so this part is optional.

## 3. Worked example by hand

**Linear and NTK**, $r = 4$, $b = 10^4$: $\theta = (1, 0.01)$. Linear with $s = 2$: $(0.5, 0.005)$. NTK with $s = 2$: $b' = 10^4 \cdot 2^{4/2} = 4 \cdot 10^4$, $\theta' = (1, (4 \cdot 10^4)^{-1/2}) = (1, 0.005)$: the fast pair kept, the slow one halved.

**YaRN**, $r = 8$, $b = 10^4$, $s = 4$, $L_0 = 2048$: $\theta = (1, 0.1, 0.01, 0.001)$. $L_0 / 2\pi = 325.95$; $c(32) = 8 \ln(10.186) / 18.421 = 1.008$, so $lo = 1$; $c(1) = 8 \ln(325.95) / 18.421 = 2.513$, so $hi = 3$. Ramp $\rho = (0, 0, 0.5, 1)$. $\theta' = (1, 0.1, 0.5 \cdot 0.01 + 0.5 \cdot 0.0025, 0.00025) = (1, 0.1, 0.00625, 0.00025)$; $t = 0.1 \ln 4 + 1 = 1.138629$.

**Llama 3**, same $\theta$, $s = 8$, $L_0 = 1024$: wavelengths $(6.28, 62.8, 628.3, 6283)$; the bands are below 256 (keep) and above 1024 (divide). Pairs 0 and 1 keep, pair 3 becomes $0.000125$, pair 2 blends with $\sigma = (1024 / 628.3 - 1) / 3 = 0.209916$: $0.01 \cdot (0.790084 / 8 + 0.209916) = 0.003087$.

**ALiBi**, 2 heads (slopes $1/16$, $1/256$), 2 queries over 3 keys: distances $(1, 0, -1)$ and $(2, 1, 0)$, head 0 biases $(-0.0625, 0, 0.0625)$ and $(-0.125, -0.0625, 0)$.

## 4. The interface

```python
def rope_inv_freq_scaled(d_rot, base, kind: Literal["default", "linear", "ntk", "yarn", "llama3"], factor=1.0,
                         original_max_pos=0, beta_fast=32, beta_slow=1, low_freq_factor=1, high_freq_factor=4
                         ) -> tuple[NDArray, float]: ...      # (inv_freq float64 [d_rot/2], attention_scaling)
def alibi_bias(n_heads: int, Tq: int, Tk: int) -> NDArray: ...  # float32 [n_heads, Tq, Tk]; optional
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: linear and NTK | unit | section 3, linear, NTK, default | the solution and the reader agree on the rules |
| hand example: YaRN | unit | section 3, YaRN ramp and temperature | Qwen-style configs |
| hand example: Llama 3 | unit | section 3, the three bands | Llama 3.1 configs |
| HF `ROPE_INIT_FUNCTIONS` | golden | ten HF cases, every kind | `rope_scaling` in `L7.9` |
| linear maps long positions onto trained angles | property | angles at $s p$ equal trained angles at $p$ | why interpolation works |
| NTK keeps the fastest pair, divides the slowest | property | fastest kept, slowest divided by $s$, monotone between | the NTK base rule |
| YaRN bands | property | fast pairs kept, slow divided, temperature, $s = 1$ is default | YaRN's design |
| Llama 3 bands keep frequencies ordered | property | bands and strictly decreasing frequencies | no reordered pairs |
| validation | boundary | unknown kind, odd width, bad base, factor below 1, missing $L_0$, swapped factors | config typos fail loudly |
| hand example: ALiBi | unit | section 3 ALiBi (optional) | ALiBi models |
| same softmax as BLOOM | golden | same weights as BLOOM after the causal mask (optional) | BLOOM's convention |
| a decode row equals the last row of prefill | property | decoding rows match prefill (optional) | incremental decoding |
| ALiBi validation | boundary | more queries than keys, zero sizes (optional) | wiring bugs |

### A model oracle test suite

[`test_ctxext.py`](../../../../course/solutions/learner-tests/L7.4/python/tests/l7-4-ctxext/test_ctxext.py) writes the rules of section 2 out pair by pair in plain Python floats and compares `rope_inv_freq_scaled` against them for linear, NTK, YaRN at several sizes, and Llama 3, checks YaRN's temperature, and checks that bad configs raise. A loop over pairs in scalar floats is a good oracle for vectorised numpy code: it is slow and obvious, so a mistake in a mask or a broadcast in the solution has nowhere to hide.

## 5. The solution, step by step

### 5.1 From first principles

1. Section 2.1 says the damage is confined to pairs whose angles leave the trained range. So the fix belongs in the frequencies $\theta_i$, not in the rotation: the output of this module is a new `inv_freq` (plus a temperature) that `L7.3`'s `rope_cos_sin` consumes unchanged.
2. Every method starts from the default ladder $\theta_i = b^{-2i/r}$ and modifies it, so the solution computes `theta = rope_inv_freq(d_rot, base)` once (`M00.3`) and branches on `kind`.
3. Linear and NTK have closed forms (sections 2.2, 2.3): divide every $\theta_i$ by $s$, or recompute the ladder with base $b \cdot s^{r/(r-2)}$. Reusing `rope_inv_freq` for NTK means the "fastest kept, slowest divided by $s$" property follows from the algebra, not from extra code.
4. YaRN and Llama 3 need per-pair decisions. Vectorise them: compute a per-pair weight in $[0, 1]$ (YaRN's ramp $\rho_i$, Llama 3's $\sigma$) and blend $\theta_i$ with $\theta_i/s$. A clamp turns the blend into "keep" and "divide" at the two ends, so three bands need no branching.
5. YaRN's band edges are pair indices derived from "how many turns over $L_0$". Inverting $L_0 / \lambda_i = n$ with $\lambda_i = 2\pi b^{2i/r}$ gives $c(n)$; the edges are rounded **outward** (floor the low edge, ceil the high) so the blend region never shrinks.
6. The return type carries the temperature because it belongs to the same config field and must reach `rope_cos_sin`'s `attention_scaling`: $1$ for every kind but YaRN.
7. Configs come from JSON written by people, so every parameter a kind needs is validated for that kind (an `original_max_pos` only matters to YaRN and Llama 3).

### 5.2 The code, annotated

**The YaRN ramp.**

```python
def _yarn_ramp(d_rot, base, original_max_pos, beta_fast, beta_slow):
    """ramp_i in [0, 1] per pair: 0 keeps theta_i, 1 divides it by factor."""
    def corr(n_rot):
        # the pair index whose wavelength fits n_rot times into the context:
        # solve L0 / (2 pi b^(2i/r)) = n_rot for i
        return d_rot * math.log(original_max_pos / (n_rot * 2 * math.pi)) / (2 * math.log(base))

    # round outward and clamp to valid indices (HF's find_correction_range)
    lo = max(math.floor(corr(beta_fast)), 0)
    hi = min(math.ceil(corr(beta_slow)), d_rot - 1)
    if lo == hi:
        hi += 0.001          # avoid 0/0 when both edges land on one pair
    i = np.arange(d_rot // 2, dtype=np.float64)
    return np.clip((i - lo) / (hi - lo), 0.0, 1.0)
```

On section 3's numbers, `corr(32)` is 1.008 and `corr(1)` is 2.513, so `lo = 1`, `hi = 3`, and the ramp over $i = 0..3$ is $(0, 0, 0.5, 1)$. The factor `2` in the denominator is the "2" in $b^{2i/r}$; dropping it moves both edges by a factor of two.

**The dispatcher.**

```python
def rope_inv_freq_scaled(d_rot, base, kind, factor=1.0, original_max_pos=0,
                         beta_fast=32, beta_slow=1, low_freq_factor=1, high_freq_factor=4):
    if kind not in ("default", "linear", "ntk", "yarn", "llama3"):
        raise ValueError(f"unknown kind {kind!r}")
    # bool is an int subclass: reject True explicitly; d_rot must be even for pairs
    if isinstance(d_rot, bool) or int(d_rot) != d_rot or d_rot < 2 or d_rot % 2:
        raise ValueError(...)
    if not base > 1:
        raise ValueError(...)
    d_rot = int(d_rot)
    theta = rope_inv_freq(d_rot, base)           # the trained ladder, step 2
    if kind == "default":
        return theta, 1.0
    if not factor >= 1:                          # factor < 1 would shrink the context
        raise ValueError(...)
    if kind == "linear":
        # position interpolation: position p behaves like p / factor
        return theta / factor, 1.0
    if kind == "ntk":
        # NTK-aware: a larger base; the fastest pair keeps its frequency and
        # the slowest (i = r/2 - 1) is divided by exactly factor.
        return rope_inv_freq(d_rot, base * factor ** (d_rot / (d_rot - 2))), 1.0
    # ... validate original_max_pos >= 1 (only yarn and llama3 need it)
```

`not factor >= 1` rather than `factor < 1` also rejects NaN, which compares false to everything.

**YaRN.**

```python
    if kind == "yarn":
        ramp = _yarn_ramp(d_rot, base, int(original_max_pos), beta_fast, beta_slow)
        keep = 1.0 - ramp  # HF's inv_freq_extrapolation_factor
        # keep = 1: extrapolate (trained theta); keep = 0: interpolate (theta / s)
        inv = (theta / factor) * (1.0 - keep) + theta * keep
        # sharpen attention: cos and sin times t, every score times t^2
        scale = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
        return inv, scale
```

Written with HF's `keep` so the expression matches `_compute_yarn_parameters` term for term; it is $(1 - \rho_i)\theta_i + \rho_i \theta_i / s$. On the example, pair 2 gets $0.5 \cdot 0.0025 + 0.5 \cdot 0.01 = 0.00625$ and $t = 1.138629$.

**Llama 3.**

```python
    if not high_freq_factor > low_freq_factor > 0:     # swapped factors flip the bands
        raise ValueError(...)
    L0 = float(original_max_pos)
    wavelen = 2 * math.pi / theta
    # long wavelengths (slow pairs, > L0 / low) are divided by factor; others kept for now
    inv = np.where(wavelen > L0 / low_freq_factor, theta / factor, theta)
    # sigma: 0 at wavelength L0 / low, 1 at L0 / high
    smooth = (L0 / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv / factor + smooth * inv
    # the medium band is neither short (< L0/high) nor long (> L0/low)
    medium = ~(wavelen < L0 / high_freq_factor) & ~(wavelen > L0 / low_freq_factor)
    return np.where(medium, smoothed, inv), 1.0
```

In the medium band `inv` is still `theta`, so `smoothed` is $(1 - \sigma)\theta_i/s + \sigma\theta_i$, section 2.5's formula. Pair 2 of the example has wavelength 628.3, between 256 and 1024, so $\sigma = 0.209916$ and $\theta'_2 = 0.003087$. Because $\sigma$ grows with frequency, the blended values sit between their neighbours and the ladder stays strictly decreasing.

**ALiBi (optional).**

```python
def alibi_bias(n_heads, Tq, Tk):
    if min(n_heads, Tq, Tk) < 1 or Tq > Tk:
        raise ValueError(...)
    slopes = alibi_slopes(n_heads)
    # queries are the LAST Tq of Tk positions (prefill: Tq = Tk; decode: Tq = 1)
    i = np.arange(Tq)[:, None] + (Tk - Tq)  # absolute index of each query
    j = np.arange(Tk)[None, :]
    dist = (i - j).astype(np.float64)
    # further back means lower score; future keys (dist < 0) are masked by the caller
    return (-slopes[:, None, None] * dist[None]).astype(np.float32)
```

The offset `Tk - Tq` is what makes a one-row decoding bias equal the last row of the prefill bias.

Full solution: [`ctxext.py`](../../../../course/solutions/python/tinyllm/modern/ctxext.py).

### 5.3 Why this solution

- **Cost.** Everything is $O(r)$ work, done once when a model is loaded. Precision, not speed, is what matters: the ladder is float64 and is cast to float32 only inside `rope_cos_sin`.
- **Frequencies, not positions.** Linear interpolation could equally be written as "divide positions by $s$". Changing the frequencies instead keeps one code path for every method and lets `L8.2` pass raw integer positions while decoding.
- **Matching HF's formulas exactly.** Several algebraically equal ways exist to write the YaRN blend or the Llama 3 bands; the solution follows HF's term order and rounding (`floor`/`ceil` of the correction range, the `+ 0.001` guard) so the golden values agree to float64 precision. A "cleaner" formula that rounds the range inward moves a pair into the wrong band.
- **Static, not dynamic.** HF's `dynamic` type recomputes the NTK base per length. That changes the frequencies under an existing KV cache (`L8.2`), so keys cached at the old frequencies no longer match. The solution's `ntk` takes one fixed factor, so the table never changes while a model is decoding.

### 5.4 Explain it back

1. Why does NTK raise the base by $s^{r/(r-2)}$ and not by $s$?
   <details><summary>Answer</summary>The last pair has exponent $-(r-2)/r$. For it to be divided by exactly $s$ we need $b'^{-(r-2)/r} = b^{-(r-2)/r}/s$, so $b' = b \cdot s^{r/(r-2)}$. Pair 0 has exponent 0 and is unchanged whatever the base.</details>
2. Why is YaRN's band defined by how many turns a pair makes over $L_0$?
   <details><summary>Answer</summary>A pair that turns many times over the trained context has seen every angle, so extrapolating it is safe and keeps fine local order. A pair that turns less than once has not, so it must be interpolated. Turns per context is exactly the quantity that separates these.</details>
3. Why must YaRN multiply cos and sin by $t = 0.1 \ln s + 1$?
   <details><summary>Answer</summary>Slowing the slow pairs makes far keys look more alike, flattening (raising the entropy of) attention at long range. Scaling $q$ and $k$ by $t$ multiplies every logit by $t^2$, sharpening the softmax back. Without it perplexity is slightly worse everywhere.</details>
4. What goes wrong if `_yarn_ramp` rounds `lo` up and `hi` down?
   <details><summary>Answer</summary>The blend region shrinks by up to a pair at each end, so a pair HF blends is instead kept or fully divided. The frequencies then disagree with the checkpoint's training and with HF's golden values.</details>
5. Why does `alibi_bias` add `Tk - Tq` to the query index?
   <details><summary>Answer</summary>During decoding the $T_q$ new queries are the last positions of a $T_k$-long sequence. Without the offset, query 0 would be treated as key position 0, and the bias would not match the prefill row it is supposed to reproduce.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. multiplying by the factor in position interpolation | the context shrinks instead of growing | `theta / factor` in the `linear` branch |
| 2. scaling the base by $s$, not $s^{r/(r-2)}$ | the slowest pair is not slowed by exactly $s$ | `base * factor ** (d_rot / (d_rot - 2))` (step 3) |
| 3. YaRN's ramp backwards | fast pairs are interpolated, slow ones extrapolated: local order blurs | ramp 0 at low indices (fast pairs) means keep: `keep = 1 - ramp` |
| 4. forgetting YaRN's temperature | long-range attention too flat; perplexity a little worse everywhere | returns `0.1 * log(factor) + 1` as the second value (step 6) |
| 5. swapping Llama 3's low and high bands | the wrong pairs are slowed | long wavelengths compare against `L0 / low_freq_factor`, short against `L0 / high_freq_factor`; swapped factors raise |
| rounding the correction range inward | one pair lands in the wrong band | `floor` for `lo`, `ceil` for `hi` (step 5) |
| a missing 2 in the correction dimension | the bands move by a factor of two | `corr` divides by `2 * math.log(base)` |
| ALiBi bias growing with distance | far tokens dominate (optional part) | the bias is `-slope * dist` |
| ALiBi queries treated as the first keys | decoding rows wrong (optional part) | query index offset by `Tk - Tq` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.3` | `rope_inv_freq` is every ladder here; `alibi_slopes` gives the ALiBi slopes |
| Back | `L7.3` | the frequencies and temperature go into `rope_cos_sin` |
| Forward | `L7.9` | `LlamaConfig.rope_scaling` becomes a `RopeSpec` through `rope_inv_freq_scaled` |
| Forward | `C1` | the long-context eval: bits per byte at twice the training context with and without YaRN |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rope_inv_freq_scaled` | HF `ROPE_INIT_FUNCTIONS` | `dynamic` (recomputed per length), `longrope` (per-pair learned factors), DeepSeek's `mscale` | `transformers/modeling_rope_utils.py` |
| YaRN | vLLM `YaRNScalingRotaryEmbedding` | precomputed scaled tables with the temperature folded in | vLLM `rotary_embedding.py` |
| context extension | LongRoPE, fine-tuning on long data | searched per-pair factors; most of the gain still comes from continued training at the longer length | Ding et al. 2024; `sq.ctx-extension` |
| `alibi_bias` | BLOOM, MPT | the bias fused into attention kernels | FlashAttention `alibi_slopes` argument |
