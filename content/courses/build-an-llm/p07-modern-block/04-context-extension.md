# Context Extension: PI, NTK, YaRN, Llama 3 Scaling, and ALiBi

A model trained on contexts of $L$ tokens has learned what every pair's angle $p\,\theta_i$ looks like for $p < L$. Feed it position $2L$ and the slow pairs, which never completed a turn during training, sit at angles it has never seen: its attention degrades and perplexity climbs past the training length.

Real checkpoints ship a fix in `config.json`. Llama 3.1 has `"rope_scaling": {"rope_type": "llama3", "factor": 8.0, ...}`, and Qwen2.5 uses `yarn`. The loader in [The Llama-Family Model: Config, Weight Loading, and Download](09-llama-family-model.md) must build exactly the frequencies Hugging Face builds from those fields (its `tiny-llama-yarn` and `tiny-llama-3` fixture models test this), and the [TinyStories capstone](../capstones/01-tinystories.md) measures what extension buys at twice the training context. This section computes those frequencies for every scaling rule, and, as an optional final part, builds ALiBi, a position scheme that uses no rotation at all.

You should have [Rotary Position Embeddings](03-rope.md) fresh, since everything here is about what its frequencies do. The default ladder `rope_inv_freq` and the ALiBi slopes `alibi_slopes` come from [Sequences, geometric series, and frequency ladders](../../mathematics-for-computing/00-precalculus/03-sequences-and-frequency-ladders.md) in *Mathematics for Computing*. The reference solution is [`ctxext.py`](../../../../course/solutions/python/tinyllm/modern/ctxext.py), with `rope_inv_freq_scaled`, `_yarn_ramp`, and the optional `alibi_bias`, against the contract [`course/contracts/py/tinyllm/modern/ctxext.pyi`](../../../../course/contracts/py/tinyllm/modern/ctxext.pyi).

> **Learning objectives**
> - Explain why only the slow RoPE pairs break past the trained context, and why every fix changes frequencies rather than the rotation.
> - Compute position interpolation, NTK-aware, YaRN, and Llama 3 frequencies by hand, including YaRN's temperature.
> - Implement the scaling rules so they match Hugging Face's `ROPE_INIT_FUNCTIONS` to float64 precision.
> - (Optional) Build an ALiBi bias whose decoding rows equal the last rows of prefill.

## Notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $r$ | `d_rot`, the rotary width | `int`, even |
| $b$ | `base` (`rope_theta`) | `float` $> 1$ |
| $\theta_i$ | default inverse frequency $b^{-2i/r}$, $i = 0 .. r/2 - 1$ | `float64[r/2]` |
| $s$ | `factor`: how many times longer the new context is | `float` $\ge 1$ |
| $L_0$ | `original_max_pos`, the trained context | `int` |
| $\lambda_i$ | wavelength $2\pi / \theta_i$: positions per full turn of pair $i$ | `float` |
| $\theta'_i$ | the scaled inverse frequency | `float64[r/2]` |
| $t$ | attention scaling, multiplies cos and sin | `float` |
| $m_h$ | ALiBi slope of head $h$ | `float` |

## What breaks

Pair $i$ completes $L_0 / \lambda_i$ turns over the trained context. Fast pairs (small $\lambda_i$) turn hundreds of times: every angle was seen, and they encode fine local order. Slow pairs turn less than once: at position $2 L_0$ they reach angles never seen. The damage is confined to frequencies, so every method below leaves the rotation code untouched and only decides which pairs to slow down and by how much (and sometimes adds a temperature).

## Position interpolation

The bluntest fix (`linear` in Hugging Face's terms) divides every frequency by $s$: $\theta'_i = \theta_i / s$. Position $s p$ then has exactly the angles position $p$ had. Nothing is new, but neighbouring tokens are now $s$ times closer in angle on the fast pairs, so the model must be fine-tuned to tell them apart.

## NTK-aware scaling

The fast pairs never needed help, so a better rule keeps them and slows only the slow ones. NTK-aware scaling (`ntk`) does it by changing the base instead of the frequencies: $b' = b \cdot s^{r / (r - 2)}$. Then $\theta'_0 = 1 = \theta_0$, and for the last pair $i = r/2 - 1$, $\theta'_i = b'^{-(r-2)/r} = b^{-(r-2)/r} / s = \theta_i / s$: exactly the factor. Pairs between are slowed by between 1 and $s$.

Hugging Face's `dynamic` type applies this rule at inference with a factor that grows with the current length, $s_\text{eff} = s \cdot T / L_0 - (s - 1)$. At $T = 2 L_0$ and $s = 2$ that is 3, which is how the golden fixture checks `ntk`.

## YaRN

NTK still slows the middle pairs by a smoothly varying amount. YaRN (`yarn`) makes the bands explicit. Pair $i$ turns $L_0 / \lambda_i$ times, and the pair index that turns exactly $n$ times is

$$c(n) = \frac{r \ln\left(L_0 / (2\pi n)\right)}{2 \ln b}.$$

With $lo = \lfloor c(\beta_\text{fast}) \rfloor$ and $hi = \lceil c(\beta_\text{slow}) \rceil$ (clamped to $[0, r - 1]$; defaults $\beta_\text{fast} = 32$, $\beta_\text{slow} = 1$), the ramp $\rho_i = \operatorname{clamp}((i - lo) / (hi - lo), 0, 1)$ blends

$$\theta'_i = (1 - \rho_i)\, \theta_i + \rho_i\, \theta_i / s .$$

Pairs turning more than 32 times keep $\theta_i$, and pairs turning less than once get $\theta_i / s$. There is one more effect to correct. Stretching the slow pairs makes attention scores flatter (more entropy) at long range, so YaRN sharpens them: cos and sin are multiplied by $t = 0.1 \ln s + 1$, which multiplies every score by $t^2$.

## Llama 3 scaling

Llama 3.1 uses wavelength bands directly (`llama3`), with $L_0 = 8192$, low factor 1, and high factor 4. A pair with $\lambda_i < L_0 / 4$ keeps $\theta_i$, a pair with $\lambda_i > L_0$ gets $\theta_i / s$, and between them, with $\sigma = (L_0 / \lambda_i - 1) / (4 - 1)$,

$$\theta'_i = (1 - \sigma)\, \theta_i / s + \sigma\, \theta_i .$$

There is no temperature, and the frequencies stay strictly decreasing in $i$.

## ALiBi (optional)

This part is optional: ALiBi has no call site in the system, since every model the book loads uses RoPE. It is worth knowing because it shows a completely different answer to the same problem.

ALiBi drops rotations entirely. Each head $h$ adds a bias $-m_h \cdot (i - j)$ to the score of query position $i$ and key position $j$: the further back, the lower. The slopes are a geometric sequence ($2^{-8/n}, 2^{-16/n}, \dots$ for $n$ heads). With $T_q$ queries that are the last of $T_k$ keys, query $i$ sits at key index $T_k - T_q + i$. Softmax ignores a constant added to a whole row, so BLOOM's form $m_h \cdot j$ gives the same weights.

## Working it by hand

Each rule fits in a small example.

**Linear and NTK.** Take $r = 4$, $b = 10^4$, so $\theta = (1, 0.01)$. Linear with $s = 2$ gives $(0.5, 0.005)$. NTK with $s = 2$ gives $b' = 10^4 \cdot 2^{4/2} = 4 \cdot 10^4$ and $\theta' = (1, (4 \cdot 10^4)^{-1/2}) = (1, 0.005)$: the fast pair kept, the slow one halved.

**YaRN.** Take $r = 8$, $b = 10^4$, $s = 4$, $L_0 = 2048$, so $\theta = (1, 0.1, 0.01, 0.001)$. Then $L_0 / 2\pi = 325.95$; $c(32) = 8 \ln(10.186) / 18.421 = 1.008$, so $lo = 1$; and $c(1) = 8 \ln(325.95) / 18.421 = 2.513$, so $hi = 3$. The ramp is $\rho = (0, 0, 0.5, 1)$, which gives $\theta' = (1, 0.1, 0.5 \cdot 0.01 + 0.5 \cdot 0.0025, 0.00025) = (1, 0.1, 0.00625, 0.00025)$, and $t = 0.1 \ln 4 + 1 = 1.138629$.

**Llama 3.** Use the same $\theta$ with $s = 8$ and $L_0 = 1024$. The wavelengths are $(6.28, 62.8, 628.3, 6283)$, and the bands are below 256 (keep) and above 1024 (divide). Pairs 0 and 1 keep, pair 3 becomes $0.000125$, and pair 2 blends with $\sigma = (1024 / 628.3 - 1) / 3 = 0.209916$: $0.01 \cdot (0.790084 / 8 + 0.209916) = 0.003087$.

**ALiBi.** With 2 heads (slopes $1/16$, $1/256$) and 2 queries over 3 keys, the distances are $(1, 0, -1)$ and $(2, 1, 0)$, so head 0's biases are $(-0.0625, 0, 0.0625)$ and $(-0.125, -0.0625, 0)$.

## The interface

```python
def rope_inv_freq_scaled(d_rot, base, kind: Literal["default", "linear", "ntk", "yarn", "llama3"], factor=1.0,
                         original_max_pos=0, beta_fast=32, beta_slow=1, low_freq_factor=1, high_freq_factor=4
                         ) -> tuple[NDArray, float]: ...      # (inv_freq float64 [d_rot/2], attention_scaling)
def alibi_bias(n_heads: int, Tq: int, Tk: int) -> NDArray: ...  # float32 [n_heads, Tq, Tk]; optional
```

`rope_inv_freq_scaled` returns a new `inv_freq` and an attention scaling, both of which feed `rope_cos_sin` from the previous section unchanged. `alibi_bias` returns one bias matrix per head for the caller to add to the scores before the causal mask.

## Building it step by step

The damage is confined to pairs whose angles leave the trained range, so the fix belongs in the frequencies $\theta_i$, not in the rotation. The output is a new `inv_freq` (plus a temperature) that `rope_cos_sin` consumes unchanged. Every method starts from the default ladder $\theta_i = b^{-2i/r}$ and modifies it, so the solution computes `theta = rope_inv_freq(d_rot, base)` once and branches on `kind`.

Linear and NTK have closed forms: divide every $\theta_i$ by $s$, or recompute the ladder with base $b \cdot s^{r/(r-2)}$. Reusing `rope_inv_freq` for NTK means the "fastest kept, slowest divided by $s$" property follows from the algebra, not from extra code.

YaRN and Llama 3 need per-pair decisions. Vectorise them: compute a per-pair weight in $[0, 1]$ (YaRN's ramp $\rho_i$, Llama 3's $\sigma$) and blend $\theta_i$ with $\theta_i/s$. A clamp turns the blend into "keep" and "divide" at the two ends, so three bands need no branching. YaRN's band edges are pair indices derived from "how many turns over $L_0$". Inverting $L_0 / \lambda_i = n$ with $\lambda_i = 2\pi b^{2i/r}$ gives $c(n)$, and the edges are rounded **outward** (floor the low edge, ceil the high) so the blend region never shrinks.

The return type carries the temperature because it belongs to the same config field and must reach `rope_cos_sin`'s `attention_scaling`; it is 1 for every kind but YaRN. Finally, configs come from JSON written by people, so every parameter a kind needs is validated for that kind (an `original_max_pos` only matters to YaRN and Llama 3).

### The YaRN ramp

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

On the worked numbers, `corr(32)` is 1.008 and `corr(1)` is 2.513, so `lo = 1`, `hi = 3`, and the ramp over $i = 0..3$ is $(0, 0, 0.5, 1)$. The factor `2` in the denominator is the "2" in $b^{2i/r}$; dropping it moves both edges by a factor of two.

### The dispatcher

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
    theta = rope_inv_freq(d_rot, base)           # the trained ladder
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

### YaRN

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

The blend is written with Hugging Face's `keep` so the expression matches `_compute_yarn_parameters` term for term; it is $(1 - \rho_i)\theta_i + \rho_i \theta_i / s$. On the example, pair 2 gets $0.5 \cdot 0.0025 + 0.5 \cdot 0.01 = 0.00625$ and $t = 1.138629$.

### Llama 3

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

In the medium band `inv` is still `theta`, so `smoothed` is $(1 - \sigma)\theta_i/s + \sigma\theta_i$, the Llama 3 formula. Pair 2 of the example has wavelength 628.3, between 256 and 1024, so $\sigma = 0.209916$ and $\theta'_2 = 0.003087$. Because $\sigma$ grows with frequency, the blended values sit between their neighbours and the ladder stays strictly decreasing.

### ALiBi (optional)

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

The offset `Tk - Tq` is what makes a one-row decoding bias equal the last row of the prefill bias. The full solution is in [`ctxext.py`](../../../../course/solutions/python/tinyllm/modern/ctxext.py).

### Why this design

Everything here is $O(r)$ work, done once when a model is loaded, so precision rather than speed is what matters: the ladder is float64 and is cast to float32 only inside `rope_cos_sin`.

Linear interpolation could equally be written as "divide positions by $s$". Changing the frequencies instead keeps one code path for every method and lets the decoder in [KV cache and generate](../p08-inference/02-kv-cache-and-generate.md) pass raw integer positions. The solution also matches Hugging Face's formulas exactly. Several algebraically equal ways exist to write the YaRN blend or the Llama 3 bands; following Hugging Face's term order and rounding (`floor`/`ceil` of the correction range, the `+ 0.001` guard) makes the golden values agree to float64 precision. A "cleaner" formula that rounds the range inward moves a pair into the wrong band.

Finally, the scaling is static, not dynamic. Hugging Face's `dynamic` type recomputes the NTK base per length, which changes the frequencies under an existing KV cache, so keys cached at the old frequencies no longer match. The solution's `ntk` takes one fixed factor, so the table never changes while a model is decoding.

## What can go wrong

The checks mirror the worked examples and the design. The hand examples for linear, NTK, default, YaRN (ramp and temperature), and Llama 3 (all three bands) must come out exactly, and ten golden cases from Hugging Face's `ROPE_INIT_FUNCTIONS`, covering every kind, must match, since `rope_scaling` configs depend on them. Property tests check the reason each method works: linear interpolation maps the angles at $s p$ onto the trained angles at $p$; NTK keeps the fastest pair, divides the slowest by $s$, and is monotone between; YaRN keeps fast pairs, divides slow ones, applies the temperature, and reduces to the default at $s = 1$; and the Llama 3 bands keep frequencies strictly decreasing so no pairs are reordered. Validation must reject an unknown kind, an odd width, a bad base, a factor below 1, a missing $L_0$, and swapped band factors. For the optional ALiBi part, the hand example must match, the softmax weights must equal BLOOM's after the causal mask, a decoding row must equal the last row of prefill, and more queries than keys or zero sizes must raise.

[`test_ctxext.py`](../../../../course/solutions/learner-tests/L7.4/python/tests/l7-4-ctxext/test_ctxext.py) writes the rules out pair by pair in plain Python floats and compares `rope_inv_freq_scaled` against them for linear, NTK, YaRN at several sizes, and Llama 3, checks YaRN's temperature, and checks that bad configs raise. A loop over pairs in scalar floats is a good oracle for vectorised numpy code: it is slow and obvious, so a mistake in a mask or a broadcast in the solution has nowhere to hide.

> **Warning:** YaRN's ramp is easy to apply backwards. Ramp 0 at low indices (the fast pairs) means keep; if you invert it, fast pairs are interpolated and slow ones extrapolated, and local order blurs. The solution writes `keep = 1 - ramp` to make the direction explicit.

The pitfalls and how the solution avoids each:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| multiplying by the factor in position interpolation | the context shrinks instead of growing | `theta / factor` in the `linear` branch |
| scaling the base by $s$, not $s^{r/(r-2)}$ | the slowest pair is not slowed by exactly $s$ | `base * factor ** (d_rot / (d_rot - 2))` |
| YaRN's ramp backwards | fast pairs are interpolated, slow ones extrapolated: local order blurs | ramp 0 at low indices (fast pairs) means keep: `keep = 1 - ramp` |
| forgetting YaRN's temperature | long-range attention too flat; perplexity a little worse everywhere | returns `0.1 * log(factor) + 1` as the second value |
| swapping Llama 3's low and high bands | the wrong pairs are slowed | long wavelengths compare against `L0 / low_freq_factor`, short against `L0 / high_freq_factor`; swapped factors raise |
| rounding the correction range inward | one pair lands in the wrong band | `floor` for `lo`, `ceil` for `hi` |
| a missing 2 in the correction dimension | the bands move by a factor of two | `corr` divides by `2 * math.log(base)` |
| ALiBi bias growing with distance | far tokens dominate (optional part) | the bias is `-slope * dist` |
| ALiBi queries treated as the first keys | decoding rows wrong (optional part) | query index offset by `Tk - Tq` |

## Where this leads

In [The Llama-Family Model: Config, Weight Loading, and Download](09-llama-family-model.md), `LlamaConfig.rope_scaling` becomes a `RopeSpec` through `rope_inv_freq_scaled`, on the way to SmolLM2-135M logits that match Hugging Face. The [TinyStories capstone](../capstones/01-tinystories.md) runs the long-context evaluation: bits per byte at twice the training context, with and without `rope_scaling = yarn`.

### Further reading

- Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (2023).
- bloc97, "NTK-Aware Scaled RoPE" (2023).
- Peng et al., ["YaRN: Efficient Context Window Extension of Large Language Models"](https://arxiv.org/abs/2309.00071) (2023), sections 3.1 to 3.4.
- Meta, "The Llama 3 Herd of Models" (2024).
- Press et al., "Train Short, Test Long: Attention with Linear Biases" (2022).
- Hugging Face `ROPE_INIT_FUNCTIONS` in `transformers/modeling_rope_utils.py`: adds `dynamic` (recomputed per length), `longrope` (per-pair learned factors), and DeepSeek's `mscale`.
- vLLM `YaRNScalingRotaryEmbedding` in `rotary_embedding.py`: precomputed scaled tables with the temperature folded in.
- Ding et al. 2024, LongRoPE: searched per-pair factors; most of the gain still comes from continued training at the longer length.
- BLOOM and MPT fuse the ALiBi bias into attention kernels; see FlashAttention's `alibi_slopes` argument.

## Key terms

- **Wavelength** ($\lambda_i$): positions per full turn of RoPE pair $i$, $2\pi / \theta_i$.
- **Position interpolation**: dividing every frequency by the factor so long positions reuse trained angles.
- **NTK-aware scaling**: raising the base by $s^{r/(r-2)}$ so the fastest pair is kept and the slowest is divided by $s$.
- **YaRN**: a per-pair ramp between keeping and dividing frequencies, by turns over the trained context, plus a temperature $0.1 \ln s + 1$.
- **Llama 3 scaling**: wavelength bands with a smooth blend between keep and divide, no temperature.
- **ALiBi**: a per-head bias that falls linearly with query-key distance, used instead of rotations.

## Summary

- Every RoPE extension method changes only the frequencies (and sometimes a temperature); the rotation code is unchanged.
- Position interpolation slows every pair by the factor, so long positions reuse trained angles exactly.
- NTK-aware scaling raises the base so the fastest pair is untouched and the slowest is slowed by exactly the factor.
- YaRN and Llama 3 split the pairs into bands by how often they turn over the original context: keep the fast ones, slow the slow ones, blend between; YaRN also sharpens attention by $0.1 \ln s + 1$.
- ALiBi (optional) uses no rotation at all: a bias that falls linearly with distance, one slope per head.

## Review questions

1. Why does NTK raise the base by $s^{r/(r-2)}$ and not by $s$?
   <details><summary>Answer</summary>The last pair has exponent $-(r-2)/r$. For it to be divided by exactly $s$ we need $b'^{-(r-2)/r} = b^{-(r-2)/r}/s$, so $b' = b \cdot s^{r/(r-2)}$. Pair 0 has exponent 0 and is unchanged whatever the base.</details>
2. Why is YaRN's band defined by how many turns a pair makes over $L_0$?
   <details><summary>Answer</summary>A pair that turns many times over the trained context has seen every angle, so extrapolating it is safe and keeps fine local order. A pair that turns less than once has not, so it must be interpolated. Turns per context is exactly the quantity that separates these.</details>
3. Why must YaRN multiply cos and sin by $t = 0.1 \ln s + 1$?
   <details><summary>Answer</summary>Slowing the slow pairs makes far keys look more alike, flattening (raising the entropy of) attention at long range. Scaling $q$ and $k$ by $t$ multiplies every logit by $t^2$, sharpening the softmax back. Without it perplexity is slightly worse everywhere.</details>
4. What goes wrong if `_yarn_ramp` rounds `lo` up and `hi` down?
   <details><summary>Answer</summary>The blend region shrinks by up to a pair at each end, so a pair Hugging Face blends is instead kept or fully divided. The frequencies then disagree with the checkpoint's training and with Hugging Face's golden values.</details>
5. Why does `alibi_bias` add `Tk - Tq` to the query index?
   <details><summary>Answer</summary>During decoding the $T_q$ new queries are the last positions of a $T_k$-long sequence. Without the offset, query 0 would be treated as key position 0, and the bias would not match the prefill row it is supposed to reproduce.</details>
6. Why does the solution avoid Hugging Face's `dynamic` scaling?
   <details><summary>Answer</summary>`dynamic` recomputes the NTK base as the sequence grows, which changes the frequencies under an existing KV cache; keys cached at the old frequencies no longer match the new queries. A fixed factor keeps the table constant while decoding.</details>
