# FP8 and microscaling formats

Float32 weights and activations move four bytes per value, and for inference that memory traffic is most of the cost. FP8 stores each value in one byte. The microscaling (MX) formats go further: they group values into blocks that share one scale, and with FP4 elements they store four bits per value. These formats reduce memory traffic dramatically, but every conversion into them has edge cases: ties, tiny magnitudes, signed zero, infinities, NaNs, and values beyond the finite range. Get any of those wrong and the quantized model disagrees with every other implementation by one code in a way that is very hard to trace.

This section writes the conversions directly from the format definitions and checks the resulting bit patterns exactly. It reuses the bit layouts and power-of-two reasoning of [IEEE 754 anatomy, ulps, and round to nearest even](01-ieee-754.md), and it assumes you have rounded values by hand in [Problem set A](08-problem-set-a.md). The key idea is that a single rounding routine, parameterized by fraction bits and bias, serves every format; the edge rules are layered on top of it, and the block scale is chosen so that it never adds a rounding of its own.

The reference implementation is [`lowp.py`](../../../../course/solutions/python/tinyllm/num/lowp.py): FP8 E4M3 and E5M2 conversion (`f32_to_fp8_bits`, `fp8_bits_to_f32`, `quantize_fp8`, `dequantize_fp8`, `fp8_scale`), FP4 E2M1 values (`f32_to_e2m1_bits`, `e2m1_bits_to_f32`), block scales (`e8m0_scale_code`, `e8m0_to_f32`), and MX quantize and dequantize (`mx_quantize`, `mx_dequantize`). Its interface is in [`lowp.pyi`](../../../../course/contracts/py/tinyllm/num/lowp.pyi). An optional standalone C version, [Low-precision conversions in C](07-low-precision-c-conversions.md), produces the same bit patterns and does not load this Python code.

> **Learning objectives**
> - Describe the layouts of E4M3, E5M2, E2M1, and E8M0, including their largest finite values and special codes.
> - Round any value into an FP8 or FP4 format with round to nearest, ties to even, using one generic routine.
> - Apply saturation, infinity, NaN, and signed-zero rules without disturbing the rounding path.
> - Choose an MX block scale from the block maximum and explain why it adds no rounding.

## The formats

An IEEE-like binary format uses a sign bit, an exponent field, and a fraction field. For a normalized value with fraction width $m$ and bias $b$, the decoded value is

$$(-1)^s (1 + f/2^m)\, 2^{e-b}.$$

Subnormals omit the implicit leading one and use the minimum exponent. E4M3 uses four exponent bits and three fraction bits, giving more precision near one than E5M2. E5M2 uses two fraction bits and therefore covers a wider range. The project defines saturation and special-value behavior explicitly, so callers do not depend on a library default. The four formats compare as follows:

| Format | Exponent bits, bias | Fraction bits | Max finite | Specials |
|---|---|---|---|---|
| E4M3 (E4M3FN) | 4, 7 | 3 | 448 (`0x7E`) | no infinity; `0x7F` is NaN; overflow and $\pm\infty$ saturate |
| E5M2 | 5, 15 | 2 | 57344 (`0x7B`) | `0x7C` is $+\infty$, `0x7D` to `0x7F` NaN; finite overflow saturates |
| FP4 E2M1 | 2, 1 | 1 | 6 (`0x7`) | none: values $0, 0.5, 1, 1.5, 2, 3, 4, 6$ |
| E8M0 (scale) | 8, 127 | 0 | $2^{127}$ | `0xFF` is NaN; no zero, no sign |

Rounding is round to nearest, ties to even. A tie at `2.5` selects `2`; a tie at `3.5` selects `4`. FP8 encoding and decoding use exact powers of two to map between values and codes. Microscaling formats apply one E8M0 power-of-two scale to a fixed-size block, then encode each scaled value in FP8 or E2M1.

## Working it by hand

Start with a scalar tie. In E4M3, `1.0625` lies halfway between the adjacent representable values `1.0` and `1.125`. Their final significand bits are even and odd respectively, so ties to even encodes `1.0` (code `0x38`). A value just above the midpoint, such as `1.07`, encodes `1.125` (`0x39`). The next midpoint, `1.1875`, lies between `1.125` (odd) and `1.25` (even) and so rounds *up* to `0x3A`: ties to even is not "ties down".

An MX block needs one more step. First find the largest absolute element. Choose the shared power-of-two scale that moves that maximum into the element format's top binade, then divide each value by the scale and round once into the element format. Decoding multiplies each value by the same scale.

Take the FP4 block $[0.3, -1.7, 5.0, 0.04]$. The largest magnitude is 5, $\lfloor\log_2 5\rfloor = 2$, and FP4's largest normal is $6 = 1.5 \cdot 2^2$, so $e_{\max} = 2$ and the scale is $X = 2^{2 - 2} = 1$ (E8M0 code $0 + 127 = 127$). Each element then rounds to the FP4 grid $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$: $0.3 \to 0.5$, $-1.7 \to -1.5$, $5.0$ is a tie between 4 (code 6, even) and 6 (code 7, odd) and goes to 4, and $0.04 \to 0$. The codes are `[1, 11, 6, 0]` (11 is code 3 with the sign bit `0x8`), the scale is `[127]`, and the decoded block is $[0.5, -1.5, 4, 0]$. Notice that the block maximum itself lost precision; that cost comes up again when we discuss the scale rule.

## Building it step by step

The solution rests on seven observations, each of which becomes a few lines of code.

A format is a grid of quanta per binade. In the binade $[2^E, 2^{E+1})$ a format with $M$ fraction bits holds the multiples of $2^{E - M}$. Below the smallest normal, $2^{1-B}$, the spacing stops shrinking: subnormals are multiples of $2^{1 - B - M}$. So for a magnitude $a$ the quantum exponent is $q_e = \max(\lfloor\log_2 a\rfloor, 1 - B) - M$.

Rounding is then one integer rounding. $a / 2^{q_e}$ is exact in float64 (a power-of-two scale), and rounding it to an integer $n$ with ties to even is exactly IEEE rounding into the format. One routine serves E4M3, E5M2, and E2M1; only $(M, B)$ differ.

The code falls out of $n$ and $q_e$. The biased exponent field is $q_e + M + B - 1$ (one less than the field, because $n$ already includes the implicit bit), and the code is $((q_e + M + B - 1) \ll M) + n$. When rounding carries $n$ from $2^{M+1} - 1$ to $2^{M+1}$, the addition carries into the exponent field by itself; for subnormals the same formula gives field 0 and $n < 2^M$, or field 1 when a subnormal rounds up to the smallest normal.

```python
# name: (fraction bits M, bias B, largest finite magnitude code)
_FMT = {"e4m3": (3, 7, 0x7E), "e5m2": (2, 15, 0x7B), "fp4_e2m1": (1, 1, 0x7)}
_MAX = {"e4m3": 448.0, "e5m2": 57344.0, "fp4_e2m1": 6.0}
_EMAX = {"fp4_e2m1": 2, "fp8_e4m3": 8}      # 6 = 1.5 * 2^2, 448 = 1.75 * 2^8


def _round_codes(a: NDArray, fmt: str) -> NDArray:
    """int64 magnitude codes of the finite float64 magnitudes a >= 0, rounded
    to nearest even, before any overflow handling (may exceed the max code)."""
    M, B, _ = _FMT[fmt]
    _, e = np.frexp(a)  # a = f * 2^e with f in [0.5, 1): floor(log2 a) = e - 1
    qe = np.maximum(e.astype(np.int64) - 1, 1 - B) - M   # the quantum exponent
    n = np.rint(np.ldexp(a, -qe)).astype(np.int64)       # exact scale, then ties to even
    code = ((qe + M + B - 1) << M) + n                   # carries into the exponent
    return np.where(n == 0, 0, code)                     # underflow to zero is code 0
```

This is the whole of the rounding logic for three formats. `frexp` gives $\lfloor\log_2 a\rfloor$ exactly, with no `log2` rounding near powers of two; `ldexp` scales without rounding; numpy's `rint` rounds half to even. Trace $1.1875$ in E4M3: $e = 1$, so $q_e = \max(0, -6) - 3 = -3$; $1.1875 \cdot 8 = 9.5$ rounds to 10; the code is $((-3 + 3 + 7 - 1) \ll 3) + 10 = 48 + 10 = 58 = $ `0x3A`. For $a = 0$, `frexp` gives $e = 0$, $n = 0$, and the `where` returns code 0.

The edge rules are kept separate from rounding. Saturation, infinity, NaN, and the sign are applied after the magnitude code is known, as a few `where`s, which keeps the rounding path free of special cases. Decoding is exact: a code splits into exponent field and mantissa; field 0 is $m \cdot 2^{1 - B - M}$, otherwise $(2^M + m)\,2^{e - B - M}$, and both are exact in float32 for every code.

```python
def _decode(mag: NDArray, fmt: str) -> NDArray:
    """float64 values of magnitude codes (no specials)."""
    M, B, _ = _FMT[fmt]
    mag = mag.astype(np.int64)
    ef, mant = mag >> M, mag & ((1 << M) - 1)
    sub = np.ldexp(mant.astype(np.float64), 1 - B - M)                       # no implicit 1
    nor = np.ldexp(((1 << M) + mant).astype(np.float64), (ef - B - M).astype(np.int64))
    return np.where(ef == 0, sub, nor)


def _encode_fp8(v: NDArray, fmt: str) -> NDArray:
    """uint8 codes of float64 values v (any shape), with the edge rules."""
    _check_fp8(fmt)
    v = np.asarray(v, dtype=np.float64)
    nan, inf = np.isnan(v), np.isinf(v)
    a = np.where(nan | inf, 0.0, np.abs(v))             # keep specials out of the rounding
    code = np.minimum(_round_codes(a, fmt), _FMT[fmt][2])  # saturate finite overflow
    if fmt == "e4m3":
        code = np.where(inf, 0x7E, code)  # no infinity: saturate
    else:
        code = np.where(inf, 0x7C, code)  # e5m2 keeps IEEE infinity
    sign = np.where(np.signbit(v), 0x80, 0)              # signbit, so -0.0 keeps its sign
    return np.where(nan, 0x7F, code | sign).astype(np.uint8)
```

The order of operations is the design. Specials are masked to 0 before rounding, so `frexp` never sees them. The rounded magnitude is clamped to the largest finite code (`0x7E` or `0x7B`), so overflow can never produce E4M3's NaN code `0x7F` or E5M2's infinity by accident. Only then are infinities, sign, and NaN set. `np.signbit` is used rather than `v < 0` because $-0.0 < 0$ is false and the sign of zero must survive.

Scaling must not round twice. `quantize_fp8` divides by the scale in float64 and rounds once to FP8; a float32 intermediate would round first and could land on the other side of a midpoint.

```python
def fp8_bits_to_f32(codes: ArrayLike, fmt: str) -> NDArray:
    # ... fmt check; codes must lie in 0..255 ...
    c = c.astype(np.int64)
    mag = c & 0x7F
    v = _decode(mag, fmt)
    if fmt == "e4m3":
        v = np.where(mag == 0x7F, np.nan, v)                  # the only NaN magnitude
    else:
        v = np.where(mag == 0x7C, np.inf, np.where(mag > 0x7C, np.nan, v))
    return np.where(c & 0x80, -v, v).astype(np.float32)      # -0.0 for 0x80


def quantize_fp8(x: ArrayLike, fmt: str, scale: float) -> NDArray:
    _check_scale(scale)
    return _encode_fp8(np.asarray(x, dtype=np.float64) / scale, fmt)   # one rounding


def dequantize_fp8(q: ArrayLike, fmt: str, scale: float) -> NDArray:
    _check_scale(scale)
    return (fp8_bits_to_f32(q, fmt).astype(np.float64) * scale).astype(np.float32)
```

`f32_to_fp8_bits` is `_encode_fp8` of the input converted to float32 first (the contract says the input *is* a float32), while `quantize_fp8` forms $x / \text{scale}$ in float64 and rounds once. `fp8_scale(amax, fmt)` is `amax / fp8_max(fmt)`, or 1 for an all-zero tensor.

The last observation concerns MX. The MX scale is a power of two chosen from the block maximum: $X = 2^{\lfloor\log_2 \text{amax}\rfloor - e_{\max}}$ puts the largest element in the element format's top binade. Because $X$ is a power of two, $x / X$ is exact, so the element rounding is the only rounding.

```python
def e8m0_scale_code(amax: float, elem: str) -> int:
    _check_elem(elem)
    # ... amax must be finite and >= 0 ...
    a = 1.0 if amax == 0 else amax            # an all-zero block gets scale 2^-emax
    e = math.frexp(a)[1] - 1  # floor(log2 a), exact
    return int(min(max(e - _EMAX[elem] + 127, 0), 254))   # 255 is NaN, never produced


def mx_quantize(x, block=32, elem="fp4_e2m1") -> tuple[NDArray, NDArray]:
    fmt = _check_elem(elem)
    v = np.asarray(x, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError("mx_quantize needs finite values")
    blk = _blocks(v, block)                    # [..., n / block, block]; raises if not divisible
    amax = np.max(np.abs(blk), axis=-1)        # the maximum, not an average
    scales = np.vectorize(lambda a: e8m0_scale_code(float(a), elem), otypes=[np.int64])(amax)
    q = np.ldexp(blk, (127 - scales)[..., None])  # x / X, exact: X is a power of two
    if fmt == "fp4_e2m1":
        codes = f32_to_e2m1_bits(q)            # rounds once, saturates at 6
    else:
        codes = _encode_fp8(q, "e4m3")
    return codes.reshape(v.shape), scales.astype(np.uint8)
```

`mx_dequantize` reverses it: view the codes as blocks, check that the scales have one entry per block, decode the elements, and multiply by `e8m0_to_f32(scales)` in float64 before one rounding to float32. In the hand example, `amax = 5`, `frexp(5)` gives exponent 3, so $e = 2$ and the code is $2 - 2 + 127 = 127$; `ldexp(blk, 0)` leaves the block unchanged, and `f32_to_e2m1_bits` produces `[1, 11, 6, 0]`. The full solution is in [`lowp.py`](../../../../course/solutions/python/tinyllm/num/lowp.py).

### Why this design

The solution works on values, not bits. Its C twin works on float32 bit fields; this file works on float64 values, where scaling by $2^{-q_e}$ and `rint` are exact. That makes one generic routine correct for three formats, vectorized over any array shape, at a cost of $O(n)$ with a few array passes.

It prefers generic rounding to a lookup table. For 256 codes, a sorted table of decoded values and a nearest search would also work, but tie handling needs care at every midpoint and the table must be rebuilt per format. The arithmetic route makes ties to even a property of `rint`.

It saturates rather than overflowing to infinity. E4M3 has no infinity, and for quantized weights an outlier that becomes NaN or infinity poisons every dot product it enters; clamping to the largest finite value bounds the damage. E5M2 keeps IEEE infinity for inputs that already are infinite.

And it follows the OCP MX scale rule. $\lfloor\log_2\text{amax}\rfloor - e_{\max}$ is what the OCP Microscaling spec defines, so codes interoperate. The rule has an accepted cost: the block maximum can land above the element format's largest value (amax $= 7$ in FP4 gives $X = 1$, and 7 saturates to 6), and, as in the hand example, the maximum itself can lose precision. A scale that always fits the maximum would need a non-power-of-two scale or one more exponent step, losing a bit of resolution for every other element.

## What can go wrong

The tests pin the scalar tie and the MX block of the hand examples, then sweep the whole code space: every 8-bit code is decoded and known values are encoded against stable expected bit patterns. Saturation and special values define behavior for overflow, infinities, and NaNs. Tiny values and signed zero protect underflow and sign preservation at zero. Encoding must invert decoding, so every finite code is stable through decode then encode. Rounding must be nearest and monotone, with ties to even, which catches incorrect midpoint handling and non-monotone encoders. Scaled quantization must round-trip and round once, so that scaling introduces no extra rounding step. Finally, FP4 values and ties, E8M0 scale codes, the MX block law, and MX shapes and errors check element codes, scale choices, block layout, and input validation.

The mistakes that those tests catch:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Rounding half away from zero | Exact midpoints choose the wrong code. | `np.rint` rounds half to even on the scaled integer |
| Treating subnormals like normalized values | Tiny values jump to zero or the minimum normal. | `qe` is floored at $1 - B - M$ and `_decode` drops the implicit 1 for field 0 |
| Producing infinity on finite overflow | Codes disagree with the saturation rule. | `np.minimum(..., max_code)` before the special rules |
| Applying the block scale after rounding | Quantized values differ by one code near boundaries. | divides by the scale first (exact `ldexp` for MX, float64 for FP8), then rounds once |
| Choosing scale from an average instead of the maximum magnitude | A large block member clips unexpectedly. | `amax = np.max(np.abs(blk), axis=-1)` |
| Accepting malformed block shapes | Values are silently dropped or grouped incorrectly. | `_blocks` raises unless the last axis is a multiple of `block`; `mx_dequantize` checks the scale shape |

## Where this leads

In *Build Your Own LLM System*, [Quantization: int8, int4 groups, fp8, and the KV cache](../../build-an-llm/09-inference/05-quantization.md) quantizes model weights for lower-bandwidth inference with these conversions, and [Candle model runner (Llama + bigram, int4), Rust KV cache, sampler and PCG32](../../build-an-llm/11-serving/02-model-runner-and-sampler.md) converts quantized model weights in the Rust engine. In *Software Craftsmanship*, [Interface migration: KV format v1 to v2](../../software-craftsmanship/07-maintenance/01-kv-format-v1-to-v2.md) stores fp8 KV values and relies on the saturation and decoding rules defined here. In parallel, [Low-precision conversions in C](07-low-precision-c-conversions.md) implements selected conversions in standalone C with the same bit-exact results.

In production, FP8 E4M3 and E5M2 are accelerator matrix formats that trade exponent range against significand precision, and the block-scaled MX formats let compact values share a power-of-two scale per block.

### Further reading

- OCP Microscaling Formats (MX) v1.0 specification.
- Micikevicius et al., "FP8 Formats for Deep Learning" (2022).
- IEEE 754 and vendor format guides for FP8 matrix instructions.

## Key terms

- **E4M3**: an 8-bit float with 4 exponent bits (bias 7) and 3 fraction bits, no infinity, largest finite value 448.
- **E5M2**: an 8-bit float with 5 exponent bits (bias 15) and 2 fraction bits, IEEE-style infinities and NaNs, largest finite value 57344.
- **E2M1 (FP4)**: a 4-bit float whose only magnitudes are $0, 0.5, 1, 1.5, 2, 3, 4, 6$.
- **E8M0**: an 8-bit unsigned power-of-two scale with bias 127, used as the shared scale of an MX block.
- **Saturation**: mapping finite overflow to the largest finite code instead of infinity or NaN.
- **Microscaling (MX)**: a block format in which a fixed number of low-precision elements share one power-of-two scale.

## Summary

- Each FP8 or FP4 format is a grid fixed by its fraction bits $M$ and bias $B$; the quantum exponent is $q_e = \max(\lfloor\log_2 a\rfloor, 1 - B) - M$.
- Dividing by $2^{q_e}$ is exact in float64, so `rint` (ties to even) is the whole rounding step, and the code formula carries into the exponent on its own.
- Special values, saturation, and the sign are applied after rounding, in a fixed order that prevents overflow from producing NaN or infinity codes by accident.
- Scaling must happen before the single rounding, in float64 for FP8 and as an exact power of two for MX.
- The MX scale $2^{\lfloor\log_2\text{amax}\rfloor - e_{\max}}$ follows the OCP spec, at the accepted cost that the block maximum can saturate or lose precision.

## Review questions

1. Why does one function, `_round_codes`, round correctly into E4M3, E5M2, and E2M1?

   <details><summary>Answer</summary>

   Each format is determined by $(M, B)$: the quantum in the binade of $a$ is $2^{q_e}$ with $q_e = \max(\lfloor\log_2 a\rfloor, 1 - B) - M$. Dividing by a power of two is exact in float64 and `rint` rounds to the nearest integer with ties to even, which is IEEE rounding onto that grid. The code formula then carries into the exponent field when the rounded mantissa overflows, so no format needs its own branch.

   </details>

2. Why is the rounded code clamped to `0x7E` before the NaN and infinity rules are applied in E4M3?

   <details><summary>Answer</summary>

   A finite value above 448 rounds to a code of `0x7F` or more, and `0x7F` is E4M3's NaN. Clamping first turns any finite overflow into 448, the contract's saturation rule; the later `where`s then set infinities to `0x7E` and NaN to `0x7F` explicitly, so NaN can only come from a NaN input.

   </details>

3. What goes wrong if `quantize_fp8` converts `x / scale` to float32 before encoding?

   <details><summary>Answer</summary>

   It rounds twice. The float32 rounding can move a value that is just above an FP8 midpoint onto the midpoint itself (or across it), and the second rounding then picks the even neighbour instead of the nearer one. Forming the quotient in float64 and rounding once into FP8 gives the correctly rounded code.

   </details>

4. Why is the MX scale a power of two, and why chosen from the block maximum?

   <details><summary>Answer</summary>

   A power of two makes $x / X$ exact, so the only error is the element rounding, and E8M0 can store it in 8 bits. The maximum fixes the range: choosing from an average lets the largest element exceed the element format and saturate, which on weights is the outlier that matters most.

   </details>

5. Why does `_encode_fp8` use `np.signbit(v)` rather than `v < 0` to set the sign bit?

   <details><summary>Answer</summary>

   $-0.0 < 0$ is false, so a comparison would encode negative zero as positive zero (`0x00`). `np.signbit` reads the sign bit itself, so $-0.0$ becomes `0x80` and decodes back to $-0.0$.

   </details>
