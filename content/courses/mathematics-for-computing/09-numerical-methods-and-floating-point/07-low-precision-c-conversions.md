# Low-precision conversions in C

[FP8 and microscaling formats](04-fp8-and-microscaling-formats.md) described, in Python, how float32 values map to compact formats. The optional C kernels of *Build Your Own LLM System*, [Paged attention for decode in C](../../build-an-llm/p09-kernels/07-paged-attention-decode.md) and [Fused int4 and int8 dequantize-matmul in C](../../build-an-llm/p09-kernels/08-fused-quantized-matmul.md), need the same conversions when they read half-precision cache values and quantization scales. They cannot call Python, and they must not drift from it: if the C decoder disagrees with the Python encoder on even one code, a cached value silently changes.

This optional section writes those conversions in standalone C: ordinary function calls inside one process, with bit patterns that agree with the Python version on every code. The heart of it is one integer rounding routine shared by four formats (E4M3, E5M2, bfloat16, and binary16), one decoder, and a thin per-format wrapper for the special values. Because the arithmetic is on integers, it is exact, sanitizer-clean, and identical on every platform.

You should read [FP8 and microscaling formats](04-fp8-and-microscaling-formats.md) first, since this section reuses its value and rounding rules and its code formula. The solution, [`lowp.c`](../../../../course/solutions/c/src/numerics/lowp.c) (`tl_f32_to_e4m3`, `tl_e4m3_to_f32`, `tl_f32_to_e5m2`, `tl_e5m2_to_f32`, `tl_f32_to_bf16`, `tl_bf16_to_f32`, `tl_f32_to_f16`, `tl_f16_to_f32`, and the shared helpers `rne_code` and `decode_mag`), is built under sanitizers and does not call Python. The interface is in [`numerics.h`](../../../../course/contracts/c/include/tinyllm/numerics.h).

> **Learning objectives**
> - Express round to nearest, ties to even as a shift and compare on a float32's integer significand.
> - Compute the destination quantum $q_e$ for any of the four formats, including the subnormal range.
> - Apply per-format special-value and overflow policies around a shared rounding core.
> - Explain why every finite code decodes and re-encodes to the same bits.

## The rounding rule in integers

These symbols describe a destination format:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $M$ | Number of retained fraction bits | integer: 3 for E4M3, 2 for E5M2, 7 for bfloat16, 10 for binary16 |
| $B$ | Exponent bias | integer: 7 for E4M3, 15 for E5M2 and binary16, 127 for bfloat16 |
| $e_{min}$ | Smallest normal exponent, $1-B$ | integer |
| $q_e$ | Exponent of the spacing between adjacent representable values | integer: $\max(\lfloor\log_2 v\rfloor,e_{min})-M$ |

A positive normal float32 is $v = s\, 2^{e_2}$ with an integer significand $s$ of at most 24 bits. A destination format with $M$ fraction bits and bias $B$ has smallest normal exponent $e_{min}=1-B$. In the binade containing $v$, adjacent destination values are separated by $2^{q_e}$, where

$$q_e = \max(\lfloor \log_2 v \rfloor, e_{min}) - M.$$

The conversion rounds $v / 2^{q_e}$ to an integer using round to nearest, ties to even. If the discarded bits are greater than half a unit, increment the retained integer. If they equal half a unit, increment only when the retained integer is odd. A carry can increment the exponent. Values below the smallest normal use the fixed subnormal quantum $2^{e_{min}-M}$.

Special values are format rules applied around that rounding core. Here E4M3 saturates finite magnitudes above 448. E5M2 uses the all-ones exponent for infinity and NaN. Signed zero preserves its sign bit. bfloat16 rounds the high half of float32 with the low half deciding ties. Binary16 uses 5 exponent bits and 10 fraction bits.

## Working it by hand

For E4M3, $M=3$, $B=7$, and $e_{min}=-6$. The float32 value 0.3 lies between $2^{-2}=0.25$ and $2^{-1}=0.5$, so its binade exponent is $-2$. Thus $q_e=-2-3=-5$ and one destination step is $2^{-5}=0.03125$. Dividing gives $0.3 / 0.03125 = 9.6$, which rounds to 10. The retained significand is 10, or binary `1010`. The low 3 bits are the fraction, `010`; the carry into the exponent makes the final code `0x2A`, whose decoded value is 0.3125.

The solution does the same computation in integers. `0.3f` has bits `0x3E99999A`, exponent field 125 and fraction `0x19999A`, so $s = $ `0x99999A` (with the implicit bit) and $e_2 = 125 - 150 = -25$. $s$ has 24 bits, so $\lfloor\log_2 v\rfloor = -25 + 23 = -2$ and $q_e = -5$. The shift is $q_e - e_2 = 20$: $n = s \gg 20 = 9$, and the 20 discarded bits are `0x9999A`, more than half (`0x80000`), so $n = 10$. The code is $((q_e + M + B - 1) \ll M) + n = (4 \ll 3) + 10 = 42 = $ `0x2A`.

## Designing the API

```c
/* tinyllm/numerics.h, low-precision section */
uint8_t  tl_f32_to_e4m3(float x);    float tl_e4m3_to_f32(uint8_t b);
uint8_t  tl_f32_to_e5m2(float x);    float tl_e5m2_to_f32(uint8_t b);
uint16_t tl_f32_to_bf16(float x);    float tl_bf16_to_f32(uint16_t b);
uint16_t tl_f32_to_f16(float x);     float tl_f16_to_f32(uint16_t b);
```

Encoders receive one `float` and return the exact stored bit pattern in an unsigned integer. Decoders receive that integer and return a float, exactly, for every code. No function allocates memory, retains a pointer, or touches the error slot. The header fixes the edge rules: E4M3 saturates $\lvert x\rvert > 448$ and $\pm\infty$ to $\pm 448$ and gives `0x7F` for NaN; E5M2 saturates finite overflow to $\pm 57344$, keeps $\pm\infty$, and gives `0x7F` for NaN; bfloat16 and binary16 overflow to $\pm\infty$ and give the quiet NaNs `0x7FC0` and `0x7E00`.

## Building it step by step

The solution follows six steps. Work on the integer significand: a finite float32 magnitude is exactly $s \cdot 2^{e_2}$ with $s < 2^{24}$, where $s = $ `frac | 0x800000` and $e_2 = \text{exp} - 150$ for normals, and $s = $ `frac` and $e_2 = -149$ for float32 subnormals; everything after this is integer arithmetic, which is exact. Find the destination quantum: $\lfloor\log_2 v\rfloor$ is $e_2$ plus the bit length of $s$ minus one; clamp it below at $e_{min}$ and subtract $M$ to get $q_e$. Round by a shift with round half to even: $v / 2^{q_e} = s \cdot 2^{e_2 - q_e}$ is a right shift of $s$ by $q_e - e_2$, and you compare the bits shifted out with half, rounding up if they are more and, if they are equal, only when the kept integer is odd. Let addition build the code: $((q_e + M + B - 1) \ll M) + n$ is the same formula as the Python version, where a subnormal has field 0 and code $n$, and a carry from $2^{M+1} - 1$ to $2^{M+1}$ moves into the exponent field.

The shared rounder implements those four steps:

```c
/* Rounded magnitude code of the finite, non-negative float32 with bits
 * `abs` in a format with M fraction bits and bias B. Never overflows a
 * uint32 for M <= 10 and B <= 127. */
static uint32_t rne_code(uint32_t abs, int M, int B) {
    uint32_t exp = abs >> 23, frac = abs & 0x7FFFFFu;
    uint32_t s;
    int e2;
    if (exp == 0) {
        if (frac == 0) return 0; /* +0 */
        s = frac;                /* subnormal float32: 0.frac * 2^-126 */
        e2 = -149;
    } else {
        s = frac | 0x800000u; /* normal: 1.frac * 2^(exp - 127) */
        e2 = (int)exp - 150;
    }
    int bits = 0; /* bit length of s */
    for (uint32_t t = s; t != 0; t >>= 1) bits++;
    int floorlog2 = e2 + bits - 1;
    int emin = 1 - B;
    int qe = (floorlog2 > emin ? floorlog2 : emin) - M;
    int shift = qe - e2; /* > 0: the target has fewer fraction bits */
    uint32_t n;
    if (shift > 24) {
        n = 0; /* v < 2^(qe - 1), half a quantum: rounds to 0 */
    } else {
        uint32_t half = 1u << (shift - 1);
        uint32_t rem = s & ((half << 1) - 1u);         /* the bits shifted out */
        n = s >> shift;
        if (rem > half || (rem == half && (n & 1u))) n++; /* ties to even */
    }
    if (n == 0) return 0;
    return ((uint32_t)(qe + M + B - 1) << M) + n;     /* carries into the exponent */
}
```

The shift is always positive because every destination has fewer than 23 fraction bits (and bfloat16's $e_{min} = -126$ equals float32's). The `shift > 24` branch is both a correctness and a safety rule: $s < 2^{24}$, so a shift of 25 or more leaves less than half a quantum, which rounds to 0, and it also keeps `1u << (shift - 1)` from reaching an undefined shift of 32 or more. For E4M3 the boundary case is $\lvert x\rvert = 2^{-10}$, exactly half the smallest subnormal $2^{-9}$: there $n = 0$, `rem == half`, and $n$ is even, so the result is 0, which is the header's "$\lvert x\rvert \le 2^{-10}$ gives $\pm 0$".

The remaining two steps are per format. Apply each format's specials around the core: NaN, infinity, sign, and overflow differ between formats, so each wrapper extracts the sign and absolute bits, answers NaN and infinity first, calls the shared rounder, then clamps or overflows by its own rule. And decode exactly: `ldexpf` of a small integer by a power of two is exact for every code of these formats, normal or subnormal.

```c
static float decode_mag(uint32_t code, int M, int B) {
    uint32_t ef = code >> M, mant = code & ((1u << M) - 1u);
    if (ef == 0) return ldexpf((float)mant, 1 - B - M);                  /* subnormal */
    return ldexpf((float)((1u << M) + mant), (int)ef - B - M);           /* implicit 1 */
}

uint8_t tl_f32_to_e4m3(float x) {
    uint32_t u = f32_bits(x), sign = (u >> 24) & 0x80u, abs = u & 0x7FFFFFFFu;
    if (abs > 0x7F800000u) return 0x7F; /* NaN */
    if (abs == 0x7F800000u) return (uint8_t)(sign | 0x7Eu); /* inf saturates */
    uint32_t c = rne_code(abs, 3, 7);
    if (c > 0x7Eu) c = 0x7Eu; /* above 448: saturate (0x7F is NaN) */
    return (uint8_t)(sign | c);
}

uint8_t tl_f32_to_e5m2(float x) {
    uint32_t u = f32_bits(x), sign = (u >> 24) & 0x80u, abs = u & 0x7FFFFFFFu;
    if (abs > 0x7F800000u) return 0x7F; /* NaN */
    if (abs == 0x7F800000u) return (uint8_t)(sign | 0x7Cu); /* inf stays inf */
    uint32_t c = rne_code(abs, 2, 15);
    if (c > 0x7Bu) c = 0x7Bu; /* finite overflow saturates to 57344 */
    return (uint8_t)(sign | c);
}
```

The wrappers share a shape. The sign is taken from bit 31 and moved to the destination's top bit, so `-0.0f` (bits `0x80000000`) becomes `0x80` through `rne_code`'s $+0$ path: the sign survives because it never went through the arithmetic. NaN and infinity are recognized by comparing the absolute bits with `0x7F800000` before rounding, since `rne_code` assumes a finite input. The clamp after rounding is the per-format overflow policy: both FP8 formats saturate, to `0x7E` (448) and `0x7B` (57344), which keeps a rounding carry from ever producing E4M3's NaN `0x7F` or E5M2's infinity `0x7C`.

bfloat16 uses the same rounder with a different policy:

```c
uint16_t tl_f32_to_bf16(float x) {
    uint32_t u = f32_bits(x), sign = (u >> 16) & 0x8000u, abs = u & 0x7FFFFFFFu;
    if (abs > 0x7F800000u) return 0x7FC0; /* NaN: quiet, sign dropped */
    if (abs == 0x7F800000u) return (uint16_t)(sign | 0x7F80u);
    uint32_t c = rne_code(abs, 7, 127);
    if (c > 0x7F80u) c = 0x7F80u; /* overflow to inf */
    return (uint16_t)(sign | c);
}

float tl_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16; /* bf16 is the top half of a float32 */
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}
```

bfloat16 has float32's exponent, so with $M = 7$, $B = 127$ the shared rounder is exactly "round the top 16 bits, the low 16 decide": the shift is always 16. Overflow goes to infinity (`0x7F80`), the IEEE rule, rather than saturating. Decoding is a 16-bit shift because bf16 *is* the top half of a float32. `tl_f32_to_f16` and `tl_f16_to_f32` follow the same pattern with $M = 10$, $B = 15$, infinity `0x7C00`, and NaN `0x7E00`; the E4M3 and E5M2 decoders check their NaN and infinity codes, then call `decode_mag` on the low 7 bits and apply the sign. The full solution is in [`lowp.c`](../../../../course/solutions/c/src/numerics/lowp.c).

### Why this design

One rounder serves four formats. Writing each conversion separately would quadruple the places a tie or a subnormal can be wrong; parameterizing by $(M, B)$ makes the formats differ only where the contract says they differ, in the special values and the overflow policy.

Integer arithmetic beats float tricks here. Adding and subtracting a magic constant (`x + 0x1.8p23f`-style rounding) or converting through `double` would also round, but each depends on the floating-point environment and FMA contraction. The integer shift-and-compare is exact, sanitizer-clean, and identical on every platform.

The C version works on bits, unlike the Python twin, which works on float64 values with `frexp`, `ldexp`, and `rint`. C has no vectorized `rint` to the format, and bit manipulation avoids any intermediate rounding. Both compute the same $q_e$, $n$, and code formula, which is why they agree on every input.

Each encode costs a bit-length loop of at most 24 iterations, a shift, and a compare; a production version would use `__builtin_clz` for the bit length. The tradeoff accepted is a few cycles for portability, since these conversions are not on the hottest path of the optional kernels.

## What can go wrong

The tests lock the hand-worked value to the interface: 0.3 must give E4M3 code `0x2A` and decode to 0.3125. FP8 saturation and specials are checked separately for E4M3 and E5M2, because a shared overflow branch would give one format the wrong policy. Subnormal ties and signed zero are checked because small values otherwise disappear or lose their sign. Every finite E4M3 and E5M2 code must survive decode then encode, an exhaustive check that finds errors across exponent transitions. Midpoint values must choose the even code, which both truncation and ties away from zero fail. For bfloat16, representative hand values and tie behavior check that the high-half shortcut still rounds correctly, and all bfloat16 codes must preserve finite values and specials. For binary16, hand examples cover normal and subnormal values (subnormal scaling differs from normal scaling), and every finite binary16 code must be a fixed point, which exercises all exponent and fraction combinations.

The mistakes those checks catch:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Always incrementing on a half-way remainder | Alternating midpoint values round away from even | `rne_code` increments on a tie only when `n & 1` |
| Applying E5M2 infinity rules to E4M3 | Large finite values become infinity instead of 448 | each format has its own wrapper and clamp (`0x7E` vs `0x7B`) |
| Using the normal quantum below $e_{min}$ | Smallest subnormal codes decode at the wrong scale | `qe` clamps at $e_{min}$ and `decode_mag` drops the implicit 1 when `ef == 0` |
| Dropping the sign before handling zero | `-0.0` returns as positive zero | the sign bit is extracted first and ORed back after rounding |

## Where this leads

In *Build Your Own LLM System*, [Paged attention for decode in C](../../build-an-llm/p09-kernels/07-paged-attention-decode.md) decodes half-precision values stored in its paged KV cache, and [Fused int4 and int8 dequantize-matmul in C](../../build-an-llm/p09-kernels/08-fused-quantized-matmul.md) decodes half-precision quantization scales, both through their C headers only.

Beyond these books, device conversion instructions provide hardware support that may round or flush subnormals differently, so the exhaustive format checks here are worth running on each new target; compiler and device conformance suites run the same vectors on additional architectures. A practical way to do that is to add a standalone C target and compare output files.

### Further reading

- The OCP FP8 specification, compared with IEEE 754 binary16's infinity and NaN encodings.
- IEEE Std 754-2019.

## Key terms

- **Quantum**: the spacing $2^{q_e}$ between adjacent representable values of the destination format in the binade of the input.
- **Integer significand**: the float32 magnitude written as $s \cdot 2^{e_2}$ with an integer $s < 2^{24}$.
- **Saturation**: clamping finite overflow to the largest finite code, as both FP8 formats do.
- **Fixed point (of a codec)**: a code that decodes and re-encodes to the same bits.

## Summary

- Encoding a value means rounding its significand in units of the destination format's quantum; for E4M3, 0.3 encodes to `0x2A`.
- One integer rounder parameterized by $(M, B)$ serves E4M3, E5M2, bfloat16, and binary16, and the code formula lets a rounding carry move into the exponent field on its own.
- E4M3 finite overflow saturates, while E5M2 reserves infinity and NaN codes; one shared overflow branch would give one of them the wrong policy.
- Ties to even depend on the low bit of the retained significand, not the sign.
- Every finite encoding decodes and re-encodes to the same bits, in both FP8 formats, bfloat16, and binary16.

## Review questions

1. Why does `rne_code` compare the discarded bits with `half` and then test `n & 1`?

   <details><summary>Answer</summary>

   Round to nearest needs to know whether the discarded part is below, at, or above half a quantum. Above half rounds up, below rounds down. Exactly half is a tie, and ties to even rounds up only when the kept integer $n$ is odd, so the result's last bit is 0. The sign plays no part: the magnitude is rounded and the sign reattached.

   </details>

2. How does a rounding carry move into the exponent field without special code?

   <details><summary>Answer</summary>

   The code is $((q_e + M + B - 1) \ll M) + n$, where a normal $n$ lies in $[2^M, 2^{M+1})$ and includes the implicit bit. If rounding takes $n$ to $2^{M+1}$, the addition adds $2 \cdot 2^M$, which is one more in the exponent field and a zero fraction: exactly the next binade's first value.

   </details>

3. Why is the `shift > 24` branch needed?

   <details><summary>Answer</summary>

   $s < 2^{24}$, so with a shift of 25 or more the value is below half a quantum and must round to 0. Without the branch, `1u << (shift - 1)` could shift by 32 or more for tiny inputs in formats with a small $e_{min}$, which is undefined behavior in C and would trip UBSan.

   </details>

4. Why do E4M3 and E5M2 clamp to different codes, and why after rounding?

   <details><summary>Answer</summary>

   E4M3 has no infinity and uses `0x7F` for NaN, so its largest finite code is `0x7E` (448). E5M2 keeps IEEE infinity at `0x7C`, so its largest finite code is `0x7B` (57344). Clamping after rounding catches both a large input and a value just below the limit that rounds up past it; clamping the input first would miss the second case.

   </details>
