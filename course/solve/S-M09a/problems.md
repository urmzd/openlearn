# S-M09a problems: floating-point representation, rounding, cancellation, stable rewrites

Answer every question in `solve/S-M09a.toml` (written by `ol start S-M09a`).
The tag after each question is its answer type: `[number]` is an exact value
(`2^-23`, `1 + 2^-6`, `1000 + log(2)`) unless the question asks for a
tolerance, `[vector]` a list `[0, 127, 0]`, `[bool]` `true` or `false`,
`[choice]` one letter, and `[proof]` a file `solve/S-M09a/qN.md` graded
against its rubric.

A float32 has 1 sign bit, 8 exponent bits (bias 127), and 23 fraction bits;
float16 (fp16) has 1, 5 (bias 15), and 10; bfloat16 (bf16) has 1, 8 (bias
127), and 7. "Round to nearest, ties to even" (RNE) rounds to the closest
representable value and, on an exact tie, to the one whose last fraction bit
is 0.

### Representation and rounding

**q1.** Give the float32 fields $[\text{sign}, \text{biased exponent}, \text{fraction}]$, the fraction as the unsigned integer stored in its 23 bits, for (a) $1.0$ and (b) $-6.5$. `[vector]`

**q2.** Give the gap between $1$ and the next larger representable number in (a) float32, (b) bf16, (c) fp16. `[number]`

**q3.** (a) Give the largest finite fp16 value. (b) The smallest positive normal float32 value is $2^k$; give $k$. `[number]`

**q4.** Round to bf16 with RNE: (a) $1 + 2^{-8}$, (b) $1 + 3 \cdot 2^{-8}$. `[number]`

**q5.** Is $0.1$ exactly representable in any binary floating-point format? `[bool]`

### Cancellation and stable rewrites

**q6.** Let $f(x) = \sqrt{x + 1} - \sqrt{x}$ and $x = 10^8$.
(a) Evaluate $f(x)$ in float32 with every operation rounded to float32 (RNE). What does it return? `[number]`
(b) Give the true value of $f(10^8)$, exactly or to at least 7 significant digits. `[number, rtol 1e-6]`

**q7.** Which formula computes $1 - \cos x$ accurately in float64 at $x = 10^{-8}$? `[choice]`
(a) `1 - cos(x)` (b) `2 * sin(x / 2)^2`

**q8.** (a) Give $\mathrm{LSE}(1000, 1000) = \log(e^{1000} + e^{1000})$ exactly. `[number]`
(b) In float64, what does the naive softmax `exp(x) / sum(exp(x))` return for its first entry at $x = (1000, 1000)$? Answer one of `0.5`, `inf`, `nan`, `0`. `[choice]`

**q9.** The data are $x = (10^8 + 1, 10^8 + 2, 10^8 + 3)$.
(a) Give their population variance $\frac{1}{3}\sum_i (x_i - \bar x)^2$ exactly. `[number]`
(b) In float64, which algorithm returns it to full precision: `one` pass (the mean of $x_i^2$ minus the square of the mean) or `two` passes (compute the mean, then average the squared deviations)? Answer `one` or `two`. `[choice]`

**q10.** You sum the float32 values $1.0$ followed by $10^4$ copies of $10^{-8}$, left to right, rounding every addition to float32.
(a) What does the naive loop return? `[number]`
(b) Give the exact real sum $1 + 10^4 \cdot 10^{-8}$, the value compensated (Kahan) summation recovers. `[number]`

**q11.** Prove that $\mathrm{softmax}(x + c\mathbf{1}) = \mathrm{softmax}(x)$ for every real $c$, and that with $c = -\max_i x_i$ every exponent is at most 0, so no $e^{x_i + c}$ overflows and the denominator is at least 1. `[proof]`
