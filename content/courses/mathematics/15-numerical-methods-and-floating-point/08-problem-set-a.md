# Problem set A: representation, rounding, and cancellation

A bigram that computes its loss in float64 numpy on tiny logits never meets a numerical problem. A trained model does. Its logits reach the hundreds, so `exp` overflows float32 at about $88.7$ and the naive softmax returns NaN. Its weights are stored in bf16 (the SmolLM2 checkpoint loaded in [The Llama-family model: config, weight loading, and download](../../ai-engineering/13-decoder-block/09-llama-family-model.md) in *AI Engineering* is), so rounding must match PyTorch exactly or parity checks fail in the last bit. Its evaluation sums $10^7$ per-token losses ([The evaluation harness and the model zoo](../../ai-engineering/12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md)), and a float32 accumulator silently stops growing.

[IEEE 754 anatomy, ulps, and round to nearest even](01-ieee-754.md) emulates bf16 and fp16 rounding, and [Stable numerics](02-stable-numerics.md) writes the stable reductions every later operation calls. This pen-and-paper set makes sure you can predict each of those failures by hand before you read the code that avoids it. It covers representation and rounding (q1 to q5) and cancellation and stable rewrites (q6 to q11, ending with a short proof). [Problem set B](09-problem-set-b.md) continues with error bounds, Newton's method, polynomial approximation, and low precision.

The proof in q11 uses the habits of [Discrete math 1 problem set](../11-discrete-math-1/03-problem-set/README.md), and q10 relies on the fact, from [Vectors, matrices, row-major layout, and matmul](../09-linear-algebra/01-vectors-matrices-and-matmul.md), that float sums depend on their order. Work each problem before opening its answer. Each answer gives the final value, the derivation, and the common wrong answer with the reason it is wrong. Exact values are written as powers of two or fractions.

> **Learning objectives**
> - Decompose a number into float32 fields and compute the gap between neighbouring floats in float32, bf16, and fp16.
> - Round exact ties to even in bf16 and recognise which decimal values are not representable in binary.
> - Predict when cancellation, overflow, or absorption in a running sum destroys a result, and rewrite the formula to avoid it.
> - Prove that softmax is shift invariant and that the max shift prevents overflow.

## The facts every problem uses

These are the symbols used in the problems. Note that here $p$ counts *stored* fraction bits, so the unit roundoff is $2^{-(p+1)}$.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $s, e, f$ | sign bit, biased exponent field, fraction field | unsigned integers |
| $p$ | fraction bits: 23 (float32), 10 (fp16), 7 (bf16) | integer |
| bias | 127 (float32, bf16), 15 (fp16) | integer |
| $\mathrm{fl}(x)$ | $x$ rounded to the nearest representable value | float |
| $\mathrm{ulp}(x)$ | gap between the two floats around $x$ | float |
| $u$ | unit roundoff, $2^{-(p+1)}$: the largest relative rounding error | float |
| $\mathrm{LSE}(x)$ | $\log \sum_i e^{x_i}$ | scalar |

### Representation

A **normal** binary float with $p$ fraction bits represents

$$x = (-1)^s \cdot \left(1 + \frac{f}{2^p}\right) \cdot 2^{\,e - \text{bias}}, \qquad 1 \le e \le 2^{w} - 2,$$

where $w$ is the exponent width. The leading 1 is implicit (not stored). The exponent field $e = 0$ holds zero and the **subnormals** $0.f \cdot 2^{1 - \text{bias}}$, which fill the gap below the smallest normal $2^{1 - \text{bias}}$; $e = 2^w - 1$ holds $\pm\infty$ ($f = 0$) and NaN ($f \ne 0$). To decompose a number, write it as $\pm 1.\text{something} \times 2^E$ in binary: $-6.5 = -110.1_2 = -1.101_2 \times 2^2$, so $s = 1$, $e = 2 + 127 = 129$, and the fraction bits are $101$ followed by 20 zeros, the integer $5 \cdot 2^{20}$. The three formats compare as follows:

| Format | Exponent bits | Fraction bits $p$ | Bias | Gap at 1 ($2^{-p}$) | Largest finite |
|---|---|---|---|---|---|
| float32 | 8 | 23 | 127 | $2^{-23}$ | $(2 - 2^{-23}) \cdot 2^{127}$ |
| bf16 | 8 | 7 | 127 | $2^{-7}$ | $(2 - 2^{-7}) \cdot 2^{127}$ |
| fp16 | 5 | 10 | 15 | $2^{-10}$ | $(2 - 2^{-10}) \cdot 2^{15}$ |

bf16 is the top 16 bits of a float32, so converting is a rounding of the low 16 bits. Within one **binade** $[2^E, 2^{E+1})$ the floats are equally spaced with gap $\mathrm{ulp} = 2^{E - p}$, and the gap doubles at each power of two: the integers above $2^{24}$ are not all float32 values, and above $10^8$ float32's gap is 8.

### Rounding

Every arithmetic result is rounded to a representable value. **Round to nearest, ties to even (RNE)**, the IEEE 754 default, picks the nearest float and, when the exact result is halfway between two, the one whose last fraction bit is 0. Ties to even avoids the upward drift that "round half up" would add over many operations. The relative error of one rounding is at most the **unit roundoff** $u = 2^{-(p+1)}$, half the gap at 1: $\mathrm{fl}(x) = x(1 + \delta)$ with $\lvert \delta \rvert \le u$. Decimal fractions such as $0.1$ have infinite binary expansions, because their denominator has the factor 5, so they are always rounded.

### Cancellation

Subtraction of two floats is exact when they are within a factor 2 of each other (Sterbenz), so the damage is never the subtraction itself. It is that the operands already carry rounding errors from earlier steps, and subtracting nearly equal numbers cancels their shared leading digits and leaves those errors as the whole result. $\sqrt{10^8 + 1} - \sqrt{10^8}$ in float32 is 0 because $10^8 + 1$ already rounded to $10^8$. The cure is an algebraically equal form that does not subtract nearly equal quantities: multiply by the conjugate, $\sqrt{x+1} - \sqrt{x} = \frac{1}{\sqrt{x+1} + \sqrt{x}}$; use $1 - \cos x = 2\sin^2(x/2)$; compute a variance from deviations rather than from $E[x^2] - E[x]^2$.

### Overflow and the max shift

$e^x$ overflows float32 above $\ln(3.4 \times 10^{38}) \approx 88.7$ and float64 above about $709.8$, and $\infty/\infty$ is NaN. Softmax and log-sum-exp are unchanged by subtracting a constant from every input (q11), so subtracting $m = \max_i x_i$ keeps every exponent at most 0:

$$\mathrm{LSE}(x) = m + \log \sum_i e^{x_i - m}, \qquad \mathrm{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}} .$$

### Summation

Adding a small number to a large running sum rounds the small number away when it is below half a gap: in float32, $1 + 10^{-8} = 1$ because the gap at 1 is $2^{-23} \approx 1.2 \times 10^{-7}$. A long sum then stops growing. **Kahan summation** carries a compensation term that captures the low-order bits each addition drops and feeds them back; **pairwise summation** adds in a balanced tree so that every partial sum is of similar size. [Stable numerics](02-stable-numerics.md) implements both.

## Two worked examples

Before the problems, here are two siblings of q1 and q6, worked in full.

First, decompose $0.15625$ as float32. $0.15625 = 5/32 = 101_2 \times 2^{-5} = 1.01_2 \times 2^{-3}$. The sign is $s = 0$, the exponent $e = -3 + 127 = 124$, and the fraction bits are $01$ followed by 21 zeros, so $f = 2^{21} = 2097152$. The answer is `[0, 124, 2097152]`. Check: $(1 + 2^{21}/2^{23}) \cdot 2^{-3} = 1.25/8 = 0.15625$.

Second, a cancellation. Evaluate $g(x) = (1 + x) - 1$ in float32 at $x = 3 \cdot 10^{-8}$. The gap at 1 is $2^{-23} \approx 1.19 \times 10^{-7}$, so $1 + x$ lies between $1$ and $1 + 2^{-23}$, closer to 1 (since $x < 2^{-24} \approx 5.96 \times 10^{-8}$), and rounds to 1. Then $1 - 1 = 0$: the answer is $0$, a 100% relative error, although the true value is $3 \cdot 10^{-8}$. Algebra gives the stable form $g(x) = x$. The same structure, a sum that has already rounded away the part you are about to subtract back out, is q6.

## The problems

A float32 has 1 sign bit, 8 exponent bits (bias 127), and 23 fraction bits; float16 (fp16) has 1, 5 (bias 15), and 10; bfloat16 (bf16) has 1, 8 (bias 127), and 7. The tag after each question is its answer type.

### Representation and rounding

**q1.** Give the float32 fields $[\text{sign}, \text{biased exponent}, \text{fraction}]$, the fraction as the unsigned integer stored in its 23 bits, for (a) $1.0$ and (b) $-6.5$. `[vector]`

<details><summary>Answer</summary>

(a) $[0, 127, 0]$. $1.0 = +1.0_2 \times 2^0$: sign 0, biased exponent $0 + 127 = 127$, and the fraction after the implicit 1 is all zeros.

(b) $[1, 129, 5242880]$, where $5242880 = 5 \cdot 2^{20}$. $-6.5 = -110.1_2 = -1.101_2 \times 2^2$: sign 1, exponent $2 + 127 = 129$, and the 23 fraction bits are $101$ followed by 20 zeros, the integer $101_2 \cdot 2^{20} = 5 \cdot 2^{20}$.

A common mistake is $[0, 0, 0]$ or $[1, 2, \ldots]$ for the exponent, which forgets the bias; another is a fraction of 1 for $1.0$, which stores the implicit leading 1 (it is never stored). Reading the fraction bits $.101$ as the decimal 5 gives $[1, 129, 5]$, which ignores their position in the 23-bit field.

</details>

**q2.** Give the gap between $1$ and the next larger representable number in (a) float32, (b) bf16, (c) fp16. `[number]`

<details><summary>Answer</summary>

(a) $2^{-23}$, (b) $2^{-7} = 1/128$, (c) $2^{-10}$. In the binade $[1, 2)$ the floats are $1 + k \cdot 2^{-p}$ with $p$ the number of stored fraction bits, so the next float after 1 is $1 + 2^{-p}$.

A common mistake is $2^{-24}$ for float32: that is the unit roundoff $u$, half the gap. For bf16, $2^{-8}$ counts the implicit bit as stored (bf16 has 8 significant bits but 7 stored fraction bits), and $2^{-10}$ is fp16's gap, not bf16's.

</details>

**q3.** (a) Give the largest finite fp16 value. (b) The smallest positive normal float32 value is $2^k$; give $k$. `[number]`

<details><summary>Answer</summary>

(a) $65504 = (2 - 2^{-10}) \cdot 2^{15}$. The largest exponent field that is not infinity/NaN is $2^5 - 2 = 30$, so the exponent is $30 - 15 = 15$, and the largest significand is all ones, $2 - 2^{-10}$. A common mistake is $65536 = 2^{16}$, which would need exponent 16 (that field is reserved for infinity), or $65535$, the largest `uint16`.

(b) $k = -126$. The smallest normal exponent field is 1, so the exponent is $1 - 127 = -126$ with significand 1. A common mistake is $-127$, which uses the field value 0 (reserved for zero and subnormals); $-149$ is the smallest *subnormal*, $2^{-126} \cdot 2^{-23}$.

</details>

**q4.** Round to bf16 with RNE: (a) $1 + 2^{-8}$, (b) $1 + 3 \cdot 2^{-8}$. `[number]`

<details><summary>Answer</summary>

(a) $1$. bf16's gap at 1 is $2^{-7}$, so the neighbours are $1$ (fraction `0000000`) and $1 + 2^{-7}$ (fraction `0000001`). $1 + 2^{-8}$ is exactly halfway; ties to even picks the neighbour whose last bit is 0, which is 1. A common mistake is $1 + 2^{-7}$, rounding the tie up.

(b) $1 + 2^{-6} = 65/64$. $1 + 3 \cdot 2^{-8} = 1 + 1.5 \cdot 2^{-7}$ is halfway between $1 + 2^{-7}$ (fraction `0000001`, odd) and $1 + 2^{-6}$ (fraction `0000010`, even), so it rounds up to the even one. Answering $1 + 2^{-7}$ truncates or rounds ties down; ties to even goes up here and down in (a).

</details>

**q5.** Is $0.1$ exactly representable in any binary floating-point format? `[bool]`

<details><summary>Answer</summary>

False. Every binary float is $m / 2^k$ for integers $m, k$ (or $m \cdot 2^k$). $0.1 = 1/10 = 1/(2 \cdot 5)$ in lowest terms has the factor 5 in its denominator, which no power of two can cancel, so its binary expansion $0.0\overline{0011}_2$ never terminates. Every format rounds it.

</details>

### Cancellation and stable rewrites

**q6.** Let $f(x) = \sqrt{x + 1} - \sqrt{x}$ and $x = 10^8$.
(a) Evaluate $f(x)$ in float32 with every operation rounded to float32 (RNE). What does it return? `[number]`
(b) Give the true value of $f(10^8)$, exactly or to at least 7 significant digits. `[number, rtol 1e-6]`

<details><summary>Answer</summary>

(a) $0$. $10^8$ lies in $[2^{26}, 2^{27})$, where the float32 gap is $2^{26 - 23} = 8$. $10^8 = 2^8 \cdot 390625$ is representable, and $10^8 + 1$ is within half a gap of it, so it rounds to $10^8$. Both square roots are then $\sqrt{10^8} = 10^4$, and the subtraction gives 0.

(b) $\dfrac{1}{\sqrt{100000001} + 10000} \approx 4.99999998750 \times 10^{-5}$. Multiply by the conjugate: $\sqrt{x+1} - \sqrt{x} = \frac{(x + 1) - x}{\sqrt{x+1} + \sqrt{x}} = \frac{1}{\sqrt{x+1} + \sqrt{x}}$, which subtracts nothing and is accurate in float32 too. A common mistake for (a) is to give this true value ($\approx 1/20000$) instead of what float32 computes. For (b), $1/10000$ is $1/\sqrt{x}$; the denominator is the *sum* of two roots, about $2\sqrt{x}$.

</details>

**q7.** Which formula computes $1 - \cos x$ accurately in float64 at $x = 10^{-8}$? `[choice]`
(a) `1 - cos(x)` (b) `2 * sin(x / 2)^2`

<details><summary>Answer</summary>

(b). $\cos(10^{-8}) = 1 - 5 \times 10^{-17} + \ldots$, and the float64 gap just below 1 is $2^{-53} \approx 1.1 \times 10^{-16}$, so $\cos x$ rounds to exactly 1 and (a) returns 0. The identity $1 - \cos x = 2\sin^2(x/2)$ computes $2 \cdot (5 \times 10^{-9})^2 = 5 \times 10^{-17}$ from a small number that was never subtracted from 1, so every digit is correct.

</details>

**q8.** (a) Give $\mathrm{LSE}(1000, 1000) = \log(e^{1000} + e^{1000})$ exactly. `[number]`
(b) In float64, what does the naive softmax `exp(x) / sum(exp(x))` return for its first entry at $x = (1000, 1000)$? Answer one of `0.5`, `inf`, `nan`, `0`. `[choice]`

<details><summary>Answer</summary>

(a) $1000 + \log 2$. Shift by $m = 1000$: $\log(e^{1000}(e^0 + e^0)) = 1000 + \log 2$. Common mistakes are $1000$ (dropping the $\log 2$) and $\log 2$ (forgetting to add the shift back).

(b) `nan`. $e^{1000}$ overflows float64 (above about $709.8$) to $\infty$, the sum is $\infty$, and $\infty / \infty$ is NaN. A common mistake is `inf`: the exponential does overflow to infinity, but the division then produces NaN. The true value, 0.5, is what the max-shifted softmax returns.

</details>

**q9.** The data are $x = (10^8 + 1, 10^8 + 2, 10^8 + 3)$.
(a) Give their population variance $\frac{1}{3}\sum_i (x_i - \bar x)^2$ exactly. `[number]`
(b) In float64, which algorithm returns it to full precision: `one` pass (the mean of $x_i^2$ minus the square of the mean) or `two` passes (compute the mean, then average the squared deviations)? Answer `one` or `two`. `[choice]`

<details><summary>Answer</summary>

(a) $2/3$. The mean is $10^8 + 2$, the deviations are $-1, 0, 1$, and $\frac{1}{3}(1 + 0 + 1) = 2/3$. A common mistake is 1, the sample variance (dividing by $n - 1 = 2$).

(b) `two`. Each $x_i^2 \approx 10^{16}$ exceeds $2^{53} \approx 9.0 \times 10^{15}$, where the float64 gap is 2, so the squares are rounded by up to 1 each; the one-pass formula subtracts two numbers near $10^{16}$ whose difference is $2/3$, so the rounding errors are the same size as the answer (it can even come out negative). The two-pass method subtracts the mean first (exact here, Sterbenz), then squares small numbers.

</details>

**q10.** You sum the float32 values $1.0$ followed by $10^4$ copies of $10^{-8}$, left to right, rounding every addition to float32.
(a) What does the naive loop return? `[number]`
(b) Give the exact real sum $1 + 10^4 \cdot 10^{-8}$, the value compensated (Kahan) summation recovers. `[number]`

<details><summary>Answer</summary>

(a) $1$. The float32 gap at 1 is $2^{-23} \approx 1.19 \times 10^{-7}$, and $10^{-8}$ is below half of it ($2^{-24} \approx 5.96 \times 10^{-8}$), so every $1 + 10^{-8}$ rounds back to 1. The running sum never moves, all $10^4$ times.

(b) $1 + 10^{-4} = 10001/10000$. Kahan summation keeps the dropped low-order part in a compensation variable; once those parts accumulate past half a gap, they enter the sum. A common mistake is to swap the two answers: (a) asks what float32 returns, not the exact sum.

</details>

**q11.** Prove that $\mathrm{softmax}(x + c\mathbf{1}) = \mathrm{softmax}(x)$ for every real $c$, and that with $c = -\max_i x_i$ every exponent is at most 0, so no $e^{x_i + c}$ overflows and the denominator is at least 1. `[proof]`

<details><summary>Answer</summary>

**Claim.** For $x \in \mathbb{R}^n$ and every real $c$, $\mathrm{softmax}(x + c\mathbf{1}) = \mathrm{softmax}(x)$, where $\mathrm{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$. With $c = -m$, $m = \max_j x_j$, every exponent $x_i + c$ is at most 0, so no term overflows, and the denominator is at least 1.

**Shift invariance.** Using $e^{a + b} = e^a e^b$,
$$\mathrm{softmax}(x + c\mathbf{1})_i = \frac{e^{x_i + c}}{\sum_j e^{x_j + c}} = \frac{e^{c} e^{x_i}}{e^{c} \sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}} = \mathrm{softmax}(x)_i,$$
where cancelling $e^c$ is allowed because $e^c > 0$.

**Bounds with $c = -m$.** For every $i$, $x_i \le m$, so $x_i - m \le 0$ and $0 < e^{x_i - m} \le 1$. No term can overflow, whatever the size of $x$. The index $k$ with $x_k = m$ contributes $e^0 = 1$, and every other term is positive, so the denominator $\sum_j e^{x_j - m} \ge 1$: it is never 0, and the division never produces $\infty$ or NaN. (The denominator is at most $n$, also finite.)

**What it avoids.** Without the shift, $e^{x}$ overflows float32 for $x$ above about $88.7$ and float64 above about $709.8$, and $\infty / \infty$ is NaN: the failure of q8(b). With it, the same mathematical value is computed from numbers in $(0, 1]$. This is how `softmax` and `logsumexp` in [Stable numerics](02-stable-numerics.md) are written.

A complete proof has four parts: the factor $e^c$ cancelled because it is positive, each shifted term in $(0, 1]$, the denominator at least 1 because the maximum term is $e^0$, and the overflow thresholds this avoids.

</details>

## Common mistakes

The errors this set is designed to expose recur in real code. Each one maps to a problem:

| Pitfall | Symptom | How to avoid it |
|---|---|---|
| Forgetting the exponent bias, or storing the implicit 1 | `decompose_f32` disagrees with the bit view | q1: add 127 to the exponent; the leading 1 is implicit |
| Confusing the gap at 1 with the unit roundoff | tolerances off by a factor 2 | q2: the gap is $2^{-p}$, the unit roundoff $2^{-(p+1)}$ is half of it |
| Thinking fp16 reaches 65536 | overflow to inf in mixed precision | q3: the top exponent field is reserved, so the maximum is $(2 - 2^{-10})2^{15} = 65504$ |
| Rounding ties up instead of to even | bf16 results differ from torch in the last bit | q4: on a tie, pick the neighbour whose last fraction bit is 0 |
| Trusting the naive formula's output | a 0 where the answer is $5 \times 10^{-5}$ | q6: rewrite with the conjugate so nothing nearly equal is subtracted |
| Exponentiating before subtracting the max | NaN from inf / inf at logit 1000 | q8 and q11: shift by the maximum first |
| One-pass variance on data far from 0 | negative or garbage variances | q9: subtract the mean before squaring |
| Accumulating a long sum naively in float32 | NLL totals that stop growing | q10: compensated (Kahan) or pairwise summation |

## Where this leads

q1 to q4 become code in [IEEE 754 anatomy, ulps, and round to nearest even](01-ieee-754.md) (`decompose_f32`, `ulp`, `round_to_bf16`, `round_to_fp16`), and q8, q10, and q11 become `logsumexp`, `softmax`, `kahan_sum`, and `pairwise_sum` in [Stable numerics](02-stable-numerics.md). In *AI Engineering*, the fused cross-entropy of [Losses with a fused backward](../../ai-engineering/06-numerics-and-autograd/04-losses-with-fused-backward.md) computes `log_softmax` with the max shift, [The Llama-family model: config, weight loading, and download](../../ai-engineering/13-decoder-block/09-llama-family-model.md) loads bf16 weights (q2 to q4), and [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](../../ai-engineering/17-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md) explains why bf16 needs no loss scaling and fp16 does (q3). [Problem set B](09-problem-set-b.md) makes q6's loss of digits quantitative with error bounds and condition numbers.

### Further reading

- Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (1991).
- Higham, *Accuracy and Stability of Numerical Algorithms*, chapters 1 to 4.

## Key terms

- **Implicit leading bit**: the 1 in front of the fraction of a normal float, which is not stored.
- **Binade**: an interval $[2^E, 2^{E+1})$ within which floats are equally spaced.
- **Unit roundoff**: the largest relative error of one rounding, half the gap at 1.
- **Cancellation**: the loss of significant digits when subtracting nearly equal numbers that already carry rounding error.
- **Absorption**: a small addend vanishing entirely because it is below half the gap of a large running sum.

## Summary

- A float is sign, biased exponent, and fraction; its value is $(-1)^s \cdot 1.f \cdot 2^{e - \text{bias}}$, and the gap between neighbours doubles every binade (q1, q2).
- bf16 keeps float32's 8 exponent bits and drops 16 fraction bits, so it has float32's range and a gap of $2^{-7}$ at 1; fp16 has a gap of $2^{-10}$ but overflows at 65504 (q2, q3).
- Round to nearest, ties to even, sends an exact halfway value to the neighbour with an even last bit, which is what makes `round_to_bf16` agree with torch bit for bit (q4).
- Subtracting nearly equal numbers loses every digit they share; the fix is an algebraically equal formula that never subtracts them (q6, q7, q9).
- Shifting softmax and log-sum-exp by the maximum changes nothing mathematically and removes overflow (q8, q11).
