<!-- ol:module S-M09a -->
# Floating point problem set, part a: representation, rounding, cancellation

## Overview

| | |
|---|---|
| **Module** | `S-M09a` · solve · none · Pass 2 · 3 to 4 h |
| **You build** | answers in `solve/S-M09a.toml` (19 checked by SymPy) and 1 proof in `solve/S-M09a/q11.md` (self-graded against its rubric) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M09a/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M09a/problems.md` and in section 4 |
| **Needs** | `S-M05` (proof habits). Reading: the [Numerical Methods and Floating Point topic](README.md) and `M03.1` section 2.6 (why float sums depend on order) |
| **Used by** | no call site (a solve set). Do it with `M09.1` (`decompose_f32`, `ulp`, `round_to_bf16`, `round_to_fp16`) and `M09.2` (`logsumexp`, `softmax`, `kahan_sum`, `pairwise_sum`); part b, `S-M09b` in Pass 6, covers error bounds, Newton, polynomial approximation, and low precision |
| **Milestone** | `MS-P2` (the Pass 2 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (1991); Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 1 to 4 |

## Key Takeaways

- A float is sign, biased exponent, and fraction; its value is $(-1)^s \cdot 1.f \cdot 2^{e - \text{bias}}$, and the gap between neighbors doubles every binade (q1, q2).
- bf16 keeps float32's 8 exponent bits and drops 16 fraction bits, so it has float32's range and a gap of $2^{-7}$ at 1; fp16 has a gap of $2^{-10}$ but overflows at 65504 (q2, q3).
- Round to nearest, ties to even, sends an exact halfway value to the neighbor with an even last bit, which is what makes `round_to_bf16` agree with torch bit for bit (q4).
- Subtracting nearly equal numbers loses every digit they share; the fix is an algebraically equal formula that never subtracts them (q6, q7, q9).
- Shifting softmax and log-sum-exp by the maximum changes nothing mathematically and removes overflow (q8, q11).

## How to work this chapter

```bash
ol start S-M09a             # writes solve/S-M09a.toml and the proof file
ol check S-M09a             # SymPy checks the answers, then asks the proof rubric (y/n)
ol check S-M09a --regrade   # ask the rubric again after you change the proof
```

---

## 1. Why now

Your Pass 1 bigram computed its loss in float64 numpy on tiny logits, and nothing went wrong. In Pass 2 the numbers stop being friendly. Logits of a trained model reach the hundreds, so `exp` overflows float32 at about $88.7$ and the naive softmax returns NaN. Weights are stored in bf16 (`L7.9` loads SmolLM2's), so you must round exactly as PyTorch does or your parity tests fail in the last bit. Evaluation sums $10^7$ per-token losses (`L6.7`), and a float32 accumulator silently stops growing. `M09.1` emulates bf16 and fp16 rounding and `M09.2` writes the stable reductions every later op calls. This set makes sure you can predict each failure by hand before you write the code that avoids it.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $s, e, f$ | sign bit, biased exponent field, fraction field | unsigned integers |
| $p$ | fraction bits: 23 (float32), 10 (fp16), 7 (bf16) | integer |
| bias | 127 (float32, bf16), 15 (fp16) | integer |
| $\mathrm{fl}(x)$ | $x$ rounded to the nearest representable value | float |
| $\mathrm{ulp}(x)$ | gap between the two floats around $x$ | float |
| $u$ | unit roundoff, $2^{-(p+1)}$: the largest relative rounding error | float |
| $\mathrm{LSE}(x)$ | $\log \sum_i e^{x_i}$ | scalar |

### 2.1 Representation

A **normal** binary float with $p$ fraction bits represents
$$x = (-1)^s \cdot \left(1 + \frac{f}{2^p}\right) \cdot 2^{\,e - \text{bias}}, \qquad 1 \le e \le 2^{w} - 2,$$
where $w$ is the exponent width. The leading 1 is implicit (not stored). The exponent field $e = 0$ holds zero and the **subnormals** $0.f \cdot 2^{1 - \text{bias}}$, which fill the gap below the smallest normal $2^{1 - \text{bias}}$; $e = 2^w - 1$ holds $\pm\infty$ ($f = 0$) and NaN ($f \ne 0$). To decompose a number, write it as $\pm 1.\text{something} \times 2^E$ in binary: $-6.5 = -110.1_2 = -1.101_2 \times 2^2$, so $s = 1$, $e = 2 + 127 = 129$, and the fraction bits are $101$ followed by 20 zeros, the integer $5 \cdot 2^{20}$.

| Format | Exponent bits | Fraction bits $p$ | Bias | Gap at 1 ($2^{-p}$) | Largest finite |
|---|---|---|---|---|---|
| float32 | 8 | 23 | 127 | $2^{-23}$ | $(2 - 2^{-23}) \cdot 2^{127}$ |
| bf16 | 8 | 7 | 127 | $2^{-7}$ | $(2 - 2^{-7}) \cdot 2^{127}$ |
| fp16 | 5 | 10 | 15 | $2^{-10}$ | $(2 - 2^{-10}) \cdot 2^{15}$ |

bf16 is the top 16 bits of a float32, so converting is a rounding of the low 16 bits. Within one **binade** $[2^E, 2^{E+1})$ the floats are equally spaced with gap $\mathrm{ulp} = 2^{E - p}$, and the gap doubles at each power of two: the integers above $2^{24}$ are not all float32 values, and above $10^8$ float32's gap is 8.

### 2.2 Rounding

Every arithmetic result is rounded to a representable value. **Round to nearest, ties to even (RNE)**, the IEEE 754 default, picks the nearest float and, when the exact result is halfway between two, the one whose last fraction bit is 0. Ties to even avoids the upward drift that "round half up" would add over many operations. The relative error of one rounding is at most the **unit roundoff** $u = 2^{-(p+1)}$, half the gap at 1: $\mathrm{fl}(x) = x(1 + \delta)$ with $\lvert \delta \rvert \le u$. Decimal fractions such as $0.1$ have infinite binary expansions, because their denominator has the factor 5, so they are always rounded.

### 2.3 Cancellation

Subtraction of two floats is exact when they are within a factor 2 of each other (Sterbenz), so the damage is never the subtraction itself: it is that the operands already carry rounding errors from earlier steps, and subtracting nearly equal numbers cancels their shared leading digits and leaves those errors as the whole result. $\sqrt{10^8 + 1} - \sqrt{10^8}$ in float32 is 0 because $10^8 + 1$ already rounded to $10^8$. The cure is an algebraically equal form that does not subtract nearly equal quantities: multiply by the conjugate, $\sqrt{x+1} - \sqrt{x} = \frac{1}{\sqrt{x+1} + \sqrt{x}}$; use $1 - \cos x = 2\sin^2(x/2)$; compute a variance from deviations rather than from $E[x^2] - E[x]^2$.

### 2.4 Overflow and the max shift

$e^x$ overflows float32 above $\ln(3.4 \times 10^{38}) \approx 88.7$ and float64 above about $709.8$; $\infty/\infty$ is NaN. Softmax and log-sum-exp are unchanged by subtracting a constant from every input (q11), so subtracting $m = \max_i x_i$ keeps every exponent at most 0:
$$\mathrm{LSE}(x) = m + \log \sum_i e^{x_i - m}, \qquad \mathrm{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}} .$$

### 2.5 Summation

Adding a small number to a large running sum rounds the small number away when it is below half a gap: in float32, $1 + 10^{-8} = 1$ because the gap at 1 is $2^{-23} \approx 1.2 \times 10^{-7}$. A long sum then stops growing. **Kahan summation** carries a compensation term that captures the low-order bits each addition drops and feeds them back; **pairwise summation** adds in a balanced tree so that every partial sum is of similar size. `M09.2` implements both.

## 3. Worked example by hand

This is a sibling of q1 and q6, not one of the graded problems.

**Decompose $0.15625$ as float32.** $0.15625 = 5/32 = 101_2 \times 2^{-5} = 1.01_2 \times 2^{-3}$. Sign $s = 0$. Exponent $e = -3 + 127 = 124$. Fraction bits: $01$ followed by 21 zeros, so $f = 2^{21} = 2097152$. Answer: `[0, 124, 2097152]`. Check: $(1 + 2^{21}/2^{23}) \cdot 2^{-3} = 1.25/8 = 0.15625$.

**Cancellation.** Evaluate $g(x) = (1 + x) - 1$ in float32 at $x = 3 \cdot 10^{-8}$. The gap at 1 is $2^{-23} \approx 1.19 \times 10^{-7}$, so $1 + x$ lies between $1$ and $1 + 2^{-23}$, closer to 1 (since $x < 2^{-24} \approx 5.96 \times 10^{-8}$), and rounds to 1. Then $1 - 1 = 0$: the answer is $0$, a 100% relative error, although the true value is $3 \cdot 10^{-8}$. Algebra gives the stable form $g(x) = x$. The same structure, a sum that has already rounded away the part you are about to subtract back out, is q6.

## 4. The problem set

Write each answer in `solve/S-M09a.toml`; lettered parts are their own tables:

```toml
[q1.a]
answer = "[0, 127, 0]"
[q2.a]
answer = "2^-23"
[q8.b]
answer = "nan"
[q11]
proof = "S-M09a/q11.md"
```

Give exact values as powers of two or fractions, not decimals. Only q6(b) accepts a decimal, within a relative tolerance of $10^{-6}$.

<!-- ol:problems S-M09a -->

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

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Forgetting the exponent bias, or storing the implicit 1 | `decompose_f32` disagrees with the bit view | q1 (canaries) |
| Confusing the gap at 1 with the unit roundoff | tolerances off by a factor 2 | q2 (canary 2^-24) |
| Thinking fp16 reaches 65536 | overflow to inf in mixed precision | q3 (canaries 65536, 65535) |
| Rounding ties up instead of to even | bf16 results differ from torch in the last bit | q4 (canaries 1 + 2^-7) |
| Trusting the naive formula's output | a 0 where the answer is $5 \times 10^{-5}$ | q6 (canary: the true value for part a) |
| Exponentiating before subtracting the max | NaN from inf / inf at logit 1000 | q8 (canary inf) |
| One-pass variance on data far from 0 | negative or garbage variances | q9 (canary one) |
| Accumulating a long sum naively in float32 | NLL totals that stop growing | q10 (canary: the exact sum for part a) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M05` | proof habits for q11 |
| Forward | `M09.1` | `decompose_f32`, `ulp`, `round_to_bf16`, `round_to_fp16` are q1 to q4 as code |
| Forward | `M09.2` | `logsumexp`, `softmax`, `kahan_sum`, `pairwise_sum` are q8, q10, q11 as code |
| Forward | `L0.3` | fused cross-entropy computes `log_softmax` with the max shift |
| Forward | `L7.9` | loads bf16 weights (q2 to q4) |
| Forward | `L11.1` | mixed precision: why bf16 needs no loss scaling and fp16 does (q3) |
| Forward | `S-M09b` | error bounds and condition numbers make q6's loss of digits quantitative |
