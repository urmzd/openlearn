# Exponents, logs, change of base, units of information

## Overview

| | |
|---|---|
| **Module** | `M00.1` · build · Python · Pass 2 · 1 to 2 h reading |
| **Solution** | [`course/solutions/python/tinyllm/num/units.py`](../../../../course/solutions/python/tinyllm/num/units.py): `LN2`, `nats_to_bits`, `bits_to_nats`, `log_base`, `bits_per_byte` |
| **Contract** | [`course/contracts/py/tinyllm/num/units.pyi`](../../../../course/contracts/py/tinyllm/num/units.pyi) |
| **Needs** | nothing to call. Reading: `lang.01` (numpy arrays), `L0.0` (the bigram's negative log-likelihood, used as the worked example) |
| **Used by** | `M11.2` perplexity and the bits-per-byte accumulator calls `bits_per_byte` · `M12.6` decibels and Whisper's log-mel call `log_base(x, 10)` · later `L1.6` tokenizer metrics · `L6.7` the model-zoo table · `C1` the capstone report |
| **Optional depth** | OpenStax, *Precalculus 2e* (free), ch. 6 (exponential and logarithmic functions); MacKay, *Information Theory, Inference, and Learning Algorithms* (free), ch. 2.4 (the information content of an outcome) |

## Key Takeaways

- A **logarithm undoes an exponential**: $\log_b y = x$ exactly when $b^x = y$. It turns products into sums, which is why every loss in the course adds log-probabilities instead of multiplying probabilities.
- **Change of base** is one division: $\log_b x = \ln x / \ln b$. Inverting it, or mixing $\log_{10}$ with $\ln$, gives numbers that look plausible and are wrong.
- The **base of the log is the unit of information**: base $e$ gives nats, base 2 gives bits, and one bit is $\ln 2 \approx 0.693$ nats.
- **Bits per byte** divides a text's total cost in nats by $n_\text{bytes} \ln 2$. It does not depend on how the text was split into tokens, so it is the one number that compares a byte bigram with a BPE transformer.
- A model that knows nothing about bytes pays exactly **8 bits per byte**; anything trained must do better.

## How to read this chapter

Sections 1 and 2 build exponents, logarithms, change of base, and units of information from first principles. Section 3 prices the `L0.0` bigram in bits per byte by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each line of the solution before reading it.

---

## 1. Why now

The bigram model from `L0.0` reports `nll`: the mean negative log-likelihood, in **nats per predicted byte**, because numpy's `log` is the natural logarithm. That number is meaningful only next to another model scored on the same units of text. In Pass 3 the tokenizers (`L1.*`) split text into multi-byte tokens, and a model over BPE tokens reports nats per *token*: a smaller vocabulary of longer pieces makes every per-token number larger, even for a better model. The passes `L2` and `L3` compare an n-gram model, a neural model, and an LSTM trained on different token streams, and the model zoo of `L6.7` ranks every architecture in one table. All of them convert to **bits per byte** through the function this module defines. This module also fixes the vocabulary the whole math track uses: exponent, logarithm, base, nat, bit.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $b$ | a base: a real number with $b > 0$ and $b \ne 1$ | `float` |
| $n, m$ | whole numbers (integers) | `int` |
| $b^x$ | $b$ raised to the power $x$ | `float` |
| $e$ | Euler's number, $2.718281828\ldots$ | `math.e` |
| $\exp(x) = e^x$ | the natural exponential function | |
| $\log_b y$ | the logarithm of $y > 0$ in base $b$: the $x$ with $b^x = y$ | `float` |
| $\ln y$ | the natural logarithm, $\log_e y$ (numpy's `np.log`) | `float` |
| $p$ | a probability, $0 \le p \le 1$ | `float` |
| $I_b(p)$ | the information content (surprisal) of an outcome of probability $p$: $-\log_b p$ | `float` |
| $N$ | the total negative log-likelihood of a text, in nats | `float` |
| $n_\text{bytes}$ | the length of the text in UTF-8 bytes | `int` |
| $\mathrm{bpb}$ | bits per byte, $N / (n_\text{bytes} \ln 2)$ | `float` |

### 2.1 Exponents

For a whole number $n \ge 1$, $b^n$ is $b$ multiplied by itself $n$ times: $2^3 = 2 \cdot 2 \cdot 2 = 8$. Counting factors gives the three laws of exponents:

$$b^m b^n = b^{m+n}, \qquad (b^m)^n = b^{mn}, \qquad (ab)^n = a^n b^n .$$

The first law forces the meaning of the other exponents. $b^0 b^n = b^{n}$, so $b^0 = 1$. $b^{-n} b^{n} = b^0 = 1$, so $b^{-n} = 1/b^n$. $(b^{1/n})^n = b^1$, so $b^{1/n} = \sqrt[n]{b}$, the positive number whose $n$-th power is $b$ (this is why $b > 0$: $(-8)^{1/2}$ is not a real number). Fractions follow: $8^{2/3} = (8^{1/3})^2 = 4$. Filling in the gaps between the fractions continuously gives $b^x$ for every real $x$, a function that is always positive, increasing when $b > 1$ and decreasing when $0 < b < 1$.

One base is special. $e = 2.71828\ldots$ is the base whose exponential grows, at every $x$, at a rate equal to its own value; `M01.1` proves this from the derivative. $\exp(x) = e^x$ is the exponential the rest of the course uses (softmax, the sigmoid, every probability a model outputs).

### 2.2 Logarithms undo exponentials

Because $b^x$ is strictly increasing (or strictly decreasing) and takes every positive value exactly once, it has an inverse: the **logarithm**.

$$\log_b y = x \quad\Longleftrightarrow\quad b^x = y, \qquad y > 0 .$$

So $\log_2 8 = 3$, $\log_{10} 0.001 = -3$, and $\log_b 1 = 0$ in every base. Each law of exponents becomes a law of logarithms (take $\log_b$ of both sides):

$$\log_b(xy) = \log_b x + \log_b y, \qquad \log_b(x^k) = k \log_b x, \qquad \log_b(x/y) = \log_b x - \log_b y .$$

The first is the reason models are trained on log-probabilities. The probability of a text is a product of one probability per token, thousands of numbers below 1 multiplied together, which underflows to 0 in floating point after a few hundred tokens. Its logarithm is a sum, which stays representable and is what `L0.0` already computes.

As $y$ approaches 0, $\log_b y$ falls without bound when $b > 1$: $\log_b 0 = -\infty$. An outcome a model calls impossible ($p = 0$) is infinitely surprising. That is a value, not an error, and `log_base` returns it.

### 2.3 Change of base

Write $x = b^{\log_b x}$ and take the natural log of both sides: $\ln x = \log_b x \cdot \ln b$ (power law). Divide:

$$\log_b x = \frac{\ln x}{\ln b} .$$

Every logarithm is the natural log divided by a constant. Example: $\log_8 32 = \ln 32 / \ln 8 = (5 \ln 2)/(3 \ln 2) = 5/3$, which checks out because $8^{5/3} = (8^{1/3})^5 = 2^5 = 32$. Two consequences: $\ln b = 0$ when $b = 1$, so base 1 is not a base (the formula divides by zero); and $\log_{1/b} x = -\log_b x$, because $\ln(1/b) = -\ln b$.

In floating point the division rounds, so change of base is accurate to about one unit in the last place, not exact: `math.log(2**29) / math.log(2)` is `29.000000000000004`. When you need an exact power of two (counting bits of a block size), use integer arithmetic or `np.log2`; results are compared with float64 tolerances, not equality.

### 2.4 Units of information

An outcome with probability $p$ carries $I_b(p) = -\log_b p$ units of **information**, also called its surprisal: a certain outcome ($p = 1$) carries none, and rarer outcomes carry more. The base names the unit:

| Base | Unit | One fair coin flip ($p = 1/2$) | One uniform byte ($p = 1/256$) |
|---|---|---|---|
| 2 | bit | 1 bit | 8 bits |
| $e$ | nat | $\ln 2 = 0.693$ nats | $\ln 256 = 5.545$ nats |

By change of base, $I_2(p) = I_e(p) / \ln 2$: **to turn nats into bits, divide by $\ln 2$**, and multiply by $\ln 2$ to go back. One nat is $1/\ln 2 = 1.4427$ bits.

A language model assigns each next token a probability, and its total negative log-likelihood on a text is the sum of the surprisals, $N = -\sum_t \ln p_t$ nats. Dividing by the number of tokens gives nats per token, which depends on the tokenizer. Dividing by the number of **bytes** does not: the same text has the same bytes however it is cut. So the course compares models by

$$\mathrm{bpb} = \frac{N}{n_\text{bytes} \ln 2} \quad\text{bits per byte.}$$

A model that gives all 256 byte values probability $1/256$ scores exactly $\log_2 256 = 8$; English text under a good model scores around 1. `M11.2` adds perplexity, $e^{N/n_\text{tokens}}$, and explains why bits per byte and cross-entropy are the same idea.

## 3. Worked example by hand

**The bigram of `L0.0`, priced in bits per byte.** The `L0.0` chapter fits an add-one bigram to the text "abbacab" (ids 0, 1, 1, 0, 2, 0, 1 over the alphabet a, b, c) and gets these next-symbol probabilities:

| context | P(a) | P(b) | P(c) |
|---|---|---|---|
| a | 1/6 | 3/6 | 2/6 |
| b | 2/5 | 2/5 | 1/5 |
| c | 2/4 | 1/4 | 1/4 |

The first symbol is never predicted, so the model makes 6 predictions, one per remaining byte:

| step | pair | $p$ | nats, $-\ln p$ | bits, $-\log_2 p$ |
|---|---|---|---|---|
| 1 | a to b | 1/2 | 0.693147 | 1.000000 |
| 2 | b to b | 2/5 | 0.916291 | 1.321928 |
| 3 | b to a | 2/5 | 0.916291 | 1.321928 |
| 4 | a to c | 1/3 | 1.098612 | 1.584963 |
| 5 | c to a | 1/2 | 0.693147 | 1.000000 |
| 6 | a to b | 1/2 | 0.693147 | 1.000000 |
| | **total** | | $N = 3\ln 2 + 2\ln 2.5 + \ln 3 = 5.010635$ | 7.228819 |

Then $\mathrm{bpb} = 5.010635 / (6 \cdot 0.693147) = 5.010635 / 4.158883 = 1.204803$. The bits column gives the same answer directly: $7.228819 / 6 = 1.204803$. Per byte in nats it is $5.010635 / 6 = 0.835106$, the `nll` that `L0.0`'s hand example computes, and $0.835106 / \ln 2 = 1.204803$ again. Against the 8 bits per byte of knowing nothing, the bigram has learned something about this (tiny) text.

**Change of base.** $\log_8 32 = 5/3$ (section 2.3), $\log_{10} 1000 = 3$, and $\log_{1/2} 8 = -3$ because $(1/2)^{-3} = 8$.

## 4. The interface

```python
# python/tinyllm/num/units.py
LN2 = math.log(2.0)
def nats_to_bits(x: float) -> float: ...                 # x / ln 2
def bits_to_nats(x: float) -> float: ...                 # x * ln 2
def log_base(x: ArrayLike, b: float) -> NDArray: ...     # ln x / ln b, float64, x's shape
def bits_per_byte(nll_nats_sum: float, n_bytes: int) -> float: ...
```

`log_base` returns $-\infty$ at $x = 0$ (for $b > 1$) without a warning, and raises `ValueError` for a base that is not positive, finite, and different from 1, and for negative or NaN inputs. `bits_per_byte` raises `ValueError` unless `n_bytes` is a positive integer and the sum is a non-negative number. The full rules are in the contract.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3: "abbacab" costs 1.204803 bits per byte | everyone agrees on the definition |
| a uniform byte model is 8 bits | unit | $n \ln 256$ nats over $n$ bytes is 8 bits per byte | the baseline every model in the zoo must beat |
| one bit is $\ln 2$ nats | unit | $\ln 2$ nats is one bit, 1 nat is 1.4427 bits, and back | the factor every later conversion uses |
| nats to bits at full precision | golden | six values against mpmath at 60 digits | full float64 precision, not an approximation |
| the conversions round-trip | property | `bits_to_nats(nats_to_bits(x)) == x` from $10^{-18}$ to $10^{17}$ | the two conversions are inverses |
| change of base by hand | unit | $\log_8 32 = 5/3$, $\log_{10} 1000 = 3$, $\log_{1/2} 8 = -3$ | the section 2.3 derivation |
| `log_base` at full precision | golden | 66 pairs, $x$ from $10^{-300}$ to $10^{300}$, bases 0.5 to 256 | float64 all the way through |
| the log laws | property | product, power, and reciprocal-base laws on random inputs | the identities later chapters rewrite losses with |
| shape and dtype | boundary | arrays keep their shape, results are float64, scalars stay scalars | `L1.6` and `M11.2` pass whole arrays |
| log of zero | boundary | $\log_2 0 = -\infty$, $\log_{1/2} 0 = +\infty$, no warning | a zero-probability token is a value, not a crash |
| bad bases | boundary | bases 1, 0, $-2$, $\infty$, NaN raise `ValueError` | base 1 silently divides by zero |
| negative or NaN $x$ | boundary | negative or NaN $x$ raises `ValueError` | a NaN surprisal poisons every later sum |
| bits per byte at full precision | golden | four totals against mpmath | the conversion `L2` and `L3` compare models with |
| bits per byte is per byte | property | doubling text and cost keeps bpb; doubling cost alone doubles it | the byte count really divides |
| bad bits-per-byte input | boundary | zero, negative, or fractional byte counts and negative or NaN totals raise `ValueError` | an empty text has no bits per byte |

## 5. The solution, step by step

### 5.1 From first principles

1. **One constant carries every conversion.** By change of base, $\log_2 p = \ln p / \ln 2$, so a quantity in nats becomes bits by dividing by $\ln 2$ (2.4). Compute $\ln 2$ once, at full float64 precision, and name it.
2. **Every logarithm is a natural log divided by a constant** (2.3). So `log_base` needs only `np.log` and one division by $\ln b$, and it inherits numpy's float64 accuracy and array handling.
3. **The base must make the division meaningful.** $\ln b$ is 0 at $b = 1$, undefined for $b \le 0$, and infinite for $b = \infty$; each would return a silent $\pm\infty$ or NaN. So the base is validated before any computation.
4. **Zero is a legitimate input; negatives and NaN are not.** $\log 0 = -\infty$ is the correct surprisal of an impossible event (2.2), so it is returned without a warning. A negative probability or a NaN is a bug upstream, and it would poison every sum after it, so it raises.
5. **Bits per byte is total nats over total bytes, in bits.** $\mathrm{bpb} = N / (n_\text{bytes} \ln 2)$: one multiplication in the denominator, with the byte count validated as a positive integer, because an empty text has no rate.

### 5.2 The constant and the scalar conversions

```python
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

LN2 = math.log(2.0)  # nats in one bit


def nats_to_bits(x: float) -> float:
    # One bit is ln 2 nats, so a quantity in nats is x / ln 2 bits.
    return float(x) / LN2


def bits_to_nats(x: float) -> float:
    return float(x) * LN2
```

`math.log(2.0)` is the correctly rounded float64 value of $\ln 2$; a typed literal such as `0.693` would cost 13 digits. `float(x)` makes the result a plain Python float even when a numpy scalar is passed. Dividing, not multiplying, in `nats_to_bits` is the whole content of pitfall 1: a nat is larger than a bit, so a count in bits must be the larger number.

### 5.3 `log_base`

```python
def log_base(x: ArrayLike, b: float) -> NDArray:
    b = float(b)
    # Reject every base where ln b is 0, undefined, or infinite. Written as
    # "not (good)", so NaN (for which every comparison is False) is rejected too.
    if not (b > 0.0 and b != 1.0 and math.isfinite(b)):
        raise ValueError(f"the base must be positive, finite, and not 1, got {b}")
    # float64 always: float32 would underflow 1e-300 to 0 and keep 7 digits.
    a = np.asarray(x, dtype=np.float64)
    if np.isnan(a).any() or (a < 0).any():
        raise ValueError(
            "log_base needs x >= 0 everywhere (the log of a negative number is not real)"
        )
    # Change of base: log_b x = ln x / ln b. log(0) = -inf is the right answer
    # (an impossible event is infinitely surprising), so silence numpy's warning.
    with np.errstate(divide="ignore"):
        return np.log(a) / math.log(b)
```

The order of the checks matters. The base is checked first, so a bad base is reported even for a bad `x`. `np.asarray(..., dtype=np.float64)` converts lists, scalars, and integer arrays alike, and keeps the shape; a 0-d input gives a numpy scalar out, so scalars stay scalars. The `errstate` block suppresses only the divide-by-zero warning that `np.log(0)` raises, and only inside this function; NaN-producing inputs never reach it because they were rejected. For $b < 1$, $\ln b < 0$, so $\log_b 0 = -\infty / (\text{negative}) = +\infty$, which is exactly the boundary case of section 4: no special case is needed.

### 5.4 `bits_per_byte`

```python
def bits_per_byte(nll_nats_sum: float, n_bytes: int) -> float:
    # bool is a subclass of int in Python: True would pass as 1 byte.
    # int(n) != n rejects 2.5; n < 1 rejects empty and negative texts.
    if isinstance(n_bytes, bool) or int(n_bytes) != n_bytes or n_bytes < 1:
        raise ValueError(f"n_bytes must be a positive integer, got {n_bytes!r}")
    nll = float(nll_nats_sum)
    # A sum of surprisals is >= 0; NaN means something upstream broke.
    if math.isnan(nll) or nll < 0:
        raise ValueError(f"a summed negative log-likelihood is >= 0, got {nll}")
    # Total nats -> total bits, then shared by every byte of the text.
    return nll / (int(n_bytes) * LN2)
```

The function takes the **sum** of the negative log-likelihood, not the mean, on purpose: the caller knows its token count, this function knows only bytes, and a sum is the one quantity that does not depend on how the text was tokenized. On section 3's numbers, `bits_per_byte(5.010635, 6)` is $5.010635 / (6 \times 0.693147) = 1.204803$. An infinite sum (a model that assigned some byte probability 0) passes validation and returns $\infty$, consistent with `log_base`.

Full solution: [`units.py`](../../../../course/solutions/python/tinyllm/num/units.py).

### 5.5 Why this solution

- **Time and space.** The scalar functions are $O(1)$; `log_base` is $O(n)$ in the array size, one `log` and one division per element, plus two $O(n)$ validation passes.
- **Change of base over `np.log2` and `np.log10`.** Those are slightly more accurate for their own bases, but `log_base` must support any base (Whisper's base 10, base $1/2$ in the boundary cases) with one code path. Accuracy is within a unit in the last place, which the golden cases allow.
- **Raising over returning NaN.** NaN propagates silently through sums and comparisons; a `ValueError` at the source names the problem where it happened.
- **Returning $-\infty$ for $\log 0$ over raising.** A zero probability is a real outcome of a badly smoothed model, and evaluation must report it, not crash.
- **A sum and a byte count over a mean.** A mean per token would bake the tokenizer into the input.

### 5.6 Explain it back

1. Why does `nats_to_bits` divide by $\ln 2$ rather than multiply?
   <details><summary>Answer</summary>One bit is $\ln 2 \approx 0.693$ nats, so a nat is the larger unit and the same quantity is a larger number in bits. $x$ nats $= x / \ln 2$ bits. Multiplying would shrink every result by a further factor of $(\ln 2)^2$.</details>
2. How does the base check reject `b = float("nan")`, and why is it written as "not (all conditions for a good base)"?
   <details><summary>Answer</summary>Every comparison with NaN is False, so <code>b > 0.0</code> is False and the whole <code>and</code> is False; <code>not</code> makes it raise. Written the other way round, as a list of bad cases (<code>b <= 0 or b == 1</code>), NaN would make every test False and slip through unless a separate check caught it. Stating the valid set fails safe.</details>
3. What does `log_base(0, 0.5)` return, and why does the code need no special case for it?
   <details><summary>Answer</summary>$+\infty$. <code>np.log(0)</code> is $-\infty$ and <code>math.log(0.5)</code> is negative, so the quotient is $+\infty$: the sign falls out of the change-of-base formula.</details>
4. Why does `bits_per_byte` take a total in nats and a byte count instead of a mean per token?
   <details><summary>Answer</summary>The total is tokenizer-independent and so is the byte count; their ratio is what makes a byte bigram and a BPE transformer comparable. A per-token mean would need the token count to undo, and mixing it up is exactly pitfall 5.</details>
5. Why compute in float64 even when the input is a float32 array?
   <details><summary>Answer</summary>float32 underflows below about $10^{-38}$ (so $10^{-300}$ becomes 0 and its log $-\infty$) and keeps about 7 significant digits. The golden cases span $10^{-300}$ to $10^{300}$ at full float64 precision.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. multiplying by $\ln 2$ to get bits (or dividing to get nats) | every number off by a factor of $(\ln 2)^2 \approx 0.48$; a 1.2 bpb model reports 0.58 | `nats_to_bits` divides by `LN2`, `bits_to_nats` multiplies, and `bits_per_byte` divides by `n_bytes * LN2` (5.2, 5.4) |
| 2. inverting change of base ($\ln b / \ln x$) or mixing logs ($\log_{10} x / \ln b$) | plausible numbers, wrong by a constant factor or a reciprocal | `np.log(a) / math.log(b)`: natural logs on both sides, $x$ on top (5.3) |
| 3. accepting base 1 | $\ln 1 = 0$: the division returns $\pm\infty$ or NaN without an error | the base is validated first: positive, finite, not 1 (5.3) |
| 4. treating $\log 0$ as an error | evaluating a model that assigns probability 0 to some byte crashes instead of reporting $\infty$ | `np.errstate(divide="ignore")` returns $\pm\infty$ without a warning (5.3) |
| 5. dividing by tokens, or not dividing at all | a BPE model looks worse than a byte model that is worse; bpb grows with text length | the input is a total in nats and the divisor is the byte count (5.4) |
| 6. computing in float32 | $10^{-300}$ underflows to 0 and the log becomes $-\infty$; 7 digits instead of 16 | `np.asarray(x, dtype=np.float64)` (5.3) |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.0` | its `nll` (nats per predicted byte) is section 3's worked example (reading) |
| Back | `lang.01` | numpy arrays and dtypes (reading) |
| Forward | `M11.2` | the perplexity and NLL accumulator reports `bpb` through `bits_per_byte` |
| Forward | `M12.6` | `power_to_db` and `log_mel_whisper` take base-10 logs with `log_base` |
| Forward | `L1.6` | tokenizer metrics: bytes per token and the bits-per-byte of each tokenizer's model |
| Forward | `L6.7` | the model-zoo table ranks every architecture by bits per byte |
| Forward | `C1` | the capstone report's headline number |
| Forward | `M00.2`, `M00.3`, `M00.4` | use exponentials and logs in their derivations (reading) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `bits_per_byte` | EleutherAI lm-evaluation-harness, perplexity tasks | reports `word_perplexity`, `byte_perplexity`, and `bits_per_byte` for every model on the same text | `lm_eval/api/metrics.py` (`bits_per_byte`) |
| bits per byte as the cross-tokenizer unit | The Pile (Gao et al., 2020) | reports BPB so that models with different tokenizers land on one scale | the paper's evaluation of GPT-2 and GPT-3 on Pile components |
| `log_base` | numpy `log2`, `log10`, `log1p`, `logaddexp` | exact-for-powers-of-two `log2`, accuracy near 1 (`log1p`, used in `M00.3`), and stable sums of exponentials (`M09.2`) | numpy reference, "Exponents and logarithms" |
