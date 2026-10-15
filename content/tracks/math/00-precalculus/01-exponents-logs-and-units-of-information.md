<!-- ol:module M00.1 -->
# Exponents, logs, change of base, units of information

## Overview

| | |
|---|---|
| **Module** | `M00.1` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/num/units.py`: `nats_to_bits`, `bits_to_nats`, `log_base`, `bits_per_byte` |
| **Contract** | [`course/contracts/py/tinyllm/num/units.pyi`](../../../../course/contracts/py/tinyllm/num/units.pyi) |
| **Tests** | `course/tests/M00.1/test_units.py` (what they check: section 4) |
| **Needs** | nothing to call. Reading: `lang.01` (numpy arrays), `L0.0` (the bigram's negative log-likelihood, used as the worked example) |
| **Used by** | `M11.2` perplexity and the bits-per-byte accumulator calls `bits_per_byte` · `M12.6` decibels and Whisper's log-mel call `log_base(x, 10)` · later `L1.6` tokenizer metrics · `L6.7` the model-zoo table · `C1` the capstone report |
| **Milestone** | `MS-P2` (the Pass 2 gate: every math module of the pass passes `ol check`) |
| **Optional depth** | OpenStax, *Precalculus 2e* (free), ch. 6 (exponential and logarithmic functions); MacKay, *Information Theory, Inference, and Learning Algorithms* (free), ch. 2.4 (the information content of an outcome) |

## Key Takeaways

- A **logarithm undoes an exponential**: $\log_b y = x$ exactly when $b^x = y$. It turns products into sums, which is why every loss in the course adds log-probabilities instead of multiplying probabilities (`test_log_laws`).
- **Change of base** is one division: $\log_b x = \ln x / \ln b$. Inverting it, or mixing $\log_{10}$ with $\ln$, gives numbers that look plausible and are wrong (`test_change_of_base_hand_values`, `test_log_base_golden`).
- The **base of the log is the unit of information**: base $e$ gives nats, base 2 gives bits, and one bit is $\ln 2 \approx 0.693$ nats (`test_one_bit_is_ln2_nats`).
- **Bits per byte** divides a text's total cost in nats by $n_\text{bytes} \ln 2$. It does not depend on how the text was split into tokens, so it is the one number that compares a byte bigram with a BPE transformer (`test_hand_example_bits_per_byte`, `test_bits_per_byte_is_per_byte`).
- A model that knows nothing about bytes pays exactly **8 bits per byte**; anything trained must do better (`test_uniform_byte_model_is_eight_bits`).

## How to work this chapter

```bash
ol start M00.1              # stubs python/tinyllm/num/units.py into your repo
ol tests M00.1              # read the test catalog first: rung R0, you write no tests here
ol check M00.1              # exit code is the verdict
ol diff  M00.1              # after passing: your code against the reference
```

---

## 1. Why now

Your bigram model from `L0.0` reports `nll`: the mean negative log-likelihood, in **nats per predicted byte**, because numpy's `log` is the natural logarithm. That number is meaningful only next to another model scored on the same units of text. In Pass 3 your tokenizers (`L1.*`) split text into multi-byte tokens, and a model over BPE tokens reports nats per *token*: a smaller vocabulary of longer pieces makes every per-token number larger, even for a better model. Milestones `MS-L2` and `MS-L3` compare an n-gram model, a neural model, and an LSTM trained on different token streams, and the model zoo of `L6.7` ranks every architecture in one table. All of them convert to **bits per byte** through the function you write here. This module also fixes the vocabulary the whole math track uses: exponent, logarithm, base, nat, bit.

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

As $y$ approaches 0, $\log_b y$ falls without bound when $b > 1$: $\log_b 0 = -\infty$. An outcome a model calls impossible ($p = 0$) is infinitely surprising. That is a value, not an error, and your `log_base` returns it.

### 2.3 Change of base

Write $x = b^{\log_b x}$ and take the natural log of both sides: $\ln x = \log_b x \cdot \ln b$ (power law). Divide:

$$\log_b x = \frac{\ln x}{\ln b} .$$

Every logarithm is the natural log divided by a constant. Example: $\log_8 32 = \ln 32 / \ln 8 = (5 \ln 2)/(3 \ln 2) = 5/3$, which checks out because $8^{5/3} = (8^{1/3})^5 = 2^5 = 32$. Two consequences: $\ln b = 0$ when $b = 1$, so base 1 is not a base (the formula divides by zero); and $\log_{1/b} x = -\log_b x$, because $\ln(1/b) = -\ln b$.

In floating point the division rounds, so change of base is accurate to about one unit in the last place, not exact: `math.log(2**29) / math.log(2)` is `29.000000000000004`. When you need an exact power of two (counting bits of a block size), use integer arithmetic or `np.log2`; the tests compare with float64 tolerances, not equality.

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

Then $\mathrm{bpb} = 5.010635 / (6 \cdot 0.693147) = 5.010635 / 4.158883 = 1.204803$. The bits column gives the same answer directly: $7.228819 / 6 = 1.204803$. Per byte in nats it is $5.010635 / 6 = 0.835106$, the `nll` that `L0.0`'s test `test_hand_example_nll` checks, and $0.835106 / \ln 2 = 1.204803$ again. Against the 8 bits per byte of knowing nothing, the bigram has learned something about this (tiny) text. These numbers are `test_hand_example_bits_per_byte`.

**Change of base.** $\log_8 32 = 5/3$ (section 2.3), $\log_{10} 1000 = 3$, and $\log_{1/2} 8 = -3$ because $(1/2)^{-3} = 8$: `test_change_of_base_hand_values`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_bits_per_byte` | unit, smoke | section 3: "abbacab" costs 1.204803 bits per byte | you and the tests agree on the definition |
| `test_uniform_byte_model_is_eight_bits` | unit | $n \ln 256$ nats over $n$ bytes is 8 bits per byte | the baseline every model in the zoo must beat |
| `test_one_bit_is_ln2_nats` | unit, smoke | $\ln 2$ nats is one bit, 1 nat is 1.4427 bits, and back | the factor every later conversion uses |
| `test_nats_to_bits_golden` | golden | six values against mpmath at 60 digits | full float64 precision, not an approximation |
| `test_nats_bits_roundtrip` | property | `bits_to_nats(nats_to_bits(x)) == x` from $10^{-18}$ to $10^{17}$ | the two conversions are inverses |
| `test_change_of_base_hand_values` | unit | $\log_8 32 = 5/3$, $\log_{10} 1000 = 3$, $\log_{1/2} 8 = -3$ | the section 2.3 derivation |
| `test_log_base_golden` | golden | 66 pairs, $x$ from $10^{-300}$ to $10^{300}$, bases 0.5 to 256 | float64 all the way through |
| `test_log_laws` | property | product, power, and reciprocal-base laws on random inputs | the identities later chapters rewrite losses with |
| `test_log_base_keeps_shape_and_dtype` | boundary | arrays keep their shape, results are float64, scalars stay scalars | `L1.6` and `M11.2` pass whole arrays |
| `test_log_of_zero_is_infinite` | boundary | $\log_2 0 = -\infty$, $\log_{1/2} 0 = +\infty$, no warning | a zero-probability token is a value, not a crash |
| `test_bad_base_rejected` | boundary | bases 1, 0, $-2$, $\infty$, NaN raise `ValueError` | base 1 silently divides by zero |
| `test_negative_or_nan_x_rejected` | boundary | negative or NaN $x$ raises `ValueError` | a NaN surprisal poisons every later sum |
| `test_bits_per_byte_golden` | golden | four totals against mpmath | the conversion `MS-L2` and `MS-L3` compare models with |
| `test_bits_per_byte_is_per_byte` | property | doubling text and cost keeps bpb; doubling cost alone doubles it | the byte count really divides |
| `test_bits_per_byte_rejects_bad_input` | boundary | zero, negative, or fractional byte counts and negative or NaN totals raise `ValueError` | an empty text has no bits per byte |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. multiplying by $\ln 2$ to get bits (or dividing to get nats) | every number off by a factor of $(\ln 2)^2 \approx 0.48$; a 1.2 bpb model reports 0.58 | `test_one_bit_is_ln2_nats` (mutants `s01`, `s08`); `test_hand_example_bits_per_byte` (mutant `s04`, bpb left in nats) |
| 2. inverting change of base ($\ln b / \ln x$) or mixing logs ($\log_{10} x / \ln b$) | plausible numbers, wrong by a constant factor or a reciprocal | `test_change_of_base_hand_values` (mutants `s02`, `s03`) |
| 3. accepting base 1 | $\ln 1 = 0$: the division returns $\pm\infty$ or NaN without an error | `test_bad_base_rejected` (mutant `s05`) |
| 4. treating $\log 0$ as an error | evaluating a model that assigns probability 0 to some byte crashes instead of reporting $\infty$ | `test_log_of_zero_is_infinite` (mutant `s11`) |
| 5. dividing by tokens, or not dividing at all | a BPE model looks worse than a byte model that is worse; bpb grows with text length | `test_bits_per_byte_is_per_byte` (mutant `s12`) |
| 6. computing in float32 | $10^{-300}$ underflows to 0 and the log becomes $-\infty$; 7 digits instead of 16 | `test_log_base_golden` (mutant `s10`) |

## 6. Where it's used next

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
