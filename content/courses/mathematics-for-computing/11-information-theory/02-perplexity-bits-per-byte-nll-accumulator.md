# Perplexity, bits per byte, NLL accumulator

## Overview

| | |
|---|---|
| **Module** | `M11.2` · build · Python · Pass 3 · 2 to 3 h |
| **Solution** | [`python/tinyllm/info/ppl.py`](../../../../course/solutions/python/tinyllm/info/ppl.py): `perplexity`, `NLLAccumulator` (`add`, `merge`, `result`) |
| **Contract** | [`course/contracts/py/tinyllm/info/ppl.pyi`](../../../../course/contracts/py/tinyllm/info/ppl.pyi) |
| **Needs** | `M00.1` `bits_per_byte` (`tinyllm/num/units.py`) · reading: `M11.1` cross-entropy, `M09.2` compensated sums |
| **Used by** | `L1.6` tokenizer metrics · later: `L2.1`, `L2.2`, `L3.6` model perplexities, `L6.7` the model-zoo bpb table, `L8.5` quantization budgets, `C1` |
| **Optional depth** | Cover and Thomas, *Elements of Information Theory* (2nd ed.), ch. 2 and 5; Gao et al., "The Pile" (2020), section 4 (bits per byte); Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 4 |

## Key Takeaways

- Perplexity is $e$ to the mean NLL per token: the geometric mean of $1/q$, so a model that is uniform over $k$ choices scores exactly $k$.
- Bits per byte divides the same total by the text's UTF-8 length instead of its token count, which makes it the one number that compares models with different tokenizers.
- A corpus mean is the total over all tokens divided by the token count, never the average of batch means, and padding is selected out, not multiplied by zero.
- Summing each batch exactly and carrying the rounding error across batches makes any split of the corpus give the same answer to float64 rounding.

## How to read this chapter

Sections 1 to 3 derive perplexity and bits per byte from the total NLL, show why batched evaluation needs totals, masks and compensated sums, and work every number by hand. Section 4 fixes the interface and lists what a correct implementation must handle. Section 5 walks through the solution, [`ppl.py`](../../../../course/solutions/python/tinyllm/info/ppl.py), and explains why each line is there. Try to predict each step before reading it.

---

## 1. Why now

The tracer bigram reports one number, `nll`: the mean negative log-likelihood of one text, in nats per token. Pass 3 breaks that number in two ways. First, tokenizers arrive: the byte tokenizer, BPE with 4096 merges (`L1.2`), Unigram (`L1.4`). A BPE model and a byte model scoring the same text report nats per *their* token, and a BPE token is three or four bytes, so their perplexities are not comparable; `L1.6`'s tokenizer metrics and the `C1` ablation need a per-byte number. Second, evaluation stops fitting in one call: the zoo (`L6.7`) scores $10^7$ tokens in padded batches, possibly on several workers. Averaging batch means, counting padding, or letting float rounding drift all change the reported number while every individual line of code looks right. This module turns per-token losses into perplexity and bits per byte, correctly, one batch at a time.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_1, \dots, x_n$ | the tokens of an evaluation text | `int64[n]` |
| $q_t$ | the probability the model gave the true token $x_t$ | `float` in $(0, 1]$ |
| $\ell_t = -\ln q_t$ | that token's negative log-likelihood, in nats | `float` $\ge 0$ |
| $S = \sum_t \ell_t$ | the total NLL of the text, in nats | `float` |
| $n$ | the number of scored tokens | `int` |
| $B$ | the number of UTF-8 bytes the scored tokens cover | `int` |
| $\bar\ell = S / n$ | the mean NLL per token (the cross-entropy, `M11.1`) | `float` |
| $\mathrm{PPL} = e^{\bar\ell}$ | perplexity | `float` $\ge 1$ |
| $\mathrm{bpb} = S / (B \ln 2)$ | bits per byte | `float` |
| $s$, $c$ | a running sum and its compensation (the low bits it lost) | `float64` |

**From NLL to perplexity.** The model gives the text probability $\prod_t q_t$ (each prediction conditioned on what came before). Its log is $-S$. The mean $\bar\ell = S/n$ is the cross-entropy between the text and the model, in nats per token. Perplexity undoes the log:

$$\mathrm{PPL} = e^{\bar\ell} = \Big(\prod_t \frac{1}{q_t}\Big)^{1/n},$$

the geometric mean of $1/q_t$. If the model spreads its mass evenly over $k$ tokens every time, $q_t = 1/k$, $\ell_t = \ln k$, and $\mathrm{PPL} = k$ exactly: perplexity is the effective number of tokens the model is choosing between. A perfect model scores 1. A model that gives a real token probability 0 has $\ell_t = \infty$ and $\mathrm{PPL} = \infty$ (the zero problem of `M07.2`). In float64, $e^{\bar\ell}$ overflows once $\bar\ell > \ln(\text{DBL\_MAX}) \approx 709.78$; an untrained model can get there, so the function returns $\infty$ instead of raising.

**Units.** One bit is $\ln 2$ nats (`M00.1`), so bits per token is $\bar\ell / \ln 2$, and $\mathrm{PPL} = 2^{\text{bits per token}}$.

**Bits per byte.** $S$ is the length, in nats, of the shortest code for the whole text under the model: an arithmetic coder driven by the model spends $-\log q_t$ for each token (`M11.3`). That total does not care how the text was cut into tokens; only the model's skill at predicting it matters. Dividing by the number of tokens brings the tokenizer back in. Dividing by the number of bytes does not, because every tokenizer sees the same $B$ bytes:

$$\mathrm{bpb} = \frac{S}{B \ln 2} = \text{bits per token} \times \frac{n}{B}.$$

A BPE model whose tokens average 4 bytes and a byte model can have per-token perplexities that differ by a factor of hundreds while their bits per byte say which one actually predicts the text better. `L6.7` reports bpb for every model in the zoo for this reason.

**One mean, many batches.** The corpus mean is $S/n$ with $S$ and $n$ summed over every batch. The average of batch means, $\frac1m \sum_j S_j / n_j$, weights each batch equally, so a short last batch or heavily padded batches pull it away from $S/n$; the two agree only when every batch has the same token count. So the accumulator keeps the two totals, never a running mean.

**Masks select.** Padded batches have slots with no real token, whose loss is garbage: a loss against a pad id, $\infty$, or NaN. A mask marks the slots that count. Multiplying by the mask looks equivalent and is not, because $\mathrm{NaN} \times 0 = \mathrm{NaN}$ and $\infty \times 0 = \mathrm{NaN}$; selecting the masked values (`x[mask]`) never touches them. The token count is the number of selected slots, not the batch size.

**Summing $10^7$ numbers.** Adding a small number to a large float64 total rounds away its low bits: near 1.0 the spacing is $2.2 \times 10^{-16}$, so $1 + 10^{-16} = 1$ exactly. Over $10^7$ tokens the lost bits add up to changes in the last digits of a reported perplexity, and worse, the answer depends on batch size and order. Two tools fix it. Within a batch, `math.fsum` returns the exactly rounded sum of the batch (after widening float32 losses to float64: summing $10^5$ float32 values in float32 loses about seven digits). Across batches, Neumaier's compensated summation keeps a second float $c$: for each batch total $x$,

$$t = s + x, \qquad c \mathrel{+}= \begin{cases} (s - t) + x & \lvert s \rvert \ge \lvert x \rvert \\ (x - t) + s & \text{otherwise} \end{cases}, \qquad s = t,$$

and the total is $s + c$. When $\lvert s \rvert \ge \lvert x \rvert$, $s - t$ is computed exactly, and $(s - t) + x$ is exactly the part of $x$ that $t$ dropped. This is the same idea as Kahan's sum in `M09.2`, run as a stream. `merge` adds another accumulator's $s$ and then its $c$, so shards scored on different workers combine without losing their low bits.

## 3. Worked example by hand

Score `the cat sat.` as four BPE tokens with the model probabilities below. The text is 12 UTF-8 bytes.

| token | `the` | ` cat` | ` sat` | `.` | total |
|---|---|---|---|---|---|
| bytes | 3 | 4 | 4 | 1 | $B = 12$ |
| $q_t$ | 1/4 | 1/2 | 1/8 | 1/2 | |
| $\ell_t = -\ln q_t$ | $2 \ln 2$ | $\ln 2$ | $3 \ln 2$ | $\ln 2$ | $S = 7 \ln 2 = 4.8520$ nats |

- Mean: $\bar\ell = 7 \ln 2 / 4 = 1.75 \ln 2 = 1.2130$ nats per token, which is 1.75 bits per token.
- Perplexity: $e^{1.75 \ln 2} = 2^{1.75} = 3.3636$. The model is as unsure as a fair choice among about 3.4 tokens.
- Bits per byte: $7 \text{ bits} / 12 \text{ bytes} = 0.5833$.

**A byte model on the same text.** Suppose a byte-level model spends 0.7 bits on each of the 12 bytes: 8.4 bits in all. Its perplexity per (byte) token is $2^{0.7} = 1.62$, which looks better than 3.36. Its bits per byte, 0.7, is worse than 0.5833. The BPE model predicts this text better; only bpb says so.

**Batches.** Score the same four tokens as two batches, `[the]` and `[ cat,  sat, ., PAD]` with mask `[1, 1, 1, 0]` and NaN in the pad slot. Batch means: $2 \ln 2$ and $(1 + 3 + 1)/3 \cdot \ln 2 = 1.667 \ln 2$. Their average is $1.833 \ln 2$: wrong. The totals give $S = 2\ln 2 + 5 \ln 2 = 7 \ln 2$ over $n = 1 + 3 = 4$ tokens: $1.75 \ln 2$, the same as one batch.

**Compensation.** Start at $s = 1$, $c = 0$ and add $10^{-16}$ ten times. Plain addition: $1 + 10^{-16}$ rounds to 1 every time, so the total stays 1. Neumaier: each step $t = 1$, $c \mathrel{+}= (1 - 1) + 10^{-16}$, so after ten steps $c = 10^{-15}$ and $s + c = 1.000000000000001$, the right answer.

These numbers are the first two cases in section 4.

## 4. The interface

```python
# python/tinyllm/info/ppl.py
def perplexity(nll_sum: float, n_tokens: int) -> float          # exp(nll_sum / n_tokens); inf on overflow
class NLLAccumulator:
    def __init__(self) -> None
    def add(self, nll: ArrayLike, mask: ArrayLike | None = None, n_bytes: int = 0) -> None
    def merge(self, other: "NLLAccumulator") -> None
    def result(self) -> dict[str, float]
    # {"nll_sum", "n_tokens", "n_bytes", "nll_mean", "ppl", "bits_per_token", "bpb"}
```

The solution calls `bits_per_byte` from `M00.1` for `bpb`; with no bytes added, `bpb` is NaN. A counted NaN or negative loss is a `ValueError`; a counted $+\infty$ is allowed and makes `ppl` infinite.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-example sentence | unit | $7 \ln 2$, 1.75 bits per token, $\mathrm{PPL} = 3.3636$, bpb 0.5833 | everyone agrees on every definition |
| two batches with padding | unit | two batches with a NaN pad give the one-batch numbers | the mean of means is wrong |
| a uniform model has perplexity $k$ | property | $\mathrm{PPL} = k$ for uniform-$k$ losses, up to $k = 50\,257$ | perplexity as a number of choices |
| perplexity overflow | boundary | 800 nats per token and an infinite total give $\infty$ | untrained models in the zoo |
| bad `perplexity` arguments | boundary | zero tokens, a negative or NaN total are rejected | sign bugs upstream |
| masked positions hold anything | boundary | NaN, $\infty$, and negative values in masked slots; 0/1 masks | padded batches in `L6.7` |
| bad batches | boundary | mask shape, counted NaN or negative, negative bytes are rejected; a counted $\infty$ is allowed | caller bugs fail loudly |
| empty totals | boundary | no tokens raises; no bytes gives NaN bpb | a table shows "missing", not "perfect" |
| streaming equals one batch | property | random splits in reverse order match one batch to $10^{-13}$ | the number does not depend on batch size |
| compensated sum over many adds | boundary | $1 + 10^5 \times 10^{-15}$ to within 2 ulps | drift over $10^7$ tokens |
| float32 batches are widened | boundary | $10^5$ float32 losses summed to $10^{-15}$ | losses arrive as float32 |
| merge equals one accumulator | property | merged shards equal one accumulator, low bits included; `other` unchanged | sharded evaluation in `dur.11` |
| bits per byte compares tokenizers | unit | the byte model has lower perplexity and higher bpb | `L1.6`, the `C1` tokenizer ablation |
| long golden cases | golden | three long cases against totals computed to 50 digits | independent of float64 rounding |

## 5. The solution, step by step

### 5.1 From first principles

1. **Every reported number is a function of three totals.** Mean NLL, perplexity, bits per token and bits per byte are all computed from $S$, $n$ and $B$. So the state of an evaluation is exactly those three numbers, and `result()` derives everything else at the end. A running mean is not a valid state: it cannot be combined correctly with a batch of a different size.
2. **Totals combine by addition, so batches and shards combine by addition.** `add` adds one batch's contribution and `merge` adds another accumulator's. Any split of the corpus gives the same totals, which is the property that makes a number reported from batched or sharded evaluation trustworthy.
3. **Only real tokens enter the totals.** A mask decides which slots are real, and the safe way to apply it is selection, `x[keep]`, because the values in pad slots are arbitrary (NaN, $\infty$) and arithmetic with them, even times 0, can poison the sum. $n$ is then the number of selected entries.
4. **The sum must not depend on how it was split.** In floating point, addition is not associative, so step 2 holds exactly only if rounding is controlled. Two levels: `math.fsum` makes each batch's total the correctly rounded exact sum, and Neumaier's compensated addition carries the rounding error of adding batch totals into a second float. With both, the result matches one big exact sum to within a couple of ulps, whatever the split and order.
5. **Bad input should fail where it enters, and $\infty$ is not bad input.** A counted NaN or negative loss means a bug upstream (a loss computed against the wrong target, a sign error), so `add` raises. A counted $+\infty$ is a legitimate result: the model gave a real token probability 0. It must flow through to `ppl = inf`, which means the compensated sum needs an infinity path, since $\infty - \infty$ in the compensation step would be NaN.
6. **Perplexity is $e^{S/n}$, computed so it cannot crash a report.** `math.exp` raises on overflow; an untrained model's mean can exceed 709.78 nats, so overflow returns $\infty$.

### 5.2 The code, annotated

```python
def perplexity(nll_sum: float, n_tokens: int) -> float:
    if n_tokens < 1:
        raise ValueError(
            f"perplexity needs at least one token, got n_tokens = {n_tokens}"
        )
    # NaN fails "nll_sum < 0" silently, so it needs its own test.
    if math.isnan(nll_sum) or nll_sum < 0:
        raise ValueError(
            f"nll_sum must be a non-negative number of nats, got {nll_sum!r}"
        )
    try:
        return math.exp(nll_sum / n_tokens)   # an infinite total gives exp(inf) = inf
    except OverflowError:  # a mean above ln(DBL_MAX) = 709.78 nats
        return math.inf
```

The state is the three totals of step 1, with $S$ split into a sum and its compensation:

```python
class NLLAccumulator:
    def __init__(self) -> None:
        self._sum = 0.0  # Neumaier: the total is _sum + _comp
        self._comp = 0.0
        self.n_tokens = 0
        self.n_bytes = 0

    def _accumulate(self, x: float) -> None:
        """Neumaier's compensated add of one float into the running total."""
        # Once anything is infinite the total is infinite; skip the error term,
        # where inf - inf would produce nan.
        if math.isinf(x) or math.isinf(self._sum):
            self._sum += x
            return
        t = self._sum + x
        # Subtract the larger operand from t first: that difference is exact,
        # so the bracket is exactly the part of the smaller operand t lost.
        if abs(self._sum) >= abs(x):
            self._comp += (self._sum - t) + x  # the low bits of x that t lost
        else:
            self._comp += (x - t) + self._sum
        self._sum = t
```

The branch on magnitudes is what distinguishes Neumaier from Kahan: Kahan's version assumes the running sum is the larger operand, and loses the correction when a single batch total is larger than everything summed so far (for example, the first big batch after a few tiny ones).

```python
    def add(
        self, nll: ArrayLike, mask: ArrayLike | None = None, n_bytes: int = 0
    ) -> None:
        x = np.asarray(nll, dtype=np.float64)       # widen float32 losses before summing
        if mask is None:
            keep = np.ones(x.shape, dtype=bool)
        else:
            keep = np.asarray(mask)
            if keep.shape != x.shape:               # no broadcasting: a wrong mask is a bug
                raise ValueError(
                    f"mask shape {keep.shape} differs from nll shape {x.shape}"
                )
            keep = keep.astype(bool)                # accepts 0/1 integer masks
        if n_bytes < 0:
            raise ValueError(f"n_bytes must be >= 0, got {n_bytes}")
        # Select, never multiply: padding may hold NaN, and NaN * 0 is NaN.
        vals = x[keep]
        # Validate only what is counted: garbage in pad slots is expected.
        if np.isnan(vals).any() or (vals < 0).any():
            raise ValueError("a counted NLL is negative or NaN")
        self._accumulate(math.fsum(vals.tolist()))  # exact within the batch
        self.n_tokens += int(vals.size)             # real tokens, not slots
        self.n_bytes += int(n_bytes)
```

Validation happens before any state changes, so a rejected batch leaves the accumulator exactly as it was.

```python
    def merge(self, other: NLLAccumulator) -> None:
        # Add the other side's sum and then its compensation: dropping
        # other._comp would throw away the low bits it carefully kept.
        self._accumulate(other._sum)
        self._accumulate(other._comp)
        self.n_tokens += other.n_tokens
        self.n_bytes += other.n_bytes             # other is read, never modified

    def result(self) -> dict[str, float]:
        if self.n_tokens == 0:
            raise ValueError("no tokens were counted")
        # The compensation is meaningless next to an infinite sum (and could be nan).
        total = self._sum + self._comp if math.isfinite(self._sum) else self._sum
        mean = total / self.n_tokens
        return {
            "nll_sum": float(total),
            "n_tokens": float(self.n_tokens),
            "n_bytes": float(self.n_bytes),
            "nll_mean": float(mean),
            "ppl": perplexity(total, self.n_tokens),
            "bits_per_token": float(mean / math.log(2.0)),
            # Same total, divided by bytes: M00.1's S / (B ln 2). No bytes: NaN, not 0.
            "bpb": bits_per_byte(total, self.n_bytes) if self.n_bytes > 0 else math.nan,
        }
```

Full solution: [`ppl.py`](../../../../course/solutions/python/tinyllm/info/ppl.py).

### 5.3 Why this solution

- **Cost.** `add` is $O(k)$ for a batch of $k$ slots: one selection, one validation pass, and `fsum`, which is linear with a small constant (Shewchuk's partials list stays short for same-sign values). The state is four numbers, so `merge` is $O(1)$, and sharded evaluation costs nothing extra to combine.
- **Why both `fsum` and Neumaier.** `fsum` alone across the whole corpus would require holding every loss, or every partial, for the whole run. Neumaier alone inside a batch would work but is a Python loop over $10^5$ elements. Exact per batch plus compensated across batches gets near-exact totals with constant state.
- **Why not `np.sum`.** NumPy uses pairwise summation, which is good (error $O(\log k)$ ulps) but still depends on the batch shape, so the same corpus split differently gives different last digits. The goal here is a number that does not depend on batching.
- **Why not a running mean** with the update $\bar\ell \mathrel{+}= (x - \bar\ell)/n$. It is numerically fine but cannot be merged across shards without also carrying $n$, at which point it is the totals again, with extra rounding.
- **Why raise on NaN rather than skip it.** Skipping would silently report a perplexity over fewer tokens than the caller thinks were scored. Raising puts the bug next to its cause.
- **Tradeoff accepted.** `vals.tolist()` copies the batch into Python floats for `fsum`, which is slower than a vectorized sum. Evaluation is dominated by the model's forward pass, so exactness wins.

### 5.4 Explain it back

1. Two batches have means 2 and 1 nats with 1 and 3 tokens. What is the corpus mean, and what does the mean of means report?
   <details><summary>Answer</summary>
   The corpus mean is $(2 \cdot 1 + 1 \cdot 3)/4 = 1.25$ nats. The mean of means is $(2 + 1)/2 = 1.5$, because it gives the one-token batch the same weight as the three-token batch. Only totals give the right answer, which is why the accumulator stores $S$ and $n$.
   </details>
2. Why does `add` select with `x[keep]` instead of computing `(x * keep).sum()` and `keep.sum()`?
   <details><summary>Answer</summary>
   Pad slots can hold NaN or $\infty$, and $\mathrm{NaN} \cdot 0$ and $\infty \cdot 0$ are both NaN, so one pad slot would make the total NaN. Selection never reads those values into arithmetic. It also makes the validation apply only to counted slots.
   </details>
3. Why does `_accumulate` take a separate path when either operand is infinite?
   <details><summary>Answer</summary>
   With $x = \infty$, $t = \infty$ and the correction $(x - t) + s$ is $\infty - \infty = \mathrm{NaN}$, which would then turn the reported total into NaN. An infinite total is a legitimate result (a counted token with probability 0), so the code just adds and skips the error term; `result` likewise ignores the compensation when the sum is infinite.
   </details>
4. A byte model and a BPE model report per-token perplexities 1.62 and 3.36 on the same text. Which predicts the text better, and how does the accumulator tell?
   <details><summary>Answer</summary>
   Per-token perplexity cannot tell, because the tokens are different sizes. The accumulator also tracks $B$, the UTF-8 bytes covered, and reports $\mathrm{bpb} = S/(B \ln 2)$. In the section 3 example the BPE model has 0.583 bpb and the byte model 0.7, so the BPE model is better.
   </details>
5. What goes wrong if `merge` adds only `other._sum`?
   <details><summary>Answer</summary>
   The other shard's compensation holds the low bits its own additions lost. Dropping it makes a sharded run differ from a single run in the last digits, which is exactly the batch-dependence the design exists to remove.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. averaging batch means | the reported loss changes with the batch size and the last batch's length | stores totals $S$, $n$, $B$ and divides only in `result()`; see step 1 |
| 2. multiplying by the mask | NaN in a pad slot turns the total into NaN; pad slots are counted as tokens | selects with `x[keep]` and counts `vals.size`; see step 3 |
| 3. plain float64 addition across batches | $10^{-15}$-sized contributions vanish into a large total; the result depends on order | Neumaier's compensated add in `_accumulate` |
| 4. summing float32 losses in float32 | seven digits of a $10^5$-token batch lost | `np.asarray(nll, dtype=np.float64)`, then `math.fsum` |
| 5. `math.exp` of a huge mean | `OverflowError` in the middle of a zoo report | `perplexity` catches `OverflowError` and returns `math.inf` |
| 6. bits per byte divided by tokens, or left in nats | a bpb that is really bits per token, or off by $\ln 2$ | passes the nats total and the byte count to `M00.1`'s `bits_per_byte` |
| 7. a merge that drops the other side's compensation or bytes | sharded and single runs disagree in the last digits, or bpb is wrong after merging | `merge` accumulates `other._sum` and `other._comp` and adds both counts |
| 8. accepting a counted NaN, or reporting on zero tokens | NaN or a `ZeroDivisionError` reaches the report | `add` validates the selected values; `result` raises on zero tokens |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | `bits_per_byte` converts the nats total to bits per byte |
| Back | `M11.1` | cross-entropy: the mean NLL is the cross-entropy of the text under the model |
| Back | `M09.2` | Kahan's compensated sum, here run as a stream across batches |
| Forward | `L1.6` | tokenizer metrics report bytes per token and the bpb of a reference model per tokenizer |
| Forward | `L2.1`, `L2.2`, `L3.6` | each statistical and recurrent model reports its validation perplexity and bpb through an accumulator |
| Forward | `L6.7` | the model zoo scores every family with one accumulator per model and prints bpb |
| Forward | `L8.5`, `C1` | quantization is allowed a fixed bpb increase; the capstone's training curve and ablations are in bpb |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `result()["bpb"]` | EleutherAI lm-evaluation-harness | `bits_per_byte`, `byte_perplexity`, and `word_perplexity` aggregations over documents, weighted by length | `lm_eval/api/metrics.py` |
| `perplexity` over a corpus | llama.cpp `perplexity` | strided windows that score only the second half of each context, so every token has enough history | `tools/perplexity/perplexity.cpp` |
| batch sums with `fsum` | CPython `math.fsum` | Shewchuk's exact partial sums, correctly rounded whatever the order | `Modules/mathmodule.c` (`math_fsum`) |
