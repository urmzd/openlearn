# Byte bigram: counts, logits, safetensors v0

Every layer of the system this book builds needs something to carry. The Rust engine in [Your first endpoint](../p10-serving/00-your-first-endpoint.md) needs a model directory to serve, the Go gateway in [the tracer gateway](../../ai-platform-engineering/12-gateway/00-streaming-proxy.md) in *AI Platform Engineering* needs that engine behind it, and the cluster in [Tracer deploy](../../infrastructure/01-containers-kubernetes/00-tracer-deploy.md) in *Infrastructure* needs both. Until `tinyllm train bigram` exists and writes a checkpoint to disk, every layer above Python has nothing to load. This section builds the smallest real language model, computes its logits with a NumPy row gather, and writes the file format that every later model in the book uses. From here on, the tracer bullet has a payload.

The model is a byte-level bigram fitted by counting: one $256 \times 256$ table whose row $i$ is the distribution of the byte that follows byte $i$. It is deliberately blunt, but it is a genuine language model with a genuine loss, and its negative log-likelihood is the number every later model must beat. The file is safetensors, written so that its bytes are identical to the reference library's, which is what lets the Rust engine (or anyone else's reader) load it.

This section builds on [Vectors, matrices, row-major layout, and matmul](../../mathematics-for-computing/03-linear-algebra/01-vectors-matrices-and-matmul.md) from *Mathematics for Computing* and on [Python and NumPy](../../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md) from *Programming Languages*. The solution code is the first, tracer version of two files: [`bigram.py`](../../../../course/solutions/history/L0.0/python/tinyllm/lm/bigram.py) (`BigramLM` with `fit_counts`, `logits`, `nll`, `sample`) and [`safetensors.py`](../../../../course/solutions/history/L0.0/python/tinyllm/io/safetensors.py) (`save_safetensors`, `load_safetensors`, F32 only), with CLI glue in [`entry/python/tinyllm/__main__.py`](../../../../course/solutions/entry/python/tinyllm/__main__.py). Later sections of this chapter take both files over; their final versions live at [`python/tinyllm/lm/bigram.py`](../../../../course/solutions/python/tinyllm/lm/bigram.py) and [`python/tinyllm/io/safetensors.py`](../../../../course/solutions/python/tinyllm/io/safetensors.py). The contracts are [`bigram.pyi`](../../../../course/contracts/py/tinyllm/lm/bigram.pyi) and [`safetensors.pyi`](../../../../course/contracts/py/tinyllm/io/safetensors.pyi), with the formats in [`safetensors.md`](../../../../course/contracts/formats/safetensors.md) and [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md) and the CLI verbs in [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md).

> **Learning objectives**
> - Fit an add-alpha smoothed bigram table by counting, and explain why the smoothing denominator is $R_i + \alpha V$.
> - Compute the model's negative log-likelihood by hand and in code, normalizing logits with a stable log-softmax.
> - Produce logits with a row gather and explain why it equals a one-hot matmul bit for bit.
> - Sample with temperature, greedy tie-breaking, and a seeded inverse CDF.
> - Write and read a canonical safetensors file whose bytes match the reference library's.

## The program around the library

Before the math, it helps to see where the code sits. The two library units are wrapped by the system's own `tinyllm` CLI, [`entry/python/tinyllm/__main__.py`](../../../../course/solutions/entry/python/tinyllm/__main__.py), and [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) fixes the three verbs the tracer needs:

| Verb | What it does with this code |
|---|---|
| `train bigram --data F --out DIR [--alpha A]` | reads the bytes of `F` as ids, calls `fit_counts`, writes `DIR/model.safetensors` (`bigram.weight`, metadata `{"format": "tinyllm"}`) and `DIR/config.json`, prints `{"out", "tokens", "nll"}` as the last line |
| `generate --model DIR --prompt P [--max-tokens N] [--greedy \| --temperature T] [--seed S]` | `load_safetensors`, `BigramLM(weight)`, `sample(list(P.encode()), N, T, S)`, prints `{"ids", "text"}` |
| `logits --model DIR --prompt P [--prefix-ids ...]` | `logits(ids)[-1]` as a JSON list |

The `config.json` for the tracer is exactly `{"tl_arch": "bigram", "tl_tokenizer": "bytes", "vocab_size": 256, "tl_format": 1}`.

The `python/` directory is a uv project that is not an installed package, and the CLI runs as a script from the repository root. The first setup is three small files (the solution's [`pyproject.toml`](../../../../course/solutions/entry/python/pyproject.toml) has since grown later dependencies):

```toml
# python/pyproject.toml
[project]
name = "tinyllm"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=2.0"]

[tool.uv]
package = false
```

```python
# python/tinyllm/__init__.py is empty. python/tinyllm/__main__.py starts with:
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script: put python/ on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyllm.lm.bigram import BigramLM  # noqa: E402
```

You use it like this:

```bash
uv run --project python python python/tinyllm/__main__.py train bigram --data some.txt --out artifacts/bigram
uv run --project python python python/tinyllm/__main__.py generate --model artifacts/bigram --prompt Once --greedy
```

With the shape of the program in view, the rest of the section fills in what `fit_counts`, `logits`, `nll`, `sample`, and the two file functions must compute.

## Just enough math

This part is "just enough" math, defined from scratch. Each idea comes back properly later in *Mathematics for Computing*: functions, `log`, and `exp` in [Exponents, logarithms, and units of information](../../mathematics-for-computing/00-precalculus/01-exponents-logs-and-units-of-information.md); probability and sampling in [Categorical sampling](../../mathematics-for-computing/07-probability-statistics/01-categorical-sampling.md); maximum likelihood in [MLE, Laplace, absolute discounting](../../mathematics-for-computing/07-probability-statistics/02-mle-laplace-absolute-discounting.md); entropy, cross-entropy, and perplexity in [Entropy, cross-entropy, and KL](../../mathematics-for-computing/11-information-theory/01-entropy-cross-entropy-and-kl.md); and stable softmax in [Stable numerics](../../mathematics-for-computing/09-numerical-methods-and-floating-point/02-stable-numerics.md). The notation used throughout is collected here:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size: the number of distinct token ids (256 for bytes) | `int` |
| $x_1, \dots, x_T$ | the training text as token ids, $x_t \in \{0, \dots, V-1\}$ | `int64[T]` |
| $C_{ij}$ | count of positions $t$ with $x_t = i$ and $x_{t+1} = j$ | `float64[V, V]` |
| $R_i = \sum_j C_{ij}$ | how often $i$ appears as a context (followed by something) | `float64[V]` |
| $\alpha > 0$ | smoothing constant (add-one: $\alpha = 1$) | `float` |
| $P(j \mid i)$ | the model's probability that $j$ follows $i$ | `float64` in $(0, 1)$ |
| $W_{ij} = \ln P(j \mid i)$ | the weight table, stored as logits | `float32[V, V]` |
| $\mathrm{softmax}(z)_j = e^{z_j} / \sum_k e^{z_k}$ | turns any real row $z$ into probabilities | `float64[V]` |
| $\mathrm{NLL}$ | mean negative log-likelihood, nats per predicted token | `float` |
| $\tau \ge 0$ | sampling temperature | `float` |
| $F_j = \sum_{k \le j} p_k$ | cumulative distribution (CDF) of a row $p$ | `float64[V]` |
| $u$ | one uniform random number in $[0, 1)$ | `float` |

### Bytes are tokens, and a bigram is a table

Text is stored as UTF-8 bytes, and each byte is a number from 0 to 255. The tracer uses the byte itself as the token id (a fixed design decision, D32 in [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md)), so $V = 256$, there is no tokenizer file, and `"hi"` is the ids `[104, 105]`. A character such as `é` takes two bytes, so it is two tokens.

A language model predicts the next token: given the text so far, it returns a probability for each of the $V$ possible next tokens. The **bigram** assumption is the bluntest one that works. Only the previous token matters, $P(x_{t+1} \mid x_1, \dots, x_t) \approx P(x_{t+1} \mid x_t)$, so the whole model is a $V \times V$ table in which row $i$ is the distribution of what follows $i$.

### From counts to probabilities, and why you must smooth

To fit the table, slide over the text and count each adjacent pair $(x_t, x_{t+1})$ into $C_{x_t, x_{t+1}}$. The row index is where you are, the column index is what comes next. The natural estimate is the observed frequency, $P(j \mid i) = C_{ij} / R_i$. It is the maximum-likelihood estimate (proved in [MLE, Laplace, absolute discounting](../../mathematics-for-computing/07-probability-statistics/02-mle-laplace-absolute-discounting.md)): no other table gives the training text a higher probability.

Zeros break it. A pair that never occurs gets probability 0, and a byte that never occurs as a context has $R_i = 0$, which makes its whole row $0/0$. Add-alpha smoothing pretends every pair was seen $\alpha$ extra times:

$$P(j \mid i) = \frac{C_{ij} + \alpha}{R_i + \alpha V}.$$

Each row still sums to one, because $\sum_j (C_{ij} + \alpha) = R_i + \alpha V$. The denominator must add $\alpha$ once for each of the $V$ columns, not once. An unseen context gets the uniform row $\alpha / (\alpha V) = 1/V$.

### Store logs, not probabilities

The natural logarithm $\ln$ turns products into sums, $\ln(ab) = \ln a + \ln b$, and $\exp$ undoes it, $e^{\ln p} = p$. The model stores $W_{ij} = \ln P(j \mid i)$, called **logits**. Every later model in the book also outputs logits and turns them into probabilities with softmax. For a row of log-probabilities, softmax gives the probabilities back exactly: $e^{\ln p_j} / \sum_k e^{\ln p_k} = p_j / 1$. Softmax also ignores a constant added to every entry, $\mathrm{softmax}(z + c) = \mathrm{softmax}(z)$, which is why you compute it as $e^{z_j - \max z}$: the largest exponent becomes $e^0 = 1$ and nothing overflows.

### Negative log-likelihood measures fit

Under the bigram assumption, the probability of the whole text (after its first token) is the product $\prod_{t=1}^{T-1} P(x_{t+1} \mid x_t)$. That number underflows for any real text, so you take its log and average:

$$\mathrm{NLL} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \ln P(x_{t+1} \mid x_t).$$

There are $T - 1$ predictions, because the first token has no context. The unit is **nats per token** (natural log). Lower is better; a model that knows nothing (uniform over 256 bytes) scores $\ln 256 \approx 5.545$. For the count model the sum regroups by pair, $\mathrm{NLL} = -\frac{1}{T-1} \sum_{i,j} C_{ij} \ln P(j \mid i)$, a closed form that the autograd bigram of [Training loop and the autograd bigram](05-training-loop-and-the-autograd-bigram.md) must reach by gradient descent.

### Logits by row gather

A one-hot vector $e_i$ has a 1 at position $i$ and 0 elsewhere. Multiplying it by the table selects row $i$: $(e_i W)_j = \sum_k [k = i] W_{kj} = W_{ij}$. NumPy expresses the same operation directly as `weight[ids]`, producing the $T \times V$ logits without constructing a $T \times V$ one-hot matrix or doing $T \cdot V \cdot V$ multiply-adds. The Rust engine of [Your first endpoint](../p10-serving/00-your-first-endpoint.md) computes the same row selection.

### Temperature and inverse-CDF sampling

To sample, divide the logits by $\tau$ before softmax: $p = \mathrm{softmax}(z / \tau)$. With log-probabilities $z = \ln q$ this gives $p_j \propto q_j^{1/\tau}$. At $\tau = 1$ you sample the model as is; $\tau < 1$ sharpens it (at $\tau = 0.5$, $p \propto q^2$); $\tau > 1$ flattens it. $\tau = 0$ is defined as **greedy**: take the largest logit, and on a tie the lowest id (decision D11 in the contracts, so every implementation agrees).

To draw $j$ with probability $p_j$, build the running sum $F_j = p_0 + \dots + p_j$ (so $F_{V-1} = 1$), draw one uniform $u \in [0, 1)$, and return the smallest $j$ with $F_j > u$. The interval of $u$ values that lands on $j$ is $[F_{j-1}, F_j)$, whose length is exactly $p_j$. One draw per token, from a generator created once with a seed, makes the same seed give the same text. The tracer may use `numpy.random.default_rng(seed)`; the autograd bigram later switches to the book's own PCG32 (from [Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md)) so that Python and Rust agree.

### The file

A safetensors file is an 8-byte little-endian length $N$, then $N$ bytes of JSON describing each tensor (dtype, shape, byte offsets), padded with spaces to a multiple of 8, then the raw tensor bytes back to back, little-endian and row-major. The full rules, including the order a canonical writer uses so its bytes match the reference library, are in [`safetensors.md`](../../../../course/contracts/formats/safetensors.md). The tracer checkpoint holds one tensor, `bigram.weight`, F32 `[256, 256]`.

## Working it by hand

All of that is easier to trust once you have pushed numbers through it. Take a three-letter alphabet so the table fits on paper: `a = 0`, `b = 1`, `c = 2`, so $V = 3$. The text is `abbacab`, ids `[0, 1, 1, 0, 2, 0, 1]`, $T = 7$. It has six adjacent pairs: `ab`, `bb`, `ba`, `ac`, `ca`, `ab`.

Counting them gives this table (row = current, column = next):

| | a | b | c | $R_i$ |
|---|---|---|---|---|
| a | 0 | 2 | 1 | 3 |
| b | 1 | 1 | 0 | 2 |
| c | 1 | 0 | 0 | 1 |

Now add one ($\alpha = 1$, so each row total grows by $\alpha V = 3$) and divide:

| | a | b | c |
|---|---|---|---|
| a | 1/6 = 0.1667 | 3/6 = 0.5 | 2/6 = 0.3333 |
| b | 2/5 = 0.4 | 2/5 = 0.4 | 1/5 = 0.2 |
| c | 2/4 = 0.5 | 1/4 = 0.25 | 1/4 = 0.25 |

Every row sums to 1. The weight table is the natural log of each entry, for example $W_{ab} = \ln 0.5 = -0.6931$ and $W_{aa} = \ln(1/6) = -1.7918$.

The NLL scores the six predictions. Their costs $-\ln P$ are:

| $t$ | context $\to$ next | $P$ | $-\ln P$ |
|---|---|---|---|
| 1 | a $\to$ b | 3/6 | $\ln 2 = 0.6931$ |
| 2 | b $\to$ b | 2/5 | $\ln 2.5 = 0.9163$ |
| 3 | b $\to$ a | 2/5 | $\ln 2.5 = 0.9163$ |
| 4 | a $\to$ c | 2/6 | $\ln 3 = 1.0986$ |
| 5 | c $\to$ a | 2/4 | $\ln 2 = 0.6931$ |
| 6 | a $\to$ b | 3/6 | $\ln 2 = 0.6931$ |

The sum is $3 \ln 2 + 2 \ln 2.5 + \ln 3 = 2.0794 + 1.8326 + 1.0986 = 5.0106$. Divided by 6, $\mathrm{NLL} = 0.8351$ nats per token, better than the uniform model's $\ln 3 = 1.0986$.

The remaining operations are quick. For ids `[0, 2]`, `weight[ids]` is rows `a` and `c` of $W$, unchanged. Greedy decoding from `a` takes the largest entry, `b` (0.5). From `b`, `a` and `b` tie at 0.4, and ties go to the lowest id, so `a`. Then `b`, then `a`: greedy from `[a]` for 4 tokens is `[1, 0, 1, 0]`.

Temperature 0.5 on row `a` squares the probabilities: $q^2 = (1/36, 9/36, 4/36)$, which renormalizes to $(1/14, 9/14, 4/14)$, so `b` goes from 50% to 64%. For inverse-CDF sampling on row `a`, $F = (1/6, 4/6, 1) = (0.1667, 0.6667, 1)$. A draw $u = 0.10$ gives `a` (the first $F_j > 0.10$), $u = 0.50$ gives `b`, and $u = 0.90$ gives `c`.

Finally the file. The format page works out a 120-byte file by hand: one tensor `w = [[1, 2], [3, 4]]` with metadata `{"format": "tinyllm"}`. Its header is 93 bytes of JSON, padded with 3 spaces to $N = 96$; the file is $8 + 96 + 16 = 120$ bytes, ending in `00 00 80 3f 00 00 00 40 00 00 40 40 00 00 80 40` (1.0, 2.0, 3.0, 4.0 as little-endian float32).

These numbers are the first things a correct implementation must reproduce, and the walkthrough below traces each of them through the code.

## Designing the API

The interface is small:

```python
# python/tinyllm/lm/bigram.py
class BigramLM:
    weight: NDArray                                   # float32 [V, V], log P(j | i)
    def __init__(self, weight: ArrayLike | None = None) -> None
    @property
    def vocab_size(self) -> int
    def fit_counts(self, ids: ArrayLike, vocab_size: int, alpha: float = 1.0) -> None
    def logits(self, ids: ArrayLike) -> NDArray       # [T, V] = W[ids]
    def nll(self, ids: ArrayLike) -> float
    def sample(self, prefix: list[int], n: int, temperature: float, seed: int) -> list[int]

# python/tinyllm/io/safetensors.py (v0: F32 only)
def save_safetensors(path: str, tensors: dict[str, NDArray], meta: dict[str, str]) -> None
def load_safetensors(path: str) -> tuple[dict[str, NDArray], dict[str, str]]
```

The contracts carry the exact rules: which errors are `ValueError`, that `sample` returns only the new ids, and that an empty `meta` omits `__metadata__`. For input ids $i_0, \dots, i_{T-1}$, `logits` is the row gather `weight[ids]`, with output shape $T \times V$.

Behind those signatures sits a list of behaviours a correct solution must get right. The hand example comes first: the probability table must come out of both `weight` and `logits`, and the NLL on `abbacab` must be $0.8351$, the number `train bigram` prints. The count direction matters: on the text `ab`, $P(b \mid a) = 2/3$ and row `b` is uniform, because a transposed table predicts the past. Every row of a 256-byte model must sum to 1, unseen bytes included, since the Rust engine samples from any row. The NLL must equal the closed form recomputed by plain loops, because that closed form is the target the autograd bigram has to reach, and it must normalize its input: rows `[0, 0]` and `[5, 5]` both give $\ln 2$, since trained logits are not normalized.

The inputs are checked rather than trusted. An $\alpha \le 0$ is a `ValueError`, so no `-inf` or `nan` reaches a checkpoint. Ids `-1` and `V` are a `ValueError`, because NumPy would wrap `-1` silently; float ids (`2.7` is not a byte) and a 2-D batch are rejected too. A weight passed to the constructor is checked, not converted: a float64, non-square, 1-D, `inf`, or `nan` weight is a `ValueError`, and asking for `vocab_size` before a table exists is a `RuntimeError`, so a bad checkpoint fails at load rather than inside the forward pass. `logits(ids)` must equal `weight[ids]` bit for bit, as float32 `[T, V]`, since that is the forward pass the engine reproduces.

Sampling has its own rules. Greedy ties go to the lowest id (greedy from `a` is `[1, 0, 1, 0]`), $\tau = 0$ is greedy and ignores the seed, and `sample` returns exactly `n` new ids with no prefix (`n = 0` gives `[]`). The same seed gives the same ids and different seeds differ. Statistically, 6000 sampled transitions must fit the table under a chi-square test with $p > 10^{-3}$, and at $\tau = 0.5$ they must fit $q^2$ renormalized, which is how you know temperature divides. An empty prefix, $\tau < 0$, or $n < 0$ is a `ValueError`; bytes have no start token to fall back on.

The file must reproduce the 120-byte worked example and be byte-identical to the pinned `safetensors` library on six golden files, so that any reader, including the Rust engine, loads it. A transposed view and a `>f4` array must still be written as row-major little-endian bytes. Saving a float64, int64, or bool array is rejected rather than silently reinterpreted. On the read side, every library file loads to the exact inputs, `load(save(x)) == x` with a writable float32 result, and the reader rejects a gap, trailing bytes, a wrong element size, a short file, a duplicate key, an unknown dtype, a JSON array header, and a non-string `__metadata__` value. It also caps the header length: an otherwise valid file with $N = 100{,}000{,}001$ is refused, so a corrupt length cannot make the reader parse gigabytes.

## Building it step by step

### The reasoning chain

Before reading any code, it is worth seeing how each design choice follows from the one before. Try to predict each step before you read it.

1. **The model is a table, and the row is the context.** The bigram assumption says the next-token distribution depends only on the current token, so there are exactly $V$ distributions to store, one per context. Storing them as the rows of a $V \times V$ matrix means "the distribution after $i$" is `weight[i]`, a contiguous row. Everything downstream (the gather in `logits`, the Rust engine's row lookup) follows from putting the context on the row axis.
2. **Fitting is counting.** The maximum-likelihood table is the normalized count matrix, so `fit_counts` needs no optimizer: count each pair $(x_t, x_{t+1})$ into $C_{x_t, x_{t+1}}$, which is a scatter-add at index pairs `(ids[:-1], ids[1:])`.
3. **Smoothing fixes the zeros, and the denominator falls out of a row sum.** Add $\alpha$ to every cell and divide by the row total. Summing the smoothed row gives $R_i + \alpha V$ automatically, so the code never writes the denominator by hand and cannot get it wrong. For `a` in the worked example, $(0+1) + (2+1) + (1+1) = 6 = 3 + 1 \cdot 3$.
4. **Store logits in float32, compute in float64.** The checkpoint format and the engine use F32, but counts can exceed $2^{24}$ (where float32 stops representing every integer) and logs of small probabilities lose digits. So the arithmetic runs in float64 and only the final $\ln P$ is cast to float32.
5. **The forward pass is a gather.** Since $e_i W = W_i$, the logits for a sequence are `weight[ids]`: $O(TV)$ copies instead of an $O(TV^2)$ matmul with a one-hot matrix.
6. **NLL must not assume normalized rows.** The count model's rows are already log-probabilities, but the autograd bigram will store trained logits that are not. So `nll` applies a stable log-softmax to each row before reading off $-\ln P(x_{t+1} \mid x_t)$. On the count model this changes nothing; on `[5, 5]` it turns $-5$ into the correct $\ln 2$.
7. **Sampling is one generator, one draw per token, and an inverse CDF.** Create the generator once from the seed, so the sequence of uniforms $u_1, u_2, \dots$ is fixed by the seed and each step consumes a fresh one. Divide by $\tau$, softmax, cumulative-sum, and return the smallest $j$ with $F_j > u$. Greedy ($\tau = 0$) is a separate branch, because dividing by zero is undefined and greedy must ignore the seed.
8. **Validate at the boundary.** Every id passes through one helper that rejects non-integers, wrong ranks, and values outside $[0, V)$, because NumPy would otherwise wrap `-1` to the last row silently. Every weight is checked at construction, so a bad checkpoint fails when it loads.
9. **The file is canonical, and the reader trusts nothing.** The writer chooses the one byte layout the reference library produces (sorted names, compact JSON, space padding, little-endian row-major data). The reader checks each rule of the format in order before it touches tensor bytes.

### Validating ids

One helper guards every entry point, so no public method can index the table with a bad id.

```python
def _as_ids(ids: ArrayLike, vocab_size: int) -> NDArray[np.int64]:
    """ids as a 1-D int64 array, every value checked against [0, vocab_size)."""
    a = np.asarray(ids)
    if a.ndim != 1:                       # a 2-D batch is not a sequence
        raise ValueError(f"ids must be 1-D, got shape {a.shape}")
    if a.size == 0:                       # np.asarray([]) is float64; an empty
        return a.astype(np.int64)         # sequence is still a valid sequence
    if not np.issubdtype(a.dtype, np.integer):   # 2.7 is not a byte
        raise ValueError(f"ids must be integers, got dtype {a.dtype}")
    a = a.astype(np.int64)
    lo, hi = int(a.min()), int(a.max())   # two reductions check every value
    if lo < 0 or hi >= vocab_size:        # -1 would silently mean "last row"
        raise ValueError(f"ids must lie in [0, {vocab_size}), got min {lo} and max {hi}")
    return a
```

The empty case comes before the dtype check because `np.asarray([])` has dtype float64, and an empty prefix list must not be reported as "not integers".

### Constructing the model

A weight passed in (from a checkpoint) is checked, never converted.

```python
def __init__(self, weight: ArrayLike | None = None) -> None:
    self.weight = None
    if weight is not None:
        w = np.asarray(weight)
        if w.dtype != np.float32:         # converting float64 would hide a wrong writer
            raise ValueError(f"weight must be float32, got {w.dtype}")
        if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] == 0:
            raise ValueError(f"weight must be a non-empty square matrix, got shape {w.shape}")
        if not np.isfinite(w).all():      # -inf from alpha = 0, or nan, poisons sampling
            raise ValueError("weight must be finite (weights must be finite)")
        self.weight = np.ascontiguousarray(w)   # rows are contiguous for the gather
```

### Fitting by counting

Steps 2 to 4 of the reasoning chain fit in five lines.

```python
def fit_counts(self, ids: ArrayLike, vocab_size: int, alpha: float = 1.0) -> None:
    if not alpha > 0:                     # also rejects nan: nan > 0 is False
        raise ValueError(f"alpha must be > 0 (alpha = 0 gives log 0 = -inf for unseen pairs), got {alpha}")
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    a = _as_ids(ids, vocab_size)
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    # Row = the token we are at, column = the token that follows it.
    # np.add.at is unbuffered: a pair that occurs twice is counted twice.
    np.add.at(counts, (a[:-1], a[1:]), 1.0)
    smoothed = counts + alpha
    # The row sum is R_i + alpha * V, so every row sums to exactly 1.
    probs = smoothed / smoothed.sum(axis=1, keepdims=True)
    self.weight = np.log(probs).astype(np.float32)   # logits, stored as F32
```

`np.add.at` is the essential call. The tempting `counts[a[:-1], a[1:]] += 1` is buffered: NumPy reads all the old values, adds one, and writes back, so a pair that appears twice (`ab` in the worked example) ends up counted once. The guard is written `not alpha > 0` rather than `alpha <= 0` so that a `nan` alpha is rejected as well.

### Logits and NLL

```python
def logits(self, ids: ArrayLike) -> NDArray:
    v = self.vocab_size                   # RuntimeError if there is no table yet
    a = _as_ids(ids, v)
    return self.weight[a]                 # row gather: [T, V] float32, bit-exact

def nll(self, ids: ArrayLike) -> float:
    a = _as_ids(ids, self.vocab_size)
    if a.size < 2:                        # T - 1 = 0 predictions: the mean is undefined
        raise ValueError("nll needs at least two ids: one context and one prediction")
    z = self.logits(a[:-1]).astype(np.float64)    # contexts only: x_1 .. x_{T-1}
    z = z - z.max(axis=1, keepdims=True)          # stable: largest exponent is e^0
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))   # log-softmax per row
    # Score the token that follows each context, then average over T - 1.
    return float(-logp[np.arange(a.size - 1), a[1:]].mean())
```

The rows come from `a[:-1]` (contexts) and the scored columns from `a[1:]` (targets). Swapping either one gives a number that no longer depends on what follows each token. On the worked example the six selected entries are exactly the six $-\ln P$ values computed by hand, and their mean is $0.8351$.

### Sampling

```python
def sample(self, prefix: list[int], n: int, temperature: float, seed: int) -> list[int]:
    v = self.vocab_size
    if len(prefix) == 0:                  # bytes have no start token to condition on
        raise ValueError("prefix must hold at least one id: the byte tokenizer has no start token")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not temperature >= 0:              # rejects negatives and nan
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    cur = int(_as_ids(prefix, v)[-1])     # a bigram only needs the last id
    rng = np.random.default_rng(seed)     # ONE generator for the whole call
    out: list[int] = []
    for _ in range(n):
        row = self.logits([cur])[0].astype(np.float64)
        if temperature == 0:
            nxt = int(np.argmax(row))     # first maximum: ties go to the lowest id
        else:
            z = row / temperature         # divide: tau < 1 sharpens
            p = np.exp(z - z.max())
            cdf = np.cumsum(p / p.sum())
            u = rng.random()              # a fresh uniform in [0, 1) per token
            # smallest j with F_j > u; the min() guards a cdf that ends at 0.9999999
            nxt = min(int(np.searchsorted(cdf, u, side="right")), v - 1)
        out.append(nxt)
        cur = nxt
    return out                            # new ids only, never the prefix
```

`searchsorted(cdf, u, side="right")` returns the first index whose value is strictly greater than `u`, which is exactly the inverse-CDF definition. With `side="left"`, a $u$ that lands exactly on a boundary $F_j$ would pick $j$ instead of $j + 1$. The clamp matters because floating-point cumulative sums can end slightly below 1; a $u$ above the last entry would otherwise return $V$, an id that does not exist. On row `a`, $F = (0.1667, 0.6667, 1)$, so $u = 0.50$ gives index 1, `b`, as in the hand example.

The full solution is [`bigram.py`](../../../../course/solutions/history/L0.0/python/tinyllm/lm/bigram.py).

### Writing safetensors

```python
def save_safetensors(path: str, tensors: dict[str, NDArray], meta: dict[str, str]) -> None:
    for k, v in meta.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"metadata must map str to str, got {k!r}: {v!r}")
    header: dict = {}
    if meta:                              # empty meta: no __metadata__ key at all, not {}
        header["__metadata__"] = {k: meta[k] for k in sorted(meta, key=lambda s: s.encode("utf-8"))}
    blobs: list[bytes] = []
    offset = 0
    # One dtype in v0, so the canonical order is simply by name, as UTF-8 bytes.
    for name in sorted(tensors, key=lambda s: s.encode("utf-8")):
        if not isinstance(name, str) or name == "__metadata__":
            raise ValueError(f"bad tensor name {name!r}")
        arr = np.asarray(tensors[name])
        if arr.dtype.kind != "f" or arr.dtype.itemsize != 4:   # F32 only, either byte order
            raise ValueError(f"tensor {name!r}: dtype {arr.dtype} is not F32 (contract v0 writes F32 only)")
        # Little-endian, row-major, whatever the array's byte order or memory layout.
        data = arr.astype("<f4", copy=False).tobytes(order="C")
        header[name] = {"dtype": "F32", "shape": [int(d) for d in arr.shape],
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)               # tensors are packed back to back, no gaps
        blobs.append(data)
    # Compact separators and raw UTF-8 (no \u escapes): the library's exact bytes.
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 8)       # pad with spaces; 0 bytes when already aligned
    with open(path, "wb") as f:
        f.write(len(text).to_bytes(8, "little"))
        f.write(text)
        for data in blobs:
            f.write(data)
```

Every canonical choice is there because a byte-for-byte match is the strongest interoperability guarantee: if the bytes are the library's, every reader that accepts the library's files accepts these. `-len(text) % 8` is the number of bytes up to the next multiple of 8, and it is 0 when the header is already aligned (in the worked example, $-93 \bmod 8 = 3$). `astype("<f4", copy=False)` fixes a big-endian array, and `tobytes(order="C")` writes a transposed view in row order, not memory order.

### Reading safetensors

The reader enforces the format's rules in the order they become checkable.

```python
def load_safetensors(path: str) -> tuple[dict[str, NDArray], dict[str, str]]:
    with open(path, "rb") as f:
        buf = f.read()
    # Rule 1: the length prefix and the header fit in the file.
    if len(buf) < 8:
        raise ValueError(f"{path}: {len(buf)} bytes is too short for the 8-byte header length")
    n = int.from_bytes(buf[:8], "little")
    if n > MAX_HEADER or 8 + n > len(buf):   # MAX_HEADER = 100_000_000: no gigabyte parses
        raise ValueError(f"{path}: header length {n} does not fit in a {len(buf)}-byte file")
    # Rule 2: UTF-8 JSON object, no duplicate keys.
    try:
        header = json.loads(buf[8:8 + n].decode("utf-8"), object_pairs_hook=_no_duplicates)
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: header is not UTF-8: {e}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: header is not JSON: {e}") from None
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header must be a JSON object")
    # Rule 3: metadata maps strings to strings.
    meta = header.pop("__metadata__", {})
    # ...
    # Rule 4: known dtype (v0: F32 only), non-negative integer dims, sizes agree.
    entries = []
    for name, e in header.items():
        # ... exactly the keys dtype, shape, data_offsets; dtype == "F32";
        # ... shape a list of ints >= 0; offsets two ints with 0 <= begin <= end
        if offs[1] - offs[0] != math.prod(shape) * 4:     # 4 bytes per F32 element
            raise ValueError(f"{path}: tensor {name!r}: {offs[1] - offs[0]} bytes for shape {shape}")
        entries.append((offs[0], offs[1], name, shape))
    # Rule 5: sorted by begin, the tensors tile the data buffer exactly.
    entries.sort()
    pos = 0
    for begin, end, name, _ in entries:
        if begin != pos:                  # a gap or an overlap
            raise ValueError(f"{path}: tensor {name!r} begins at {begin}, expected {pos} (gap or overlap)")
        pos = end
    if 8 + n + pos != len(buf):           # trailing bytes are an error too
        raise ValueError(f"{path}: the data buffer holds {len(buf) - 8 - n} bytes, the tensors cover {pos}")
    base = 8 + n
    tensors = {}
    for begin, end, name, shape in entries:
        arr = np.frombuffer(buf, dtype="<f4", count=(end - begin) // 4, offset=base + begin)
        # astype copies (frombuffer views are read-only) and gives native float32;
        # reshape keeps a scalar's shape [] as ()
        tensors[name] = arr.astype(np.float32).reshape(shape)
    return tensors, dict(meta)
```

`_no_duplicates` is a `json` object hook that raises on a repeated key, because Python's default is last-one-wins and two readers could then disagree about the same file. `type(d) is int` (in the elided shape check) rejects `true`, which `isinstance(True, int)` would accept.

The full solution is [`safetensors.py`](../../../../course/solutions/history/L0.0/python/tinyllm/io/safetensors.py).

### The CLI verb

`train bigram` is glue, and it shows the whole pipeline in one place:

```python
ids = np.frombuffer(Path(a.data).read_bytes(), dtype=np.uint8).astype(np.int64)  # bytes are ids
if ids.size < 2:
    raise ValueError(f"{a.data}: need at least two bytes to count a bigram")
model = BigramLM()
model.fit_counts(ids, VOCAB, alpha=a.alpha)
# ... mkdir
save_safetensors(
    str(out / "model.safetensors"),
    {TENSOR: np.asarray(model.weight, dtype=np.float32)},   # TENSOR = "bigram.weight"
    {"format": "tinyllm"},
)
(out / "config.json").write_text(json.dumps(CONFIG) + "\n")
# ...
return {"out": str(out), "tokens": int(ids.size), "nll": model.nll(ids)}
```

The full file is [`entry/python/tinyllm/__main__.py`](../../../../course/solutions/entry/python/tinyllm/__main__.py).

### Why this design

The costs are modest. `fit_counts` is $O(T + V^2)$ time and $O(V^2)$ memory (65,536 cells for bytes). `logits` is $O(TV)$, a copy of $T$ rows, and so is `nll`. `sample` is $O(nV)$: each step does a softmax and a cumulative sum over one row. The checkpoint is $8 + N + 4V^2 \approx 262$ KB.

Counting beats gradient descent here because the count table is the exact maximum-likelihood solution, computed in one pass. Gradient descent (which the autograd bigram uses on purpose, to test the autograd engine) reaches the same NLL only approximately and needs a learning rate and many steps. The gather beats the one-hot matmul for a similar reason: the matmul is how the math reads, but it allocates a $T \times V$ matrix and does $TV^2$ multiply-adds to copy $T$ rows. The gather is exact, and it is what production engines call an embedding lookup.

The explicit inverse CDF is preferred to `rng.choice(V, p=p)`, which would also sample correctly, because NumPy's internal algorithm is NumPy's to change, and the Rust engine must reproduce the solution draw for draw once the sampler moves to PCG32. "One uniform, smallest $j$ with $F_j > u$" is a rule you can port to any language. Likewise a strict, canonical file is preferred to "whatever `json.dumps` gives". A looser writer would produce files the reference library still reads, but byte-identity is what lets a hash or a golden file check the checkpoint. A strict reader costs a few comparisons and turns a corrupt download into an error message instead of a model that loads garbage.

Two tradeoffs are accepted for now. This first version reads the whole file into memory and copies each tensor; memory-mapping and every other dtype wait for [Safetensors for every dtype, atomic checkpoints, and the token stream](06-safetensors-checkpoints-and-token-streams.md). And the sampler uses NumPy's generator, which only Python can reproduce; [Training loop and the autograd bigram](05-training-loop-and-the-autograd-bigram.md) replaces it with PCG32.

## What can go wrong

Most of the guards in the code exist because of one specific mistake. The table below collects them, with the symptom you would see and the line that prevents it.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| no smoothing ($\alpha = 0$) | `-inf` logits for unseen pairs; a row of `nan` for a byte never seen as a context; `0 * -inf = nan` inside a matmul | `fit_counts` rejects `not alpha > 0`, and the constructor rejects non-finite weights |
| counting `C[next, cur]` | the model predicts the previous byte; the hand table comes out transposed | `np.add.at(counts, (a[:-1], a[1:]), 1.0)`: contexts index rows |
| trusting ids | `-1` silently counts into the last row (NumPy wraps negative indexes) | every entry point goes through `_as_ids`, which checks the range |
| reading logits as log-probabilities in `nll` | correct for the count model, wrong the day the table holds trained logits | `nll` applies a stable log-softmax first |
| a one-hot matrix multiply in `logits` | allocates an unnecessary $T \times V$ array instead of gathering rows | `logits` returns `self.weight[a]` |
| seeding a new generator at every step | still repeatable, but every step draws the same $u$, so the text follows one quantile of each row | one `default_rng(seed)` before the loop |
| writing the array's memory as is | a transposed view writes columns; a big-endian array writes swapped bytes | `astype("<f4", copy=False).tobytes(order="C")` |
| a reader that trusts the header | reads past a gap or ignores trailing bytes, so a corrupt checkpoint loads | rule 5: sorted entries must tile the buffer exactly, with no trailing bytes |
| smoothing denominator $R_i + \alpha$ | rows sum to more than 1 | the denominator is the row sum of the smoothed counts |
| multiplying by the temperature | $\tau = 0.5$ flattens instead of sharpening | `z = row / temperature` |
| `np.argmax` on a reversed row, or `>=` in a manual loop | greedy ties go to the highest id, and engines disagree | plain `np.argmax(row)`, which returns the first maximum |
| scoring the context token in `nll` | an NLL that does not depend on what follows | rows from `a[:-1]`, scored columns from `a[1:]` |
| insertion order, zero padding, padding when already aligned, `{}` for empty metadata, `\u` escapes | bytes differ from the library's, so a strict reader or a hash check rejects the file | sorted names, `-len(text) % 8` spaces, `__metadata__` only when non-empty, `ensure_ascii=False` |
| converting a loaded tensor with `np.ascontiguousarray` | a scalar tensor (shape `[]`) comes back with shape `(1,)`, because `ascontiguousarray` returns at least 1-D (a pitfall covered in [Python and NumPy](../../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md)) | the loader uses `astype(np.float32).reshape(shape)`, which keeps shape `()` |

> **Warning:** The buffered `counts[a[:-1], a[1:]] += 1` passes on most short inputs and fails only when a pair repeats. Always use `np.add.at` for a scatter-add.

## Where this leads

This section leaned on two earlier pieces: the Python matrix multiplication of [Vectors, matrices, row-major layout, and matmul](../../mathematics-for-computing/03-linear-algebra/01-vectors-matrices-and-matmul.md), which the row gather replaces, and the NumPy arrays, dtypes, `np.add.at`, and broadcasting of [Python and NumPy](../../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md).

Looking forward, the Rust engine of [Your first endpoint](../p10-serving/00-your-first-endpoint.md) reads `bigram.weight` from this file (`tl_arch = bigram`, `tl_tokenizer = bytes`) and computes the same logits by indexing it with token ids, which completes the tracer: every layer of the system runs end to end on this model. Within this chapter, [Training loop and the autograd bigram](05-training-loop-and-the-autograd-bigram.md) takes over `bigram.py`, trains the same table with autograd until it reaches the count model's NLL within $10^{-3}$, and samples with PCG32; and [Safetensors for every dtype, atomic checkpoints, and the token stream](06-safetensors-checkpoints-and-token-streams.md) takes over `safetensors.py` to add every dtype, atomic checkpoints, and the token-stream reader.

### Further reading

- Jurafsky and Martin, *Speech and Language Processing* (3rd ed. draft), ch. 3 "N-gram Language Models".
- KenLM n-gram models (`lm/builder/`): the production version of the add-alpha bigram, with 5-gram contexts, modified Kneser-Ney smoothing, backoff, and compact tries.
- Embedding lookup (`ggml_get_rows` in llama.cpp's `ggml/src/ggml.c`, `torch.nn.Embedding`): the same row gather used for logits here.
- vLLM and SGLang samplers (vLLM `v1/sample/sampler.py`): top-k, top-p, min-p, and penalties, batched on GPU with per-request seeds.
- Hugging Face `safetensors` (`safetensors/src/tensor.rs`): every dtype, memory-mapped zero-copy loading, and the Rust core the engine mirrors.

## Key terms

- **bigram model**: a language model that predicts the next token from the current token alone, stored as a $V \times V$ table.
- **add-alpha smoothing**: adding $\alpha$ to every count before normalizing, so no probability is zero.
- **logits**: unnormalized log-scores that softmax turns into probabilities; here the table stores $\ln P(j \mid i)$.
- **negative log-likelihood (NLL)**: the mean of $-\ln P$ over the predicted tokens, in nats per token.
- **row gather**: indexing a table by ids to select rows, equivalent to multiplying by one-hot vectors.
- **temperature**: the divisor applied to logits before softmax; below 1 sharpens, above 1 flattens, 0 means greedy.
- **inverse CDF sampling**: drawing $u \in [0, 1)$ and returning the smallest $j$ with $F_j > u$.
- **safetensors**: a checkpoint format of an 8-byte header length, a JSON header, and raw little-endian row-major tensor bytes.

## Summary

- A bigram language model is one table: row $i$ is the distribution of the next byte after byte $i$, and add-alpha smoothing makes every row a proper distribution with no zeros.
- The negative log-likelihood of the count model is a closed form in the counts, and it is the number every later model in the book must beat.
- Indexing the table by token ids picks its rows exactly, so NumPy returns the model's logits bit for bit.
- Sampling is temperature, softmax, one uniform draw, and an inverse CDF; a seed makes it repeatable.
- The solution's safetensors file is byte-identical to the reference library's, so the Rust engine (and anyone else's reader) loads it.

## Review questions

1. Why is the smoothing denominator $R_i + \alpha V$ and not $R_i + \alpha$, and where does the code compute it?

   <details><summary>Answer</summary>

   Smoothing adds $\alpha$ to each of the $V$ cells of a row, so the row total grows by $\alpha V$; dividing by $R_i + \alpha$ would leave rows summing to more than 1. The code never writes the formula: `smoothed.sum(axis=1, keepdims=True)` sums the already-smoothed row, which is $R_i + \alpha V$ by construction.

   </details>

2. What goes wrong if `fit_counts` uses `counts[a[:-1], a[1:]] += 1` instead of `np.add.at`?

   <details><summary>Answer</summary>

   Fancy-index assignment is buffered: every index pair reads the original value, adds one, and writes back, so a pair that occurs $k$ times is counted once. In `abbacab` the pair `ab` occurs twice, so $C_{ab}$ would come out as 1 instead of 2 and the hand table would be wrong. `np.add.at` applies each addition in turn.

   </details>

3. The count model's rows are already log-probabilities. Why does `nll` apply log-softmax anyway?

   <details><summary>Answer</summary>

   Because the interface is "a table of logits", not "a table of log-probabilities". The autograd bigram stores trained logits whose rows do not sum to 1 after `exp`; reading them directly as $\ln P$ would give a wrong NLL. Normalizing costs $O(TV)$ and is a no-op (up to rounding) on the count model, so it is always correct.

   </details>

4. What breaks if the sampler creates `default_rng(seed)` inside the loop instead of once?

   <details><summary>Answer</summary>

   The output is still repeatable, but every step draws the same first uniform $u$. Every token is then the same quantile of its row, so the samples no longer follow the model's distribution: a chi-square test over many transitions fails even though each individual draw looks plausible.

   </details>

5. Why does the writer sort names by their UTF-8 bytes, and pad with spaces rather than zeros?

   <details><summary>Answer</summary>

   Both choices reproduce the reference library's bytes exactly ([`safetensors.md`](../../../../course/contracts/formats/safetensors.md) gives the canonical rules). For a Python `str`, UTF-8 byte order happens to equal code-point order, but stating the rule in bytes makes it unambiguous across languages: a UTF-16 based sort (Java, JavaScript) orders some characters above U+FFFF differently. Spaces keep the header valid JSON (trailing whitespace is allowed), while zero bytes are not valid JSON and a strict parser rejects them.

   </details>

6. On the text `abbacab` with $\alpha = 1$, what does greedy decoding produce from `[a]` for four tokens, and why does the second token come out as `a`?

   <details><summary>Answer</summary>

   `[1, 0, 1, 0]`. From `a` the largest probability is `b` at 0.5. Row `b` has `a` and `b` tied at 0.4, and the tie rule picks the lowest id, so `a`; the pattern then repeats.

   </details>
