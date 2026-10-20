# n-gram language model with interpolated modified Kneser-Ney

The [byte bigram](../06-numerics-and-autograd/01-byte-bigram.md) predicts the next byte from one byte of history, and the [autograd bigram](../06-numerics-and-autograd/06-training-loop-and-the-autograd-bigram.md) fits the same table by gradient descent. Neither can use more context. A table indexed by the last three bytes has $256^3 \approx 1.7 \times 10^7$ rows, almost all of them never seen in training, and an unseen row means probability 0 for the true next token and an infinite loss.

This section builds the classic answer, the one speech recognition and machine translation relied on for twenty years before neural models: count n-grams, then smooth the counts so that every context still gives every token some probability. Interpolated modified Kneser-Ney is the strongest of those smoothers. It becomes the baseline every neural model in [the model zoo](../12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) has to beat on bits per byte, a cheap draft model for [speculative decoding](../21-inference/06-speculative-decoding.md), and, in [the TinyStories capstone](../capstones/01-tinystories.md), the perplexity filter that throws out garbage documents before training.

You will build Kneser-Ney up from the zero-probability problem one fix at a time, compute a small bigram model by hand to exact fractions, and then walk through the solution: counting, adjusted counts, discounts, the recursion, scoring, and the saved file. Try to predict each step of the code before reading it. The section builds on [MLE, Laplace, absolute discounting](../../mathematics/13-probability-statistics/03-mle-laplace-absolute-discounting.md) (for `ney_discount`) and [Perplexity, bits per byte, NLL accumulator](../../mathematics/17-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md) (for `NLLAccumulator`), both in *Mathematics*, and on [Safetensors for every dtype, atomic checkpoints, and the token stream](../06-numerics-and-autograd/07-safetensors-checkpoints-and-token-streams.md) for `save_safetensors` and `load_safetensors`. If conditional probability feels rusty, [the probability problem set, part a](../../mathematics/13-probability-statistics/09-problem-set-a.md) is good preparation.

The reference solution is [`ngram.py`](../../../../course/solutions/python/tinyllm/lm/ngram.py): `NGramLM(n, discount, vocab_size)` with `fit`, `_build`, `_estimate`, `prob`, `logprobs`, `nll`, `perplexity`, `discounts`, `save`, and `load`, an n-gram model with interpolated Kneser-Ney smoothing (modified, or with one fixed discount). Its contract is [`ngram.pyi`](../../../../course/contracts/py/tinyllm/lm/ngram.pyi) and its tests are [`test_ngram.py`](../../../../course/solutions/learner-tests/L2.1/python/tests/l2-1-ngram/test_ngram.py).

> **Learning objectives**
> - Explain why maximum-likelihood n-grams give infinite perplexity on new text, and how interpolated discounting fixes it while keeping every distribution summing to 1.
> - Compute continuation counts, adjusted counts, interpolation weights, and Kneser-Ney probabilities by hand for a small corpus.
> - Estimate the three modified-KN discounts from counts of counts, and recognize when the formulas must fall back.
> - Implement `NGramLM` with matching scalar (`prob`) and vectorized (`logprobs`) paths, correct context rules, scoring, and a byte-stable save format.

## Notation

The section uses the following symbols throughout; the table is a reference to come back to, and each symbol is introduced again in the text where it first matters.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size; tokens are ids $0 \dots V-1$ | `int` |
| $n$ | the order: an n-gram model conditions on $n - 1$ previous tokens | `int` |
| $w$, $v$ | tokens | ids |
| `<s>` | the start symbol, put before every sequence; a context symbol, never predicted (stored as $-1$) | |
| $h$ | a history: the $k - 1$ symbols before the predicted token at order $k$ | tuple |
| $h'$ | $h$ without its first symbol (the next lower order's history) | tuple |
| $c_k(g)$ | raw count of the $k$-gram $g$ | `int` |
| $N_{1+}(\bullet\, g)$ | continuation count: number of distinct symbols seen right before $g$ | `int` |
| $a_k(g)$ | adjusted count: $c_k(g)$ at the top order or when $g$ starts with `<s>`, else $N_{1+}(\bullet\, g)$ | `int` |
| $T_k(h) = \sum_v a_k(h v)$ | total adjusted count after $h$ | `int` |
| $N_j(h)$ | number of tokens $v$ with $a_k(hv) = j$ ($N_{3+}$: at least 3) | `int` |
| $D_{k1}, D_{k2}, D_{k3}$ | discounts for adjusted counts 1, 2, and 3 or more at order $k$ | `float` |
| $\gamma_k(h)$ | interpolation weight: the mass the discounts removed after $h$ | `float` |
| $p_k(w \mid h)$ | the order-$k$ estimate; $p_0(w) = 1/V$ | `float` |
| $n_j$ | counts of counts: number of $k$-grams with $a_k = j$ | `int` |
| $Y$ | $n_1 / (n_1 + 2 n_2)$, Ney's discount | `float` |

## n-gram models and the zero problem

The chain rule writes the probability of a sequence as a product of next-token probabilities, $P(w_1 \dots w_T) = \prod_t P(w_t \mid w_{<t})$. An **n-gram model** assumes the next token depends only on the last $n - 1$: $P(w_t \mid w_{<t}) \approx P(w_t \mid w_{t-n+1} \dots w_{t-1})$. The maximum-likelihood estimate, from [MLE, Laplace, absolute discounting](../../mathematics/13-probability-statistics/03-mle-laplace-absolute-discounting.md), is a ratio of counts, $c(h w) / \sum_v c(h v)$.

That estimate fails badly on real data. On a 2 KB corpus a byte 4-gram model has seen a few thousand of the possible $256^4$ four-byte strings. Any test sentence contains n-grams it never saw, each gets probability 0, and the perplexity is infinite. For count-based models, smoothing is the whole game. Before we can smooth, though, we have to be precise about what is being counted.

## Sequences, `<s>`, and contexts

Each training sequence (a line, a document) is padded with one start symbol: `a b c` becomes `<s> a b c`. The first real token is then predicted from the context `<s>`, which says "start of a sequence", so the model can learn that sentences start with capitals and stories with "Once". The n-grams of a padded sequence are the runs of $k \le n$ symbols that **end at a token**, so `<s>` can only be first. Because `<s>` is never predicted, every distribution is over the $V$ real tokens and sums to 1.

A query passes the tokens before the predicted one in the same sequence. Two rules decide what the model conditions on. A context with fewer than $n - 1$ tokens is the **start** of a sequence, so `<s>` goes in front of it. A longer one is cut to its **last** $n - 1$ tokens and gets no `<s>`. For a trigram, `[b]` means `<s> b`, while `[a, b]` and `[c, c, a, b]` both mean `a b`. The order actually used is $K = \min(n, \text{len(padded context)} + 1)$.

## Absolute discounting and interpolation

The first smoother subtracts a fixed $D \in (0, 1]$ from every seen count after a history $h$ and gives the removed mass to the next lower order. It does so at every context (interpolation), not only when the count is zero (backoff):

$$p_k(w \mid h) = \frac{\max(c(hw) - D, 0)}{T(h)} + \gamma(h)\, p_{k-1}(w \mid h'), \qquad \gamma(h) = \frac{D \cdot \#\{v : c(hv) > 0\}}{T(h)} .$$

The `absolute_discount` function from *Mathematics* is exactly this formula for one context. The recursion bottoms out at $p_0(w) = 1/V$, so every token has positive probability.

You can check that it sums to 1. The first term sums to $(T(h) - D \cdot \#\{\text{seen}\})/T(h) = 1 - \gamma(h)$, and $\gamma(h)$ times a distribution that sums to 1 adds back exactly $\gamma(h)$. This needs $D$ to be at most every count it is subtracted from; for $D \le 1$ and integer counts that always holds.

## Kneser-Ney: continuation counts

Absolute discounting still builds its lower orders from raw counts, and that is wrong in a specific way. The lower-order model is only consulted in proportion to $\gamma(h)$, that is, for histories where the higher order has little evidence. It should therefore answer "how likely is $w$ to appear in a **new** context?", not "how frequent is $w$?". In English "Francisco" is frequent but almost always follows "San"; as a continuation of an unfamiliar word it deserves little probability. Kneser and Ney's fix is to replace, below the top order, the count of a $k$-gram $g$ by its **continuation count** $N_{1+}(\bullet\, g)$, the number of distinct symbols seen right before it.

Two exceptions keep raw counts. The top order $n$ keeps them because nothing backs off to it from above. N-grams that start with `<s>` keep them because nothing ever precedes `<s>`, so their continuation count would be 0 (this is KenLM's rule). Call the result the adjusted count $a_k(g)$. The model, from $p_0(w) = 1/V$ up to order $K$, is

$$p_k(w \mid h) = \frac{\max\big(a_k(hw) - D_k(a_k(hw)), 0\big)}{T_k(h)} + \gamma_k(h)\, p_{k-1}(w \mid h'), \qquad \gamma_k(h) = \frac{D_{k1} N_1(h) + D_{k2} N_2(h) + D_{k3} N_{3+}(h)}{T_k(h)} ,$$

with $D_k(0) = 0$, $D_k(1) = D_{k1}$, $D_k(2) = D_{k2}$, $D_k(c \ge 3) = D_{k3}$. The weight $\gamma_k(h)$ is exactly the mass the discounts removed after $h$, bucket by bucket, so the distribution still sums to 1 as long as $D_{kj} \le j$. If no $k$-gram starts with $h$ (a history never seen), there is nothing to discount and $p_k(w \mid h) = p_{k-1}(w \mid h')$: the model falls through to the next lower order, never to the uniform distribution or to 0.

The solution computes this in two ways. `logprobs` computes all $V$ probabilities at once with numpy, order by order; `prob` computes one with dictionary lookups. Both must give the same numbers.

## Modified Kneser-Ney: three discounts

One discount is a compromise. A count of 1 is mostly noise and deserves a large discount relative to its size; a count of 50 deserves almost none. Chen and Goodman estimate three discounts per order from the **counts of counts** of the adjusted counts at that order, $n_j = \#\{g : a_k(g) = j\}$:

$$Y = \frac{n_1}{n_1 + 2 n_2}, \qquad D_{k1} = 1 - 2Y\frac{n_2}{n_1}, \qquad D_{k2} = 2 - 3Y\frac{n_3}{n_2}, \qquad D_{k3} = 3 - 4Y\frac{n_4}{n_3} .$$

$Y$ is Ney's single discount, `ney_discount` applied to the adjusted counts. On small data the formulas break: some $n_j$ is 0 (division by zero), or a discount falls outside $(0, j]$, either negative or larger than the count it is subtracted from, which would make probabilities negative. The contract then uses KenLM's fallback $(0.5, 1.0, 1.5)$ for that order. The worked example's trigram falls back at every order; the 2 KB byte corpus of the tests has enough data from order 2 up. With a fixed discount $d$, $D_{k1} = D_{k2} = D_{k3} = d$ everywhere.

Here is a hand case where the formulas do work. A unigram model whose adjusted counts are 1, 1, 1, 1, 2, 2, 2, 3, 3, 4 has $n_1 = 4$, $n_2 = 3$, $n_3 = 2$, $n_4 = 1$, so $Y = 4 / (4 + 6) = 0.4$ and

$$D_1 = 1 - 2(0.4)\tfrac34 = 0.4, \qquad D_2 = 2 - 3(0.4)\tfrac23 = 1.2, \qquad D_3 = 3 - 4(0.4)\tfrac12 = 2.2 .$$

All three lie in $(0, j]$, so there is no fallback.

## Scoring and saving

With probabilities defined, scoring follows directly. `nll(ids)` scores one sequence token by token, $-\ln p(w_t \mid w_{<t})$, the first token from `<s>`. `perplexity(ids)` is $\exp$ of their mean, summed with `NLLAccumulator` from [Perplexity, bits per byte, NLL accumulator](../../mathematics/17-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md) so the result is exact over long texts. Bits per byte, the zoo's metric, is the same total over the byte count.

`save` writes one safetensors file through the writer from [Safetensors for every dtype, atomic checkpoints, and the token stream](../06-numerics-and-autograd/07-safetensors-checkpoints-and-token-streams.md). For every order $k$ it stores an `I32` tensor `order{k}.grams` with every $k$-gram seen (`<s>` as $-1$, rows in ascending order) and `order{k}.counts` with their raw counts, plus metadata `tl_arch = "ngram"`, `n`, `discount`, `vocab_size`. Everything else (adjusted counts, discounts, $\gamma$) is recomputed by `load`, so the file holds only data and saving a loaded model writes the same bytes.

## Working it by hand

Take three sequences over $V = 3$ tokens `a`, `b`, `c` (ids 0, 1, 2), a bigram model ($n = 2$), and a fixed discount $d = 1/2$:

```
<s> a b c
<s> a b a b
<s> c a b c
```

The bigram counts are the top order, so they stay raw: `<s> a` 2, `<s> c` 1, `a b` 4, `b c` 2, `b a` 1, `c a` 1.

At the unigram level we use continuation counts. `a` follows `<s>`, `b`, and `c`: 3. `b` follows only `a`: 1. `c` follows `b` and `<s>`: 2. So $T_1 = 6$, there are three seen types, $\gamma_1 = 3 \cdot \frac12 / 6 = \frac14$, and

$$p_1(a) = \frac{3 - \frac12}{6} + \frac14 \cdot \frac13 = \frac12, \qquad p_1(b) = \frac{1 - \frac12}{6} + \frac1{12} = \frac16, \qquad p_1(c) = \frac{2 - \frac12}{6} + \frac1{12} = \frac13 .$$

Notice that `b` is the most frequent token (4 raw occurrences, like `a`) but gets the least unigram mass, because it only ever follows `a`. With raw counts (4, 4, 3) the same formula would give $p_1(b) = \frac{3.5}{11} + \frac{3}{22} \cdot \frac13 = \frac{4}{11} \approx 0.36$, more than twice as much. That gap is the whole point of Kneser-Ney.

Now condition on `b`. The bigrams after it are `b a` 1 and `b c` 2, so $T = 3$ and $\gamma = 2 \cdot \frac12 / 3 = \frac13$. The table below adds the discounted count to the interpolated unigram for each token:

| $w$ | discounted count / $T$ | $+\ \gamma\, p_1(w)$ | $p(w \mid b)$ |
|---|---|---|---|
| `a` | $0.5 / 3 = 1/6$ | $\frac13 \cdot \frac12 = \frac16$ | $1/3$ |
| `b` | $0$ | $\frac13 \cdot \frac16 = \frac1{18}$ | $1/18$ |
| `c` | $1.5 / 3 = 1/2$ | $\frac13 \cdot \frac13 = \frac19$ | $11/18$ |

They sum to $\frac{6 + 1 + 11}{18} = 1$. After `c` there is only `c a`, so $T = 1$, $\gamma = \frac12$, and $p(\cdot \mid c) = (\frac34, \frac1{12}, \frac16)$.

At the start of a sequence, `<s> a` 2 and `<s> c` 1 give $T = 3$ and $\gamma = \frac13$: $p(a \mid \texttt{<s>}) = \frac{1.5}{3} + \frac16 = \frac23$, $p(b \mid \texttt{<s>}) = \frac1{18}$, $p(c \mid \texttt{<s>}) = \frac{0.5}{3} + \frac19 = \frac5{18}$. The sequence `a b` therefore costs $-\ln\frac23 - \ln p(b \mid a)$ nats with $p(b \mid a) = \frac{3.5}{4} + \frac18 \cdot \frac16 = \frac{43}{48}$.

These numbers are the first hand cases the tests pin, to $10^{-15}$.

## Designing the API

The interface is small. A model is constructed with an order, a discount policy, and optionally a vocabulary size; `fit` consumes sequences of token ids and replaces any earlier counts; the rest are queries and persistence.

```python
class NGramLM:
    def __init__(self, n: int, discount: float | Literal["modified"] = "modified",
                 vocab_size: int | None = None) -> None: ...
    def fit(self, sequences: Iterable[Sequence[int]]) -> None: ...      # replaces earlier counts
    def discounts(self, order: int) -> tuple[float, float, float]: ...
    def prob(self, context: Sequence[int], token: int) -> float: ...
    def logprobs(self, context: Sequence[int]) -> NDArray: ...         # [V]
    def nll(self, ids: Sequence[int]) -> NDArray: ...                  # [len(ids)], first token from <s>
    def perplexity(self, ids: Sequence[int]) -> float: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "NGramLM": ...
FALLBACK_DISCOUNTS = (0.5, 1.0, 1.5)
```

`discount` is either `"modified"` (three estimated discounts per order) or a single float used everywhere. If `vocab_size` is omitted, $V$ is inferred as one more than the largest id seen. `logprobs` returns a length-$V$ vector of natural-log probabilities; `nll` returns one value per input token.

## Building it step by step

Before reading code, it helps to see that the design is forced by a handful of requirements.

Every probability must be positive and every distribution must sum to 1. Positivity forces a floor that is never zero, so the recursion ends at $p_0 = 1/V$. Summing to 1 forces the interpolation weight to be exactly the mass the discounts removed, so $\gamma_k(h)$ is not a free parameter but $\sum_j D_{kj} N_j(h) / T_k(h)$.

The lower order answers a different question. It is consulted in proportion to $\gamma$, for histories with little evidence, so it should model "a token in a new context", which is the number of distinct left neighbours. That forces the adjusted count: raw at the top order, continuation counts below it. And because `<s>` has no left neighbour, a continuation count for a `<s>`-initial gram would be 0 and the start of every sequence would lose its information; those grams keep raw counts at every order.

Everything the queries need depends only on the history. For each order and history we need the adjusted counts of what follows, their total, and $\gamma$. Precomputing those once in `fit` makes a query $O(n)$ dictionary lookups, and lets the saved file hold only raw counts, since everything else is a function of them.

Finally, discounts come from the data, but formulas can break. Chen and Goodman's estimates divide by $n_1, n_2, n_3$ and can leave $(0, j]$ on small data; a discount above $j$ makes a numerator negative, so the solution must check and fall back. And the context rules are part of the model: `nll` must apply the same rule as `prob`, token by token.

### Counting

`fit` makes one pass over each padded sequence. Position `i` of `padded` ends one gram of every length up to `min(n, i + 1)`, so `<s>` (at position 0) can only ever be the first symbol of a gram and is never counted as a predicted token.

```python
def fit(self, sequences: Iterable[Sequence[int]]) -> None:
    raw: dict[int, Counter] = {k: Counter() for k in range(1, self.n + 1)}  # fresh: fit replaces
    biggest = -1
    for seq in sequences:
        ids = [int(t) for t in seq]
        if not ids:
            continue
        lo, hi = min(ids), max(ids)
        if lo < 0 or (self._fixed_vocab is not None and hi >= self._fixed_vocab):
            raise ValueError(...)
        biggest = max(biggest, hi)
        padded = [BOS] + ids                      # BOS = -1, never a token id
        for i in range(1, len(padded)):           # every gram ends at a token
            for k in range(1, min(self.n, i + 1) + 1):
                raw[k][tuple(padded[i - k + 1 : i + 1])] += 1
    if biggest < 0:
        raise ValueError("NGramLM.fit: no sequence holds a token")
    self.vocab_size = self._fixed_vocab or biggest + 1   # ids 0 .. biggest: V is one more
    self._build(raw)
```

### Adjusted counts and per-history tables

Continuation counts of order $k$ come from the distinct $(k+1)$-grams: each one $(v, g)$ adds 1 to $g$. The rest is grouping by history and precomputing $(\text{next counts}, T, \gamma)$.

```python
def _build(self, raw: dict[int, Counter]) -> None:
    self._raw = raw                               # only raw counts are saved
    n = self.n
    adj = {}
    for k in range(n, 0, -1):
        if k == n:
            adj[k] = dict(raw[k])                 # top order: raw counts
            continue
        cont = Counter()
        for g in raw[k + 1]:                      # each distinct (v, g) is one left neighbour of g
            cont[g[1:]] += 1
        adj[k] = {g: (c if g[0] == BOS else cont[g]) for g, c in raw[k].items()}  # <s>-initial: raw
    self._disc, self._hist = {}, {}
    for k in range(1, n + 1):
        D = self._estimate(adj[k])
        self._disc[k] = D
        by_hist = {}
        for g, a in adj[k].items():
            by_hist.setdefault(g[:-1], {})[g[-1]] = a  # history -> {next token: adjusted count}
        table = {}
        for h, nxt in by_hist.items():
            total = float(sum(nxt.values()))
            n1 = sum(1 for a in nxt.values() if a == 1)
            n2 = sum(1 for a in nxt.values() if a == 2)
            n3 = sum(1 for a in nxt.values() if a >= 3)
            gamma = (D[0] * n1 + D[1] * n2 + D[2] * n3) / total  # exactly the mass removed
            table[h] = (nxt, total, gamma)
        self._hist[k] = table
    self._fitted = True
```

Trace it on the hand example. `raw[2]` has `<s> a`, `<s> c`, `a b`, `b c`, `b a`, `c a`; `cont` gives `a` 3 (from `<s> a`, `b a`, `c a`), `b` 1, `c` 2, which are the continuation counts computed by hand. The unigram history is `()`, with $T_1 = 6$ and $\gamma_1 = 1/4$.

### Discounts

The counts of counts are over the adjusted counts of one order, which is why `ney_discount` is called on `adj_k` and not on raw counts.

```python
def _estimate(self, adj_k: dict[tuple, int]) -> tuple[float, float, float]:
    if self.discount != "modified":
        d = float(self.discount)
        return (d, d, d)                           # plain interpolated KN
    nj = Counter(a for a in adj_k.values() if 1 <= a <= 4)
    n1, n2, n3, n4 = nj[1], nj[2], nj[3], nj[4]
    if min(n1, n2, n3, n4) == 0:
        return FALLBACK_DISCOUNTS                  # an empty bucket: a formula divides by 0 or is degenerate
    Y = ney_discount(adj_k)                        # n1 / (n1 + 2 n2)
    D = (1.0 - 2.0 * Y * n2 / n1, 2.0 - 3.0 * Y * n3 / n2, 3.0 - 4.0 * Y * n4 / n3)
    if not all(0.0 < d <= j for d, j in zip(D, (1, 2, 3))):
        return FALLBACK_DISCOUNTS                  # negative or larger than the count: unsafe
    return D
```

Note that the $D_3$ formula uses $n_4 / n_3$, not $n_3 / n_2$: each $D_j$ compares the $j$ and $j + 1$ buckets.

### The context rule

```python
def _padded(self, context: Sequence[int]) -> tuple:
    ctx = tuple(int(t) for t in context)
    for t in ctx:
        self._check_id(t)
    if len(ctx) < self.n - 1:
        return (BOS,) + ctx                        # too short: it is a sequence start
    return ctx[len(ctx) - (self.n - 1):]           # long: keep the LAST n - 1 tokens
```

For $n = 1$, `self.n - 1` is 0, so the slice is empty and the model is the KN unigram regardless of context.

### One probability

The recursion runs bottom-up: start at the uniform floor and, order by order, replace `p` with "discounted count plus $\gamma$ times the lower order". An unseen history skips its order, so `p` stays the lower-order estimate, which is exactly $p_k = p_{k-1}$ when nothing follows $h$.

```python
def prob(self, context: Sequence[int], token: int) -> float:
    self._ready()
    self._check_id(int(token))
    w = int(token)
    hist = self._padded(context)
    K = min(self.n, len(hist) + 1)                 # the highest order this context supports
    p = 1.0 / self.vocab_size                      # p_0
    for k in range(1, K + 1):
        h = hist[len(hist) - (k - 1):] if k > 1 else ()   # the last k - 1 symbols
        entry = self._hist[k].get(h)
        if entry is None:
            continue                               # unseen history: fall through, not to 0
        nxt, total, gamma = entry
        D = self._disc[k]
        a = nxt.get(w, 0)
        d = 0.0 if a == 0 else D[min(a, 3) - 1]    # D_1, D_2, or D_3+ by the count's bucket
        p = max(a - d, 0.0) / total + gamma * p
    return p
```

### All $V$ at once

`logprobs` is the same loop with arrays: scatter the next-token counts into a length-$V$ vector, choose each entry's discount with `np.select`, and apply the same update.

```python
p = np.full(V, 1.0 / V)
for k in range(1, K + 1):
    # ... same h and entry lookup as prob ...
    a = np.zeros(V)
    idx = np.fromiter(nxt.keys(), dtype=np.int64, count=len(nxt))
    a[idx] = np.fromiter(nxt.values(), dtype=np.float64, count=len(nxt))
    d = np.select([a == 0, a == 1, a == 2], [0.0, D[0], D[1]], default=D[2])  # same buckets as prob
    p = np.maximum(a - d, 0.0) / total + gamma * p
with np.errstate(divide="ignore"):
    return np.log(p)
```

### Scoring

`nll` gives each token the same context `prob` would get from the whole prefix, without copying the prefix: while `t < n - 1` the slice is the whole prefix (a sequence start), afterwards the last $n - 1$ tokens. The first token is scored from `<s>`.

```python
def nll(self, ids: Sequence[int]) -> NDArray:
    seq = [int(t) for t in ids]
    m = self.n - 1
    return np.array([-math.log(self.prob(seq[max(0, t - m) : t], seq[t]))
                     for t in range(len(seq))], dtype=np.float64)

def perplexity(self, ids: Sequence[int]) -> float:
    if len(ids) == 0:
        raise ValueError("NGramLM.perplexity: empty sequence")
    acc = NLLAccumulator()                         # exact summation over long texts (M11.2)
    acc.add(self.nll(ids))
    return acc.result()["ppl"]
```

### The file

`save` writes, per order, the sorted raw grams (`<s>` as $-1$) as an `I32` tensor of shape `[count, k]` and their counts, plus four metadata strings. `load` rebuilds the `Counter`s and calls the same `_build`, so a reloaded model is the same model, and sorting makes re-saving byte-identical.

The full solution is [`ngram.py`](../../../../course/solutions/python/tinyllm/lm/ngram.py).

### Why this design

The costs are modest. `fit` is $O(n \cdot N)$ counter updates for $N$ training tokens, plus $O(G)$ to build the tables for $G$ distinct grams; memory is $O(G)$. `prob` is $O(n)$ dictionary lookups, `logprobs` is $O(nV)$, and `nll` is $O(nT)$ for $T$ tokens.

Precomputing per history is what makes queries cheap. Without it, $\gamma_k(h)$ and $T_k(h)$ would be recomputed by summing over every continuation of $h$ on each query, which is $O(V)$ per order; precomputing turns queries into lookups and costs one pass at `fit` time.

Interpolation is preferred over backoff because backoff consults the lower order only for unseen continuations and needs extra normalization weights to sum to 1, while interpolation mixes at every context, sums to 1 by construction, and empirically does better with KN (Chen and Goodman).

The file holds raw counts because adjusted counts, discounts, and $\gamma$ are derived; storing them would duplicate state that could disagree with the counts. KenLM's ARPA format does store probabilities, trading file size for query speed, which the zoo does not need. Likewise, plain Python dicts rather than suffix arrays or tries are the right choice for megabyte corpora: they are simple and exact, and KenLM and infini-gram exist for the 100 GB case. The tradeoffs accepted are that memory grows with distinct grams, and `logprobs` builds a dense $V$-vector per order, which is fine for $V = 256$ and acceptable for subword vocabularies.

## What can go wrong

The tests check a specific list of behaviours, and each corresponds to something a later chapter depends on. The hand example's $p(\cdot \mid b)$, $p(\cdot \mid c)$, and $p(\cdot \mid \texttt{<s>}) = (2/3, 1/18, 5/18)$ must match to $10^{-15}$, since every sequence starts at `<s>` and the unigram level exercises continuation counts. Golden cases compare every context of the example, as a bigram and as a modified-KN trigram, against exact fractions from an independent implementation, which catches `<s>`-initial grams losing their raw counts at lower orders. A golden byte corpus (the story corpus from Part 0 as bytes, $n = 1 \dots 4$, modified and $d = 0.75$) checks discounts, held-out perplexities, and probabilities after "the "; end-to-end perplexity must match the reference within 0.5%.

Several properties hold for any input. $\sum_v p(v \mid \text{ctx}) = 1$ for seen, unseen, and start contexts at $n = 1 \dots 4$, because speculative decoding needs a proper distribution for exact acceptance. The vectorized path equals `prob` for every token and has logsumexp 0, since the zoo and speculative decoding both call `logprobs`. An unseen history gives the KN unigram, $(23, 7, 15, 3)/48$ in the test, with no jump to uniform or to 0. A short context means sequence start and a long one keeps its last $n - 1$ tokens, because the draft model passes whole prefixes. Zero counts of counts and a negative $D_3$ both give $(0.5, 1, 1.5)$ and the result still sums to 1. Per-token NLL starts from `<s>`, perplexity is $\exp(\text{mean})$, and empty input raises. A save and load round trip keeps header `tl_arch = "ngram"`, I32 tensors, equal probabilities, and identical bytes on re-save. A second `fit` forgets the first. Bad $n$, bad discounts, ids outside $0 \dots V-1$, and queries before `fit` all fail early, and $V$ is inferred correctly.

The bugs that break these are worth knowing by name. The table lists each one, the symptom you would see, and the line of the solution that prevents it.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| lower orders from raw counts (absolute discounting, not Kneser-Ney) | frequent tokens get too much mass as fresh continuations; `p(b)` at the unigram level is 4/11 instead of 1/6 | `_build` replaces counts below the top order with `cont[g]` |
| $\gamma$ charged with $D_1$ for every seen token | with three discounts, the distribution no longer sums to 1 | `gamma` weights $N_1$, $N_2$, $N_{3+}$ by their own discounts |
| continuation counts for n-grams that start with `<s>` | they become 0: start-of-sequence predictions fall to the lower order | `c if g[0] == BOS else cont[g]` |
| no fallback when a discount leaves $(0, j]$ | a negative $D_3$ gives negative probabilities on small data | `_estimate` checks `0 < d <= j` and returns `FALLBACK_DISCOUNTS` |
| an unseen history jumps to the uniform distribution | every rare context forgets what the unigram level knows | `prob` and `logprobs` `continue` past a missing history, keeping `p` |
| keeping the first $n - 1$ tokens of a long context | the draft model conditions on the start of the prompt, not its end | `_padded` slices `ctx[len(ctx) - (n - 1):]` |
| counts of 3 or more discounted with $D_2$ | $\gamma$ and the numerators disagree: no longer sums to 1 | `D[min(a, 3) - 1]`, the same bucket rule as `gamma` |
| the first token of a sequence not scored | perplexity skips the hardest prediction | `nll` loops from `t = 0`, with an empty prefix padded to `<s>` |
| `<s>`-initial n-grams dropped by `save` | a reloaded model predicts sequence starts differently | `save` writes every gram in `_raw`, `<s>` as $-1$ |
| `logprobs` using $D_1$ for every count | the vectorized path disagrees with `prob` | `np.select` over the same three buckets |
| $D_3$ computed from $n_3 / n_2$ | wrong discounts at every order with enough data | $D_3 = 3 - 4Y n_4 / n_3$ |
| `fit` adding to the previous counts | a reused model mixes two corpora | `fit` builds new `Counter`s each call |
| inferred $V$ = the largest id instead of 1 + it | the largest token becomes unpredictable | `biggest + 1` |

> **Warning:** The two most common Kneser-Ney bugs are silent. Building lower orders from raw counts still gives a valid distribution, just a worse one, and charging $\gamma$ with $D_1$ for everything only breaks normalization when the three discounts differ. A sums-to-one property test across many contexts catches the second; only the hand example catches the first.

## Where this leads

End to end, `tinyllm lm train ngram --n 4` followed by `tinyllm eval` (see [`cli_lm.py`](../../../../course/solutions/entry/python/tinyllm/cli_lm.py)) gives the statistical-LM perplexity baseline. [The model zoo](../12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) trains KN-4 on TinyStories and reports its bits per byte, the floor the neural models must beat. [Speculative decoding](../21-inference/06-speculative-decoding.md) uses the n-gram model as a draft that proposes tokens from `logprobs` for the target model to verify. In [the TinyStories capstone](../capstones/01-tinystories.md), the corpus pipeline's `ppl_filter` from [Extract, normalize, quality filters](../../data-engineering/05-corpus-pipeline/02-extract-normalize-and-quality-filters.md) in *Data Engineering* drops documents whose KN perplexity is too high, and quality checks score samples with it. The next section, [Bengio's neural probabilistic language model](02-bengio-nplm.md), replaces the count table with learned embeddings and beats this baseline.

Each piece here has a production counterpart worth reading:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| counting in dicts | KenLM `lmplz` | disk-based streaming sorts, so a 100 GB corpus is counted in bounded memory; pruning of rare n-grams | `lm/builder/` (`corpus_count.cc`, `adjust_counts.cc`, `initial_probabilities.cc`) |
| `save` with every n-gram | KenLM's ARPA and binary formats | probabilities and backoff weights precomputed per n-gram; a trie or probing hash table with quantized values for fast queries | `lm/binary_format.cc`, `lm/trie.cc` |
| `NGramLM` as a quality filter | CCNet and RedPajama | a KenLM 5-gram trained on Wikipedia scores every web document; perplexity buckets decide what is kept | `facebookresearch/cc_net`, `cc_net/perplexity.py` |
| n-gram draft model | prompt-lookup and n-gram speculative decoding in vLLM | drafts from n-gram matches against the prompt, no extra model | `vllm/v1/spec_decode/ngram_proposer.py` |
| interpolated KN | the infini-gram index (Liu et al. 2024) | unbounded-order n-gram counts over trillions of tokens with suffix arrays | `liujch1998/infini-gram` |

### Further reading

- Chen and Goodman, "An empirical study of smoothing techniques for language modeling" (1998), sections 2.7, 3, and 4.1.6.
- Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), 3rd ed., chapter 3.
- Heafield et al., "Scalable modified Kneser-Ney language model estimation" (ACL 2013, [KenLM](https://kheafield.com/code/kenlm/)).

## Key terms

- **n-gram model**: a language model that conditions each token only on the previous $n - 1$ tokens.
- **interpolation**: mixing a discounted higher-order estimate with the lower-order estimate at every context, weighted by $\gamma$.
- **continuation count**: $N_{1+}(\bullet\, g)$, the number of distinct symbols seen immediately before $g$.
- **adjusted count**: the raw count at the top order or for `<s>`-initial grams, the continuation count otherwise.
- **counts of counts**: $n_j$, the number of grams at one order whose adjusted count is $j$.
- **modified Kneser-Ney**: Kneser-Ney with three discounts per order ($D_1$, $D_2$, $D_{3+}$) estimated from counts of counts.

## Summary

- Maximum-likelihood n-grams give probability 0 to every unseen continuation, so their perplexity on new text is infinite; smoothing moves a little mass from seen counts to a lower-order model, at every context.
- Kneser-Ney's lower orders count how many different words a token **follows** (its continuation count), not how often it occurs: a frequent token that only ever follows one word gets little mass as a fresh continuation.
- The start symbol `<s>` has nothing before it, so n-grams that begin with it keep their raw counts.
- Modified KN uses three discounts per order, $D_1, D_2, D_{3+}$, estimated from the counts of counts, with a fixed fallback $(0.5, 1, 1.5)$ when the data are too small for the formulas.
- The same numbers come out of `prob`, the vectorized `logprobs`, and a reloaded file.

## Review questions

1. Why is $\gamma_k(h) = (D_1 N_1 + D_2 N_2 + D_3 N_{3+}) / T_k(h)$ and not $D_1 \cdot \#\{\text{seen}\} / T_k(h)$?
   <details><summary>Answer</summary>$\gamma$ must equal the mass the numerators lost, so that the first term sums to $1 - \gamma$ and $\gamma$ times a normalised lower order adds it back. A token with count 2 lost $D_2$, not $D_1$; charging $D_1$ for everything makes the distribution sum to something other than 1 whenever the three discounts differ.</details>
2. Why do `<s>`-initial grams keep raw counts at lower orders?
   <details><summary>Answer</summary>A continuation count is the number of distinct symbols seen right before the gram, and nothing ever precedes `<s>`, so it would be 0. Every start-of-sequence context would then have no evidence and fall through to the lower order, forgetting that stories start with "Once".</details>
3. What does the model predict after a history it never saw, and why not the uniform distribution?
   <details><summary>Answer</summary>The order with the unseen history is skipped, so the result is the next lower order's estimate, down to the KN unigram. Jumping to uniform would discard everything the lower orders know: a bigram over the hand example's corpus would answer an unseen history with the KN unigram $(1/2, 1/6, 1/3)$, not $1/3$ each.</details>
4. Why must a discount be at most $j$ for counts in bucket $j$?
   <details><summary>Answer</summary>The numerator is $\max(a - D, 0)$; if $D > a$ the max clips at 0 but $\gamma$ still adds $D$, so the total exceeds 1. The formulas can produce such values (or negative ones) on small data, which is why `_estimate` checks $0 < D_j \le j$ and falls back to $(0.5, 1, 1.5)$.</details>
5. Why is the context `[c, c, a, b]` the same as `[a, b]` for a trigram, but `[b]` is not the same as `[a, b]`?
   <details><summary>Answer</summary>A trigram conditions on two symbols. A context of 2 or more tokens keeps its last two, so both long contexts are `a b`. A context of 1 token is shorter than $n - 1$, which can only happen at the start of a sequence, so it becomes `<s> b`.</details>
