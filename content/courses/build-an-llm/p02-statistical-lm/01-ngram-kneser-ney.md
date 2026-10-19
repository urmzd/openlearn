# n-gram language model with interpolated modified Kneser-Ney

## Overview

| | |
|---|---|
| **Module** | `L2.1` · build · Python · Pass 3 |
| **Solution** | [`ngram.py`](../../../../course/solutions/python/tinyllm/lm/ngram.py): `NGramLM(n, discount, vocab_size)` with `fit`, `_build`, `_estimate`, `prob`, `logprobs`, `nll`, `perplexity`, `discounts`, `save`, `load`: an n-gram model with interpolated Kneser-Ney smoothing (modified, or one fixed discount) · model tests [`test_ngram.py`](../../../../course/solutions/learner-tests/L2.1/python/tests/l2-1-ngram/test_ngram.py) |
| **Contract** | [`course/contracts/py/tinyllm/lm/ngram.pyi`](../../../../course/contracts/py/tinyllm/lm/ngram.pyi) |
| **Needs** | `M07.2` `ney_discount` · `M11.2` `NLLAccumulator` for perplexity · `L0.6` `save_safetensors` and `load_safetensors`. Reading: `S-M07a` (conditional probability), `L0.0` (the bigram this generalizes) |
| **Used by** | later `L6.7` the zoo's KN baseline, `L8.6` the n-gram draft model of speculative decoding, `C1` the corpus perplexity filter plugged into `data.02`'s `ppl_filter` · end to end: `tinyllm lm train ngram --n 4`, then `tinyllm eval` ([`cli_lm.py`](../../../../course/solutions/entry/python/tinyllm/cli_lm.py)), gives the statistical-LM layer's perplexity baseline |
| **Optional depth** | Chen and Goodman, "An empirical study of smoothing techniques for language modeling" (1998), sections 2.7, 3, and 4.1.6; Jurafsky and Martin, *Speech and Language Processing*, 3rd ed., ch. 3; Heafield et al., "Scalable modified Kneser-Ney language model estimation" (ACL 2013, KenLM) |

## Key Takeaways

- Maximum-likelihood n-grams give probability 0 to every unseen continuation, so their perplexity on new text is infinite; smoothing moves a little mass from seen counts to a lower-order model, at every context.
- Kneser-Ney's lower orders count how many different words a token **follows** (its continuation count), not how often it occurs: a frequent token that only ever follows one word gets little mass as a fresh continuation.
- The start symbol `<s>` has nothing before it, so n-grams that begin with it keep their raw counts.
- Modified KN uses three discounts per order, $D_1, D_2, D_{3+}$, estimated from the counts of counts, with a fixed fallback when the data are too small for the formulas.
- The same numbers come out of `prob`, the vectorized `logprobs`, and a reloaded file.

## How to read this chapter

Sections 1 and 2 build Kneser-Ney from the zero-probability problem, one fix at a time. Section 3 computes a bigram model by hand, to exact fractions. Section 4 fixes the interface. Section 5 walks through the solution: counting, adjusted counts, discounts, the recursion, scoring, and the file, and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The Pass 1 bigram (`L0.0`) predicts the next byte from one byte of history, and the Pass 2 trainer (`L0.5`) fits the same table by gradient descent. Neither can use more context: a table indexed by the last three bytes has $256^3 \approx 1.7 \times 10^7$ rows, almost all never seen, and an unseen row means probability 0 for the true next token and an infinite loss. This module is the classic answer, used in speech recognition and machine translation for twenty years before neural models: count n-grams, then smooth the counts so that every context still gives every token some probability. Interpolated modified Kneser-Ney is the strongest of those smoothers, the baseline every neural model in the zoo (`L6.7`) has to beat on bits per byte, a cheap draft model for speculative decoding (`L8.6`), and in the capstone the perplexity filter that throws out garbage documents before training.

## 2. Principles

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
| $Y$ | $n_1 / (n_1 + 2 n_2)$, Ney's discount (`M07.2`) | `float` |

### 2.1 n-gram models and the zero problem

The chain rule writes the probability of a sequence as a product of next-token probabilities, $P(w_1 \dots w_T) = \prod_t P(w_t \mid w_{<t})$. An **n-gram model** assumes the next token depends only on the last $n - 1$: $P(w_t \mid w_{<t}) \approx P(w_t \mid w_{t-n+1} \dots w_{t-1})$. The maximum-likelihood estimate (`M07.2`) is a ratio of counts, $c(h w) / \sum_v c(h v)$. On a 2 KB corpus a byte 4-gram model has seen a few thousand of the possible $256^4$ four-byte strings; any test sentence contains n-grams it never saw, each gets probability 0, and the perplexity is infinite. Smoothing is the whole game.

### 2.2 Sequences, `<s>`, and contexts

Each training sequence (a line, a document) is padded with one start symbol: `a b c` becomes `<s> a b c`. The first real token is then predicted from the context `<s>`, which says "start of a sequence"; a model learns that sentences start with capitals and stories with "Once". The n-grams of a padded sequence are the runs of $k \le n$ symbols that **end at a token**, so `<s>` can only be first. `<s>` is never predicted, so every distribution is over the $V$ real tokens and sums to 1.

A query passes the tokens before the predicted one in the same sequence. Two rules decide what the model conditions on: a context with fewer than $n - 1$ tokens is the **start** of a sequence, so `<s>` goes in front of it; a longer one is cut to its **last** $n - 1$ tokens and gets no `<s>`. For a trigram, `[b]` means `<s> b` while `[a, b]` and `[c, c, a, b]` both mean `a b`. The order actually used is $K = \min(n, \text{len(padded context)} + 1)$.

### 2.3 Absolute discounting and interpolation

Absolute discounting subtracts a fixed $D \in (0, 1]$ from every seen count after a history $h$ and gives the removed mass to the next lower order, at every context (interpolation), not only when the count is zero (backoff):

$$p_k(w \mid h) = \frac{\max(c(hw) - D, 0)}{T(h)} + \gamma(h)\, p_{k-1}(w \mid h'), \qquad \gamma(h) = \frac{D \cdot \#\{v : c(hv) > 0\}}{T(h)} .$$

`M07.2`'s `absolute_discount` is this formula for one context. The recursion bottoms out at $p_0(w) = 1/V$, so every token has positive probability. Why it sums to 1: the first term sums to $(T(h) - D \cdot \#\{\text{seen}\})/T(h) = 1 - \gamma(h)$, and $\gamma(h)$ times a distribution that sums to 1 adds back exactly $\gamma(h)$. This needs $D \le$ every count it is subtracted from; for $D \le 1$ and integer counts that always holds.

### 2.4 Kneser-Ney: continuation counts

Absolute discounting still builds its lower orders from raw counts, and that is wrong in a specific way. The lower-order model is only consulted in proportion to $\gamma(h)$, that is, for histories where the higher order has little evidence. It should answer "how likely is $w$ to appear in a **new** context?", not "how frequent is $w$?". In English "Francisco" is frequent but almost always follows "San"; as a continuation of an unfamiliar word it deserves little probability. Kneser and Ney's fix: below the top order, replace the count of a $k$-gram $g$ by its **continuation count** $N_{1+}(\bullet\, g)$, the number of distinct symbols seen right before it.

Two exceptions keep raw counts: the top order $n$ (nothing is backed off to from above), and n-grams that start with `<s>`, since nothing ever precedes `<s>` and their continuation count would be 0 (KenLM's rule). Call the result the adjusted count $a_k(g)$. The model, from $p_0(w) = 1/V$ up to order $K$:

$$p_k(w \mid h) = \frac{\max\big(a_k(hw) - D_k(a_k(hw)), 0\big)}{T_k(h)} + \gamma_k(h)\, p_{k-1}(w \mid h'), \qquad \gamma_k(h) = \frac{D_{k1} N_1(h) + D_{k2} N_2(h) + D_{k3} N_{3+}(h)}{T_k(h)} ,$$

with $D_k(0) = 0$, $D_k(1) = D_{k1}$, $D_k(2) = D_{k2}$, $D_k(c \ge 3) = D_{k3}$. The weight $\gamma_k(h)$ is exactly the mass the discounts removed after $h$, bucket by bucket, so the distribution still sums to 1 as long as $D_{kj} \le j$. If no $k$-gram starts with $h$ (a history never seen), there is nothing to discount and $p_k(w \mid h) = p_{k-1}(w \mid h')$: the model falls through to the next lower order, never to the uniform distribution or to 0.

`logprobs` computes all $V$ probabilities at once with numpy, order by order; `prob` computes one with dictionary lookups. Both must give the same numbers.

### 2.5 Modified Kneser-Ney: three discounts

One discount is a compromise: a count of 1 is mostly noise and deserves a large discount relative to its size, a count of 50 almost none. Chen and Goodman estimate three discounts per order from the **counts of counts** of the adjusted counts at that order, $n_j = \#\{g : a_k(g) = j\}$:

$$Y = \frac{n_1}{n_1 + 2 n_2}, \qquad D_{k1} = 1 - 2Y\frac{n_2}{n_1}, \qquad D_{k2} = 2 - 3Y\frac{n_3}{n_2}, \qquad D_{k3} = 3 - 4Y\frac{n_4}{n_3} .$$

$Y$ is Ney's single discount, `M07.2`'s `ney_discount` applied to the adjusted counts. On small data the formulas break: some $n_j$ is 0 (division by zero), or a discount falls outside $(0, j]$ (negative, or larger than the count it is subtracted from, which would make probabilities negative). The contract then uses KenLM's fallback $(0.5, 1.0, 1.5)$ for that order. The worked example's trigram falls back at every order; the 2 KB byte corpus of the tests has enough data from order 2 up. With a fixed discount $d$, $D_{k1} = D_{k2} = D_{k3} = d$ everywhere.

### 2.6 Scoring and saving

`nll(ids)` scores one sequence token by token, $-\ln p(w_t \mid w_{<t})$, the first token from `<s>`; `perplexity(ids)` is $\exp$ of their mean, summed with `M11.2`'s `NLLAccumulator` so the result is exact over long texts. Bits per byte, the zoo's metric, is the same total over the byte count (`M11.2`).

`save` writes one safetensors file through `L0.6`'s writer: for every order $k$, an `I32` tensor `order{k}.grams` with every $k$-gram seen (`<s>` as $-1$, rows in ascending order) and `order{k}.counts` with their raw counts, plus metadata `tl_arch = "ngram"`, `n`, `discount`, `vocab_size`. Everything else (adjusted counts, discounts, $\gamma$) is recomputed by `load`, so the file holds only data and saving a loaded model writes the same bytes.

## 3. Worked example by hand

Three sequences over $V = 3$ tokens `a`, `b`, `c` (ids 0, 1, 2), a bigram model ($n = 2$), fixed discount $d = 1/2$:

```
<s> a b c
<s> a b a b
<s> c a b c
```

**Bigram counts** (the top order: raw counts): `<s> a` 2, `<s> c` 1, `a b` 4, `b c` 2, `b a` 1, `c a` 1.

**Unigram level: continuation counts.** `a` follows `<s>`, `b`, and `c`: 3. `b` follows only `a`: 1. `c` follows `b` and `<s>`: 2. So $T_1 = 6$, three seen types, $\gamma_1 = 3 \cdot \frac12 / 6 = \frac14$, and

$$p_1(a) = \frac{3 - \frac12}{6} + \frac14 \cdot \frac13 = \frac12, \qquad p_1(b) = \frac{1 - \frac12}{6} + \frac1{12} = \frac16, \qquad p_1(c) = \frac{2 - \frac12}{6} + \frac1{12} = \frac13 .$$

`b` is the most frequent token (4 raw occurrences, like `a`) but gets the least unigram mass: it only ever follows `a`. With raw counts (4, 4, 3) the same formula would give $p_1(b) = \frac{3.5}{11} + \frac{3}{22} \cdot \frac13 = \frac{4}{11} \approx 0.36$, more than twice as much.

**After `b`**: `b a` 1, `b c` 2, so $T = 3$ and $\gamma = 2 \cdot \frac12 / 3 = \frac13$:

| $w$ | discounted count / $T$ | $+\ \gamma\, p_1(w)$ | $p(w \mid b)$ |
|---|---|---|---|
| `a` | $0.5 / 3 = 1/6$ | $\frac13 \cdot \frac12 = \frac16$ | $1/3$ |
| `b` | $0$ | $\frac13 \cdot \frac16 = \frac1{18}$ | $1/18$ |
| `c` | $1.5 / 3 = 1/2$ | $\frac13 \cdot \frac13 = \frac19$ | $11/18$ |

They sum to $\frac{6 + 1 + 11}{18} = 1$. After `c` (only `c a`, $T = 1$, $\gamma = \frac12$): $p(\cdot \mid c) = (\frac34, \frac1{12}, \frac16)$. These are the first hand case of section 4.

**At the start** (`<s> a` 2, `<s> c` 1, $T = 3$, $\gamma = \frac13$): $p(a \mid \texttt{<s>}) = \frac{1.5}{3} + \frac16 = \frac23$, $p(b \mid \texttt{<s>}) = \frac1{18}$, $p(c \mid \texttt{<s>}) = \frac{0.5}{3} + \frac19 = \frac5{18}$. This is the start-of-sequence hand case, and the sequence `a b` costs $-\ln\frac23 - \ln p(b \mid a)$ nats with $p(b \mid a) = \frac{3.5}{4} + \frac18 \cdot \frac16 = \frac{43}{48}$.

## 4. The interface

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


### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example, bigram | unit, smoke | section 3's $p(\cdot \mid b)$ and $p(\cdot \mid c)$ to $10^{-15}$ | continuation counts at the unigram level |
| Hand example, sequence start | unit, smoke | $p(\cdot \mid \texttt{<s>}) = (2/3, 1/18, 5/18)$ | every sequence starts here |
| Golden hand cases | golden | every context of the example, as a bigram and as a modified-KN trigram, against exact fractions from an independent implementation | `<s>`-initial grams keep raw counts at lower orders |
| Golden byte corpus | golden | the Pass 1 story corpus as bytes, $n = 1 \dots 4$, modified and $d = 0.75$: discounts, held-out perplexities, probabilities after "the " | the end-to-end perplexity must match the reference within 0.5% |
| Probabilities sum to one | property | $\sum_v p(v \mid \text{ctx}) = 1$ for seen, unseen, and start contexts, $n = 1 \dots 4$ | a proper distribution, as `L8.6` needs for exact acceptance |
| `logprobs` agrees with `prob` | property | the vectorized path equals `prob` for every token; logsumexp 0 | `L6.7` and `L8.6` call `logprobs` |
| An unseen history backs off | boundary | an unseen history gives the KN unigram $(23, 7, 15, 3)/48$ | no jump to uniform or to 0 |
| Context length rules | unit | short context means sequence start; long context keeps its last $n - 1$ tokens | the draft model passes whole prefixes |
| Discount fallback | boundary | zero counts of counts and a negative $D_3$ both give $(0.5, 1, 1.5)$, and the result still sums to 1 | small corpora in tests and filters |
| NLL and perplexity | unit | per-token NLL from `<s>`, perplexity $= \exp(\text{mean})$, empty input raises | the corpus filter thresholds perplexity |
| Save and load round trip | unit | header `tl_arch = "ngram"`, I32 tensors, equal probabilities after reload, identical bytes on re-save | the zoo and the capstone load the file later |
| `fit` replaces previous counts | boundary | a second `fit` forgets the first | one model object per corpus source |
| Input validation | boundary | bad $n$, bad discounts, ids outside $0 \dots V-1$, queries before `fit`, inferred $V$ | caller bugs fail early |

**A second hand case for the discounts.** A unigram model whose adjusted counts are 1, 1, 1, 1, 2, 2, 2, 3, 3, 4 has $n_1 = 4$, $n_2 = 3$, $n_3 = 2$, $n_4 = 1$, so $Y = 4 / (4 + 6) = 0.4$ and

$$D_1 = 1 - 2(0.4)\tfrac34 = 0.4, \qquad D_2 = 2 - 3(0.4)\tfrac23 = 1.2, \qquad D_3 = 3 - 4(0.4)\tfrac12 = 2.2 .$$

All three lie in $(0, j]$, so no fallback. The model tests in [`test_ngram.py`](../../../../course/solutions/learner-tests/L2.1/python/tests/l2-1-ngram/test_ngram.py) pin this case beside the section 3 numbers, the fallback, sums to 1, start versus mid-sequence contexts, an unseen history, `logprobs` against `prob`, `nll` of a two-token sequence, a save and load round trip, and a second `fit`.

## 5. The solution, step by step

### 5.1 From first principles

1. **Every probability must be positive and every distribution must sum to 1.** Positivity forces a floor that is never zero: the recursion ends at $p_0 = 1/V$. Summing to 1 forces the interpolation weight to be exactly the mass the discounts removed, so $\gamma_k(h)$ is not a free parameter but $\sum_j D_{kj} N_j(h) / T_k(h)$.
2. **The lower order answers a different question.** It is consulted in proportion to $\gamma$, for histories with little evidence, so it should model "a token in a new context", which is the number of distinct left neighbours. That forces the adjusted count: raw at the top order, continuation counts below it.
3. **`<s>` has no left neighbour,** so a continuation count for a `<s>`-initial gram would be 0 and the start of every sequence would lose its information. Those grams keep raw counts at every order.
4. **Everything the queries need depends only on the history.** For each order and history: the adjusted counts of what follows, their total, and $\gamma$. Precomputing those once in `fit` makes a query $O(n)$ dictionary lookups, and makes the saved file only raw counts, since everything else is a function of them.
5. **Discounts come from the data, but formulas can break.** Chen and Goodman's estimates divide by $n_1, n_2, n_3$ and can leave $(0, j]$ on small data; a discount above $j$ makes a numerator negative, so the solution must check and fall back.
6. **The context rules are part of the model.** A context shorter than $n - 1$ is a sequence start and gets `<s>`; a longer one keeps its last $n - 1$ tokens. `nll` must use the same rule as `prob`, token by token.

### 5.2 The code, annotated

**Counting.** One pass over each padded sequence. Position `i` of `padded` ends one gram of every length up to `min(n, i + 1)`, so `<s>` (at position 0) can only ever be the first symbol of a gram and is never counted as a predicted token.

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

**Adjusted counts and per-history tables.** Continuation counts of order $k$ come from the distinct $(k+1)$-grams: each one $(v, g)$ adds 1 to $g$. The rest is grouping by history and precomputing $(\text{next counts}, T, \gamma)$.

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

On section 3: `raw[2]` has `<s> a`, `<s> c`, `a b`, `b c`, `b a`, `c a`; `cont` gives `a` 3 (from `<s> a`, `b a`, `c a`), `b` 1, `c` 2, which are the continuation counts computed by hand. The unigram history is `()`, with $T_1 = 6$ and $\gamma_1 = 1/4$.

**Discounts.** The counts of counts are over the adjusted counts of one order, which is why `ney_discount` is called on `adj_k` and not on raw counts.

```python
def _estimate(self, adj_k: dict[tuple, int]) -> tuple[float, float, float]:
    if self.discount != "modified":
        d = float(self.discount)
        return (d, d, d)                           # plain interpolated KN
    nj = Counter(a for a in adj_k.values() if 1 <= a <= 4)
    n1, n2, n3, n4 = nj[1], nj[2], nj[3], nj[4]
    if min(n1, n2, n3, n4) == 0:
        return FALLBACK_DISCOUNTS                  # a formula would divide by 0 (or give D = j exactly)
    Y = ney_discount(adj_k)                        # n1 / (n1 + 2 n2)
    D = (1.0 - 2.0 * Y * n2 / n1, 2.0 - 3.0 * Y * n3 / n2, 3.0 - 4.0 * Y * n4 / n3)
    if not all(0.0 < d <= j for d, j in zip(D, (1, 2, 3))):
        return FALLBACK_DISCOUNTS                  # negative or larger than the count: unsafe
    return D
```

The $D_3$ formula uses $n_4 / n_3$, not $n_3 / n_2$: each $D_j$ compares the $j$ and $j + 1$ buckets.

**The context rule.**

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

**One probability.** The recursion runs bottom-up: start at the uniform floor and, order by order, replace `p` with "discounted count plus $\gamma$ times the lower order". An unseen history skips its order, so `p` stays the lower-order estimate, which is exactly $p_k = p_{k-1}$ when nothing follows $h$.

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

**All $V$ at once.** `logprobs` is the same loop with arrays: scatter the next-token counts into a length-$V$ vector, choose each entry's discount with `np.select`, and apply the same update.

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

**Scoring.** `nll` gives each token the same context `prob` would get from the whole prefix, without copying the prefix: while `t < n - 1` the slice is the whole prefix (a sequence start), afterwards the last $n - 1$ tokens. The first token is scored from `<s>`.

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

**The file.** `save` writes, per order, the sorted raw grams (`<s>` as $-1$) as an `I32` tensor of shape `[count, k]` and their counts, plus four metadata strings. `load` rebuilds the `Counter`s and calls the same `_build`, so a reloaded model is the same model, and sorting makes re-saving byte-identical.

Full solution: [`ngram.py`](../../../../course/solutions/python/tinyllm/lm/ngram.py).

### 5.3 Why this solution

- **Complexity.** `fit` is $O(n \cdot N)$ counter updates for $N$ training tokens, plus $O(G)$ to build the tables for $G$ distinct grams; memory is $O(G)$. `prob` is $O(n)$ dictionary lookups; `logprobs` is $O(nV)$; `nll` is $O(nT)$ for $T$ tokens.
- **Why precompute per history?** $\gamma_k(h)$ and $T_k(h)$ would otherwise be recomputed by summing over every continuation of $h$ on each query, which is $O(V)$ per order. Precomputing turns queries into lookups and costs one pass at `fit` time.
- **Why interpolation, not backoff?** Backoff consults the lower order only for unseen continuations and needs extra normalisation weights to sum to 1; interpolation mixes at every context, sums to 1 by construction, and empirically does better with KN (Chen and Goodman).
- **Why raw counts in the file?** Adjusted counts, discounts, and $\gamma$ are derived; storing them would duplicate state that could disagree with the counts. KenLM's ARPA format does store probabilities, trading file size for query speed, which the zoo does not need.
- **Why Python dicts, not suffix arrays or tries?** The corpora here are megabytes; dicts are simple and exact. KenLM and infini-gram exist for the 100 GB case (see Going further).
- **Tradeoffs accepted.** Memory grows with distinct grams, and `logprobs` builds a dense $V$-vector per order, which is fine for $V = 256$ and acceptable for subword vocabularies.

### 5.4 Explain it back

1. Why is $\gamma_k(h) = (D_1 N_1 + D_2 N_2 + D_3 N_{3+}) / T_k(h)$ and not $D_1 \cdot \#\{\text{seen}\} / T_k(h)$?
   <details><summary>Answer</summary>$\gamma$ must equal the mass the numerators lost, so that the first term sums to $1 - \gamma$ and $\gamma$ times a normalised lower order adds it back. A token with count 2 lost $D_2$, not $D_1$; charging $D_1$ for everything makes the distribution sum to something other than 1 whenever the three discounts differ.</details>
2. Why do `<s>`-initial grams keep raw counts at lower orders?
   <details><summary>Answer</summary>A continuation count is the number of distinct symbols seen right before the gram, and nothing ever precedes `<s>`, so it would be 0. Every start-of-sequence context would then have no evidence and fall through to the lower order, forgetting that stories start with "Once".</details>
3. What does the model predict after a history it never saw, and why not the uniform distribution?
   <details><summary>Answer</summary>The order with the unseen history is skipped, so the result is the next lower order's estimate, down to the KN unigram. Jumping to uniform would discard everything the lower orders know; on section 3's corpus with an unseen history the answer is the unigram $(23, 7, 15, 3)/48$ in the four-token test, not $1/4$ each.</details>
4. Why must a discount be at most $j$ for counts in bucket $j$?
   <details><summary>Answer</summary>The numerator is $\max(a - D, 0)$; if $D > a$ the max clips at 0 but $\gamma$ still adds $D$, so the total exceeds 1. The formulas can produce such values (or negative ones) on small data, which is why `_estimate` checks $0 < D_j \le j$ and falls back to $(0.5, 1, 1.5)$.</details>
5. Why is the context `[c, c, a, b]` the same as `[a, b]` for a trigram, but `[b]` is not the same as `[a, b]`?
   <details><summary>Answer</summary>A trigram conditions on two symbols. A context of 2 or more tokens keeps its last two, so both long contexts are `a b`. A context of 1 token is shorter than $n - 1$, which can only happen at the start of a sequence, so it becomes `<s> b`.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. lower orders from raw counts (absolute discounting, not Kneser-Ney) | frequent tokens get too much mass as fresh continuations; `p(b)` at the unigram level is 4/11 instead of 1/6 | `_build` replaces counts below the top order with `cont[g]` |
| 2. $\gamma$ charged with $D_1$ for every seen token | with three discounts, the distribution no longer sums to 1 | `gamma` weights $N_1$, $N_2$, $N_{3+}$ by their own discounts |
| 3. continuation counts for n-grams that start with `<s>` | they become 0: start-of-sequence predictions fall to the lower order | `c if g[0] == BOS else cont[g]` |
| 4. no fallback when a discount leaves $(0, j]$ | a negative $D_3$ gives negative probabilities on small data | `_estimate` checks `0 < d <= j` and returns `FALLBACK_DISCOUNTS` |
| 5. an unseen history jumps to the uniform distribution | every rare context forgets what the unigram level knows | `prob` and `logprobs` `continue` past a missing history, keeping `p` |
| 6. keeping the first $n - 1$ tokens of a long context | the draft model conditions on the start of the prompt, not its end | `_padded` slices `ctx[len(ctx) - (n - 1):]` |
| counts of 3 or more discounted with $D_2$ | $\gamma$ and the numerators disagree: no longer sums to 1 | `D[min(a, 3) - 1]`, the same bucket rule as `gamma` |
| the first token of a sequence not scored | perplexity skips the hardest prediction | `nll` loops from `t = 0`, with an empty prefix padded to `<s>` |
| `<s>`-initial n-grams dropped by `save` | a reloaded model predicts sequence starts differently | `save` writes every gram in `_raw`, `<s>` as $-1$ |
| `logprobs` using $D_1$ for every count | the vectorized path disagrees with `prob` | `np.select` over the same three buckets |
| $D_3$ computed from $n_3 / n_2$ | wrong discounts at every order with enough data | $D_3 = 3 - 4Y n_4 / n_3$ |
| `fit` adding to the previous counts | a reused model mixes two corpora | `fit` builds new `Counter`s each call |
| inferred $V$ = the largest id instead of 1 + it | the largest token becomes unpredictable | `biggest + 1` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.2` | `ney_discount` gives $Y = n_1/(n_1 + 2n_2)$ for each order's discounts |
| Back | `M11.2` | `perplexity` sums the per-token NLLs with `NLLAccumulator` |
| Back | `L0.6` | `save` and `load` go through `save_safetensors` and `load_safetensors` |
| Back | `L0.0` | the byte bigram this generalizes (reading) |
| Forward | `L6.7` | the zoo trains KN-4 on TinyStories and reports its bits per byte, the floor the neural models must beat |
| Forward | `L8.6` | the n-gram draft model proposes tokens from `logprobs` that the target model verifies |
| Forward | `C1` | the corpus pipeline's `ppl_filter` (`data.02`) drops documents whose KN perplexity is too high, and quality checks score samples with it |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| counting in dicts | KenLM `lmplz` | disk-based streaming sorts, so a 100 GB corpus is counted in bounded memory; pruning of rare n-grams | `lm/builder/` (`corpus_count.cc`, `adjust_counts.cc`, `initial_probabilities.cc`) |
| `save` with every n-gram | KenLM's ARPA and binary formats | probabilities and backoff weights precomputed per n-gram; a trie or probing hash table with quantized values for fast queries | `lm/binary_format.cc`, `lm/trie.cc` |
| `NGramLM` as a quality filter | CCNet and RedPajama | a KenLM 5-gram trained on Wikipedia scores every web document; perplexity buckets decide what is kept | `facebookresearch/cc_net`, `cc_net/perplexity.py` |
| n-gram draft model | prompt-lookup and n-gram speculative decoding in vLLM | drafts from n-gram matches against the prompt, no extra model | `vllm/v1/spec_decode/ngram_proposer.py` |
| interpolated KN | the infini-gram index (Liu et al. 2024) | unbounded-order n-gram counts over trillions of tokens with suffix arrays | `liujch1998/infini-gram` |
