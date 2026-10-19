# Bengio's neural probabilistic language model

## Overview

| | |
|---|---|
| **Module** | `L2.2` · build · Python · Pass 3 |
| **Solution** | [`nplm.py`](../../../../course/solutions/python/tinyllm/lm/nplm.py): `windows`, `NPLM(vocab, context, d_emb, d_hidden, direct, rng)` with `forward`, `nll`, `perplexity`, `generate`; `train_nplm`; `save_nplm` and `load_nplm` (the model directory) · the CLI verbs `lm train nplm`, `eval`, `generate` in [`cli_lm.py`](../../../../course/solutions/entry/python/tinyllm/cli_lm.py) · model tests [`test_nplm.py`](../../../../course/solutions/learner-tests/L2.2/python/tests/l2-2-nplm/test_nplm.py) |
| **Contract** | [`course/contracts/py/tinyllm/lm/nplm.pyi`](../../../../course/contracts/py/tinyllm/lm/nplm.pyi) |
| **Needs** | `L0.1` `Tensor`, `no_grad` · `L0.2` `F.reshape`, `F.tanh`, `F.embedding` · `L0.3` `cross_entropy` · `L0.4` `Module`, `Embedding`, `Linear` · `L0.5` `DataLoader`, `train_step` · `L0.6` `save_safetensors`, `load_safetensors`, `open_tokens` · `M10.2` `SGD` · `M11.2` `NLLAccumulator` · `M06.3` `PCG32`. Reading: `L2.1` (the count model this one beats), `L0.0` |
| **Used by** | later `L6.7`: the model zoo loads the NPLM directory and reports its bits per byte · end to end: `tinyllm lm train nplm`, then `tinyllm eval`, reaches a perplexity below the bigram's and the 4-gram's; `tinyllm generate` twice with one seed gives one story |
| **Optional depth** | Bengio, Ducharme, Vincent, and Jauvin, "A Neural Probabilistic Language Model" (JMLR 2003), sections 1 to 3; Goodfellow, Bengio, Courville, *Deep Learning*, section 12.4; Jurafsky and Martin, *Speech and Language Processing*, 3rd ed., ch. 7 |

## Key Takeaways

- An NPLM replaces the count table with a learned embedding per token and a small network over a fixed window, so a window never seen in training still gets a sensible prediction from the windows it resembles.
- The window is concatenated **oldest first** and the embedding lookup is an autograd op: the order is part of the parameter layout, and a lookup that bypasses the graph leaves the table untrained.
- The whole model is ordinary layers, so the forward is one line of math and backward is free; torch, given the same weights, computes the same logits, loss, and gradients.
- Training is exactly the L0.5 loop with one SGD across epochs, and every random choice comes from one PCG32 per purpose, so a seed reproduces the run bit for bit.
- The model directory (config plus F32 safetensors) is what the CLI writes, `generate` reads, and the zoo loads.

## How to read this chapter

Sections 1 and 2 derive the model from the failure of count tables and fix every detail that changes numbers (window order, init stream, sampling). Section 3 runs the forward and the first gradient by hand. Section 4 fixes the interface. Section 5 walks through the solution and explains why each line is written the way it is. Try to predict each step of the solution before reading it.

The solution's CLI exposes the model as three verbs: `lm train nplm` trains and writes the model directory, `eval` scores a token file, and `generate` reads an `nplm` directory and samples from it.

---

## 1. Why now

The Kneser-Ney model (`L2.1`) is the best a count table can do, and it hits a wall that smoothing cannot move. To a count table, "the cat ran to the park" and "the dog ran to the park" are unrelated strings: everything learned about one tells it nothing about the other, so it needs to see every word in every context. More context makes this exponentially worse: a byte 8-gram has $256^8 \approx 1.8 \times 10^{19}$ possible histories, and almost all the ones you meet at test time were never counted. Bengio's 2003 answer is the idea every later model in this course keeps: give each token a learned vector, and let a network predict from the vectors of the window. Tokens that appear in similar windows get similar vectors, so the model generalizes across them. It is also the first model the course trains with its own autograd (`L0.1` to `L0.4`) and training loop (`L0.5`) on real text, and on the Pass 3 story corpus it beats the 4-gram (perplexity about 1.54 against 1.87) and the count bigram (8.52) by a wide margin.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size; tokens are ids $0 \dots V-1$ (256 bytes in the Pass 3 corpus) | `int` |
| $n$ | the order: the model sees the $n - 1$ previous tokens | `int` |
| $c = n - 1$ | the window (`context`) | `int` |
| $m$ | embedding size (`d_emb`) | `int` |
| $h$ | hidden size (`d_hidden`) | `int` |
| $B$ | batch size: windows per step | `int` |
| $C$ | embedding table, row $C[w]$ is token $w$'s vector | `float32[V, m]` |
| $x$ | the window's embeddings concatenated, oldest first | `float32[B, c m]` |
| $H, d$ | hidden layer weight and bias | `float32[h, c m]`, `float32[h]` |
| $U, b$ | output weight and bias | `float32[V, h]`, `float32[V]` |
| $W$ | direct path from the embeddings to the logits (optional) | `float32[V, c m]` |
| $y$ | logits | `float32[B, V]` |
| $p = \mathrm{softmax}(y)$ | the predicted distribution of the next token | `float32[B, V]` |
| $\mathcal{L}$ | mean cross-entropy over the batch, in nats | scalar |
| $\eta$, $\mu$ | SGD learning rate and momentum (`M10.2`) | `float` |

### 2.1 From counts to vectors

An n-gram model estimates $P(w_t \mid w_{t-c} \dots w_{t-1})$ with one free number per (history, token) pair: a table with $V^{c} \times V$ entries, nearly all of them estimated from zero data. A neural model shares parameters across histories. Each token gets one vector $C[w] \in \mathbb{R}^m$, used in every window where the token appears, and one function maps a window of vectors to a distribution. The parameter count is now $V m + (\text{network})$, linear in $V$ and in $c$, instead of exponential in $c$. If "cat" and "dog" end up with nearby vectors (because they appear in similar windows), then every window containing one is also evidence about windows containing the other.

### 2.2 The architecture

Bengio et al. write the model as one equation:

$$x = [C[w_{t-c}]; \dots; C[w_{t-1}]], \qquad y = b + W x + U \tanh(d + H x), \qquad p = \mathrm{softmax}(y).$$

Read it from the inside out. Look up the $c$ rows of $C$ for the window and concatenate them, oldest first, into one vector of length $c m$; that is exactly a row-major reshape of the `[B, c, m]` lookup to `[B, c m]`. Apply one hidden layer with a $\tanh$ nonlinearity. Map the hidden vector to $V$ logits. The paper also adds $W x$, a **direct** linear path from the embeddings to the logits: a log-linear model in parallel with the hidden layer, which the paper found speeds up training. In the course's layers: `emb = Embedding(V, m)`, `hidden = Linear(c m, h)`, `out = Linear(h, V)`, and `direct = Linear(c m, V, bias=False)`. Those attribute names, in that order, are the state-dict keys and so the safetensors keys of the model directory.

The order of the concatenation is part of the model. Position $j$ of the window always multiplies the same columns $H[:, jm:(j+1)m]$, so the network learns what the token two back means differently from the token one back. Concatenating newest first is not wrong in itself, but it is a different parameter layout: weights trained one way are garbage read the other way, and a checkpoint is no longer interchangeable with anyone else's.

### 2.3 Training

The loss is the mean cross-entropy (`L0.3`) of each window's logits against the next token. Its gradient at the logits is $p - \mathrm{onehot}(\text{target})$ per row, divided by $B$; everything below the logits is the chain rule through `Linear`, `tanh`, the reshape, and the embedding lookup, which the autograd already does. The one place a gradient can silently vanish is the lookup: `F.embedding` is an op whose backward adds each row's gradient into the rows it read (a token used twice in a window gets both). Reading `emb.weight.data[ids]` with numpy instead gives the same forward numbers and a constant: $C$ never learns.

`train_nplm` is the `L0.5` loop with nothing new. The windows of the whole sequence become arrays `x` (`int64[N, c]`) and `y` (`int64[N]`); a `DataLoader` shuffles them with the given PCG32 (a new permutation each epoch) in batches of $B$, dropping the last partial batch; one `SGD` (`M10.2`) is created before the first step, so its momentum buffers carry across epochs; and each step is `train_step` with the given `clip`. It runs for exactly `steps` steps, wrapping into new epochs as needed.

### 2.4 Scoring

`nll(ids)` scores every token that has a full window: positions $c \dots \text{len} - 1$. Each value is $\mathrm{logsumexp}(y) - y_{\text{target}}$ in float64, from float32 logits, computed under `no_grad` in batches (batching changes nothing but rounding). `perplexity` is $\exp$ of the mean, summed with `M11.2`'s `NLLAccumulator`; with byte tokens, bits per byte is the mean divided by $\ln 2$. The first $c$ tokens of a file are not scored: an n-gram model predicts them from `<s>`, the NPLM has no start symbol, so on the Pass 3 validation file the NPLM scores 14 667 tokens where the 4-gram scores 14 675. Over that many tokens the difference does not move the comparison.

### 2.5 Generation and the model directory

`generate(prefix, n, temperature, seed)` repeats: take the last $c$ ids, compute the logits, choose the next id, append it. At temperature 0 it is the first maximum (ties to the lowest id, D11). Otherwise it draws exactly as `L0.5`'s `BigramLM.sample`: one `PCG32(seed)` (stream 54) per call, one `uniform()` per token, weights $e^{z_j - \max z}$ of $z = y / T$ in float64 summed in id order, and the first id whose running sum exceeds $u \cdot \sum$. A prompt shorter than the window is an error; a longer prompt conditions only on its end.

`save_nplm(model, dir)` writes `config.json` (`tl_arch = "nplm"`, the sizes as `tl_context`, `tl_d_emb`, `hidden_size`, `tl_direct`) and `model.safetensors` (the state dict as F32, metadata `{"format": "tinyllm", "tl_arch": "nplm"}`) with `L0.6`'s writer; `load_nplm` rebuilds the model from the config and loads the tensors strictly.

## 3. Worked example by hand

$V = 3$ tokens `a`, `b`, `c` (ids 0, 1, 2), a window of $c = 2$, embedding size $m = 1$, one hidden unit $h = 1$, with the direct path:

$$C = \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix}, \quad H = \begin{pmatrix} 0.5 & -0.5 \end{pmatrix}, \quad d = 0, \quad U = \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix}, \quad b = \begin{pmatrix} 0 \\ 0.5 \\ 0 \end{pmatrix}, \quad W = \begin{pmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 1 \end{pmatrix}.$$

**Windows.** The sequence 5 6 7 8 with $c = 2$ gives the windows (5, 6) with target 7 and (6, 7) with target 8; the first two tokens are never targets.

**Forward for the window (a, c).** Oldest first: $x = (C[a], C[c]) = (1, -1)$. Hidden: $H x + d = 0.5 \cdot 1 - 0.5 \cdot (-1) = 1$, so the hidden activation is $\tanh 1 = 0.761594$. Direct path: $W x = (1, 0, -1)$. Output path: $U \tanh 1 = (0.761594, 0, -0.761594)$. Logits:

$$y = b + W x + U \tanh 1 = (1.761594,\ 0.5,\ -1.761594).$$

The reversed window (c, a) has $x = (-1, 1)$, hidden pre-activation $-1$, and logits $(-1.761594, 0.5, 1.761594)$: the mirror image. The order matters. Without the direct path only $b + U \tanh 1 = (0.761594, 0.5, -0.761594)$ remains.

**Probabilities and loss.** $e^{y} = (5.821711, 1.648721, 0.171771)$, sum $7.642203$, so $p = (0.761784, 0.215739, 0.022477)$. With target `a` the loss is $-\ln 0.761784 = 0.272092$ nats.

**Gradient at the output bias.** $\partial \mathcal{L} / \partial y = p - \mathrm{onehot}(a) = (-0.238216, 0.215739, 0.022477)$, and since $y = b + \dots$, that is exactly the gradient of $b$. Going one layer down, the hidden unit receives $\sum_v (p_v - \mathrm{onehot}_v) U_v = -0.238216 - 0.022477 = -0.260692$, times $1 - \tanh^2 1 = 0.419974$ gives $-0.109484$ at the pre-activation; from there $H$, and through both paths $C[a]$ and $C[c]$, receive their shares. $C[b]$ was not in the window: its gradient is exactly 0.

## 4. The interface

```python
def windows(ids: ArrayLike, context: int) -> tuple[NDArray, NDArray]: ...   # ([N, c], [N]), N = len - c

class NPLM(Module):
    def __init__(self, vocab: int, context: int, d_emb: int, d_hidden: int,
                 direct: bool = True, rng=None) -> None: ...                 # emb, hidden, out, direct
    def forward(self, ctx_ids: ArrayLike) -> Tensor: ...                      # [B, c] -> [B, V] logits
    def nll(self, ids: ArrayLike, batch_size: int = 1024) -> NDArray: ...     # float64 [len - c]
    def perplexity(self, ids: ArrayLike) -> float: ...
    def generate(self, prefix, n, temperature, seed) -> list[int]: ...

def train_nplm(model, ids, steps, batch_size, lr, rng, momentum=0.0,
               weight_decay=0.0, clip=None) -> list[float]: ...               # one loss per step
def save_nplm(model: NPLM, dir: str, tokenizer: str = "bytes") -> None: ...
def load_nplm(dir: str) -> NPLM: ...
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Windows of the hand example | unit, smoke | (5, 6) -> 7 and (6, 7) -> 8; `context` 1 too | every training example and every scored token comes from here |
| Forward of the hand example | unit, smoke | section 3's logits and the mirrored window; the model without the direct path | the parameter layout every checkpoint shares |
| The gradient reaches the bias | unit | loss $0.272092$, $\partial b = p - \mathrm{onehot}$, $C[b]$ untouched | the loss is right before any training |
| Golden against torch | golden | logits, loss, and every parameter's gradient against torch, with and without the direct path; ids repeat in and across windows | torch is the oracle for every later model (`L3`, `L5`) |
| Gradcheck every parameter | gradcheck | frozen central differences in float64 for all six parameters | a table cut out of the graph never learns |
| State-dict names and shapes | unit | keys and shapes in registration order, F32, `direct.weight` only with the direct path | the zoo (`L6.7`) loads by these names |
| Init draws from one generator | unit | layers drawn from one rng in order; two default models equal; no repeated draws across layers | a seed fixes the whole model |
| NLL matches cross-entropy per window | unit | per-token NLL in float64, batch size changes only rounding, mean = the loss, perplexity = exp(mean) | `eval` and the zoo's bits per byte |
| Seeded and greedy generation | unit | the contract's draw rebuilt with PCG32; same seed, same ids; greedy is the step-by-step argmax | generating twice with one seed gives one text |
| Greedy ties go to the lowest id | boundary | an all-zero model decodes greedily to id 0 every step | the D11 tie rule every engine shares |
| Generation uses the last window | boundary | a long prompt conditions on its end | prompts are longer than the window |
| `train_nplm` is the L0.5 loop | differential | 10 steps over 4-batch epochs with momentum and weight decay: equal bit for bit to the loop rebuilt from `L0.5` and `M10.2` | the training is reproducible from a seed |
| `train_nplm` runs exactly `steps` | boundary | data of exactly one batch: 3 steps are 3 epochs | `steps` counts optimizer steps, not epochs |
| `train_nplm` passes the clip | unit | with clip 0.5 the run equals the rebuilt loop with the same clip | long runs survive one bad batch |
| The NPLM learns the corpus | learning | 300 steps on 50 000 bytes reach the reference's validation perplexity (mean + 3 sd over 5 seeds), below the bigram's 8.52 | the model actually learns |
| Save and load round trip | unit | `config.json` exactly, F32 tensors with the metadata, identical logits after reload, with and without the direct path | `generate` and the zoo load what `lm train nplm` wrote |
| Input validation | boundary | sizes below 1, wrong window width, ids outside $0 \dots V-1$, short prompts, too few windows, bad tokenizer, wrong `tl_arch` | caller bugs fail at the call |

The model tests in [`test_nplm.py`](../../../../course/solutions/learner-tests/L2.2/python/tests/l2-2-nplm/test_nplm.py) pin the hand example directly: an `NPLM(3, 2, 1, 1)` loaded with section 3's weights must give logits $(1 + \tanh 1, 0.5, -1 - \tanh 1)$ for the window `[[0, 2]]`, and they add the windows of a short sequence, gradients on every parameter including the table, a finite-difference check of the embedding gradient, `nll` against `cross_entropy`, greedy ties and the last-window rule, `train_nplm` against the rebuilt loop across an epoch boundary, and a save and load round trip.

## 5. The solution, step by step

### 5.1 From first principles

1. **Share parameters across histories.** A count table has one free number per (history, token); to generalise, the same parameters must serve many histories. So each token gets one vector, reused in every window, and one function maps any window of vectors to a distribution.
2. **The simplest function that can tell positions apart** is a linear map of the concatenated vectors: position $j$ always multiplies the same block of columns. That fixes both the concatenation (oldest first, as a row-major reshape of `[B, c, m]`) and the meaning of every weight column.
3. **A nonlinearity, plus a shortcut.** One `tanh` layer gives the model capacity beyond log-linear; the direct path $W x$ keeps a log-linear model alongside, which the paper found speeds training. Both are `Linear` layers, so the model is a composition of ops the autograd already differentiates.
4. **The lookup must be in the graph.** The table is a parameter like any other; its gradient arrives only if the lookup is an op with a backward (`Embedding` calls `F.embedding`). Everything else follows from the chain rule.
5. **Reproducibility is a design constraint.** A seed must fix the whole model and run: one generator shared by all layers in registration order, one loader rng shuffling every epoch, one optimizer for the whole run (momentum carries across epochs), and one sampler generator per `generate` call.
6. **Scoring is exact and comparable.** NLL per scored token in float64 from float32 logits, summed with `NLLAccumulator`, so `eval` and the zoo report numbers comparable with `L2.1`.

### 5.2 The code, annotated

**Windows.** Row $i$ of the context matrix is `ids[i : i + c]` and its target is `ids[i + c]`, so the first $c$ tokens are never targets.

```python
def windows(ids: ArrayLike, context: int) -> tuple[NDArray, NDArray]:
    a = np.asarray(ids)
    # ... 1-D integer array, context >= 1, at least one full window ...
    a = a.astype(np.int64)
    n = a.size - context
    # Row i is ids[i : i + context]: a strided view, copied so callers own it.
    ctx = np.lib.stride_tricks.sliding_window_view(a, context)[:n].copy()
    return ctx, a[context:].copy()          # target of row i is ids[i + context]
```

`sliding_window_view` has `a.size - context + 1` rows; the last one has no target, so `[:n]` drops it. On 5 6 7 8 with $c = 2$: rows (5, 6), (6, 7) and targets 7, 8.

**Construction.** The layers are registered in the order of the state dict (`emb`, `hidden`, `out`, `direct`), and all draw from one generator.

```python
# ONE generator for every layer, in registration order: passing None to
# each layer would restart PCG32(0).substream("init") per layer and give
# correlated weights.
r = rng if rng is not None else PCG32(0).substream("init")
width = self.context * self.d_emb
self.emb = Embedding(self.vocab, self.d_emb, rng=r)          # C
self.hidden = Linear(width, self.d_hidden, rng=r)            # H, d
self.out = Linear(self.d_hidden, self.vocab, rng=r)          # U, b
self.direct = (
    Linear(width, self.vocab, bias=False, rng=r) if self.use_direct else None  # W, no bias: b is in out
)
```

**Forward.** Bengio's equation, one op per term.

```python
def forward(self, ctx_ids: ArrayLike) -> Tensor:
    ids = np.asarray(ctx_ids)
    if ids.ndim != 2 or ids.shape[1] != self.context:
        raise ValueError(...)
    e = self.emb(ids)  # [B, context, d_emb]: the op, so C gets gradients
    # Row-major reshape concatenates the rows oldest first: x = [C[w1], C[w2], ...].
    x = F.reshape(e, (ids.shape[0], self.context * self.d_emb))
    y = self.out(F.tanh(self.hidden(x)))      # b + U tanh(d + H x)
    if self.direct is not None:
        y = y + self.direct(x)                # + W x
    return y                                  # logits; softmax lives in the loss
```

Returning logits, not probabilities, lets `cross_entropy` and `nll` use a stable logsumexp. On section 3's window (a, c): `e` is `[[[1], [-1]]]`, `x` is `[[1, -1]]`, and `y` is $(1.761594, 0.5, -1.761594)$.

**Scoring.** Each scored token costs $\mathrm{logsumexp}(z) - z_{\text{target}}$, computed with the max subtracted so `exp` cannot overflow.

```python
def nll(self, ids: ArrayLike, batch_size: int = 1024) -> NDArray:
    ctx, tgt = windows(ids, self.context)
    # ... batch_size >= 1 ...
    out = np.empty(tgt.size, dtype=np.float64)
    with no_grad():                                        # no graph: scoring only
        for i in range(0, tgt.size, batch_size):
            z = self.forward(ctx[i : i + batch_size]).data.astype(np.float64)
            m = z.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
            rows = np.arange(z.shape[0])
            out[i : i + batch_size] = lse - z[rows, tgt[i : i + batch_size]]
    return out

def perplexity(self, ids: ArrayLike) -> float:
    acc = NLLAccumulator()
    acc.add(self.nll(ids))
    return float(acc.result()["ppl"])                      # exp(mean), not the mean
```

**Training.** Nothing new: the `L0.5` loop. The loader and optimizer are built once, before the loop, so each pass over `loader` is a fresh permutation from the same rng and the momentum buffers persist.

```python
def train_nplm(model, ids, steps, batch_size, lr, rng,
               momentum=0.0, weight_decay=0.0, clip=None) -> list[float]:
    # ... steps >= 0 ...
    ctx, tgt = windows(ids, model.context)
    if tgt.size < batch_size:
        raise ValueError(...)                              # drop_last would yield no batch at all
    loader = DataLoader({"x": ctx, "y": tgt}, batch_size, shuffle=True, rng=rng, drop_last=True)
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)  # once
    losses: list[float] = []
    model.train()
    while len(losses) < steps:                             # wrap into new epochs as needed
        for batch in loader:                               # each pass is a new permutation
            losses.append(train_step(model, batch, _window_loss, opt, clip=clip)["loss"])
            if len(losses) == steps:
                break
    return losses
```

`_window_loss` is `cross_entropy(model(batch["x"]), batch["y"])`. The `tgt.size < batch_size` check matters: with `drop_last=True` such data yields no batch, and `while` would spin forever.

**Generation.** The window is always the last $c$ ids of the growing history; one `PCG32(seed)` per call.

```python
rng = PCG32(seed)  # one generator per call, stream 54 (L0.5's sampler)
out: list[int] = []
with no_grad():
    for _ in range(n):
        window = np.asarray([hist[-self.context :]], dtype=np.int64)   # the LAST c ids
        row = self.forward(window).data[0].astype(np.float64)
        if temperature == 0:
            nxt = int(np.argmax(row))  # first maximum: ties to the lowest id
        else:
            nxt = _draw(row, float(temperature), rng.uniform())        # one uniform per token
        out.append(nxt)
        hist.append(nxt)
```

```python
def _draw(row: NDArray, temperature: float, u: float) -> int:
    z = row / temperature
    w = np.exp(z - z.max()).tolist()           # stable weights; the max weight is 1
    total = 0.0
    for wi in w:  # in id order, the order the Rust engine sums in
        total += wi
    target = u * total
    cum, last = 0.0, 0
    for i, wi in enumerate(w):
        if wi > 0.0:                           # skip zero weights so they are never chosen
            cum += wi
            last = i
            if cum > target:
                return i
    return last                                # rounding: fall back to the last positive id
```

Summing in a Python loop rather than `np.sum` fixes the summation order (numpy uses pairwise summation), so the draw matches every other engine bit for bit.

**The directory.** `save_nplm` writes `config.json` (`tl_arch`, `tl_tokenizer`, `vocab_size`, `tl_format`, `tl_context`, `tl_d_emb`, `hidden_size`, `tl_direct`) and the state dict as F32 safetensors; `load_nplm` checks `tl_arch`, rebuilds the model from the config, and calls `load_state_dict(tensors, strict=True)`, so a missing or extra tensor (`direct.weight`) is an error, not a silently ignored key.

Full solution: [`nplm.py`](../../../../course/solutions/python/tinyllm/lm/nplm.py).

### 5.3 Why this solution

- **Complexity.** Per batch of $B$ windows, the forward costs $O(B(c m h + h V + c m V))$: the output layers dominate when $V$ is large, which is why later models replace the full softmax (see Going further). Parameters: $V m + h c m + h + V h + V + V c m$.
- **Why concatenate instead of averaging the window?** Averaging (a bag of words) loses position: "dog bites man" and "man bites dog" would get the same prediction. Concatenation keeps one block of weights per position at the cost of parameters linear in $c$.
- **Why logits out of `forward`?** Softmax followed by `log` loses precision for small probabilities; the losses compute logsumexp from logits directly. It also lets `generate` apply temperature to logits.
- **Why one generator for all layers?** Re-seeding per layer makes layers of the same shape start identical or correlated, and makes the model depend on how many layers share a default. One stream in registration order is the simplest rule that makes "seed fixes the model" true.
- **Why rebuild nothing in training?** Reusing `DataLoader`, `train_step`, and `SGD` means the only new code is the model; the differential case against the `L0.5` loop then pins training bit for bit.
- **Tradeoffs accepted.** A fixed window cannot use context beyond $c$ tokens, and the first $c$ tokens of a file are never scored; the recurrent models of part 3 remove the first limit.

### 5.4 Explain it back

1. Why does the reshape of `[B, c, m]` to `[B, c m]` give "oldest first"?
   <details><summary>Answer</summary>`windows` puts the oldest token in column 0 of each row, `Embedding` keeps that order along axis 1, and a row-major reshape lays axis 1 out before axis 2, so $x = [C[w_{t-c}], \dots, C[w_{t-1}]]$. Column block $j$ of $H$ and $W$ therefore always sees the token $c - j$ positions back.</details>
2. What goes wrong if the lookup reads `emb.weight.data[ids]` with numpy?
   <details><summary>Answer</summary>The forward gives the same numbers, but the result is a constant with no link to `emb.weight`, so the table receives no gradient and stays at its initialisation. Numpy would also silently read id $-1$ as the last row instead of raising.</details>
3. Why is the SGD optimizer created once, outside the epoch loop?
   <details><summary>Answer</summary>With momentum, the optimizer holds a velocity buffer per parameter. A new optimizer per epoch resets the buffers at every epoch boundary, so the run differs from the `L0.5` loop and from any reference trained with the same seed.</details>
4. Section 3's window (a, c) has loss 0.272092 for target `a`. What is the gradient of the output bias `out.bias`, and why does $C[b]$ get none?
   <details><summary>Answer</summary>$\partial \mathcal{L}/\partial b = p - \mathrm{onehot}(a) = (-0.238216, 0.215739, 0.022477)$, because $y = b + \dots$ and the softmax cross-entropy gradient at the logits is $p - \mathrm{onehot}$. $C[b]$ is not read by this window, and the embedding backward only adds into the rows it read.</details>
5. Why does `_draw` sum the weights in a Python loop instead of calling `np.sum`?
   <details><summary>Answer</summary>Floating-point addition is not associative. `np.sum` uses pairwise summation, so its total can differ in the last bits from a left-to-right sum. Summing in id order matches the contract and the Rust engine, so the same seed draws the same id everywhere.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the window's rows read from `emb.weight.data` instead of through `F.embedding` | the forward is right, the table gets no gradient and never learns (and numpy reads id -1 as the last row) | `forward` calls `self.emb(ids)`, the autograd op |
| 2. the window concatenated newest first | another parameter layout: torch's weights and every saved checkpoint read wrong | `windows` is oldest first and `F.reshape` keeps that order |
| 3. the direct path registered but never added to the logits | the parameter exists, saves, and loads, and has no effect | `y = y + self.direct(x)` when the path exists |
| 4. `rng=None` passed on to every layer | each layer restarts the default stream: correlated initial weights | resolves one `r` and passes it to every layer |
| 5. targets shifted by one (predicting the last window token) | a copy task: training loss collapses, nothing is learned about the next token | targets are `a[context:]`, one past each window |
| 6. generation conditioned on the first ids of the prompt | every step sees the same window: the text loops on the prompt's start | the window is `hist[-self.context:]` |
| a new `SGD` every epoch | the momentum buffers reset at each epoch boundary | one `SGD` before the `while` loop |
| training stops at the end of the first epoch | fewer steps than asked, silently | `while len(losses) < steps` wraps into new epochs |
| the direct matrix not saved | the reloaded model cannot load strictly | `save_nplm` writes the whole `state_dict()` |
| greedy ties broken toward the highest id | a different text from every other engine (D11) | `np.argmax` returns the first maximum |
| `clip` not passed to `train_step` | long runs blow up on one bad batch | `train_step(..., clip=clip)` |
| perplexity returned as the mean NLL | a number in nats reported as a perplexity | `NLLAccumulator(...)["ppl"]`, the exponential of the mean |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | `Tensor` carries the gradients; `no_grad` for scoring and generation |
| Back | `L0.2` | `F.embedding` (the lookup with a backward), `F.reshape`, `F.tanh` |
| Back | `L0.3` | `cross_entropy` is the training loss |
| Back | `L0.4` | `Module`, `Embedding`, `Linear`: the layers and their state-dict names |
| Back | `L0.5` | `DataLoader` and `train_step` are the whole training loop |
| Back | `L0.6` | `save_safetensors`, `load_safetensors` for the model directory; `open_tokens` reads the corpus |
| Back | `M10.2` | `SGD` with momentum and weight decay |
| Back | `M11.2` | `NLLAccumulator` for perplexity and bits per byte |
| Back | `M06.3` | `PCG32`: the default init stream and the sampler |
| Back | `L2.1` | the count model this one is compared with (reading) |
| Forward | `L6.7` | the zoo loads the NPLM directory and reports its bits per byte next to KN-4 and the transformers |
| Forward | `L3.6` | the recurrent LM replaces the fixed window with a state, and its bits per byte must beat this model's on the same file |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a full softmax over $V$ | hierarchical softmax (Morin and Bengio 2005), adaptive softmax (Grave et al. 2017) | the output layer costs $O(\log V)$ or a few clusters instead of $O(V)$ per token | `torch.nn.AdaptiveLogSoftmaxWithLoss` (`torch/nn/modules/adaptive.py`) |
| fixed window of $c$ tokens | recurrent and attention models | unbounded context with a state, or every earlier token at once | `L3.6` (RNN LM) and `L5.1` (attention) in this course |
| embedding table trained with the model | word2vec and GloVe (`L2.3`), then tied input and output embeddings (Press and Wolf 2017) | the table pretrained on more text, or shared with the output layer to halve the parameters | `transformers` `tie_word_embeddings` |
| the direct path $W x$ | residual connections (He et al. 2016) | an identity-like shortcut around the nonlinearity, now in every transformer block | `L5.5` in this course |
| CPU numpy, one process | the paper's parallel training on 40 CPUs | data-parallel and parameter-parallel SGD over the output layer | Bengio et al. (2003), section 3 |
