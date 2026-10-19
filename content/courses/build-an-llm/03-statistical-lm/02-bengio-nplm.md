# Bengio's neural probabilistic language model

The [Kneser-Ney model](01-ngram-kneser-ney.md) is the best a count table can do, and it hits a wall that smoothing cannot move. To a count table, "the cat ran to the park" and "the dog ran to the park" are unrelated strings: everything learned about one tells it nothing about the other, so it needs to see every word in every context. More context makes this exponentially worse. A byte 8-gram has $256^8 \approx 1.8 \times 10^{19}$ possible histories, and almost all the ones you meet at test time were never counted.

Bengio's 2003 answer is the idea every later model in this book keeps: give each token a learned vector, and let a network predict from the vectors of the window. Tokens that appear in similar windows get similar vectors, so the model generalizes across them. It is also the first model in the book trained with your own autograd and training loop on real text, and on the story corpus it beats the 4-gram (perplexity about 1.54 against 1.87) and the count bigram (8.52) by a wide margin.

You will derive the model from the failure of count tables, pin down every detail that changes numbers (window order, the init stream, sampling), run the forward pass and the first gradient by hand, and then walk through the solution line by line. The section builds on the whole of Part 0: [Tensor, broadcasting backward, and no_grad](../01-foundations/02-tensor-and-broadcasting-backward.md), [The op library and gradcheck_all](../01-foundations/03-op-library-and-gradcheck.md) (`F.reshape`, `F.tanh`, `F.embedding`), [Losses with a fused backward](../01-foundations/04-losses-with-fused-backward.md) (`cross_entropy`), [Module system and basic layers](../01-foundations/05-module-system-and-layers.md) (`Module`, `Embedding`, `Linear`), [Training loop and the autograd bigram](../01-foundations/06-training-loop-and-the-autograd-bigram.md) (`DataLoader`, `train_step`), and [Safetensors for every dtype, atomic checkpoints, and the token stream](../01-foundations/07-safetensors-checkpoints-and-token-streams.md) (`save_safetensors`, `load_safetensors`, `open_tokens`). From *Mathematics for Computing* it uses [the Optimizer protocol and SGD](../../mathematics-for-computing/11-optimization/02-optimizer-protocol-and-sgd.md), [the NLL accumulator](../../mathematics-for-computing/12-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md), and [PCG32](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md).

The reference solution is [`nplm.py`](../../../../course/solutions/python/tinyllm/lm/nplm.py): `windows`, `NPLM(vocab, context, d_emb, d_hidden, direct, rng)` with `forward`, `nll`, `perplexity`, and `generate`, plus `train_nplm`, `save_nplm`, and `load_nplm` for the model directory. Its contract is [`nplm.pyi`](../../../../course/contracts/py/tinyllm/lm/nplm.pyi) and its tests are [`test_nplm.py`](../../../../course/solutions/learner-tests/L2.2/python/tests/l2-2-nplm/test_nplm.py). The CLI in [`cli_lm.py`](../../../../course/solutions/entry/python/tinyllm/cli_lm.py) exposes the model as three verbs: `lm train nplm` trains and writes the model directory, `eval` scores a token file, and `generate` reads an `nplm` directory and samples from it.

> **Learning objectives**
> - Explain how shared embeddings let a neural LM generalize to windows never seen in training, and why parameter count becomes linear rather than exponential in the context length.
> - Compute the NPLM forward pass, loss, and output-bias gradient by hand, including the direct path.
> - Implement the model from existing layers so that the embedding table is trained, the window is concatenated oldest first, and a seed reproduces the run bit for bit.
> - Score, generate from, save, and reload the model in a way that matches torch, the n-gram baseline, and other engines.

## Notation

The symbols below are used throughout the section.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size; tokens are ids $0 \dots V-1$ (256 bytes in the story corpus) | `int` |
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
| $\eta$, $\mu$ | SGD learning rate and momentum | `float` |

## From counts to vectors

An n-gram model estimates $P(w_t \mid w_{t-c} \dots w_{t-1})$ with one free number per (history, token) pair: a table with $V^{c} \times V$ entries, nearly all of them estimated from zero data. A neural model shares parameters across histories instead. Each token gets one vector $C[w] \in \mathbb{R}^m$, used in every window where the token appears, and one function maps a window of vectors to a distribution. The parameter count is now $V m + (\text{network})$, linear in $V$ and in $c$, instead of exponential in $c$. If "cat" and "dog" end up with nearby vectors (because they appear in similar windows), then every window containing one is also evidence about windows containing the other.

That leaves the question of which function to use.

## The architecture

Bengio et al. write the model as one equation:

$$x = [C[w_{t-c}]; \dots; C[w_{t-1}]], \qquad y = b + W x + U \tanh(d + H x), \qquad p = \mathrm{softmax}(y).$$

Read it from the inside out. Look up the $c$ rows of $C$ for the window and concatenate them, oldest first, into one vector of length $c m$; that is exactly a row-major reshape of the `[B, c, m]` lookup to `[B, c m]`. Apply one hidden layer with a $\tanh$ nonlinearity. Map the hidden vector to $V$ logits. The paper also adds $W x$, a **direct** linear path from the embeddings to the logits: a log-linear model in parallel with the hidden layer, which the paper found speeds up training.

In the book's layers this is `emb = Embedding(V, m)`, `hidden = Linear(c m, h)`, `out = Linear(h, V)`, and `direct = Linear(c m, V, bias=False)`. Those attribute names, in that order, are the state-dict keys and so the safetensors keys of the model directory.

The order of the concatenation is part of the model. Position $j$ of the window always multiplies the same columns $H[:, jm:(j+1)m]$, so the network learns what the token two back means differently from the token one back. Concatenating newest first is not wrong in itself, but it is a different parameter layout: weights trained one way are garbage read the other way, and a checkpoint is no longer interchangeable with anyone else's.

## Training

The loss is the mean cross-entropy of each window's logits against the next token. Its gradient at the logits is $p - \mathrm{onehot}(\text{target})$ per row, divided by $B$; everything below the logits is the chain rule through `Linear`, `tanh`, the reshape, and the embedding lookup, which the autograd already does.

The one place a gradient can silently vanish is the lookup. `F.embedding` is an op whose backward adds each row's gradient into the rows it read (a token used twice in a window gets both). Reading `emb.weight.data[ids]` with numpy instead gives the same forward numbers and a constant: $C$ never learns.

`train_nplm` is the [Part 0 training loop](../01-foundations/06-training-loop-and-the-autograd-bigram.md) with nothing new. The windows of the whole sequence become arrays `x` (`int64[N, c]`) and `y` (`int64[N]`). A `DataLoader` shuffles them with the given PCG32 (a new permutation each epoch) in batches of $B$, dropping the last partial batch. One `SGD` is created before the first step, so its momentum buffers carry across epochs, and each step is `train_step` with the given `clip`. It runs for exactly `steps` steps, wrapping into new epochs as needed.

## Scoring

`nll(ids)` scores every token that has a full window: positions $c \dots \text{len} - 1$. Each value is $\mathrm{logsumexp}(y) - y_{\text{target}}$ in float64, from float32 logits, computed under `no_grad` in batches (batching changes nothing but rounding). `perplexity` is $\exp$ of the mean, summed with `NLLAccumulator`; with byte tokens, bits per byte is the mean divided by $\ln 2$.

The first $c$ tokens of a file are not scored. An n-gram model predicts them from `<s>`, but the NPLM has no start symbol, so on the story corpus's validation file the NPLM scores 14 667 tokens where the 4-gram scores 14 675. Over that many tokens the difference does not move the comparison.

## Generation and the model directory

`generate(prefix, n, temperature, seed)` repeats a simple loop: take the last $c$ ids, compute the logits, choose the next id, append it. At temperature 0 the choice is the first maximum (ties go to the lowest id, the book-wide tie rule recorded as decision D11). Otherwise it draws exactly as `BigramLM.sample` from Part 0 does: one `PCG32(seed)` (stream 54) per call, one `uniform()` per token, weights $e^{z_j - \max z}$ of $z = y / T$ in float64 summed in id order, and the first id whose running sum exceeds $u \cdot \sum$. A prompt shorter than the window is an error; a longer prompt conditions only on its end.

`save_nplm(model, dir)` writes `config.json` (`tl_arch = "nplm"`, the sizes as `tl_context`, `tl_d_emb`, `hidden_size`, `tl_direct`) and `model.safetensors` (the state dict as F32, metadata `{"format": "tinyllm", "tl_arch": "nplm"}`) with the Part 0 safetensors writer. `load_nplm` rebuilds the model from the config and loads the tensors strictly.

## Working it by hand

Take $V = 3$ tokens `a`, `b`, `c` (ids 0, 1, 2), a window of $c = 2$, embedding size $m = 1$, and one hidden unit $h = 1$, with the direct path:

$$C = \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix}, \quad H = \begin{pmatrix} 0.5 & -0.5 \end{pmatrix}, \quad d = 0, \quad U = \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix}, \quad b = \begin{pmatrix} 0 \\ 0.5 \\ 0 \end{pmatrix}, \quad W = \begin{pmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 1 \end{pmatrix}.$$

Start with the windows. The sequence 5 6 7 8 with $c = 2$ gives the windows (5, 6) with target 7 and (6, 7) with target 8; the first two tokens are never targets.

Now run the forward for the window (a, c). Oldest first, $x = (C[a], C[c]) = (1, -1)$. The hidden pre-activation is $H x + d = 0.5 \cdot 1 - 0.5 \cdot (-1) = 1$, so the hidden activation is $\tanh 1 = 0.761594$. The direct path gives $W x = (1, 0, -1)$ and the output path gives $U \tanh 1 = (0.761594, 0, -0.761594)$. The logits are

$$y = b + W x + U \tanh 1 = (1.761594,\ 0.5,\ -1.761594).$$

The reversed window (c, a) has $x = (-1, 1)$, hidden pre-activation $-1$, and logits $(-1.761594, 0.5, 1.761594)$: the mirror image. The order matters. Without the direct path only $b + U \tanh 1 = (0.761594, 0.5, -0.761594)$ remains.

Next, probabilities and loss. $e^{y} = (5.821711, 1.648721, 0.171771)$ with sum $7.642203$, so $p = (0.761784, 0.215739, 0.022477)$. With target `a` the loss is $-\ln 0.761784 = 0.272092$ nats.

Finally, the gradient at the output bias. $\partial \mathcal{L} / \partial y = p - \mathrm{onehot}(a) = (-0.238216, 0.215739, 0.022477)$, and since $y = b + \dots$, that is exactly the gradient of $b$. One layer down, the hidden unit receives $\sum_v (p_v - \mathrm{onehot}_v) U_v = -0.238216 - 0.022477 = -0.260692$; times $1 - \tanh^2 1 = 0.419974$ that gives $-0.109484$ at the pre-activation. From there $H$, and through both paths $C[a]$ and $C[c]$, receive their shares. $C[b]$ was not in the window, so its gradient is exactly 0.

The tests pin this example directly: an `NPLM(3, 2, 1, 1)` loaded with these weights must give logits $(1 + \tanh 1, 0.5, -1 - \tanh 1)$ for the window `[[0, 2]]`.

## Designing the API

The public surface is one helper that turns a token sequence into training windows, the model class, a training function, and a pair of directory functions.

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

`forward` returns logits, not probabilities. `train_nplm` returns one loss per optimizer step, and `steps` counts steps, not epochs.

## Building it step by step

The design follows from six requirements. First, parameters must be shared across histories: a count table has one free number per (history, token), and to generalize the same parameters must serve many histories, so each token gets one vector reused in every window and one function maps any window of vectors to a distribution. Second, the simplest function that can tell positions apart is a linear map of the concatenated vectors, where position $j$ always multiplies the same block of columns; that fixes both the concatenation (oldest first, as a row-major reshape of `[B, c, m]`) and the meaning of every weight column. Third, one `tanh` layer gives the model capacity beyond log-linear, and the direct path $W x$ keeps a log-linear model alongside. Both are `Linear` layers, so the model is a composition of ops the autograd already differentiates.

Fourth, the lookup must be in the graph. The table is a parameter like any other, and its gradient arrives only if the lookup is an op with a backward (`Embedding` calls `F.embedding`). Fifth, reproducibility is a design constraint: a seed must fix the whole model and run, which means one generator shared by all layers in registration order, one loader rng shuffling every epoch, one optimizer for the whole run (so momentum carries across epochs), and one sampler generator per `generate` call. Sixth, scoring is exact and comparable: NLL per scored token in float64 from float32 logits, summed with `NLLAccumulator`, so `eval` and the zoo report numbers comparable with the n-gram model.

### Windows

Row $i$ of the context matrix is `ids[i : i + c]` and its target is `ids[i + c]`, so the first $c$ tokens are never targets.

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

`sliding_window_view` has `a.size - context + 1` rows; the last one has no target, so `[:n]` drops it. On 5 6 7 8 with $c = 2$ this gives rows (5, 6), (6, 7) and targets 7, 8.

### Construction

The layers are registered in the order of the state dict (`emb`, `hidden`, `out`, `direct`), and all draw from one generator.

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

### Forward

The forward is Bengio's equation, one op per term.

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

Returning logits, not probabilities, lets `cross_entropy` and `nll` use a stable logsumexp. On the hand example's window (a, c), `e` is `[[[1], [-1]]]`, `x` is `[[1, -1]]`, and `y` is $(1.761594, 0.5, -1.761594)$.

### Scoring

Each scored token costs $\mathrm{logsumexp}(z) - z_{\text{target}}$, computed with the max subtracted so `exp` cannot overflow.

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

### Training

Training is nothing new: the Part 0 loop. The loader and optimizer are built once, before the loop, so each pass over `loader` is a fresh permutation from the same rng and the momentum buffers persist.

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

`_window_loss` is `cross_entropy(model(batch["x"]), batch["y"])`. The `tgt.size < batch_size` check matters: with `drop_last=True` such data yields no batch, and the `while` would spin forever.

### Generation

The window is always the last $c$ ids of the growing history, with one `PCG32(seed)` per call.

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

### The directory

`save_nplm` writes `config.json` (`tl_arch`, `tl_tokenizer`, `vocab_size`, `tl_format`, `tl_context`, `tl_d_emb`, `hidden_size`, `tl_direct`) and the state dict as F32 safetensors. `load_nplm` checks `tl_arch`, rebuilds the model from the config, and calls `load_state_dict(tensors, strict=True)`, so a missing or extra tensor (`direct.weight`) is an error, not a silently ignored key.

The full solution is [`nplm.py`](../../../../course/solutions/python/tinyllm/lm/nplm.py).

### Why this design

Per batch of $B$ windows, the forward costs $O(B(c m h + h V + c m V))$. The output layers dominate when $V$ is large, which is why later models replace the full softmax (see the production table at the end). The parameter count is $V m + h c m + h + V h + V + V c m$.

Concatenation rather than averaging the window is what keeps position. Averaging (a bag of words) would give "dog bites man" and "man bites dog" the same prediction; concatenation keeps one block of weights per position at the cost of parameters linear in $c$. Returning logits out of `forward` avoids the precision loss of softmax followed by `log` for small probabilities, since the losses compute logsumexp from logits directly, and it lets `generate` apply temperature to logits.

One generator for all layers is the simplest rule that makes "a seed fixes the model" true. Re-seeding per layer makes layers of the same shape start identical or correlated, and makes the model depend on how many layers share a default. Reusing `DataLoader`, `train_step`, and `SGD` means the only new code is the model, and a differential test against the Part 0 loop then pins training bit for bit. The tradeoffs accepted are that a fixed window cannot use context beyond $c$ tokens and the first $c$ tokens of a file are never scored; the recurrent models of Part 3 remove the first limit.

## What can go wrong

The tests cover the hand example and a set of properties that later chapters rely on. The windows of the hand example must be (5, 6) -> 7 and (6, 7) -> 8, with `context` 1 checked too, because every training example and every scored token comes from here. The forward must reproduce the hand logits, the mirrored window, and the model without the direct path, which pins the parameter layout every checkpoint shares. The loss must be $0.272092$ with $\partial b = p - \mathrm{onehot}$ and $C[b]$ untouched.

Against torch as the oracle (as it is for the recurrent models and the transformer later), logits, loss, and every parameter's gradient must match, with and without the direct path, with ids repeating within and across windows. A frozen central-difference gradcheck in float64 covers all six parameters, because a table cut out of the graph never learns. The state dict must have its keys and shapes in registration order, F32, with `direct.weight` only when the direct path exists, since [the model zoo](../07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) loads by these names. Layers must draw from one rng in order, two default models must be equal, and no draws may repeat across layers.

Scoring must give per-token NLL in float64 where batch size changes only rounding, the mean equals the loss, and perplexity is exp(mean). Seeded generation is rebuilt from the contract's draw with PCG32: same seed, same ids; greedy is the step-by-step argmax; an all-zero model decodes greedily to id 0 every step; and a long prompt conditions on its end. `train_nplm` must equal, bit for bit, the Part 0 loop rebuilt with SGD over 10 steps across 4-batch epochs with momentum and weight decay; must run exactly `steps` (data of exactly one batch: 3 steps are 3 epochs); and must pass `clip` through (with clip 0.5 the run equals the rebuilt loop with the same clip). The model must actually learn: 300 steps on 50 000 bytes reach the reference's validation perplexity (mean + 3 sd over 5 seeds), below the bigram's 8.52. A save and load round trip must reproduce `config.json` exactly, F32 tensors with the metadata, and identical logits after reload. Sizes below 1, a wrong window width, ids outside $0 \dots V-1$, short prompts, too few windows, a bad tokenizer, and a wrong `tl_arch` all fail at the call.

The bugs behind these checks are listed below with their symptoms and the line that prevents each.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| the window's rows read from `emb.weight.data` instead of through `F.embedding` | the forward is right, the table gets no gradient and never learns (and numpy reads id -1 as the last row) | `forward` calls `self.emb(ids)`, the autograd op |
| the window concatenated newest first | another parameter layout: torch's weights and every saved checkpoint read wrong | `windows` is oldest first and `F.reshape` keeps that order |
| the direct path registered but never added to the logits | the parameter exists, saves, and loads, and has no effect | `y = y + self.direct(x)` when the path exists |
| `rng=None` passed on to every layer | each layer restarts the default stream: correlated initial weights | resolves one `r` and passes it to every layer |
| targets shifted by one (predicting the last window token) | a copy task: training loss collapses, nothing is learned about the next token | targets are `a[context:]`, one past each window |
| generation conditioned on the first ids of the prompt | every step sees the same window: the text loops on the prompt's start | the window is `hist[-self.context:]` |
| a new `SGD` every epoch | the momentum buffers reset at each epoch boundary | one `SGD` before the `while` loop |
| training stops at the end of the first epoch | fewer steps than asked, silently | `while len(losses) < steps` wraps into new epochs |
| the direct matrix not saved | the reloaded model cannot load strictly | `save_nplm` writes the whole `state_dict()` |
| greedy ties broken toward the highest id | a different text from every other engine | `np.argmax` returns the first maximum |
| `clip` not passed to `train_step` | long runs blow up on one bad batch | `train_step(..., clip=clip)` |
| perplexity returned as the mean NLL | a number in nats reported as a perplexity | `NLLAccumulator(...)["ppl"]`, the exponential of the mean |

> **Warning:** The embedding bug is the dangerous one, because nothing looks wrong. The forward numbers are correct and the loss still falls (the other layers learn), but the table stays at its initialization. A gradcheck on `emb.weight` catches it immediately.

## Where this leads

End to end, `tinyllm lm train nplm` followed by `tinyllm eval` reaches a perplexity below the bigram's and the 4-gram's, and `tinyllm generate` run twice with one seed gives one story. [The model zoo](../07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) loads the NPLM directory and reports its bits per byte next to KN-4 and the transformers. [The RNN language model](../04-recurrent/06-rnn-language-model-stateful-tbptt.md) replaces the fixed window with a state, and its bits per byte must beat this model's on the same file. The next section, [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](03-word2vec-and-ppmi-svd.md), learns the embedding table on its own, without a language model around it.

Each piece has a production descendant:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a full softmax over $V$ | hierarchical softmax (Morin and Bengio 2005), adaptive softmax (Grave et al. 2017) | the output layer costs $O(\log V)$ or a few clusters instead of $O(V)$ per token | `torch.nn.AdaptiveLogSoftmaxWithLoss` (`torch/nn/modules/adaptive.py`) |
| fixed window of $c$ tokens | recurrent and attention models | unbounded context with a state, or every earlier token at once | [the RNN LM](../04-recurrent/06-rnn-language-model-stateful-tbptt.md) and [scaled dot-product attention](../06-transformer-2017/01-scaled-dot-product-attention.md) in this book |
| embedding table trained with the model | word2vec and GloVe ([next section](03-word2vec-and-ppmi-svd.md)), then tied input and output embeddings (Press and Wolf 2017) | the table pretrained on more text, or shared with the output layer to halve the parameters | `transformers` `tie_word_embeddings` |
| the direct path $W x$ | residual connections (He et al. 2016) | an identity-like shortcut around the nonlinearity, now in every transformer block | [the encoder-decoder Transformer](../06-transformer-2017/05-encoder-decoder-transformer.md) in this book |
| CPU numpy, one process | the paper's parallel training on 40 CPUs | data-parallel and parameter-parallel SGD over the output layer | Bengio et al. (2003), section 3 |

### Further reading

- Bengio, Ducharme, Vincent, and Jauvin, "A Neural Probabilistic Language Model" (JMLR 2003), sections 1 to 3.
- Goodfellow, Bengio, and Courville, *Deep Learning*, section 12.4.
- Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), 3rd ed., chapter 7.

## Key terms

- **embedding table**: the matrix $C$ whose row $C[w]$ is the learned vector for token $w$, shared across every window.
- **window**: the fixed $c$ previous tokens the model conditions on, concatenated oldest first.
- **direct path**: the optional linear map $W x$ from the embeddings straight to the logits, in parallel with the hidden layer.
- **logits**: the unnormalized scores $y$ whose softmax is the next-token distribution.
- **registration order**: the order layers are created in a `Module`, which fixes both the state-dict keys and the order of random initialization draws.

## Summary

- An NPLM replaces the count table with a learned embedding per token and a small network over a fixed window, so a window never seen in training still gets a sensible prediction from the windows it resembles.
- The window is concatenated **oldest first** and the embedding lookup is an autograd op: the order is part of the parameter layout, and a lookup that bypasses the graph leaves the table untrained.
- The whole model is ordinary layers, so the forward is one line of math and backward is free; torch, given the same weights, computes the same logits, loss, and gradients.
- Training is exactly the Part 0 loop with one SGD across epochs, and every random choice comes from one PCG32 per purpose, so a seed reproduces the run bit for bit.
- The model directory (config plus F32 safetensors) is what the CLI writes, `generate` reads, and the zoo loads.

## Review questions

1. Why does the reshape of `[B, c, m]` to `[B, c m]` give "oldest first"?
   <details><summary>Answer</summary>`windows` puts the oldest token in column 0 of each row, `Embedding` keeps that order along axis 1, and a row-major reshape lays axis 1 out before axis 2, so $x = [C[w_{t-c}], \dots, C[w_{t-1}]]$. Column block $j$ of $H$ and $W$ therefore always sees the token $c - j$ positions back.</details>
2. What goes wrong if the lookup reads `emb.weight.data[ids]` with numpy?
   <details><summary>Answer</summary>The forward gives the same numbers, but the result is a constant with no link to `emb.weight`, so the table receives no gradient and stays at its initialisation. Numpy would also silently read id $-1$ as the last row instead of raising.</details>
3. Why is the SGD optimizer created once, outside the epoch loop?
   <details><summary>Answer</summary>With momentum, the optimizer holds a velocity buffer per parameter. A new optimizer per epoch resets the buffers at every epoch boundary, so the run differs from the Part 0 loop and from any reference trained with the same seed.</details>
4. The hand example's window (a, c) has loss 0.272092 for target `a`. What is the gradient of the output bias `out.bias`, and why does $C[b]$ get none?
   <details><summary>Answer</summary>$\partial \mathcal{L}/\partial b = p - \mathrm{onehot}(a) = (-0.238216, 0.215739, 0.022477)$, because $y = b + \dots$ and the softmax cross-entropy gradient at the logits is $p - \mathrm{onehot}$. $C[b]$ is not read by this window, and the embedding backward only adds into the rows it read.</details>
5. Why does `_draw` sum the weights in a Python loop instead of calling `np.sum`?
   <details><summary>Answer</summary>Floating-point addition is not associative. `np.sum` uses pairwise summation, so its total can differ in the last bits from a left-to-right sum. Summing in id order matches the contract and the Rust engine, so the same seed draws the same id everywhere.</details>
