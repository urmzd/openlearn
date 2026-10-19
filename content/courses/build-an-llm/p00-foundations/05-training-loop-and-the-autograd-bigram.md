# Training loop and the autograd bigram

The four sections before this one give you every piece of a training run and no run. The digits MLP that closes this chapter needs batches drawn in a reproducible order, a step that does the things a step does in the right order, and an evaluation that does not train. And the tracer model that the engine has served since [Byte bigram: counts, logits, safetensors v0](00-byte-bigram.md) is still a table of counts: it cannot be improved, only recounted.

This section writes the loop every later model trains with, then retrains the tracer's own table with it. If the loop is right, gradient descent finds the count model again, because the bigram problem has one minimum and the counts are it. That makes the bigram the most precise end-to-end test of the autograd there is: the answer is known to many digits. The section also moves the bigram's sampler onto PCG32 with the Rust engine's exact arithmetic, so a seed means the same text in Python and in Rust.

The section builds on everything earlier in the chapter: `no_grad` from [Tensor, broadcasting backward, and no_grad](01-tensor-and-broadcasting-backward.md), `F.embedding` from [The op library and gradcheck_all](02-op-library-and-gradcheck.md), the losses of [Losses with a fused backward](03-losses-with-fused-backward.md), and `Module` and `Linear` from [Module system and basic layers](04-module-system-and-layers.md). From *Mathematics for Computing* it uses PCG32 ([Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md)), SGD ([The Optimizer protocol](../../mathematics-for-computing/10-optimization/02-optimizer-protocol-and-sgd.md)), AdamW ([Adam and AdamW](../../mathematics-for-computing/10-optimization/03-adam-and-adamw.md)), `clip_grad_norm_` ([Learning-rate schedules and gradient clipping](../../mathematics-for-computing/10-optimization/04-schedules-and-gradient-clipping.md)), and the `EMA` used for the smoothed loss ([Series convergence, EMA as a geometric series, bias correction](../../mathematics-for-computing/02-calculus-2/02-series-and-the-ema.md)). The solution is [`loop.py`](../../../../course/solutions/python/tinyllm/train/loop.py) (`DataLoader`, `train_step`, `evaluate`) and [`bigram.py`](../../../../course/solutions/python/tinyllm/lm/bigram.py), taken over from the byte bigram (its earlier version is in [`history/L0.0/`](../../../../course/solutions/history/L0.0/)), which adds `BigramLogits` (the table as a trainable `Module`), `to_lm`, and `BigramLM.sample` on PCG32. The `train` verbs live in [`cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py). The contracts are [`loop.pyi`](../../../../course/contracts/py/tinyllm/train/loop.pyi) and [`bigram.pyi`](../../../../course/contracts/py/tinyllm/lm/bigram.pyi).

The CLI uses this code for two verbs. `train bigram --method autograd --data F --out DIR` runs full-batch AdamW on every byte pair of `F` and writes the same model directory as the counting verb, from the trained table. `train mlp --data digits.npz --hidden 64 --epochs 30 --ckpt DIR` trains a two-layer ReLU MLP on the UCI digits through `DataLoader`, with each pixel standardized by the training set's mean and standard deviation. `--method counts` remains the original counting verb.

> **Learning objectives**
> - Write a training step that zeroes, computes the loss, refuses a non-finite loss, backpropagates, clips, and updates, and explain why each step is in that place.
> - Build a seeded data loader whose shuffle is the spec's Fisher-Yates, fresh each epoch.
> - Write an evaluation that weights batches by rows, runs in eval mode under `no_grad`, and always restores the model's mode.
> - Prove that the bigram's minimum is the unsmoothed count model and compute its NLL floor by hand.
> - Sample with the engine's exact arithmetic so a seed gives the same ids in Python and Rust.

## The pieces of a training run

The notation for this section:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\theta$ | every parameter of the model | list of `Tensor` |
| $B$ | batch size; $n$ rows in the dataset | ints |
| $\mathcal{L}(\theta; \text{batch})$ | the loss `loss_fn` returns, one element | `Tensor` |
| $\eta$ | learning rate | float |
| $c$ | gradient clipping threshold, global norm | float |
| $\lVert g \rVert = \sqrt{\sum_p \sum_i g_{p,i}^2}$ | global gradient norm over all parameters | float |
| $W \in \mathbb{R}^{V \times V}$ | the bigram table, row $i$ = logits after token $i$ | `float32[V, V]` |
| $C_{ij}$, $R_i = \sum_j C_{ij}$ | bigram counts and row totals | |
| $N = T - 1$ | predictions in a text of $T$ tokens | int |
| $u$ | one uniform draw in $[0, 1)$ from PCG32 | float |

### The loader

`DataLoader(arrays, batch_size, shuffle, rng)` holds named arrays that share their first dimension $n$ and yields dictionaries of rows. Without shuffle the order is $0, \dots, n - 1$. With shuffle, when each epoch **starts** it builds the list $0, \dots, n - 1$ and calls `rng.shuffle` on it: the Fisher-Yates of `spec/pcg32.md`, going from the end and swapping $i$ with `below(i + 1)`. Each epoch advances the generator, so epoch 2 has a new permutation and a seeded run repeats exactly. `drop_last` drops a final partial batch, so every step sees $B$ rows, and `len(loader)` says how many batches an epoch has. Arrays of different lengths, or a shuffle without a generator, are errors: one pairs inputs with the wrong labels, the other hides an unseeded order.

### The step

`train_step(model, batch, loss_fn, opt, clip)` does seven things, in this order:

1. `opt.zero_grad()`. Gradients **accumulate** across `backward` calls, so without this step $k$ trains on the sum of the gradients of steps $1, \dots, k$.
2. `loss = loss_fn(model, batch)`: a one-element Tensor, or a pair `(loss, {"acc": ...})` of the loss and extra metrics. More than one element is an error, because `backward()` on a vector would need an upstream gradient nobody chose.
3. If the loss is not finite, raise `FloatingPointError` **before** any update. One `nan` step turns every weight into `nan` and the run is lost; raising first lets the caller skip the batch.
4. `loss.backward()`.
5. With `clip`, `clip_grad_norm_(parameters, c)` rescales all gradients together when $\lVert g \rVert > c$. Report the norm **before** clipping, because that is the number that tells you training is unstable.
6. `opt.step()`, then return `{"loss": ..., "grad_norm": ...}` plus the extra metrics.
7. With `ema` (an `EMA`), feed the loss to `ema.update` and report `ema.value_debiased()` as `"loss_ema"`. One batch's loss is noise; the debiased EMA is the curve a training log plots, and without the correction its first steps sit near zero. A non-finite loss raised in step 3, so it never reaches the average.

### Evaluation

`evaluate(model, loader, loss_fn)` switches the model to eval mode (dropout off), runs every batch under `no_grad` (no graph, no stored activations), and averages the loss and metrics **weighted by rows**: a last batch of 2 rows counts as 2 rows, not as a full batch. It puts the model back in the mode it found it, in a `finally`, so an exception in a batch cannot leave a training run with dropout silently off.

## The bigram, trained

`BigramLogits(V)` is a `Module` with one parameter, `weight` of shape `[V, V]`, starting at zeros (every next token equally likely, NLL $\ln V$). Its forward is a row lookup, `F.embedding(weight, ids)`: the logits `onehot(ids) @ W` of the counting bigram without the multiplications by zero, for ids of any shape (the trainer feeds `[B, T]` windows). Backward adds each position's $p - e_{\text{next}}$ into its row.

### Why gradient descent finds the counts

The mean NLL of the table on a text is

$$\mathcal{L}(W) = \frac{1}{N}\sum_{i,j} C_{ij}\,\big(\log\textstyle\sum_k e^{W_{ik}} - W_{ij}\big),$$

a sum over rows of convex functions (log-sum-exp is convex, $-W_{ij}$ is linear). Setting the gradient of row $i$ to zero gives $R_i\,\mathrm{softmax}(W_i) = C_i$, so at the minimum $\mathrm{softmax}(W_i)_j = C_{ij}/R_i$: the **unsmoothed** count model. Its NLL, $-\frac{1}{N}\sum_{ij} C_{ij}\log(C_{ij}/R_i)$, is the floor no bigram can beat on that text.

The add-one model of the counting bigram sits slightly above that floor; the trained table gets within $10^{-3}$ nats of it, never below. Pairs that never occur push their logits toward $-\infty$ forever, so the loss approaches the floor without reaching it. AdamW with a large learning rate gets there in a few hundred full-batch steps.

### Sampling like the engine

`BigramLM.sample` keeps its signature and switches its generator: one `PCG32(seed)` (stream 54) per call, one `uniform()` per token, and the Rust engine's draw from [Your first endpoint](../p10-serving/00-your-first-endpoint.md). That draw computes weights $w_j = e^{z_j - \max z}$ of $z = \text{row}/\tau$ in float64, sums them in id order, and picks the first $j$ whose running sum exceeds $u \cdot \sum_j w_j$. Same arithmetic, same order, same generator: `generate --seed S` and the engine's completion with seed `S` give the same ids. `to_lm()` turns the trained module into a `BigramLM` holding a float32 **copy** of the table, the object the CLI saves as `bigram.weight`.

## Working it by hand

Start with one training step. Take `Linear(1, 1)` with $w = 2$, $b = 0$; the batch $x = [[1], [2]]$, $y = [[3], [5]]$; loss `mse`; and SGD with $\eta = 0.1$.

1. The predictions $wx + b$ are $[2, 4]$, the residuals $[2 - 3, 4 - 5] = [-1, -1]$, and the loss $((-1)^2 + (-1)^2)/2 = 1$.
2. The gradients of $\frac{1}{2}\sum r_k^2$ are $\partial/\partial w = \frac{2}{2}\sum r_k x_k = -1 \cdot 1 - 1 \cdot 2 = -3$ and $\partial/\partial b = \sum r_k = -2$.
3. The update gives $w = 2 - 0.1 \cdot (-3) = 2.3$ and $b = 0 - 0.1 \cdot (-2) = 0.2$.
4. A second step must use only the new gradients. The residuals are $[2.5 - 3, 4.8 - 5] = [-0.5, -0.2]$, so $\partial/\partial w = -0.5 - 0.4 = -0.9$ and $\partial/\partial b = -0.7$. A loop that forgot `zero_grad` would add the first step's $-3$ and $-2$.
5. With `clip=1`, the global norm is $\sqrt{3^2 + 2^2} = \sqrt{13} = 3.606$. It is reported, and both gradients are scaled by $1/3.606$ before the update.

Now the bigram, from zeros. Take `BigramLogits(3)` on the text `abbacab` (ids `[0, 1, 1, 0, 2, 0, 1]`, $N = 6$ predictions).

1. Every row of zeros has softmax $[1/3, 1/3, 1/3]$, so the NLL is $\ln 3 = 1.0986$.
2. Row `a` makes three predictions (the next tokens are `b`, `c`, `b`). Each contributes $p - e_{\text{next}}$, and the mean divides by 6: $\big(3 \cdot [\tfrac13, \tfrac13, \tfrac13] - [0, 2, 1]\big)/6 = [\tfrac16, -\tfrac16, 0]$. Descent raises $W_{ab}$ and lowers $W_{aa}$; $W_{ac}$ is already right on average.
3. The floor: row `a` goes to $b$ twice and $c$ once ($2/3$, $1/3$), row `b` to $a$ and $b$ once each, row `c` to $a$ always. The NLL is $\frac{1}{6}\big(2\ln\tfrac32 + \ln 3 + 2\ln 2 + 0\big) = \frac{3\ln 3}{6} = 0.5493$, below the add-one model's $0.8351$.

These numbers are the first cases a correct solution must reproduce.

## Designing the API

```python
# python/tinyllm/train/loop.py
class DataLoader:
    def __init__(self, arrays: Mapping[str, NDArray], batch_size: int, shuffle: bool, rng, drop_last: bool = True)
    def __iter__(self) -> Iterator[dict[str, NDArray]]; def __len__(self) -> int
def train_step(model, batch, loss_fn, opt, clip: Optional[float] = None,
               ema: Optional[EMA] = None) -> dict[str, float]   # EMA from tinyllm.num.ema (M02.2)
def evaluate(model, loader, loss_fn) -> dict[str, float]          # mean loss, mean metrics, "n"

# python/tinyllm/lm/bigram.py (taken over from L0.0; BigramLM keeps its v0 API)
class BigramLogits(Module):
    def __init__(self, vocab: int = 256); def forward(self, ids) -> Tensor; def to_lm(self) -> BigramLM
```

A correct solution reproduces the hand-worked step (loss 1, gradients $-3$ and $-2$, then $w = 2.3$ and $b = 0.2$), so everyone agrees on what a step is. The second step must use $-0.9$ and $-0.7$ alone, because gradients accumulate. With clipping, `grad_norm` is $\sqrt{13}$ and the update uses the clipped gradients, which keeps long runs stable. A `nan` loss raises and no weight moves, so one bad batch cannot poison the run. `(loss, metrics)` pairs are reported, so accuracy can sit beside the loss, and a vector loss is an error. With $\beta = 0.9$, the losses 1 and 0.145 must report `loss_ema` 1.0 and 0.55, and a `nan` loss never reaches the EMA; that smoothed curve is what every later training log plots.

The loader must yield in-order batches with `drop_last` and `len` right, so evaluation sees every row once; its shuffle must be the spec's Fisher-Yates, new each epoch, so the Go and Rust ports shuffle the same way; and unequal lengths or a shuffle without a generator must fail, to prevent silent mispairing. Evaluation must count a 2-row batch as 2 rows, run in eval mode under `no_grad`, and restore the mode, including after an exception, so validation numbers are honest (as [LM evaluation harness and the model zoo](../p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) requires) and dropout cannot stay off.

Two learning checks exercise the whole stack: a 2-8-1 tanh MLP must solve XOR, which puts every piece of the chapter on one path, and minibatch AdamW with clipping must reach 95% accuracy on two moons, a real minibatch run. These two are judged against a loss bar, the solution's own mean over five seeds plus three standard deviations, so a correct but differently ordered implementation still passes and a broken one does not. The same seed must give bitwise-identical weights and metrics, and another seed must differ, for resuming in [Safetensors for every dtype, atomic checkpoints, and the token stream](06-safetensors-checkpoints-and-token-streams.md) and for regression tests.

For the bigram: the hand-worked NLL $\ln 3$ and row `a`'s gradient $[1/6, -1/6, 0]$ show it trains the right rows; logits must be table rows for both 1-D and `[B, T]` ids, with bad ids failing, since the trainer feeds token windows; the autograd bigram must reach within $10^{-3}$ nats of the count floor and never go below, which is the chapter's end-to-end run in miniature; `to_lm()` must give the same logits through `BigramLM.logits`, as a copy, so the engine serves what was trained; and the sampled ids must equal the engine's PCG32 draw at three temperatures, so Python and Rust agree on a seed.

## Building it step by step

### The reasoning chain

1. **A step must use this batch's gradient and nothing else.** `backward` adds into `.grad`, so the step zeroes first. Putting `zero_grad` at the start rather than the end also covers the first step after construction or after a failed batch.
2. **A step must never write a non-finite update.** The only cheap place to check is the loss value, before `backward` and `opt.step`. A `nan` loss gives `nan` gradients, and one `opt.step` with them destroys every weight. So: compute the loss, test it, raise if needed, and only then backpropagate.
3. **Clipping sits between backward and step**, because it rescales gradients that exist only after backward and must be applied before the optimizer reads them. The useful number to report is the norm before clipping: after clipping it is always at most $c$ and tells you nothing.
4. **Reproducibility comes from one generator and one algorithm.** If the permutation is drawn from PCG32 with the spec's Fisher-Yates, the Go and Rust ports shuffle identically, and drawing it when each epoch starts makes epoch 2 differ from epoch 1 while the whole run stays a function of the seed.
5. **Evaluation must not train and must not leak state.** Eval mode turns dropout off; `no_grad` builds no graph. Whatever happens inside, the mode must be restored, which forces a `try`/`finally`. Averages are over rows, so a short last batch is weighted by its size.
6. **The bigram's optimum is known.** The loss is convex with the unsmoothed counts as its minimizer, with NLL $0.5493$ on `abbacab`. A table starting at zeros (NLL $\ln 3 = 1.0986$) and trained by the loop must approach that floor from above. That makes the bigram a precise check of every piece of autograd in between.
7. **Sampling must be the engine's arithmetic, not just its distribution.** Two correct samplers with different summation orders or different generators give different ids for one seed. Copying the engine's exact recipe (one `PCG32(seed)`, one `uniform()` per token, float64 weights summed in id order) is the only way to make a seed mean one text in both languages.

### The loader

All validation happens in the constructor, so a bad loader fails where it is built, not at the first batch.

```python
class DataLoader:
    def __init__(self, arrays, batch_size, shuffle, rng, drop_last=True) -> None:
        if not arrays:
            raise ValueError("DataLoader needs at least one array")
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        lengths = {len(v) for v in self.arrays.values()}
        if len(lengths) != 1:
            # Unequal lengths would silently pair inputs with the wrong labels.
            raise ValueError(f"arrays have different lengths: {sorted(lengths)}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if shuffle and rng is None:
            raise ValueError("shuffle needs an rng (a PCG32)")   # no hidden unseeded order
        self.n = lengths.pop()
        # ... store batch_size, shuffle, rng, drop_last

    def __iter__(self):
        order = list(range(self.n))
        if self.shuffle:
            # Drawn when the epoch starts: each epoch advances the generator,
            # so epoch 2 differs from epoch 1 and a seeded run repeats exactly.
            self.rng.shuffle(order)          # spec/pcg32.md Fisher-Yates (M06.3)
        idx = np.asarray(order, dtype=np.int64)
        b = self.batch_size
        stop = (self.n // b) * b if self.drop_last else self.n
        for i in range(0, stop, b):
            rows = idx[i : i + b]
            yield {k: v[rows] for k, v in self.arrays.items()}   # same rows from every array

    def __len__(self) -> int:
        return self.n // self.batch_size if self.drop_last else math.ceil(self.n / self.batch_size)
```

Because `__iter__` is a generator, the shuffle runs when a `for` loop starts, which is exactly "when each epoch starts".

### The loss contract

One helper normalizes what `loss_fn` may return, used by both the step and evaluation.

```python
def _split(out: Any) -> tuple[Tensor, dict[str, float]]:
    """loss_fn's result as (loss, extra metrics)."""
    loss, extra = (out[0], dict(out[1])) if isinstance(out, tuple) else (out, {})
    if not isinstance(loss, Tensor):
        raise TypeError(f"loss_fn must return a Tensor (or a (Tensor, dict) pair), got {type(loss).__name__}")
    if loss.data.size != 1:
        # backward() on a vector would need an upstream gradient nobody chose.
        raise ValueError(f"the loss must have one element, got shape {loss.shape}")
    return loss, {k: float(v) for k, v in extra.items()}
```

### The step

The order of the lines is the content: zero, loss, finite check, backward, clip, step, report.

```python
def train_step(model, batch, loss_fn, opt, clip=None, ema=None) -> dict[str, float]:
    # Zero first: gradients accumulate across backward calls (L0.1).
    opt.zero_grad()
    loss, extra = _split(loss_fn(model, batch))
    value = float(np.asarray(loss.data).reshape(()))
    if not math.isfinite(value):
        # Stop before the update: one nan step would poison every weight.
        raise FloatingPointError(f"the loss is {value}; no parameter was updated")
    loss.backward()
    stats = {"loss": value}
    if clip is not None:
        # clip_grad_norm_ returns the norm *before* rescaling: the diagnostic.
        stats["grad_norm"] = float(clip_grad_norm_(model.parameters(), clip))
    opt.step()
    stats.update(extra)
    if ema is not None:
        # One step's loss is noisy; the curve to read is its EMA (M02.2),
        # debiased so the first steps are not pulled toward m_0 = 0.
        ema.update(value)          # never reached by a nan: we raised above
        stats["loss_ema"] = float(ema.value_debiased())
    return stats
```

On the hand-worked example, `value` is 1, backward fills $-3$ and $-2$, and with `clip=1` the reported norm is $\sqrt{13} = 3.606$ while SGD sees $-3/3.606$ and $-2/3.606$. For the EMA, with $\beta = 0.9$: after loss 1 the raw average is $0.1$ and the debiased one $0.1/(1 - 0.9) = 1.0$; after $0.145$ it is $(0.09 + 0.0145)/(1 - 0.81) = 0.55$.

### Evaluation

The `finally` is the reason this function is longer than a loop.

```python
def evaluate(model, loader, loss_fn) -> dict[str, float]:
    was_training = model.training
    model.eval()                                   # dropout off, everywhere (L0.4)
    sums: dict[str, float] = {}
    n = 0
    try:
        with no_grad():                            # no graph, no stored activations
            for batch in loader:
                loss, extra = _split(loss_fn(model, batch))
                rows = len(next(iter(batch.values())))
                # Weight by rows: a smaller last batch must count less.
                for k, v in {"loss": float(np.asarray(loss.data).reshape(())), **extra}.items():
                    sums[k] = sums.get(k, 0.0) + v * rows
                n += rows
    finally:
        model.train(was_training)                  # restored on success and on error
    if n == 0:
        raise ValueError("evaluate: the loader produced no batches")
    out = {k: v / n for k, v in sums.items()}
    out["n"] = float(n)
    return out
```

It restores `was_training`, not `True`: evaluating a model that was already in eval mode must leave it there.

### The bigram as a module

One parameter, zeros, and a gather.

```python
class BigramLogits(Module):
    def __init__(self, vocab: int = 256) -> None:
        super().__init__()
        if vocab < 1:
            raise ValueError(f"vocab must be >= 1, got {vocab}")
        # Zeros: every next token starts equally likely (NLL ln V), and no
        # random init is needed for a convex problem.
        self.weight = Tensor(np.zeros((vocab, vocab)), requires_grad=True)

    def forward(self, ids: ArrayLike) -> Tensor:
        # A row gather is the one-hot matmul without the zeros: same logits,
        # and backward adds each position's gradient into its row. Any shape
        # of ids: the trainer feeds [B, T] windows. F.embedding checks them.
        return F.embedding(self.weight, ids)

    def to_lm(self) -> BigramLM:
        # A float32 copy: training can continue without changing the served model.
        return BigramLM(self.weight.data.astype(np.float32, copy=True))
```

Zeros are safe here because the problem is convex: there is no symmetry to break, unlike a hidden layer, where identical rows would receive identical gradients forever.

The CLI's `train bigram --method autograd` drives `BigramLogits` with full-batch AdamW at learning rate 0.5 and no weight decay, using `cross_entropy` on `model(x)` for every `(byte, next byte)` pair; see `train_bigram_full_batch` in [`cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py).

### Sampling like the engine

The sampler keeps the counting bigram's signature; what changed is the generator and the draw.

```python
    def sample(self, prefix: list[int], n: int, temperature: float, seed: int) -> list[int]:
        # ... validate prefix (non-empty), n >= 0, temperature >= 0
        cur = int(_as_ids(prefix, v)[-1])
        # One generator per call, stream 54: the tracer engine's PCG32(seed, 54).
        rng = PCG32(seed)
        out: list[int] = []
        for _ in range(n):
            row = self.logits([cur])[0].astype(np.float64)
            if temperature == 0:
                nxt = int(np.argmax(row))  # first maximum: ties go to the lowest id
            else:
                nxt = _draw(row, float(temperature), rng.uniform())   # one uniform per token
            out.append(nxt)
            cur = nxt
        return out


def _draw(row: NDArray, temperature: float, u: float) -> int:
    """The tracer engine's draw (L10.0): weights exp(z - max z) of z = row /
    temperature in f64, summed in id order, NaN never picked; the first id
    whose running sum exceeds u * total."""
    z = [float(x) / temperature for x in row]
    finite = [x for x in z if not math.isnan(x)]
    m = max(finite) if finite else -math.inf
    w = [0.0 if math.isnan(x) or m == -math.inf else math.exp(x - m) for x in z]
    total = 0.0
    for wi in w:                 # a plain left-to-right sum, as the Rust loop does
        total += wi
    target = u * total
    cum, last = 0.0, 0
    for i, wi in enumerate(w):
        if wi > 0.0:
            cum += wi
            last = i
            if cum > target:
                return i
    return last                  # rounding left target unreached: the last live id
```

The explicit Python loops are deliberate: `np.sum` uses pairwise summation, which can round differently from the engine's sequential sum and move the chosen id at a boundary.

The full solution is [`loop.py`](../../../../course/solutions/python/tinyllm/train/loop.py) and [`bigram.py`](../../../../course/solutions/python/tinyllm/lm/bigram.py), which also keeps `BigramLM`'s `fit_counts`, `logits`, and `nll` from the counting version.

### Why this design

One step is one forward, one backward, and $O(|\theta|)$ for clipping and the update; the loop adds nothing asymptotic. The loader's shuffle is $O(n)$ per epoch, and fancy indexing copies one batch at a time, so memory stays at the dataset plus one batch. The bigram's forward is a gather, $O(BT \cdot V)$, instead of the one-hot matmul's $O(BT \cdot V^2)$. (The tracer's C matmul `tl_matmul_f32`, from [Vectors, matrices, row-major layout, and matmul in Python](../../mathematics-for-computing/03-linear-algebra/01-vectors-matrices-and-matmul.md) and [the kernels chapter](../p09-kernels/README.md), is not needed here: `BigramLM.logits` reads the same rows with a NumPy gather.)

Each alternative fails in a specific way. Shuffling by `argsort` of uniforms is a correct random permutation but not the spec's, so the ports would disagree. Checking for `nan` after `opt.step` detects the problem one step too late. Averaging batch means in `evaluate` is simpler and biased whenever the last batch is short. Sampling with `np.random.choice` gives the right distribution and the wrong ids.

Two tradeoffs are accepted. The step has no gradient accumulation, mixed precision, or schedule; later chapters add those around it rather than inside it. And `grad_norm` is reported only when clipping is on, because computing a global norm costs a pass over every gradient.

## What can go wrong

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| no `zero_grad` at the start of the step | the loss falls, then oscillates: every step adds all earlier gradients | `opt.zero_grad()` is the first line of `train_step` |
| evaluation that averages batches, leaves dropout on, or restores the mode only on success | validation loss biased by the last batch; noisy evaluations; training after a failed eval with dropout off | sums `v * rows` and divides by total rows; `model.eval()` then `model.train(was_training)` in `finally` |
| shuffling by sorting uniforms, or one permutation for every epoch | runs are not reproducible across languages, or every epoch sees the same order | `rng.shuffle(order)` (the spec's Fisher-Yates) inside `__iter__`, once per epoch |
| checking the loss for `nan` after the update | the run is already lost when the error is raised | `math.isfinite(value)` is tested before `backward` and `opt.step` |
| a new generator for every token, or a stream other than the engine's | Python and Rust disagree on the same seed; repeated draws of the same $u$ | one `PCG32(seed)` (stream 54) per call, one `uniform()` per token, the engine's `_draw` |

> **Warning:** A sampler can be statistically perfect and still fail to match the engine. Summation order and generator choice change which id a given $u$ selects at a boundary, so copy the engine's arithmetic exactly.

## Where this leads

This section is where the chapter closes end to end: together with the next section, the autograd bigram reaches the count MLE and the digits MLP trains through the loader. The Rust engine of [Your first endpoint](../p10-serving/00-your-first-endpoint.md) serves the table `to_lm()` produces, unchanged (`tl_arch = bigram`). [Safetensors for every dtype, atomic checkpoints, and the token stream](06-safetensors-checkpoints-and-token-streams.md) takes over `safetensors.py` and keeps the counting behaviour of `bigram.py` as its regression.

Later chapters call `train_step` and `evaluate` as they are. The NPLM of [Bengio's neural probabilistic language model](../p02-statistical-lm/02-bengio-nplm.md) and the [encoder-decoder Transformer](../p05-transformer-2017/05-encoder-decoder-transformer.md) train through `DataLoader` and `train_step`; the [RNN language model](../p03-recurrent/06-rnn-language-model-stateful-tbptt.md) trains through `train_step`; [GPT pre-training](../p06-objectives/01-gpt-decoder-only.md) calls `train_step` each step, as does the capstone; and the model zoo of [LM evaluation harness and the model zoo](../p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) loads the bigram as a `BigramLM` and scores it with the others.

### Further reading

- Karpathy, "A Recipe for Training Neural Networks" (2019, free).
- Goodfellow, Bengio, Courville, *Deep Learning*, ch. 8.
- `torch.utils.data.DataLoader` (`torch/utils/data/dataloader.py`): worker processes, pinned memory, samplers, and a `generator` argument for seeded shuffles.
- Hugging Face `Trainer.training_step` (`transformers/trainer.py`) and Lightning's `LightningModule`: mixed precision, gradient accumulation, distributed reduction, and callbacks.
- `torch.cuda.amp.GradScaler` (`torch/amp/grad_scaler.py`): skips steps with `inf` gradients and lowers the loss scale, as in [mixed-precision training](../p11-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md).
- vLLM's per-request seeded sampling (vLLM `v1/sample/sampler.py`): a generator per request on GPU, with the same draws across batch compositions.

## Key terms

- **training step**: one zero-grad, loss, finiteness check, backward, clip, and optimizer update on one batch.
- **global gradient norm**: the square root of the sum of squares of every gradient entry across all parameters.
- **gradient clipping**: rescaling all gradients together when their global norm exceeds a threshold.
- **Fisher-Yates shuffle**: the in-place permutation algorithm that swaps each position $i$, from the end, with a uniformly chosen position at or below it.
- **debiased EMA**: an exponential moving average divided by $1 - \beta^t$ so its early values are not pulled toward zero.
- **NLL floor**: the NLL of the unsmoothed count model, the minimum any bigram can reach on a given text.

## Summary

- One training step is zero the gradients, compute the loss, refuse a non-finite loss, backpropagate, clip, update: in that order, every time.
- Data order comes from the seeded PCG32 through the spec's Fisher-Yates shuffle, a fresh permutation each epoch, so two runs with one seed are bitwise identical.
- Evaluation weights every batch by its rows, runs in eval mode under `no_grad`, and always hands the model back in the mode it found it.
- The bigram is a convex problem whose minimum is the unsmoothed count model; the autograd-trained table reaches its NLL within $10^{-3}$ nats.
- `sample` now draws exactly as the Rust engine does, so a seed gives the same text in Python and Rust.

## Review questions

1. Why does `train_step` call `opt.zero_grad()` at the start instead of after `opt.step()`?

   <details><summary>Answer</summary>

   Between two normal steps either placement works. Zeroing at the start makes the step independent of whatever ran before it: a freshly built model, a gradient check or a manual `backward` in a notebook, or a previous step whose `opt.step` raised after `backward`. The step's contract becomes "the update uses this batch's gradient", with no assumption about the caller.

   </details>

2. Why is the finiteness check on the loss value and not on the gradients?

   <details><summary>Answer</summary>

   A non-finite loss is the first observable symptom and is one number, so it is checked before any work is done. Checking gradients would need a pass over every parameter and still has to happen before `opt.step`. Raising before `backward` also keeps `nan` out of the EMA and the moment buffers.

   </details>

3. In `evaluate`, why restore `was_training` in a `finally` instead of calling `model.train()` at the end?

   <details><summary>Answer</summary>

   An exception in any batch would skip a trailing `model.train()`, leaving a training run with dropout silently off. And `model.train()` would wrongly switch on a model that was in eval mode before the call. Restoring the saved flag in `finally` is right in both cases.

   </details>

4. Why can `BigramLogits` start from zeros when an MLP cannot?

   <details><summary>Answer</summary>

   The bigram loss is convex in $W$ with a unique optimum (the counts' softmax), so any start reaches it, and zeros give the honest NLL $\ln V$. In an MLP, hidden units with identical weights get identical gradients and stay identical; random init is what breaks that symmetry.

   </details>

5. What breaks if `sample` created a new `PCG32(seed)` for every token?

   <details><summary>Answer</summary>

   Every token would use the same first `uniform()` $u$, so the sampler would repeat the same quantile of each row's distribution, and it would no longer match the engine, which draws a fresh $u$ per token from one generator. A seed would mean different text in Python and Rust.

   </details>

6. On `abbacab`, why can the trained bigram approach an NLL of $0.5493$ but never go below it, while the add-one model sits at $0.8351$?

   <details><summary>Answer</summary>

   $0.5493 = 3\ln 3 / 6$ is the NLL of the unsmoothed count model, which is the unique minimizer of the convex bigram loss, so no table can do better. The add-one model is a different table (smoothed toward uniform), so its NLL is higher. Training approaches the floor from above without reaching it, because the logits of pairs that never occur keep moving toward $-\infty$.

   </details>
