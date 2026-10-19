# Training loop and the autograd bigram

## Overview

| | |
|---|---|
| **Module** | `L0.5` · build · Python · Pass 2 · 2 to 3 h |
| **Solution** | [`loop.py`](../../../../course/solutions/python/tinyllm/train/loop.py): `DataLoader`, `train_step`, `evaluate` · [`bigram.py`](../../../../course/solutions/python/tinyllm/lm/bigram.py), taken over from `L0.0` (its earlier version is [`history/L0.0/`](../../../../course/solutions/history/L0.0/)): `BigramLogits` (the table as a trainable `Module`), `to_lm`, and `BigramLM.sample` on PCG32 · the `train` verbs in [`cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py) |
| **Contract** | [`course/contracts/py/tinyllm/train/loop.pyi`](../../../../course/contracts/py/tinyllm/train/loop.pyi) · [`course/contracts/py/tinyllm/lm/bigram.pyi`](../../../../course/contracts/py/tinyllm/lm/bigram.pyi) |
| **Needs** | `L0.1` `no_grad` · `L0.2` `F.embedding` · `L0.3` the losses the loop trains with · `L0.4` `Module`, `Linear` · `M06.3` PCG32 · `M10.2` SGD · `M10.3` AdamW · `M10.4` `clip_grad_norm_` · `M02.2` `EMA` (the smoothed loss) · `rt.01` and `M03.1` (the C matmul of Pass 1) · reading: `L0.0` |
| **Used by** | `L10.0` the engine serves the table this module trains (inherited from `L0.0` with `bigram.py`) · `L0.6` keeps `L0.0`'s behaviour of `bigram.py` as its regression · later: `L2.2`, `L3.6`, `L6.1`, and the capstone train with this loop · later: `L5.5`, `L6.7`. Together with `L0.6`, this module closes Part 0 end to end: the autograd bigram reaches the count MLE, and the digits MLP trains through the loader |
| **Optional depth** | Karpathy, "A Recipe for Training Neural Networks" (2019, free); Goodfellow, Bengio, Courville, *Deep Learning*, ch. 8 |

## Key Takeaways

- One training step is zero the gradients, compute the loss, refuse a non-finite loss, backpropagate, clip, update: in that order, every time.
- Data order comes from the seeded PCG32 through the spec's Fisher-Yates shuffle, a fresh permutation each epoch, so two runs with one seed are bitwise identical.
- Evaluation weights every batch by its rows, runs in eval mode under `no_grad`, and always hands the model back in the mode it found it.
- The bigram is a convex problem whose minimum is the unsmoothed count model; the autograd-trained table reaches its NLL within $10^{-3}$ nats.
- `sample` now draws exactly as the Rust engine does, so a seed gives the same text in Python and Rust.

## How to read this chapter

Sections 1 to 3 build the training step and the bigram's optimum from first principles and work both by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through `loop.py` and the new parts of `bigram.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

The solution's command-line entry point, [`cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py), uses this module for two verbs: `train bigram --method autograd --data F --out DIR` (full-batch AdamW on every byte pair of `F`, then the same model directory as Pass 1, from the trained table) and `train mlp --data digits.npz --hidden 64 --epochs 30 --ckpt DIR` (a two-layer ReLU MLP on the UCI digits through `DataLoader`, with each pixel standardized by the training set's mean and standard deviation). `--method counts` stays the Pass 1 verb.

---

## 1. Why now

`L0.1` to `L0.4` give you every piece of a training run and no run. The digits MLP that closes Part 0 needs batches drawn in a reproducible order, a step that does the five things of a step in the right order, and an evaluation that does not train. And the tracer model the engine has served since Pass 1 is still a table of counts: it cannot be improved, only recounted. This module writes the loop every later model trains with, then retrains the tracer's own table with it. If the loop is right, gradient descent finds the count model again (the bigram problem has one minimum, and the counts are it), which makes the bigram the most precise end-to-end test of the autograd there is: the answer is known to many digits.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\theta$ | every parameter of the model | list of `Tensor` |
| $B$ | batch size; $n$ rows in the dataset | ints |
| $\mathcal{L}(\theta; \text{batch})$ | the loss `loss_fn` returns, one element | `Tensor` |
| $\eta$ | learning rate | float |
| $c$ | gradient clipping threshold, global norm (`M10.4`) | float |
| $\lVert g \rVert = \sqrt{\sum_p \sum_i g_{p,i}^2}$ | global gradient norm over all parameters | float |
| $W \in \mathbb{R}^{V \times V}$ | the bigram table, row $i$ = logits after token $i$ | `float32[V, V]` |
| $C_{ij}$, $R_i = \sum_j C_{ij}$ | bigram counts and row totals (`L0.0`) | |
| $N = T - 1$ | predictions in a text of $T$ tokens | int |
| $u$ | one uniform draw in $[0, 1)$ from PCG32 | float |

**The loader.** `DataLoader(arrays, batch_size, shuffle, rng)` holds named arrays that share their first dimension $n$ and yields dictionaries of rows. Without shuffle the order is $0, \dots, n - 1$. With shuffle, when each epoch **starts** it builds the list $0, \dots, n - 1$ and calls `rng.shuffle` on it: `spec/pcg32.md`'s Fisher-Yates (`M06.3`), going from the end and swapping $i$ with `below(i + 1)`. Each epoch advances the generator, so epoch 2 has a new permutation and a seeded run repeats exactly. `drop_last` drops a final partial batch, so every step sees $B$ rows; `len(loader)` says how many batches an epoch has. Arrays of different lengths, or a shuffle without a generator, are errors: one pairs inputs with the wrong labels, the other hides an unseeded order.

**The step.** `train_step(model, batch, loss_fn, opt, clip)`:

1. `opt.zero_grad()`. Gradients **accumulate** across `backward` calls (`L0.1`), so without this step $k$ trains on the sum of the gradients of steps $1, \dots, k$.
2. `loss = loss_fn(model, batch)`: a one-element Tensor, or a pair `(loss, {"acc": ...})` of the loss and extra metrics. More than one element is an error: `backward()` on a vector would need an upstream gradient nobody chose.
3. If the loss is not finite, raise `FloatingPointError` **before** any update: one `nan` step turns every weight into `nan` and the run is lost; raising first lets the caller skip the batch.
4. `loss.backward()`.
5. With `clip`, `clip_grad_norm_(parameters, c)` (`M10.4`) rescales all gradients together when $\lVert g \rVert > c$; report the norm **before** clipping, because that is the number that tells you training is unstable.
6. `opt.step()`, then return `{"loss": ..., "grad_norm": ...}` plus the extra metrics.
7. With `ema` (an `M02.2` `EMA`), feed the loss to `ema.update` and report `ema.value_debiased()` as `"loss_ema"`. One batch's loss is noise; the debiased EMA is the curve a training log plots, and without the correction its first steps sit near zero. A non-finite loss raised in step 3, so it never reaches the average.

**Evaluation.** `evaluate(model, loader, loss_fn)` switches the model to eval mode (dropout off, `L0.4`), runs every batch under `no_grad` (no graph, no stored activations, `L0.1`), and averages the loss and metrics **weighted by rows**: a last batch of 2 rows counts as 2 rows, not as a full batch. It puts the model back in the mode it found it, in a `finally`, so an exception in a batch cannot leave a training run with dropout silently off.

**The same model, trained.** `BigramLogits(V)` is a `Module` with one parameter, `weight` of shape `[V, V]`, starting at zeros (every next token equally likely, NLL $\ln V$). Its forward is a row lookup, `F.embedding(weight, ids)`: the logits `onehot(ids) @ W` of `L0.0` without the multiplications by zero, for ids of any shape (the trainer feeds `[B, T]` windows). Backward adds each position's $p - e_{\text{next}}$ into its row.

**Why gradient descent finds the counts.** The mean NLL of the table on a text is

$$\mathcal{L}(W) = \frac{1}{N}\sum_{i,j} C_{ij}\,\big(\log\textstyle\sum_k e^{W_{ik}} - W_{ij}\big),$$

a sum over rows of convex functions (log-sum-exp is convex, $-W_{ij}$ is linear). Setting the gradient of row $i$ to zero gives $R_i\,\mathrm{softmax}(W_i) = C_i$, so at the minimum $\mathrm{softmax}(W_i)_j = C_{ij}/R_i$: the **unsmoothed** count model. Its NLL, $-\frac{1}{N}\sum_{ij} C_{ij}\log(C_{ij}/R_i)$, is the floor no bigram can beat on that text. Pass 1's add-one model sits slightly above it; the trained table gets within $10^{-3}$ nats of the floor, never below it. Pairs that never occur push their logits toward $-\infty$ forever, so the loss approaches the floor without reaching it; AdamW (`M10.3`) with a large learning rate gets there in a few hundred full-batch steps.

**Sampling like the engine.** `BigramLM.sample` keeps its signature and switches its generator: one `PCG32(seed)` (stream 54) per call, one `uniform()` per token, and the Rust engine's draw (`L10.0`): weights $w_j = e^{z_j - \max z}$ of $z = \text{row}/\tau$ in float64, summed in id order, and the first $j$ whose running sum exceeds $u \cdot \sum_j w_j$. Same arithmetic, same order, same generator: `generate --seed S` and the engine's completion with seed `S` give the same ids. `to_lm()` turns the trained module into a `BigramLM` holding a float32 **copy** of the table, the object the CLI saves as `bigram.weight`.

## 3. Worked example by hand

**One training step.** `Linear(1, 1)` with $w = 2$, $b = 0$; the batch $x = [[1], [2]]$, $y = [[3], [5]]$; loss `mse`; SGD with $\eta = 0.1$.

1. Predictions $wx + b = [2, 4]$; residuals $[2 - 3, 4 - 5] = [-1, -1]$; loss $((-1)^2 + (-1)^2)/2 = 1$.
2. Gradients of $\frac{1}{2}\sum r_k^2$: $\partial/\partial w = \frac{2}{2}\sum r_k x_k = -1 \cdot 1 - 1 \cdot 2 = -3$, and $\partial/\partial b = \sum r_k = -2$.
3. Update: $w = 2 - 0.1 \cdot (-3) = 2.3$, $b = 0 - 0.1 \cdot (-2) = 0.2$.
4. A second step must use only the new gradients: residuals $[2.5 - 3, 4.8 - 5] = [-0.5, -0.2]$, so $\partial/\partial w = -0.5 - 0.4 = -0.9$, $\partial/\partial b = -0.7$. A loop that forgot `zero_grad` would add the first step's $-3$ and $-2$.
5. With `clip=1`, the global norm is $\sqrt{3^2 + 2^2} = \sqrt{13} = 3.606$; it is reported, and both gradients are scaled by $1/3.606$ before the update.

**The bigram, from zeros.** `BigramLogits(3)` on the text `abbacab` (ids `[0, 1, 1, 0, 2, 0, 1]`, $N = 6$ predictions):

1. Every row of zeros has softmax $[1/3, 1/3, 1/3]$, so the NLL is $\ln 3 = 1.0986$.
2. Row `a` makes three predictions (the next tokens are `b`, `c`, `b`). Each contributes $p - e_{\text{next}}$, and the mean divides by 6: $\big(3 \cdot [\tfrac13, \tfrac13, \tfrac13] - [0, 2, 1]\big)/6 = [\tfrac16, -\tfrac16, 0]$. Descent raises $W_{ab}$ and lowers $W_{aa}$; $W_{ac}$ is already right on average.
3. The floor: row `a` goes to $b$ twice and $c$ once ($2/3$, $1/3$), row `b` to $a$ and $b$ once each, row `c` to $a$ always. The NLL is $\frac{1}{6}\big(2\ln\tfrac32 + \ln 3 + 2\ln 2 + 0\big) = \frac{3\ln 3}{6} = 0.5493$, below the add-one model's $0.8351$ of `L0.0`.

These numbers are the first cases a correct solution must reproduce.

## 4. The interface

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked step | unit | section 3: loss 1, gradients $-3$ and $-2$, then $w = 2.3$, $b = 0.2$ | everyone agrees on what a step is |
| zero_grad every step | unit | the second step uses $-0.9$ and $-0.7$ alone | gradients accumulate (`L0.1`) |
| clipping reports the norm | unit | `grad_norm` is $\sqrt{13}$, the update uses clipped gradients | long runs stay stable (`M10.4`) |
| a non-finite loss stops before the update | boundary | a `nan` loss raises and no weight moves | one bad batch cannot poison the run |
| loss_fn extras and shape | boundary | `(loss, metrics)` pairs are reported; a vector loss is an error | accuracy beside the loss |
| loss_ema is the debiased average | unit | with $\beta = 0.9$ the losses 1 and 0.145 report `loss_ema` 1.0 and 0.55; a `nan` loss never reaches the EMA | the smoothed curve every later training log plots (`M02.2`) |
| in-order batches | unit | in-order batches, `drop_last`, `len` | evaluation sees every row once |
| the shuffle is the spec's permutation | property | the order is the spec's Fisher-Yates, new each epoch | the Go and Rust ports shuffle the same way |
| bad loader input | boundary | unequal lengths, shuffle without a generator | silent mispairing |
| evaluation weights by rows and restores the mode | unit | a 2-row batch counts 2 rows; eval mode, `no_grad`, mode restored | honest validation numbers (`L6.7`) |
| evaluation restores the mode after an error | boundary | an exception still restores training mode | dropout cannot stay off |
| learns XOR | learning | a 2-8-1 tanh MLP solves XOR | every piece of `L0.1` to `L0.5` on one path |
| learns two moons | learning | minibatch AdamW with clipping reaches 95% accuracy | a real minibatch run |
| same seed, bitwise identical | property | same seed, identical weights and metrics; another seed differs | resuming (`L0.6`) and regression tests |
| the bigram's hand-worked gradient | unit | section 3: NLL $\ln 3$, row `a`'s gradient $[1/6, -1/6, 0]$ | the bigram trains the right rows |
| forward is a row gather | unit | logits are table rows for 1-D and `[B, T]` ids; bad ids fail | the trainer feeds token windows |
| the autograd bigram reaches the count MLE | property | within $10^{-3}$ nats of the floor, never below | the Part 0 end-to-end run in miniature |
| `to_lm` serves the same table | unit | `to_lm()` gives the same logits through `BigramLM.logits`, as a copy | the engine serves what was trained |
| sampling draws like the engine | unit | ids equal the engine's PCG32 draw at three temperatures | Python and Rust agree on a seed |

The two learning cases are judged against a loss bar: the solution's own mean over five seeds plus three standard deviations, so a correct but differently ordered implementation still passes and a broken one does not.

## 5. The solution, step by step

### From first principles

1. **A step must use this batch's gradient and nothing else.** `backward` adds into `.grad` (`L0.1`), so the step zeroes first. Putting `zero_grad` at the start rather than the end also covers the first step after construction or after a failed batch.
2. **A step must never write a non-finite update.** The only cheap place to check is the loss value, before `backward` and `opt.step`. A `nan` loss gives `nan` gradients, and one `opt.step` with them destroys every weight. So: compute the loss, test it, raise if needed, and only then backpropagate.
3. **Clipping sits between backward and step**, because it rescales gradients that exist only after backward and must be applied before the optimizer reads them. The useful number to report is the norm before clipping: after clipping it is always at most $c$ and tells you nothing.
4. **Reproducibility comes from one generator and one algorithm.** If the permutation is drawn from PCG32 with the spec's Fisher-Yates, the Go and Rust ports shuffle identically, and drawing it when each epoch starts makes epoch 2 differ from epoch 1 while the whole run stays a function of the seed.
5. **Evaluation must not train and must not leak state.** Eval mode turns dropout off; `no_grad` builds no graph. Whatever happens inside, the mode must be restored, which forces a `try`/`finally`. Averages are over rows, so a short last batch is weighted by its size.
6. **The bigram's optimum is known.** Section 2 showed the loss is convex with the unsmoothed counts as its minimizer, with NLL $0.5493$ on `abbacab`. A table starting at zeros (NLL $\ln 3 = 1.0986$) and trained by the loop must approach that floor from above. That makes the bigram a precise check of every piece of autograd in between.
7. **Sampling must be the engine's arithmetic, not just its distribution.** Two correct samplers with different summation orders or different generators give different ids for one seed. Copying the engine's exact recipe (one `PCG32(seed)`, one `uniform()` per token, float64 weights summed in id order) is the only way to make a seed mean one text in both languages.

### The code, annotated

**The loader.** All validation happens in the constructor, so a bad loader fails where it is built, not at the first batch.

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

**The loss contract.** One helper normalizes what `loss_fn` may return, used by both the step and evaluation.

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

**The step.** The order of the lines is the content: zero, loss, finite check, backward, clip, step, report.

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

On section 3's example: `value` is 1, backward fills $-3$ and $-2$, and with `clip=1` the reported norm is $\sqrt{13} = 3.606$ while SGD sees $-3/3.606$ and $-2/3.606$. For the EMA case, with $\beta = 0.9$: after loss 1 the raw average is $0.1$ and the debiased one $0.1/(1 - 0.9) = 1.0$; after $0.145$ it is $(0.09 + 0.0145)/(1 - 0.81) = 0.55$.

**Evaluation.** The `finally` is the reason this function is longer than a loop.

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

**The bigram as a Module.** One parameter, zeros, and a gather.

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

The CLI's `train bigram --method autograd` drives this module with full-batch AdamW at learning rate 0.5 and no weight decay, using `cross_entropy` on `model(x)` for every `(byte, next byte)` pair; see `train_bigram_full_batch` in [`cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py).

**Sampling like the engine.** The sampler keeps `L0.0`'s signature; what changed is the generator and the draw.

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

Full solution: [`loop.py`](../../../../course/solutions/python/tinyllm/train/loop.py) · [`bigram.py`](../../../../course/solutions/python/tinyllm/lm/bigram.py) (also `BigramLM` from `L0.0`: `fit_counts`, `logits`, `nll`).

### Why this solution

- **Cost.** One step is one forward, one backward, and $O(|\theta|)$ for clipping and the update; the loop adds nothing asymptotic. The loader's shuffle is $O(n)$ per epoch and fancy indexing copies one batch at a time, so memory stays at the dataset plus one batch. The bigram's forward is a gather, $O(BT \cdot V)$, instead of the one-hot matmul's $O(BT \cdot V^2)$.
- **Alternatives.** Shuffling by `argsort` of uniforms is a correct random permutation but not the spec's, so the ports would disagree. Checking for `nan` after `opt.step` detects the problem one step too late. Averaging batch means in `evaluate` is simpler and biased whenever the last batch is short. Sampling with `np.random.choice` gives the right distribution and the wrong ids.
- **Accepted tradeoffs.** The step has no gradient accumulation, mixed precision, or schedule; later modules add those around it rather than inside it. `grad_norm` is reported only when clipping is on, because computing a global norm costs a pass over every gradient.

### Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. no `zero_grad` at the start of the step | the loss falls, then oscillates: every step adds all earlier gradients | `opt.zero_grad()` is the first line of `train_step` |
| 2. evaluation that averages batches, leaves dropout on, or restores the mode only on success | validation loss biased by the last batch; noisy evaluations; training after a failed eval with dropout off | sums `v * rows` and divides by total rows; `model.eval()` then `model.train(was_training)` in `finally` |
| 3. shuffling by sorting uniforms, or one permutation for every epoch | runs are not reproducible across languages, or every epoch sees the same order | `rng.shuffle(order)` (the spec's Fisher-Yates) inside `__iter__`, once per epoch |
| 4. checking the loss for `nan` after the update | the run is already lost when the error is raised | `math.isfinite(value)` is tested before `backward` and `opt.step` |
| 5. a new generator for every token, or a stream other than the engine's | Python and Rust disagree on the same seed; repeated draws of the same $u$ | one `PCG32(seed)` (stream 54) per call, one `uniform()` per token, the engine's `_draw` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | `evaluate` runs under `no_grad`; the loss is a `Tensor` |
| Back | `L0.2` | `BigramLogits.forward` is `F.embedding` |
| Back | `L0.3` | the loop is exercised with, and the CLI trains on, `cross_entropy`, `mse`, `bce_with_logits` |
| Back | `L0.4` | `BigramLogits` is a `Module`; the tests train `Linear` stacks |
| Back | `M06.3` | `PCG32`: the loader's shuffle and the sampler's draws |
| Back | `M10.2` | `SGD` steps the hand example |
| Back | `M10.3` | `AdamW` trains the MLPs and the bigram |
| Back | `M10.4` | `clip_grad_norm_` in `train_step` |
| Back | `M02.2` | `EMA.update` and `value_debiased` give `train_step`'s `loss_ema` |
| Back | Python reference | This model has no native library boundary. |
| Back | `M03.1` | `tl_matmul_f32` is the C matmul of Pass 1; the solution's `BigramLM.logits` reads the same rows with a NumPy gather |
| Forward | `L10.0` | the Rust engine serves the table `to_lm()` produces, unchanged (`tl_arch = bigram`) |
| Forward | `L0.6` | takes over `safetensors.py`; and keeps `bigram.py`'s `L0.0` behaviour as its regression |
| Forward | `L2.2` | the NPLM trains through `DataLoader` and `train_step` |
| Forward | `L3.6` | the RNN LM trains through `train_step` |
| Forward | `L5.5` | the transformer trains through `DataLoader` and `train_step` |
| Forward | `L6.1` | GPT pre-training calls `train_step` each step |
| Forward | `L6.7` | the model zoo loads the bigram as a `BigramLM` and scores it with the others |

Later passes (`L2.2`, `L3.6`, `L6.1`, the capstone) call `train_step` and `evaluate` as they are.

## Going further

| Piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `DataLoader` | `torch.utils.data.DataLoader` | worker processes, pinned memory, samplers, a `generator` argument for seeded shuffles | `torch/utils/data/dataloader.py` |
| `train_step` | HF `Trainer.training_step`, Lightning `LightningModule` | mixed precision, gradient accumulation, distributed reduction, callbacks | `transformers/trainer.py` |
| non-finite guard | `torch.cuda.amp.GradScaler` | skips steps with `inf` gradients and lowers the loss scale (`L11.1`) | `torch/amp/grad_scaler.py` |
| PCG32 sampling shared with Rust | vLLM's per-request seeded sampling | a generator per request on GPU, the same draws across batch compositions | vLLM `v1/sample/sampler.py` |
