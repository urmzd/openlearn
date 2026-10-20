# Losses with a fused backward

[The op library and gradcheck_all](03-op-library-and-gradcheck.md) provides `log_softmax` and `gather`, so cross-entropy could be written as three ops: `-gather(log_softmax(z), t).mean()`. It would be correct, slow, and fragile. Backward through that chain keeps a `[N, V]` array alive at each step, and for the bigram (`V = 256`) or a real vocabulary (`V = 49,152` for SmolLM2) that is most of the memory a training step uses. Padding makes it worse: batches of token windows have rows that must not count, and a mean that divides by the padded length silently shrinks the loss.

Every training loop from here on, starting with [Training loop and the autograd bigram](06-training-loop-and-the-autograd-bigram.md) and running through [Bengio's neural probabilistic language model](../08-statistical-language-models/02-bengio-nplm.md), [GPT decoder-only](../12-objectives-and-adaptation/01-gpt-decoder-only.md), and the capstone, calls this file once per step. So the loss must be one graph node with a closed-form gradient, correct about padding, and finite for any logits. This section builds three such losses (softmax cross-entropy with `ignore_index`, label smoothing, and three reductions; mean squared error; and binary cross-entropy on logits with `pos_weight`), and then shows a small test suite that pins them down.

The section builds on `from_op` from [Tensor, broadcasting backward, and no_grad](02-tensor-and-broadcasting-backward.md) and, from *Mathematics*, on the stable `log_softmax` of [Stable numerics](../../mathematics/15-numerical-methods-and-floating-point/02-stable-numerics.md), the independent `cross_entropy_vjp` derivation of [Matrix differentials, the trace trick, and closed-form VJPs](../../mathematics/14-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md), and the definition of cross-entropy $H(q, p)$ in [Entropy, cross-entropy, KL, JS, and the k3 estimator](../../mathematics/17-information-theory/01-entropy-cross-entropy-and-kl.md). For the test suite, it helps to have read [TDD, unit tests, and mutation testing](../../software-engineering/03-testing/01-tdd-unit-tests-and-mutation-grading.md) in *Software Engineering*, on writing tests that catch faults. The solution is [`losses.py`](../../../../course/solutions/python/tinyllm/autograd/losses.py) with its test suite [`test_losses.py`](../../../../course/solutions/learner-tests/L0.3/python/tests/l0-3-losses/test_losses.py); the contract is [`losses.pyi`](../../../../course/contracts/py/tinyllm/autograd/losses.pyi).

> **Learning objectives**
> - Derive the fused cross-entropy gradient $\mathrm{softmax}(z) - q$ for one-hot and smoothed targets.
> - Implement `ignore_index` so padding adds no loss, gets no gradient, and does not count in the mean, with an all-padding batch giving 0 rather than `nan`.
> - Apply label smoothing over all $V$ classes and compute a smoothed loss by hand.
> - Write binary cross-entropy in a softplus form that stays finite at logits of $\pm 1000$.
> - Design tests whose expected values come from hand computation and each catch one plausible mistake.

## Deriving the losses

The notation for this section:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $z_i \in \mathbb{R}^V$ | logits of row $i$ ($N$ rows, $V$ classes) | `float[N, V]` (or `[..., V]`) |
| $t_i$ | target class of row $i$, or `ignore_index` | `int[N]` |
| $p_i = \mathrm{softmax}(z_i)$ | predicted distribution | `float[V]` |
| $k_i \in \{0, 1\}$ | 1 when row $i$ is kept ($t_i \ne$ `ignore_index`) | |
| $K = \sum_i k_i$ | number of kept rows | int |
| $\varepsilon$ | label smoothing, in $[0, 1]$ | float |
| $q_i = (1 - \varepsilon)\,e_{t_i} + \varepsilon/V$ | smoothed target, $e_t$ the one-hot vector | `float[V]` |
| $\ell_i = -\sum_j q_{ij} \log p_{ij}$ | row loss, $H(q_i, p_i)$ | float |
| $x, y$ | logit and target of binary cross-entropy, $y \in [0, 1]$ | float |
| $w$ | `pos_weight`, the weight on positive examples | float, default 1 |
| $\sigma(x) = 1/(1 + e^{-x})$, $\mathrm{softplus}(x) = \log(1 + e^x)$ | | functions |

### The fused gradient

For one row with a one-hot target, $\ell = -\log p_t = -z_t + \log\sum_j e^{z_j}$. Differentiating, $\partial\ell/\partial z_j = -[j = t] + e^{z_j}/\sum_k e^{z_k} = p_j - [j = t]$. With a smoothed target, $\ell = -\sum_j q_j z_j + \log\sum_k e^{z_k}$ (because $\sum_j q_j = 1$), so $\partial\ell/\partial z = p - q$. That is the whole backward: one subtraction on the probabilities, which the forward already computed as $e^{\log p}$. A correct fused VJP equals the `cross_entropy_vjp` derived in *Mathematics* on the same inputs.

The forward has to be stable. $\log p = \mathrm{log\_softmax}(z) = z - \max z - \log\sum_j e^{z_j - \max z}$. Computing $-\log(\mathrm{softmax}(z))$ instead underflows: at $z = [0, 10^3]$ softmax returns exactly $[0, 1]$ in float64, and $\log 0 = -\infty$.

### Reductions and padding

The kept mask $k$ zeroes ignored rows in both directions: their loss is 0 and their gradient row is 0. `"none"` returns the per-row losses ($\ell_i k_i$); `"sum"` is $\sum_i k_i \ell_i$; `"mean"` is that sum divided by $K$, the number of **kept** rows. Dividing by $N$ would make a batch of 3 real rows and 5 padding rows report 3/8 of its true loss and train at 3/8 of the learning rate.

When $K = 0$ the mean is $0/0$. This book defines it as loss 0 with a zero gradient. Torch returns `nan`, and one `nan` step poisons every weight. The upstream gradient $\bar\ell$ scales the rows: by $\bar\ell/K$ for the mean, by $\bar\ell$ for the sum, and row by row for `"none"`.

### Label smoothing

A target of exactly one class asks the model to push $p_t$ to 1, which needs $z_t - z_j \to \infty$. Smoothing moves $\varepsilon$ of the target mass onto a uniform distribution over all $V$ classes, so $q_{t} = 1 - \varepsilon + \varepsilon/V$ and every other class gets $\varepsilon/V$. The contract writes the same loss as $(1 - \varepsilon)(-\log p_t) + \varepsilon\,\mathrm{mean}_j(-\log p_j)$. Spreading over $V - 1$ classes (excluding the target) is a different, also published, variant; torch and this book use $V$.

### Mean squared error and binary cross-entropy

Mean squared error is $\mathrm{mse} = \frac{1}{n}\sum (\hat{y} - y)^2$ over all $n$ elements, and its gradient is $\frac{2}{n}(\hat{y} - y)$.

Binary cross-entropy on logits takes one logit $x$ and target $y$: $-[w y \log\sigma(x) + (1 - y)\log(1 - \sigma(x))]$. Using $-\log\sigma(x) = \mathrm{softplus}(-x)$ and $-\log(1 - \sigma(x)) = x + \mathrm{softplus}(-x)$, this is

$$(1 - y)\,x + \big(1 + (w - 1)y\big)\,\mathrm{softplus}(-x),$$

and $\mathrm{softplus}(-x) = \max(-x, 0) + \log(1 + e^{-|x|})$ never exponentiates a positive number. The gradient is $(1 - y) - (1 + (w - 1)y)\,\sigma(-x)$, divided by the element count for the mean.

## Working it by hand

Two rows are enough to see every rule at work. Take logits for two rows over $V = 3$ classes: row 0 is $z_0 = [0, \ln 2, \ln 3]$ with target 2, and row 1 has target `ignore_index` (-100), whatever its logits.

1. $e^{z_0} = [1, 2, 3]$, sum 6, so $p_0 = [1/6, 1/3, 1/2]$.
2. $\ell_0 = -\log p_{0,2} = -\log(1/2) = \ln 2 \approx 0.6931$.
3. Row 1 is ignored: $k = [1, 0]$, $K = 1$, so the mean loss is $\ell_0 / 1 = \ln 2$.
4. The gradient of row 0 is $p_0 - e_2 = [1/6, 1/3, -1/2]$, divided by $K = 1$. Row 1 gets zeros.

Now add label smoothing to the same row 0, with $\varepsilon = 0.3$. Each class gets $0.3/3 = 0.1$ and the target keeps $0.7$ more, so $q = [0.1, 0.1, 0.8]$.

1. $\ell = -(0.1 \log\tfrac16 + 0.1 \log\tfrac13 + 0.8 \log\tfrac12) = 0.1\ln 6 + 0.1\ln 3 + 0.8\ln 2 \approx 0.1792 + 0.1099 + 0.5545 = 0.8436$.
2. The gradient is $p - q = [\tfrac16 - 0.1, \tfrac13 - 0.1, \tfrac12 - 0.8] = [\tfrac{1}{15}, \tfrac{7}{30}, -\tfrac{3}{10}]$.

Both gradients sum to zero, as every softmax gradient must. The walkthrough below traces both examples through the code.

## Designing the API

```python
# python/tinyllm/autograd/losses.py
def cross_entropy(logits: Tensor, targets: ArrayLike, ignore_index: int = -100,
                  label_smoothing: float = 0.0, reduction: Literal["mean", "sum", "none"] = "mean") -> Tensor
def mse(pred: Tensor, target: ArrayLike) -> Tensor
def bce_with_logits(logits: Tensor, targets: ArrayLike, pos_weight: Optional[ArrayLike] = None) -> Tensor
```

Each returns one `from_op` node whose VJP is the closed form above. Targets are validated before anything else: integers, shape `logits.shape[:-1]`, each in $[0, V)$ or equal to `ignore_index`.

A correct solution reproduces both hand examples: loss $\ln 2$ with gradient $[1/6, 1/3, -1/2]$ and zeros on the ignored row, and the smoothed $q = [0.1, 0.1, 0.8]$ with its loss and $p - q$. Its loss and gradient equal `torch.nn.functional` on every option, so training loops ported from torch behave the same. The fused VJPs pass central differences (a closed form off by a factor fails), the fused gradient equals the independently derived `cross_entropy_vjp` matrix, and each smoothed row loss equals $H(q, p)$ computed the slow way from the definition.

The padding rules are checked as properties. Appending ignored rows changes neither the loss nor the kept gradients, which padded batches (the token windows of [Safetensors for every dtype, atomic checkpoints, and the token stream](07-safetensors-checkpoints-and-token-streams.md), the packing of [Supervised fine-tuning](../18-post-training/01-supervised-fine-tuning.md)) rely on. A batch of only padding gives loss 0 and a zero gradient, because one `nan` step ruins a run. Logits of $10^4$ and bce logits of 1000 stay finite, because trained logits are large. The `none`, `sum`, and `mean` reductions agree with each other, which matters because evaluation sums `none` in [The evaluation harness and the model zoo](../12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md). Float32 logits give a float32 loss and gradient. And out-of-range or float targets, an unknown reduction, a bad $\varepsilon$, and shape mismatches are all rejected, so data and config bugs fail here.

## Building it step by step

### The reasoning chain

1. **One node per loss.** The loss is the last op of every training step and its input is the largest array in the graph (`[N, V]` logits). Composing `log_softmax`, `gather`, and `mean` would keep an `[N, V]` intermediate per op alive until backward. A single `from_op` node keeps only `log_softmax(z)` and recomputes $p = e^{\log p}$ in backward.
2. **The gradient is $p - q$, scaled.** The derivation gave $\partial \ell_i / \partial z_i = p_i - q_i$ for any target distribution $q_i$ that sums to 1. So backward is: exponentiate the stored log-probabilities, subtract $\varepsilon / V$ everywhere, subtract $1 - \varepsilon$ more at the target, then apply the keep mask and the reduction's scale.
3. **Validate, then neutralize ignored targets.** A target of `-100` cannot be used as an index (NumPy would read class $V - 100$, or fail). So the target array is validated over kept rows only, ignored entries are replaced by 0 for indexing, and the keep mask zeroes their loss and gradient afterwards. This way the indexing code never branches per row.
4. **The mean divides by kept rows, and $K = 0$ is defined.** The mean's denominator is $K$, and the case $K = 0$ returns 0 in both directions instead of $0/0$.
5. **Stable forms everywhere.** Cross-entropy reads $\log p$ from a stable `log_softmax`. Binary cross-entropy is rewritten with softplus so that no exponent is ever positive, and its gradient uses a stable $\sigma(-x)$.
6. **Keep the input's dtype.** Every output and gradient is cast back to the logits' dtype, so float32 training stays float32.

### Validating targets

```python
def _targets(targets, shape, v: int, ignore_index: int):
    """(targets as int64 with ignored rows set to 0, the keep mask), validated."""
    t = np.asarray(targets)
    if t.size and not np.issubdtype(t.dtype, np.integer):   # 2.0 is not a class
        raise ValueError(f"targets must be integers, got dtype {t.dtype}")
    t = t.astype(np.int64)
    if t.shape != shape:                  # one target per row of logits
        raise ValueError(f"targets have shape {t.shape}, logits need {shape}")
    keep = t != ignore_index
    kept = t[keep]                        # range-check only the rows that count
    if kept.size and (kept.min() < 0 or kept.max() >= v):    # -1 would wrap to class V-1
        raise ValueError(f"targets must lie in [0, {v}) or equal ignore_index {ignore_index}")
    return np.where(keep, t, 0), keep     # 0 is a safe index; keep masks it out later
```

### The cross-entropy forward

```python
def cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
    eps = float(label_smoothing)
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1], got {eps}")
    z = logits if isinstance(logits, Tensor) else Tensor(logits)
    a = z.data
    # ... logits need a class axis
    v = a.shape[-1]
    t, keep = _targets(targets, a.shape[:-1], v, ignore_index)
    lp = np.asarray(stable.log_softmax(a, axis=-1), dtype=a.dtype)   # M09.2: never log(softmax)
    nll = -np.take_along_axis(lp, t[..., None], axis=-1)[..., 0]     # -log p_t per row
    # smoothed: (1 - eps)(-log p_t) + eps * mean_j(-log p_j), the contract's form of H(q, p)
    rows = nll if eps == 0.0 else (1.0 - eps) * nll - eps * lp.mean(axis=-1)
    rows = np.where(keep, rows, 0.0).astype(a.dtype)    # padding adds no loss
    n = int(keep.sum())                                 # K, the kept rows
    if reduction == "mean":
        # Divide by the rows that count: padding must not dilute the loss.
        out = rows.sum() / n if n else np.zeros((), dtype=a.dtype)
    elif reduction == "sum":
        out = rows.sum()
    else:
        out = rows                                      # "none": per-row, ignored rows are 0
    out = np.asarray(out, dtype=a.dtype)
```

`lp.mean(axis=-1)` is the average of $\log p_j$ over all $V$ classes, which is what makes the smoothing spread $\varepsilon / V$ over every class, the target included. On the hand-worked row with $\varepsilon = 0.3$: $0.7 \ln 2 + 0.3 \cdot \tfrac13(\ln 6 + \ln 3 + \ln 2) = 0.4852 + 0.3584 = 0.8436$, the hand value.

### The cross-entropy backward

```python
    def vjp(g):
        # d loss_i / d z_i = softmax(z_i) - q_i, q_i the (smoothed) target.
        d = np.exp(lp)                    # p, from the stored stable log-probabilities
        if eps:
            d = d - eps / v               # the uniform part of q, on every class
        # the one-hot part of q: subtract (1 - eps) at each row's target
        np.put_along_axis(d, t[..., None], np.take_along_axis(d, t[..., None], axis=-1) - (1.0 - eps), axis=-1)
        d = d * keep[..., None]           # ignored rows get exactly zero
        if reduction == "mean":
            d = d * (g / n if n else 0.0) # K = 0: a zero gradient, not nan
        elif reduction == "sum":
            d = d * g
        else:
            d = d * g[..., None]          # "none": one upstream value per row
        return (d.astype(a.dtype, copy=False),)

    return from_op(out, (z,), vjp, "cross_entropy")
```

On the hand-worked row, `exp(lp) = [1/6, 1/3, 1/2]`; without smoothing the target entry becomes $1/2 - 1 = -1/2$, giving $[1/6, 1/3, -1/2]$. With $\varepsilon = 0.3$, subtract $0.1$ everywhere, then $0.7$ more at class 2, giving $[1/15, 7/30, -3/10]$. The ignored row in the first example was indexed at class 0 (by `_targets`), and the keep mask erases whatever that produced.

### Mean squared error

```python
def mse(pred, target):
    p = pred if isinstance(pred, Tensor) else Tensor(pred)
    y = np.asarray(target, dtype=p.dtype)        # the target takes pred's dtype
    if y.shape != p.shape:                       # no silent broadcasting of a target
        raise ValueError(f"target has shape {y.shape}, pred has {p.shape}")
    d = p.data - y
    n = d.size
    out = np.asarray((d * d).sum() / n, dtype=p.dtype)
    return from_op(out, (p,), lambda g: (g * (2.0 / n) * d,), "mse")   # (2/n)(pred - y)
```

The shape check matters more than it looks: `pred` of shape `[N, 1]` against a target of shape `[N]` would broadcast to `[N, N]` and compute a meaningless loss without any error.

### Binary cross-entropy on logits

```python
def bce_with_logits(logits, targets, pos_weight=None):
    z = logits if isinstance(logits, Tensor) else Tensor(logits)
    x = z.data
    y = np.asarray(targets, dtype=x.dtype)
    # ... shape check, as in mse
    w = np.ones((), dtype=x.dtype) if pos_weight is None else np.asarray(pos_weight, dtype=x.dtype)
    c = 1.0 + (w - 1.0) * y  # the weight on softplus(-x) = -log sigmoid(x)
    # softplus(-x) = max(-x, 0) + log1p(exp(-|x|)): the exponent is never positive.
    sp = np.maximum(-x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    n = x.size
    out = np.asarray(((1.0 - y) * x + c * sp).sum() / n, dtype=x.dtype)
    sig_neg = np.exp(-np.logaddexp(0.0, x))  # sigmoid(-x) = 1 / (1 + e^x), stable for either sign

    def vjp(g):
        # d/dx [(1 - y) x + c softplus(-x)] = (1 - y) - c sigmoid(-x), averaged over n
        return ((g / n * ((1.0 - y) - c * sig_neg)).astype(x.dtype, copy=False),)

    return from_op(out, (z,), vjp, "bce_with_logits")
```

`np.logaddexp(0, x)` is $\log(1 + e^x)$ computed without overflow, so $\sigma(-x) = e^{-\mathrm{softplus}(x)}$ is finite for $x = \pm 1000$. You can check the `pos_weight` hand value: at $x = 0$, $y = 1$, $w = 2$, $c = 2$ and $\mathrm{softplus}(0) = \ln 2$, so the loss is $0 + 2 \ln 2$.

The full solution is [`losses.py`](../../../../course/solutions/python/tinyllm/autograd/losses.py).

### A test suite that catches faults

The solution ships a small test suite of its own, written against the contract only (it imports `tinyllm.autograd.losses` and `tinyllm.autograd.tensor`, plus NumPy and pytest). Each test's expected value is computed by hand from the worked examples, never by running the code, which is what lets it catch a wrong implementation. Each test targets one plausible mistake:

| Test | What it shows | The mistake it catches |
|---|---|---|
| `test_hand_example_matches_section_3` | logits $[0, \ln 2, \ln 3]$, target 2: loss $\ln 2$, gradient $[1/6, 1/3, -1/2]$ | a wrong sign or a missing subtraction at the target |
| `test_ignored_rows_get_no_loss_and_no_gradient` | a row whose target is `ignore_index` adds no loss and gets a zero gradient, also with a custom `ignore_index` | padding that trains toward a fake target |
| `test_mean_divides_by_kept_rows` | two kept rows and one ignored row: the mean is $(\ln 2 + \ln 6)/2$, the sum is $\ln 2 + \ln 6$ | a mean that divides by $N$ |
| `test_all_ignored_batch_is_zero` | only padding gives loss exactly 0 and a zero gradient | $0/0 = $ `nan` |
| `test_label_smoothing_hand_value` | $\varepsilon = 0.3$ on $V = 3$: loss $0.1 \ln 6 + 0.1 \ln 3 + 0.8 \ln 2$, gradient $p - q$ | smoothing over $V - 1$ classes, or summing instead of averaging |
| `test_large_logits_are_finite` | logits $[10^4, 0, -10^4]$ with target 2 give exactly $2 \times 10^4$; bce at $\pm 1000$ gives 1000 | $\log(\mathrm{softmax})$ and $\log \sigma$ overflow |
| `test_out_of_range_target_raises` | targets 3 and $-1$ on $V = 3$ are a `ValueError` | trusting targets |
| `test_mse_and_bce_gradients_by_finite_differences` | both gradients match central differences; bce with `pos_weight` 2 at $x = 0$, $y = 1$ is $2 \ln 2$ | a closed form off by a factor, a misplaced `pos_weight` |

Two of them are worth reading in full:

```python
def test_mean_divides_by_kept_rows():
    """Two kept rows and one ignored row: the mean is the kept sum over 2."""
    z = leaf([ROW, ROW, ROW])                      # ROW = [0, ln 2, ln 3]
    loss = cross_entropy(z, np.array([2, 0, -100]))
    # target 2: -ln(1/2) = ln 2; target 0: -ln(1/6) = ln 6; padding: nothing
    np.testing.assert_allclose(loss.data, (LN2 + LN6) / 2, rtol=1e-12)
    total = cross_entropy(leaf([ROW, ROW, ROW]), np.array([2, 0, -100]), reduction="sum")
    np.testing.assert_allclose(total.data, LN2 + LN6, rtol=1e-12)

def test_large_logits_are_finite():
    """Logits of 1e4 and bce logits of 1000 give finite losses and gradients."""
    z = leaf([[1e4, 0.0, -1e4]])
    loss = cross_entropy(z, np.array([2]))
    assert np.isfinite(loss.data)
    # log p_2 = -1e4 - 1e4 - log(1 + e^-1e4 + e^-2e4) = -2e4 to double precision
    np.testing.assert_allclose(loss.data, 2e4, rtol=1e-12)
    b = leaf([-1000.0, 1000.0])
    lb = bce_with_logits(b, np.array([1.0, 0.0]))  # both maximally wrong: 1000 each
    np.testing.assert_allclose(lb.data, 1000.0, rtol=1e-12)
```

The second test asserts the exact value, not just finiteness: a loss that is finite but clipped (for example by adding a tiny constant inside the log) would pass `isfinite` and fail here. The full suite is [`test_losses.py`](../../../../course/solutions/learner-tests/L0.3/python/tests/l0-3-losses/test_losses.py).

### Why this design

Forward and backward are each $O(NV)$ with a constant number of passes over the logits. Memory between forward and backward is one `[N, V]` array (`lp`) plus the small target and mask arrays; the three-op composition would hold three. Composition is correct, but it pays for generality the loss does not need: three VJPs, three saved arrays, and a backward that multiplies a softmax Jacobian-vector product by a scatter. The fused form is one subtraction on probabilities the forward already computed.

Returning loss 0 for an all-padding batch, instead of torch's `nan`, is a deliberate departure. Matching torch here would mean one empty batch (which data loaders do produce at the end of an epoch or with aggressive filtering) poisons every weight. Defining $0/0$ as 0 with zero gradient makes such a step a no-op. Smoothing over $V$ rather than $V - 1$ is torch's choice too, and it keeps $q$ a simple mixture of the one-hot and the uniform distribution, so the gradient is $p - q$ with no special case.

Two tradeoffs are accepted. The `[N, V]` logits themselves are still materialized; fusing the output projection into the loss (Liger Kernel) is the next step and is left out for clarity. And `mse` and `bce_with_logits` support only the mean reduction, which is all the book's training loops use.

## What can go wrong

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| counting ignored rows in the mean, or giving them gradient | padded batches report a smaller loss; padding logits train toward a fake target | rows and gradients are multiplied by the keep mask, and the mean divides by `keep.sum()` |
| trusting targets | `-1` silently trains the last class (NumPy indexes from the end) | `_targets` range-checks every kept target |
| dividing by $K = 0$ | an all-padding batch gives `nan`, then every weight is `nan` | `rows.sum() / n if n else 0`, and the VJP scale is 0 when `n == 0` |
| $-\log(\mathrm{softmax}(z))$ and $-\log\sigma(x)$ | `inf` loss at logits of $10^3$ | `stable.log_softmax` and the softplus form with `log1p(exp(-abs(x)))` |
| smoothing over $V - 1$ classes, or summing $-\log p_j$ instead of averaging | disagrees with torch; the loss grows with the vocabulary size | `eps * lp.mean(axis=-1)` in the forward, `eps / v` on every class in the backward |

> **Warning:** A mean that divides by $N$ instead of $K$ does not crash or produce `nan`. It quietly scales the learning rate by the fraction of real rows, which varies from batch to batch.

## Where this leads

This section built on one `from_op` node per loss, the stable `log_softmax`, the independently derived `cross_entropy_vjp` that the fused gradient must equal, and the definition $H(q, p)$ of a smoothed row's loss.

Next, the bigram and the MLPs of [Training loop and the autograd bigram](06-training-loop-and-the-autograd-bigram.md) minimize `cross_entropy`, `mse`, and `bce_with_logits`. Packed batches in [Supervised fine-tuning](../18-post-training/01-supervised-fine-tuning.md) mark their padding with `ignore_index`. And every language-model training step in the book ends in `cross_entropy`: [mixed-precision training](../17-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md), [Bengio's neural probabilistic language model](../08-statistical-language-models/02-bengio-nplm.md), [RNN language model with stateful TBPTT](../09-recurrent-networks/06-rnn-language-model-stateful-tbptt.md), [Encoder-decoder with teacher forcing](../10-attention/01-encoder-decoder-teacher-forcing.md), [Encoder-decoder Transformer](../11-transformer/05-encoder-decoder-transformer.md), [GPT decoder-only](../12-objectives-and-adaptation/01-gpt-decoder-only.md), [BERT: the bidirectional encoder and masked language modeling](../12-objectives-and-adaptation/02-bert-masked-lm.md), [ELECTRA: replaced-token detection](../12-objectives-and-adaptation/03-electra-replaced-token-detection.md), and [Fine-tuning heads and the linear-head export](../12-objectives-and-adaptation/05-fine-tuning-heads-and-the-linear-head-export.md).

### Further reading

- Goodfellow, Bengio, Courville, *Deep Learning* (free online), sections 6.2.2 and 7.5.
- Szegedy et al., "Rethinking the Inception Architecture" (2016), section 7 (label smoothing).
- Liger Kernel `fused_linear_cross_entropy` (`liger_kernel/ops/fused_linear_cross_entropy.py`): fuses the output projection too, so the `[N, V]` logits are never materialized, chunked over rows.
- PyTorch `nll_loss` (`aten/src/ATen/native/LossNLL.cpp`): the same `ignore_index` semantics, plus per-class weights.
- `torch.nn.CrossEntropyLoss(label_smoothing=...)` (`aten/src/ATen/native/Loss.cpp`): label smoothing combined with class weights and probability targets.
- `torch.nn.functional.binary_cross_entropy_with_logits` (`aten/src/ATen/native/Loss.cpp`): the same softplus form, vectorized on GPU.

## Key terms

- **fused loss**: a loss implemented as one graph node with a closed-form gradient, instead of a chain of ops.
- **`ignore_index`**: a target value marking a row (usually padding) that contributes no loss, no gradient, and no count to the mean.
- **label smoothing**: mixing the one-hot target with a uniform distribution over all $V$ classes, weight $\varepsilon$.
- **softplus**: $\log(1 + e^x)$, used to write binary cross-entropy without positive exponents.
- **`pos_weight`**: a multiplier on the positive-class term of binary cross-entropy.

## Summary

- Softmax cross-entropy is one graph node: its forward uses `log_softmax`, and its gradient is the closed form $\mathrm{softmax}(z) - q$.
- Padding rows (`ignore_index`) add no loss, get no gradient, and do not count in the mean; a batch of only padding is loss 0, not `nan`.
- Label smoothing spreads $\varepsilon/V$ over **every** class, the target included, so a row's loss is the cross-entropy $H(q, p)$ against the smoothed target.
- Stable forms keep logits of $10^3$ finite: `log_softmax` for cross-entropy, softplus for binary cross-entropy.
- A good test suite for a loss pins hand-computed values, the padding rules, and the stability limits.

## Review questions

1. Why is the gradient of smoothed cross-entropy still $p - q$, with no extra term from the smoothing?

   <details><summary>Answer</summary>

   With any target distribution $q$ that sums to 1, $\ell = -\sum_j q_j \log p_j = -\sum_j q_j z_j + \log \sum_k e^{z_k}$, because $\sum_j q_j \log\sum_k e^{z_k} = \log\sum_k e^{z_k}$. Differentiating gives $-q_j + p_j$. Smoothing only changes $q$, and $q$ still sums to 1, so the form is unchanged.

   </details>

2. Why does `_targets` replace ignored targets with 0 instead of leaving them as `-100`?

   <details><summary>Answer</summary>

   Both the forward (`take_along_axis`) and the backward (`put_along_axis`) index with the targets. `-100` would read class $V - 100$ for $V \ge 100$ and raise for smaller $V$. Class 0 is always a valid index, and the keep mask then zeroes the loss and the gradient of those rows, so the value read there never matters.

   </details>

3. A batch has 3 real rows and 5 padding rows. What goes wrong if the mean divides by 8?

   <details><summary>Answer</summary>

   The reported loss is $3/8$ of the true mean, and the gradient is scaled by $1/8$ instead of $1/3$, so the batch trains at $3/8$ of the intended learning rate. Worse, the factor changes with the amount of padding, so loss curves are not comparable across batches.

   </details>

4. Why compute $\sigma(-x)$ as `exp(-logaddexp(0, x))` rather than `1 / (1 + exp(x))`?

   <details><summary>Answer</summary>

   At $x = 1000$, `exp(x)` overflows to `inf` with a warning; `1 / inf = 0` is actually the right answer, but at $x = -1000$ the other common form `exp(-x) / (1 + exp(-x))` gives `inf / inf = nan`. `logaddexp(0, x)` is computed stably for either sign, and $e^{-\mathrm{softplus}(x)}$ is then a number in $(0, 1)$ without any intermediate overflow.

   </details>

5. Why does `test_large_logits_are_finite` assert the exact value $2 \times 10^4$ rather than only `np.isfinite`?

   <details><summary>Answer</summary>

   An implementation that hides overflow by clipping (for example adding a tiny constant inside the log) produces a finite but wrong loss. Asserting the exact value catches that, while `isfinite` alone would pass it.

   </details>
