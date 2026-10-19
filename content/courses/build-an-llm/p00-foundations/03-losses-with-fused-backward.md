# Losses with a fused backward

## Overview

| | |
|---|---|
| **Module** | `L0.3` · build · Python · Pass 2 · about 2 h reading |
| **Solution** | [`losses.py`](../../../../course/solutions/python/tinyllm/autograd/losses.py): `cross_entropy` (ignore_index, label smoothing, three reductions), `mse`, `bce_with_logits` (pos_weight); a model test suite, [`test_losses.py`](../../../../course/solutions/learner-tests/L0.3/python/tests/l0-3-losses/test_losses.py) |
| **Contract** | [`course/contracts/py/tinyllm/autograd/losses.pyi`](../../../../course/contracts/py/tinyllm/autograd/losses.pyi) |
| **Needs** | `L0.1` `from_op` · `M09.2` stable `log_softmax` · `M08.3` `cross_entropy_vjp` (an independent derivation of the gradient) · `M11.1` cross-entropy $H(q, p)$ (the definition, computed the slow way) · reading: [`craft.03`](../../../tracks/software-craftsmanship/03-testing-mentality/01-tdd-unit-tests-and-mutation-grading.md) on writing tests that catch faults |
| **Used by** | `L0.5` every training step of the course computes one of these; the Pass 2 bigram and the digits MLP train on `cross_entropy` · later: `L11.1`, `L12.1`, `L2.2`, `L3.6`, `L4.1`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5` |
| **Optional depth** | Goodfellow, Bengio, Courville, *Deep Learning* (free online), sections 6.2.2 and 7.5; Szegedy et al., "Rethinking the Inception Architecture" (2016), section 7 (label smoothing) |

## Key Takeaways

- Softmax cross-entropy is one graph node: its forward uses `log_softmax`, and its gradient is the closed form $\mathrm{softmax}(z) - q$.
- Padding rows (`ignore_index`) add no loss, get no gradient, and do not count in the mean; a batch of only padding is loss 0, not `nan`.
- Label smoothing spreads $\varepsilon/V$ over **every** class, the target included, so a row's loss is the cross-entropy $H(q, p)$ against the smoothed target.
- Stable forms keep logits of $10^3$ finite: `log_softmax` for cross-entropy, softplus for binary cross-entropy.
- A good test suite for a loss pins hand-computed values, the padding rules, and the stability limits; section 5 shows one that does.

## How to read this chapter

Sections 1 to 3 derive the fused gradients and the padding rules and work two rows by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through `losses.py` line by line, then the model test suite and why each test is there. Try to write each VJP from section 2 before reading the code.

---

## 1. Why now

`L0.2` provides `log_softmax` and `gather`, so cross-entropy could be written as three ops: `-gather(log_softmax(z), t).mean()`. It would be correct and slow, and it would be fragile. Backward through that chain keeps a `[N, V]` array alive at each step, and for the bigram (`V = 256`) or a real vocabulary (`V = 49,152` for SmolLM2) that is most of the memory a training step uses. Padding makes it worse: batches of token windows have rows that must not count, and a mean that divides by the padded length silently shrinks the loss. Every training loop from here on (`L0.5`, `L2.2`, `L6.1`, the capstone) calls this file once per step, so the loss must be one node with a closed-form gradient, correct about padding, and finite for any logits.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $z_i \in \mathbb{R}^V$ | logits of row $i$ ($N$ rows, $V$ classes) | `float[N, V]` (or `[..., V]`) |
| $t_i$ | target class of row $i$, or `ignore_index` | `int[N]` |
| $p_i = \mathrm{softmax}(z_i)$ | predicted distribution | `float[V]` |
| $k_i \in \{0, 1\}$ | 1 when row $i$ is kept ($t_i \ne$ `ignore_index`) | |
| $K = \sum_i k_i$ | number of kept rows | int |
| $\varepsilon$ | label smoothing, in $[0, 1]$ | float |
| $q_i = (1 - \varepsilon)\,e_{t_i} + \varepsilon/V$ | smoothed target, $e_t$ the one-hot vector | `float[V]` |
| $\ell_i = -\sum_j q_{ij} \log p_{ij}$ | row loss, $H(q_i, p_i)$ (`M11.1`) | float |
| $x, y$ | logit and target of binary cross-entropy, $y \in [0, 1]$ | float |
| $w$ | `pos_weight`, the weight on positive examples | float, default 1 |
| $\sigma(x) = 1/(1 + e^{-x})$, $\mathrm{softplus}(x) = \log(1 + e^x)$ | | functions |

**The fused gradient.** For one row with a one-hot target, $\ell = -\log p_t = -z_t + \log\sum_j e^{z_j}$. Differentiate: $\partial\ell/\partial z_j = -[j = t] + e^{z_j}/\sum_k e^{z_k} = p_j - [j = t]$. With a smoothed target, $\ell = -\sum_j q_j z_j + \log\sum_k e^{z_k}$ (because $\sum_j q_j = 1$), so $\partial\ell/\partial z = p - q$. That is the whole backward: one subtraction on the probabilities, which the forward already computed as $e^{\log p}$. `M08.3` derived it; a correct fused VJP equals `M08.3`'s `cross_entropy_vjp` on the same inputs.

**The forward is stable.** $\log p = \mathrm{log\_softmax}(z) = z - \max z - \log\sum_j e^{z_j - \max z}$ (`M09.2`). Computing $-\log(\mathrm{softmax}(z))$ instead underflows: at $z = [0, 10^3]$ softmax returns exactly $[0, 1]$ in float64 and $\log 0 = -\infty$.

**Reductions and padding.** The kept mask $k$ zeroes ignored rows in both directions: their loss is 0 and their gradient row is 0. `"none"` returns the per-row losses ($\ell_i k_i$); `"sum"` is $\sum_i k_i \ell_i$; `"mean"` is that sum divided by $K$, the number of **kept** rows. Dividing by $N$ would make a batch of 3 real rows and 5 padding rows report 3/8 of its true loss and train at 3/8 of the learning rate. When $K = 0$ the mean is $0/0$; the course defines it as loss 0 with a zero gradient (torch returns `nan`, and one `nan` step poisons every weight). The upstream gradient $\bar\ell$ scales the rows: by $\bar\ell/K$ for the mean, by $\bar\ell$ for the sum, row by row for `"none"`.

**Label smoothing.** A target of exactly one class asks the model to push $p_t$ to 1, which needs $z_t - z_j \to \infty$. Smoothing moves $\varepsilon$ of the target mass onto a uniform distribution over all $V$ classes, so $q_{t} = 1 - \varepsilon + \varepsilon/V$ and every other class gets $\varepsilon/V$. The contract writes the same loss as $(1 - \varepsilon)(-\log p_t) + \varepsilon\,\mathrm{mean}_j(-\log p_j)$. Spreading over $V - 1$ classes (excluding the target) is a different, also published, variant; torch and this course use $V$.

**Mean squared error.** $\mathrm{mse} = \frac{1}{n}\sum (\hat{y} - y)^2$ over all $n$ elements; the gradient is $\frac{2}{n}(\hat{y} - y)$.

**Binary cross-entropy on logits.** For one logit $x$ and target $y$: $-[w y \log\sigma(x) + (1 - y)\log(1 - \sigma(x))]$. Using $-\log\sigma(x) = \mathrm{softplus}(-x)$ and $-\log(1 - \sigma(x)) = x + \mathrm{softplus}(-x)$, this is

$$(1 - y)\,x + \big(1 + (w - 1)y\big)\,\mathrm{softplus}(-x),$$

and $\mathrm{softplus}(-x) = \max(-x, 0) + \log(1 + e^{-|x|})$ never exponentiates a positive number. The gradient is $(1 - y) - (1 + (w - 1)y)\,\sigma(-x)$, divided by the element count for the mean.

## 3. Worked example by hand

**Cross-entropy with a padding row.** Logits for two rows over $V = 3$ classes: row 0 is $z_0 = [0, \ln 2, \ln 3]$ with target 2; row 1 has target `ignore_index` (-100), whatever its logits.

1. $e^{z_0} = [1, 2, 3]$, sum 6, so $p_0 = [1/6, 1/3, 1/2]$.
2. $\ell_0 = -\log p_{0,2} = -\log(1/2) = \ln 2 \approx 0.6931$.
3. Row 1 is ignored: $k = [1, 0]$, $K = 1$, so the mean loss is $\ell_0 / 1 = \ln 2$.
4. Gradient of row 0: $p_0 - e_2 = [1/6, 1/3, -1/2]$, divided by $K = 1$. Row 1: zeros.

**Label smoothing.** Same row 0, $\varepsilon = 0.3$: each class gets $0.3/3 = 0.1$ and the target keeps $0.7$ more, so $q = [0.1, 0.1, 0.8]$.

1. $\ell = -(0.1 \log\tfrac16 + 0.1 \log\tfrac13 + 0.8 \log\tfrac12) = 0.1\ln 6 + 0.1\ln 3 + 0.8\ln 2 \approx 0.1792 + 0.1099 + 0.5545 = 0.8436$.
2. Gradient $p - q = [\tfrac16 - 0.1, \tfrac13 - 0.1, \tfrac12 - 0.8] = [\tfrac{1}{15}, \tfrac{7}{30}, -\tfrac{3}{10}]$.

Both gradients sum to zero, as every softmax gradient must.

These are the first two cases in section 4, and section 5 traces both through the code.

## 4. The interface

```python
# python/tinyllm/autograd/losses.py
def cross_entropy(logits: Tensor, targets: ArrayLike, ignore_index: int = -100,
                  label_smoothing: float = 0.0, reduction: Literal["mean", "sum", "none"] = "mean") -> Tensor
def mse(pred: Tensor, target: ArrayLike) -> Tensor
def bce_with_logits(logits: Tensor, targets: ArrayLike, pos_weight: Optional[ArrayLike] = None) -> Tensor
```

Each returns one `from_op` node whose VJP is the closed form above. Targets are validated before anything else: integers, shape `logits.shape[:-1]`, each in $[0, V)$ or equal to `ignore_index`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand-example cross-entropy | unit | section 3: loss $\ln 2$, gradient $[1/6, 1/3, -1/2]$, zeros on the ignored row | the code and the math agree on the definition |
| hand-example label smoothing | unit | section 3: $q = [0.1, 0.1, 0.8]$, loss and $p - q$ | the smoothing every LM config can turn on |
| agreement with torch | golden | loss and gradient equal `torch.nn.functional` on every option | training loops ported from torch |
| central differences | gradcheck | the fused VJPs against central differences | a closed form off by a factor fails |
| matches `M08.3`'s `cross_entropy_vjp` | differential | the fused gradient equals `M08.3`'s matrix | the derivation and the code agree |
| rows match `M11.1`'s cross-entropy | differential | each smoothed row loss is `M11.1`'s $H(q, p)$ | the definition, computed the slow way |
| ignored rows excluded | property | appending ignored rows changes neither the loss nor the kept gradients | padded batches (`L0.6` windows, `L12.1` packing) |
| all rows ignored | boundary | only padding gives loss 0 and a zero gradient | one `nan` step ruins a run |
| large logits stay finite | boundary | logits of $10^4$ and bce logits of 1000 stay finite | trained logits are large |
| reductions agree | property | `none`, `sum`, `mean` agree with each other | evaluation sums `none` (`L6.7`) |
| float32 stays float32 | boundary | float32 logits give float32 loss and gradient | in-place float32 updates |
| bad inputs rejected | boundary | out-of-range or float targets, unknown reduction, bad $\varepsilon$, shape mismatches | data and config bugs fail here |

## 5. The solution, step by step

### From first principles

1. **One node per loss.** The loss is the last op of every training step and its input is the largest array in the graph (`[N, V]` logits). Composing `log_softmax`, `gather`, and `mean` would keep an `[N, V]` intermediate per op alive until backward. A single `from_op` node keeps only `log_softmax(z)` and recomputes $p = e^{\log p}$ in backward.
2. **The gradient is $p - q$, scaled.** Section 2 derived $\partial \ell_i / \partial z_i = p_i - q_i$ for any target distribution $q_i$ that sums to 1. So backward is: exponentiate the stored log-probabilities, subtract $\varepsilon / V$ everywhere, subtract $1 - \varepsilon$ more at the target, then apply the keep mask and the reduction's scale.
3. **Validate, then neutralize ignored targets.** A target of `-100` cannot be used as an index (numpy would read class $V - 100$, or fail). So the target array is validated over kept rows only, ignored entries are replaced by 0 for indexing, and the keep mask zeroes their loss and gradient afterwards. This way the indexing code never branches per row.
4. **The mean divides by kept rows, and $K = 0$ is defined.** The mean's denominator is $K$, and the case $K = 0$ returns 0 in both directions instead of $0/0$.
5. **Stable forms everywhere.** Cross-entropy reads $\log p$ from a stable `log_softmax`. Binary cross-entropy is rewritten with softplus so that no exponent is ever positive, and its gradient uses a stable $\sigma(-x)$.
6. **Keep the input's dtype.** Every output and gradient is cast back to the logits' dtype, so float32 training stays float32.

### The code, annotated

**Target validation.**

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

**Cross-entropy forward.**

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

`lp.mean(axis=-1)` is the average of $\log p_j$ over all $V$ classes, which is what makes the smoothing spread $\varepsilon / V$ over every class, the target included. On the section 3 row with $\varepsilon = 0.3$: $0.7 \ln 2 + 0.3 \cdot \tfrac13(\ln 6 + \ln 3 + \ln 2) = 0.4852 + 0.3584 = 0.8436$, the hand value.

**Cross-entropy backward.**

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

On section 3's row: `exp(lp) = [1/6, 1/3, 1/2]`; without smoothing the target entry becomes $1/2 - 1 = -1/2$, giving $[1/6, 1/3, -1/2]$. With $\varepsilon = 0.3$: subtract $0.1$ everywhere, then $0.7$ more at class 2, giving $[1/15, 7/30, -3/10]$. The ignored row in the first example was indexed at class 0 (by `_targets`), and the keep mask erases whatever that produced.

**Mean squared error.**

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

**Binary cross-entropy on logits.**

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

`np.logaddexp(0, x)` is $\log(1 + e^x)$ computed without overflow, so $\sigma(-x) = e^{-\mathrm{softplus}(x)}$ is finite for $x = \pm 1000$. Check the `pos_weight` hand value: at $x = 0$, $y = 1$, $w = 2$: $c = 2$, $\mathrm{softplus}(0) = \ln 2$, so the loss is $0 + 2 \ln 2$.

Full solution: [`losses.py`](../../../../course/solutions/python/tinyllm/autograd/losses.py).

### The model test suite

The solution ships a small test suite of its own, written against the contract only (it imports `tinyllm.autograd.losses` and `tinyllm.autograd.tensor`, plus numpy and pytest). Each test's expected value is computed by hand from section 3, never by running the code, which is what lets it catch a wrong implementation. Each test targets one plausible mistake:

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

Two of them, quoted:

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

The second test asserts the exact value, not just finiteness: a loss that is finite but clipped (for example by adding a tiny constant inside the log) would pass `isfinite` and fail here.

Full suite: [`test_losses.py`](../../../../course/solutions/learner-tests/L0.3/python/tests/l0-3-losses/test_losses.py).

### Why this solution

- **Cost.** Forward and backward are each $O(NV)$ with a constant number of passes over the logits. Memory between forward and backward is one `[N, V]` array (`lp`) plus the small target and mask arrays; the three-op composition would hold three.
- **A fused node over composition.** Composition is correct, but it pays for generality the loss does not need: three VJPs, three saved arrays, and a backward that multiplies a softmax Jacobian-vector product by a scatter. The fused form is one subtraction on probabilities the forward already computed.
- **Loss 0 for an all-padding batch over torch's `nan`.** Matching torch here would mean one empty batch (which data loaders do produce at the end of an epoch or with aggressive filtering) poisons every weight. Defining $0/0$ as 0 with zero gradient makes such a step a no-op.
- **Smoothing over $V$, not $V - 1$.** Both variants are published; $V$ is torch's and keeps $q$ a simple mixture of the one-hot and the uniform distribution, so the gradient is $p - q$ with no special case.
- **Tradeoffs accepted.** The `[N, V]` logits themselves are still materialized; fusing the output projection into the loss (Liger Kernel) is the next step and is left out for clarity. `mse` and `bce_with_logits` only support the mean reduction, which is all the course's training loops use.

### Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. counting ignored rows in the mean, or giving them gradient | padded batches report a smaller loss; padding logits train toward a fake target | rows and gradients are multiplied by the keep mask, and the mean divides by `keep.sum()` |
| 2. trusting targets | `-1` silently trains the last class (numpy indexes from the end) | `_targets` range-checks every kept target |
| 3. dividing by $K = 0$ | an all-padding batch gives `nan`, then every weight is `nan` | `rows.sum() / n if n else 0`, and the VJP scale is 0 when `n == 0` |
| 4. $-\log(\mathrm{softmax}(z))$ and $-\log\sigma(x)$ | `inf` loss at logits of $10^3$ | `stable.log_softmax` and the softplus form with `log1p(exp(-abs(x)))` |
| 5. smoothing over $V - 1$ classes, or summing $-\log p_j$ instead of averaging | disagrees with torch; the loss grows with the vocabulary size | `eps * lp.mean(axis=-1)` in the forward, `eps / v` on every class in the backward |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | each loss is one `from_op` node |
| Back | `M09.2` | `log_softmax` for the stable forward |
| Back | `M08.3` | `cross_entropy_vjp`, the independent derivation the fused gradient must equal |
| Back | `M11.1` | $H(q, p)$, the definition of a smoothed row's loss |
| Forward | `L0.5` | the bigram and the MLPs of the training loop minimize `cross_entropy`, `mse`, and `bce_with_logits` |
| Forward | `L12.1` | packed batches mark their padding with `ignore_index` |
| Forward | `L11.1`, `L2.2`, `L3.6`, `L4.1`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5` | every language-model training step ends in `cross_entropy` |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| fused `cross_entropy` | Liger Kernel `fused_linear_cross_entropy` | fuses the output projection too: the `[N, V]` logits are never materialized, chunked over rows | `liger_kernel/ops/fused_linear_cross_entropy.py` |
| `ignore_index` | PyTorch `nll_loss` | the same semantics, plus per-class weights | `aten/src/ATen/native/LossNLL.cpp` |
| label smoothing | `torch.nn.CrossEntropyLoss(label_smoothing=...)` | combined with class weights and probability targets | `aten/src/ATen/native/Loss.cpp` |
| `bce_with_logits` | `torch.nn.functional.binary_cross_entropy_with_logits` | the same softplus form, vectorized on GPU | `aten/src/ATen/native/Loss.cpp` |
