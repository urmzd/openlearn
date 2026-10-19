# The op library and gradcheck_all

## Overview

| | |
|---|---|
| **Module** | `L0.2` · build · Python · Pass 2 · about 3 h reading |
| **Solution** | [`functional.py`](../../../../course/solutions/python/tinyllm/autograd/functional.py), imported as `F`: 25 ops (activations, reductions, shape, selection, the softmax family, dropout, matmul) and `gradcheck_all()`; the `gradcheck --suite all` verb in [`entry/python/tinyllm/cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py) |
| **Contract** | [`course/contracts/py/tinyllm/autograd/functional.pyi`](../../../../course/contracts/py/tinyllm/autograd/functional.pyi) |
| **Needs** | `L0.1` `Tensor` and `from_op` · `M01.3` activations and their derivatives · `M09.2` stable softmax, log-softmax, logsumexp · `M08.3` the softmax VJPs and `unbroadcast` · `M08.1` dual numbers · `M04.1` `gradcheck` · `M06.3` PCG32 |
| **Used by** | `L0.4` layers are written in `F` · `L0.5` `BigramLogits` is an `F.embedding` · `L0.6` the checkpoint tests compute their loss with `F` · Pass 2's end-to-end run starts with `{tinyllm} gradcheck --suite all` · later: every model module (see section 7) |
| **Optional depth** | Griewank and Walther, *Evaluating Derivatives* (2nd ed.), ch. 3 and 4; the PyTorch `derivatives.yaml` file, read as a table of VJPs |

## Key Takeaways

- Every op is one numpy forward plus one closed-form VJP joined by `from_op`; nothing in the library calls another op's backward.
- Reductions put the reduced axes back before broadcasting the gradient, a maximum shares its gradient among ties, and a mean divides by the reduced count only.
- Selection ops route gradient by addition: a token or index used twice gets both gradients, and an out-of-range id is an error where it enters.
- The softmax family's VJPs are computed from the stable outputs, so logits of 1000 and fully masked rows give finite gradients.
- `gradcheck_all` checks every op against central differences with a random output weighting, plus forward-mode duals for the elementwise ops, and it fails on a planted 0.1% or 1e-7 error.

## How to read this chapter

Sections 1 to 3 derive each family of VJPs and work three of them by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through the solution family by family, then `gradcheck_all`, the check that proves the rest; try to write each VJP down before reading it.

The system's CLI exposes the library through its first Pass 2 verb, `{tinyllm} gradcheck --suite all` ([`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md)): it calls `F.gradcheck_all(rtol=1e-5)`, prints one line per op, and ends with `{"suite": "all", "checks": N, "failed": K, "max_rel_err": E, "worst": "<op>"}`; it exits 0 only when `failed` is 0.

---

## 1. Why now

`L0.1` built a `Tensor` that knows `+ - * / @ **` and indexing. A model needs more: a nonlinearity between layers, a softmax over the vocabulary, an embedding lookup for token ids, a mean for LayerNorm, a dropout mask. Written inline in each model, each of those is a chance to get a VJP wrong silently, because a wrong gradient still trains, just badly. This module puts every op in one library with one VJP each, checks each against torch and against numerical differentiation, and gives the system's CLI the `gradcheck` verb that Pass 2's end-to-end run starts with. From here on, model code (`L0.4`, `L2`, `L3`, `L5`) composes these ops and never writes a backward pass.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | an op's input | `Tensor`, any shape |
| $y = f(x)$ | its output | `Tensor` |
| $\bar{y}$, $\bar{x}$ | upstream gradient and the gradient handed back (`L0.1`) | shapes of $y$, $x$ |
| $A$ | the reduced axes of a reduction | tuple of ints |
| $n = \prod_{a \in A} s_a$ | number of elements reduced into each output | int |
| $\mathrm{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$ | along one axis | same shape as $x$ |
| $\langle u, v \rangle = \sum_i u_i v_i$ | inner product along the softmax axis | |
| $\phi$, $\Phi$ | standard normal density and CDF (`gelu`) | functions |
| $p$ | dropout probability | float in $[0, 1]$ |
| $u_k$ | the $k$-th uniform draw of a PCG32 (`M06.3`) | float in $[0, 1)$ |
| $w$ | random output weights inside `gradcheck_all` | shape of $y$ |
| $\epsilon$ | central-difference step, $10^{-6}$ (`M04.1`) | float |

**Elementwise ops.** For $y = f(x)$ applied entry by entry, $\bar{x} = \bar{y} \odot f'(x)$. The derivatives are `M01.3`'s: $\exp' = \exp$ (reuse the output), $\log' = 1/x$, $\tanh' = 1 - \tanh^2$, $\sigma' = \sigma(1 - \sigma)$, $\mathrm{relu}'(x) = [x > 0]$ (0 at the kink, as torch), $\mathrm{silu}(x) = x\sigma(x)$, and two `gelu` forms: exact $x\Phi(x)$ and the tanh approximation $\tfrac{x}{2}(1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$. `approximate="none"` or `"tanh"` picks one; anything else is an error, because HF configs name exactly these two.

**Reductions put axes back.** `sum(x, axis=A)` adds every input element into one output with weight 1, so $\bar{x}$ is $\bar{y}$ copied back over $A$. Without `keepdims` the output lost those axes; `np.expand_dims(g, A)` restores them as size 1, and broadcasting copies. `mean` is `sum` times $1/n$ with $n$ the size of the **reduced** axes only. `max` routes the gradient to the entries equal to the maximum; with ties torch (`amax`) splits it equally, so the shares still add up to $\bar{y}$. `var` with `correction` $c$ is $\sum (x - \mu)^2 / (n - c)$: $c = 0$ is the population variance LayerNorm uses, $c = 1$ the sample variance. Its gradient is $\bar{y} \cdot 2(x_i - \mu)/(n - c)$; the terms through $\mu$ vanish because $\sum_i (x_i - \mu) = 0$.

**Shape ops move gradients back.** `reshape` reshapes $\bar{y}$ to the input shape. `transpose(a, b)` is its own inverse. `permute(dims)` is undone by the inverse permutation, `argsort(dims)`, not by applying `dims` again. `concat` splits $\bar{y}$ at the **cumulative** sizes of its inputs, and `stack` takes slice $i$ along the stacking axis for input $i$.

**Selection ops add.** `where(cond, a, b)` sends $\bar{y}$ to $a$ where `cond` holds and to $b$ elsewhere, then unbroadcasts each. `masked_fill(x, mask, v)` sends nothing to the filled positions: an attention mask must not train the scores it hid. `gather(x, idx, axis)` (torch's) and `embedding(W, ids)` read entries, possibly the same one twice, so their VJPs are scatter-adds (`np.add.at`). Both check their indices: numpy would read `-1` as the last row and never complain.

**The softmax family, stably.** Forward values come from `M09.2` (subtract the max first). The VJPs come from `M08.3` and use the *outputs*:

$$\bar{x} = y \odot (\bar{y} - \langle \bar{y}, y \rangle) \quad (\mathrm{softmax}), \qquad \bar{x} = \bar{y} - \mathrm{softmax}(x)\,\textstyle\sum_i \bar{y}_i \quad (\mathrm{log\_softmax}),$$

and for $\ell = \mathrm{logsumexp}(x)$, $\bar{x} = \bar{\ell}\, \mathrm{softmax}(x)$. None of these ever computes $e^{x}$ of a raw logit, so a logit of 1000 is as safe as one of 1. A row masked entirely to $-\infty$ has softmax zeros (`M09.2`'s rule) and therefore a zero gradient.

**Inverted dropout.** During training each element is kept with probability $1 - p$ and scaled by $1/(1 - p)$, so the expected output equals the input and evaluation needs no rescaling: in eval mode, or with $p = 0$, dropout returns its input object and draws nothing. The mask uses one uniform per element in C order, $\text{keep}_k = [u_k \ge p]$, from the PCG32 the caller passes in. The draw count is fixed (exactly `x.size` draws per call), which is what lets a resumed run (`L0.6`) reproduce the same masks. $p = 1$ drops everything (no division by zero).

**gradcheck_all: a test that can fail.** For each op, `gradcheck_all` draws float64 inputs from `PCG32(0)`, keeps them away from kinks (relu at 0, max ties), and compares the analytic gradient of $f(x) = \sum y \odot w$ with central differences $\big(f(x + \epsilon e_i) - f(x - \epsilon e_i)\big)/2\epsilon$ through `M04.1`'s `gradcheck` at `rtol`. The weights $w$ are random on purpose: $\sum_i \mathrm{softmax}(x)_i = 1$ for every $x$, so the gradient of the plain sum is zero and *any* softmax VJP, right or wrong, would pass. Central differences resolve relative errors near $10^{-5}$; a smaller error in an elementwise derivative is caught by a second check, forward-mode dual numbers (`M08.1`), which are exact to rounding and flag a gap above $10^{-9}$. Ops are looked up in the module when the check runs, so a monkeypatched op is the one checked.

## 3. Worked example by hand

**Softmax and its VJP.** Take $x = [0, \ln 2, \ln 3]$. Then $e^{x} = [1, 2, 3]$, the sum is 6, and $y = [1/6, 2/6, 3/6]$. With the upstream gradient $\bar{y} = [1, 0, 0]$ (the loss looks only at the first output):

1. $\langle \bar{y}, y \rangle = 1 \cdot 1/6 = 1/6$.
2. $\bar{y} - 1/6 = [5/6, -1/6, -1/6]$.
3. Multiply by $y$: $\bar{x} = [5/36, -2/36, -3/36]$.

The entries sum to zero: raising every logit by the same amount leaves softmax unchanged, so no direction along $(1, 1, 1)$ can change the loss. Check $\bar{x}_0$ by perturbation: $y_0 = e^{x_0}/(e^{x_0} + 5)$, whose derivative at $x_0 = 0$ is $5/36$.

**A tie in max.** $\max([1, 3, 3]) = 3$ is reached twice. torch's `amax` gives each maximal entry half: $\bar{x} = [0, 0.5, 0.5]$ for $\bar{y} = 1$.

**Variance.** For $[1, 2, 3, 4]$: $\mu = 2.5$, squared deviations $2.25 + 0.25 + 0.25 + 2.25 = 5$, so `var` is $5/4 = 1.25$ with correction 0 and $5/3$ with correction 1.

These are the first cases in section 4, and section 5 traces each through the code.

## 4. The interface

```python
# python/tinyllm/autograd/functional.py   (from tinyllm.autograd import functional as F)
exp, log, tanh, sigmoid, relu, silu; gelu(x, approximate="none" | "tanh")
sum(x, axis=None, keepdims=False); mean(...); max(...); var(x, axis=None, keepdims=False, correction=0)
reshape(x, shape); transpose(x, a, b); permute(x, dims); concat(xs, axis=0); stack(xs, axis=0)
where(cond, a, b); gather(x, idx, axis); embedding(weight, ids); masked_fill(x, mask, value)
softmax(x, axis=-1); log_softmax(x, axis=-1); logsumexp(x, axis=-1, keepdims=False)
dropout(x, p, training, rng); matmul(a, b)
def gradcheck_all(rtol: float = 1e-5) -> dict[str, GradcheckReport]
```


`sum`, `max`, and `mean` shadow Python builtins inside the module, so the solution calls `builtins.max` where it needs the builtin. Float32 inputs give float32 outputs and gradients: derivative arrays and masks are cast to the input's dtype.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand-example softmax backward | unit | section 3: $y = [1/6, 2/6, 3/6]$, $\bar{x} = [5/36, -2/36, -3/36]$ | the code and the math agree on the softmax VJP |
| max ties share the gradient | boundary | `max([1, 3, 3])` gives $[0, 0.5, 0.5]$ | ReLU-max pooling and torch parity |
| agreement with torch | golden | value and gradients equal torch 2.14 on 45 cases | models ported from torch (`L5` to `L7`) |
| every op against central differences | gradcheck | every VJP against central differences, float64 | a wrong VJP trains badly, not visibly |
| float32 in, float32 out | boundary | float32 in, float32 out for twelve ops, and a float32 gradient through dropout | Python agrees with the float32 C kernels (`L9`) |
| relu derivative at zero | boundary | $\mathrm{relu}'(0) = 0$ | torch's convention |
| repeated embedding ids accumulate | boundary | a repeated token gets both rows of gradient | every embedding table |
| repeated gather indices accumulate | boundary | an index read twice gets gradient 2 | the cross-entropy gather in `L0.3` |
| embedding rejects bad ids | boundary | `-1`, `n`, and float ids are a `ValueError` | tokenizer bugs fail where they enter |
| dropout in eval mode | unit | eval mode returns the input object, draws nothing | evaluation is deterministic |
| dropout mask and scale | statistical | about 75% kept at $p = 0.25$, survivors scaled by $4/3$ | the expected output is the input |
| dropout draw accounting | unit | the mask is $u \ge p$ for the same-seed uniforms, exactly `x.size` draws | a resumed run replays the masks (`L0.6`) |
| dropout bounds on $p$ | boundary | $p = 1$ gives zeros; $p \notin [0, 1]$ is an error | config errors fail early |
| a fully masked softmax row | boundary | an all-masked row gives zeros and a zero gradient | padded rows in attention (`L5.2`) |
| logsumexp without overflow | boundary | `logsumexp([1000, 1000])` and its gradient $[0.5, 0.5]$ | trained logits are large |
| variance correction | unit | 1.25 and $5/3$; $n \le c$ is an error | LayerNorm's population variance (`L0.4`) |
| `where` and `masked_fill` routing | unit | gradients go to the chosen operand; filled positions get none | attention masks (`L5`) |
| unknown gelu approximation | boundary | only `"none"` and `"tanh"` | HF configs name exactly these |
| no graph under `no_grad` | property | every op returns a constant under `no_grad` | evaluation memory |
| `gradcheck_all` reports every op | unit | one ok report per op, 33 names | `gradcheck --suite all` |
| `gradcheck_all` catches a wrong op | boundary | a 0.1% softmax error and a $10^{-7}$ sigmoid error are both flagged | a checker that always passes is worse than none |

## 5. The solution, step by step

### From first principles

1. **One op, one forward, one VJP, one `from_op` call.** `L0.1` already handles graph bookkeeping, dtype casting of gradients, and shape checks. So an op only has to compute its forward with numpy and return a closure that maps $\bar{y}$ to $\bar{x}$. No op calls another op's backward, which keeps every VJP independently checkable.
2. **Elementwise ops share one pattern.** $\bar{x} = \bar{y} \odot f'(x)$, so a helper that takes `f` and `df` (from `M01.3`) covers tanh, sigmoid, relu, silu, and both gelus. `exp` and `log` are written out because `exp` reuses its output as its derivative and `log`'s derivative is a division.
3. **A reduction's VJP is "put the axes back, then broadcast".** Normalize the axis argument once (negative axes, tuples, `None`), so `np.expand_dims(g, axes)` can restore exactly the reduced axes as size 1. Then `sum` broadcasts, `max` multiplies by an equal-share mask, and `var` multiplies by $2(x - \mu)/(n - c)$.
4. **Shape ops invert themselves.** Each forward is a pure rearrangement, so the backward applies the inverse rearrangement to $\bar{y}$: reshape back, swap the same two axes, transpose by the inverse permutation, split at cumulative sizes, take slice $i$.
5. **Selection ops scatter with `np.add.at` and check their indices.** Any op that reads an entry more than once must add each contribution back. Index validation sits in one helper so `gather` and `embedding` reject `-1` the same way.
6. **The softmax family reuses the stable forward.** The VJPs from `M08.3` are written in terms of the output $y$, which the stable forward already produced, so backward never exponentiates a raw logit.
7. **Dropout's randomness is an explicit input.** Taking the generator as an argument and drawing exactly `x.size` uniforms makes the mask a pure function of the generator state, which is what makes training resumable bit for bit.
8. **The checker must be able to fail.** A random output weighting defeats the softmax blind spot, kinks are pushed out of reach, and a second, exact check (duals) covers errors below what central differences resolve. Looking ops up at call time makes the checker check whatever is actually installed.

### The code, annotated

**Lifting and the elementwise helper.**

```python
def _t(x) -> Tensor:
    """x as a Tensor: Tensors pass through, anything else is a constant."""
    if isinstance(x, Tensor):
        return x
    a = np.asarray(x)
    # a float array keeps its dtype; ints and bools become float32 constants
    return Tensor(a, dtype=a.dtype if a.dtype.kind == "f" else np.float32)

def _elementwise(x, f, df, name: str) -> Tensor:
    """y = f(x) with vjp g * df(x), both in x's dtype."""
    x = _t(x)
    a = x.data
    y = np.asarray(f(a), dtype=a.dtype)          # M01.3 helpers may return float64
    return from_op(y, (x,), lambda g: (g * np.asarray(df(a), dtype=a.dtype),), name)

def exp(x):
    x = _t(x)
    y = np.exp(x.data)
    return from_op(y, (x,), lambda g: (g * y,), "exp")  # d e^x = e^x: reuse the output

def gelu(x, approximate="none"):
    if approximate == "none":
        return _elementwise(x, act.gelu_erf, act.dgelu_erf, "gelu")
    if approximate == "tanh":
        return _elementwise(x, act.gelu_tanh, act.dgelu_tanh, "gelu_tanh")
    raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
```

The two `dtype=a.dtype` casts are the whole float32 story for elementwise ops: without them a float64 derivative array would promote the gradient, and a float64 forward would promote every later activation. `relu` uses `act.drelu`, which is $[x > 0]$, so $\mathrm{relu}'(0) = 0$ as in torch.

**Reductions.** Axis normalization comes first, because every reduction's backward depends on knowing exactly which axes disappeared.

```python
def _axes(axis, ndim: int) -> tuple[int, ...]:
    """axis as a sorted tuple of non-negative axes (None: every axis)."""
    if axis is None:
        return tuple(range(ndim))
    ax = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    out = []
    for a in ax:
        a = int(a)
        if not -ndim <= a < builtins.max(ndim, 1):   # builtins.max: F.max shadows it
            raise ValueError(f"axis {a} is out of range for {ndim} dimensions")
        out.append(a % ndim if ndim else 0)
    return tuple(sorted(set(out)))      # sorted: expand_dims inserts in order

def _expand(g, axes, keepdims: bool):
    """An upstream gradient of a reduction, with the reduced axes put back as size 1."""
    return g if keepdims else np.expand_dims(g, axes)

def sum(x, axis=None, keepdims=False):
    x = _t(x)
    axes = _axes(axis, x.ndim)
    shape = x.shape
    y = np.sum(x.data, axis=axes, keepdims=keepdims)
    # Every input element contributed with weight 1: copy g back over the axes.
    # .copy(): broadcast_to returns a read-only view, and backward adds into gradients
    return from_op(
        y, (x,), lambda g: (np.broadcast_to(_expand(g, axes, keepdims), shape).copy(),), "sum"
    )

def mean(x, axis=None, keepdims=False):
    x = _t(x)
    axes = _axes(axis, x.ndim)
    n = math.prod(x.shape[a] for a in axes)    # the REDUCED count, not x.size
    if n == 0:
        raise ValueError("mean over an empty axis")
    return sum(x, axis, keepdims) * (1.0 / n)  # two existing ops: no new VJP to get wrong

def max(x, axis=None, keepdims=False):
    x = _t(x)
    axes = _axes(axis, x.ndim)
    a = x.data
    y = np.max(a, axis=axes, keepdims=keepdims)

    def vjp(g):
        ym = np.max(a, axis=axes, keepdims=True)
        hit = (a == ym).astype(a.dtype)          # 1 at every maximal entry
        # Ties share the gradient equally (torch.amax), so it still sums to g.
        share = hit / np.sum(hit, axis=axes, keepdims=True)
        return (share * _expand(g, axes, keepdims),)

    return from_op(y, (x,), vjp, "max")

def var(x, axis=None, keepdims=False, correction=0):
    x = _t(x)
    axes = _axes(axis, x.ndim)
    n = math.prod(x.shape[a] for a in axes)
    if n - correction <= 0:                    # n = 1 with Bessel's correction divides by 0
        raise ValueError(f"var over {n} elements with correction {correction}")
    a = x.data
    d = a - np.mean(a, axis=axes, keepdims=True)
    y = np.sum(d * d, axis=axes, keepdims=keepdims) / (n - correction)
    # d/dx_i of sum_j (x_j - mu)^2 is 2 (x_i - mu): the terms through mu sum
    # to zero because sum_j (x_j - mu) = 0.
    return from_op(
        y.astype(a.dtype, copy=False), (x,),
        lambda g: (_expand(g, axes, keepdims) * (2.0 / (n - correction)) * d,),
        "var",
    )
```

On `max([1, 3, 3])`: `hit = [0, 1, 1]`, its sum is 2, `share = [0, 0.5, 0.5]`, as section 3 computed. On `var([1, 2, 3, 4])`: `d = [-1.5, -0.5, 0.5, 1.5]`, $\sum d^2 = 5$, divided by $4 - 0$ or $4 - 1$.

**Shape ops.**

```python
def reshape(x, shape):
    x = _t(x)
    s = x.shape
    return from_op(x.data.reshape(tuple(shape)), (x,), lambda g: (g.reshape(s),), "reshape")

def transpose(x, a, b):
    x = _t(x)
    # swapping the same two axes again is the inverse
    return from_op(np.swapaxes(x.data, a, b), (x,), lambda g: (np.swapaxes(g, a, b),), "transpose")

def permute(x, dims):
    x = _t(x)
    dims = tuple(int(d) % builtins.max(x.ndim, 1) for d in dims)
    if sorted(dims) != list(range(x.ndim)):
        raise ValueError(f"dims {dims} is not a permutation of {x.ndim} axes")
    inv = tuple(int(i) for i in np.argsort(dims))  # undo the permutation in backward
    return from_op(np.transpose(x.data, dims), (x,), lambda g: (np.transpose(g, inv),), "permute")

def concat(xs, axis=0):
    ts = [_t(x) for x in xs]
    # ... empty list is an error
    sizes = [t.shape[axis] for t in ts]
    cuts = np.cumsum(sizes)[:-1]             # split points are CUMULATIVE sizes
    return from_op(np.concatenate([t.data for t in ts], axis=axis), ts,
                   lambda g: tuple(np.split(g, cuts, axis=axis)), "concat")

def stack(xs, axis=0):
    ts = [_t(x) for x in xs]
    # ... empty list is an error
    y = np.stack([t.data for t in ts], axis=axis)
    ax = axis % y.ndim                       # normalized against the OUTPUT's rank
    return from_op(y, ts, lambda g: tuple(np.take(g, i, axis=ax) for i in range(len(ts))), "stack")
```

For `permute(x, (2, 0, 1))`, output axis 0 is input axis 2. Applying `(2, 0, 1)` again would rotate further; `argsort((2, 0, 1)) = (1, 2, 0)` is the permutation that brings each axis home.

**Selection ops.**

```python
def _int_index(idx, n: int, what: str):
    """An integer index array checked against [0, n)."""
    a = np.asarray(idx)
    if a.size and not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{what} must be integers, got dtype {a.dtype}")
    a = a.astype(np.int64)
    if a.size and (a.min() < 0 or a.max() >= n):    # numpy would wrap -1 silently
        raise ValueError(f"{what} must lie in [0, {n}), got min {a.min()} and max {a.max()}")
    return a

def embedding(weight, ids):
    w = _t(weight)
    # ... weight must be 2-D [n, d]
    i = _int_index(ids, w.shape[0], "ids")
    W = w.data

    def vjp(g):
        out = np.zeros_like(W)
        np.add.at(out, i, g)  # a token that appears twice gets both rows of gradient
        return (out,)

    return from_op(W[i], (w,), vjp, "embedding")

def gather(x, idx, axis):
    x = _t(x)
    ax = axis % x.ndim
    i = _int_index(idx, x.shape[ax], "gather index")
    # ... idx must have x's rank (torch.gather semantics)
    a = x.data

    def vjp(g):
        out = np.zeros_like(a)
        # Full coordinates of every output element, with axis replaced by the
        # index: add.at so a position picked twice gets both gradients.
        coords = list(np.indices(i.shape, sparse=True))
        coords[ax] = i
        np.add.at(out, tuple(coords), g)
        return (out,)

    return from_op(np.take_along_axis(a, i, axis=ax), (x,), vjp, "gather")

def masked_fill(x, mask, value):
    x = _t(x)
    m = np.broadcast_to(np.asarray(mask, dtype=bool), x.shape)
    y = np.where(m, np.asarray(value, dtype=x.dtype), x.data)
    # a filled position's output does not depend on x: its gradient is zero
    return from_op(y, (x,), lambda g: (np.where(m, np.zeros((), dtype=g.dtype), g),), "masked_fill")
```

`gather`'s backward needs the full coordinate of every output element: `np.indices(i.shape, sparse=True)` gives open grids for every axis, and replacing the gathered axis with `i` turns them into the input coordinates `take_along_axis` read. `where` follows the same idea with two operands: `g` where `cond` holds goes to `a`, the rest to `b`, and each is unbroadcast to its operand's shape, because `where(cond, a, b)` broadcasts all three. A zero is built with `g`'s dtype so the gradient is not promoted.

**The softmax family.**

```python
def softmax(x, axis=-1):
    x = _t(x)
    y = np.asarray(stable.softmax(x.data, axis=axis), dtype=x.dtype)   # M09.2: max subtracted
    return from_op(y, (x,), lambda g: (softmax_vjp(g, y, axis),), "softmax")

def log_softmax(x, axis=-1):
    x = _t(x)
    y = np.asarray(stable.log_softmax(x.data, axis=axis), dtype=x.dtype)
    return from_op(y, (x,), lambda g: (log_softmax_vjp(g, y, axis),), "log_softmax")

def logsumexp(x, axis=-1, keepdims=False):
    x = _t(x)
    a = x.data
    y = np.asarray(stable.logsumexp(a, axis=axis, keepdims=keepdims), dtype=x.dtype)
    ax = axis % builtins.max(x.ndim, 1)
    # d logsumexp(x) / dx_i = softmax(x)_i, computed stably, never exp(x) / exp(y)
    return from_op(y, (x,),
        lambda g: (_expand(g, (ax,), keepdims) * np.asarray(stable.softmax(a, axis=ax), dtype=a.dtype),),
        "logsumexp")
```

with the VJPs from `M08.3` ([`vjp.py`](../../../../course/solutions/python/tinyllm/autograd/vjp.py)):

```python
def softmax_vjp(g, y, axis=-1):
    # y * (g - <g, y>): uses only the output, so a masked row (y = 0) gives 0
    return y * (g - np.sum(g * y, axis=axis, keepdims=True))

def log_softmax_vjp(g, y, axis=-1):
    # y is log-softmax, so exp(y) is softmax: never exp of a raw logit
    return g - np.exp(y) * np.sum(g, axis=axis, keepdims=True)
```

Trace section 3: $y = [1/6, 2/6, 3/6]$, $g = [1, 0, 0]$, `np.sum(g * y) = 1/6`, `g - 1/6 = [5/6, -1/6, -1/6]`, times $y$ is $[5/36, -2/36, -3/36]$.

**Dropout and matmul.**

```python
def dropout(x, p, training, rng):
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout p must be in [0, 1], got {p}")
    x = _t(x)
    if not training or p == 0.0:
        return x                               # the same object, and no draws consumed
    u = np.asarray(rng.uniforms(x.data.size), dtype=np.float64).reshape(x.shape)  # C order
    keep = u >= p                              # P(keep) = 1 - p
    # Inverted dropout: scale at training time so evaluation is the identity.
    scale = 0.0 if p == 1.0 else 1.0 / (1.0 - p)   # p = 1: all zeros, no division by 0
    m = (keep * scale).astype(x.dtype)         # the mask in x's dtype: no float64 leak
    return from_op(x.data * m, (x,), lambda g: (g * m,), "dropout")

def matmul(a, b):
    return _t(a) @ b                           # L0.1's VJP, unchanged
```

Returning `x` itself in eval mode (rather than a copy through `from_op`) means evaluation draws nothing from the generator, so whether an evaluation ran between two training steps cannot change the next training mask.

**`gradcheck_all`.**

```python
def gradcheck_all(rtol: float = 1e-5) -> dict[str, GradcheckReport]:
    reports = {}
    rng = PCG32(0)                             # fixed inputs: the check is reproducible
    for name, fn, shapes, (lo, hi) in _cases():  # 33 cases, ops looked up via globals()
        xs = [lo + (hi - lo) * rng.uniforms(math.prod(s)).reshape(s) for s in shapes]
        # Keep kinks out of reach: relu at 0 and max ties have no derivative.
        xs = [np.where(np.abs(x) < 0.05, x + 0.1, x) for x in xs]
        out_shape = fn(*[Tensor(x, dtype=np.float64) for x in xs]).shape
        # random weights, centred on 0: sum(softmax) = 1 would make a plain sum blind
        w = rng.uniforms(math.prod(out_shape)).reshape(out_shape) - 0.5

        def f(*arrays, fn=fn, w=w) -> float:   # default args bind this iteration's fn, w
            return float(np.sum(fn(*[Tensor(a, dtype=np.float64) for a in arrays]).data * w))

        ts = [Tensor(x, requires_grad=True, dtype=np.float64) for x in xs]
        fn(*ts).backward(w)                    # analytic gradient of sum(y * w)
        analytic = [t.grad if t.grad is not None else np.zeros_like(t.data) for t in ts]
        rep = gradcheck(f, xs, analytic, rtol=rtol)   # M04.1 central differences
        # elementwise ops: forward-mode duals are exact to rounding
        if name in _DUAL and _dual_error(name, fn) > 1e-9:
            rep = dataclasses.replace(rep, ok=False)
        reports[name] = rep
    return reports
```

Three details make it a real test. `_cases()` reads ops from `globals()` when it runs, so if a test (or a debugging session) replaces `F.softmax`, the replacement is what gets checked. Dropout is checked with a `_Rng` whose `uniforms` restarts `PCG32(7)` on every call, so every evaluation of `f` sees the same mask and the op is a fixed linear map that finite differences can probe. And `_dual_error` evaluates each elementwise op's backward at four points against `dual.derivative` of an independent forward-mode formula (sigmoid written as $e^x / (e^x + 1)$, gelu through `dual_erf`), so a $10^{-7}$ error, invisible to central differences at $\epsilon = 10^{-6}$, still fails.

The CLI verb wraps it:

```python
reports = F.gradcheck_all(rtol=1e-5)
failed = sorted(name for name, r in reports.items() if not r.ok)
worst = max(reports, key=lambda n: reports[n].max_rel_err) if reports else ""
# ... one "ok  <op> max_rel_err ..." or "FAIL <op> ..." line per op
result = {"suite": a.suite, "checks": len(reports), "failed": len(failed),
          "max_rel_err": float(max((r.max_rel_err for r in reports.values()), default=0.0)),
          "worst": worst}
if failed:
    raise CheckFailed(result, f"gradcheck failed for {', '.join(failed)}")   # exit 1
```

Full solution: [`functional.py`](../../../../course/solutions/python/tinyllm/autograd/functional.py); the verb is `cmd_gradcheck` in [`cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py).

### Why this solution

- **Cost.** Every VJP costs about as much as its forward: one pass over the input for elementwise ops and reductions, a scatter-add for selection, a matmul pair for matmul. `gradcheck_all` is expensive by design (two forward passes per input element for central differences) and runs on tiny shapes, so the whole suite takes well under a second.
- **Closed-form VJPs over composing primitives.** `softmax` could be built as `exp(x - max) / sum(exp(...))` from existing ops and differentiated automatically. That graph has four nodes, keeps four intermediate arrays, and its backward passes through `exp` of shifted logits; the closed form keeps one array (the output) and is stable by construction. `mean` is the one place composition wins: `sum * (1/n)` is exactly right and adds no new VJP.
- **Equal-share ties over first-index ties.** Routing the whole gradient to the first maximum is also a valid subgradient, but it disagrees with torch's `amax`, and torch parity is how later modules port models. Giving every tie the full gradient is simply wrong: the shares then sum to $k\bar{y}$.
- **An explicit generator over a global RNG.** `np.random` global state would make dropout masks depend on everything else that ever drew a number. Passing the PCG32 in and fixing the draw count makes masks reproducible across a crash and resume, and portable to Rust.
- **Two independent checks over one.** Central differences catch structural errors (a missing transpose, a wrong sign) on every op; duals catch tiny analytic errors on elementwise ops. Either alone misses a class of bugs.
- **Tradeoffs accepted.** Selection ops allocate a dense zero gradient the size of the table (a `[50257, 768]` embedding gets a dense gradient even if a batch touches 100 rows); sparse gradients are left to production frameworks.

### Explain it back

1. Why does `gradcheck_all` weight the output by random $w$ instead of differentiating `sum(y)`?

   <details><summary>Answer</summary>

   For softmax, $\sum_i y_i = 1$ for every input, so the true gradient of the plain sum is zero, and any VJP whose output sums to zero along the axis (including many wrong ones) would match central differences of a constant. Random weights make the scalar depend on every output differently, so every part of the VJP is exercised.

   </details>

2. Why does the `max` VJP divide the hit mask by its sum?

   <details><summary>Answer</summary>

   When $k$ entries tie for the maximum, the subgradient that matches torch's `amax` gives each $1/k$ of $\bar{y}$, so the shares sum to $\bar{y}$ as the chain rule requires. Without the division each tie gets the full gradient and the total is $k\bar{y}$.

   </details>

3. What goes wrong if `permute`'s backward applies `dims` again instead of `argsort(dims)`?

   <details><summary>Answer</summary>

   A permutation applied twice is not the identity unless it is its own inverse (a swap). For `(2, 0, 1)` applying it twice gives `(1, 2, 0)` of the original, so the gradient comes back with axes in the wrong order: wrong values for any shape, and a shape error whenever the axis sizes differ.

   </details>

4. Why does dropout in eval mode return its input object instead of `x * 1`?

   <details><summary>Answer</summary>

   Two reasons: evaluation must be deterministic and free, and it must not consume generator draws. Returning `x` draws nothing, builds no node, and leaves the generator where it was, so the next training step's mask is the same whether or not an evaluation happened in between.

   </details>

5. Why does `gradcheck_all` shift inputs with $|x| < 0.05$ away from zero?

   <details><summary>Answer</summary>

   Central differences assume the function is smooth between $x - \epsilon$ and $x + \epsilon$. At relu's kink an input within $\epsilon$ of 0 gives a numerical slope near $0.5$ while the analytic VJP says 0 or 1, so a correct op would be reported as broken. Moving every input at least 0.05 from zero (far more than $\epsilon = 10^{-6}$) keeps each probe on one side of the kink, and the check stays sharp everywhere else.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. `out[ids] = g` in `embedding` or `gather` | frequent tokens train slower than rare ones | `np.add.at` in both VJPs; see step 5 |
| 2. trusting ids | `-1` silently trains the last row of the table | `_int_index` rejects non-integers and values outside $[0, n)$ |
| 3. ties in `max`: every maximal entry gets the full gradient, or only the first | gradients double, or disagree with torch | `share = hit / sum(hit)` |
| 4. the `logsumexp` VJP as `exp(x) / exp(y)` | `inf / inf = nan` at logit 1000 | the VJP multiplies by `stable.softmax(a)` |
| 5. feeding the softmax VJP the input $x$ instead of the output $y$ | wrong gradients everywhere, `nan` on masked rows | `softmax_vjp(g, y, axis)` closes over the stable output |
| 6. a `gradcheck_all` that cannot fail (no dual check, or a loose tolerance) | `gradcheck --suite all` passes with a broken op | random output weights, a dual check at $10^{-9}$, and `rtol=1e-5` |
| 7. float64 leaking from a mask or a derivative array | a float32 model becomes float64, twice the memory | every derivative, mask, and zero is cast to the input's or the gradient's dtype |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | every op is `from_op(forward, parents, vjp)` on its `Tensor` |
| Back | `M01.3` | the activation functions and their derivatives |
| Back | `M09.2` | stable `softmax`, `log_softmax`, `logsumexp` forwards |
| Back | `M08.3` | `softmax_vjp`, `log_softmax_vjp`, `unbroadcast` |
| Back | `M08.1` | `derivative` on dual numbers, the second check in `gradcheck_all` |
| Back | `M04.1` | `gradcheck`, the central-difference check |
| Back | `M06.3` | `PCG32`: dropout's uniforms and `gradcheck_all`'s inputs |
| Forward | `L0.4` | `Linear`, `LayerNorm`, `Embedding`, `Dropout` are compositions of `F` ops |
| Forward | `L0.5` | `BigramLogits.forward` is `F.embedding`; the trainer reshapes with `F.reshape` |
| Forward | `L0.6` | the checkpoint tests' training loss is `F.mean` of a squared error |
| Forward | `L13.1` | `reshape` and `relu` around the conv and pooling ops |
| Forward | `L13.3` | GELU variants and residual sums in every vision block |
| Forward | `L13.5` | normalization, `matmul`, and log-softmax build the similarity losses |
| Forward | `L14.2` | GELU after each stem conv and inside every block |
| Forward | `L14.3` | Whisper decoding composes `F` ops |
| Forward | `L2.2`, `L3.1` to `L3.6`, `L4.1` to `L4.3`, `L5.3` to `L5.5`, `L6.1` to `L6.6`, `L7.1` to `L7.9` | every later model is written in `F` and never defines its own backward |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one VJP per op | PyTorch `derivatives.yaml` | a generated backward for about 2,000 ops, double-backward, forward-mode formulas | `tools/autograd/derivatives.yaml` |
| `gradcheck_all` | `torch.autograd.gradcheck`, `gradgradcheck` | complex inputs, sparse layouts, second derivatives, fast mode with random projections | `torch/autograd/gradcheck.py` |
| `dropout` with an explicit generator | `torch.nn.functional.dropout`, JAX `random.bernoulli` | Philox counter-based RNG on GPU, fused into matmul epilogues | `aten/src/ATen/native/Dropout.cpp` |
| stable softmax VJP | fused softmax and log-softmax kernels | one pass over the row, online max (`L9.2`) | `aten/src/ATen/native/SoftMax.cpp` |
