# Oracles, golden and differential tests, gradcheck as a test

Everything built in *Build Your Own LLM System* up to the modern transformer block got its gradients from the autograd of [Tensor, broadcasting backward, and no_grad](../../build-an-llm/01-foundations/02-tensor-and-broadcasting-backward.md) and [The op library and gradcheck_all](../../build-an-llm/01-foundations/03-op-library-and-gradcheck.md): each op's vector-Jacobian product was written once and composed. The C kernels and the Rust engine later in that book have no autograd and no torch to lean on. FlashAttention's backward, a fused RMSNorm, a SwiGLU kernel: each is a forward and a backward written by hand, and each is checked against the Python reference. That only works if a numerical kernel can be tested with nothing but an **oracle**, which is what this section shows, and what the oracle suites of the modern-block chapter, from [Pre-LN and RMSNorm](../../build-an-llm/08-modern-block/01-pre-ln-rmsnorm.md) to [Llama-family model](../../build-an-llm/08-modern-block/09-llama-family-model.md), do.

This section builds on [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md) for how a suite is judged and on [Property-based tests](02-property-based-tests.md) for laws and models. It uses the central-difference idea from [Partial derivatives, gradients, and gradcheck](../../mathematics-for-computing/05-calculus-3/01-gradients-and-gradcheck.md) in *Mathematics for Computing*, and two of its ops come from [Pre-LN and RMSNorm](../../build-an-llm/08-modern-block/01-pre-ln-rmsnorm.md) and [Gated MLPs](../../build-an-llm/08-modern-block/02-gated-mlps.md). The kata is [`kernels.py`](../../../../course/solutions/primers/craft.05/kernels.py): four forward passes (`linear`, `rmsnorm`, `swiglu`, `attention`) and their hand-written backward passes, in numpy. Its oracle suite is [`test_oracles.py`](../../../../course/solutions/primers/craft.05/test_oracles.py), which catches all ten planted faults. The suite runs with pytest and numpy; `TINYLLM_FIXTURES` names the directory that holds the torch-recorded golden values:

```bash
TINYLLM_FIXTURES=/path/to/fixtures uv run --no-project --with pytest --with numpy \
  python -m pytest -q primers/craft.05
```

Try to derive each backward yourself before reading it.

> **Learning objectives**
> - Name the three oracles (golden values, gradcheck, differential tests) and what each can and cannot catch.
> - Derive the backward of a linear layer, RMSNorm, SwiGLU, and attention from $L = \sum f(x) \odot g$.
> - Write a float64 gradcheck with a random upstream gradient and justify its tolerance.
> - Explain why a shape check is not an oracle, using the transposed weight gradient.

## What an oracle is

A test needs an **oracle**: something other than the code under test that knows the right answer. This section uses three, and every kernel from here on is tested with them. The notation is small:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f$ | the op under test, $x \mapsto y$ | function |
| $g$ | an upstream gradient, the shape of $y$ | array |
| $L = \sum f(x) \odot g$ | a scalar whose gradient with respect to $x$ is the VJP of $f$ at $g$ | `float` |
| $\epsilon$ | the central-difference step, $10^{-6}$ in float64 | `float` |
| $e_i$ | the unit vector of element $i$ | array |

### Golden tests

A golden test compares with numbers that an independent implementation produced and a maintainer recorded: here torch's float64 forward values and autograd gradients for fixed inputs (`oracles.npz`). It catches every bug that changes those numbers, including the ones that keep the shape. Its weakness is coverage: it only knows the recorded inputs. The suite reads the file through `TINYLLM_FIXTURES`, never through a hard-coded path.

### Gradcheck as a test

The second oracle covers any input. For any differentiable $f$ and any $g$, the VJP is the gradient of $L(x) = \sum f(x) \odot g$:

$$\frac{\partial L}{\partial x_i} \approx \frac{L(x + \epsilon e_i) - L(x - \epsilon e_i)}{2\epsilon}, \qquad \text{error } O(\epsilon^2) + O(u / \epsilon).$$

In float64 ($u \approx 1.1 \times 10^{-16}$) the step $\epsilon = 10^{-6}$ balances truncation and rounding near $10^{-10}$, so a tolerance of $10^{-6}$ relative is safe for a correct backward and far below any real bug. A random $g$ (not all ones) matters: with $g = 1$ a backward that sums where it should weigh can still pass. Check every input of the op (`x`, `W`, and `b` for a linear layer), with shapes where a transposed gradient would not fit **and** with square ones where it would.

### Differential tests

The third oracle is a second implementation. Two implementations of the same contract must agree: the causal mask against attention over each query's prefix, chunked against whole, C against Python, Rust against Python. A differential test needs no recorded numbers, so it covers any input that can be generated. When both sides come from the same author, one side must also be checked against another oracle, or two equally wrong implementations pass together.

### Tolerances are part of the oracle

Exact comparison fails for correct code as soon as two implementations sum in a different order. Use the dtype's rounding: float64 differences near $10^{-15}$ relative for a few operations, more for long reductions (with $\sqrt{K}$ growth, as [Error analysis, condition numbers, and tolerance budgets](../../mathematics-for-computing/10-numerical-methods-and-floating-point/03-error-analysis-condition-numbers-and-tolerance-budgets.md) explains). Too loose a tolerance hides real bugs: a missing $1/\sqrt{d}$ is a factor of 2 for $d = 4$, but a dropped term of size $10^{-3}$ needs a tolerance below that.

## Working it by hand: the transposed gradient

The bug that motivates all three oracles is small enough to work by hand. Take $y = x W^\top$ with $x = (1, 2)$ and $W = \begin{pmatrix} 1 & 0 \\ 3 & 1 \end{pmatrix}$, so $y = (1 \cdot 1 + 2 \cdot 0,\ 1 \cdot 3 + 2 \cdot 1) = (1, 5)$. With upstream $g = (1, 0)$:

- $\partial L / \partial W = g^\top x = \begin{pmatrix} 1 \\ 0 \end{pmatrix} (1, 2) = \begin{pmatrix} 1 & 2 \\ 0 & 0 \end{pmatrix}$: only row 0 of $W$ made $y_0$.
- The transposed bug $x^\top g = \begin{pmatrix} 1 & 0 \\ 2 & 0 \end{pmatrix}$ has the same $2 \times 2$ shape and is wrong.
- The central difference on $W_{0,1}$ gives $L(W_{0,1} \pm \epsilon) = 1 \pm 2\epsilon$, so the quotient is $2$, matching the correct entry and not the transposed one (0).

A shape check passes both; the golden file and the gradcheck reject the bug. Shapes are not an oracle.

## The interface

The kata, `primers/craft.05/kernels.py`, works in numpy float64, and its docstring is the spec:

```python
def linear(x, W, b): ...                         # x @ W.T + b
def linear_backward(x, W, gy): ...               # (gx, gW, gb)
def rmsnorm(x, w, eps=1e-6): ...                 # x / sqrt(mean(x^2) + eps) * w      (L7.1)
def rmsnorm_backward(x, w, gy, eps=1e-6): ...    # (gx, gw)
def swiglu(a, b): ...                            # silu(a) * b                        (L7.2)
def swiglu_backward(a, b, gh): ...               # (ga, gb)
def attention(q, k, v, causal=False): ...        # softmax(q k^T / sqrt(d)) v         (L5.1)
def attention_backward(q, k, v, go, causal=False): ...   # (gq, gk, gv)
```

The attention op is the one from [Scaled dot-product attention](../../build-an-llm/06-transformer-2017/01-scaled-dot-product-attention.md). The oracle suite, `primers/craft.05/test_oracles.py`, imports the kata as `kernels` and uses no torch, no autograd, and no unseeded randomness. It applies four oracles, each catching some planted faults:

| Oracle | For |
|---|---|
| golden: every forward and every gradient equals `oracles.npz` (`linear`, `linear_sq`, `rmsnorm`, `swiglu`, `attn`, `attn_causal`) | all four ops, square and non-square |
| gradcheck: central differences of $\sum f \odot g$ for every input, random $g$ | every backward, on generated inputs |
| differential: causal attention equals attention over each query's prefix | the mask |
| stability: scores of 300 give finite outputs (the mean of the values) | the softmax's max subtraction |

A correct solution reproduces the hand-worked case ($g_W = g^\top x$, not $x^\top g$), matches the golden file in every value and gradient, and comes with a suite made of oracles (golden values read through `TINYLLM_FIXTURES`, a gradcheck, no torch or autograd) that accepts a correct implementation and fails at least once against each of the 10 planted faults.

## Deriving the backward passes

Seven steps take you from the definition to the kata and its suite.

1. **Derive each backward from $L = \sum f(x) \odot g$.** For $y = xW^\top + b$, $\partial L/\partial W_{oi} = \sum_n g_{no} x_{ni}$, so $g_W = g^\top x$ with the shape of $W$; $g_x = gW$; $g_b = \sum_n g_n$. Every backward of the kata comes from writing $L$ out and differentiating one index at a time.
2. **RMSNorm's rms depends on $x$.** With $n = x/r$ and $u = g \odot w$, differentiating through $r$ adds $-n\,\mathrm{mean}(u \odot n)$: $g_x = (u - n\,\mathrm{mean}(u \odot n))/r$. Treating $r$ as a constant drops that term.
3. **SwiGLU's gate derivative is the product rule.** $\frac{d}{da}\, a\sigma(a) = \sigma + a\sigma(1 - \sigma)$.
4. **Attention's backward is three chain-rule steps.** $g_v = P^\top g_o$; $g_P = g_o v^\top$; the softmax VJP is $g_S = P \odot (g_P - \sum_j P \odot g_P)$; then $g_q = g_S k/\sqrt d$ and $g_k = g_S^\top q/\sqrt d$.
5. **Keep the forward numerically safe.** Subtract the row maximum before `exp`, and write the sigmoid as $\frac12(1 + \tanh(a/2))$, so neither overflows.
6. **Test with three independent oracles.** Golden values from torch for fixed inputs, square and non-square; a float64 gradcheck with random upstream $g$ for every input; and a differential test of the causal mask against prefix attention. Each covers what the others cannot.
7. **Choose shapes and inputs that expose the faults.** A square $W$ lets a transposed gradient fit; a non-square $W$ makes a wrong product fail on shape. Large $a$ makes SwiGLU's missing term large; scores of 300 make an unshifted softmax overflow.

The linear, RMSNorm, and SwiGLU backwards follow directly from steps 1 to 3:

```python
def linear_backward(x, W, gy):
    gx = gy @ W            # [N, o] @ [o, i] -> [N, i]
    gW = gy.T @ x          # [o, N] @ [N, i] -> [o, i]: the shape of W, NOT x.T @ gy
    gb = gy.sum(axis=0)    # summed over the batch: each example contributes
    return gx, gW, gb


def rmsnorm_backward(x, w, gy, eps=1e-6):
    r = _rms(x, eps)                                   # sqrt(mean(x^2) + eps), per row
    n = x / r
    gw = (gy * n).reshape(-1, x.shape[-1]).sum(axis=0) # w is shared by every row
    u = gy * w
    gx = (u - n * np.mean(u * n, axis=-1, keepdims=True)) / r   # the second term: r depends on x
    return gx, gw


def _sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))  # no overflow for large |a|


def swiglu_backward(a, b, gh):
    s = _sigmoid(a)
    ga = gh * b * (s + a * s * (1.0 - s))  # product rule on a * sigmoid(a)
    gb = gh * a * s
    return ga, gb
```

Attention follows step 4, with step 5's max subtraction in the shared probability helper:

```python
def _probs(q, k, causal):
    S = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if causal:
        Tq, Tk = S.shape[-2], S.shape[-1]
        # key j visible to query i iff j <= i: the diagonal is visible, so no row is empty
        S = np.where(np.arange(Tk)[None, :] <= np.arange(Tq)[:, None], S, -np.inf)
    S = S - S.max(axis=-1, keepdims=True)  # max subtraction: exp never overflows
    P = np.exp(S)                          # hidden keys: exp(-inf) = 0
    return P / P.sum(axis=-1, keepdims=True)


def attention_backward(q, k, v, go, causal=False):
    P = _probs(q, k, causal)                           # recomputed, as a fused kernel would
    gv = np.swapaxes(P, -1, -2) @ go
    gP = go @ np.swapaxes(v, -1, -2)
    gS = P * (gP - np.sum(P * gP, axis=-1, keepdims=True))   # softmax VJP; zero where P is zero
    scale = 1.0 / np.sqrt(q.shape[-1])
    gq = gS @ k * scale
    gk = np.swapaxes(gS, -1, -2) @ q * scale           # gS transposed: keys index the columns
    return gq, gk, gv
```

The backward recomputes $P$ instead of storing it, which is what FlashAttention does to save memory. Masked keys have $P = 0$, so the softmax VJP gives them zero gradient without any extra masking. The full kata is [`kernels.py`](../../../../course/solutions/primers/craft.05/kernels.py).

## Building the oracle suite

The gradcheck helper is the oracle that needs no recorded numbers:

```python
RNG = np.random.Generator(np.random.PCG64(5))          # seeded: the same inputs every run


def gradcheck(f, xs, grads, eps=1e-6, tol=1e-6):
    """Central differences of the scalar f(*xs) against grads, element by element."""
    for i, (x, g) in enumerate(zip(xs, grads)):
        assert g.shape == x.shape, f"input {i}: gradient shape {g.shape} != {x.shape}"
        for idx in np.ndindex(x.shape):
            old = x[idx]
            x[idx] = old + eps
            hi = f(*xs)
            x[idx] = old - eps
            lo = f(*xs)
            x[idx] = old                               # restore before the next element
            num = (hi - lo) / (2 * eps)
            assert abs(num - g[idx]) <= tol * (1 + abs(num)), f"input {i} at {idx}: {g[idx]} vs {num}"
```

The tolerance $10^{-6}$ relative (plus absolute near 0) sits between the $\sim 10^{-10}$ error of a correct central difference in float64 and the size of any real bug, as derived in [Gradcheck as a test](#gradcheck-as-a-test).

The tests, grouped by oracle:

```python
@pytest.mark.parametrize("p", ["linear", "linear_sq"])     # non-square and square W
def test_linear_golden(p):
    x, W, b, g = FIX[p + ".x"], FIX[p + ".W"], FIX[p + ".b"], FIX[p + ".g"]
    np.testing.assert_allclose(K.linear(x, W, b), FIX[p + ".y"], atol=1e-12)
    for got, name in zip(K.linear_backward(x, W, g), ("gx", "gW", "gb")):
        np.testing.assert_allclose(got, FIX[f"{p}.{name}"], atol=1e-12, err_msg=name)
# ... test_rmsnorm_golden, test_swiglu_golden, and test_attention_golden for "attn" and
#     "attn_causal" compare every forward and gradient the same way


def test_linear_gradcheck_square_and_not():
    for o in (4, 3):                                   # o == i: square W; o != i: not
        x, W, b, g = RNG.normal(size=(2, 4)), RNG.normal(size=(o, 4)), RNG.normal(size=o), RNG.normal(size=(2, o))
        gx, gW, gb = K.linear_backward(x, W, g)
        gradcheck(lambda x, W, b: float(np.sum(K.linear(x, W, b) * g)), [x, W, b], [gx, gW, gb])


def test_swiglu_gradcheck():
    a, b, g = RNG.normal(size=(2, 6)) * 3, RNG.normal(size=(2, 6)), RNG.normal(size=(2, 6))   # large a
    ga, gb = K.swiglu_backward(a, b, g)
    gradcheck(lambda a, b: float(np.sum(K.swiglu(a, b) * g)), [a, b], [ga, gb])
# ... test_rmsnorm_gradcheck and test_attention_gradcheck (causal and not) in the same form


def test_causal_equals_prefix_attention():             # differential: the mask vs. the definition
    q, k, v = RNG.normal(size=(5, 3)), RNG.normal(size=(5, 3)), RNG.normal(size=(5, 2))
    full = K.attention(q, k, v, causal=True)
    for i in range(5):
        np.testing.assert_allclose(full[i], K.attention(q[i : i + 1], k[: i + 1], v[: i + 1])[0], atol=1e-12)


def test_large_scores_stay_finite():                   # stability: scores of 300 * 300 * 3 / sqrt(3)
    q = np.full((2, 3), 300.0)
    k = np.full((4, 3), 300.0)
    v = RNG.normal(size=(4, 2))
    o = K.attention(q, k, v)
    assert np.isfinite(o).all()
    np.testing.assert_allclose(o, np.tile(v.mean(axis=0), (2, 1)), atol=1e-12)   # equal scores: the mean of v
```

The golden fixture (`oracles.npz`, torch's float64 forwards and autograd gradients for fixed inputs) is read through the `TINYLLM_FIXTURES` directory. Every gradcheck uses a random upstream $g$, never ones. The stability test knows its exact answer without any oracle file: when all scores are equal, softmax is uniform and the output is the mean of the values. The full suite is [`test_oracles.py`](../../../../course/solutions/primers/craft.05/test_oracles.py).

## Why this design

Three oracles beat one. Golden values catch everything on their inputs, including non-square bugs, but only there; gradcheck covers any input but costs two forward passes per element; the differential test checks the mask's meaning with no numbers recorded at all. A fault one misses, another catches.

Checking runs in float64, whatever the kernel's dtype. Central differences in float32 have errors near $10^{-3}$, too large to separate a correct backward from a slightly wrong one; the reference runs in float64 so the tolerance can be tight.

The gradcheck goes element by element rather than along a random projection. Perturbing each element costs $O(\text{size})$ forward passes, fine for the small shapes here; production tools (`torch.autograd.gradcheck` fast mode) project onto random directions to scale.

One tradeoff is accepted: golden files must be regenerated if a contract changes, and they tie the suite to torch's conventions. That is the point, because those are the conventions the C and Rust ports must match.

## What can go wrong

The table lists the ten planted faults, followed by three mistakes in the suite itself:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| A transposed weight gradient, $x^\top g$ for $g^\top x$ | right shape on a square $W$, wrong values | `gW = gy.T @ x`; caught by the `linear_sq` golden case and the square gradcheck |
| $g W^\top$ for the input gradient | fits only a square $W$ | `gx = gy @ W`; caught by the non-square golden case and gradcheck |
| Averaging the bias gradient over the batch | every bias learns $N$ times too slowly | `gb = gy.sum(axis=0)`; caught by the golden test |
| RMSNorm's backward treating the rms as a constant | the gradient misses the $-n \cdot \mathrm{mean}(u n)$ term | the second term in `gx`; caught by `test_rmsnorm_gradcheck` |
| SwiGLU's gate derivative as $\sigma(a)$ alone | wrong by $a \sigma (1 - \sigma)$, small near 0 | the product rule `s + a * s * (1 - s)`; caught by the gradcheck with $a$ scaled by 3 |
| A softmax backward without the row sum | `gq` and `gk` off by a term that vanishes only when $g$ is uniform | `gP - sum(P * gP)`; caught by gradchecks with random $g$ |
| Losing $1/\sqrt{d}$ in the query gradient | `gq` too large by $\sqrt{d}$ | `scale` on both `gq` and `gk`; caught by the golden test |
| $g_S q$ for $g_S^\top q$ in the key gradient | fits square scores only | `np.swapaxes(gS, -1, -2) @ q`; caught by the causal golden case and the gradcheck |
| A softmax without the max subtraction | `inf / inf` at scores near 700 in float64 | `S - S.max(...)`; caught by `test_large_scores_stay_finite` |
| A causal mask that hides the diagonal | query 0 sees nothing: NaN | `j <= i`; caught by the causal golden case and the prefix differential |
| `g = ones` in a gradcheck | sum-instead-of-weigh bugs pass | every gradcheck draws $g$ from the seeded generator |
| A tolerance tighter than float64 rounding | the correct kata fails | gradcheck at $10^{-6}$ relative, golden at $10^{-12}$ absolute |
| Comparing shapes only | every transposed bug on square inputs survives | every test compares values |

> **Warning:** The transposed weight gradient is the most important fault on the list. On a square matrix it has the right shape and plausible magnitudes, and training with it still moves the loss, just in the wrong direction for most weights.

## Where this leads

This section combined mutation testing and faults from pitfalls from [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md), the model property of [Property-based tests](02-property-based-tests.md) (which is a differential oracle), and central differences from [Partial derivatives, gradients, and gradcheck](../../mathematics-for-computing/05-calculus-3/01-gradients-and-gradcheck.md). The same three oracles carry forward:

- The modern-block chapter of *Build Your Own LLM System*, from [Pre-LN and RMSNorm](../../build-an-llm/08-modern-block/01-pre-ln-rmsnorm.md) through [Llama-family model](../../build-an-llm/08-modern-block/09-llama-family-model.md), has oracle suites of golden, differential, and gradcheck tests.
- The C kernels [Softmax in C](../../build-an-llm/10-kernels/04-softmax-three-pass-and-online.md), [FlashAttention forward in C](../../build-an-llm/10-kernels/05-flash-attention-forward.md), and [Elementwise kernels in C](../../build-an-llm/10-kernels/08-elementwise-kernels.md) are checked against the Python reference with the same three oracles.
- The Rust engine's [Candle model runner](../../build-an-llm/11-serving/02-model-runner-and-sampler.md) checks its logits against the Python reference and Hugging Face's fixture.
- [Benchmarks and perf gates](04-benchmarks-and-perf-gates.md) comes next: once the numbers are right, make them fast.

Each piece of the suite has a production counterpart:

- The gradcheck corresponds to `torch.autograd.gradcheck` and `gradgradcheck` (`torch/autograd/gradcheck.py`), which add complex inputs, a fast mode that uses a random projection instead of every element, and second derivatives.
- Golden files correspond to PyTorch OpInfo (`torch/testing/_internal/common_methods_invocations.py`), where one description per op drives forward, backward, dtype, and device tests.
- Differential tests correspond to FlashAttention's tests (`flash-attention/tests/test_flash_attn.py`), which compare the fused kernel against a float64 reference with a tolerance set from the reference's own float32 error.
- Tolerances correspond to `torch.testing.assert_close` and its per-dtype defaults (PyTorch testing docs).

### Further reading

- Barr et al., "The Oracle Problem in Software Testing: A Survey" (IEEE TSE, 2015).
- McKeeman, "Differential Testing for Software" (1998).
- The PyTorch `torch.autograd.gradcheck` documentation.

## Key terms

- **Oracle**: something other than the code under test that knows the right answer.
- **Golden test**: a comparison against numbers an independent implementation produced and a maintainer recorded.
- **Gradcheck**: a comparison of a hand-written gradient against central differences of $\sum f(x) \odot g$.
- **Differential test**: a check that two implementations of the same contract agree on generated inputs.
- **VJP (vector-Jacobian product)**: the gradient of $\sum f(x) \odot g$ with respect to $x$; what a backward pass computes.
- **Tolerance**: the allowed difference between result and oracle, set from the dtype's rounding.

## Summary

- A test needs an oracle. Golden files (torch's numbers), a second implementation (differential), and a mathematical identity (central differences for a gradient) are the three used for every kernel from here on.
- A hand-written backward is where the subtle bugs live: a transposed weight gradient has the right shape on a square matrix and the wrong values. Shapes are not an oracle.
- Gradcheck is a test: central differences of $\sum f(x) \odot g$ for a random upstream $g$ check every input's gradient, in float64, with a tolerance that float64 rounding cannot break.
- Tolerances are part of the oracle: too tight and the correct kata fails; too loose and a lost $1/\sqrt{d}$ survives.
- A good oracle suite catches every one of the ten planted faults, the transposed weight gradient above all.

## Review questions

1. Why can a transposed weight gradient pass a shape check, and which oracle catches it?

   <details><summary>Answer</summary>

   On a square $W$ ($o = i$), $x^\top g$ and $g^\top x$ are both $o \times i$, so the shape is right and the values are the transpose. The worked example shows the difference: the correct $g_W$ is $\begin{pmatrix} 1 & 2 \\ 0 & 0 \end{pmatrix}$, the transposed one $\begin{pmatrix} 1 & 0 \\ 2 & 0 \end{pmatrix}$. The square golden case (`linear_sq`) and the gradcheck with $o = 4$ both compare values and fail.

   </details>

2. Why must the gradcheck's upstream $g$ be random rather than all ones?

   <details><summary>Answer</summary>

   With $g = 1$ every output is weighted equally, so a backward that sums where it should weigh by $g$ gives the same numbers. The softmax VJP is the clearest case: with uniform $g_P$ the row-sum term cancels exactly, so a backward that forgets it passes. A random $g$ makes each output's weight distinct.

   </details>

3. What does `test_causal_equals_prefix_attention` check that the golden causal case does not?

   <details><summary>Answer</summary>

   It checks the meaning of the mask on generated inputs: causal attention for query $i$ must equal plain attention over keys $0..i$. The golden case only covers the recorded inputs. A mask that hides the diagonal gives query 0 no keys, so its row is NaN and the differential fails.

   </details>

4. Why is the gradcheck tolerance $10^{-6}$ and not $10^{-12}$ or $10^{-3}$?

   <details><summary>Answer</summary>

   A correct central difference in float64 with $\epsilon = 10^{-6}$ has error near $10^{-10}$ (truncation $O(\epsilon^2)$ plus rounding $O(u/\epsilon)$), so $10^{-12}$ would reject correct code. $10^{-3}$ would let a small missing term survive, for example SwiGLU's $a\sigma(1 - \sigma)$ near $a = 0$. $10^{-6}$ sits safely between.

   </details>

5. Why does `attention_backward` recompute $P$ instead of taking it from the forward?

   <details><summary>Answer</summary>

   The kata mirrors how a fused kernel works: storing $P$ costs $T_q \times T_k$ memory per head, while recomputing it from $q$ and $k$ costs one more matrix product. FlashAttention makes the same trade. Recomputing also keeps the backward's signature a function of the inputs alone, which makes it easy to test.

   </details>
