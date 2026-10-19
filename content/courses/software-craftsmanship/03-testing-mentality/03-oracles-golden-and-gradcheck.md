# Oracles, golden and differential tests, gradcheck as a test

## Overview

| | |
|---|---|
| **Module** | `craft.05` · practice · Python · Pass 5 · 1.5 to 2 h reading |
| **Solution** | the kata [`course/solutions/primers/craft.05/kernels.py`](../../../../course/solutions/primers/craft.05/kernels.py) (four forward passes of Parts 5 to 7 and their hand-written backward passes, in numpy: `linear`, `rmsnorm`, `swiglu`, `attention` and their `_backward`s) and its oracle suite [`course/solutions/primers/craft.05/test_oracles.py`](../../../../course/solutions/primers/craft.05/test_oracles.py), which catches all ten planted faults of section 6 |
| **Contract** | the rules in the kata's docstring (derived step by step in section 5) |
| **Needs** | reading: [`craft.03`](01-tdd-unit-tests-and-mutation-grading.md) mutation testing · [`craft.04`](02-property-based-tests.md) laws and models · `M04.1` central differences (the `gradcheck` idea these tests use) · [`L7.1`](../../../courses/build-an-llm/p07-modern-block/01-pre-ln-rmsnorm.md) RMSNorm and [`L7.2`](../../../courses/build-an-llm/p07-modern-block/02-gated-mlps.md) SwiGLU, two of the kata's ops |
| **Used by** | no call site (a practice): the oracle suites of `L7.1` to `L7.9` are written this way, and the same three oracles prove the C kernels (`L9`) and the Rust engine (`L10.1`) |
| **Optional depth** | Barr et al., "The Oracle Problem in Software Testing: A Survey" (IEEE TSE, 2015); McKeeman, "Differential Testing for Software" (1998); the PyTorch `torch.autograd.gradcheck` documentation |

## Key Takeaways

- A test needs an **oracle**: something other than the code under test that knows the right answer. Golden files (torch's numbers), a second implementation (differential), and a mathematical identity (central differences for a gradient) are the three used for every kernel from here on.
- A hand-written backward is where the subtle bugs live: a transposed weight gradient has the right shape on a square matrix and the wrong values. Shapes are not an oracle.
- Gradcheck **is** a test: central differences of $\sum f(x) \odot g$ for a random upstream $g$ check every input's gradient, in float64, with a tolerance that float64 rounding cannot break.
- Tolerances are part of the oracle: too tight and the correct kata fails; too loose and a lost $1/\sqrt{d}$ survives.
- A good oracle suite catches every one of the ten planted faults of section 6, the transposed weight gradient above all.

## How to read this chapter

Sections 1 and 2 explain the three oracles (golden values, gradcheck, differential tests) and how to set tolerances. Section 3 works the transposed-gradient bug by hand. Section 4 fixes the kata's interface and the oracles to use. Section 5 derives each backward from first principles, walks through the kata and its oracle suite, and shows which fault each test exists to catch. Try to derive each backward yourself before reading it.

The suite runs with pytest and numpy; `TINYLLM_FIXTURES` names the directory that holds the torch-recorded golden values:

```bash
TINYLLM_FIXTURES=/path/to/fixtures uv run --no-project --with pytest --with numpy \
  python -m pytest -q primers/craft.05
```

---

## 1. Why now

Everything built in Parts 5 to 7 got its gradients from the autograd (`L0.1`, `L0.2`): each op's VJP was written once and composed. The C kernels of Part 9 and the Rust engine of Part 10 have no autograd and no torch to lean on. FlashAttention's backward, a fused RMSNorm, a SwiGLU kernel: each is a forward and a backward written by hand, and each one is checked against **the Python reference** (P6). That only works if a numerical kernel can be tested with nothing but an oracle, which is what this chapter shows, and what the oracle suites of `L7.1` to `L7.9` do.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f$ | the op under test, $x \mapsto y$ | function |
| $g$ | an upstream gradient, the shape of $y$ | array |
| $L = \sum f(x) \odot g$ | a scalar whose gradient with respect to $x$ is the VJP of $f$ at $g$ | `float` |
| $\epsilon$ | the central-difference step, $10^{-6}$ in float64 | `float` |
| $e_i$ | the unit vector of element $i$ | array |

### 2.1 Golden tests

A golden test compares with numbers an independent implementation produced and a maintainer recorded: here torch's float64 forward values and autograd gradients for fixed inputs (`oracles.npz`). It catches every bug that changes those numbers, including the ones that keep the shape. Its weakness is coverage: only the recorded inputs. The suite reads the file through `TINYLLM_FIXTURES`, never through a hard-coded path.

### 2.2 Gradcheck as a test

For any differentiable $f$ and any $g$, the VJP is the gradient of $L(x) = \sum f(x) \odot g$:

$$\frac{\partial L}{\partial x_i} \approx \frac{L(x + \epsilon e_i) - L(x - \epsilon e_i)}{2\epsilon}, \qquad \text{error } O(\epsilon^2) + O(u / \epsilon).$$

In float64 ($u \approx 1.1 \times 10^{-16}$) the step $\epsilon = 10^{-6}$ balances truncation and rounding near $10^{-10}$, so a tolerance of $10^{-6}$ relative is safe for a correct backward and far below any real bug. A random $g$ (not all ones) matters: with $g = 1$ a backward that sums where it should weigh can still pass. Check every input of the op (`x`, `W`, and `b` for a linear layer), with shapes where a transposed gradient would not fit **and** with square ones where it would.

### 2.3 Differential tests

Two implementations of the same contract must agree: the causal mask against attention over each query's prefix, chunked against whole, C against Python, Rust against Python. A differential test needs no recorded numbers, so it covers any input that can be generated. When both sides come from the same author, one side must also be checked against an oracle, or two equally wrong implementations pass together.

### 2.4 Tolerances

Exact comparison fails for correct code as soon as two implementations sum in a different order. Use the dtype's rounding: float64 differences near $10^{-15}$ relative for a few operations, more for long reductions ($\sqrt{K}$ growth, `M09.3`). Too loose a tolerance hides real bugs: a missing $1/\sqrt{d}$ is a factor of 2 for $d = 4$, but a dropped term of size $10^{-3}$ needs a tolerance below that.

## 3. Worked example by hand

$y = x W^\top$ with $x = (1, 2)$ and $W = \begin{pmatrix} 1 & 0 \\ 3 & 1 \end{pmatrix}$: $y = (1 \cdot 1 + 2 \cdot 0,\ 1 \cdot 3 + 2 \cdot 1) = (1, 5)$. With upstream $g = (1, 0)$:

- $\partial L / \partial W = g^\top x = \begin{pmatrix} 1 \\ 0 \end{pmatrix} (1, 2) = \begin{pmatrix} 1 & 2 \\ 0 & 0 \end{pmatrix}$: only row 0 of $W$ made $y_0$.
- The transposed bug $x^\top g = \begin{pmatrix} 1 & 0 \\ 2 & 0 \end{pmatrix}$ has the same $2 \times 2$ shape and is wrong.
- Central difference on $W_{0,1}$: $L(W_{0,1} \pm \epsilon) = 1 \pm 2\epsilon$, so the quotient is $2$, matching the correct entry and not the transposed one (0).

A shape check passes both; the golden file and the gradcheck reject the bug.

## 4. The interface

**The kata**, `primers/craft.05/kernels.py` (its docstring is the spec), numpy float64:

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

**The oracle suite**, `primers/craft.05/test_oracles.py`: tests importing the kata as `kernels`; no torch, no autograd, no unseeded randomness. Each oracle catches some planted faults:

| Oracle | For |
|---|---|
| golden: every forward and every gradient equals `oracles.npz` (`linear`, `linear_sq`, `rmsnorm`, `swiglu`, `attn`, `attn_causal`) | all four ops, square and non-square |
| gradcheck: central differences of $\sum f \odot g$ for every input, random $g$ | every backward, on generated inputs |
| differential: causal attention equals attention over each query's prefix | the mask |
| stability: scores of 300 give finite outputs (the mean of the values) | the softmax's max subtraction |

### What a correct solution must handle

| Case | Checks |
|---|---|
| the hand-worked transposed gradient | section 3: $g_W = g^\top x$, not $x^\top g$ |
| the kata matches torch | every value and gradient equals the golden file |
| the suite is made of oracles | golden values read through `TINYLLM_FIXTURES`, a gradcheck, no torch or autograd |
| the suite accepts a correct implementation | every oracle holds for the kata |
| the suite catches the planted faults | against each of the 10 faults of section 6, at least one test fails |

## 5. The solution, step by step

### From first principles

1. **Derive each backward from $L = \sum f(x) \odot g$.** For $y = xW^\top + b$, $\partial L/\partial W_{oi} = \sum_n g_{no} x_{ni}$, so $g_W = g^\top x$ with the shape of $W$; $g_x = gW$; $g_b = \sum_n g_n$. Every backward of the kata comes from writing $L$ out and differentiating one index at a time.
2. **RMSNorm's rms depends on $x$.** With $n = x/r$ and $u = g \odot w$, differentiating through $r$ adds $-n\,\mathrm{mean}(u \odot n)$: $g_x = (u - n\,\mathrm{mean}(u \odot n))/r$. Treating $r$ as a constant drops that term.
3. **SwiGLU's gate derivative is the product rule.** $\frac{d}{da}\, a\sigma(a) = \sigma + a\sigma(1 - \sigma)$.
4. **Attention's backward is three chain-rule steps.** $g_v = P^\top g_o$; $g_P = g_o v^\top$; the softmax VJP $g_S = P \odot (g_P - \sum_j P \odot g_P)$; then $g_q = g_S k/\sqrt d$ and $g_k = g_S^\top q/\sqrt d$.
5. **Numerics in the forward.** Subtract the row maximum before `exp`, and write the sigmoid as $\frac12(1 + \tanh(a/2))$, so neither overflows.
6. **Test with three independent oracles.** Golden values from torch for fixed inputs, square and non-square; a float64 gradcheck with random upstream $g$ for every input; and a differential test of the causal mask against prefix attention. Each covers what the others cannot.
7. **Choose shapes and inputs that expose the faults.** Square $W$ hides nothing from values but a transposed gradient fits it; non-square $W$ makes a wrong product fail on shape. Large $a$ makes SwiGLU's missing term large; scores of 300 make an unshifted softmax overflow.

### The kata, annotated

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

The backward recomputes $P$ instead of storing it, which is what FlashAttention does to save memory; masked keys have $P = 0$, so the softmax VJP gives them zero gradient without any extra masking. Full kata: [`kernels.py`](../../../../course/solutions/primers/craft.05/kernels.py).

### The oracle suite, annotated

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

The tolerance $10^{-6}$ relative (plus absolute near 0) sits between the $\sim 10^{-10}$ error of a correct central difference in float64 and the size of any real bug (section 2.2).

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

The golden fixture (`oracles.npz`, torch's float64 forwards and autograd gradients for fixed inputs) is read through the `TINYLLM_FIXTURES` directory. Every gradcheck uses a random upstream $g$, never ones. The stability test knows its exact answer without any oracle file: when all scores are equal, softmax is uniform and the output is the mean of the values. Full suite: [`test_oracles.py`](../../../../course/solutions/primers/craft.05/test_oracles.py).

### Why this solution

- **Three oracles over one.** Golden values catch everything on their inputs, including non-square bugs, but only there; gradcheck covers any input but costs two forward passes per element; the differential test checks the mask's meaning with no numbers recorded at all. A fault one misses, another catches.
- **Float64 for checking, whatever the kernel's dtype.** Central differences in float32 have errors near $10^{-3}$, too large to separate a correct backward from a slightly wrong one; the reference runs in float64 so the tolerance can be tight.
- **Element-by-element gradcheck over a random projection.** Perturbing each element costs $O(\text{size})$ forward passes, fine for the small shapes here; production tools (`torch.autograd.gradcheck` fast mode) project onto random directions to scale.
- **Tradeoffs accepted.** Golden files must be regenerated if a contract changes, and they tie the suite to torch's conventions; that is the point, because those are the conventions the C and Rust ports must match.

### Explain it back

**1. Why can a transposed weight gradient pass a shape check, and which oracle catches it?**

<details><summary>Answer</summary>

On a square $W$ ($o = i$), $x^\top g$ and $g^\top x$ are both $o \times i$, so the shape is right and the values are the transpose. Section 3 shows the difference: the correct $g_W$ is $\begin{pmatrix} 1 & 2 \\ 0 & 0 \end{pmatrix}$, the transposed one $\begin{pmatrix} 1 & 0 \\ 2 & 0 \end{pmatrix}$. The square golden case (`linear_sq`) and the gradcheck with $o = 4$ both compare values and fail.

</details>

**2. Why must the gradcheck's upstream $g$ be random rather than all ones?**

<details><summary>Answer</summary>

With $g = 1$ every output is weighted equally, so a backward that sums where it should weigh by $g$ gives the same numbers. The softmax VJP is the clearest case: with uniform $g_P$ the row-sum term cancels exactly, so a backward that forgets it passes. A random $g$ makes each output's weight distinct.

</details>

**3. What does `test_causal_equals_prefix_attention` check that the golden causal case does not?**

<details><summary>Answer</summary>

It checks the meaning of the mask on generated inputs: causal attention for query $i$ must equal plain attention over keys $0..i$. The golden case only covers the recorded inputs. A mask that hides the diagonal gives query 0 no keys, so its row is NaN and the differential fails.

</details>

**4. Why is the gradcheck tolerance $10^{-6}$ and not $10^{-12}$ or $10^{-3}$?**

<details><summary>Answer</summary>

A correct central difference in float64 with $\epsilon = 10^{-6}$ has error near $10^{-10}$ (truncation $O(\epsilon^2)$ plus rounding $O(u/\epsilon)$), so $10^{-12}$ would reject correct code. $10^{-3}$ would let a small missing term survive, for example SwiGLU's $a\sigma(1 - \sigma)$ near $a = 0$. $10^{-6}$ sits safely between.

</details>

**5. Why does `attention_backward` recompute $P$ instead of taking it from the forward?**

<details><summary>Answer</summary>

The kata mirrors how a fused kernel works: storing $P$ costs $T_q \times T_k$ memory per head, while recomputing it from $q$ and $k$ costs one more matrix product. FlashAttention makes the same trade. Recomputing also keeps the backward's signature a function of the inputs alone, which makes it easy to test.

</details>

## 6. Pitfalls

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

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.03` | mutation testing, the correct-code baseline, faults from pitfalls |
| Back | `craft.04` | the model property is a differential oracle |
| Back | `M04.1` | central differences, the gradcheck these tests may use |
| Forward | `L7.1` to `L7.9` | oracle suites: golden, differential, gradcheck |
| Forward | `L9.2`, `L9.3`, `L9.6` | C kernels checked against the Python reference with the same three oracles |
| Forward | `L10.1` | the Rust engine's logits against the Python reference and HF's fixture |
| Forward | `craft.06` | benchmarks: once the numbers are right, make them fast |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the gradcheck | `torch.autograd.gradcheck`, `gradgradcheck` | complex inputs, fast mode (a random projection instead of every element), second derivatives | `torch/autograd/gradcheck.py` |
| golden files | PyTorch OpInfo | one description per op drives forward, backward, dtype, and device tests | `torch/testing/_internal/common_methods_invocations.py` |
| differential tests | FlashAttention's tests | the fused kernel against a float64 reference, tolerance set from the reference's own float32 error | `flash-attention/tests/test_flash_attn.py` |
| tolerances | `torch.testing.assert_close` | per-dtype default tolerances | PyTorch testing docs |
