# Activation functions and their derivatives

## Overview

| | |
|---|---|
| **Module** | `M01.3` · build · Python · Pass 2 · 1.5 to 2 h reading |
| **Solution** | [`course/solutions/python/tinyllm/num/activations.py`](../../../../course/solutions/python/tinyllm/num/activations.py): `sigmoid`, `tanh`, `relu`, `softplus`, `gelu_tanh`, `gelu_erf`, `silu`, each with its derivative `d<name>` |
| **Contract** | [`course/contracts/py/tinyllm/num/activations.pyi`](../../../../course/contracts/py/tinyllm/num/activations.pyi) |
| **Needs** | `M02.1` `erf_series`, which `gelu_erf` calls. Reading: `M01.1` (derivatives), `M00.1` (exp and log) |
| **Used by** | `M08.1` checks its dual-number derivatives against these closed forms · later `L0.2` wraps them as autograd ops, `L3.2` uses sigmoid and tanh for LSTM gates, `L7.2` uses SiLU in SwiGLU, `L9.6` ports SiLU to C |
| **Optional depth** | OpenStax, *Calculus Volume 1* (free), sections 3.3 to 3.6 (derivative rules, the chain rule) and 3.9 (exponential and logarithm); Hendrycks and Gimpel, "Gaussian Error Linear Units" (2016); Ramachandran, Zoph, and Le, "Searching for Activation Functions" (2017, SiLU/swish) |

## Key Takeaways

- An **activation** is a nonlinear function applied elementwise between linear layers; without one, any stack of layers is a single matrix. Training needs each one's **derivative**, which the chain rule multiplies into every gradient.
- The derivatives come from three rules: $(e^x)' = e^x$, the quotient or chain rule, and the product rule. $\sigma' = \sigma(x)\sigma(-x)$, $\tanh' = 1 - \tanh^2$, $\mathrm{softplus}' = \sigma$, $\mathrm{silu}' = \sigma(x)(1 + x\sigma(-x))$, $\mathrm{gelu}' = \Phi + x\varphi$.
- **Overflow is avoided, not silenced**: $e^{-|x|}$ never overflows, so sigmoid and softplus are written in terms of it, and every function is finite and warning-free at $x = \pm 10^{30}$ in float32 and float64.
- Writing $\sigma' $ as $\sigma(x)\sigma(-x)$ instead of $\sigma(1 - \sigma)$ keeps full **relative accuracy** in the tails.
- The two GELUs are **different functions** (they differ in the fourth digit at $x = 1$), and $\mathrm{relu}'(0) = 0$ by convention: match the checkpoint you load.

## How to read this chapter

Sections 1 and 2 derive each activation and its derivative from the rules of calculus, and show how to evaluate them without overflow or loss of relative accuracy. Section 3 works every one by hand at $x = 1$. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The tracer's bigram is one table lookup: no layers, nothing between them. Pass 2 replaces it with models that have hidden layers (the MLP of `L0.4`, later recurrent cells and transformer blocks), and a hidden layer is a matrix product followed by an activation function. The autograd engine (`L0.2`) needs both halves of every activation: the function for the forward pass and its derivative for the backward pass. Get a derivative wrong and the model still trains, slowly and to a worse loss, with no error anywhere; get the forward pass numerically careless and one large logit after a bad optimizer step turns the whole batch into nan. This module derives each activation and its derivative from the rules of calculus, writes them so that no input can overflow, and checks them against PyTorch at 1000 points from $-40$ to $40$.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | the input, one number per element | float32 or float64 array |
| $\sigma(x)$ | the logistic sigmoid $\frac{1}{1 + e^{-x}}$ | same shape as $x$ |
| $\tanh x$ | hyperbolic tangent $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | same |
| $\mathrm{relu}(x)$ | $\max(x, 0)$ | same |
| $\mathrm{softplus}(x)$ | $\ln(1 + e^x)$ | same |
| $\Phi(x)$ | standard normal CDF, $\frac{1}{2}\left(1 + \operatorname{erf}(x/\sqrt 2)\right)$ | same |
| $\varphi(x)$ | standard normal density, $\frac{1}{\sqrt{2\pi}} e^{-x^2/2}$; note $\Phi' = \varphi$ | same |
| $\operatorname{erf}(z)$ | $\frac{2}{\sqrt\pi}\int_0^z e^{-t^2}\,dt$, computed by `M02.1`'s `erf_series` | same |
| $f'$ or `df` | the derivative of $f$ with respect to $x$ | same |

### 2.1 Why a nonlinearity

A linear layer computes $y = Wx + b$. Two in a row compute $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$: again one linear layer, so depth would add nothing. Inserting a nonlinear $f$ between them, $W_2 f(W_1 x + b_1) + b_2$, breaks the collapse, and with enough hidden units such networks can approximate any continuous function on a bounded set. Every $f$ below is applied **elementwise**: output $i$ depends only on input $i$.

### 2.2 Sigmoid and tanh

$\sigma(x) = \frac{1}{1 + e^{-x}}$ squashes the line into $(0, 1)$: $\sigma(0) = \frac12$, $\sigma(x) \to 1$ as $x \to \infty$, $\to 0$ as $x \to -\infty$, and $\sigma(-x) = 1 - \sigma(x)$. It turns a score into a probability (LSTM gates in `L3.2`, logistic regression in `M07.7`). Its derivative, by the chain rule on $(1 + e^{-x})^{-1}$:

$$\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x)\,\sigma(-x) = \sigma(x)(1 - \sigma(x)) .$$

$\tanh$ is a rescaled sigmoid, $\tanh x = 2\sigma(2x) - 1$, with range $(-1, 1)$. By the quotient rule, $\tanh' x = 1 - \tanh^2 x$.

**Evaluating without overflow.** For $x = -1000$, $e^{-x} = e^{1000}$ overflows to inf. The value $1/(1 + \infty) = 0$ happens to be right, but the overflow raises a floating-point warning (an error under `np.errstate(over="raise")`), and the same pattern elsewhere produces $\infty / \infty = $ nan. The fix uses only $z = e^{-|x|} \in (0, 1]$, which can never overflow:

$$\sigma(x) = \begin{cases} \frac{1}{1 + z} & x \ge 0 \\ \frac{z}{1 + z} & x < 0 \end{cases}$$

(the second line is the first with numerator and denominator multiplied by $e^{x}$).

**Relative accuracy in the tails.** $\sigma'(30) = 9.3576 \times 10^{-14}$. Computed as $s(1 - s)$ with $s = \sigma(30) = 1 - 9.36 \times 10^{-14}$, the subtraction $1 - s$ leaves only the last few bits of $s$, and the result is $9.348 \times 10^{-14}$: three correct digits. $\sigma(30)\sigma(-30)$ never subtracts and keeps all sixteen. Tiny derivatives matter because the chain rule multiplies them into other factors.

### 2.3 ReLU and softplus

$\mathrm{relu}(x) = \max(x, 0)$ is the identity for positive inputs and zero otherwise; its derivative is 1 for $x > 0$ and 0 for $x < 0$. At $x = 0$ the graph has a corner and no derivative exists. Frameworks pick a value: PyTorch uses 0, and so does this course, because a gradient that differs from the framework's exactly at zeros makes gradients disagree with any checkpoint's training history there.

$\mathrm{softplus}(x) = \ln(1 + e^x)$ is a smooth relu: about $x$ for large $x$, about $e^x$ for very negative $x$. Its derivative is $\frac{e^x}{1 + e^x} = \sigma(x)$. Naively, $\ln(1 + e^{1000})$ overflows, and $\ln(1 + e^{-40})$ gives exactly 0 because $1 + 4.2 \times 10^{-18}$ rounds to 1. The stable form splits off the large part and uses `log1p`, which computes $\ln(1 + u)$ accurately for tiny $u$:

$$\mathrm{softplus}(x) = \max(x, 0) + \operatorname{log1p}\!\left(e^{-|x|}\right) .$$

### 2.4 GELU, exact and approximate

The Gaussian error linear unit weights its input by the probability that a standard normal variable is below it:

$$\mathrm{gelu}(x) = x\,\Phi(x) = \frac{x}{2}\left(1 + \operatorname{erf}\frac{x}{\sqrt 2}\right), \qquad \mathrm{gelu}'(x) = \Phi(x) + x\,\varphi(x)$$

by the product rule, since $\Phi' = \varphi$. The solution's `gelu_erf` calls `erf_series` from `M02.1`. Before erf was fast on GPUs, GPT-2 and BERT used a tanh approximation:

$$\mathrm{gelu}_{\tanh}(x) = \frac{x}{2}\left(1 + \tanh u\right), \quad u = \sqrt{2/\pi}\,(x + 0.044715\, x^3) .$$

Its derivative needs the product rule and the chain rule: $\frac12(1 + \tanh u) + \frac{x}{2}(1 - \tanh^2 u)\,u'$, with $u' = \sqrt{2/\pi}(1 + 3 \cdot 0.044715\, x^2)$. The two GELUs differ by up to $4.7 \times 10^{-4}$ (near $x = 2.7$); a checkpoint trained with one must be served with the same one. For $|x| > 10$, $\tanh u$ is already $\pm 1$ exactly, and $x^3$ would overflow for $|x| > 10^{102}$ ($7 \times 10^{12}$ in float32), so $x$ is clamped to $[-10, 10]$ inside $u$. Likewise $\varphi$ underflows to 0 beyond $|x| = 40$, and $x^2$ inside it is clamped there.

### 2.5 SiLU

$\mathrm{silu}(x) = x\,\sigma(x)$ (also called swish) is the gate inside SwiGLU, the MLP of Llama-family models (`L7.2`). Product rule:

$$\mathrm{silu}'(x) = \sigma(x) + x\,\sigma(x)(1 - \sigma(x)) = \sigma(x)\left(1 + x\,\sigma(-x)\right) .$$

Forgetting the product rule and writing $\sigma(x)$ alone is the classic bug: it is right at $x = 0$ and wrong everywhere else.

### 2.6 Identities a correct solution satisfies

$\sigma(x) + \sigma(-x) = 1$; $\tanh x = 2\sigma(2x) - 1$; and for $f$ in $\{\mathrm{softplus}, \mathrm{silu}, \mathrm{gelu}, \mathrm{gelu}_{\tanh}\}$, $f(x) - f(-x) = x$ (each is $x$ times something that adds to 1 at $\pm x$, or for softplus $\ln\frac{1 + e^x}{1 + e^{-x}} = \ln e^x$). These hold for every $x$, so they are an independent check of signs and symmetry at any point.

### 2.7 Dtypes

A float32 tensor stays float32 and a float64 tensor stays float64: promoting silently doubles memory and breaks the bit-for-bit comparisons with the C kernels of Pass 6. Integer input is computed in float64. The GELU with erf is computed in float64 internally and cast back.

## 3. Worked example by hand

At $x = 1$, with $e^{-1} = 0.3678794$:

| Function | Computation | Value | Derivative | Value |
|---|---|---|---|---|
| sigmoid | $1 / 1.3678794$ | 0.7310586 | $0.7310586 \times 0.2689414$ | 0.1966119 |
| tanh | $2\sigma(2) - 1$ | 0.7615942 | $1 - 0.7615942^2$ | 0.4199743 |
| relu | $\max(1, 0)$ | 1 | $x > 0$ | 1 |
| softplus | $\ln(1 + 2.7182818)$ | 1.3132617 | $\sigma(1)$ | 0.7310586 |
| silu | $1 \times 0.7310586$ | 0.7310586 | $0.7310586 \times (1 + 0.2689414)$ | 0.9276705 |
| gelu (erf) | $1 \times \Phi(1)$ | 0.8413447 | $\Phi(1) + \varphi(1) = 0.8413447 + 0.2419707$ | 1.0833155 |
| gelu (tanh) | $u = 0.7978846 \times 1.044715 = 0.8335620$, $\frac12(1 + \tanh u) = \frac12(1 + 0.6823840)$ | 0.8411920 | (section 2.4) | |

The exact and approximate GELU differ by $1.5 \times 10^{-4}$ here.

## 4. The interface

```python
def sigmoid(x: ArrayLike) -> NDArray: ...      # and dsigmoid
def tanh(x: ArrayLike) -> NDArray: ...         # and dtanh
def relu(x: ArrayLike) -> NDArray: ...         # and drelu (0 at x = 0)
def softplus(x: ArrayLike) -> NDArray: ...     # and dsoftplus (= sigmoid)
def gelu_tanh(x: ArrayLike) -> NDArray: ...    # and dgelu_tanh
def gelu_erf(x: ArrayLike) -> NDArray: ...     # and dgelu_erf, through erf_series (M02.1)
def silu(x: ArrayLike) -> NDArray: ...         # and dsilu
```

Every function keeps the shape and the float32 or float64 dtype of its input, returns finite values for finite inputs, and emits no overflow, invalid, or divide warning.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked table at 1 | unit | section 3's table for sigmoid, tanh, relu, softplus, silu | everyone agrees on every definition |
| the hand-worked GELUs | unit | both GELUs and the exact derivative at 1 | the two GELUs are not interchangeable |
| PyTorch reference values | golden | each function and derivative against PyTorch autograd at 1000 points in $[-40, 40]$, float64 tolerances | `L0.2` is checked against PyTorch forward and backward |
| derivatives against finite differences | gradcheck | each smooth derivative against the frozen central-difference gradcheck | the derivative belongs to its function, not just to a formula |
| relu away from the kink | gradcheck | relu's derivative away from 0 | the same, for relu |
| the identities of section 2.6 | property | at 500 seeded points | sign and symmetry slips |
| relative accuracy in the tails | boundary | $\sigma'(\pm 30)$ and $\mathrm{softplus}(-40)$ to 14 digits | tiny gradients still multiply other factors |
| no overflow for any finite input | boundary | $\pm 10^{30}$ and the largest float, float32 and float64, with warnings raised as errors | one bad logit must not become nan |
| dtype and shape | unit | float32 stays float32, shapes kept, ints become float64 | memory and C-kernel comparisons |
| $\mathrm{relu}'(0)$ | boundary | $\mathrm{relu}'(0) = 0$ | PyTorch's convention |

## 5. The solution, step by step

### From first principles

1. **Fix the dtype once, at the door.** Section 2.7: float32 stays float32, float64 stays float64, anything else becomes float64. One helper does this, and every constant afterwards is built in the input's dtype (`x.dtype.type(1)`) so that no scalar silently promotes the array.
2. **Sigmoid from $e^{-|x|}$.** Section 2.2: the only exponential that cannot overflow for every finite $x$ is $z = e^{-|x|} \in (0, 1]$. Both $\frac{1}{1+z}$ (for $x \ge 0$) and $\frac{z}{1+z}$ (for $x < 0$) are then bounded, warning-free expressions.
3. **Derivatives without subtracting from 1.** $\sigma' = \sigma(x)\sigma(-x)$ is a product of two accurately computed numbers; $\sigma(1 - \sigma)$ subtracts a number close to 1 and loses the tail (section 2.2). The same idea gives $\mathrm{silu}' = \sigma(x)(1 + x\sigma(-x))$.
4. **Softplus by splitting off the large part.** $\ln(1 + e^x) = \max(x, 0) + \ln(1 + e^{-|x|})$, and the second term is `log1p` of a number in $(0, 1]$: no overflow, and full accuracy when $e^{-|x|}$ is tiny. Its derivative is exactly $\sigma$, so it reuses `sigmoid`.
5. **ReLU's kink by convention.** $\mathrm{relu}'(x) = [x > 0]$, strict, so the value at 0 is 0, matching PyTorch (section 2.3).
6. **Tanh-GELU with a clamp.** $\tanh u$ is exactly $\pm1$ in float64 once $|x| > 10$, so clamping $x$ to $[-10, 10]$ inside $u$ changes no result and removes the overflow of $x^3$. The *outer* $x$ is not clamped: it multiplies $\tfrac12(1 + \tanh u)$ and must keep its true value.
7. **Exact GELU in float64.** $\Phi$ comes from `erf_series` (`M02.1`), which works in float64; computing in float64 and casting back once gives float32 callers a correctly rounded result rather than float32 rounding inside the series. The density $\varphi$ squares $x$, so $x$ is clamped to $[-40, 40]$ there, beyond which $e^{-x^2/2}$ is 0 anyway.

### The code, annotated

```python
def _arr(x: ArrayLike) -> NDArray:
    """x as a float array: float32 and float64 keep their dtype."""
    a = np.asarray(x)
    if a.dtype in (np.float32, np.float64):
        return a
    return a.astype(np.float64)          # ints, bools, float16 compute in float64


def sigmoid(x: ArrayLike) -> NDArray:
    x = _arr(x)
    # exp(-|x|) is in (0, 1]: it can never overflow. For x >= 0 that is
    # 1 / (1 + e^-x); for x < 0 multiply top and bottom by e^x.
    z = np.exp(-np.abs(x))
    one = x.dtype.type(1)                # a 1 of the input's dtype: no promotion
    return np.where(x >= 0, one / (one + z), z / (one + z))


def dsigmoid(x: ArrayLike) -> NDArray:
    x = _arr(x)
    # s(x)(1 - s(x)) = s(x) s(-x). The second form never subtracts from 1,
    # so dsigmoid(30) keeps its relative accuracy instead of 1e-3.
    return sigmoid(x) * sigmoid(-x)
```

`np.where` evaluates both branches on every element and then picks one. That is why both branches must be safe for *all* $x$, not only for the half where they are selected: with $z \in (0, 1]$, both $1/(1+z)$ and $z/(1+z)$ are. (The naive `np.where(x >= 0, 1/(1+np.exp(-x)), ...)` would still compute `np.exp(-x)` for $x = -1000$ and warn.)

```python
def dtanh(x: ArrayLike) -> NDArray:
    t = np.tanh(_arr(x))                 # np.tanh is already overflow-safe
    return t.dtype.type(1) - t * t       # 1 - tanh^2


def drelu(x: ArrayLike) -> NDArray:
    x = _arr(x)
    return (x > 0).astype(x.dtype)  # strict: the derivative at the kink is 0


def softplus(x: ArrayLike) -> NDArray:
    x = _arr(x)
    # max(x, 0) carries the large part; log1p(e^-|x|) is in (0, ln 2] and accurate when tiny
    return np.maximum(x, x.dtype.type(0)) + np.log1p(np.exp(-np.abs(x)))


def dsoftplus(x: ArrayLike) -> NDArray:
    return sigmoid(x)  # d/dx log(1 + e^x) = e^x / (1 + e^x)
```

For $x = -40$, `softplus` returns $0 + \operatorname{log1p}(4.25 \times 10^{-18}) = 4.25 \times 10^{-18}$, all digits correct, where $\ln(1 + e^{-40})$ would round $1 + 4.25 \times 10^{-18}$ to 1 and return 0.

```python
def _gelu_u(x: NDArray) -> NDArray:
    """sqrt(2/pi) (x + 0.044715 x^3) with x clamped to [-10, 10]: tanh is
    already +-1 there (u(10) = 43.6), and x^3 would overflow for |x| > 1e102
    (|x| > 7e12 in float32)."""
    t = x.dtype.type
    xc = np.clip(x, t(-GELU_CLAMP), t(GELU_CLAMP))
    return t(GELU_C) * (xc + t(GELU_A) * xc * xc * xc)


def gelu_tanh(x: ArrayLike) -> NDArray:
    x = _arr(x)
    t = x.dtype.type
    return t(0.5) * x * (t(1) + np.tanh(_gelu_u(x)))      # unclamped outer x


def dgelu_tanh(x: ArrayLike) -> NDArray:
    x = _arr(x)
    t = x.dtype.type
    xc = np.clip(x, t(-GELU_CLAMP), t(GELU_CLAMP))
    th = np.tanh(_gelu_u(x))
    du = t(GELU_C) * (t(1) + t(3 * GELU_A) * xc * xc)    # u'(x), on the clamped x
    # product rule on 0.5 x (1 + tanh u), with tanh' = 1 - tanh^2
    return t(0.5) * (t(1) + th) + t(0.5) * x * (t(1) - th * th) * du
```

Beyond the clamp, `th` is exactly $\pm 1$, so $1 - \text{th}^2 = 0$ and the second term is $0$ whatever `du` is; the derivative is then exactly 1 (for $x > 10$) or 0 (for $x < -10$), as it should be. Writing `t(0.5) * x` first also keeps $\tfrac12 x \cdot 2$ finite at the largest float, where `x * 2` would overflow.

```python
def _phi_cdf(x: NDArray) -> NDArray:
    """Phi(x) = (1 + erf(x / sqrt 2)) / 2 in float64."""
    return 0.5 * (1.0 + erf_series(x.astype(np.float64) / math.sqrt(2.0), ERF_TERMS))


def gelu_erf(x: ArrayLike) -> NDArray:
    x = _arr(x)
    return (x.astype(np.float64) * _phi_cdf(x)).astype(x.dtype)   # one cast back at the end


def dgelu_erf(x: ArrayLike) -> NDArray:
    x = _arr(x)
    xd = x.astype(np.float64)
    xc = np.clip(xd, -PDF_CLAMP, PDF_CLAMP)  # xd * xd overflows for |x| > 1e154
    pdf = np.exp(-0.5 * xc * xc) / math.sqrt(2.0 * math.pi)
    return (_phi_cdf(x) + xd * pdf).astype(x.dtype)               # Phi + x phi
```

`ERF_TERMS = 160` is the series length `M02.1` shows is enough for $3 \times 10^{-15}$ accuracy on the whole line.

```python
def silu(x: ArrayLike) -> NDArray:
    x = _arr(x)
    return x * sigmoid(x)


def dsilu(x: ArrayLike) -> NDArray:
    x = _arr(x)
    # d/dx [x s(x)] = s(x) + x s(x) (1 - s(x)) = s(x) (1 + x s(-x))
    return sigmoid(x) * (x.dtype.type(1) + x * sigmoid(-x))
```

At $x = 1$: $\sigma(1)(1 + 1 \cdot \sigma(-1)) = 0.7310586 \times 1.2689414 = 0.9276705$, the table's value. At $x = 10^{30}$, $\sigma(-x) = 0$ (its $z$ underflows to 0, quietly), so `dsilu` is exactly 1.

Full solution: [`activations.py`](../../../../course/solutions/python/tinyllm/num/activations.py).

### Why this solution

- **Cost.** Every function is $O(n)$ elementwise work. `sigmoid` computes one exponential per element and both `np.where` branches; `gelu_erf` is the expensive one, a 160-term series per element, which is acceptable for a reference implementation and is exactly why production code (Going further) uses minimax `erf`.
- **`np.where` with safe branches over masking.** Boolean indexing (`out[x >= 0] = ...`) also avoids computing the wrong branch, but costs copies and more code, and is easy to get wrong for 0-d inputs. With $z = e^{-|x|}$ both branches are safe everywhere, so the simple form is correct.
- **Clamping over `np.errstate`.** Suppressing overflow warnings would hide the problem but still produce inf intermediates, which turn into nan the moment they meet a 0 (for example $\infty \cdot 0$ in the derivative). Clamping where the result is already saturated changes nothing numerically and produces no inf at all.
- **Float64 for the erf path only.** Running everything in float64 would break the float32 comparisons of Pass 6 and double memory; running `erf_series` in float32 would lose accuracy in a 160-term sum. The solution promotes only where the algorithm needs it and casts back once.
- **Tradeoff accepted.** `dtanh` uses $1 - \tanh^2$, which subtracts from 1 in the tails: $\tanh'(20)$ is computed as exactly 0 instead of about $1.7 \times 10^{-17}$. PyTorch uses the same formula, and the golden comparison is against PyTorch, so matching it is the right choice here.

### Explain it back

**1. Why does `sigmoid` use $e^{-|x|}$ in both branches of `np.where`, rather than $e^{-x}$ in one and $e^{x}$ in the other?**

<details><summary>Answer</summary>

`np.where` evaluates both branch expressions for every element before choosing. With $e^{-x}$ in one branch and $e^{x}$ in the other, one of them overflows for every large $|x|$ even though its result is discarded, raising a warning (or an error under `np.errstate(over="raise")`). $e^{-|x|}$ is at most 1 for every input, so neither branch can overflow.

</details>

**2. $\sigma(x)(1 - \sigma(x))$ and $\sigma(x)\sigma(-x)$ are equal on paper. Why does the solution use the second?**

<details><summary>Answer</summary>

At $x = 30$, $\sigma(x) = 1 - 9.36 \times 10^{-14}$, and $1 - \sigma(x)$ is a subtraction of two numbers that agree in 13 digits: only about 3 correct digits survive. $\sigma(-30) = z/(1+z)$ with $z = e^{-30}$ is computed directly to full relative precision, and multiplying two accurate numbers loses nothing.

</details>

**3. In `gelu_tanh`, $x$ is clamped inside $u$ but not in the outer factor. What would clamping the outer $x$ do?**

<details><summary>Answer</summary>

For $x > 10$, $\mathrm{gelu}(x) = x$ to float precision. Clamping the outer factor would return $10$ for every $x > 10$, a function that saturates instead of growing linearly, with derivative 0 instead of 1. The clamp is safe only where its effect is invisible: inside $\tanh u$, which is already $\pm 1$.

</details>

**4. Why is $\mathrm{relu}'(0)$ defined as 0, when the derivative does not exist there?**

<details><summary>Answer</summary>

Any value in $[0, 1]$ is a valid subgradient, so the choice is a convention. Exact zeros are common in practice (padding, relu of a relu, quantized inputs), and the gradients there must match the framework the checkpoint and the reference implementation use. PyTorch uses 0, so `drelu` uses the strict comparison `x > 0`.

</details>

**5. Why does `gelu_erf` convert to float64 and back, while `gelu_tanh` stays in the input dtype?**

<details><summary>Answer</summary>

`gelu_tanh` uses only `np.tanh` and a few multiplications, which are accurate in either dtype and must match a float32 C kernel operation for operation. `gelu_erf` calls `erf_series`, a long sum whose accuracy depends on float64 intermediates; computing it in float32 would add many float32 roundings. Promoting for the computation and casting once at the end gives a float32 result that is the float64 answer correctly rounded, while the output dtype still matches the input.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. exponentials that can overflow: $1/(1 + e^{-x})$, $\ln(1 + e^x)$, an unclamped $x^3$ | overflow warnings, inf, or nan at $x = \pm 1000$ | only $e^{-\lvert x\rvert}$ is ever exponentiated; $x$ is clamped inside $u$ and inside $\varphi$; steps 2, 4, 6, 7 |
| 2. subtracting from 1 in the tails: $\sigma(1 - \sigma)$, $\ln(1 + e^{-40})$ | three correct digits at $x = 30$; $\mathrm{softplus}(-40) = 0$ | `sigmoid(x) * sigmoid(-x)` and `log1p`; steps 3 and 4 |
| 3. $\mathrm{relu}'(0) = 1$ | gradients differ from PyTorch's exactly at zeros | `(x > 0)`, strict; step 5 |
| 4. using the tanh approximation for the exact GELU (or the reverse) | $1.5 \times 10^{-4}$ off at $x = 1$; a loaded checkpoint drifts | two separate functions, `gelu_erf` through `erf_series` and `gelu_tanh` through `np.tanh` |
| 5. forgetting the product rule: $\mathrm{silu}' = \sigma$, $\mathrm{gelu}_{\tanh}'$ without the $\tanh'$ term | right at 0, wrong everywhere else | `dsilu` has the $x\sigma(-x)$ term; `dgelu_tanh` has both product-rule terms |
| 6. promoting float32 to float64 | a float64 array where the kernel expects float32 | `_arr` keeps float32; constants built with `x.dtype.type`; erf path casts back; step 1 |
| dropping the square in $1 - \tanh^2$ | $\tanh'(1) = 0.238$ instead of 0.420 | `t.dtype.type(1) - t * t` |
| a wrong GELU constant (0.0447 for 0.044715) | the 6th digit of every GELU | `GELU_A = 0.044715` |
| subtracting the log1p term in softplus | $\mathrm{softplus}(1) = 0.687$ | `np.maximum(x, 0) + np.log1p(...)` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M02.1` | `gelu_erf` and `dgelu_erf` call `erf_series(x / sqrt(2), 160)` |
| Back | `M01.1` | derivatives as limits; the frozen gradcheck in the tests is `M01.1`'s central difference (reading) |
| Back | `M00.1` | $e^x$ and $\ln$ (reading) |
| Forward | `M08.1` | dual numbers compute the same derivatives by forward-mode autodiff; its tests compare them with these to $10^{-15}$ |
| Forward | `L0.2` | the elementwise ops of `tinyllm.F` (forward and backward) |
| Forward | `L3.2` | LSTM gates: sigmoid for input, forget, output; tanh for the cell |
| Forward | `L7.2` | SwiGLU $= \mathrm{silu}(xW) \odot xV$ |
| Forward | `L9.6` | `tl_silu_mul_f32` in C, checked against this Python |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `sigmoid`, `silu`, `gelu_*` | PyTorch `torch.nn.functional` | fused CUDA kernels, autocast to bf16, in-place variants | `aten/src/ATen/native/Activation.cpp`, `cuda/ActivationGeluKernel.cu` |
| stable `softplus` | PyTorch `F.softplus(beta, threshold)` | returns $x$ above a threshold (default 20) instead of computing the log, trading $2 \times 10^{-9}$ for speed | `aten/src/ATen/native/cpu/Activation.cpp` |
| `gelu_erf` via a series | `erf` in libm, `erff` on GPUs | minimax rational approximations, a few ulps everywhere | glibc `sysdeps/ieee754/dbl-64/s_erf.c` |
| the activation zoo | llama.cpp `ggml_silu`, `ggml_gelu` | lookup tables for f16 GELU, vectorized SiLU | `ggml/src/ggml-cpu/vec.h` |
