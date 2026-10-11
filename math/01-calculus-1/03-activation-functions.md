<!-- ol:module M01.3 -->
# Activation functions and their derivatives

## Overview

| | |
|---|---|
| **Module** | `M01.3` · build · Python · Pass 2 · 3 to 4 h |
| **You build** | `python/tinyllm/num/activations.py`: `sigmoid`, `tanh`, `relu`, `softplus`, `gelu_tanh`, `gelu_erf`, `silu`, each with its derivative `d<name>` |
| **Contract** | [`course/contracts/py/tinyllm/num/activations.pyi`](../../course/contracts/py/tinyllm/num/activations.pyi) |
| **Tests** | `course/tests/M01.3/test_activations.py`, with the PyTorch golden values in `course/fixtures/M01.3/activations_torch.npz` (what they check: section 4) |
| **Needs** | `M02.1` `erf_series`, which `gelu_erf` calls (or `--ref-deps`). Reading: `M01.1` (derivatives), `M00.1` (exp and log) |
| **Used by** | `M08.1` checks its dual-number derivatives against these closed forms · later `L0.2` wraps them as autograd ops, `L3.2` uses sigmoid and tanh for LSTM gates, `L7.2` uses SiLU in SwiGLU, `L9.6` ports SiLU to C |
| **Milestone** | `MS-P2` (the Pass 2 gate) |
| **Optional depth** | OpenStax, *Calculus Volume 1* (free), sections 3.3 to 3.6 (derivative rules, the chain rule) and 3.9 (exponential and logarithm); Hendrycks and Gimpel, "Gaussian Error Linear Units" (2016); Ramachandran, Zoph, and Le, "Searching for Activation Functions" (2017, SiLU/swish) |

## Key Takeaways

- An **activation** is a nonlinear function applied elementwise between linear layers; without one, any stack of layers is a single matrix. Training needs each one's **derivative**, which the chain rule multiplies into every gradient (`test_hand_example_at_one`, `test_derivatives_pass_gradcheck`).
- The derivatives come from three rules: $(e^x)' = e^x$, the quotient or chain rule, and the product rule. $\sigma' = \sigma(x)\sigma(-x)$, $\tanh' = 1 - \tanh^2$, $\mathrm{softplus}' = \sigma$, $\mathrm{silu}' = \sigma(x)(1 + x\sigma(-x))$, $\mathrm{gelu}' = \Phi + x\varphi$ (`test_matches_torch_golden`).
- **Overflow is avoided, not silenced**: $e^{-|x|}$ never overflows, so sigmoid and softplus are written in terms of it, and every function is finite and warning-free at $x = \pm 10^{30}$ in float32 and float64 (`test_no_overflow_for_any_finite_input`).
- Writing $\sigma' $ as $\sigma(x)\sigma(-x)$ instead of $\sigma(1 - \sigma)$ keeps full **relative accuracy** in the tails (`test_derivative_tails_keep_relative_accuracy`).
- The two GELUs are **different functions** (they differ in the fourth digit at $x = 1$), and $\mathrm{relu}'(0) = 0$ by convention: match the checkpoint you load (`test_hand_example_gelus`, `test_relu_derivative_at_zero_is_zero`).

## How to work this chapter

```bash
ol start M01.3              # stubs python/tinyllm/num/activations.py into your repo
ol tests M01.3              # read the test catalog first: rung R0, you write no tests here
ol check M01.3              # exit code is the verdict
ol check M01.3 --ref-deps   # only if your M02.1 is not passing yet
ol diff  M01.3              # after passing: your code against the reference
```

---

## 1. Why now

The tracer's bigram is one table lookup: no layers, nothing between them. Pass 2 replaces it with models that have hidden layers (the MLP of `L0.4`, later recurrent cells and transformer blocks), and a hidden layer is a matrix product followed by an activation function. Your autograd engine (`L0.2`) needs both halves of every activation: the function for the forward pass and its derivative for the backward pass. Get a derivative wrong and the model still trains, slowly and to a worse loss, with no error anywhere; get the forward pass numerically careless and one large logit after a bad optimizer step turns the whole batch into nan. This module derives each activation and its derivative from the rules of calculus, writes them so that no input can overflow, and checks them against PyTorch at 1000 points from $-40$ to $40$.

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

$\mathrm{relu}(x) = \max(x, 0)$ is the identity for positive inputs and zero otherwise; its derivative is 1 for $x > 0$ and 0 for $x < 0$. At $x = 0$ the graph has a corner and no derivative exists. Frameworks pick a value: PyTorch uses 0, and so does this course, because a gradient that differs from the framework's exactly at zeros makes your gradients disagree with any checkpoint's training history there.

$\mathrm{softplus}(x) = \ln(1 + e^x)$ is a smooth relu: about $x$ for large $x$, about $e^x$ for very negative $x$. Its derivative is $\frac{e^x}{1 + e^x} = \sigma(x)$. Naively, $\ln(1 + e^{1000})$ overflows, and $\ln(1 + e^{-40})$ gives exactly 0 because $1 + 4.2 \times 10^{-18}$ rounds to 1. The stable form splits off the large part and uses `log1p`, which computes $\ln(1 + u)$ accurately for tiny $u$:

$$\mathrm{softplus}(x) = \max(x, 0) + \operatorname{log1p}\!\left(e^{-|x|}\right) .$$

### 2.4 GELU, exact and approximate

The Gaussian error linear unit weights its input by the probability that a standard normal variable is below it:

$$\mathrm{gelu}(x) = x\,\Phi(x) = \frac{x}{2}\left(1 + \operatorname{erf}\frac{x}{\sqrt 2}\right), \qquad \mathrm{gelu}'(x) = \Phi(x) + x\,\varphi(x)$$

by the product rule, since $\Phi' = \varphi$. Your `gelu_erf` calls `erf_series` from `M02.1`. Before erf was fast on GPUs, GPT-2 and BERT used a tanh approximation:

$$\mathrm{gelu}_{\tanh}(x) = \frac{x}{2}\left(1 + \tanh u\right), \quad u = \sqrt{2/\pi}\,(x + 0.044715\, x^3) .$$

Its derivative needs the product rule and the chain rule: $\frac12(1 + \tanh u) + \frac{x}{2}(1 - \tanh^2 u)\,u'$, with $u' = \sqrt{2/\pi}(1 + 3 \cdot 0.044715\, x^2)$. The two GELUs differ by up to $4.7 \times 10^{-4}$ (near $x = 2.7$); a checkpoint trained with one must be served with the same one. For $|x| > 10$, $\tanh u$ is already $\pm 1$ exactly, and $x^3$ would overflow for $|x| > 10^{102}$ ($7 \times 10^{12}$ in float32), so $x$ is clamped to $[-10, 10]$ inside $u$. Likewise $\varphi$ underflows to 0 beyond $|x| = 40$, and $x^2$ inside it is clamped there.

### 2.5 SiLU

$\mathrm{silu}(x) = x\,\sigma(x)$ (also called swish) is the gate inside SwiGLU, the MLP of Llama-family models (`L7.2`). Product rule:

$$\mathrm{silu}'(x) = \sigma(x) + x\,\sigma(x)(1 - \sigma(x)) = \sigma(x)\left(1 + x\,\sigma(-x)\right) .$$

Forgetting the product rule and writing $\sigma(x)$ alone is the classic bug: it is right at $x = 0$ and wrong everywhere else.

### 2.6 Identities the tests use

$\sigma(x) + \sigma(-x) = 1$; $\tanh x = 2\sigma(2x) - 1$; and for $f$ in $\{\mathrm{softplus}, \mathrm{silu}, \mathrm{gelu}, \mathrm{gelu}_{\tanh}\}$, $f(x) - f(-x) = x$ (each is $x$ times something that adds to 1 at $\pm x$, or for softplus $\ln\frac{1 + e^x}{1 + e^{-x}} = \ln e^x$). These hold for every $x$, so `test_identities` checks them at 500 seeded points.

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

The exact and approximate GELU differ by $1.5 \times 10^{-4}$ here. These numbers are the first two tests, `test_hand_example_at_one` and `test_hand_example_gelus`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_at_one` | unit, smoke | section 3's table for sigmoid, tanh, relu, softplus, silu | you and the tests agree on every definition |
| `test_hand_example_gelus` | unit, smoke | both GELUs and the exact derivative at 1 | the two GELUs are not interchangeable |
| `test_matches_torch_golden` | golden | each function and derivative against PyTorch autograd at 1000 points in $[-40, 40]$, float64 tolerances | `L0.2` is checked against PyTorch forward and backward |
| `test_derivatives_pass_gradcheck` | gradcheck | each smooth derivative against the frozen central-difference gradcheck | the derivative belongs to your function, not just to a formula |
| `test_relu_gradcheck_away_from_zero` | gradcheck | relu's derivative away from the kink | the same, for relu |
| `test_identities` | property | the identities of section 2.6 at 500 seeded points | sign and symmetry slips |
| `test_derivative_tails_keep_relative_accuracy` | boundary | $\sigma'(\pm 30)$ and $\mathrm{softplus}(-40)$ to 14 digits | tiny gradients still multiply other factors |
| `test_no_overflow_for_any_finite_input` | boundary | $\pm 10^{30}$ and the largest float, float32 and float64, with warnings raised as errors | one bad logit must not become nan |
| `test_dtype_and_shape_preserved` | unit | float32 stays float32, shapes kept, ints become float64 | memory and C-kernel comparisons |
| `test_relu_derivative_at_zero_is_zero` | boundary | $\mathrm{relu}'(0) = 0$ | PyTorch's convention |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. exponentials that can overflow: $1/(1 + e^{-x})$, $\ln(1 + e^x)$, an unclamped $x^3$ | overflow warnings, inf, or nan at $x = \pm 1000$ | `test_no_overflow_for_any_finite_input` (mutants `s10`, `s11`, `s12`) |
| 2. subtracting from 1 in the tails: $\sigma(1 - \sigma)$, $\ln(1 + e^{-40})$ | three correct digits at $x = 30$; $\mathrm{softplus}(-40) = 0$ | `test_derivative_tails_keep_relative_accuracy` (mutants `s03`, `s09`) |
| 3. $\mathrm{relu}'(0) = 1$ | gradients differ from PyTorch's exactly at zeros | `test_relu_derivative_at_zero_is_zero`, `test_matches_torch_golden` (mutant `s04`) |
| 4. using the tanh approximation for the exact GELU (or the reverse) | $1.5 \times 10^{-4}$ off at $x = 1$; a loaded checkpoint drifts | `test_hand_example_gelus` (mutant `s05`) |
| 5. forgetting the product rule: $\mathrm{silu}' = \sigma$, $\mathrm{gelu}_{\tanh}'$ without the $\tanh'$ term | right at 0, wrong everywhere else | `test_hand_example_at_one` (mutant `s02`), `test_derivatives_pass_gradcheck` (mutant `s06`) |
| 6. promoting float32 to float64 | a float64 array where the kernel expects float32 | `test_dtype_and_shape_preserved` (mutant `s13`) |
| dropping the square in $1 - \tanh^2$ | $\tanh'(1) = 0.238$ instead of 0.420 | `test_hand_example_at_one` (mutant `s01`) |
| a wrong GELU constant (0.0447 for 0.044715) | the 6th digit of every GELU | `test_hand_example_gelus` (mutant `s07`) |
| subtracting the log1p term in softplus | $\mathrm{softplus}(1) = 0.687$ | `test_hand_example_at_one` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M02.1` | `gelu_erf` and `dgelu_erf` call `erf_series(x / sqrt(2), 160)` |
| Back | `M01.1` | derivatives as limits; the frozen gradcheck in the tests is `M01.1`'s central difference (reading) |
| Back | `M00.1` | $e^x$ and $\ln$ (reading) |
| Forward | `M08.1` | dual numbers compute the same derivatives by forward-mode autodiff; its tests compare them with yours to $10^{-15}$ |
| Forward | `L0.2` | the elementwise ops of `tinyllm.F` (forward and backward) |
| Forward | `L3.2` | LSTM gates: sigmoid for input, forget, output; tanh for the cell |
| Forward | `L7.2` | SwiGLU $= \mathrm{silu}(xW) \odot xV$ |
| Forward | `L9.6` | `tl_silu_mul_f32` in C, checked against your Python |

`L0.2`, `L3.2`, `L7.2`, and `L9.6` join `used_by` when they are authored (`course/DEVIATIONS.md` row B31-02).

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `sigmoid`, `silu`, `gelu_*` | PyTorch `torch.nn.functional` | fused CUDA kernels, autocast to bf16, in-place variants | `aten/src/ATen/native/Activation.cpp`, `cuda/ActivationGeluKernel.cu` |
| stable `softplus` | PyTorch `F.softplus(beta, threshold)` | returns $x$ above a threshold (default 20) instead of computing the log, trading $2 \times 10^{-9}$ for speed | `aten/src/ATen/native/cpu/Activation.cpp` |
| `gelu_erf` via a series | `erf` in libm, `erff` on GPUs | minimax rational approximations, a few ulps everywhere | glibc `sysdeps/ieee754/dbl-64/s_erf.c` |
| the activation zoo | llama.cpp `ggml_silu`, `ggml_gelu` | lookup tables for f16 GELU, vectorized SiLU | `ggml/src/ggml-cpu/vec.h` |
