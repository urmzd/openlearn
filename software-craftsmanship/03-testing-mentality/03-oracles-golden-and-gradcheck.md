<!-- ol:module craft.05 -->
# Oracles, golden and differential tests, gradcheck as a test (R5)

## Overview

| | |
|---|---|
| **Module** | `craft.05` · practice · Python · Pass 5 · 3 to 4 h |
| **You build** | `primers/craft.05/kernels.py` (a kata: four forward passes of Parts 5 to 7 and their hand-written backward passes, in numpy) and `primers/craft.05/test_oracles.py`, your oracle tests for it |
| **Contract** | the rules in the kata's docstring (`ol check craft.05` writes the kata with stubs on its first run) |
| **Tests** | `course/tests/craft.05/`: the grade of your oracle tests by planted faults (section 4), with torch's values and gradients in `course/fixtures/craft.05/oracles.npz` (`course/oracle/craft.05/oracles_torch.py`) |
| **Needs** | reading: [`craft.03`](01-tdd-unit-tests-and-mutation-grading.md) how tests are graded · [`craft.04`](02-property-based-tests.md) laws and models · `M04.1` central differences (your `gradcheck`, which these tests may use) · [`L7.1`](../../ml/08-tinyllm/p07-modern-block/01-pre-ln-rmsnorm.md) RMSNorm and [`L7.2`](../../ml/08-tinyllm/p07-modern-block/02-gated-mlps.md) SwiGLU, two of the kata's ops |
| **Used by** | no call site (a practice): rung R5 grades your oracle tests in `L7.1` to `L7.9`, and the same three oracles prove your C kernels (`L9`) and Rust engine (`L10.1`) |
| **Milestone** | `MS-P5` (the Pass 5 gate requires a passing `craft.05`) |
| **Optional depth** | Barr et al., "The Oracle Problem in Software Testing: A Survey" (IEEE TSE, 2015); McKeeman, "Differential Testing for Software" (1998); the PyTorch `torch.autograd.gradcheck` documentation |

## Key Takeaways

- A test needs an **oracle**: something other than the code under test that knows the right answer. Golden files (torch's numbers), a second implementation (differential), and a mathematical identity (central differences for a gradient) are the three you will use for every kernel from here on.
- A hand-written backward is where the subtle bugs live: a transposed weight gradient has the right shape on a square matrix and the wrong values (`test_hand_example_transposed_gradient`). Shapes are not an oracle.
- Gradcheck **is** a test: central differences of $\sum f(x) \odot g$ for a random upstream $g$ check every input's gradient, in float64, with a tolerance that float64 rounding cannot break.
- Tolerances are part of the oracle: too tight and the correct kata fails (`test_your_tests_accept_the_reference`); too loose and a lost $1/\sqrt{d}$ survives.
- The grade: your tests must catch at least 80% of ten planted faults, and always the transposed weight gradient (`test_planted_faults_are_caught`).

## How to work this chapter

```bash
ol check craft.05           # first run: writes primers/craft.05/kernels.py (stubs) and fails
# write primers/craft.05/test_oracles.py (section 4), then the kata; run your tests yourself:
TINYLLM_FIXTURES=$PWD/.ol/openlearn/course/fixtures uv run --no-project --with pytest --with numpy \
  python -m pytest -q primers/craft.05
ol check craft.05           # grades your oracles against the planted faults
```

(`ol check` sets `TINYLLM_FIXTURES` itself; when you run pytest by hand, point it at the course's `fixtures/` directory of your openlearn checkout.) Write the tests first, against the stub: they must fail. `ol check` never shows a planted fault's code; a survivor prints only its one-line description.

---

## 1. Why now

Everything you built in Parts 5 to 7 got its gradients from your autograd (`L0.1`, `L0.2`): you wrote each op's VJP once and composed them. The C kernels of Part 9 and the Rust engine of Part 10 have no autograd and no torch to lean on. FlashAttention's backward, a fused RMSNorm, a SwiGLU kernel: each is a forward and a backward written by hand, and each one is checked against **your Python** (P6). That only works if you know how to test a numerical kernel with nothing but an oracle. The course tests of `L7.1` to `L7.9` did it for you with frozen helpers; from this chapter on, rung R5 grades the oracle tests you write for them.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f$ | the op under test, $x \mapsto y$ | function |
| $g$ | an upstream gradient, the shape of $y$ | array |
| $L = \sum f(x) \odot g$ | a scalar whose gradient with respect to $x$ is the VJP of $f$ at $g$ | `float` |
| $\epsilon$ | the central-difference step, $10^{-6}$ in float64 | `float` |
| $e_i$ | the unit vector of element $i$ | array |

### 2.1 Golden tests

A golden test compares with numbers an independent implementation produced and a maintainer recorded: here torch's float64 forward values and autograd gradients for fixed inputs (`oracles.npz`). It catches every bug that changes those numbers, including the ones that keep the shape. Its weakness is coverage: only the recorded inputs. Read the file through `TINYLLM_FIXTURES`, never through a path into the course tree.

### 2.2 Gradcheck as a test

For any differentiable $f$ and any $g$, the VJP is the gradient of $L(x) = \sum f(x) \odot g$:

$$\frac{\partial L}{\partial x_i} \approx \frac{L(x + \epsilon e_i) - L(x - \epsilon e_i)}{2\epsilon}, \qquad \text{error } O(\epsilon^2) + O(u / \epsilon).$$

In float64 ($u \approx 1.1 \times 10^{-16}$) the step $\epsilon = 10^{-6}$ balances truncation and rounding near $10^{-10}$, so a tolerance of $10^{-6}$ relative is safe for a correct backward and far below any real bug. A random $g$ (not all ones) matters: with $g = 1$ a backward that sums where it should weigh can still pass. Check every input of the op (`x`, `W`, and `b` for a linear layer), with shapes where a transposed gradient would not fit **and** with square ones where it would.

### 2.3 Differential tests

Two implementations of the same contract must agree: the causal mask against attention over each query's prefix, chunked against whole, C against Python, Rust against Python. A differential test needs no recorded numbers, so it covers any input you can generate. When both sides come from you, one side must also be checked against an oracle, or two equally wrong implementations pass together.

### 2.4 Tolerances

Exact comparison fails for correct code as soon as two implementations sum in a different order. Use the dtype's rounding: float64 differences near $10^{-15}$ relative for a few operations, more for long reductions ($\sqrt{K}$ growth, `M09.3`). Too loose a tolerance hides real bugs: a missing $1/\sqrt{d}$ is a factor of 2 for $d = 4$, but a dropped term of size $10^{-3}$ needs a tolerance below that.

## 3. Worked example by hand

$y = x W^\top$ with $x = (1, 2)$ and $W = \begin{pmatrix} 1 & 0 \\ 3 & 1 \end{pmatrix}$: $y = (1 \cdot 1 + 2 \cdot 0,\ 1 \cdot 3 + 2 \cdot 1) = (1, 5)$. With upstream $g = (1, 0)$:

- $\partial L / \partial W = g^\top x = \begin{pmatrix} 1 \\ 0 \end{pmatrix} (1, 2) = \begin{pmatrix} 1 & 2 \\ 0 & 0 \end{pmatrix}$: only row 0 of $W$ made $y_0$.
- The transposed bug $x^\top g = \begin{pmatrix} 1 & 0 \\ 2 & 0 \end{pmatrix}$ has the same $2 \times 2$ shape and is wrong.
- Central difference on $W_{0,1}$: $L(W_{0,1} \pm \epsilon) = 1 \pm 2\epsilon$, so the quotient is $2$, matching the correct entry and not the transposed one (0).

A shape check passes both; the golden file and the gradcheck reject the bug. This is `test_hand_example_transposed_gradient`.

## 4. The artifact and its check

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

**Your tests**, `primers/craft.05/test_oracles.py`: at least six test functions, importing the kata as `kernels` (and, if you like, your own `tinyllm.num.gradcheck` from `M04.1`); no torch, no autograd, no `random`. Each oracle catches some planted faults:

| Oracle | For |
|---|---|
| golden: every forward and every gradient equals `oracles.npz` (`linear`, `linear_sq`, `rmsnorm`, `swiglu`, `attn`, `attn_causal`) | all four ops, square and non-square |
| gradcheck: central differences of $\sum f \odot g$ for every input, random $g$ | every backward, on inputs you generate |
| differential: causal attention equals attention over each query's prefix | the mask |
| stability: scores of 300 give finite outputs (the mean of the values) | the softmax's max subtraction |

**The check** (`ol check craft.05`, `course/tests/craft.05/check`) runs six tests:

| Test | KIND | Checks |
|---|---|---|
| `test_hand_example_transposed_gradient` | unit | section 3 on the course's kata |
| `test_your_tests_are_oracle_tests` | unit | the file exists, at least 6 tests, reads the fixture through `TINYLLM_FIXTURES`, has a gradcheck, imports nothing forbidden |
| `test_your_kata_matches_torch` | golden | your kata's values and gradients equal torch's |
| `test_your_kata_passes_your_tests` | unit | your oracles hold for your kata |
| `test_your_tests_accept_the_reference` | conformance | your oracles hold for the course's kata |
| `test_planted_faults_are_caught` | fault | against each of the 10 planted faults your tests fail; score at least 0.80 and `s01` always caught |

Each run copies your test file next to one version of the kata in a scratch directory and runs pytest in its own process group with a timeout; your tree is never touched.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| A transposed weight gradient, $x^\top g$ for $g^\top x$ | right shape on a square $W$, wrong values | a golden test on the square linear case or a gradcheck with a square $W$ (mutant `s01`) |
| $g W^\top$ for the input gradient | fits only a square $W$ | the golden test or a gradcheck with a non-square $W$ (mutant `s02`) |
| Averaging the bias gradient over the batch | every bias learns $N$ times too slowly | the golden test (mutant `s03`) |
| RMSNorm's backward treating the rms as a constant | the gradient misses the $-n \cdot \mathrm{mean}(u n)$ term | a gradcheck of RMSNorm (mutant `s04`) |
| SwiGLU's gate derivative as $\sigma(a)$ alone | wrong by $a \sigma (1 - \sigma)$, small near 0 | a gradcheck with large $a$ (mutant `s05`) |
| A softmax backward without the row sum | `gq` and `gk` off by a term that vanishes only when $g$ is uniform | gradcheck with a random $g$ (mutant `s06`) |
| Losing $1/\sqrt{d}$ in the query gradient | `gq` too large by $\sqrt{d}$ | the golden test (mutant `s07`) |
| $g_S q$ for $g_S^\top q$ in the key gradient | fits square scores only | the causal golden case or a square gradcheck (mutant `s08`) |
| A softmax without the max subtraction | `inf / inf` at scores near 700 in float64 | the stability test (mutant `s09`) |
| A causal mask that hides the diagonal | query 0 sees nothing: NaN | the causal golden case, the prefix differential (mutant `s10`) |
| `g = ones` in a gradcheck | sum-instead-of-weigh bugs pass | `test_planted_faults_are_caught` reports the survivors |
| A tolerance tighter than float64 rounding | the correct kata fails | `test_your_tests_accept_the_reference` |
| Comparing shapes only | every transposed bug on square inputs survives | `test_planted_faults_are_caught` (`s01` is required) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.03` | mutation grading, baselines A and B, required faults |
| Back | `craft.04` | the model property is a differential oracle |
| Back | `M04.1` | central differences, the gradcheck these tests may use |
| Forward | `L7.1` to `L7.9` | your rung R5 suites: golden, differential, gradcheck |
| Forward | `L9.2`, `L9.3`, `L9.6` | C kernels checked against your Python with the same three oracles |
| Forward | `L10.1` | the Rust engine's logits against your Python and HF's fixture |
| Forward | `craft.06` | benchmarks: once the numbers are right, make them fast |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| your gradcheck | `torch.autograd.gradcheck`, `gradgradcheck` | complex inputs, fast mode (a random projection instead of every element), second derivatives | `torch/autograd/gradcheck.py` |
| golden files | PyTorch OpInfo | one description per op drives forward, backward, dtype, and device tests | `torch/testing/_internal/common_methods_invocations.py` |
| differential tests | FlashAttention's tests | the fused kernel against a float64 reference, tolerance set from the reference's own float32 error | `flash-attention/tests/test_flash_attn.py` |
| tolerances | `torch.testing.assert_close` | per-dtype defaults, like the course's `_lib.close` | PyTorch testing docs |
