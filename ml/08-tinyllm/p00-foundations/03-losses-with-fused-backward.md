<!-- ol:module L0.3 -->
# Losses with a fused backward

## Overview

| | |
|---|---|
| **Module** | `L0.3` · build · Python · Pass 2 · 3 to 4 h, plus your graded tests (rung R2) |
| **You build** | `python/tinyllm/autograd/losses.py`: `cross_entropy` (ignore_index, label smoothing, three reductions), `mse`, `bce_with_logits` (pos_weight); and your own tests in `python/tests/l0-3-losses/` |
| **Contract** | [`course/contracts/py/tinyllm/autograd/losses.pyi`](../../../course/contracts/py/tinyllm/autograd/losses.pyi) |
| **Tests** | `course/tests/L0.3/` (what they check: section 4), golden values from torch 2.14 in `course/fixtures/L0.3/losses_torch.npz`; your tests are graded by mutation, threshold 0.60 |
| **Needs** | `L0.1` `from_op` · `M09.2` stable `log_softmax` · `M08.3` `cross_entropy_vjp` (the differential test) · `M11.1` cross-entropy $H(q, p)$ (the differential test) · reading: `craft.03` how your tests are graded (or `--ref-deps`) |
| **Used by** | `L0.5` every training step of the course computes one of these · later: `L11.1`, `L12.1`, `L2.2`, `L3.6`, `L4.1`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5` |
| **Milestone** | `MS-L0` (the bigram and the digits MLP train on `cross_entropy`) |
| **Optional depth** | Goodfellow, Bengio, Courville, *Deep Learning* (free online), sections 6.2.2 and 7.5; Szegedy et al., "Rethinking the Inception Architecture" (2016), section 7 (label smoothing) |

## Key Takeaways

- Softmax cross-entropy is one graph node: its forward uses `log_softmax`, and its gradient is the closed form $\mathrm{softmax}(z) - q$ (`test_matches_m08_cross_entropy_vjp`, `test_gradcheck_losses`).
- Padding rows (`ignore_index`) add no loss, get no gradient, and do not count in the mean; a batch of only padding is loss 0, not `nan` (`test_ignore_index_excluded`, `test_all_ignored_is_zero`).
- Label smoothing spreads $\varepsilon/V$ over **every** class, the target included, so a row's loss is the cross-entropy $H(q, p)$ against the smoothed target (`test_hand_example_label_smoothing`, `test_rows_match_m11_cross_entropy`).
- Stable forms keep logits of $10^3$ finite: `log_softmax` for cross-entropy, softplus for binary cross-entropy (`test_large_logits_stay_finite`).
- Your own tests, written from the names in section 4, must kill at least 60% of the planted faults: the first grade of your tests in the course.

## How to work this chapter

```bash
ol start L0.3              # stubs losses.py into your repo; prints your test path and rung
ol tests L0.3              # the course tests: the exemplars your own tests imitate
ol check L0.3              # course tests first, then the mutation grade of your tests
ol mutate L0.3             # the full grade, cached by your test files' hash
ol check L0.3 --ref-deps   # only if L0.1 or a math dependency is not passing yet
ol diff  L0.3              # after passing: your code against the reference
```

Read [`craft.03`](../../../software-craftsmanship/03-testing-mentality/01-tdd-unit-tests-and-mutation-grading.md) first if you have not: it explains what a mutant is, how the score is computed, and why your tests may import only the contract.

---

## 1. Why now

`L0.2` gives you `log_softmax` and `gather`, so cross-entropy could be written as three ops: `-gather(log_softmax(z), t).mean()`. It would be correct and slow, and it would be fragile. Backward through that chain keeps a `[N, V]` array alive at each step, and for the bigram (`V = 256`) or a real vocabulary (`V = 49,152` for SmolLM2) that is most of the memory a training step uses. Padding makes it worse: batches of token windows have rows that must not count, and a mean that divides by the padded length silently shrinks the loss. Every training loop from here on (`L0.5`, `L2.2`, `L6.1`, the capstone) calls this file once per step, so the loss must be one node with a closed-form gradient, correct about padding, and finite for any logits.

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

**The fused gradient.** For one row with a one-hot target, $\ell = -\log p_t = -z_t + \log\sum_j e^{z_j}$. Differentiate: $\partial\ell/\partial z_j = -[j = t] + e^{z_j}/\sum_k e^{z_k} = p_j - [j = t]$. With a smoothed target, $\ell = -\sum_j q_j z_j + \log\sum_k e^{z_k}$ (because $\sum_j q_j = 1$), so $\partial\ell/\partial z = p - q$. That is the whole backward: one subtraction on the probabilities, which the forward already computed as $e^{\log p}$. `M08.3` derived it; the course test compares your fused VJP with `M08.3`'s `cross_entropy_vjp` on the same inputs.

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

These are `test_hand_example_cross_entropy` and `test_hand_example_label_smoothing`.

## 4. The interface

```python
# python/tinyllm/autograd/losses.py
def cross_entropy(logits: Tensor, targets: ArrayLike, ignore_index: int = -100,
                  label_smoothing: float = 0.0, reduction: Literal["mean", "sum", "none"] = "mean") -> Tensor
def mse(pred: Tensor, target: ArrayLike) -> Tensor
def bce_with_logits(logits: Tensor, targets: ArrayLike, pos_weight: Optional[ArrayLike] = None) -> Tensor
```

Each returns one `from_op` node whose VJP is the closed form above. Validate targets before anything else: integers, shape `logits.shape[:-1]`, each in $[0, V)$ or equal to `ignore_index`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_cross_entropy` | unit | section 3: loss $\ln 2$, gradient $[1/6, 1/3, -1/2]$, zeros on the ignored row | you and the test agree on the definition |
| `test_hand_example_label_smoothing` | unit | section 3: $q = [0.1, 0.1, 0.8]$, loss and $p - q$ | the smoothing every LM config can turn on |
| `test_matches_torch` | golden | loss and gradient equal `torch.nn.functional` on every option | training loops ported from torch |
| `test_gradcheck_losses` | gradcheck | the fused VJPs against frozen central differences | a closed form off by a factor fails |
| `test_matches_m08_cross_entropy_vjp` | differential | your fused gradient equals `M08.3`'s matrix | the derivation and the code agree |
| `test_rows_match_m11_cross_entropy` | differential | each smoothed row loss is `M11.1`'s $H(q, p)$ | the definition, computed the slow way |
| `test_ignore_index_excluded` | property | appending ignored rows changes neither the loss nor the kept gradients | padded batches (`L0.6` windows, `L12.1` packing) |
| `test_all_ignored_is_zero` | boundary | only padding gives loss 0 and a zero gradient | one `nan` step ruins a run |
| `test_large_logits_stay_finite` | boundary | logits of $10^4$ and bce logits of 1000 stay finite | trained logits are large |
| `test_reductions_agree` | property | `none`, `sum`, `mean` agree with each other | evaluation sums `none` (`L6.7`) |
| `test_float32_stays_float32` | boundary | float32 logits give float32 loss and gradient | in-place float32 updates |
| `test_rejects_bad_inputs` | boundary | out-of-range or float targets, unknown reduction, bad $\varepsilon$, shape mismatches | data and config bugs fail here |

### Your graded tests (rung R2)

Write them in `python/tests/l0-3-losses/` (any `test_*.py` file there). The names and what each must show are given; the bodies are yours. Import only names the contract declares (`tinyllm.autograd.losses`, `tinyllm.autograd.tensor`), plus numpy, pytest, and the standard library: your tests run against the reference with one fault planted at a time, so they must not depend on anything else of yours. Compute expected values by hand (this is rung R1's habit: the number in the assertion comes from you, not from running the code).

- `test_hand_example_matches_section_3`: logits $[0, \ln 2, \ln 3]$, target 2: loss $\ln 2$, gradient $[1/6, 1/3, -1/2]$.
- `test_ignored_rows_get_no_loss_and_no_gradient`: a row whose target is `ignore_index` adds no loss and gets a zero gradient.
- `test_mean_divides_by_kept_rows`: two kept rows and one ignored row: the mean is the kept sum over 2.
- `test_all_ignored_batch_is_zero`: only padding gives loss 0 and a zero gradient, never `nan`.
- `test_label_smoothing_hand_value`: $\varepsilon = 0.3$ on $V = 3$: the loss and $p - q$ of section 3.
- `test_large_logits_are_finite`: logits of $10^4$ and bce logits of 1000 give finite losses and gradients.
- `test_out_of_range_target_raises`: targets outside $[0, V)$, other than `ignore_index`, are a `ValueError`.
- `test_mse_and_bce_gradients_by_finite_differences`: both gradients match central differences you write; bce with `pos_weight` 2 at $x = 0$, $y = 1$ is $2\ln 2$.

`ol check L0.3` passes when the course tests pass and your tests kill at least 60% of the planted faults (`ol mutate L0.3` prints the score and the survivors).

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. counting ignored rows in the mean, or giving them gradient | padded batches report a smaller loss; padding logits train toward a fake target | `test_ignore_index_excluded`, `test_hand_example_cross_entropy` (mutants `s01`, `s02`) |
| 2. trusting targets | `-1` silently trains the last class (numpy indexes from the end) | `test_rejects_bad_inputs` (mutant `s07`) |
| 3. dividing by $K = 0$ | an all-padding batch gives `nan`, then every weight is `nan` | `test_all_ignored_is_zero` (mutant `s03`) |
| 4. $-\log(\mathrm{softmax}(z))$ and $-\log\sigma(x)$ | `inf` loss at logits of $10^3$ | `test_large_logits_stay_finite` (mutants `s04`, `s09`) |
| 5. smoothing over $V - 1$ classes, or summing $-\log p_j$ instead of averaging | disagrees with torch; the loss grows with the vocabulary size | `test_hand_example_label_smoothing` (mutants `s05`, `s06`) |

## 6. Where it's used next
| Forward | `L12.1` | Registered module relationship. |
| Forward | `L11.1` | Registered call site uses this module. |
| Forward | `L2.2` | Registered call site uses this module. |
| Forward | `L3.6` | Registered call site uses this module. |
| Forward | `L4.1` | Registered call site uses this module. |
| Forward | `L5.5` | Registered call site uses this module. |
| Forward | `L6.1` | Registered call site uses this module. |
| Forward | `L6.2` | Registered call site uses this module. |
| Forward | `L6.3` | Registered call site uses this module. |
| Forward | `L6.5` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | each loss is one `from_op` node |
| Back | `M09.2` | `log_softmax` for the stable forward |
| Back | `M08.3` | `cross_entropy_vjp`, the derivation the differential test compares with |
| Back | `M11.1` | $H(q, p)$, the definition of a smoothed row's loss |
| Forward | `L0.5` | the bigram and the MLPs of the training-loop tests minimize `cross_entropy`, `mse`, and `bce_with_logits` |

If you skip this module, `ol check L0.5` stops with `L0.5 needs L0.3`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| fused `cross_entropy` | Liger Kernel `fused_linear_cross_entropy` | fuses the output projection too: the `[N, V]` logits are never materialized, chunked over rows | `liger_kernel/ops/fused_linear_cross_entropy.py` |
| `ignore_index` | PyTorch `nll_loss` | the same semantics, plus per-class weights | `aten/src/ATen/native/LossNLL.cpp` |
| label smoothing | `torch.nn.CrossEntropyLoss(label_smoothing=...)` | combined with class weights and probability targets | `aten/src/ATen/native/Loss.cpp` |
| `bce_with_logits` | `torch.nn.functional.binary_cross_entropy_with_logits` | the same softplus form, vectorized on GPU | `aten/src/ATen/native/Loss.cpp` |
