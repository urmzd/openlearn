<!-- ol:module L0.2 -->
# The op library and gradcheck_all

## Overview

| | |
|---|---|
| **Module** | `L0.2` · build · Python · Pass 2 · 5 to 7 h |
| **You build** | `python/tinyllm/autograd/functional.py`, imported as `F`: 25 ops (activations, reductions, shape, selection, the softmax family, dropout, matmul) and `gradcheck_all()` |
| **Contract** | [`course/contracts/py/tinyllm/autograd/functional.pyi`](../../../../course/contracts/py/tinyllm/autograd/functional.pyi) |
| **Tests** | `course/tests/L0.2/` (what they check: section 4), golden values from torch 2.14 in `course/fixtures/L0.2/ops_torch.npz` |
| **Needs** | `L0.1` `Tensor` and `from_op` · `M01.3` activations and their derivatives · `M09.2` stable softmax, log-softmax, logsumexp · `M08.3` the softmax VJPs and `unbroadcast` · `M08.1` dual numbers · `M04.1` `gradcheck` · `M06.3` PCG32 (or `--ref-deps`) |
| **Used by** | `L0.4` layers are written in `F` · `L0.5` `BigramLogits` is an `F.embedding` · `L0.6` the checkpoint tests compute their loss with `F` · later: `L2.2`, `L3.1`, `L3.2`, `L3.3`, `L3.4`, `L3.6`, `L4.1`, `L4.2`, `L4.3`, `L5.3`, `L5.4`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5`, `L6.6`, `L7.1`, `L7.2`, `L7.3`, `L7.5`, `L7.6`, `L7.8`, `L7.9`, `L13.1`, `L13.3`, `L13.5`, `L14.2`  · `L14.3` Whisper decoding |
| **Milestone** | `MS-L0` (step 1: `{tinyllm} gradcheck --suite all`) |
| **Optional depth** | Griewank and Walther, *Evaluating Derivatives* (2nd ed.), ch. 3 and 4; the PyTorch `derivatives.yaml` file, read as a table of VJPs |

## Key Takeaways

- Every op is one numpy forward plus one closed-form VJP joined by `from_op`; nothing in the library calls another op's backward (`test_gradcheck_each_op`, `test_matches_torch`).
- Reductions put the reduced axes back before broadcasting the gradient, a maximum shares its gradient among ties, and a mean divides by the reduced count only (`test_max_ties_share_the_gradient`, `test_var_correction`).
- Selection ops route gradient by addition: a token or index used twice gets both gradients, and an out-of-range id is an error where it enters (`test_embedding_repeated_ids_accumulate`, `test_embedding_rejects_bad_ids`).
- The softmax family's VJPs are computed from the stable outputs, so logits of 1000 and fully masked rows give finite gradients (`test_logsumexp_no_overflow`, `test_softmax_fully_masked_row`).
- `gradcheck_all` checks every op against central differences with a random output weighting, plus forward-mode duals for the elementwise ops, and it fails on a planted 0.1% or 1e-7 error (`test_gradcheck_all_catches_a_wrong_op`).

## How to work this chapter

```bash
ol start L0.2              # stubs functional.py into your repo
ol tests L0.2              # read the test catalog first: rung R0
ol check L0.2              # exit code is the verdict
ol check L0.2 --ref-deps   # only if a math dependency is not passing yet
ol diff  L0.2              # after passing: your code against the reference
```

Your CLI gains its first Pass 2 verb, `{tinyllm} gradcheck --suite all` ([`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md), fixed by `MS-L0`): call `F.gradcheck_all(rtol=1e-5)`, print one line per op, and end with `{"suite": "all", "checks": N, "failed": K, "max_rel_err": E, "worst": "<op>"}`; exit 0 only when `failed` is 0.

---

## 1. Why now

`L0.1` gave you a `Tensor` that knows `+ - * / @ **` and indexing. A model needs more: a nonlinearity between layers, a softmax over the vocabulary, an embedding lookup for token ids, a mean for LayerNorm, a dropout mask. Written inline in each model, each of those is a chance to get a VJP wrong silently, because a wrong gradient still trains, just badly. This module puts every op in one library with one VJP each, checks each against torch and against numerical differentiation, and gives your CLI the `gradcheck` verb `MS-L0` runs first. From here on, model code (`L0.4`, `L2`, `L3`, `L5`) composes these ops and never writes a backward pass.

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

**Inverted dropout.** During training each element is kept with probability $1 - p$ and scaled by $1/(1 - p)$, so the expected output equals the input and evaluation needs no rescaling: in eval mode, or with $p = 0$, dropout returns its input object and draws nothing. The mask uses one uniform per element in C order, $\text{keep}_k = [u_k \ge p]$, from the PCG32 you pass in. The draw count is fixed (exactly `x.size` draws per call), which is what lets a resumed run (`L0.6`) reproduce the same masks. $p = 1$ drops everything (no division by zero).

**gradcheck_all: a test that can fail.** For each op, `gradcheck_all` draws float64 inputs from `PCG32(0)`, keeps them away from kinks (relu at 0, max ties), and compares the analytic gradient of $f(x) = \sum y \odot w$ with central differences $\big(f(x + \epsilon e_i) - f(x - \epsilon e_i)\big)/2\epsilon$ through `M04.1`'s `gradcheck` at `rtol`. The weights $w$ are random on purpose: $\sum_i \mathrm{softmax}(x)_i = 1$ for every $x$, so the gradient of the plain sum is zero and *any* softmax VJP, right or wrong, would pass. Central differences resolve relative errors near $10^{-5}$; a smaller error in an elementwise derivative is caught by a second check, forward-mode dual numbers (`M08.1`), which are exact to rounding and flag a gap above $10^{-9}$. Ops are looked up in the module when the check runs, so a monkeypatched op is the one checked.

## 3. Worked example by hand

**Softmax and its VJP.** Take $x = [0, \ln 2, \ln 3]$. Then $e^{x} = [1, 2, 3]$, the sum is 6, and $y = [1/6, 2/6, 3/6]$. With the upstream gradient $\bar{y} = [1, 0, 0]$ (the loss looks only at the first output):

1. $\langle \bar{y}, y \rangle = 1 \cdot 1/6 = 1/6$.
2. $\bar{y} - 1/6 = [5/6, -1/6, -1/6]$.
3. Multiply by $y$: $\bar{x} = [5/36, -2/36, -3/36]$.

The entries sum to zero: raising every logit by the same amount leaves softmax unchanged, so no direction along $(1, 1, 1)$ can change the loss. Check $\bar{x}_0$ by perturbation: $y_0 = e^{x_0}/(e^{x_0} + 5)$, whose derivative at $x_0 = 0$ is $5/36$.

**A tie in max.** $\max([1, 3, 3]) = 3$ is reached twice. torch's `amax` gives each maximal entry half: $\bar{x} = [0, 0.5, 0.5]$ for $\bar{y} = 1$.

**Variance.** For $[1, 2, 3, 4]$: $\mu = 2.5$, squared deviations $2.25 + 0.25 + 0.25 + 2.25 = 5$, so `var` is $5/4 = 1.25$ with correction 0 and $5/3$ with correction 1.

These are `test_hand_example_softmax_backward`, `test_max_ties_share_the_gradient`, and `test_var_correction`.

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

`sum`, `max`, and `mean` shadow Python builtins inside the module; use `builtins.max` where you need the builtin. Float32 inputs give float32 outputs and gradients (cast derivative arrays and masks to the input's dtype).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_softmax_backward` | unit | section 3: $y = [1/6, 2/6, 3/6]$, $\bar{x} = [5/36, -2/36, -3/36]$ | you and the test agree on the softmax VJP |
| `test_max_ties_share_the_gradient` | boundary | `max([1, 3, 3])` gives $[0, 0.5, 0.5]$ | ReLU-max pooling and torch parity |
| `test_matches_torch` | golden | value and gradients equal torch 2.14 on 45 cases | models ported from torch (`L5` to `L7`) |
| `test_gradcheck_each_op` | gradcheck | every VJP against the frozen central differences, float64 | a wrong VJP trains badly, not visibly |
| `test_float32_in_float32_out` | boundary | float32 in, float32 out for twelve ops, and a float32 gradient through dropout | Python agrees with the float32 C kernels (`L9`) |
| `test_relu_derivative_at_zero_is_zero` | boundary | $\mathrm{relu}'(0) = 0$ | torch's convention |
| `test_embedding_repeated_ids_accumulate` | boundary | a repeated token gets both rows of gradient | every embedding table |
| `test_gather_repeated_indices_accumulate` | boundary | an index read twice gets gradient 2 | the cross-entropy gather in `L0.3` |
| `test_embedding_rejects_bad_ids` | boundary | `-1`, `n`, and float ids are a `ValueError` | tokenizer bugs fail where they enter |
| `test_dropout_eval_is_identity` | unit | eval mode returns the input object, draws nothing | evaluation is deterministic |
| `test_dropout_mask_and_scale` | statistical | about 75% kept at $p = 0.25$, survivors scaled by $4/3$ | the expected output is the input |
| `test_dropout_draw_accounting` | unit | the mask is $u \ge p$ for the same-seed uniforms, exactly `x.size` draws | a resumed run replays the masks (`L0.6`) |
| `test_dropout_p_bounds` | boundary | $p = 1$ gives zeros; $p \notin [0, 1]$ is an error | config errors fail early |
| `test_softmax_fully_masked_row` | boundary | an all-masked row gives zeros and a zero gradient | padded rows in attention (`L5.2`) |
| `test_logsumexp_no_overflow` | boundary | `logsumexp([1000, 1000])` and its gradient $[0.5, 0.5]$ | trained logits are large |
| `test_var_correction` | unit | 1.25 and $5/3$; $n \le c$ is an error | LayerNorm's population variance (`L0.4`) |
| `test_where_and_masked_fill_route_gradients` | unit | gradients go to the chosen operand; filled positions get none | attention masks (`L5`) |
| `test_gelu_rejects_unknown_approximation` | boundary | only `"none"` and `"tanh"` | HF configs name exactly these |
| `test_ops_build_no_graph_under_no_grad` | property | every op returns a constant under `no_grad` | evaluation memory |
| `test_gradcheck_all_reports_every_op` | unit | one ok report per op, 33 names | `MS-L0`'s `gradcheck --suite all` |
| `test_gradcheck_all_catches_a_wrong_op` | boundary | a 0.1% softmax error and a $10^{-7}$ sigmoid error are both flagged | a checker that always passes is worse than none |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `out[ids] = g` in `embedding` or `gather` | frequent tokens train slower than rare ones | `test_embedding_repeated_ids_accumulate` (mutant `s06`), `test_gather_repeated_indices_accumulate` (mutant `s07`) |
| 2. trusting ids | `-1` silently trains the last row of the table | `test_embedding_rejects_bad_ids` (mutant `s08`) |
| 3. ties in `max`: every maximal entry gets the full gradient, or only the first | gradients double, or disagree with torch | `test_max_ties_share_the_gradient` (mutants `s03`, `s04`) |
| 4. the `logsumexp` VJP as `exp(x) / exp(y)` | `inf / inf = nan` at logit 1000 | `test_logsumexp_no_overflow` (mutant `s13`) |
| 5. feeding the softmax VJP the input $x$ instead of the output $y$ | wrong gradients everywhere, `nan` on masked rows | `test_softmax_fully_masked_row`, `test_hand_example_softmax_backward` (mutant `s01`) |
| 6. a `gradcheck_all` that cannot fail (no dual check, or a loose tolerance) | `MS-L0` step 1 passes with a broken op | `test_gradcheck_all_catches_a_wrong_op` (mutants `s19`, `s20`) |
| 7. float64 leaking from a mask or a derivative array | a float32 model becomes float64, twice the memory | `test_float32_in_float32_out` (mutant `m04`) |

## 6. Where it's used next
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L2.2` | Registered call site uses this module. |
| Forward | `L3.1` | Registered call site uses this module. |
| Forward | `L3.2` | Registered call site uses this module. |
| Forward | `L3.3` | Registered call site uses this module. |
| Forward | `L3.4` | Registered call site uses this module. |
| Forward | `L3.6` | Registered call site uses this module. |
| Forward | `L4.1` | Registered call site uses this module. |
| Forward | `L4.2` | Registered call site uses this module. |
| Forward | `L4.3` | Registered call site uses this module. |
| Forward | `L5.3` | Registered call site uses this module. |
| Forward | `L5.4` | Registered call site uses this module. |
| Forward | `L5.5` | Registered call site uses this module. |
| Forward | `L6.1` | Registered call site uses this module. |
| Forward | `L6.2` | Registered call site uses this module. |
| Forward | `L6.3` | Registered call site uses this module. |
| Forward | `L6.5` | Registered call site uses this module. |
| Forward | `L6.6` | Registered call site uses this module. |
| Forward | `L7.1` | Registered call site uses this module. |
| Forward | `L7.2` | Registered call site uses this module. |
| Forward | `L7.3` | Registered call site uses this module. |
| Forward | `L7.5` | Registered call site uses this module. |
| Forward | `L7.6` | Registered call site uses this module. |
| Forward | `L7.8` | Registered call site uses this module. |
| Forward | `L7.9` | Registered call site uses this module. |

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

If you skip this module, `ol check L0.4` stops with `L0.4 needs L0.2`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one VJP per op | PyTorch `derivatives.yaml` | a generated backward for about 2,000 ops, double-backward, forward-mode formulas | `tools/autograd/derivatives.yaml` |
| `gradcheck_all` | `torch.autograd.gradcheck`, `gradgradcheck` | complex inputs, sparse layouts, second derivatives, fast mode with random projections | `torch/autograd/gradcheck.py` |
| `dropout` with an explicit generator | `torch.nn.functional.dropout`, JAX `random.bernoulli` | Philox counter-based RNG on GPU, fused into matmul epilogues | `aten/src/ATen/native/Dropout.cpp` |
| stable softmax VJP | fused softmax and log-softmax kernels | one pass over the row, online max (`L9.2`) | `aten/src/ATen/native/SoftMax.cpp` |
