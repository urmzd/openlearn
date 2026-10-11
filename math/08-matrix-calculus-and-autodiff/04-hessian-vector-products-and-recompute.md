<!-- ol:module M08.4 -->
# Hessian-vector products and the recompute-versus-memory schedule

## Overview

| | |
|---|---|
| **Module** | `M08.4` · build · Python · Pass 9 · 3 h |
| **You build** | `python/tinyllm/autograd/hvp.py`: `hvp_fd`, `hessian_fd`, `checkpoint_cost`, `min_checkpoint_memory`, `checkpoint_schedule` |
| **Contract** | [`course/contracts/py/tinyllm/autograd/hvp.pyi`](../../course/contracts/py/tinyllm/autograd/hvp.pyi) |
| **Tests** | `course/tests/M08.4/` (what they check: section 4) |
| **Needs** | nothing to build first · reading: `M08.3` closed-form VJPs (the golden tests differentiate its cross-entropy gradient), `S-M08` forward versus reverse cost, `M04.2` the Hessian and the second-derivative test |
| **Used by** | `L11.1` activation checkpointing plans its segments with `checkpoint_schedule` · later: `M10.5` (optional) estimates the top Hessian eigenvalue with `hvp_fd` |
| **Milestone** | `MS-P9` (Pass 9 gate: the math of the pass checks green, then the capstone trains with checkpointing) |
| **Optional depth** | Pearlmutter, ["Fast Exact Multiplication by the Hessian"](https://doi.org/10.1162/neco.1994.6.1.147) (Neural Computation, 1994); Chen, Xu, Zhang, and Guestrin, [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174) (2016); Griewank and Walther, *Evaluating Derivatives*, chapter 12 (checkpointing) |

## Key Takeaways

- The Hessian-vector product $Hv$ is the derivative of the gradient along $v$, so two gradient calls give it to $O(\epsilon^2)$ without forming the $n \times n$ Hessian (`test_hand_example_quartic`, `test_error_shrinks_as_eps_squared`, `test_matches_torch_hvp`).
- On a quadratic the central difference is exact at every $\epsilon$, because the gradient is affine (`test_quadratic_hvp_is_exact`).
- Reverse mode keeps one saved input per layer. Recomputing segments trades forward time for memory: the peak is $\max_i (i + s_i)$ and the extra work is every layer outside the last segment (`test_checkpoint_cost_extremes`).
- Uniform segments of $\sqrt n$ layers peak near $2\sqrt n$; shrinking segments (sizes $P, P-1, \dots, 1$) reach the optimum $P \approx \sqrt{2n}$, and under a budget the best schedule is found greedily (`test_min_memory_is_triangular`, `test_schedule_is_optimal`).

## How to work this chapter

```bash
ol start M08.4              # stubs hvp.py into your repo, contract alongside
ol tests M08.4              # read the test catalog first: rung R0, you write no tests here
ol check M08.4              # exit code is the verdict
ol diff  M08.4              # after passing: your code against the reference
```

---

## 1. Why now

Pass 9 trains the capstone, an 8-layer Llama with a 512-token context, on a laptop. Its backward pass needs every layer's saved activations at once: at batch 16 that is several hundred megabytes in float32 before the logits, and the activations, not the 10M weights, are what runs the process out of memory. `L11.1` fixes it with activation checkpointing: keep only some layer inputs, recompute the rest during backward. Which layers to keep is a counting problem with an exact answer, and you need that answer before you write the mechanism. The same pass is where curvature questions come up ("is my learning rate past $2/\lambda_{\max}$?", `M10.5`), and the tool for those is the Hessian-vector product, which never forms the Hessian. This module gives you both: `hvp_fd` and the schedule functions `L11.1` calls.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f: \mathbb{R}^n \to \mathbb{R}$ | a scalar loss of $n$ parameters | function |
| $x \in \mathbb{R}^n$ | the point (parameters) | `float64[...]` |
| $g(x) = \nabla f(x)$ | the gradient (`grad_fn`) | same shape as $x$ |
| $H(x) = \nabla^2 f(x)$ | the Hessian: $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ | $n \times n$, symmetric |
| $v$ | a direction | same shape as $x$ |
| $\epsilon$ | the finite-difference step along $v$ (`eps`) | float |
| $D^k g(x)[v, \dots, v]$ | the $k$-th directional derivative of $g$ along $v$ (a derivative of order $k + 1$ of $f$) | same shape as $x$ |
| $e_j$ | the $j$-th unit vector | `float64[n]` |
| $n$ (`n_layers`) | layers in a stack | `int` |
| $k$ | segments in a schedule | `int` |
| $s_i$ | layers in segment $i$ ($i = 0, \dots, k-1$), $\sum_i s_i = n$ | `int` |
| `starts` | first layer of each segment: $0, s_0, s_0 + s_1, \dots$ | `list[int]` |
| $B$ (`mem_budget_layers`) | the memory budget, in saved layer inputs | `int` |
| $P$ | the peak number of saved layer inputs held at once | `int` |

**The Hessian-vector product is a directional derivative of the gradient.** Taylor-expand the gradient around $x$ along $v$:

$$g(x + \epsilon v) = g(x) + \epsilon\, H v + \tfrac{\epsilon^2}{2} D^2 g(x)[v, v] + \tfrac{\epsilon^3}{6} D^3 g(x)[v, v, v] + O(\epsilon^4).$$

The first-order term is $\epsilon H v$, so $Hv = \lim_{\epsilon \to 0} (g(x + \epsilon v) - g(x))/\epsilon$. That limit is a derivative, and you already know how to take derivatives numerically.

**Central differences cancel the even terms.** Subtract the expansion at $-\epsilon$ from the one at $+\epsilon$: $g(x)$ and the $\epsilon^2$ term cancel, and

$$\frac{g(x + \epsilon v) - g(x - \epsilon v)}{2\epsilon} = Hv + \frac{\epsilon^2}{6} D^3 g(x)[v, v, v] + O(\epsilon^4).$$

The one-sided quotient keeps the $\tfrac{\epsilon}{2} D^2 g(x)[v,v]$ term: halving $\epsilon$ halves its error, while halving it in the central quotient quarters the error. The step cannot be made arbitrarily small either: each gradient carries rounding error of about $u\,|g|$ ($u = 2^{-53} \approx 1.1 \times 10^{-16}$ in float64), and dividing by $2\epsilon$ magnifies it to about $u\,|g|/\epsilon$. Balancing $\epsilon^2$ against $u/\epsilon$ puts the best step near $u^{1/3} \approx 5 \times 10^{-6}$ times the scale of the problem; the default $10^{-4}$ gives about $10^{-8}$ relative accuracy on smooth losses and is far from the rounding floor.

**Quadratics are exact.** For $f(x) = \tfrac12 x^\top A x + b^\top x$, $g(x) = \tfrac12(A + A^\top) x + b$ is affine, $D^3 g = 0$, and the central difference returns $\tfrac12 (A + A^\top) v$ for every $\epsilon$ (up to rounding). A non-symmetric $A$ shows that the Hessian is the symmetric part.

**Cost.** Two gradient calls, each about one forward and one backward pass, for one $Hv$. Forming $H$ would take $n$ of them and $n^2$ numbers of memory: for a 10M-parameter model, $10^{14}$ entries. Everything that needs curvature uses products instead: Newton-CG solves with $Hv$, and the power iteration $v \leftarrow Hv / \lVert Hv \rVert$ finds the top eigenvalue $\lambda_{\max}$ (`M10.5`). For small $n$, `hessian_fd` builds $H$ column by column from $H e_j$ and returns $(H + H^\top)/2$: the columns carry $O(\epsilon^2)$ errors that differ between $H_{ij}$ and $H_{ji}$, and code that assumes symmetry (eigenvalues, Cholesky) must get it exactly. Pearlmutter's $\mathcal{R}$-operator (forward mode over reverse mode) computes $Hv$ exactly at the same cost, which is what frameworks do; it needs an autograd that differentiates its own backward pass, which yours does not.

**Reverse mode stores activations.** A stack of layers $y_{l+1} = f_l(y_l)$, $l = 0, \dots, n-1$, computes the VJP of layer $l$ from its saved input $y_l$ (a matmul's $\bar W = \bar y^\top x$ needs $x$). Count memory in units of one saved layer input. A plain forward pass keeps all $n$, and backward frees them from the top down: the peak is $n$.

**Segments and recomputation.** Split the stack into $k$ consecutive segments of $s_0, \dots, s_{k-1}$ layers. Every segment but the last runs forward keeping only its input (one unit) and discards its internal activations; the last segment runs normally and keeps its $s_{k-1}$. During backward, segment $i$ is rerun from its saved input to rebuild its $s_i$ activations, then backpropagated and freed. This is `torch.utils.checkpoint.checkpoint_sequential`. Memory while segment $i$ is backpropagated: the inputs of segments $0, \dots, i-1$ (still needed later) plus segment $i$'s $s_i$ activations (the first of which is its saved input). The end of the forward pass is the case $i = k - 1$. So

$$P = \max_{0 \le i < k} (i + s_i), \qquad \text{recomputed layers} = n - s_{k-1}.$$

Both extremes cost $n$: one segment ($k = 1$) is no checkpointing, and one layer per segment holds $n$ inputs while recomputing $n - 1$ layers for nothing.

**Uniform segments give $\sqrt n$.** With $k$ segments of $s$ layers ($ks = n$), $P = (k - 1) + s = n/s + s - 1$. Setting the derivative $-n/s^2 + 1$ to zero gives $s = \sqrt n$ and $P \approx 2\sqrt n - 1$: the classic sublinear-memory result of Chen et al.

**Shrinking segments do better.** The term $i + s_i$ grows with $i$, so later segments should be smaller. If every $i + s_i \le P$, then $s_i \le P - i$, and $k$ segments hold at most

$$C(k, P) = \sum_{i=0}^{k-1} (P - i) = kP - \frac{k(k-1)}{2}$$

layers. That is largest at $k = P$ (the segment sizes $P, P-1, \dots, 1$), where it is the triangular number $P(P+1)/2$. So the least peak of any schedule is the least $P$ with $P(P+1)/2 \ge n$, about $\sqrt{2n}$: for $n = 16$, $P = 6$ against 7 for uniform segments of 4. Compute it with integers (`math.isqrt`): a float square root is wrong by one for large $n$.

**The budgeted schedule.** Given a budget $B \ge P_{\min}$, minimize recomputation, that is, maximize the last segment. With $k$ segments the last holds at most $\min(B - k + 1,\, n - k + 1)$ layers (its $i + s_i \le B$, and every other segment needs at least one), and the other $k-1$ segments must cover the rest within $C(k-1, B)$. That bound shrinks as $k$ grows, so the fewest feasible segments is optimal. Fill the first $k - 1$ segments greedily, largest first, $s_i = \min(B - i,\ \text{rest} - (k - 2 - i))$, leaving at least one layer for each later segment; the order fixes one answer among ties, so your trainer and the reference checkpoint the same layers. A brute force over all $2^{n-1}$ schedules agrees for every $n \le 10$ (`test_schedule_is_optimal`).

## 3. Worked example by hand

**A quartic.** $f(x) = x^4/24$ in one dimension: $g(x) = x^3/6$, $H(x) = g'(x) = x^2/2$, $g''(x) = x$, and $g'''(x) = 1$. At $x = 1$, $v = 1$, $\epsilon = 0.1$:

| quantity | value |
|---|---|
| $g(1.1) = 1.331/6$, $g(0.9) = 0.729/6$ | $0.2218333$, $0.1215$ |
| central: $(1.331 - 0.729)/(6 \cdot 0.2)$ | $0.602/1.2 = 0.5016667$ |
| exact $H v$ | $0.5$ |
| error, predicted $\epsilon^2 v^3 g'''/6$ | $0.01/6 = 0.0016667$ (exact here: $g$ is cubic, so the series stops) |
| one-sided: $(1.331 - 1)/(6 \cdot 0.1)$ | $0.5516667$, error $0.0517 \approx \tfrac{\epsilon}{2} g''(1)$, thirty times larger |
| forgetting the 2: $0.602/0.6$ | $1.0033$ |


**A six-layer stack.** $n = 6$:

| schedule (`starts`) | sizes $s_i$ | $i + s_i$ | peak $P$ | recomputed |
|---|---|---|---|---|
| `[0]` | 6 | 6 | 6 | 0 |
| `[0, 1, 2, 3, 4, 5]` | 1, 1, 1, 1, 1, 1 | 1, 2, 3, 4, 5, 6 | 6 | 5 |
| `[0, 2, 4]` (uniform, $s \approx \sqrt 6$) | 2, 2, 2 | 2, 3, 4 | 4 | 4 |
| `[0, 3, 5]` | 3, 2, 1 | 3, 3, 3 | 3 | 5 |
| `[0, 3]` | 3, 3 | 3, 4 | 4 | 3 |

The least peak is 3, because $3 \cdot 4/2 = 6 \ge 6 > 2 \cdot 3/2$, and `[0, 3, 5]` reaches it. With a budget of 4: one segment needs 6 units, too many; two segments can keep a last segment of $\min(4 - 1, 6 - 1) = 3$ layers if the first covers the other 3 within $C(1, 4) = 4$, which it does. So `checkpoint_schedule(6, 4) == [0, 3]`: the same peak as uniform segments and one recomputed layer fewer. These numbers are the first two tests, `test_hand_example_quartic` and `test_hand_example_schedule`.

## 4. The interface

```python
# python/tinyllm/autograd/hvp.py
def hvp_fd(grad_fn, x, v, eps: float = 1e-4) -> NDArray          # (g(x + eps v) - g(x - eps v)) / (2 eps)
def hessian_fd(grad_fn, x, eps: float = 1e-4) -> NDArray          # 1-D x: columns H e_j, then (H + H^T) / 2
def checkpoint_cost(n_layers: int, starts) -> tuple[int, int]     # (peak, recomputed)
def min_checkpoint_memory(n_layers: int) -> int                   # least P with P (P + 1) / 2 >= n
def checkpoint_schedule(n_layers: int, mem_budget_layers: int) -> list[int]   # segment starts
```

`hvp_fd` calls `grad_fn` exactly twice, each time with a new float64 array, and copies each result, so a gradient that returns its own argument is safe. The step is $\epsilon v$, not normalized: scale $v$ yourself if its norm is far from 1. Shape mismatches, a non-positive or non-finite step, and an infeasible budget raise `ValueError`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_quartic` | unit | the section 3 quartic numbers, error $\epsilon^2/6$ | you and the test agree on the formula |
| `test_hand_example_schedule` | unit | the section 3 schedule table | `L11.1` checkpoints exactly these layers |
| `test_quadratic_hvp_is_exact` | property | $\tfrac12 (A + A^\top) v$ at three step sizes, $v$ not unit | the step is $\epsilon v$ |
| `test_error_shrinks_as_eps_squared` | property | halving $\epsilon$ divides the error by 4 | a central difference, not one-sided |
| `test_matches_torch_hvp` | golden | torch's exact double-backward $Hv$ of a cross-entropy loss whose gradient is `M08.3`'s rule | the curvature `M10.5` estimates |
| `test_hessian_fd_matches_torch` | golden | a dense $6 \times 6$ Hessian, exactly symmetric | small-model Newton steps |
| `test_hessian_is_exactly_symmetric` | unit | Rosenbrock at $(-1.2, 1)$: $H = H^\top$ bit for bit, close to the analytic Hessian | eigenvalue code assumes symmetry |
| `test_grad_fn_may_return_its_argument` | unit | a gradient that returns its input; two calls, `x` untouched | cheap gradients alias |
| `test_hvp_rejects_bad_inputs` | boundary | shape and step errors raise; the default step is accurate | caller bugs surface |
| `test_checkpoint_cost_extremes` | unit | no checkpointing and all checkpoints both peak at $n$; invalid schedules raise | the cost model itself |
| `test_min_memory_is_triangular` | unit | $P(P+1)/2 \ge n > (P-1)P/2$ up to $n = 2^{61} - 1$ | integer arithmetic at scale |
| `test_schedule_is_optimal` | property | within budget and as few recomputed layers as brute force, $n \le 10$ | the schedule is the best one |
| `test_schedule_ties_go_to_the_largest_first_segment` | unit | the tie rule on four cases | reference and learner agree |
| `test_schedule_rejects_impossible_budget` | boundary | a budget below the least peak raises | no silent overrun |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a one-sided difference | error $O(\epsilon)$: 0.5517 instead of 0.5017 in section 3 | `test_hand_example_quartic` (mutant `s01`) |
| 2. dividing by $\epsilon$ instead of $2\epsilon$ | every $Hv$ twice too large | `test_hand_example_quartic` (mutant `s02`) |
| 3. reusing one buffer for $x \pm \epsilon v$ | a gradient that aliases its input changes under you; $Hv = 0$ | `test_grad_fn_may_return_its_argument` (mutant `s03`) |
| 4. not symmetrizing the finite-difference Hessian | $H_{ij} \ne H_{ji}$ by $O(\epsilon^2)$ | `test_hessian_is_exactly_symmetric` (mutant `s04`) |
| 5. peak as (segments $-$ 1) + (largest segment) | overcounts when the largest segment comes first; the optimizer picks worse schedules | `test_hand_example_schedule` (mutant `s05`) |
| 6. the uniform $\sqrt n$ rule whatever the budget | over budget when it is tight, needless recomputation when it is loose | `test_hand_example_schedule` (mutant `s09`) |
| 7. a float `ceil(sqrt(2n))` for the least peak | one too many at $n = 6$, wrong at large $n$ | `test_min_memory_is_triangular` (mutant `s07`) |
| 8. counting the last segment as recomputed | recomputation off by $s_{k-1}$ | `test_checkpoint_cost_extremes` (mutant `s06`) |
| 9. a budget check off by one | peaks one unit over the budget | `test_schedule_rejects_impossible_budget` (mutant `s10`) |
| 10. normalizing $v$ inside `hvp_fd` | returns $Hv/\lVert v \rVert$ | `test_quadratic_hvp_is_exact` (mutant `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M08.3` | its cross-entropy VJP, $X^\top(\mathrm{softmax} - \mathrm{onehot})/n$, is the gradient whose $Hv$ the golden tests compare with torch |
| Back | `S-M08` | the forward versus reverse cost counts behind "two gradients per $Hv$" |
| Back | `M04.2` | the Hessian and the second-derivative test |
| Forward | `L11.1` | `checkpoint_sequential` runs your segments with `checkpoint_schedule` and recomputes them in backward |
| Forward | `M10.5` | (optional) power iteration on `hvp_fd` estimates $\lambda_{\max}$ to compare the learning rate with $2/\lambda_{\max}$ |

If you skip this module, `ol check L11.1` stops with `L11.1 needs M08.4`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `hvp_fd` | `torch.autograd.functional.hvp`, `torch.func.jvp(grad(f))` | exact forward-over-reverse products (Pearlmutter's $\mathcal{R}$-operator) at the cost of two backward passes | `torch/autograd/functional.py` |
| `hessian_fd` | PyHessian | Hessian spectra of real networks by power iteration and stochastic Lanczos, from $Hv$ only | `pyhessian/hessian.py` |
| `checkpoint_schedule` | `torch.utils.checkpoint.checkpoint_sequential` | uniform segments, RNG state replay, non-reentrant saved-tensor hooks | `torch/utils/checkpoint.py` |
| the cost model | Checkmate (Jain et al., 2020) | optimal rematerialization for arbitrary graphs as an integer program | `checkmate` repository, `checkmate/core/solvers/` |
