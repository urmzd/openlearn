# Matrix Calculus and Autodiff

## Overview

- **Primary references**: Parr and Howard, [*The Matrix Calculus You Need for Deep Learning*](https://arxiv.org/abs/1802.01528) (free); Baydin, Pearlmutter, Radul, and Siskind, [*Automatic Differentiation in Machine Learning: a Survey*](https://www.jmlr.org/papers/v18/17-468.html) (JMLR 2018, free)
- **Supplementary**: Griewank and Walther, *Evaluating Derivatives* (SIAM, 2nd ed.); Petersen and Pedersen, [*The Matrix Cookbook*](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) (free); Karpathy, [micrograd](https://github.com/karpathy/micrograd) (free); Chen et al., [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174) (free)
- **Prerequisites**: [Calculus 1](../01-calculus-1/) (the chain rule), [Calculus 3](../04-calculus-3/) (gradients, Jacobians), [Linear Algebra](../03-linear-algebra/), [Discrete Math 2](../06-discrete-math-2/) (topological sort)
- **Estimated time**: 3 weeks at 10 to 12 h/week; in the course, Pass 2 (M08.1 to M08.3) and Pass 9 (M08.4)

## Key Takeaways

- **Autodiff is the chain rule applied to a program, not to a formula.** Forward mode carries a derivative alongside each value; reverse mode records the computation and walks it backward once.
- **Reverse mode costs about one extra forward pass per scalar output**, which is why every neural network trains with it: one loss, millions of parameters.
- **Every op needs one rule: its vector-Jacobian product (VJP).** The trace trick and matrix differentials derive the VJPs of matmul, softmax, LayerNorm, RMSNorm, and cross-entropy in a few lines each.
- **Gradients are checked, never trusted.** Central differences in float64 against the analytic VJP catch nearly every wrong rule.
- **Memory is the price of reverse mode.** Activation checkpointing trades recomputation for memory on a $\sqrt{n}$ schedule.

## How to Study

Read Parr and Howard first, deriving every identity by hand, then the Baydin survey sections 2 and 3. Build micrograd from memory once before the course modules. In the course, `S-M08` checks the derivations and `M08.1` to `M08.3` are the code: dual numbers, a scalar `Value`, then the closed-form VJPs that `L0.2` registers as ops.

---

# Concepts & Techniques

## Core Insight

A gradient is a linear map, and a program is a composition of small differentiable steps. Forward mode pushes a tangent through each step (a Jacobian-vector product); reverse mode pulls a cotangent back through each step (a vector-Jacobian product). Knowing the VJP of each primitive is enough to differentiate any program built from them, and knowing it in closed form is what makes it fast and stable.

## 1. Forward mode with dual numbers

**Key ideas**:
- **Dual number** $a + b\varepsilon$ with $\varepsilon^2 = 0$: evaluating $f(x + \varepsilon) = f(x) + f'(x)\varepsilon$ gives value and derivative together.
- **Cost**: one pass per input direction, so it suits few inputs and many outputs.
- **Call site**: `M08.1` is the derivative oracle for the elementwise ops of `L0.2`.

## 2. Reverse mode on a graph

**Key ideas**:
- **Tape**: record each op and its inputs; `backward` visits nodes in reverse topological order (`M06.1`) and accumulates $\bar{x} \mathrel{+}= \bar{y}\, \partial y / \partial x$.
- **Broadcasting**: the gradient of a broadcast input is the sum of the upstream gradient over the broadcast axes (`unbroadcast`).
- **Call site**: `M08.2` is the scalarized oracle for `L0.1` broadcasting backward.

## 3. Matrix differentials and closed-form VJPs

**Key ideas**:
- **Trace trick**: for scalar $L$, $dL = \operatorname{tr}(G^{\top} dY)$; with $Y = AB$, $dY = dA\,B + A\,dB$ gives $\bar{A} = G B^{\top}$, $\bar{B} = A^{\top} G$.
- **Softmax**: $\bar{x} = y \odot (g - \langle g, y \rangle)$; fused with cross-entropy the gradient is $\text{softmax}(z) - \text{onehot}(t)$.
- **Normalization**: LayerNorm and RMSNorm VJPs reuse the saved reciprocal standard deviation; beat 2 of `M08.3` defines both functions before deriving them.

## 4. Second order and memory

**Key ideas**:
- **Hessian-vector product** by finite differences of gradients, or forward-over-reverse, without forming the Hessian.
- **Checkpointing**: keep activations at segment boundaries and recompute inside a segment; $\sqrt{n}$ segments balance memory and time (`M08.4`, used by `L11.1`).

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `S-M08` | Matrix calculus problem set (VJP derivations, checked by SymPy) | solve | 2 |
| `M08.1` | Dual numbers, forward mode | build | 2 |
| `M08.2` | Scalar reverse mode (`Value`) | build | 2 |
| `M08.3` | Matrix differentials, trace trick, closed-form VJPs. Beat 2 defines LayerNorm and RMSNorm as functions before deriving their VJPs | build | 2 |
| `M08.4` | Hessian-vector products, recompute vs memory schedule | build | 9 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M08.1` | [Dual numbers and forward-mode autodiff](01-dual-numbers-forward-mode.md) | build | 2 |
| 2 | `M08.2` | [Scalar reverse mode: the Value graph](02-scalar-reverse-mode.md) | build | 2 |
| 3 | `M08.3` | [Matrix differentials, the trace trick, and closed-form VJPs](03-matrix-differentials-and-vjps.md) | build | 2 |
| 4 | `M08.4` | [Hessian-vector products and the recompute-versus-memory schedule](04-hessian-vector-products-and-recompute.md) | build | 9 |
| 5 | `S-M08` | [Matrix calculus problem set: differentials, trace trick, VJPs, SDPA Jacobian](90-problem-set.md) | solve | 2 |

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Calculus 3](../04-calculus-3/) | gradients, Jacobians, and the multivariable chain rule; `M04.1` gradcheck |
| [Numerical Methods and Floating Point](../09-numerical-methods-and-floating-point/) | stable softmax and log-sum-exp inside the VJPs |
| [Optimization](../10-optimization/) | the optimizers consume these gradients |
| [tinyllm Part 0](../../../courses/build-an-llm/p00-foundations/) | `L0.1` to `L0.3` turn these rules into the learner's autograd engine |
| [Deep Learning](../../machine-learning/02-deep-learning/) | backpropagation in the textbook setting |
