# Optimization

## Overview

- **Primary references**: Boyd and Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/) (free PDF); Nocedal and Wright, *Numerical Optimization* (Springer, 2nd ed.)
- **Supplementary**: Kingma and Ba, [*Adam*](https://arxiv.org/abs/1412.6980) (free); Loshchilov and Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) (free); Goh, [*Why Momentum Really Works*](https://distill.pub/2017/momentum/) (free); Cohen et al., [*Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*](https://arxiv.org/abs/2103.00065) (free); Jordan, [*Muon*](https://kellerjordan.github.io/posts/muon/) (free)
- **Prerequisites**: [Calculus 3](../04-calculus-3/) (gradients, Hessians), [Linear Algebra](../03-linear-algebra/) (eigenvalues), [Precalculus](../00-precalculus/) (geometric series for schedules)
- **Estimated time**: 3 weeks at 10 to 12 h/week; in the course, Pass 2 (M10.1 to M10.4), with optional curvature and Muon in Pass 9

## Key Takeaways

- **Gradient descent converges at a rate set by the condition number** $\kappa = L/\mu$: the gap shrinks by about $(1 - 1/\kappa)$ per step on a quadratic.
- **Momentum and Nesterov** average past gradients to move faster along shallow directions; **Adam** adds a per-parameter scale from the second moment, with bias correction for early steps.
- **Weight decay is not L2 regularization under Adam**: AdamW decouples it from the adaptive scale.
- **The schedule is part of the optimizer.** Warmup, cosine or warmup-stable-decay, and gradient clipping decide whether a run trains at all.
- **Optimizers are stateful**: resuming a run bitwise needs the moments, the step count, and the schedule position in the checkpoint.

## How to Study

Read Boyd and Vandenberghe chapters 2, 3, and 9 for convexity and descent methods, then the Adam and AdamW papers in full. Plot every optimizer on a 2D ill-conditioned quadratic before you trust it on a network. In the course, `M10.1` to `M10.4` are built in Pass 2 and train every model after; `S-M10a` checks the derivations; the optional `M10.5` and `M10.6` are C1 diagnostics and an A/B experiment.

---

# Concepts & Techniques

## Core Insight

Training is minimization of an average loss with noisy gradients. Everything in an optimizer answers one of three questions: which direction (the gradient, smoothed by momentum), how far in each coordinate (the learning rate, scaled per parameter by Adam), and how that step size changes over time (the schedule). Convex theory gives the vocabulary and the rates; neural networks break its assumptions but keep its intuitions.

## 1. Convexity, smoothness, and gradient descent

**Key ideas**:
- **Definitions**: $f$ is $L$-smooth if its gradient is $L$-Lipschitz and $\mu$-strongly convex if $f(y) \ge f(x) + \nabla f(x)^{\top}(y - x) + \tfrac{\mu}{2}\|y - x\|^2$.
- **Armijo line search** backtracks until the decrease is at least $c\,\alpha\,\nabla f^{\top} d$ (`M10.1`).

## 2. The `Optimizer` protocol: SGD, momentum, AdamW

**Key ideas**:
- **Protocol**: `step`, `zero_grad`, `state_dict`, `load_state_dict`, shared by every optimizer the course trains with (`M10.2`, `M10.3`).
- **AdamW**: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$, step $\eta\,\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \eta\lambda\theta$.

## 3. Schedules and clipping

**Key ideas**:
- **Cosine with warmup**, **warmup-stable-decay** (the C1 schedule), and **Noam** (the 2017 transformer), plus global-norm gradient clipping (`M10.4`).

## 4. Curvature and beyond (optional)

**Key ideas**:
- **Top Hessian eigenvalue** by power iteration on Hessian-vector products; gradient descent is stable only while $\eta < 2/\lambda_{\max}$ (`M10.5`).
- **Muon** orthogonalizes the momentum of 2D weights with Newton-Schulz iterations (`M10.6`).
- **Lagrangians and KKT** conditions are solve-only (`M10.7`); they derive the closed form behind DPO in `L12.2`.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `S-M10a` | Descent methods and optimizers problem set | solve | 2 |
| `S-M10b` | Lagrangians and KL-regularized objectives problem set | solve | 10, optional |
| `M10.1` | Convexity, L-smoothness, gradient descent, Armijo line search. Beat 2 defines the condition number κ = L/μ (M09.3 later generalizes it to matrices) | build | 2 |
| `M10.2` | Optimizer protocol, SGD, momentum, Nesterov, weight decay | build | 2 |
| `M10.3` | Adam/AdamW, bias correction, decoupled decay | build | 2 |
| `M10.4` | Schedules (cosine, WSD, Noam) and gradient clipping | build | 2 |
| `M10.5` | Curvature: Newton step, top Hessian eigenvalue, edge of stability | build | 9, optional |
| `M10.6` | Muon: Newton-Schulz orthogonalized momentum | build | 9, optional |
| `M10.7` | Lagrangians, KKT, duality, KL-regularized objectives | solve | solve set |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M10.1` | [Convexity, smoothness, gradient descent, and Armijo line search](01-gradient-descent-and-line-search.md) | build | 2 |
| 2 | `M10.2` | [The Optimizer protocol: SGD, momentum, Nesterov, weight decay](02-optimizer-protocol-and-sgd.md) | build | 2 |
| 3 | `M10.3` | [Adam and AdamW: bias correction and decoupled weight decay](03-adam-and-adamw.md) | build | 2 |
| 4 | `M10.4` | [Learning-rate schedules (cosine, WSD, Noam) and gradient clipping](04-schedules-and-gradient-clipping.md) | build | 2 |
| 5 | `S-M10a` | [Optimization problem set, part a: convexity, GD rates, momentum, Adam](90-problem-set-a.md) | solve | 2 |
| 6 | `S-M10b` | [Optimization problem set, part b: constrained optimization, duality, and DPO](91-problem-set-b.md) | solve | 10 |

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Calculus 3](../04-calculus-3/) | gradients, Hessians, and Taylor models of the loss |
| [Matrix Calculus and Autodiff](../08-matrix-calculus-and-autodiff/) | where the gradients come from; Hessian-vector products |
| [tinyllm Part 0](../../../courses/build-an-llm/p00-foundations/) | `L0.5` trains with these optimizers |
| [Training and Post-Training](../../ml/07-training-and-post-training/) | optimizer memory and large-scale training practice |
