# Optimization

Training a model means minimizing an average loss using noisy gradients. Everything an optimizer does answers one of three questions. Which direction should the step go? That is the gradient, smoothed by momentum. How far should it go in each coordinate? That is the learning rate, which Adam scales per parameter. How should the step size change over time? That is the schedule. Convex theory gives the vocabulary and the convergence rates. Neural networks break its assumptions, but its intuitions survive, and the tools in this chapter are the ones every model in the series trains with.

The chapter starts with plain gradient descent, where the theory is exact: on a quadratic, the step must stay under $2/L$, and the number of steps grows with the condition number $\kappa = L/\mu$. The gap shrinks by about $1 - 1/\kappa$ per step. Next it wraps that update behind a common `Optimizer` protocol and adds momentum, Nesterov, and weight decay. Then it builds Adam and AdamW, where bias correction fixes the first steps and decoupled weight decay keeps regularization out of the adaptive scale. Last come learning-rate schedules and gradient clipping, which often decide whether a run trains at all. Two problem sets check the derivations by hand. The second one also covers Lagrangians, duality, and the KL-regularized objective behind direct preference optimization.

Optimizers are stateful objects, and that fact runs through the whole chapter. To resume a run bit for bit, the checkpoint must hold the moments, the step count, and the position in the schedule. Each implementation is checked against PyTorch's trajectory, so recipes written for PyTorch carry over unchanged.

> **Learning objectives**
> - Derive the stability limit $\eta < 2/L$ and the $1 - 1/\kappa$ rate of gradient descent, and use a backtracking line search when $L$ is unknown.
> - Implement SGD with momentum, Nesterov momentum, and coupled weight decay behind a resumable `Optimizer` protocol.
> - Implement Adam and AdamW with bias correction and decoupled weight decay, and match PyTorch step for step.
> - Build cosine, warmup-stable-decay, and Noam schedules and global-norm gradient clipping, with correct phase boundaries.
> - Solve KKT, duality, and KL-regularized problems by hand, including the derivation behind DPO.

## Before you begin

You should be comfortable with gradients and Hessians from [Calculus 3](../10-calculus-3/README.md), eigenvalues from [Linear Algebra](../09-linear-algebra/README.md), and geometric series from [Precalculus](../06-precalculus/README.md), which explain both bias correction and warmup schedules. The gradients these optimizers consume come from [Matrix Calculus and Autodiff](../14-matrix-calculus-and-autodiff/README.md). Its section [Hessian-vector products and the recompute-versus-memory schedule](../14-matrix-calculus-and-autodiff/04-hessian-vector-products-and-recompute.md) supplies the curvature tools for going beyond first-order methods.

## In this chapter

1. [Convexity, smoothness, gradient descent, and Armijo line search](01-gradient-descent-and-line-search.md) derives the step-size limit $2/L$ and the $1 - 1/\kappa$ rate, then builds gradient descent and a backtracking line search that handle divergence and NaN correctly.
2. [The Optimizer protocol: SGD, momentum, Nesterov, weight decay](02-optimizer-protocol-and-sgd.md) defines the four-method interface every optimizer shares and builds a PyTorch-compatible SGD that resumes bit for bit.
3. [Adam and AdamW: bias correction and decoupled weight decay](03-adam-and-adamw.md) derives Adam's two moving averages, the bias correction, and why weight decay must be decoupled, then implements both to match torch exactly.
4. [Learning-rate schedules (cosine, WSD, Noam) and gradient clipping](04-schedules-and-gradient-clipping.md) builds three schedules as pure functions of the step, along with global-norm clipping that keeps the update direction.
5. [Optimization problem set, part a: convexity, GD rates, momentum, Adam](05-problem-set-a.md) has 17 problems with worked answers on convexity, gradient descent rates, momentum, and Adam.
6. [Optimization problem set, part b: constrained optimization, duality, and DPO](06-problem-set-b.md) has 18 optional problems on KKT conditions, duality, and the KL-regularized optimum that DPO relies on.

## Where this chapter leads

In *AI Engineering*, [Training loop and the autograd bigram](../../ai-engineering/06-numerics-and-autograd/06-training-loop-and-the-autograd-bigram.md) trains with these optimizers, and every later model in that book does too. [Training and Post-Training](../../ai-engineering/19-training-methods/README.md) in *AI Engineering* covers optimizer memory and large-scale training practice. Beyond this chapter, the same ideas extend to curvature. The top Hessian eigenvalue can be found by power iteration on Hessian-vector products, and gradient descent stays stable only while $\eta < 2/\lambda_{\max}$, which is the edge of stability. Muon orthogonalizes the momentum of 2D weights with Newton-Schulz iterations.

## Further reading

- Boyd and Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/) (free PDF), chapters 2, 3, and 9 for convexity and descent methods; Nocedal and Wright, *Numerical Optimization* (Springer, 2nd ed.).
- Kingma and Ba, [*Adam*](https://arxiv.org/abs/1412.6980), and Loshchilov and Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101), both worth reading in full.
- Goh, [*Why Momentum Really Works*](https://distill.pub/2017/momentum/). It pairs well with plotting every optimizer on a 2D ill-conditioned quadratic before you trust it on a network.
- Cohen et al., [*Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*](https://arxiv.org/abs/2103.00065).
- Jordan, [*Muon*](https://kellerjordan.github.io/posts/muon/).
