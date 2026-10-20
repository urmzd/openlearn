# Matrix Calculus and Autodiff

A gradient is a linear map, and a program is a composition of small differentiable steps. Automatic differentiation is the chain rule applied to a program rather than to a formula. Forward mode pushes a tangent through each step, a Jacobian-vector product; reverse mode records the computation and pulls a cotangent back through each step, a vector-Jacobian product (VJP). If you know the VJP of each primitive, you can differentiate any program built from them, and knowing it in closed form is what makes it fast and stable.

This chapter builds that machinery from first principles, in the order you would build an autograd engine. It starts with dual numbers, which carry a derivative alongside each value and give exact derivatives with no step size; they become the oracle that every later rule is checked against. It then builds reverse mode one scalar at a time, where the graph and the backward walk are small enough to print. Because reverse mode costs about one extra forward pass per scalar output, it is how every neural network trains: one loss, millions of parameters. Scalars are too slow for real work, so the third section derives closed-form VJPs for matmul, softmax, log-softmax, cross-entropy, LayerNorm, and RMSNorm with matrix differentials and the trace trick, a few lines each. The last section turns to the costs of reverse mode: Hessian-vector products that give curvature without forming the Hessian, and activation checkpointing, which trades recomputation for memory on a $\sqrt{n}$-type schedule. A problem set drills the derivations on paper.

Two habits run through the whole chapter. Gradients are checked, never trusted: central differences in float64 against the analytic VJP catch nearly every wrong rule. And every rule is worked by hand on small numbers before it is coded.

> **Learning objectives**
> - Compute derivatives and Jacobian-vector products with dual numbers, and explain why they are exact to rounding.
> - Implement scalar reverse mode over a recorded graph and explain why adjoints accumulate and why the walk follows reverse topological order.
> - Derive closed-form VJPs with matrix differentials and the trace trick, including the correction terms of normalization layers and the sum over broadcast axes.
> - Compute Hessian-vector products from two gradient calls, and plan an optimal activation-checkpointing schedule under a memory budget.
> - Choose between forward and reverse mode by counting passes, and estimate the FLOPs and memory of a backward pass.

## Before you begin

You should be comfortable with the chain rule from [Calculus 1](../07-calculus-1/README.md), gradients and Jacobians from [Calculus 3](../10-calculus-3/README.md), matrix products and shapes from [Linear Algebra](../09-linear-algebra/README.md), and topological sorting from [Discrete Math 2](../12-discrete-math-2/README.md), which orders the backward walk. The stable softmax and log-sum-exp from [Numerical Methods and Floating Point](../15-numerical-methods-and-floating-point/README.md) appear inside the VJPs. A good companion read is Parr and Howard's *The Matrix Calculus You Need for Deep Learning*, deriving every identity by hand, followed by sections 2 and 3 of the Baydin survey. If you have never built micrograd, doing so once from memory is excellent preparation.

## In this chapter

1. [Dual numbers and forward-mode autodiff](01-dual-numbers-forward-mode.md). Dual numbers $a + b\varepsilon$ with $\varepsilon^2 = 0$ turn Taylor's theorem into an identity, so one evaluation returns a value and its exact derivative; the section builds a numpy-compatible `Dual` and its JVP.
2. [Scalar reverse mode: the Value graph](02-scalar-reverse-mode.md). A scalar `Value` records each operation and its local rule, and `backward` walks the graph once in reverse topological order to give the gradient with respect to every input.
3. [Matrix differentials, the trace trick, and closed-form VJPs](03-matrix-differentials-and-vjps.md). The differential method derives the VJPs of matmul, softmax, log-softmax, fused cross-entropy, LayerNorm, and RMSNorm, and `unbroadcast` sums gradients over broadcast axes.
4. [Hessian-vector products and the recompute-versus-memory schedule](04-hessian-vector-products-and-recompute.md). Two gradient calls give $Hv$ to $O(\epsilon^2)$; a counting argument gives the least peak memory of activation checkpointing and the optimal schedule under a budget.
5. [Matrix calculus problem set: differentials, trace trick, VJPs, SDPA Jacobian](05-problem-set.md). Pen-and-paper problems with worked answers on shapes, differentials, the trace trick, VJPs, forward versus reverse cost, and the attention backward.

## Key terms

- **Automatic differentiation / autodiff**: computing exact derivatives of a program by applying the chain rule to each primitive operation it executes, in forward mode (Jacobian-vector products) or reverse mode (vector-Jacobian products); it is how neural networks get their gradients, with no step size and no symbolic formula.

## Where this chapter leads

The gradients built here feed the optimizers of [Optimization](../16-optimization/README.md). In *AI Engineering*, the [Foundations](../../ai-engineering/06-numerics-and-autograd/README.md) chapter turns these rules into a tensor autograd engine, registering each VJP as an op. In *AI Engineering*, [Deep Learning](../../ai-engineering/02-deep-learning/README.md) treats backpropagation in the textbook setting.

## Further reading

- Parr and Howard, [*The Matrix Calculus You Need for Deep Learning*](https://arxiv.org/abs/1802.01528) (free).
- Baydin, Pearlmutter, Radul, and Siskind, [*Automatic Differentiation in Machine Learning: a Survey*](https://www.jmlr.org/papers/v18/17-468.html) (JMLR 2018, free).
- Griewank and Walther, *Evaluating Derivatives* (SIAM, 2nd ed.).
- Petersen and Pedersen, [*The Matrix Cookbook*](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) (free).
- Karpathy, [micrograd](https://github.com/karpathy/micrograd) (free).
- Chen et al., [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174) (free).
