# Scalar reverse mode: the Value graph

A byte-level bigram model has $256 \times 256 = 65536$ weights and one loss. Forward mode, from [Dual numbers and forward-mode autodiff](01-dual-numbers-forward-mode.md), would need 65536 passes to get the gradient, one per weight, and a training run takes thousands of steps. Reverse mode gets all 65536 partial derivatives from one forward pass and one backward pass. That asymmetry is why every neural network trains with reverse mode, and it is the engine that [Tensor, broadcasting backward, and no_grad](../../build-an-llm/p00-foundations/01-tensor-and-broadcasting-backward.md) in *Build Your Own LLM System* builds over numpy tensors, on the way to a bigram trained by the book's own autograd.

Tensors add broadcasting, shapes, and in-place buffers, all of which hide the mechanics. So this section builds the same engine one scalar at a time, where every node is a number you can print. The tensor engine then uses it as an oracle: a tensor op's gradient must equal the gradient of the same computation spelled out in scalar `Value`s.

You will need the iterative topological sort from [Graphs, DAGs, and an iterative topological sort](../06-discrete-math-2/01-graphs-dags-and-topological-sort.md) (its `toposort` lives in [`graph.py`](../../../../course/solutions/python/tinyllm/autograd/graph.py)), the dual numbers of the previous section (forward mode must agree with reverse mode), and the numeric VJP of [Jacobians, the multivariable chain rule, and numeric JVP/VJP](../04-calculus-3/02-jacobians-and-the-chain-rule.md). The complete implementation is [`scalar.py`](../../../../course/solutions/python/tinyllm/autograd/scalar.py), with its interface in the contract [`course/contracts/py/tinyllm/autograd/scalar.pyi`](../../../../course/contracts/py/tinyllm/autograd/scalar.pyi).

> **Learning objectives**
> - Explain how reverse mode turns the multivariable chain rule into a backward walk over a recorded graph.
> - Run a forward and backward pass by hand on a small graph with a reused node.
> - Implement a scalar `Value` with closures for local rules and a backward pass in reverse topological order.
> - Explain why adjoints accumulate with `+=`, why the walk must visit each node once, and why backward costs a constant times the forward pass.

## Recording the computation as a graph

The notation is short. $L$ is the final scalar output, the loss. $v$ is any node of the graph: an input, a constant, or an intermediate result. Its adjoint $\bar v = \partial L / \partial v$ says how much $L$ changes per unit change of $v$, and lives in `v.grad`. An operation $y = g(x_1, \dots, x_k)$ has inputs $x_j$ (recorded in `_prev`) and local derivatives $\partial g / \partial x_j$. For a vector-valued output, $u \in \mathbb{R}^m$ is a weight vector, as in $u^\top J$.

Every operation creates a new `Value` holding its result, the inputs it came from (`_prev`), and a closure (`_backward`) that knows the local derivatives. Inputs are leaves. Because a node is created only after its inputs exist, the graph has no cycles: it is a directed acyclic graph, exactly the object that *Discrete Math 2* taught you to sort.

## The chain rule, pushed backward

If $L$ depends on $x$ only through the nodes $y$ that consume $x$, the multivariable chain rule says

$$\bar x = \sum_{y \text{ consumes } x} \bar y\, \frac{\partial y}{\partial x}.$$

So each node $y$, once its own $\bar y$ is complete, adds $\bar y\, \partial y / \partial x_j$ into each input's adjoint. The sum over consumers is why the update is `x.grad += ...`: a node used twice (the variable $a$ in the worked example below, or every weight shared across time steps in [Vanilla RNN with manual BPTT and truncation](../../build-an-llm/p03-recurrent/01-vanilla-rnn-manual-bptt.md)) collects one term per use.

That raises the question of order. A node may push its adjoint only after every consumer has pushed into it; otherwise it pushes a partial sum. Seed $\bar L = 1$ ($L$ changes one-for-one with itself) and visit nodes so that every node comes before its inputs: a reverse topological order. `toposort(root, parents)` returns exactly that list for `parents = lambda v: v._prev`, each reachable node once, and it is iterative, so a graph 50000 nodes deep does not hit Python's recursion limit of about 1000 frames. Walking paths instead of nodes (a stack that pushes every input every time) visits a shared node once per path: its `_backward` runs several times and its inputs get multiples of the right gradient.

## Local rules

Each operation's `_backward` adds $\bar y$ times its partial derivatives into its inputs:

| $y$ | adds to its inputs |
|---|---|
| $x_1 + x_2$ | $\bar x_1 \mathrel{+}= \bar y$, $\bar x_2 \mathrel{+}= \bar y$ |
| $x_1 x_2$ | $\bar x_1 \mathrel{+}= x_2\,\bar y$, $\bar x_2 \mathrel{+}= x_1\,\bar y$ |
| $x^k$, constant $k$ | $\bar x \mathrel{+}= k x^{k-1}\,\bar y$ |
| $e^x$ | $\bar x \mathrel{+}= e^x\,\bar y$ (reuse the output) |
| $\log x$ | $\bar x \mathrel{+}= \bar y / x$ |
| $\tanh x$ | $\bar x \mathrel{+}= (1 - y^2)\,\bar y$ (reuse the output) |
| $\mathrm{relu}(x)$ | $\bar x \mathrel{+}= [x > 0]\,\bar y$, so 0 at exactly $x = 0$ |

Subtraction and division need no rules of their own: $a - b = a + (-1)b$ and $a / b = a \cdot b^{-1}$. A Python number in an operation becomes a constant leaf; Python calls the reflected method (`__rsub__`, `__rtruediv__`) when the number is on the left, and the operand order must survive.

## What it costs, and what it computes

The forward pass does one operation per node; the backward pass does one local-derivative product per edge. So the full gradient of a scalar costs a small constant times one forward pass, whatever the number of inputs. That is the asymmetry with forward mode, which costs one pass per input.

For a vector output $y = f(x) \in \mathbb{R}^m$, run backward from $L = \sum_i u_i y_i$. The leaves end up holding $u^\top J_f(x)$, a vector-Jacobian product (VJP), the row-vector counterpart of the JVP from the previous section.

One behavior surprises people: backward accumulates, it never resets. Calling `backward` twice adds every gradient twice, and PyTorch behaves the same. Training loops clear gradients before each step for this reason (`zero_grad` in [The Optimizer protocol: SGD, momentum, Nesterov, weight decay](../10-optimization/02-optimizer-protocol-and-sgd.md)).

## Working it by hand

Take $L = (ab + c)\,a$ at $a = 2$, $b = -3$, $c = 10$. The forward pass creates three nodes:

| node | operation | value |
|---|---|---|
| $d$ | $a \cdot b$ | $-6$ |
| $e$ | $d + c$ | $4$ |
| $L$ | $e \cdot a$ | $8$ |

The backward pass visits them in reverse topological order $L, e, d$, then the leaves:

| visit | its adjoint | pushes |
|---|---|---|
| $L$ | $\bar L = 1$ (seed) | $\bar e \mathrel{+}= a \cdot 1 = 2$, $\bar a \mathrel{+}= e \cdot 1 = 4$ |
| $e$ | $\bar e = 2$ | $\bar d \mathrel{+}= 2$, $\bar c \mathrel{+}= 2$ |
| $d$ | $\bar d = 2$ | $\bar a \mathrel{+}= b \cdot 2 = -6$, $\bar b \mathrel{+}= a \cdot 2 = 4$ |

The result is $\bar a = 4 - 6 = -2$, $\bar b = 4$, $\bar c = 2$. Check it by hand: $L = a^2 b + ac$, so $\partial L/\partial a = 2ab + c = -12 + 10 = -2$, $\partial L/\partial b = a^2 = 4$, and $\partial L/\partial c = a = 2$. The two paths into $a$ (through $L$ directly and through $d$) add up; with `=` instead of `+=`, $\bar a$ would end as $-6$ or $4$ depending on the order. These numbers become the first test of the implementation.

## Designing the interface

The whole engine is one class:

```python
# python/tinyllm/autograd/scalar.py
class Value:
    data: float; grad: float; _prev: tuple[Value, ...]; _op: str; _backward: Callable[[], None]
    def __init__(self, data: float, _children: tuple = (), _op: str = '') -> None
    # + - * / with Values or numbers on either side, unary -, ** with a constant exponent
    def exp(self) -> Value; def log(self) -> Value; def tanh(self) -> Value; def relu(self) -> Value
    def backward(self) -> None      # self.grad = 1, then _backward in toposort order
```

`backward` calls `toposort(self, lambda v: v._prev)` from `tinyllm.autograd.graph`: each reachable node once, every node before the nodes it was computed from. `Value ** Value` is a `TypeError`.

## Building it step by step

The design follows from the chain rule in six steps.

1. The backward chain rule needs, at each node, its inputs and its local derivatives. A node is created by an operation, so the operation is the one place that knows both: each operator method builds the output `Value`, records its inputs in `_prev`, and attaches a closure `_backward` that adds $\bar y\,\partial y/\partial x_j$ into each input.
2. Because a node can have several consumers, its adjoint is a sum. So every closure uses `+=` on its inputs' `grad`, and `grad` starts at 0.
3. A node may push only once its adjoint is complete, which forces the order: every node before the nodes it was computed from. `toposort(root, parents)` returns that order with each node once, iteratively.
4. The seed is $\bar L = 1$, set (not added) on the output before the walk.
5. Subtraction, division, and negation are compositions of `+`, `*`, and `**`, so they get their gradients from those rules and need no closures of their own. Numbers become constant leaves via `_lift`, and reflected methods keep the operand order.
6. The closures for $e^x$ and $\tanh x$ reuse the forward output; `relu` passes the gradient only where the input is strictly positive.

Start with the node and the two basic rules:

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                 # adjoints are sums: start at 0
        self._prev = tuple(_children)   # the inputs this node was computed from
        self._op = _op                  # for printing the graph only
        self._backward = _noop          # leaves have nothing to push

    def __add__(self, other):
        o = _lift(other)
        out = Value(self.data + o.data, (self, o), "+")

        def _backward():
            # d(x1 + x2)/dx_j = 1; += because self may be used elsewhere too
            self.grad += out.grad
            o.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        o = _lift(other)
        out = Value(self.data * o.data, (self, o), "*")

        def _backward():
            # each factor gets the OTHER factor's value; for x * x both lines
            # hit the same node and add 2x, as they should
            self.grad += o.data * out.grad
            o.grad += self.data * out.grad

        out._backward = _backward
        return out
```

The closure captures `self`, `o`, and `out`, so it reads `out.grad` at the time it runs, after the walk has completed it, not at the time it was created.

Everything else is built from those two, or has a one-input rule:

```python
    def __radd__(self, other):          # 2 + x: addition commutes
        return self + other

    def __sub__(self, other):           # a - b = a + (-1) b
        return self + (-_lift(other))

    def __rsub__(self, other):          # 1 - x: the number stays on the left
        return _lift(other) + (-self)

    def __truediv__(self, other):       # a / b = a * b^-1
        return self * _lift(other) ** -1

    def __rtruediv__(self, other):      # 6 / x = 6 * x^-1
        return _lift(other) * self**-1

    def __neg__(self):
        return self * -1.0

    def __pow__(self, k):
        # a Value exponent would need the x^y rule (with log x); refuse it
        # rather than apply k x^(k-1) with the wrong meaning
        if isinstance(k, Value) or not isinstance(k, (int, float)):
            raise TypeError("Value ** k needs a constant int or float k")
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad   # (e^x)' = e^x: reuse the output

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad  # 1 - y^2, y the OUTPUT

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # strict >: gradient 0 at exactly 0, as in PyTorch
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out
```

And finally the walk:

```python
    def backward(self):
        # root first, every node before the nodes it was computed from,
        # each reachable node once; iterative, so deep chains are fine
        order = toposort(self, lambda v: v._prev)
        self.grad = 1.0                 # seed: set, not +=
        for v in order:
            v._backward()               # v.grad is complete when v is reached


def _lift(x):
    """A Value as is; a number as a constant leaf."""
    return x if isinstance(x, Value) else Value(x)
```

Run the worked example through it. `toposort` returns $L, e, d$ and then the leaves (in some order, all after the nodes that consume them). $L$'s closure adds 2 to $\bar e$ and 4 to $\bar a$; $e$'s adds 2 to $\bar d$ and $\bar c$; $d$'s adds $-6$ to $\bar a$ and 4 to $\bar b$. Leaves run `_noop`. $\bar a = 4 - 6 = -2$.

The full solution is [`scalar.py`](../../../../course/solutions/python/tinyllm/autograd/scalar.py); the walk order comes from [`graph.py`](../../../../course/solutions/python/tinyllm/autograd/graph.py).

## Why this design

The cost is what the theory promised. Forward: one node per operation. Backward: one topological sort, $O(\text{nodes} + \text{edges})$, then one closure call per node, each $O(\text{inputs})$. So the whole gradient costs a small constant times the forward pass, independent of the number of inputs. Memory is one `Value` per intermediate, which is why scalar autograd is an oracle, not an engine: a $576 \times 576$ matmul on one vector would be a third of a million nodes.

Closures beat a table of op rules because the closure captures exactly the values the rule needs (`t` for tanh, `out` for exp) at the time they are computed. A table keyed by `_op` would have to re-derive them from `_prev` and could not reuse the forward output.

A precomputed order is simpler than a ready queue. PyTorch counts each node's pending consumers and runs a node when its count hits zero, which handles graphs that are freed as they go, and multiple devices. For a scalar oracle, one `toposort` gives the same order with less machinery and reuses the sort you already have.

Composite subtraction and division mean fewer rules and therefore fewer places to get a sign wrong. The cost is a few extra nodes per operation, irrelevant for an oracle.

The tradeoff you accept is that `backward` accumulates across calls instead of zeroing first. That matches PyTorch and makes gradient accumulation across micro-batches possible; the price is that a training loop must clear gradients itself.

## What a correct implementation must handle

The hand example pins the walk: the forward and backward tables above must come out exactly. Every operation is gradchecked against frozen central differences at three points, so each local rule is right on its own. Two differential tests tie reverse mode to the earlier tools: one expression using every operation is differentiated in both partials and compared with `Dual` at 25 points, so forward and reverse mode agree; and $u^\top J$ for a map $\mathbb{R}^3 \to \mathbb{R}^3$ is compared with `vjp_numeric`, confirming that reverse mode is a VJP.

The structural cases matter most downstream. A reused node ($x + x$, $x \cdot x$, $x \cdot x + x$) models weights shared across positions. A diamond, $3e^x + 4e^x$, must give $7e^x$, since shared subexpressions are everywhere. A 50000-node chain models unrolled recurrences in the RNN chapter of *Build Your Own LLM System*. ReLU at exactly zero must have gradient 0, for agreement with PyTorch at kinks. Constants on either side ($2x$, $x2$, $2 + x$, $1 - x$, $x - 1$, $6/x$, $x/2$, $-x$) cover numbers mixed into expressions. `Value ** Value` must be a `TypeError`, so that no rule is silently wrong. And a second call to `backward` must double the leaf gradient, which is why training zeroes gradients.

The common mistakes, with the symptom each produces:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| `x.grad = ...` instead of `+=` | a node used twice keeps only its last contribution | every closure adds into `grad` |
| Walking inputs first, or walking every path | partial or repeated pushes; shared nodes get wrong multiples | `toposort` gives each node once, consumers first |
| A recursive topological sort inside `backward` | `RecursionError` on a deep chain | `toposort` uses an explicit stack |
| Giving each factor its own value in the product rule | $\bar a$ is wrong in the worked example | `self.grad += o.data * out.grad`: each factor gets the other's value |
| $\tanh'$ from the input ($1 - x^2$) instead of the output ($1 - y^2$) | wrong slopes away from 0 | captures the output `t` and uses $1 - t^2$ |
| `>=` in relu's backward | gradient 1 at the kink, unlike PyTorch | strict `self.data > 0` |
| A reflected method that reverses the operands | $1 - x$ has slope $+1$ | `__rsub__` builds `_lift(other) + (-self)` |
| Adding to the seed instead of setting it | the second backward starts from 2 | `self.grad = 1.0` before the walk |

## Where this leads

Looking back, `toposort(root, parents)` from [Graphs, DAGs, and an iterative topological sort](../06-discrete-math-2/01-graphs-dags-and-topological-sort.md) supplies the order of the walk without recursion; `Dual` from [Dual numbers and forward-mode autodiff](01-dual-numbers-forward-mode.md) computes the same derivatives forward, and the two must agree; and `vjp_numeric` from [Jacobians, the multivariable chain rule, and numeric JVP/VJP](../04-calculus-3/02-jacobians-and-the-chain-rule.md) approximates the same $u^\top J$ by central differences. Looking forward, the tensor engine's broadcasting backward in [Tensor, broadcasting backward, and no_grad](../../build-an-llm/p00-foundations/01-tensor-and-broadcasting-backward.md) is checked against the same computation spelled out in `Value`s. Within this chapter, [Matrix differentials, the trace trick, and closed-form VJPs](03-matrix-differentials-and-vjps.md) replaces graphs of scalars with one closed-form VJP per matrix operation.

Production engines follow the same pattern with more machinery. micrograd is this same engine plus a tiny neural-network library on top (`micrograd/engine.py`). PyTorch's autograd engine uses dependency counting and a ready queue instead of a precomputed order, runs multithreaded per device, and supports retained or freed graphs (`torch/csrc/autograd/engine.cpp`). JAX's `jax.vjp` derives reverse mode by linearizing and transposing the forward trace (`jax/_src/interpreters/ad.py`).

Further reading:

- Karpathy, [micrograd](https://github.com/karpathy/micrograd) and the lecture "The spelled-out intro to neural networks and backpropagation".
- Baydin et al., [*Automatic Differentiation in Machine Learning: a Survey*](https://www.jmlr.org/papers/v18/17-468.html), section 3.2.

## Key terms

- **Adjoint**: $\bar v = \partial L / \partial v$, the sensitivity of the final output to a node, stored in `v.grad`.
- **Reverse mode**: automatic differentiation that records the computation and walks it backward once from the output, pushing adjoints into inputs.
- **Reverse topological order**: an ordering in which every node comes before the nodes it was computed from.
- **Vector-Jacobian product (VJP)**: $u^\top J_f(x)$, what one backward pass from $L = \sum_i u_i y_i$ leaves in the leaves.
- **Leaf**: a node with no inputs, such as a parameter or a constant.
- **Seed**: the initial adjoint $\bar L = 1$ set on the output before the walk.

## Summary

- Reverse mode records the computation as a graph, then walks it once from the output, pushing $\bar v = \partial L / \partial v$ into each node's inputs by the chain rule.
- A node used several times receives several contributions, so gradients are accumulated with `+=`, never assigned.
- The walk must visit each node once, after every consumer has added to it: reverse topological order, which an iterative `toposort` provides without recursion.
- One backward pass gives the gradient with respect to every input, at a constant times the cost of the forward pass, and it agrees with forward mode on every input.
- `backward` accumulates across calls, so training loops clear gradients each step.

## Review questions

1. In $x \cdot x$, `__mul__`'s closure runs `self.grad += o.data * out.grad` and `o.grad += self.data * out.grad` with `self is o`. What is the result, and why is it right?

   <details><summary>Answer</summary>

   Both lines add $x\,\bar y$ to the same node, giving $2x\,\bar y$, the derivative of $x^2$. With `=` instead of `+=`, the second line would overwrite the first and give $x\,\bar y$.

   </details>

2. Why must the walk visit nodes in reverse topological order rather than, say, depth-first from the output pushing as it goes?

   <details><summary>Answer</summary>

   A node's adjoint is a sum over its consumers. Pushing from a node before all its consumers have added to it pushes a partial sum. A depth-first walk that pushes on every visit would also run a shared node's closure once per path, so in the diamond $3e^x + 4e^x$ the inputs would get multiples of the right gradient.

   </details>

3. Why does `tanh` capture `t` and use $1 - t^2$, rather than recomputing from `self.data`?

   <details><summary>Answer</summary>

   $\tanh'(x) = 1 - \tanh^2 x$ is a function of the output. Using $1 - x^2$ with the input is a different, wrong function (correct only at 0). Capturing `t` also avoids recomputing $\tanh$.

   </details>

4. Why is the seed `self.grad = 1.0` and not `self.grad += 1.0`?

   <details><summary>Answer</summary>

   $\partial L/\partial L = 1$ by definition, and the root has no consumers that could contribute to it, so the seed is a definition, not a contribution. With `+=`, a second call would find the 1 left by the first call and start the walk from 2. (Intermediate nodes also still hold their first-call adjoints, so a second call on the same graph re-pushes them; that is why training builds a fresh graph and clears gradients each step.)

   </details>

5. Why does `Value ** Value` raise instead of using the power rule?

   <details><summary>Answer</summary>

   The rule $k x^{k-1}$ treats the exponent as a constant. With a `Value` exponent, the gradient with respect to the exponent ($x^y \ln x$) would be silently missing. Raising makes the unsupported case loud.

   </details>

6. For $L = (ab + c)\,a$ at $a = 2$, $b = -3$, $c = 10$, what would $\bar a$ be if the closures assigned instead of accumulating?

   <details><summary>Answer</summary>

   $a$ receives two contributions, 4 from $L = e \cdot a$ and $-6$ from $d = a \cdot b$. With `=`, whichever closure runs last wins, so $\bar a$ ends as $-6$ or $4$ depending on the order, instead of the correct $-2$.

   </details>
