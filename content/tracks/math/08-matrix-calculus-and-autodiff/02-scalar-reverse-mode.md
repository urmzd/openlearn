# Scalar reverse mode: the Value graph

## Overview

| | |
|---|---|
| **Module** | `M08.2` · build · Python · Pass 2 · about 1 to 1.5 h reading |
| **Solution** | [`scalar.py`](../../../../course/solutions/python/tinyllm/autograd/scalar.py): `Value` (`+ - * / **`, `exp`, `log`, `tanh`, `relu`, `backward`) |
| **Contract** | [`course/contracts/py/tinyllm/autograd/scalar.pyi`](../../../../course/contracts/py/tinyllm/autograd/scalar.pyi) |
| **Needs** | `M06.1` iterative topological sort (`toposort`, [`graph.py`](../../../../course/solutions/python/tinyllm/autograd/graph.py)) · `M08.1` dual numbers (forward mode must agree) · `M04.2` numeric VJP |
| **Used by** | `L0.1` (the scalarized oracle for broadcasting backward in the Tensor engine, on the way to the Pass 2 result: a bigram trained by the course's own autograd) |
| **Optional depth** | Karpathy, [micrograd](https://github.com/karpathy/micrograd) and the lecture "The spelled-out intro to neural networks and backpropagation"; Baydin et al., [*Automatic Differentiation in Machine Learning: a Survey*](https://www.jmlr.org/papers/v18/17-468.html), section 3.2 |

## Key Takeaways

- Reverse mode records the computation as a graph, then walks it once from the output, pushing $\bar v = \partial L / \partial v$ into each node's inputs by the chain rule.
- A node used several times receives several contributions, so gradients are accumulated with `+=`, never assigned.
- The walk must visit each node once, after every consumer has added to it: reverse topological order, which `M06.1`'s iterative `toposort` provides without recursion.
- One backward pass gives the gradient with respect to every input, and it agrees with forward mode on every input.

## How to read this chapter

Sections 1 to 3 build reverse mode from the chain rule and work one graph by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The Pass 2 bigram has $256 \times 256 = 65536$ weights and one loss. Forward mode (`M08.1`) would need 65536 passes to get the gradient, one per weight; `L0.5` trains it for thousands of steps. Reverse mode gets all 65536 partial derivatives from one forward pass and one backward pass, and that is the engine `L0.1` builds over numpy tensors. Tensors add broadcasting, shapes, and in-place buffers, which hide the mechanics. This module builds the same engine one scalar at a time, where every node is a number that can be printed, and `L0.1` then uses it as an oracle: a tensor op's gradient must equal the gradient of the same computation spelled out in scalar `Value`s.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $L$ | the final scalar output (the loss) | `Value` |
| $v$ | any node of the graph: an input, a constant, or an intermediate result | `Value` |
| $\bar v = \partial L / \partial v$ | the adjoint of $v$: how much $L$ changes per unit change of $v$ | `v.grad` |
| $y = g(x_1, \dots, x_k)$ | one operation with inputs $x_j$ | `Value` with `_prev = (x_1, ..., x_k)` |
| $\partial g / \partial x_j$ | the operation's local derivative | float |
| $u$ | a weight vector for a vector-valued output, in $u^\top J$ | `float64[m]` |

**The graph.** Every operation creates a new `Value` holding its result, the inputs it came from (`_prev`), and a closure (`_backward`) that knows the local derivatives. Inputs are leaves. Because a node is created only after its inputs exist, the graph has no cycles: it is a directed acyclic graph, the object `M06.1` sorts.

**The chain rule, pushed backward.** If $L$ depends on $x$ only through the nodes $y$ that consume $x$, the multivariable chain rule says

$$\bar x = \sum_{y \text{ consumes } x} \bar y\, \frac{\partial y}{\partial x}.$$

So each node $y$, once its own $\bar y$ is complete, adds $\bar y\, \partial y / \partial x_j$ into each input's adjoint. The sum over consumers is why the update is `x.grad += ...`: a node used twice (the variable $a$ in the worked example, every weight shared across time steps in `L3.1`) collects one term per use.

**The order of the walk.** A node may push its adjoint only after every consumer has pushed into it, otherwise it pushes a partial sum. Seed $\bar L = 1$ ($L$ changes one-for-one with itself) and visit nodes so that every node comes before its inputs: a reverse topological order. `M06.1`'s `toposort(root, parents)` returns exactly that list for `parents = lambda v: v._prev`, each reachable node once, and it is iterative, so a graph 50000 nodes deep does not hit Python's recursion limit of about 1000 frames. Walking paths instead of nodes (a stack that pushes every input every time) visits a shared node once per path: its `_backward` runs several times and its inputs get multiples of the right gradient.

**Local rules.** Each operation's `_backward` adds $\bar y$ times its partial derivatives:

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

**What it costs, and what it computes.** The forward pass does one operation per node; the backward pass does one local-derivative product per edge. So the full gradient of a scalar costs a small constant times one forward pass, whatever the number of inputs. That is the asymmetry with forward mode, which costs one pass per input. For a vector output $y = f(x) \in \mathbb{R}^m$, run backward from $L = \sum_i u_i y_i$: the leaves end up holding $u^\top J_f(x)$, a vector-Jacobian product (VJP), the row-vector counterpart of `M08.1`'s JVP.

**Backward accumulates, it never resets.** Calling `backward` twice adds every gradient twice (PyTorch behaves the same). Training loops clear gradients before each step for this reason (`zero_grad` in `M10.2`).

## 3. Worked example by hand

$L = (ab + c)\,a$ at $a = 2$, $b = -3$, $c = 10$.

**Forward.**

| node | operation | value |
|---|---|---|
| $d$ | $a \cdot b$ | $-6$ |
| $e$ | $d + c$ | $4$ |
| $L$ | $e \cdot a$ | $8$ |

**Backward**, in reverse topological order $L, e, d$, then the leaves:

| visit | its adjoint | pushes |
|---|---|---|
| $L$ | $\bar L = 1$ (seed) | $\bar e \mathrel{+}= a \cdot 1 = 2$, $\bar a \mathrel{+}= e \cdot 1 = 4$ |
| $e$ | $\bar e = 2$ | $\bar d \mathrel{+}= 2$, $\bar c \mathrel{+}= 2$ |
| $d$ | $\bar d = 2$ | $\bar a \mathrel{+}= b \cdot 2 = -6$, $\bar b \mathrel{+}= a \cdot 2 = 4$ |

Result: $\bar a = 4 - 6 = -2$, $\bar b = 4$, $\bar c = 2$. By hand, $L = a^2 b + ac$, so $\partial L/\partial a = 2ab + c = -12 + 10 = -2$, $\partial L/\partial b = a^2 = 4$, $\partial L/\partial c = a = 2$. The two paths into $a$ (through $L$ directly and through $d$) add up; with `=` instead of `+=`, $\bar a$ would end as $-6$ or $4$ depending on the order.

These numbers are the first case in section 4.

## 4. The interface

```python
# python/tinyllm/autograd/scalar.py
class Value:
    data: float; grad: float; _prev: tuple[Value, ...]; _op: str; _backward: Callable[[], None]
    def __init__(self, data: float, _children: tuple = (), _op: str = '') -> None
    # + - * / with Values or numbers on either side, unary -, ** with a constant exponent
    def exp(self) -> Value; def log(self) -> Value; def tanh(self) -> Value; def relu(self) -> Value
    def backward(self) -> None      # self.grad = 1, then _backward in M06.1's toposort order
```

`backward` calls `toposort(self, lambda v: v._prev)` from `tinyllm.autograd.graph` (`M06.1`): each reachable node once, every node before the nodes it was computed from. `Value ** Value` is a `TypeError`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example | unit | the section 3 forward and backward tables | pins the walk |
| Each op against central differences | gradcheck | every operation against frozen central differences at three points | each local rule is right |
| Agreement with dual numbers | differential | one expression using every operation, both partials, against `M08.1` at 25 points | forward and reverse mode agree |
| VJP against numeric VJP | differential | $u^\top J$ for a map $\mathbb{R}^3 \to \mathbb{R}^3$ against `M04.2`'s `vjp_numeric` | reverse mode is a VJP |
| Reused node | unit | $x + x$, $x \cdot x$, $x \cdot x + x$ | weights shared across positions |
| Diamond | unit | $3e^x + 4e^x$ gives $7e^x$ | shared subexpressions everywhere |
| Deep chain | boundary | a 50000-node chain | unrolled recurrences (`L3.1`) |
| ReLU at zero | boundary | gradient 0 at exactly 0 | agreement with PyTorch at kinks |
| Constants on either side | unit | $2x$, $x2$, $2 + x$, $1 - x$, $x - 1$, $6/x$, $x/2$, $-x$ | numbers mixed into expressions |
| Power needs a constant exponent | boundary | `Value ** Value` is a `TypeError` | no silently wrong rule |
| Backward twice | unit | a second call doubles the leaf gradient | why training zeroes gradients |

## 5. The solution, step by step

### From first principles

1. The chain rule pushed backward (section 2) needs, at each node, its inputs and its local derivatives. A node is created by an operation, so the operation is the one place that knows both: each operator method builds the output `Value`, records its inputs in `_prev`, and attaches a closure `_backward` that adds $\bar y\,\partial y/\partial x_j$ into each input.
2. Because a node can have several consumers, its adjoint is a sum. So every closure uses `+=` on its inputs' `grad`, and `grad` starts at 0.
3. A node may push only once its adjoint is complete, which forces the order: every node before the nodes it was computed from. `M06.1`'s `toposort(root, parents)` returns that order with each node once, iteratively.
4. The seed is $\bar L = 1$, set (not added) on the output before the walk.
5. Subtraction, division, and negation are compositions of `+`, `*`, and `**`, so they get their gradients from those rules and need no closures of their own. Numbers become constant leaves via `_lift`, and reflected methods keep the operand order.
6. The closure for $e^x$ and $\tanh x$ reuses the forward output; `relu` passes the gradient only where the input is strictly positive.

### The code, annotated

The node and the two basic rules:

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

Everything else is built from those, or has a one-input rule:

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

And the walk:

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

On the worked example, `toposort` returns $L, e, d$ and then the leaves (in some order, all after the nodes that consume them). $L$'s closure adds 2 to $\bar e$ and 4 to $\bar a$; $e$'s adds 2 to $\bar d$ and $\bar c$; $d$'s adds $-6$ to $\bar a$ and 4 to $\bar b$. Leaves run `_noop`. $\bar a = 4 - 6 = -2$.

Full solution: [`scalar.py`](../../../../course/solutions/python/tinyllm/autograd/scalar.py); the walk order comes from [`graph.py`](../../../../course/solutions/python/tinyllm/autograd/graph.py).

### Why this solution

- **Cost.** Forward: one node per operation. Backward: one topological sort, $O(\text{nodes} + \text{edges})$, then one closure call per node, each $O(\text{inputs})$. So the whole gradient costs a small constant times the forward pass, independent of the number of inputs. Memory is one `Value` per intermediate, which is why scalar autograd is an oracle, not an engine: a $576 \times 576$ matmul on one vector would be a third of a million nodes.
- **Why closures instead of a table of op rules.** The closure captures exactly the values the rule needs (`t` for tanh, `out` for exp) at the time they are computed. A table keyed by `_op` would have to re-derive them from `_prev` and could not reuse the forward output.
- **Why a precomputed order, not a ready queue.** PyTorch counts each node's pending consumers and runs a node when its count hits zero. That handles graphs that are freed as they go, and multiple devices. For a scalar oracle, one `toposort` gives the same order with less machinery, and reuses `M06.1`.
- **Why composite subtraction and division.** Fewer rules means fewer places to get a sign wrong. The cost is a few extra nodes per operation, irrelevant for an oracle.
- **Tradeoff accepted.** `backward` accumulates across calls instead of zeroing first. That matches PyTorch and makes gradient accumulation across micro-batches possible; the price is that a training loop must clear gradients itself.

### Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. `x.grad = ...` instead of `+=` | a node used twice keeps only its last contribution | every closure adds into `grad` |
| 2. walking inputs first, or walking every path | partial or repeated pushes; shared nodes get wrong multiples | `toposort` gives each node once, consumers first |
| 3. a recursive topological sort inside `backward` | `RecursionError` on a deep chain | `M06.1`'s `toposort` uses an explicit stack |
| 4. giving each factor its own value in the product rule | $\bar a$ is wrong in the worked example | `self.grad += o.data * out.grad`: each factor gets the other's value |
| 5. $\tanh'$ from the input ($1 - x^2$) instead of the output ($1 - y^2$) | wrong slopes away from 0 | captures the output `t` and uses $1 - t^2$ |
| 6. `>=` in relu's backward | gradient 1 at the kink, unlike PyTorch | strict `self.data > 0` |
| 7. a reflected method that reverses the operands | $1 - x$ has slope $+1$ | `__rsub__` builds `_lift(other) + (-self)` |
| 8. adding to the seed instead of setting it | the second backward starts from 2 | `self.grad = 1.0` before the walk |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M06.1` | `toposort(root, parents)`: the order of the walk, without recursion |
| Back | `M08.1` | `Dual` computes the same derivatives forward; the two must agree |
| Back | `M04.2` | `vjp_numeric` approximates the same $u^\top J$ by central differences |
| Forward | `L0.1` | the Tensor engine's broadcasting backward is checked against the same computation spelled out in `Value`s |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Value` | micrograd | the same engine plus a tiny neural-network library on top | `micrograd/engine.py` |
| `backward` | PyTorch's autograd engine | dependency counting and a ready queue instead of a precomputed order, multithreaded per device, retained or freed graphs | `torch/csrc/autograd/engine.cpp` |
| VJP from `backward` | JAX `jax.vjp` | reverse mode derived by linearizing and transposing the forward trace | `jax/_src/interpreters/ad.py` |
