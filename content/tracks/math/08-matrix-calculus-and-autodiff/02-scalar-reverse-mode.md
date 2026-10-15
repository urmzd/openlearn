<!-- ol:module M08.2 -->
# Scalar reverse mode: the Value graph

## Overview

| | |
|---|---|
| **Module** | `M08.2` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/autograd/scalar.py`: `Value` (`+ - * / **`, `exp`, `log`, `tanh`, `relu`, `backward`) |
| **Contract** | [`course/contracts/py/tinyllm/autograd/scalar.pyi`](../../../../course/contracts/py/tinyllm/autograd/scalar.pyi) |
| **Tests** | `course/tests/M08.2/` (what they check: section 4) |
| **Needs** | `M06.1` iterative topological sort (`toposort`) · `M08.1` dual numbers (the tests compare) · `M04.2` numeric VJP (or `--ref-deps`) |
| **Used by** | `L0.1` (the scalarized oracle for broadcasting backward in your Tensor engine) |
| **Milestone** | `MS-P2` (Pass 2 gate: every math module of the pass checks green, then your autograd bigram trains) |
| **Optional depth** | Karpathy, [micrograd](https://github.com/karpathy/micrograd) and the lecture "The spelled-out intro to neural networks and backpropagation"; Baydin et al., [*Automatic Differentiation in Machine Learning: a Survey*](https://www.jmlr.org/papers/v18/17-468.html), section 3.2 |

## Key Takeaways

- Reverse mode records the computation as a graph, then walks it once from the output, pushing $\bar v = \partial L / \partial v$ into each node's inputs by the chain rule (`test_hand_example`).
- A node used several times receives several contributions, so gradients are accumulated with `+=`, never assigned (`test_reused_node_accumulates`).
- The walk must visit each node once, after every consumer has added to it: reverse topological order, which `M06.1`'s iterative `toposort` provides without recursion (`test_diamond_visits_each_node_once`, `test_deep_chain_no_recursion_error`).
- One backward pass gives the gradient with respect to every input, and it agrees with forward mode on every input (`test_matches_dual`, `test_gradcheck_each_op`).

## How to work this chapter

```bash
ol start M08.2              # stubs scalar.py into your repo, contract alongside
ol tests M08.2              # read the test catalog first: rung R0, you write no tests here
ol check M08.2              # exit code is the verdict
ol check M08.2 --ref-deps   # only if your M06.1, M08.1, or M04.2 is not passing yet
ol diff  M08.2              # after passing: your code against the reference
```

---

## 1. Why now

Your bigram has $256 \times 256 = 65536$ weights and one loss. Forward mode (`M08.1`) would need 65536 passes to get the gradient, one per weight; `L0.5` trains it for thousands of steps. Reverse mode gets all 65536 partial derivatives from one forward pass and one backward pass, and that is the engine `L0.1` builds over numpy tensors. Tensors add broadcasting, shapes, and in-place buffers, which hide the mechanics. This module builds the same engine one scalar at a time, where every node is a number you can print, and `L0.1` then uses it as an oracle: a tensor op's gradient must equal the gradient of the same computation spelled out in scalar `Value`s.

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

These numbers are the first case in section 4, `test_hand_example`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | the section 3 forward and backward tables | you and the test agree on the walk |
| `test_gradcheck_each_op` | gradcheck | every operation against the frozen central differences at three points | each local rule is right |
| `test_matches_dual` | differential | one expression using every operation, both partials, against `M08.1` at 25 points | forward and reverse mode agree |
| `test_vjp_matches_numeric_vjp` | differential | $u^\top J$ for a map $\mathbb{R}^3 \to \mathbb{R}^3$ against `M04.2`'s `vjp_numeric` | reverse mode is a VJP |
| `test_reused_node_accumulates` | unit | $x + x$, $x \cdot x$, $x \cdot x + x$ | weights shared across positions |
| `test_diamond_visits_each_node_once` | unit | $3e^x + 4e^x$ gives $7e^x$ | shared subexpressions everywhere |
| `test_deep_chain_no_recursion_error` | boundary | a 50000-node chain | unrolled recurrences (`L3.1`) |
| `test_relu_gradient_at_zero` | boundary | gradient 0 at exactly 0 | agreement with PyTorch at kinks |
| `test_constants_on_either_side` | unit | $2x$, $x2$, $2 + x$, $1 - x$, $x - 1$, $6/x$, $x/2$, $-x$ | numbers mixed into expressions |
| `test_pow_needs_a_constant_exponent` | boundary | `Value ** Value` is a `TypeError` | no silently wrong rule |
| `test_backward_twice_accumulates` | unit | a second call doubles the leaf gradient | why training zeroes gradients |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `x.grad = ...` instead of `+=` | a node used twice keeps only its last contribution | `test_reused_node_accumulates` (mutant `s01`) |
| 2. walking inputs first, or walking every path | partial or repeated pushes; shared nodes get wrong multiples | `test_diamond_visits_each_node_once` (mutants `s03`, `s13`) |
| 3. a recursive topological sort inside `backward` | `RecursionError` on a deep chain | `test_deep_chain_no_recursion_error` (mutant `s05`) |
| 4. giving each factor its own value in the product rule | $\bar a$ is wrong in the worked example | `test_hand_example` (mutant `s02`) |
| 5. $\tanh'$ from the input ($1 - x^2$) instead of the output ($1 - y^2$) | wrong slopes away from 0 | `test_gradcheck_each_op` (mutant `s06`) |
| 6. `>=` in relu's backward | gradient 1 at the kink, unlike PyTorch | `test_relu_gradient_at_zero` (mutant `s07`) |
| 7. a reflected method that reverses the operands | $1 - x$ has slope $+1$ | `test_constants_on_either_side` (mutants `s11`, `s12`) |
| 8. adding to the seed instead of setting it | the second backward starts from 2 | `test_backward_twice_accumulates` (mutant `s04`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M06.1` | `toposort(root, parents)`: the order of the walk, without recursion |
| Back | `M08.1` | `Dual` computes the same derivatives forward; the tests compare |
| Back | `M04.2` | `vjp_numeric` approximates the same $u^\top J$ by central differences |
| Forward | `L0.1` | your Tensor engine's broadcasting backward is checked against the same computation spelled out in `Value`s |

If you skip this module, `ol check L0.1` stops with `L0.1 needs M08.2`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Value` | micrograd | the same engine plus a tiny neural-network library on top | `micrograd/engine.py` |
| `backward` | PyTorch's autograd engine | dependency counting and a ready queue instead of a precomputed order, multithreaded per device, retained or freed graphs | `torch/csrc/autograd/engine.cpp` |
| VJP from `backward` | JAX `jax.vjp` | reverse mode derived by linearizing and transposing the forward trace | `jax/_src/interpreters/ad.py` |
