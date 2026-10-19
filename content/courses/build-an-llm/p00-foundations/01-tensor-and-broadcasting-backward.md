# Tensor, broadcasting backward, and no_grad

## Overview

| | |
|---|---|
| **Module** | `L0.1` · build · Python · Pass 2 · about 2 to 3 h reading |
| **Solution** | [`tensor.py`](../../../../course/solutions/python/tinyllm/autograd/tensor.py): `Tensor` (arithmetic, `@`, `**`, indexing, `backward`, `detach`, `numpy`) and `from_op`; [`mode.py`](../../../../course/solutions/python/tinyllm/autograd/mode.py): `is_grad_enabled`, `no_grad`. Helpers from earlier modules: [`graph.py`](../../../../course/solutions/python/tinyllm/autograd/graph.py) (`toposort`), [`vjp.py`](../../../../course/solutions/python/tinyllm/autograd/vjp.py) (`unbroadcast`) |
| **Contract** | [`course/contracts/py/tinyllm/autograd/tensor.pyi`](../../../../course/contracts/py/tinyllm/autograd/tensor.pyi) · [`course/contracts/py/tinyllm/autograd/mode.pyi`](../../../../course/contracts/py/tinyllm/autograd/mode.pyi) |
| **Needs** | `M06.1` iterative topological sort (`toposort`) · `M08.3` `unbroadcast` and the matmul VJP · `M08.2` the scalar `Value` (an independent oracle for the gradients) · reading: `lang.01` broadcasting |
| **Used by** | `L0.2` op library · `L0.3` fused losses · `L0.4` modules · `L0.5` training loop, where this autograd retrains the tracer bigram · `L0.6` checkpoint tests · later: every trained model in the course (see section 7) |
| **Optional depth** | Baydin, Pearlmutter, Radul, Siskind, "Automatic Differentiation in Machine Learning: a Survey" (JMLR 2018, free); Karpathy's `micrograd` (the scalar version of this module) |

## Key Takeaways

- A tensor op records its inputs and one vector-Jacobian product; `backward` calls each VJP once, in reverse topological order, so a node's gradient is complete before it is split among its inputs.
- A value used in several places gets the **sum** of the gradients from every place, whether the reuse is explicit (`a * a + a`) or hidden in broadcasting.
- Undoing broadcasting is a sum over the stretched axes, and indexing with repeated indices is a scatter-add, never an assignment.
- Grad mode is a per-thread flag that `no_grad` turns off and always restores, so evaluation builds no graph and a failed batch cannot leave training broken.
- A tensor keeps its dtype: constants are converted to it, gradients have it, so float32 models stay float32.

## How to read this chapter

Sections 1 to 3 build reverse mode on arrays from first principles and work one broadcasting example by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through the solution: `from_op`, `backward`, each operator's VJP, and `no_grad`, and explains why each line is there. Try to predict each VJP before reading it.

`tinyllm` stays a namespace package: there is no `__init__.py` under `python/tinyllm/autograd/`. Import as `from tinyllm.autograd.tensor import Tensor`.

---


## 1. Why now

The bigram from Pass 1 is trained by counting. That works for exactly one model: the moment a model has a hidden layer (the digits MLP that Pass 2 trains), a gate, or attention, there is no count to take and the weights must be found by gradient descent. `M08.2` built reverse mode on scalars, one `Value` per number. A `[256, 256]` table is 65,536 `Value` objects and one Python call per multiply; the digits MLP would take minutes per step. This module moves reverse mode from scalars to numpy arrays, so one node holds a whole matrix and one VJP is one vectorized numpy expression. Every model in the rest of the course trains through the `Tensor` built here.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x, w$ | input tensors of an op | `ndarray`, shapes $s_x$, $s_w$ |
| $y = f(x, w)$ | the op's output | `ndarray`, shape $s_y$ |
| $L$ | the scalar loss at the end of the graph | `float` |
| $\bar{y} = \partial L / \partial y$ | upstream gradient ("y-bar"), same shape as $y$ | `ndarray`, $s_y$ |
| $\bar{x} = \partial L / \partial x$ | the gradient this op hands back to $x$ | `ndarray`, $s_x$ |
| $J_x = \partial y / \partial x$ | the Jacobian of the op in $x$ (never built) | $|s_y| \times |s_x|$ |
| $\mathrm{vjp}(\bar{y})$ | vector-Jacobian product: $\bar{x} = \bar{y}^\top J_x$, computed without $J_x$ | function |
| $G$ | the graph: tensors as nodes, an edge from each input to each output | DAG |
| $\mathrm{unbroadcast}(g, s)$ | $g$ summed over every axis numpy stretched to reach shape $s$ | `M08.3` |

**A tensor is an array plus its history.** `Tensor(data, requires_grad=True)` is a **leaf**: a parameter or an input you want gradients for. An op on tensors returns a new tensor that remembers its inputs (its *parents*) and a function, the VJP, that turns the gradient of the loss with respect to the output into gradients with respect to the inputs. That record is the graph. `from_op(data, parents, vjp, op)` is the one constructor every op uses: it attaches the parents and the VJP only when grad mode is on and some parent requires grad, otherwise it returns a plain constant.

**One VJP per op, never a Jacobian.** The chain rule says $\bar{x} = \bar{y}^\top J_x$. For $y = x \odot w$ (elementwise), $J_x$ is diagonal with $w$ on it, so $\bar{x} = \bar{y} \odot w$: a product, not a matrix. Every op in this module has such a closed form (`M08.3` derived them):

| Op | $\bar{x}$ | $\bar{w}$ |
|---|---|---|
| $x + w$ | $\bar{y}$ | $\bar{y}$ |
| $x - w$ | $\bar{y}$ | $-\bar{y}$ |
| $x \odot w$ | $\bar{y} \odot w$ | $\bar{y} \odot x$ |
| $x / w$ | $\bar{y} / w$ | $-\bar{y} \odot x / w^2$ |
| $x^p$ ($p$ a number) | $\bar{y} \cdot p\, x^{p-1}$ | |
| $X W$ (matmul) | $\bar{Y} W^\top$ | $X^\top \bar{Y}$ |
| $x[\text{idx}]$ | zeros with $\bar{y}$ **added** at idx | |

**Reverse topological order.** Calling `loss.backward()` seeds $\bar{L} = 1$ and walks the graph from the loss back to the leaves. A node may feed several consumers (a diamond: $b = 2a$, $c = a^2$, $d = bc$). Its gradient is complete only after every consumer has added its share, so the walk must visit every consumer before the node: reverse topological order. `M06.1`'s `toposort(root, parents)` gives exactly that order iteratively. A recursive walk would hit Python's recursion limit (about 1000) on an unrolled network of a thousand ops.

**Reuse means sum.** If $a$ feeds two ops, $L$ depends on $a$ through two paths and $\partial L/\partial a$ is the sum of both. So `backward` keeps a dictionary of pending gradients keyed by node and **adds** each contribution; it never overwrites. Leaves go one step further, like torch: their `.grad` adds up across separate `backward()` calls until you set it back to `None` (that is how an optimizer's `zero_grad` and gradient accumulation over micro-batches both work).

**Broadcasting is reuse, so its gradient is a sum.** `x * w` with `x` of shape `(2, 3)` and `w` of shape `(3,)` uses each `w[j]` twice, once per row. numpy never copies it, but mathematically the op saw `w` repeated. The gradient for the repeated copy is the sum over the repeated axis: `unbroadcast(g, (3,))` sums `g` over axis 0. The rule in general (`M08.3`): sum over the leading axes numpy added, then over every axis where the input had size 1 and the output did not, keeping it as size 1. Every binary op calls `unbroadcast` on both gradients, because either operand may have been stretched.

**Indexing scatters with addition.** `x[[0, 2, 0]]` reads row 0 twice. Its gradient must carry both upstream rows back into row 0. `out[idx] = g` with a repeated index writes once and keeps the last; `np.add.at(out, idx, g)` adds every occurrence. An embedding lookup with a repeated token (`L0.2`) is this exact case.

**Constants.** A number, an ndarray, or a tensor with `requires_grad=False` is a constant: it takes part in the forward pass and gets no gradient. A constant is converted to the **other operand's dtype**: a Python float is float64, and promoting a float32 model to float64 doubles its memory and makes it disagree with the float32 C and Rust ports. Gradients are cast to their tensor's dtype too, so `p.data -= lr * p.grad` keeps the parameter's type.

**numpy must defer.** `ndarray + Tensor` normally runs numpy's own `__add__`, which broadcasts the Tensor *object* into an array of objects. Setting the class attribute `__array_ufunc__ = None` tells numpy to return `NotImplemented`, so Python calls `Tensor.__radd__` and the result is a Tensor with a graph.

**Grad mode.** Evaluation and sampling need no gradients, and a graph keeps every intermediate array alive until the output dies. `mode.py` holds a per-thread flag (`threading.local`), on by default. `no_grad()` is a context manager that saves the current value, sets it off, and restores **the saved value** in a `finally`, so blocks nest and an exception inside the block cannot leave training without gradients. `from_op` reads the flag: with it off, outputs have `requires_grad=False` and no parents.

## 3. Worked example by hand

Take $x = \begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$ (shape `(2, 3)`), $w = [10, 20, 30]$ (shape `(3,)`, broadcast down the rows), and $y = x \odot w + w$, with an upstream gradient of ones, $\bar{y} = \mathbf{1}_{2\times 3}$.

**Forward.** $x \odot w = \begin{bmatrix}10&40&90\\40&100&180\end{bmatrix}$, and adding $w$ to each row gives $y = \begin{bmatrix}20&60&120\\50&120&210\end{bmatrix}$.

**The graph.** Two ops: $m = x \odot w$, then $y = m + w$. The leaf $w$ has two consumers, the multiply and the add, so it must wait until both have reported.

**Backward through the add.** $\bar{m} = \bar{y} = \mathbf{1}$. The add's gradient for $w$ is $\bar{y}$ too, shape `(2, 3)`, but $w$ has shape `(3,)`: unbroadcast sums over axis 0, giving $[2, 2, 2]$.

**Backward through the multiply.** $\bar{x} = \bar{m} \odot w = \begin{bmatrix}10&20&30\\10&20&30\end{bmatrix}$: $w$ copied into every row, already $x$'s shape. The multiply's gradient for $w$ is $\bar{m} \odot x = x$, shape `(2, 3)`; unbroadcast sums the rows: $[1+4, 2+5, 3+6] = [5, 7, 9]$.

**Sum the two paths into $w$.** $\bar{w} = [5, 7, 9] + [2, 2, 2] = [7, 9, 11]$.

**Check one entry by perturbation.** Raise $w_0$ from 10 to $10 + h$: column 0 of $y$ becomes $(1 + 1)(10 + h)$ and $(4 + 1)(10 + h)$, so the summed output grows by $(2 + 5)h = 7h$. The slope is 7, the first entry of $\bar{w}$.

These numbers are the first case in section 4, and section 5 traces them through `backward`.

## 4. The interface

```python
# python/tinyllm/autograd/tensor.py
class Tensor:
    __array_ufunc__ = None
    data: NDArray; grad: Optional[NDArray]; requires_grad: bool
    def __init__(self, data: ArrayLike, requires_grad: bool = False, dtype=np.float32) -> None
    shape, dtype, ndim (properties); __len__
    def backward(self, grad: Optional[ArrayLike] = None) -> None
    def detach(self) -> "Tensor"; def numpy(self) -> NDArray
    # + - * / @ ** (by a number) unary -, reflected forms, __getitem__
def from_op(data, parents, vjp, op: str = "") -> Tensor

# python/tinyllm/autograd/mode.py
def is_grad_enabled() -> bool
@contextmanager
def no_grad() -> Iterator[None]
```


The contracts carry the exact rules: the constructor copies its data and rejects integer dtypes, `backward()` without an argument needs a one-element tensor, a VJP that returns a wrong-shaped gradient is a `RuntimeError` naming the op. The solution builds every operator on `from_op`, keeps `_parents` and `_vjp` as private attributes, and walks the graph with `toposort(self, lambda t: [p for p in t._parents if p.requires_grad])`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example, broadcast backward | unit | the section 3 numbers: $\bar{w} = [7, 9, 11]$, $\bar{x}$ = $w$ per row | the code and the math agree on broadcasting and reuse |
| matches the scalar `Value` | differential | the same expression built from `M08.2`'s scalar `Value` gives the same gradients | a Tensor op is a batch of scalar ops |
| add, sub, mul gradients | gradcheck | every binary op under five broadcasting patterns against central differences | biases, scales, and masks broadcast everywhere |
| div gradients | gradcheck | $\partial(a/b)/\partial b = -a/b^2$ | normalization layers divide |
| pow gradients | gradcheck | $p\,x^{p-1}$ for $p$ integer, fractional, negative | `L0.4`'s LayerNorm uses $(\cdot)^{-1/2}$ |
| matmul gradients | gradcheck | $\bar{Y}W^\top$, $X^\top\bar{Y}$ with 1-D operands and batch broadcasting | every linear layer |
| basic indexing gradients | gradcheck | slices and ints route the gradient to the selected entries | slicing activations |
| repeated indices accumulate | boundary | `x[[0, 2, 0]]`: row 0 gets both upstream rows | embedding lookups with repeated tokens |
| a shared input accumulates | unit | $a \cdot a + a$ at $a = 3$ gives 7 | weight tying, residual connections |
| diamond graph | unit | $d = 2a \cdot a^2$ at $a = 2$ gives 24 | any graph with a fork |
| `.grad` accumulates across `backward()` calls | property | `.grad` adds up over two `backward()` calls | micro-batch accumulation (`L11.1`) |
| `backward` on a constant | boundary | `backward` on a tensor without grad is an error | a forgotten `requires_grad` fails loudly |
| `backward` on a non-scalar without a gradient | boundary | no-argument `backward` on a vector is an error | an unreduced loss fails loudly |
| VJP shape mismatch | boundary | a wrong-shaped VJP result names the op | a missing unbroadcast in `L0.2` is found at once |
| a custom op on `from_op` | unit | a custom op on `from_op` with a constant parent | the seam `L0.2` builds every op on |
| gradient dtype | boundary | `.grad` has the data's dtype | in-place optimizer updates |
| constants mix in | unit | numbers, arrays, and flagless tensors are constants | masks and targets in model code |
| float32 stays float32 | boundary | a float32 tensor times a Python float stays float32 | Python agrees with the C kernels (`L9`) |
| reflected operators | unit | `2 - x`, `3 / x`, `A @ x` keep their operand order | constants on the left |
| an ndarray on the left | boundary | `ndarray + Tensor` is a Tensor with a graph | `mask * scores` in attention |
| the constructor copies and checks dtype | boundary | the leaf owns a copy; integer dtypes are rejected | a parameter is not an alias of a batch |
| `detach` stops the gradient | unit | `detach()` shares data and passes no gradient | targets and frozen values |
| `no_grad` builds no graph | property | ops inside `no_grad` keep no parents | evaluation memory (`L0.5`) |
| `no_grad` restores after an exception | boundary | an exception inside the block restores grad mode | a failed eval batch cannot stop training |
| `no_grad` nests | boundary | the inner block restores the outer block's "off" | nested evaluation helpers |
| a deep chain without recursion | boundary | a chain of 5,000 additions backpropagates | unrolled RNNs (`L3`) |
| shape and length | unit | `shape`, `dtype`, `ndim`, `len` read through | model code reads them every forward |

## 5. The solution, step by step

### From first principles

1. **Each op must carry its own backward rule.** The chain rule composes local derivatives, so the only thing an op has to know is how to turn $\bar{y}$ into $\bar{x}$ for its own inputs. That is one closure per output tensor: the VJP, capturing whatever forward values it needs (`x`, `y`, shapes).
2. **One constructor decides whether to record.** Every op does the same bookkeeping: compute the forward array, then attach parents and a VJP only if gradients are wanted. Putting that in `from_op` means the rule "record only when grad mode is on and some parent requires grad" is written once, and `L0.2` can add new ops without touching `Tensor`.
3. **Backward needs an order in which each node is finished before it is used.** A node's gradient is the sum over all its consumers, so it can only be passed to its own VJP after every consumer has run. Reverse topological order (root first, every node before its parents) guarantees that, and `M06.1`'s iterative `toposort` produces it without recursion.
4. **Pending gradients live in a dictionary keyed by identity, and every contribution is added.** Tensors overload `==` elementwise, so they cannot be dictionary keys themselves; `id(node)` can. Adding (never assigning) makes reuse correct: $a \cdot a + a$ at $a = 3$ collects $3 + 3$ from the multiply and $1$ from the add, total 7.
5. **Every binary VJP ends in `unbroadcast`.** Either operand may have been stretched by numpy, and broadcasting is reuse, so its gradient is a sum over the stretched axes. Calling `unbroadcast(g, operand.shape)` on both outputs is correct whether or not broadcasting happened (it is the identity when the shapes already match).
6. **Constants take the other operand's dtype, and gradients take their tensor's dtype.** That keeps a float32 model in float32 end to end: `x * 0.5` stays float32, and `p.data -= lr * p.grad` never promotes.
7. **Grad mode is a restored flag, not a toggled one.** `no_grad` must save the old value and restore it in `finally`, so nested blocks and exceptions both leave the flag as they found it. It is thread-local so that one thread evaluating cannot switch off gradients in another thread that is training.

### The code, annotated

**`from_op`: the seam every op is built on.**

```python
def from_op(data, parents, vjp, op: str = "") -> Tensor:
    parents = tuple(parents)
    t = Tensor.__new__(Tensor)        # skip __init__: no copy, keep the op's dtype
    t.data = np.asarray(data)
    t.grad = None
    t._op = op                        # named in error messages
    track = is_grad_enabled() and any(p.requires_grad for p in parents)
    t.requires_grad = track
    # Without tracking, keep nothing: no_grad must not hold the inputs alive.
    t._parents = parents if track else ()
    t._vjp = vjp if track else None
    return t
```

`Tensor.__new__` bypasses the constructor on purpose: `__init__` copies its data and defaults to float32, which is right for a leaf the caller hands in but wasteful (and dtype-changing) for an op's output. Dropping `parents` and `vjp` when not tracking is what makes `no_grad` save memory: the VJP closure captures the forward arrays, so keeping it would keep every intermediate alive.

**The leaf constructor.**

```python
def __init__(self, data, requires_grad: bool = False, dtype=np.float32) -> None:
    dt = np.dtype(dtype)
    if not np.issubdtype(dt, np.floating):     # an int tensor has no meaningful gradient
        raise ValueError(f"Tensor dtype must be floating point, got {dt}")
    self.data = np.array(data, dtype=dt)  # a copy: a leaf never aliases the caller's array
    self.grad = None
    self.requires_grad = bool(requires_grad)
    self._parents, self._vjp, self._op = (), None, ""   # a leaf has no history
```

**`backward`: steps 3 and 4.**

```python
def backward(self, grad=None) -> None:
    if not self.requires_grad:
        raise RuntimeError("backward() on a tensor that does not require grad")
    if grad is None:
        if self.data.size != 1:           # refuse to guess "sum of the outputs"
            raise RuntimeError(
                f"backward() without a gradient needs a one-element tensor, got shape {self.shape}"
            )
        g0 = np.ones_like(self.data)      # dL/dL = 1
    else:
        g0 = np.asarray(grad, dtype=self.data.dtype)
        # ... shape must equal self.shape
    # Root first, every node before its parents: when a node comes up, every
    # consumer has already added its share, so its gradient is complete.
    order = toposort(self, lambda t: [p for p in t._parents if p.requires_grad])
    grads: dict[int, NDArray] = {id(self): g0}
    for node in order:
        g = grads.pop(id(node), None)     # pop: the finished gradient is freed after use
        if g is None:
            continue
        if node._vjp is None:
            # A leaf: accumulate across backward() calls, as torch does.
            node.grad = g.copy() if node.grad is None else node.grad + g
            continue
        pgrads = node._vjp(g)
        # ... len(pgrads) must equal len(node._parents)
        for p, pg in zip(node._parents, pgrads):
            if pg is None or not p.requires_grad:   # constants get nothing
                continue
            pg = np.asarray(pg, dtype=p.data.dtype)  # gradients keep their tensor's dtype
            if pg.shape != p.shape:                  # a missing unbroadcast, caught here
                raise RuntimeError(
                    f"op {node._op or '?'}: gradient of shape {pg.shape} for an input of shape {p.shape}"
                )
            # A tensor used by several ops (or twice by one) sums every path.
            prev = grads.get(id(p))
            grads[id(p)] = pg if prev is None else prev + pg
```

The traversal filter `p.requires_grad` prunes constant subgraphs, so a mask or a target never enters the walk. The leaf branch copies on first write so `.grad` never aliases an array a VJP might still own. Walk the worked example through it: the order is $y, m, x, w$ (or $y, m, w, x$; both are valid). At $y$ the add's VJP posts $\bar{m} = \mathbf{1}$ and $[2,2,2]$ for $w$. At $m$ the multiply posts $\bar{x}$ and adds $[5,7,9]$ to $w$'s pending $[2,2,2]$. Only then does $w$ come up, with $[7, 9, 11]$.

**Elementwise operators: step 5 and step 6.**

```python
def __add__(self, other):
    a, b = self, _lift(other, self)       # a number or array becomes a constant of self's dtype
    sa, sb = a.shape, b.shape             # capture shapes, not the tensors
    return from_op(
        a.data + b.data, (a, b), lambda g: (unbroadcast(g, sa), unbroadcast(g, sb)), "add"
    )

def __sub__(self, other):
    # ... as __add__, with the second gradient negated
    return from_op(
        a.data - b.data, (a, b), lambda g: (unbroadcast(g, sa), unbroadcast(-g, sb)), "sub"
    )

def __mul__(self, other):
    a, b = self, _lift(other, self)
    x, y = a.data, b.data
    return from_op(
        x * y, (a, b),
        # each operand's gradient is g times the OTHER operand, reduced to its OWN shape
        lambda g: (unbroadcast(g * y, x.shape), unbroadcast(g * x, y.shape)),
        "mul",
    )

def __truediv__(self, other):
    a, b = self, _lift(other, self)
    x, y = a.data, b.data
    # d(x/y)/dx = 1/y, d(x/y)/dy = -x/y^2
    return from_op(
        x / y, (a, b),
        lambda g: (unbroadcast(g / y, x.shape), unbroadcast(-g * x / (y * y), y.shape)),
        "div",
    )

def __rsub__(self, other):
    return _lift(other, self) - self      # 2 - x, not x - 2: the operand order is kept

def _lift(other, like: Tensor) -> Tensor:
    """other as a Tensor; numbers and arrays become constants of like's dtype."""
    if isinstance(other, Tensor):
        return other
    return Tensor(np.asarray(other), requires_grad=False, dtype=like.data.dtype)
```

The reflected forms (`__radd__`, `__rmul__`, `__rtruediv__`, `__rmatmul__`) all lift the constant and call the forward operator with the operands in their original order, which is what keeps `2 - x` and `3 / x` correct. `unbroadcast` computes in float64 and the cast back to the parameter's dtype happens once, in `backward`.

**Matmul.** The only VJP with real shape bookkeeping, because numpy promotes 1-D operands.

```python
def __matmul__(self, other):
    a, b = self, _lift(other, self)
    A, B = a.data, b.data
    if A.ndim == 0 or B.ndim == 0:
        raise ValueError("matmul needs operands with at least one dimension")
    out = A @ B

    def vjp(g):
        # Promote 1-D operands to matrices the way numpy does: a vector on
        # the left is a row (1, k), on the right a column (k, 1).
        A2 = A[None, :] if A.ndim == 1 else A
        B2 = B[:, None] if B.ndim == 1 else B
        g2 = g
        if B.ndim == 1:  # restore the column axis first: a dot product's g is 0-d
            g2 = np.expand_dims(g2, -1)
        if A.ndim == 1:
            g2 = np.expand_dims(g2, -2)
        # C = A B  =>  dA = G B^T, dB = A^T G (per batch), then sum the
        # batch axes an operand was broadcast along.
        dA = unbroadcast(g2 @ np.swapaxes(B2, -1, -2), A2.shape)
        dB = unbroadcast(np.swapaxes(A2, -1, -2) @ g2, B2.shape)
        return dA.reshape(A.shape), dB.reshape(B.shape)   # drop the promoted axes again

    return from_op(out, (a, b), vjp, "matmul")
```

numpy removes the promoted axis from the output, so `g` is missing it too. The VJP puts the axes back in the same places, applies the 2-D rule batched over leading axes (`swapaxes(-1, -2)` is a transpose of the last two), sums any batch axes that were broadcast, and reshapes to the operand's original shape. The order of the two `expand_dims` matters: for a vector-vector dot product `g` is 0-d, and adding the column axis first then the row axis gives the `(1, 1)` matrix that matches `A2 @ B2`.

**Power, negation, indexing.**

```python
def __pow__(self, exponent: float):
    if isinstance(exponent, Tensor) or not np.isscalar(exponent):
        raise TypeError("Tensor ** x needs a Python number x")   # x ** w needs log x: not here
    x = self.data
    p = float(exponent)
    return from_op(x**p, (self,), lambda g: (g * p * x ** (p - 1),), "pow")

def __neg__(self):
    return from_op(-self.data, (self,), lambda g: (-g,), "neg")

def __getitem__(self, index):
    idx = index
    x = self.data

    def vjp(g):
        out = np.zeros_like(x)            # every entry not selected gets zero
        # add.at, not out[idx] = g: an id that appears twice in an integer
        # index gets both gradients.
        np.add.at(out, idx, g)
        return (out,)

    return from_op(x[idx], (self,), vjp, "getitem")
```

**`detach` and `numpy`.** `detach` builds a new tensor with `Tensor.__new__` that shares `self.data` (no copy) and has no parents and `requires_grad=False`, so it is a constant view of the same numbers. `numpy()` returns `self.data`.

Full solution: [`tensor.py`](../../../../course/solutions/python/tinyllm/autograd/tensor.py).

**Grad mode.**

```python
_local = threading.local()

def is_grad_enabled() -> bool:
    # A thread that never entered no_grad has no attribute yet: on by default.
    return getattr(_local, "enabled", True)

@contextmanager
def no_grad() -> Iterator[None]:
    prev = is_grad_enabled()
    _local.enabled = False
    try:
        yield
    finally:
        # Restore the previous value, not True: no_grad blocks nest, and an
        # exception inside the block must not leave grad mode off.
        _local.enabled = prev
```

Full solution: [`mode.py`](../../../../course/solutions/python/tinyllm/autograd/mode.py).

### Why this solution

- **Cost.** `backward` is $O(N + E)$ for the sort plus the cost of the VJPs, which mirror the forward ops (a matmul's backward is two matmuls of the same size). Memory is the forward graph: every VJP closure holds the arrays it needs until the graph is released, and `grads.pop` frees each finished gradient as soon as it has been handed on.
- **Graph on arrays, not scalars.** `M08.2`'s scalar `Value` is correct but needs one Python object and one call per number; a `[256, 256]` table is 65,536 nodes per op. One node per array turns each VJP into a single vectorized numpy expression.
- **Closures over a class per op.** A VJP as a lambda that captures `x`, `y`, and shapes is the smallest correct unit; a class hierarchy (`AddBackward`, `MulBackward`) would add boilerplate with no extra power at this size. PyTorch uses classes because its graph is in C++ and its nodes need saved-tensor version checks.
- **A full topological sort over dependency counting.** Counting each node's pending consumers and processing a node when its count reaches zero (PyTorch's engine) avoids the sort but needs a first pass to count anyway; the sort is simpler and has the same complexity.
- **Errors at the seam.** Checking each parent gradient's shape in `backward` costs one comparison per edge and turns a missing `unbroadcast` in any future op into an immediate error naming the op, instead of a silent broadcast inside a later addition.
- **Tradeoffs accepted.** No in-place op support and no version counters, so mutating `t.data` after using `t` in a graph silently corrupts the gradient; no higher-order gradients (the VJPs operate on raw arrays, not tensors).

### Explain it back

1. Why must `backward` visit nodes in reverse topological order rather than, say, breadth-first from the loss?

   <details><summary>Answer</summary>

   A node's gradient is the sum of its consumers' contributions, and its VJP must see that complete sum. Breadth-first order can reach a node through a short path before a longer path has delivered its share (in the diamond $d = b \cdot c$ with $b = 2a$, $c = a^2$, any order that processes $a$ before both $b$ and $c$ loses a term). Reverse topological order is exactly the condition "every consumer before the node".

   </details>

2. In the worked example, why does $w$ get $[7, 9, 11]$ and not $[5, 7, 9]$ or $[2, 2, 2]$?

   <details><summary>Answer</summary>

   $w$ feeds two ops, so its gradient is the sum of both paths: the multiply contributes $\mathrm{unbroadcast}(\bar{m} \odot x) = [5, 7, 9]$ and the add contributes $\mathrm{unbroadcast}(\bar{y}) = [2, 2, 2]$. Each path's contribution is itself a sum over the broadcast row axis. Either value alone means `backward` overwrote instead of adding.

   </details>

3. Why does `from_op` drop the parents and the VJP when it is not tracking, instead of just setting `requires_grad=False`?

   <details><summary>Answer</summary>

   The VJP closure captures forward arrays and the parents tuple holds the input tensors. Keeping them would keep the whole computation alive as long as the output lives, which defeats the point of evaluating under `no_grad`. Dropping them makes the output a plain constant.

   </details>

4. What happens to a float32 model if `_lift` converts `0.5` with numpy's default dtype?

   <details><summary>Answer</summary>

   `np.asarray(0.5)` is float64, and float32 times float64 is float64, so the output and everything downstream (and eventually the parameters, through their gradients) become float64. Memory doubles and results stop matching the float32 C and Rust kernels. `_lift` converts the constant to the other operand's dtype, so the product stays float32.

   </details>

5. Why does `no_grad` restore `prev` in a `finally` instead of setting the flag to `True` after the `yield`?

   <details><summary>Answer</summary>

   Two failure cases. Nesting: an inner block that sets `True` on exit turns gradients back on inside the outer `no_grad`. Exceptions: code after `yield` does not run when the block raises, so an exception during evaluation would leave the flag off and every later training step would build no graph and leave the loss flat. `finally` plus the saved value handles both.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. returning $\bar{y}$ as is for a broadcast operand | `ValueError` or a gradient of the output's shape for a bias; with unbroadcast to the *other* operand's shape, wrong sums | every binary VJP calls `unbroadcast` on each gradient with that operand's own shape; `backward` rejects a wrong-shaped gradient |
| 2. assigning a node's gradient instead of adding, or visiting a node before its consumers | $a \cdot a + a$ gives 3 or 6 instead of 7; the diamond gives half its gradient | `grads[id(p)] = prev + pg` and the reverse topological order from `toposort`; see steps 3 and 4 |
| 3. a leaf's `.grad` overwritten on each `backward()` | gradient accumulation over micro-batches silently uses the last one | the leaf branch adds `node.grad + g` when a gradient is already there |
| 4. `out[idx] = g` in the indexing VJP | a token used twice in a batch trains half as fast | `np.add.at(out, idx, g)` |
| 5. `no_grad` that sets the flag back to `True`, or restores only on a normal exit | after a failed eval batch, or inside nested blocks, training runs without gradients and the loss stays flat | saves `prev` and restores it in `finally` |
| 6. no `__array_ufunc__ = None` | `mask * scores` with an ndarray mask on the left is an array of Tensor objects | the class sets `__array_ufunc__ = None`, so numpy defers to `__rmul__` |
| 7. a missing transpose in the matmul VJP | square matrices give a gradient of the right shape and the wrong values | `g2 @ swapaxes(B2, -1, -2)` and `swapaxes(A2, -1, -2) @ g2` |
| 8. $\partial(a/b)/\partial b = +a/b^2$ | division layers train in the wrong direction | `-g * x / (y * y)` |
| 9. `backward()` on a non-scalar silently seeding ones | an unreduced `[B]` loss trains on its sum without anyone deciding so | `backward()` without a gradient requires `data.size == 1` |
| 10. constants converted to their own dtype | `x * 0.5` makes a float32 model float64 | `_lift` uses the other operand's dtype |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M06.1` | `toposort` orders the graph from the loss to the leaves, iteratively |
| Back | `M08.3` | `unbroadcast` and the matmul VJP are the derivations this module runs |
| Back | `M08.2` | the scalar `Value` graph, an independent oracle for the same gradients |
| Back | `lang.01` | numpy broadcasting rules and `np.add.at` |
| Forward | `L0.2` | every op of the library is a `from_op` call with one VJP |
| Forward | `L0.3` | the fused losses are single `from_op` nodes |
| Forward | `L0.4` | parameters are Tensors with `requires_grad=True` |
| Forward | `L0.5` | `train_step` calls `backward`, `evaluate` runs under `no_grad` |
| Forward | `L0.6` | the checkpoint tests train a model with this autograd before saving it |
| Forward | `L13.1` | `Conv2d` and the pools are `from_op` nodes with an im2col backward |
| Forward | `L13.2` | position tables are constant `Tensor`s outside the graph |
| Forward | `L13.3` | patch tokens, the CLS token, and position tables are `Tensor`s |
| Forward | `L13.5` | both losses are `from_op` nodes over the similarity matrix |
| Forward | `L14.2` | the Whisper stem and blocks run on `Tensor` |
| Forward | `L14.3` | Whisper decoding runs on `Tensor` |
| Forward | `L11.1`, `L2.2`, `L3.1` to `L3.6`, `L4.1` to `L4.3`, `L5.1` to `L5.5`, `L6.1` to `L6.7`, `L7.1` to `L7.9`, `L8.2`, `L8.5` | every trained model, objective, and inference module builds its graph on `Tensor` and `from_op` |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Tensor` + `from_op` | PyTorch autograd | a C++ graph of `Node`s, saved tensors with version counters to catch in-place edits, hooks | `torch/csrc/autograd/engine.cpp`, `function.h` |
| `backward` in topological order | PyTorch's engine | a ready queue per device and dependency counts instead of a full sort | `Engine::evaluate_function` in `engine.cpp` |
| `unbroadcast` | `sum_to_size` | the same reduction, emitted by the derivative generator for every broadcasting op | `tools/autograd/derivatives.yaml` |
| `no_grad` | `torch.no_grad`, `torch.inference_mode` | inference mode also skips version counters and view tracking | `torch/autograd/grad_mode.py` |
| reverse mode on arrays | JAX `vjp` | a functional transform over traced programs, with `jit` compiling the backward pass | `jax/_src/interpreters/ad.py` |
