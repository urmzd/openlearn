<!-- ol:module L0.1 -->
# Tensor, broadcasting backward, and no_grad

## Overview

| | |
|---|---|
| **Module** | `L0.1` · build · Python · Pass 2 · 4 to 6 h |
| **You build** | `python/tinyllm/autograd/tensor.py`: `Tensor` (arithmetic, `@`, `**`, indexing, `backward`, `detach`, `numpy`) and `from_op`; `python/tinyllm/autograd/mode.py`: `is_grad_enabled`, `no_grad` |
| **Contract** | [`course/contracts/py/tinyllm/autograd/tensor.pyi`](../../../../course/contracts/py/tinyllm/autograd/tensor.pyi) · [`course/contracts/py/tinyllm/autograd/mode.pyi`](../../../../course/contracts/py/tinyllm/autograd/mode.pyi) |
| **Tests** | `course/tests/L0.1/` (what they check: section 4) |
| **Needs** | `M06.1` iterative topological sort (`toposort`) · `M08.3` `unbroadcast` and the matmul VJP · `M08.2` the scalar `Value` (the tests' oracle) · reading: `lang.01` broadcasting (or `--ref-deps`) |
| **Used by** | `L0.2` op library · `L0.3` fused losses · `L0.4` modules · `L0.5` training loop · `L0.6` checkpoint tests · later: `L11.1`, `L2.2`, `L3.1`, `L3.2`, `L3.3`, `L3.4`, `L3.6`, `L4.1`, `L4.2`, `L4.3`, `L5.1`, `L5.3`, `L5.4`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5`, `L6.6`, `L6.7`, `L7.1`, `L7.2`, `L7.3`, `L7.5`, `L7.6`, `L7.7`, `L7.8`, `L7.9`, `L8.2`, `L8.5`, `L13.1`, `L13.2`, `L13.3`, `L13.5`, `L14.2`  · `L14.3` Whisper decoding |
| **Milestone** | `MS-L0` (your autograd retrains the tracer and survives a kill) |
| **Optional depth** | Baydin, Pearlmutter, Radul, Siskind, "Automatic Differentiation in Machine Learning: a Survey" (JMLR 2018, free); Karpathy's `micrograd` (the scalar version of this module) |

## Key Takeaways

- A tensor op records its inputs and one vector-Jacobian product; `backward` calls each VJP once, in reverse topological order, so a node's gradient is complete before it is split among its inputs (`test_diamond_graph`, `test_deep_chain_no_recursion`).
- A value used in several places gets the **sum** of the gradients from every place, whether the reuse is explicit (`a * a + a`) or hidden in broadcasting (`test_shared_input_accumulates`, `test_hand_example_broadcast_backward`).
- Undoing broadcasting is a sum over the stretched axes, and indexing with repeated indices is a scatter-add, never an assignment (`test_add_sub_mul_grads`, `test_getitem_repeated_indices_accumulate`).
- Grad mode is a per-thread flag that `no_grad` turns off and always restores, so evaluation builds no graph and a failed batch cannot leave training broken (`test_no_grad_builds_no_graph`, `test_no_grad_restores_after_exception`).
- A tensor keeps its dtype: constants are converted to it, gradients have it, so float32 models stay float32 (`test_float32_stays_float32`, `test_grad_dtype_matches_data`).

## How to work this chapter

```bash
ol start L0.1              # stubs tensor.py and mode.py into your repo
ol tests L0.1              # read the test catalog first: rung R0, you write no graded tests here
ol check L0.1              # exit code is the verdict
ol check L0.1 --ref-deps   # only if M06.1, M08.2, or M08.3 is not passing yet
ol diff  L0.1              # after passing: your code against the reference
```

`tinyllm` stays a namespace package: no `__init__.py` under `python/tinyllm/autograd/`. Import as `from tinyllm.autograd.tensor import Tensor`.

---

## 1. Why now

Your bigram from Pass 1 is trained by counting. That works for exactly one model: the moment a model has a hidden layer (the digits MLP of `MS-L0`), a gate, or attention, there is no count to take and the weights must be found by gradient descent. `M08.2` gave you reverse mode on scalars, one `Value` per number. A `[256, 256]` table is 65,536 `Value` objects and one Python call per multiply; the digits MLP would take minutes per step. This module moves reverse mode from scalars to numpy arrays, so one node holds a whole matrix and one VJP is one vectorized numpy expression. Every model in the rest of the course trains through the `Tensor` you write here.

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

These numbers are `test_hand_example_broadcast_backward`, the first test in section 4.

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

The contracts carry the exact rules: the constructor copies its data and rejects integer dtypes, `backward()` without an argument needs a one-element tensor, a VJP that returns a wrong-shaped gradient is a `RuntimeError` naming the op. Build every operator on `from_op`, keep `_parents` and `_vjp` as private attributes, and walk the graph with `toposort(self, lambda t: [p for p in t._parents if p.requires_grad])`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_broadcast_backward` | unit | the section 3 numbers: $\bar{w} = [7, 9, 11]$, $\bar{x}$ = $w$ per row | you and the test agree on broadcasting and reuse |
| `test_matches_scalarized_value` | differential | the same expression built from `M08.2`'s scalar `Value` gives the same gradients | a Tensor op is a batch of scalar ops |
| `test_add_sub_mul_grads` | gradcheck | every binary op under five broadcasting patterns against central differences | biases, scales, and masks broadcast everywhere |
| `test_div_grads` | gradcheck | $\partial(a/b)/\partial b = -a/b^2$ | normalization layers divide |
| `test_pow_grads` | gradcheck | $p\,x^{p-1}$ for $p$ integer, fractional, negative | `L0.4`'s LayerNorm uses $(\cdot)^{-1/2}$ |
| `test_matmul_grads` | gradcheck | $\bar{Y}W^\top$, $X^\top\bar{Y}$ with 1-D operands and batch broadcasting | every linear layer |
| `test_getitem_basic_grads` | gradcheck | slices and ints route the gradient to the selected entries | slicing activations |
| `test_getitem_repeated_indices_accumulate` | boundary | `x[[0, 2, 0]]`: row 0 gets both upstream rows | embedding lookups with repeated tokens |
| `test_shared_input_accumulates` | unit | $a \cdot a + a$ at $a = 3$ gives 7 | weight tying, residual connections |
| `test_diamond_graph` | unit | $d = 2a \cdot a^2$ at $a = 2$ gives 24 | any graph with a fork |
| `test_grad_accumulates_across_backward_calls` | property | `.grad` adds up over two `backward()` calls | micro-batch accumulation (`L11.1`) |
| `test_backward_on_constant_raises` | boundary | `backward` on a tensor without grad is an error | a forgotten `requires_grad` fails loudly |
| `test_backward_needs_grad_for_non_scalar` | boundary | no-argument `backward` on a vector is an error | an unreduced loss fails loudly |
| `test_vjp_shape_mismatch_raises` | boundary | a wrong-shaped VJP result names the op | a missing unbroadcast in `L0.2` is found at once |
| `test_from_op_custom_op` | unit | a custom op on `from_op` with a constant parent | the seam `L0.2` builds every op on |
| `test_grad_dtype_matches_data` | boundary | `.grad` has the data's dtype | in-place optimizer updates |
| `test_constants_mix_in` | unit | numbers, arrays, and flagless tensors are constants | masks and targets in model code |
| `test_float32_stays_float32` | boundary | a float32 tensor times a Python float stays float32 | Python agrees with the C kernels (`L9`) |
| `test_reflected_operators` | unit | `2 - x`, `3 / x`, `A @ x` keep their operand order | constants on the left |
| `test_ndarray_on_the_left` | boundary | `ndarray + Tensor` is a Tensor with a graph | `mask * scores` in attention |
| `test_constructor_copies_and_checks_dtype` | boundary | the leaf owns a copy; integer dtypes are rejected | a parameter is not an alias of a batch |
| `test_detach_stops_gradient` | unit | `detach()` shares data and passes no gradient | targets and frozen values |
| `test_no_grad_builds_no_graph` | property | ops inside `no_grad` keep no parents | evaluation memory (`L0.5`) |
| `test_no_grad_restores_after_exception` | boundary | an exception inside the block restores grad mode | a failed eval batch cannot stop training |
| `test_no_grad_nests` | boundary | the inner block restores the outer block's "off" | nested evaluation helpers |
| `test_deep_chain_no_recursion` | boundary | a chain of 5,000 additions backpropagates | unrolled RNNs (`L3`) |
| `test_shape_and_len` | unit | `shape`, `dtype`, `ndim`, `len` read through | model code reads them every forward |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. returning $\bar{y}$ as is for a broadcast operand | `ValueError` or a gradient of the output's shape for a bias; with unbroadcast to the *other* operand's shape, wrong sums | `test_add_sub_mul_grads` (mutants `s01`, `s20`), `test_hand_example_broadcast_backward` |
| 2. assigning a node's gradient instead of adding, or visiting a node before its consumers | $a \cdot a + a$ gives 3 or 6 instead of 7; the diamond gives half its gradient | `test_shared_input_accumulates`, `test_diamond_graph` (mutants `s02`, `s17`) |
| 3. a leaf's `.grad` overwritten on each `backward()` | gradient accumulation over micro-batches silently uses the last one | `test_grad_accumulates_across_backward_calls` (mutant `s03`) |
| 4. `out[idx] = g` in the indexing VJP | a token used twice in a batch trains half as fast | `test_getitem_repeated_indices_accumulate` (mutant `s04`) |
| 5. `no_grad` that sets the flag back to `True`, or restores only on a normal exit | after a failed eval batch, or inside nested blocks, training runs without gradients and the loss stays flat | `test_no_grad_restores_after_exception` (mutant `s05`), `test_no_grad_nests` (mutant `s06`) |
| 6. no `__array_ufunc__ = None` | `mask * scores` with an ndarray mask on the left is an array of Tensor objects | `test_ndarray_on_the_left` (mutant `s14`) |
| 7. a missing transpose in the matmul VJP | square matrices give a gradient of the right shape and the wrong values | `test_matmul_grads` (mutant `s07`) |
| 8. $\partial(a/b)/\partial b = +a/b^2$ | division layers train in the wrong direction | `test_div_grads` (mutant `s08`) |
| 9. `backward()` on a non-scalar silently seeding ones | an unreduced `[B]` loss trains on its sum without anyone deciding so | `test_backward_needs_grad_for_non_scalar` (mutant `s13`) |
| 10. constants converted to their own dtype | `x * 0.5` makes a float32 model float64 | `test_float32_stays_float32` (mutant `s10`) |

## 6. Where it's used next
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L11.1` | Registered call site uses this module. |
| Forward | `L2.2` | Registered call site uses this module. |
| Forward | `L3.1` | Registered call site uses this module. |
| Forward | `L3.2` | Registered call site uses this module. |
| Forward | `L3.3` | Registered call site uses this module. |
| Forward | `L3.4` | Registered call site uses this module. |
| Forward | `L3.6` | Registered call site uses this module. |
| Forward | `L4.1` | Registered call site uses this module. |
| Forward | `L4.2` | Registered call site uses this module. |
| Forward | `L4.3` | Registered call site uses this module. |
| Forward | `L5.1` | Registered call site uses this module. |
| Forward | `L5.3` | Registered call site uses this module. |
| Forward | `L5.4` | Registered call site uses this module. |
| Forward | `L5.5` | Registered call site uses this module. |
| Forward | `L6.1` | Registered call site uses this module. |
| Forward | `L6.2` | Registered call site uses this module. |
| Forward | `L6.3` | Registered call site uses this module. |
| Forward | `L6.5` | Registered call site uses this module. |
| Forward | `L6.6` | Registered call site uses this module. |
| Forward | `L6.7` | Registered call site uses this module. |
| Forward | `L7.1` | Registered call site uses this module. |
| Forward | `L7.2` | Registered call site uses this module. |
| Forward | `L7.3` | Registered call site uses this module. |
| Forward | `L7.5` | Registered call site uses this module. |
| Forward | `L7.6` | Registered call site uses this module. |
| Forward | `L7.7` | Registered call site uses this module. |
| Forward | `L7.8` | Registered call site uses this module. |
| Forward | `L7.9` | Registered call site uses this module. |
| Forward | `L8.2` | Registered call site uses this module. |
| Forward | `L8.5` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M06.1` | `toposort` orders the graph from the loss to the leaves, iteratively |
| Back | `M08.3` | `unbroadcast` and the matmul VJP are the derivations this module runs |
| Back | `M08.2` | the scalar `Value` graph, the oracle for the differential test |
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

If you skip this module, `ol check L0.2` stops with `L0.2 needs L0.1`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Tensor` + `from_op` | PyTorch autograd | a C++ graph of `Node`s, saved tensors with version counters to catch in-place edits, hooks | `torch/csrc/autograd/engine.cpp`, `function.h` |
| `backward` in topological order | PyTorch's engine | a ready queue per device and dependency counts instead of a full sort | `Engine::evaluate_function` in `engine.cpp` |
| `unbroadcast` | `sum_to_size` | the same reduction, emitted by the derivative generator for every broadcasting op | `tools/autograd/derivatives.yaml` |
| `no_grad` | `torch.no_grad`, `torch.inference_mode` | inference mode also skips version counters and view tracking | `torch/autograd/grad_mode.py` |
| reverse mode on arrays | JAX `vjp` | a functional transform over traced programs, with `jit` compiling the backward pass | `jax/_src/interpreters/ad.py` |
