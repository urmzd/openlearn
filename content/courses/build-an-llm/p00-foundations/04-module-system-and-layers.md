# Module system and basic layers

## Overview

| | |
|---|---|
| **Module** | `L0.4` · build · Python · Pass 2 · 2 to 3 h |
| **Solution** | [`module.py`](../../../../course/solutions/python/tinyllm/nn/module.py): `Module` (registration in `__setattr__`, `named_parameters`, `named_modules`, `state_dict`, `load_state_dict`, `train`/`eval`, `zero_grad`) · [`layers.py`](../../../../course/solutions/python/tinyllm/nn/layers.py): `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `ReLU`, `Tanh`, `GELU`, `Sequential`, `ModuleList` · model tests: [`test_module.py`](../../../../course/solutions/learner-tests/L0.4/python/tests/l0-4-module/test_module.py) |
| **Contract** | [`course/contracts/py/tinyllm/nn/module.pyi`](../../../../course/contracts/py/tinyllm/nn/module.pyi) · [`course/contracts/py/tinyllm/nn/layers.pyi`](../../../../course/contracts/py/tinyllm/nn/layers.pyi) |
| **Needs** | `L0.1` `Tensor` (parameters) · `L0.2` the ops every forward is written in · `M07.3` `normal_init` · `M06.3` PCG32 init and dropout streams · reading: `craft.03` red then green |
| **Used by** | `L0.5` `BigramLogits` is a `Module`, the loop trains `Linear` stacks, and the digits MLP that closes Part 0 is two `Linear` layers · `L0.6` a checkpoint is a module's `state_dict` · later: `L11.1`, `L2.2`, `L3.2`, `L3.3`, `L3.6`, `L4.1`, `L4.2`, `L4.3`, `L5.3`, `L5.4`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5`, `L6.6`, `L7.1`, `L7.2`, `L7.5`, `L7.6`, `L7.8`, `L7.9`, `L8.5`, `L13.1`, `L13.3`, `L13.5`, `L14.2`  · `L14.3` Whisper decoding |
| **Optional depth** | the PyTorch `torch.nn.Module` documentation and source (`torch/nn/modules/module.py`); Ba, Kiros, Hinton, "Layer Normalization" (2016) |

## Key Takeaways

- A module finds its parameters and children by attribute assignment, in assignment order, and the dotted names this produces are the safetensors keys of every checkpoint in the course, identical to PyTorch's.
- `load_state_dict` matches by name, checks every key and shape before copying anything, and writes into the existing arrays, so an optimizer built earlier keeps training the loaded weights.
- `Linear` stores one row per output, $W \in \mathbb{R}^{\text{out} \times \text{in}}$, and computes $xW^\top + b$, PyTorch's layout, which is what lets HF weights load by name.
- LayerNorm uses the population variance with $\epsilon$ inside the square root.
- `eval()` and `train()` reach every descendant, and a weight tied under two names is one parameter.

## How to read this chapter

Sections 1 to 3 build the idea from first principles and work a `Linear` layer and a LayerNorm row by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through the solution in `module.py` and `layers.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

With `L0.1` to `L0.3` you can train anything, but a model is still a loose set of Tensors you have to pass to the optimizer and to the checkpoint writer by hand, in the right order, every time. The digits MLP has four parameter arrays; a Llama layer (`L7.9`) has nine, a 30-layer model 270. Worse, the checkpoint format (`formats/safetensors.md`) names tensors by strings such as `model.layers.3.self_attn.q_proj.weight`, and the Rust engine (`L10.1`) and HF's loaders find weights by those names. This module gives every model one way to list its parameters, name them exactly as PyTorch does, save and load them, and switch dropout off for evaluation.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | input batch, rows of $d_{\text{in}}$ features | `float32[..., d_in]` |
| $W$ | `Linear` weight, one row per output | `float32[d_out, d_in]` |
| $b$ | `Linear` bias | `float32[d_out]` |
| $y = xW^\top + b$ | `Linear` output | `float32[..., d_out]` |
| $\bar{y}$ | upstream gradient | shape of $y$ |
| $\mu, \sigma^2$ | mean and population variance of one row (over the last axis, $d$ entries) | scalars per row |
| $\epsilon$ | LayerNorm's guard, $10^{-5}$ | float |
| $\gamma, \beta$ | LayerNorm's affine weight and bias | `float32[d]` |
| $\mathcal{N}(0, s^2)$ | normal distribution with standard deviation $s$ (`M07.3`) | |

**Registration by assignment.** `Module.__init__` creates two ordered dictionaries, parameters and children, before anything else. `__setattr__` then sorts every assignment: a `Tensor` with `requires_grad=True` is a parameter, a `Module` is a child, anything else (a number, a constant Tensor such as a causal mask, `None`) is plain state and unregisters the name. That is why a subclass must call `super().__init__()` first: without the dictionaries there is nowhere to register, and the error should say so. Python dictionaries keep insertion order, so registration order is assignment order.

**Names are the contract.** `named_parameters()` yields the module's own parameters first, then each child's with the child's name and a dot as prefix, depth first. A two-layer `Sequential(Linear(2, 3), ReLU(), Linear(3, 1))` yields `0.weight`, `0.bias`, `2.weight`, `2.bias`: exactly `torch.nn.Sequential`'s `state_dict` keys. A Tensor registered under two names (tied input and output embeddings in `L7.9`) is listed once, under its first name, by identity; otherwise the optimizer would step it twice.

**state_dict and load_state_dict.** `state_dict()` is `{name: copy of the array}` in `named_parameters` order; copies, so a saved state does not change when training continues. `load_state_dict(sd, strict=True)` first compares keys: with `strict`, any missing or unexpected key is a `KeyError` that names all of them, raised **before** anything is copied (a half-loaded model is worse than none). A shape mismatch is a `ValueError`, also before copying. Then each array is written **into** the existing parameter array (`p.data[...] = a`), converted to the parameter's dtype. The optimizer (`M10.2`, `M10.3`) holds references to those Tensors and arrays; replacing them would leave it updating arrays the model no longer uses. `strict=False` loads the names that match, which is how you load a pretrained backbone under a new head.

**train and eval.** `training` is a flag on every module; `Dropout` reads it. `train(mode)` sets it on the module and every descendant and returns the module; `eval()` is `train(False)`. Setting it only on the top module leaves every nested dropout active during evaluation.

**Linear in PyTorch's layout.** $W$ has shape `[d_out, d_in]`, so $y = xW^\top + b$, and backward (from `L0.1`'s matmul VJP) gives $\bar{W} = \bar{y}^\top x$, $\bar{b} = \sum_{\text{rows}} \bar{y}$, $\bar{x} = \bar{y}W$. Storing $W$ as `[d_in, d_out]` computes the same function from its own initialization, and then fails the day you load torch weights by name. Initialization follows `M07.3`: $W \sim \mathcal{N}(0, 1/d_{\text{in}})$ (standard deviation $1/\sqrt{d_{\text{in}}}$), so unit-variance inputs give unit-variance outputs, and $b = 0$. `Embedding(n, d)` is a table `[n, d]` with $\mathcal{N}(0, 1)$ rows and `F.embedding` as its forward.

**Seeded initialization.** Each layer takes an optional `rng`, a PCG32 (`M06.3`), and draws its weights from it once, at construction, in parameter registration order. With no `rng`, layers use the spec's sub-streams of seed 0: `PCG32(0).substream("init")` for weights and `substream("dropout")` for dropout masks, so adding a dropout layer does not change any initial weight.

**LayerNorm.** For each row (the last axis): $\hat{x} = (x - \mu)/\sqrt{\sigma^2 + \epsilon}$, then $y = \gamma \odot \hat{x} + \beta$, with $\sigma^2$ the **population** variance (`F.var` with correction 0, as torch). $\epsilon$ belongs inside the square root: it guards rows whose variance is tiny, and outside the root it does almost nothing there. The forward is written with `F.mean`, `F.var`, and `**`, so its backward comes from `L0.2` and needs no code here.

**Containers.** `Sequential(*mods)` registers its children as `"0"`, `"1"`, ... and applies them in order. `ModuleList(mods)` only holds them (no forward); `append` registers the next index and returns the list.

## 3. Worked example by hand

`Linear(2, 3)` with
$W = \begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}$, $b = [0.5, -0.5, 1]$, and one input row $x = [[1, 1]]$.

**Forward.** $xW^\top$ takes the dot product of $x$ with each row of $W$: $[1 + 2, 3 + 4, 5 + 6] = [3, 7, 11]$. Add $b$: $y = [[3.5, 6.5, 12]]$.

**Backward** with $\bar{y} = [[1, 1, 1]]$:

1. $\bar{W} = \bar{y}^\top x = \begin{bmatrix}1\\1\\1\end{bmatrix}[1, 1]$, a $3 \times 2$ matrix of ones: each weight $W_{ij}$ multiplied $x_j = 1$ once.
2. $\bar{b} = [1, 1, 1]$: the bias is added once per output.
3. $\bar{x} = \bar{y}W = [1 + 3 + 5, 2 + 4 + 6] = [[9, 12]]$: each input feeds all three outputs.

**LayerNorm on a nearly constant row.** $x = [0, 0.002]$: $\mu = 0.001$, $\sigma^2 = 10^{-6}$. With $\epsilon = 10^{-5}$ inside the root, $\hat{x} = \pm 0.001/\sqrt{1.1 \times 10^{-5}} = \pm 0.30151$, torch's value. With $\epsilon$ added after the root, $\pm 0.001/(0.001 + 10^{-5}) = \pm 0.990$.

These two hand results are the first cases a correct solution must reproduce.

## 4. The interface

```python
# python/tinyllm/nn/module.py
class Module:
    training: bool
    def __init__(self) -> None; def __setattr__(self, name, value) -> None
    def forward(self, *args, **kwargs); def __call__(self, *args, **kwargs)
    def named_parameters(self, prefix="") -> Iterator[tuple[str, Tensor]]; def parameters(self)
    def named_modules(self, prefix="") -> Iterator[tuple[str, "Module"]]
    def train(self, mode=True) -> "Module"; def eval(self) -> "Module"; def zero_grad(self) -> None
    def state_dict(self) -> dict[str, NDArray]; def load_state_dict(self, sd, strict=True) -> None

# python/tinyllm/nn/layers.py
Linear(in_f, out_f, bias=True, rng=None); Embedding(n, d, rng=None); LayerNorm(d, eps=1e-5)
Dropout(p, rng=None); ReLU(); Tanh(); GELU(approximate="none"); Sequential(*mods); ModuleList(mods=())
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3: $y = [[3.5, 6.5, 12]]$, $\bar{W}$ ones, $\bar{b}$ ones, $\bar{x} = [[9, 12]]$ | the layout is the one everyone agrees on |
| parity with torch | golden | every layer with torch's weights copied in by name gives torch's outputs and gradients | HF weights load by name (`L7.9`) |
| state_dict names match torch | golden | the exact keys and order torch produces, through nested containers | the safetensors key contract |
| state_dict round trip | property | save, rebuild with another seed, load: same function; saved arrays are copies | checkpoints (`L0.6`) |
| load is in place | unit | the Tensors and arrays an optimizer holds are the ones loaded into | resuming a run (`L0.6`) |
| strict loading | boundary | strict names every missing and unexpected key and copies nothing; wrong shapes fail; non-strict loads the rest | a wrong checkpoint fails loudly |
| parameter order and tying | unit | own parameters before children's; a tied Tensor once | weight tying (`L7.9`) |
| plain state is not a parameter | boundary | constants and numbers are not parameters; `None` unregisters | masks and buffers in attention (`L5`) |
| forgetting `super().__init__()` | boundary | the error names `super().__init__()` | the most common first bug |
| train/eval reach every child | unit | `eval()` reaches nested dropout; both return the model | evaluation is deterministic (`L0.5`) |
| zero_grad | unit | every parameter's `.grad` is cleared | gradients accumulate (`L0.1`) |
| LayerNorm normalizes | property | rows come out with mean 0 and variance 1 | every transformer block (`L5`) |
| $\epsilon$ inside the square root | boundary | section 3's $\pm 0.30151$ | torch parity on small-variance rows |
| init statistics and seeding | statistical | $\mathcal{N}(0, 1/d_{\text{in}})$ weights, zero bias, seed-reproducible | stable signal through depth (`M07.3`) |
| Sequential and ModuleList | unit | order, indexing, `append` | stacks of blocks (`L5`, `L7`) |

### The model tests

This module is one where tests come first (`craft.03`): each behaviour is pinned by a test that fails before the code exists and passes after. The model suite, [`test_module.py`](../../../../course/solutions/learner-tests/L0.4/python/tests/l0-4-module/test_module.py), imports only contract names and covers, in order: the hand example above; state_dict names in registration order with no key for a missing bias; loading by name (a reordered dict gives the same parameters); loading in place; strict mode rejecting missing and unexpected keys without changing anything; a tied parameter listed once; `eval()` turning dropout off in nested modules; LayerNorm against its formula. Its first test is the hand example:

```python
# python/tests/l0-4-module/test_module.py
def test_hand_example_linear():
    """W = [[1, 2], [3, 4], [5, 6]], b = [0.5, -0.5, 1], x = [[1, 1]]: y = [[3.5, 6.5, 12]], dx = [[9, 12]]."""
    lin = Linear(2, 3, rng=Gen())  # Gen: a tiny deterministic stand-in generator defined in the file
    lin.load_state_dict({"weight": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), "bias": np.array([0.5, -0.5, 1.0])})
    x = Tensor([[1.0, 1.0]], requires_grad=True)
    y = lin(x)
    np.testing.assert_allclose(y.data, [[3.5, 6.5, 12.0]])
    y.backward(np.ones((1, 3)))
    np.testing.assert_allclose(x.grad, [[9.0, 12.0]])
    np.testing.assert_allclose(lin.weight.grad, np.ones((3, 2)))
    np.testing.assert_allclose(lin.bias.grad, [1.0, 1.0, 1.0])
```

The one fault a good suite must catch is the last pitfall in section 6: a `Linear` that stores or applies its weight in the wrong orientation. A model trained from scratch cannot see it; only a test with known weights, like this one, can.

## 5. The solution, step by step

### From first principles

1. **Every model needs one list of its parameters, in a fixed order, with names.** The optimizer needs the list; the checkpoint needs the names; the order must be reproducible so that two builds of the same model agree. The names must also be PyTorch's, because HF checkpoints and the Rust engine look weights up by those strings.
2. **The cheapest place to collect that list is attribute assignment.** Model code already writes `self.fc = Linear(...)`. If `__setattr__` sorts each assignment into "parameter", "child module", or "plain state", the model author writes nothing extra, and Python's insertion-ordered dicts make registration order equal to assignment order. This is exactly what torch does, so torch's names fall out for free.
3. **The dictionaries must exist before the first assignment.** So `Module.__init__` creates them with `object.__setattr__` (bypassing its own hook), and the hook checks for them first and raises an error naming `super().__init__()` if they are missing.
4. **Names come from a depth-first walk.** Own parameters first, then each child with `child_name + "."` as prefix. For `Sequential(Linear(2, 3), ReLU(), Linear(3, 1))` this gives `0.weight`, `0.bias`, `2.weight`, `2.bias`; `ReLU` has no parameters, so `1` never appears.
5. **A tied weight must appear once.** Deduplicate by object identity, not by value or name, and keep the first name. Otherwise the optimizer steps the shared Tensor twice per update.
6. **Loading must be all or nothing, and in place.** Validate all keys, then all shapes, and only then copy. Copy into the existing arrays (`data[...] = a`), because the optimizer built before the load holds references to those arrays.
7. **Layers are compositions of `L0.2` ops.** `Linear` is `matmul` with a transposed weight, `LayerNorm` is `mean`, `var`, a power, and two elementwise ops. Their backward passes then come from autograd and need no code, and the worked example's $\bar{W} = \bar{y}^\top x$ is produced by the matmul VJP.

### The code, annotated

**Registration.** The hook is the whole module system. Note the two `pop` calls: a name can move between categories, and it must live in at most one dict.

```python
class Module:
    def __init__(self) -> None:
        # object.__setattr__: our own __setattr__ needs these dicts to exist.
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if "_params" not in self.__dict__:
            # Pitfall 1 surfaces here, with the fix in the message.
            raise AttributeError(
                f"{type(self).__name__}: call super().__init__() before assigning attributes"
            )
        params, children = self.__dict__["_params"], self.__dict__["_children"]
        if isinstance(value, Tensor) and value.requires_grad:
            children.pop(name, None)
            params[name] = value          # trainable Tensor: a parameter
        elif isinstance(value, Module):
            params.pop(name, None)
            children[name] = value        # a sub-module: walked for names
        else:
            # Re-assigning a name to plain state (None, a constant Tensor)
            # unregisters it: a Linear built with bias=False has no "bias".
            params.pop(name, None)
            children.pop(name, None)
        object.__setattr__(self, name, value)  # still a normal attribute
```

The test is `requires_grad`, not "is a Tensor": a causal mask stored as `self.mask = Tensor(...)` is state the model reads, not a weight the optimizer should move.

**Names and the walk.** One shared `seen` set across the whole recursion is what makes tying work across sub-modules (an embedding in one child, the output head in another).

```python
    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        seen: set[int] = set()
        out: list[tuple[str, Tensor]] = []

        def walk(pre: str, m: Module) -> None:
            for name, p in m._params.items():
                if id(p) not in seen:  # a tied weight is listed once, first name wins
                    seen.add(id(p))
                    out.append((pre + name, p))
            for name, c in m._children.items():
                walk(pre + name + ".", c)

        # Depth first, own parameters before children, in registration order.
        walk(prefix, self)
        return iter(out)

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        out = [(prefix, self)]
        for name, c in self._children.items():
            out.extend(c.named_modules(prefix + name if not prefix else f"{prefix}.{name}"))
        return iter(out)

    def train(self, mode: bool = True) -> "Module":
        # Every descendant, not just self: nested Dropout must switch too.
        for _, m in self.named_modules():
            object.__setattr__(m, "training", bool(mode))
        return self   # so `model.eval()` can be chained
```

**Saving and loading.** `state_dict` returns copies; `load_state_dict` runs in three phases: key check, shape check, copy.

```python
    def state_dict(self) -> dict[str, NDArray]:
        # Copies: a saved state must not change when training continues.
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True) -> None:
        own = dict(self.named_parameters())
        if strict:
            # Phase 1: every bad key reported at once, before any write.
            missing = sorted(set(own) - set(sd))
            unexpected = sorted(set(sd) - set(own))
            if missing or unexpected:
                raise KeyError(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
        arrays = {}
        for name, p in own.items():
            if name not in sd:
                continue                   # only reachable with strict=False
            a = np.asarray(sd[name])
            # Phase 2: exact shape, so a broadcastable array cannot slip in.
            if a.shape != p.shape:
                raise ValueError(f"{name}: shape {a.shape} in the state dict, {p.shape} in the model")
            arrays[name] = a
        for name, a in arrays.items():
            # Phase 3. In place: the optimizer holds these Tensors and their arrays.
            own[name].data[...] = a.astype(own[name].data.dtype, copy=False)
```

Matching is by name through `own[name]`, never by position, so a checkpoint whose keys were written in another order loads correctly.

**Linear and the default streams.** The weight is `[out_f, in_f]` and the forward transposes it, which is the torch layout from Principles.

```python
def _rng(rng: Any, purpose: str) -> Any:
    """rng, or the default stream for `purpose` (spec/pcg32.md sub-streams)."""
    return rng if rng is not None else PCG32(0).substream(purpose)

class Linear(Module):
    def __init__(self, in_f: int, out_f: int, bias: bool = True, rng: Any = None) -> None:
        super().__init__()
        if in_f < 1 or out_f < 1:
            raise ValueError(f"Linear({in_f}, {out_f}): sizes must be positive")
        r = _rng(rng, "init")
        # torch's layout: one row per output, so y = x @ W^T.
        # std 1/sqrt(in_f): unit-variance inputs give unit-variance outputs.
        self.weight = _param(normal_init((out_f, in_f), 1.0 / math.sqrt(in_f), r))
        self.bias = _param(np.zeros(out_f)) if bias else None  # None: unregistered

    def forward(self, x: Tensor) -> Tensor:
        y = F.matmul(x, F.transpose(self.weight, 0, 1))
        return y + self.bias if self.bias is not None else y
```

`Embedding` is the same pattern with an `[n, d]` table, $\mathcal{N}(0, 1)$ rows, and `F.embedding` as the forward. `Dropout` keeps its own generator from the `"dropout"` sub-stream and passes `self.training` to `F.dropout`, which is how `eval()` reaches it.

**LayerNorm.** Four lines, and each one is a decision from Principles.

```python
class LayerNorm(Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)              # plain state, not a parameter
        self.weight = _param(np.ones(d))   # gamma starts at 1: identity affine
        self.bias = _param(np.zeros(d))    # beta starts at 0

    def forward(self, x: Tensor) -> Tensor:
        mu = F.mean(x, axis=-1, keepdims=True)
        var = F.var(x, axis=-1, keepdims=True)  # population variance, as torch
        xhat = (x - mu) * (var + self.eps) ** -0.5   # eps inside the root
        return xhat * self.weight + self.bias
```

On section 3's row, `var + eps` is $1.1 \times 10^{-5}$ and `xhat` is $\pm 0.30151$.

**Containers.** `Sequential` registers through `setattr` with the names `"0"`, `"1"`, ..., so the ordinary hook does all the work and the keys come out as torch's.

```python
class Sequential(Module):
    def __init__(self, *mods: Module) -> None:
        super().__init__()
        for i, m in enumerate(mods):
            if not isinstance(m, Module):
                raise TypeError(f"Sequential item {i} is {type(m).__name__}, not a Module")
            setattr(self, str(i), m)  # names "0", "1", ...: torch's state_dict keys

    def forward(self, x: Any) -> Any:
        for m in self._children.values():   # registration order is call order
            x = m(x)
        return x

class ModuleList(Module):
    def append(self, m: Module) -> "ModuleList":
        if not isinstance(m, Module):
            raise TypeError(f"ModuleList holds Modules, got {type(m).__name__}")
        setattr(self, str(len(self._children)), m)  # next index
        return self
    # ...
```

Full solution: [`module.py`](../../../../course/solutions/python/tinyllm/nn/module.py) · [`layers.py`](../../../../course/solutions/python/tinyllm/nn/layers.py) (also `ReLU`, `Tanh`, `GELU`, and indexing for both containers).

### Why this solution

- **Cost.** `named_parameters` is $O(P + M)$ for $P$ parameters and $M$ modules; it builds a list and returns an iterator over it, so callers see a consistent snapshot even if they mutate the model while iterating. `load_state_dict` is $O(P)$ plus one copy of the weights. None of this is on the hot path; it runs once per optimizer build or checkpoint.
- **Alternatives.** An explicit `register_parameter` call per weight works but makes every model author repeat each name, and a forgotten call is a weight that silently never trains. Discovering parameters by scanning `vars(self)` on demand loses the ordering guarantee once attributes are reassigned and cannot distinguish a tied Tensor from a copy without the same identity check. Storing `Linear` weights as `[in, out]` saves a transpose and breaks every torch and HF checkpoint.
- **Accepted tradeoffs.** No buffers (saved but not trained arrays) and no hooks; constant Tensors are plain state and are not saved. Loading converts to the parameter's dtype rather than refusing a dtype mismatch, which is what torch does and what makes float64 test fixtures load into float32 models.

### Explain it back

1. Why does `Module.__init__` use `object.__setattr__` instead of `self._params = {}`?
   <details><summary>Answer</summary>
   `self._params = {}` would call the module's own `__setattr__`, which looks for `_params` in `__dict__` before it exists and raises. Bypassing the hook is the only way to create the dictionaries the hook depends on.
   </details>
2. Why is `seen` keyed by `id(p)` and shared across the whole walk?
   <details><summary>Answer</summary>
   Tying means the same Tensor object is registered under two names, possibly in different sub-modules. Identity is the property that distinguishes "the same weight" from "two weights that happen to be equal"; a per-module set would miss a tie that crosses modules. Without it the optimizer would apply two updates per step to one array.
   </details>
3. What breaks if `load_state_dict` did `own[name].data = a` instead of `own[name].data[...] = a`?
   <details><summary>Answer</summary>
   The parameter would point at a new array while the optimizer's moment buffers and any references it holds still refer to the old one. A resumed run would then update arrays the model no longer reads, and the loss would stop moving. Writing through `[...]` keeps the array identity.
   </details>
4. Why are the shape checks done in a separate loop before the copy loop?
   <details><summary>Answer</summary>
   So a bad checkpoint changes nothing. If shapes were checked while copying, the first few layers would be overwritten before the error, leaving a half-loaded model that looks valid.
   </details>
5. With $\epsilon$ outside the square root, why is the error large only on near-constant rows?
   <details><summary>Answer</summary>
   For a row with $\sigma^2 \gg \epsilon$, $\sqrt{\sigma^2 + \epsilon} \approx \sigma \approx \sigma + \epsilon$ and both forms agree. When $\sigma^2$ is comparable to $\epsilon$, $\sqrt{\epsilon}$ (about $3 \times 10^{-3}$) dominates $\epsilon$ itself, so inside the root it damps the row strongly and outside it barely does: section 3 gets $0.30151$ against $0.990$.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. assigning a parameter before `super().__init__()` | an `AttributeError` about `_params`, far from the cause | `__setattr__` checks for `_params` first and raises an error that names `super().__init__()` |
| 2. `train()`/`eval()` setting only the top module's flag | dropout stays on in nested blocks during evaluation | `train` sets the flag on every module from `named_modules()` |
| 3. `load_state_dict` replacing the arrays | a resumed run's optimizer updates arrays the model no longer uses: the loss stops moving | copies with `data[...] = a`, keeping each array's identity |
| 4. loading by position, checking keys after copying, or skipping the shape check | a reordered checkpoint loads into the wrong layers; a wrong one half-loads; a broadcastable array loads silently | looks up `own[name]`; checks all keys, then all exact shapes, then copies |
| 5. LayerNorm with the sample variance, or $\epsilon$ outside the root | small disagreements with torch everywhere, large ones on near-constant rows | `F.var` with its default correction 0, and `(var + eps) ** -0.5` |
| 6. `Linear` weight as `[in, out]`, or a forward without the transpose | trains fine from scratch, then torch and HF weights load transposed or fail | weight shape `(out_f, in_f)` and `matmul(x, transpose(W))`; pinned by the hand example |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | parameters are Tensors with `requires_grad=True` |
| Back | `L0.2` | `Linear` is `F.matmul` and `F.transpose`; `LayerNorm` is `F.mean` and `F.var`; `Embedding` is `F.embedding` |
| Back | `M07.3` | `normal_init` draws the initial weights |
| Back | `M06.3` | `PCG32(0).substream("init")` and `substream("dropout")` are the defaults |
| Forward | `L0.5` | `BigramLogits` subclasses `Module`; the loop trains `Sequential` MLPs |
| Forward | `L0.6` | `save_checkpoint` writes `state_dict()` and resume calls `load_state_dict` |
| Forward | `L13.1` | `Conv2d`, `Conv1d`, and LeNet-5 are `Module`s; the classifier head is `Linear` |
| Forward | `L13.3` | `ViT` stacks `LayerNorm`, `Linear`, and `Embedding` in a `ModuleList` |
| Forward | `L13.5` | the text tower and both projections are `Module`s |
| Forward | `L14.2` | `WhisperEncoder` stacks `LayerNorm` and `Linear` in a `ModuleList` |
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L11.1` | `train_step_mixed` steps `model.parameters()`; `checkpoint_sequential` segments a list of `Module` layers and gathers their `parameters()` |
| Forward | `L2.2` | the NPLM is a `Module` of `Embedding` and `Linear` layers |
| Forward | `L3.2` | the LSTM cell is a `Module` with `Dropout` and `xavier_uniform` weights |
| Forward | `L3.3` | the GRU cell is a `Module` with `Dropout` and `xavier_uniform` weights |
| Forward | `L3.6` | the RNN LM wraps the cell between an `Embedding` and a `Linear` read-out |
| Forward | `L4.1` | the encoder and decoder are `Module`s built from `Embedding` and `Linear` |
| Forward | `L4.2` | additive (Bahdanau) attention is a `Module` of `Linear` projections |
| Forward | `L4.3` | Luong attention is a `Module` of `Linear` projections |
| Forward | `L5.3` | multi-head attention's Q, K, V, and output projections are `Linear` |
| Forward | `L5.4` | the learned position table is a `Module` initialised with `normal_init` |
| Forward | `L5.5` | the transformer stacks `LayerNorm`, `Linear`, `Embedding`, and `Dropout` in a `ModuleList` |
| Forward | `L6.1` | GPT stacks its blocks in a `ModuleList` of `LayerNorm`, `Linear`, `Embedding`, and `Dropout` |
| Forward | `L6.2` | BERT's encoder stacks the same layers in a `ModuleList` |
| Forward | `L6.3` | the ELECTRA generator and discriminator heads are `Module`s over `Linear` |
| Forward | `L6.5` | the task heads are `Module`s of `Dropout` and `Linear` |
| Forward | `L6.6` | `LoRALinear` wraps a frozen `Linear` and adds two trainable `Linear` factors |
| Forward | `L7.1` | `RMSNorm` is a `Module` |
| Forward | `L7.2` | SwiGLU's gate, up, and down projections are `Linear` |
| Forward | `L7.5` | grouped-query attention's projections are `Linear` |
| Forward | `L7.6` | MLA's down- and up-projections are `Linear` |
| Forward | `L7.8` | the MoE experts sit in a `ModuleList`; the router is a `Linear` |
| Forward | `L7.9` | the Llama model registers `Embedding`, `ModuleList`, and `Linear` in HF's order, so `state_dict` keys match HF's |
| Forward | `L8.5` | `QuantLinear` replaces `Linear` layers found through `named_modules()`; `export_q4` walks `named_parameters()` |

## Going further

| Piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Module` registration | `torch.nn.Module` | buffers (saved, not trained), forward and backward hooks, `_load_from_state_dict` per module for versioned formats | `torch/nn/modules/module.py` |
| `state_dict` names | HF `transformers` key mapping | renames between checkpoint versions, sharded `model-0000x-of-0000y.safetensors` with an index file | `modeling_utils.py`, `_load_state_dict_into_model` |
| `Linear`, `LayerNorm` | fused kernels in Apex and Liger | LayerNorm and RMSNorm forward and backward in one pass, fused with the residual add | `liger_kernel/ops/layer_norm.py` |
| explicit `rng` for init | JAX and Flax `PRNGKey` splitting | every random draw takes a key, so initialization is a pure function of the seed | `flax/linen/module.py` (`make_rng`) |
