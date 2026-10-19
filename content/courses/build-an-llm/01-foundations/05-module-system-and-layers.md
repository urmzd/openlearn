# Module system and basic layers

With [the `Tensor`](02-tensor-and-broadcasting-backward.md), [the op library](03-op-library-and-gradcheck.md), and [the fused losses](04-losses-with-fused-backward.md) you can train anything, but a model is still a loose set of Tensors that you have to pass to the optimizer and to the checkpoint writer by hand, in the right order, every time. The digits MLP has four parameter arrays; a Llama layer in [Llama-family model](../08-modern-block/09-llama-family-model.md) has nine, and a 30-layer model 270.

Worse, the checkpoint format ([`safetensors.md`](../../../../course/contracts/formats/safetensors.md)) names tensors by strings such as `model.layers.3.self_attn.q_proj.weight`, and the Rust engine of [Candle model runner](../11-serving/02-model-runner-and-sampler.md) and Hugging Face's loaders find weights by those names. This section gives every model one way to list its parameters, name them exactly as PyTorch does, save and load them, and switch dropout off for evaluation. It then builds the first layers on top: `Linear`, `Embedding`, `LayerNorm`, `Dropout`, three activations, and two containers.

The section builds on the `Tensor` (parameters are Tensors), the op library (every forward is written in `F`), and, from *Mathematics for Computing*, on `normal_init` from [Variance propagation and initialization](../../mathematics-for-computing/08-probability-statistics/04-variance-propagation-and-initialization.md) and PCG32 sub-streams from [Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md). This is a section where tests come first, so [TDD, unit tests, and mutation testing](../../software-craftsmanship/03-testing-mentality/01-tdd-unit-tests-and-mutation-grading.md) in *Software Craftsmanship* is useful background. The solution is [`module.py`](../../../../course/solutions/python/tinyllm/nn/module.py) (`Module` with registration in `__setattr__`, `named_parameters`, `named_modules`, `state_dict`, `load_state_dict`, `train`/`eval`, `zero_grad`) and [`layers.py`](../../../../course/solutions/python/tinyllm/nn/layers.py) (`Linear`, `Embedding`, `LayerNorm`, `Dropout`, `ReLU`, `Tanh`, `GELU`, `Sequential`, `ModuleList`), with model tests in [`test_module.py`](../../../../course/solutions/learner-tests/L0.4/python/tests/l0-4-module/test_module.py). The contracts are [`module.pyi`](../../../../course/contracts/py/tinyllm/nn/module.pyi) and [`layers.pyi`](../../../../course/contracts/py/tinyllm/nn/layers.pyi).

> **Learning objectives**
> - Implement parameter and child registration through `__setattr__`, and explain why the resulting dotted names match PyTorch's.
> - Write a `load_state_dict` that matches by name, validates every key and shape before copying, and loads in place.
> - Compute a `Linear` layer's forward and backward by hand in PyTorch's `[out, in]` layout.
> - Implement LayerNorm with the population variance and $\epsilon$ inside the square root, and show numerically why the placement matters.
> - Make `train()` and `eval()` reach every descendant, and list a tied weight once.

## How a module finds its parameters

The notation for this section:

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
| $\mathcal{N}(0, s^2)$ | normal distribution with standard deviation $s$ | |

### Registration by assignment

`Module.__init__` creates two ordered dictionaries, parameters and children, before anything else. `__setattr__` then sorts every assignment: a `Tensor` with `requires_grad=True` is a parameter, a `Module` is a child, and anything else (a number, a constant Tensor such as a causal mask, `None`) is plain state and unregisters the name. That is why a subclass must call `super().__init__()` first: without the dictionaries there is nowhere to register, and the error should say so. Python dictionaries keep insertion order, so registration order is assignment order.

### Names are the contract

`named_parameters()` yields the module's own parameters first, then each child's with the child's name and a dot as prefix, depth first. A two-layer `Sequential(Linear(2, 3), ReLU(), Linear(3, 1))` yields `0.weight`, `0.bias`, `2.weight`, `2.bias`: exactly `torch.nn.Sequential`'s `state_dict` keys. A Tensor registered under two names (tied input and output embeddings in the Llama model) is listed once, under its first name, by identity; otherwise the optimizer would step it twice.

### Saving and loading

`state_dict()` is `{name: copy of the array}` in `named_parameters` order. It returns copies, so a saved state does not change when training continues. `load_state_dict(sd, strict=True)` first compares keys: with `strict`, any missing or unexpected key is a `KeyError` that names all of them, raised **before** anything is copied, because a half-loaded model is worse than none. A shape mismatch is a `ValueError`, also before copying.

Then each array is written **into** the existing parameter array (`p.data[...] = a`), converted to the parameter's dtype. The optimizers of [The Optimizer protocol](../../mathematics-for-computing/11-optimization/02-optimizer-protocol-and-sgd.md) and [Adam and AdamW](../../mathematics-for-computing/11-optimization/03-adam-and-adamw.md) hold references to those Tensors and arrays; replacing them would leave the optimizer updating arrays the model no longer uses. `strict=False` loads the names that match, which is how you load a pretrained backbone under a new head.

### Train and eval

`training` is a flag on every module, and `Dropout` reads it. `train(mode)` sets it on the module and every descendant and returns the module; `eval()` is `train(False)`. Setting it only on the top module would leave every nested dropout active during evaluation.

## The basic layers

With registration in place, the layers are short. Each one is a composition of `F` ops, so none needs its own backward.

`Linear` uses PyTorch's layout. $W$ has shape `[d_out, d_in]`, so $y = xW^\top + b$, and backward (from the matmul VJP) gives $\bar{W} = \bar{y}^\top x$, $\bar{b} = \sum_{\text{rows}} \bar{y}$, and $\bar{x} = \bar{y}W$. Storing $W$ as `[d_in, d_out]` computes the same function from its own initialization, and then fails the day you load torch weights by name. Initialization follows the variance argument of [Variance propagation and initialization](../../mathematics-for-computing/08-probability-statistics/04-variance-propagation-and-initialization.md): $W \sim \mathcal{N}(0, 1/d_{\text{in}})$ (standard deviation $1/\sqrt{d_{\text{in}}}$), so unit-variance inputs give unit-variance outputs, and $b = 0$. `Embedding(n, d)` is a table `[n, d]` with $\mathcal{N}(0, 1)$ rows and `F.embedding` as its forward.

Initialization is seeded. Each layer takes an optional `rng`, a PCG32, and draws its weights from it once, at construction, in parameter registration order. With no `rng`, layers use the spec's sub-streams of seed 0: `PCG32(0).substream("init")` for weights and `substream("dropout")` for dropout masks, so adding a dropout layer does not change any initial weight.

LayerNorm normalizes each row (the last axis): $\hat{x} = (x - \mu)/\sqrt{\sigma^2 + \epsilon}$, then $y = \gamma \odot \hat{x} + \beta$, with $\sigma^2$ the **population** variance (`F.var` with correction 0, as in torch). $\epsilon$ belongs inside the square root. It guards rows whose variance is tiny, and outside the root it does almost nothing there. The forward is written with `F.mean`, `F.var`, and `**`, so its backward comes from the op library and needs no code here.

The containers are thin. `Sequential(*mods)` registers its children as `"0"`, `"1"`, ... and applies them in order. `ModuleList(mods)` only holds them (no forward); `append` registers the next index and returns the list.

## Working it by hand

Take `Linear(2, 3)` with $W = \begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}$, $b = [0.5, -0.5, 1]$, and one input row $x = [[1, 1]]$.

The forward pass $xW^\top$ takes the dot product of $x$ with each row of $W$: $[1 + 2, 3 + 4, 5 + 6] = [3, 7, 11]$. Adding $b$ gives $y = [[3.5, 6.5, 12]]$.

The backward pass with $\bar{y} = [[1, 1, 1]]$ gives three gradients:

1. $\bar{W} = \bar{y}^\top x = \begin{bmatrix}1\\1\\1\end{bmatrix}[1, 1]$, a $3 \times 2$ matrix of ones: each weight $W_{ij}$ multiplied $x_j = 1$ once.
2. $\bar{b} = [1, 1, 1]$: the bias is added once per output.
3. $\bar{x} = \bar{y}W = [1 + 3 + 5, 2 + 4 + 6] = [[9, 12]]$: each input feeds all three outputs.

Now LayerNorm on a nearly constant row, $x = [0, 0.002]$. Here $\mu = 0.001$ and $\sigma^2 = 10^{-6}$. With $\epsilon = 10^{-5}$ inside the root, $\hat{x} = \pm 0.001/\sqrt{1.1 \times 10^{-5}} = \pm 0.30151$, torch's value. With $\epsilon$ added after the root, $\pm 0.001/(0.001 + 10^{-5}) = \pm 0.990$.

These two hand results are the first cases a correct solution must reproduce.

## Designing the API

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

A correct solution reproduces the hand example ($y = [[3.5, 6.5, 12]]$, $\bar{W}$ and $\bar{b}$ all ones, $\bar{x} = [[9, 12]]$), because that layout is the one everyone agrees on. With torch's weights copied in by name, every layer must give torch's outputs and gradients, which is what lets Hugging Face weights load by name. The `state_dict` keys and their order must be exactly torch's, through nested containers, since they are the safetensors key contract.

Saving and loading are checked as properties. Save, rebuild with another seed, and load, and you must get the same function, with the saved arrays being copies; checkpoints in [Safetensors for every dtype, atomic checkpoints, and the token stream](07-safetensors-checkpoints-and-token-streams.md) depend on this. Loading must be in place, so the Tensors and arrays an optimizer holds are the ones loaded into, which is how a run resumes. Strict loading names every missing and unexpected key and copies nothing, wrong shapes fail, and non-strict loading loads the rest, so a wrong checkpoint fails loudly.

The registration rules are checked too. Own parameters come before children's, and a tied Tensor appears once (weight tying in the Llama model). Constants and numbers are not parameters, and assigning `None` unregisters a name, which matters for masks and buffers in attention. Forgetting `super().__init__()` gives an error that names `super().__init__()`, because it is the most common first bug. `eval()` reaches nested dropout and both `train` and `eval` return the model, so evaluation in [Training loop and the autograd bigram](06-training-loop-and-the-autograd-bigram.md) is deterministic. `zero_grad` clears every parameter's `.grad`, since gradients accumulate.

For the layers: LayerNorm rows come out with mean 0 and variance 1, as every transformer block expects; the hand-worked $\pm 0.30151$ pins $\epsilon$ inside the square root for torch parity on small-variance rows; weights follow $\mathcal{N}(0, 1/d_{\text{in}})$ with zero bias and are seed-reproducible, which keeps the signal stable through depth; and `Sequential` and `ModuleList` get order, indexing, and `append` right, for the stacks of blocks in later chapters.

### Tests first

This is a section where the tests come first: each behaviour is pinned by a test that fails before the code exists and passes after. The model suite, [`test_module.py`](../../../../course/solutions/learner-tests/L0.4/python/tests/l0-4-module/test_module.py), imports only contract names and covers, in order: the hand example above; `state_dict` names in registration order with no key for a missing bias; loading by name (a reordered dict gives the same parameters); loading in place; strict mode rejecting missing and unexpected keys without changing anything; a tied parameter listed once; `eval()` turning dropout off in nested modules; and LayerNorm against its formula. Its first test is the hand example:

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

The one fault a good suite must catch is a `Linear` that stores or applies its weight in the wrong orientation (the last pitfall below). A model trained from scratch cannot see it; only a test with known weights, like this one, can.

## Building it step by step

### The reasoning chain

1. **Every model needs one list of its parameters, in a fixed order, with names.** The optimizer needs the list; the checkpoint needs the names; the order must be reproducible so that two builds of the same model agree. The names must also be PyTorch's, because Hugging Face checkpoints and the Rust engine look weights up by those strings.
2. **The cheapest place to collect that list is attribute assignment.** Model code already writes `self.fc = Linear(...)`. If `__setattr__` sorts each assignment into "parameter", "child module", or "plain state", the model author writes nothing extra, and Python's insertion-ordered dicts make registration order equal to assignment order. This is exactly what torch does, so torch's names fall out for free.
3. **The dictionaries must exist before the first assignment.** So `Module.__init__` creates them with `object.__setattr__` (bypassing its own hook), and the hook checks for them first and raises an error naming `super().__init__()` if they are missing.
4. **Names come from a depth-first walk.** Own parameters first, then each child with `child_name + "."` as prefix. For `Sequential(Linear(2, 3), ReLU(), Linear(3, 1))` this gives `0.weight`, `0.bias`, `2.weight`, `2.bias`; `ReLU` has no parameters, so `1` never appears.
5. **A tied weight must appear once.** Deduplicate by object identity, not by value or name, and keep the first name. Otherwise the optimizer steps the shared Tensor twice per update.
6. **Loading must be all or nothing, and in place.** Validate all keys, then all shapes, and only then copy. Copy into the existing arrays (`data[...] = a`), because the optimizer built before the load holds references to those arrays.
7. **Layers are compositions of library ops.** `Linear` is `matmul` with a transposed weight, `LayerNorm` is `mean`, `var`, a power, and two elementwise ops. Their backward passes then come from autograd and need no code, and the worked example's $\bar{W} = \bar{y}^\top x$ is produced by the matmul VJP.

### Registration

The hook is the whole module system. Note the two `pop` calls: a name can move between categories, and it must live in at most one dict.

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

### Names and the walk

One shared `seen` set across the whole recursion is what makes tying work across sub-modules (an embedding in one child, the output head in another).

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

### Saving and loading

`state_dict` returns copies; `load_state_dict` runs in three phases: key check, shape check, copy.

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

### Linear and the default streams

The weight is `[out_f, in_f]` and the forward transposes it, which is the torch layout described above.

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

### LayerNorm

Four lines, and each one is a decision made above.

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

On the hand-worked row, `var + eps` is $1.1 \times 10^{-5}$ and `xhat` is $\pm 0.30151$.

### Containers

`Sequential` registers through `setattr` with the names `"0"`, `"1"`, ..., so the ordinary hook does all the work and the keys come out as torch's.

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

The full solution is [`module.py`](../../../../course/solutions/python/tinyllm/nn/module.py) and [`layers.py`](../../../../course/solutions/python/tinyllm/nn/layers.py), which also has `ReLU`, `Tanh`, `GELU`, and indexing for both containers.

### Why this design

`named_parameters` is $O(P + M)$ for $P$ parameters and $M$ modules; it builds a list and returns an iterator over it, so callers see a consistent snapshot even if they mutate the model while iterating. `load_state_dict` is $O(P)$ plus one copy of the weights. None of this is on the hot path; it runs once per optimizer build or checkpoint.

The alternatives are each worse in a specific way. An explicit `register_parameter` call per weight works, but it makes every model author repeat each name, and a forgotten call is a weight that silently never trains. Discovering parameters by scanning `vars(self)` on demand loses the ordering guarantee once attributes are reassigned, and it cannot distinguish a tied Tensor from a copy without the same identity check. Storing `Linear` weights as `[in, out]` saves a transpose and breaks every torch and Hugging Face checkpoint.

Two tradeoffs are accepted. There are no buffers (saved but not trained arrays) and no hooks; constant Tensors are plain state and are not saved. And loading converts to the parameter's dtype rather than refusing a dtype mismatch, which is what torch does and what makes float64 test fixtures load into float32 models.

## What can go wrong

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| assigning a parameter before `super().__init__()` | an `AttributeError` about `_params`, far from the cause | `__setattr__` checks for `_params` first and raises an error that names `super().__init__()` |
| `train()`/`eval()` setting only the top module's flag | dropout stays on in nested blocks during evaluation | `train` sets the flag on every module from `named_modules()` |
| `load_state_dict` replacing the arrays | a resumed run's optimizer updates arrays the model no longer uses: the loss stops moving | copies with `data[...] = a`, keeping each array's identity |
| loading by position, checking keys after copying, or skipping the shape check | a reordered checkpoint loads into the wrong layers; a wrong one half-loads; a broadcastable array loads silently | looks up `own[name]`; checks all keys, then all exact shapes, then copies |
| LayerNorm with the sample variance, or $\epsilon$ outside the root | small disagreements with torch everywhere, large ones on near-constant rows | `F.var` with its default correction 0, and `(var + eps) ** -0.5` |
| `Linear` weight as `[in, out]`, or a forward without the transpose | trains fine from scratch, then torch and Hugging Face weights load transposed or fail | weight shape `(out_f, in_f)` and `matmul(x, transpose(W))`; pinned by the hand example |

> **Warning:** A transposed `Linear` is invisible to training from scratch: loss curves look normal. Only a test with known weights, or loading a real checkpoint, reveals it.

## Where this leads

Within this chapter, `BigramLogits` in [Training loop and the autograd bigram](06-training-loop-and-the-autograd-bigram.md) subclasses `Module` and the loop trains `Sequential` MLPs (the digits MLP that closes the chapter is two `Linear` layers), while `save_checkpoint` in [Safetensors for every dtype, atomic checkpoints, and the token stream](07-safetensors-checkpoints-and-token-streams.md) writes `state_dict()` and resuming calls `load_state_dict`.

After that, nearly every model in the book is a tree of these modules. The NPLM of [Bengio's neural probabilistic language model](../03-statistical-lm/02-bengio-nplm.md) is a `Module` of `Embedding` and `Linear` layers. The [LSTM](../04-recurrent/02-lstm.md) and [GRU](../04-recurrent/03-gru.md) cells are `Module`s with `Dropout` and `xavier_uniform` weights, and the [RNN language model](../04-recurrent/06-rnn-language-model-stateful-tbptt.md) wraps a cell between an `Embedding` and a `Linear` read-out. The [encoder-decoder](../05-attention-origins/01-encoder-decoder-teacher-forcing.md) is built from `Embedding` and `Linear`, and both [Bahdanau](../05-attention-origins/02-bahdanau-additive-attention.md) and [Luong](../05-attention-origins/03-luong-attention-input-feeding.md) attention are `Module`s of `Linear` projections.

In the Transformer chapters, [multi-head attention](../06-transformer-2017/03-multi-head-attention.md) projects Q, K, V, and the output with `Linear`; the learned position table of [Positional encodings](../06-transformer-2017/04-positional-encodings.md) is a `Module` initialized with `normal_init`; and the [encoder-decoder Transformer](../06-transformer-2017/05-encoder-decoder-transformer.md), [GPT](../07-objectives/01-gpt-decoder-only.md), and [BERT](../07-objectives/02-bert-masked-lm.md) stack `LayerNorm`, `Linear`, `Embedding`, and `Dropout` in a `ModuleList`. The [ELECTRA](../07-objectives/03-electra-replaced-token-detection.md) heads and the [fine-tuning heads](../07-objectives/05-fine-tuning-heads-and-the-linear-head-export.md) are `Module`s over `Linear`, and `LoRALinear` in [LoRA: low-rank adaptation with PiSSA initialization and merging](../07-objectives/06-lora-pissa-and-merge.md) wraps a frozen `Linear` with two trainable `Linear` factors.

The modern block follows the same pattern: [RMSNorm](../08-modern-block/01-pre-ln-rmsnorm.md) is a `Module`; the gate, up, and down projections of [Gated MLPs](../08-modern-block/02-gated-mlps.md), the projections of [grouped-query attention](../08-modern-block/05-gqa-attention.md), and the down- and up-projections of [multi-head latent attention](../08-modern-block/06-multi-head-latent-attention.md) are `Linear`; the [Mixture of Experts](../08-modern-block/08-mixture-of-experts.md) keeps its experts in a `ModuleList` with a `Linear` router; and the [Llama-family model](../08-modern-block/09-llama-family-model.md) registers `Embedding`, `ModuleList`, and `Linear` in Hugging Face's order, so its `state_dict` keys match. `QuantLinear` in [Quantization](../09-inference/05-quantization.md) replaces `Linear` layers found through `named_modules()`, and `export_q4` walks `named_parameters()`. `train_step_mixed` in [mixed-precision training](../12-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md) steps `model.parameters()`, and `checkpoint_sequential` segments a list of `Module` layers. In vision, `Conv2d`, `Conv1d`, and LeNet-5 in [Conv layers and pooling](../14-vision/01-conv-layers-and-pooling.md) are `Module`s with a `Linear` classifier head, the [vision transformer](../14-vision/03-vision-transformer.md) stacks `LayerNorm`, `Linear`, and `Embedding` in a `ModuleList`, and the text tower and projections of [CLIP and SigLIP](../14-vision/05-contrastive-image-text.md) are `Module`s. In speech, `WhisperEncoder` in [Whisper encoder](../15-speech/02-whisper-encoder.md) stacks `LayerNorm` and `Linear` in a `ModuleList`, and [Whisper decoding](../15-speech/03-whisper-decoder-and-weights.md) uses the same module contract.

### Further reading

- The PyTorch `torch.nn.Module` documentation and source (`torch/nn/modules/module.py`): buffers (saved, not trained), forward and backward hooks, and `_load_from_state_dict` per module for versioned formats.
- Ba, Kiros, Hinton, "Layer Normalization" (2016).
- Hugging Face `transformers` key mapping (`modeling_utils.py`, `_load_state_dict_into_model`): renames between checkpoint versions, and sharded `model-0000x-of-0000y.safetensors` with an index file.
- Fused kernels in Apex and Liger (`liger_kernel/ops/layer_norm.py`): LayerNorm and RMSNorm forward and backward in one pass, fused with the residual add.
- JAX and Flax `PRNGKey` splitting (`flax/linen/module.py`, `make_rng`): every random draw takes a key, so initialization is a pure function of the seed.

## Key terms

- **module**: an object that owns parameters and child modules, discovered through attribute assignment.
- **parameter**: a `Tensor` with `requires_grad=True` registered on a module; the optimizer updates it.
- **state dict**: an ordered mapping from dotted parameter names to arrays, used for checkpoints.
- **strict loading**: refusing a state dict with any missing or unexpected key, before copying anything.
- **weight tying**: one parameter registered under two names, listed once by identity.
- **LayerNorm**: per-row normalization by the population mean and variance with $\epsilon$ inside the square root, then an affine map.

## Summary

- A module finds its parameters and children by attribute assignment, in assignment order, and the dotted names this produces are the safetensors keys of every checkpoint in the book, identical to PyTorch's.
- `load_state_dict` matches by name, checks every key and shape before copying anything, and writes into the existing arrays, so an optimizer built earlier keeps training the loaded weights.
- `Linear` stores one row per output, $W \in \mathbb{R}^{\text{out} \times \text{in}}$, and computes $xW^\top + b$, PyTorch's layout, which is what lets Hugging Face weights load by name.
- LayerNorm uses the population variance with $\epsilon$ inside the square root.
- `eval()` and `train()` reach every descendant, and a weight tied under two names is one parameter.

## Review questions

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

   For a row with $\sigma^2 \gg \epsilon$, $\sqrt{\sigma^2 + \epsilon} \approx \sigma \approx \sigma + \epsilon$ and both forms agree. When $\sigma^2$ is comparable to $\epsilon$, $\sqrt{\epsilon}$ (about $3 \times 10^{-3}$) dominates $\epsilon$ itself, so inside the root it damps the row strongly and outside it barely does: the hand example gets $0.30151$ against $0.990$.

   </details>

6. What `state_dict` keys does `Sequential(Linear(2, 3), ReLU(), Linear(3, 1))` produce, and why is there no key starting with `1.`?

   <details><summary>Answer</summary>

   `0.weight`, `0.bias`, `2.weight`, `2.bias`. The children are registered as `"0"`, `"1"`, `"2"`, but `ReLU` has no parameters, so the walk emits nothing under `1.`.

   </details>
