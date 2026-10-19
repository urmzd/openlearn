# LoRA: low-rank adaptation with PiSSA initialization and merging

A [BERT encoder](02-bert-masked-lm.md) can now be trained, and the [fine-tuning heads](05-fine-tuning-heads-and-the-linear-head-export.md) make it classify sentences. Fine-tuning every weight works on a laptop for a 50k-parameter model, but the habit does not scale. The [AdamW](../../mathematics-for-computing/10-optimization/03-adam-and-adamw.md) state is two extra copies of every weight, and each fine-tuned task stores a full copy of the model. At the 135M parameters of SmolLM2, which the engine serves later, that is half a gigabyte of optimizer state and of checkpoint per task. The end of this chapter needs the opposite: a classifier fine-tune in which less than 5% of the parameters train.

LoRA is the tool for that, and it is built from a piece already in place: the low-rank approximation of [SVD, Eckart-Young low rank, and least squares](../../mathematics-for-computing/03-linear-algebra/05-svd-low-rank-and-least-squares.md) in *Mathematics for Computing*. This section builds the LoRA layer, its two initializations (zero and PiSSA), freezing, merging for serving, and PEFT-compatible adapter files.

Beyond the SVD, you need only the Part 0 foundations: [the `Tensor`](../p00-foundations/01-tensor-and-broadcasting-backward.md), [the op library](../p00-foundations/02-op-library-and-gradcheck.md), and [`Linear`, `Dropout`, and `Module`](../p00-foundations/04-module-system-and-layers.md). The reference implementation is [`python/tinyllm/obj/lora.py`](../../../../course/solutions/python/tinyllm/obj/lora.py) (`LoRALinear`, `inject_lora`, `merge_lora`, `lora_state_dict`, `load_lora_state_dict`, `trainable_fraction`), with the contract in [`course/contracts/py/tinyllm/obj/lora.pyi`](../../../../course/contracts/py/tinyllm/obj/lora.pyi).

> **Learning objectives**
> - Compute a LoRA forward pass and its merged weight by hand, including the scale $s = \alpha / r$.
> - Explain why zero-initialized $B$ makes the adapted model equal the base bit for bit.
> - Derive PiSSA's split of $W$ into a trainable top-$r$ part and a frozen residual.
> - Freeze a model so that only adapters (and an explicit head) train, while every weight stays in the checkpoint.
> - Merge adapters for serving and save them under PEFT's key names.

## The low-rank update

Here is the notation for this section.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $W$ | the base Linear's weight, frozen | `float32[out, in]` |
| $b$ | the base Linear's bias, frozen | `float32[out]` |
| $x$ | one input row | `float32[in]` |
| $r$ | the adapter rank, $1 \le r \le \min(\text{in}, \text{out})$ | `int` |
| $A$ | `lora_A.weight`, the down projection | `float32[r, in]` |
| $B$ | `lora_B.weight`, the up projection | `float32[out, r]` |
| $\alpha$ | `alpha`, the adapter's scale knob | `float` $> 0$ |
| $s = \alpha / r$ | `scaling` | `float` |
| $\Delta W = s B A$ | the update the adapter adds, rank at most $r$ | `float32[out, in]` |
| $U, S, V^\top$ | the SVD of $W$: $W = U\,\mathrm{diag}(S)\,V^\top$ | as `svd` returns |
| $W_r$ | the best rank-$r$ approximation of $W$ (Eckart-Young) | `[out, in]` |

A fine-tune changes $W$ into $W + \Delta W$. Hu et al. observed that the useful $\Delta W$ of a fine-tune has low "intrinsic" rank, so they write it as a product $\Delta W = s B A$ with $r$ much smaller than either dimension. The layer computes

$$y = W x + b + s\,B (A\,x).$$

The order of the product matters for cost. $A x$ is $r$ numbers, so the adapter costs $r(\text{in} + \text{out})$ multiply-adds per row instead of $\text{in} \cdot \text{out}$. The parameters are $A$ and $B$, $r(\text{in} + \text{out})$ numbers per layer. For a $768 \times 768$ projection at $r = 8$ that is 12 288 trainable numbers instead of 589 824, about 2%.

The scale $s = \alpha / r$ is a convention with a reason. With $A$ initialized at a fixed size, the update $B A x$ sums $r$ terms, so its size grows with $r$. Dividing by $r$ keeps the adapter's effective learning rate roughly independent of the rank, so $\alpha$ is tuned once.

LoRA's dropout acts on the adapter's input only: $y = W x + b + s B A\,\mathrm{dropout}(x)$. The frozen path must see exactly the base model's input.

## Two initializations: zero and PiSSA

The update's form is fixed; the next question is where $A$ and $B$ start.

With the **default** initialization, $A$ is drawn like any `Linear` weight and $B = 0$. Then $\Delta W = 0$ exactly: the adapted model starts as the pretrained one, and $s B A x$ is a vector of exact zeros, so $y$ is bit for bit the base output. The first gradient step moves $B$ only ($\partial y / \partial A$ contains $B$, which is zero), and after that both.

**PiSSA**, from Meng et al., starts the adapter at the most important part of $W$ instead. With the SVD $W = U\,\mathrm{diag}(S)\,V^\top$ and singular values in decreasing order, Eckart-Young says the best rank-$r$ approximation is $W_r = U_{:, :r}\,\mathrm{diag}(S_{:r})\,V_{:, :r}^\top$. `low_rank(W, r)` returns it as balanced factors $P = U_{:, :r}\sqrt{S_{:r}}$ and $Q = \sqrt{S_{:r}}\,V_{:, :r}^\top$, with $P Q = W_r$. PiSSA sets

$$B = P / \sqrt{s}, \qquad A = Q / \sqrt{s}, \qquad W \leftarrow W - W_r,$$

so $s B A = W_r$ and the frozen weight keeps only the residual. At step 0 the layer computes $(W - W_r)x + W_r x = W x$, the base function again (to float32 rounding, since it is now a sum of two products). What changed is which directions train. The adapter now owns the top singular directions of $W$, the ones a fine-tune most often needs to move, and PiSSA converges faster than zero-initialized LoRA in the paper's experiments. Dividing by $\sqrt{s}$ is what makes $s B A$ come out as $W_r$ and not $s W_r$.

## Freezing

An adapter is only cheap if nothing else trains. Freezing is `requires_grad = False` on a **registered** parameter. The module system registers a Tensor when it is assigned with `requires_grad` set, and it stays registered afterwards. A frozen weight is therefore still in `named_parameters` and `state_dict`, so the checkpoint keeps it, but it produces no gradient: an op (`from_op` in the `Tensor`) records a vjp only when some input requires grad.

`inject_lora` freezes the **whole** model first, then wraps the targeted Linears. After it, the adapters are the only trainable parameters, and an optimizer built from `p for p in model.parameters() if p.requires_grad` holds $A$ and $B$ only. `trainable_fraction` counts elements: trainable over all registered, frozen included.

A `LoRALinear` reuses the base Linear's own Tensors under the base's names (`weight`, `bias`) and adds `lora_A.weight` and `lora_B.weight`. That keeps every base checkpoint key valid on the adapted model.

## Merging

For serving, fold the update into the weight: $W' = W + s B A$, computed in float64 and stored in float32, then put the original `Linear` object back in its parent. The merged model has the base's keys in the base's order, no adapter left, and the same outputs as the adapted model up to float32 rounding (the two compute $W' x$ and $W x + s B (A x)$, which round differently). A merged model never runs the adapter again; forgetting to remove it counts the update twice.

Adapters are shared without the base, under PEFT's names in `adapter_model.safetensors`: `base_model.model.<module name>.lora_A.weight` and `.lora_B.weight`. `lora_state_dict` writes those keys; `load_lora_state_dict` reads them back and refuses a mismatched file before copying anything.

## Working it by hand

Start with a **forward pass and merge**. Let $W = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $b = (0.5, -0.5)$, $r = 1$, $\alpha = 2$, so $s = 2$. Take $A = (1, -1)$, $B = (0.5, 1)^\top$, and $x = (2, 1)$. The table follows the computation.

| step | value |
|---|---|
| frozen path $W x + b$ | $(1 \cdot 2 + 2 \cdot 1, 3 \cdot 2 + 4 \cdot 1) + b = (4, 10) + (0.5, -0.5) = (4.5, 9.5)$ |
| $A x$ | $2 - 1 = 1$ |
| $B (A x)$ | $(0.5, 1)$ |
| times $s = 2$ | $(1, 2)$ |
| $y$ | $(5.5, 11.5)$ |
| $\Delta W = s B A$ | $2 \begin{pmatrix} 0.5 & -0.5 \\ 1 & -1 \end{pmatrix} = \begin{pmatrix} 1 & -1 \\ 2 & -2 \end{pmatrix}$ |
| merged $W'$ | $\begin{pmatrix} 2 & 1 \\ 5 & 2 \end{pmatrix}$, and $W' x + b = (5, 12) + b = (5.5, 11.5)$ |

The merged layer gives the same output as the adapted one, as it must.

Now a **PiSSA split**. Let $W = \mathrm{diag}(3, 1)$, $r = 1$, $\alpha = 4$, so $s = 4$. The SVD is $U = V = I$, $S = (3, 1)$, so $W_1 = \mathrm{diag}(3, 0)$, $P = (\sqrt 3, 0)^\top$, and $Q = (\sqrt 3, 0)$. Divide each by $\sqrt s = 2$: $B = (\sqrt 3 / 2, 0)^\top$, $A = (\sqrt 3 / 2, 0)$, and $s B A = 4 \cdot \tfrac{3}{4} \mathrm{diag}(1, 0) = \mathrm{diag}(3, 0)$. The frozen residual is $\mathrm{diag}(0, 1)$. (The pair $(-B, -A)$ is the same split.)

## Designing the interface

```python
class LoRALinear(Module):
    def __init__(self, base: Linear, r: int, alpha: float, dropout: float = 0.0,
                 init: Literal["default", "pissa"] = "default", rng=None) -> None: ...
    def delta_weight(self) -> NDArray: ...           # s * B @ A, float32 [out, in]
    def forward(self, x: Tensor) -> Tensor: ...
def inject_lora(model, target: Callable[[str, Module], bool], r: int, alpha: float,
                dropout: float = 0.0, init="default", rng=None) -> list[str]: ...
def merge_lora(model) -> list[str]: ...
def lora_state_dict(model) -> dict[str, NDArray]: ...    # PEFT key names
def load_lora_state_dict(model, sd) -> None: ...
def trainable_fraction(model) -> float: ...
```

`inject_lora` takes a predicate over module names and modules rather than a list of names, so callers can say "every `q_proj` and `v_proj` under the backbone" once and have it apply to any architecture.

## Building it step by step

The reasoning runs as follows.

1. The goal is a fine-tune whose trainable state is small. If the useful update $\Delta W$ has low rank, write it as $B A$ with $A \in \mathbb{R}^{r \times \text{in}}$ and $B \in \mathbb{R}^{\text{out} \times r}$: $r(\text{in} + \text{out})$ numbers instead of $\text{in} \cdot \text{out}$.
2. The base must not change, so $W$ and $b$ are frozen and the layer adds $s B (A x)$. Computing $A x$ first keeps the cost at $r(\text{in} + \text{out})$ per row and never materializes $B A$.
3. Fine-tuning should start from the pretrained function, so at default init $\Delta W = 0$. Setting $B = 0$ (not $A$) is the right zero: then $A$ is random and $\partial L / \partial B = s\,(\partial L/\partial y)(A x)^\top$ is non-zero, so training can start. With $A = 0$ and $B$ random the same holds the other way, but the convention, and PEFT, use $B = 0$.
4. The size of $B A x$ grows with $r$, so $s = \alpha / r$ keeps the effective step roughly rank-independent.
5. PiSSA instead puts the best rank-$r$ part $W_r$ into the adapter and freezes $W - W_r$. Eckart-Young says $W_r$ comes from the top $r$ singular triples, which `low_rank` returns as balanced factors $P, Q$; dividing each by $\sqrt s$ makes $s B A = W_r$. In the worked example $s = 4$, $\sqrt s = 2$, and $s B A = \mathrm{diag}(3, 0)$.
6. For checkpoint compatibility, the wrapped layer must still expose `weight` and `bias` under the old names, so a base checkpoint loads into an adapted model. The wrapper therefore re-registers the base's own Tensors, frozen.
7. Only adapters train and everything else is frozen: `inject_lora` freezes the whole model first, then wraps. A caller that adds a new head unfreezes it explicitly.
8. Serving should not pay for the adapter, so merge $W' = W + s B A$ and put the plain `Linear` back, so that the keys and module tree are the base's again.

### Registering a frozen Tensor

The module system registers a Tensor as a parameter only when it is assigned with `requires_grad` set. The helper uses that, then freezes:

```python
def _register_frozen(owner, name, t):
    """Register t under name, then freeze it: L0.4 registers a parameter when
    it is assigned with requires_grad True, and keeps it registered after."""
    if t is None:
        setattr(owner, name, None)
        return
    t.requires_grad = True
    setattr(owner, name, t)      # now in named_parameters and the state_dict
    t.requires_grad = False      # ... but no gradient, so out of the optimizer
    t.grad = None
```

### The layer

```python
class LoRALinear(Module):
    def __init__(self, base, r, alpha, dropout=0.0, init="default", rng=None):
        super().__init__()
        # ... base must be a Linear; 1 <= r <= min(in, out); alpha > 0; init known ...
        out_f, in_f = base.weight.shape
        self.in_f, self.out_f, self.r = in_f, out_f, int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        # The base's own Tensors, registered here under the base's names and
        # frozen: the checkpoint keys stay "<name>.weight" and "<name>.bias".
        object.__setattr__(self, "_base", base)    # kept for merging, NOT registered as a child
        _register_frozen(self, "weight", base.weight)
        _register_frozen(self, "bias", base.bias)
        self.lora_A = Linear(in_f, self.r, bias=False, rng=rng)      # A: a normal Linear init
        self.lora_B = Linear(self.r, out_f, bias=False, rng=rng)
        self.lora_B.weight.data[...] = 0.0  # B = 0: the adapted model starts as the base
        self.lora_dropout = Dropout(dropout, rng=rng)
        if init == "pissa":
            # W = U S V^T. low_rank gives balanced factors P = U_r sqrt(S_r),
            # Q = sqrt(S_r) V_r^T with P Q = W_r. Divide each by sqrt(s) so
            # s B A = W_r exactly, and keep only the residual W - W_r frozen.
            W = self.weight.data.astype(np.float64)
            P, Q = low_rank(W, self.r)
            root = math.sqrt(self.scaling)
            self.lora_B.weight.data[...] = (P / root).astype(np.float32)
            self.lora_A.weight.data[...] = (Q / root).astype(np.float32)
            self.weight.data[...] = (W - P @ Q).astype(np.float32)
```

`object.__setattr__` bypasses `Module`'s registration, so `_base` is not a child module and its Tensors are not listed twice. The SVD runs in float64, and the residual is written in place into the same Tensor the base Linear holds.

```python
    def forward(self, x):
        y = F.matmul(x, F.transpose(self.weight, 0, 1))
        if self.bias is not None:
            y = y + self.bias
        # Dropout on the adapter's input only: the frozen path sees x as it is.
        h = self.lora_B(self.lora_A(self.lora_dropout(x)))   # B (A x): r-wide in the middle
        return y + h * self.scaling
```

With $B = 0$, `h` is exact zeros, and $y + 0 \cdot s$ is $y$ bit for bit. The bitwise-equal start is a consequence of floating point adding exact zeros, not of a tolerance.

### Injection

```python
def inject_lora(model, target, r, alpha, dropout=0.0, init="default", rng=None):
    # A LoRALinear's own lora_A and lora_B are never adapted again.
    inside = [n + "." for n, _ in _lora_modules(model)]
    chosen = [
        n for n, m in list(model.named_modules())
        if n and isinstance(m, Linear) and not any(n.startswith(p) for p in inside) and target(n, m)
    ]
    if not chosen:
        raise ValueError("inject_lora: target matched no Linear")
    # Freeze everything first; the adapters built below are the only
    # trainable parameters (a caller may unfreeze a head afterwards).
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None
    for n in chosen:
        parent, attr = _resolve(model, n)     # walk the dotted name to the parent module
        setattr(parent, attr, LoRALinear(getattr(parent, attr), r, alpha, dropout, init, rng))
    return chosen
```

The list of targets is computed before any replacement, so wrapping does not change what is being iterated. Freezing happens before wrapping, so the new `lora_A` and `lora_B` (created trainable) are the only parameters with `requires_grad`.

### Merging

```python
def merge_lora(model):
    merged = []
    for n, m in _lora_modules(model):
        base = m._base
        W = m.weight.data.astype(np.float64)       # for PiSSA, this is the residual
        base.weight.data[...] = (W + m.delta_weight().astype(np.float64)).astype(np.float32)
        parent, attr = _resolve(model, n)
        setattr(parent, attr, base)  # the plain Linear again: no adapter left
        merged.append(n)
    return merged
```

`delta_weight` computes $s B A$ in float64 and returns float32. Because `m.weight` and `base.weight` are the same Tensor, the merged weight is $W + s B A$ both in the default case, where $W$ is the original, and for PiSSA, where it is $W - W_r + W_r$ up to rounding. Putting `base` back restores the original module and therefore the original key order.

### Adapter files and the trainable fraction

```python
PEFT_PREFIX = "base_model.model."

def lora_state_dict(model):
    out = {}
    for n, m in _lora_modules(model):
        out[f"{PEFT_PREFIX}{n}.lora_A.weight"] = m.lora_A.weight.data.copy()
        out[f"{PEFT_PREFIX}{n}.lora_B.weight"] = m.lora_B.weight.data.copy()
    return out


def load_lora_state_dict(model, sd):
    own = lora_state_dict(model)
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    if missing or unexpected:       # checked before anything is copied
        raise KeyError(f"adapter mismatch: missing {missing}, unexpected {unexpected}")
    # ... copy each array into its adapter, checking shapes ...


def trainable_fraction(model):
    total = trainable = 0
    for _, p in model.named_parameters():      # frozen parameters are still listed
        total += p.data.size
        trainable += p.data.size if p.requires_grad else 0
    # ... a model with no parameters raises ...
    return trainable / total
```

The full solution is in [`lora.py`](../../../../course/solutions/python/tinyllm/obj/lora.py).

### Why this design

Per adapted layer, LoRA stores $r(\text{in} + \text{out})$ parameters, gradients, and two AdamW moments, against $\text{in} \cdot \text{out}$ for a full fine-tune, and the forward pass adds $r(\text{in} + \text{out})$ multiply-adds per row. PiSSA adds one SVD per layer at injection, $O(\text{in} \cdot \text{out} \cdot \min(\text{in}, \text{out}))$, paid once.

The solution **wraps rather than subclasses**: replacing the `Linear` with a wrapper that reuses its Tensors keeps every base key valid and lets `merge_lora` restore the exact original object. It **freezes the whole model** rather than only the wrapped layers, because otherwise embeddings, norms, and untargeted Linears would still train, and the "small" fine-tune would hold a full optimizer state. It **merges in float64**, so the merged weight differs from the unmerged computation only by the final rounding. The tradeoff accepted is that a rank-$r$ update cannot express every fine-tune; for this chapter's classifier at $r = 8$ it is enough, and the full-rank alternative costs about 50 times more trainable state.

## Testing it

A correct solution has to pass these checks.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand-worked forward and merge | unit | $y = (5.5, 11.5)$ before and after merging, $W' = [[2, 1], [5, 2]]$ | everyone agrees on the formula and on $s$ |
| hand-worked PiSSA split | unit | $B$, $A$, and the residual $\mathrm{diag}(0, 1)$ | the PiSSA split, sign aside |
| default init equals the base bit for bit | differential | adapted outputs equal the base's bit for bit; $B = 0$, $A \ne 0$ | fine-tuning starts from the pretrained function |
| scaling is $\alpha / r$ | unit | $s = \alpha / r$ and $\Delta W = s B A$ for four $(r, \alpha)$ | one $\alpha$ works across ranks |
| only adapter parameters get gradients | property | flags and gradients: frozen weight and bias get none | the optimizer state is the adapter's only |
| injection freezes everything else | property | embeddings, norms, untargeted Linears frozen; a target matching nothing raises | LoRA trains adapters, nothing else |
| adapter gradients | gradcheck | $\partial / \partial A$ and $\partial / \partial B$ against central differences | the gradients the classifier trains with |
| PiSSA reproduces the base | differential | PiSSA outputs equal the base's within float32 | step 0 is still the pretrained model |
| PiSSA's adapter is the top singular part | golden | $s B A = W_r$ from LAPACK; $B$'s columns are $\pm\sqrt{S_j / s}\,u_j$ | PiSSA's point: train the principal directions |
| merged equals unmerged | differential | merged plain Linears give the adapted outputs within 1e-5, default and PiSSA | serving pays nothing for the adapter |
| merging restores the base keys | property | same keys, same order as before injection; with $B = 0$ the same values | every loader reads a merged checkpoint |
| PEFT key names | golden | `base_model.model.q_proj.lora_A.weight` and friends, shapes $[r, \text{in}]$ and $[\text{out}, r]$ | adapters move between tools |
| adapter roundtrip | property | an adapter file restores the adapter; wrong keys fail before copying | resuming and sharing adapters |
| dropout only on the adapter path | property | with $B = 0$ dropout changes nothing; in eval mode it is the identity | the frozen path sees the true input |
| trainable fraction | unit | 16 trainable elements out of the total | the chapter's "under 5% trainable" target |
| bad arguments | boundary | rank bounds, $\alpha \le 0$, unknown init, a non-Linear base | caller bugs fail loudly |

The model tests in [`test_lora.py`](../../../../course/solutions/learner-tests/L6.6/python/tests/l6-6-lora/test_lora.py) use numpy itself as the oracle: $x W^\top + b + s\,x A^\top B^\top$, LAPACK's `np.linalg.svd` for PiSSA, and float64 finite differences for the gradients.

## What can go wrong

Most LoRA bugs either start the fine-tune from a damaged model or quietly train far more than the adapters.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| scaling by $\alpha$ instead of $\alpha / r$ | a learning rate that works at $r = 4$ diverges at $r = 16$ | `self.scaling = self.alpha / self.r` |
| a random $B$ at default init | the "fine-tune" starts from a damaged model; step-0 loss above the base's | `self.lora_B.weight.data[...] = 0.0` |
| the wrapped weight left trainable, or only the wrapped layers frozen | the whole model trains: optimizer memory and checkpoints as large as a full fine-tune | `_register_frozen` for `weight` and `bias`; `inject_lora` freezes every parameter first |
| PiSSA without the residual, or without dividing by $\sqrt s$ | step 0 computes $W + W_r$ or $W - W_r + s W_r$: not the base model | the frozen weight becomes `W - P @ Q`; both factors are divided by `sqrt(scaling)` |
| merging without $s$, or leaving the adapter in place | merged outputs differ, or the update is counted twice | `delta_weight` includes `scaling`; `merge_lora` puts the plain `Linear` back |
| adapter keys without PEFT's prefix; dropout on the frozen path | adapters that no other tool loads; a noisy base model in training | keys start with `base_model.model.`; dropout wraps only the input of `lora_A` |

> **Tip:** `trainable_fraction` is the cheapest LoRA test there is. If it reports more than a few percent after `inject_lora`, something other than the adapters is still trainable.

## Where this leads

The [fine-tuning heads](05-fine-tuning-heads-and-the-linear-head-export.md) use `lora_classifier` to adapt a classifier's attention queries and values while keeping the new head trainable. Later, [supervised fine-tuning](../p12-post-training/01-supervised-fine-tuning.md) of the chat model trains LoRA adapters (optional). A natural extension is multi-LoRA serving, in which the engine serves many adapters over one base, unmerged.

The production tools build on the same core. Hugging Face PEFT's `LoraConfig` and `get_peft_model` (`peft/tuners/lora/layer.py`, `peft/tuners/lora/model.py`) add adapters for Conv1D and embeddings, several named adapters per layer, `modules_to_save`, and per-module rank patterns. PEFT's `init_lora_weights="pissa"` and `"pissa_niter_4"` use a fast randomized SVD for large weights, alongside the OLoRA, LoftQ, and EVA initializations. Its `merge_and_unload()` and `add_weighted_adapter` merge several adapters with weights (TIES, DARE). vLLM (`vllm/lora/`) and S-LoRA serve thousands of adapters over one base, batched with custom kernels (Punica SGMV). And QLoRA and DoRA put adapters over a 4-bit base or split $W$ into magnitude and direction.

Further reading:

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021), sections 4 and 7.
- Meng, Wang, and Zhang, "PiSSA: Principal Singular Values and Singular Vectors Adaptation" (2024), sections 3 and 4.
- Aghajanyan et al., "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" (2020).
- Sheng et al., "S-LoRA" (2023).
- Dettmers et al. (2023), QLoRA; Liu et al. (2024), DoRA.

## Key terms

- **LoRA**: a frozen Linear plus a trainable low-rank update $s B A$.
- **Rank $r$**: the inner dimension of the adapter, bounding the rank of $\Delta W$.
- **Scaling $s = \alpha / r$**: the multiplier on the adapter output that keeps the effective step rank-independent.
- **PiSSA**: initializing the adapter at the top-$r$ singular part of $W$ and freezing the residual.
- **Merging**: folding $s B A$ into $W$ and restoring the plain Linear for serving.

## Summary

- A LoRA layer computes $W x + b + s\,B A x$ with the base $W$ frozen; at default init $B = 0$, so the adapted model equals the base **bit for bit**.
- The scale is $s = \alpha / r$, not $\alpha$: doubling the rank at a fixed $\alpha$ halves each direction's step.
- Only $A$ and $B$ get gradients; the frozen weight stays in the checkpoint but out of the optimizer.
- PiSSA starts the adapter at the top-$r$ singular part of $W$ and freezes the residual, so step 0 still computes the base function.
- Merging folds $s B A$ into $W$ and puts the plain `Linear` back: same outputs, same keys, no serving cost.

## Review questions

1. Why is $B$ set to zero and not $A$?
   <details><summary>Answer</summary>Either makes $\Delta W = 0$. With $B = 0$ and $A$ random, the first gradient reaches $B$ (it is proportional to $A x$, which is non-zero), so training starts; $A$ starts moving once $B$ is non-zero. This is the LoRA paper's and PEFT's convention, so adapters are compatible.</details>
2. Why does PiSSA divide both factors by $\sqrt s$?
   <details><summary>Answer</summary>The layer multiplies $B A$ by $s$. `low_rank` returns $P Q = W_r$; with $B = P/\sqrt s$ and $A = Q/\sqrt s$, $s B A = W_r$ exactly. Without the division the adapter would contribute $s W_r$, and step 0 would not be the base function.</details>
3. A frozen weight is still in `state_dict`. Why does that not make it trainable?
   <details><summary>Answer</summary>Registration and training are separate. The weight is registered (so it is saved and loaded) but `requires_grad` is False, so no op records a vjp for it and an optimizer built from parameters that require grad never sees it.</details>
4. What goes wrong if `merge_lora` folds $s B A$ into $W$ but leaves the `LoRALinear` in place?
   <details><summary>Answer</summary>The forward would compute $(W + sBA)x + sBAx$: the update counted twice. Swapping the plain `Linear` back removes the adapter path.</details>
5. Why is dropout applied only to the adapter's input?
   <details><summary>Answer</summary>The frozen path must compute exactly what the base model computes. Dropping inputs there would add noise to the pretrained function in training; on the adapter path it regularizes only what is being learned.</details>
