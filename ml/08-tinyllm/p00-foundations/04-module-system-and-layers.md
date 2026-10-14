<!-- ol:module L0.4 -->
# Module system and basic layers

## Overview

| | |
|---|---|
| **Module** | `L0.4` · build · Python · Pass 2 · 4 to 5 h, plus your graded tests (rung R3) |
| **You build** | `python/tinyllm/nn/module.py`: `Module` (registration, `named_parameters`, `state_dict`, `load_state_dict`, `train`/`eval`, `zero_grad`); `python/tinyllm/nn/layers.py`: `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `ReLU`, `Tanh`, `GELU`, `Sequential`, `ModuleList`; and your own tests in `python/tests/l0-4-module/`, written first |
| **Contract** | [`course/contracts/py/tinyllm/nn/module.pyi`](../../../course/contracts/py/tinyllm/nn/module.pyi) · [`course/contracts/py/tinyllm/nn/layers.pyi`](../../../course/contracts/py/tinyllm/nn/layers.pyi) |
| **Tests** | `course/tests/L0.4/` (what they check: section 4), golden values from torch 2.14 in `course/fixtures/L0.4/layers_torch.npz`; your tests are graded by mutation, threshold 0.70 plus one required fault, with a red-then-green journal |
| **Needs** | `L0.1` `Tensor` (parameters) · `L0.2` the ops every forward is written in · `M07.3` `normal_init` · `M06.3` PCG32 init and dropout streams · reading: `craft.03` red then green (or `--ref-deps`) |
| **Used by** | `L0.5` `BigramLogits` is a `Module`, the loop trains `Linear` stacks · `L0.6` a checkpoint is a module's `state_dict` · later: `L11.1`, `L2.2`, `L3.2`, `L3.3`, `L3.6`, `L4.1`, `L4.2`, `L4.3`, `L5.3`, `L5.4`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5`, `L6.6`, `L7.1`, `L7.2`, `L7.5`, `L7.6`, `L7.8`, `L7.9`, `L8.5`, `L13.1`, `L13.3`, `L13.5`, `L14.2`  · `L14.3` Whisper decoding |
| **Milestone** | `MS-L0` (the digits MLP is two `Linear` layers) |
| **Optional depth** | the PyTorch `torch.nn.Module` documentation and source (`torch/nn/modules/module.py`); Ba, Kiros, Hinton, "Layer Normalization" (2016) |

## Key Takeaways

- A module finds its parameters and children by attribute assignment, in assignment order, and the dotted names this produces are the safetensors keys of every checkpoint in the course, identical to PyTorch's (`test_state_dict_names_match_torch`).
- `load_state_dict` matches by name, checks every key and shape before copying anything, and writes into the existing arrays, so an optimizer built earlier keeps training the loaded weights (`test_load_state_dict_strict`, `test_load_is_in_place`).
- `Linear` stores one row per output, $W \in \mathbb{R}^{\text{out} \times \text{in}}$, and computes $xW^\top + b$, PyTorch's layout, which is what lets HF weights load by name (`test_hand_example_linear`, `test_matches_torch`).
- LayerNorm uses the population variance with $\epsilon$ inside the square root (`test_layernorm_eps_inside_the_square_root`).
- `eval()` and `train()` reach every descendant, and a weight tied under two names is one parameter (`test_train_eval_reaches_every_child`, `test_parameters_order_and_tying`).

## How to work this chapter

```bash
ol start L0.4              # stubs module.py and layers.py; prints your test path and rung (R3)
ol tests L0.4              # the course tests
# write ONE test in python/tests/l0-4-module/, then:
ol tdd red L0.4            # must FAIL against your current code: records the red
# make it pass, then:
ol tdd green L0.4          # must PASS with the same test files: records the green
# repeat for each test; then:
ol check L0.4              # course tests, the red-then-green journal, the mutation grade
ol mutate L0.4             # the full grade, cached by your test files' hash
```

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

These are `test_hand_example_linear` and `test_layernorm_eps_inside_the_square_root`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_linear` | unit | section 3: $y = [[3.5, 6.5, 12]]$, $\bar{W}$ ones, $\bar{b}$ ones, $\bar{x} = [[9, 12]]$ | you and the test agree on the layout |
| `test_matches_torch` | golden | every layer with torch's weights copied in by name gives torch's outputs and gradients | HF weights load by name (`L7.9`) |
| `test_state_dict_names_match_torch` | golden | the exact keys and order torch produces, through nested containers | the safetensors key contract |
| `test_state_dict_roundtrip` | property | save, rebuild with another seed, load: same function; saved arrays are copies | checkpoints (`L0.6`) |
| `test_load_is_in_place` | unit | the Tensors and arrays an optimizer holds are the ones loaded into | resuming a run (`L0.6`) |
| `test_load_state_dict_strict` | boundary | strict names every missing and unexpected key and copies nothing; wrong shapes fail; non-strict loads the rest | a wrong checkpoint fails loudly |
| `test_parameters_order_and_tying` | unit | own parameters before children's; a tied Tensor once | weight tying (`L7.9`) |
| `test_plain_state_is_not_a_parameter` | boundary | constants and numbers are not parameters; `None` unregisters | masks and buffers in attention (`L5`) |
| `test_forgetting_super_init_is_explained` | boundary | the error names `super().__init__()` | the most common first bug |
| `test_train_eval_reaches_every_child` | unit | `eval()` reaches nested dropout; both return the model | evaluation is deterministic (`L0.5`) |
| `test_zero_grad` | unit | every parameter's `.grad` is cleared | gradients accumulate (`L0.1`) |
| `test_layernorm_normalizes` | property | rows come out with mean 0 and variance 1 | every transformer block (`L5`) |
| `test_layernorm_eps_inside_the_square_root` | boundary | section 3's $\pm 0.30151$ | torch parity on small-variance rows |
| `test_init_statistics_and_seeding` | statistical | $\mathcal{N}(0, 1/d_{\text{in}})$ weights, zero bias, seed-reproducible | stable signal through depth (`M07.3`) |
| `test_sequential_and_modulelist` | unit | order, indexing, `append` | stacks of blocks (`L5`, `L7`) |

### Your graded tests (rung R3)

Rung R3 gives you the interface (above) and one test; you write the rest **before** the code that makes each pass. The given test:

```python
# python/tests/l0-4-module/test_module.py
import numpy as np
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear

def test_hand_example_linear():
    """W = [[1, 2], [3, 4], [5, 6]], b = [0.5, -0.5, 1], x = [[1, 1]]: y = [[3.5, 6.5, 12]], dx = [[9, 12]]."""
    lin = Linear(2, 3)
    lin.load_state_dict({"weight": np.array([[1, 2], [3, 4], [5, 6]]), "bias": np.array([0.5, -0.5, 1.0])})
    x = Tensor([[1.0, 1.0]], requires_grad=True)
    y = lin(x)
    assert np.allclose(y.data, [[3.5, 6.5, 12.0]])
    y.backward(np.ones((1, 3)))
    assert np.allclose(x.grad, [[9.0, 12.0]])
```

Then, one at a time, red then green: state_dict names in registration order with no key for a missing bias; loading by name (a reordered dict gives the same parameters); loading in place; strict mode rejecting missing and unexpected keys without changing anything; a tied parameter listed once; `eval()` turning dropout off in nested modules; LayerNorm against its formula. Import only contract names (`tinyllm.nn.module`, `tinyllm.nn.layers`, `tinyllm.autograd.tensor`). `ol check L0.4` requires, for every test file, a `ol tdd red` record before its last `ol tdd green`, a mutation score of at least 0.70, and the required fault killed (it is the one section 5's last pitfall describes).

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. assigning a parameter before `super().__init__()` | an `AttributeError` about `_params`, far from the cause | `test_forgetting_super_init_is_explained` (mutant `m06`) |
| 2. `train()`/`eval()` setting only the top module's flag | dropout stays on in nested blocks during evaluation | `test_train_eval_reaches_every_child` (mutant `s14`) |
| 3. `load_state_dict` replacing the arrays | a resumed run's optimizer updates arrays the model no longer uses: the loss stops moving | `test_load_is_in_place` (mutant `s09`) |
| 4. loading by position, checking keys after copying, or skipping the shape check | a reordered checkpoint loads into the wrong layers; a wrong one half-loads; a broadcastable array loads silently | `test_state_dict_roundtrip` (mutant `s08`), `test_load_state_dict_strict` (mutants `s10`, `s11`) |
| 5. LayerNorm with the sample variance, or $\epsilon$ outside the root | small disagreements with torch everywhere, large ones on near-constant rows | `test_layernorm_normalizes` (mutant `s03`), `test_layernorm_eps_inside_the_square_root` (mutant `s04`) |
| 6. `Linear` weight as `[in, out]`, or a forward without the transpose | trains fine from scratch, then torch and HF weights load transposed or fail | `test_hand_example_linear`, `test_matches_torch` (mutants `s01`, `s02`) |

## 6. Where it's used next
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L11.1` | Registered call site uses this module. |
| Forward | `L2.2` | Registered call site uses this module. |
| Forward | `L3.2` | Registered call site uses this module. |
| Forward | `L3.3` | Registered call site uses this module. |
| Forward | `L3.6` | Registered call site uses this module. |
| Forward | `L4.1` | Registered call site uses this module. |
| Forward | `L4.2` | Registered call site uses this module. |
| Forward | `L4.3` | Registered call site uses this module. |
| Forward | `L5.3` | Registered call site uses this module. |
| Forward | `L5.4` | Registered call site uses this module. |
| Forward | `L5.5` | Registered call site uses this module. |
| Forward | `L6.1` | Registered call site uses this module. |
| Forward | `L6.2` | Registered call site uses this module. |
| Forward | `L6.3` | Registered call site uses this module. |
| Forward | `L6.5` | Registered call site uses this module. |
| Forward | `L6.6` | Registered call site uses this module. |
| Forward | `L7.1` | Registered call site uses this module. |
| Forward | `L7.2` | Registered call site uses this module. |
| Forward | `L7.5` | Registered call site uses this module. |
| Forward | `L7.6` | Registered call site uses this module. |
| Forward | `L7.8` | Registered call site uses this module. |
| Forward | `L7.9` | Registered call site uses this module. |
| Forward | `L8.5` | Registered call site uses this module. |

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

If you skip this module, `ol check L0.5` stops with `L0.5 needs L0.4`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Module` registration | `torch.nn.Module` | buffers (saved, not trained), forward and backward hooks, `_load_from_state_dict` per module for versioned formats | `torch/nn/modules/module.py` |
| `state_dict` names | HF `transformers` key mapping | renames between checkpoint versions, sharded `model-0000x-of-0000y.safetensors` with an index file | `modeling_utils.py`, `_load_state_dict_into_model` |
| `Linear`, `LayerNorm` | fused kernels in Apex and Liger | LayerNorm and RMSNorm forward and backward in one pass, fused with the residual add | `liger_kernel/ops/layer_norm.py` |
| explicit `rng` for init | JAX and Flax `PRNGKey` splitting | every random draw takes a key, so initialization is a pure function of the seed | `flax/linen/module.py` (`make_rng`) |
