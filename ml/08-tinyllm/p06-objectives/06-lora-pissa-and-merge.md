<!-- ol:module L6.6 -->
# LoRA with PiSSA init and merge

## Overview

| | |
|---|---|
| **Module** | `L6.6` · build · Python · Pass 5 · 3 to 4 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/obj/lora.py`: `LoRALinear`, `inject_lora`, `merge_lora`, `lora_state_dict`, `load_lora_state_dict`, `trainable_fraction`; and your own oracle tests in `python/tests/l6-6-lora/` |
| **Contract** | [`course/contracts/py/tinyllm/obj/lora.pyi`](../../../course/contracts/py/tinyllm/obj/lora.pyi) |
| **Tests** | `course/tests/L6.6/test_lora.py` (what they check: section 4); the PiSSA oracle is LAPACK's SVD; your tests are graded by mutation, threshold 0.80 with every required pitfall fault killed |
| **Needs** | `L0.1` Tensor · `L0.2` ops · `L0.4` `Linear`, `Dropout`, `Module` · [`M03.5` SVD and `low_rank`](../../../math/03-linear-algebra/05-svd-low-rank-and-least-squares.md) (or `--ref-deps`) |
| **Used by** | `L6.5` `lora_classifier` (the `finetune classify --lora r=8` step of MS-L6) · later: `L12.1` SFT, `sq.multi-lora` |
| **Milestone** | `MS-L6` (`trainable_frac < 0.05` for the LoRA fine-tune) |
| **Optional depth** | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021), sections 4 and 7; Meng, Wang, and Zhang, "PiSSA: Principal Singular Values and Singular Vectors Adaptation" (2024), sections 3 and 4; Aghajanyan et al., "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" (2020) |

## Key Takeaways

- A LoRA layer computes $W x + b + s\,B A x$ with the base $W$ frozen; at default init $B = 0$, so the adapted model equals the base **bit for bit** (`test_default_init_equals_base_bitwise`).
- The scale is $s = \alpha / r$, not $\alpha$: doubling the rank at a fixed $\alpha$ halves each direction's step (`test_scaling_is_alpha_over_r`).
- Only $A$ and $B$ get gradients; the frozen weight stays in the checkpoint but out of the optimizer (`test_only_adapter_parameters_get_gradients`, `test_inject_freezes_everything_else`).
- PiSSA starts the adapter at the top-$r$ singular part of $W$ and freezes the residual, so step 0 still computes the base function (`test_pissa_reproduces_the_base`, `test_pissa_adapter_is_the_top_singular_part`).
- Merging folds $s B A$ into $W$ and puts the plain `Linear` back: same outputs, same keys, no serving cost (`test_merge_equals_unmerged`, `test_merge_restores_the_base_keys`).

## How to work this chapter

```bash
ol start L6.6              # stubs lora.py; prints your test path and rung (R5)
ol tests L6.6              # the course tests
# write your oracle tests in python/tests/l6-6-lora/ (section 4 lists what to cover), then:
ol check L6.6              # course tests and the mutation grade of your tests
ol mutate L6.6             # the full grade, cached by your test files' hash
ol diff  L6.6              # after passing: your code against the reference
```

---

## 1. Why now

You can now train a BERT encoder (`L6.2`) and want it to classify sentences (`L6.5`). Fine-tuning every weight works on a laptop for a 50k-parameter model, but the habit does not scale: the AdamW state (`M10.3`) is two extra copies of every weight, and each task you fine-tune stores a full copy of the model. At the 135M parameters of SmolLM2, which your engine serves later, that is half a gigabyte of optimizer state and of checkpoint per task. The milestone of this part asks for the opposite: a classifier fine-tune where less than 5% of the parameters train (`trainable_frac < 0.05`). LoRA is the tool, and it is built from a piece you already own: the low-rank approximation of `M03.5`.

## 2. Principles

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
| $U, S, V^\top$ | the SVD of $W$ (`M03.5`): $W = U\,\mathrm{diag}(S)\,V^\top$ | as `svd` returns |
| $W_r$ | the best rank-$r$ approximation of $W$ (Eckart-Young) | `[out, in]` |

### 2.1 The update

A fine-tune changes $W$ into $W + \Delta W$. Hu et al. observed that the useful $\Delta W$ of a fine-tune has low "intrinsic" rank, so they write it as a product $\Delta W = s B A$ with $r$ much smaller than either dimension. The layer computes

$$y = W x + b + s\,B (A\,x).$$

The order of the product matters for cost: $A x$ is $r$ numbers, so the adapter costs $r(\text{in} + \text{out})$ multiply-adds per row instead of $\text{in} \cdot \text{out}$. The parameters are $A$ and $B$, $r(\text{in} + \text{out})$ numbers per layer. For a $768 \times 768$ projection at $r = 8$ that is 12 288 trainable numbers instead of 589 824: about 2%.

The scale $s = \alpha / r$ is a convention with a reason. With $A$ initialized at a fixed size, the update $B A x$ sums $r$ terms, so its size grows with $r$; dividing by $r$ keeps the effective learning rate of the adapter roughly independent of the rank, and you tune $\alpha$ once.

LoRA's dropout (`dropout`) acts on the adapter's input only: $y = W x + b + s B A\,\mathrm{dropout}(x)$. The frozen path must see exactly the base model's input.

### 2.2 Initialization: default and PiSSA

**Default.** $A$ is drawn like any `Linear` weight (`L0.4`) and $B = 0$. Then $\Delta W = 0$ exactly: the adapted model starts as the pretrained one, and $s B A x$ is a vector of exact zeros, so $y$ is bit for bit the base output. The first gradient step moves $B$ only ($\partial y / \partial A$ contains $B$, which is zero), then both.

**PiSSA.** Meng et al. start the adapter at the most important part of $W$ instead. With the SVD $W = U\,\mathrm{diag}(S)\,V^\top$ and singular values in decreasing order, Eckart-Young (`M03.5`) says the best rank-$r$ approximation is $W_r = U_{:, :r}\,\mathrm{diag}(S_{:r})\,V_{:, :r}^\top$. `low_rank(W, r)` returns it as balanced factors $P = U_{:, :r}\sqrt{S_{:r}}$ and $Q = \sqrt{S_{:r}}\,V_{:, :r}^\top$, with $P Q = W_r$. PiSSA sets

$$B = P / \sqrt{s}, \qquad A = Q / \sqrt{s}, \qquad W \leftarrow W - W_r,$$

so $s B A = W_r$ and the frozen weight keeps only the residual. At step 0 the layer computes $(W - W_r)x + W_r x = W x$: the base function again (to float32 rounding, since it is now a sum of two products). What changed is which directions train: the adapter now owns the top singular directions of $W$, the ones a fine-tune most often needs to move, and PiSSA converges faster than zero-initialized LoRA in the paper's experiments. Dividing by $\sqrt{s}$ is what makes $s B A$ come out as $W_r$ and not $s W_r$.

### 2.3 Freezing

Freezing is `requires_grad = False` on a **registered** parameter. `L0.4` registers a Tensor when it is assigned with `requires_grad` set, and it stays registered afterwards, so a frozen weight is still in `named_parameters` and `state_dict` (the checkpoint keeps it) but produces no gradient: an op (`L0.1`'s `from_op`) records a vjp only when some input requires grad. `inject_lora` freezes the **whole** model first, then wraps the targeted Linears; after it, the adapters are the only trainable parameters, and an optimizer built from `p for p in model.parameters() if p.requires_grad` holds $A$ and $B$ only. `trainable_fraction` counts elements: trainable over all registered, frozen included.

A `LoRALinear` reuses the base Linear's own Tensors under the base's names (`weight`, `bias`) and adds `lora_A.weight` and `lora_B.weight`. That keeps every base checkpoint key valid on the adapted model.

### 2.4 Merging

To serve, fold the update into the weight: $W' = W + s B A$, computed in float64 and stored in float32, then put the original `Linear` object back in its parent. The merged model has the base's keys in the base's order, no adapter left, and the same outputs as the adapted model up to float32 rounding (the two compute $W' x$ and $W x + s B (A x)$, which round differently). A merged model never runs the adapter again: forgetting to remove it counts the update twice.

Adapters are shared without the base, under PEFT's names in `adapter_model.safetensors`: `base_model.model.<module name>.lora_A.weight` and `.lora_B.weight`. `lora_state_dict` writes those keys; `load_lora_state_dict` reads them back and refuses a mismatched file before copying anything.

## 3. Worked example by hand

**Forward and merge.** $W = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $b = (0.5, -0.5)$, $r = 1$, $\alpha = 2$, so $s = 2$. Take $A = (1, -1)$ and $B = (0.5, 1)^\top$, and $x = (2, 1)$.

| step | value |
|---|---|
| frozen path $W x + b$ | $(1 \cdot 2 + 2 \cdot 1, 3 \cdot 2 + 4 \cdot 1) + b = (4, 10) + (0.5, -0.5) = (4.5, 9.5)$ |
| $A x$ | $2 - 1 = 1$ |
| $B (A x)$ | $(0.5, 1)$ |
| times $s = 2$ | $(1, 2)$ |
| $y$ | $(5.5, 11.5)$ |
| $\Delta W = s B A$ | $2 \begin{pmatrix} 0.5 & -0.5 \\ 1 & -1 \end{pmatrix} = \begin{pmatrix} 1 & -1 \\ 2 & -2 \end{pmatrix}$ |
| merged $W'$ | $\begin{pmatrix} 2 & 1 \\ 5 & 2 \end{pmatrix}$, and $W' x + b = (5, 12) + b = (5.5, 11.5)$ |

**PiSSA split.** $W = \mathrm{diag}(3, 1)$, $r = 1$, $\alpha = 4$, so $s = 4$. The SVD is $U = V = I$, $S = (3, 1)$, so $W_1 = \mathrm{diag}(3, 0)$ and $P = (\sqrt 3, 0)^\top$, $Q = (\sqrt 3, 0)$. Divide each by $\sqrt s = 2$: $B = (\sqrt 3 / 2, 0)^\top$, $A = (\sqrt 3 / 2, 0)$, and $s B A = 4 \cdot \tfrac{3}{4} \mathrm{diag}(1, 0) = \mathrm{diag}(3, 0)$. The frozen residual is $\mathrm{diag}(0, 1)$. (The pair $(-B, -A)$ is the same split.)

These are `test_hand_example_forward_and_merge` and `test_hand_example_pissa_split`.

## 4. The interface

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_forward_and_merge` | unit | section 3: $y = (5.5, 11.5)$ before and after merging, $W' = [[2, 1], [5, 2]]$ | you and the test agree on the formula and on $s$ |
| `test_hand_example_pissa_split` | unit | section 3: $B$, $A$, and the residual $\mathrm{diag}(0, 1)$ | the PiSSA split, sign aside |
| `test_default_init_equals_base_bitwise` | differential | adapted outputs equal the base's bit for bit; $B = 0$, $A \ne 0$ | fine-tuning starts from the pretrained function |
| `test_scaling_is_alpha_over_r` | unit | $s = \alpha / r$ and $\Delta W = s B A$ for four $(r, \alpha)$ | one $\alpha$ works across ranks |
| `test_only_adapter_parameters_get_gradients` | property | flags and gradients: frozen weight and bias get none | the optimizer state is the adapter's only |
| `test_inject_freezes_everything_else` | property | embeddings, norms, untargeted Linears frozen; a target matching nothing raises | LoRA trains adapters, nothing else |
| `test_adapter_gradcheck` | gradcheck | $\partial / \partial A$ and $\partial / \partial B$ against the frozen central differences | the gradients `L6.5` trains with |
| `test_pissa_reproduces_the_base` | differential | PiSSA outputs equal the base's within float32 | step 0 is still the pretrained model |
| `test_pissa_adapter_is_the_top_singular_part` | golden | $s B A = W_r$ from LAPACK; $B$'s columns are $\pm\sqrt{S_j / s}\,u_j$ | PiSSA's point: train the principal directions |
| `test_merge_equals_unmerged` | differential | merged plain Linears give the adapted outputs within 1e-5, default and PiSSA | serving pays nothing for the adapter |
| `test_merge_restores_the_base_keys` | property | same keys, same order as before injection; with $B = 0$ the same values | every loader reads a merged checkpoint |
| `test_peft_key_names` | golden | `base_model.model.q_proj.lora_A.weight` and friends, shapes $[r, \text{in}]$ and $[\text{out}, r]$ | adapters move between tools |
| `test_adapter_roundtrip` | property | an adapter file restores the adapter; wrong keys fail before copying | resuming and sharing adapters |
| `test_dropout_only_on_the_adapter_path` | property | with $B = 0$ dropout changes nothing; in eval mode it is the identity | the frozen path sees the true input |
| `test_trainable_fraction` | unit | 16 trainable elements out of the total | the MS-L6 `trainable_frac` bar |
| `test_validation` | boundary | rank bounds, $\alpha \le 0$, unknown init, a non-Linear base | caller bugs fail loudly |

### Your graded tests (rung R5)

Rung R5 asks for **oracles**: expected values from an independent computation. For LoRA the oracles are numpy itself: recompute $x W^\top + b + s\,x A^\top B^\top$, take LAPACK's `np.linalg.svd` for PiSSA, and check gradients with your own finite differences in float64. Cover the formula with $r > 1$, the bitwise start, which parameters train and get gradients, PiSSA against LAPACK, merging (outputs, removed adapters, unchanged keys), PEFT names, dropout on the adapter path only, and the rank bound. Import only the contract (`tinyllm.obj.lora` and the `L0.4` layers). `ol check L6.6` requires a mutation score of at least 0.80 with every required pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. scaling by $\alpha$ instead of $\alpha / r$ | a learning rate that works at $r = 4$ diverges at $r = 16$ | `test_scaling_is_alpha_over_r`, `test_adapter_gradcheck` (mutant `s01`) |
| 2. a random $B$ at default init | the "fine-tune" starts from a damaged model; step-0 loss above the base's | `test_default_init_equals_base_bitwise` (mutant `s02`) |
| 3. the wrapped weight left trainable, or only the wrapped layers frozen | the whole model trains: optimizer memory and checkpoints as large as a full fine-tune | `test_only_adapter_parameters_get_gradients` (mutant `s03`), `test_inject_freezes_everything_else` (mutant `s09`) |
| 4. PiSSA without the residual, or without dividing by $\sqrt s$ | step 0 computes $W + W_r$ or $W - W_r + s W_r$: not the base model | `test_pissa_reproduces_the_base`, `test_pissa_adapter_is_the_top_singular_part` (mutants `s04`, `s05`) |
| 5. merging without $s$, or leaving the adapter in place | merged outputs differ, or the update is counted twice | `test_merge_equals_unmerged` (mutants `s06`, `s07`) |
| 6. adapter keys without PEFT's prefix; dropout on the frozen path | adapters that no other tool loads; a noisy base model in training | `test_peft_key_names` (mutant `s08`), `test_dropout_only_on_the_adapter_path` (mutant `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | freezing is `requires_grad`; `from_op` records no vjp for frozen inputs |
| Back | `L0.2` | the adapter path is `matmul` and `transpose` of the op library |
| Back | `L0.4` | `LoRALinear` wraps a `Linear`, reuses its Tensors, and builds `lora_A` and `lora_B` as Linears |
| Back | `M03.5` | `low_rank` gives PiSSA's balanced factors |
| Forward | `L6.5` | `lora_classifier` adapts a classifier's attention queries and values and keeps the new head trainable |
| Forward | `L12.1` | supervised fine-tuning of the chat model trains LoRA adapters (optional, Pass 10) |
| Forward | `sq.multi-lora` | the engine serves many adapters over one base, unmerged |

If you skip this module, `ol check L6.5` stops with `needs L6.6`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `LoRALinear`, `inject_lora` | Hugging Face PEFT `LoraConfig`, `get_peft_model` | adapters for Conv1D and embeddings, several named adapters per layer, `modules_to_save`, rank patterns per module | `peft/tuners/lora/layer.py`, `peft/tuners/lora/model.py` |
| `init="pissa"` | PEFT `init_lora_weights="pissa"` and `"pissa_niter_4"` | a fast randomized SVD for large weights; also OLoRA, LoftQ, EVA initializations | `peft/tuners/lora/layer.py` (`pissa_init`) |
| `merge_lora` | `merge_and_unload()`, `add_weighted_adapter` | merging several adapters with weights (TIES, DARE) | `peft/tuners/lora/model.py` |
| unmerged adapters at serving time | vLLM and S-LoRA multi-LoRA serving | thousands of adapters over one base, batched with custom kernels (Punica SGMV) | vLLM `vllm/lora/`; Sheng et al., "S-LoRA" (2023) |
| low-rank training | QLoRA, DoRA | adapters over a 4-bit base; a magnitude and direction split of $W$ | Dettmers et al. 2023; Liu et al. 2024 |
