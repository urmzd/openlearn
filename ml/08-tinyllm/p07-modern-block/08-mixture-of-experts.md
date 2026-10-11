<!-- ol:module L7.8 -->
# Mixture of Experts: routing, sorted dispatch, Switch aux loss, aux-free bias

## Overview

| | |
|---|---|
| **Module** | `L7.8` · build · Python · Pass 5 · 3 to 4 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/modern/moe.py`: `MoE`, `topk_ids`, `dispatch`, `combine`, `load_balance_loss`; and your own oracle tests in `python/tests/l7-8-moe/` |
| **Contract** | [`course/contracts/py/tinyllm/modern/moe.pyi`](../../../course/contracts/py/tinyllm/modern/moe.pyi) |
| **Tests** | `course/tests/L7.8/test_moe.py` (what they check: section 4), golden values from transformers 5.19.0 `MixtralSparseMoeBlock` (and `load_balancing_loss_func`), `Qwen3MoeSparseMoeBlock`, and `DeepseekV3MoE` in `course/fixtures/L7.8/moe_hf.npz` (`course/oracle/L7.8/moe_hf.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L7.2` `GatedMLP` (every expert) · `L0.4` `Linear`, `ModuleList`, `Module` · `L0.2` the op library · `L0.1` `Tensor` · `M06.3` `PCG32` (or `--ref-deps`) |
| **Used by** | `L7.9` builds MoE layers when `tl_num_experts > 0` · later: C1's MoE-vs-dense ablation at equal active parameters |
| **Milestone** | `MS-L7` (your decoder loads and matches Hugging Face checkpoints) |
| **Optional depth** | Shazeer et al., "Outrageously Large Neural Networks" (2017); Fedus et al., "Switch Transformers" (2021), sections 2.1 and 2.2; Wang et al., "Auxiliary-Loss-Free Load Balancing" (2024); DeepSeek-V3 report, section 2.1.2 |

## Key Takeaways

- A router picks the top $k$ of $E$ experts per token and weighs their outputs; parameters grow with $E$, compute with $k$ (`test_hand_example`).
- Sorted dispatch gives each expert one contiguous slice of tokens and puts results back with the inverse permutation, and equals the per-token loop exactly (`test_sorted_dispatch_equals_dense_loop`).
- The Switch loss $E \sum_e f_e P_e$ is $k$ at perfect balance; $f$ is a count with no gradient, so the router learns through $P$ (`test_load_balance_loss_bounds_and_gradient`).
- DeepSeek-V3's bias chooses experts but never weighs them, and a sign rule balances the load with no loss term (`test_aux_free_bias`, `test_bias_updates_balance_the_load`).

## How to work this chapter

```bash
ol start L7.8              # stubs moe.py; prints your test path and rung (R5)
ol tests L7.8              # the course tests
# write your oracle tests in python/tests/l7-8-moe/, then:
ol check L7.8              # course tests and the mutation grade of your tests
ol diff  L7.8              # after passing: your code against the reference
```

---

## 1. Why now

Your gated MLP (`L7.2`) holds two thirds of a Llama block's parameters, and every token pays for all of them. Mixtral, Qwen-MoE, and DeepSeek-V3 replace it with many smaller MLPs and a router: each token uses two (or eight of 256), so a model can hold far more knowledge at the same cost per token. C1 asks a concrete question with it: at equal active parameters, does an MoE beat the dense MLP on TinyStories? Answering it needs a router that matches the published ones, a dispatch that does not loop over tokens in Python, and a way to stop every token from crowding onto the same expert.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $N$ | tokens (every leading axis flattened) | `int` |
| $E$, $k$ | experts, experts per token | `int` |
| $z_n \in \mathbb{R}^E$ | router logits of token $n$: $W_g x_n$ | `float32[E]` |
| $p_n = \mathrm{softmax}(z_n)$ | router probabilities | `float32[E]` |
| $\mathcal{T}_n$ | the $k$ experts chosen for token $n$ | ids |
| $w_{n,e}$ | the weight of expert $e$ for token $n$ | `float32` |
| $f_e$ | assignments to expert $e$, divided by $N$ | `float` |
| $P_e$ | mean of $p_{n,e}$ over tokens | `float` |
| $b_e$ | the aux-free correction bias | `float32[E]` |

### 2.1 Routers

$$y_n = \sum_{e \in \mathcal{T}_n} w_{n,e}\, \mathrm{Expert}_e(x_n) \;+\; \mathrm{Shared}(x_n).$$

| `router` | chooses $\mathcal{T}_n$ by | weights $w_{n,e}$ | used by |
|---|---|---|---|
| `softmax_topk` | top $k$ of $p_n$ | $p_{n,e}$, renormalized over $\mathcal{T}_n$ when `norm_topk` | Mixtral (norm), Qwen-MoE (configurable) |
| `topk_softmax` | top $k$ of $z_n$ | softmax over the chosen logits | Switch-style |
| `sigmoid` | top $k$ of $\sigma(z_n) + b$ | $\sigma(z_{n,e})$, renormalized when `norm_topk` | DeepSeek-V3 |

Then every weight is multiplied by `routed_scaling` (DeepSeek-V3 uses 2.5). Ties go to the lowest id, the course's one tie rule. Shared experts (DeepSeek) see every token; $S$ shared experts of width $f$ are one gated MLP of width $S f$.

### 2.2 Sorted dispatch

Flatten the $N k$ assignments as $a = n k + j$. A stable sort by expert id gives `perm`; `x_sorted = x[perm // k]` puts each expert's tokens in one slice, `offsets` (a prefix sum of the counts) bounds the slices, each expert runs once on its slice, and `inv_perm` (the inverse permutation, `inv_perm[perm[i]] = i`) puts each output back at its assignment before the weighted sum over the $k$ slots. This is the shape of every fast MoE kernel: one matrix product per expert instead of one per token.

### 2.3 The load-balancing loss

A router left alone learns to favor a few experts: they get more gradient, improve, and attract more tokens. Switch Transformer's loss, as Hugging Face computes it:

$$L_{aux} = E \sum_{e=1}^{E} f_e P_e .$$

At perfect balance $f_e = k / E$ and $P_e = 1 / E$, so $L_{aux} = k$; concentrating raises it. $f$ comes from an argmax and has no gradient; the router learns through $P$: $\partial L_{aux} / \partial p_{n,e} = E f_e / N$, so the experts that got the most tokens get their probabilities pushed down hardest. `MoE.route` returns `aux_loss_coef * L_aux`.

### 2.4 Gradients through a discrete choice

The choice of $\mathcal{T}_n$ is piecewise constant: no gradient flows through it. The router still learns, through the weights $w_{n,e}$ that multiply the chosen experts' outputs and through $P$ in the loss. Away from ties, central differences agree with the analytic gradient.

### 2.5 Balancing without a loss

An auxiliary loss also pulls the main objective. DeepSeek-V3 adds a bias only to the scores used for choosing, and after each step moves it by a fixed rate $\gamma$:

$$b_e \leftarrow b_e + \gamma\, \mathrm{sign}\big(\overline{\mathrm{load}} - \mathrm{load}_e\big).$$

An overloaded expert's bias falls until it is chosen less. The bias never enters the weights, so the output is still a mixture of the experts' own scores. It is plain state, not a trained parameter: no optimizer touches it and it is not in `state_dict`.

## 3. Worked example by hand

Four experts, top 2, two tokens with logits $z_0 = (\ln 2, \ln 4, 0, 0)$ and $z_1 = (\ln 3, 0, \ln 3, \ln 2)$.

- Softmax: $p_0 = (2, 4, 1, 1)/8 = (1/4, 1/2, 1/8, 1/8)$ and $p_1 = (3, 1, 3, 2)/9$.
- Token 0 takes experts 1 and 0 (largest first), weights $(1/2, 1/4)$ renormalized to $(2/3, 1/3)$. Token 1 has a tie between experts 0 and 2 at $1/3$; both are in the top 2, lowest id first: $(0, 2)$, weights $(1/2, 1/2)$.
- Assignments $a = n k + j$: $(t_0, e_1), (t_0, e_0), (t_1, e_0), (t_1, e_2)$, experts $[1, 0, 0, 2]$. Stable sort: `perm` $= (1, 2, 0, 3)$, tokens `perm // 2` $= (0, 1, 0, 1)$, counts $(2, 1, 1, 0)$, `offsets` $= (0, 2, 3, 4, 4)$, `inv_perm` $= (2, 0, 1, 3)$.
- If the experts output $(10, 20, 30, 40)$ in sorted order, `inv_perm` puts back $(30, 10, 20, 40)$ in assignment order: token 0 gets $\frac{2}{3} 30 + \frac{1}{3} 10 = 70/3$, token 1 gets $\frac{1}{2} 20 + \frac{1}{2} 40 = 30$.
- Loss: $f = (2, 1, 1, 0)/2$, $P = (7/24, 11/36, 11/48, 25/144)$, $L_{aux} = 4\,(1 \cdot \frac{7}{24} + \frac{1}{2} \cdot \frac{11}{36} + \frac{1}{2} \cdot \frac{11}{48}) = 161/72 \approx 2.236$, above $k = 2$.

This is `test_hand_example`.

## 4. The interface

```python
def topk_ids(scores, k) -> NDArray                                   # [N, k], ties to the lowest id
def load_balance_loss(router_probs: Tensor, topk_idx, n_experts) -> Tensor
def dispatch(x: Tensor, topk_idx, n_experts) -> tuple[Tensor, NDArray, NDArray]   # x_sorted, offsets, inv_perm
def combine(y_sorted: Tensor, topk_w: Tensor, inv_perm) -> Tensor
class MoE(Module):
    def __init__(self, d, d_ff_expert, n_experts, top_k, n_shared=0, router="softmax_topk", norm_topk=True,
                 aux_loss_coef=0.01, bias_update_rate=0.0, routed_scaling=1.0, act="silu", rng=None)
    def route(self, x) -> tuple[NDArray, Tensor, Tensor]               # idx, weights, aux
    def forward(self, x) -> Tensor;  aux_loss, expert_load (properties of the last forward)
    def update_bias(self, expert_load) -> None
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3: routing with a tie, dispatch arrays, combine, the loss | you and the test agree on every step |
| `test_moe_golden` | golden | Mixtral, Qwen3-MoE without renormalizing, DeepSeek-V3 with bias, shared experts, scaling | MoE checkpoints and configs in `L7.9` |
| `test_sorted_dispatch_equals_dense_loop` | differential | batched path vs per-token loop for three routers | the speed path is the correct path |
| `test_dispatch_is_a_stable_grouping` | property | contiguous slices, stable order, exact inverse, gradients back to x | kernels that read one slice per expert |
| `test_topk_ties_go_to_lowest_id` | boundary | the tie rule | reproducible routing in Python and C |
| `test_load_balance_loss_bounds_and_gradient` | property | $k$ at balance, larger when concentrated, $E f_e / N$ gradient | the router learns to spread load |
| `test_gradcheck_router_and_experts` | gradcheck | float64 central differences to x and the router | C1 trains the MoE |
| `test_aux_free_bias` | unit | the sign rule by hand; bias chooses, never weighs | DeepSeek-V3's balancing |
| `test_bias_updates_balance_the_load` | property | 60 updates spread a skewed router's 64 tokens | balance without a loss term |
| `test_state_and_idle_experts` | unit | Hub key names; an idle expert; stats not parameters | checkpoints load; training loops read `aux_loss` |
| `test_validation` | boundary | bad top_k, router, rates, expert ids | config bugs fail loudly |

### Your graded tests (rung R5)

Your oracle is the per-token loop in numpy float64, with each router's weights computed from the logits by the table in section 2.1 (sorting with an explicit `(score, id)` key for the tie rule), run for every router; add a hand-sized dispatch with known `offsets` and roundtrip, the tie rule, the loss and its gradient on a two-token case, a central difference for the router weight, the bias rule, and the validation errors. Import only `tinyllm.modern.moe` and `tinyllm.autograd`. `ol check L7.8` requires 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. renormalizing when the config says `norm_topk_prob = false` | Qwen-MoE outputs scaled wrong | `test_moe_golden` (mutant `s01`) |
| 2. weighing by the biased score | the bias leaks into the output; DeepSeek disagrees | `test_moe_golden`, `test_aux_free_bias` (mutant `s02`) |
| 3. scattering back with `perm` instead of its inverse | outputs land on the wrong tokens | `test_hand_example`, `test_dispatch_is_a_stable_grouping` (mutant `s03`) |
| 4. reading tokens in token order, not grouped by expert | each expert runs on another expert's tokens | `test_dispatch_is_a_stable_grouping` (mutant `s04`) |
| 5. counting $f$ from probabilities instead of assignments | the loss no longer measures the routing | `test_hand_example`, `test_load_balance_loss_bounds_and_gradient` (mutant `s05`) |
| a bias rule with the sign flipped | overloaded experts get more tokens | `test_bias_updates_balance_the_load` (mutant `s06`) |
| ties to the highest id | routing differs between implementations | `test_topk_ties_go_to_lowest_id` (mutant `s07`) |
| shared experts skipped | DeepSeek outputs miss a term | `test_moe_golden` (mutant `s08`) |
| `routed_scaling` ignored | DeepSeek outputs 2.5 times too small in the routed part | `test_moe_golden` (mutant `s09`) |
| router weights detached | the router never learns | `test_gradcheck_router_and_experts` (mutant `s10`) |
| offsets that hold each expert's end | slices shifted by one expert | `test_dispatch_is_a_stable_grouping` (mutant `s11`) |
| `topk_softmax` weighing by the full softmax | weights do not sum to 1 | `test_sorted_dispatch_equals_dense_loop` (mutant `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L7.2` | every expert and the shared experts are a `GatedMLP` |
| Back | `L0.4` | the router `Linear`, the experts' `ModuleList` |
| Back | `L0.2` | softmax, sigmoid, gather, reshape, concat with their gradients |
| Back | `L0.1` | `Tensor` indexing for dispatch |
| Back | `M06.3` | the default initialization stream |
| Forward | `L7.9` | `tl_num_experts`, `tl_top_k_experts`, `first_k_dense_replace` build MoE layers |
| Forward | C1 | MoE vs dense at equal active parameters (M05.1's `active` count) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `dispatch`, `combine` | MegaBlocks, vLLM `fused_moe` | grouped GEMMs over sorted tokens, no padding to a capacity | `vllm/model_executor/layers/fused_moe/`, MegaBlocks paper |
| routing | DeepSeek-V3 group-limited routing | top groups first (`n_group`, `topk_group`), then experts inside them | HF `DeepseekV3TopkRouter` |
| capacity | Switch Transformer capacity factor | drop tokens over an expert's capacity | Switch paper, section 2.2 |
| expert parallelism | DeepEP, all-to-all dispatch | experts on different GPUs, tokens moved by all-to-all | DeepSeek DeepEP |
