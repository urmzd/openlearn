# Mixture of Experts

The gated MLP of [Gated MLPs: SwiGLU and GeGLU](02-gated-mlps.md) holds two thirds of a Llama block's parameters, and every token pays for all of them. Mixtral, Qwen-MoE, and DeepSeek-V3 replace it with many smaller MLPs and a router: each token uses two of them (or eight of 256), so a model can hold far more knowledge at the same cost per token.

The [TinyStories capstone](../capstones/01-tinystories.md) asks a concrete question with this layer: at equal active parameters, does a mixture of experts (MoE) beat the dense MLP on TinyStories? Answering it needs three things, and this section builds each: a router that matches the published ones, a dispatch that does not loop over tokens in Python, and a way to stop every token from crowding onto the same expert, either with Switch Transformer's auxiliary loss or with DeepSeek-V3's loss-free bias.

Every expert is a `GatedMLP`. You will also use `Linear`, `ModuleList`, and `Module` from [Module system and basic layers](../06-numerics-and-autograd/05-module-system-and-layers.md), the op library from [The op library and gradcheck_all](../06-numerics-and-autograd/03-op-library-and-gradcheck.md), `Tensor` indexing from [Tensor, broadcasting backward, and no_grad](../06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md), and `PCG32` from [Modular arithmetic, hashing, and PCG32 in Python](../../mathematics/12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics*. The reference solution is [`moe.py`](../../../../course/solutions/python/tinyllm/modern/moe.py), with `MoE` (`route`, `forward`, `update_bias`), `topk_ids`, `dispatch`, `combine`, and `load_balance_loss`, against the contract [`course/contracts/py/tinyllm/modern/moe.pyi`](../../../../course/contracts/py/tinyllm/modern/moe.pyi).

> **Learning objectives**
> - Route tokens with each of the three router types and explain how the choice and the weights are computed separately.
> - Implement sorted dispatch and its inverse permutation, and trace them by hand.
> - Compute the Switch load-balancing loss, explain why it equals $k$ at balance, and explain where its gradient comes from.
> - Implement DeepSeek-V3's aux-free bias rule and explain why the bias chooses experts but never weighs them.

## Notation

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

## Routers

An MoE layer computes, for each token,

$$y_n = \sum_{e \in \mathcal{T}_n} w_{n,e}\, \mathrm{Expert}_e(x_n) \;+\; \mathrm{Shared}(x_n).$$

Parameters grow with $E$, and compute grows with $k$. Published models differ in how they choose $\mathcal{T}_n$ and how they compute the weights; the three routers the solution supports are:

| `router` | chooses $\mathcal{T}_n$ by | weights $w_{n,e}$ | used by |
|---|---|---|---|
| `softmax_topk` | top $k$ of $p_n$ | $p_{n,e}$, renormalized over $\mathcal{T}_n$ when `norm_topk` | Mixtral (norm), Qwen-MoE (configurable) |
| `topk_softmax` | top $k$ of $z_n$ | softmax over the chosen logits | Switch-style |
| `sigmoid` | top $k$ of $\sigma(z_n) + b$ | $\sigma(z_{n,e})$, renormalized when `norm_topk` | DeepSeek-V3 |

Every weight is then multiplied by `routed_scaling` (DeepSeek-V3 uses 2.5). Ties go to the lowest id, the one tie rule used throughout the book. Shared experts (DeepSeek) see every token, and $S$ shared experts of width $f$ are one gated MLP of width $S f$.

## Sorted dispatch

The obvious implementation loops over tokens and calls $k$ experts for each. That is correct but runs $N k$ tiny matmuls. Sorted dispatch rearranges the work so each expert runs once.

Flatten the $N k$ assignments as $a = n k + j$. A stable sort by expert id gives `perm`; `x_sorted = x[perm // k]` puts each expert's tokens in one slice, and `offsets` (a prefix sum of the counts) bounds the slices. Each expert runs once on its slice, and `inv_perm` (the inverse permutation, `inv_perm[perm[i]] = i`) puts each output back at its assignment before the weighted sum over the $k$ slots. This is the shape of every fast MoE kernel: one matrix product per expert instead of one per token.

## The load-balancing loss

A router left alone learns to favor a few experts: they get more gradient, improve, and attract more tokens. Switch Transformer's loss, as Hugging Face computes it, pushes back:

$$L_{aux} = E \sum_{e=1}^{E} f_e P_e .$$

At perfect balance $f_e = k / E$ and $P_e = 1 / E$, so $L_{aux} = k$; concentrating raises it. $f$ comes from an argmax and has no gradient, so the router learns through $P$: $\partial L_{aux} / \partial p_{n,e} = E f_e / N$, which means the experts that got the most tokens get their probabilities pushed down hardest. `MoE.route` returns `aux_loss_coef * L_aux`.

That raises a broader question: how does the router learn at all, if the choice of experts is discrete? The choice of $\mathcal{T}_n$ is piecewise constant, so no gradient flows through it. The router still learns through the weights $w_{n,e}$ that multiply the chosen experts' outputs, and through $P$ in the loss. Away from ties, central differences agree with the analytic gradient.

## Balancing without a loss

An auxiliary loss also pulls on the main objective. DeepSeek-V3 avoids that by adding a bias only to the scores used for choosing, and after each step moving it by a fixed rate $\gamma$:

$$b_e \leftarrow b_e + \gamma\, \mathrm{sign}\big(\overline{\mathrm{load}} - \mathrm{load}_e\big).$$

An overloaded expert's bias falls until it is chosen less. The bias never enters the weights, so the output is still a mixture of the experts' own scores. It is plain state, not a trained parameter: no optimizer touches it and it is not in `state_dict`.

## Working it by hand

Take four experts, top 2, and two tokens with logits $z_0 = (\ln 2, \ln 4, 0, 0)$ and $z_1 = (\ln 3, 0, \ln 3, \ln 2)$.

- The softmax gives $p_0 = (2, 4, 1, 1)/8 = (1/4, 1/2, 1/8, 1/8)$ and $p_1 = (3, 1, 3, 2)/9$.
- Token 0 takes experts 1 and 0 (largest first), with weights $(1/2, 1/4)$ renormalized to $(2/3, 1/3)$. Token 1 has a tie between experts 0 and 2 at $1/3$; both are in the top 2, lowest id first, so it takes $(0, 2)$ with weights $(1/2, 1/2)$.
- The assignments $a = n k + j$ are $(t_0, e_1), (t_0, e_0), (t_1, e_0), (t_1, e_2)$, with experts $[1, 0, 0, 2]$. The stable sort gives `perm` $= (1, 2, 0, 3)$, tokens `perm // 2` $= (0, 1, 0, 1)$, counts $(2, 1, 1, 0)$, `offsets` $= (0, 2, 3, 4, 4)$, and `inv_perm` $= (2, 0, 1, 3)$.
- If the experts output $(10, 20, 30, 40)$ in sorted order, `inv_perm` puts back $(30, 10, 20, 40)$ in assignment order. Token 0 gets $\frac{2}{3} 30 + \frac{1}{3} 10 = 70/3$ and token 1 gets $\frac{1}{2} 20 + \frac{1}{2} 40 = 30$.
- For the loss, $f = (2, 1, 1, 0)/2$ and $P = (7/24, 11/36, 11/48, 25/144)$, so $L_{aux} = 4\,(1 \cdot \frac{7}{24} + \frac{1}{2} \cdot \frac{11}{36} + \frac{1}{2} \cdot \frac{11}{48}) = 161/72 \approx 2.236$, above $k = 2$.

## The interface

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

The free functions are the reusable pieces: tie-safe top-k, the loss, and the two halves of dispatch. `MoE` composes them; after each `forward` it exposes `aux_loss` and `expert_load` for the training loop, which passes the load to `update_bias` when aux-free balancing is on.

## Building it step by step

Routing is a discrete choice followed by continuous weights, so split it that way. Choose ids from plain numpy scores (no gradient can flow through a choice anyway), then **gather** the weights from the autograd tensor, so the router learns through them. Ties must break the same way in every implementation; a stable sort of `-scores` keeps equal scores in id order, giving "lowest id wins" with no extra code.

Each expert should run once on a contiguous batch. Flatten the $N k$ assignments, stable-sort them by expert id (`perm`), and gather the token rows `perm // k`. A prefix sum of per-expert counts gives slice boundaries, and the inverse permutation puts outputs back. Gathering rows by index is differentiable, so gradients flow back to `x` through `dispatch` and `combine` without any custom backward.

The Switch loss needs a count $f$ (no gradient) and a mean probability $P$ (gradient). Computing $f$ with `np.bincount` and $P$ with `F.mean` encodes exactly that split. The DeepSeek bias must change the choice and nothing else, so it is added only to the numpy scores fed to `topk_ids`, and the weights are gathered from the unbiased sigmoid. It is updated by a rule, so it is a numpy array, not a `Tensor` parameter. Finally, shared experts see every token, so $S$ of width $f$ are one `GatedMLP` of width $S f$, added after the combine.

### Top-k with the tie rule

```python
def topk_ids(scores, k):
    s = np.asarray(scores)
    if not 1 <= k <= s.shape[-1]:
        raise ValueError(...)
    # A stable sort of -s keeps equal scores in id order: ties go to the lowest id.
    return np.argsort(-s, axis=-1, kind="stable")[..., :k].astype(np.int64)
```

On token 1 of the hand example, $p_1 = (3, 1, 3, 2)/9$; sorting $-p_1$ stably gives ids $(0, 2, 3, 1)$, and the first two are $(0, 2)$.

### The loss

```python
def load_balance_loss(router_probs, topk_idx, n_experts):
    idx = _ids(topk_idx, n_experts)          # integer [N, k], ids in range
    # ... router_probs must be [N, E]
    N = idx.shape[0]
    # f_e counts assignments: a constant for autograd. P_e carries the gradient.
    f = np.bincount(idx.ravel(), minlength=n_experts).astype(router_probs.dtype) / N
    P = F.mean(router_probs, axis=0)
    return F.sum(P * f) * float(n_experts)
```

$f$ is divided by $N$, not $N k$, as Hugging Face does; that is why the balanced value is $k$, not 1. The gradient with respect to $p_{n,e}$ is $E f_e / N$: `F.mean` contributes $1/N$ and the product contributes $f_e$.

### Dispatch and combine

```python
def dispatch(x, topk_idx, n_experts):
    idx = _ids(topk_idx, n_experts)
    # ... x must be [N, d]
    k = idx.shape[1]
    perm = np.argsort(idx.ravel(), kind="stable")  # assignment n*k + j, grouped by expert
    counts = np.bincount(idx.ravel(), minlength=n_experts)
    # offsets[e] .. offsets[e+1] is expert e's slice; E + 1 entries, starting at 0
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.size)          # sorted slot of each assignment
    # Assignment a belongs to token a // k.
    return x[perm // k], offsets, inv_perm.astype(np.int64)

def combine(y_sorted, topk_w, inv_perm):
    N, k = topk_w.shape
    y = y_sorted[np.asarray(inv_perm)]  # back to assignment order n*k + j
    y = F.reshape(y, (N, k, y_sorted.shape[-1]))
    return F.sum(y * F.reshape(topk_w, (N, k, 1)), axis=1)   # weighted sum over the k slots
```

Run the hand example through the code. The experts per assignment are `[1, 0, 0, 2]`; stable `argsort` gives `perm = (1, 2, 0, 3)`, `perm // 2 = (0, 1, 0, 1)`, counts `(2, 1, 1, 0)`, and `offsets = (0, 2, 3, 4, 4)`. `inv_perm[perm] = arange` writes `inv_perm = (2, 0, 1, 3)`. Sorted outputs $(10, 20, 30, 40)$ indexed by `inv_perm` give $(30, 10, 20, 40)$, reshaped to token rows $(30, 10)$ and $(20, 40)$, weighted to $70/3$ and $30$.

Stability matters twice: within an expert's slice tokens stay in order, so a kernel reading the slice sees a predictable layout, and the result is deterministic.

### Routing

```python
    def route(self, x):
        # ... x must be [N, d]
        z = self.gate(x)                               # router logits, a Tensor
        k = self.top_k
        if self.router == "sigmoid":
            s = F.sigmoid(z)
            idx = topk_ids(s.data + self.e_score_correction_bias, k)  # the bias chooses ...
            w = F.gather(s, idx, axis=-1)  # ... the unbiased score weighs
            if self.norm_topk:
                w = w / (F.sum(w, axis=-1, keepdims=True) + 1e-20)    # HF's epsilon
            probs = s / F.sum(s, axis=-1, keepdims=True)              # for the loss only
        elif self.router == "softmax_topk":
            probs = F.softmax(z, axis=-1)
            idx = topk_ids(probs.data, k)
            w = F.gather(probs, idx, axis=-1)
            if self.norm_topk:
                w = w / F.sum(w, axis=-1, keepdims=True)
        else:  # topk_softmax
            probs = F.softmax(z, axis=-1)
            idx = topk_ids(z.data, k)
            w = F.softmax(F.gather(z, idx, axis=-1), axis=-1)         # softmax over the chosen k only
        if self.routed_scaling != 1.0:
            w = w * self.routed_scaling
        aux = load_balance_loss(probs, idx, self.n_experts) * self.aux_loss_coef
        return idx, w, aux
```

Every choice reads `.data` (numpy) and every weight is built from the tensor `z`, which is the split between choice and weight in code. For the sigmoid router, the loss needs probabilities that sum to 1, so it normalizes the sigmoid scores; this never affects the output weights.

### The forward

```python
    def forward(self, x):
        lead, d = x.shape[:-1], x.shape[-1]
        xf = F.reshape(x, (-1, d))                     # every leading axis becomes N
        idx, w, aux = self.route(xf)
        xs, offsets, inv_perm = dispatch(xf, idx, self.n_experts)
        parts = []
        for e in range(self.n_experts):
            a, b = int(offsets[e]), int(offsets[e + 1])
            if b > a:  # an expert with no token does no work
                parts.append(self.experts[e](xs[a:b]))
        y = combine(F.concat(parts, axis=0), w, inv_perm)   # parts concatenated are in sorted order
        if self.shared_experts is not None:
            y = y + self.shared_experts(xf)            # shared experts see every token
        self._last = {"aux": aux, "load": np.bincount(idx.ravel(), minlength=self.n_experts).astype(np.int64)}
        return F.reshape(y, lead + (d,))
```

The Python loop is over experts ($E$ iterations), never over tokens. The per-step statistics are kept in a plain dict because, as the constructor's comment notes, a `Tensor` that requires grad assigned as an attribute would register as a parameter.

### The aux-free bias rule

```python
    def update_bias(self, expert_load):
        load = np.asarray(expert_load, dtype=np.float64)
        # ... must be [E]
        # Overloaded experts (load above the mean) get a lower bias, so the
        # router picks them less next step; underloaded ones a higher bias.
        step = self.bias_update_rate * np.sign(load.mean() - load)
        self.e_score_correction_bias = (self.e_score_correction_bias + step).astype(np.float32)
```

The name `e_score_correction_bias` and the module names `gate`, `experts`, and `shared_experts` follow the Hub checkpoints, so the Llama-family loader reads them by name. The full solution is in [`moe.py`](../../../../course/solutions/python/tinyllm/modern/moe.py).

### Why this design

Each token runs $k$ experts, so expert FLOPs are $k/E$ of running all of them; routing adds an $O(N d E)$ gate and an $O(N k \log(N k))$ sort.

Compare the alternatives. A per-token loop gives the same numbers but runs $N k$ small matmuls in Python; sorting gives $E$ batched matmuls, the shape that MegaBlocks and vLLM's `fused_moe` turn into grouped GEMMs. A dense mask, running every expert on every token and masking, is simple but costs $E/k$ times the FLOPs, which removes the point of an MoE.

The solution also has no capacity limit. Switch Transformer drops tokens beyond an expert's capacity to keep shapes static; the solution keeps every token (slices can be any length, even zero), which matches the Hugging Face reference outputs exactly, whereas dropping would make the output depend on batch composition. And keeping `e_score_correction_bias` out of the parameters ensures no optimizer or weight decay touches it, so the sign rule is the only thing that moves it.

## What can go wrong

The hand example (routing with a tie, the dispatch arrays, combine, and the loss) must come out exactly. Hugging Face goldens cover Mixtral, Qwen3-MoE without renormalizing, DeepSeek-V3 with bias, shared experts, and routed scaling, so MoE checkpoints and configs load correctly. Sorted dispatch must equal the dense per-token loop for all three routers, which proves the speed path is the correct path, and dispatch must be a stable grouping: contiguous slices, stable order, an exact inverse, and gradients back to `x`. Ties must go to the lowest id so routing reproduces across Python and C. The loss must be $k$ at balance, larger when concentrated, with gradient $E f_e / N$. A float64 central-difference gradcheck reaches `x` and the router, since the capstone trains the MoE. The bias rule is checked by hand, including that the bias chooses but never weighs, and 60 updates must spread a skewed router's 64 tokens. Hub key names must match, an idle expert must work, and the statistics must not be parameters. Bad `top_k`, routers, rates, and expert ids must raise.

[`test_moe.py`](../../../../course/solutions/learner-tests/L7.8/python/tests/l7-8-moe/test_moe.py) uses the per-token loop in numpy float64 as its oracle, with each router's weights computed from the logits by the router table above (sorting with an explicit `(score, id)` key for the tie rule), run for every router. It adds a hand-sized dispatch with known `offsets` and roundtrip, the tie rule, the loss and its gradient on a two-token case, a central difference for the router weight, the bias rule, and the validation errors. The per-token loop is the definition of an MoE; sorted dispatch is an optimisation, so the loop is exactly the right oracle for it.

> **Warning:** Scattering outputs back with `perm` instead of `inv_perm` produces outputs of the right shape on the wrong tokens. `perm[i]` names the assignment in sorted slot $i$; to read assignment $a$'s output you need its slot, `inv_perm[a]`.

The full list:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| renormalizing when the config says `norm_topk_prob = false` | Qwen-MoE outputs scaled wrong | renormalizes only under `if self.norm_topk` |
| weighing by the biased score | the bias leaks into the output; DeepSeek disagrees | the bias is added to `s.data` for `topk_ids` only; weights gather from `s` |
| scattering back with `perm` instead of its inverse | outputs land on the wrong tokens | `combine` indexes with `inv_perm` |
| reading tokens in token order, not grouped by expert | each expert runs on another expert's tokens | `dispatch` returns `x[perm // k]`, grouped by expert |
| counting $f$ from probabilities instead of assignments | the loss no longer measures the routing | `f = bincount(idx) / N` |
| a bias rule with the sign flipped | overloaded experts get more tokens | `sign(load.mean() - load)`: above-mean load lowers the bias |
| ties to the highest id | routing differs between implementations | stable `argsort` of `-scores` |
| shared experts skipped | DeepSeek outputs miss a term | `y + self.shared_experts(xf)` when `n_shared > 0` |
| `routed_scaling` ignored | DeepSeek outputs 2.5 times too small in the routed part | `w * self.routed_scaling` |
| router weights detached | the router never learns | weights are `F.gather`/`F.softmax` of the gate tensor, not of `.data` |
| offsets that hold each expert's end | slices shifted by one expert | `offsets` starts with 0 and has $E + 1$ entries |
| `topk_softmax` weighing by the full softmax | weights do not sum to 1 | softmax over `F.gather(z, idx)`, the chosen logits only |

## Where this leads

[The Llama-family model](09-llama-family-model.md) builds MoE layers from `tl_num_experts`, `tl_top_k_experts`, and `first_k_dense_replace`, so the decoder loads and matches Hugging Face MoE checkpoints. The [TinyStories capstone](../capstones/01-tinystories.md) compares MoE against dense at equal active parameters, using the `active` count from [Counting: parameters, FLOPs, KV bytes, and memory plans](../../mathematics/11-discrete-math-1/01-counting-params-flops-kv-bytes.md).

### Further reading

- Shazeer et al., "Outrageously Large Neural Networks" (2017).
- Fedus et al., ["Switch Transformers"](https://arxiv.org/abs/2101.03961) (2021), sections 2.1 and 2.2; section 2.2 covers the capacity factor that drops tokens over an expert's capacity.
- Wang et al., "Auxiliary-Loss-Free Load Balancing" (2024).
- DeepSeek-V3 report, section 2.1.2; Hugging Face `DeepseekV3TopkRouter` implements group-limited routing (top groups first with `n_group` and `topk_group`, then experts inside them).
- MegaBlocks and vLLM `fused_moe` (`vllm/model_executor/layers/fused_moe/`): grouped GEMMs over sorted tokens, no padding to a capacity.
- DeepSeek DeepEP: expert parallelism, with experts on different GPUs and tokens moved by all-to-all.

## Key terms

- **Mixture of experts (MoE)**: a layer of many expert MLPs of which a router picks $k$ per token.
- **Router**: the linear gate and selection rule that chooses experts and computes their weights.
- **Sorted dispatch**: grouping assignments by expert with a stable sort so each expert runs once on a contiguous slice.
- **Inverse permutation** (`inv_perm`): the map from each assignment to its sorted slot, used to put outputs back.
- **Load-balancing loss**: Switch Transformer's $E \sum_e f_e P_e$, equal to $k$ at perfect balance.
- **Aux-free bias**: DeepSeek-V3's per-expert correction added to choice scores only, moved by a sign rule.
- **Shared expert**: an expert that sees every token with weight 1.

## Summary

- A router picks the top $k$ of $E$ experts per token and weighs their outputs; parameters grow with $E$, compute with $k$.
- Sorted dispatch gives each expert one contiguous slice of tokens, puts results back with the inverse permutation, and equals the per-token loop exactly.
- The Switch loss $E \sum_e f_e P_e$ is $k$ at perfect balance; $f$ is a count with no gradient, so the router learns through $P$.
- DeepSeek-V3's bias chooses experts but never weighs them, and a sign rule balances the load with no loss term.
- The choice uses numpy scores and the weights are gathered from tensors, so the router learns despite the discrete choice.

## Review questions

1. Why does `inv_perm`, not `perm`, put the outputs back?
   <details><summary>Answer</summary>`perm[i]` says which assignment sits in sorted slot $i$. To read assignment $a$'s output we need its sorted slot, which is `inv_perm[a]`. Indexing with `perm` applies the permutation a second time and lands outputs on the wrong tokens.</details>
2. Why does the router still learn when top-k has no gradient?
   <details><summary>Answer</summary>The weights $w_{n,e}$ are gathered from the router's softmax or sigmoid tensor and multiply the chosen experts' outputs, so the task loss reaches the gate through them. The aux loss adds gradient through $P$.</details>
3. Why is $L_{aux} = k$ at perfect balance, and why does $f$ carry no gradient?
   <details><summary>Answer</summary>Balanced, $f_e = k/E$ (counts over $N$ with $k$ per token) and $P_e = 1/E$, so $E \sum_e (k/E)(1/E) = k$. $f$ is a count of argmax choices, a step function of the logits, so its derivative is zero almost everywhere; the solution makes it an explicit constant.</details>
4. What goes wrong if the DeepSeek bias is added to the weights as well as to the choice?
   <details><summary>Answer</summary>The output becomes a mixture weighted by the bias, which is not a trained quantity and drifts every step. DeepSeek's checkpoints and Hugging Face's outputs use the unbiased sigmoid scores as weights.</details>
5. Why are $S$ shared experts one `GatedMLP` of width $S f$?
   <details><summary>Answer</summary>Every shared expert sees every token with weight 1, so their outputs add. The sum of $S$ gated MLPs is a gated MLP whose hidden layer is the concatenation of theirs, width $S f$; that is also how the checkpoints store them.</details>
6. Why does the solution not impose an expert capacity, as Switch Transformer does?
   <details><summary>Answer</summary>Dropping tokens over capacity makes a token's output depend on what else is in the batch and breaks exact agreement with the Hugging Face references. Sorted dispatch handles slices of any length, including zero, so no token needs to be dropped.</details>
