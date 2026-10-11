<!-- ol:module L6.1 -->
# GPT decoder-only, causal LM loss, GPT-2 weight loading

## Overview

| | |
|---|---|
| **Module** | `L6.1` · build · Python · Pass 5 · 3 to 4 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/obj/gpt.py`: `GPTConfig`, `Block`, `GPT` (`hidden`, `forward`), `clm_loss`, `load_hf_gpt2`, `fit_gpt`, `save_gpt`, `load_gpt`; and your own oracle tests in `python/tests/l6-1-gpt/` |
| **Contract** | [`course/contracts/py/tinyllm/obj/gpt.pyi`](../../../course/contracts/py/tinyllm/obj/gpt.pyi) |
| **Tests** | `course/tests/L6.1/test_gpt.py` (what they check: section 4); the oracle is a random tiny Hugging Face `GPT2LMHeadModel`; the learning test compares with the reference's bar over 5 seeds |
| **Needs** | `L5.2` `causal_mask` · `L5.3` attention · `L5.4` `LearnedPE` · `L0.1` · `L0.2` · `L0.3` `cross_entropy` · `L0.4` layers · `L0.5` `train_step` · `L0.6` safetensors, `TokenStream` · `M10.3` `AdamW` · `M10.4` `cosine_with_warmup` · `M07.3` `normal_init`, `scaled_residual_std` · `M06.3` `PCG32` (or `--ref-deps`) |
| **Used by** | `L6.5` sequence, token, and reward heads · `L6.7` the zoo's `gpt` rows · later `L6.6` LoRA targets, `L8.2`'s first KV cache |
| **Milestone** | `MS-L6` |
| **Optional depth** | Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019), section 2.3; Karpathy, nanoGPT `model.py` |

## Key Takeaways

- A GPT is `L5.5`'s decoder without cross-attention: one causal stack, so every position predicts the next token and one forward pass trains on all of them (`test_logits_are_causal`).
- The causal LM loss shifts by one: logits at position $t$ against the token at $t + 1$; the last position has no target, and a padded target counts for nothing (`test_hand_example_clm_loss`, `test_forward_targets_equals_clm_loss`).
- GPT-2's block is pre-LN with a **tanh-approximated** GELU and a final `ln_f`; the output layer is the token table (`test_block_matches_the_formula`, `test_head_is_tied_to_the_embedding`).
- Hugging Face's GPT-2 stores projections as `Conv1D`, weight `[in, out]`: every weight is transposed on load, and `c_attn` is cut into three column blocks q, k, v (`test_golden_hf_gpt2`).
- The two projections that write into the residual stream start at $0.02/\sqrt{2L}$ (`test_gpt2_init`).

## How to work this chapter

```bash
ol start L6.1              # stubs gpt.py; prints your test path and rung (R5)
ol tests L6.1              # the course tests
ol check L6.1              # course tests and the mutation grade of your tests
ol diff  L6.1              # after passing: your code against the reference
```

---

## 1. Why now

Your encoder-decoder (`L5.5`) maps one sequence to another, but the system you are building serves a language model: a model that continues text. Every token of a corpus is a training target for the prefix before it, and nothing needs to be encoded separately. GPT keeps exactly the half of `L5.5` that does this: the decoder, minus its cross-attention. This chapter builds it to GPT-2's exact layout, so that a checkpoint trained by Hugging Face loads into your code and gives the same logits. That loader is the bridge the rest of the course walks over: `L7.9` loads SmolLM2 the same way, and `L8.2` puts the first KV cache on this model.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $t_0 \dots t_{T-1}$ | the token ids of one window | `int[T]` |
| $V, d, L, H$ | vocabulary, width, layers, heads | `int` |
| $n_{ctx}$ | the longest window (rows of `wpe`) | `int` |
| $W_{te} \in \mathbb{R}^{V \times d}$ | token table `wte` | `float32[V, d]` |
| $W_{pe} \in \mathbb{R}^{n_{ctx} \times d}$ | learned positions `wpe` (`L5.4`) | `float32[n_ctx, d]` |
| $h^{(\ell)}_t$ | the residual stream at layer $\ell$, position $t$ | `[B, T, d]` |
| $z_t \in \mathbb{R}^V$ | logits at position $t$ | `[B, T, V]` |
| $\text{GELU}_{\tanh}(u)$ | $\tfrac12 u (1 + \tanh(\sqrt{2/\pi}(u + 0.044715 u^3)))$ | elementwise |

### 2.1 One stack, one mask

$$P(t_0 \dots t_{T-1}) = \prod_{t=1}^{T-1} P(t_t \mid t_0 \dots t_{t-1}) .$$

With the causal mask (`L5.2`), position $t$ attends to positions $0 \dots t$ only, so its output is a function of the prefix and can be trained to predict $t_{t+1}$. One forward pass computes all $T$ predictions at once; without the mask position $t$ could read $t_{t+1}$ and the loss would teach copying.

### 2.2 The causal LM loss

$$\mathcal{L} = \frac{1}{|K|} \sum_{t \in K} -\log \operatorname{softmax}(z_t)_{t_{t+1}}, \qquad K = \{ t < T - 1 : t_{t+1} \text{ is a real token} \}.$$

In code: logits `[:, :-1]` against ids `[:, 1:]`, with `-100` where the target is padding (`L0.3` ignores it). The shift happens **once**: either `clm_loss` shifts a full window, or the data already gives (inputs, targets) pairs shifted by one, as `L0.6`'s `TokenStream` does, and `forward(x, y)` uses them as they are.

### 2.3 The GPT-2 block

$$h \leftarrow h + \text{Attn}(\text{LN}_1(h)), \qquad h \leftarrow h + W_{proj}\,\text{GELU}_{\tanh}(W_{fc}\,\text{LN}_2(h)),$$

pre-LN (`L5.5`'s `norm="pre"`), then $\text{LN}_f$ after the last block and $z = \text{LN}_f(h) W_{te}^\top$: the output layer is the token table (tied). GPT-2 used the tanh approximation of GELU; the exact form differs by up to about $10^{-3}$ at $|u| \approx 2$, enough to fail a golden comparison.

**Init.** $\mathcal{N}(0, 0.02^2)$ for the tables and every `Linear` weight; biases 0; LayerNorms $(1, 0)$. Each block adds two outputs to the residual stream (`attn.out_proj`, `mlp_proj`), so after $L$ blocks the stream holds $2L$ such terms; their weights start at $0.02/\sqrt{2L}$ (`M07.3`'s `scaled_residual_std`) to keep its variance from growing with depth.

### 2.4 Loading GPT-2

Hugging Face's `GPT2LMHeadModel` names and shapes:

| HF key (after `transformer.`) | Shape | Yours |
|---|---|---|
| `wte.weight`, `wpe.weight` | `[V, d]`, `[n_ctx, d]` | same names |
| `h.i.ln_1.*`, `h.i.ln_2.*`, `ln_f.*` | `[d]` | same names |
| `h.i.attn.c_attn.weight`, `.bias` | `[d, 3d]`, `[3d]` | `attn.q_proj`, `k_proj`, `v_proj`: columns $0..d$, $d..2d$, $2d..3d$, each **transposed** |
| `h.i.attn.c_proj.weight` | `[d, d]` | `attn.out_proj.weight` **transposed** |
| `h.i.mlp.c_fc.weight` | `[d, 4d]` | `mlp_fc.weight` **transposed** |
| `h.i.mlp.c_proj.weight` | `[4d, d]` | `mlp_proj.weight` **transposed** |

`Conv1D` computes $x W + b$ with $W$ `[in, out]`; `Linear` computes $x W^\top + b$ with $W$ `[out, in]`. The square `c_proj` of attention loads either way without a shape error, which is why the golden test compares values, not shapes. Old checkpoints also carry the buffers `attn.bias` and `attn.masked_bias` (a stored causal mask): they are not weights, and the loader ignores them. `lm_head.weight` is the same tensor as `wte.weight`.

## 3. Worked example by hand

$V = 2$, ids $(0, 1, 1)$, logits rows $z_0 = (0, \ln 3)$, $z_1 = (0, 0)$, $z_2 = (5, -5)$.

- Position 0 predicts $t_1 = 1$: $\operatorname{softmax}(0, \ln 3) = (1/4, 3/4)$, loss $-\ln 0.75 = 0.287682$.
- Position 1 predicts $t_2 = 1$: $(1/2, 1/2)$, loss $\ln 2 = 0.693147$.
- Position 2 has no next token: dropped.

$\mathcal{L} = (0.287682 + 0.693147)/2 = 0.490415$. With the third token marked as padding, position 1's target is padding: $\mathcal{L} = 0.287682$, and the gradient on $z_0$ is $\operatorname{softmax}(z_0) - \text{onehot}(1) = (0.25, -0.25)$, zero on the other rows. This is `test_hand_example_clm_loss`.

## 4. The interface

```python
@dataclass
class GPTConfig: vocab: int; n_ctx: int; d_model: int; n_heads: int; n_layers: int; d_ff: int
                 dropout: float = 0.0; ln_eps: float = 1e-5; tie: bool = True
class GPT(Module):
    def hidden(self, ids) -> Tensor                                       # ln_f output [B, T, d]
    def forward(self, ids, targets=None) -> tuple[Tensor, Optional[Tensor]]
def clm_loss(logits: Tensor, ids, mask=None) -> Tensor
def load_hf_gpt2(model: GPT, sd: Mapping[str, Any]) -> None
def fit_gpt(model, stream, steps, lr, warmup=0, weight_decay=0.1, clip=1.0) -> list[float]
def save_gpt(model, dir, tokenizer="bytes") / load_gpt(dir)
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_clm_loss` | unit | section 3: 0.490415, the padded 0.287682, the gradient | you and the test agree on the shift |
| `test_golden_hf_gpt2` | golden | HF logits, loss, and the gradient of every HF tensor through the loader | real GPT-2 checkpoints load (`L7.9` does the same for SmolLM2) |
| `test_load_hf_gpt2_keys` | boundary | bare `GPT2Model` keys and old buffers load; a missing key is named; a wrong config is a shape error | checkpoints from many sources |
| `test_logits_are_causal` | property | changing tokens after $t$ leaves logits $0..t$ bitwise equal | training on all positions is honest |
| `test_block_matches_the_formula` | differential | one block in float64 numpy with large weights | tanh GELU, pre-LN, both residuals |
| `test_forward_targets_equals_clm_loss` | property | `forward(x[:, :-1], x[:, 1:])` equals `clm_loss` | the shift happens once |
| `test_head_is_tied_to_the_embedding` | unit | no `lm_head`; logits $= h W_{te}^\top$; `tie=False` adds one | half the output parameters |
| `test_gpt2_init` | statistical | residual projections at $0.02/\sqrt{2L}$, others 0.02, seeded | training starts stable at depth |
| `test_save_load_roundtrip` | unit | the model directory loads back exactly | the zoo (`L6.7`) |
| `test_validation` | boundary | windows past `n_ctx`, one-token loss, float ids | errors, not silent garbage |
| `test_learns_byte_stories` | learning | 150 AdamW updates of a 2-layer byte GPT reach the reference's held-out loss | the model learns language |

### Your graded tests (rung R5)

Make up a GPT-2 state dict in Hugging Face's layout yourself (random `[d, 3d]` `c_attn`, `[in, out]` Conv1D weights, perturbed LayerNorms), write the HF forward in float64 numpy from **that dict** (so your oracle never sees your transposes), load it with `load_hf_gpt2`, and compare logits. Add causality, the section 3 loss, bare keys, the tied head's gradient, and the residual init. `ol check L6.1` requires 0.80 with the pitfall faults killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a Conv1D weight loaded without the transpose | `c_proj` is square: no error, wrong logits | `test_golden_hf_gpt2` (mutant `s01`) |
| 2. `c_attn` split into interleaved columns instead of three blocks | q, k, v mixed | `test_golden_hf_gpt2` (mutant `s02`) |
| 3. no causal mask | loss near zero in training, nonsense when generating | `test_logits_are_causal`, `test_forward_targets_equals_clm_loss` (mutant `s03`) |
| 4. no shift (each position predicts its own token), or the padding mask applied to inputs | the model learns to copy; padded targets counted | `test_hand_example_clm_loss` (mutants `s04`, `s09`) |
| 5. exact GELU | off by about $10^{-3}$: HF parity fails | `test_block_matches_the_formula` (mutant `s05`) |
| 6. `ln_f` forgotten | logits too large; HF disagrees | `test_golden_hf_gpt2` (mutant `s06`) |
| 7. residual projections at the plain 0.02 | the stream's variance grows with depth | `test_gpt2_init` (mutant `s08`) |
| position embeddings never added | the model is a bag of tokens | `test_golden_hf_gpt2` (mutant `s07`) |
| no gradient through the tied head | `wte` learns only from the input side | `test_golden_hf_gpt2` (mutant `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L5.2` | `causal_mask` for every block |
| Back | `L5.3` | the attention of every block |
| Back | `L5.4` | `LearnedPE` is `wpe` |
| Back | `L0.1` | `Tensor` |
| Back | `L0.2` | the tanh GELU, `matmul` for the tied head |
| Back | `L0.3` | `cross_entropy` for both losses |
| Back | `L0.4` | `Linear`, `Embedding`, `LayerNorm` |
| Back | `L0.5` | `train_step` inside `fit_gpt` |
| Back | `L0.6` | safetensors model directory, `TokenStream` batches |
| Back | `M10.3` | `AdamW` |
| Back | `M10.4` | `cosine_with_warmup` |
| Back | `M07.3` | `normal_init`, `scaled_residual_std` |
| Back | `M06.3` | `PCG32` init stream |
| Forward | `L6.5` | sequence, token, and reward heads on `GPT.hidden` |
| Forward | `L6.6` | LoRA adapters on the attention projections |
| Forward | `L6.7` | the zoo's `gpt` rows (bits per byte) |
| Forward | `L8.2` | the first KV cache and `generate` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `GPT` | nanoGPT, HF `GPT2LMHeadModel` | flash attention, weight decay groups, `torch.compile` | `karpathy/nanoGPT/model.py`; `transformers/models/gpt2/modeling_gpt2.py` |
| `load_hf_gpt2` | HF `from_pretrained` | sharded safetensors, dtype casting, key renaming hooks | `transformers/modeling_utils.py` |
| learned `wpe` | RoPE in Llama-style models | no length cap from the table (`L7.3`) | Part 7 |
| `fit_gpt` | llm.c, Megatron | mixed precision, gradient accumulation, data parallelism | `L11.1`; `karpathy/llm.c` |
