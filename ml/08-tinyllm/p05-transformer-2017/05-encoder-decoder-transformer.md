<!-- ol:module L5.5 -->
# Encoder-decoder Transformer: post-LN and pre-LN, Noam, label smoothing

## Overview

| | |
|---|---|
| **Module** | `L5.5` · build · Python · Pass 5 · 4 to 6 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/xfmr/transformer.py`: `TransformerConfig`, `EncoderLayer`, `DecoderLayer`, `Transformer` (`encode`, `decode`, `decode_step`), `label_smoothed_loss`, `noam_rate`, `make_optimizer`, `fit`, `translate`, `save_transformer`, `load_transformer`; and your own oracle tests in `python/tests/l5-5-transformer/` |
| **Contract** | [`course/contracts/py/tinyllm/xfmr/transformer.pyi`](../../../course/contracts/py/tinyllm/xfmr/transformer.pyi) |
| **Tests** | `course/tests/L5.5/test_transformer.py` (what they check: section 4); the oracle is torch's `TransformerEncoder`/`TransformerDecoder` with copied weights, both LayerNorm placements; the learning test compares with the reference's bar over 5 seeds |
| **Needs** | `L5.2` masks · `L5.3` attention · `L5.4` `SinusoidalPE` · `L0.1` · `L0.2` · `L0.3` `cross_entropy` · `L0.4` layers · `L0.5` `DataLoader`, `train_step` · `L0.6` safetensors · `M10.3` `Adam` · `M10.4` `noam` · `M07.3` `normal_init` · `M06.3` `PCG32` · `L4.4` `beam_search` (or `--ref-deps`) |
| **Used by** | `L6.7` the zoo's `transformer` rows on the addition task |
| **Milestone** | `MS-L5` (exact match 0.98 on unseen sums; post-LN fails without warmup where pre-LN trains) |
| **Optional depth** | Vaswani et al. (2017), sections 3 and 5; Xiong et al., "On Layer Normalization in the Transformer Architecture" (ICML 2020); Szegedy et al. (2016), section 7 (label smoothing); Rush, *The Annotated Transformer* |

## Key Takeaways

- The encoder reads the whole source once; the decoder writes one token at a time with three sublayers: masked self-attention, cross-attention into the encoder's output, and a feed-forward network (`test_golden_torch`).
- Two masks keep it honest: the decoder may not read the future (`test_decoder_is_causal`), and nobody may read padding (`test_source_padding_is_never_read`).
- Where the LayerNorm sits decides trainability: post-LN (the paper) normalizes after the residual sum and needs warmup; pre-LN normalizes the sublayer's input, needs one final norm per stack, and trains without warmup (`test_golden_torch`, `MS-L5`).
- The paper's recipe is three numbers you can check by hand: embeddings times $\sqrt{d}$, the Noam rate with update $s$ at step $s + 1$, and label smoothing that moves $\varepsilon$ of the target mass onto every id (`test_hand_example_noam`, `test_hand_example_label_smoothing`).
- Decoding one token at a time must give the logits of the full forward pass (`test_decode_step_equals_forward`); beam search (`L4.4`) runs on top of that step.

## How to work this chapter

```bash
ol start L5.5              # stubs transformer.py; prints your test path and rung (R5)
ol tests L5.5              # the course tests
ol check L5.5              # course tests and the mutation grade of your tests
ol milestone MS-L5 --smoke # your CLI: train transformer, translate, the LayerNorm demo
ol diff  L5.5              # after passing: your code against the reference
```

---

## 1. Why now

Your seq2seq model of Part 4 (`L4.1` to `L4.3`) reads a sentence one token at a time through a GRU: training cannot run the positions in parallel, and everything the decoder knows about the source passes through attention over states that were themselves built sequentially. You now have the three parts that remove the recurrence: attention with heads (`L5.3`), masks (`L5.2`), and positions (`L5.4`). This module assembles them into the 2017 architecture, trains it with the paper's recipe, and decodes it with your beam search. The test task of `MS-L5` is three-digit addition, which needs exact attention to two positions per output digit and carries between them: an RNN encoder-decoder of the same size learns it slowly; the transformer reaches 0.98 exact match in a few thousand updates. The same build also exposes the first architectural decision every modern LLM made differently from the paper: where to put the LayerNorm.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in \mathbb{R}^{S \times d}$ | source states; $S$ source length | `[B, S, d]` |
| $y \in \mathbb{R}^{T \times d}$ | target states; $T$ target length | `[B, T, d]` |
| $d, H, d_{ff}$ | model width, heads, feed-forward width | `int` |
| $N_e, N_d$ | encoder and decoder layers | `int` |
| $E$ | the shared embedding table | `float32[V, d]` |
| $\text{LN}$ | LayerNorm over the last axis (`L0.4`) | |
| $\text{FF}(x) = W_2\,\text{relu}(W_1 x + b_1) + b_2$ | position-wise feed-forward | `[.., d] -> [.., d]` |
| $V$ | vocabulary size | `int` |
| $\varepsilon$ | label smoothing | `float` in $[0, 1]$ |
| $s$ | update index, 0 for the first update | `int` |
| $w$ | warmup updates | `int` |

### 2.1 The stacks

**Input.** $x_0 = E[\text{src}] \sqrt{d} + \text{PE}$, the same for the target with the same table $E$ (the paper shares one matrix between the two embeddings and the output layer). $E$ starts at $\mathcal{N}(0, 1/d)$, so $E\sqrt d$ has unit scale, the scale of $\text{PE}$.

**Encoder layer** (post-LN): $x \leftarrow \text{LN}_1(x + \text{SelfAttn}(x))$, then $x \leftarrow \text{LN}_2(x + \text{FF}(x))$. Self-attention may read every real source position.

**Decoder layer** (post-LN): $y \leftarrow \text{LN}_1(y + \text{SelfAttn}(y))$ with the causal and target-padding mask, $y \leftarrow \text{LN}_2(y + \text{CrossAttn}(y, m))$ where $m$ is the encoder's output and the mask is the source padding, then $y \leftarrow \text{LN}_3(y + \text{FF}(y))$.

**Output.** $\text{logits} = y E^\top$. With `tie_embeddings` and equal vocabularies, `src_emb`, `tgt_emb`, and `out` are one tensor, listed once in the `state_dict`.

### 2.2 Where the LayerNorm goes

Post-LN computes $\text{LN}(x + F(x))$: every layer's output is normalized, but the gradient that reaches early layers passes through every LN on the residual path, and at initialization the gradients of the last layers are large. With Adam at a fixed learning rate the first updates are too big and training stalls. Xiong et al. show this, and that the fix the paper used, a learning-rate **warmup**, is what makes post-LN trainable.

Pre-LN computes $x + F(\text{LN}(x))$: the residual path is the identity, gradients reach every layer undamped, and no warmup is needed. Because nothing normalizes the stream any more, a pre-LN stack ends with one more LN (`enc_norm`, `dec_norm`). torch's `norm_first=True` is this; GPT-2 (`L6.1`) and every model of Part 7 use it. The `MS-L5` demo trains a 6 + 6 layer model at a constant rate $3 \times 10^{-3}$ with no warmup: post-LN keeps about 91% of its first loss after 150 updates, pre-LN about 54%.

### 2.3 Three kinds of attention

| Sublayer | Queries | Keys and values | Mask |
|---|---|---|---|
| encoder self | source | source | source padding `[B, 1, 1, S]` |
| decoder self | target | target | causal `[T, T]` AND target padding `[B, 1, 1, T]` (`L5.2`'s `combine`) |
| decoder cross | target | encoder output | source padding |

The caller passes padding masks only (`src != pad`, `tgt_in != pad`); the model builds these attention masks, so it cannot forget causality.

### 2.4 Training recipe

**Teacher forcing.** A target $y_1 \dots y_T$ is fed as `tgt_in` = (bos, $y_1, \dots, y_{T-1}$) and predicted as $(y_1, \dots, y_T)$, every position at once: the causal mask is what makes this one forward pass equal to $T$ sequential steps.

**Label smoothing.** With $p = \text{softmax}(\text{logits})$ the smoothed target is $q = (1 - \varepsilon)\,\text{onehot}(t) + \varepsilon / V$ and the loss is $-\sum_j q_j \log p_j = (1 - \varepsilon)(-\log p_t) + \varepsilon \cdot \text{mean}_j(-\log p_j)$, with gradient $p - q$. It stops the model from pushing one logit to infinity. `L0.3`'s `cross_entropy(label_smoothing=)` already implements it; `label_smoothed_loss` adds the padding rule: a target equal to the pad id is not a prediction and counts for nothing, not even in the mean.

**Noam schedule** (`M10.4`): $\text{lr}(n) = f\, d^{-1/2} \min(n^{-1/2}, n\, w^{-3/2})$, a linear warmup for $w$ steps, then a decay as $n^{-1/2}$. The paper counts steps from $n = 1$; update $s$ (counting from 0) therefore runs at $n = s + 1$. Evaluating at $n = s$ gives rate 0 to the first update.

**Optimizer** (`M10.3`): Adam with $\beta = (0.9, 0.98)$, $\epsilon = 10^{-9}$, no weight decay.

## 3. Worked example by hand

**Label smoothing**, $V = 3$, logits $(1, 0, 0)$, target 0, $\varepsilon = 0.1$:
$p = (e, 1, 1)/(e + 2) = (0.576117, 0.211942, 0.211942)$, so
$-\log p = (0.551445, 1.551445, 1.551445)$ and

$$\text{loss} = 0.9 \times 0.551445 + 0.1 \times \tfrac{0.551445 + 2 \times 1.551445}{3} = 0.496300 + 0.121811 = 0.618111 .$$

$q = (0.9 + 0.1/3, 0.1/3, 0.1/3) = (0.933333, 0.033333, 0.033333)$, gradient $p - q = (-0.357216, 0.178608, 0.178608)$. A second row whose target is the pad id contributes no loss, no gradient, and does not count in the mean: `test_hand_example_label_smoothing`.

**Noam**, $d = 16$, $w = 4$, $f = 1$: $d^{-1/2} = 0.25$.

| update $s$ | $n = s + 1$ | $\min(n^{-1/2}, n w^{-3/2})$ | rate |
|---|---|---|---|
| 0 | 1 | $\min(1, 1/8) = 0.125$ | 0.03125 |
| 3 | 4 | $\min(0.5, 0.5) = 0.5$ | 0.125 (peak) |
| 15 | 16 | $\min(0.25, 2) = 0.25$ | 0.0625 |

This is `test_hand_example_noam`.

## 4. The interface

```python
@dataclass
class TransformerConfig: src_vocab: int; tgt_vocab: int; d_model: int = 512; n_heads: int = 8; d_ff: int = 2048
                         n_enc: int = 6; n_dec: int = 6; dropout: float = 0.1; norm: Literal['post','pre'] = 'post'
                         tie_embeddings: bool = True; max_len: int = 256; ln_eps: float = 1e-5
class Transformer(Module):
    def encode(self, src, src_mask) -> Tensor                                # [B, S, d]
    def decode(self, tgt_in, memory, src_mask, tgt_mask) -> Tensor          # logits [B, T, V]
    def forward(self, src, tgt_in, src_mask, tgt_mask) -> Tensor
    def init_state(self, src_mask) -> DecodeState
    def decode_step(self, y_prev, memory, state) -> tuple[Tensor, DecodeState]   # last position's logits
def label_smoothed_loss(logits, targets, eps, pad_id) -> Tensor
def noam_rate(step, d_model, warmup, factor=1.0) -> float
def fit(model, src, tgt, steps, batch, pad_id, warmup=4000, factor=1.0, smoothing=0.1, lr=None, ...) -> list[float]
def translate(model, src, bos, eos, pad_id, max_len, beam_size=1, length_penalty=1.0) -> list[list[int]]
def save_transformer(model, dir) / load_transformer(dir)
```

Masks passed in are padding masks, bool `[B, S]` and `[B, T]`, True = a real token.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_label_smoothing` | unit | section 3: 0.618111, the gradient, the ignored pad row | the loss every update of `MS-L5` uses |
| `test_hand_example_noam` | unit | section 3: 0.03125, 0.125, 0.0625 | the first update learns |
| `test_golden_torch` | golden | torch post-LN and pre-LN: logits, smoothed loss, every gradient | both placements match the reference implementation |
| `test_decoder_is_causal` | property | changing targets after $t$ leaves logits $0..t$ bitwise equal | teacher forcing is honest |
| `test_source_padding_is_never_read` | property | ids in padded source positions change nothing | batches of different lengths |
| `test_decoder_reads_the_source` | property | a different source changes the logits | cross-attention is wired to the encoder |
| `test_decode_step_equals_forward` | differential | step-by-step decoding equals the full forward pass | beam search scores the right logits |
| `test_embeddings_scaled_and_tied` | unit | $E\sqrt d + \text{PE}$; one tied table, listed once | the paper's section 3.4 |
| `test_parameter_names_follow_the_norm` | unit | pre-LN adds exactly `enc_norm`, `dec_norm`; decoder layers have `cross_attn`, `norm3` | zoo checkpoints (`L6.7`) |
| `test_optimizer_is_the_papers_adam` | unit | betas (0.9, 0.98), eps 1e-9 | the recipe |
| `test_fit_follows_the_schedule_and_is_reproducible` | property | exact step count, same seed same losses, a constant `lr` | the milestone's comparisons |
| `test_translate_is_greedy_with_beam_one_and_drops_eos` | differential | beam 1 equals a hand argmax loop; no eos; mode restored; save/load | `{tinyllm} translate` |
| `test_validation` | boundary | unknown norm, non-bool masks, batch mismatch | wiring bugs fail loudly |
| `test_learns_reversed_addition` | learning | 600 updates on two-digit reversed addition reach the reference's held-out loss | the recipe trains |

### Your graded tests (rung R5)

Write the whole forward pass again in float64 numpy from the model's `state_dict`: embeddings times $\sqrt d$ plus `sinusoidal_pe`, the attention of `L5.3` written out, LayerNorm, ReLU, and the post-LN and pre-LN orders, for a one-layer model with padding in both source and target, and compare the logits. Add the section 3 numbers, the causal and padding laws, `decode_step` against the forward pass, and a `translate` whose eos is chosen to finish. `ol check L5.5` requires 0.80 with the pitfall faults (`s01` to `s08`) killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. post-LN written as pre-LN (normalizing the sublayer's input) | a different model from the paper's; torch disagrees | `test_golden_torch` (mutant `s01`) |
| 2. pre-LN without the final LayerNorm | the stack's output is unnormalized; logits blow up with depth | `test_golden_torch` (mutant `s02`) |
| 3. no causal mask in the decoder | training loss near zero, generation garbage | `test_decoder_is_causal` (mutant `s03`) |
| 4. cross-attention reading the decoder instead of the encoder | trains as a language model of the targets, never learns the task | `test_decoder_reads_the_source` (mutant `s04`) |
| 5. embeddings not multiplied by $\sqrt d$ | positions swamp the tokens at init | `test_embeddings_scaled_and_tied` (mutant `s05`) |
| 6. source padding readable in cross-attention | outputs depend on how long the batch's other sentences are | `test_source_padding_is_never_read` (mutant `s06`) |
| 7. label smoothing counting pad targets | short targets in a batch pull every logit toward the pad id | `test_hand_example_label_smoothing` (mutant `s07`) |
| 8. Noam at $n = s$ | the first update has rate 0, the peak is one late | `test_hand_example_noam` (mutant `s08`) |
| the output layer not tied | twice the parameters; the zoo's keys differ | `test_embeddings_scaled_and_tied` (mutant `s09`) |
| `decode_step` returning the first position | beam search scores the wrong token | `test_decode_step_equals_forward` (mutant `s10`) |
| `translate` keeping eos | exact match is 0 | `test_translate_is_greedy_with_beam_one_and_drops_eos` (mutant `s11`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L5.2` | `causal_mask` and `combine` build the decoder mask |
| Back | `L5.3` | every attention sublayer |
| Back | `L5.4` | `SinusoidalPE` on source and target |
| Back | `L0.1` | `Tensor` and `no_grad` for decoding |
| Back | `L0.2` | `relu`, the ops of every layer |
| Back | `L0.3` | `cross_entropy` with `label_smoothing` |
| Back | `L0.4` | `Linear`, `Embedding`, `LayerNorm`, `ModuleList` |
| Back | `L0.5` | `DataLoader` and `train_step` inside `fit` |
| Back | `L0.6` | safetensors for the model directory |
| Back | `M10.3` | `Adam` |
| Back | `M10.4` | `noam` |
| Back | `M07.3` | `normal_init` for the $\mathcal{N}(0, 1/d)$ table |
| Back | `M06.3` | `PCG32` init and shuffle streams |
| Back | `L4.4` | `beam_search` over `decode_step` |
| Forward | `L6.1` | GPT keeps only the decoder, in pre-LN |
| Forward | `L6.2` | BERT keeps only the encoder, in post-LN |
| Forward | `L6.7` | the zoo loads `tl_arch = transformer` and reports exact match with beam search |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Transformer` | torch `nn.Transformer`, fairseq `TransformerModel` | fused attention, `norm_first`, incremental state caching | `torch/nn/modules/transformer.py`; `fairseq/models/transformer/` |
| post-LN vs pre-LN | DeepNorm, Sandwich-LN, RMSNorm pre-norm | train 1000-layer post-LN; modern LLMs use pre-RMSNorm (`L7.1`) | Wang et al., "DeepNet" (2022) |
| `noam_rate` | linear warmup + cosine, WSD | the schedules LLM training uses (`M10.4`, `C1`) | HF `get_cosine_schedule_with_warmup` |
| `decode_step` recomputing the prefix | the KV cache | $O(T)$ instead of $O(T^2)$ work per token | `L8.2`; HF `past_key_values` |
