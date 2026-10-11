<!-- ol:module L6.2 -->
# BERT encoder and MLM masking

## Overview

| | |
|---|---|
| **Module** | `L6.2` · build · Python · Pass 5 · 3 to 4 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/obj/bert.py`: `mlm_mask`, `special_ids`, `mlm_batch`, `BertConfig`, `BertLayer`, `BertEncoder`, `BertForMLM`, `hf_bert_encoder_sd`, `load_hf_bert`, `save_bert`, `load_bert`; and your own oracle tests in `python/tests/l6-2-bert/` |
| **Contract** | [`course/contracts/py/tinyllm/obj/bert.pyi`](../../../course/contracts/py/tinyllm/obj/bert.pyi) |
| **Tests** | `course/tests/L6.2/test_bert.py` (what they check: section 4); the oracles are a random tiny Hugging Face `BertForMaskedLM` and the masking op order replayed on the frozen PCG32 |
| **Needs** | `L5.3` attention · `L5.4` `LearnedPE` · `L1.3` `WordPieceTokenizer` · `M07.1` `sample_categorical` · `M06.3` `PCG32` · `L0.1` · `L0.2` · `L0.3` `cross_entropy` · `L0.4` layers · `L0.6` safetensors · `M07.3` `normal_init` (or `--ref-deps`) |
| **Used by** | `L6.3` ELECTRA's generator and discriminator · `L6.5`'s classification heads · `L6.7` the zoo's classification rows |
| **Milestone** | `MS-L6` |
| **Optional depth** | Devlin et al., "BERT" (NAACL 2019), sections 3.1 and appendix A.1; Liu et al., "RoBERTa" (2019), section 4.1 (dynamic masking) |

## Key Takeaways

- BERT keeps `L5.5`'s encoder (post-LN, exact GELU) and reads in both directions: only padding is masked (`test_reads_both_directions`, `test_padding_is_never_read`).
- The objective is fill-in-the-blank: pick 15% of the non-special positions, show [MASK] at 80% of them, a random token at 10%, the token itself at 10%, and predict the **original** token at exactly the picked positions (`test_hand_example_mlm_mask`, `test_keep_action_still_predicts`).
- One fixed op order over one PCG32 stream makes masking reproducible in every implementation (`test_mask_follows_the_op_order`), and its rates are checked statistically (`test_mask_rates`).
- The MLM head transforms, normalizes, and reuses the word table as its decoder, with its own bias (`test_golden_hf_bert`, `test_decoder_is_tied_and_has_a_bias`).

## How to work this chapter

```bash
ol start L6.2              # stubs bert.py; prints your test path and rung (R5)
ol tests L6.2              # the course tests
ol check L6.2              # course tests and the mutation grade of your tests
ol diff  L6.2              # after passing: your code against the reference
```

---

## 1. Why now

Your GPT (`L6.1`) reads left to right, which is what generation needs and what classification does not: to decide whether "not bad at all" is positive, every word should see every other. A bidirectional encoder cannot be trained by next-token prediction, because each position could simply read the answer. BERT's fix is to hide some tokens and ask for them back. The system needs this encoder twice: ELECTRA (`L6.3`) trains a small BERT as its generator and a BERT-shaped discriminator, and the classification heads of `L6.5` (the usage-policy head the gateway evaluates) sit on top of it. Its tokenizer is your WordPiece of `L1.3`.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_t$ | input id at position $t$ | `int` |
| $s_t \in \{0, 1\}$ | special position ([CLS], [SEP], [PAD], ...): never picked | `bool` |
| $p$ | picking probability, 0.15 | `float` |
| $u$ | one uniform draw from the PCG32 | `float` in $[0, 1)$ |
| $V$ | vocabulary size | `int` |
| $y_t$ | label: the original id at picked positions, $-100$ elsewhere | `int` |
| $\text{tt}_t$ | token type (segment A = 0, B = 1) | `int` |
| $W_{word}, W_{pos}, W_{type}$ | the three embedding tables | `float32[V, d]`, `[L, d]`, `[2, d]` |
| $\Phi$ | the standard normal CDF; exact GELU is $u\,\Phi(u)$ | |

### 2.1 The encoder

$$h_0 = \text{LN}(W_{word}[x] + W_{pos}[0..T-1] + W_{type}[\text{tt}]),$$

then $L$ post-LN layers: $h \leftarrow \text{LN}(h + \text{Attn}(h))$, $h \leftarrow \text{LN}(h + W_2\,\text{GELU}(W_1 h))$, with the exact GELU $u\,\Phi(u)$ (Hugging Face's `"gelu"`) and LayerNorm $\epsilon = 10^{-12}$. The attention mask is padding only, `[B, 1, 1, T]`: every real position reads every other real position, left and right. Changing the last word changes the first position's output, which a GPT can never do.

### 2.2 Masking 15%, and 80/10/10

Visit positions in C order (row by row):

1. a special position draws nothing; label $-100$;
2. draw $u$; if $u \ge p$ the position is not picked; label $-100$;
3. otherwise label $y_t = x_t$ (the original), draw $a$ = `sample_categorical([0.8, 0.1, 0.1], u')` (`M07.1`, inverse CDF);
4. $a = 0$: show [MASK]; $a = 1$: show `rng.below(V)`, a uniform random id; $a = 2$: show $x_t$ itself.

Why not always [MASK]? [MASK] never appears when the encoder is fine-tuned or used, so a model trained only on [MASK] positions would learn that real tokens need no thought. The random 10% teaches it that any visible token may be wrong, and the kept 10% that a visible token may be right. The label is always the original token, also at kept positions: those positions are predictions too. Draws happen only where the steps say, so the same seed gives the same batch in Python, Rust, and Go.

Because the random id is uniform over all $V$ ids, it can be [MASK] or the original token by chance: the exact shares of what a picked position **shows** are $0.8 + 0.1/V$ ([MASK]), $0.1 (V - 2)/V$ (another token), and $0.1 + 0.1/V$ (itself), which is what `test_mask_rates` checks.

### 2.3 The MLM head and loss

$$z_t = \text{LN}(\text{GELU}(W_T h_t + b_T))\,W_{word}^\top + b_{dec}, \qquad \mathcal{L} = \frac{1}{|\{t : y_t \ne -100\}|} \sum_{y_t \ne -100} -\log \operatorname{softmax}(z_t)_{y_t}.$$

The decoder weight is the word table (tied); only its bias $b_{dec}$ is new. Positions that were not picked contribute no loss and no gradient (`L0.3`'s `ignore_index`).

### 2.4 Hugging Face names

HF's BERT uses `nn.Linear`, so no transpose (unlike GPT-2's Conv1D, `L6.1`): `embeddings.{word,position,token_type}_embeddings`, `embeddings.LayerNorm`, `encoder.layer.i.attention.self.{query,key,value}`, `attention.output.dense`, `attention.output.LayerNorm`, `intermediate.dense`, `output.dense`, `output.LayerNorm`, and the head `cls.predictions.transform.dense`, `transform.LayerNorm`, `cls.predictions.bias`. HF's attention mask adds the most negative float instead of $-\infty$; a padded key gets weight 0 either way.

## 3. Worked example by hand

Ids `[CLS] the cat sat [SEP] [PAD]` = $(2, 5, 6, 7, 3, 0)$, [MASK] = 4, $V = 40$, $p = 0.15$, and a generator that will return the uniforms $0.05, 0.50, 0.90, 0.10, 0.85$ and then `below(40)` = 9.

| position | token | special? | draws | picked? | action | shown | label |
|---|---|---|---|---|---|---|---|
| 0 | [CLS] | yes | none | | | 2 | -100 |
| 1 | the | no | $u = 0.05 < 0.15$; $a$: $0.50 < 0.8$ | yes | [MASK] | 4 | 5 |
| 2 | cat | no | $u = 0.90$ | no | | 6 | -100 |
| 3 | sat | no | $u = 0.10$; $a$: $0.8 \le 0.85 < 0.9$; below = 9 | yes | random | 9 | 7 |
| 4 | [SEP] | yes | none | | | 3 | -100 |
| 5 | [PAD] | yes | none | | | 0 | -100 |

Inputs $(2, 4, 6, 9, 3, 0)$, labels $(-100, 5, -100, 7, -100, -100)$; exactly five uniforms and one `below`. This is `test_hand_example_mlm_mask`, with a generator that fails on any extra draw.

## 4. The interface

```python
def mlm_mask(ids, special_mask, mask_id: int, vocab: int, p: float, rng) -> tuple[NDArray, NDArray]  # inputs, labels
def special_ids(tok: WordPieceTokenizer) -> list[int]
def mlm_batch(tok, texts, max_len: int, p: float, rng) -> dict[str, NDArray]   # input_ids, token_type_ids, attention_mask, labels
@dataclass
class BertConfig: vocab: int; max_len: int = 512; d_model: int = 768; n_heads: int = 12; n_layers: int = 12
                  d_ff: int = 3072; type_vocab: int = 2; dropout: float = 0.1; ln_eps: float = 1e-12
class BertEncoder(Module): def forward(self, ids, token_type_ids=None, attn_mask=None) -> Tensor
class BertForMLM(Module):  def forward(self, ids, token_type_ids=None, attn_mask=None, labels=None) -> (logits, loss)
def load_hf_bert(model, sd) -> None;  def save_bert(model, dir); def load_bert(dir) -> BertForMLM
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_mlm_mask` | unit | section 3, draw for draw | you and the test agree on the op order |
| `test_keep_action_still_predicts` | unit | a kept token is still labelled | the 10% that teaches "visible may be right" |
| `test_mask_follows_the_op_order` | golden | a `[4, 32]` batch equals the op order replayed on the frozen PCG32 | the same batch in every language |
| `test_mask_rates` | statistical | picked share within 4 sd of $p$; 80/10/10 shares by chi-square | the objective's proportions |
| `test_golden_hf_bert` | golden | HF logits, MLM loss, and gradients through the loader | real BERT checkpoints load (`L6.3`, `L6.5`) |
| `test_layer_matches_the_formula` | differential | one layer in float64 numpy with the exact GELU and post-LN | the activation and the norm placement |
| `test_padding_is_never_read` | property | ids at padded positions change no real output | batches of different lengths |
| `test_reads_both_directions` | property | the last token changes the first position | bidirectional, unlike GPT |
| `test_token_types_are_added` | property | segment B changes the output; default is A | sentence-pair tasks |
| `test_loss_only_at_labelled_positions` | unit | the loss is the cross-entropy at the labelled position | no loss from visible tokens |
| `test_decoder_is_tied_and_has_a_bias` | unit | parameter names and order; no decoder weight | checkpoint keys for `L6.3` and the zoo |
| `test_mlm_batch_from_text` | unit | WordPiece ids, padding, attention mask; specials never picked at $p = 1$ | batches straight from text |
| `test_load_encoder_alone_and_roundtrip` | unit | a bare encoder loads from an MLM checkpoint; the model directory roundtrips | classifiers start from pretrained encoders |
| `test_validation` | boundary | bad ids, shapes, rates, mask id, module type | errors, not silent garbage |

### Your graded tests (rung R5)

Two oracles. For masking, replay the op order yourself with your own `PCG32` (two generators from one seed: one for `mlm_mask`, one for your loop) and compare exactly; add the section 3 example with a scripted generator that fails on extra draws. For the model, write one layer and the MLM head in float64 numpy (exact GELU with `math.erf`) and compare; build a Hugging Face-named state dict yourself to test `load_hf_bert`. `ol check L6.2` requires 0.80 with the pitfall faults killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. special or padded positions picked | the model learns to predict [CLS] and [PAD] | `test_hand_example_mlm_mask`, `test_mlm_batch_from_text` (mutants `s01`, `s12`) |
| 2. every position labelled | the loss is dominated by copying visible tokens | `test_hand_example_mlm_mask`, `test_mask_rates` (mutant `s02`) |
| 3. random and keep swapped | the rates look right; the batch differs from every other implementation | `test_mask_follows_the_op_order`, `test_keep_action_still_predicts` (mutant `s03`) |
| 4. the label is what was shown, not the original | the model learns to predict [MASK] | `test_hand_example_mlm_mask` (mutant `s04`) |
| 5. a second uniform drawn for every position | a different stream from the second position on | `test_mask_follows_the_op_order` (mutant `s05`) |
| 6. GPT-2's tanh GELU instead of the exact one | HF parity off by about $10^{-3}$ | `test_layer_matches_the_formula` (mutant `s06`) |
| 7. pre-LN instead of BERT's post-LN | a different model; HF disagrees | `test_golden_hf_bert`, `test_layer_matches_the_formula` (mutant `s07`) |
| 8. padding keys not masked | outputs depend on the batch's longest sentence | `test_padding_is_never_read` (mutant `s08`) |
| 9. a causal mask copied from GPT | the encoder is one-directional | `test_reads_both_directions` (mutant `s11`) |
| token types never added | sentence pairs look like one sentence | `test_token_types_are_added` (mutant `s09`) |
| the decoder bias left out | HF disagrees | `test_golden_hf_bert` (mutant `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L5.3` | the attention of every layer |
| Back | `L5.4` | `LearnedPE` is the position embedding |
| Back | `L1.3` | `WordPieceTokenizer` encodes text in `mlm_batch` |
| Back | `M07.1` | `sample_categorical` picks the 80/10/10 action |
| Back | `M06.3` | `PCG32`: every draw of the masking, and the init stream |
| Back | `L0.1` | `Tensor` |
| Back | `L0.2` | exact `gelu`, `matmul` for the tied decoder |
| Back | `L0.3` | `cross_entropy` with `ignore_index` |
| Back | `L0.4` | `Linear`, `Embedding`, `LayerNorm` |
| Back | `L0.6` | safetensors model directory |
| Back | `M07.3` | `normal_init` |
| Forward | `L6.3` | ELECTRA: a small `BertForMLM` generator and a `BertEncoder` discriminator, masked with `mlm_mask` |
| Forward | `L6.5` | sequence and token classification heads on `BertEncoder` |
| Forward | `L6.7` | the zoo's classification rows |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `mlm_mask` | HF `DataCollatorForLanguageModeling` | masking at batch time, whole-word masking | `transformers/data/data_collator.py` |
| static 15% | RoBERTa dynamic masking; 40% masking | a new mask every epoch; higher rates for large models | Liu et al. 2019; Wettig et al. 2023 |
| `BertEncoder` | ModernBERT | RoPE, GeGLU, alternating local and global attention, 8k context | Warner et al. 2024 |
| MLM | ELECTRA's replaced-token detection | a loss on every position, not 15% | `L6.3`; Clark et al. 2020 |
