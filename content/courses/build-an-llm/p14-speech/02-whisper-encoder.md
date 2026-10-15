<!-- ol:module L14.2 -->
# Whisper encoder: conv stem, sinusoids, pre-LN blocks

## Overview

| | |
|---|---|
| **Module** | `L14.2` · build · Python · Pass 12 · 4 to 6 h |
| **You build** | `python/tinyllm/speech/encoder.py`: `WhisperConfig` (with `from_hf`), `whisper_sinusoids`, `WhisperAttention` (no key bias), `WhisperEncoderLayer`, `WhisperEncoder` |
| **Contract** | [`course/contracts/py/tinyllm/speech/encoder.pyi`](../../../../course/contracts/py/tinyllm/speech/encoder.pyi) |
| **Tests** | `course/tests/L14.2/test_encoder.py` (what they check: section 4), golden values from `course/oracle/L14.2/whisper_tiny_hf.py` (tiny random HF Whisper models, transformers 5.19) |
| **Needs** | `L13.1` `Conv1d` · `M12.1` `conv_out_size` · `L5.3` `MultiHeadAttention` · `L0.4` `Linear`, `LayerNorm`, `ModuleList`, `Module` · `L0.2` `gelu`, `transpose` · `L0.1` `Tensor` · `M06.3` `PCG32` (or `--ref-deps`). Reading: `L5.1` attention, `L5.4` sinusoids, `L14.1` the features it reads |
| **Used by** | `L14.3` (Whisper composes it) · later `L15.6` (the Rust encoder is proven against these fixtures) |
| **Milestone** | `MS-L14` step 2 (tiny Whisper logits within $10^{-5}$) |
| **Optional depth** | Radford et al., "Robust speech recognition via large-scale weak supervision" (2022), section 2.2 and figure 1; Gulati et al., "Conformer" (2020), for what a speech encoder can add; Xiong et al., "On layer normalization in the transformer architecture" (2020), for pre-LN |

## Key Takeaways

- The stem is two width-3 `Conv1d`s with GELU; the second has stride 2, so 3000 mel frames (10 ms each) become 1500 positions (20 ms each) (`test_stem_output_length`).
- Whisper's position table concatenates all the sines and then all the cosines, with frequencies $10000^{-i/(C/2-1)}$; L5.4's table interleaves them (`test_hand_example_sinusoids`).
- The blocks are pre-LN with unmasked self-attention and a final LayerNorm; the key projection has no bias (`test_encoder_matches_hf_tiny_whisper`, `test_parameter_names_are_hf_names`).
- A float32 table at angle 1499 rad is only good to about $3 \cdot 10^{-5}$ across libraries, so model parity uses the checkpoint's stored table (`test_sinusoids_match_hf_tables`).

## How to work this chapter

```bash
ol start L14.2              # stubs python/tinyllm/speech/encoder.py into your repo
ol tests L14.2              # read the test catalog first: rung R0, you write no tests here
ol check L14.2              # exit code is the verdict
ol check L14.2 --ref-deps   # only if one of your Needs modules is not passing yet
ol diff  L14.2              # after passing: your code against the reference
```

---

## 1. Why now

`L14.1` turns any WAV file into the $80 \times 3000$ picture Whisper expects. This module is the half of Whisper that looks at the picture: the encoder runs once per 30 s window and produces 1500 vectors that every decoding step (`L14.3`, `L14.4`) attends to. Nothing in it is new: a convolution from `L13.1`, attention from `L5.3`, LayerNorm and MLPs from Part 0, sinusoids like `L5.4`'s. What is new is fitting them to a published checkpoint exactly, where three small conventions (concatenated sinusoids, a key projection without bias, pre-LN with a final norm) decide whether 37.8 million loaded weights compute anything meaningful. The engine's candle encoder (`L15.6`) is later proven against the fixtures this module passes.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $M$ | log-mel input | `float32[N, n_mels, 2A]` |
| $A$ | audio positions (`n_audio_ctx`, 1500) | int |
| $d$, $H$ | width and heads | int |
| $P$ | sinusoid table | `float32[A, d]` |
| $x_\ell$ | hidden state after block $\ell$ | `float32[N, A, d]` |

### 2.1 The convolutional stem

Speech features are local: a phone lasts tens of milliseconds. Two `Conv1d` layers with kernel 3 and padding 1 mix each frame with its neighbours before attention sees anything:
$$h_1 = \text{gelu}(\text{conv}_{n_\text{mels} \to d}^{k=3, p=1}(M)), \qquad h_2 = \text{gelu}(\text{conv}_{d \to d}^{k=3, s=2, p=1}(h_1)).$$
With `M12.1`'s output size $\lfloor (n + 2p - k)/s \rfloor + 1$, the first keeps $3000$ frames and the second gives $\lfloor 2999/2 \rfloor + 1 = 1500$. Halving the length quarters the cost of attention. GELU is the exact form $x\,\Phi(x)$.

### 2.2 Sinusoidal positions, Whisper's way

With $C = d$ channels and half $= C/2$, Whisper uses $\text{inc} = \ln(10000)/(\text{half} - 1)$, frequencies $\omega_i = e^{-\text{inc}\cdot i}$ for $i < \text{half}$, and
$$P_{p} = [\sin(p\,\omega_0), \dots, \sin(p\,\omega_{\text{half}-1}),\ \cos(p\,\omega_0), \dots, \cos(p\,\omega_{\text{half}-1})].$$
Compared with `L5.4`: the sines and cosines are two blocks, not interleaved pairs, and the ladder ends exactly at $1/10000$ (divided by half $- 1$, not half). The table is fixed, added once after the stem, and never trained. Computed in float32, an angle near 1499 rad carries the error of one ulp of `exp`, about $3 \cdot 10^{-5}$ after the multiplication, and different libraries' float32 `exp` differ by that ulp. Checkpoints store the table they were trained with, so `L14.3`'s loader checks the stored table against `whisper_sinusoids` (within $10^{-4}$) and then uses the stored one.

### 2.3 Pre-LN blocks

Each block is
$$x \leftarrow x + \text{attn}(\text{LN}_1(x)), \qquad x \leftarrow x + W_2\,\text{gelu}(W_1\,\text{LN}_2(x) + b_1) + b_2,$$
and the encoder ends with one more LayerNorm. Attention has no mask: every position attends to all 1500. It is `L5.3`'s multi-head attention with $Q, V, O$ biases but no $K$ bias (a bias on keys adds the same amount to every score of a query, which the softmax cancels, so Whisper drops it). Parameter names follow Hugging Face (`self_attn.k_proj.weight`, `self_attn_layer_norm`, `fc1`, `fc2`, `final_layer_norm`, `layer_norm`) so a checkpoint loads by name.

## 3. Worked example by hand

The table for $C = 4$ channels: half $= 2$, $\text{inc} = \ln(10000)/1 = 9.2103$, $\omega = (e^0, e^{-9.2103}) = (1, 10^{-4})$. Row $p$ is $[\sin p,\ \sin(10^{-4} p),\ \cos p,\ \cos(10^{-4} p)]$, so row 1 is $[0.8415,\ 0.0001,\ 0.5403,\ 1.0000]$ and row 2 is $[0.9093,\ 0.0002,\ -0.4161,\ 1.0000]$. `L5.4`'s interleaved table for the same size would put $\cos p$ second (`test_hand_example_sinusoids`).

The stem's lengths for a 30 s window: conv 1 maps 3000 frames to $(3000 + 2 - 3)/1 + 1 = 3000$; conv 2 maps them to $\lfloor (3000 + 2 - 3)/2 \rfloor + 1 = 1500$, exactly the table's rows. A 2998-frame input would give 1499 positions and is refused, as HF refuses it (`test_stem_output_length`, on the small model: 16 frames give 8).

## 4. The interface

```python
@dataclass
class WhisperConfig:                       # whisper-tiny's sizes by default
    n_mels=80; n_audio_ctx=1500; d=384; n_heads=6; enc_layers=4; dec_layers=4
    n_text_ctx=448; vocab=51865; d_ff=1536
    @classmethod
    def from_hf(cls, config_json: str) -> "WhisperConfig": ...
    def validate(self) -> None: ...
def whisper_sinusoids(length, channels, max_timescale=10000.0) -> NDArray: ...
class WhisperAttention(MultiHeadAttention): ...      # k_proj without bias
class WhisperEncoderLayer(Module): ...
class WhisperEncoder(Module):
    def forward(self, mel) -> Tensor: ...             # [N, n_mels, 2A] -> [N, A, d]
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_sinusoids` | unit, smoke | section 3's 4-channel table | the layout of positions |
| `test_sinusoids_match_hf_tables` | golden | HF's stored tables: $10^{-6}$ short, $10^{-4}$ at 1500 rows | why the loader keeps the stored copy |
| `test_stem_output_length` | unit | $2A$ frames give $A$ positions; other lengths, mel counts, ranks refused | 30 s windows only |
| `test_encoder_matches_hf_tiny_whisper` | golden | the full encoder on a real clip within $10^{-5}$ of HF | `L14.3`'s logits, `L15.6`'s fixtures |
| `test_encoder_matches_hf_small_batch` | golden | 128 mels, a batch of two, items independent | other checkpoints, batching |
| `test_parameter_names_are_hf_names` | unit | every name and shape; no `k_proj.bias` | loading by name |
| `test_config_from_hf` | unit | HF config keys; unsupported variants refused | refusing what would load wrong |
| `test_blocks_are_pre_ln_residual` | property | zeroed branches leave layer_norm(stem + positions) | the block structure |
| `test_every_frame_sees_every_frame` | property | the last frame reaches the first output | no mask in the encoder |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. sin and cos interleaved as in `L5.4` | loads fine, transcribes nothing | `test_hand_example_sinusoids`, `test_sinusoids_match_hf_tables` (mutant `s01`) |
| 2. cosines first | the same, more subtly | `test_hand_example_sinusoids` (mutant `s02`) |
| 3. the second conv without stride 2 | 3000 positions against a 1500-row table | `test_stem_output_length`, `test_encoder_matches_hf_tiny_whisper` (mutant `s03`) |
| 4. tanh GELU in the stem | outputs off by about $10^{-3}$ | `test_encoder_matches_hf_tiny_whisper` (mutant `s04`) |
| 5. a bias on `k_proj` | strict loading fails: an unexpected key | `test_parameter_names_are_hf_names` (mutant `s05`) |
| 6. post-LN blocks | normalized residual stream, wrong outputs | `test_blocks_are_pre_ln_residual`, `test_encoder_matches_hf_tiny_whisper` (mutant `s06`) |
| 7. no final LayerNorm | unnormalized states reach the decoder | `test_blocks_are_pre_ln_residual` (mutant `s07`) |
| 8. a causal mask copied from the decoder | early frames never hear later ones | `test_every_frame_sees_every_frame` (mutant `s08`) |
| 9. positions never added | the model cannot tell when a sound happened | `test_blocks_are_pre_ln_residual` (mutant `s09`) |
| 10. dividing the frequency step by $C/2$ | the last frequency is not $10^{-4}$ | `test_hand_example_sinusoids` (mutant `s10`) |
| accepting zero layers | an encoder that is only a stem | `test_config_from_hf` (mutant `m01`) |
| accepting a shorter window | positions silently misaligned | `test_stem_output_length` (mutant `m02`) |
| accepting an odd width | sin and cos halves of unequal size | `test_config_from_hf` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L13.1` | `Conv1d(n_mels, d, 3, pad=1)` and `Conv1d(d, d, 3, stride=2, pad=1)` |
| Back | `M12.1` | `conv_out_size` for the length check |
| Back | `L5.3` | `WhisperAttention` subclasses `MultiHeadAttention` |
| Back | `L0.4` | `Linear`, `LayerNorm`, `ModuleList`, `Module` |
| Back | `L0.2` | `gelu` (exact) and `transpose` |
| Back | `L0.1` | arrays become a `Tensor` at the input |
| Back | `M06.3` | `PCG32(0).substream("init")` when no rng is given |
| Back | `L5.1`, `L5.4`, `L14.1` | attention, the other sinusoid layout, the features (reading) |
| Forward | `L14.3` | `Whisper` runs the encoder once per window and hands its output to every decoder layer's cross-attention |
| Forward | `L15.6` | the Rust encoder on candle is checked against these outputs (with B14's serving group) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `WhisperEncoder` | HF `WhisperEncoder` | SDPA and flash attention, gradient checkpointing, a conv stem in fp16 | `transformers/models/whisper/modeling_whisper.py` |
| `WhisperEncoder` | `whisper.cpp` | the same encoder in C with quantized weights and Metal or CUDA kernels | `ggerganov/whisper.cpp`, `whisper_build_graph_encoder` |
| the 30 s window | Distil-Whisper, `faster-whisper` | batched chunks and CTranslate2 kernels; the window itself never changes | `huggingface/distil-whisper` |
| the stem | Conformer encoders | convolution modules inside every block, not only at the input | Gulati et al. (2020) |
