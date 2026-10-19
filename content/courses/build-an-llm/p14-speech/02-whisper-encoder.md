# Whisper encoder: conv stem, sinusoids, pre-LN blocks

## Overview

| | |
|---|---|
| **Module** | `L14.2` · build · Python · Pass 12 · 2 to 3 h |
| **Solution** | [`encoder.py`](../../../../course/solutions/python/tinyllm/speech/encoder.py): `WhisperConfig` (with `from_hf`), `whisper_sinusoids`, `WhisperAttention` (no key bias), `WhisperEncoderLayer`, `WhisperEncoder` |
| **Contract** | [`course/contracts/py/tinyllm/speech/encoder.pyi`](../../../../course/contracts/py/tinyllm/speech/encoder.pyi) |
| **Needs** | `L13.1` `Conv1d` · `M12.1` `conv_out_size` · `L5.3` `MultiHeadAttention` · `L0.4` `Linear`, `LayerNorm`, `ModuleList`, `Module` · `L0.2` `gelu`, `transpose` · `L0.1` `Tensor` · `M06.3` `PCG32`. Reading: `L5.1` attention, `L5.4` sinusoids, `L14.1` the features it reads |
| **Used by** | `L14.3` (Whisper composes it; tiny Whisper logits match HF within $10^{-5}$) · later `L15.6` (the Rust encoder is proven against the same reference outputs) |
| **Optional depth** | Radford et al., "Robust speech recognition via large-scale weak supervision" (2022), section 2.2 and figure 1; Gulati et al., "Conformer" (2020), for what a speech encoder can add; Xiong et al., "On layer normalization in the transformer architecture" (2020), for pre-LN |

## Key Takeaways

- The stem is two width-3 `Conv1d`s with GELU; the second has stride 2, so 3000 mel frames (10 ms each) become 1500 positions (20 ms each).
- Whisper's position table concatenates all the sines and then all the cosines, with frequencies $10000^{-i/(C/2-1)}$; L5.4's table interleaves them.
- The blocks are pre-LN with unmasked self-attention and a final LayerNorm; the key projection has no bias.
- A float32 table at angle 1499 rad is only good to about $3 \cdot 10^{-5}$ across libraries, so model parity uses the checkpoint's stored table.

## How to read this chapter

Sections 1 to 3 derive the stem, the position table and the block structure and work the small cases by hand. Section 4 fixes the interface and the cases a correct encoder must handle. Section 5 walks through the solution class by class and explains why each convention is the one the checkpoint needs. Try to predict each line before reading it.

---

## 1. Why now

`L14.1` turns any WAV file into the $80 \times 3000$ picture Whisper expects. This module is the half of Whisper that looks at the picture: the encoder runs once per 30 s window and produces 1500 vectors that every decoding step (`L14.3`, `L14.4`) attends to. Nothing in it is new: a convolution from `L13.1`, attention from `L5.3`, LayerNorm and MLPs from Part 0, sinusoids like `L5.4`'s. What is new is fitting them to a published checkpoint exactly, where three small conventions (concatenated sinusoids, a key projection without bias, pre-LN with a final norm) decide whether 37.8 million loaded weights compute anything meaningful. The engine's candle encoder (`L15.6`) is later proven against the same reference outputs.

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

The table for $C = 4$ channels: half $= 2$, $\text{inc} = \ln(10000)/1 = 9.2103$, $\omega = (e^0, e^{-9.2103}) = (1, 10^{-4})$. Row $p$ is $[\sin p,\ \sin(10^{-4} p),\ \cos p,\ \cos(10^{-4} p)]$, so row 1 is $[0.8415,\ 0.0001,\ 0.5403,\ 1.0000]$ and row 2 is $[0.9093,\ 0.0002,\ -0.4161,\ 1.0000]$. `L5.4`'s interleaved table for the same size would put $\cos p$ second.

The stem's lengths for a 30 s window: conv 1 maps 3000 frames to $(3000 + 2 - 3)/1 + 1 = 3000$; conv 2 maps them to $\lfloor (3000 + 2 - 3)/2 \rfloor + 1 = 1500$, exactly the table's rows. A 2998-frame input would give 1499 positions and is refused, as HF refuses it (on a small model with $A = 8$: 16 frames give 8).

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Section 3's 4-channel table | unit, smoke | the hand values | the layout of positions |
| HF's stored tables | golden | $10^{-6}$ short, $10^{-4}$ at 1500 rows | why the loader keeps the stored copy |
| Stem lengths | unit | $2A$ frames give $A$ positions; other lengths, mel counts, ranks refused | 30 s windows only |
| Full encoder vs HF tiny Whisper | golden | a real clip within $10^{-5}$ of HF | `L14.3`'s logits, `L15.6`'s parity |
| 128 mels and batches | golden | a batch of two, items independent | other checkpoints, batching |
| Parameter names | unit | every name and shape; no `k_proj.bias` | loading by name |
| `WhisperConfig.from_hf` | unit | HF config keys; unsupported variants refused | refusing what would load wrong |
| Zeroed branches | property | leave `layer_norm(stem + positions)` | the pre-LN block structure |
| Last frame vs first output | property | the last frame reaches the first output | no mask in the encoder |

## 5. The solution, step by step

### From first principles

1. **The checkpoint fixes the architecture.** The goal is not a good speech encoder but the exact function HF's `WhisperEncoder` computes with given weights. Every convention (layout of the table, which projections have biases, where the norms are) must match, and names must match HF's so a state dict loads by name with no mapping table.
2. **Length is fixed by the table.** The position table has $A = 1500$ rows and is added elementwise, so the stem must output exactly $A$ positions. With section 2.1's formula, only $2A$ frames give $A$, so any other length is refused instead of padded or truncated silently.
3. **Positions are data, not parameters.** The table is a deterministic function of $(A, d)$; computing it in the constructor is enough for random-init models, and `L14.3` swaps in the checkpoint's stored copy for parity (section 2.2).
4. **The key bias is redundant.** A key bias $b$ adds $q \cdot b$ to every score of query $q$, the same for all keys, and softmax is invariant to adding a constant to all its inputs. So `k_proj` has no bias, and loading must not expect one.
5. **Pre-LN puts the norm inside the branch.** The residual stream $x$ is never normalized between blocks; each branch reads $\text{LN}(x)$ and adds its output back. That leaves the stream unnormalized at the end, which is why a final LayerNorm is needed.
6. **The encoder hears the whole window.** Speech recognition benefits from right context, and the window is processed once, so attention takes no mask.

### The code, annotated

**The sinusoid table.** Angles are computed in float32 exactly as Whisper does; `sin` and `cos` are then evaluated in float64 and rounded once, so the only remaining error is the float32 `exp` ulp discussed in section 2.2.

```python
def whisper_sinusoids(length, channels, max_timescale=10000.0):
    if channels % 2 or channels < 4:          # two equal halves; half - 1 must be > 0
        raise ValueError(...)
    # ... reject length < 1
    half = channels // 2
    # half - 1, not half: the last frequency is exactly 1 / max_timescale.
    inc = np.float32(math.log(max_timescale) / (half - 1))
    inv = np.exp(-inc * np.arange(half, dtype=np.float32))
    t = (np.arange(length, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float64)
    # Concatenated blocks [sin | cos], not L5.4's interleaved pairs.
    return np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32)
```

For $C = 4$: `half = 2`, `inc = ln(10000)`, `inv = (1, 1e-4)`, and row 1 is `[sin 1, sin 1e-4, cos 1, cos 1e-4]`, section 3's table. `channels >= 4` is required because with `half = 1` the step divides by zero.

**Attention without a key bias.** Subclassing L5.3 keeps the attention math in one place; only the key projection is replaced.

```python
class WhisperAttention(MultiHeadAttention):
    def __init__(self, d, n_heads, rng=None):
        r = rng if rng is not None else PCG32(0).substream("init")
        super().__init__(d, n_heads, bias=True, rng=r)   # q, v, out with biases
        # Re-assigning keeps k_proj's place in the registration order.
        self.k_proj = Linear(d, d, bias=False, rng=r)
```

**The block.** Two pre-LN residual branches. Attention is called as `self_attn(h, h)` with no mask: queries and keys are the same normalized sequence, and every position sees every other.

```python
class WhisperEncoderLayer(Module):
    def __init__(self, cfg, rng=None):
        # ... names are HF's, so a checkpoint loads by name
        self.self_attn = WhisperAttention(cfg.d, cfg.n_heads, rng=r)
        self.self_attn_layer_norm = LayerNorm(cfg.d)
        self.fc1 = Linear(cfg.d, cfg.d_ff, rng=r)
        self.fc2 = Linear(cfg.d_ff, cfg.d, rng=r)
        self.final_layer_norm = LayerNorm(cfg.d)    # the MLP's pre-norm, despite the name

    def forward(self, x):
        h = self.self_attn_layer_norm(x)
        x = x + self.self_attn(h, h)                # unmasked self-attention
        h = self.final_layer_norm(x)
        return x + self.fc2(F.gelu(self.fc1(h)))    # exact GELU
```

**The encoder.** Validate the shape, run the stem, add positions, run the blocks, and normalize once at the end.

```python
def forward(self, mel):
    x = mel if isinstance(mel, Tensor) else Tensor(np.asarray(mel, dtype=np.float32))
    cfg = self.cfg
    if x.ndim != 3 or x.shape[1] != cfg.n_mels:
        raise ValueError(...)
    frames = x.shape[2]
    a = conv_out_size(conv_out_size(frames, 3, 1, 1), 3, 2, 1)  # for the message
    if frames != 2 * cfg.n_audio_ctx:            # only 2A frames fill the A-row table
        raise ValueError(...)
    x = F.gelu(self.conv1(x))                    # [N, d, 2A], k=3 pad=1
    x = F.gelu(self.conv2(x))  # [N, d, A]       # k=3 stride=2 pad=1
    x = F.transpose(x, 1, 2) + self.positions  # [N, A, d]
    for layer in self.layers:
        x = layer(x)
    return self.layer_norm(x)                    # pre-LN needs a final norm
```

**The config.** `WhisperConfig.from_hf` reads HF's key names (`num_mel_bins`, `max_source_positions`, `d_model`, ...) and refuses variants this code would load wrongly: a non-`whisper` `model_type`, decoder heads or FFN size different from the encoder's (the solution uses one value for both), an activation other than `gelu`, and `scale_embedding`. `validate` requires positive sizes, `d` divisible by `n_heads`, and an even `d` for the two halves of the table.

Full solution: [`encoder.py`](../../../../course/solutions/python/tinyllm/speech/encoder.py).

### Why this solution

- **Cost.** The stem costs $O(N \cdot 2A \cdot 3 d \cdot n_\text{mels} + N \cdot A \cdot 3 d^2)$. Each block costs $O(N A^2 d)$ for attention and $O(N A d\, d_\text{ff})$ for the MLP; the stride-2 conv halves $A$ and therefore quarters the attention term.
- **Alternatives.** Reusing L5.4's interleaved table, or giving `k_proj` a bias "for symmetry", would produce a model that loads (or nearly loads) and outputs nonsense. A name-mapping layer between the solution's names and HF's would work but is one more place to get wrong; using HF's names directly makes loading a dictionary lookup. Accepting any frame count and slicing the table would hide a misconfigured frontend.
- **Tradeoffs accepted.** Only full 30 s windows are accepted, as in Whisper itself; shorter audio is padded by `L14.1`. Encoder and decoder share head count and FFN width, which holds for all published Whisper sizes.

### Explain it back

<details><summary>Why can Whisper drop the key bias but not the query or value bias?</summary>

A key bias adds $q \cdot b$ to all scores of a query equally, and softmax cancels a constant shift. A query bias adds $b_q \cdot k_j$, which differs per key, and a value bias changes the output vectors; neither cancels.

</details>

<details><summary>Why does the frequency step divide by <code>half - 1</code>?</summary>

So that the last frequency, $i = \text{half} - 1$, is exactly $e^{-\ln 10000} = 10^{-4}$. Dividing by `half` (L5.4's convention) stops one step short, and every column but the first has a different frequency from the one the checkpoint was trained with.

</details>

<details><summary>What would a zero-initialized attention and MLP branch produce, and why does that test the structure?</summary>

With both branches outputting zero, each block is the identity on the residual stream, so the output is `layer_norm(stem + positions)`. A post-LN block would normalize the stream at every layer and a missing final norm would leave it unnormalized, so either mistake changes this output.

</details>

<details><summary>Why refuse 2998 frames instead of letting the stem produce 1499 positions?</summary>

The positions are added row by row from a 1500-row table, and the checkpoint was trained only on full windows. Accepting a shorter input would need a slice of the table, silently changing what each position means to the model; refusing makes a frontend bug visible.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. sin and cos interleaved as in `L5.4` | loads fine, transcribes nothing | `np.concatenate([np.sin(t), np.cos(t)], axis=1)` |
| 2. cosines first | the same, more subtly | sines are the first block |
| 3. the second conv without stride 2 | 3000 positions against a 1500-row table | `conv2` is `Conv1d(d, d, 3, stride=2, pad=1)` |
| 4. tanh GELU in the stem | outputs off by about $10^{-3}$ | `F.gelu` is the exact $x\,\Phi(x)$ |
| 5. a bias on `k_proj` | strict loading fails: an unexpected key | `WhisperAttention` replaces `k_proj` with `Linear(d, d, bias=False)` |
| 6. post-LN blocks | normalized residual stream, wrong outputs | each branch normalizes its input, `x + branch(LN(x))` |
| 7. no final LayerNorm | unnormalized states reach the decoder | `forward` returns `self.layer_norm(x)` |
| 8. a causal mask copied from the decoder | early frames never hear later ones | `self_attn(h, h)` with no mask |
| 9. positions never added | the model cannot tell when a sound happened | `transpose(x, 1, 2) + self.positions` after the stem |
| 10. dividing the frequency step by $C/2$ | the last frequency is not $10^{-4}$ | divides by `half - 1` |
| accepting zero layers | an encoder that is only a stem | `validate` requires every size $\ge 1$ |
| accepting a shorter window | positions silently misaligned | `forward` requires exactly $2A$ frames |
| accepting an odd width | sin and cos halves of unequal size | `validate` requires an even `d` |

## 7. Where it's used next

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

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `WhisperEncoder` | HF `WhisperEncoder` | SDPA and flash attention, gradient checkpointing, a conv stem in fp16 | `transformers/models/whisper/modeling_whisper.py` |
| `WhisperEncoder` | `whisper.cpp` | the same encoder in C with quantized weights and Metal or CUDA kernels | `ggerganov/whisper.cpp`, `whisper_build_graph_encoder` |
| the 30 s window | Distil-Whisper, `faster-whisper` | batched chunks and CTranslate2 kernels; the window itself never changes | `huggingface/distil-whisper` |
| the stem | Conformer encoders | convolution modules inside every block, not only at the input | Gulati et al. (2020) |
