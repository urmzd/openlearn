# Whisper encoder: conv stem, sinusoids, pre-LN blocks

[The audio frontend](01-audio-frontend.md) turns any WAV file into the $80 \times 3000$ picture Whisper expects. This section builds the half of Whisper that looks at that picture. The encoder runs once per 30 s window and produces 1500 vectors that every decoding step in [Whisper decoder and checkpoint weights](03-whisper-decoder-and-weights.md) and [Whisper decoding and long-form transcription](04-whisper-decoding.md) attends to.

Nothing in the encoder is new. The stem is a convolution from [Conv layers and pooling on autograd](../14-vision/01-conv-layers-and-pooling.md), the blocks use [Multi-head attention](../06-transformer-2017/03-multi-head-attention.md), LayerNorm and MLPs come from [Module system and basic layers](../01-foundations/05-module-system-and-layers.md), and the position table is a close cousin of the one in [Positional encodings: sinusoidal, learned](../06-transformer-2017/04-positional-encodings.md). What is new is fitting these parts to a published checkpoint exactly. Three small conventions (concatenated sinusoids, a key projection without bias, pre-LN with a final norm) decide whether 37.8 million loaded weights compute anything meaningful. Get one wrong and the model loads, runs, and transcribes nothing.

Besides the sections above, the encoder uses `conv_out_size` from [Convolution as a linear map](../../mathematics-for-computing/13-signals-and-images/01-convolution-as-a-linear-map.md) and `PCG32` from [Modular arithmetic, hashing, and PCG32 in Python](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md), both in *Mathematics for Computing*, plus `gelu` and `transpose` from [The op library and gradcheck_all](../01-foundations/03-op-library-and-gradcheck.md) and `Tensor` from [Tensor, broadcasting backward, and no_grad](../01-foundations/02-tensor-and-broadcasting-backward.md). It is also worth having [Scaled dot-product attention](../06-transformer-2017/01-scaled-dot-product-attention.md) in mind. The result is checked against Hugging Face's tiny Whisper within $10^{-5}$. The solution is [`encoder.py`](../../../../course/solutions/python/tinyllm/speech/encoder.py), which defines `WhisperConfig` (with `from_hf`), `whisper_sinusoids`, `WhisperAttention` (no key bias), `WhisperEncoderLayer` and `WhisperEncoder`; the contract is [`encoder.pyi`](../../../../course/contracts/py/tinyllm/speech/encoder.pyi).

> **Learning objectives**
> - Derive why two width-3 `Conv1d` layers, the second with stride 2, turn 3000 mel frames into exactly 1500 positions.
> - Compute Whisper's sinusoid table by hand and explain how it differs from the interleaved table of the original Transformer.
> - Explain why Whisper's key projection can drop its bias while the query and value projections cannot.
> - Describe the pre-LN block structure and why it needs a final LayerNorm.
> - Explain why model parity uses the checkpoint's stored position table rather than a freshly computed one.

## The notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $M$ | log-mel input | `float32[N, n_mels, 2A]` |
| $A$ | audio positions (`n_audio_ctx`, 1500) | int |
| $d$, $H$ | width and heads | int |
| $P$ | sinusoid table | `float32[A, d]` |
| $x_\ell$ | hidden state after block $\ell$ | `float32[N, A, d]` |

## The convolutional stem

Speech features are local: a phone lasts tens of milliseconds. So before attention sees anything, two `Conv1d` layers with kernel 3 and padding 1 mix each frame with its neighbours:

$$h_1 = \text{gelu}(\text{conv}_{n_\text{mels} \to d}^{k=3, p=1}(M)), \qquad h_2 = \text{gelu}(\text{conv}_{d \to d}^{k=3, s=2, p=1}(h_1)).$$

With the output size formula $\lfloor (n + 2p - k)/s \rfloor + 1$, the first convolution keeps 3000 frames and the second gives $\lfloor 2999/2 \rfloor + 1 = 1500$. Each of the 1500 positions therefore covers 20 ms instead of 10. Halving the length quarters the cost of attention, which is the whole point of the stride. The GELU is the exact form $x\,\Phi(x)$, not the tanh approximation.

## Sinusoidal positions, Whisper's way

After the stem, a fixed position table is added once. With $C = d$ channels and half $= C/2$, Whisper uses $\text{inc} = \ln(10000)/(\text{half} - 1)$, frequencies $\omega_i = e^{-\text{inc}\cdot i}$ for $i < \text{half}$, and

$$P_{p} = [\sin(p\,\omega_0), \dots, \sin(p\,\omega_{\text{half}-1}),\ \cos(p\,\omega_0), \dots, \cos(p\,\omega_{\text{half}-1})].$$

Compare this with the sinusoidal table in [Positional encodings](../06-transformer-2017/04-positional-encodings.md) and you find two differences. The sines and cosines are two blocks, not interleaved pairs. And the frequency ladder ends exactly at $1/10000$, because the step is divided by half $- 1$, not by half. The table is fixed and never trained.

There is one numerical wrinkle. Computed in float32, an angle near 1499 rad carries the error of one ulp of `exp`, about $3 \cdot 10^{-5}$ after the multiplication, and different libraries' float32 `exp` differ by exactly that ulp. A table computed here and one computed by PyTorch can therefore disagree at the $3 \cdot 10^{-5}$ level in the last rows. Checkpoints store the table they were trained with, so the loader in [the next section](03-whisper-decoder-and-weights.md) checks the stored table against `whisper_sinusoids` (within $10^{-4}$) and then uses the stored one.

## Pre-LN blocks

The rest of the encoder is a stack of blocks, each

$$x \leftarrow x + \text{attn}(\text{LN}_1(x)), \qquad x \leftarrow x + W_2\,\text{gelu}(W_1\,\text{LN}_2(x) + b_1) + b_2,$$

and the encoder ends with one more LayerNorm. Because each branch normalizes only its own input, the residual stream itself is never normalized between blocks, and the final norm is what brings it to a sensible scale before the decoder reads it.

Attention has no mask: every position attends to all 1500. It is ordinary multi-head attention with $Q$, $V$ and output biases but no $K$ bias. The reason is a small piece of algebra worth remembering: a bias on keys adds the same amount to every score of a query, and the softmax cancels any constant shift, so Whisper drops it. Parameter names follow Hugging Face (`self_attn.k_proj.weight`, `self_attn_layer_norm`, `fc1`, `fc2`, `final_layer_norm`, `layer_norm`) so that a checkpoint loads by name.

## Working it by hand

Take a table for $C = 4$ channels. Then half $= 2$, $\text{inc} = \ln(10000)/1 = 9.2103$, and $\omega = (e^0, e^{-9.2103}) = (1, 10^{-4})$. Row $p$ is $[\sin p,\ \sin(10^{-4} p),\ \cos p,\ \cos(10^{-4} p)]$, so row 1 is $[0.8415,\ 0.0001,\ 0.5403,\ 1.0000]$ and row 2 is $[0.9093,\ 0.0002,\ -0.4161,\ 1.0000]$. The interleaved table of the original Transformer would put $\cos p$ second.

The stem's lengths for a 30 s window: conv 1 maps 3000 frames to $(3000 + 2 - 3)/1 + 1 = 3000$, and conv 2 maps them to $\lfloor (3000 + 2 - 3)/2 \rfloor + 1 = 1500$, exactly the table's rows. A 2998-frame input would give 1499 positions and is refused, as Hugging Face refuses it. On a small test model with $A = 8$, 16 frames give 8 positions.

## Designing the API

The configuration defaults to whisper-tiny's sizes, and the encoder maps `[N, n_mels, 2A]` to `[N, A, d]`:

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

A correct encoder reproduces the hand table for 4 channels, and matches Hugging Face's stored tables within $10^{-6}$ for short tables and $10^{-4}$ at 1500 rows (which is why the loader keeps the stored copy). The stem must turn $2A$ frames into $A$ positions and refuse other lengths, mel counts and ranks, since only 30 s windows are valid. The full encoder must match Hugging Face's tiny Whisper on a real clip within $10^{-5}$; that is what the decoder's logits and the later Rust encoder's parity both rest on. It must also handle 128 mels and batches (a batch of two, items independent). Every parameter name and shape must match, with no `k_proj.bias`, so loading by name works. `WhisperConfig.from_hf` must read Hugging Face config keys and refuse variants that would load wrong. Two structural properties make good tests: with zeroed branches the output is `layer_norm(stem + positions)`, which pins the pre-LN structure; and the last frame must reach the first output, which shows there is no mask.

## Building it step by step

The reasoning chain:

1. **The checkpoint fixes the architecture.** The goal is not a good speech encoder but the exact function Hugging Face's `WhisperEncoder` computes with given weights. Every convention (layout of the table, which projections have biases, where the norms are) must match, and names must match Hugging Face's so that a state dict loads by name with no mapping table.
2. **Length is fixed by the table.** The position table has $A = 1500$ rows and is added elementwise, so the stem must output exactly $A$ positions. By the output size formula, only $2A$ frames give $A$, so any other length is refused instead of padded or truncated silently.
3. **Positions are data, not parameters.** The table is a deterministic function of $(A, d)$. Computing it in the constructor is enough for random-init models, and the loader swaps in the checkpoint's stored copy for parity.
4. **The key bias is redundant.** A key bias $b$ adds $q \cdot b$ to every score of query $q$, the same for all keys, and softmax is invariant to adding a constant to all its inputs. So `k_proj` has no bias, and loading must not expect one.
5. **Pre-LN puts the norm inside the branch.** The residual stream $x$ is never normalized between blocks; each branch reads $\text{LN}(x)$ and adds its output back. That leaves the stream unnormalized at the end, which is why a final LayerNorm is needed.
6. **The encoder hears the whole window.** Speech recognition benefits from right context, and the window is processed once, so attention takes no mask.

### The sinusoid table

Angles are computed in float32 exactly as Whisper does; `sin` and `cos` are then evaluated in float64 and rounded once, so the only remaining error is the float32 `exp` ulp discussed above.

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
    # Concatenated blocks [sin | cos], not the original Transformer's interleaved pairs.
    return np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32)
```

For $C = 4$: `half = 2`, `inc = ln(10000)`, `inv = (1, 1e-4)`, and row 1 is `[sin 1, sin 1e-4, cos 1, cos 1e-4]`, the hand table. `channels >= 4` is required because with `half = 1` the step divides by zero.

### Attention without a key bias

Subclassing `MultiHeadAttention` keeps the attention math in one place; only the key projection is replaced.

```python
class WhisperAttention(MultiHeadAttention):
    def __init__(self, d, n_heads, rng=None):
        r = rng if rng is not None else PCG32(0).substream("init")
        super().__init__(d, n_heads, bias=True, rng=r)   # q, v, out with biases
        # Re-assigning keeps k_proj's place in the registration order.
        self.k_proj = Linear(d, d, bias=False, rng=r)
```

### The block

Two pre-LN residual branches. Attention is called as `self_attn(h, h)` with no mask: queries and keys are the same normalized sequence, and every position sees every other.

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

Note the naming trap: Hugging Face's `final_layer_norm` inside a layer is the MLP branch's pre-norm, not the encoder's final norm. The encoder's final norm is the top-level `layer_norm`.

### The encoder

Validate the shape, run the stem, add positions, run the blocks, and normalize once at the end.

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
    x = F.gelu(self.conv2(x))  # [N, d, A]; k=3 stride=2 pad=1
    x = F.transpose(x, 1, 2) + self.positions  # [N, A, d]
    for layer in self.layers:
        x = layer(x)
    return self.layer_norm(x)                    # pre-LN needs a final norm
```

### The config

`WhisperConfig.from_hf` reads Hugging Face's key names (`num_mel_bins`, `max_source_positions`, `d_model`, ...) and refuses variants this code would load wrongly: a non-`whisper` `model_type`, decoder heads or FFN size different from the encoder's (the solution uses one value for both), an activation other than `gelu`, and `scale_embedding`. `validate` requires positive sizes, `d` divisible by `n_heads`, and an even `d` for the two halves of the table. The full solution is [`encoder.py`](../../../../course/solutions/python/tinyllm/speech/encoder.py).

### Why this design

The stem costs $O(N \cdot 2A \cdot 3 d \cdot n_\text{mels} + N \cdot A \cdot 3 d^2)$. Each block costs $O(N A^2 d)$ for attention and $O(N A d\, d_\text{ff})$ for the MLP; the stride-2 convolution halves $A$ and therefore quarters the attention term.

The alternatives all fail in instructive ways. Reusing the interleaved table, or giving `k_proj` a bias "for symmetry", produces a model that loads (or nearly loads) and outputs nonsense. A name-mapping layer between the solution's names and Hugging Face's would work, but it is one more place to get wrong; using Hugging Face's names directly makes loading a dictionary lookup. Accepting any frame count and slicing the table would hide a misconfigured frontend.

Two tradeoffs are accepted. Only full 30 s windows are accepted, as in Whisper itself, and shorter audio is padded by the frontend. Encoder and decoder share head count and FFN width, which holds for all published Whisper sizes.

## What can go wrong

The dangerous mistakes here are the ones that load cleanly. Interleaving sin and cos as in the original Transformer gives a model that loads fine and transcribes nothing; putting the cosines first does the same, more subtly. The solution concatenates `[np.sin(t), np.cos(t)]` with sines first, and divides the frequency step by `half - 1` so that the last frequency really is $10^{-4}$. Forgetting to add the positions at all leaves a model that cannot tell when a sound happened.

> **Warning:** A causal mask copied from the decoder means early frames never hear later ones. The encoder calls `self_attn(h, h)` with no mask.

The remaining mistakes, and how the solution avoids each:

| Mistake | Symptom | Fix in the solution |
|---|---|---|
| second conv without stride 2 | 3000 positions against a 1500-row table | `conv2` is `Conv1d(d, d, 3, stride=2, pad=1)` |
| tanh GELU in the stem | outputs off by about $10^{-3}$ | `F.gelu` is the exact $x\,\Phi(x)$ |
| a bias on `k_proj` | strict loading fails on an unexpected key | `Linear(d, d, bias=False)` |
| post-LN blocks | normalized residual stream, wrong outputs | `x + branch(LN(x))` |
| no final LayerNorm | unnormalized states reach the decoder | returns `self.layer_norm(x)` |
| accepting zero layers | an encoder that is only a stem | `validate` requires every size $\ge 1$ |
| accepting a shorter window | positions silently misaligned | requires exactly $2A$ frames |
| accepting an odd width | sin and cos halves of unequal size | `validate` requires an even `d` |

## Where this leads

[Whisper decoder and checkpoint weights](03-whisper-decoder-and-weights.md) composes this encoder into `Whisper`, runs it once per window, and hands its output to every decoder layer's cross-attention. Later, the engine's Rust encoder on candle is checked against these same outputs.

Looking back, the pieces came from across the book: `Conv1d(n_mels, d, 3, pad=1)` and `Conv1d(d, d, 3, stride=2, pad=1)` from the convolution layers, `conv_out_size` for the length check, `MultiHeadAttention` as the base class, `Linear`, `LayerNorm`, `ModuleList` and `Module` from the module system, the exact `gelu` and `transpose` from the op library, `Tensor` at the input, and `PCG32(0).substream("init")` when no rng is given.

### Further reading

- Radford et al., "Robust speech recognition via large-scale weak supervision" (2022), section 2.2 and figure 1.
- Gulati et al., "Conformer" (2020), for what a speech encoder can add: convolution modules inside every block, not only at the input.
- Xiong et al., "On layer normalization in the transformer architecture" (2020), for pre-LN.
- Hugging Face's `WhisperEncoder` adds SDPA and flash attention, gradient checkpointing and a conv stem in fp16; see `transformers/models/whisper/modeling_whisper.py`.
- `whisper.cpp` implements the same encoder in C with quantized weights and Metal or CUDA kernels; see `ggerganov/whisper.cpp`, `whisper_build_graph_encoder`.
- Distil-Whisper and `faster-whisper` add batched chunks and CTranslate2 kernels; the 30 s window itself never changes. See `huggingface/distil-whisper`.

## Key terms

- **conv stem**: the two width-3 `Conv1d` layers with GELU that precede attention; the second's stride 2 halves the sequence length.
- **concatenated sinusoids**: Whisper's position layout, all sines then all cosines, with frequencies $10000^{-i/(C/2-1)}$.
- **pre-LN block**: a residual block whose branches normalize their input, `x + branch(LN(x))`, leaving the stream itself unnormalized.
- **key bias redundancy**: a bias on keys shifts every score of a query equally, so softmax cancels it.

## Summary

- The stem is two width-3 `Conv1d`s with exact GELU; the second has stride 2, so 3000 mel frames (10 ms each) become 1500 positions (20 ms each).
- Whisper's position table concatenates all the sines and then all the cosines, with frequencies $10000^{-i/(C/2-1)}$; the original Transformer's table interleaves them.
- The blocks are pre-LN with unmasked self-attention and a final LayerNorm, and the key projection has no bias.
- A float32 table at angle 1499 rad is only good to about $3 \cdot 10^{-5}$ across libraries, so model parity uses the checkpoint's stored table.
- Only exactly $2A$ frames are accepted; anything else is a frontend bug made visible.

## Review questions

1. Why can Whisper drop the key bias but not the query or value bias?
   <details><summary>Answer</summary>

   A key bias adds $q \cdot b$ to all scores of a query equally, and softmax cancels a constant shift. A query bias adds $b_q \cdot k_j$, which differs per key, and a value bias changes the output vectors; neither cancels.

   </details>

2. Why does the frequency step divide by `half - 1`?
   <details><summary>Answer</summary>

   So that the last frequency, $i = \text{half} - 1$, is exactly $e^{-\ln 10000} = 10^{-4}$. Dividing by `half` (the original Transformer's convention) stops one step short, and every column but the first has a different frequency from the one the checkpoint was trained with.

   </details>

3. What would a zero-initialized attention and MLP branch produce, and why does that test the structure?
   <details><summary>Answer</summary>

   With both branches outputting zero, each block is the identity on the residual stream, so the output is `layer_norm(stem + positions)`. A post-LN block would normalize the stream at every layer and a missing final norm would leave it unnormalized, so either mistake changes this output.

   </details>

4. Why refuse 2998 frames instead of letting the stem produce 1499 positions?
   <details><summary>Answer</summary>

   The positions are added row by row from a 1500-row table, and the checkpoint was trained only on full windows. Accepting a shorter input would need a slice of the table, silently changing what each position means to the model; refusing makes a frontend bug visible.

   </details>

5. Compute row 2 of the 4-channel table.
   <details><summary>Answer</summary>

   With $\omega = (1, 10^{-4})$, row 2 is $[\sin 2, \sin(2 \cdot 10^{-4}), \cos 2, \cos(2 \cdot 10^{-4})] = [0.9093, 0.0002, -0.4161, 1.0000]$.

   </details>
