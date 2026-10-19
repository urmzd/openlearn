# Whisper decoder and checkpoint weights

## Overview

| | |
|---|---|
| **Module** | `L14.3` · build · Python · Pass 12 · 2 to 3 h |
| **Solution** | [`decoder.py`](../../../../course/solutions/python/tinyllm/speech/decoder.py): `CrossKV`, `WhisperDecoderLayer`, `WhisperDecoder`, `Whisper` (`encode`, `encode_wav`, `new_cache`, `param_count`, `from_pretrained`); [`hf_whisper.py`](../../../../course/solutions/python/tinyllm/io/hf_whisper.py): `whisper_state_from_hf`, `read_whisper` |
| **Contract** | [`decoder.pyi`](../../../../course/contracts/py/tinyllm/speech/decoder.pyi), [`hf_whisper.pyi`](../../../../course/contracts/py/tinyllm/io/hf_whisper.pyi) |
| **Needs** | `L14.1`; `L14.2`, `L8.2`, `L0.6`, `L0.4`, `L0.2`, `L0.1`, `L5.1`, `L5.2`, `L5.4`, `M06.3`. Reading: `L7.9` |
| **Used by** | `ethics.08`, `L14.4`. With this module a published Whisper checkpoint produces logits that match HF |
| **Optional depth** | Compare token-at-a-time and chunked decoding, including a batch with different prompts. |

## Key Takeaways

- Self-attention sees only the decoder prefix; cross-attention sees the complete encoded audio window.
- Cross keys and values are projected once per window. Self keys and values grow at each decoding step.
- Output logits use the token embedding matrix transposed. A separate untied output weight is an invalid checkpoint.

## How to read this chapter

Sections 1 to 3 describe the decoder and the loader and trace a two-step cached decode by hand. Section 4 lists the cases a correct implementation must handle. Section 5 walks through the solution: the decoder layer, the cache discipline in `WhisperDecoder.forward`, then the strict loader. Try to predict each validation before reading it.

## 1. Why now

L14.2 turns audio features into vectors, but cannot generate a transcript. The decoder compares its text prefix with those vectors and predicts the next token. This module supplies logits and checkpoint loading. L14.4 will supply language, timestamp and search policies.

## 2. Principles

At absolute decoder position `o + t`, add the learned positional row to the token embedding. Each block applies pre-normalized causal self-attention, pre-normalized cross-attention, and a pre-normalized GELU feed-forward network, with a residual addition after each operation. Apply the final layer norm, then multiply by the transpose of the token embedding matrix.

Cross keys and values depend on the audio window and the decoder layer, so compute them once and retain one pair per layer. Self-attention caches retain the prefix. Read the common committed cache length before any layer appends; every layer uses that same position offset and causal mask. Validate every cache shape before the first append so a rejected request cannot partially mutate the cache.

The loader removes `model.` from parameter names, verifies an optional tied output projection, and separates the encoder's frozen sinusoid table from trainable parameters. Strict state loading rejects missing, unexpected or wrongly shaped tensors. For sharded files, each tensor must occur in the file named by the index, exactly once.

## 3. Worked example

The same calculation, run on a real checkpoint, reproduces HF's logits.

For a two-token prefix `[sot, language]`, a fresh cache has offset zero. The self-attention permission matrix is `[[True, False], [True, True]]`. Both tokens may attend to every encoded audio position. Every decoder layer appends two self-key and self-value rows, making the committed length two.

The next one-token call uses positional row two and a `1 x 3` all-true self-attention mask. It reuses the unchanged cross keys and values. Its last logits must agree with recomputing all three prefix tokens in one uncached call. If the checkpoint stores an output projection, each element must equal the corresponding token-embedding element before the redundant tensor is discarded.

## 4. The interface

### What a correct solution must handle

| Case | Kind of check | Contract |
|---|---|---|
| Logits and parameter count vs HF | golden | Independent HF logits within absolute tolerance `1e-5`; parameter count includes the frozen encoder position table. |
| Cached chunks vs one full call | property | Chunked and full decoding agree; cross keys and values remain unchanged. |
| Causality and 1-D ids | property | A changed suffix cannot affect prefix logits; one-dimensional ids preserve the documented output shape. |
| Bad tokens and positions | boundary | Invalid token ids, ranks and positions leave the cache untouched. |
| Cross-KV and cache shapes | boundary | Batch, layer, head and feature widths are checked before decoding. |
| Half-precision position table | boundary | Rounded fixed positions retain checkpoint values; corrupted entries fail. |
| Untied weights and wrong sinusoids | boundary | Incompatible weight names, positions and output ties fail explicitly. |
| Sharded checkpoints | conformance | Sharded and unsharded states agree; paths cannot escape the checkpoint directory. |
| Missing tensors and wrong shapes | boundary | Incomplete checkpoints cannot leave random parameters in the model. |
| WAV entry point | conformance | WAV channel mixing, padding and mel features produce the same cross-attention keys and logits as the explicit pipeline. |

Published checkpoints may store the fixed sinusoid table in float16. Such a table is retained exactly, while a corrupted entry is rejected. The loader permits 5e-4 absolute deviation only for tables exactly representable in float16; other tables retain the 1e-4 bound.

See the [decoder contract](../../../../course/contracts/py/tinyllm/speech/decoder.pyi) and the [loader contract](../../../../course/contracts/py/tinyllm/io/hf_whisper.pyi).

## 5. The solution, step by step

### From first principles

1. **Two kinds of context, two caches.** Self-attention keys and values depend on the text prefix, which grows by one or more tokens per call, so they live in an append-only `KVCache` (L8.2). Cross-attention keys and values depend only on the encoder output and the layer's weights, so they are computed once per 30 s window (`CrossKV`) and passed unchanged to every call.
2. **One offset for all layers.** Position $o + t$ must mean the same token in every layer, so the offset is read once, before any layer appends. If each layer read its own length, layer 1 would see layer 0's append and use positions one chunk ahead.
3. **Validate before mutating.** `cache.update` appends. If a check fails halfway through the layer loop, earlier layers would hold an extra chunk and the cache would be inconsistent. So every check (ids, ranks, cross-KV shapes, cache shape, capacity, equal fills) runs before the loop.
4. **Causality comes from the mask.** For a fresh call, `causal_mask(t, o + t, q_offset=o)`; with a cache, `cache.mask(t)` builds the same `t x (o + t)` lower-triangular-with-offset matrix. In section 3's second step that is `1 x 3`, all true.
5. **Tied output.** Whisper's logits are $\text{LN}(x)\,E^\top$ with $E$ the token embedding. A checkpoint may also store `proj_out.weight`; it is redundant, and if it differs from $E$ the checkpoint is not one this model can represent, so it is an error rather than a second weight.
6. **Strict loading.** A missing tensor would leave a random parameter that still produces plausible-looking logits. So the loader maps every name, rejects unknown ones, and `load_state_dict(strict=True)` rejects missing or wrongly shaped ones.

### The code, annotated

**The decoder layer.** Three pre-LN residual branches. Self-attention is written out (instead of calling `MultiHeadAttention.forward`) so that the new keys and values can go through the cache; cross-attention uses precomputed keys and values and no mask.

```python
def forward(self, x, xk, xv, mask, cache=None, layer=0):
    h = self.self_attn_layer_norm(x)
    a = self.self_attn
    q, k, v = [a.split_heads(p(h)) for p in (a.q_proj, a.k_proj, a.v_proj)]
    if cache is not None:
        # Append this chunk and attend over the whole committed prefix.
        keys, values = cache.update(layer, k.data, v.data)
        k, v = Tensor(keys), Tensor(values)
    context, _ = scaled_dot_product_attention(q, k, v, mask=mask)   # causal
    x = x + a.out_proj(a.merge_heads(context))
    a = self.encoder_attn
    q = a.split_heads(a.q_proj(self.encoder_attn_layer_norm(x)))
    # Cross-attention: the whole audio window, no mask, keys/values precomputed.
    context, _ = scaled_dot_product_attention(q, Tensor(xk), Tensor(xv))
    x = x + a.out_proj(a.merge_heads(context))
    return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
```

**Cross keys and values, once per window.** Each layer's `encoder_attn` projects the audio with its own `k_proj` (no bias, from L14.2's `WhisperAttention`) and `v_proj`. The copies detach the arrays from the autograd graph so later calls cannot alter them.

```python
def cross_kv(self, audio):
    # ... wrap arrays as Tensor; require [N, A, d] with A >= 1
    keys, values = [], []
    for layer in self.layers:
        a = layer.encoder_attn
        keys.append(a.split_heads(a.k_proj(audio)).data.astype(np.float32, copy=True))
        values.append(a.split_heads(a.v_proj(audio)).data.astype(np.float32, copy=True))
    return CrossKV(keys, values)
```

**The decoder forward: all checks, then one offset, then the loop.**

```python
ids = np.asarray(ids)
one = ids.ndim == 1
if one:
    ids = ids[None, :]                      # [T] is a batch of one
# ... require a nonempty integer [N, T] array with ids in [0, vocab)
# ... require len(xkv.k) == dec_layers and each k, v of shape [N, heads, A, d_head]
offset = 0 if cache is None else cache.seq_len()   # read once, before any append
if offset + t > cfg.n_text_ctx:             # learned positions stop at n_text_ctx
    raise ValueError("decoder positions exceed n_text_ctx")
if cache is not None:
    # ... cache (layers, batch, heads, d_head) must match the decoder
    if offset + t > cache.max_len or any(
        cache.seq_len(i) != offset for i in range(cfg.dec_layers)
    ):                                       # full, or layers out of step
        raise ValueError("self-KV cache is full or has inconsistent layer lengths")
mask = (
    causal_mask(t, offset + t, q_offset=offset)
    if cache is None
    else cache.mask(t)
)
x = self.embed_positions(self.embed_tokens(ids), offset)   # rows offset .. offset+t-1
for i, layer in enumerate(self.layers):
    x = layer(x, xkv.k[i], xkv.v[i], mask, cache, i)
logits = F.matmul(
    self.layer_norm(x), F.transpose(self.embed_tokens.weight, 0, 1)   # tied output
)
return F.reshape(logits, (t, cfg.vocab)) if one else logits
```

**The model.** `Whisper` composes L14.2's encoder with this decoder. `encode_wav` runs the L14.1 frontend with the model's mel count and window length (`2 * n_audio_ctx * hop_length` samples), so the WAV path and the explicit path produce the same features. `param_count` adds `n_audio_ctx * d` because the encoder position table is part of the published parameter count but is not a trainable `Parameter` here.

```python
def encode_wav(self, data):
    cfg = FeatureConfig(feature_size=self.config.n_mels)
    samples = to_mono_16k(read_wav(data), cfg)
    window = pad_or_trim(samples, 2 * self.config.n_audio_ctx * cfg.hop_length)
    mel = whisper_log_mel(window, cfg)
    return self.encode(Tensor(mel[None].astype(np.float32)))

@classmethod
def from_pretrained(cls, dir):
    cfg, state, positions = read_whisper(dir)
    model = cls(cfg)
    model.load_state_dict(state, strict=True)   # missing/unexpected/shape errors raise
    if positions is not None:
        model.encoder.positions = positions.copy()   # the checkpoint's own table
    return model.eval()
```

Full solution: [`decoder.py`](../../../../course/solutions/python/tinyllm/speech/decoder.py).

**Mapping HF names.** Every tensor goes to exactly one of three places: verified and dropped (`proj_out.weight`), kept aside as the fixed table, or stripped of `model.` into the state dict. Anything else is an error.

```python
for name, value in tensors.items():
    value = np.asarray(value, dtype=np.float32)
    if name == "proj_out.weight":
        embedding = tensors.get("model.decoder.embed_tokens.weight")
        if embedding is None or not np.array_equal(value, np.asarray(embedding, dtype=np.float32)):
            raise ValueError("Whisper output projection is not tied to token embeddings")
    elif name == "model.encoder.embed_positions.weight":
        expected = whisper_sinusoids(cfg.n_audio_ctx, cfg.d)
        # Published Whisper checkpoints can store the fixed table after
        # float16 rounding. At unit magnitude its rounding error is up to
        # 2**-12, plus float32 phase differences across math libraries.
        quantized = np.array_equal(value, value.astype(np.float16).astype(np.float32))
        tolerance = 5e-4 if quantized else 1e-4
        if (value.shape != expected.shape or not np.all(np.isfinite(value))
                or np.max(np.abs(value - expected)) > tolerance):
            raise ValueError("stored encoder positions are not Whisper sinusoids")
        positions = value.copy()
    elif name.startswith("model."):
        state[name[len("model."):]] = value.copy()
    else:
        raise ValueError(f"unexpected Whisper tensor name: {name}")
```

**Sharded checkpoints.** The index's `weight_map` names a file per tensor. The loader reads each named file once, refuses paths that resolve outside the checkpoint directory, and checks that every tensor is in the file the index says, exactly once, and that the index names nothing missing.

```python
root = Path(dir).resolve()
# ...
for name in names:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Whisper shard path escapes the checkpoint directory")
    shard, _ = load_safetensors(str(path))
    for key, value in shard.items():
        if key in tensors or (weight_map is not None and weight_map.get(key) != name):
            raise ValueError(f"duplicate or misindexed Whisper tensor: {key}")
        tensors[key] = value
if weight_map is not None and set(tensors) != set(weight_map):
    raise ValueError("Whisper shard index names missing tensors")
```

Full solution: [`hf_whisper.py`](../../../../course/solutions/python/tinyllm/io/hf_whisper.py).

### Why this solution

- **Cost.** With a cache, decoding token $o$ costs $O(o\,d)$ per layer for self-attention and $O(A\,d)$ for cross-attention, instead of $O(o^2 d)$ for recomputing the prefix. Cross keys and values cost $O(L\,A\,d^2)$ once per window rather than once per token; for whisper-tiny that is 1500 positions times 4 layers saved on every step.
- **Alternatives.** Recomputing cross keys and values per step is correct but repeats the most expensive projection hundreds of times per window. Loading leniently (`strict=False`, ignoring `proj_out.weight`) accepts fine-tuned checkpoints with an untied head and silently gives them the wrong output layer. Trusting the stored position table without checking it would accept a checkpoint from a different architecture.
- **Tradeoffs accepted.** The tied-output requirement rejects some community fine-tunes; that is deliberate, since this model has no slot for a separate head. The loader converts everything to float32, doubling memory for float16 checkpoints.

### Explain it back

<details><summary>Why is <code>offset</code> read once before the layer loop instead of from each layer's cache?</summary>

Layer 0's `cache.update` appends before layer 1 runs. Reading inside the loop would give layer 1 an offset one chunk larger, so its mask and its notion of the prefix would disagree with layer 0's. One read of the committed length makes every layer use the same positions.

</details>

<details><summary>Why must all checks happen before the first <code>cache.update</code>?</summary>

The update mutates the cache. A failure after some layers appended would leave layers with different fills, which the next call rejects as inconsistent. Checking first means a rejected request leaves the cache exactly as it was.

</details>

<details><summary>Why is a different <code>proj_out.weight</code> an error instead of a weight to load?</summary>

The model computes logits as $\text{LN}(x)E^\top$; there is no separate output matrix to load it into. Ignoring it would produce plausible but wrong scores; failing tells the user the checkpoint is not a standard Whisper.

</details>

<details><summary>Why does <code>param_count</code> add <code>n_audio_ctx * d</code>?</summary>

The encoder's sinusoid table is stored in checkpoints and counted in published sizes (about 37.8 million for tiny), but here it is a fixed array, not a trainable parameter, so `parameters()` does not include it.

</details>

## 6. Pitfalls

| Mistake | Symptom | How the solution avoids it |
|---|---|---|
| Reset position offset at every incremental call | Full and cached logits disagree | `offset = cache.seq_len()` and `embed_positions(..., offset)` |
| Omit the causal mask | Future tokens change earlier predictions | passes `causal_mask` or `cache.mask(t)` to self-attention |
| Load an untied output matrix silently | Plausible but incorrect transcript scores | `whisper_state_from_hf` requires `proj_out.weight` to equal the embedding |
| Advance the committed length once per layer | Later layers use different positions | reads the offset once before the loop and requires equal fills |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `L14.2` | Encoder output and Whisper attention projections. |
| Back | `L8.2` | Self-attention cache and offset masks. |
| Back | `L0.6` | Safetensors reader. |
| Back | `L0.4` | Registered modules, embeddings, normalization and strict state loading. |
| Back | `L0.2` | Differentiable tensor operations. |
| Back | `L0.1` | Tensor storage and shapes. |
| Back | `L5.1` | Scaled dot-product attention. |
| Back | `L5.2` | Causal mask construction. |
| Back | `L5.4` | Learned positional embeddings. |
| Back | `M06.3` | Reproducible parameter initialization. |
| Back | `L7.9` | Download the checkpoint before calling the local loader. |
| Back | `L14.1` | `encode_wav` decodes, mixes channels, resamples and computes model-sized log-mel features. |
| Forward | `ethics.08` | The consent and model-card evidence practice calls this implementation. |
| Forward | `L14.4` | Transcription policy consumes these logits and reuses cross-KV per window. |

## Going further

| Resource | Exercise |
|---|---|
| [Whisper contract](../../../../course/contracts/py/tinyllm/speech/decoder.pyi) | Derive the parameter count, then compare one-token and multi-token cache updates. |
