# Whisper decoder and checkpoint weights

[The encoder](02-whisper-encoder.md) turns audio features into 1500 vectors, but it cannot produce a single word. Generating a transcript needs the other half of Whisper: a decoder that compares its text prefix with those vectors and predicts the next token. This section builds that decoder and the loader that fills it with a published checkpoint. With both in place, a real Whisper checkpoint produces logits that match Hugging Face's. Choosing languages, timestamps and search strategies is policy, and it waits for [Whisper decoding and long-form transcription](04-whisper-decoding.md).

Two ideas carry the section. The first is that a Whisper decoder sees two kinds of context, the growing text prefix and the fixed audio window, and each wants its own cache discipline. The second is that a checkpoint is a claim about an architecture, and the loader's job is to verify that claim strictly rather than load whatever fits.

The decoder builds directly on [The KV cache, incremental decoding, and generate](../21-inference/02-kv-cache-and-generate.md) for its self-attention cache and offset masks, on [Safetensors for every dtype, atomic checkpoints, and the token stream](../06-numerics-and-autograd/07-safetensors-checkpoints-and-token-streams.md) for reading weights, and on [Module system and basic layers](../06-numerics-and-autograd/05-module-system-and-layers.md) for registered modules, embeddings, normalization and strict state loading. Attention comes from [Scaled dot-product attention](../11-transformer/01-scaled-dot-product-attention.md), causal masks from [Masks: causal, padding, sliding window, additive](../11-transformer/02-masks.md), and learned positional embeddings from [Positional encodings](../11-transformer/04-positional-encodings.md); tensors and differentiable operations come from [Tensor, broadcasting backward, and no_grad](../06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md) and [The op library and gradcheck_all](../06-numerics-and-autograd/03-op-library-and-gradcheck.md), and reproducible initialization from PCG32 in *Mathematics* ([Modular arithmetic, hashing, and PCG32 in Python](../../mathematics/12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md)). To download a checkpoint before calling the local loader, use the downloader from [The Llama-Family Model: Config, Weight Loading, and Download](../13-decoder-block/09-llama-family-model.md). The solution is [`decoder.py`](../../../../course/solutions/python/tinyllm/speech/decoder.py) (`CrossKV`, `WhisperDecoderLayer`, `WhisperDecoder`, and `Whisper` with `encode`, `encode_wav`, `new_cache`, `param_count`, `from_pretrained`) and [`hf_whisper.py`](../../../../course/solutions/python/tinyllm/io/hf_whisper.py) (`whisper_state_from_hf`, `read_whisper`); the contracts are the [decoder contract](../../../../course/contracts/py/tinyllm/speech/decoder.pyi) and the [loader contract](../../../../course/contracts/py/tinyllm/io/hf_whisper.pyi).

> **Learning objectives**
> - Describe a Whisper decoder block: causal self-attention over the prefix, unmasked cross-attention over the audio window, and a GELU MLP, all pre-normalized.
> - Explain why cross keys and values are projected once per window while self keys and values grow at each step.
> - Explain why the position offset is read once before the layer loop, and why every check runs before the first cache append.
> - Load a Hugging Face checkpoint strictly, including a tied output projection, a fixed sinusoid table, and sharded files.

## Self-attention, cross-attention, and the tied output

At absolute decoder position $o + t$, the decoder adds the learned positional row to the token embedding. Each block then applies pre-normalized causal self-attention, pre-normalized cross-attention, and a pre-normalized GELU feed-forward network, with a residual addition after each. A final layer norm follows, and logits come from multiplying by the transpose of the token embedding matrix: $\text{LN}(x)\,E^\top$.

Self-attention sees only the decoder prefix; cross-attention sees the complete encoded audio window. That asymmetry decides the caching. Cross keys and values depend only on the audio window and the decoder layer, so you compute them once and retain one pair per layer. Self-attention keys and values depend on the prefix, so they live in a cache that grows. The cache rule that follows is the subtle one: read the common committed cache length before any layer appends, so that every layer uses the same position offset and the same causal mask. And validate every cache shape before the first append, so a rejected request cannot partially mutate the cache.

## Loading a checkpoint strictly

The loader removes `model.` from parameter names, verifies an optional tied output projection, and separates the encoder's frozen sinusoid table from the trainable parameters. Strict state loading then rejects missing, unexpected or wrongly shaped tensors. For sharded files, each tensor must occur in the file the index names, exactly once.

One complication comes from how checkpoints are published. Some store the fixed sinusoid table in float16. Such a table is retained exactly, while a corrupted entry is rejected: the loader permits $5 \cdot 10^{-4}$ absolute deviation only for tables exactly representable in float16, and other tables keep the $10^{-4}$ bound.

## Tracing a cached decode by hand

Run on a real checkpoint, the following calculation reproduces Hugging Face's logits.

For a two-token prefix `[sot, language]`, a fresh cache has offset zero. The self-attention permission matrix is `[[True, False], [True, True]]`, and both tokens may attend to every encoded audio position. Every decoder layer appends two self-key and self-value rows, making the committed length two.

The next one-token call uses positional row two and a `1 x 3` all-true self-attention mask. It reuses the unchanged cross keys and values. Its last logits must agree with recomputing all three prefix tokens in one uncached call. And if the checkpoint stores an output projection, each element must equal the corresponding token-embedding element before the redundant tensor is discarded.

## What a correct implementation must handle

The tests follow from that trace. Logits must match independent Hugging Face logits within an absolute tolerance of `1e-5`, and the parameter count must include the frozen encoder position table. Cached chunked decoding must agree with one full call, with cross keys and values unchanged. Causality must hold (a changed suffix cannot affect prefix logits), and one-dimensional ids must preserve the documented output shape. Invalid token ids, ranks and positions must leave the cache untouched, and batch, layer, head and feature widths of the cross-KV and the cache must be checked before decoding. A half-precision position table must keep its rounded values while a corrupted entry fails; incompatible weight names, positions and output ties must fail explicitly. Sharded and unsharded states must agree, and shard paths must not escape the checkpoint directory. Incomplete checkpoints must not leave random parameters in the model. Finally, the WAV entry point's channel mixing, padding and mel features must produce the same cross-attention keys and logits as the explicit pipeline.

## Building it step by step

The reasoning:

1. **Two kinds of context, two caches.** Self-attention keys and values depend on the text prefix, which grows by one or more tokens per call, so they live in an append-only `KVCache`. Cross-attention keys and values depend only on the encoder output and the layer's weights, so they are computed once per 30 s window (`CrossKV`) and passed unchanged to every call.
2. **One offset for all layers.** Position $o + t$ must mean the same token in every layer, so the offset is read once, before any layer appends. If each layer read its own length, layer 1 would see layer 0's append and use positions one chunk ahead.
3. **Validate before mutating.** `cache.update` appends. If a check failed halfway through the layer loop, earlier layers would hold an extra chunk and the cache would be inconsistent. So every check (ids, ranks, cross-KV shapes, cache shape, capacity, equal fills) runs before the loop.
4. **Causality comes from the mask.** For a fresh call, `causal_mask(t, o + t, q_offset=o)`; with a cache, `cache.mask(t)` builds the same `t x (o + t)` lower-triangular-with-offset matrix. In the trace's second step that is `1 x 3`, all true.
5. **Tied output.** Whisper's logits are $\text{LN}(x)\,E^\top$ with $E$ the token embedding. A checkpoint may also store `proj_out.weight`. It is redundant, and if it differs from $E$ the checkpoint is not one this model can represent, so it is an error rather than a second weight.
6. **Strict loading.** A missing tensor would leave a random parameter that still produces plausible-looking logits. So the loader maps every name, rejects unknown ones, and `load_state_dict(strict=True)` rejects missing or wrongly shaped ones.

### The decoder layer

Three pre-LN residual branches. Self-attention is written out, instead of calling `MultiHeadAttention.forward`, so that the new keys and values can go through the cache. Cross-attention uses precomputed keys and values and no mask.

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

### Cross keys and values, once per window

Each layer's `encoder_attn` projects the audio with its own `k_proj` (no bias, from the encoder's `WhisperAttention`) and `v_proj`. The copies detach the arrays from the autograd graph so later calls cannot alter them.

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

### The decoder forward: all checks, then one offset, then the loop

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

### The model

`Whisper` composes the encoder with this decoder. `encode_wav` runs the [audio frontend](01-audio-frontend.md) with the model's mel count and window length (`2 * n_audio_ctx * hop_length` samples), so the WAV path and the explicit path produce the same features. `param_count` adds `n_audio_ctx * d`, because the encoder position table is part of the published parameter count but is not a trainable `Parameter` here.

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

The full model is in [`decoder.py`](../../../../course/solutions/python/tinyllm/speech/decoder.py).

### Mapping Hugging Face names

Every tensor goes to exactly one of three places: verified and dropped (`proj_out.weight`), kept aside as the fixed table, or stripped of `model.` into the state dict. Anything else is an error.

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

### Sharded checkpoints

The index's `weight_map` names a file per tensor. The loader reads each named file once, refuses paths that resolve outside the checkpoint directory, and checks that every tensor is in the file the index says, exactly once, and that the index names nothing missing.

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

The full loader is in [`hf_whisper.py`](../../../../course/solutions/python/tinyllm/io/hf_whisper.py).

### Why this design

With a cache, decoding token $o$ costs $O(o\,d)$ per layer for self-attention and $O(A\,d)$ for cross-attention, instead of $O(o^2 d)$ for recomputing the prefix. Cross keys and values cost $O(L\,A\,d^2)$ once per window rather than once per token; for whisper-tiny that avoids two $1500 \times 384 \times 384$ projections per layer on every step.

Recomputing cross keys and values per step is correct, but it repeats the most expensive projection hundreds of times per window. Loading leniently (`strict=False`, ignoring `proj_out.weight`) would accept fine-tuned checkpoints with an untied head and silently give them the wrong output layer. Trusting the stored position table without checking it would accept a checkpoint from a different architecture.

Two tradeoffs are accepted. The tied-output requirement rejects some community fine-tunes; that is deliberate, since this model has no slot for a separate head. And the loader converts everything to float32, doubling memory for float16 checkpoints.

## What can go wrong

Each of the classic decoder bugs produces plausible output, which is what makes them dangerous. Resetting the position offset at every incremental call makes full and cached logits disagree; the solution reads `offset = cache.seq_len()` and passes it to `embed_positions(..., offset)`. Omitting the causal mask lets future tokens change earlier predictions; the solution always passes `causal_mask` or `cache.mask(t)` to self-attention. Advancing the committed length once per layer gives later layers different positions; the solution reads the offset once before the loop and requires equal fills.

> **Warning:** Silently loading an untied output matrix gives plausible but incorrect transcript scores. `whisper_state_from_hf` requires `proj_out.weight` to equal the token embedding exactly.

## Where this leads

[Whisper decoding and long-form transcription](04-whisper-decoding.md) consumes these logits, reuses the cross-KV for every step, beam and retry within a window, and turns them into timestamped segments. In *AI Engineering*, the consent and model-card evidence practice in [Consent and biometric evidence in model cards](../34-privacy-and-pii/02-consent-and-biometric-evidence.md) calls this implementation.

### Further reading

- The [Whisper decoder contract](../../../../course/contracts/py/tinyllm/speech/decoder.pyi). As an exercise, derive the parameter count from it, then compare one-token and multi-token cache updates.
- Compare token-at-a-time and chunked decoding, including a batch with different prompts.

## Key terms

- **cross-attention**: attention whose queries come from the decoder and whose keys and values come from the encoded audio window, with no mask.
- **`CrossKV`**: the per-layer cross keys and values, projected once per window and reused for every decoding call.
- **committed length**: the number of prefix positions every layer's self-attention cache holds; read once per call as the position offset.
- **tied output**: computing logits as $\text{LN}(x)\,E^\top$ with the token embedding $E$, so there is no separate output matrix.
- **strict loading**: rejecting any missing, unexpected or wrongly shaped tensor rather than leaving parameters at random values.

## Summary

- Self-attention sees only the decoder prefix; cross-attention sees the complete encoded audio window.
- Cross keys and values are projected once per window; self keys and values grow at each decoding step.
- The offset is read once before any layer appends, and every check runs before the first append, so a rejected request leaves the cache untouched.
- Output logits use the token embedding matrix transposed; a separate untied output weight is an invalid checkpoint.
- The loader keeps the checkpoint's own sinusoid table after verifying it, with a looser bound only for float16-representable tables.

## Review questions

1. Why is `offset` read once before the layer loop instead of from each layer's cache?
   <details><summary>Answer</summary>

   Layer 0's `cache.update` appends before layer 1 runs. Reading inside the loop would give layer 1 an offset one chunk larger, so its mask and its notion of the prefix would disagree with layer 0's. One read of the committed length makes every layer use the same positions.

   </details>

2. Why must all checks happen before the first `cache.update`?
   <details><summary>Answer</summary>

   The update mutates the cache. A failure after some layers appended would leave layers with different fills, which the next call rejects as inconsistent. Checking first means a rejected request leaves the cache exactly as it was.

   </details>

3. Why is a different `proj_out.weight` an error instead of a weight to load?
   <details><summary>Answer</summary>

   The model computes logits as $\text{LN}(x)E^\top$; there is no separate output matrix to load it into. Ignoring it would produce plausible but wrong scores; failing tells the user the checkpoint is not a standard Whisper.

   </details>

4. Why does `param_count` add `n_audio_ctx * d`?
   <details><summary>Answer</summary>

   The encoder's sinusoid table is stored in checkpoints and counted in published sizes (about 37.8 million for tiny), but here it is a fixed array, not a trainable parameter, so `parameters()` does not include it.

   </details>

5. In the hand trace, what mask does the second (one-token) call use, and which positional row?
   <details><summary>Answer</summary>

   The committed length is two, so the new token uses positional row two and a `1 x 3` all-true self-attention mask. Cross keys and values are reused unchanged.

   </details>
