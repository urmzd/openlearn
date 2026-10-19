# The Llama-Family Model: Config, Weight Loading, and Download

This chapter has built every piece of a modern decoder as a separate module, and none of them has produced a token yet. The transformer from [Encoder-decoder Transformer: post-LN and pre-LN, Noam, label smoothing](../06-transformer-2017/05-encoder-decoder-transformer.md) is a 2017 design with its own parameter names, so it cannot load a single published checkpoint.

This section assembles RMSNorm, gated MLPs or experts, RoPE with scaled frequencies, grouped-query or latent attention, and windows into one model whose config and tensor names are Hugging Face's. It then fetches SmolLM2-135M from the Hub with its own downloader and matches Hugging Face's logits. Everything after this in the book (the sampler, caches, kernels, the Rust engine, the capstone) runs this model.

Nearly every earlier section of the chapter feeds in: `RMSNorm` and `pre_norm_residual` from [Pre-LN and RMSNorm](01-pre-ln-rmsnorm.md), `GatedMLP` from [Gated MLPs](02-gated-mlps.md), `RopeSpec` from [Rotary Position Embeddings](03-rope.md), `rope_inv_freq_scaled` from [Context Extension](04-context-extension.md), `GQAttention` from [Grouped-Query Attention](05-gqa-attention.md), `MLAttention` from [Multi-Head Latent Attention](06-multi-head-latent-attention.md), `attention_mask` from [Sliding Windows and Attention Sinks](07-sliding-window-and-sinks.md), and `MoE` from [Mixture of Experts](08-mixture-of-experts.md). From the foundations you need `load_safetensors` and `save_safetensors` from [Safetensors for every dtype, atomic checkpoints, and the token stream](../01-foundations/07-safetensors-checkpoints-and-token-streams.md), `Embedding`, `Linear`, and `ModuleList` from [Module system and basic layers](../01-foundations/05-module-system-and-layers.md), the op library, and `Tensor`. From *Mathematics for Computing* you need `ModelConfig` from [Counting: parameters, FLOPs, KV bytes, and memory plans](../../mathematics-for-computing/06-discrete-math-1/01-counting-params-flops-kv-bytes.md) and `PCG32` from [Modular arithmetic, hashing, and PCG32 in Python](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md); [IEEE 754 anatomy, ulps, and round to nearest even](../../mathematics-for-computing/10-numerical-methods-and-floating-point/01-ieee-754.md) explains BF16, and [Byte-level BPE](../02-tokenizers/02-byte-level-bpe.md) covers the SmolLM2 tokenizer.

The reference solution is [`llama.py`](../../../../course/solutions/python/tinyllm/modern/llama.py), with `LlamaConfig` (`from_hf`, `to_hf`, `model_config`, `rope_spec`), `LlamaDecoderLayer`, `LlamaModel`, and `LlamaForCausalLM`, plus [`hf.py`](../../../../course/solutions/python/tinyllm/io/hf.py), with `hf_download` and `_fetch`. The contracts are [`course/contracts/py/tinyllm/modern/llama.pyi`](../../../../course/contracts/py/tinyllm/modern/llama.pyi) and [`course/contracts/py/tinyllm/io/hf.pyi`](../../../../course/contracts/py/tinyllm/io/hf.pyi). The solution's entry point exposes the model as four CLI verbs in [`cli_modern.py`](../../../../course/solutions/entry/python/tinyllm/cli_modern.py): `pull <owner/name>` runs `hf_download` into `<cache>/<owner>--<name>/<revision>`, `info --model DIR` prints `param_count()`, `logits --model DIR --prompt TEXT` prints next-token logits, and `generate --model DIR --prompt TEXT` decodes greedily or with sampling through `ConcatKVCache`. Each prints one JSON object as its last line. By the end of the section, that CLI loads a Llama-family checkpoint exactly as transformers saves it, reports Hugging Face's parameter count, gives Hugging Face's next-token logits, and decodes Hugging Face's greedy continuation.

> **Learning objectives**
> - Assemble a Llama decoder block from two Pre-LN residual steps and map every `config.json` key onto a module.
> - Count SmolLM2-135M's 134,515,008 parameters by hand from its config.
> - Load a checkpoint strictly by name, including tied heads, shards, and both transformers 4.x and 5.x configs.
> - Continue positions from the cache so cached decoding equals the full forward.
> - Build a downloader that resumes, verifies sha256, and renames atomically.

## Notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$, $d$, $L$ | vocabulary, hidden width, layers | `int` |
| $H$, $H_{kv}$, $d_h$ | query heads, kv heads, head width | `int` |
| $f$ | `intermediate_size` | `int` |
| $h_l$ | the residual stream after layer $l$ | `float32[B, T, d]` |
| $E \in \mathbb{R}^{V \times d}$ | `model.embed_tokens.weight` | matrix |
| $p_t$ | absolute position of token $t$ | `int` |

## The block

A Llama model is an embedding, $L$ identical blocks of two Pre-LN residual steps, a final norm, and a head:

$$h_0 = E[\mathrm{ids}], \quad h' = h_{l-1} + \mathrm{Attn}(\mathrm{RMSNorm}_1(h_{l-1})), \quad h_l = h' + \mathrm{MLP}(\mathrm{RMSNorm}_2(h')),$$
$$\mathrm{logits} = \mathrm{RMSNorm}_f(h_L)\, W_{head}^\top, \qquad W_{head} = E \text{ when tied}.$$

Attention is `GQAttention` (or `MLAttention`), the MLP is `GatedMLP` (or `MoE` from layer `first_k_dense_replace` on), and the norms are `RMSNorm`. RoPE gets its frequencies from `rope_inv_freq_scaled`, and windows and sink tokens become a mask from `attention_mask`.

## From config.json to modules

`LlamaConfig.from_hf` reads a `config.json` from transformers 4.x (`rope_theta`, and `rope_scaling` with `"type"` or `"rope_type"`) or 5.x (`rope_parameters`, with `rope_theta` inside). Defaults matter: SmolLM2's config has no `head_dim` (it is $d / H$), and Mistral's `sliding_window` is a top-level key. `model_type` qwen2 means q, k, and v biases; a Llama config with `attention_bias` would also bias `o_proj`, which `GQAttention` does not model, so it is refused. The book's own extensions use `tl_*` keys from `formats/config.schema.json`: `tl_attention = "mla"` with `tl_mla_rank`, `tl_num_experts`, `tl_sliding_window`, and `tl_sink_tokens`. `to_hf()` writes the dict back, and `from_hf(to_hf(c)) == c`.

## The checkpoint contract

The module system made dotted parameter names the safetensors keys. Registered in Hugging Face's order (decoder layer: `self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`; model: `embed_tokens`, `layers`, `norm`; then `lm_head`), the model's `state_dict` has exactly Hugging Face's keys. A tied model registers the embedding once and has no `lm_head.weight`. Some tied checkpoints store the copy anyway, and old ones store `rotary_emb.inv_freq`; both are dropped before a strict load. BF16 weights arrive as float32 through the safetensors loader, and large checkpoints come in shards listed by `model.safetensors.index.json`.

`param_count()` is the number of parameters, a tied tensor counted once; it equals Hugging Face's `num_parameters()` and the count of `model_config()` from *Mathematics for Computing*. Building SmolLM2 with `init=False` allocates the tensors without 135 million Box-Muller draws, since `from_pretrained` overwrites every value anyway.

## Positions, the cache, and training

Without explicit positions, a forward with a cache starts at `cache.seq_len(0)`, because the cache already holds that many tokens. Every layer gets the same cache object and its own `layer` index. With a window, the mask comes from `attention_mask`, which understands both a bounded `SinkWindowCache` and a cache that keeps everything.

Built from the op library end to end, the model also trains with the book's autograd: the cross-entropy reaches every parameter, and the tied embedding collects gradient from both of its uses, the lookup and the head.

## Downloading weights

The model is useless without the bytes, and a 270 MB download over a real network fails in interesting ways. The Hub serves `{endpoint}/{repo}/resolve/{revision}/{file}`. A large file answers with a redirect to a CDN, and the redirect carries `X-Linked-Etag` (the sha256 of the content) and `X-Linked-Size`.

`hf_download` writes to `<file>.part`. After a dropped connection it sends `Range: bytes=<size>-` and appends, hashing the bytes already on disk first; a server that ignores the range answers 200 with the whole file, so the download restarts. Only a file whose size and sha256 check out is renamed into place (`os.replace` is atomic), so a crash or a corrupt byte never leaves a bad file under the final name.

## Working it by hand

SmolLM2-135M-Instruct's `config.json` has $V = 49152$, $d = 576$, $L = 30$, $H = 9$, $H_{kv} = 3$, $f = 1536$, `rms_norm_eps` $10^{-5}$, `rope_theta` $10^5$, tied embeddings, and no `head_dim`, so $d_h = 576 / 9 = 64$ and the kv width is $3 \cdot 64 = 192$. One layer breaks down as follows:

| Part | Shape | Parameters |
|---|---|---|
| `q_proj`, `o_proj` | $576 \times 576$ each | 663,552 |
| `k_proj`, `v_proj` | $192 \times 576$ each | 221,184 |
| gate, up, down | $3 \times 576 \times 1536$ | 2,654,208 |
| two RMSNorms | $2 \times 576$ | 1,152 |
| **layer** | | **3,540,096** |

Thirty layers make 106,202,880. The embedding is $49152 \times 576 = 28{,}311{,}552$ (it is also the head, tied, so it is counted once), and the final norm adds 576. The total is **134,515,008**, Hugging Face's `num_parameters()`.

The download works the same way by hand. `hf_download("org/tiny", ["config.json", "model.safetensors"], cache, revision="abc123")` asks for `/org/tiny/resolve/abc123/config.json` (served directly), then `/org/tiny/resolve/abc123/model.safetensors` (a 302 to the CDN carrying the sha256), and leaves both files in `cache/org--tiny/abc123/` with no `.part` file.

## The interface

```python
@dataclass
class LlamaConfig:                     # HF names; tl_* extensions (contract table)
    @classmethod
    def from_hf(cls, config_json: str) -> "LlamaConfig"
    def to_hf(self) -> dict;  def model_config(self) -> ModelConfig;  def rope_spec(self) -> RopeSpec
class LlamaForCausalLM(Module):
    def __init__(self, cfg, rng=None, init=True)
    def forward(self, ids, positions=None, cache=None) -> Tensor         # logits [B, T, V]
    def aux_loss(self) -> Optional[Tensor];  def param_count(self) -> int
    @classmethod
    def from_pretrained(cls, dir: str) -> "LlamaForCausalLM"
    def save_pretrained(self, dir: str, dtype: str = "F32") -> None
def hf_download(repo_id, filenames, cache_dir, revision="main", endpoint=None, sha256=None, timeout=60.0) -> str
```

`LlamaConfig` normalizes every config dialect into one dataclass. `LlamaForCausalLM` takes token ids of shape `[T]` or `[B, T]` and returns logits; `aux_loss` collects the MoE balancing losses, and `from_pretrained` and `save_pretrained` read and write Hugging Face directories. `hf_download` returns the directory it filled.

## Building it step by step

The checkpoint is the contract. Matching Hugging Face means two things: build the same modules from the same `config.json` keys, and register parameters under the same dotted names in the same order. Everything else follows from those two.

`config.json` comes in several dialects (transformers 4.x and 5.x, Mistral, Qwen2, the book's `tl_*` extensions). One reader normalizes them into one dataclass, so every later step sees a single shape, and a writer (`to_hf`) makes the round trip exact. A block is two Pre-LN residual steps; which attention (GQA or MLA) and which MLP (dense or MoE, by layer index) are config decisions made in the layer's constructor, and the forward is the same for all.

Windows and sink tokens are a property of the mask, not of the attention module, because sink tokens must be exempt from the window. So the layer asks `attention_mask` for a mask and gives `GQAttention` no window of its own. With a cache, positions must continue from what the cache holds, and every layer must see the same cache with its own index. A tied head is one tensor with one name; it must be counted once, saved once, and receive gradient from both its uses.

Two practical requirements finish the design. Loading 135 M parameters should not spend time drawing random numbers that are about to be overwritten, so the model builds shapes without draws and then loads strictly by name. And a download over a flaky network must be resumable, verified, and atomic: write to `.part`, hash everything including resumed bytes, and rename only after the check.

### Reading `config.json`

These are the parts that decide correctness:

```python
        H = int(c["num_attention_heads"])
        # transformers 5.x nests theta and scaling in rope_parameters; 4.x has
        # rope_theta and rope_scaling at the top.
        rp = dict(c.get("rope_parameters") or {})
        theta = float(rp.pop("rope_theta", c.get("rope_theta", 10000.0)))
        scaling = rp or c.get("rope_scaling")          # what is left of rope_parameters is the scaling
        if scaling:
            scaling = dict(scaling)
            # 5.x and newer 4.x say "rope_type"; older 4.x configs say "type"
            kind = scaling.pop("rope_type", scaling.pop("type", "default"))
            if kind not in ROPE_KINDS:
                raise ValueError(...)
            scaling = None if kind == "default" else {"rope_type": kind, **scaling}
        bias = bool(c.get("attention_bias", False))
        if mt == "llama" and bias:
            # HF Llama's attention_bias also biases o_proj, which GQAttention has not
            raise ValueError(...)
        # Mistral: top-level sliding_window; Qwen2 only when use_sliding_window; tl_* overrides
        window = c.get("tl_sliding_window", c.get("sliding_window") if mt != "qwen2" or c.get("use_sliding_window") else None)
        # ...
        head_dim = int(c.get("head_dim") or c["hidden_size"] // H)   # d / H, never d / Hkv
        cfg = cls(
            # ...
            num_key_value_heads=int(c.get("num_key_value_heads") or H),   # absent means MHA
            qkv_bias=bias or mt == "qwen2" or bool(c.get("tl_qkv_bias", False)),  # Qwen2 implies q, k, v biases
            # ...
            first_k_dense_replace=int(c.get("first_k_dense_replace", 0) or 0) if n_exp else 0,
            # ...
        )
        cfg.model_config()  # M05.1 rejects inconsistent sizes
        return cfg
```

For SmolLM2 there is no `head_dim`, so it is $576 / 9 = 64$, and `num_key_value_heads` is 3. Every `c.get(..., default)` mirrors Hugging Face's own default for that key, which is why a sparse config such as SmolLM2's builds the same model as in transformers. Calling `model_config()` at the end routes the sizes through the `ModelConfig` validator, so an inconsistent config fails at load time, not in a matmul.

### From config to RoPE

```python
    def rope_spec(self):
        if self.attention == "mla":
            r = self.qk_rope_head_dim                    # MLA rotates only its rope part
        else:
            r = int(self.head_dim * self.partial_rotary_factor)
        s = dict(self.rope_scaling or {})
        kind = s.get("rope_type", "default")
        inv, scale = rope_inv_freq_scaled(
            r, self.rope_theta, kind, factor=float(s.get("factor", 1.0)),
            original_max_pos=int(s.get("original_max_position_embeddings") or self.max_position_embeddings),
            # ... beta_fast, beta_slow, low_freq_factor, high_freq_factor with HF's defaults
        )
        if s.get("attention_factor") is not None:
            scale = float(s["attention_factor"])         # an explicit YaRN temperature wins
        return RopeSpec(inv_freq=inv, attention_scaling=scale,
                        layout="interleaved" if self.rope_interleaved else "half", rotary_dim=r)
```

There is one spec per model, shared by every layer: the context-extension code computes the frequencies, and the RoPE code applies them.

### The decoder layer

```python
class LlamaDecoderLayer(Module):
    def __init__(self, cfg, layer_idx, rng=None):
        # ...
        spec = cfg.rope_spec()
        self.window, self.n_sink = cfg.sliding_window, cfg.sink_tokens
        # Registration order = HF LlamaDecoderLayer: self_attn, mlp, then the norms.
        if cfg.attention == "mla":
            self.self_attn = MLAttention(d, H, cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_nope_head_dim,
                                         cfg.qk_rope_head_dim, cfg.v_head_dim, spec, rng=r)
        else:
            # The window lives in the mask (L7.7), so sink tokens can be exempt from it.
            self.self_attn = GQAttention(d, H, cfg.num_key_value_heads, cfg.head_dim, spec, qkv_bias=cfg.qkv_bias,
                                         window=None, sinks=cfg.learned_sinks, rng=r)
        # layers 0 .. first_k_dense_replace - 1 stay dense
        if cfg.num_experts and layer_idx >= cfg.first_k_dense_replace:
            self.mlp = MoE(d, cfg.moe_intermediate_size, cfg.num_experts, cfg.num_experts_per_tok, ...)
        else:
            self.mlp = GatedMLP(d, cfg.intermediate_size, act=cfg.hidden_act, bias=cfg.mlp_bias, rng=r)
        self.input_layernorm = RMSNorm(d, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(d, cfg.rms_norm_eps)

    def forward(self, h, positions, cache=None, layer=0):
        mask = attention_mask(cache, layer, h.shape[1], self.window, self.n_sink)   # None without a window
        h = pre_norm_residual(h, self.input_layernorm, lambda u: self.self_attn(u, positions, mask, cache, layer))
        return pre_norm_residual(h, self.post_attention_layernorm, self.mlp)
```

The forward is the block's two equations, one line each. The registration order is what makes `state_dict()` list `self_attn.*`, `mlp.*`, `input_layernorm.weight`, and `post_attention_layernorm.weight` in Hugging Face's order.

### Building without draws

With `init=False`, `LlamaModel` uses `_FlatRNG`, a `PCG32` stand-in that returns constants, builds one layer of each kind (dense, MoE) and deep-copies it for the others, and widens a one-row embedding to $V \times d$ zeros. Building SmolLM2 this way allocates the right shapes without 135 million Box-Muller draws, and `from_pretrained` then overwrites every value.

### The forward

```python
    def forward(self, ids, positions=None, cache=None):
        a = np.asarray(ids)
        # ... integer [T] or [B, T]
        single = a.ndim == 1
        a = a[None] if single else a
        T = a.shape[1]
        if positions is None:
            # continue from what the cache holds: decoding token s has position s
            s = int(cache.seq_len(0)) if cache is not None and hasattr(cache, "seq_len") else 0
            positions = np.arange(s, s + T)
        h = self.model.embed_tokens(a)  # ValueError for an id outside [0, vocab)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, positions, cache, i)          # one cache, one index per layer
        h = self.model.norm(h)                         # the final RMSNorm
        if self.lm_head is None:
            # tied: the same tensor as the lookup, so it collects gradient from both uses
            logits = F.matmul(h, F.transpose(self.model.embed_tokens.weight, 0, 1))
        else:
            logits = self.lm_head(h)
        return logits[0] if single else logits
```

Positions are read from the cache **before** any layer updates it; `seq_len(0)` is the count of tokens already seen. That one line is what makes prefill-then-decode equal the full forward.

### Counting and loading

```python
    def param_count(self):
        return int(sum(p.data.size for p in self.parameters()))   # a tied tensor is one parameter

    @classmethod
    def from_pretrained(cls, dir):
        d = Path(dir)
        cfg = LlamaConfig.from_hf(str(d))
        model = cls(cfg, init=False)
        # old checkpoints store the RoPE table as a buffer; it is derived, not loaded
        sd = {k: v for k, v in _weights(d).items() if not k.endswith("rotary_emb.inv_freq")}
        if cfg.tie_word_embeddings:
            sd.pop("lm_head.weight", None)  # some tied checkpoints store the copy too
        model.load_state_dict(sd, strict=True)          # every key, no more, no fewer
        return model
```

`_weights` reads `model.safetensors`, or every shard named in `model.safetensors.index.json`. `save_pretrained` writes `to_hf()` plus `torch_dtype` and one safetensors file in F32, BF16, or F16. For SmolLM2, `param_count()` sums the per-layer 3,540,096 thirty times, the embedding once, and the final norm: 134,515,008.

### The downloader

`hf_download` validates the repo id and filenames (no absolute paths, no `..`, so nothing is written outside the cache), skips a file that exists and matches a given sha256, and calls `_fetch` for the rest:

```python
def _fetch(url, final, want_sha, timeout):
    part = final.with_name(final.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    h = hashlib.sha256()
    if start:
        with open(part, "rb") as f:  # the resumed hash must cover the bytes already there
            for block in iter(lambda: f.read(CHUNK), b""):
                h.update(block)
    seen = {}                                          # X-Linked-* from the redirect
    req = urllib.request.Request(url, headers={"User-Agent": "tinyllm-hf-download/1"})
    # ... Authorization from HF_TOKEN
    if start:
        req.add_header("Range", f"bytes={start}-")    # the first byte we do not have
    opener = urllib.request.build_opener(_KeepHeaders(seen))
    try:
        resp = opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code != 416 or not start:
            raise
        resp = None  # 416: nothing after `start`; the .part file is complete
    total = None
    if resp is not None:
        with resp:
            # ... record X-Linked-* if the server sent them directly
            if resp.status == 200 and start:
                # The server ignored the range and sends the whole file: start over.
                start, h = 0, hashlib.sha256()
            # ... total from Content-Range (206) or Content-Length (200)
            with open(part, "ab" if start else "wb") as f:     # append only to a real resume
                for block in iter(lambda: resp.read(CHUNK), b""):
                    f.write(block)
                    h.update(block)
    if seen.get("X-Linked-Size", "").isdigit():
        total = int(seen["X-Linked-Size"])
    etag = (seen.get("X-Linked-Etag") or "").strip('"').lower()
    expect = (want_sha or (etag if _HEX64.match(etag) else "")).lower()
    size = part.stat().st_size
    if total is not None and size < total:
        # A short body is a network failure: keep the bytes for the next resume.
        raise OSError(...)
    if total is not None and size > total:
        part.unlink()
        raise ValueError(...)
    if expect and h.hexdigest() != expect:
        part.unlink()                                  # a corrupt file is never resumed from
        raise ValueError(...)
    os.replace(part, final)  # atomic: the final name only ever holds a checked file
```

`_KeepHeaders` is a redirect handler that remembers `X-Linked-Etag` and `X-Linked-Size` from the Hub's 302, because the CDN it redirects to does not repeat them. Without it, large files (the ones that are redirected) would never be verified. The etag is trusted as the expected hash only when it is 64 hex digits (a sha256); any other etag is ignored rather than compared, so a non-sha256 value can never cause a false rejection.

The full solution is in [`llama.py`](../../../../course/solutions/python/tinyllm/modern/llama.py) and [`hf.py`](../../../../course/solutions/python/tinyllm/io/hf.py).

### Why this design

Registering modules under Hugging Face's names, rather than keeping a mapping table, means `load_state_dict(strict=True)` is the whole loader, and the system's own checkpoints load in transformers. A renaming table would be one more thing that can silently drop or misplace a tensor. Loading is strict because a missing or extra key should raise; lenient loading would leave a parameter at its random init, which produces plausible-looking but wrong logits.

Masks come from `attention_mask`, not from a window inside the attention. A window in `GQAttention` could not exempt sink tokens and would not know about a bounded cache; one mask function handles every cache type. No-draw construction matters because drawing 135 M Gaussians in Python takes noticeable time and is thrown away; `_FlatRNG` and layer copying make loading cost about what reading the file costs.

The downloader uses `urllib` with a redirect handler rather than `huggingface_hub`. The library adds snapshots, symlinks, and Xet storage; the solution needs three properties (resume, verify, atomic rename) and gets them in about 100 lines with no dependency.

As for cost, the forward is $L$ blocks of attention and MLP cost plus an $O(B T d V)$ head, which dominates for small models with large vocabularies such as SmolLM2.

## What can go wrong

The tests cover the model, the loader, and the downloader. SmolLM2's config must give 134,515,008 parameters, as the CLI's `info` reports. Five tiny Hugging Face checkpoints (covering GQA, MQA, a window, biases, tied and untied heads, and Llama 3 and YaRN rope) must give Hugging Face's logits, and `param_count` must equal both Hugging Face and the `ModelConfig` count, which the capstone's budget uses. State-dict keys must be in Hugging Face's order, with `lm_head.weight` only when untied, so the system's checkpoints load in transformers.

Prefill then decode must equal the full forward, including through a bounded cache. An MLA model with MoE after a dense layer must decode through the latent cache and report its aux loss and count, for the capstone's ablations. Window, sink tokens, and learned sinks must stream through `SinkWindowCache` like full attention. `to_hf` and `from_hf` must round-trip, and 4.x and 5.x rope keys must agree. Saving and loading must be bit-exact for F32, rounded for BF16, and must handle shards. The tied embedding's gradient must match central differences, with no silent parameter. Unexpected keys, unsupported rope types, a Llama `attention_bias`, and bad ids must all raise.

For the downloader, the hand-worked URL, redirect, and cache layout must match. A `.part` file must resume with a Range request whose hash covers the whole file, so a dropped connection costs nothing; a 200 answer to a range request must restart the file, for mirrors without range support; corrupt bytes must be rejected with nothing left under the final name; a verified file must be skipped and a stale one replaced, so running `pull` twice is free; and `HF_ENDPOINT` must be honored while bad repo ids and filenames are refused, so mirrors work and nothing is written outside the cache.

[`test_llama.py`](../../../../course/solutions/learner-tests/L7.9/python/tests/l7-9-llama/test_llama.py) uses as oracles the golden logits and parameter counts that transformers produced for five tiny checkpoints, the full forward as the oracle of the cached one, the `ModelConfig` count for SmolLM2 built without weights, and a fake Hub on `127.0.0.1` (`http.server`) that redirects with `X-Linked-Etag` and can cut a response short, ignore ranges, or corrupt a byte. It adds config round-trips (with a 4.x `"type"` key), MLA and MoE variants, sink tokens through `SinkWindowCache`, BF16 saving, shards, and the validation errors. A local fake Hub is the right oracle for a downloader: every network failure the real one can produce is reproducible on demand.

> **Warning:** Deriving a missing `head_dim` from the kv heads ($576 / 3$) instead of the query heads ($576 / 9$) builds SmolLM2 with the wrong shapes. Hugging Face's default is `hidden_size // num_attention_heads`.

The full list of pitfalls:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| deriving a missing `head_dim` from the kv heads | SmolLM2 builds with the wrong shapes | `head_dim = hidden_size // num_attention_heads` |
| ignoring transformers 5's `rope_parameters` | YaRN and Llama-3 checkpoints load with plain RoPE | reads `rope_parameters` first and treats what is left after `rope_theta` as the scaling |
| restarting positions at 0 for every chunk | cached decode drifts after the prompt | positions start at `cache.seq_len(0)` |
| skipping the final RMSNorm | logits on the wrong scale; Hugging Face disagrees | `h = self.model.norm(h)` before the head |
| ignoring `sliding_window` | Mistral agrees with Hugging Face only for short prompts | the reader picks up `sliding_window` and the layer turns it into a mask |
| a sha256 that is never compared | corrupt weights load silently | `h.hexdigest() != expect` raises and deletes the `.part` file |
| appending a 200 answer to the partial file | a resumed file is the start twice | a 200 with `start > 0` resets `start` and the hash |
| Qwen2's biases not implied by its `model_type` | strict load fails on the bias keys | `qkv_bias = ... or mt == "qwen2"` |
| counting a tied embedding twice | `info` reports 163 million for SmolLM2 | `lm_head = None` when tied; `param_count` sums `parameters()` |
| one more dense layer than `first_k_dense_replace` | MoE configs build the wrong model | MoE when `layer_idx >= first_k_dense_replace` |
| a mask that forgets the sink tokens | streams degrade once token 0 leaves the window | the layer passes `n_sink` to `attention_mask` |
| `tl_attention = "mla"` building GQA | MLA checkpoints do not load | `attention="mla" if attn == "mla"`, and the layer builds `MLAttention` |
| the 4.x `"type"` key ignored | old checkpoints lose their rope scaling | `scaling.pop("rope_type", scaling.pop("type", "default"))` |
| `tl_learned_sinks` not built | gpt-oss-style configs lose their sinks | `GQAttention(..., sinks=cfg.learned_sinks)` |
| the redirect's `X-Linked-Etag` dropped | nothing is verified for large files | `_KeepHeaders` records it from the 302 |
| a resumed hash that skips the bytes on disk | every resumed download fails its check | the hash is seeded with the `.part` file's bytes |
| a range that starts one byte late | resumed files lose a byte | `Range: bytes={start}-` where `start` is the current size |
| downloading a verified file again | `pull` costs 270 MB every time | `hf_download` skips a file that exists and matches |

## Where this leads

This model is what the rest of the book runs. [KV cache and generate](../09-inference/02-kv-cache-and-generate.md) adds KV-cached generation over it, and the rest of the inference chapter builds on it: [Sampling and logit processors](../09-inference/01-sampling-and-logit-processors.md), [A paged KV cache in Python](../09-inference/03-paged-kv-cache.md), [Radix prefix cache](../09-inference/04-radix-prefix-cache.md), [Quantization](../09-inference/05-quantization.md), and [Speculative decoding](../09-inference/06-speculative-decoding.md). The Rust engine's logits in [Candle model runner](../11-serving/02-model-runner-and-sampler.md) are compared with this model's, and the [TinyStories capstone](../capstones/01-tinystories.md) uses it as its architecture and release format.

### Further reading

- Touvron et al., "Llama 2" (2023), section 2.2.
- Hugging Face `LlamaForCausalLM` in `transformers/models/llama/modeling_llama.py`: attention backends (SDPA, FlashAttention), `logits_to_keep`, and tensor-parallel plans.
- Hugging Face `PretrainedConfig` and the `rope_parameters` standardization in `transformers/modeling_rope_utils.py`: one schema for every rope variant, plus config validation.
- Hugging Face weight loading and safetensors `safe_open` in `transformers/modeling_utils.py`: memory-mapped lazy loading and key renaming (`conversion_mapping`).
- `huggingface_hub.hf_hub_download` in `huggingface_hub/file_download.py`: ETag caching by commit, symlinked snapshots, and Xet storage; see also the `safetensors` and Hub download documentation.

## Key terms

- **Decoder block**: two Pre-LN residual steps, attention then MLP.
- **Tied embedding**: one tensor used both as the token lookup and as the output head, counted and saved once.
- **Strict loading**: loading a state dict that must contain exactly the model's keys, no more and no fewer.
- **Shard**: one of several safetensors files listed in `model.safetensors.index.json` for a large checkpoint.
- **`X-Linked-Etag`**: the Hub redirect header carrying a large file's sha256.
- **Atomic rename**: `os.replace` of a verified `.part` file, so the final name only ever holds checked bytes.

## Summary

- A Llama block is two Pre-LN residual steps, attention then MLP, and the whole model is Hugging Face's config keys and tensor names: SmolLM2's 134,515,008 parameters fall out of its config by hand.
- Loading by name is strict: every key in Hugging Face's order, a tied head stored once, and transformers 4.x and 5.x configs read alike.
- Windows and sinks live in the mask, so one function serves every cache type.
- Positions continue from what the cache holds, so decoding one token at a time equals the full forward, with windows and sinks too.
- A download is resumable, verified against the published sha256, and atomic.

## Review questions

1. Why is SmolLM2's `head_dim` 64 and not $576 / 3$?
   <details><summary>Answer</summary>Hugging Face's default is `hidden_size // num_attention_heads`: the query heads set the head width; kv heads only set how many key/value heads there are. $576 / 9 = 64$, and the kv projections are $3 \cdot 64 = 192$ wide.</details>
2. Why does `forward` read `cache.seq_len(0)` before running any layer?
   <details><summary>Answer</summary>`seq_len` counts tokens already cached; the new tokens' positions start there. Reading after layer 0 had updated the cache would shift every position by $T$; restarting at 0 per chunk would re-rotate decoded tokens to the prompt's positions.</details>
3. Why does the decoder layer pass `window=None` to `GQAttention`?
   <details><summary>Answer</summary>Sink tokens must stay visible outside the window, and a bounded cache needs a mask built from held positions. `attention_mask` handles both; a window inside `GQAttention` could not.</details>
4. Why does a tied model count, save, and load its embedding once?
   <details><summary>Answer</summary>It is one tensor used twice. `parameters()` yields it once, so `param_count` matches Hugging Face; there is no `lm_head` module, so `state_dict` has no `lm_head.weight`, and a checkpoint that stores the copy has it dropped before the strict load.</details>
5. What would go wrong if `_fetch` appended a 200 response to an existing `.part` file?
   <details><summary>Answer</summary>A server that ignores the Range header sends the whole file again; appending it would produce the first `start` bytes twice. The solution resets `start` and the hash and opens the file with `"wb"`.</details>
6. Where does the figure of 163 million come from if the tied embedding is counted twice?
   <details><summary>Answer</summary>Adding a second copy of the $49152 \times 576 = 28{,}311{,}552$ embedding to 134,515,008 gives 162,826,560, about 163 million.</details>
