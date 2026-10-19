# Llama-family model, HF config, weight loading, downloader

## Overview

| | |
|---|---|
| **Module** | `L7.9` · build · Python · Pass 5 · 3 h reading |
| **Solution** | [`llama.py`](../../../../course/solutions/python/tinyllm/modern/llama.py): `LlamaConfig` (`from_hf`, `to_hf`, `model_config`, `rope_spec`), `LlamaDecoderLayer`, `LlamaModel`, `LlamaForCausalLM`; [`hf.py`](../../../../course/solutions/python/tinyllm/io/hf.py): `hf_download`, `_fetch`; the CLI verbs `pull`, `info`, `logits`, `generate` in [`cli_modern.py`](../../../../course/solutions/entry/python/tinyllm/cli_modern.py); model oracle tests in [`test_llama.py`](../../../../course/solutions/learner-tests/L7.9/python/tests/l7-9-llama/test_llama.py) |
| **Contract** | [`course/contracts/py/tinyllm/modern/llama.pyi`](../../../../course/contracts/py/tinyllm/modern/llama.pyi), [`course/contracts/py/tinyllm/io/hf.pyi`](../../../../course/contracts/py/tinyllm/io/hf.pyi) |
| **Needs** | `L7.1` `RMSNorm`, `pre_norm_residual` · `L7.2` `GatedMLP` · `L7.3` `RopeSpec` · `L7.4` `rope_inv_freq_scaled` · `L7.5` `GQAttention` · `L7.6` `MLAttention` · `L7.7` `attention_mask` · `L7.8` `MoE` · `L0.6` `load_safetensors`, `save_safetensors` · `M05.1` `ModelConfig` · `L0.4` `Embedding`, `Linear`, `ModuleList` · `L0.2` the op library · `L0.1` `Tensor` · `M06.3` `PCG32` · reading: `M09.1` BF16, `L1.2` the SmolLM2 tokenizer |
| **Used by** | the end of Part 7: the system's CLI loads a Llama-family checkpoint exactly as transformers saves it, reports HF's parameter count, gives HF's next-token logits, and decodes HF's greedy continuation · later: `L8.1` to `L8.6` (inference), `L10.1` (logits parity), C1 (the capstone architecture) · later: `L8.2` |
| **Optional depth** | Touvron et al., "Llama 2" (2023), section 2.2; Hugging Face, `modeling_llama.py` and the `safetensors` and Hub download documentation |

## Key Takeaways

- A Llama block is two Pre-LN residual steps, attention then MLP, and the whole model is HF's config keys and tensor names: SmolLM2's 134,515,008 parameters fall out of its config by hand.
- Loading by name is strict: every key in HF's order, a tied head stored once, transformers 4.x and 5.x configs read alike.
- Positions continue from what the cache holds, so decoding one token at a time equals the full forward, with windows and sinks too.
- A download is resumable, verified against the published sha256, and atomic.

## How to read this chapter

Sections 1 to 3 assemble the block, map `config.json` onto modules, state the checkpoint and download contracts, and count SmolLM2's parameters by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution: the config reader, the layer, the model, loading, and the downloader. Try to predict each step of the solution before reading it.

The solution's entry point exposes this module as four CLI verbs ([`cli_modern.py`](../../../../course/solutions/entry/python/tinyllm/cli_modern.py)): `pull <owner/name>` runs `hf_download` into `<cache>/<owner>--<name>/<revision>`, `info --model DIR` prints `param_count()`, `logits --model DIR --prompt TEXT` prints next-token logits, and `generate --model DIR --prompt TEXT` decodes greedily or with sampling through `L7.5`'s `ConcatKVCache`. Each prints one JSON object as its last line.

---

## 1. Why now

Part 7 built every piece of a modern decoder as a separate module, and none of them has produced a token yet. The transformer from Part 5 (`L5.5`) is a 2017 design with its own parameter names, so it cannot load one published checkpoint. This module assembles RMSNorm, gated MLPs or experts, RoPE with scaled frequencies, GQA or latent attention, and windows into one model whose config and tensor names are Hugging Face's, then fetches SmolLM2-135M from the Hub with its own downloader and matches HF's logits. Everything after this (the sampler, caches, kernels, the Rust engine, the capstone) runs this model.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$, $d$, $L$ | vocabulary, hidden width, layers | `int` |
| $H$, $H_{kv}$, $d_h$ | query heads, kv heads, head width | `int` |
| $f$ | `intermediate_size` | `int` |
| $h_l$ | the residual stream after layer $l$ | `float32[B, T, d]` |
| $E \in \mathbb{R}^{V \times d}$ | `model.embed_tokens.weight` | matrix |
| $p_t$ | absolute position of token $t$ | `int` |

### 2.1 The block

$$h_0 = E[\mathrm{ids}], \quad h' = h_{l-1} + \mathrm{Attn}(\mathrm{RMSNorm}_1(h_{l-1})), \quad h_l = h' + \mathrm{MLP}(\mathrm{RMSNorm}_2(h')),$$
$$\mathrm{logits} = \mathrm{RMSNorm}_f(h_L)\, W_{head}^\top, \qquad W_{head} = E \text{ when tied}.$$

Attention is `L7.5`'s `GQAttention` (or `L7.6`'s `MLAttention`), the MLP is `L7.2`'s `GatedMLP` (or `L7.8`'s `MoE` from layer `first_k_dense_replace` on), the norms are `L7.1`'s. RoPE (`L7.3`) gets its frequencies from `L7.4`, and windows and sink tokens become a mask from `L7.7`.

### 2.2 From config.json to modules

`LlamaConfig.from_hf` reads a `config.json` from transformers 4.x (`rope_theta`, `rope_scaling` with `"type"` or `"rope_type"`) or 5.x (`rope_parameters`, with `rope_theta` inside). Defaults matter: SmolLM2's config has no `head_dim` (it is $d / H$) and Mistral's `sliding_window` is a top-level key. `model_type` qwen2 means q, k, v biases; a Llama config with `attention_bias` would also bias `o_proj`, which `L7.5` does not model, so it is refused. The course's own extensions use `tl_*` keys from `formats/config.schema.json`: `tl_attention = "mla"` with `tl_mla_rank`, `tl_num_experts`, `tl_sliding_window`, `tl_sink_tokens`. `to_hf()` writes the dict back, and `from_hf(to_hf(c)) == c`.

### 2.3 The checkpoint contract

`L0.4` made the dotted parameter names the safetensors keys. Registered in HF's order (decoder layer: `self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`; model: `embed_tokens`, `layers`, `norm`; then `lm_head`), the model's `state_dict` has exactly HF's keys. A tied model registers the embedding once and has no `lm_head.weight`; some tied checkpoints store the copy anyway, and old ones store `rotary_emb.inv_freq`: both are dropped before a strict load. BF16 weights arrive as float32 through `L0.6` (the conversion is `M09.1`'s). Large checkpoints come in shards listed by `model.safetensors.index.json`.

### 2.4 Counting what was loaded

`param_count()` is the number of parameters, a tied tensor once: HF's `num_parameters()` and `M05.1`'s count of `model_config()`. Building SmolLM2 with `init=False` allocates the tensors without 135 million Box-Muller draws (`from_pretrained` overwrites every value anyway).

### 2.5 Positions and the cache

Without explicit positions, a forward with a cache starts at `cache.seq_len(0)`: the cache already holds that many tokens. Every layer gets the same cache object and its own `layer` index. With a window, the mask comes from `L7.7`'s `attention_mask`, which understands both a bounded `SinkWindowCache` and a cache that keeps everything.

### 2.6 Training it

Built from the op library end to end, the model trains with the course's autograd: the cross-entropy reaches every parameter, and the tied embedding collects gradient from both of its uses, the lookup and the head.

### 2.7 Downloading weights

The Hub serves `{endpoint}/{repo}/resolve/{revision}/{file}`. A large file answers with a redirect to a CDN, and the redirect carries `X-Linked-Etag` (the sha256 of the content) and `X-Linked-Size`. `hf_download` writes to `<file>.part`; after a dropped connection it sends `Range: bytes=<size>-` and appends, hashing the bytes already on disk first; a server that ignores the range answers 200 with the whole file, so the download restarts. Only a file whose size and sha256 check out is renamed into place (`os.replace` is atomic), so a crash or a corrupt byte never leaves a bad file under the final name.

## 3. Worked example by hand

SmolLM2-135M-Instruct's `config.json`: $V = 49152$, $d = 576$, $L = 30$, $H = 9$, $H_{kv} = 3$, $f = 1536$, `rms_norm_eps` $10^{-5}$, `rope_theta` $10^5$, tied, no `head_dim`, so $d_h = 576 / 9 = 64$ and the kv width is $3 \cdot 64 = 192$.

One layer:

| Part | Shape | Parameters |
|---|---|---|
| `q_proj`, `o_proj` | $576 \times 576$ each | 663,552 |
| `k_proj`, `v_proj` | $192 \times 576$ each | 221,184 |
| gate, up, down | $3 \times 576 \times 1536$ | 2,654,208 |
| two RMSNorms | $2 \times 576$ | 1,152 |
| **layer** | | **3,540,096** |

Thirty layers make 106,202,880; the embedding $49152 \times 576 = 28{,}311{,}552$ (also the head: tied, counted once); the final norm 576. Total: **134,515,008**, HF's `num_parameters()`.

The download of section 2.7, by hand: `hf_download("org/tiny", ["config.json", "model.safetensors"], cache, revision="abc123")` asks for `/org/tiny/resolve/abc123/config.json` (served directly), then `/org/tiny/resolve/abc123/model.safetensors` (a 302 to the CDN carrying the sha256), and leaves both files in `cache/org--tiny/abc123/` with no `.part` file.

## 4. The interface

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the SmolLM2 config by hand | unit | section 3: SmolLM2's config and its 134,515,008 parameters | the CLI's `info` |
| tiny HF checkpoints give HF's logits | golden | five HF checkpoints (GQA, MQA, window, biases, tied and untied, Llama-3 and YaRN rope) | SmolLM2 logits parity |
| parameter counts match HF and `M05.1` | property | `param_count` equals HF and M05.1 | `info`, C1's budget |
| state-dict keys are HF's | unit | HF's key order; `lm_head.weight` only when untied | the system's checkpoints load in transformers |
| cached decode equals the full forward | differential | prefill then decode equals the full forward, also through a bounded cache | `L8.2`'s generation |
| latent attention and experts | differential | MLA plus MoE after a dense layer, latent-cache decode, aux loss, count | C1's ablations |
| sink tokens stream like full attention | differential | window, sink tokens, and learned sinks through `SinkWindowCache` | streaming chats in `L8.2` |
| config round trip, both rope forms | property | `to_hf` and `from_hf` round-trip; 4.x and 5.x rope keys agree | old and new checkpoints |
| save/load round trip and shards | property | F32 bit-exact, BF16 rounded, sharded checkpoints | C1's release, larger Hub models |
| gradients reach every parameter | gradcheck | the tied embedding's gradient vs central differences; no silent parameter | C1 trains this model |
| validation | boundary | strict keys, unsupported rope, Llama `attention_bias`, bad ids | wrong checkpoints fail loudly |
| the download by hand | unit | the URL, the redirect, the cache layout | the CLI's `pull` |
| resume from a `.part` file | fault | Range resume hashes the whole file | a dropped connection costs nothing |
| a server that ignores ranges | fault | a 200 answer restarts the file | mirrors without range support |
| corrupt bytes are rejected | fault | sha256 checked; nothing left under the final name | the parity numbers are about these bytes |
| existing files are not downloaded again | unit | a verified file is skipped; a stale one replaced | `pull` twice is free |
| endpoint from the environment, validation | boundary | `HF_ENDPOINT`; bad repo ids and filenames | mirrors; no writes outside the cache |

### A model oracle test suite

[`test_llama.py`](../../../../course/solutions/learner-tests/L7.9/python/tests/l7-9-llama/test_llama.py) uses as oracles the golden logits and parameter counts that transformers produced for five tiny checkpoints, the full forward as the oracle of the cached one, M05.1's count for SmolLM2 built without weights, and a fake Hub on `127.0.0.1` (`http.server`) that redirects with `X-Linked-Etag`, can cut a response short, ignore ranges, or corrupt a byte. It adds config round-trips (with a 4.x `"type"` key), MLA and MoE variants, sink tokens through `SinkWindowCache`, BF16 saving, shards, and the validation errors. A local fake Hub is the right oracle for a downloader: every network failure the real one can produce is reproducible on demand.

## 5. The solution, step by step

### 5.1 From first principles

1. The checkpoint is the contract. Matching HF means two things: build the same modules from the same `config.json` keys, and register parameters under the same dotted names in the same order. Everything else in the module follows from those two.
2. `config.json` comes in several dialects (transformers 4.x and 5.x, Mistral, Qwen2, the course's `tl_*` extensions). One reader normalizes them into one dataclass, so every later step sees a single shape; a writer (`to_hf`) makes the round trip exact.
3. A block is two Pre-LN residual steps. Which attention (GQA or MLA) and which MLP (dense or MoE, by layer index) are config decisions made in the layer's constructor; the forward is the same for all.
4. Windows and sink tokens are a property of the mask, not of the attention module, because sink tokens must be exempt from the window. So the layer asks `L7.7`'s `attention_mask` for a mask and gives `GQAttention` no window of its own.
5. With a cache, positions must continue from what the cache holds, and every layer must see the same cache with its own index.
6. A tied head is one tensor with one name. It must be counted once, saved once, and receive gradient from both its uses.
7. Loading 135 M parameters should not spend time drawing random numbers that are about to be overwritten: build shapes without draws, then load strictly by name.
8. A download over a flaky network must be resumable, verified, and atomic: write to `.part`, hash everything including resumed bytes, rename only after the check.

### 5.2 The code, annotated

**Reading `config.json`.** The parts that decide correctness:

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

For SmolLM2 there is no `head_dim`, so it is $576 / 9 = 64$; `num_key_value_heads` is 3. Every `c.get(..., default)` mirrors HF's own default for that key, which is why a sparse config such as SmolLM2's builds the same model as in transformers. Calling `model_config()` at the end routes the sizes through `M05.1`'s validator, so an inconsistent config fails at load time, not in a matmul.

**From config to RoPE.**

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

One spec per model, shared by every layer: `L7.4` computes the frequencies, `L7.3` applies them.

**The decoder layer.**

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

This is section 2.1's two equations, one line each. The registration order is what makes `state_dict()` list `self_attn.*`, `mlp.*`, `input_layernorm.weight`, `post_attention_layernorm.weight` in HF's order.

**Building without draws.** With `init=False`, `LlamaModel` uses `_FlatRNG`, a `PCG32` stand-in that returns constants, builds one layer of each kind (dense, MoE) and deep-copies it for the others, and widens a one-row embedding to $V \times d$ zeros. Building SmolLM2 this way allocates the right shapes without 135 million Box-Muller draws; `from_pretrained` then overwrites every value.

**The forward.**

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

**Counting and loading.**

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

**The downloader.** `hf_download` validates the repo id and filenames (no absolute paths, no `..`, so nothing is written outside the cache), skips a file that exists and matches a given sha256, and calls `_fetch` for the rest:

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

`_KeepHeaders` is a redirect handler that remembers `X-Linked-Etag` and `X-Linked-Size` from the Hub's 302, because the CDN it redirects to does not repeat them. Without it, large files (the ones that are redirected) would never be verified. The etag is used only when it looks like a sha256 (64 hex digits); small files served directly carry a git blob hash instead.

Full solution: [`llama.py`](../../../../course/solutions/python/tinyllm/modern/llama.py), [`hf.py`](../../../../course/solutions/python/tinyllm/io/hf.py).

### 5.3 Why this solution

- **HF names, not a mapping table.** Registering modules under HF's names means `load_state_dict(strict=True)` is the whole loader, and the system's own checkpoints load in transformers. A renaming table would be one more thing that can silently drop or misplace a tensor.
- **Strict loading.** A missing or extra key raises. Lenient loading would leave a parameter at its random init, which produces plausible-looking but wrong logits.
- **Masks from `L7.7`, not windows in the attention.** Putting the window in `GQAttention` could not exempt sink tokens and would not know about a bounded cache; one mask function handles every cache type.
- **No-draw construction.** Drawing 135 M Gaussians in Python takes noticeable time and is thrown away. `_FlatRNG` and layer copying make loading cost about what reading the file costs.
- **`urllib` with a redirect handler, not `huggingface_hub`.** The library adds snapshots, symlinks, and Xet storage. The solution needs three properties (resume, verify, atomic rename) and gets them in about 100 lines with no dependency.
- **Cost.** The forward is $L$ blocks of `L7.5`/`L7.6` and `L7.2`/`L7.8` cost plus an $O(B T d V)$ head, which dominates for small models with large vocabularies such as SmolLM2.

### 5.4 Explain it back

1. Why is SmolLM2's `head_dim` 64 and not $576 / 3$?
   <details><summary>Answer</summary>HF's default is `hidden_size // num_attention_heads`: the query heads set the head width; kv heads only set how many key/value heads there are. $576 / 9 = 64$, and the kv projections are $3 \cdot 64 = 192$ wide.</details>
2. Why does `forward` read `cache.seq_len(0)` before running any layer?
   <details><summary>Answer</summary>`seq_len` counts tokens already cached; the new tokens' positions start there. Reading after layer 0 had updated the cache would shift every position by $T$; restarting at 0 per chunk would re-rotate decoded tokens to the prompt's positions.</details>
3. Why does the decoder layer pass `window=None` to `GQAttention`?
   <details><summary>Answer</summary>Sink tokens must stay visible outside the window, and a bounded cache needs a mask built from held positions. `attention_mask` handles both; a window inside `GQAttention` could not.</details>
4. Why does a tied model count, save, and load its embedding once?
   <details><summary>Answer</summary>It is one tensor used twice. `parameters()` yields it once, so `param_count` matches HF; there is no `lm_head` module, so `state_dict` has no `lm_head.weight`, and a checkpoint that stores the copy has it dropped before the strict load.</details>
5. What would go wrong if `_fetch` appended a 200 response to an existing `.part` file?
   <details><summary>Answer</summary>A server that ignores the Range header sends the whole file again; appending it would produce the first `start` bytes twice. The solution resets `start` and the hash and opens the file with `"wb"`.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. deriving a missing `head_dim` from the kv heads | SmolLM2 builds with the wrong shapes | `head_dim = hidden_size // num_attention_heads` |
| 2. ignoring transformers 5's `rope_parameters` | YaRN and Llama-3 checkpoints load with plain RoPE | reads `rope_parameters` first and treats what is left after `rope_theta` as the scaling |
| 3. restarting positions at 0 for every chunk | cached decode drifts after the prompt | positions start at `cache.seq_len(0)` (step 5) |
| 4. skipping the final RMSNorm | logits on the wrong scale; HF disagrees | `h = self.model.norm(h)` before the head |
| 5. ignoring `sliding_window` | Mistral agrees with HF only for short prompts | the reader picks up `sliding_window` and the layer turns it into a mask |
| 6. a sha256 that is never compared | corrupt weights load silently | `h.hexdigest() != expect` raises and deletes the `.part` file |
| 7. appending a 200 answer to the partial file | a resumed file is the start twice | a 200 with `start > 0` resets `start` and the hash |
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

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L7.1` | RMSNorm and the Pre-LN residual step |
| Back | `L7.2` | the dense MLP of every block |
| Back | `L7.3` | `RopeSpec` for every attention layer |
| Back | `L7.4` | `rope_scaling` frequencies and YaRN's temperature |
| Back | `L7.5` | GQA attention with the cache hook |
| Back | `L7.6` | MLA for `tl_attention = "mla"` |
| Back | `L7.7` | window and sink-token masks |
| Back | `L7.8` | MoE layers for `tl_num_experts > 0` |
| Back | `L0.6` | safetensors in every dtype |
| Back | `M05.1` | `ModelConfig` and the counts `info` reports |
| Back | `L0.4` | `Embedding`, `Linear`, `ModuleList`, the checkpoint names |
| Back | `L0.2` | logits through the tied embedding |
| Back | `L0.1` | `Tensor` |
| Back | `M06.3` | the initialization stream of a fresh model |
| Forward | `L8.2` | KV-cached generation over this model |
| Forward | `L8.1`, `L8.3` to `L8.6` | sampling, paged caches, quantization, speculative decoding |
| Forward | `L10.1` | the Rust engine's logits compared with this model's |
| Forward | C1 | the capstone's architecture and release format |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `LlamaForCausalLM` | HF `LlamaForCausalLM` | attention backends (SDPA, FlashAttention), `logits_to_keep`, tensor parallel plans | `transformers/models/llama/modeling_llama.py` |
| `from_hf` | HF `PretrainedConfig`, `rope_parameters` standardization | one schema for every rope variant, config validation | `transformers/modeling_rope_utils.py` |
| `from_pretrained` | HF weight loading, safetensors `safe_open` | memory-mapped lazy loading, key renaming (`conversion_mapping`) | `transformers/modeling_utils.py` |
| `hf_download` | `huggingface_hub.hf_hub_download` | ETag caching by commit, symlinked snapshots, Xet storage | `huggingface_hub/file_download.py` |
