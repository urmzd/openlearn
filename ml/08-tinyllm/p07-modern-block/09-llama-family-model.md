<!-- ol:module L7.9 -->
# Llama-family model, HF config, weight loading, downloader

## Overview

| | |
|---|---|
| **Module** | `L7.9` · build · Python · Pass 5 · 5 to 6 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/modern/llama.py`: `LlamaConfig`, `LlamaDecoderLayer`, `LlamaModel`, `LlamaForCausalLM`; `python/tinyllm/io/hf.py`: `hf_download`; and your own oracle tests in `python/tests/l7-9-llama/` |
| **Contract** | [`course/contracts/py/tinyllm/modern/llama.pyi`](../../../course/contracts/py/tinyllm/modern/llama.pyi), [`course/contracts/py/tinyllm/io/hf.pyi`](../../../course/contracts/py/tinyllm/io/hf.pyi) |
| **Tests** | `course/tests/L7.9/test_llama.py` and `test_hf.py` (what they check: section 4); five tiny random checkpoints saved by transformers 5.19.0 with their float32 logits and parameter counts in `course/fixtures/L7.9/` (`course/oracle/L7.9/llama_hf.py`); a local fake Hub for the downloader; your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L7.1` `RMSNorm`, `pre_norm_residual` · `L7.2` `GatedMLP` · `L7.3` `RopeSpec` · `L7.4` `rope_inv_freq_scaled` · `L7.5` `GQAttention` · `L7.6` `MLAttention` · `L7.7` `attention_mask` · `L7.8` `MoE` · `L0.6` `load_safetensors`, `save_safetensors` · `M05.1` `ModelConfig` · `L0.4` `Embedding`, `Linear`, `ModuleList` · `L0.2` the op library · `L0.1` `Tensor` · `M06.3` `PCG32` · reading: `M09.1` BF16, `L1.2` the SmolLM2 tokenizer (or `--ref-deps`) |
| **Used by** | MS-L7 through your CLI · later: `L8.1` to `L8.6` (inference), `L10.1` (logits parity), C1 (the capstone architecture) · later: `L8.2` |
| **Milestone** | `MS-L7` (your decoder loads and matches Hugging Face checkpoints) |
| **Optional depth** | Touvron et al., "Llama 2" (2023), section 2.2; Hugging Face, `modeling_llama.py` and the `safetensors` and Hub download documentation |

## Key Takeaways

- A Llama block is two Pre-LN residual steps, attention then MLP, and the whole model is HF's config keys and tensor names: SmolLM2's 134,515,008 parameters fall out of its config by hand (`test_hand_example_smollm2_config`).
- Loading by name is strict: every key in HF's order, a tied head stored once, transformers 4.x and 5.x configs read alike (`test_tiny_hf_logits_golden`, `test_config_roundtrip_and_both_rope_forms`).
- Positions continue from what the cache holds, so decoding one token at a time equals the full forward, with windows and sinks too (`test_cached_decode_equals_full_forward`).
- A download is resumable, verified against the published sha256, and atomic (`test_resume_from_part_file`, `test_corrupt_bytes_are_rejected`).

## How to work this chapter

```bash
ol start L7.9              # stubs llama.py and hf.py; prints your test path and rung (R5)
ol tests L7.9              # the course tests
# write your oracle tests in python/tests/l7-9-llama/, then:
ol check L7.9              # course tests and the mutation grade of your tests
ol milestone MS-L7 --smoke # your CLI on the committed tiny model
ol diff  L7.9              # after passing: your code against the reference
```

---

## 1. Why now

Part 7 gave you every piece of a modern decoder as a separate tested module, and none of them has produced a token yet. Your transformer from Part 5 (`L5.5`) is a 2017 design with its own parameter names, so it cannot load one published checkpoint. This module assembles RMSNorm, gated MLPs or experts, RoPE with scaled frequencies, GQA or latent attention, and windows into one model whose config and tensor names are Hugging Face's, then fetches SmolLM2-135M from the Hub with a downloader you wrote and matches HF's logits. Everything after this (the sampler, caches, kernels, the Rust engine, the capstone) runs this model.

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

`L0.4` made the dotted parameter names the safetensors keys. Registered in HF's order (decoder layer: `self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`; model: `embed_tokens`, `layers`, `norm`; then `lm_head`), your `state_dict` has exactly HF's keys. A tied model registers the embedding once and has no `lm_head.weight`; some tied checkpoints store the copy anyway, and old ones store `rotary_emb.inv_freq`: both are dropped before a strict load. BF16 weights arrive as float32 through `L0.6` (the conversion is `M09.1`'s). Large checkpoints come in shards listed by `model.safetensors.index.json`.

### 2.4 Counting what you loaded

`param_count()` is the number of parameters, a tied tensor once: HF's `num_parameters()` and `M05.1`'s count of `model_config()`. Building SmolLM2 with `init=False` allocates the tensors without 135 million Box-Muller draws (`from_pretrained` overwrites every value anyway).

### 2.5 Positions and the cache

Without explicit positions, a forward with a cache starts at `cache.seq_len(0)`: the cache already holds that many tokens. Every layer gets the same cache object and its own `layer` index. With a window, the mask comes from `L7.7`'s `attention_mask`, which understands both a bounded `SinkWindowCache` and a cache that keeps everything.

### 2.6 Training it

Built from the op library end to end, the model trains with your autograd: the cross-entropy reaches every parameter, and the tied embedding collects gradient from both of its uses, the lookup and the head.

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

Thirty layers make 106,202,880; the embedding $49152 \times 576 = 28{,}311{,}552$ (also the head: tied, counted once); the final norm 576. Total: **134,515,008**, HF's `num_parameters()`. This is `test_hand_example_smollm2_config`.

The download of section 2.7, by hand: `hf_download("org/tiny", ["config.json", "model.safetensors"], cache, revision="abc123")` asks for `/org/tiny/resolve/abc123/config.json` (served directly), then `/org/tiny/resolve/abc123/model.safetensors` (a 302 to the CDN carrying the sha256), and leaves both files in `cache/org--tiny/abc123/` with no `.part` file: `test_hand_example_layout_and_redirect`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_smollm2_config` | unit | section 3: SmolLM2's config and its 134,515,008 parameters | `{tinyllm} info` in MS-L7 |
| `test_tiny_hf_logits_golden` | golden | five HF checkpoints (GQA, MQA, window, biases, tied and untied, Llama-3 and YaRN rope) give HF's logits | SmolLM2 parity in MS-L7 |
| `test_param_counts_match_hf_and_m05_1` | property | `param_count` equals HF and M05.1 | `info`, C1's budget |
| `test_state_dict_keys_are_hf` | unit | HF's key order; `lm_head.weight` only when untied | your checkpoints load in transformers |
| `test_cached_decode_equals_full_forward` | differential | prefill then decode equals the full forward, also through a bounded cache | `L8.2`'s generation |
| `test_latent_attention_and_experts` | differential | MLA plus MoE after a dense layer, latent-cache decode, aux loss, count | C1's ablations |
| `test_sink_tokens_stream_like_full_attention` | differential | window, sink tokens, and learned sinks through `SinkWindowCache` | streaming chats in `L8.2` |
| `test_config_roundtrip_and_both_rope_forms` | property | `to_hf` and `from_hf` round-trip; 4.x and 5.x rope keys agree | old and new checkpoints |
| `test_save_load_roundtrip_and_shards` | property | F32 bit-exact, BF16 rounded, sharded checkpoints | C1's release, larger Hub models |
| `test_gradients_reach_every_parameter` | gradcheck | the tied embedding's gradient vs central differences; no silent parameter | C1 trains this model |
| `test_validation` | boundary | strict keys, unsupported rope, Llama `attention_bias`, bad ids | wrong checkpoints fail loudly |
| `test_hand_example_layout_and_redirect` | unit | the URL, the redirect, the cache layout | `{tinyllm} pull` |
| `test_resume_from_part_file` | fault | Range resume hashes the whole file | a dropped connection costs nothing |
| `test_server_that_ignores_ranges` | fault | a 200 answer restarts the file | mirrors without range support |
| `test_corrupt_bytes_are_rejected` | fault | sha256 checked; nothing left under the final name | the parity numbers are about these bytes |
| `test_existing_files_are_not_downloaded_again` | unit | a verified file is skipped; a stale one replaced | `pull` twice is free |
| `test_endpoint_from_environment_and_validation` | boundary | `HF_ENDPOINT`; bad repo ids and filenames | mirrors; no writes outside the cache |

### Your graded tests (rung R5)

Your oracles are the course's golden files (`$TINYLLM_FIXTURES/L7.9/hf_logits.npz` and `hf_params.json`) for all five tiny checkpoints, the full forward as the oracle of the cached one, M05.1's count for SmolLM2 built without weights, and a fake Hub on `127.0.0.1` (`http.server`) that redirects with `X-Linked-Etag`, can cut a response short, ignore ranges, or corrupt a byte. Add config round-trips (with a 4.x `"type"` key), MLA and MoE variants, sink tokens through `SinkWindowCache`, BF16 saving, shards, and the validation errors. Import only the contract modules. `ol check L7.9` requires 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. deriving a missing `head_dim` from the kv heads | SmolLM2 builds with the wrong shapes | `test_hand_example_smollm2_config` (mutant `s01`) |
| 2. ignoring transformers 5's `rope_parameters` | YaRN and Llama-3 checkpoints load with plain RoPE | `test_tiny_hf_logits_golden`, `test_config_roundtrip_and_both_rope_forms` (mutant `s02`) |
| 3. restarting positions at 0 for every chunk | cached decode drifts after the prompt | `test_cached_decode_equals_full_forward` (mutant `s03`) |
| 4. skipping the final RMSNorm | logits on the wrong scale; HF disagrees | `test_tiny_hf_logits_golden`, `test_gradients_reach_every_parameter` (mutant `s04`) |
| 5. ignoring `sliding_window` | Mistral agrees with HF only for short prompts | `test_tiny_hf_logits_golden`, `test_cached_decode_equals_full_forward` (mutant `s05`) |
| 6. a sha256 that is never compared | corrupt weights load silently | `test_corrupt_bytes_are_rejected` (mutant `s13`) |
| 7. appending a 200 answer to the partial file | a resumed file is the start twice | `test_server_that_ignores_ranges` (mutant `s14`) |
| Qwen2's biases not implied by its `model_type` | strict load fails on the bias keys | `test_tiny_hf_logits_golden` (mutant `s06`) |
| counting a tied embedding twice | `info` reports 163 million for SmolLM2 | `test_param_counts_match_hf_and_m05_1` (mutant `s07`) |
| one more dense layer than `first_k_dense_replace` | MoE configs build the wrong model | `test_latent_attention_and_experts` (mutant `s08`) |
| a mask that forgets the sink tokens | streams degrade once token 0 leaves the window | `test_sink_tokens_stream_like_full_attention` (mutant `s09`) |
| `tl_attention = "mla"` building GQA | MLA checkpoints do not load | `test_latent_attention_and_experts` (mutant `s10`) |
| the 4.x `"type"` key ignored | old checkpoints lose their rope scaling | `test_config_roundtrip_and_both_rope_forms` (mutant `s11`) |
| `tl_learned_sinks` not built | gpt-oss-style configs lose their sinks | `test_sink_tokens_stream_like_full_attention` (mutant `s12`) |
| the redirect's `X-Linked-Etag` dropped | nothing is verified for large files | `test_corrupt_bytes_are_rejected` (mutant `s15`) |
| a resumed hash that skips the bytes on disk | every resumed download fails its check | `test_resume_from_part_file` (mutant `s16`) |
| a range that starts one byte late | resumed files lose a byte | `test_resume_from_part_file` (mutant `s17`) |
| downloading a verified file again | `pull` costs 270 MB every time | `test_existing_files_are_not_downloaded_again` (mutant `s18`) |

## 6. Where it's used next

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

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `LlamaForCausalLM` | HF `LlamaForCausalLM` | attention backends (SDPA, FlashAttention), `logits_to_keep`, tensor parallel plans | `transformers/models/llama/modeling_llama.py` |
| `from_hf` | HF `PretrainedConfig`, `rope_parameters` standardization | one schema for every rope variant, config validation | `transformers/modeling_rope_utils.py` |
| `from_pretrained` | HF weight loading, safetensors `safe_open` | memory-mapped lazy loading, key renaming (`conversion_mapping`) | `transformers/modeling_utils.py` |
| `hf_download` | `huggingface_hub.hf_hub_download` | ETag caching by commit, symlinked snapshots, Xet storage | `huggingface_hub/file_download.py` |
